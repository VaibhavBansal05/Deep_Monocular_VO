"""Exercise M3 notebook controls without GPU, subprocesses or dataset access."""

import ast
import contextlib
import io
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("m3_*.ipynb"))


def cells(path):
    return ["".join(cell["source"]) for cell in json.loads(path.read_text())["cells"]]


def execute(source, namespace):
    exec(compile(source, "<M3 notebook cell>", "exec"), namespace)


def runtime(tmp_path, **settings):
    calls = []

    def run(*args):
        calls.append(args)
        if "--out" in args:
            out = Path(args[args.index("--out") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "smoke.json").write_text("{}")
            (out / "metrics.json").write_text("{}")

    namespace = dict(Path=Path, CONFIG="configs/m3.yaml", CHECKPOINT_ROOT=tmp_path / "checkpoints",
                     RESULT_ROOT=tmp_path / "results", WORK=tmp_path, DATA_ROOT=tmp_path / "data",
                     RUN_PILOT=False, PILOT_STEPS=200, RUN_FULL=False, EVALUATE=False,
                     EVAL_CKPT=None, EVAL_DEPTH=False, HAS_CHECKPOINT_RESUME=False,
                     RGB_DATASET_VERSION="vaibhavkrishnabansal/vkitti2-subset/1",
                     DEPTH_DATASET_VERSION="suryanshbajaj/vkitti2-depth-m3/1",
                     NUM_WORKERS=2, PYTHON="unused", run=run, shutil=shutil,
                     subprocess=SimpleNamespace(check_output=lambda *a, **k: ""))
    namespace.update(settings)
    return namespace, calls


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.name)
def test_cells_compile_and_training_defaults_are_safe(notebook):
    doc = json.loads(notebook.read_text())
    for index, cell in enumerate(doc["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{notebook.name}:{index}", "exec")
            assert cell["execution_count"] is None and not cell["outputs"]
    assignments = [node for node in ast.parse(cells(notebook)[1]).body if isinstance(node, ast.Assign)]
    switches = {node.targets[0].id: ast.literal_eval(node.value)
                for node in assignments if isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in {"RUN_PILOT", "RUN_FULL", "EVALUATE"}}
    assert switches == {"RUN_PILOT": notebook.name == "m3_kaggle.ipynb",
                        "RUN_FULL": False, "EVALUATE": False}


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.name)
@pytest.mark.parametrize("available", [False, True])
def test_resume_probe_handles_standalone_and_future_integration(notebook, available, monkeypatch):
    from mvo.common.train import engine

    if available:
        monkeypatch.setattr(engine, "restore_rng", lambda state: None, raising=False)
    else:
        monkeypatch.delattr(engine, "restore_rng", raising=False)
    setup = ast.parse(cells(notebook)[3])
    probe = next(node for node in setup.body if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "HAS_CHECKPOINT_RESUME" for t in node.targets))

    def check_output(command, **kwargs):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            execute(command[-1], {})
        return output.getvalue()

    namespace = dict(PYTHON="unused", subprocess=SimpleNamespace(check_output=check_output))
    execute(ast.unparse(probe), namespace)
    assert namespace["HAS_CHECKPOINT_RESUME"] is available


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.name)
@pytest.mark.parametrize("pilot,full", [(False, False), (False, True), (True, False), (True, True)])
def test_pilot_runs_only_when_requested(notebook, tmp_path, pilot, full):
    namespace, calls = runtime(tmp_path, RUN_PILOT=pilot, RUN_FULL=full, PILOT_STEPS=3)
    execute(cells(notebook)[6], namespace)
    assert len(calls) == (2 if pilot else 0)
    if pilot:
        assert calls[0][calls[0].index("--steps") + 1] == "3"
        assert "smoke" in calls[0] and "visualize" in calls[1]


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.name)
def test_fresh_full_run_works_without_resume_integration(notebook, tmp_path):
    namespace, calls = runtime(tmp_path, RUN_FULL=True)
    execute(cells(notebook)[6], namespace)
    execute(cells(notebook)[8], namespace)
    assert len(calls) == 1 and calls[0][0] == "scripts/train.py"
    assert "--resume" not in calls[0]


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.name)
@pytest.mark.parametrize("available", [False, True])
def test_existing_checkpoint_resumes_only_with_integration(notebook, tmp_path, available):
    namespace, calls = runtime(tmp_path, RUN_FULL=True, HAS_CHECKPOINT_RESUME=available)
    execute(cells(notebook)[6], namespace)
    last = namespace["OUT"] / "last.pt"
    last.parent.mkdir(parents=True)
    last.touch()
    if available:
        execute(cells(notebook)[8], namespace)
        assert "--resume" in calls[0] and str(last) in calls[0]
    else:
        with pytest.raises(RuntimeError, match="fresh output directory"):
            execute(cells(notebook)[8], namespace)
        assert not calls


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.name)
@pytest.mark.parametrize("external", [False, True])
def test_evaluation_only_uses_selected_checkpoint_without_history(notebook, tmp_path, external):
    namespace, calls = runtime(tmp_path, EVALUATE=True)
    execute(cells(notebook)[6], namespace)
    checkpoint = tmp_path / "attached/best.pt" if external else namespace["OUT"] / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    if external:
        namespace["EVAL_CKPT"] = str(checkpoint)
    execute(cells(notebook)[8], namespace)
    execute(cells(notebook)[9], namespace)
    assert len(calls) == 3 and not any(c[0] == "scripts/train.py" for c in calls)
    assert all(str(checkpoint) in command for command in calls)
    assert "--eval-depth" not in calls[0]


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.name)
def test_missing_evaluation_checkpoint_fails_before_any_action(notebook, tmp_path):
    namespace, calls = runtime(tmp_path, EVALUATE=True)
    execute(cells(notebook)[6], namespace)
    with pytest.raises(FileNotFoundError, match="Evaluation checkpoint missing"):
        execute(cells(notebook)[9], namespace)
    assert not calls


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.name)
@pytest.mark.parametrize("available", [False, True])
def test_optional_depth_evaluation_requires_supported_script(notebook, tmp_path, available):
    namespace, calls = runtime(tmp_path, EVALUATE=True, EVAL_DEPTH=True)
    namespace["subprocess"] = SimpleNamespace(
        check_output=lambda *a, **k: "--eval-depth" if available else "--model --ckpt")
    execute(cells(notebook)[6], namespace)
    checkpoint = namespace["OUT"] / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    if available:
        execute(cells(notebook)[9], namespace)
        assert "--eval-depth" in calls[0]
        if notebook.name == "m3_kaggle.ipynb":
            assert len(calls) == 2  # no repeated robustness evaluation in a depth-only version
            assert "--depth-dataset-version" in calls[0]
            assert "depth_evaluation" in str(namespace["RESULTS"])
    else:
        with pytest.raises(RuntimeError, match="depth-evaluation integration"):
            execute(cells(notebook)[9], namespace)
        assert not calls
