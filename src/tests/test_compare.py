import importlib.util
import json
import sys
from pathlib import Path

import pytest

COMPARE_PATH = Path(__file__).parents[1] / "scripts" / "compare.py"
SPEC = importlib.util.spec_from_file_location("compare", COMPARE_PATH)
assert SPEC is not None and SPEC.loader is not None
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)

load = compare.load
main = compare.main
render = compare.render


def _write_metrics(results_dir, model, metrics):
    model_dir = results_dir / model
    model_dir.mkdir(parents=True)
    (model_dir / "metrics.json").write_text(json.dumps(metrics))


def test_load_reads_model_metrics_from_expected_structure(tmp_path):
    _write_metrics(tmp_path, "m1", {"owner": "Pranav", "ate_sim3": 2.0})
    _write_metrics(tmp_path, "m3", {"owner": "Suryansh", "ate_sim3": 1.0})

    rows = load(tmp_path)

    assert set(rows) == {"m1", "m3"}
    assert rows["m3"]["owner"] == "Suryansh"
    assert rows["m1"]["ate_sim3"] == 2.0


def test_render_markdown_has_columns_models_and_best_value():
    rows = {
        "m1": {"owner": "Pranav", "ate_sim3": 2.0},
        "m3": {"owner": "Suryansh", "ate_sim3": 1.0},
    }

    table = render(rows)

    assert "ATE sim(3) [m]" in table
    assert "Classical ORB/LK + MLP" in table
    assert "Self-supervised PoseNet+DepthNet" in table
    assert "**1.000**" in table
    assert "**2.000**" not in table


def test_render_uses_em_dash_for_missing_metric_values():
    table = render({"m1": {"owner": "Pranav", "ate_sim3": 1.0}})

    assert "—" in table


def test_render_latex_produces_table_structure():
    table = render({"m1": {"owner": "Pranav", "ate_sim3": 1.0}}, latex=True)

    assert table.startswith("\\begin{tabular}")
    assert "\\hline" in table
    assert r"t\_rel [\%]" in table
    assert r"r\_rel [deg/100m]" in table
    assert "\\textbf{1.000}" in table
    assert table.endswith("\\end{tabular}")


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
def test_render_treats_non_finite_metrics_as_unavailable(non_finite):
    rows = {
        "m1": {"owner": "Pranav", "t_rel_pct": non_finite},
        "m2": {"owner": "Abhishek", "t_rel_pct": 2.0},
    }

    table = render(rows)
    rendered_rows = {
        cells[0]: cells
        for line in table.splitlines()[2:]
        if (cells := line.strip("| ").split(" | "))
    }

    assert rendered_rows["Classical ORB/LK + MLP"][6] == "—"
    assert rendered_rows["CNN + GRU/LSTM"][6] == "**2.00**"


def test_main_adds_scale_note_and_reports_missing_models(tmp_path, monkeypatch, capsys):
    _write_metrics(tmp_path, "m3", {"owner": "Suryansh", "ate_sim3": 1.0})
    monkeypatch.setattr(sys, "argv", ["compare.py", "--results", str(tmp_path)])

    main()

    output = capsys.readouterr().out
    assert "Scale handling differs by design" in output
    assert "Not yet evaluated: m1, m2, m4." in output


def test_main_fails_clearly_when_no_metrics_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["compare.py", "--results", str(tmp_path)])

    with pytest.raises(SystemExit, match="no metrics.json"):
        main()
