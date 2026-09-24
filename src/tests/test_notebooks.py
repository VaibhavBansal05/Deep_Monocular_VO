"""Structural checks on the training notebooks.

Notebooks are not exercised by CI in any other way, and two defects have
already shipped in them: cell sources written without trailing newlines (Colab
collapsed each cell onto one line and handed the whole Python cell to bash),
and a double backslash that broke a shell line continuation.
"""

import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))


def _cells(path):
    return json.loads(path.read_text())["cells"]


@pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda p: p.name)
def test_is_valid_notebook_json(nb):
    doc = json.loads(nb.read_text())
    assert doc.get("nbformat") == 4
    assert doc["cells"], "notebook has no cells"


@pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda p: p.name)
def test_every_source_line_keeps_its_newline(nb):
    """The .ipynb format stores each line WITH its trailing newline. Without
    them Jupyter and Colab concatenate a cell onto a single line, which for a
    cell beginning with a shell escape hands the entire cell to bash."""
    for i, c in enumerate(_cells(nb)):
        for ln in c["source"][:-1]:
            assert ln.endswith("\n"), f"cell {i}: line missing newline: {ln!r}"


@pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda p: p.name)
def test_shell_line_continuations_are_single_backslashes(nb):
    r"""In bash, a line ending in '\' continues; one ending in '\\' is an
    escaped backslash and terminates the command, silently splitting it."""
    for i, c in enumerate(_cells(nb)):
        if c["cell_type"] != "code":
            continue
        for ln in c["source"]:
            stripped = ln.rstrip("\n")
            if stripped.endswith("\\") and stripped.endswith("\\\\"):
                pytest.fail(f"cell {i}: double backslash continuation: {ln!r}")


@pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda p: p.name)
def test_no_credentials_are_embedded(nb):
    """Tokens must come from Colab userdata, Kaggle Secrets or getpass, never
    from a literal. Cell outputs are saved into the file when it is committed."""
    text = "\n".join("".join(c["source"]) for c in _cells(nb))
    assert "github_pat_" not in text
    assert "ghp_" not in text
    for c in _cells(nb):
        assert not c.get("outputs"), "committed notebooks must have no outputs"
