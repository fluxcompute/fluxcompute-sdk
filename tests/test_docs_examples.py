"""Structural checks for docs and examples: every published snippet must
parse, and docs must not reference symbols the SDK no longer has. The code
is not executed."""

from __future__ import annotations

import ast
import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
NOTEBOOKS = sorted((ROOT / "examples").glob("*.ipynb"))

# Names removed in 0.3.0; docs must not reference them.
REMOVED_SYMBOLS = (
    "RedisSessionManager",
    "redis_session",
    "[server]",
)


def _readme_python_blocks() -> list[tuple[int, str]]:
    md = (ROOT / "README.md").read_text(encoding="utf-8")
    return list(enumerate(re.findall(r"```python\n(.*?)```", md, re.S)))


def _notebook_code_cells(path: pathlib.Path) -> list[tuple[int, str]]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        # %pip / %matplotlib and friends aren't Python; skip magic-only cells.
        if any(ln.strip().startswith(("%", "!")) for ln in source.splitlines()):
            continue
        out.append((i, source))
    return out


def _notebook_text(path: pathlib.Path) -> str:
    nb = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(c.get("source", [])) for c in nb.get("cells", []))


@pytest.mark.parametrize("index,code", _readme_python_blocks())
def test_readme_python_blocks_parse(index, code):
    """A copy-pasteable snippet that raises SyntaxError is worse than none."""
    compile(code, f"README.md:block{index}", "exec")


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_valid_and_has_no_stored_output(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    assert nb.get("nbformat") == 4
    for i, cell in enumerate(nb.get("cells", [])):
        assert not cell.get("outputs"), f"{path.name} cell {i} has stored output"
        assert not cell.get("execution_count"), f"{path.name} cell {i} has an exec count"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_code_cells_parse(path):
    # Notebooks legitimately use top-level `await` — IPython compiles cells
    # with this flag, so the check has to match the real runtime rather than
    # plain script semantics.
    for index, code in _notebook_code_cells(path):
        compile(code, f"{path.name}:cell{index}", "exec",
                flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebooks_do_not_reference_removed_symbols(path):
    text = _notebook_text(path)
    for symbol in REMOVED_SYMBOLS:
        assert symbol not in text, f"{path.name} references removed {symbol!r}"


def test_readme_does_not_reference_removed_symbols():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for symbol in REMOVED_SYMBOLS:
        assert symbol not in text, f"README references removed {symbol!r}"


def _literal_list_from_notebook(path: pathlib.Path, name: str) -> list[str]:
    """Extract a list-of-strings literal by name using Python's own parser, so
    implicitly concatenated strings are joined exactly as the notebook sees
    them (a regex splits them and silently reads the wrong prompt)."""
    for _, code in _notebook_code_cells(path):
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign)
                    and any(getattr(t, "id", None) == name for t in node.targets)
                    and isinstance(node.value, ast.List)):
                try:
                    return [ast.literal_eval(e) for e in node.value.elts]
                except ValueError:
                    return []
    return []


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_tier_demo_prompts_actually_span_the_tiers(path):
    """The tier demo labels three prompts easy/medium/hard; the classifier
    must actually route them to those three tiers."""
    from fluxcompute.classifier.heuristic import classify

    samples = _literal_list_from_notebook(path, "samples")
    if not samples:
        pytest.skip(f"{path.name} has no tier-demo `samples` list")
    labels = [classify([{"role": "user", "content": s}]).label for s in samples]
    assert labels == ["easy", "medium", "hard"], (
        f"{path.name} tier demo routes {labels}, not easy/medium/hard"
    )


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_workload_bank_is_actually_mixed(path):
    """The savings demo's prompt bank must produce all three difficulty
    tiers."""
    from fluxcompute.classifier.heuristic import classify

    bank = _literal_list_from_notebook(path, "bank")
    if not bank:
        pytest.skip(f"{path.name} has no `bank` list")
    labels = {classify([{"role": "user", "content": q}]).label for q in bank}
    assert labels == {"easy", "medium", "hard"}, (
        f"{path.name} workload bank only produces {sorted(labels)}"
    )
