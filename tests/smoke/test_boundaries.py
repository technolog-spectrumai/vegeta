"""Boundaries, checked from the source text (no imports):

- the lab rules (``tools/labs.py``): libs never import labs, a lab never imports another lab, nothing imports a lab's
  ``exp/`` (limbo) — labs exchange machines only as files written by Yax;
- Piccolo's code (``piccolo``, ``onboard``) is imported nowhere: Vegeta hands Piccolo files, never code."""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HATCHED = ("piccolo", "onboard")
SKIP = (".venv", "vendor", "__pycache__", "out")


def imports(path):
    for node in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module


def test_nothing_imports_piccolo():
    bad = []
    for p in sorted(ROOT.rglob("*.py")):
        if set(p.relative_to(ROOT).parts) & set(SKIP):
            continue
        bad += [f"{p.relative_to(ROOT)}: {n}" for n in imports(p) if n.split(".")[0] in HATCHED]
    assert bad == []


def test_lab_rules():
    """libs never import labs, a lab never imports another lab, nothing imports a lab's exp/ (tools/labs.py)."""
    sys.path.insert(0, str(ROOT / "tools"))
    import labs
    bad = labs.check(ROOT)
    assert bad == [], "\n".join(bad)
