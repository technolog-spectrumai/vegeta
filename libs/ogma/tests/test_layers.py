"""Ogma's layer rule: numpy (and the standard library) at import time; it never imports Vegeta's tools or designs, a
simulator or CAD at import time (mujoco, cadquery only inside the strategies' functions; pandas only for display)."""
import ast
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "ogma"


def _module_level(tree):
    """Every node outside function bodies: what runs when the module is imported."""
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node
            yield from _module_level(node)


def _names(node):
    if isinstance(node, ast.Import):
        return [a.name for a in node.names]
    if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
        return [node.module]
    return []


def test_ogma_imports_only_numpy_and_the_standard_library():
    allowed = set(sys.stdlib_module_names) | {"ogma", "numpy"}
    bad = [f"{p.name}: {n}" for p in SRC.rglob("*.py") for node in _module_level(ast.parse(p.read_text()))
           for n in _names(node) if n.split(".")[0] not in allowed]
    assert bad == []


def test_ogma_never_imports_vegeta_anywhere():
    """Designs import Ogma, never the reverse — not even lazily."""
    bad = [f"{p.name}: {n}" for p in SRC.rglob("*.py") for node in ast.walk(ast.parse(p.read_text()))
           for n in _names(node) if n.split(".")[0] in {"vegeta", "piccolo", "chiron", "races"}]
    assert bad == []
