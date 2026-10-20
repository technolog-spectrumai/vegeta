"""The layer rules: Yax is standard library only; Ogma is numpy (+ Yax) at import time and never imports a host or an
engine (pandas only lazily, for display tables)."""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {"vegeta", "piccolo", "chiron", "mujoco", "cadquery", "ozaru", "races", "onboard", "pandas"}


def _module_level(tree):
    """Every node outside function bodies: what runs when the module is imported (a lazy import inside a
    function, e.g. pandas for a display table, is allowed)."""
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node
            yield from _module_level(node)


def imports(lib):
    for p in (ROOT / lib / "src").rglob("*.py"):
        for node in _module_level(ast.parse(p.read_text())):
            if isinstance(node, ast.Import):
                yield p, [a.name for a in node.names], node
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                yield p, [node.module], node


def test_yax_is_standard_library_only():
    allowed = set(sys.stdlib_module_names) | {"yax"}
    assert [f"{p}: {n}" for p, ns, _ in imports("yax") for n in ns if n.split(".")[0] not in allowed] == []


def test_ogma_imports_only_numpy_yax_and_the_standard_library():
    allowed = set(sys.stdlib_module_names) | {"ogma", "yax", "numpy"}
    bad = [f"{p}: {n}" for p, ns, _ in imports("ogma") for n in ns if n.split(".")[0] not in allowed]
    assert bad == []
    assert not [n for _, ns, _ in imports("ogma") for n in ns if n.split(".")[0] in FORBIDDEN]
