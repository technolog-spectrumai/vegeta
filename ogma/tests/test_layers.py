"""The layer rules: Yax is standard library only; Ogma is numpy (+ Yax) and never imports a host or an engine."""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {"vegeta", "piccolo", "chiron", "mujoco", "cadquery", "ozaru", "races", "onboard", "pandas"}


def imports(lib):
    for p in (ROOT / lib / "src").rglob("*.py"):
        for node in ast.walk(ast.parse(p.read_text())):
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
