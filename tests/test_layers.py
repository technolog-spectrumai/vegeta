"""Yax imports nobody's code: no host (Vegeta, Piccolo, Chiron, Ogma), and at import time only the standard library,
numpy and USD (pxr). The engines a format needs (mujoco, cadquery/OCP, jsbsim) are imported inside its functions."""
import ast
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "yax"
HOSTS = {"vegeta", "piccolo", "chiron", "ogma", "races", "onboard", "ozaru"}
AT_IMPORT = set(sys.stdlib_module_names) | {"yax", "numpy", "pxr"}


def _module_level(tree):
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node
            yield from _module_level(node)


def _imports(nodes):
    for node in nodes:
        if isinstance(node, ast.Import):
            yield from (a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module.split(".")[0]


def test_no_host_anywhere():
    bad = [f"{p.name}: {m}" for p in SRC.rglob("*.py") for m in _imports(ast.walk(ast.parse(p.read_text()))) if m in HOSTS]
    assert bad == []


def test_engines_only_inside_functions():
    bad = [f"{p.name}: {m}" for p in SRC.rglob("*.py") for m in _imports(_module_level(ast.parse(p.read_text())))
           if m not in AT_IMPORT]
    assert bad == []
