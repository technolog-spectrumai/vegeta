"""Every component is geometry only: it imports CAD and geometry (cadquery, numpy, math, vegeta.dedalus, sibling
components) and no analysis tool, and it defines at least one Dedalus design whose defaults resolve."""
from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
MODULES = sorted(p.stem for p in HERE.glob("*.py") if p.stem != "__init__")
ALLOWED = {"__future__", "math", "cmath", "copy", "dataclasses", "functools", "itertools", "typing", "pathlib",
           "numpy", "cadquery", "vegeta.dedalus", "OCP"}


def _imports(path: Path) -> set[str]:
    out = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            out.add("." * node.level + (node.module or ""))
    return out


def test_there_are_components():
    assert MODULES, "no component modules"


@pytest.mark.parametrize("name", MODULES)
def test_imports_geometry_only(name):
    bad = [m for m in _imports(HERE / f"{name}.py")
           if not m.startswith(".") and m not in ALLOWED and not any(m.startswith(a + ".") for a in ALLOWED)]
    assert bad == [], f"components/{name}.py imports {bad}: geometry only (analysis stays in the notebooks)"


@pytest.mark.parametrize("name", MODULES)
def test_defines_designs_whose_defaults_resolve(name):
    pytest.importorskip("cadquery")
    from vegeta.dedalus import Design
    mod = importlib.import_module(f"components.{name}")
    designs = [c for _, c in inspect.getmembers(mod, inspect.isclass)
               if issubclass(c, Design) and c is not Design and c.__module__ == mod.__name__]
    assert designs, f"components/{name}.py defines no Design"
    for cls in designs:
        assert isinstance(cls().resolve(), dict)
