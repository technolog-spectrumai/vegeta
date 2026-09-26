"""What the model writes, and the static check it must pass before it is run.

The model writes one Python file: a Dedalus ``Design`` whose ``build(p)`` returns a CadQuery Workplane/Shape (one
part) or a ``cq.Assembly`` of named, coloured parts. ``validate_source`` is a static screen (an import allow-list and
banned names/attributes); it is a speed bump, not a security boundary — the code still runs only in the walled
subprocess of ``sandbox``.
"""
from __future__ import annotations

import ast

ALLOWED_IMPORTS = ("cadquery", "math", "numpy", "vegeta.dedalus", "typing", "__future__")
BANNED_NAMES = frozenset({
    "open", "eval", "exec", "compile", "__import__", "getattr", "setattr", "delattr", "globals", "locals", "vars",
    "breakpoint", "input", "memoryview", "exit", "quit", "help", "__builtins__", "__loader__", "__spec__",
})
BANNED_ATTRS = frozenset({
    "exporters", "importers", "export", "save", "load", "tofile", "fromfile", "loadtxt", "savetxt", "genfromtxt",
    "memmap", "ctypeslib", "lib", "occ_impl", "system", "popen", "spawn", "fork", "remove", "unlink", "rmtree",
    "write_text", "write_bytes", "read_text", "read_bytes",
})
ALLOWED_DUNDERS = frozenset({"__init__", "__name__"})

CONTRACT = """Write ONE Python file that defines a Dedalus design:

```python
import cadquery as cq
from vegeta.dedalus import Design, Parameter

class Model(Design):
    parameters = [Parameter("height", 120.0, "mm", min=20, description="...")]   # the key dimensions
    def build(self, p):
        ...                                   # CadQuery; lengths in millimetres, Z up, the object resting on z = 0
        return assembly                       # cq.Assembly of named, coloured parts (or one Workplane)
```

Rules:
- Imports: only cadquery, math, numpy, and `from vegeta.dedalus import Design, Parameter`. No file, network,
  process or OS access; no open/eval/exec/getattr/__import__, no dunder attributes, no exporters or save/load.
- One class deriving from Design; `build(self, p)` must return either a cq.Workplane / Shape (one part), or a
  cq.Assembly whose parts are added with `assy.add(workplane, name="seat", color=cq.Color(r, g, b))` (r, g, b in 0..1).
  Give every part a short snake_case name that matches the plan, and a colour.
- Build every part as a closed solid from primitives (box, cylinder, sphere, extrude, revolve, loft), booleans
  (union, cut, intersect) and modest fillets/chamfers (smaller than the edges they round). Avoid tiny features,
  zero-thickness walls, coplanar booleans and self-intersections: every part must be a valid, watertight solid.
- Parts should touch or overlap where they are joined (no floating parts), unless the plan says otherwise.
- Put the main dimensions in `parameters` with sensible defaults and ranges; keep the code short and plain.
"""

EXAMPLE_SOURCE = '''import cadquery as cq
from vegeta.dedalus import Design, Parameter


class Model(Design):
    """A small table: a round top on one central column and a disc foot."""

    parameters = [
        Parameter("top_diameter", 400.0, "mm", min=100),
        Parameter("height", 450.0, "mm", min=100),
        Parameter("column_diameter", 50.0, "mm", min=10),
    ]

    def build(self, p):
        foot = cq.Workplane("XY").circle(p["top_diameter"] * 0.3).extrude(20)
        column = cq.Workplane("XY").workplane(offset=20).circle(p["column_diameter"] / 2).extrude(p["height"] - 45)
        top = cq.Workplane("XY").workplane(offset=p["height"] - 25).circle(p["top_diameter"] / 2).extrude(25).edges(">Z").fillet(4)
        assy = cq.Assembly(name="table")
        assy.add(foot, name="foot", color=cq.Color(0.2, 0.2, 0.22))
        assy.add(column, name="column", color=cq.Color(0.6, 0.6, 0.62))
        assy.add(top, name="top", color=cq.Color(0.55, 0.36, 0.2))
        return assy
'''


def _module_allowed(name: str) -> bool:
    if any(seg in BANNED_ATTRS or seg in BANNED_NAMES for seg in name.split(".")):
        return False  # e.g. cadquery.occ_impl.exporters
    return any(name == a or name.startswith(a + ".") for a in ALLOWED_IMPORTS)


def validate_source(source: str) -> list[str]:
    """Problems that stop ``source`` from being run (empty list = it may run in the sandbox)."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"syntax error at line {exc.lineno}: {exc.msg}"]
    problems: list[str] = []
    for node in ast.walk(tree):
        line = getattr(node, "lineno", "?")
        if isinstance(node, ast.Import):
            problems += [f"line {line}: import of '{a.name}' is not allowed" for a in node.names if not _module_allowed(a.name)]
        elif isinstance(node, ast.ImportFrom):
            mod = ("." * node.level) + (node.module or "")
            if node.level or not _module_allowed(node.module or ""):
                problems.append(f"line {line}: import from '{mod}' is not allowed")
            elif any(a.name in BANNED_ATTRS or a.name in BANNED_NAMES for a in node.names):
                problems.append(f"line {line}: importing {', '.join(a.name for a in node.names)} from '{mod}' is not allowed")
            elif node.module == "vegeta.dedalus":
                problems += [f"line {line}: only Design and Parameter may be imported from vegeta.dedalus"
                             for a in node.names if a.name not in ("Design", "Parameter")]
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            problems.append(f"line {line}: '{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute):
            a = node.attr
            if a.startswith("__") and a.endswith("__") and a not in ALLOWED_DUNDERS:
                problems.append(f"line {line}: attribute '{a}' is not allowed")
            elif a in BANNED_ATTRS:
                problems.append(f"line {line}: '.{a}' is not allowed (no file access from the model's code)")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            problems.append(f"line {line}: global/nonlocal is not allowed")
    designs = [n for n in tree.body if isinstance(n, ast.ClassDef) and any(
        (isinstance(b, ast.Name) and b.id == "Design") or (isinstance(b, ast.Attribute) and b.attr == "Design") for b in n.bases)]
    if len(designs) != 1:
        problems.append(f"the file must define exactly one class deriving from Design (found {len(designs)})")
    elif not any(isinstance(n, ast.FunctionDef) and n.name == "build" for n in designs[0].body):
        problems.append(f"class {designs[0].name} has no build(self, p) method")
    return sorted(set(problems), key=problems.index)
