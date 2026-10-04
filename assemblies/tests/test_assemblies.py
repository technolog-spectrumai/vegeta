"""Assemblies are geometry only, load by spec, and put each component where the notebooks and ``scenarios/`` put it.
Quick: one CAD build per assembly (seconds), no solver."""
from __future__ import annotations

import ast
import importlib
import math
from pathlib import Path

import pytest

cq = pytest.importorskip("cadquery")
from vegeta import dedalus  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
MODULES = sorted(p.stem for p in HERE.glob("*.py") if p.stem not in ("__init__", "base"))
SPECS = {"quadcopter": "Quadcopter", "fixed_wing": "FixedWingDrone", "survey_boat": "SurveyBoatAssembly",
         "submarine": "SubmarineAssembly"}
ALLOWED = {"__future__", "math", "typing", "dataclasses", "numpy", "cadquery", "vegeta.dedalus", "components"}


def _imports(path):
    out = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            out.add("." * node.level + (node.module or ""))
    return out


def test_every_module_is_listed():
    assert sorted(SPECS) == MODULES


@pytest.mark.parametrize("name", MODULES + ["base"])
def test_imports_geometry_only(name):
    bad = [m for m in _imports(HERE / f"{name}.py")
           if not m.startswith(".") and m not in ALLOWED and not any(m.startswith(a + ".") for a in ALLOWED)]
    assert bad == [], f"assemblies/{name}.py imports {bad}: geometry only (analysis stays in the notebooks)"


@pytest.fixture(scope="module")
def built():
    cache = {}

    def get(name):
        if name not in cache:
            design = dedalus.load_design(f"assemblies.{name}:{SPECS[name]}")
            p = design.resolve()
            cache[name] = design, p, design.parts(p)
        return cache[name]
    return get


def _box(shape):
    return shape.BoundingBox()


@pytest.mark.parametrize("name", sorted(SPECS))
def test_parts_are_valid_and_build_is_their_compound(name, built):
    design, p, parts = built(name)
    assert all(s.isValid() for s in parts.values())
    compound = design.build(p)
    assert len(list(compound)) == len(parts)
    assert design.source_identity()["qualname"].endswith(".parts")


def test_quadcopter_props_on_motors_on_pads(built):
    _, p, parts = built("quadcopter")
    assert set(parts) == {"frame", *(f"motor_{i}" for i in range(1, 5)), *(f"propeller_{i}" for i in range(1, 5))}
    r = p["wheelbase"] / 2
    for k in range(4):
        ang = math.radians(45 + 90 * k)
        m = _box(parts[f"motor_{k + 1}"])
        assert m.center.x == pytest.approx(r * math.cos(ang), abs=0.1) and m.center.y == pytest.approx(r * math.sin(ang), abs=0.1)
        assert m.zmin == pytest.approx(p["arm_height"], abs=0.01)
        assert _box(parts[f"propeller_{k + 1}"]).zmin > p["arm_height"] + p["motor_height"] - 6    # hub on the shaft
    # neighbours are mirror images (counter-rotating), diagonals the same
    p1, p2, p3 = (_box(parts[f"propeller_{i}"]) for i in (1, 2, 3))
    off = lambda b, i: (b.center.x - r * math.cos(math.radians(45 + 90 * (i - 1))), b.center.y - r * math.sin(math.radians(45 + 90 * (i - 1))))
    assert off(p1, 1) == pytest.approx(off(p3, 3), abs=0.1)
    assert off(p2, 2)[1] == pytest.approx(-off(p1, 1)[1], abs=0.1)


def test_fixed_wing_props_ahead_of_the_nacelles(built):
    _, p, parts = built("fixed_wing")
    x = -p["nacelle_forward"] - p["propeller_gap"]
    for name, y in (("propeller_right", -p["nacelle_y"]), ("propeller_left", p["nacelle_y"])):
        b = _box(parts[name])
        assert b.xlen < 20 and b.zlen == pytest.approx(228.6, abs=0.5)          # disk across the flight direction
        assert b.center.y == pytest.approx(y, abs=1) and abs(b.center.x - x) < 8
    assert _box(parts["airframe"]).xmin == pytest.approx(-p["nose_length"], abs=1)       # fuselage nose


def test_fixed_wing_angle_of_attack_turns_the_props_too():
    design = dedalus.load_design("assemblies.fixed_wing:FixedWingDrone")
    p = design.resolve(angle_of_attack_deg=10, propeller="10x6, 2 blades")      # two blades: the box centre is the hub
    b = _box(design.parts(p)["propeller_left"])
    th = math.radians(10)
    x0 = -p["nacelle_forward"] - p["propeller_gap"]
    assert b.center.z == pytest.approx(-x0 * math.sin(th), abs=3)


def test_boat_prop_behind_the_pod(built):
    _, p, parts = built("survey_boat")
    b = _box(parts["propeller"])
    assert b.center.x == pytest.approx(-p["bracket_thickness"] - 90 - p["propeller_gap"], abs=6)
    assert b.center.z == pytest.approx(0.55 * p["depth"] - p["bracket_height"], abs=6)
    assert b.xlen < 15                                                           # disk across +x


def test_submarine_prop_behind_the_tail(built):
    _, p, parts = built("submarine")
    b = _box(parts["propeller"])
    assert b.center.x == pytest.approx(-p["propeller_gap"], abs=6) and b.xlen < 25
    assert _box(parts["vehicle"]).xmin == pytest.approx(0, abs=1)


def test_component_parameters_pass_through():
    from assemblies import pick
    from components.submarine import Submarine
    design = dedalus.load_design("assemblies.submarine:SubmarineAssembly")
    p = design.resolve(length=900)
    assert "part" not in p and pick(p, Submarine, part="body")["length"] == 900
    assert importlib.import_module("assemblies").Assembly
