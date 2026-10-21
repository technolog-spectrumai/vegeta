"""og.cad reproduces Dedalus's mass properties (tests/golden/cad.json) and agrees with og.mass's shapes."""
import json
from pathlib import Path

import numpy as np
import pytest

cq = pytest.importorskip("cadquery")
from ogma import cad, mass as om  # noqa: E402

G = json.loads((Path(__file__).parent / "golden" / "cad.json").read_text())
SOLIDS = {
    "box": lambda: cq.Workplane("XY").box(10, 20, 30).translate((5, -3, 12)),
    "cylinder": lambda: cq.Workplane("XZ").circle(8).extrude(25, both=True).translate((0, 4, 0)),
    "bracket": lambda: cq.Workplane("XY").box(60, 40, 8).faces(">Z").workplane().hole(12).edges("|Z").fillet(4).translate((10, 0, 4)),
}


@pytest.mark.parametrize("name", sorted(SOLIDS))
def test_mass_properties_are_dedalus(name):
    g = G["solids"][name]
    p = cad.mass_properties(SOLIDS[name]())
    assert p["volume"] == pytest.approx(g["props"]["volume"], rel=1e-12)
    np.testing.assert_allclose(p["com"], g["props"]["com"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(p["inertia"], g["props"]["inertia"], rtol=1e-12, atol=1e-9)
    for key, kw in (("density", {"density_kg_m3": 1240.0}), ("given", {"mass_kg": 0.25})):
        s = cad.mass_section(g["props"], "mm", **kw)
        assert s == g[key]


@pytest.mark.parametrize("shape,pos", [(om.Box(0.04, 0.02, 0.01), (0.1, 0.0, -0.02)), (om.Cylinder(0.015, 0.06, "x"), (0.0, 0.05, 0.0)),
                                       (om.Cylinder(0.01, 0.03, "z"), (0, 0, 0)), (om.Sphere(0.02), (0.01, 0.02, 0.03)),
                                       (om.Capsule(0.01, 0.05, "y"), (0.02, 0.0, 0.0)), (om.Capsule(0.008, 0.03, "z"), (0, 0, 0.1)),
                                       (om.Ellipsoid(0.03, 0.02, 0.01), (0.0, -0.01, 0.02))])
def test_solids_agree_with_the_budget_shapes(shape, pos):
    """A solid built from an og.mass shape has the shape's own inertia (MuJoCo's formulas) and sits at ``pos``."""
    kg = 0.3
    mp = cad.properties(cad.solid(shape, pos), mass_kg=kg)
    assert mp.mass == kg
    np.testing.assert_allclose(mp.com, pos, atol=1e-9)
    np.testing.assert_allclose(mp.inertia, shape.inertia(kg), rtol=1e-6, atol=1e-12)


def test_thin_shapes_have_no_solid_but_a_stand_in():
    with pytest.raises(NotImplementedError):
        cad.solid(om.Rod("x", 0.3))
    for shape in (om.Rod("x", 0.3), om.Plate(0.2, 0.1, "xy"), om.Shell(0.1, 0.05, 0.04)):
        assert cad.mass_properties(cad.solid(shape, thin=0.002))["volume"] > 0
    with pytest.raises(ValueError):
        cad.properties(cad.solid(om.Sphere(0.01)))


ROVER = om.Budget([om.Item("hull", 2.0, (0.0, 0.0, 0.0), om.Box(0.4, 0.2, 0.1), group="structure"),
                   om.Item("battery", 0.6, (0.08, 0.02, -0.03), om.Box(0.1, 0.05, 0.03), group="power"),
                   om.Item("arm", 0.3, (-0.1, 0.1, 0.05), om.Rod("x", 0.3), group="structure"),
                   om.Item("wheel", 0.4, (0.15, -0.12, -0.05), om.Cylinder(0.06, 0.04, "y"), group="drive"),
                   om.Item("mast", 0.2, (0.0, 0.0, 0.15), om.Capsule(0.015, 0.2, "z"), group="sensors"),
                   om.Item("dome", 0.15, (0.12, 0.0, 0.07), om.Ellipsoid(0.04, 0.04, 0.02), group="sensors"),
                   om.Item("sensor", 0.1, (0.2, 0.0, 0.06), group="sensors")])


def test_assembly_is_the_budget(tmp_path):
    """The CAD built from a budget is the same machine: its mass properties (solids from OpenCascade) equal the
    budget's; the assembly holds one part per shaped item and exports."""
    want, got = ROVER.properties(), cad.assembly_properties(ROVER)
    assert got.mass == pytest.approx(want.mass, rel=1e-12)
    np.testing.assert_allclose(got.com, want.com, atol=1e-12)
    np.testing.assert_allclose(got.inertia, want.inertia, rtol=1e-5, atol=1e-9)
    asm = cad.assembly(ROVER)
    assert len(asm.children) == 6                                   # the point mass has no geometry
    assert len(cad.assembly(ROVER, thin=None).children) == 5        # nor, then, the rod
    for suffix in (".step", ".stl"):
        p = cad.export(asm, tmp_path / f"rover{suffix}")
        assert Path(p).stat().st_size > 0
