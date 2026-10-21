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
                                       (om.Cylinder(0.01, 0.03, "z"), (0, 0, 0)), (om.Sphere(0.02), (0.01, 0.02, 0.03))])
def test_solids_agree_with_the_budget_shapes(shape, pos):
    """A solid built from an og.mass shape has the shape's own inertia (MuJoCo's formulas) and sits at ``pos``."""
    kg = 0.3
    mp = cad.properties(cad.solid(shape, pos), mass_kg=kg)
    assert mp.mass == kg
    np.testing.assert_allclose(mp.com, pos, atol=1e-9)
    np.testing.assert_allclose(mp.inertia, shape.inertia(kg), rtol=1e-6, atol=1e-12)


def test_thin_shapes_have_no_solid():
    with pytest.raises(NotImplementedError):
        cad.solid(om.Rod("x", 0.3))
    with pytest.raises(ValueError):
        cad.properties(cad.solid(om.Sphere(0.01)))
