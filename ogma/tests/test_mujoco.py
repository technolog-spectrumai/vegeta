"""og.mujoco: the strings (no MuJoCo needed) and, with MuJoCo installed, what MuJoCo compiles from them — every
Ogma solid shape has MuJoCo's own inertia (inertiafromgeom), and a budget body has the budget's mass properties."""
import numpy as np
import pytest

from ogma import mass as om, mujoco as omj

SHAPES = [om.Box(0.04, 0.02, 0.01), om.Sphere(0.02), om.Cylinder(0.015, 0.06, "z"), om.Cylinder(0.015, 0.06, "x"),
          om.Capsule(0.01, 0.05, "z"), om.Capsule(0.01, 0.05, "y"), om.Ellipsoid(0.03, 0.02, 0.01)]


def test_strings():
    assert omj.geom(om.Box(0.04, 0.02, 0.01), "b", (0.1, 0, 0), kg=0.2) == \
        '<geom name="b" type="box" size="0.02 0.01 0.005" pos="0.1 0 0" mass="0.2"/>'
    assert 'quat="0.707106781187 0 0.707106781187 0"' in omj.geom(om.Cylinder(0.01, 0.1, "x"), "c")
    mp = om.MassProperties(1.5, np.array([0.1, 0.0, -0.02]), np.array([[0.02, -0.001, 0.0], [-0.001, 0.03, 0.0], [0.0, 0.0, 0.04]]))
    assert omj.inertial(mp) == '<inertial pos="0.1 0 -0.02" mass="1.5" fullinertia="0.02 0.03 0.04 -0.001 0 0"/>'
    with pytest.raises(NotImplementedError):
        omj.geom(om.Rod("x", 0.3), "r")


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"{type(s).__name__}-{getattr(s, 'axis', '')}")
def test_every_shape_has_mujocos_inertia(shape):
    pytest.importorskip("mujoco")
    kg = 0.35
    xml = omj.model(f'<body name="b">{omj.geom(shape, "g", kg=kg)}</body>')
    got = omj.compile_check(xml)["b"]
    assert got["mass"] == pytest.approx(kg, rel=1e-12)
    np.testing.assert_allclose(np.sort(got["inertia"]), np.sort(np.linalg.eigvalsh(shape.inertia(kg))), rtol=1e-9)


def test_budget_body_is_the_budget():
    pytest.importorskip("mujoco")
    b = om.Budget([om.Item("hull", 2.0, (0.0, 0.0, 0.0), om.Box(0.4, 0.2, 0.1)),
                   om.Item("battery", 0.6, (0.08, 0.02, -0.03), om.Box(0.1, 0.05, 0.03)),
                   om.Item("arm", 0.3, (-0.1, 0.1, 0.05), om.Rod("x", 0.3)),
                   om.Item("wheel", 0.4, (0.15, -0.12, -0.05), om.Cylinder(0.06, 0.04, "y")),
                   om.Item("sensor", 0.1, (0.2, 0.0, 0.06))])
    p = b.properties()
    got = omj.compile_check(omj.model(omj.body("rover", b)))["rover"]
    assert got["mass"] == pytest.approx(p.mass, rel=1e-12)
    np.testing.assert_allclose(got["com"], p.com, atol=1e-12)
    np.testing.assert_allclose(np.sort(got["inertia"]), np.sort(np.linalg.eigvalsh(p.inertia)), rtol=1e-9)
