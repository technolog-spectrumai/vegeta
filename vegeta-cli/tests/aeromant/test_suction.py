"""The suction_hood template: rendering in both dialects, the duct column, parameter checks, the results reader."""

import numpy as np
import pytest

from vegeta import aeromant
from vegeta.aeromant import CFDCase, OpenFOAMEnvironment, write_stl_ascii
from vegeta.aeromant.stl import Surface

COM = OpenFOAMEnvironment(env={"WM_PROJECT_VERSION": "v2412"})
ORG = OpenFOAMEnvironment(env={"WM_PROJECT_VERSION": "14"})
PARAMS = dict(flow_rate=0.35, duct_center=(-0.1, 0.0, 0.0), duct_inner=0.14, duct_wall=0.01, floor_height=0.30,
              kinematic_viscosity=1.5e-5, density=1.2, reference_length=0.5)


def _box(lo, hi):
    """Closed box surface (12 triangles, outward normals)."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    f = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    return np.array([[v[a], v[b], v[c]] for a, b, c in f])


@pytest.fixture(scope="module")
def hood_stl(tmp_path_factory):
    """A hood block on the road (lips at 0.03) with a duct block through the floor (to 0.32)."""
    tris = np.concatenate([_box((-0.225, -0.25, 0.03), (0.025, 0.25, 0.17)), _box((-0.18, -0.08, 0.17), (-0.02, 0.08, 0.32))])
    return write_stl_ascii(Surface(tris), tmp_path_factory.mktemp("suction") / "hood.stl")


@pytest.mark.parametrize("env", [COM, ORG], ids=["com", "org"])
def test_suction_template_renders(tmp_path, hood_stl, env):
    case = CFDCase("suction_hood", hood_stl, PARAMS, tmp_path / "c", geometry_units="m", environment=env)
    res = case.prepare()
    assert res.ok, res.messages
    for f in (tmp_path / "c").rglob("*"):
        if f.is_file() and f.suffix != ".stl" and f.name != "aeromant_case.json":
            assert "{{" not in f.read_text().replace("{{...}}", ""), f
    d = aeromant.open_case(tmp_path / "c")["derived"]
    assert float(d["SUCTION_W"]) == pytest.approx(0.35 / 0.14 ** 2)
    assert float(d["X1"]) == pytest.approx(-0.1 - 0.075) and float(d["X2"]) == pytest.approx(-0.1 + 0.075)   # through the duct wall
    assert float(d["ZTOP"]) == 0.3 and float(d["GROUND_U"]) == 0.0
    bm = (tmp_path / "c/system/blockMeshDict").read_text()
    assert bm.count("hex (") == 9 and "suction" in bm and "ground" in bm
    u = (tmp_path / "c/0.orig/U").read_text()
    assert "pressureInletOutletVelocity" in u and f"uniform (0 0 {float(d['SUCTION_W']):.9g})" in u
    assert res.metrics["reynolds_number"] == pytest.approx(0.35 / 0.14 ** 2 * 0.5 / 1.5e-5)


def test_suction_parameter_checks(tmp_path, hood_stl):
    t = aeromant.get_template("suction_hood")
    with pytest.raises(ValueError, match="requires explicit"):
        t.resolve({"flow_rate": 0.3})
    low = CFDCase("suction_hood", hood_stl, dict(PARAMS, floor_height=0.5), tmp_path / "low", geometry_units="m", environment=COM).prepare()
    assert not low.ok and "through the floor" in low.messages[0]
    off = CFDCase("suction_hood", hood_stl, dict(PARAMS, duct_center=(5.0, 0.0, 0.0)), tmp_path / "off", geometry_units="m", environment=COM).prepare()
    assert not off.ok and "inside the domain" in off.messages[0]
    assert "suction" in t.patches and "ground" in t.patches


def test_suction_results_from_function_objects(tmp_path, hood_stl):
    case = CFDCase("suction_hood", hood_stl, dict(PARAMS, ground_speed=1.0), tmp_path / "c", geometry_units="m", environment=COM)
    assert case.prepare().ok
    for name, col in (("suctionPressure", -350.0), ("suctionFlow", 0.35), ("hoodPressure", -60.0)):
        d = tmp_path / "c/postProcessing" / name / "0"
        d.mkdir(parents=True)
        (d / "surfaceFieldValue.dat").write_text("# Region type : patch\n# Time\tvalue\n" + "".join(f"{k}\t{col * (1 + 0.01 * (k % 2))}\n" for k in range(1, 61)))
    res = aeromant.read_case_results(tmp_path / "c")
    m = res.metrics
    assert m["fan_static_pressure_Pa"] == pytest.approx(1.2 * 350 * 1.005, rel=1e-6)
    assert m["flow_rate_m3_s"] == pytest.approx(0.35 * 1.005, rel=1e-6)
    assert m["air_power_W"] == pytest.approx(m["flow_rate_m3_s"] * m["fan_static_pressure_Pa"])
    assert m["hood_wall_pressure_Pa"] == pytest.approx(-1.2 * 60 * 1.005, rel=1e-6)
    assert m["suction_velocity_m_s"] == pytest.approx(0.35 / 0.14 ** 2)
    assert not [msg for msg in res.messages if "differs" in msg]
    (tmp_path / "c/postProcessing/suctionFlow/0/surfaceFieldValue.dat").write_text("# x\n1\t0.2\n2\t0.2\n")
    assert [msg for msg in aeromant.read_case_results(tmp_path / "c").messages if "differs" in msg]
