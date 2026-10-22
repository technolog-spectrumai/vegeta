"""rotor_mrf_installed: a propeller (rotating zone) next to a standing body — rendering in both dialects, the domain
growing to hold the body, parameter checks, separate propeller and body forces; one live tractor run."""
import math
import re

import numpy as np
import pytest

from vegeta.aeromant import CFDCase, OpenFOAMEnvironment, write_stl_ascii
from vegeta.aeromant.stl import Surface
from _rotor import flat_plate_propeller

COM = OpenFOAMEnvironment(env={"WM_PROJECT_VERSION": "v2412"})
ORG = OpenFOAMEnvironment(env={"WM_PROJECT_VERSION": "14"})
PARAMS = dict(rpm=6000.0, airspeed=10.0, diameter=0.127, kinematic_viscosity=1.5e-5, density=1.2)


def box(lo, hi, name="pod"):
    """A closed, outward-wound box surface."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 4, 7, 3)]
    tris = [v[[a, b, c]] for a, b, c, d in quads] + [v[[a, c, d]] for a, b, c, d in quads]
    s = Surface(np.array(tris), name)
    assert s.volume > 0
    return s


@pytest.fixture(scope="module")
def stls(tmp_path_factory):
    d = tmp_path_factory.mktemp("installed")
    prop = write_stl_ascii(flat_plate_propeller(), d / "prop.stl")
    behind = write_stl_ascii(box((0.012, -0.012, -0.012), (0.35, 0.012, 0.012)), d / "pod_behind.stl")     # tractor
    ahead = write_stl_ascii(box((-0.35, -0.012, -0.012), (-0.012, 0.012, 0.012)), d / "pod_ahead.stl")     # pusher
    return prop, behind, ahead


@pytest.mark.parametrize("env", [COM, ORG], ids=["com", "org"])
def test_installed_renders_with_the_standing_body(tmp_path, stls, env):
    prop, behind, _ = stls
    case = CFDCase("rotor_mrf_installed", prop, PARAMS, tmp_path / "c", geometry_units="m", environment=env, static_geometry=behind)
    res = case.prepare()
    assert res.ok, res.messages
    geo = tmp_path / "c" / case.files.geometry_dir
    assert (geo / "body.stl").is_file() and (geo / "static.stl").is_file()
    for f in (tmp_path / "c").rglob("*"):
        if f.is_file() and f.parent.name not in ("triSurface", "geometry", "inputs"):
            assert not re.search(r"\{\{[A-Z0-9_]+\}\}", f.read_text()), f
    assert "nonRotatingPatches  (static)" in (tmp_path / "c/constant/MRFProperties").read_text()
    control = (tmp_path / "c/system/controlDict").read_text()
    assert "staticForces" in control and '"static.*"' in control and '"body.*"' in control
    snappy = (tmp_path / "c/system/snappyHexMeshDict").read_text()
    assert "static.eMesh" in snappy and "staticBox" in snappy
    assert '"(body|static).*"' in (tmp_path / "c/0.orig/U").read_text()
    feat = (tmp_path / "c/system" / ("surfaceFeatureExtractDict" if env is COM else "surfaceFeaturesDict")).read_text()
    assert "static.stl" in feat
    assert res.metrics["static_bbox_max_m"][0] == pytest.approx(0.35, abs=1e-9)


def test_domain_grows_to_hold_the_body(tmp_path, stls):
    prop, behind, ahead = stls
    D = PARAMS["diameter"]
    for name, pod in (("t", behind), ("p", ahead)):
        case = CFDCase("rotor_mrf_installed", prop, dict(PARAMS, upstream=1.0, downstream=1.0), tmp_path / name,
                       geometry_units="m", environment=COM, static_geometry=pod)
        assert case.prepare().ok
        info = __import__("json").loads((tmp_path / name / "aeromant_case.json").read_text())["derived"]
        xmin, xmax = float(info["XMIN"]), float(info["XMAX"])
        if name == "t":
            assert xmax >= 0.35 + 1.5 * D - 1e-9 and xmin == pytest.approx(min(-1.0 * D, 0.012 - 1.5 * D))
        else:
            assert xmin <= -0.35 - 1.5 * D + 1e-9 and xmax >= 1.0 * D - 1e-9


def test_installed_parameter_checks(tmp_path, stls):
    prop, behind, _ = stls
    with pytest.raises(ValueError, match="needs static_geometry"):
        CFDCase("rotor_mrf_installed", prop, PARAMS, tmp_path / "a", geometry_units="m", environment=COM)
    with pytest.raises(ValueError, match="no standing body"):
        CFDCase("rotor_mrf", prop, PARAMS, tmp_path / "b", geometry_units="m", environment=COM, static_geometry=behind)


def _fake(case_dir, fn, fx, mx, n=60):
    pp = case_dir / "postProcessing" / fn / "0"
    pp.mkdir(parents=True)
    (pp / "force.dat").write_text("# Time forces\n" + "".join(f"{i}\t({fx} 0 0)\t(0 0 0)\t(0 0 0)\n" for i in range(1, n + 1)))
    (pp / "moment.dat").write_text("# Time moments\n" + "".join(f"{i}\t({mx} 0 0)\t(0 0 0)\t(0 0 0)\n" for i in range(1, n + 1)))


def test_propeller_and_body_forces_are_separate(tmp_path, stls):
    prop, behind, _ = stls
    case = CFDCase("rotor_mrf_installed", prop, PARAMS, tmp_path / "s", geometry_units="m", environment=COM, static_geometry=behind)
    assert case.prepare().ok
    _fake(tmp_path / "s", "forces", fx=-2.0, mx=-0.05)
    _fake(tmp_path / "s", "staticForces", fx=0.3, mx=0.0)        # the slipstream drags the pod downstream
    m = case.results().metrics
    omega = 6000 * 2 * math.pi / 60
    assert m["thrust_N"] == pytest.approx(2.0) and m["torque_Nm"] == pytest.approx(0.05)
    assert m["body_drag_N"] == pytest.approx(0.3) and m["net_thrust_N"] == pytest.approx(1.7)
    assert m["efficiency"] == pytest.approx(2.0 * 10 / (0.05 * omega))
    assert m["net_efficiency"] == pytest.approx(1.7 * 10 / (0.05 * omega))


@pytest.mark.requires_openfoam
@pytest.mark.slow
def test_installed_tractor_live(tmp_path, openfoam, stls):
    """Flat-plate rotor ahead of a square pod: positive thrust and torque, the pod sees a drag."""
    prop, behind, _ = stls
    case = CFDCase("rotor_mrf_installed", prop, dict(PARAMS, rpm=5000.0, airspeed=5.0, iterations=200, cells_per_diameter=6.0,
                                                   surface_level=3, near_level=2, rotor_level=2, wake_level=1, static_level=2,
                                                   upstream=2.0, downstream=3.0, lateral=2.0),
                   tmp_path / "live", geometry_units="m", environment=openfoam, static_geometry=behind)
    assert case.prepare().ok
    res = case.run()
    assert res.ok, res.messages
    m = res.metrics
    assert m["thrust_N"] > 0 and m["torque_Nm"] > 0, m
    assert m["body_drag_N"] > 0 and m["net_thrust_N"] == pytest.approx(m["thrust_N"] - m["body_drag_N"])
