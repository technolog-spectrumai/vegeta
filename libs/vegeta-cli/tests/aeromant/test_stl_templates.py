import json
import math
import struct

import numpy as np
import pytest

from vegeta import aeromant
from vegeta.aeromant import CFDCase, get_template, read_stl, write_stl_ascii
from vegeta.aeromant.stl import Surface

REQ = dict(velocity=1.0, kinematic_viscosity=0.01, density=1.0, reference_area=math.pi / 4,
           reference_length=1.0, center_of_rotation=(0, 0, 0))


def test_stl_ascii_and_binary_round_trip(tmp_path, sphere_stl):
    s = read_stl(sphere_stl)
    assert s.triangles.shape[1:] == (3, 3)
    assert s.volume == pytest.approx(4 / 3 * math.pi * 0.125, rel=0.02)
    lo, hi = s.bbox
    assert lo == pytest.approx([-0.5] * 3, abs=1e-6) and hi == pytest.approx([0.5] * 3, abs=1e-6)
    tri = s.triangles.astype(np.float32)
    rec = b"".join(struct.pack("<12fH", *np.zeros(3), *t.ravel(), 0) for t in tri)
    binary = tmp_path / "b.stl"
    binary.write_bytes(b"\0" * 80 + struct.pack("<I", len(tri)) + rec)
    assert read_stl(binary).triangles == pytest.approx(s.triangles, abs=1e-6)
    assert s.scaled(1e-3).area == pytest.approx(s.area * 1e-6)


def test_bad_stl(tmp_path):
    p = tmp_path / "x.stl"
    p.write_text("hello")
    with pytest.raises(ValueError):
        read_stl(p)


def test_template_requires_explicit_values():
    t = get_template("laminar_external")
    with pytest.raises(ValueError, match="requires explicit"):
        t.resolve({"velocity": 1.0})
    with pytest.raises(ValueError, match="no parameter"):
        t.resolve(dict(REQ, turbulence_model="guess"))
    with pytest.raises(ValueError, match="> 0"):
        t.resolve(dict(REQ, velocity=-1.0))
    with pytest.raises(ValueError, match="3 components"):
        t.resolve(dict(REQ, center_of_rotation=(0, 0)))
    p = t.resolve(REQ)
    assert p["iterations"] == 400 and p["center_of_rotation"] == [0.0, 0.0, 0.0]
    assert "REQUIRED" in t.describe()
    with pytest.raises(ValueError, match="unknown template"):
        get_template("magic")


def test_prepare_renders_every_placeholder(tmp_path, sphere_stl):
    case = CFDCase("rans_ksst_external", sphere_stl, REQ, tmp_path / "c", geometry_units="m")
    res = case.prepare()
    assert res.ok, res.messages
    assert case.is_prepared
    for f in (tmp_path / "c").rglob("*"):
        if f.is_file() and f.suffix != ".stl" and f.name != "aeromant_case.json":
            assert "{{" not in f.read_text().replace("{{...}}", ""), f
    ctrl = (tmp_path / "c/system/controlDict").read_text()
    assert "magUInf         1;" in ctrl and "Aref            0.785398163;" in ctrl
    assert (tmp_path / "c/constant/triSurface/body.stl").read_text().startswith("solid body")
    assert (tmp_path / "c/inputs/sphere.stl").is_file()
    assert res.metrics["reynolds_number"] == pytest.approx(100)
    k = 1.5 * (1.0 * 0.005) ** 2
    assert f"uniform {k:.9g}" in (tmp_path / "c/0.orig/k").read_text()


def test_prepare_scales_mm_and_checks_body_size(tmp_path):
    from _sphere import icosphere

    stl = write_stl_ascii(icosphere(500.0, 1), tmp_path / "big.stl")  # 1 m sphere drawn in mm
    ok = CFDCase("laminar_external", stl, REQ, tmp_path / "mm", geometry_units="mm").prepare()
    assert ok.ok and ok.metrics["body_bbox_max_m"][0] == pytest.approx(0.5, rel=1e-3)
    bad = CFDCase("laminar_external", stl, REQ, tmp_path / "m", geometry_units="m").prepare()
    assert not bad.ok and "geometry_units" in bad.messages[0]
    with pytest.raises(ValueError, match="geometry_units"):
        CFDCase("laminar_external", stl, REQ, tmp_path / "x", geometry_units="furlong")


def test_prepare_refuses_non_empty_dir(tmp_path, sphere_stl):
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "keep.txt").write_text("mine")
    res = CFDCase("laminar_external", sphere_stl, REQ, tmp_path / "c", geometry_units="m").prepare()
    assert not res.ok and "not empty" in res.messages[0]
    assert (tmp_path / "c" / "keep.txt").is_file()


def test_run_requires_prepare_and_known_steps(tmp_path, sphere_stl):
    case = CFDCase("laminar_external", sphere_stl, REQ, tmp_path / "c", geometry_units="m")
    assert "prepare() first" in case.run().messages[0]
    case.prepare()
    assert "unknown step" in case.run(steps=["magicFoam"]).messages[0]
    changed = CFDCase("laminar_external", sphere_stl, dict(REQ, velocity=2.0), tmp_path / "c",
                      geometry_units="m")
    assert not changed.is_prepared


def test_missing_openfoam_is_failed_result(tmp_path, sphere_stl):
    env = aeromant.OpenFOAMEnvironment(env={"PATH": str(tmp_path)})
    case = CFDCase("laminar_external", sphere_stl, REQ, tmp_path / "c", geometry_units="m",
                   environment=env)
    case.prepare()
    res = case.run(steps=["blockMesh"])
    assert res.status == "failed" and "not found" in res.messages[0]
    assert res.execution[0].returncode is None


def test_openfoam_flavor_detection(tmp_path):
    from vegeta.aeromant.environment import flavor_of

    assert flavor_of("v2412") == "openfoam.com" and flavor_of("v1912") == "openfoam.com"
    assert flavor_of("14") == "openfoam.org" and flavor_of("14-7b05503f98a8") == "openfoam.org"
    assert flavor_of("dev") == "openfoam.org" and flavor_of(None) is None and flavor_of("weird") is None
    rc = tmp_path / "bashrc"
    rc.write_text("# fake\nexport WM_PROJECT_VERSION=14\nexport FOAM_INST_DIR=/opt\n")
    env = aeromant.OpenFOAMEnvironment(bashrc=str(rc))
    assert env.version() == "14" and env.flavor() == "openfoam.org"
    assert aeromant.OpenFOAMEnvironment(env={"PATH": str(tmp_path)}).flavor() is None


def test_template_aliases_and_flavors():
    t = get_template("laminar_external_simplefoam")  # old name still works
    assert t.name == "laminar_external" and t.flavor_names == ["openfoam.com", "openfoam.org"]
    assert t.step_names("openfoam.com")[1] == "features" and t.step_names("openfoam.org")[-1] == "solver"
    assert t.flavor("openfoam.org").pipeline[-1].argv == ("foamRun",)
    with pytest.raises(ValueError, match="no case files"):
        t.flavor("openfoam.net")
    assert "openfoam.org (12 and later)" in t.describe()


def test_prepare_uses_case_files_of_the_detected_flavor(tmp_path, sphere_stl):
    rc = tmp_path / "bashrc"
    rc.write_text("export WM_PROJECT_VERSION=14\n")
    org = aeromant.OpenFOAMEnvironment(bashrc=str(rc))
    case = CFDCase("rans_ksst_external", sphere_stl, REQ, tmp_path / "org", geometry_units="m", environment=org)
    assert case.flavor == "openfoam.org"
    res = case.prepare()
    assert res.ok, res.messages
    c = tmp_path / "org"
    assert (c / "constant/geometry/body.stl").is_file() and not (c / "constant/triSurface").exists()
    assert (c / "constant/physicalProperties").is_file() and (c / "constant/momentumTransport").is_file()
    assert "solver          incompressibleFluid;" in (c / "system/controlDict").read_text()
    assert "type triSurface;" in (c / "system/snappyHexMeshDict").read_text()
    assert 'insidePoint (' in (c / "system/snappyHexMeshDict").read_text()
    assert "{{" not in (c / "system/snappyHexMeshDict").read_text()
    assert res.artifacts["body_stl"] == c / "constant/geometry/body.stl"
    info = json.loads((c / "aeromant_case.json").read_text())
    assert info["config"]["openfoam_flavor"] == "openfoam.org" and info["config"]["openfoam_version"] == "14"
    # the openfoam.com case files are used by default when nothing is detected
    com = CFDCase("rans_ksst_external", sphere_stl, REQ, tmp_path / "com", geometry_units="m",
                  environment=aeromant.OpenFOAMEnvironment(env={"PATH": str(tmp_path)}))
    assert com.flavor == "openfoam.com" and com.prepare().ok
    assert (tmp_path / "com/constant/triSurface/body.stl").is_file()
    assert (tmp_path / "com/constant/transportProperties").is_file()


def test_setup_plot_from_prepared_case(tmp_path, sphere_stl):
    pv = pytest.importorskip("pyvista")
    pv.OFF_SCREEN = True
    from vegeta.aeromant import viz

    case = CFDCase("laminar_external", sphere_stl, REQ, tmp_path / "c", geometry_units="m")
    assert case.prepare().ok
    pl = viz.plot_setup(case)
    assert pl.screenshot(str(tmp_path / "s.png")).shape[0] > 0
    pl.close()
