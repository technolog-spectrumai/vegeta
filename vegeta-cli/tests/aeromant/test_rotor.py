"""Rotor (MRF) templates: rendering in both dialects, parameter checks, forces parsing, metrics; one live run."""
import math
import re

import numpy as np
import pytest

from vegeta import aeromant
from vegeta.aeromant import CFDCase, OpenFOAMEnvironment, write_stl_ascii
from vegeta.aeromant.results import read_force_history
from _rotor import flat_plate_propeller

COM = OpenFOAMEnvironment(env={"WM_PROJECT_VERSION": "v2412"})
ORG = OpenFOAMEnvironment(env={"WM_PROJECT_VERSION": "14"})
PARAMS = dict(rpm=6000.0, diameter=0.127, kinematic_viscosity=1.5e-5, density=1.2)


@pytest.fixture(scope="module")
def prop_stl(tmp_path_factory):
    surf = flat_plate_propeller()
    assert surf.volume > 0
    return write_stl_ascii(surf, tmp_path_factory.mktemp("rotor") / "prop.stl")


@pytest.mark.parametrize("template,extra", [("rotor_mrf", {"airspeed": 8.0}), ("rotor_mrf_static", {})])
@pytest.mark.parametrize("env", [COM, ORG], ids=["com", "org"])
def test_rotor_templates_render(tmp_path, prop_stl, template, extra, env):
    case = CFDCase(template, prop_stl, dict(PARAMS, rotation=-1, **extra), tmp_path / "c", geometry_units="m", environment=env)
    res = case.prepare()
    assert res.ok, res.messages
    assert case.flavor == env.flavor() and case.template.step_names(case.flavor)[-1] == "solver"
    mrf = (tmp_path / "c/constant/MRFProperties").read_text()
    assert "cellZone            rotorZone;" in mrf
    omega = float([l.split()[1].rstrip(";") for l in mrf.splitlines() if l.strip().startswith("omega")][0])
    assert omega == pytest.approx(-6000 * 2 * math.pi / 60)
    snappy = (tmp_path / "c/system/snappyHexMeshDict").read_text()
    assert "cellZone rotorZone" in snappy and not re.search(r"\{\{[A-Z0-9_]+\}\}", snappy)
    assert "forces" in (tmp_path / "c/system/controlDict").read_text()
    u = (tmp_path / "c/0.orig/U").read_text()
    assert "fixedValue" in u
    if template == "rotor_mrf_static":                       # residual inflow: 2 % of the tip speed
        tip = 6000 * 2 * math.pi / 60 * 0.127 / 2
        assert f"uniform ({0.02 * tip:.9g} 0 0)" in u or f"({0.02 * tip:.6g}" in u
    assert res.metrics["reynolds_number"] > 1e4 and len(res.metrics["background_cells"]) == 3


def test_rotor_parameter_checks(tmp_path, prop_stl):
    with pytest.raises(ValueError, match="requires explicit"):
        CFDCase("rotor_mrf", prop_stl, PARAMS, tmp_path / "a", geometry_units="m", environment=COM)
    with pytest.raises(ValueError, match="no parameter"):
        CFDCase("rotor_mrf_static", prop_stl, dict(PARAMS, airspeed=1.0), tmp_path / "b", geometry_units="m", environment=COM)
    with pytest.raises(ValueError, match="rpm must be > 0"):
        CFDCase("rotor_mrf_static", prop_stl, dict(PARAMS, rpm=-10), tmp_path / "c", geometry_units="m", environment=COM)
    bad = CFDCase("rotor_mrf_static", prop_stl, dict(PARAMS, rotation=2.0), tmp_path / "d", geometry_units="m", environment=COM)
    assert not bad.prepare().ok and "rotation" in bad.prepare(overwrite=True).messages[0]
    zero = CFDCase("rotor_mrf", prop_stl, dict(PARAMS, airspeed=0.0), tmp_path / "e", geometry_units="m", environment=COM)
    assert "airspeed must be > 0" in zero.prepare().messages[0]
    off = CFDCase("rotor_mrf_static", prop_stl, dict(PARAMS, center=(0.5, 0, 0)), tmp_path / "f", geometry_units="m", environment=COM)
    assert "not inside the STL bounding box" in off.prepare().messages[0]


def test_read_force_history_both_dialects(tmp_path):
    com = tmp_path / "force_com.dat"
    com.write_text("# Force\n# CofR : (0 0 0)\n# Time forces(pressure viscous porous)\n"
                   "1\t(-2.0 0.1 0.2)\t(-1.5 0.05 0.1)\t(-0.5 0.05 0.1)\t(0 0 0)\n2\t(-2.2 0.0 0.0)\t(-1.6 0 0)\t(-0.6 0 0)\t(0 0 0)\n")
    org = tmp_path / "force_org.dat"
    org.write_text("# Forces\n# Time  total_x total_y total_z  pressure_x pressure_y pressure_z  viscous_x viscous_y viscous_z\n"
                   "1 -2.0 0.1 0.2 -1.5 0.05 0.1 -0.5 0.05 0.1\n2 -2.2 0 0 -1.6 0 0 -0.6 0 0\n")
    for f in (com, org):
        h = read_force_history(f)
        assert h.shape == (2, 4) and h[0].tolist() == [1.0, -2.0, 0.1, 0.2] and h[1, 1] == -2.2
    empty = tmp_path / "empty.dat"
    empty.write_text("# only comments\n")
    with pytest.raises(ValueError, match="no force data"):
        read_force_history(empty)


def _fake_forces(case_dir, fx, mx, n=60):
    pp = case_dir / "postProcessing" / "forces" / "0"
    pp.mkdir(parents=True)
    (pp / "force.dat").write_text("# Time forces\n" + "".join(f"{i}\t({fx} 0.01 -0.02)\t(0 0 0)\t(0 0 0)\n" for i in range(1, n + 1)))
    (pp / "moment.dat").write_text("# Time moments\n" + "".join(f"{i}\t({mx} 0.001 0.0)\t(0 0 0)\t(0 0 0)\n" for i in range(1, n + 1)))


def test_rotor_metrics_from_forces(tmp_path, prop_stl):
    case = CFDCase("rotor_mrf_static", prop_stl, PARAMS, tmp_path / "s", geometry_units="m", environment=COM)
    assert case.prepare().ok
    assert not case.results().ok                           # solver NOT RUN
    _fake_forces(tmp_path / "s", fx=-2.0, mx=-0.05)         # fluid pushes the rotor to -x, resists rotation about +x
    m = case.results().metrics
    omega = 6000 * 2 * math.pi / 60
    assert m["thrust_N"] == pytest.approx(2.0) and m["torque_Nm"] == pytest.approx(0.05)
    assert m["power_W"] == pytest.approx(0.05 * omega) and m["averaging_window"] == 50
    assert m["ct"] == pytest.approx(2.0 / (1.2 * 100 ** 2 * 0.127 ** 4))
    ideal = 2.0 * math.sqrt(2.0 / (2 * 1.2 * math.pi * 0.0635 ** 2))
    assert m["figure_of_merit"] == pytest.approx(ideal / (0.05 * omega))
    axial = CFDCase("rotor_mrf", prop_stl, dict(PARAMS, airspeed=10.0, rotation=-1), tmp_path / "a", geometry_units="m", environment=COM)
    assert axial.prepare().ok
    _fake_forces(tmp_path / "a", fx=-2.0, mx=0.05)          # reversed rotation: the resisting moment is +x
    m = axial.results().metrics
    assert m["torque_Nm"] == pytest.approx(0.05) and m["efficiency"] == pytest.approx(2.0 * 10 / (0.05 * omega))
    assert m["advance_ratio"] == pytest.approx(10 / (100 * 0.127))
    wrong = CFDCase("rotor_mrf_static", prop_stl, PARAMS, tmp_path / "w", geometry_units="m", environment=COM)
    wrong.prepare()
    _fake_forces(tmp_path / "w", fx=+1.0, mx=-0.05)
    r = wrong.results()
    assert r.metrics["thrust_N"] == pytest.approx(-1.0) and any("negative thrust" in x for x in r.messages)


@pytest.mark.requires_openfoam
@pytest.mark.slow
def test_static_rotor_live(tmp_path, openfoam, prop_stl):
    """Flat-plate two-blade rotor at 3000 rpm: positive thrust and torque with rotation=1, plausible figure of merit."""
    case = CFDCase("rotor_mrf_static", prop_stl, dict(PARAMS, rpm=3000.0, iterations=250, cells_per_diameter=6.0,
                                                    surface_level=3, near_level=2, rotor_level=2, wake_level=1),
                   tmp_path / "live", geometry_units="m", environment=openfoam)
    assert case.prepare().ok
    res = case.run()
    assert res.ok, res.messages
    m = res.metrics
    assert m["mesh_cells"] > 20000
    assert m["thrust_N"] > 0 and m["torque_Nm"] > 0, m
    assert 0.1 < m["figure_of_merit"] < 1.0
    assert m["power_W"] == pytest.approx(m["torque_Nm"] * 3000 * 2 * math.pi / 60)
    assert case.results().metrics["thrust_N"] == pytest.approx(m["thrust_N"])


def test_every_template_field_file_declares_dimensions():
    from vegeta.aeromant.templates import TEMPLATE_ROOT, TEMPLATES

    for name, spec in TEMPLATES.items():
        for fl in spec.flavors:
            for f in (TEMPLATE_ROOT / name / fl.directory / "0.orig").iterdir():
                text = f.read_text()
                assert "dimensions" in text and "internalField" in text and "boundaryField" in text, f"{name}/{fl.directory}/0.orig/{f.name}"
