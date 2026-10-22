"""The compressible templates (openfoam.com, rhoSimpleFoam): jet_external (an aircraft with a running jet engine) and
compressor_mrf (a radial impeller in its casing) — rendering with the named surfaces, parameter checks, and results read
from synthetic postProcessing output (no OpenFOAM needed)."""
import json
import math
import re

import numpy as np
import pytest

from vegeta.aeromant import CFDCase, OpenFOAMEnvironment, write_stl_ascii
from vegeta.aeromant.stl import Surface

COM = OpenFOAMEnvironment(env={"WM_PROJECT_VERSION": "v2412"})
ORG = OpenFOAMEnvironment(env={"WM_PROJECT_VERSION": "12"})


def box(lo, hi, name):
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 4, 7, 3)]
    return Surface(np.array([v[[a, b, c]] for a, b, c, d in quads] + [v[[a, c, d]] for a, b, c, d in quads]), name)


def square_x(x, half, name, c=(0.0, 0.0)):
    y0, z0 = c
    v = np.array([[x, y0 - half, z0 - half], [x, y0 + half, z0 - half], [x, y0 + half, z0 + half], [x, y0 - half, z0 + half]])
    return Surface(np.array([v[[0, 1, 2]], v[[0, 2, 3]]]), name)


def no_placeholders(case_dir):
    for f in case_dir.rglob("*"):
        if f.is_file() and f.parent.name not in ("triSurface", "geometry", "inputs"):
            assert not re.search(r"\{\{[A-Z0-9_]+\}\}", f.read_text()), f


# -- jet_external -----------------------------------------------------------------------------------------------
JET = dict(velocity=150.0, pressure=101325.0, temperature=288.15, reference_area=0.3, reference_length=0.35,
           center_of_rotation=[0.3, 0, 0], intake_mass_flow=0.29, exhaust_mass_flow=0.297, exhaust_temperature=860.0)


@pytest.fixture(scope="module")
def jet_stls(tmp_path_factory):
    d = tmp_path_factory.mktemp("jet")
    body = write_stl_ascii(box((0.0, -0.06, -0.06), (1.2, 0.06, 0.06), "body"), d / "body.stl")
    intake = write_stl_ascii(square_x(0.4, 0.03, "intake"), d / "intake.stl")
    exhaust = write_stl_ascii(square_x(1.2, 0.025, "exhaust"), d / "exhaust.stl")
    return body, {"intake": intake, "exhaust": exhaust}


def test_jet_renders_with_its_engine_faces(tmp_path, jet_stls):
    body, surfaces = jet_stls
    case = CFDCase("jet_external", body, JET, tmp_path / "c", geometry_units="m", environment=COM, surfaces=surfaces)
    res = case.prepare()
    assert res.ok, res.messages
    c = tmp_path / "c"
    no_placeholders(c)
    for n in ("body", "intake", "exhaust"):
        assert (c / "constant/triSurface" / f"{n}.stl").is_file()
        assert f"{n}.eMesh" in (c / "system/snappyHexMeshDict").read_text()
    assert "rhoSimpleFoam" in (c / "system/controlDict").read_text()
    u = (c / "0.orig/U").read_text()
    assert "massFlowRate    0.29;" in u and "massFlowRate    0.297;" in u
    assert "uniform 860" in (c / "0.orig/T").read_text()
    control = (c / "system/controlDict").read_text()
    rho = 101325.0 / (287.05 * 288.15)
    assert f"rhoInf          {rho:.9g};" in control and "pRef            101325;" in control
    derived = json.loads((c / "aeromant_case.json").read_text())["derived"]
    start = [float(v) for v in derived["PLUME_START"].split()]
    assert start[0] > 1.2 and abs(start[1]) < 1e-9
    assert res.metrics["exhaust_area_m2"] == pytest.approx(0.05 ** 2)
    d_jet = math.sqrt(4 * 0.05 ** 2 / math.pi)
    assert float(derived["JET_MIXING_LENGTH"]) == pytest.approx(0.07 * d_jet)


def test_jet_parameter_and_surface_checks(tmp_path, jet_stls):
    body, surfaces = jet_stls
    with pytest.raises(ValueError, match="needs the surface"):
        CFDCase("jet_external", body, JET, tmp_path / "a", geometry_units="m", environment=COM, surfaces={"intake": surfaces["intake"]})
    with pytest.raises(ValueError, match="has no surface"):
        CFDCase("rans_ksst_external", body, dict(velocity=10, kinematic_viscosity=1.5e-5, density=1.2, reference_area=1,
                reference_length=0.3, center_of_rotation=[0, 0, 0]), tmp_path / "b", geometry_units="m", environment=COM,
                surfaces=surfaces)
    with pytest.raises(ValueError, match="openfoam.org"):
        CFDCase("jet_external", body, JET, tmp_path / "c", geometry_units="m", environment=ORG, surfaces=surfaces)
    bad = CFDCase("jet_external", body, dict(JET, exhaust_mass_flow=0.1), tmp_path / "d", geometry_units="m",
                  environment=COM, surfaces=surfaces).prepare()
    assert not bad.ok and "exhaust_mass_flow" in " ".join(bad.messages)


def _sfv(case, name, rows):
    d = case / "postProcessing" / name / "0"
    d.mkdir(parents=True)
    (d / "surfaceFieldValue.dat").write_text("# Time value\n" + "\n".join(" ".join(f"{v:g}" for v in r) for r in rows) + "\n")


def test_jet_results_from_postprocessing(tmp_path, jet_stls):
    body, surfaces = jet_stls
    case = CFDCase("jet_external", body, JET, tmp_path / "c", geometry_units="m", environment=COM, surfaces=surfaces)
    assert case.prepare().ok
    c = tmp_path / "c"
    fc = c / "postProcessing/forceCoeffs/0"
    fc.mkdir(parents=True)
    (fc / "coefficient.dat").write_text("# Time Cd Cs Cl CmRoll CmPitch CmYaw\n" + "\n".join(
        f"{i} 0.03 0 0.2 0 -0.01 0" for i in range(1, 101)) + "\n")
    _sfv(c, "intakeFlow", [(i, -0.29) for i in range(1, 51)])
    _sfv(c, "exhaustFlow", [(i, -0.297) for i in range(1, 51)])
    s = c / "postProcessing/plumeLine/1500"
    s.mkdir(parents=True)
    x = np.linspace(1.22, 2.6, 50)
    T = 288.15 + 570 * np.exp(-(x - 1.22) / 0.4)
    (s / "plume.csv").write_text("x,T,exhaust,p\n" + "\n".join(f"{a},{b},{(b - 288.15) / 571.85},101325" for a, b in zip(x, T)))
    res = case.results()
    m = res.metrics
    assert res.ok, res.messages
    assert m["Cd"] == pytest.approx(0.03) and m["mach_number"] == pytest.approx(150 / math.sqrt(1.4 * 287.05 * 288.15))
    q = 0.5 * 101325.0 / (287.05 * 288.15) * 150 ** 2
    assert m["drag_force_N"] == pytest.approx(0.03 * q * 0.3)
    assert m["intake_mass_flow_kg_s"] == pytest.approx(0.29) and m["exhaust_mass_flow_kg_s"] == pytest.approx(0.297)
    assert m["jet_excess_T_at_1m_K"] == pytest.approx(570 * math.exp(-1 / 0.4), rel=0.02)
    assert m["exhaust_fraction_at_0.5m"] > m["exhaust_fraction_at_1m"] > 0
    assert len(m["plume"]["distance_m"]) == 50


# -- compressor_mrf -----------------------------------------------------------------------------------------------
D = 0.068
COMP = dict(rpm=125000.0, diameter=D, inlet_total_pressure=101325.0, inlet_total_temperature=288.15,
            outlet_pressure=180000.0, location_in_mesh=[-0.05, 0.02, 0.0])


@pytest.fixture(scope="module")
def comp_stls(tmp_path_factory):
    d = tmp_path_factory.mktemp("comp")
    impeller = write_stl_ascii(box((-0.02, -D / 2, -D / 2), (0.01, D / 2, D / 2), "impeller"), d / "impeller.stl")
    shroud = write_stl_ascii(box((-0.08, -0.06, -0.06), (0.03, 0.06, 0.06), "shroud"), d / "shroud.stl")
    inlet = write_stl_ascii(square_x(-0.08, 0.03, "inlet"), d / "inlet.stl")
    outlet = write_stl_ascii(square_x(0.03, 0.05, "outlet"), d / "outlet.stl")
    return impeller, {"shroud": shroud, "inlet": inlet, "outlet": outlet}


def test_compressor_renders(tmp_path, comp_stls):
    imp, surfaces = comp_stls
    case = CFDCase("compressor_mrf", imp, COMP, tmp_path / "c", geometry_units="m", environment=COM, surfaces=surfaces)
    res = case.prepare()
    assert res.ok, res.messages
    c = tmp_path / "c"
    no_placeholders(c)
    mrf = (c / "constant/MRFProperties").read_text()
    omega = 125000 * 2 * math.pi / 60
    assert f"omega               {omega:.9g};" in mrf and '"shroud.*"' in mrf
    assert "uniform 180000" in (c / "0.orig/p").read_text() and "totalPressure" in (c / "0.orig/p").read_text()
    assert "farfield" in (c / "system/blockMeshDict").read_text()
    assert "upwind" in (c / "system/fvSchemes").read_text()
    derived = json.loads((c / "aeromant_case.json").read_text())["derived"]
    assert float(derived["ZONE_RADIUS"]) == pytest.approx(D / 2 + 0.002 * D)
    assert res.metrics["reynolds_number"] == pytest.approx(omega * D / 2 * D / 1.5e-5)


def test_compressor_checks(tmp_path, comp_stls):
    imp, surfaces = comp_stls
    bad = CFDCase("compressor_mrf", imp, dict(COMP, location_in_mesh=[1.0, 0, 0]), tmp_path / "a", geometry_units="m",
                  environment=COM, surfaces=surfaces).prepare()
    assert not bad.ok and "location_in_mesh" in " ".join(bad.messages)
    with pytest.raises(ValueError, match="needs the surface"):
        CFDCase("compressor_mrf", imp, COMP, tmp_path / "b", geometry_units="m", environment=COM)


def test_compressor_results(tmp_path, comp_stls):
    imp, surfaces = comp_stls
    case = CFDCase("compressor_mrf", imp, COMP, tmp_path / "c", geometry_units="m", environment=COM, surfaces=surfaces)
    assert case.prepare().ok
    c = tmp_path / "c"
    mdot, cp, g = 0.29, 1005.0, 1.4
    t_in, u_in, p_in = 280.0, 120.0, 93000.0
    t_out, u_out, p_out = 430.0, 150.0, 330000.0
    _sfv(c, "inletFlow", [(i, -mdot) for i in range(1, 60)])
    _sfv(c, "outletFlow", [(i, mdot) for i in range(1, 60)])
    _sfv(c, "inletState", [(i, p_in, t_in, u_in) for i in range(1, 60)])
    _sfv(c, "outletState", [(i, p_out, t_out, u_out) for i in range(1, 60)])
    omega = 125000 * 2 * math.pi / 60
    t0_in, t0_out = t_in + u_in ** 2 / (2 * cp), t_out + u_out ** 2 / (2 * cp)
    power = mdot * cp * (t0_out - t0_in)
    fo = c / "postProcessing/forces/0"
    fo.mkdir(parents=True)
    (fo / "moment.dat").write_text("# Time total\n" + "\n".join(f"{i} ({-power / omega} 0 0)" for i in range(1, 60)))
    (fo / "force.dat").write_text("# Time total\n" + "\n".join(f"{i} (0 0 0)" for i in range(1, 60)))
    m = case.results().metrics
    p0_in, p0_out = p_in * (t0_in / t_in) ** 3.5, p_out * (t0_out / t_out) ** 3.5
    pr = p0_out / p0_in
    assert m["mass_flow_kg_s"] == pytest.approx(mdot) and m["pressure_ratio_tt"] == pytest.approx(pr)
    eta = (pr ** (1 / 3.5) - 1) / (t0_out / t0_in - 1)
    assert m["efficiency_tt"] == pytest.approx(eta) and m["efficiency_from_torque"] == pytest.approx(eta, rel=1e-6)
    assert m["shaft_power_W"] == pytest.approx(power) and m["work_coefficient"] == pytest.approx(power / mdot / (omega * D / 2) ** 2)
