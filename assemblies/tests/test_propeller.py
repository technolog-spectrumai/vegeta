"""The propeller component against the notebooks it came from: each catalogue entry equals the notebook's own
``from_pitch`` call and CAD arguments, the rotor-disk polar equals scenarios/run_scenario.py's, the rotor STL is
reproducible (stable CFD case keys). The notebook calls are written out literally here, as they are in the notebooks."""
import hashlib
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
from vegeta import aeromant, boreas

from assemblies.components import propeller as pr

REPO = Path(__file__).resolve().parents[2]


def _same(a: boreas.Propeller, b: boreas.Propeller):
    assert a.name == b.name and a.blades == b.blades and a.mass_kg == b.mass_kg and a.rotor_mass_kg == b.rotor_mass_kg
    np.testing.assert_allclose(a.r, b.r, rtol=1e-12)
    np.testing.assert_allclose(a.chord, b.chord, rtol=1e-12)
    np.testing.assert_allclose(a.beta_deg, b.beta_deg, rtol=1e-12)


def test_quadcopter_propeller_is_notebook_08s():
    d, p = boreas.inches(5, 4.3)
    nb = boreas.Propeller.from_pitch("5x4.3 tri-blade", d, p, blades=3, chord_root_m=0.010, chord_max_m=0.016,
                                     chord_tip_m=0.006, mass_kg=0.0045, rotor_mass_kg=0.020,
                                     notes="generic planform; fit chord/beta to the real propeller for better numbers")
    s = pr.get("5x4.3 tri-blade")
    _same(s.model(), nb)
    assert s.cad_kw() == dict(diameter=d * 1000, pitch=p * 1000, blades=3, hub_diameter=12, hub_height=7, bore=5,
                              chord_root=10, chord_max=16, chord_tip=6, thickness=0.10, camber=0.05)
    assert s.airfoil() == boreas.Airfoil(name="thin cambered section", cl_alpha=2 * math.pi * 0.9, alpha0_deg=-3.0, cl_max=1.1,
                                         cd0=0.025, k=0.045, source="assumed for a moulded 5-inch blade at Re ~ 1e5")


def test_fixed_wing_propeller_is_notebook_09as():
    d, p = boreas.inches(9, 6)
    nb = boreas.Propeller.from_pitch("9x6 electric", d, p, blades=2, chord_root_m=0.014, chord_max_m=0.022, chord_tip_m=0.006,
                                     mass_kg=0.012, rotor_mass_kg=0.045, notes="generic planform")
    s = pr.get("9x6 electric")
    _same(s.model(), nb)
    assert s.cad_kw(1)["blades"] == 1 and s.cad_kw()["chord_max"] == 22 and s.airfoil().alpha0_deg == -2.5


def test_notebook_25_blade_with_both_blade_counts():
    s = pr.get("10x6, 2 blades")
    for B in (2, 3):
        nb = boreas.Propeller.from_pitch(f"10x6, {B} blades", 0.254, 0.1524, blades=B, chord_root_m=0.018,
                                         chord_max_m=0.026, chord_tip_m=0.010, hub_radius_m=0.010)
        _same(s.with_blades(B).model(), nb)
    assert s.with_blades(3).cad_kw() == dict(diameter=254.0, pitch=152.4, blades=3, chord_root=18.0, chord_max=26.0,
                                             chord_tip=10.0, thickness=0.10, camber=0.04, stations=8, hub_diameter=20.0,
                                             hub_height=10.0, bore=5.0)


@pytest.mark.parametrize("name,args,cad_root", [
    ("60 mm 3-blade marine", (0.060, 0.050, 0.012, 0.018, 0.008, 0.02, 0.05, "generic planform; fit to the real propeller"), 10.0),
    ("120 mm 3-blade", (0.120, 0.100, 0.018, 0.030, 0.012, 0.06, 0.15, "generic planform"), 18.0)])
def test_marine_propellers_are_notebooks_12_and_13s(name, args, cad_root):
    d, p, c0, c1, c2, m, mr, notes = args
    nb = boreas.Propeller.from_pitch(name, d, p, blades=3, chord_root_m=c0, chord_max_m=c1, chord_tip_m=c2, mass_kg=m,
                                     rotor_mass_kg=mr, notes=notes)
    s = pr.get(name)
    _same(s.model(), nb)
    assert s.cad_kw()["chord_root"] == cad_root                          # 12's CAD root chord differs from its BEMT one
    assert s.airfoil() == boreas.Airfoil(name="marine blade section", cl_alpha=5.5, alpha0_deg=-2.0, cl_max=1.0, cd0=0.02,
                                         k=0.05, source="assumed")


def test_spec_round_trip_and_unknown_name():
    s = pr.get("9x6 electric")
    assert pr.PropellerSpec.from_dict(s.to_dict()) == s
    from assemblies.vida import Assembly
    Assembly("p", "propeller", params={"spec": s.to_dict()})            # plain enough to be a node parameter
    with pytest.raises(KeyError, match="catalogue has"):
        pr.get("11x7")


def test_polar_table_is_run_scenarios():
    spec = importlib.util.spec_from_file_location("run_scenario", REPO / "scenarios" / "run_scenario.py")
    rs = importlib.util.module_from_spec(spec)
    import sys
    sys.modules["run_scenario"] = rs                                    # it defines dataclasses
    try:
        spec.loader.exec_module(rs)
    finally:
        sys.modules.pop("run_scenario", None)
    for section in ("electric 9 in", "marine"):
        assert pr.polar_table(pr.airfoil(section)) == rs._polar_table(pr.airfoil(section))
    prop = pr.get("9x6 electric").model()
    assert pr.blade_table(prop) == [[r, b, c] for r, b, c in zip(prop.r, prop.beta_deg, prop.chord)]


def test_rotor_params_follow_the_notebooks():
    s = pr.get("60 mm 3-blade marine")
    p = pr.rotor_params(s, 2184.0, airspeed=1.35, medium="sea water", fidelity="quick")
    assert p == dict(rpm=2184.0, diameter=0.060, kinematic_viscosity=1.05e-6, density=1025.0, rotation=1, iterations=400,
                     cells_per_diameter=6.0, surface_level=3, near_level=2, rotor_level=2, wake_level=1, airspeed=1.35)
    with pytest.raises(ValueError, match="fidelity"):
        pr.rotor_params(s, 1000.0, fidelity="huge")


def test_blade_loads():
    s = pr.get("5x4.3 tri-blade")
    prop, sec = s.model(), s.airfoil()
    op = boreas.solve(prop, sec, 20000.0, 0.0)
    loads = pr.blade_loads(prop, op)
    assert loads["thrust_N"] == pytest.approx(op.thrust / 3)
    assert loads["tangential_N"] == pytest.approx(op.torque / (3 * 0.7 * prop.radius))


def test_cad_files_kept_and_reproducible(tmp_path):
    pytest.importorskip("cadquery")
    s = pr.get("5x4.3 tri-blade")
    a = pr.cad_files(s, tmp_path / "a")
    step_time = a["step"].stat().st_mtime_ns
    assert pr.cad_files(s, tmp_path / "a")["step"].stat().st_mtime_ns == step_time          # kept, not rewritten
    b = pr.cad_files(s, tmp_path / "b")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()                              # noqa: E731
    assert sha(a["stl_axis_x"]) == sha(b["stl_axis_x"])                                    # same STL -> same CFD key
    env = aeromant.OpenFOAMEnvironment()
    params = pr.rotor_params(s, 20000.0, fidelity="smoke")
    ka = pr.rotor_case(a["stl_axis_x"], params, tmp_path / "ca", env)
    assert ka.template.name == "rotor_mrf_static"
    lo, hi = aeromant.read_stl(a["stl_axis_x"]).bbox
    assert hi[0] - lo[0] < 15 and hi[1] - lo[1] > 100                                       # thin along x: the axis is +x
    one = pr.cad_files(s, tmp_path / "blade", blades=1)
    model = pr.blade_model(s, one["step"], {"thrust_N": 2.0, "tangential_N": 0.5}, element_mm=0.8)
    assert [r.name for r in model.regions] == ["hub", "blade"] and model.loads[0].fz == 2.0
