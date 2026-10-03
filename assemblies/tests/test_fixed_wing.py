"""The fixed-wing workflow (notebook 09a): the drive equals the notebook's cells run literally, the rotor-disk case
equals scenarios/run_scenario.py's air preset, the hand-off has the keys notebook 09b reads; tree and reuse with the
solvers off or faked; the real FEA is slow."""
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cadquery")
from vegeta import aeromant, boreas                                   # noqa: E402

from assemblies.workflows import fixed_wing as wf                     # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def test_drive_is_notebook_09as():
    AUW_KG, S, AR, OSWALD, RHO, V = 1.0, 0.17, 5.9, 0.8, 1.2, 14.0
    CL_CFD, CD_CFD = 0.40, 0.070
    d, p = boreas.inches(9, 6)
    prop = boreas.Propeller.from_pitch("9x6 electric", d, p, blades=2, chord_root_m=0.014, chord_max_m=0.022, chord_tip_m=0.006,
                                       mass_kg=0.012, rotor_mass_kg=0.045, notes="generic planform")
    airfoil = boreas.Airfoil(name="thin cambered section", cl_alpha=2 * math.pi * 0.9, alpha0_deg=-2.5, cl_max=1.1, cd0=0.02,
                             k=0.04, source="assumed for a 9-inch blade at Re ~ 1.5e5")
    motor = boreas.Motor("2212-920KV", kv_rpm_per_volt=920, resistance_ohm=0.12, no_load_current_a=0.6, max_current_a=20, mass_kg=0.055)
    battery = boreas.Battery("3S 5000 mAh", cells=3, capacity_ah=5.0, usable_fraction=0.8, mass_kg=0.380)
    system = boreas.Propulsion(prop, airfoil, motor, battery, rho=RHO)
    K = 1 / (math.pi * AR * OSWALD)
    CD0 = CD_CFD - K * CL_CFD ** 2
    W = AUW_KG * 9.81
    q = 0.5 * RHO * V ** 2
    cl = W / (q * S)
    D = q * S * (CD0 + K * cl ** 2)
    cruise = system.for_thrust(D / 2, V)
    endurance = battery.usable_wh / (2 * cruise.electrical_power) * 60
    _, _, _, pts, perf = wf.drive(AUW_KG, S, AR, CL_CFD, CD_CFD)
    assert pts["cruise"].rpm == pytest.approx(cruise.rpm) and perf["endurance_min"] == pytest.approx(endurance)
    assert pts["engine_out"].thrust == pytest.approx(system.at_throttle(1.0, V).thrust)


def _run_scenario():
    spec = importlib.util.spec_from_file_location("run_scenario", REPO / "scenarios" / "run_scenario.py")
    rs = importlib.util.module_from_spec(spec)
    sys.modules["run_scenario"] = rs
    try:
        spec.loader.exec_module(rs)
    finally:
        sys.modules.pop("run_scenario", None)
    return rs


@pytest.fixture
def fake_cfd(monkeypatch):
    seen = {}

    def run_cases(cases, *, jobs=1, processors=1, run=True, progress=False, cancel=None, timeout=None):
        for c in cases:
            seen[c.template.name] = c
        if not run:
            return [aeromant.Result(kind="aeromant.run").fail("NOT RUN") for _ in cases]
        r = aeromant.Result(kind="aeromant.results")
        r.metrics.update(Cl=0.42, Cd=0.065, lift_force_N=10.0, drag_force_N=1.5, thrust_N=2.0, torque_Nm=0.05, power_W=40.0,
                         efficiency=0.6, converged=True, mesh_cells=1000)
        return [r for _ in cases]

    import assemblies.workflows._common as common
    monkeypatch.setattr(common.aeromant, "run_cases", run_cases)
    monkeypatch.setattr(wf, "environment", lambda run=True: aeromant.OpenFOAMEnvironment())
    return seen


def test_rotor_disks_are_the_air_scenarios(tmp_path, fake_cfd):
    air = _run_scenario().air()
    root = wf.run(fidelity="full", aircraft={"thickness": 0.15}, run_fea=False, run_cfd=True, out=tmp_path / "runs",
                  vida_path=tmp_path / "f.vida", export=False, progress=False)
    ours = root.child("installed_cfd").params["cfd"]
    for k in ("disk1_center", "disk2_center", "disk_axis"):
        np.testing.assert_allclose(ours[k], air.cfd[k], atol=1e-12)
    np.testing.assert_allclose(ours["blade"], air.cfd["blade"], rtol=1e-12)
    np.testing.assert_allclose(ours["polar"], air.cfd["polar"], rtol=1e-12)
    for k in ("diameter", "blades", "rotation1", "rotation2", "disk_level", "velocity", "reference_length", "surface_level"):
        assert ours[k] == pytest.approx(air.cfd[k])
    assert air.design_params["thickness"] == 0.15 and air.design_params["angle_of_attack_deg"] == wf.AOA_DEG
    assert fake_cfd["aircraft_rotor_disks"].template.name == "aircraft_rotor_disks"


def test_workflow_tree_reuse_and_handoff(tmp_path, fake_cfd):
    kw = dict(fidelity="smoke", run_fea=False, out=tmp_path / "runs", vida_path=tmp_path / "f.vida",
              export_path=tmp_path / "f.json", progress=False)
    first = wf.run(run_cfd=False, **kw)
    assert [p for p, _ in first.walk()] == ["", "airframe", "propeller", "wing_fea", "aero_cfd", "rotor_cfd", "blade_fea",
                                            "installed_cfd"]
    assert first.results["polar_point"] == wf.RECORDED_POLAR and "recorded" in first.meta["not_run"][0]
    h = json.loads((tmp_path / "f.json").read_text())
    for k in ("auw_kg", "wing_area_m2", "v_cruise_m_s", "rho", "endurance_min", "preferred_wing", "parts_g", "points",
              "polar_point"):                                                     # what 09b cell 4 reads
        assert k in h
    assert set(h["points"]) == {"cruise", "climb", "engine_out"}
    assert boreas.load(tmp_path / "f_propulsion.json")["points"]
    second = wf.run(run_cfd=True, **kw)                                           # the CFD polar now drives everything
    assert second.results["polar_source"] == "aero_cfd" and second.results["polar_point"]["Cl"] == 0.42
    assert second.child("airframe").status() == "reused" and second.child("rotor_cfd").status() == "computed"
    third = wf.run(run_cfd=True, **kw)
    assert all(third.child(n).status() == "reused" for n in ("aero_cfd", "rotor_cfd", "installed_cfd"))


@pytest.mark.slow
@pytest.mark.requires_ccx
def test_wing_and_blade_fea_for_real(tmp_path):
    if shutil.which("ccx") is None:
        pytest.skip("CalculiX not on PATH")
    kw = dict(fidelity="smoke", run_cfd=False, out=tmp_path / "runs", vida_path=tmp_path / "f.vida", export=False, progress=False)
    r = wf.run(**kw)
    s = r.child("wing_fea").results["stress"]
    assert all(v["ok"] for v in s.values()) and s["pull_up"]["max_displacement_mm"] > s["engine_out"]["max_displacement_mm"]
    assert r.child("blade_fea").results["complete"]
    assert wf.run(**kw).child("wing_fea").status() == "reused"
