"""The quadcopter workflow (notebook 08): the drive numbers equal the notebook's cells run literally, the frame load
cases, the tree, reuse from the saved .vida, the export in boreas' format. Solvers off, or faked; the real FEA is slow."""
import shutil

import pytest

pytest.importorskip("cadquery")
from vegeta import aeromant, boreas                                    # noqa: E402

from assemblies import vida                                            # noqa: E402
from assemblies.components import impeller, quad_frame_analysis as qa  # noqa: E402
from assemblies.components.quad_frame import QuadFrame                 # noqa: E402
from assemblies.workflows import quadcopter as wf                      # noqa: E402


def test_drive_is_notebook_08s():
    import math
    frame = QuadFrame().generate()
    parts_g = sum(wf.PARTS_G.values()) + frame.volume * 1.25e-3       # cells 5 and 9
    AUW_KG, MOTORS, RHO = parts_g / 1000, 4, 1.2                      # cell 44
    d, p = boreas.inches(5, 4.3)                                       # cell 46
    prop = boreas.Propeller.from_pitch("5x4.3 tri-blade", d, p, blades=3, chord_root_m=0.010, chord_max_m=0.016,
                                       chord_tip_m=0.006, mass_kg=0.0045, rotor_mass_kg=0.020,
                                       notes="generic planform; fit chord/beta to the real propeller for better numbers")
    airfoil = boreas.Airfoil(name="thin cambered section", cl_alpha=2 * math.pi * 0.9, alpha0_deg=-3.0, cl_max=1.1, cd0=0.025,
                             k=0.045, source="assumed for a moulded 5-inch blade at Re ~ 1e5")
    motor = boreas.Motor("2306-2400KV", kv_rpm_per_volt=2400, resistance_ohm=0.06, no_load_current_a=1.2, max_current_a=40, mass_kg=0.030)
    battery = boreas.Battery("4S 1500 mAh", cells=4, capacity_ah=1.5, usable_fraction=0.8, mass_kg=0.180)
    system = boreas.Propulsion(prop, airfoil, motor, battery, rho=RHO)
    hover = system.for_thrust(AUW_KG * 9.81 / MOTORS)                  # cell 56
    hover_minutes = battery.usable_wh / (MOTORS * hover.electrical_power) * 60
    _, _, _, pts = wf.drive_points(AUW_KG)
    assert pts["hover"].rpm == pytest.approx(hover.rpm) and pts["full"].thrust == pytest.approx(system.at_throttle(1.0).thrust)
    from assemblies.components import propeller as pr
    assert pr.battery("4S 1500 mAh").usable_wh / (4 * pts["hover"].electrical_power) * 60 == pytest.approx(hover_minutes)


def test_frame_models():
    p = QuadFrame().resolve()
    ms = qa.frame_models("frame.step", p, max_thrust_per_motor_N=8.0, landing_N=40.0, element_mm=4.0)
    assert list(ms) == ["max_thrust", "hard_landing"] and len(ms["max_thrust"].regions) == 8
    assert [l.fz for l in ms["max_thrust"].loads] == [8.0] * 4 and ms["hard_landing"].loads[0].fz == 40.0


@pytest.fixture
def fake_cfd(monkeypatch):
    calls = []

    def run_cases(cases, *, jobs=1, processors=1, run=True, progress=False, cancel=None, timeout=None):
        calls.append(cases[0].template.name)
        if not run:
            return [aeromant.Result(kind="aeromant.run").fail("NOT RUN") for _ in cases]
        r = aeromant.Result(kind="aeromant.results")
        r.metrics.update(thrust_N=1.1, torque_Nm=0.01, power_W=9.0, figure_of_merit=0.5, drag_force_N=0.8, lift_force_N=0.1,
                         Cd=1.1, Cl=0.1, converged=True, mesh_cells=1000)
        return [r for _ in cases]

    monkeypatch.setattr(wf.aeromant, "run_cases", run_cases)
    monkeypatch.setattr(wf, "environment", lambda run=True: aeromant.OpenFOAMEnvironment())
    return calls


def test_workflow_tree_reuse_and_export(tmp_path, fake_cfd):
    kw = dict(fidelity="smoke", run_fea=False, out=tmp_path / "runs", vida_path=tmp_path / "q.vida",
              export_path=tmp_path / "q.json", progress=False)
    first = wf.run(run_cfd=False, **kw)
    assert [p for p, _ in first.walk()] == ["", "frame", "frame_fea", "canopy_cfd", "propeller", "rotor_cfd", "blade_fea"]
    assert first.child("frame_fea").status() == "NOT RUN" and first.child("rotor_cfd").status() == "NOT RUN"
    doc = boreas.load(tmp_path / "q.json")                              # notebook 08's export format
    assert set(doc["points"]) == {"hover", "cruise", "full"}
    second = wf.run(run_cfd=True, **kw)
    assert second.child("frame").status() == "reused" and second.child("rotor_cfd").results["forces"]["thrust_N"] == 1.1
    assert sorted(fake_cfd[-2:]) == ["rans_ksst_external", "rotor_mrf_static"]
    third = wf.run(run_cfd=True, **kw)
    assert third.child("rotor_cfd").status() == "reused" and third.child("canopy_cfd").status() == "reused"
    assert len(fake_cfd) == 4                                          # nothing new solved
    other = wf.run(run_cfd=True, **dict(kw, frame={"arm_width": 16.0}))
    assert other.child("frame").status() == "computed" and other.child("propeller").status() == "reused"
    assert vida.load(tmp_path / "q.vida").child("frame").params["frame"]["arm_width"] == 16.0


@pytest.mark.real_solvers
@pytest.mark.slow
@pytest.mark.requires_ccx
def test_frame_and_blade_fea_for_real(tmp_path):
    if shutil.which("ccx") is None:
        pytest.skip("CalculiX not on PATH")
    kw = dict(fidelity="smoke", run_cfd=False, out=tmp_path / "runs", vida_path=tmp_path / "q.vida", export=False, progress=False)
    r = wf.run(**kw)
    s = r.child("frame_fea").results["stress"]
    assert all(v["ok"] for v in s.values()) and s["hard_landing"]["max_von_mises_MPa"] > s["max_thrust"]["max_von_mises_MPa"]
    assert r.child("blade_fea").results["complete"]
    assert wf.run(**kw).child("frame_fea").status() == "reused"
