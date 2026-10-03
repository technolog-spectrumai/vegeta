"""The hull component and the boat and submarine workflows against notebooks 12 and 13 and scenarios/run_scenario.py:
the resistance formulas as written in the notebooks, the draft / wetted surface / buoyancy centre / cruise rpm the
scenario presets recorded from the notebooks, and the scenes' CFD inputs equal to the presets at those numbers."""
import importlib.util
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cadquery")
from vegeta import aeromant                                            # noqa: E402

from assemblies.components import hull as H, propeller as pr          # noqa: E402
from assemblies.workflows import boat, submarine                      # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def _run_scenario():
    spec = importlib.util.spec_from_file_location("run_scenario", REPO / "scenarios" / "run_scenario.py")
    rs = importlib.util.module_from_spec(spec)
    sys.modules["run_scenario"] = rs
    try:
        spec.loader.exec_module(rs)
    finally:
        sys.modules.pop("run_scenario", None)
    return rs


def test_resistance_is_the_notebooks():
    RHO_W, NU_W, G = 1025.0, 1.05e-6, 9.81
    L_WL, S_wet, V = 1.0, 0.231, 1.5                                   # 12 cell 12, written out
    Re = V * L_WL / NU_W
    Cf = 0.075 / (math.log10(Re) - 2) ** 2
    Fn = V / math.sqrt(G * L_WL)
    Cw = 0.004 * (Fn / 0.45) ** 4 / (1 + (Fn / 0.45) ** 4) * 1.6
    assert H.boat_resistance(V, L_WL, S_wet)[0] == pytest.approx(0.5 * RHO_W * V ** 2 * S_wet * (1.25 * Cf + Cw))
    L_M, D_M, S_WET, S_BODY = 1.2, 0.18, 0.694, 0.600                  # 13 cell 8 / 14 cell 2
    k = 1.5 * (D_M / L_M) ** 1.5 + 7 * (D_M / L_M) ** 3
    Cf = 0.075 / (math.log10(V * L_M / NU_W) - 2) ** 2
    R = 0.5 * RHO_W * V ** 2 * S_BODY * Cf * (1 + k) + 0.5 * RHO_W * V ** 2 * (S_WET - S_BODY) * Cf * 1.5 * 1.3
    assert H.body_of_revolution_resistance(V, L_M, D_M, S_WET, S_BODY)[0] == pytest.approx(R)


@pytest.fixture
def fake_cfd(monkeypatch):
    seen = []

    def run_cases(cases, *, jobs=1, processors=1, run=True, progress=False, cancel=None, timeout=None):
        seen.extend(c.template.name for c in cases)
        if not run:
            return [aeromant.Result(kind="aeromant.run").fail("NOT RUN") for _ in cases]
        r = aeromant.Result(kind="aeromant.results")
        r.metrics.update(drag_force_N=5.0, lift_force_N=0.0, Cd=0.01, thrust_N=5.0, torque_Nm=0.1, power_W=20.0, efficiency=0.6,
                         converged=True, mesh_cells=1000)
        return [r for _ in cases]

    import assemblies.workflows._common as common
    monkeypatch.setattr(common.aeromant, "run_cases", run_cases)
    monkeypatch.setattr(boat, "environment", lambda run=True: aeromant.OpenFOAMEnvironment())
    monkeypatch.setattr(submarine, "environment", lambda run=True: aeromant.OpenFOAMEnvironment())
    return seen


def test_boat_reproduces_notebook_12_and_the_scenario(tmp_path, fake_cfd):
    kw = dict(fidelity="smoke", run_fea=False, out=tmp_path / "runs", vida_path=tmp_path / "b.vida", export_path=tmp_path / "b.json",
              progress=False)
    r = boat.run(run_cfd=False, **kw)
    hs = r.child("hull").results["hydrostatics"]
    assert hs["draft_mm"] == pytest.approx(53.1, abs=0.1)                       # run_scenario.boat(): draft = 53.1 recorded from 12
    assert hs["wetted_surface_m2"] == pytest.approx(0.231, abs=0.001)           # its reference area 2 x 0.231
    assert r.results["points"]["cruise"]["rpm"] == pytest.approx(2184.0, rel=1e-3)   # its rpm = 2184 recorded from 12
    pre = _run_scenario().boat()
    spec = pr.get("60 mm 3-blade marine")
    ours = boat.scene_params(r.child("hull").params["boat"], 53.1, 2184.0, spec.model(), spec.airfoil())
    for k, v in pre.cfd.items():
        assert np.allclose(ours[k], v, rtol=1e-12, atol=1e-12), k
    again = boat.run(run_cfd=True, **kw)
    assert again.child("hull").status() == "reused" and {"rans_ksst_external", "rotor_mrf", "aircraft_rotor_disks"} <= set(fake_cfd)
    assert boat.run(run_cfd=True, **kw).child("scene_cfd").status() == "reused"


def test_submarine_reproduces_notebook_13_and_the_scenario(tmp_path, fake_cfd):
    kw = dict(fidelity="smoke", run_fea=False, out=tmp_path / "runs", vida_path=tmp_path / "s.vida", export_path=tmp_path / "s.json",
              progress=False)
    r = submarine.run(run_cfd=False, **kw)
    h = r.child("hull").results
    assert h["wetted_surface_m2"] == pytest.approx(0.6936, abs=1e-4)            # run_scenario.sub(): reference area 0.6936
    assert h["CB_mm"][0] / 1000 == pytest.approx(0.6393, abs=1e-4)              # its cb_x = 0.6393 recorded from 13
    assert r.results["points"]["cruise"]["rpm"] == pytest.approx(910.0, rel=1e-3)    # its rpm = 910 recorded from 13
    assert h["mass_kg"] == pytest.approx(25.7, abs=0.05)                        # 14 cell 2: M_VEHICLE = 25.7 copied from 13
    pre = _run_scenario().sub()
    spec = pr.get("120 mm 3-blade")
    ours = submarine.scene_params(0.6936, 0.6393, 1.2, 910.0, spec.model(), spec.airfoil())
    for k, v in pre.cfd.items():
        assert np.allclose(ours[k], v, rtol=1e-12, atol=1e-12), k
    again = submarine.run(run_cfd=True, **kw)
    assert again.child("hull").status() == "reused" and "hull_rotor_disk" in fake_cfd
    assert submarine.run(run_cfd=True, **kw).child("rotor_cfd").status() == "reused"


@pytest.mark.real_solvers
@pytest.mark.slow
@pytest.mark.requires_ccx
def test_water_fea_for_real(tmp_path):
    if shutil.which("ccx") is None:
        pytest.skip("CalculiX not on PATH")
    b = boat.run(fidelity="smoke", run_cfd=False, out=tmp_path / "b", vida_path=tmp_path / "b.vida", export=False, progress=False)
    assert all(b.child(n).results["complete"] for n in ("hull_fea", "bracket_fea", "blade_fea"))
    s = submarine.run(fidelity="smoke", run_cfd=False, out=tmp_path / "s", vida_path=tmp_path / "s.vida", export=False, progress=False)
    ph = s.child("pressure_hull_fea").results["stress"]["design_depth"]
    assert ph["ok"] and ph["safety_factor"] > 1
