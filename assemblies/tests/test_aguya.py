"""AGUYA's components and workflow. The component tests are notebook 29's (notebooks/designs/tests/test_aguya.py) on
the promoted copies; then the workflow: the engine from the microjet assembly, the tree, reuse, the export. The jet CFD
is replaced by known results; the wing FEA for real is marked slow."""
import math

import numpy as np
import pytest

from assemblies.components import aguya as ag, aguya_flight as F, aguya_jet, aguya_wing
from vegeta.boreas import microjet as mj


@pytest.fixture(scope="module")
def engine():
    return mj.from_catalogue("140 N class")


@pytest.fixture(scope="module")
def unit(engine):
    return F.jet_unit(engine)


@pytest.fixture(scope="module")
def plane(engine, unit):
    return F.airframe(ag.for_engine(engine), unit)


def test_jet_unit_matches_the_cycle(engine, unit):
    assert unit.full(0.0)[0] == pytest.approx(142.0, rel=1e-3)
    assert unit.full(0.0)[1] == pytest.approx(450.0 / 60e3, rel=1e-3)
    T = 60.0
    ff, rpm, ok = unit.for_thrust(100.0, T)
    assert ok and engine.rpm_idle < rpm < engine.rpm_max
    assert mj.solve(engine, rpm, 100.0).thrust == pytest.approx(T, rel=0.05)
    assert unit.for_thrust(100.0, 1e4)[2] is False


def test_top_speed_and_drag(plane, unit):
    m = plane.dry_mass_kg + 1.0
    v = F.top_speed(plane, unit, m)
    assert 140 < v < 200
    assert unit.full(v)[0] == pytest.approx(plane.drag(v, m), rel=1e-3)


def test_mission_burns_the_fuel_and_keeps_the_reserve(plane, unit):
    f = F.fuel_for(plane, unit, 10000.0)
    assert f.feasible and 0 < f.fuel_kg < plane.tank_kg
    assert f.fuel_left_kg == pytest.approx(0.15 * f.fuel_kg, abs=0.01)
    burned = sum(v[1] for v in f.phases.values())
    assert burned == pytest.approx(f.fuel_kg - f.fuel_left_kg, rel=1e-6)
    assert f.track["mass"][-1] == pytest.approx(plane.dry_mass_kg + f.fuel_left_kg, rel=1e-6)
    assert np.all(np.diff(f.track["mass"]) <= 1e-12)
    assert f.phases["sample"][0] == pytest.approx(180.0)
    assert 10000.0 / f.dash_speed < f.time_to_fire_s < 10000.0 / f.dash_speed + 40
    far = F.fuel_for(plane, unit, 20000.0)
    assert far.fuel_kg > f.fuel_kg and far.time_to_fire_s > f.time_to_fire_s


def test_out_of_reach_and_tank_sizing(engine, plane, unit):
    f = F.fuel_for(plane, unit, 60000.0)
    assert not f.feasible
    q, a, g = F.size_for(ag.for_engine(engine), unit, 30000.0)
    assert g.feasible and a.tank_kg >= g.fuel_kg and q["fuselage_diameter"] > 120


def test_cad_and_cfd_surfaces(engine, tmp_path):
    pytest.importorskip("cadquery")
    from vegeta.aeromant.stl import Surface, read_stl
    p = ag.for_engine(engine)
    for part in ("aircraft", "wing", "nacelle"):
        g = ag.Aguya().generate(**dict(p, part=part))
        assert g.shape.isValid() and len(g.shape.Solids()) == 1
    out = ag.cfd_surfaces(p, tmp_path)
    assert out["faces"]["intake"] == 1 and out["faces"]["exhaust"] == 1
    s = {k: read_stl(v) for k, v in out["paths"].items()}
    assert s["exhaust"].area == pytest.approx(out["nozzle_area_m2"] * 1e6, rel=0.02)
    assert s["intake"].area == pytest.approx(out["intake_area_m2"] * 1e6, rel=0.02)
    assert Surface(np.vstack([x.triangles for x in s.values()])).volume > 0
    assert p["nozzle_diameter"] == pytest.approx(2 * math.sqrt(engine.a8 / math.pi) * 1000)


# ------------------------------------------------------------------------------------------------- the workflow
from vegeta import aeromant                                             # noqa: E402

from assemblies import vida                                             # noqa: E402
from assemblies.components import impeller                              # noqa: E402
from assemblies.workflows import aguya as wf, microjet                  # noqa: E402


@pytest.fixture(scope="module")
def engine_vida(tmp_path_factory):
    pytest.importorskip("cadquery")
    d = tmp_path_factory.mktemp("microjet")
    microjet.run(fidelity="smoke", run_cfd=False, run_fea=False, out=d / "runs", vida_path=d / "microjet.vida",
                 export=False, progress=False)
    return d / "microjet.vida"


@pytest.fixture
def fake_jet_cfd(monkeypatch):
    calls = []

    def run_cases(cases, *, jobs=1, processors=1, run=True, progress=False, cancel=None, timeout=None):
        calls.append(len(cases))
        if not run:
            return [aeromant.Result(kind="aeromant.run").fail("NOT RUN") for _ in cases]
        r = aeromant.Result(kind="aeromant.results", metadata={"reused": False})
        r.metrics.update(drag_force_N=121.0, lift_force_N=50.0, Cd=0.031, Cl=0.2, mach_number=0.47, converged=True,
                         mesh_cells=1000, plume={"distance_m": [0.1, 0.5], "excess_temperature_K": [600.0, 120.0],
                                                 "exhaust_fraction": [1.0, 0.2]}, jet_excess_T_at_1m_K=60.0)
        return [r for _ in cases]

    monkeypatch.setattr(aguya_jet.aeromant, "run_cases", run_cases)
    monkeypatch.setattr(impeller, "environment", lambda run=True: aeromant.OpenFOAMEnvironment())
    return calls


def test_gust_numbers(engine, plane):
    g = aguya_wing.gust(ag.for_engine(engine), plane, 160.0)
    assert 0 < g["alleviation"] < 0.88 and g["load_factor"] > 6
    assert 1 + g["alleviation"] * 1.225 * g["penetration_speed"] * 7.5 * g["lift_slope"] * g["wing_area_m2"] / (2 * g["weight_N"]) \
        == pytest.approx(6.0)


def test_workflow_needs_the_engine(tmp_path):
    with pytest.raises(FileNotFoundError, match="assemblies.workflows.microjet"):
        wf.run(engine_vida=tmp_path / "none.vida", out=tmp_path / "runs", vida_path=tmp_path / "a.vida", export=False)


def test_workflow_tree_reuse_and_export(tmp_path, engine_vida, fake_jet_cfd):
    kw = dict(fidelity="smoke", engine_vida=engine_vida, run_fea=False, out=tmp_path / "runs", vida_path=tmp_path / "a.vida",
              export_path=tmp_path / "a.json", progress=False, merlin_vida=tmp_path / "merlin.vida",
              propulsors_vida=tmp_path / "propulsors.vida")
    first = wf.run(run_cfd=False, **kw)
    assert [p for p, _ in first.walk()] == ["", "engine", "engine/parts", "engine/compressor", "engine/wheels", "airframe",
                                            "jet_cfd", "wing", "merlin_race"]
    assert first.child("engine").status() == "reused" and first.child("airframe").status() == "computed"
    assert first.child("jet_cfd").status() == "NOT RUN" and first.child("wing").status() == "NOT RUN"
    assert first.child("engine").key == vida.load(engine_vida).key                 # grafted as it is
    assert first.child("airframe").results["feasible"] and first.child("merlin_race").status() == "NOT RUN"
    assert "merlin.vida" in first.child("merlin_race").meta["not_run"][0]

    second = wf.run(run_cfd=True, **kw)
    assert second.child("airframe").status() == "reused" and fake_jet_cfd == [1, 1]
    assert second.child("jet_cfd").results["forces"]["drag_force_N"] == 121.0
    assert second.results["summary"]["drag_at_dash_cfd_N"] == 121.0
    third = wf.run(run_cfd=True, **kw)
    assert third.child("jet_cfd").status() == "reused" and fake_jet_cfd == [1, 1]

    import json
    doc = json.loads((tmp_path / "a.json").read_text())
    assert doc["jet"]["cfd"]["drag_force_N"] == 121.0 and doc["source"]["jet_cfd_complete"] is True
    assert doc["design_range_km"] == 30 and doc["sizing"]["30"]["feasible"] is True

    with pytest.raises(ValueError, match="microjet"):
        wf.run(redo=["engine/compressor"], **kw)
    redo = wf.run(run_cfd=True, redo=["jet_cfd"], **kw)
    assert redo.child("jet_cfd").status() == "computed" and fake_jet_cfd == [1, 1, 1]


def test_a_new_engine_changes_every_node_on_it(tmp_path, engine_vida, fake_jet_cfd):
    kw = dict(fidelity="smoke", run_cfd=True, run_fea=False, out=tmp_path / "runs", vida_path=tmp_path / "a.vida",
              export=False, progress=False, merlin_vida=tmp_path / "merlin.vida", propulsors_vida=tmp_path / "propulsors.vida")
    a = wf.run(engine_vida=engine_vida, **kw)
    other = vida.load(engine_vida)
    other.params["engine_class"] = "200 N class"                                   # a different engine assembly
    other.save(tmp_path / "other.vida")
    b = wf.run(engine_vida=tmp_path / "other.vida", **kw)
    assert b.child("engine").key != a.child("engine").key and b.key != a.key
    assert b.child("airframe").key == a.child("airframe").key                       # same start geometry: still reused
    assert b.child("airframe").status() == "reused"


@pytest.mark.slow
@pytest.mark.requires_ccx
def test_wing_fea_for_real(tmp_path, engine_vida):
    import shutil
    if shutil.which("ccx") is None:
        pytest.skip("CalculiX not on PATH")
    kw = dict(fidelity="smoke", engine_vida=engine_vida, run_cfd=False, out=tmp_path / "runs", vida_path=tmp_path / "a.vida",
              export=False, progress=False, merlin_vida=tmp_path / "merlin.vida", propulsors_vida=tmp_path / "propulsors.vida")
    w = wf.run(**kw).child("wing")
    assert w.results["complete"] and all(v["ok"] for v in w.results["stress"].values())
    s = w.results["stress"]
    assert s["gust_at_dash"]["max_von_mises_MPa"] / s["limit_6g"]["max_von_mises_MPa"] == pytest.approx(
        w.params["gust"]["load_factor"] / 6.0, rel=1e-3)                            # linear: stress scales with the load
    assert len(w.results["modes"]["frequency_hz"]) == 6
    assert wf.run(**kw).child("wing").status() == "reused"
