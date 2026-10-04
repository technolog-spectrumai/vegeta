"""Every workflow end to end with its solvers ON, the solver run functions mocked and answered by stubs (``stubs.py``,
installed for every test by ``conftest.py``): the real CAD, the real FEA models and CFD cases are built and handed to
the mocks. Checks that no node is left NOT RUN, that a second run reuses everything without calling a solver again,
and what the mocks were asked to solve."""
from __future__ import annotations

from pathlib import Path

import pytest
import stubs

from assemblies import results

pytest.importorskip("cadquery")

from assemblies.workflows import (aguya, boat, fixed_wing, merlin, microjet, onager, quadcopter, rover,  # noqa: E402
                                  submarine, walkers)

DATA = Path(__file__).resolve().parents[1] / "data"
GRAFTED = ("engine", "propulsors")      # saved trees grafted as they are: they keep the status they were saved with


def not_run(root) -> list[str]:
    return [p or root.name for p, n in root.walk() if n.status() == "NOT RUN"]


def solved_twice(run, solvers, **kw):
    """Run, check nothing is NOT RUN and that the solvers were called as the real ones take it, run again: the second
    run asks no solver and reuses every leaf of its own (grafted trees keep their saved status)."""
    first = run(**kw)
    assert not_run(first) == []
    if kw.get("export", True):                            # the plain data for notebooks, next to the .vida
        assert results.load(results.path_for(kw["vida_path"])).status().loc[first.name, "kind"] == first.kind
    stubs.check_calls(solvers, progress=kw.get("progress"))
    asked = (len(solvers.fea), len(solvers.cfd), len(solvers.sim))
    again = run(**kw)
    assert (len(solvers.fea), len(solvers.cfd), len(solvers.sim)) == asked
    leaves = [p for p, n in again.walk() if p and not n.children and n.results and p.split("/")[0] not in GRAFTED]
    assert leaves and all(again.child(p).status() == "reused" for p in leaves), [
        (p, again.child(p).status()) for p in leaves if again.child(p).status() != "reused"]
    return first


def common(tmp_path, name):
    return dict(fidelity="smoke", out=tmp_path / name, vida_path=tmp_path / f"{name}.vida", progress=False)


def test_microjet_and_aguya(tmp_path, solvers):
    m = solved_twice(microjet.run, solvers, export_path=tmp_path / "microjet.json", **common(tmp_path, "microjet"))
    assert m.child("compressor").results["complete"] and m.child("wheels").results["complete"]
    a = solved_twice(aguya.run, solvers, engine_vida=tmp_path / "microjet.vida", merlin_vida=DATA / "merlin.vida",
                     propulsors_vida=DATA / "propulsors.vida", export_path=tmp_path / "aguya.json", **common(tmp_path, "aguya"))
    assert a.child("jet_cfd").results["complete"] and a.child("wing").results["complete"]
    assert {"compressor_mrf", "jet_external"} <= set(solvers.cfd)
    assert "jet_external" in solvers.cfd


def test_quadcopter(tmp_path, solvers):
    q = solved_twice(quadcopter.run, solvers, export_path=tmp_path / "q.json", **common(tmp_path, "quadcopter"))
    assert all(q.child(n).results["complete"] for n in ("frame_fea", "canopy_cfd", "rotor_cfd", "blade_fea"))
    assert set(solvers.cfd) == {"rans_ksst_external", "rotor_mrf_static"}
    # the life (08 Part 3): one mesh, the modes with the 4 motor and 4 stack masses, 3 unit cases, 3 missions + 3 balanced
    assert solvers.mesh.call_count == 1 and solvers.solve_modes.call_count == 1
    modal = solvers.solve_modes.call_args.args[0]
    assert len(modal.masses) == 8 and not modal.loads and solvers.solve_modes.call_args.args[2] == 8
    assert {"thrust", "unbalance", "landing"} <= set(solvers.fea) and solvers.assess_fatigue.call_count == 6
    fat = q.child("life/fatigue").results
    assert set(fat["life"]) == {"inspection", "freestyle", "cruise"} and fat["worst"] in fat["life"]
    assert set(fat["mixes"]) == {"inspection-heavy", "freestyle-heavy", "cruise-only"} and fat["hours_to_failure"] > 0
    life_json = __import__("json").loads((tmp_path / "q_life.json").read_text())
    assert life_json["modes_hz"] == q.child("life/unit_fea").results["modes_hz"] and "hours_to_failure" in life_json


def test_fixed_wing(tmp_path, solvers):
    f = solved_twice(fixed_wing.run, solvers, export_path=tmp_path / "fw.json", **common(tmp_path, "fixed_wing"))
    assert f.results["polar_source"] == "aero_cfd"
    assert all(f.child(n).results["complete"] for n in ("wing_fea", "aero_cfd", "rotor_cfd", "blade_fea", "installed_cfd"))
    assert set(solvers.cfd) == {"rans_ksst_external", "rotor_mrf", "aircraft_rotor_disks"}
    # the life (09b): wing 2415 and 2412, fuselage: 3 meshes and modal solves, 4 + 4 + 3 unit cases, 3 + 3 + 3 + 1 fatigue
    assert solvers.mesh.call_count == 3 and solvers.solve_modes.call_count == 3
    assert [c.args[2] for c in solvers.solve_modes.call_args_list] == [8, 8, 6]
    assert solvers.assess_fatigue.call_count == 10
    steps = [c.args[0].geometry for c in solvers.solve_modes.call_args_list]
    assert len(set(steps)) == 3                                    # three different STEP files: 2415, 2412, fuselage
    w = f.child("wing_life/fatigue").results
    assert set(w["damage_per_mission"]) == set(w["damage_naca2412"]) == {"survey", "patrol", "windy_hops"}
    assert len(w["nacelle_amplitude_mm"]) == 3 and w["damage_per_1000h"] > 0
    fu = f.child("fuselage_life/fatigue").results
    assert fu["static"]["bound_MPa"] >= fu["static"]["hotspot_stress_MPa"] > 0 and fu["mass_kg"]["shell"] > 0
    life_json = __import__("json").loads((tmp_path / "fw_life.json").read_text())
    assert set(life_json) >= {"wing", "fuselage"}


def test_merlin(tmp_path, solvers):
    out = tmp_path / "merlin_design.json"
    m = solved_twice(merlin.run, solvers, propulsors_vida=DATA / "propulsors.vida", export_path=out, **common(tmp_path, "merlin"))
    assert m.child("wing_fea").results["complete"] and set(m.child("polar_cfd").results["polar"]) == set(merlin.KINDS)
    assert out.is_file()
    assert solvers.cfd == ["rans_ksst_external"] * 3


def test_boat_and_submarine(tmp_path, solvers):
    b = solved_twice(boat.run, solvers, export_path=tmp_path / "b.json", **common(tmp_path, "boat"))
    assert set(solvers.cfd) == {"rans_ksst_external", "rotor_mrf", "aircraft_rotor_disks"}
    assert all(b.child(n).results["complete"] for n in ("hull_cfd", "hull_fea", "bracket_fea", "rotor_cfd", "blade_fea", "scene_cfd"))
    before = len(solvers.cfd)
    s = solved_twice(submarine.run, solvers, export_path=tmp_path / "s.json", **common(tmp_path, "submarine"))
    assert set(solvers.cfd[before:]) == {"rans_ksst_external", "rotor_mrf", "hull_rotor_disk"}
    assert all(s.child(n).results["complete"] for n in ("hull_cfd", "pressure_hull_fea", "rotor_cfd", "blade_fea", "scene_cfd"))


def test_ground(tmp_path, solvers):
    r = solved_twice(rover.run, solvers, export=False, **common(tmp_path, "rover"))
    assert r.child("arm_fea").results["complete"] and r.child("chassis_fea").results["complete"]
    assert len(r.child("arm_fea").results["stress"]) == 6 and len(r.child("chassis_fea").results["stress"]) == 3
    assert len(solvers.fea) == 9 and solvers.cfd == []
    o = solved_twice(onager.run, solvers, **common(tmp_path, "onager"))
    assert o.child("sentinel/leg_fea").results["complete"] and len(solvers.fea) == 9 + 8
    assert sorted(solvers.sim) == ["scene"] * 4
    w = solved_twice(walkers.run, solvers, vida_path=tmp_path / "walkers.vida", progress=False)
    assert w.child("apheloria/pack").results["stub"] and solvers.sim.count("pack") == 1


def test_race_table_per_power_is_study(capsys):
    """The workflow's race under a progress bar gives exactly the table of one ``merlin_race.study`` call."""
    import pandas as pd

    from assemblies import vida
    from assemblies.components import merlin_race as mr
    from assemblies.workflows.propulsors import load_library
    lib = load_library(vida.load(DATA / "propulsors.vida"))
    ok = [e for e in lib["entries"] if "error" not in e]
    small = {**lib, "entries": [e for e in ok if e["kind"] == "edf"][:2] + [e for e in ok if e["kind"] != "edf"][:2]}
    small["by_id"] = {e["id"]: e for e in small["entries"]}
    expected = mr.study(small, powers=(1900.0, 2700.0))
    got = merlin.race_table(small, None, powers=(1900.0, 2700.0), progress=True)
    pd.testing.assert_frame_equal(got, expected)
    assert "MERLIN race" in capsys.readouterr().err
