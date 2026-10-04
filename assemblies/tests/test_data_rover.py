"""Notebook data the rover workflow records (audit gaps rovers rover-1, rover-2): the rainflow counts of the wheel forces
per terrain (11 cell 21), the arm's modes with the wheel at the axle and its 100 N unit cases (cells 14, 18, 23), body
bounce, wheel hop and the arm's frequency check (cells 6, 8, 18, 19). Each lift against the notebook cell run as it is
over the same inputs, then the new keys through ``run()`` with the solvers mocked (``stubs.py``), twice: the second run
asks no solver and reuses every node."""
from __future__ import annotations

import ast
import json
import math
from types import SimpleNamespace

import pandas as pd
import pytest
import stubs

pytest.importorskip("cadquery")
from vegeta import chronos, talos

from assemblies import results, vida
from assemblies.components.rover import Rover
from assemblies.workflows import rover
from test_life import NB, cell

N = "11_rover_mechanics.ipynb"
SHORT_M = 20.0                     # the terrains cut to 20 m: the quarter car in a fraction of a second


@pytest.fixture
def short_terrains(monkeypatch):
    short = {k: dict(v, length=SHORT_M) for k, v in rover.TERRAINS.items()}
    monkeypatch.setattr(rover, "TERRAINS", short)
    return short


@pytest.fixture(scope="module")
def p():
    return Rover().resolve(part="rover")


def _sims(runs, terrains):
    """The workflow's quarter-car runs in the shape of 11 cell 8's ``sims`` (and cell 11's ``Fx``)."""
    return {k: dict(r, duration_s=r["t"][-1], speed=terrains[k]["speed"], length=terrains[k]["length"]) for k, r in runs.items()}


# ------------------------------------------------------------------------------------------------ rover-1: rainflow
def test_rainflow_blocks_are_cell_21(short_terrains, tmp_path, monkeypatch):
    s = rover.SUSPENSION
    runs = rover.terrain_runs(2.0, s["k_susp"], s["zeta"], s["k_tyre"])
    sims = _sims(runs, short_terrains)
    monkeypatch.setattr(chronos.LoadSpectrum, "plot", lambda self, *a, **k: None)          # the cell draws each spectrum
    ns = cell(N, 21, sims=sims, RUNS=tmp_path)                                              # the whole cell, as it is
    for name, sp in ns["spectra"].items():
        scale = ns["MISSION_MIN"][name] * 60 / sims[name]["duration_s"]
        for pattern, series in (("wheel_z", sims[name]["Ft"]), ("wheel_x", sims[name]["Fx"])):
            nb = {(b.mean, b.amplitude): b.cycles for b in sp.blocks if b.pattern == pattern}
            assert {(m, a): c for m, a, c in rover.rainflow_blocks(series, scale)} == nb     # the cell's own scaling: identical
            unscaled = rover.rainflow_blocks(series)                                          # what the workflow records
            assert {(m, a) for m, a, _ in unscaled} == set(nb)
            for m, a, c in unscaled:
                assert c * scale == pytest.approx(nb[(m, a)], rel=1e-12)
        assert len(sp.blocks) > 0 and (tmp_path / f"spectrum_{name.replace(' ', '_')}.json").is_file()


# ------------------------------------------------------------------------------------------------ rover-2: modes, unit cases
def test_suspension_frequencies_are_cells_6_8_18():
    m_total = 2.08
    c6 = cell(N, 6, ["M_WHEEL", "M_SPRUNG_CORNER"], M_TOTAL=m_total)
    c8 = cell(N, 8, ["K_SUSP", "ZETA", "K_TYRE", "C_SUSP"], M_SPRUNG_CORNER=c6["M_SPRUNG_CORNER"])
    c18 = cell(N, 18, ["f_hop", "f_body"], K_TYRE=c8["K_TYRE"], K_SUSP=c8["K_SUSP"], M_WHEEL=c6["M_WHEEL"],
               M_SPRUNG_CORNER=c6["M_SPRUNG_CORNER"])
    assert rover.M_WHEEL == c6["M_WHEEL"]
    assert rover.SUSPENSION == {"k_susp": c8["K_SUSP"], "zeta": c8["ZETA"], "k_tyre": c8["K_TYRE"]}
    s = rover.SUSPENSION
    assert rover.suspension_frequencies(m_total, s["k_susp"], s["zeta"], s["k_tyre"]) == {
        "m_sprung_corner_kg": c6["M_SPRUNG_CORNER"], "c_susp_N_s_m": c8["C_SUSP"], "f_body_hz": c18["f_body"], "f_hop_hz": c18["f_hop"]}


def test_frequency_table_is_cell_19(p):
    modes_hz = [3.0, 4.0, 190.0, 320.0]                         # low modes, so the nearest mode and margin differ per terrain
    c18 = cell(N, 18, ["structure"], modes=SimpleNamespace(metrics={"frequencies_hz": modes_hz}))
    c19 = cell(N, 19, ["D_WHEEL"], p=p)
    assert c18["structure"].damping_ratio == rover.ARM_DAMPING
    src = "".join(json.loads((NB / N).read_text())["cells"][19]["source"])
    expr = ast.parse(src).body[-1].value                                                     # the cell's table
    g = {"pd": pd, "math": math, "sims": {k: {"speed": v["speed"]} for k, v in rover.TERRAINS.items()},
         "terrains": rover.TERRAINS, "structure": c18["structure"], "D_WHEEL": c19["D_WHEEL"]}
    table = eval(compile(ast.Expression(expr), f"{N}:19", "eval"), g)
    mine = rover.frequency_table(modes_hz, p["wheel_diameter"], rover.TERRAINS)
    pd.testing.assert_frame_equal(pd.DataFrame(mine).T.round(2), table)                       # the cell shows .round(2)
    assert len({v["nearest_arm_mode_hz"] for v in mine.values()}) > 1


def test_arm_unit_models_are_cells_14_18_23(p, tmp_path):
    step = tmp_path / "arm.step"
    c14 = cell(N, 14, ["PA12CF", "L", "h", "rp", "ra", "ARM_REGIONS", "WHEEL_MASS_T", "arm_model"], p=p, M_WHEEL=rover.M_WHEEL,
               cad={"arm": SimpleNamespace(artifacts={"step": step})})
    c18 = cell(N, 18, ["arm_modal"], arm_model=c14["arm_model"], WHEEL_MASS_T=c14["WHEEL_MASS_T"])
    c23 = cell(N, 23, ["unit_models"], arm_model=c14["arm_model"])
    base, unit = rover.arm_unit_models(step, p, 2.0)                                         # the notebook meshes at 2 mm
    assert base.config() == c18["arm_modal"].config() and base.key == c18["arm_modal"].key
    assert base.masses == [talos.PointMass("axle", rover.M_WHEEL * 1e-3)]
    assert list(unit) == list(c23["unit_models"]) == ["wheel_z", "wheel_x"]
    for k, (m, load) in unit.items():
        nm, nload = c23["unit_models"][k]
        assert m.config() == nm.config() and m.key == nm.key and load == nload == rover.UNIT_N


# ------------------------------------------------------------------------------------------------ through run()
def test_rover_records_the_new_keys_and_reuses_them(short_terrains, tmp_path, solvers):
    kw = dict(fidelity="smoke", out=tmp_path / "r", vida_path=tmp_path / "r.vida", progress=False)
    first = rover.run(run_fea=False, **kw)                       # no FEA: the arm modes NOT RUN, the rest recorded
    assert first.child("arm_modes").status() == "NOT RUN" and first.results["arm_frequency_table"] is None
    T = first.child("terrains").results
    s = rover.SUSPENSION
    runs = rover.terrain_runs(first.child("terrains").params["m_total"], s["k_susp"], s["zeta"], s["k_tyre"])
    assert set(T["rainflow"]) == set(short_terrains)
    for k, r in runs.items():
        rf = T["rainflow"][k]
        assert rf["duration_s"] == T["summary"][k]["duration_s"] and rf["dt_s"] == pytest.approx(2e-3, rel=1e-12)
        assert rf["wheel_z"] == rover.rainflow_blocks(r["Ft"]) and rf["wheel_x"] == rover.rainflow_blocks(r["Fx"])
        assert rf["wheel_z"] and all(len(b) == 3 for b in rf["wheel_z"] + rf["wheel_x"])

    root = rover.run(**kw)                                       # with FEA (mocked): arm modes and unit cases
    m = root.child("arm_modes")
    assert m.results["complete"] and m.results["modes_hz"] == stubs.MODES_HZ[:rover.ARM_MODES]
    assert set(m.results["unit"]) == {"wheel_z", "wheel_x"} and all(v["load"] == 100.0 for v in m.results["unit"].values())
    assert m.results["point_mass_total"] == pytest.approx(rover.M_WHEEL * 1e-3)
    (base, _, n_modes), = [c.args for c in solvers.solve_modes.call_args_list]
    assert n_modes == 4 and base.masses == [talos.PointMass("axle", rover.M_WHEEL * 1e-3)]
    assert {"unit_z", "unit_x"} <= set(solvers.fea) and len(solvers.fea) == 9 + 2
    stubs.check_calls(solvers, progress=False)
    m_total = root.results["mass_kg"]
    assert root.results["f_body_hz"] == pytest.approx(math.sqrt(s["k_susp"] / ((m_total - 4 * rover.M_WHEEL) / 4)) / (2 * math.pi))
    assert root.results["f_hop_hz"] == pytest.approx(math.sqrt((s["k_tyre"] + s["k_susp"]) / rover.M_WHEEL) / (2 * math.pi))
    table = rover.frequency_table(m.results["modes_hz"], root.child("parts").params["rover"]["wheel_diameter"], short_terrains)
    pd.testing.assert_frame_equal(pd.DataFrame(root.results["arm_frequency_table"]), pd.DataFrame(table))     # NaN: no rocks
    assert root.child("terrains").status() == "reused"           # the rainflow came with the node: nothing again

    asked = (len(solvers.fea), solvers.solve_modes.call_count, solvers.mesh.call_count)
    again = rover.run(**kw)
    assert (len(solvers.fea), solvers.solve_modes.call_count, solvers.mesh.call_count) == asked
    assert all(again.child(n).status() == "reused" for n in ("parts", "terrains", "arm_fea", "chassis_fea", "arm_modes"))
    R = results.load(results.path_for(kw["vida_path"]))
    assert R["arm_modes"]["modes_hz"] == stubs.MODES_HZ[:4] and set(R["terrains"]["rainflow"]) == set(short_terrains)
    assert {"f_body_hz", "f_hop_hz", "m_sprung_corner_kg", "c_susp_N_s_m", "arm_frequency_table"} <= set(R[""])


def test_a_saved_terrains_node_without_rainflow_gets_it(short_terrains, tmp_path, solvers):
    kw = dict(fidelity="smoke", out=tmp_path / "r", vida_path=tmp_path / "r.vida", export=False, progress=False)
    first = rover.run(**kw)
    old = vida.load(kw["vida_path"])                             # as a tree saved before the rainflow was recorded
    rf = old.child("terrains").results.pop("rainflow")
    kept = {k: v for k, v in old.child("terrains").results.items()}
    old.save(kw["vida_path"])
    again = rover.run(**kw)
    t = again.child("terrains")
    assert t.status() == "computed" and t.results["rainflow"] == rf == first.child("terrains").results["rainflow"]
    assert {k: v for k, v in t.results.items() if k != "rainflow"} == kept                   # the old keys as they were
    assert all(again.child(n).status() == "reused" for n in ("arm_fea", "chassis_fea", "arm_modes"))
    assert rover.run(**kw).child("terrains").status() == "reused"
