"""Notebook data the MERLIN and propulsor workflows record (audit gaps merlin M1, M2, M3, M6, M7, P1), each lift against the
notebook cell it came from, run as it is over the same inputs: the propulsor tables at 1900 W (25 cell 48, 25b cell 30,
25c cell 10) on each library node; 26b's race table and winners (cells 8, 9), the propulsors and airframes flown
(cells 4, 6), the mountain (cell 12), where the sensor breathes (cell 14) and the smoke mission (cells 18, 20) on MERLIN's
mission node. Then the new keys through ``run()`` (solvers mocked; nothing here calls one), in the .vida and in
<name>_results.json, and a second run that reuses them."""
from __future__ import annotations

import ast
import dataclasses
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("cadquery")
from assemblies import DATA, results, vida                                  # noqa: E402
from assemblies.components import merlin_flight as mf, propulsor_maps as pm  # noqa: E402
from assemblies.workflows import merlin as wm, propulsors as wp             # noqa: E402
from assemblies.workflows.propulsors import load_library                    # noqa: E402

NB = Path(__file__).resolve().parents[2] / "notebooks"
P25, P25B, P25C, M26B = ("25_air_propeller.ipynb", "25b_ducted_fan.ipynb", "25c_exotic_propellers.ipynb",
                         "26b_merlin_mission.ipynb")


def _source(notebook: str, index: int) -> list:
    return ast.parse("".join(json.loads((NB / notebook).read_text())["cells"][index]["source"])).body


def run_cell(notebook: str, index: int, *, names=None, upto=None, pick=None, **globals_) -> dict:
    """Run notebook ``notebook`` cell ``index`` over ``globals_`` and return the namespace: the whole cell, or only its
    function definitions and assignments to ``names``, or its statements up to and including the first that ``upto``
    accepts, or those ``pick`` accepts."""
    body = _source(notebook, index)
    if names is not None:
        def assigned(s):
            return {n.id for t in s.targets for n in ast.walk(t) if isinstance(n, ast.Name)} if isinstance(s, ast.Assign) else set()
        body = [s for s in body if (isinstance(s, ast.FunctionDef) and s.name in names) or (assigned(s) and assigned(s) <= set(names))]
    if upto is not None:
        stop = next(i for i, s in enumerate(body) if upto(s))
        body = body[:stop + 1]
    if pick is not None:
        body = [s for s in body if pick(s)]
    ns = {"math": math, "np": np, "pd": pd, "os": os, "dataclasses": dataclasses, **globals_}
    exec(compile(ast.Module(body, []), f"{notebook}:{index}", "exec"), ns)
    return ns


def last_expression(notebook: str, index: int, **ns):
    """The value of the cell's last line (the table it shows)."""
    return eval(compile(ast.Expression(_source(notebook, index)[-1].value), f"{notebook}:{index}", "eval"), dict(ns))


def assigns_to(name):
    return lambda s: isinstance(s, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in s.targets)


# ------------------------------------------------------------------------------------------------ P1: the tables at 1900 W
@pytest.fixture
def pmaps(monkeypatch):
    """``import propulsor_maps as pmaps`` in the cells: the assemblies copy, its committed map files."""
    monkeypatch.setitem(sys.modules, "propulsor_maps", pm)
    monkeypatch.delenv("AIR_PROP_MAPS", raising=False)
    return pm


def test_propeller_table_is_25_cell_48(pmaps):
    ns = run_cell(P25, 48, upto=assigns_to("MAP_TAB"))
    ours = wp.map_table(pm.load(pm.LIBRARIES["propellers"]))
    assert ours == ns["rows"]
    pd.testing.assert_frame_equal(pd.DataFrame(ours).T, ns["MAP_TAB"])


def test_fan_table_is_25b_cell_30(pmaps):
    ns = run_cell(P25B, 30, upto=assigns_to("FAN_TAB"))
    assert wp.fan_table(pm.load(pm.LIBRARIES["edf"])) == ns["rows"]


def test_exotic_table_is_25c_cell_10(pmaps):
    layouts = run_cell(P25C, 4, names=["LAYOUTS"])["LAYOUTS"]
    ns = run_cell(P25C, 10, upto=lambda s: isinstance(s, ast.For), LAYOUTS=layouts)
    assert wp.exotic_table(pm.load(pm.LIBRARIES["exotic"]), layouts) == ns["rows"]


def test_propulsors_run_records_the_tables_once(tmp_path):
    committed = vida.load(DATA / "propulsors.vida")
    if any(wp.build("full").child(k).key != committed.child(k).key for k in wp.KINDS):
        pytest.skip("the committed propulsors.vida is not of the current spaces (it would solve, and write data/)")
    shutil.copy(DATA / "propulsors.vida", tmp_path / "p.vida")
    kw = dict(fidelity="full", out=tmp_path / "runs", vida_path=tmp_path / "p.vida", progress=False)
    t = wp.run(**kw)
    tables = {"propellers": wp.map_table, "exotic": wp.exotic_table, "edf": wp.fan_table}
    for k in wp.KINDS:
        node = t.child(k)
        assert node.results["source"] == committed.child(k).results["source"]               # the library itself reused
        recs = node.results["at_1900W"]
        rows = tables[k]({"entries": load_library(vida.Assembly("one", "x", children=[committed.child(k)]))["entries"]})
        assert len(recs) == len(rows) == node.results["n"] * (1 if k == "edf" else 2)
        for r in recs:
            key = (r["label"], r["layout"]) if k != "edf" else r["label"]
            assert {c: r[c] for c in rows[key]} == rows[key]
            assert r["id"] == next(e["id"] for e in node.results["entries"] if e["label"] == r["label"])
    saved = results.load(tmp_path / "p_results.json")
    assert saved.table("edf", "at_1900W").shape[0] == 360
    assert set(saved.table("propellers", "at_1900W").columns) >= {"id", "label", "layout", "static thrust at 1900 W [N]",
                                                                    "at 30 m/s [N]", "at 60 m/s [N]", "w", "t"}
    again = wp.run(**kw)
    assert all(again.child(k).status() == "reused" for k in wp.KINDS)


# ------------------------------------------------------------------------------------------------ MERLIN's mission (26b)
@pytest.fixture(scope="module")
def merlins():
    design = vida.load(DATA / "merlin.vida").results["design"]
    lib = load_library(vida.load(DATA / "propulsors.vida"))
    return design, wm._merlins(design, lib)


@pytest.fixture(scope="module")
def extras(merlins):
    return wm.mission_extras(merlins[1])


@pytest.fixture(scope="module")
def nb26b():
    """26b cell 2's names (``NAME``, ``KINDS``, ``RHO``)."""
    return run_cell(M26B, 2, names=["NAME", "KINDS", "RHO", "NU", "G"])


@pytest.fixture(scope="module")
def race_tab(merlins):
    return run_cell(M26B, 8, names=["RACE_TAB"], mf=mf, MERLINS=merlins[1], DISTANCES_KM=mf.DISTANCES_KM,
                    MISSION=wm.MISSION)["RACE_TAB"]


def _kind(name, nb):
    return next(k for k in nb["KINDS"] if nb["NAME"][k] == name)


def test_race_table_is_26b_cell_8(race_tab, extras):
    assert race_tab.reset_index(names=["distance", "propulsor"]).to_dict("records") == extras["race_table"]
    assert results.plain(extras["race_table"])[0]["limited by"]


def test_winners_are_26b_cell_9(race_tab, extras, nb26b, capsys):
    ns = run_cell(M26B, 9, RACE_TAB=race_tab, DISTANCES_KM=mf.DISTANCES_KM, KINDS=nb26b["KINDS"], NAME=nb26b["NAME"])
    assert extras["winner_per_distance"] == {f"{d:.0f} km": ns["WINNER"][d] for d in mf.DISTANCES_KM}      # cell 24's
    home = re.findall(r"first home: the (.+?) \(", capsys.readouterr().out)
    assert list(extras["first_home_per_distance"].values()) == [_kind(n, nb26b) for n in home]
    assert ns["FASTEST"] == extras["smoke"]["propulsor"]


def test_units_and_airframes_are_26b_cells_4_and_6(merlins, extras, nb26b, capsys):
    design, M = merlins
    shown = []
    ns = run_cell(M26B, 4, mf=mf, MAX_ELECTRICAL_W=wm.MAX_ELECTRICAL_W, DRIVE_EFF=wm.DRIVE_EFF, MISSION=wm.MISSION,
                  DESIGN=design, NAME=nb26b["NAME"], display=shown.append)
    out = capsys.readouterr().out
    u = extras["units"]
    assert (f"EDF ({u['edf']['id']}): duct_loss {u['edf']['duct_loss']:.3f}, exit area ratio {u['edf']['exit_area_ratio']:.3f}, "
            f"jet scrubbing {u['edf']['scrub_area_m2'] * 1e4:.0f} cm²") in out
    for k in ("tractor", "pusher"):
        assert (f"{nb26b['NAME'][k]}: rpm limit {u[k]['rpm_max']:.0f}; chosen {u[k]['prop']}, effective wake w = {u[k]['w']:.4f}, "
                f"thrust deduction t = {u[k]['t']:.4f}") in out
    ours = pd.DataFrame(extras["pitch_study"])
    for k, table in zip(("tractor", "pusher"), shown):                          # cell 4 displays the two pitch studies
        pd.testing.assert_frame_equal(ours[ours.propulsor == k].drop(columns="propulsor").set_index("pitch [in]").round(4), table,
                                      check_index_type=False)
    MERLINS = ns["MERLINS"]
    assert {k: v["info"]["entry"]["id"] for k, v in MERLINS.items()} == {k: v["id"] for k, v in u.items()}
    perf = run_cell(M26B, 6, names=["PERF"], mf=mf, MERLINS=M)["PERF"]
    table = last_expression(M26B, 6, PERF=perf, MERLINS=M, NAME=nb26b["NAME"], pd=pd)
    for k in M:                                                                 # cell 6's static thrust (shown at 2 decimals)
        assert table.loc["static thrust [N]", nb26b["NAME"][k]] == round(u[k]["static_thrust_N"], 2)
        a = extras["airframes"][k]                                              # cell 24's polar and masses
        assert (a["cd0"], a["wing_area_m2"], a["aspect_ratio"], a["mass_kg"]) == (
            M[k]["airframe"].cd0, M[k]["airframe"].wing_area_m2, M[k]["airframe"].aspect_ratio, M[k]["airframe"].mass_kg)


def test_mountain_is_26b_cell_12(merlins, race_tab, extras, nb26b, capsys):
    M = merlins[1]
    ns = run_cell(M26B, 12, names=["FIRE_ELEVATION_M", "MOUNTAIN_TAB", "cmp_"], mf=mf, MERLINS=M, DISTANCES_KM=mf.DISTANCES_KM,
                  MISSION=wm.MISSION, RACE_TAB=race_tab)
    assert ns["FIRE_ELEVATION_M"] == wm.FIRE_ELEVATION_M
    mountain = ns["MOUNTAIN_TAB"].assign(**{"fire elevation [m]": ns["FIRE_ELEVATION_M"]})
    assert mountain.reset_index(names=["distance", "propulsor"]).to_dict("records") == extras["race_mountain"]
    cmp_ = ns["cmp_"]
    flat = cmp_.set_axis([" / ".join(c) for c in cmp_.columns], axis=1).reset_index(names=["distance", "propulsor"])
    assert flat.to_dict("records") == extras["race_mountain_vs_flat"]
    assert "reach [min] / uphill costs [s]" in extras["race_mountain_vs_flat"][0]
    run_cell(M26B, 12, pick=lambda s: isinstance(s, ast.For), MOUNTAIN_TAB=ns["MOUNTAIN_TAB"], DISTANCES_KM=mf.DISTANCES_KM)
    uphill = re.findall(r"uphill: first over the fire: the (.+?) \(", capsys.readouterr().out)
    assert list(extras["winner_per_distance_mountain"].values()) == [_kind(n, nb26b) for n in uphill]


def test_sensor_air_is_26b_cell_14(merlins, extras, nb26b):
    ns = run_cell(M26B, 14, MERLINS=merlins[1], RHO=nb26b["RHO"])
    assert extras["sensor_air"] == ns["rows"]
    assert extras["sensor_air"]["15 m/s"]["EDF capture area / highlight area"] == pytest.approx(3.13, abs=0.01)


def test_smoke_mission_is_26b_cells_18_and_20(merlins, extras):
    smoke = extras["smoke"]
    ns = run_cell(M26B, 18, names=["PLUME", "FOREST", "af", "unit", "R20", "EP"], mf=mf, MERLINS=merlins[1],
                  FASTEST=smoke["propulsor"], MISSION=wm.MISSION)
    EP = ns["EP"]
    assert smoke["plume"] == dataclasses.asdict(ns["PLUME"])
    assert smoke["summary"] == EP.summary()
    assert smoke["phases"] == EP.phase_table().reset_index().to_dict("records")
    est = run_cell(M26B, 20, names=["EST"], mf=mf, PLUME=ns["PLUME"], EP=EP)["EST"]
    assert {k: v for k, v in smoke["source_estimate"].items() if np.ndim(v) == 0} == {k: v for k, v in est.items() if np.ndim(v) == 0}
    check = last_expression(M26B, 20, EST=est, PLUME=ns["PLUME"], pd=pd)
    assert smoke["ignition_check"] == check["ignition check"].to_dict()
    np.testing.assert_array_equal(smoke["track"]["ppm"], EP.ppm[::wm.SMOKE_TRACK_EVERY])
    assert smoke["summary"]["passes"] >= 1 and smoke["ignition_check"]["true [MW]"] == 40.0


def test_extras_without_the_ducted_fan_and_out_of_reach(merlins):
    """The smoke-fidelity library has no 90 mm fan: no sensor table. On a third of the battery no MERLIN reaches the
    30 km fire: its rows are NaN (null in the JSON) with the reason, its winners None; the rest as usual."""
    x = wm.mission_extras({k: v for k, v in merlins[1].items() if k != "edf"}, dataclasses.replace(wm.MISSION, battery_wh=60.0))
    assert x["sensor_air"] is None and set(x) == set(wm.MISSION_EXTRAS)
    assert len(x["race_table"]) == 2 * len(mf.DISTANCES_KM) and x["smoke"]["propulsor"] in ("tractor", "pusher")
    far = [r for r in results.plain(x["race_table"]) if r["distance"] == "30 km"]
    assert all(r["reach [min]"] is None and r["landing [min]"] is None and "energy" in r["limited by"] for r in far)
    assert x["winner_per_distance"]["30 km"] is None and x["winner_per_distance_mountain"]["30 km"] is None
    assert x["winner_per_distance"]["5 km"] in ("tractor", "pusher")
    vida.Assembly("m", "merlin_mission").record(**x)                           # storable as it is


def test_extras_missing_follows_the_constants(extras):
    assert wm.extras_missing({}) == list(wm.MISSION_EXTRAS)
    assert wm.extras_missing(dict(extras)) == []
    other = dict(extras, race_mountain=[dict(r, **{"fire elevation [m]": 300.0}) for r in extras["race_mountain"]])
    assert wm.extras_missing(other) == list(wm.MOUNTAIN_KEYS)
    other = dict(extras, smoke=dict(extras["smoke"], plume=dict(extras["smoke"]["plume"], heat_mw=10.0)))
    assert wm.extras_missing(other) == ["smoke"]


def test_merlin_run_records_the_extras_once(tmp_path):
    """On the committed tree (the wing FEA and polar CFD marked done without results, so no CAD and the same design):
    the reused mission gets the new keys, keeps its old ones, and a second run reuses it."""
    prior = vida.load(DATA / "merlin.vida")
    prior.child("wing_fea").results = {"complete": True}
    prior.child("polar_cfd").results = {"complete": True}
    prior.save(tmp_path / "m.vida")
    kw = dict(fidelity="full", run_cfd=False, run_fea=False, out=tmp_path / "runs", vida_path=tmp_path / "m.vida",
              export_path=tmp_path / "merlin_design.json", progress=False)
    root = wm.run(**kw)
    mission = root.child("mission")
    assert root.child("race").status() == "reused"                    # same design: the race is not run again
    assert set(wm.MISSION_EXTRAS) <= set(mission.results)
    old = prior.child("mission").results
    for k in ("reach", "mass_kg"):
        assert mission.results[k] == old[k]
    np.testing.assert_array_equal(mission.results["performance"]["pusher"]["V"], old["performance"]["pusher"]["V"])
    saved = results.load(tmp_path / "m_results.json")
    assert saved.table("mission", "race_table").shape == (15, 11)
    assert saved.table("mission", "race_mountain")["fire elevation [m]"].eq(600.0).all()
    assert saved["mission"]["winner_per_distance"]["20 km"] == saved["mission"]["smoke"]["propulsor"]
    assert saved.table("mission", "pitch_study").shape[0] == 6
    again = wm.run(**dict(kw, export=False))
    assert again.child("mission").status() == "reused"
    assert wm.extras_missing(vida.load(tmp_path / "m.vida").child("mission").results) == []
