"""Notebook data the fixed-wing workflow records on its root (audit gaps air-1, air-2): the level-flight drag (09a cell
42), the speed range and climb / engine-out envelope (09a cell 49) with the wing loading (cell 10), and the noise table
with its tone spectra (09a cell 56), each against the notebook cell run as it is over the workflow's own drive; then the
new keys through ``run()`` with the solvers mocked, in the .vida and in fixed_wing_results.json."""
from __future__ import annotations

import ast
import json
import math
import shutil
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from vegeta import boreas

from assemblies import DATA, results, vida
from assemblies.components import propeller as pr
from assemblies.workflows import fixed_wing as wf
from test_life import NB, cell

A = "09a_fixed_wing_design.ipynb"
POINTS = ("cruise", "climb", "static", "engine_out")
NB_POINTS = {"cruise": "cruise 14 m/s", "climb": "climb 12 m/s full", "static": "static full",
             "engine_out": "engine-out cruise (1 motor full)"}                    # the names cell 51 gives them
SPEED_COLUMNS = ["airspeed_m_s", "drag_N", "cl_level", "thrust_full_both_N", "thrust_full_one_N", "rate_of_climb_m_s"]
NOISE_COLUMNS = ["rpm", "BPF_hz", "tonal_dB", "broadband_dB", "one_prop_dB_at_1m", "aircraft_dB_at_1m", "aircraft_dB_at_100m"]
AUW, S, AR = 1.0, 0.17, 5.9                                                     # 09a: AUW ~1 kg, S 0.17 m^2, AR 5.9


@pytest.fixture(scope="module")
def drive():
    prop, sec, battery, pts, perf = wf.drive(AUW, S, AR, wf.RECORDED_POLAR["Cl"], wf.RECORDED_POLAR["Cd"])
    system = boreas.Propulsion(prop, sec, pr.motor(wf.DRIVE["motor"]), battery, rho=wf.DRIVE["rho"])
    return SimpleNamespace(prop=prop, system=system, pts=pts, perf=perf, W=AUW * wf.G)


def _plt():
    plt, axes = mock.MagicMock(name="plt"), [mock.MagicMock(name="ax0"), mock.MagicMock(name="ax1")]
    plt.subplots.return_value = (mock.MagicMock(name="fig"), axes)
    return plt, axes


# ------------------------------------------------------------------------------------------------ air-1 (09a cells 42, 49, 10)
def test_level_drag_is_cell_42(drive):
    g = dict(RHO=wf.DRIVE["rho"], W=drive.W, WING_AREA=S, CD0=drive.perf["cd0"], K_INDUCED=drive.perf["k_induced"])
    nb_drag = cell(A, 42, ["drag"], **g)["drag"]
    ours = wf.level_drag(g["W"], g["RHO"], g["WING_AREA"], g["CD0"], g["K_INDUCED"])
    for v in (*wf.SPEEDS, wf.V_CRUISE, np.float64(9.5)):
        assert ours(v) == nb_drag(v)                                            # (D, cl), exactly
    assert ours(wf.V_CRUISE)[0] == pytest.approx(drive.perf["drag_cruise_N"], rel=1e-12)   # drive()'s cruise drag


def test_speed_range_is_cell_49(drive, capsys):
    g = dict(RHO=wf.DRIVE["rho"], W=drive.W, WING_AREA=S, CD0=drive.perf["cd0"], K_INDUCED=drive.perf["k_induced"])
    nb_drag = cell(A, 42, ["drag"], **g)["drag"]
    solved, asked = {}, []

    def at_throttle(throttle, v):                                               # the real BEMT once per airspeed
        asked.append((throttle, v))
        if (throttle, v) not in solved:
            solved[throttle, v] = drive.system.at_throttle(throttle, v)
        return solved[throttle, v]

    system = SimpleNamespace(at_throttle=at_throttle)
    plt, axes = _plt()
    ns = cell(A, 49, None, system=system, drag=nb_drag, MOTORS=wf.DRIVE["motors"], W=drive.W, RHO=g["RHO"],
              WING_AREA=S, plt=plt)                                             # the whole cell, the plots to mocks
    printed = capsys.readouterr().out.splitlines()
    cell_asked, asked[:] = list(asked), []
    ours_drag = wf.level_drag(drive.W, g["RHO"], S, drive.perf["cd0"], drive.perf["k_induced"])
    table, env = wf.speed_range(system, ours_drag, W=drive.W, WING_AREA=S)
    assert asked == cell_asked and len(solved) == 19                            # the same 19 full-throttle points
    np.testing.assert_array_equal(wf.SPEEDS, ns["Vs"])
    assert list(table) == SPEED_COLUMNS
    assert all(type(x) is float and len(col) == 19 for col in table.values() for x in col)
    assert table["airspeed_m_s"] == ns["Vs"].tolist() and table["drag_N"] == ns["D"].tolist()
    assert table["thrust_full_one_N"] == ns["T_full"].tolist()
    assert table["thrust_full_both_N"] == (ns["MOTORS"] * ns["T_full"]).tolist()
    assert table["rate_of_climb_m_s"] == ns["roc"].tolist()
    assert table["cl_level"] == [nb_drag(v)[1] for v in ns["Vs"]]
    (v0, d0, *_), _ = axes[0].plot.call_args_list[0]                            # the drag curve the cell plots
    assert table["drag_N"] == list(d0) and table["airspeed_m_s"] == list(v0)
    eo = ns["eo"]
    assert env["stall_speed_m_s"] == ns["stall_v"] and env["cl_max_assumed"] == 1.1
    assert env["v_max_level_m_s"] == ns["v_max"] and env["best_climb_m_s"] == ns["roc"].max()
    assert env["best_climb_speed_m_s"] == ns["Vs"][ns["roc"].argmax()]
    assert env["engine_out_level_m_s"] == [ns["Vs"][eo.min()], ns["Vs"][eo.max()]] and env["sweep_m_s"] == [8.0, 26.0]
    assert printed == [                                                         # the cell's two lines, from our numbers
        f"stall speed (Cl_max 1.1 assumed): {env['stall_speed_m_s']:.1f} m/s | max level speed ~{env['v_max_level_m_s']:.0f} m/s"
        f" | best climb {env['best_climb_m_s']:.1f} m/s at {env['best_climb_speed_m_s']:.0f} m/s",
        "engine out: level flight possible between {:.0f} and {:.0f} m/s".format(*env["engine_out_level_m_s"])]
    i14 = table["airspeed_m_s"].index(wf.V_CRUISE)                              # 1 motor full at cruise = the engine-out point
    assert table["thrust_full_one_N"][i14] == pytest.approx(drive.pts["engine_out"].thrust, rel=1e-12)


def test_speed_range_without_level_flight_is_cell_49(capsys):
    """Too heavy for its thrust: the cell's nan maximum speed and 'NOT possible on one motor' are None here."""
    system = SimpleNamespace(at_throttle=lambda throttle, v: SimpleNamespace(thrust=0.2 + 0.0 * v))
    drag = wf.level_drag(30.0, 1.2, 0.17, 0.03, 0.07)
    plt, _ = _plt()
    ns = cell(A, 49, None, system=system, drag=drag, MOTORS=2, W=30.0, RHO=1.2, WING_AREA=0.17, plt=plt)
    assert capsys.readouterr().out.splitlines()[-1] == "engine out: NOT possible on one motor"
    table, env = wf.speed_range(system, drag, W=30.0, WING_AREA=0.17, RHO=1.2, MOTORS=2)
    assert math.isnan(ns["v_max"]) and env["v_max_level_m_s"] is None and env["engine_out_level_m_s"] is None
    assert table["rate_of_climb_m_s"] == ns["roc"].tolist() and env["best_climb_m_s"] == ns["roc"].max() < 0


def test_wing_loading_is_cell_10(drive):
    from assemblies.components import wing
    from assemblies.components.fixed_wing import FixedWing
    p0 = FixedWing().resolve()
    ns = cell(A, 10, ["b", "c0", "lam", "S", "AR", "V_CRUISE", "q"], p0=p0, RHO=wf.DRIVE["rho"])
    W = drive.W
    src = "".join(json.loads((NB / A).read_text())["cells"][10]["source"])
    shown = [ast.unparse(n.value) for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FormattedValue)]
    assert "W / S" in shown and "W / (q * S)" in shown                          # what the cell prints
    ours = wf.wing_loading(W, ns["S"])
    assert ours == {"wing_loading_N_m2": W / ns["S"], "cl_cruise": W / (ns["q"] * ns["S"])}
    assert ns["V_CRUISE"] == wf.V_CRUISE
    assert wing.WingSpec.from_params(p0).planform()["area_m2"] == pytest.approx(ns["S"], rel=1e-12)  # the workflow's S


# ------------------------------------------------------------------------------------------------ air-2 (09a cell 56)
def test_noise_table_is_cell_56(drive):
    nb_pts = {NB_POINTS[k]: drive.pts[k] for k in POINTS}
    plt, ax = mock.MagicMock(name="plt"), mock.MagicMock(name="ax")
    plt.subplots.return_value = (mock.MagicMock(name="fig"), ax)
    ns = cell(A, 56, None, prop=drive.prop, pts=nb_pts, MOTORS=wf.DRIVE["motors"], plt=plt)   # the whole cell
    assert (wf.NOISE_DIST, wf.NOISE_ANGLE) == (ns["DIST"], ns["ANGLE"])
    rows_nb, _ = wf.noise_table(drive.prop, nb_pts)                             # under the notebook's own point names
    assert rows_nb == ns["noise_rows"]
    rows, tones = wf.noise_table(drive.prop, drive.pts)                         # under the workflow's
    assert list(rows) == list(tones) == list(POINTS)
    for k in POINTS:
        assert list(rows[k]) == NOISE_COLUMNS == list(ns["noise_rows"][NB_POINTS[k]])
        assert rows[k] == ns["noise_rows"][NB_POINTS[k]]                        # the cell's numbers, exactly
        assert all(type(v) is float for v in rows[k].values())
    assert rows["engine_out"]["aircraft_dB_at_1m"] == rows["engine_out"]["one_prop_dB_at_1m"]      # one motor running
    assert rows["cruise"]["aircraft_dB_at_1m"] == rows["cruise"]["one_prop_dB_at_1m"] + 10 * math.log10(2)
    expected = ns["noise"].astype(float).rename(index={v: k for k, v in NB_POINTS.items()})
    pd.testing.assert_frame_equal(pd.DataFrame(rows).T.round(1), expected)
    (freq, spl), _ = ax.bar.call_args                                           # the cruise spectrum the cell plots
    assert ax.bar.call_count == 1
    assert tones["cruise"] == {"frequency_hz": [float(x) for x in freq], "spl_db": [float(x) for x in spl]}
    assert tones["engine_out"] == {"frequency_hz": [float(x) for x in ns["tones"]["frequency_hz"]],   # the cell's last pass
                                   "spl_db": [float(x) for x in ns["tones"]["spl_db"]]}
    assert all(len(t["frequency_hz"]) == len(t["spl_db"]) == 6 for t in tones.values())


# ------------------------------------------------------------------------------------------------ through run()
def test_root_records_envelope_and_noise_through_the_workflow(tmp_path, solvers):
    pytest.importorskip("cadquery")
    shutil.copy(DATA / "fixed_wing.vida", tmp_path / "f.vida")                  # the committed tree: nodes reused
    kw = dict(fidelity="full", run_cfd=False, run_fea=False, out=tmp_path / "runs", vida_path=tmp_path / "f.vida",
              export_path=tmp_path / "f.json", progress=False)
    first = wf.run(**kw)
    r = first.results
    for old in ("auw_kg", "polar_point", "polar_source", "polar", "points"):
        assert old in r                                                          # the keys there were, all still there
    committed = vida.load(DATA / "fixed_wing.vida").results
    assert r["points"] == committed["points"] and r["polar"] == committed["polar"]   # their values unchanged
    assert list(r["speed_range"]) == SPEED_COLUMNS and all(len(c) == 19 for c in r["speed_range"].values())
    sr, env = r["speed_range"], r["envelope"]
    i14 = sr["airspeed_m_s"].index(wf.V_CRUISE)
    assert sr["thrust_full_one_N"][i14] == pytest.approx(r["points"]["engine_out"]["thrust_N"], rel=1e-12)
    assert sr["drag_N"][i14] == pytest.approx(r["polar"]["drag_cruise_N"], rel=1e-12)
    W, area = r["auw_kg"] * wf.G, first.child("airframe").results["planform"]["area_m2"]
    assert env["stall_speed_m_s"] == pytest.approx(math.sqrt(2 * W / (wf.DRIVE["rho"] * area * 1.1)))
    assert env["wing_loading_N_m2"] == pytest.approx(W / area) and 8.0 <= env["best_climb_speed_m_s"] <= 26.0
    assert env["engine_out_level_m_s"][0] < env["engine_out_level_m_s"][1] <= env["v_max_level_m_s"]
    assert list(r["noise"]) == list(r["noise_tones"]) == list(POINTS)
    for k in POINTS:
        assert r["noise"][k]["rpm"] == r["points"][k]["rpm"] and list(r["noise"][k]) == NOISE_COLUMNS
    saved = vida.load(tmp_path / "f.vida").results
    for k in ("speed_range", "envelope", "noise", "noise_tones"):
        assert saved[k] == r[k]
    f = results.load(tmp_path / "f_results.json")
    assert f.table("", "speed_range").shape == (19, 6) and list(f.table("", "noise").index) == list(POINTS)
    assert f[""]["envelope"] == env and f[""]["noise_tones"]["cruise"] == r["noise_tones"]["cruise"]
    before = {p: n.status() for p, n in first.walk()}
    second = wf.run(**kw)                                                        # nothing new below the root
    after = {p: n.status() for p, n in second.walk()}
    assert after[""].startswith("computed") and all(s in ("reused", "NOT RUN", "-") for p, s in after.items() if p)
    assert {p for p, s in after.items() if s == "NOT RUN"} == {p for p, s in before.items() if s == "NOT RUN"}
    assert solvers.fea == [] and solvers.cfd == []
    for k in ("speed_range", "envelope", "noise", "noise_tones"):
        assert second.results[k] == r[k]
