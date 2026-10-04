"""Notebook data the quadcopter workflow records on its root (audit gaps air-2, air-3): the throttle sweep (08 cell 55)
and the noise table with its tone spectra (08 cell 60), each against the notebook cell run as it is over the workflow's
own drive; then the new keys through ``run()`` with the solvers mocked, in the .vida and in quadcopter_results.json."""
from __future__ import annotations

import shutil
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from assemblies import DATA, results, vida
from assemblies.workflows import quadcopter as wq
from test_life import cell

Q = "08_quadcopter.ipynb"
POINTS = ("hover", "cruise", "full")
SWEEP_COLUMNS = ["throttle", "rpm", "thrust_N", "current_A", "electrical_W", "motor_eff", "current_limited"]


@pytest.fixture(scope="module")
def drive():
    prop, sec, system, pts = wq.drive_points(0.48)
    return SimpleNamespace(prop=prop, system=system, pts=pts)


def test_throttle_sweep_is_cell_55(drive):
    ns = cell(Q, 55, ["throttles", "sweep", "sw"], system=drive.system)          # the cell's sweep, its table
    np.testing.assert_array_equal(wq.THROTTLES, ns["throttles"])
    calls = []

    def sweep(throttles, airspeed):                                              # hand the lift the cell's own sweep
        calls.append((np.asarray(throttles), airspeed))
        return ns["sweep"]

    ours = wq.throttle_sweep(SimpleNamespace(sweep=sweep))
    assert len(calls) == 1 and np.array_equal(calls[0][0], ns["throttles"]) and calls[0][1] == 0.0
    assert list(ours) == SWEEP_COLUMNS
    pd.testing.assert_frame_equal(pd.DataFrame(ours), ns["sw"], check_exact=True)
    assert all(type(v) is float for k, col in ours.items() if k != "current_limited" for v in col)
    assert all(type(v) is bool for v in ours["current_limited"])
    assert ours["current_limited"][-1] is True and ours["current_limited"][0] is False   # where the motor limit starts


def test_noise_table_is_cell_60(drive):
    plt, ax = mock.MagicMock(), mock.MagicMock()
    plt.subplots.return_value = (mock.MagicMock(), ax)
    ns = cell(Q, 60, None, prop=drive.prop, hover=drive.pts["hover"], cruise=drive.pts["cruise"], punch=drive.pts["full"],
              MOTORS=wq.DRIVE["motors"], plt=plt)                                # the whole cell, the plot to a mock
    rows, tones = wq.noise_table(drive.prop, drive.pts)
    assert (wq.NOISE_DIST, wq.NOISE_ANGLE) == (ns["DIST"], ns["ANGLE"])
    assert list(rows) == list(ns["noise_rows"]) == list(POINTS)
    for name in POINTS:
        assert list(rows[name]) == list(ns["noise_rows"][name])                  # the cell's columns, in its order
        assert rows[name] == ns["noise_rows"][name]                              # and its numbers, exactly
        assert all(type(v) is float for v in rows[name].values())
    assert list(rows["hover"])[-3:] == ["4_rotors_dB_at_1m", "4_rotors_dB_at_10m", "4_rotors_dB_at_50m"]
    pd.testing.assert_frame_equal(pd.DataFrame(rows).T.round(1), ns["noise"].astype(float))
    (freq, spl), _ = ax.bar.call_args                                            # the hover spectrum the cell plots
    assert tones["hover"] == {"frequency_hz": [float(x) for x in freq], "spl_db": [float(x) for x in spl]}
    assert tones["full"] == {"frequency_hz": [float(x) for x in ns["tones"]["frequency_hz"]],   # the cell's last pass
                             "spl_db": [float(x) for x in ns["tones"]["spl_db"]]}
    assert set(tones) == set(POINTS) and all(len(t["frequency_hz"]) == len(t["spl_db"]) == 6 for t in tones.values())


def test_root_records_sweep_and_noise_through_the_workflow(tmp_path, solvers):
    pytest.importorskip("cadquery")
    shutil.copy(DATA / "quadcopter.vida", tmp_path / "q.vida")                   # the committed tree: nodes reused
    kw = dict(fidelity="full", run_cfd=False, run_fea=False, out=tmp_path / "runs", vida_path=tmp_path / "q.vida",
              export_path=tmp_path / "q.json", progress=False)
    first = wq.run(**kw)
    r = first.results
    for old in ("auw_kg", "points", "hover_endurance_min", "thrust_to_weight", "map", "hover_thrust_bemt_vs_cfd"):
        assert old in r                                                          # the keys there were, all still there
    assert list(r["throttle_sweep"]) == SWEEP_COLUMNS and all(len(c) == 17 for c in r["throttle_sweep"].values())
    assert r["throttle_sweep"]["rpm"][-1] == pytest.approx(r["points"]["full"]["rpm"])   # throttle 1.0 is the full point
    assert set(r["noise"]) == set(r["noise_tones"]) == set(POINTS)
    assert r["noise"]["hover"]["rpm"] == r["points"]["hover"]["rpm"] and "4_rotors_dB_at_50m" in r["noise"]["full"]
    saved = vida.load(tmp_path / "q.vida").results
    assert saved["throttle_sweep"] == r["throttle_sweep"] and saved["noise"] == r["noise"]
    q = results.load(tmp_path / "q_results.json")
    assert q.table("", "throttle_sweep").shape == (17, 7) and list(q.table("", "noise").index) == list(POINTS)
    assert q[""]["noise_tones"]["hover"]["frequency_hz"] == r["noise_tones"]["hover"]["frequency_hz"]
    before = {p: n.status() for p, n in first.walk()}
    second = wq.run(**kw)                                                        # nothing new below the root
    after = {p: n.status() for p, n in second.walk()}
    assert after[""] == "computed" and all(s in ("reused", "NOT RUN", "-") for p, s in after.items() if p)
    assert {p: s for p, s in after.items() if s == "NOT RUN"} == {p: s for p, s in before.items() if s == "NOT RUN"}
    assert solvers.fea == [] and solvers.cfd == []
    assert second.results["noise"] == r["noise"] and second.results["throttle_sweep"] == r["throttle_sweep"]
