"""Notebook data the submarine workflow records (audit gaps water W1, W3, W4, W5, W6, W8): the resistance curve, the drive
at every speed and the survey point, the dive thrust grid (13 cells 8, 13, 29; 14 cell 4), the pressure hull's
thin-shell theory, collapse, ring mode and pressure vs depth (13 cells 16, 26, 28), the blade's modes dry and wet and the
blade-pass margins (13 cells 25, 26), the open-water curves (14 cell 6) and map, noise and cavitation (13 cell 23), the
mass budget with positions (13 cell 6) and the wake labels. Each lift against the notebook cell run as it is over the
same inputs (whatever solves goes to the mocks, ``stubs.py``), then the new keys through ``run()`` with the solvers
mocked, in the .vida, in submarine_results.json and in submarine.json."""
from __future__ import annotations

import ast
import json
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import stubs
from vegeta import boreas, talos

from assemblies import DATA, results, vida
from assemblies.components import propeller as pr
from assemblies.workflows import submarine as S
from test_life import NB, cell

N13, N14 = "13_submarine.ipynb", "14_submarine_propeller.ipynb"
NB_NAMES = {"cruise": "cruise", "full": "top speed"}                  # the workflow's point -> 13's
SAVED = json.loads((DATA / "submarine_results.json").read_text())["nodes"]


def _last_expr(notebook: str, index: int, ns: dict, unwrap: int = 0, which: int = -1):
    """The value of the cell's last expression (``which``: its statement index) over ``ns``; ``unwrap`` drops that many
    trailing ``.round(...)``/``.T``."""
    src = "".join(json.loads((NB / notebook).read_text())["cells"][index]["source"])
    expr = ast.parse(src).body[which].value
    for _ in range(unwrap):
        expr = expr.func.value if isinstance(expr, ast.Call) else expr.value
    return eval(compile(ast.Expression(expr), f"{notebook}:{index}", "eval"), {"pd": pd, "np": np, **ns})


def _same(a, b) -> bool:
    """Equal: bools exactly, numbers to 1e-12."""
    if isinstance(b, (bool, np.bool_)) or isinstance(a, (bool, np.bool_)):
        return bool(a) == bool(b) and isinstance(a, (bool, np.bool_)) and isinstance(b, (bool, np.bool_))
    return a == pytest.approx(b, rel=1e-12)


def _plt(n_axes=1):
    plt = mock.MagicMock()
    axes = [mock.MagicMock() for _ in range(n_axes)]
    plt.subplots.return_value = (mock.MagicMock(), axes if n_axes > 1 else axes[0])
    return plt, axes


@pytest.fixture(scope="module")
def wf():
    """The workflow's drive on the committed hull (``submarine.drive``, about 10 s, once)."""
    h, p = SAVED["hull"]["results"], SAVED["hull"]["params"]["sub"]
    L, D, S_WET, S_BODY = p["length"] / 1000, p["diameter"] / 1000, h["wetted_surface_m2"], h["body_surface_m2"]
    prop, sec, battery, pts, perf = S.drive(L, D, S_WET, S_BODY)
    ns8 = cell(N13, 8, ["L_M", "D_M", "S_WET", "S_BODY", "resistance", "Vs", "R", "R_cruise", "_", "Cf_c", "k_form"],
               p=p, mv={"surface_area": S_WET * 1e6}, mb={"surface_area": S_BODY * 1e6}, RHO_W=S.RHO_W, NU_W=S.NU_W,
               V_CRUISE=S.V_CRUISE)
    return SimpleNamespace(prop=prop, sec=sec, battery=battery, pts=pts, perf=perf, p=p, L=L, D=D, S_WET=S_WET, S_BODY=S_BODY,
                           v_max=perf["top_speed_m_s"], ns8=ns8)


def test_constants_are_the_notebooks():
    ns = cell(N13, 2, ["RHO_W", "NU_W", "G", "DESIGN_DEPTH_M", "TEST_FACTOR", "V_CRUISE"])
    assert (ns["RHO_W"], ns["NU_W"], ns["G"]) == (S.RHO_W, S.NU_W, S.G)
    assert (ns["DESIGN_DEPTH_M"], ns["TEST_FACTOR"], ns["V_CRUISE"]) == (S.DESIGN_DEPTH_M, S.TEST_FACTOR, S.V_CRUISE)
    ns = cell(N13, 13, ["WAKE", "T_DED", "HOTEL_W"])
    assert (ns["WAKE"], ns["T_DED"], ns["HOTEL_W"]) == (S.DRIVE["wake"], S.DRIVE["thrust_deduction"], S.DRIVE["hotel_W"])
    ns = cell(N13, 23, ["CP_MIN", "DIST"])
    assert (ns["CP_MIN"], ns["DIST"]) == (S.CP_MIN, S.NOISE_DIST)
    assert cell(N13, 25, ["ADDED_MASS_FACTOR"])["ADDED_MASS_FACTOR"] == S.ADDED_MASS_FACTOR
    assert cell(N13, 28, ["P_ATM"])["P_ATM"] == S.P_ATM
    ns = cell(N14, 2, ["SPEEDS", "W_MEAN", "T_DED", "HOTEL_W"])
    assert ns["SPEEDS"]["survey"] == S.V_SURVEY and ns["SPEEDS"]["cruise"] == S.V_CRUISE
    assert ns["W_MEAN"] == pytest.approx(1 - S.DRIVE["wake"], abs=1e-15)          # 14's wake fraction is 1 - drive.wake
    assert (ns["T_DED"], ns["HOTEL_W"]) == (S.DRIVE["thrust_deduction"], S.DRIVE["hotel_W"])               # 14 copied them from 13


# ------------------------------------------------------------------------------------------------ W8: the mass budget
def test_mass_budget_is_cell_6():
    h = SAVED["hull"]["results"]
    mv = {"volume_mm3": h["vehicle_volume_L"] * 1e6, "center_of_mass_mm": h["CB_mm"], "surface_area_mm2": h["wetted_surface_m2"] * 1e6}
    mh = {"volume_mm3": h["items_kg"]["pressure hull (6061-T6, from CAD)"] / S.RHO_AL, "center_of_mass_mm": [250.0, 0.0, 0.0]}
    ns = cell(N13, 6, None, RHO_W=S.RHO_W, G=S.G, X_PH=S.X_PH, p=SAVED["hull"]["params"]["sub"], display=lambda *a: None,
              print=lambda *a: None, mv={"volume": mv["volume_mm3"], "center_of_mass": mv["center_of_mass_mm"],
                                         "surface_area": mv["surface_area_mm2"]},
              mh={"volume": mh["volume_mm3"], "center_of_mass": mh["center_of_mass_mm"]})
    ours = S.mass_and_trim(mv, mh)
    assert ours["mass_budget"] == ns["items"].to_dict(orient="index")                # 13 cell 33's mass_budget, exactly
    assert list(ours["mass_budget"]) == list(ns["items"].index)
    assert {n: b["mass_kg"] for n, b in ours["mass_budget"].items()} == ours["items_kg"]
    assert ours["CG_mm"][0] == ns["CG"][0] and ours["CG_mm"][2] == ns["CG"][2] and ours["BG_mm"] == ns["BG"]


# ------------------------------------------------------------------------------------------------ W1: the drive
def test_resistance_curve_is_cell_8(wf):
    ns, perf = wf.ns8, wf.perf
    rc = perf["resistance_curve"]
    assert list(rc["speed_m_s"]) == list(ns["Vs"])
    for i, k in enumerate(("total_N", "bare_body_N", "appendages_N")):
        assert np.allclose(rc[k], ns["R"][:, i], rtol=1e-12, atol=0), k
    assert np.allclose(rc["effective_power_W"], ns["R"][:, 0] * ns["Vs"], rtol=1e-12, atol=0)
    assert perf["Cf_cruise"] == pytest.approx(ns["Cf_c"], rel=1e-12)
    assert perf["form_factor_1pk"] == pytest.approx(1 + ns["k_form"], rel=1e-12)
    assert perf["Re_cruise"] == pytest.approx(S.V_CRUISE * ns["L_M"] / S.NU_W, rel=1e-15)
    assert perf["effective_power_cruise_W"] == pytest.approx(ns["R_cruise"] * S.V_CRUISE, rel=1e-12)
    assert perf["resistance_curve"]["total_N"][0] < perf["resistance_at_cruise_N"] < perf["resistance_curve"]["total_N"][-1]


def test_speed_sweep_and_table_are_cell_13(wf):
    """Cell 13 as it is (the top speed given: its bisection is the workflow's), the workflow's drive at the same speeds."""
    ns = cell(N13, 13, ["WAKE", "T_DED", "HOTEL_W", "prop", "section", "motor", "battery", "drive", "point_at", "cruise", "full",
                        "speeds", "pts", "endurance_h", "range_km"], resistance=wf.ns8["resistance"], RHO_W=S.RHO_W,
              V_CRUISE=S.V_CRUISE, V_MAX=wf.v_max)
    perf = wf.perf
    assert list(perf["speeds_m_s"]) == list(ns["speeds"]) and list(perf["endurance_h"]) == pytest.approx(list(ns["endurance_h"]), rel=1e-12)
    sweep = S.speed_sweep(ns["pts"])                                                   # the cell's 14 points as columns
    for k, v in sweep.items():
        assert perf[k] == pytest.approx(v, rel=1e-12), k
    assert perf["electrical_W"] == pytest.approx([pt.electrical_power for pt in ns["pts"]], rel=1e-12)   # what the cell plots
    assert perf["cruise_throttle"] == pytest.approx(ns["cruise"].throttle, rel=1e-12)
    table = _last_expr(N13, 13, {**ns, "V_CRUISE": S.V_CRUISE, "V_MAX": wf.v_max}, unwrap=1)       # the cell's table, unrounded
    cols = {"speed_m_s": "boat_speed_m_s", "prop_efficiency": "prop_efficiency", "motor_efficiency": "motor_efficiency"}
    for k, name in NB_NAMES.items():
        row = perf["drive_points"][k]
        for c, v in table[name].items():
            assert _same(row[cols.get(c, c)], v), (k, c)


def test_survey_point_is_14_cell_4_row(wf):
    """14 cell 4's table row, over 13's drive (14 recomputes its own section; the survey speed is its SPEEDS["survey"])."""
    ns = cell(N13, 13, ["WAKE", "T_DED", "HOTEL_W", "prop", "section", "motor", "battery", "drive", "point_at"],
              resistance=wf.ns8["resistance"], RHO_W=S.RHO_W)
    pts = {"survey": ns["point_at"](S.V_SURVEY), "cruise": wf.pts["cruise"], "top speed": wf.pts["full"]}
    speeds = {"survey": S.V_SURVEY, "cruise": S.V_CRUISE, "top speed": wf.v_max}
    table = _last_expr(N14, 4, {"pts": pts, "speeds": speeds, "W_MEAN": 1 - S.DRIVE["wake"]}, unwrap=1)
    for k, name in (("survey", "survey"), *NB_NAMES.items()):
        row = wf.perf["drive_points"][k]
        assert set(table[name].index) <= set(row)
        for c, v in table[name].items():
            assert _same(row[c], v), (k, c)
    assert wf.perf["drive_points"]["full"]["throttle"] == 1.0


def test_dive_thrust_is_cell_29(wf):
    """Cell 29's grid lines over the same (fake, fast) drive; ``propulsion()`` is the workflow's drive."""
    drive = SimpleNamespace(at_throttle=lambda thr, u: SimpleNamespace(thrust=12.0 * thr - 1.5 * u + 0.3 * thr * u ** 2))
    cruise = SimpleNamespace(throttle=wf.perf["cruise_throttle"])
    ns = cell(N13, 29, ["u_grid", "T_grid"], drive=drive, cruise=cruise, WAKE=S.DRIVE["wake"], T_DED=S.DRIVE["thrust_deduction"],
              V_MAX=wf.v_max)
    resistance = wf.ns8["resistance"]
    ours = S.dive_thrust(drive, resistance, cruise, wf.v_max)
    assert list(ours["speed_m_s"]) == list(ns["u_grid"])
    assert list(ours["net_thrust_N"]["cruise_throttle"]) == list(ns["T_grid"][cruise.throttle])
    assert list(ours["net_thrust_N"]["full_throttle"]) == list(ns["T_grid"][1.0])
    assert list(ours["resistance_N"]) == [resistance(max(u, 0.05))[0] for u in ns["u_grid"]]   # the cell's drag(u)
    assert ours["thrust_deduction"] == S.DRIVE["thrust_deduction"] and ours["cruise_throttle"] == cruise.throttle
    prop, sec, battery, d = S.propulsion()
    assert d.at_throttle(1.0, S.DRIVE["wake"] * wf.v_max).thrust == pytest.approx(wf.pts["full"].thrust, rel=1e-12)


# ------------------------------------------------------------------------------------------------ W3: the pressure hull
def test_pressure_hull_is_cells_16_26_28(wf):
    p = wf.p
    ns15 = cell(N13, 15, ["P_DESIGN", "AL", "R_I", "T_SH", "D_CAP", "L_PH"], p=p, RHO_W=S.RHO_W, G=S.G,
                DESIGN_DEPTH_M=S.DESIGN_DEPTH_M)
    assert ns15["AL"] == talos.Material(**S.AL6061)
    g = {k: ns15[k] for k in ("P_DESIGN", "AL", "R_I", "T_SH", "L_PH")}
    ns16 = cell(N13, 16, ["r_m", "hoop", "axial", "vm_theory", "dr_theory", "D_m", "p_yield", "p_buckle", "collapse"],
                **g, TEST_FACTOR=S.TEST_FACTOR, RHO_W=S.RHO_W, G=S.G)
    ns26 = cell(N13, 26, ["f_ring"], AL=g["AL"], r_m=ns16["r_m"])
    res = SimpleNamespace(metrics={"max_von_mises": 21.0})
    ns28 = cell(N13, 28, ["P_ATM", "HOTSPOT_PER_MPA", "table_depths", "pressure_table"], res=res, P_DESIGN=g["P_DESIGN"],
                DESIGN_DEPTH_M=S.DESIGN_DEPTH_M, TEST_FACTOR=S.TEST_FACTOR, RHO_W=S.RHO_W, G=S.G, AL=g["AL"],
                p_buckle=ns16["p_buckle"])
    ours = S.pressure_hull_theory(p, 21.0)
    assert ours["design_pressure_MPa"] == g["P_DESIGN"] and ours["mean_radius_mm"] == ns16["r_m"]
    assert ours["theory"] == {"hoop_MPa": ns16["hoop"], "axial_MPa": ns16["axial"], "von_mises_MPa": ns16["vm_theory"],
                              "radial_displacement_mm": ns16["dr_theory"]}               # 13 cell 33's "theory"
    assert ours["collapse"] == ns16["collapse"].to_dict(orient="index")               # the cell's table, row by row
    assert (ours["p_yield_MPa"], ours["p_buckle_MPa"]) == (ns16["p_yield"], ns16["p_buckle"])
    assert ours["collapse_depth_m"]["elastic_buckling"] == ns16["collapse"].loc["elastic buckling (Windenburg–Trilling)", "depth_m"]
    assert ours["test_depth_m"] == S.TEST_FACTOR * S.DESIGN_DEPTH_M
    assert ours["ring_mode_hz"] == ns26["f_ring"] and ours["hotspot_MPa_per_MPa"] == ns28["HOTSPOT_PER_MPA"]
    table = ns28["pressure_table"]
    assert [r["depth_m"] for r in ours["pressure_vs_depth"]] == list(table.index) == ns28["table_depths"]
    for r, (d, row) in zip(ours["pressure_vs_depth"], table.iterrows()):
        assert {k: v for k, v in r.items() if k != "depth_m"} == row.to_dict(), d
    # the audit's numbers (6061-T6, 180 mm, 6 mm wall, 500 mm cylinder) and 14 cell 24's hand copy of f_ring
    assert ours["p_yield_MPa"] == pytest.approx(19.2) and ours["p_buckle_MPa"] == pytest.approx(17.96, abs=0.01)
    assert ours["ring_mode_hz"] == pytest.approx(11400.0, rel=0.01)
    bare = S.pressure_hull_theory(p)                                                   # no FEA: the theory, no hotspot
    assert bare["hotspot_MPa_per_MPa"] is None and all(r["hull_hotspot_MPa"] is None and r["SF_yield"] is None
                                                        for r in bare["pressure_vs_depth"])
    assert [r["SF_buckling"] for r in bare["pressure_vs_depth"]] == list(table["SF_buckling"])


# ------------------------------------------------------------------------------------------------ W4: the blade's modes
def test_blade_modes_are_cell_25(tmp_path, solvers, wf):
    step = tmp_path / "blade.step"
    step.write_text("ISO-10303-21; a stand-in, hashed into the mesh key\n")
    names = ["hub_r", "hub_h", "R_tip", "BLADE_REGIONS", "T_blade", "F_tan", "blade_model", "blade_modes", "ADDED_MASS_FACTOR",
             "f_air", "f_water"]
    ns = cell(N13, 25, names, CAD_KW=cell(N13, 18, ["CAD_KW"])["CAD_KW"], bfiles=SimpleNamespace(artifacts={"step": step}),
              AL=talos.Material(**S.AL6061), full=wf.pts["full"], prop=wf.prop, RUNS=tmp_path)
    (theirs, workdir, n_modes), _ = solvers.solve_modes.call_args                    # the cell's modal solve, to the mock
    ps = pr.get(S.DRIVE["propeller"])
    ours = pr.blade_model(ps, step, pr.blade_loads(wf.prop, wf.pts["full"]), element_mm=S.ELEMENT["blade"]["full"],
                          material=S.AL6061, name="blade_top_speed", root_gap_mm=2.0)   # the workflow's, at full (2 mm)
    assert ours.config() == theirs.config() and ours.key == theirs.key
    assert n_modes == S.BLADE_MODES and workdir == tmp_path / "blade_modal"
    assert S.blade_modes(ns["f_air"]) == {"modes_hz_air": ns["f_air"], "added_mass_factor": ns["ADDED_MASS_FACTOR"],
                                          "modes_hz_water_assumed": ns["f_water"]}
    assert ns["f_air"] == stubs.MODES_HZ[:4]


def test_blade_margins_are_cell_26(wf):
    f_water = [f * S.ADDED_MASS_FACTOR for f in (60.0, 175.0, 420.0, 700.0)]      # cruise and top speed near different modes
    pts = {"cruise": wf.pts["cruise"], "top speed": wf.pts["full"]}
    ns = cell(N13, 26, ["blade_structure"], f_water=f_water)
    table = _last_expr(N13, 26, {"blade_structure": ns["blade_structure"], "prop": wf.prop, "cruise": pts["cruise"],
                                 "full": pts["top speed"]}, unwrap=1, which=-2)
    ours = S.blade_margins(wf.prop, wf.pts, f_water)
    assert ns["blade_structure"].damping_ratio == ours["damping_ratio"] == S.BLADE_DAMPING
    assert ours["modes_hz_water_assumed"] == f_water
    for k, name in NB_NAMES.items():
        assert ours["points"][k] == table.loc[name].to_dict(), k
    assert ours["points"]["cruise"]["nearest_blade_mode_hz"] != ours["points"]["full"]["nearest_blade_mode_hz"]


# ------------------------------------------------------------------------------------------------ W5: open water
def test_open_water_is_14_cell_6(wf):
    plt, (ax,) = _plt(1)
    ns = cell(N14, 6, None, plt=plt, prop=wf.prop, sec=wf.sec, pts={"cruise": wf.pts["cruise"], "top speed": wf.pts["full"]},
              RHO_W=S.RHO_W)
    ours = S.open_water(wf.prop, wf.sec, wf.pts["cruise"].rpm)
    assert ours["rpm"] == wf.pts["cruise"].rpm and ours["rho"] == S.RHO_W
    assert (ours["J"], ours["KT"], ours["KQ"], ours["eta0"]) == (ns["Js"], ns["KT"], ns["KQ"], ns["ETA"])
    assert ours["inflow_m_s"] == list(np.linspace(0.05, 1.3, 30) * ns["n_c"] * wf.prop.diameter)
    (J, KT), kw = ax.plot.call_args_list[0]
    assert kw["label"] == "K_T" and list(J) == ours["J"] and list(KT) == ours["KT"]


# ------------------------------------------------------------------------------------------------ W6: noise, cavitation
def test_noise_and_cavitation_are_cell_23(wf):
    plt, (ax,) = _plt(1)
    ns = cell(N13, 23, None, plt=plt, prop=wf.prop, cruise=wf.pts["cruise"], full=wf.pts["full"], V_CRUISE=S.V_CRUISE,
              V_MAX=wf.v_max, WAKE=S.DRIVE["wake"], DESIGN_DEPTH_M=S.DESIGN_DEPTH_M)
    rows, tones, cav = S.noise_rows(wf.prop, wf.pts, wf.v_max)
    assert list(rows) == list(wf.pts) and set(tones) == set(cav) == set(wf.pts)
    for k, name in NB_NAMES.items():
        assert list(rows[k]) == list(ns["noise_rows"][name])                         # the cell's columns, in its order
        assert rows[k] == ns["noise_rows"][name], k                                  # and its numbers, exactly
        V = S.V_CRUISE if k == "cruise" else wf.v_max
        for depth, key in (("5m", "sigma_at_5m"), ("rated_depth", "sigma_at_rated_depth")):
            full_check = boreas.cavitation(wf.prop, wf.pts[k].rpm, S.DRIVE["wake"] * V, S.CAV_DEPTHS_M[depth], boreas.SEA_WATER,
                                           cp_min=S.CP_MIN)
            assert cav[k][depth] == full_check and full_check["cavitation_number"] == rows[k][key]
    (freq, spl), _ = ax.bar.call_args                                                # the cruise spectrum the cell plots
    assert tones["cruise"] == {"frequency_hz": [float(x) for x in freq], "spl_db": [float(x) for x in spl]}
    assert all(len(t["frequency_hz"]) == S.NOISE_HARMONICS for t in tones.values())
    assert cav["full"]["5m"]["rpm_at_inception"] > wf.pts["full"].rpm                # no cavitation at top speed at 5 m


# ------------------------------------------------------------------------------------------------ through run()
def test_new_keys_through_the_workflow(tmp_path, solvers, wf, monkeypatch):
    """A tree saved before these keys existed (the committed one with the new keys and the solver results taken out, so
    the test does not depend on what the committed data holds): first run with the solvers off, then on, then again;
    then a tree whose blade FEA was solved before the modes existed and whose dive grid is stale. The drive is the
    fixture's (unchanged by run), the dive grid's drive a fast stand-in (counted), the open-water map a small grid."""
    pytest.importorskip("cadquery")
    monkeypatch.setattr(S, "drive", lambda L_m, D_m, S_wet, S_body: (wf.prop, wf.sec, wf.battery, wf.pts, wf.perf))
    made = []
    fake = SimpleNamespace(at_throttle=lambda thr, u: SimpleNamespace(thrust=12.0 * thr - 1.5 * u))
    monkeypatch.setattr(S, "propulsion", lambda: made.append(1) or (wf.prop, wf.sec, wf.battery, fake))
    monkeypatch.setattr(S, "MAP_RPM", np.array([400.0, 900.0, 2400.0]))
    monkeypatch.setattr(S, "MAP_V", np.array([0.0, 1.0, 3.0]))
    old = vida.load(DATA / "submarine.vida")                 # the saved hull without mass_budget, the root without its keys
    old.child("hull").results.pop("mass_budget", None)
    old.results = {}
    for k in ("computed_at", "reused", "not_run"):
        old.meta.pop(k, None)
    for n in ("hull_cfd", "pressure_hull_fea", "rotor_cfd", "blade_fea", "scene_cfd"):   # as written with the solvers off
        old.child(n).forget()
    old.save(tmp_path / "s.vida")
    kw = dict(fidelity="full", out=tmp_path / "runs", vida_path=tmp_path / "s.vida", export_path=tmp_path / "s.json",
              progress=False)

    # 1. solvers off: the analytic keys are there, the solver ones wait
    off = S.run(run_cfd=False, run_fea=False, **kw)
    r = off.results
    assert {"points", "performance"} <= set(r) and set(r["points"]) == {"cruise", "full"}             # the keys there were
    assert {"resistance_curve", "drive_points", "rpm", "torque_Nm", "Cf_cruise", "form_factor_1pk"} <= set(r["performance"])
    assert set(r["performance"]["drive_points"]) == {"survey", "cruise", "full"}
    assert r["dive_thrust"]["net_thrust_N"]["full_throttle"][0] == pytest.approx((12.0 - 1.5 * S.DRIVE["wake"] * 0.05) * 0.9)
    assert r["blade_structure"] is None and r["pressure_hull"]["hotspot_MPa_per_MPa"] is None
    assert r["pressure_hull"]["ring_mode_hz"] == pytest.approx(11364.2, abs=0.1)
    assert set(r["noise"]["points"]) == set(r["cavitation"]["points"]) == {"cruise", "full"}
    assert r["noise"]["reference"] == "dB re 1 uPa" and r["cavitation"]["depths_m"] == {"5m": 5.0, "rated_depth": 200.0}
    assert r["cavitation"]["inflow_m_s"]["full"] == pytest.approx(0.85 * wf.v_max)
    assert r["drive_wake"]["inflow_ratio"] == 0.85 and r["drive_wake"]["wake_fraction"] == 0.15
    assert r["open_water"]["rpm"] == wf.pts["cruise"].rpm and len(r["open_water"]["J"]) == 30
    assert list(r["open_water_map"]["rpm"]) == [400.0, 900.0, 2400.0] and r["open_water_map"]["rho"] == S.RHO_W
    hull = off.child("hull").results
    assert {n: b["mass_kg"] for n, b in hull["mass_budget"].items()} == pytest.approx(hull["items_kg"], rel=1e-12)
    assert hull["mass_budget"]["trim lead (computed)"]["x_mm"] == pytest.approx(hull["trim_lead_x_mm"], rel=1e-12)
    assert off.child("hull").status() == "computed"                                     # mass_budget added to the saved hull
    for n in ("blade_fea", "hull_cfd", "pressure_hull_fea"):
        assert off.child(n).status() == "NOT RUN" and not off.child(n).results
    exported = json.loads((tmp_path / "s.json").read_text())
    assert np.allclose(np.asarray(exported["map"]["thrust_n"], float), r["open_water_map"]["thrust_n"], rtol=1e-12, equal_nan=True)
    assert exported["rho"] == submarine.RHO_W and set(exported["points"]) == {"cruise", "full"}   # sea water (was the air default)
    assert solvers.solve_modes.call_count == 0 and made == [1]

    # 2. solvers on: the blade's modes (the mesh of the solved static case copied), the margins, the hotspot
    monkeypatch.setattr(talos.StructuralModel, "mesh_is_current",
                        lambda self, d: d.name == "blade_top_speed" or (d / "mesh.msh").is_file())
    d = tmp_path / "runs" / "blade_fea" / "blade_top_speed"
    d.mkdir(parents=True, exist_ok=True)
    (d / "mesh.msh").write_text("meshed\n")
    on = S.run(**kw)
    assert solvers.solve_modes.call_count == 1 and solvers.mesh.call_count == 0
    (bm, bdir, bn), _ = solvers.solve_modes.call_args
    static = next(m for c in solvers.solve_models.call_args_list for m in c.args[0] if m.name == "blade_top_speed")
    assert bn == S.BLADE_MODES and bm.name == "blade_top_speed" and bm._mesh_key() == static._mesh_key()
    assert bdir == tmp_path / "runs" / "blade_fea" / "modal" and (bdir / "mesh.msh").is_file()
    bl = on.child("blade_fea").results
    assert bl["modes_hz_air"] == stubs.MODES_HZ[:4] and bl["modes_hz_water_assumed"] == [f * 0.65 for f in stubs.MODES_HZ[:4]]
    assert bl["added_mass_factor"] == 0.65 and bl["complete"] and "modal" in on.child("blade_fea").files
    r = on.results
    assert r["blade_structure"]["modes_hz_water_assumed"] == bl["modes_hz_water_assumed"]
    assert set(r["blade_structure"]["points"]) == {"cruise", "full"}
    P = S.RHO_W * S.G * S.DESIGN_DEPTH_M / 1e6
    assert r["pressure_hull"]["hotspot_MPa_per_MPa"] == pytest.approx(stubs.FEA_METRICS["max_von_mises"] / P, rel=1e-12)
    assert made == [1]                                                                  # the dive grid kept with the root
    nodes = json.loads(results.path_for(tmp_path / "s.vida").read_text())["nodes"]
    assert nodes["blade_fea"]["results"]["modes_hz_air"] == bl["modes_hz_air"] and nodes["hull"]["results"]["mass_budget"]
    assert set(nodes[""]["results"]) >= {"dive_thrust", "pressure_hull", "open_water", "open_water_map", "cavitation", "noise",
                                        "blade_structure", "drive_wake"}
    assert nodes[""]["results"]["pressure_hull"]["pressure_vs_depth"][0]["SF_yield"] is None   # inf at the surface -> null
    q = results.load(results.path_for(tmp_path / "s.vida"))
    assert list(q.table("", "performance")) and len(q.table("hull", "mass_budget")) == len(hull["mass_budget"])

    # 3. again: nothing solved, every node of its own reused
    asked = (len(solvers.fea), len(solvers.cfd), solvers.solve_modes.call_count)
    again = S.run(**kw)
    assert (len(solvers.fea), len(solvers.cfd), solvers.solve_modes.call_count) == asked
    assert all(n.status() == "reused" for p, n in again.walk() if p), [(p, n.status()) for p, n in again.walk()]
    assert made == [1]

    # 4. a tree solved before the modes were recorded: only the modal solve runs, the static case is not solved again;
    #    its dive grid made for another drive (cruise throttle) is made again
    old = vida.load(tmp_path / "s.vida")
    for k in ("modes_hz_air", "added_mass_factor", "modes_hz_water_assumed"):
        old.child("blade_fea").results.pop(k)
    old.results["dive_thrust"]["cruise_throttle"] = 0.5 * wf.perf["cruise_throttle"]
    old.save(tmp_path / "s.vida")
    later = S.run(**kw)
    assert (len(solvers.fea), len(solvers.cfd)) == asked[:2] and solvers.solve_modes.call_count == asked[2] + 1
    assert later.child("blade_fea").results["modes_hz_air"] == stubs.MODES_HZ[:4]
    assert made == [1, 1] and later.results["dive_thrust"]["cruise_throttle"] == wf.perf["cruise_throttle"]
    stubs.check_calls(solvers, progress=False)
