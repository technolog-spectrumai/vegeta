"""Notebook data the boat workflow records (audit gaps water W2, W4, W5, W6, W8): the bracket's modes with the pod
(12 cell 18), the blade's modes dry and wet and the blade-pass margins (cell 51), the frequency table (cell 53), the
open-water curves (cell 36) and map, cavitation (cell 38) and noise (cell 55), the mass budget with positions (cell 7),
the hull's half of the double-body drag (cells 13, 31) and the wake labels. Each lift against the notebook cell run as it
is over the same inputs (whatever solves goes to the mocks, ``stubs.py``), then the new keys through ``run()`` with the
solvers mocked, in the .vida, in boat_results.json and in boat.json."""
from __future__ import annotations

import json
import shutil
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import stubs
from vegeta import boreas, talos

from assemblies import DATA, results, vida
from assemblies.components import hull as H, propeller as pr
from assemblies.components.survey_boat import SurveyBoat
from assemblies.workflows import boat
from test_life import cell

N = "12_boat_at_sea.ipynb"
NB_NAMES = {"cruise": "cruise", "full": "full speed", "bollard": "bollard pull"}      # the workflow's point -> the notebook's
SAVED = json.loads((DATA / "boat_results.json").read_text())["nodes"]


@pytest.fixture(scope="module")
def drive():
    """The boat's drive at its three points (12 cells 15, 35), the hull and the top speed as the committed tree has them."""
    spec = pr.get(boat.DRIVE["propeller"])
    prop, sec = spec.model(), spec.airfoil()
    battery = pr.battery(boat.DRIVE["battery"])
    d = boreas.Propulsion(prop, sec, pr.motor(boat.DRIVE["motor"]), battery, rho=boat.RHO_W)
    p = SAVED["hull"]["params"]["boat"]
    S_wet, perf = SAVED["hull"]["results"]["hydrostatics"]["wetted_surface_m2"], SAVED[""]["results"]["performance"]
    w, v_max = boat.DRIVE["wake"], perf["top_speed_m_s"]
    R = H.boat_resistance(boat.V_CRUISE, p["length"] / 1000, S_wet, form_factor=boat.FORM_FACTOR)[0]
    pts = {"cruise": d.for_thrust(R, w * boat.V_CRUISE), "full": d.at_throttle(1.0, w * v_max), "bollard": d.at_throttle(1.0, 0.0)}
    return SimpleNamespace(prop=prop, sec=sec, battery=battery, pts=pts, perf=perf, v_max=v_max, p=p,
                           nb_pts={NB_NAMES[k]: v for k, v in pts.items()})


def _plt(n_axes=1):
    plt = mock.MagicMock()
    axes = [mock.MagicMock() for _ in range(n_axes)]
    plt.subplots.return_value = (mock.MagicMock(), axes if n_axes > 1 else axes[0])
    return plt, axes


def test_constants_are_the_notebooks():
    ns = cell(N, 15, ["WAKE", "V_CRUISE"])
    assert (ns["WAKE"], ns["V_CRUISE"]) == (boat.DRIVE["wake"], boat.V_CRUISE)
    ns = cell(N, 35, ["WATER", "DEPTH_M", "CP_MIN"])
    assert ns["WATER"] == boreas.SEA_WATER and (ns["DEPTH_M"], ns["CP_MIN"]) == (boat.DEPTH_M, boat.CP_MIN)


# ------------------------------------------------------------------------------------------------ W8: the mass budget
def test_mass_budget_is_cell_7():
    p = SurveyBoat().resolve(part="boat")
    info = {"boat": {"volume_mm3": 2.41e6}, "bracket": {"volume_mm3": 2.43e5}}
    ns = cell(N, 7, ["RHO_PETG", "items", "M_KG", "CG"], p=p, boat=SimpleNamespace(volume=info["boat"]["volume_mm3"]),
              bracket=SimpleNamespace(volume=info["bracket"]["volume_mm3"]))
    ours = boat.mass_budget(info, p, [list(i) for i in boat.ITEMS])
    assert ns["RHO_PETG"] == boat.RHO_PETG and list(ours) == list(ns["items"].index)            # the cell's rows, in order
    table = pd.DataFrame(ours).T
    pd.testing.assert_frame_equal(table[["mass_g", "x_mm", "z_mm"]], ns["items"].astype(float), check_names=False)
    assert all(r["mass_kg"] == r["mass_g"] / 1000 for r in ours.values())
    m = np.array([r["mass_g"] for r in ours.values()])
    assert m.sum() / 1000 == pytest.approx(ns["M_KG"], rel=1e-15)
    assert (m * [r["x_mm"] for r in ours.values()]).sum() / m.sum() == pytest.approx(ns["CG"][0], rel=1e-15)


# ------------------------------------------------------------------------------------------------ W2: the bracket's modes
def test_bracket_modal_model_is_cell_18(tmp_path, solvers):
    p = SurveyBoat().resolve(part="boat")
    step = tmp_path / "bracket.step"
    step.write_text("ISO-10303-21; a stand-in, hashed into the mesh key\n")
    (tmp_path / "bracket_mesh").mkdir()
    ns = cell(N, 18, ["t_b", "w_b", "h_b", "BR_REGIONS", "bracket_model", "bracket_case", "modes"], p=p, shutil=shutil,
              RUNS=tmp_path, PETG_CF=talos.Material(**boat.PETG_CF), cad={"bracket": SimpleNamespace(artifacts={"step": step})})
    (theirs, workdir, n_modes), _ = solvers.solve_modes.call_args                   # the cell's modal solve, to the mock
    ours = boat.bracket_modal_model(step, p, boat.ELEMENT["bracket"]["full"])       # 3 mm at full: the cell's
    assert ours.config() == theirs.config() and ours.key == theirs.key
    assert n_modes == boat.BRACKET_MODES and workdir == tmp_path / "bracket_modal"
    assert [(m.region, m.mass) for m in ours.masses] == [("pod", 300e-6)] and not ours.loads
    assert ns["modes"].metrics["frequencies_hz"] == stubs.MODES_HZ[:6]


# ------------------------------------------------------------------------------------------------ W4: the blade's modes
def test_blade_modes_and_margins_are_cell_51(tmp_path, solvers, drive):
    step = tmp_path / "blade.step"
    step.write_text("ISO-10303-21; a stand-in\n")
    (tmp_path / "blade_fea").mkdir()
    ps = pr.get(boat.DRIVE["propeller"])
    model = pr.blade_model(ps, step, pr.blade_loads(drive.prop, drive.pts["bollard"]), element_mm=1.0, material=boat.PA12CF,
                           name="bollard", half_width_mm=25.0)
    printed = []
    ns = cell(N, 51, None, shutil=shutil, RUNS=tmp_path, blade_model=model, prop=drive.prop, pts=drive.nb_pts,
              tviz=mock.MagicMock(), print=lambda *a: printed.append(a))                  # the whole cell, solve_modes mocked
    (m, workdir, n_modes), _ = solvers.solve_modes.call_args
    assert m is model and n_modes == boat.BLADE_MODES and workdir == tmp_path / "blade_modal"
    ours = boat.blade_modes(ns["f_air"])
    assert ours == {"modes_hz_air": ns["f_air"], "added_mass_factor": ns["ADDED_MASS_FACTOR"],
                    "modes_hz_water_assumed": ns["f_water"]}
    margin = boat.blade_pass_margin(drive.prop, drive.pts, ours["modes_hz_water_assumed"])
    assert {NB_NAMES[k]: v for k, v in margin["blade_pass_hz"].items()} == ns["bpf"]
    assert margin["first_wet_mode_hz"] == ns["f_water"][0]
    shown = printed[1][3]                                                             # the cell's margins, as it prints them
    assert {NB_NAMES[k]: f"{v:.0%}" for k, v in margin["margin_to_first_wet_mode"].items()} == shown


# ------------------------------------------------------------------------------------------------ W2 + W4: cell 53
def test_frequency_table_is_cell_53(drive):
    bracket_hz, f_water = stubs.MODES_HZ[:6], boat.blade_modes(stubs.MODES_HZ[:4])["modes_hz_water_assumed"]
    margin = boat.blade_pass_margin(drive.prop, drive.pts, f_water)
    bpf = {NB_NAMES[k]: v for k, v in margin["blade_pass_hz"].items()}
    plt, _ = _plt()
    ns = cell(N, 53, None, plt=plt, prop=drive.prop, cruise=drive.pts["cruise"], full=drive.pts["full"],
              bollard=drive.pts["bollard"], pts=drive.nb_pts, modes=SimpleNamespace(metrics={"frequencies_hz": bracket_hz}),
              f_water=f_water, bpf=bpf)
    ours = boat.frequency_table(bracket_hz, f_water, drive.prop, drive.pts["cruise"].rpm, margin["blade_pass_hz"]["cruise"])
    ft = ns["freq_table"]
    assert list(ours) == list(ft.index)
    assert [r["hz"] for r in ours.values()] == [float(h) for h in ft["hz"]]
    assert [r["nearest_line"] for r in ours.values()] == list(ft["nearest_line"])
    assert [f"{r['margin_to_BPF_at_cruise']:.0%}" for r in ours.values()] == list(ft["margin_to_BPF_at_cruise"])
    assert {r["nearest_line"] for r in ours.values()} == {"1P", "BPF"}                 # both lines picked: the test bites


# ------------------------------------------------------------------------------------------------ W5: open water
def test_open_water_is_cell_36(drive):
    plt, (a0, a1, a2) = _plt(3)
    cell(N, 36, None, plt=plt, prop=drive.prop, section=drive.sec, WATER=boreas.SEA_WATER)
    ours = boat.open_water(drive.prop, drive.sec)
    assert ours["rho"] == boreas.SEA_WATER.density and list(ours["vs_J"]) == ["1500", "2500", "3500"]
    for (eta, ct), (rpm, c) in zip(zip(a0.plot.call_args_list, a1.plot.call_args_list), ours["vs_J"].items()):
        assert eta.kwargs["label"] == ct.kwargs["label"] == f"{rpm} rpm"
        assert c["J"] == list(eta.args[0]) == list(ct.args[0])
        assert c["efficiency"] == list(eta.args[1]) and c["ct"] == list(ct.args[1])
        assert c["inflow_m_s"] == list(np.linspace(0.2, 6, 25))
    assert len(a2.plot.call_args_list) == 3
    for call, (v, thrust) in zip(a2.plot.call_args_list, ours["thrust_vs_rpm"]["thrust_N"].items()):
        assert call.kwargs["label"] == f"{v} m/s inflow"
        assert ours["thrust_vs_rpm"]["rpm"] == list(call.args[0]) and thrust == list(call.args[1])


# ------------------------------------------------------------------------------------------------ W6: cavitation, noise
def test_cavitation_is_cell_38(drive):
    rpms = cell(N, 36, ["rpms"])["rpms"]
    ns = cell(N, 38, ["cav", "incept"], prop=drive.prop, rpms=rpms, WAKE=boat.DRIVE["wake"], V_CRUISE=boat.V_CRUISE,
              DEPTH_M=boat.DEPTH_M, WATER=boreas.SEA_WATER, CP_MIN=boat.CP_MIN, full=drive.pts["full"], V_MAX=drive.v_max)
    ours = boat.cavitation_sweep(drive.prop)
    cav = ns["cav"]
    assert ours["rpm"] == list(cav.index) and set(ours) == {"rpm", *cav.columns}
    for k in cav.columns:
        assert ours[k] == list(cav[k]), k
    _, _, points = boat.noise_rows(drive.prop, drive.pts, drive.v_max)
    assert points["full"] == ns["incept"]                                              # full speed: the cell's incept
    assert ns["incept"]["cavitates"] is True and points["cruise"]["cavitates"] is False


def test_noise_is_cell_55(drive):
    plt, (ax,) = _plt(1)
    ns = cell(N, 55, None, plt=plt, prop=drive.prop, pts=drive.nb_pts, WATER=boreas.SEA_WATER, WAKE=boat.DRIVE["wake"],
              V_CRUISE=boat.V_CRUISE, V_MAX=drive.v_max, DEPTH_M=boat.DEPTH_M, CP_MIN=boat.CP_MIN)
    assert ns["DIST"] == boat.NOISE_DIST
    rows, tones, cav = boat.noise_rows(drive.prop, drive.pts, drive.v_max)
    assert list(rows) == list(drive.pts) and set(tones) == set(cav) == set(drive.pts)
    for k, name in NB_NAMES.items():
        assert list(rows[k]) == list(ns["rows"][name])                                 # the cell's columns, in its order
        assert rows[k] == ns["rows"][name], k                                          # and its numbers, exactly
    (freq, spl), _ = ax.bar.call_args                                                  # the cruise spectrum the cell plots
    assert tones["cruise"] == {"frequency_hz": [float(x) for x in freq], "spl_db": [float(x) for x in spl]}
    assert all(len(t["frequency_hz"]) == boat.NOISE_HARMONICS for t in tones.values())
    pd.testing.assert_frame_equal(pd.DataFrame(rows).T.rename(index={k: v for k, v in NB_NAMES.items()}).round(1),
                                  pd.DataFrame(ns["rows"]).T.round(1))


# ------------------------------------------------------------------------------------------------ through run()
SOLVED = ("hull_cfd", "hull_fea", "bracket_fea", "rotor_cfd", "blade_fea", "scene_cfd")


def test_new_keys_through_the_workflow(tmp_path, solvers, drive, monkeypatch):
    """The committed tree, put back to what it was before these keys (no mass budget, nothing solved): a run with the
    solvers off; then on with the modal solves failing; then with run_fea off on static cases solved without modes; then
    on (only the modal solves run); then again (nothing runs). The drive is the fixture's (``boat.drive`` takes 17 s and
    is unchanged); the open-water curves are computed once and the map on a 3 x 3 grid, to keep the five runs short."""
    pytest.importorskip("cadquery")
    monkeypatch.setattr(boat, "drive", lambda S_wet, length_mm: (drive.prop, drive.sec, drive.battery, drive.pts, drive.perf))
    real_open_water, seen = boat.open_water, {}

    def open_water_once(prop, section):
        if "curves" not in seen:
            seen["curves"] = real_open_water(prop, section)
        return seen["curves"]
    monkeypatch.setattr(boat, "open_water", open_water_once)
    monkeypatch.setattr(boat, "MAP_RPM", np.array([1000.0, 3000.0, 5000.0]))
    monkeypatch.setattr(boat, "MAP_V", np.array([0.0, 1.5, 3.0]))
    shutil.copy(DATA / "boat.vida", tmp_path / "b.vida")
    t = vida.load(tmp_path / "b.vida")                     # whatever is committed: the state the steps below start from
    t.child("hull").results.pop("mass_budget", None)
    for n in SOLVED:
        t.child(n).forget()
    t.save(tmp_path / "b.vida")
    kw = dict(fidelity="full", out=tmp_path / "runs", vida_path=tmp_path / "b.vida", export_path=tmp_path / "b.json",
              progress=False)

    # 1. solvers off: the analytic keys are there, the solver ones wait
    off = boat.run(run_cfd=False, run_fea=False, **kw)
    r = off.results
    assert {"points", "performance"} <= set(r)                                          # the keys there were
    assert r["frequency_table"] is None and r["blade_pass_margin"] is None
    assert set(r["noise"]["points"]) == set(r["cavitation"]["points"]) == {"cruise", "full", "bollard"}
    assert r["noise"]["reference"] == "dB re 1 uPa" and r["cavitation"]["depth_m"] == 0.15
    assert r["cavitation"]["inflow_m_s"]["full"] == pytest.approx(0.9 * drive.v_max)
    assert r["drive_wake"]["inflow_ratio"] == 0.9 and r["drive_wake"]["wake_fraction"] == 0.1
    assert r["open_water"] == seen["curves"]
    assert list(r["open_water_map"]["rpm"]) == [1000.0, 3000.0, 5000.0] and r["open_water_map"]["rho"] == boat.RHO_W
    hull = off.child("hull").results
    assert {n: b["mass_g"] for n, b in hull["mass_budget"].items()} == pytest.approx(hull["items"], rel=1e-9)
    assert off.child("hull").status() == "computed"                                     # mass_budget added to the saved hull
    for n in SOLVED:
        assert off.child(n).status() == "NOT RUN" and not off.child(n).results, n
    exported = json.loads((tmp_path / "b.json").read_text())
    assert exported["map"]["thrust_n"] == r["open_water_map"]["thrust_n"]
    assert exported["rho"] == boat.RHO_W     # the drive and the map are in sea water (it said 1.225, boreas' air default)
    assert solvers.solve_modes.call_count == 0 and not solvers.fea and not solvers.cfd

    # 2. solvers on, the modal solves failing: the static cases are solved, the modes are noted and not recorded
    monkeypatch.setattr(talos.StructuralModel, "mesh_is_current",
                        lambda self, d: d.name in ("thrust", "bollard") or (d / "mesh.msh").is_file())
    for d in ("bracket_fea/thrust", "blade_fea/bollard"):
        (tmp_path / "runs" / d).mkdir(parents=True, exist_ok=True)
        (tmp_path / "runs" / d / "mesh.msh").write_text("meshed\n")
    modes_ok = solvers.solve_modes.side_effect
    solvers.solve_modes.side_effect = lambda *a, **k: talos.Result(kind="talos.modes").fail("boom")
    failed = boat.run(**kw)
    assert solvers.solve_modes.call_count == 2 and solvers.mesh.call_count == 0
    for n, key in (("bracket_fea", "modes_hz"), ("blade_fea", "modes_hz_air")):
        node = failed.child(n)
        assert node.results["complete"] and key not in node.results and "modal" not in node.files, n
        assert node.meta["not_run"] == ["modes: boom"] and node.status() == "computed (part NOT RUN)", n
    assert failed.results["frequency_table"] is None and failed.results["blade_pass_margin"] is None
    solved = (len(solvers.fea), len(solvers.cfd))
    assert solved[0] > 0 and solved[1] > 0

    # 3. run_fea off on the static cases solved without modes: noted, still reused, nothing solved
    solvers.solve_modes.side_effect = modes_ok
    no_fea = boat.run(run_fea=False, **kw)
    assert (len(solvers.fea), len(solvers.cfd), solvers.solve_modes.call_count) == (*solved, 2)
    for n, key in (("bracket_fea", "modes_hz"), ("blade_fea", "modes_hz_air")):
        node = no_fea.child(n)
        assert node.status() == "reused" and node.meta["not_run"] == ["modes: run_fea=False"], n
        assert node.results["complete"] and key not in node.results, n
    assert no_fea.results["frequency_table"] is None

    # 4. solvers on: only the two modal solves run (in the meshes copied from the solved static cases), nothing static
    on = boat.run(**kw)
    assert (len(solvers.fea), len(solvers.cfd)) == solved
    assert solvers.solve_modes.call_count == 4 and solvers.mesh.call_count == 0
    (bm, bdir, bn), (lm, ldir, ln) = (c.args for c in solvers.solve_modes.call_args_list[-2:])
    thrust = next(m for c in solvers.solve_models.call_args_list for m in c.args[0] if m.name == "thrust")
    assert bn == 6 and len(bm.masses) == 1 and not bm.loads and bm._mesh_key() == thrust._mesh_key()
    assert ln == 4 and lm.name == "bollard" and (bdir / "mesh.msh").is_file() and (ldir / "mesh.msh").is_file()
    bf, bl, hc = (on.child(n).results for n in ("bracket_fea", "blade_fea", "hull_cfd"))
    assert bf["modes_hz"] == stubs.MODES_HZ[:6] and bf["pod_mass_kg"] == 0.3 and bf["complete"]
    assert bl["modes_hz_air"] == stubs.MODES_HZ[:4] and bl["modes_hz_water_assumed"] == [f * 0.65 for f in stubs.MODES_HZ[:4]]
    assert hc["half_drag_N"] == hc["forces"]["drag_force_N"] / 2 and hc["speed_m_s"] == boat.V_CFD
    assert "modal" in on.child("bracket_fea").files and "modal" in on.child("blade_fea").files
    assert not on.child("bracket_fea").meta.get("not_run") and on.child("bracket_fea").status() == "computed"
    r = on.results
    assert r["blade_pass_margin"]["first_wet_mode_hz"] == bl["modes_hz_water_assumed"][0]
    assert len(r["frequency_table"]) == 6 + 4
    nodes = json.loads(results.path_for(tmp_path / "b.vida").read_text())["nodes"]
    assert nodes["bracket_fea"]["results"]["modes_hz"] == bf["modes_hz"] and nodes["hull"]["results"]["mass_budget"]
    assert set(nodes[""]["results"]) >= {"open_water", "open_water_map", "cavitation", "noise", "blade_pass_margin",
                                        "frequency_table", "drive_wake"}

    # 5. again: nothing solved, every node of its own reused
    again = boat.run(**kw)
    assert (len(solvers.fea), len(solvers.cfd), solvers.solve_modes.call_count) == (*solved, 4)
    assert all(n.status() == "reused" for p, n in again.walk() if p), [(p, n.status()) for p, n in again.walk()]
    stubs.check_calls(solvers, progress=False)
