"""Notebook data the Onager workflow records (audit gaps rovers onager-1 ... onager-6): each lift in
``components.onager_sizing`` and ``workflows.onager`` against the notebook cells it came from, the cell code read out of
the .ipynb and run as it is over the same inputs (plots, prints and displays to stand-ins; what MuJoCo would give, a
made-up episode), the result compared with what the notebook's export cell writes; then the new nodes and keys
through ``run()`` with the solvers mocked (``stubs.py``), twice: the second run asks no solver and reuses every node."""
from __future__ import annotations

import ast
import json
import math
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from vegeta import talos

from assemblies.components import (actuators as act, gait, onager_atlas_robot as oar, onager_atlas_scenario as oas,
                                   onager_manus_robot as omr, onager_manus_scenario as oms, onager_robot as orb,
                                   onager_sizing as sizing, onager_sweeper_cfd as cfd, onager_sweeper_controller as swc,
                                   onager_sweeper_robot as osr, onager_sweeper_scenario as oss, road_wheel as rw)
from assemblies.results import plain

NB = Path(__file__).resolve().parents[2] / "notebooks"
DATA = Path(__file__).resolve().parents[1] / "data"
N20, N21, N22, N23 = "20_onager_sentinel.ipynb", "21_onager_atlas.ipynb", "22_onager_manus.ipynb", "23_onager_sweeper.ipynb"
G = 9.81
SHORT_M = 20.0                       # terrains and roads cut to 20 m: the quarter cars in a fraction of a second


def _src(notebook: str, index: int) -> str:
    return "".join(json.loads((NB / notebook).read_text())["cells"][index]["source"])


def _plt():
    """``matplotlib.pyplot`` as a stand-in: ``plt.subplots`` gives a figure and axes that take any call."""
    plt = mock.MagicMock(name="plt")
    plt.subplots.return_value = (mock.MagicMock(name="fig"), mock.MagicMock(name="axes"))
    return plt


def run_cell(notebook: str, index: int, ns: dict, *, names=None, stop: str | None = None, override: dict | None = None) -> dict:
    """Run notebook ``notebook`` cell ``index`` statement by statement in ``ns`` (updated and returned): the whole cell,
    or only its function definitions and the assignments to ``names``; ``stop``: stop before the first top-level
    statement whose source starts with it; ``override`` ``{name: f}``: right after a statement assigns ``name``,
    ``ns[name] = f(ns[name])`` (a shorter road, as an input)."""
    src = _src(notebook, index)
    tree = ast.parse(src)
    base = {"math": math, "np": np, "pd": pd, "plt": _plt(), "display": lambda *a, **k: None, "tqdm": lambda x, **k: x,
            "print": lambda *a, **k: None}
    for k, v in base.items():
        ns.setdefault(k, v)

    def assigned(node):
        out = set()
        for t in (node.targets if isinstance(node, ast.Assign) else []):
            out |= {n.id for n in ast.walk(t) if isinstance(n, ast.Name)}
        return out
    for node in tree.body:
        if stop is not None and ast.get_source_segment(src, node).startswith(stop):
            break
        if names is not None and not ((isinstance(node, ast.FunctionDef) and node.name in names)
                                      or (isinstance(node, ast.Assign) and assigned(node) and assigned(node) <= set(names))):
            continue
        exec(compile(ast.Module([node], []), f"{notebook}:{index}", "exec"), ns)
        for name, f in (override or {}).items():
            if name in assigned(node):
                ns[name] = f(ns[name])
    return ns


def export(notebook: str, index: int, var: str, key: str, ns: dict):
    """The value the notebook's export cell writes under ``var[key]`` (the dict literal assigned to ``var``), evaluated
    over ``ns``."""
    tree = ast.parse(_src(notebook, index))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == var for t in node.targets):
            for k, v in zip(node.value.keys, node.value.values):
                if isinstance(k, ast.Constant) and k.value == key:
                    return eval(compile(ast.Expression(v), f"{notebook}:{index}", "eval"), ns)
    raise KeyError(f"{notebook} cell {index}: no {var}[{key!r}]")


def same(a, b) -> bool:
    """Equal as plain data (numpy to numbers, NaN to None)."""
    return plain(a) == plain(b)


# ------------------------------------------------------------------------------------------------ the inputs
@pytest.fixture(scope="module")
def sentinel():
    p = orb.design_params()
    cad = orb.cad_numbers(p)                                        # the recorded CAD numbers (no CAD rebuilt)
    budget = orb.mass_budget(p, cad)
    M = sum(budget.values())
    return SimpleNamespace(p=p, cad=cad, budget=budget, M=M, W=M * G, geo=orb.geometry(p))


def _variant(module):
    p = module.design_params()
    cad = module.cad_numbers(p)
    budget = module.mass_budget(p, cad)
    return SimpleNamespace(p=p, cad=cad, budget=budget, M=sum(budget.values()))


def stand_log(fz=(1150.0, 1130.0, 1290.0, 1310.0), com=(-0.048, 0.002, 1.02), z0=1.04, z1=1.025):
    """A made-up standing episode's log, as ChironLab writes it."""
    T = 5
    ff = np.zeros((T, 4, 3)); ff[:, :, 2] = np.asarray(fz)
    bp = np.zeros((T, 3, 3)); bp[:, 0, 2] = np.linspace(z0, z1, T)
    return {"foot_force": ff, "com": np.tile(np.asarray(com), (T, 1)), "body_pos": bp, "feet": ["FL", "FR", "RL", "RR"],
            "t": np.linspace(0.0, 2.0, T)}


class FakeLab:
    """What ``<variant>_lab(...)`` gives: ``run`` records its call and returns an episode with ``log``."""

    def __init__(self, log):
        self.log, self.calls = log, []

    def run(self, *a, **k):
        self.calls.append((a, k))
        return SimpleNamespace(log=self.log, save=lambda *a, **k: None)


# ------------------------------------------------------------------------------------------------ onager-2: the drive
def _nb20_drive(s, rear_share):
    """Notebook 20 cells 10, 11, 12, 14, 15, 17 (the corner), 26 and 27 run in one namespace, as the notebook runs them."""
    ns = {"act": act, "gait": gait, "orb": orb, "G": G, "M_TOTAL": s.M, "W": s.W, "geo": s.geo, "budget": s.budget,
          "REAR_SHARE": rear_share, "p": s.p}
    run_cell(N20, 2, ns, names=["SPEC", "G"])
    for i in (10, 11, 12, 14, 15):
        run_cell(N20, i, ns)
    run_cell(N20, 17, ns, names=["M_UNSPRUNG", "M_SPRUNG_CORNER", "LEVER_KNEE", "K_SUSP", "C_SUSP", "ZETA", "K_TYRE", "f_heave", "f_hop"])
    run_cell(N20, 26, ns)
    run_cell(N20, 27, ns)
    return ns


def test_drive_inputs_are_notebook_20s(sentinel):
    ns = _nb20_drive(sentinel, 0.53)
    d = sizing.SENTINEL_DRIVE
    assert d["V_SPEC_KMH"] == ns["SPEC"]["speed, wheels [km/h]"]
    for k in ("RHO_AIR", "C_D", "A_FRONT", "C_RR", "MU_TYRE", "GRADES_PCT", "E_BATT_WH", "USABLE", "ETA_DRIVE", "P_HOTEL", "BETA",
              "STANCE", "SWING_SPEED_FRACTION", "CROUCH"):
        assert d[k] == ns[k], k


def test_sentinel_drive_is_cells_10_to_27(sentinel):
    rear_share = 0.53                                               # cell 8's ChironLab number, made up here
    ns = _nb20_drive(sentinel, rear_share)
    mine = sizing.sentinel_drive(sentinel.M, sentinel.budget, sentinel.geo, sentinel.p, REAR_SHARE=rear_share)
    for key in ("wheel_mode", "walking_mode", "thermal"):                                         # cell 33's export
        assert same(mine[key], export(N20, 33, "summary_doc", key, ns)), key
    nb_stand_up = export(N20, 33, "summary_doc", "stand_up", ns)
    assert {k: mine["stand_up"][k] for k in nb_stand_up} == nb_stand_up
    assert mine["stand_up"]["peak_current_A"] == ns["su"].i.max()                                # cell 26's print
    assert set(mine["walking_mode"]["torques"]) == {"walk", "amble", "standing, rear wheel"}
    assert "standing, brakes off (knee)" in mine["thermal"] and mine["walking_mode"]["holding_power_W"] is not None


def test_sentinel_drive_without_the_stand(sentinel):
    full = sizing.sentinel_drive(sentinel.M, sentinel.budget, sentinel.geo, sentinel.p, REAR_SHARE=0.53)
    bare = sizing.sentinel_drive(sentinel.M, sentinel.budget, sentinel.geo, sentinel.p)
    assert bare["wheel_mode"] == full["wheel_mode"] and bare["stand_up"] == full["stand_up"]
    assert bare["walking_mode"]["holding_power_W"] is None
    assert bare["walking_mode"]["speed_limits"] == full["walking_mode"]["speed_limits"]
    assert bare["walking_mode"]["torques"] == {k: v for k, v in full["walking_mode"]["torques"].items() if k != "standing, rear wheel"}
    assert bare["thermal"] == {k: v for k, v in full["thermal"].items() if k != "standing, brakes off (knee)"}


# ------------------------------------------------------------------------------------------------ onager-3: terrains, pins
def _short(terrains):
    return {k: dict(v, length=SHORT_M) for k, v in terrains.items()}


def test_terrains_and_rock_strike_are_cells_17_22(sentinel):
    ns = _nb20_drive(sentinel, 0.53) | {"sag": 0.01}                 # sag: cell 8's, printed by cell 17
    run_cell(N20, 17, ns, override={"TERRAINS": _short})
    assert sizing.TERRAINS == {k: v for k, v in run_cell(N20, 17, {}, names=["TERRAINS"])["TERRAINS"].items()}
    assert sizing.K_TYRE_20 == ns["K_TYRE"]
    corner = sizing.sentinel_corner(sentinel.M, sentinel.budget, sentinel.geo)
    assert corner == {"m_unsprung_kg": ns["M_UNSPRUNG"], "m_sprung_corner_kg": ns["M_SPRUNG_CORNER"], "lever_knee_m": ns["LEVER_KNEE"],
                      "k_susp_N_m": ns["K_SUSP"], "c_susp_N_s_m": ns["C_SUSP"], "zeta": ns["ZETA"], "k_tyre_N_m": ns["K_TYRE"],
                      "f_heave_hz": ns["f_heave"], "f_hop_hz": ns["f_hop"]}
    t = sizing.sentinel_terrains(sentinel.M, sentinel.geo, corner, _short(sizing.TERRAINS))
    susp = export(N20, 33, "summary_doc", "suspension", ns)                                     # cell 33's export
    assert same(t["summary"], susp["terrains"])
    assert {k: corner[k] for k in ("k_susp_N_m", "zeta", "k_tyre_N_m", "f_heave_hz", "f_hop_hz")} == {k: v for k, v in susp.items() if k != "terrains"}
    # cell 22: the rock strike case (and the standing case from the stand)
    log = stand_log()
    ns.update(talos=talos, fz_stand=np.asarray(log["foot_force"])[-1, :, 2])
    run_cell(N20, 22, ns, names=["STEEL_PIN", "a1s", "a2s", "DROP_M", "S_LEG", "landing_mean", "rock", "i_rock", "slope_rock", "CASES"])
    assert t["rock_strike"] == ns["CASES"][sizing.ROCK_STRIKE]
    assert ns["CASES"][sizing.STANDING]["fz"] == sizing.stand_summary(log)["rear_wheel_max_N"]


def test_pins_are_cell_23(sentinel):
    p = sentinel.p
    cases = {"walk peak (β 0.75)": dict(fx=628.0, fy=0.0, fz=2093.5), sizing.ROCK_STRIKE: dict(fx=7514.6, fy=0.0, fz=7514.6),
             "cornering 0.4 g, outer wheel": dict(fx=0.0, fy=601.0, fz=1502.0), sizing.STANDING: dict(fx=0.0, fy=0.0, fz=1310.0)}
    ns = {"talos": talos, "p": p, "CASES": cases}
    run_cell(N20, 22, ns, names=["STEEL_PIN"])
    run_cell(N20, 19, ns, names=["t_low", "t_up"])
    run_cell(N20, 23, ns, stop="def rows")
    geom = sizing.pin_geometry(p)
    assert geom["yield_MPa"] == ns["STEEL_PIN"].yield_strength
    assert (geom["lever_axle_mm"], geom["lever_pin_mm"]) == (ns["lever_axle"], ns["lever_pin"])
    mine = sizing.sentinel_pins(cases, geom)
    hand = ns["hand"]
    for name, r in mine.items():
        a, k = hand[f"stub axle — {name}"], hand[f"knee pin — {name}"]
        assert (r["stub_axle_MPa"], r["stub_axle_SF"]) == (a["max_von_mises_MPa"], a["SF_yield"])
        assert (r["knee_pin_MPa"], r["knee_pin_SF"]) == (k["max_von_mises_MPa"], k["SF_yield"])
        assert a["load"] == f"{r['resultant_N']:.0f} N resultant"


# ------------------------------------------------------------------------------------------------ onager-4: the stand
@pytest.mark.parametrize("variant, notebook, index, names, robot, lab", [
    ("sentinel", N20, 8, ["robot", "lab_flat", "ep_stand", "fz_stand", "com_stand", "sag", "REAR_SHARE"], "onager", "onager_lab"),
    ("atlas", N21, 6, ["robot", "lab_flat", "ep_stand", "fz_stand", "com", "X_CG", "Z_CG"], "atlas", "atlas_lab"),
    ("manus", N22, 6, ["robot", "lab_flat", "ep_stand", "fz", "com"], "manus", "manus_lab"),
    ("sweeper", N23, 6, ["robot", "lab_flat", "ep_stand", "fz", "com"], "sweeper", "sweeper_lab")])
def test_stand_is_the_cells_standing_episode(variant, notebook, index, names, robot, lab, monkeypatch):
    from vegeta import chiron
    from assemblies.workflows import onager
    log = stand_log()
    labs = {"nb": FakeLab(log), "wf": FakeLab(log)}
    made = {}

    def builders(who):
        return (lambda cad=None, **k: made.setdefault(who, ("robot", cad)),
                lambda terrain, robot=None, **k: (made.setdefault(f"{who} lab", (terrain, robot)), labs[who])[1])
    mk_robot, mk_lab = builders("nb")
    module = {"sentinel": orb, "atlas": oar, "manus": omr, "sweeper": osr}[variant]
    fake = SimpleNamespace(**{robot: mk_robot, lab: mk_lab})
    alias = {"sentinel": "orb", "atlas": "oar", "manus": "omr", "sweeper": "osr"}[variant]
    monkeypatch.setattr(chiron, "Flat", lambda: "flat ground")
    stand = SimpleNamespace(Stand=lambda: "stand")
    ns = run_cell(notebook, index, {alias: fake, "chiron": chiron, "oc": stand, "cad_num": {"cad": 1}}, names=names)
    s = sizing.stand_summary(log)
    fz = ns.get("fz_stand", ns.get("fz"))
    com = ns.get("com_stand", ns.get("com"))
    assert s["corner_loads_N"] == dict(zip(log["feet"], fz.tolist())) and s["com_m"] == com.tolist()
    if variant == "sentinel":
        assert (s["rear_share"], s["sag_m"], s["rear_wheel_max_N"]) == (ns["REAR_SHARE"], ns["sag"], fz[2:].max())
    if variant == "atlas":
        assert s["cg_m"] == {"x": ns["X_CG"], "z": ns["Z_CG"]}
    # the workflow's episode builds and runs the same
    monkeypatch.setitem(onager.STAND, variant, builders("wf"))
    monkeypatch.setattr(onager.oc, "Stand", lambda: "stand")
    assert onager._stand(variant, {"cad": 1}) == s
    assert labs["wf"].calls == labs["nb"].calls and made["wf"] == made["nb"] and made["wf lab"] == made["nb lab"]
    assert getattr(module, robot) and getattr(module, lab)                       # the names the cell calls exist


def test_stand_builders_are_the_notebooks():
    from assemblies.workflows import onager
    assert onager.STAND == {"sentinel": (orb.onager, orb.onager_lab), "atlas": (oar.atlas, oar.atlas_lab),
                            "manus": (omr.manus, omr.manus_lab), "sweeper": (osr.sweeper, osr.sweeper_lab)}


# ------------------------------------------------------------------------------------------------ onager-1: the scenes' outcome
def _mission(ts, log, real, phases=None):
    """A scenario module whose ``run`` gives a made-up episode (``log``) and whose ``timeseries`` gives ``ts``."""
    ep = SimpleNamespace(log=log, save=lambda *a, **k: None)
    phases = phases if phases is not None else pd.DataFrame({"duration_s": {"drive": 5.0, "lift": 3.0}})
    return SimpleNamespace(Scene=real.Scene, make_lab=lambda scene, **k: "lab", run=lambda lab, scene, **k: ep,
                           timeseries=lambda ep: ts, phase_table=lambda ep: phases, in_basket=getattr(real, "in_basket", None),
                           collected=lambda ep: pd.DataFrame({"t": ep.log.get("collected", {})})), ep


MISSION_LOG = {"mission_finished": True, "events": [[12.5, "wire", "cut: both jaws (peak squeeze 9100 N)"]],
               "mission": [[0.0, "drive", "start"], [5.0, "drive", "done"], [5.0, "lift", "start"], [8.0, "lift", "timeout"]],
               "t": np.linspace(0.0, 49.98, 51)}


def _check_common(out, log):
    from assemblies.workflows._scene import plain as scene_plain
    assert out["finished"] is bool(log["mission_finished"]) and out["mission"] == log["mission"]
    assert out["duration_s"] == float(log["t"][-1]) and out["events"] == log["events"]
    json.dumps({k: scene_plain(v) for k, v in out.items()})                        # what run_scene records


def test_atlas_outcome_is_cells_20_24(tmp_path):
    from assemblies.workflows import onager
    ts = pd.DataFrame({"t": [0.0, 25.0, 50.0], "x": [0.0, 4.0, 8.0], "pallet_x": [3.0, 6.0, 11.02], "pallet_z": [0.07, 0.4, 0.071],
                       "pallet_tilt_deg": [0.1, 5.2, 0.4]})
    fake, ep = _mission(ts, dict(MISSION_LOG), oas)
    ns = run_cell(N21, 20, {"oas": fake, "robot": None, "RUNS": tmp_path})
    nb = export(N21, 24, "doc", "pallet_job", ns | {"movie": "movie.mp4"})
    out = onager._scene(fake, onager._atlas_outcome)(50.0)
    assert out["pallet_final"] == nb["pallet_final"] and out["finished"] == nb["finished"] and out["events"] == nb["events"]
    _check_common(out, ep.log)


def test_manus_outcome_is_cells_16_20(tmp_path):
    from assemblies.workflows import onager
    ts = pd.DataFrame({"t": [0.0, 30.0, 66.0], "x": [0.0, 6.0, 14.0], "log_z": [0.07, 0.62, 0.08], "log_y": [0.0, 0.3, 1.45]})
    fake, ep = _mission(ts, dict(MISSION_LOG, cut_at=12.5, cutter_history=np.zeros((0, 3))), oms)
    ns = run_cell(N22, 16, {"oms": fake, "robot": None, "RUNS": tmp_path, "orb": orb, "omr": omr})
    nb = export(N22, 20, "doc", "mission", ns | {"movie": "movie.mp4"})
    out = onager._scene(fake, onager._manus_outcome)(66.0)
    assert {k: out[k] for k in ("finished", "events", "log_max_height_m", "log_final_y_m")} == {
        k: nb[k] for k in ("finished", "events", "log_max_height_m", "log_final_y_m")}
    assert out["cut_at"] == ep.log["cut_at"]                                       # cell 17 marks it
    _check_common(out, ep.log)


def test_sweeper_outcome_is_cells_23_27(tmp_path):
    from assemblies.workflows import onager
    hz = osr.sweeper_geometry()["hull_z"]
    x0, y0, z0, x1, y1, z1 = osr.sweeper_geometry()["basket"]
    hull = np.array([18.0, 0.1, hz])
    props = ["brick", "box", "can_0"]
    pp = np.zeros((3, 3, 3))
    pp[-1, 0] = hull + [(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2]               # the brick in the basket
    pp[-1, 1] = hull + [1.0, 0.0, -hz + 0.07]                                       # the box left on the road
    bp, bq = np.zeros((3, 2, 3)), np.zeros((3, 2, 4))
    bp[:, 0] = hull; bq[:, 0, 0] = 1.0
    log = dict(MISSION_LOG, collected={"can_0": 7.1, "packet_1": 9.3}, props=props, prop_pos=pp, body_pos=bp, body_quat=bq)
    ts = pd.DataFrame({"t": [0.0, 95.0], "x": [0.0, 18.0]})
    fake, ep = _mission(ts, log, oss)
    ns = run_cell(N23, 23, {"oss": fake, "robot": None, "RUNS": tmp_path, "suction": dict(cfd.SUCTION)})
    nb = export(N23, 27, "doc", "mission", ns | {"movie": "movie.mp4"})
    out = onager._scene(fake, onager._sweeper_outcome)(95.0)
    assert out["vacuumed"] == nb["vacuumed"] and out["in_basket"] == nb["in_basket"] == {"brick": True, "box": False}
    assert out["n_litter"] == len(ns["scene"].litter) and out["finished"] == nb["finished"]
    _check_common(out, ep.log)


# ------------------------------------------------------------------------------------------------ onager-5: the variants' sizing
def test_atlas_drives_are_cell_12():
    a = _variant(oar)
    ns = {"act": act, "oar": oar, "orb": orb, "G": G, "cad_num": a.cad, "fl": oar.forklift_geometry()}
    run_cell(N21, 2, ns, names=["TARGETS", "G"])
    run_cell(N21, 10, ns, names=["m_load"])
    run_cell(N21, 12, ns)
    assert sizing.ATLAS["m_load"] == ns["m_load"] and sizing.ATLAS["FRICTION"] == ns["FRICTION"]
    assert sizing.atlas_drives(a.p, a.cad) == export(N21, 24, "doc", "drives", ns)


def test_atlas_load_chart_is_cells_8_10_18():
    a = _variant(oar)
    X_CG, Z_CG = -0.061, 1.087                                     # cell 6's ChironLab numbers, made up here
    ns = {"act": act, "oar": oar, "orb": orb, "oas": oas, "G": G, "M": a.M, "X_CG": X_CG, "Z_CG": Z_CG, "budget": a.budget}
    run_cell(N21, 2, ns, names=["TARGETS", "G"])
    run_cell(N21, 4, ns, names=["geo", "fl", "x_front"])
    run_cell(N21, 8, ns)
    run_cell(N21, 10, ns)
    n = len(rw.iso8608_profile(SHORT_M, 0.005, 256e-6, 5)[0])
    run_cell(N21, 18, ns, override={"x_road": lambda x: x[:n], "z_road": lambda z: z[:n]})
    assert np.array_equal(ns["z_road"], rw.iso8608_profile(SHORT_M, 0.005, 256e-6, 5)[1])     # a shorter road, the same road
    assert sizing.ATLAS["SF_TIP"] == ns["SF_TIP"] and sizing.ATLAS["K_TYRE"] == ns["K_TYRE"]
    assert sizing.ATLAS["road"][1:] == [0.005, 256e-6, 5]
    mine = sizing.atlas_load_chart(a.M, X_CG, Z_CG, a.budget, a.p, dict(sizing.ATLAS, road=[SHORT_M, 0.005, 256e-6, 5]))
    for key in ("load_chart", "stance", "ride"):
        assert same(mine[key], export(N21, 24, "doc", key, ns)), key


def test_manus_sizing_is_cells_8_10_12_14():
    m = _variant(omr)
    ns = {"act": act, "omr": omr, "orb": orb, "oms": oms, "G": G, "cad_num": m.cad, "p": m.p, "ag": omr.arm_geometry()}
    for i in (8, 10):
        run_cell(N22, i, ns)
    run_cell(N22, 12, ns, names=["Lu", "uw", "ud", "outboard", "lever_out"])
    run_cell(N22, 14, ns, names=["kp_sh", "I_out", "f_servo"])
    mine = sizing.manus_sizing(m.p, m.cad)
    for key in ("arm_torques_holding_log", "cutting", "shoulder_servo_mode_hz"):
        assert same(mine[key], export(N22, 20, "doc", key, ns)), key
    assert mine["heaviest_log_kg"] == ns["heaviest"]
    assert {k: tuple(v) for k, v in sizing.MANUS["WIRES"].items()} == ns["WIRES"] and sizing.MANUS["lever_out"] == ns["lever_out"]


def test_sweeper_sizing_is_cells_8_10_15_21():
    s = _variant(osr)
    ns = {"act": act, "omr": omr, "orb": orb, "osr": osr, "oss": oss, "osc": swc, "cfd": cfd, "G": G, "M": s.M, "cad_num": s.cad,
          "sg": osr.sweeper_geometry(), "ag": omr.arm_geometry(osr.design_params())}
    for i in (8, 10):
        run_cell(N23, i, ns)
    run_cell(N23, 13, ns, names=["suction"])
    run_cell(N23, 15, ns, names=["pick"])
    run_cell(N23, 21, ns, names=["m_disc", "e_disc", "m_imp", "grade", "f_broom", "f_fan", "F_broom", "e_fan", "F_fan", "k_tyre", "f_ride"])
    mine = sizing.sweeper_sizing(s.p, s.cad, s.M)
    for key in ("power_W", "endurance_h", "arms", "pickup"):
        assert same(mine[key], export(N23, 27, "doc", key, ns)), key
    assert mine["unbalance"] == {k: ns[k] for k in ("f_broom", "F_broom", "f_fan", "F_fan", "f_ride")}
    for k in ("N_bristle", "mu_bristle", "dp_downstream", "eta_fan", "eta_motor", "P_elec", "E_usable", "e_disc", "m_imp", "grade", "k_tyre"):
        assert sizing.SWEEPER[{"E_usable": "E_usable_Wh"}.get(k, k)] == ns[k], k


# ------------------------------------------------------------------------------------------------ onager-6: the sub-mass tables
def test_sub_masses_are_cell_6():
    from assemblies.workflows import onager
    a, m, s = _variant(oar), _variant(omr), _variant(osr)
    nb21 = run_cell(N21, 6, {"oar": oar, "cad_num": a.cad}, names=["fork_mass"])
    nb22 = run_cell(N22, 6, {"omr": omr, "cad_num": m.cad}, names=["arm"])
    nb23 = run_cell(N23, 6, {"osr": osr, "cad_num": s.cad}, names=["gear"])
    sa, sm, ss = (onager.sub_masses(v, x.p, x.cad) for v, x in (("atlas", a), ("manus", m), ("sweeper", s)))
    assert sa == {"sub_masses_kg": {"forklift_masses": nb21["fork_mass"]["mass [kg]"].to_dict()}, "counterweight": oar.COUNTERWEIGHT}
    assert sm["sub_masses_kg"]["arm_masses"] == nb22["arm"]["one arm [kg]"].drop("TOTAL").to_dict()       # 22 cell 20's arm_kg
    assert ss["sub_masses_kg"]["sweeper_masses"] == nb23["gear"].iloc[:, 0].drop("TOTAL").to_dict()       # 23 cell 27's sweeping_gear_kg
    assert ss["sub_masses_kg"]["arm_masses"] == omr.arm_masses(osr.design_params(), s.cad)                 # 23 cell 10's m_arm
    assert sum(sa["sub_masses_kg"]["forklift_masses"].values()) == a.budget["forklift (see forklift_masses)"]
    assert 2 * sum(sm["sub_masses_kg"]["arm_masses"].values()) == m.budget["arms 2x (see arm_masses)"]
    assert sum(ss["sub_masses_kg"]["sweeper_masses"].values()) == s.budget["sweeping gear (see sweeper_masses)"]
    assert onager.sub_masses("sentinel", orb.design_params(), orb.cad_numbers()) == {"sub_masses_kg": None, "counterweight": None}


# ------------------------------------------------------------------------------------------------ the tree, through the mocks
NEW = {"sentinel": ("stand", "drive", "terrains", "pins"), "atlas": ("stand", "sizing", "load_chart"), "manus": ("stand", "sizing"),
       "sweeper": ("stand", "sizing")}


@pytest.fixture
def quick(tmp_path, monkeypatch):
    """The committed ``onager.vida`` (its bodies reused: no CAD), the legs' STEP files as stand-ins, the terrains and
    the Atlas's road cut short."""
    import stubs
    from assemblies.workflows import onager

    def export_kept(make, params, out, name, formats=("step",)):
        out = Path(out); out.mkdir(parents=True, exist_ok=True)
        f = out / f"{name}.step"
        f.write_text("ISO-10303-21; stand-in\n")
        return {"step": f}, None
    monkeypatch.setattr(onager, "export_kept", export_kept)
    monkeypatch.setattr(sizing, "TERRAINS", _short(sizing.TERRAINS))
    monkeypatch.setattr(sizing, "ATLAS", dict(sizing.ATLAS, road=[SHORT_M, 0.005, 256e-6, 5]))
    shutil.copy(DATA / "onager.vida", tmp_path / "x.vida")
    return dict(fidelity="full", out=tmp_path / "o", vida_path=tmp_path / "x.vida", progress=False), stubs


def _had_results(tree, path) -> bool:
    try:
        return bool(tree.child(path).results)
    except KeyError:
        return False


def test_new_nodes_through_the_mocks(quick, solvers):
    """The new nodes and keys on the first run (computed, or reused when the copied data already has them), what
    needs the stand following the stand's results, every scene and stand without results simulated once; the second
    run asks no solver and reuses every node."""
    from assemblies import results, vida
    from assemblies.workflows import onager
    kw, stubs = quick
    before = vida.load(kw["vida_path"])
    root = onager.run(**kw)
    stubs.check_calls(solvers, progress=False)
    for v, names in NEW.items():
        b = root.child(f"{v}/body")
        assert b.status() in ("computed", "reused") and "sub_masses_kg" in b.results and "counterweight" in b.results
        for n in names:
            if (v, n) != ("atlas", "load_chart"):
                assert root.child(f"{v}/{n}").status() in ("computed", "reused"), (v, n)
        assert root.child(f"{v}/stand").results
    assert [c.name for c in root.child("sentinel").children] == ["body", "scene", "leg_fea", "stand", "drive", "terrains", "pins"]
    assert root.child("atlas/body").results["counterweight"] == oar.COUNTERWEIGHT
    assert set(root.child("atlas/body").results["sub_masses_kg"]) == {"forklift_masses"}
    s = root.child("sentinel")
    st = s.child("stand").results
    assert set(s.child("drive").results) == {"wheel_mode", "walking_mode", "stand_up", "thermal"}
    assert (s.child("drive").results["walking_mode"]["holding_power_W"] is None) == (st.get("rear_share") is None)
    assert set(s.child("terrains").results) == {"suspension", "summary", "rock_strike"}
    standing = {sizing.STANDING} if st.get("rear_wheel_max_N") is not None else set()
    assert set(s.child("pins").results["pins"]) == set(s.child("leg_fea").params["cases"]) | {sizing.ROCK_STRIKE} | standing
    lc = root.child("atlas/load_chart")
    if root.child("atlas/stand").results.get("com_m") is None:                  # the stubs' stand: no CG yet
        assert lc.status() == "-" and lc.results == {}
    else:
        assert set(lc.results) == {"load_chart", "stance", "ride"}
    assert set(root.child("atlas/sizing").results) == {"drives"}
    assert set(root.child("manus/sizing").results) == {"arm_torques_holding_log", "heaviest_log_kg", "cutting", "shoulder_servo_mode_hz"}
    assert set(root.child("sweeper/sizing").results) == {"power_W", "endurance_h", "arms", "pickup", "unbalance"}
    data = results.load(results.path_for(kw["vida_path"]))
    assert data["sentinel/pins"]["pins"][sizing.ROCK_STRIKE]["stub_axle_SF"] > 0 and "atlas/load_chart" in data
    fresh = [f"{v}/{n}" for v in NEW for n in ("scene", "stand") if not _had_results(before, f"{v}/{n}")]
    assert sorted(solvers.sim) == sorted(path.split("/")[1] for path in fresh)       # each one simulated once
    assert all(root.child(path).status() == "computed" for path in fresh)
    asked = (len(solvers.fea), len(solvers.cfd), len(solvers.sim))

    again = onager.run(**kw)
    assert (len(solvers.fea), len(solvers.cfd), len(solvers.sim)) == asked
    for v, names in NEW.items():
        for n in ("body", "scene") + names:
            if again.child(f"{v}/{n}").results:
                assert again.child(f"{v}/{n}").status() == "reused", (v, n)
    assert again.child("atlas/load_chart").status() == root.child("atlas/load_chart").status().replace("computed", "reused")


def test_the_stand_fills_what_needs_it(quick, solvers):
    """Once the stand has a ChironLab result, the Sentinel's standing rows, the standing pin case and the Atlas's load
    chart are computed (their parameters carry the stand's numbers); a run after that reuses them."""
    from assemblies.workflows import onager
    kw, stubs = quick
    onager.run(**kw)
    log = stand_log()

    def run_scene(node, episode, *, run):
        if node.results:
            return node
        if not run:
            return node.not_run("run_sim=False")
        return node.record(**(sizing.stand_summary(log) if node.name == "stand" else stubs.scene_result(node)))
    solvers.run_scene.side_effect = run_scene
    root = onager.run(redo=("stand",), **kw)
    s = root.child("sentinel")
    st = s.child("stand").results
    assert st["rear_share"] == sizing.stand_summary(log)["rear_share"]
    d = s.child("drive")
    assert d.status() == "computed" and d.params["rear_share"] == st["rear_share"]
    assert "standing, rear wheel" in d.results["walking_mode"]["torques"] and d.results["walking_mode"]["holding_power_W"]
    assert "standing, brakes off (knee)" in d.results["thermal"]
    assert s.child("pins").results["pins"][sizing.STANDING]["resultant_N"] == st["rear_wheel_max_N"]
    assert s.child("terrains").status() == "reused"                                            # independent of the stand
    lc = root.child("atlas/load_chart")
    assert lc.status() == "computed" and set(lc.results) == {"load_chart", "stance", "ride"}
    assert set(lc.results["load_chart"]) == {"400 mm", "500 mm", "600 mm", "800 mm"} and set(lc.results["ride"]) == {
        "empty, 3 m/s", "rated pallet, 1 m/s", "rated pallet, 2 m/s"}
    again = onager.run(**kw)
    for path in ("sentinel/stand", "sentinel/drive", "sentinel/pins", "atlas/stand", "atlas/load_chart"):
        assert again.child(path).status() == "reused", path


def test_redo_without_the_solver_stays_not_run(quick, solvers):
    """``--redo`` of a solver node whose solver is off leaves it NOT RUN, as the workflow did before the new nodes (they
    reuse only themselves): the Atlas's, Manus's and Sweeper's scenes after --redo scene --no-sim, every stand after
    --redo stand --no-sim (the Sentinel's too: it comes after leg_fea), the legs after --redo leg_fea --no-fea. The new
    analytic nodes are reused all the same."""
    from assemblies.workflows import onager
    kw, stubs = quick
    onager.run(**kw)                                              # the scenes, the stands and the legs solved (stubs)
    root = onager.run(redo=("scene", "stand"), run_sim=False, **kw)
    for v in ("atlas", "manus", "sweeper"):
        assert root.child(f"{v}/scene").status() == "NOT RUN", v
    for v in NEW:
        assert root.child(f"{v}/stand").status() == "NOT RUN", v
    assert root.child("sentinel/leg_fea").status() == "reused"
    for path in ("sentinel/drive", "sentinel/terrains", "sentinel/pins", "atlas/sizing", "manus/sizing", "sweeper/sizing"):
        assert root.child(path).status() == "reused", path
    root = onager.run(redo=("leg_fea",), run_sim=False, run_fea=False, **kw)
    assert root.child("sentinel/leg_fea").status() == "NOT RUN"
    for path in ("sentinel/drive", "sentinel/terrains", "sentinel/pins"):
        assert root.child(path).status() == "reused", path
