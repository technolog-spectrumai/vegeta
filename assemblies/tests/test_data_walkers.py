"""The walkers' notebook data (notebooks 16-19) against the cells they came from: the cell code is read out of the
.ipynb and run as it is over the same inputs, next to the function lifted into ``workflows/walkers.py``; then the
workflow, through the solver mocks, records the new keys and reuses every node on a second run."""
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

pytest.importorskip("cadquery")

from assemblies import DATA, results  # noqa: E402
from assemblies.components import actuators as act, apheloria, gait, myropod, myropod_robot as mr  # noqa: E402
from assemblies.components import robot_dog_robot as rdr  # noqa: E402
from assemblies.components.robot_dog import RobotDog  # noqa: E402
from assemblies.workflows import walkers  # noqa: E402

NB = Path(__file__).resolve().parents[2] / "notebooks"
D, P, C, A = "16_robot_dog.ipynb", "17_myropod_persephone.ipynb", "18_myropod_cleopatra.ipynb", "19_myropod_apheloria.ipynb"
G = 9.81


class _Plt:
    """matplotlib for a cell that also draws: every figure is a mock."""

    @staticmethod
    def subplots(*a, **k):
        return mock.MagicMock(), mock.MagicMock()


def cell(notebook: str, index: int, names=None, **globals_) -> dict:
    """Run notebook ``notebook`` cell ``index`` over ``globals_``: the whole cell, or only its function definitions and
    assignments to ``names``; returns the namespace (``print``/``display`` silenced, ``plt`` a mock)."""
    src = "".join(json.loads((NB / notebook).read_text())["cells"][index]["source"])
    tree = ast.parse(src)
    ns = {"math": math, "np": np, "pd": pd, "plt": _Plt(), "print": lambda *a, **k: None, "display": lambda *a, **k: None,
          **globals_}
    if names is None:
        body = tree.body
    else:
        def assigned(node):
            out = set()
            for t in (node.targets if isinstance(node, ast.Assign) else []):
                out |= {n.id for n in ast.walk(t) if isinstance(n, ast.Name)}
            return out
        body = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in names)
                or (isinstance(n, ast.Assign) and assigned(n) and assigned(n) <= set(names))]
    exec(compile(ast.Module(body, []), f"{notebook}:{index}", "exec"), ns)
    return ns


def close(a, b) -> bool:
    if isinstance(b, (bool, np.bool_, str)) or b is None:
        return a == b
    if isinstance(b, float) and math.isnan(b):
        return a is None or math.isnan(a)
    return a == pytest.approx(float(b), rel=1e-12, abs=1e-12)


def nested_close(a, b) -> bool:
    """Equal dicts / lists of numbers (within 1e-12), strings and booleans."""
    if isinstance(b, dict):
        return isinstance(a, dict) and set(a) == set(b) and all(nested_close(a[k], b[k]) for k in b)
    if isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(nested_close(x, y) for x, y in zip(a, b))
    return close(a, b)


def same_table(df: pd.DataFrame, rows: dict, names: dict | None = None):
    """A notebook table (rows x columns) equals a lifted ``{row: {key: v}}``, the columns renamed by ``names``."""
    names = names or {}
    nb = df.to_dict(orient="index")
    assert set(nb) == set(rows)
    for r, cols in nb.items():
        for c, v in cols.items():
            assert close(rows[r][names.get(c, c)], v), (r, c, rows[r][names.get(c, c)], v)


# ------------------------------------------------------------------------------------------------ the workflow, twice
@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """``walkers.run`` twice through the mocks, from a copy of the committed ``walkers.vida`` (its nodes are reused and
    the new keys filled in; without it everything is built)."""
    import stubs
    mp = pytest.MonkeyPatch()
    s = stubs.install(mp)
    try:
        tmp = tmp_path_factory.mktemp("walkers")
        if (DATA / "walkers.vida").is_file():
            shutil.copy(DATA / "walkers.vida", tmp / "w.vida")
        first = walkers.run(vida_path=tmp / "w.vida", progress=False)
        sim1 = list(s.sim)
        second = walkers.run(vida_path=tmp / "w.vida", progress=False)
        stubs.check_calls(s, progress=False)
        return SimpleNamespace(first=first, second=second, sim1=sim1, sim2=list(s.sim), json=results.path_for(tmp / "w.vida"))
    finally:
        mp.undo()


def test_new_keys_recorded_and_reused(runs):
    r1, r2 = runs.first, runs.second
    want = {"dog/body": walkers.DOG_BODY_KEYS, "dog/gaits": walkers.DOG_TORQUE_KEYS + walkers.DOG_LOAD_KEYS,
            "cleopatra/body": walkers.CLEO_KEYS, "persephone/body": walkers.PERSEPHONE_KEYS, "apheloria/body": walkers.APH_KEYS}
    for path, keys in want.items():
        assert set(keys) <= set(r1.child(path).results), path
    assert r1.results["persephone_mass_kg"] == r1.child("persephone/body").results["mass_kg"]
    assert "persephone" not in r1.results["masses_kg"]
    assert r1.results["masses_kg"]["apheloria"] == pytest.approx(33.116, abs=1e-3)       # the robot, unchanged
    leaves = [p for p, n in r2.walk() if p and not n.children]
    assert all(r2.child(p).status() == "reused" for p in leaves), [(p, r2.child(p).status()) for p in leaves]
    assert runs.sim2 == runs.sim1                                                          # no scene asked again
    data = results.load(runs.json)
    assert data["dog/gaits"]["joint_torques"]["bound, with payload"]["peak_knee_Nm"] == pytest.approx(32.71, abs=0.01)
    assert data[""]["persephone_mass_kg"] == pytest.approx(r1.results["persephone_mass_kg"])


def test_old_keys_and_values_unchanged(runs):
    path = DATA / f"walkers{results.SUFFIX}"
    if not path.is_file():
        pytest.skip("no committed walkers_results.json")
    before, now = results.load(path), results.load(runs.json)
    for node in before:
        assert now.nodes[node]["params"] == before.nodes[node]["params"], node
        for k, v in before[node].items():
            assert now[node][k] == v, (node, k)


# ------------------------------------------------------------------------------------------------ the dog (16)
@pytest.fixture(scope="module")
def dog():
    p = rdr.design_params()
    M_DOG = sum(rdr.mass_budget().values())
    PAYLOAD = walkers.DOG_PAYLOAD_KG
    ns4 = cell(D, 4, ["p", "H_STAND"], dog_design=RobotDog())
    ns8 = cell(D, 8, ["GAITS", "foot_peak", "X_CG", "WHEELBASE", "static_split"], p=p)
    W_DOG, W_LOADED = M_DOG * G, (M_DOG + PAYLOAD) * G
    ns9 = cell(D, 9, None, p=p, gait=gait, act=act, H_STAND=ns4["H_STAND"], GAITS=ns8["GAITS"], foot_peak=ns8["foot_peak"],
               W_DOG=W_DOG, W_LOADED=W_LOADED)
    return SimpleNamespace(p=p, M_DOG=M_DOG, PAYLOAD=PAYLOAD, W_DOG=W_DOG, W_LOADED=W_LOADED, ns4=ns4, ns8=ns8, ns9=ns9)


def test_dog_torques_are_16_cell_9(dog, runs):
    assert {k: v for k, v in dog.ns4["p"].items() if k != "part"} == dog.p and dog.ns8["GAITS"] == walkers.DOG_GAITS   # the design
    q = act.get(rdr.ACTUATOR_KEY)
    assert dog.ns9["MOTOR"] == {"name": q.key, "continuous_Nm": q.rated_Nm, "peak_Nm": q.stall_Nm}
    rows = walkers.dog_joint_torques(dog.p, dog.ns4["H_STAND"], dog.W_DOG, dog.W_LOADED, walkers.DOG_GAITS, dog.ns9["MOTOR"])
    same_table(dog.ns9["torques"], rows, {"peak hip [Nm]": "peak_hip_Nm", "peak knee [Nm]": "peak_knee_Nm",
                                          "hip / peak motor": "hip_over_peak_motor", "knee / peak motor": "knee_over_peak_motor"})
    g = runs.first.child("dog/gaits").results
    assert g["motor"] == dog.ns9["MOTOR"]
    same_table(dog.ns9["torques"], g["joint_torques"], {"peak hip [Nm]": "peak_hip_Nm", "peak knee [Nm]": "peak_knee_Nm",
                                                        "hip / peak motor": "hip_over_peak_motor", "knee / peak motor": "knee_over_peak_motor"})


def test_dog_geometry_is_16_cells_4_5_8_11_36(dog, runs):
    p, H = dog.p, dog.ns4["H_STAND"]
    y_leg = cell(D, 5, ["y_leg"], p=p)["y_leg"]
    ns11 = cell(D, 11, ["DECK_Z", "H_CG_DOG"], p=p, H_STAND=H)
    src = "".join(json.loads((NB / D).read_text())["cells"][36]["source"])
    doc = next(n.value for n in ast.parse(src).body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Dict))
    stance = next(v for k, v in zip(doc.keys, doc.values) if isinstance(k, ast.Constant) and k.value == "stance_width_mm")
    stance_width = eval(compile(ast.Expression(stance), "c36", "eval"), {"y_leg": y_leg, "p": p})
    geo = walkers.dog_geometry(p)
    assert geo == pytest.approx({"standing_height": H, "foot_x": RobotDog.foot_x(p), "leg_stretched": p["upper_leg_length"] + p["lower_leg_length"],
                                 "y_leg": y_leg, "stance_width": stance_width, "hip_x": p["hip_x"], "wheelbase": dog.ns8["WHEELBASE"],
                                 "deck_z": ns11["DECK_Z"], "cg_height": ns11["H_CG_DOG"]}, rel=1e-12)
    assert runs.first.child("dog/body").results["geometry_mm"] == pytest.approx(geo, rel=1e-12)
    assert geo["standing_height"] == pytest.approx(324.7, abs=0.05) and geo["wheelbase"] == 380.0


def test_dog_stairs_are_16_cells_8_11(dog, runs):
    ns = cell(D, 11, None, p=dog.p, H_STAND=dog.ns4["H_STAND"], leg_ik=dog.ns9["leg_ik"], M_DOG=dog.M_DOG,
              PAYLOAD_ALLOW_KG=dog.PAYLOAD, W_DOG=dog.W_DOG, W_LOADED=dog.W_LOADED, WHEELBASE=dog.ns8["WHEELBASE"])
    s = walkers.DOG_STAIRS
    assert (ns["RISER"], ns["TREAD"], ns["CLEARANCE"], ns["MU_FEET"], ns["HIP_RANGE"], ns["KNEE_RANGE"], ns["PAYLOAD_CG_ABOVE_DECK"]) == (
        s["riser_mm"], s["tread_mm"], s["clearance_mm"], s["mu_feet"], s["hip_range_deg"], s["knee_range_deg"], s["payload_cg_above_deck_mm"])
    assert dog.ns8["X_CG"] == rdr.X_CG_MM
    out = walkers.dog_stairs(dog.p, dog.M_DOG, dog.PAYLOAD)
    same_table(ns["stairs"], out["stairs"], {"rear pair [N]": "rear_pair_N", "front pair [N]": "front_pair_N",
                                             "tangential needed [N]": "tangential_needed_N", "friction available [N]": "friction_available_N",
                                             "friction SF": "friction_SF", "tipping margin [mm]": "tipping_margin_mm"})
    assert out["foot_over_riser_deg"] == pytest.approx({"hip": ns["lift_ok"][0], "knee": ns["lift_ok"][1]}, rel=1e-12)
    assert out["lowest_crouch_mm"] == ns["crouch"]
    assert (out["slope_deg"], out["cg_height_mm"], out["cg_height_loaded_mm"], out["deck_z_mm"]) == pytest.approx(
        (ns["SLOPE_DEG"], ns["H_CG_DOG"], ns["H_CG_LOADED"], ns["DECK_Z"]), rel=1e-12)
    split = dog.ns8["static_split"]
    assert nested_close(out["static_split_N"], {"dog": split(dog.W_DOG, dog.ns8["X_CG"]), "with payload": split(dog.W_LOADED, dog.ns8["X_CG"])})
    g = runs.first.child("dog/gaits").results
    assert nested_close(g["stairs"], out["stairs"]) and g["stairs_inputs"]["riser_mm"] == 170.0


def test_dog_landing_is_16_cell_13(dog, runs):
    ns = cell(D, 13, None, M_DOG=dog.M_DOG, PAYLOAD_ALLOW_KG=dog.PAYLOAD, G=G)
    assert (ns["DROP_M"], ns["S_LEG"]) == (walkers.DOG_LANDING["drop_m"], walkers.DOG_LANDING["stroke_m"])
    out = walkers.dog_landing(dog.M_DOG, dog.PAYLOAD)
    same_table(ns["landing"], out, {"mean per leg [N]": "mean_per_leg_N", "peak per leg [N]": "peak_per_leg_N",
                                    "contact time [ms]": "contact_time_ms"})
    landing = runs.first.child("dog/gaits").results["landing"]
    assert landing["with payload"]["peak_per_leg_N"] == pytest.approx(661.1, abs=0.1)
    assert landing["dog"]["peak_per_leg_N"] == pytest.approx(478.2, abs=0.1)


# ------------------------------------------------------------------------------------------------ Cleopatra (18)
def test_cleopatra_gaits_are_18_cell_7(runs):
    assert cell(C, 2, ["CLEO"])["CLEO"] == mr.CLEO_MM
    ACTUATOR = cell(C, 5, ["ACTUATOR"], act=act)["ACTUATOR"]
    assert ACTUATOR == act.get(walkers.CLEO_LEG_ACTUATOR).as_dict()
    p = myropod.Myropod().resolve(**mr.CLEO_MM)
    mb = mr.mass_budget()
    g = {k: mb[k] * 1000 for k in ("per segment", "head", "battery", "compute")}
    ns = cell(C, 7, None, p=p, per_segment=pd.Series({"all": g["per segment"]}), head_g=g["head"], battery_g=g["battery"],
              compute_g=g["compute"], ACTUATOR=ACTUATOR, G=G)
    assert ns["GAITS"] == walkers.CLEO_GAITS
    W_SEG, rows = walkers.cleopatra_gaits(p, g["per segment"], g["head"], g["battery"], g["compute"], ACTUATOR)
    assert W_SEG == pytest.approx(ns["W_SEG"], rel=1e-12)
    names = {"W_seg [N]": "W_seg_N", "peak foot [N]": "peak_foot_N", "servo SF": "servo_SF"}
    same_table(ns["tq"], rows, names)
    b = runs.first.child("cleopatra/body").results
    same_table(ns["tq"], b["gaits"], names)
    assert b["segment_weight_N"] == pytest.approx(ns["W_SEG"], rel=1e-12)
    assert b["mass_budget_kg"] == mb                                                     # the budget it is built on


def test_cleopatra_envelope_is_18_cell_4(runs):
    b = runs.first.child("cleopatra/body").results
    p = myropod.Myropod().resolve(**mr.CLEO_MM)
    ns = cell(C, 4, ["N", "sheet"], p=p, cleo=SimpleNamespace(dimensions=b["envelope_mm"]))
    assert ns["sheet"]["sheet"].to_dict() == b["sheet"] == walkers.CLEO_SHEET
    design = ns["sheet"]["this design"]
    assert [design["length [m]"], design["width [m]"], design["height [m]"]] == [x / 1000 for x in b["envelope_mm"]]
    assert b["envelope_mm"][0] == pytest.approx(652.0, abs=0.5) and b["envelope_mm"][2] / 1000 > 0.16   # taller than the sheet


# ------------------------------------------------------------------------------------------------ Persephone (17)
@pytest.fixture(scope="module")
def persephone(runs):
    b = runs.first.child("persephone/body").results
    d = myropod.Myropod()
    vol = b["part_volume_mm3"]
    parts = {k: SimpleNamespace(volume=vol[k]) for k in ("segment", "leg", "head")}
    ns4 = cell(P, 4, ["p", "PITCH", "N", "REACH", "LENGTH"], myropod=d)
    ns7 = cell(P, 7, ["RHO_PA12CF", "SERVOS", "JOINT_SERVO", "mass_budget"], act=act, p=ns4["p"], N=ns4["N"],
               seg=parts["segment"], leg=parts["leg"], head=parts["head"])
    ns9 = cell(P, 9, ["SEGMENT_TYPES", "CONFIG", "MODULES_G", "_mass_budget_base", "mass_budget"], **ns7)
    ns11 = cell(P, 11, None, p=ns4["p"], PITCH=ns4["PITCH"], REACH=ns4["REACH"])
    ns14 = cell(P, 14, None, **{**ns9, "reach_y": ns11["reach_y"]})
    return SimpleNamespace(b=b, d=d, vol=vol, ns4=ns4, ns9=ns9, ns11=ns11, ns14=ns14)


def test_persephone_geometry_is_17_cell_4(persephone):
    b, ns4 = persephone.b, persephone.ns4
    assert (b["pitch_mm"], b["length_mm"], b["n_segments"], b["leg_reach_mm"]) == (ns4["PITCH"], ns4["LENGTH"], ns4["N"], ns4["REACH"])
    assert set(b["part_volume_mm3"]) == {"segment", "leg", "head"} and b["segment_surface_area_mm2"] > 0
    assert b["part_volume_mm3"]["segment"] == pytest.approx(35204, rel=1e-3)


def test_persephone_fit_is_17_cell_11(persephone):
    b, ns = persephone.b, persephone.ns11
    assert ns["JOINT_RANGE_DEG"] == walkers.PERSEPHONE_JOINT_RANGE_DEG == b["joint_range_deg"]
    assert b["fit_mm"] == pytest.approx({"bracing_min": ns["D_MIN_BRACE"], "bracing_max": ns["D_MAX_BRACE"],
                                         "min_bend_radius": ns["R_MIN_BEND"], "mitre_min": ns["D_MIN_MITRE"]}, rel=1e-12)
    same_table(pd.DataFrame(ns["flues"]).T, b["flues"], {
        "bracing possible": "bracing_possible", "legs for bracing: hip/knee [deg]": "legs_for_bracing",
        "joint angle in a tight 90° elbow [deg]": "elbow_joint_angle_deg", "elbow ok": "elbow_ok", "mitre corner ok": "mitre_corner_ok",
        "sagging body in a horizontal run: legs reach the floor": "legs_reach_floor"})


def test_persephone_mass_is_17_cells_7_9_14(persephone, runs):
    b, ns9, ns = persephone.b, persephone.ns9, persephone.ns14
    assert ns9["CONFIG"] == walkers.PERSEPHONE_CONFIG == b["config"] and ns9["MODULES_G"] == b["modules_g"] == 285.0
    assert {t: {"payload_g": r["payload_g"], "what": r["what"]} for t, r in ns9["SEGMENT_TYPES"].iterrows()} == b["segment_types"]
    inputs = walkers.PERSEPHONE_BRACING
    assert (ns["MU_SOOT"], ns["MU_CLEAN"], ns["SF_SLIP"], ns["SF_TORQUE"], ns["TETHER_G_PER_M"], ns["HEIGHT_M"], ns["STANCE_FRACTION"], ns["D_WORK"]) == (
        inputs["mu_soot"], inputs["mu_clean"], inputs["sf_slip"], inputs["sf_torque"], inputs["tether_g_per_m"], inputs["height_m"],
        inputs["stance_fraction"], inputs["d_work_mm"])
    lifted = walkers.persephone_mass(persephone.ns4["p"], persephone.vol["segment"], persephone.vol["leg"], persephone.vol["head"])
    for out in (lifted, b):
        assert out["leg_servo"] == ns["choice"] == "standard 60 g worm"
        assert out["mass_kg"] == pytest.approx(ns["M_CRAWLER"], rel=1e-12)
        assert out["per_segment_g"] == pytest.approx(ns["per_segment"].to_dict(), rel=1e-12)
        assert out["bracing_design_point"] == pytest.approx(ns["B"], rel=1e-12)
        same_table(ns["brace_tab"], out["bracing"], {"servo SF": "servo_SF"})
        assert out["joint_servo"] == ns["JOINT_SERVO"] and out["leg_servo_data"] == act.get(ns["choice"]).as_dict()
    assert runs.first.results["persephone_mass_kg"] == pytest.approx(ns["M_CRAWLER"], rel=1e-12)


# ------------------------------------------------------------------------------------------------ Apheloria (19)
@pytest.fixture(scope="module")
def aph(runs):
    b = runs.first.child("apheloria/body").results
    a = apheloria.Apheloria()
    vol = b["part_volume_mm3"]
    parts = {k: SimpleNamespace(volume=vol[k]) for k in ("segment", "plate", "leg", "head")}
    ns4 = cell(A, 4, ["p", "N", "REACH"], aph=a)
    ns7 = cell(A, 7, None, aph=a, p=ns4["p"], act=act, seg=parts["segment"], plate=parts["plate"], leg=parts["leg"], head=parts["head"])
    ns9 = cell(A, 9, None, aph=a, p=ns4["p"], cfg=ns7["cfg"], G=G)
    ns11 = cell(A, 11, None, aph=a, p=ns4["p"], N=ns4["N"], MODULES=ns7["MODULES"], ACT=ns7["ACT"], G=G)
    return SimpleNamespace(b=b, a=a, vol=vol, ns4=ns4, ns7=ns7, ns9=ns9, ns11=ns11)


def test_apheloria_geometry_is_19_cell_4(aph):
    a, b, p = aph.a, aph.b, aph.ns4["p"]
    assert b["cad_design"] == p and b["leg_reach_mm"] == aph.ns4["REACH"]
    assert (b["pitch_mm"], b["ball_radius_mm"], b["coil_radius_mm"], b["coil_angle_deg"]) == (
        a.pitch_length(p), a.ball_radius(p), a.coil_radius(p), a.coil_angle_deg(p))
    assert b["pitch_mm"] == 168.0 and b["ball_radius_mm"] == pytest.approx(312.6, abs=0.05) and b["coil_angle_deg"] == 40.0


def test_apheloria_configs_are_19_cell_7(aph):
    ns, b = aph.ns7, aph.b
    assert list(ns["MODULES"]["payload_g"].items()) == walkers.APH_MODULES and ns["CONFIGS"] == walkers.APH_CONFIGS
    lifted = walkers.apheloria_configs(aph.a, aph.ns4["p"], aph.vol["segment"], aph.vol["plate"], aph.vol["leg"], aph.vol["head"])
    names = {"mass [kg]": "mass_kg", "batteries [Wh]": "batteries_Wh", "length walk [mm]": "length_walk_mm", "ball diameter [mm]": "ball_diameter_mm"}
    for out in (lifted, b):
        assert out["base_segment_g"] == pytest.approx(ns["base_segment"].to_dict(), rel=1e-12)
        assert out["head_g"] == pytest.approx(ns["head_g"], rel=1e-12)
        same_table(ns["MODULES"], out["modules_g"])
        same_table(ns["cfg"], out["configs"], names)
        assert out["actuators"] == ns["ACT"]
    assert b["configs"]["standard (8)"]["mass_kg"] == pytest.approx(39.23, abs=0.01)
    assert b["mass_kg"] == pytest.approx(33.116, abs=1e-3) and "no payload modules" in b["mass_note"]


def test_apheloria_ball_is_19_cell_9(aph):
    ns, b = aph.ns9, aph.b
    keys = {"standard config [kg]": "standard_config_kg", "ball diameter [m]": "ball_diameter_m", "slope to start rolling [deg]": "slope_to_start_deg",
            "kinetic energy at 15 cm/s [J]": "kinetic_energy_J", "rolling 10 m downhill at 5° — speed [m/s]": "roll_10m_at_5deg_m_s",
            f"drop {ns['DROP_M']} m: impact force [N]": "impact_force_N", "impact peak on the plate pad [N]": "impact_peak_pad_N",
            "impact as g": "impact_g"}
    assert set(ns["roll"].index) == set(keys)
    inputs = walkers.APH_BALL
    assert (ns["CG_OFFSET"], ns["V_ROLL"], ns["DROP_M"], ns["DELTA_M"]) == (inputs["cg_offset_m"], inputs["v_roll_m_s"], inputs["drop_m"], inputs["delta_m"])
    lifted = walkers.apheloria_ball(aph.a, aph.ns4["p"], aph.ns7["cfg"].loc["standard (8)", "mass [kg]"])
    for out in (lifted, b["ball"]):
        for k, v in ns["roll"].items():
            assert out[keys[k]] == pytest.approx(float(v), rel=1e-12), k


def test_apheloria_curl_is_19_cell_11(aph):
    ns, b = aph.ns11, aph.b
    N = aph.ns4["N"]
    lifted = walkers.apheloria_curl(aph.a, aph.ns4["p"], aph.ns7["MODULES"].loc["camera", "segment_g"], aph.ns7["ACT"]["joint"]["stall_Nm"])
    for out in (lifted, b["curl"]):
        assert out["k_max_lifted"] == ns["K_MAX"] == 3
        assert out["segment_weight_N"] == pytest.approx(ns["W_SEG"], rel=1e-12)
        assert out["moment_k_max_Nm"] == pytest.approx(ns["M_CURL"], rel=1e-12)
        assert out["moment_half_chain_Nm"] == pytest.approx(ns["curl_moment"](N // 2), rel=1e-12)
        assert out["sf_half_chain"] == pytest.approx(ns["ACT"]["joint"]["stall_Nm"] / ns["curl_moment"](N // 2), rel=1e-12)
        assert out["moment_by_segments_lifted_Nm"] == pytest.approx({str(k): ns["curl_moment"](k) for k in range(1, N + 1)}, rel=1e-12)
