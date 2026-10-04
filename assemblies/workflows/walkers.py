"""The walkers: the robot dog (notebook 16), the Myropods Persephone (17) and Cleopatra (18), and Apheloria (19).

The tree::

    walkers (walker_family)
      dog (walker)
        body (walker_body)           mass budget from the CAD numbers, the Chiron robot (mass, joints) against it;
                                     the standing geometry (16 cells 4, 5, 8, 11)
        gaits (foot_peaks)           peak foot force of walk, trot and bound, alone and with the payload (16 cell 8);
                                     joint torques against the qdd 24 Nm (cell 9), stairs and slopes (cells 8, 11),
                                     the drop landing (cell 13)
      cleopatra (walker)
        body (walker_body)           notebook 18's budget and the Chiron robot; weight per segment, the gait and servo
                                     table (18 cell 7), the CAD envelope against the sheet (cell 4)
      persephone (walker)
        body (walker_cad)            the crawler's CAD (myropod.Myropod defaults): volume, area, size; segment, leg
                                     and head volumes, pitch, length, reach (17 cell 4), the fit (cell 11), the mass
                                     budget with the bracing servo loop and the bracing table (cells 7, 9, 14)
      apheloria (walker)
        body (walker_body)           the Chiron robot: head + 8 segments, 96 leg servos, 8 body pitch joints;
                                     notebook 19's design: geometry (cell 4), base segment, modules, configurations
                                     (cell 7), the ball (cell 9), curling up (cell 11)
        pack (episode)               packs into its ball in MuJoCo   (scenarios/apheloria_pack.py, without the movie)
        unpack (episode)             and opens again

The root also records ``persephone_mass_kg`` (17 cell 14's crawler with its modules; ``masses_kg`` has no Persephone).
Apheloria's ``mass_kg`` (also in ``masses_kg``) is the Chiron robot, head + 8 base segments without payload modules;
notebook 19 designs to ``configs["standard (8)"]``.

What stays in the notebooks: the leg torque table of 19 (cell 13) and its endurance (cell 20), the gait simulations of
16 and 17, Persephone's flue climb. The trials of the dog and Cleopatra are in ``benchmark/``.
"""
from __future__ import annotations

import copy
import dataclasses
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .. import DATA, vida, results
from .._cli import main, parser
from ..components import actuators as act, apheloria, gait
from ..components import apheloria_pack as ap_scene, apheloria_robot as ar, leg, myropod, myropod_robot as mr
from ..components import robot_dog_robot as rdr
from ..components.robot_dog import RobotDog
from ..vida import Assembly
from ._common import add_after
from ._scene import run_scene

NAME = "walkers"
G = 9.81
MACHINES = ("dog", "cleopatra", "persephone", "apheloria")
DOG_GAITS = {"walk": 0.75, "trot": 0.50, "bound": 0.30}                                   # 16 cell 8
DOG_PAYLOAD_KG = 5.0                                                                       # 16 cell 6: the deck allowance
DOG_STRIDE_FRACTION = 0.45                                                                 # 16 cell 9: STRIDE = 0.45 H_STAND
DOG_STAIRS = {"riser_mm": 170.0, "tread_mm": 280.0, "clearance_mm": 40.0,                  # 16 cell 11 (inputs)
              "mu_feet": {"dry concrete": 0.7, "wet tiles": 0.35, "soot on stone": 0.3},
              "hip_range_deg": (-30.0, 110.0), "knee_range_deg": (0.0, 150.0), "payload_cg_above_deck_mm": 60.0}
DOG_LANDING = {"drop_m": 0.30, "stroke_m": 0.080}                                          # 16 cell 13 (inputs)
CLEO_GAITS = {"crawl 0.4 m/s": (0.75, 0.4), "walk 1.2 m/s": (0.50, 1.2), "run 2.0 m/s": (0.35, 2.0)}   # 18 cell 7: (duty, speed)
CLEO_LEG_ACTUATOR = "smart servo 6 Nm"                                                     # 18 cell 5
CLEO_SHEET = {"length [m]": "0.55–0.70", "width [m]": "0.22–0.30", "height [m]": "0.10–0.16", "mass [kg]": "3.5–7.5",   # 18 cell 4
              "legs": "12 (6 x 2)", "walk [m/s]": "to 1.2", "crawl [m/s]": "to 0.4", "run time [h]": "2–8"}
PERSEPHONE_JOINT_RANGE_DEG = 45.0                                                          # 17 cell 11 (input)
PERSEPHONE_SEGMENT_TYPES = [                                                               # 17 cell 9: (type, payload_g, what)
    ("plain", 0.0, "structure and legs only"),
    ("battery backup", 45.0, "2 x 18650 for a 10 min retreat when the tether fails"),
    ("sensor", 30.0, "gas, temperature, humidity, a side camera"),
    ("relay dropper", 60.0, "two LoRa relay pucks it leaves behind at elbows"),
    ("sampler", 55.0, "a soot scraper and a sample cup"),
    ("lights", 20.0, "extra illuminators for the cameras behind"),
]
PERSEPHONE_CONFIG = ["lights", "sensor", "plain", "battery backup", "plain", "relay dropper", "plain", "sensor", "plain",   # 17 cell 9
                     "sampler", "plain", "battery backup"]
PERSEPHONE_BRACING = {"mu_soot": 0.30, "mu_clean": 0.60, "sf_slip": 2.0, "sf_torque": 1.5, "tether_g_per_m": 45.0,         # 17 cell 14
                      "height_m": 15.0, "stance_fraction": 2 / 3, "d_work_mm": 150.0}
APH_MODULES = [("camera", 180), ("IR sensor", 220), ("lidar", 350), ("IMU", 60), ("env sensor", 120), ("fibre optic spool", 900),   # 19 cell 7
               ("battery (Li-ion 300 Wh)", 1800), ("AI core", 400), ("tool / sampler", 1200), ("medical", 800), ("odor module", 250),
               ("dummy / armour", 0)]
APH_CONFIGS = {"mini (4)": ["camera", "battery (Li-ion 300 Wh)", "AI core", "env sensor"],                              # 19 cell 7
               "standard (8)": ["camera", "IR sensor", "lidar", "battery (Li-ion 300 Wh)", "AI core", "env sensor",
                                "battery (Li-ion 300 Wh)", "tool / sampler"],
               "max (14)": ["camera", "IR sensor", "lidar", "IMU", "battery (Li-ion 300 Wh)", "AI core", "env sensor",
                            "fibre optic spool", "battery (Li-ion 300 Wh)", "tool / sampler", "medical", "odor module",
                            "battery (Li-ion 300 Wh)", "dummy / armour"]}
APH_BALL = {"cg_offset_m": 0.025, "v_roll_m_s": 0.15, "drop_m": 0.5, "delta_m": 0.030}     # 19 cell 9 (inputs)
APH_MASS_NOTE = ("mass_kg is the Chiron robot: head + 8 base segments, no payload modules; notebook 19 designs to "
                 "configs['standard (8)']")
#: the result keys each fill-in adds (a saved node without one of them gets them all on the next run)
DOG_BODY_KEYS = ("geometry_mm",)
DOG_TORQUE_KEYS = ("joint_torques", "motor")
DOG_LOAD_KEYS = ("static_split_N", "stairs", "landing")
CLEO_KEYS = ("segment_weight_N", "gaits", "envelope_mm")
PERSEPHONE_KEYS = ("part_volume_mm3", "fit_mm", "mass_kg")
APH_KEYS = ("configs", "modules_g", "ball_radius_mm", "curl", "ball")


def _robot(robot) -> dict:
    s = robot.summary()
    return {"robot_mass_kg": robot.total_mass(), "servos": len(robot.actuated_joints()),
            "links": s.get("n_links"), "passive_joints": s.get("n_passive"), "link_mass_kg": s.get("link_mass_kg")}


def _design(machine: str) -> dict:
    if machine == "dog":
        return rdr.design_params()
    if machine == "cleopatra":
        return dataclasses.asdict(mr.CleopatraParams())
    if machine == "persephone":
        return myropod.Myropod().resolve()
    return {**dataclasses.asdict(ar.PARAMS), "n_segments": ar.N_SEGMENTS, "plates": True}


def build(machines=MACHINES) -> Assembly:
    root = Assembly(NAME, "walker_family")
    for m in machines:
        p = _design(m)
        node = root.add(Assembly(m, "walker", params={"design": p}))
        node.add(Assembly("body", "walker_cad" if m == "persephone" else "walker_body", params={"design": p}))
        if m == "dog":
            node.add(Assembly("gaits", "foot_peaks", params={"duty": DOG_GAITS, "payload_kg": DOG_PAYLOAD_KG}))
    return root


def body(node: Assembly, machine: str) -> Assembly:
    """The machine's mass: the budget the notebook writes down and the Chiron robot built from the same numbers."""
    if machine == "dog":
        budget = rdr.mass_budget()
        return node.record(mass_budget_kg=budget, mass_kg=sum(budget.values()), **_robot(rdr.dog_robot()))
    if machine == "cleopatra":
        budget = mr.mass_budget()
        return node.record(mass_budget_kg=budget, mass_kg=budget["total"], **_robot(mr.cleopatra()))
    if machine == "persephone":
        m = myropod.Myropod().generate().measure()
        return node.record(volume_mm3=m["volume"], surface_area_mm2=m["surface_area"], dimensions_mm=m["dimensions"],
                           center_of_mass_mm=m["center_of_mass"])
    r = ar.apheloria()
    return node.record(mass_kg=r.total_mass(), **_robot(r))


def dog_gaits(node: Assembly, mass_kg: float) -> Assembly:
    """16 cell 8: the half-sine peak foot force of each gait, the dog alone and with the payload allowance."""
    W, W_loaded = mass_kg * G, (mass_kg + node.params["payload_kg"]) * G
    rows = {g: {"duty": b, "feet_down_mean": 4 * b, "peak_foot_N": leg.foot_peak(W, b), "peak_foot_loaded_N": leg.foot_peak(W_loaded, b),
                "F_over_W_per_foot": leg.foot_peak(1.0, b)} for g, b in node.params["duty"].items()}
    return node.record(gaits=rows)


# ------------------------------------------------------------------------------------------------ the dog (notebook 16)
def _table(df, names: dict | None = None) -> dict:
    """A notebook DataFrame (rows x columns) as ``{row: {column: value}}``, columns renamed by ``names``, numbers plain."""
    names = names or {}
    plain = lambda v: v.item() if isinstance(v, np.generic) else v                     # noqa: E731
    return {str(r): {names.get(c, c): plain(v) for c, v in row.items()} for r, row in df.to_dict(orient="index").items()}


def dog_geometry(p: dict) -> dict:
    """The standing geometry of the design ``p`` [mm]: 16 cell 4 (H_STAND, foot_x, the stretched leg), cell 5 (y_leg),
    cell 8 (WHEELBASE), cell 11 (DECK_Z, H_CG_DOG) and cell 36 (the stance width)."""
    H_STAND = RobotDog.standing_height(p)                                                  # cell 4
    y_leg = p["body_width"] / 2 + p["hip_boss_length"] + p["upper_leg_thickness"] / 2 + 1.0   # cell 5
    WHEELBASE = 2 * p["hip_x"]                                                             # cell 8
    DECK_Z = H_STAND + 8.0 + p["body_height"] / 2 - p["deck_depth"]                        # cell 11
    H_CG_DOG = H_STAND + 8.0                                                               # body centre
    return {"standing_height": H_STAND, "foot_x": RobotDog.foot_x(p), "leg_stretched": p["upper_leg_length"] + p["lower_leg_length"],
            "y_leg": y_leg, "stance_width": 2 * y_leg + p["upper_leg_thickness"], "hip_x": p["hip_x"], "wheelbase": WHEELBASE,
            "deck_z": DECK_Z, "cg_height": H_CG_DOG}


def dog_joint_torques(p: dict, H_STAND: float, W_DOG: float, W_LOADED: float, GAITS: dict, MOTOR: dict) -> dict:
    """16 cell 9: hip-pitch and knee torque [N m] over a stance (planar two-link leg at the standing height, half-sine
    Fz, Fx = 0.3 Fz cos), the peak per gait, the dog alone and with the payload, and the ratio to the motor's peak:
    ``{"<gait>, dog" | "<gait>, with payload": {peak_hip_Nm, peak_knee_Nm, hip_over_peak_motor, knee_over_peak_motor}}``."""
    L1, L2 = p["upper_leg_length"], p["lower_leg_length"]

    def leg_ik(x_foot, z_foot):
        """Hip and knee angles (upper leg behind vertical, lower leg ahead of vertical, knee back) for a foot at (x, z) relative to the hip."""
        return gait.ik_two_link_planar(x_foot, z_foot, L1, L2)

    def joint_torques(x_foot, z_foot, fx, fz):
        """Hip-pitch and knee torques [N m] holding a foot force (fx forward, fz up on the foot) at the given foot position [mm]."""
        ang = leg_ik(x_foot, z_foot)
        if ang is None:
            return float("nan"), float("nan")
        a1, a2 = map(math.radians, ang)
        knee = np.array([-L1 * math.sin(a1), -L1 * math.cos(a1)])
        foot = np.array([x_foot, z_foot])
        F = np.array([fx, fz])
        tau_hip = (foot[0] * F[1] - foot[1] * F[0]) / 1000
        tau_knee = ((foot[0] - knee[0]) * F[1] - (foot[1] - knee[1]) * F[0]) / 1000
        return tau_hip, tau_knee

    STRIDE = DOG_STRIDE_FRACTION * H_STAND          # foot travel during one stance at a trot [mm] (input: ~ half the leg height)
    xs = np.linspace(STRIDE / 2, -STRIDE / 2, 41)   # the foot lands ahead and leaves behind
    torque_rows = {}
    for gname, beta in GAITS.items():
        for W, tag in ((W_DOG, "dog"), (W_LOADED, "with payload")):
            Fz = leg.foot_peak(W, beta) * np.sin(np.pi * np.linspace(0, 1, len(xs)))      # half-sine over the stance
            Fx = 0.3 * Fz * np.cos(np.pi * np.linspace(0, 1, len(xs)))                   # braking then pushing
            th = np.array([joint_torques(x, -H_STAND, fx, fz) for x, fx, fz in zip(xs, Fx, Fz)])
            torque_rows[f"{gname}, {tag}"] = {"peak_hip_Nm": float(np.nanmax(np.abs(th[:, 0]))), "peak_knee_Nm": float(np.nanmax(np.abs(th[:, 1])))}
    for row in torque_rows.values():
        row["hip_over_peak_motor"] = row["peak_hip_Nm"] / MOTOR["peak_Nm"]
        row["knee_over_peak_motor"] = row["peak_knee_Nm"] / MOTOR["peak_Nm"]
    return torque_rows


def dog_stairs(p: dict, M_DOG: float, PAYLOAD_ALLOW_KG: float, X_CG: float = rdr.X_CG_MM, stairs: dict = DOG_STAIRS) -> dict:
    """16 cell 8 (the static front/rear split) and cell 11 (stairs and slopes with the payload on the back): the foot
    over the riser, the lowest crouch, the CG heights, and per surface the pair loads, friction SF and tipping margin."""
    L1, L2 = p["upper_leg_length"], p["lower_leg_length"]
    H_STAND = RobotDog.standing_height(p)
    W_DOG, W_LOADED = M_DOG * G, (M_DOG + PAYLOAD_ALLOW_KG) * G

    def leg_ik(x_foot, z_foot):                                                            # cell 9
        return gait.ik_two_link_planar(x_foot, z_foot, L1, L2)

    WHEELBASE = 2 * p["hip_x"]                                                             # cell 8

    def static_split(W, x_cg):                # front/rear share of the weight from the CG position
        front = 0.5 + x_cg / WHEELBASE
        return {"front_N": W * front / 2, "rear_N": W * (1 - front) / 2}

    RISER, TREAD = stairs["riser_mm"], stairs["tread_mm"]                                  # cell 11
    SLOPE_DEG = math.degrees(math.atan(RISER / TREAD))
    CLEARANCE = stairs["clearance_mm"]
    MU_FEET = stairs["mu_feet"]
    HIP_RANGE, KNEE_RANGE = tuple(stairs["hip_range_deg"]), tuple(stairs["knee_range_deg"])
    PAYLOAD_CG_ABOVE_DECK = stairs["payload_cg_above_deck_mm"]
    DECK_Z = H_STAND + 8.0 + p["body_height"] / 2 - p["deck_depth"]
    H_CG_DOG = H_STAND + 8.0                           # body centre
    H_CG_LOADED = (M_DOG * H_CG_DOG + PAYLOAD_ALLOW_KG * (DECK_Z + PAYLOAD_CG_ABOVE_DECK)) / (M_DOG + PAYLOAD_ALLOW_KG)
    # (a) foot lift: the swing foot must reach z = -(H_STAND - RISER - CLEARANCE) at x = +TREAD/2 while the body stays level
    lift_ok = leg_ik(TREAD / 2, -(H_STAND - RISER - CLEARANCE))
    # (b) a crouch: the lowest standing height within the joint ranges
    crouch = min((z for z in np.arange(100, H_STAND, 5) if (a := leg_ik(0.0, -z)) and HIP_RANGE[0] <= a[0] <= HIP_RANGE[1] and KNEE_RANGE[0] <= a[1] <= KNEE_RANGE[1]), default=None)

    # (c) slope: weight split and the friction needed, plus the backward-tipping margin
    def slope_check(W, h_cg, theta_deg, mu):
        th = math.radians(theta_deg)
        shift = h_cg / WHEELBASE * math.tan(th)                                  # share moved to the downhill (rear) pair
        rear = W * math.cos(th) * (0.5 + shift); front = W * math.cos(th) * (0.5 - shift)
        needed = W * math.sin(th)                                                  # total tangential force to hold / climb
        available = mu * (rear + front)
        tip_margin = WHEELBASE / 2 - h_cg * math.tan(th)                           # CG projection inside the rear feet [mm]
        return {"rear_pair_N": rear, "front_pair_N": front, "tangential_needed_N": needed, "friction_available_N": available,
                "friction_SF": available / needed, "tipping_margin_mm": tip_margin}
    rows = {}
    for surf, mu in MU_FEET.items():
        rows[f"dog only, {surf}"] = slope_check(W_DOG, H_CG_DOG, SLOPE_DEG, mu)
        rows[f"with payload, {surf}"] = slope_check(W_LOADED, H_CG_LOADED, SLOPE_DEG, mu)
    return {"static_split_N": {"dog": static_split(W_DOG, X_CG), "with payload": static_split(W_LOADED, X_CG)}, "x_cg_mm": X_CG,
            "slope_deg": SLOPE_DEG, "cg_height_mm": H_CG_DOG, "cg_height_loaded_mm": H_CG_LOADED, "deck_z_mm": DECK_Z, "stairs": rows,
            "foot_over_riser_deg": None if lift_ok is None else {"hip": lift_ok[0], "knee": lift_ok[1]},
            "lowest_crouch_mm": None if crouch is None else float(crouch)}


def dog_landing(M_DOG: float, PAYLOAD_ALLOW_KG: float, landing: dict = DOG_LANDING) -> dict:
    """16 cell 13: a drop from a step onto the front pair, mean and half-sine peak per leg and the contact time."""
    DROP_M, S_LEG = landing["drop_m"], landing["stroke_m"]

    def landing_force(mass, n_legs=2):
        mean = mass * G * (DROP_M / S_LEG + 1) / n_legs
        return {"mean_per_leg_N": mean, "peak_per_leg_N": mean * math.pi / 2, "contact_time_ms": 2 * S_LEG / math.sqrt(2 * G * DROP_M) * 1000}
    return {"dog": landing_force(M_DOG), "with payload": landing_force(M_DOG + PAYLOAD_ALLOW_KG)}


def dog_torques(node: Assembly, design: dict, mass_kg: float) -> Assembly:
    """On the gaits node: 16 cell 9's joint torques and the motor (``qdd 24 Nm``) they are compared with."""
    q = act.get(rdr.ACTUATOR_KEY)
    MOTOR = {"name": q.key, "continuous_Nm": q.rated_Nm, "peak_Nm": q.stall_Nm}
    W, W_loaded = mass_kg * G, (mass_kg + node.params["payload_kg"]) * G
    rows = dog_joint_torques(design, RobotDog.standing_height(design), W, W_loaded, node.params["duty"], MOTOR)
    return node.record(joint_torques=rows, motor=MOTOR)


def dog_loads(node: Assembly, design: dict, mass_kg: float) -> Assembly:
    """On the gaits node: stairs and slopes (16 cells 8, 11) and the drop landing (cell 13), with their inputs."""
    payload = node.params["payload_kg"]
    return node.record(**dog_stairs(design, mass_kg, payload), stairs_inputs=copy.deepcopy(DOG_STAIRS),
                       landing=dog_landing(mass_kg, payload), landing_inputs=copy.deepcopy(DOG_LANDING))


# ------------------------------------------------------------------------------------------------ Cleopatra (notebook 18)
def cleopatra_gaits(p: dict, per_segment_g: float, head_g: float, battery_g: float, compute_g: float, ACTUATOR: dict,
                    GAITS: dict = CLEO_GAITS) -> tuple[dict, dict]:
    """18 cell 7: the weight each segment carries, and per gait and segment the peak foot force, the hip, knee and
    longitudinal torques and the leg servo's SF (``per_segment_g`` is the cell's ``per_segment.sum()``)."""
    W_SEG = {"segment 1 (head)": (per_segment_g + head_g + compute_g) / 1000 * G, "segment 2 (battery)": (per_segment_g + battery_g) / 1000 * G,
             "segment 3": per_segment_g / 1000 * G}

    def torques(Fz, Fx, hip=p["hip_angle_deg"], knee=p["knee_angle_deg"]):
        a, b = math.radians(hip), math.radians(knee)
        y_knee, z_knee = p["femur_length"] * math.cos(a), -p["femur_length"] * math.sin(a)
        y_foot, z_foot = y_knee + p["tibia_length"] * math.cos(b), z_knee - p["tibia_length"] * math.sin(b)
        tau_hip = (Fz * y_foot) / 1000; tau_knee = (Fz * (y_foot - y_knee)) / 1000          # vertical force over the horizontal levers
        tau_hip_roll = (Fx * abs(z_foot)) / 1000                                               # the longitudinal force on the leg's vertical drop
        return {"tau_hip_Nm": tau_hip, "tau_knee_Nm": tau_knee, "tau_long_Nm": tau_hip_roll}
    rows = {}
    for gname, (beta, v) in GAITS.items():
        for sname, Ws in W_SEG.items():
            F = leg.foot_peak(Ws, beta); Fx = 0.3 * F
            rows[f"{gname}, {sname}"] = {"W_seg_N": Ws, "duty": beta, "peak_foot_N": F, **torques(F, Fx)}
    for row in rows.values():
        row["servo_SF"] = ACTUATOR["stall_Nm"] / max(row["tau_hip_Nm"], row["tau_knee_Nm"], row["tau_long_Nm"])
    return W_SEG, rows


def cleopatra_loads(node: Assembly) -> Assembly:
    """On Cleopatra's body: 18 cell 7 over the recorded budget (``mass_budget_kg``: per segment, head, battery, compute)
    and the CAD envelope of 18 cell 4 (``Myropod`` with ``CLEO``) next to the sheet."""
    mb = node.results["mass_budget_kg"]
    d = myropod.Myropod()
    p = d.resolve(**mr.CLEO_MM)
    ACTUATOR = act.get(CLEO_LEG_ACTUATOR).as_dict()
    W_SEG, rows = cleopatra_gaits(p, mb["per segment"] * 1000, mb["head"] * 1000, mb["battery"] * 1000, mb["compute"] * 1000, ACTUATOR)
    envelope = [float(x) for x in d.generate(**mr.CLEO_MM).dimensions]
    return node.record(segment_weight_N=W_SEG, gaits=rows, leg_actuator=ACTUATOR, envelope_mm=envelope, sheet=copy.deepcopy(CLEO_SHEET))


# ------------------------------------------------------------------------------------------------ Persephone (notebook 17)
def persephone_fit(p: dict, PITCH: float, REACH: dict, JOINT_RANGE_DEG: float = PERSEPHONE_JOINT_RANGE_DEG) -> dict:
    """17 cell 11: the bracing range of flue diameters, the tightest bend radius, the mitre corner, and the flue table."""
    def reach_y(hip_deg, knee_deg):
        return p["femur_length"] * math.cos(math.radians(hip_deg)) + p["tibia_length"] * math.cos(math.radians(knee_deg))
    D_MIN_BRACE = p["seg_width"] + 2 * 6 + 2 * max(reach_y(-30, 120), p["femur_length"] * math.cos(math.radians(-30))) + p["foot_diameter"]   # legs folded: knees out
    D_MAX_BRACE = p["seg_width"] + 2 * 6 + 2 * reach_y(0, 0) * 0.95 + p["foot_diameter"]                                                        # 95 % of the full stretch
    R_MIN_BEND = PITCH / (2 * math.sin(math.radians(JOINT_RANGE_DEG) / 2))
    D_MIN_MITRE = (p["seg_length"] + p["seg_height"]) / math.sqrt(2)
    flues = {}
    for D in (100, 120, 150, 180, 200, 250, 300):
        R_bend = D                                                             # a tight elbow: centre-line radius = one diameter (input)
        phi = math.degrees(PITCH / R_bend)
        flues[f"{D} mm"] = {"bracing_possible": D_MIN_BRACE <= D <= D_MAX_BRACE,
                            "legs_for_bracing": "folded" if D < D_MIN_BRACE else ("stretched" if D > D_MAX_BRACE else "ok"),
                            "elbow_joint_angle_deg": phi, "elbow_ok": phi <= JOINT_RANGE_DEG, "mitre_corner_ok": D >= D_MIN_MITRE,
                            "legs_reach_floor": REACH["drop_z"] >= 0}
    return {"fit_mm": {"bracing_min": D_MIN_BRACE, "bracing_max": D_MAX_BRACE, "min_bend_radius": R_MIN_BEND, "mitre_min": D_MIN_MITRE},
            "joint_range_deg": JOINT_RANGE_DEG, "flues": flues}


def persephone_mass(p: dict, seg_volume: float, leg_volume: float, head_volume: float, segment_types=PERSEPHONE_SEGMENT_TYPES,
                    config=PERSEPHONE_CONFIG, bracing_inputs: dict = PERSEPHONE_BRACING) -> dict:
    """17 cells 7 (the mass budget over the CAD volumes), 9 (the segment modules) and 14 (bracing a vertical flue: the
    fixed-point loop that picks the lightest self-locking leg servo, the crawler's mass, the bracing table)."""
    N = p["n_segments"]
    RHO_PA12CF = 1.1e-3                      # g/mm^3 printed PA12-CF (shell, legs)
    SERVOS = act.table("servo")              # candidate leg servos from the shared library, lightest first
    SERVOS = SERVOS[SERVOS["self_locking"]]  # a braced leg must hold for free: worm-output servos only
    JOINT_SERVO = act.get("mini 32 g").as_dict()   # pitch + yaw per body joint (moves, never holds a bracing load)

    def mass_budget_base(servo):                                                           # cell 7
        s = SERVOS.loc[servo]
        per_segment = pd.Series({
            "shell + hatch + rib (from CAD)": seg_volume * RHO_PA12CF,
            "legs 4x (from CAD)": 4 * leg_volume * RHO_PA12CF,
            f"leg servos 12x ({servo}; hip yaw, hip pitch, knee per leg)": 12 * s["mass_g"],
            "joint servos 2x (pitch, yaw)": 2 * JOINT_SERVO["mass_g"],
            "segment PCB, IMU, contact sensors": 14.0,
            "wiring, pins, pads": 12.0,
        })
        head_g = head_volume * RHO_PA12CF + 2 * 6.0 + 25.0 + 8.0 + 30.0      # lights, camera, IMU, radio + antenna
        tail_g = 25.0                                                          # tether gland, strain relief
        return per_segment, N * per_segment.sum() + head_g + tail_g
    SEGMENT_TYPES = pd.DataFrame(list(segment_types), columns=["type", "payload_g", "what"]).set_index("type")   # cell 9
    CONFIG = list(config)
    MODULES_G = float(SEGMENT_TYPES.loc[CONFIG, "payload_g"].sum())

    def mass_budget(servo):
        per_segment, total_g = mass_budget_base(servo)
        return per_segment, total_g + MODULES_G

    def reach_y(hip_deg, knee_deg):                                                        # cell 11
        return p["femur_length"] * math.cos(math.radians(hip_deg)) + p["tibia_length"] * math.cos(math.radians(knee_deg))
    MU_SOOT = bracing_inputs["mu_soot"]                                                    # cell 14
    SF_SLIP, SF_TORQUE = bracing_inputs["sf_slip"], bracing_inputs["sf_torque"]
    TETHER_G_PER_M = bracing_inputs["tether_g_per_m"]
    HEIGHT_M = bracing_inputs["height_m"]
    STANCE_FRACTION = bracing_inputs["stance_fraction"]
    D_WORK = bracing_inputs["d_work_mm"]

    def legs_for(D):
        """Hip and knee angles that put the foot on the wall of a flue of diameter D (both legs of a pair reach the same wall distance)."""
        wall = (D - p["seg_width"]) / 2 - 6 - p["foot_diameter"] / 2        # hip to wall
        best = min(((abs(reach_y(h, k) - wall), h, k) for h in np.arange(-30, 60, 2.0) for k in np.arange(20, 120, 2.0)), key=lambda t: t[0])
        return best[1], best[2]

    def bracing(m_kg, D=D_WORK, mu=MU_SOOT, height_m=HEIGHT_M):
        W = (m_kg + TETHER_G_PER_M / 1000 * height_m) * G
        n_stance = STANCE_FRACTION * 4 * N
        N_leg = SF_SLIP * W / (mu * n_stance)
        hip, knee = legs_for(D)
        lever_hip = reach_y(hip, knee)                                        # normal force x horizontal reach (the pad pushes straight at the wall)
        lever_knee = p["tibia_length"] * math.cos(math.radians(knee))
        friction_leg = W / n_stance                                           # the share of the weight each stance leg carries along the wall
        tau_hip = (N_leg * lever_hip + friction_leg * (p["femur_length"] * math.sin(math.radians(hip)) + p["tibia_length"] * math.sin(math.radians(knee)))) / 1000
        tau_knee = (N_leg * lever_knee + friction_leg * p["tibia_length"] * math.sin(math.radians(knee))) / 1000
        return {"W_total_N": W, "N_leg_N": N_leg, "friction_leg_N": friction_leg, "hip_deg": hip, "knee_deg": knee, "tau_hip_Nm": tau_hip, "tau_knee_Nm": tau_knee}
    choice = None
    for servo in SERVOS.index:                                                # lightest first
        per_segment, m_total_g = mass_budget(servo)
        b = bracing(m_total_g / 1000)
        if SF_TORQUE * max(b["tau_hip_Nm"], b["tau_knee_Nm"]) <= SERVOS.loc[servo, "stall_Nm"]:
            choice = servo; break
    out = {"segment_types": {t: {"payload_g": float(r["payload_g"]), "what": r["what"]} for t, r in SEGMENT_TYPES.iterrows()},
           "config": CONFIG, "modules_g": MODULES_G, "joint_servo": JOINT_SERVO, "bracing_inputs": dict(bracing_inputs),
           "leg_servo": choice, "leg_servo_data": None, "per_segment_g": None, "mass_kg": None, "bracing_design_point": None,
           "bracing": None}
    if choice is None:                                                        # no catalogue servo holds: said, not hidden
        return out
    per_segment, M_G = mass_budget(choice); M_CRAWLER = M_G / 1000
    B = bracing(M_CRAWLER)
    rows = {}
    for D in (120, 150, 180, 200):
        for h in (0.0, 7.5, 15.0):
            rows[f"{D} mm flue, {h:.0f} m up"] = bracing(M_CRAWLER, D=D, height_m=h)
    brace_tab = pd.DataFrame(rows).T
    brace_tab["servo SF"] = SERVOS.loc[choice, "stall_Nm"] / brace_tab[["tau_hip_Nm", "tau_knee_Nm"]].max(axis=1)
    out.update(leg_servo_data=act.get(choice).as_dict(), per_segment_g={k: float(v) for k, v in per_segment.items()},
               mass_kg=float(M_CRAWLER), bracing_design_point={k: float(v) for k, v in B.items()},
               bracing={r: {k: float(v) for k, v in row.items()} for r, row in _table(brace_tab, {"servo SF": "servo_SF"}).items()})
    return out


def persephone_parts(node: Assembly) -> Assembly:
    """On Persephone's body: the segment, leg and head CAD (17 cell 4: volumes, the segment's area, pitch, length, leg
    reach), the fit (cell 11) and the mass budget with the bracing servo (cells 7, 9, 14)."""
    d = myropod.Myropod()
    p = d.resolve()
    parts = {k: d.generate(part=k).measure() for k in ("segment", "leg", "head")}
    PITCH = d.pitch_length(p); N = p["n_segments"]; REACH = d.leg_reach(p)                 # cell 4
    LENGTH = p["head_length"] + N * PITCH
    mass = persephone_mass(p, parts["segment"]["volume"], parts["leg"]["volume"], parts["head"]["volume"])
    return node.record(part_volume_mm3={k: m["volume"] for k, m in parts.items()}, segment_surface_area_mm2=parts["segment"]["surface_area"],
                       pitch_mm=PITCH, length_mm=LENGTH, n_segments=N, leg_reach_mm=REACH, **persephone_fit(p, PITCH, REACH), **mass)


# ------------------------------------------------------------------------------------------------ Apheloria (notebook 19)
def apheloria_configs(aph, p: dict, seg_volume: float, plate_volume: float, leg_volume: float, head_volume: float,
                      modules=APH_MODULES, configs: dict = APH_CONFIGS) -> dict:
    """19 cell 7: the base segment over the CAD volumes, the module table, the head, and the mini / standard / max
    configurations (segments, mass, batteries, length in walk mode, ball diameter)."""
    RHO_PA12CF, RHO_AL = 1.1e-3, 2.7e-3
    ACT = {"leg": act.get("smart servo 12 Nm").as_dict(), "joint": act.get("qdd 60 Nm").as_dict()}      # from the shared library
    base_segment = pd.Series({"body shell (CAD, PA12-CF)": (seg_volume - plate_volume) * RHO_PA12CF, "armour plate (CAD, aluminium)": plate_volume * RHO_AL,
                              "legs 4x (CAD)": 4 * leg_volume * RHO_PA12CF, "leg servos 12x (hip yaw, hip pitch, knee per leg)": 12 * ACT["leg"]["mass_g"], "joint actuators 2x": 2 * ACT["joint"]["mass_g"], "MCU, IMU, bus, sensors": 90.0, "wiring, pins, pads": 80.0})
    MODULES = pd.DataFrame(list(modules), columns=["module", "payload_g"]).set_index("module")
    MODULES["segment_g"] = base_segment.sum() + MODULES["payload_g"]
    head_g = head_volume * RHO_PA12CF + 2 * 180 + 350 + 150 + 200 + ACT["joint"]["mass_g"]   # stereo cameras, lidar, illuminators, computer, its joint half
    CONFIGS = configs
    cfg = pd.DataFrame({k: {"segments": len(v), "mass [kg]": (head_g + MODULES.loc[v, "segment_g"].sum()) / 1000, "batteries [Wh]": 300 * v.count("battery (Li-ion 300 Wh)"),
                            "length walk [mm]": p["head_length"] + len(v) * aph.pitch_length(p), "ball diameter [mm]": 2 * aph.ball_radius(aph.resolve(n_segments=len(v)))} for k, v in CONFIGS.items()}).T
    configs_out = _table(cfg, {"mass [kg]": "mass_kg", "batteries [Wh]": "batteries_Wh", "length walk [mm]": "length_walk_mm",
                               "ball diameter [mm]": "ball_diameter_mm"})
    for row in configs_out.values():
        row["segments"] = int(row["segments"])
    return {"base_segment_g": {k: float(v) for k, v in base_segment.items()}, "head_g": float(head_g),
            "modules_g": {m: {"payload_g": float(r["payload_g"]), "segment_g": float(r["segment_g"])} for m, r in MODULES.iterrows()},
            "configs": configs_out, "config_modules": {k: list(v) for k, v in CONFIGS.items()}, "actuators": ACT}


def apheloria_ball(aph, p: dict, M_STD: float, inputs: dict = APH_BALL) -> dict:
    """19 cell 9: the standard configuration as a ball: slope to start rolling, kinetic energy at 15 cm/s, rolling 10 m
    downhill at 5°, a deployment drop on one plate (impact force, half-sine peak on the pad, as g)."""
    R_BALL = aph.ball_radius(p) / 1000
    CG_OFFSET = inputs["cg_offset_m"]                                                          # the CG off the ball's centre [m]
    slope_start = math.degrees(math.asin(CG_OFFSET / R_BALL))
    V_ROLL = inputs["v_roll_m_s"]
    E_ROLL = 0.5 * M_STD * V_ROLL**2 * (1 + 0.6)                                             # translation + rotation (a thick ring: I ≈ 0.6 m R²)
    DROP_M, DELTA_M = inputs["drop_m"], inputs["delta_m"]                                     # deployment drop and the crush of the foam liner
    F_IMPACT = M_STD * G * (DROP_M / DELTA_M + 1)
    F_IMPACT_PAD = F_IMPACT * math.pi / 2                                                     # half-sine peak
    return {"standard_config_kg": M_STD, "ball_diameter_m": 2 * R_BALL, "slope_to_start_deg": slope_start, "kinetic_energy_J": E_ROLL,
            "roll_10m_at_5deg_m_s": math.sqrt(2 * G * 10 * math.sin(math.radians(5)) / 1.6), "impact_force_N": F_IMPACT,
            "impact_peak_pad_N": F_IMPACT_PAD, "impact_g": F_IMPACT_PAD / (M_STD * G), "drop_m": DROP_M, "inputs": dict(inputs)}


def apheloria_curl(aph, p: dict, segment_g: float, joint_stall_Nm: float, sf: float = 1.5) -> dict:
    """19 cell 11: curling up, the moment at a joint lifting k segments straight out (``segment_g``: the camera
    segment), the most it lifts with SF 1.5 against the joint actuator, and half the chain."""
    N = p["n_segments"]
    PITCH_M = aph.pitch_length(p) / 1000
    W_SEG = segment_g / 1000 * G                                                              # a typical segment with a light module

    def curl_moment(k):                                                                        # k segments hanging straight out from a joint
        return sum(W_SEG * (PITCH_M * (i + 0.5)) for i in range(k))
    ks = np.arange(1, N + 1)
    K_MAX = int(max((k for k in ks if curl_moment(int(k)) * sf <= joint_stall_Nm), default=0))
    M_CURL = curl_moment(K_MAX)
    return {"segment_weight_N": W_SEG, "k_max_lifted": K_MAX, "moment_k_max_Nm": M_CURL, "moment_half_chain_Nm": curl_moment(N // 2),
            "joint_stall_Nm": joint_stall_Nm, "sf_half_chain": joint_stall_Nm / curl_moment(N // 2), "sf_required": sf,
            "moment_by_segments_lifted_Nm": {str(int(k)): curl_moment(int(k)) for k in ks}}


def apheloria_design(node: Assembly) -> Assembly:
    """On Apheloria's body: notebook 19's design (``apheloria.Apheloria`` defaults, mm): geometry (cell 4), base
    segment, modules and configurations over the CAD volumes (cell 7), the ball (cell 9), curling up (cell 11)."""
    a = apheloria.Apheloria()
    p = a.resolve()
    vol = {k: a.generate(part=k).volume for k in ("segment", "plate", "leg", "head")}
    c = apheloria_configs(a, p, vol["segment"], vol["plate"], vol["leg"], vol["head"])
    M_STD = c["configs"]["standard (8)"]["mass_kg"]
    curl = apheloria_curl(a, p, c["modules_g"]["camera"]["segment_g"], c["actuators"]["joint"]["stall_Nm"])
    return node.record(cad_design=p, part_volume_mm3=vol, pitch_mm=a.pitch_length(p), ball_radius_mm=a.ball_radius(p),
                       coil_radius_mm=a.coil_radius(p), coil_angle_deg=a.coil_angle_deg(p), leg_reach_mm=a.leg_reach(p),
                       **c, ball=apheloria_ball(a, p, M_STD), curl=curl, mass_note=APH_MASS_NOTE)


def run(*, machines=MACHINES, run_sim: bool = True, vida_path: Path | None = None, include: str = "results", force: bool = False,
        redo=(), progress: bool = True, export: bool = True, **_ignored) -> Assembly:
    vida_path = Path(vida_path or DATA / f"{NAME}.vida")
    prior = vida.load(vida_path) if vida_path.is_file() and not force else None
    root = build(machines)
    root.reuse(prior)
    from tqdm.auto import tqdm
    for m in tqdm(machines, desc="walkers (CAD, robots, MuJoCo scenes)", disable=not progress):
        node = root.child(m)
        b = node.child("body")
        if m in redo or "body" in redo:
            b.forget()
        if not b.results:
            body(b, m)
        # the notebooks' design numbers on the body, also on a saved body made before them (16, 17, 18, 19)
        if m == "dog" and any(k not in b.results for k in DOG_BODY_KEYS):
            b.record(geometry_mm=dog_geometry(node.params["design"]))
        if m == "cleopatra" and any(k not in b.results for k in CLEO_KEYS):
            cleopatra_loads(b)
        if m == "persephone" and any(k not in b.results for k in PERSEPHONE_KEYS):
            persephone_parts(b)
        if m == "apheloria" and any(k not in b.results for k in APH_KEYS):
            apheloria_design(b)
        if m == "dog":
            g = node.child("gaits")
            if "gaits" in redo:
                g.forget()
            if not g.results:
                dog_gaits(g, b.results["mass_kg"])
            if any(k not in g.results for k in DOG_TORQUE_KEYS):          # 16 cell 9, also on a saved gaits node
                dog_torques(g, node.params["design"], b.results["mass_kg"])
            if any(k not in g.results for k in DOG_LOAD_KEYS):            # 16 cells 8, 11, 13
                dog_loads(g, node.params["design"], b.results["mass_kg"])
        if m == "apheloria":
            for scene in ("pack", "unpack"):
                sc = add_after(node, Assembly(scene, "episode", params={"design": node.params["design"], "scene": scene}),
                               prior and _child(prior, m), redo)
                run_scene(sc, lambda scene=scene: ap_scene.scene(scene), run=run_sim)
    root.record(masses_kg={m: root.child(f"{m}/body").results["mass_kg"] for m in machines if m != "persephone"})
    if "persephone" in machines:                                  # 17 cell 14's crawler (masses_kg stays as it was)
        root.record(persephone_mass_kg=root.child("persephone/body").results.get("mass_kg"))
    root.meta["workflow"] = f"assemblies.workflows.{NAME}"
    root.meta["saved_to"] = str(root.save(vida_path, include=include))
    if export:                                                    # plain data for notebooks (assemblies.results)
        root.meta["results_to"] = str(results.write(root, vida_path))
    return root


def _child(tree: Assembly, name: str):
    try:
        return tree.child(name)
    except KeyError:
        return None


def _parser():
    ap = parser("The walkers: masses, the dog's foot loads, Persephone's CAD, Apheloria packing in MuJoCo", cfd=False, fea=False)
    ap.add_argument("--machine", dest="machines", action="append", choices=MACHINES, help="only these (repeatable)")
    ap.add_argument("--no-sim", dest="run_sim", action="store_false", help="run no MuJoCo scene (simulated ones are kept)")
    return ap


def _run_cli(*, machines=None, **kw):
    return run(machines=tuple(machines or MACHINES), **kw)


if __name__ == "__main__":
    raise SystemExit(main(_run_cli, _parser(), DATA / f"{NAME}.vida"))
