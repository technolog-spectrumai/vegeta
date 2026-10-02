"""Sikarian Lobster Nefri (notebook 24) as a Chiron robot underwater: the design (``lobster.py``, mm) in SI, the mass
and volume budget (CAD volumes × densities, the catalogue's sealed servos, listed parts, syntactic foam sized so
the robot sinks by ``NET_FRACTION`` of its mass), and the water: MuJoCo's fluid drag (``density``, ``viscosity``,
ellipsoid fluid shapes) plus the ``Water`` hook — buoyancy per body at its centre of buoyancy (MuJoCo applies none)
and the tail thruster's force along its shroud axis, from a Boreas BEMT table in water.

==================  ======================  ======================================================================
joint               drive                   convention
==================  ======================  ======================================================================
``<leg>_yaw``       sealed servo 1.5 Nm     hip yaw: + swings the foot forward (left legs about −z, right about +z)
``<leg>_lift``      sealed servo 1.5 Nm     hip pitch: + lifts the foot (left legs about +x, right about −x)
``<s>_shoulder_yaw``sealed servo 1.5 Nm     claw arm yaw about +z (+ to the left)
``<s>_shoulder``    sealed servo 1.5 Nm     upper arm about −y: + raises it; its elevation is q
``<s>_elbow``       sealed servo 1.5 Nm     palm about +y: + bends it down; its elevation is q_sh − q_el
``<s>_wrist``       sealed servo 1.5 Nm     hand about +y: + bends it down; the jaws' elevation is q_sh − q_el − q_wr
``<s>_jaw_upper``   sealed servo 12 Nm,     about the jaw pin: + opens the upper jaw; the lower jaw mirrors it (one
``<s>_jaw_lower``   worm (one drive)        worm drive and a linkage in the machine: here each jaw has its torque)
``tail{1,2}_yaw``   sealed servo 3 Nm       about +z: + swings the tail tip to the right (−y)
``tail{1,2}_pitch`` sealed servo 3 Nm       about +y: + lifts the tail tip
==================  ======================  ======================================================================

Legs: ``FL``, ``ML``, ``RL`` (left), ``FR``, ``MR``, ``RR``. Sides of the claws: ``L``, ``R`` (the right claw cuts).
The thruster's thrust acts on the ``thruster`` body along its +x axis (towards the body); the jet leaves along −x.

    import lobster_robot as lr
    robot = lr.lobster()                  # ~7 kg
    lab = lr.lobster_lab(terrain, props=...)          # water on, the Water hook attached (lab.water)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from vegeta import boreas
from vegeta.chiron import ChironLab, FootSpec, Geom, Joint, Link, PointMass, Robot, Servo

import actuators as act

__all__ = ["Variant", "VARIANTS", "ORNATUS", "ORNATUS_CAD", "variant_of", "NEFRI", "CAD", "WATER", "MATERIALS", "PARTS_KG", "SERVO_KEYS", "SERVO_DENSITY", "GAINS", "NET_FRACTION",
           "LEGS", "TRIPODS", "CLAWS", "TAIL_JOINTS", "THRUSTER", "ASSUMPTIONS", "LAB_OPTIONS", "design_params",
           "cad_numbers", "geometry", "budget", "thruster_table", "thrust", "leg_joints", "claw_joints", "nominal_qpos",
           "lobster", "lobster_lab", "Water"]

#: The Nefri design (lobster.py defaults, mm and degrees).
NEFRI = {
    "shell_length": 320.0, "shell_width": 200.0, "shell_height": 130.0, "shell_thickness": 3.0, "shell_chamfer": 28.0,
    "shell_bottom": 75.0, "housing_diameter": 110.0, "housing_wall": 3.0, "housing_length": 230.0, "cap_thickness": 8.0,
    "hip_spacing": 95.0, "hip_height": 35.0, "coxa_length": 22.0, "femur_length": 70.0, "tibia_length": 105.0,
    "tibia_splay_deg": 10.0, "leg_width": 16.0, "leg_thickness": 8.0, "foot_diameter": 20.0, "claw_y": 55.0,
    "claw_height": 70.0, "upper_arm_length": 90.0, "palm_length": 45.0, "hand_length": 22.0, "arm_width": 22.0, "jaw_length": 70.0,
    "jaw_thickness": 6.0, "jaw_depth": 18.0, "jaw_pin_diameter": 6.0, "cutter_x": 14.0, "cutter_depth": 4.0,
    "hook_length": 0.0, "hook_width": 8.0, "claw_pitch_deg": 20.0, "claw_elbow_deg": 40.0, "claw_wrist_deg": 0.0, "jaw_open_deg": 20.0,
    "tail_height": 70.0, "tail_segment_1": 85.0, "tail_segment_2": 70.0, "tail_diameter": 50.0, "tail_yaw_deg": 0.0,
    "tail_pitch_deg": 0.0, "shroud_length": 60.0, "shroud_inner": 82.0, "shroud_wall": 5.0, "prop_diameter": 76.0,
    "prop_pitch": 70.0, "prop_blades": 3, "prop_hub": 20.0, "mast_height": 90.0, "mast_diameter": 7.0,
}

#: CAD of the default Nefri (notebook 24 §1; mm³, mm; ``cad_numbers(recompute=True)`` rebuilds them).
CAD = {"shell_volume": 614803.3, "shell_com_z": 142.8, "housing_volume": 367861.7, "leg_volume": 37246.8, "upper_arm_volume": 47471.6, "jaw_volume": 7180.7, "tail_segment_volume": 186293.3, "shroud_volume": 101627.1, "propeller_volume": 6064.5}

#: Water (input): fresh water at 15 °C — rivers, canals, outfalls (sea water: 1025 kg/m³, the notebook compares).
WATER = {"density": 1000.0, "viscosity": 1.14e-3, "g": 9.81}
#: Densities [kg/m³] of the materials.
MATERIALS = {"ASA shell": 1070.0, "Al 6082 housing": 2700.0, "PA12-CF legs and arms": 1150.0, "tool steel jaws": 7850.0,
             "PA12 shroud and propeller": 1010.0, "syntactic foam (10 m class)": 400.0}
#: Listed parts [kg] (and where their volume counts: inside the housing they displace nothing extra).
PARTS_KG = {"battery 4S3P Li-ion 21700, 207 Wh (in the housing)": 0.80, "computer, IMU, depth sensor (in the housing)": 0.12,
            "acoustic modem and radio relay (in the housing)": 0.10, "power electronics, thruster ESC (in the housing)": 0.10,
            "O-rings, penetrators, cable glands": 0.12, "cameras 3x low-light (sealed)": 0.09, "imaging sonar (sealed)": 0.15,
            "hydrophone array and antenna masts": 0.09, "wiring outside the housing": 0.08,
            "thruster motor, sealed BLDC": 0.18, "tail segments 2x (PA12 tube, foam-filled)": 0.14,
            "palm housings 2x": 0.06, "foot balls 6x (rubber)": 0.03}
#: Volumes [m³] of the listed parts outside the housing (they displace water; the rest are inside it).
PARTS_VOLUME = {"cameras 3x low-light (sealed)": 45e-6, "imaging sonar (sealed)": 90e-6, "hydrophone array and antenna masts": 40e-6,
                "thruster motor, sealed BLDC": 60e-6, "tail segments 2x (PA12 tube, foam-filled)": 2 * 140e-6,
                "palm housings 2x": 2 * 25e-6, "foot balls 6x (rubber)": 6 * 4.2e-6, "wiring outside the housing": 50e-6}
SERVO_KEYS = {"leg": "sealed servo 1.5 Nm", "arm": "sealed servo 1.5 Nm", "jaw": "sealed servo 12 Nm, worm", "tail": "sealed servo 3 Nm"}
SERVO_DENSITY = 2200.0                    # kg/m³: an aluminium servo case with its motor and gears (displaced volume)
#: Position-loop gains (kp N·m/rad, kd N·m·s/rad, armature kg·m²) per joint kind (ASSUMPTIONS).
GAINS = {"leg": (12.0, 0.25, 0.004), "arm": (15.0, 0.25, 0.004), "jaw": (20.0, 0.3, 0.01), "tail": (10.0, 0.2, 0.004)}
NET_FRACTION = 0.10                       # the robot sinks with 10 % of its mass (the datasheet: it stands on the bottom)
DUCT_GAIN = 1.20                          # bollard thrust of the shrouded propeller over the open one (notebook 24 §3)

LEGS = {"FL": ("F", 1), "ML": ("M", 1), "RL": ("R", 1), "FR": ("F", -1), "MR": ("M", -1), "RR": ("R", -1)}
TRIPODS = (("FL", "MR", "RL"), ("FR", "ML", "RR"))
CLAWS = {"L": 1, "R": -1}
TAIL_JOINTS = ("tail1_yaw", "tail1_pitch", "tail2_yaw", "tail2_pitch")
#: Tail pitch ranges [deg]: the first knuckle curls the tail up over the back like a crayfish's (to press the robot
#: onto the bottom the thrust must come down from above the body); yaw ±50° at both knuckles.
TAIL_PITCH_RANGE = {1: (-50.0, 150.0), 2: (-60.0, 100.0)}
#: The thruster (inputs): a 76 mm 3-blade propeller in its shroud, a 4S sealed BLDC (Boreas models).
THRUSTER = {"motor": boreas.Motor("sealed BLDC 250 kV", kv_rpm_per_volt=250.0, resistance_ohm=0.15, no_load_current_a=0.4,
                                  max_current_a=12.0, mass_kg=0.18, source="assumed: underwater thruster motor class, 4S"),
            "battery": boreas.Battery("4S3P Li-ion 21700", cells=4, capacity_ah=14.4, cell_voltage_nominal=3.6,
                                      usable_fraction=0.85, mass_kg=0.80),
            "section": boreas.Airfoil(name="marine blade section", cl_alpha=5.5, alpha0_deg=-2.0, cl_max=1.0, cd0=0.02,
                                      k=0.05, source="assumed (as notebooks 13/14)"),
            "rpm_max": 3300.0, "spool_s": 0.15}

ASSUMPTIONS = {
    "water": "fresh water 1000 kg/m³, 1.14 mPa·s; MuJoCo's ellipsoid fluid model on the shell, legs, arms and tail "
             "(drag, added mass); buoyancy by the Water hook (MuJoCo applies none) at each body's centre of buoyancy",
    "buoyancy_volumes": "hull: the housing's envelope, the shell's material, the foam, the servos in the hull and "
                        "the sealed parts outside the housing; legs, arms, tail: their CAD or listed volumes; the "
                        "shell is free-flooding (its interior is water)",
    "foam": "syntactic foam (400 kg/m³, 10 m class) in the shell's corners beside the housing, sized so the robot "
            "sinks with NET_FRACTION of its mass; its centre 35 mm above the housing axis, the battery 35 mm below it (the robot rights itself) "
            "and placed along x so the centre of buoyancy is over the centre of gravity standing (lobster(trim=True)); "
            "the battery at the rear of the housing (BATTERY_X): the steel jaws and worm drives are heavy forward, "
            "the tail is buoyant aft",
    "gains": "kp so a leg sags < 1° under its share of the wet weight and the current; kd for ζ ≈ 0.5 with the leg's "
             "inertia and added mass; armature: a 1 g·cm² rotor through 1:200 (sealed servo class)",
    "thruster": "thrust from Boreas BEMT in water at the axial inflow (a table over rpm and speed) × DUCT_GAIN 1.2 "
                "(an accelerating duct at bollard, momentum theory with a 1.2 expansion ratio, notebook 24 §3); "
                "first-order spool-up 0.15 s; reaction torque −Q on the thruster body",
    "jaws": "two hinges with the worm drive's torque each (one drive and a linkage in the machine), as the Onager "
            "Manus's pincer; the cutter notch 14 mm from the pin: 12 N·m / 0.014 m = 860 N on a rope there; the "
            "notch's outer flank is a 4 mm tooth on each inner face (NOTCH_TOOTH): without it the closing jaws push "
            "a round rope out to ~36 mm (the scissors' push-out) where the squeeze is only ~340 N",
}

LAB_OPTIONS = {"timestep": 0.0005, "control_dt": 0.002, "log_dt": 0.02, "density": WATER["density"],
               "viscosity": WATER["viscosity"], "contact_solref": (0.005, 1.0), "heightfield_cell": 0.02,
               "course_extent": (-2.0, 8.0, -2.0, 2.0)}


# ----------------------------------------------------------------------------------------------- the design
def design_params(overrides: dict | None = None, variant=None) -> dict:
    """The variant's parameters (Nefri by default) with ``overrides``."""
    p = dict(variant_of(variant).params)
    for k, v in (overrides or {}).items():
        if k not in NEFRI:
            raise KeyError(f"unknown SikarianLobster parameter {k!r}")
        p[k] = float(v)
    return p


def cad_numbers(p: dict | None = None, *, recompute: bool = False, variant=None) -> dict:
    """CAD volumes [mm³] (and the shell's centroid height [mm]); the stored numbers (``CAD``, ``ORNATUS_CAD``) for a
    variant's own design."""
    v = variant_of(variant)
    p = dict(v.params) if p is None else dict(p)
    if not recompute:
        for vv in VARIANTS.values():
            if vv.cad and p == vv.params:
                return dict(vv.cad)
    import lobster

    d = lobster.SikarianLobster()
    over = {k: v for k, v in p.items() if k in {prm.name for prm in d.parameters}}
    vol = lambda part: d.generate(part=part, **over).measure()["volume"]          # noqa: E731
    shell = d.generate(part="shell", **over).measure()
    return {"shell_volume": shell["volume"], "shell_com_z": shell["center_of_mass"][2], "housing_volume": vol("housing"),
            "leg_volume": vol("leg"), "upper_arm_volume": vol("upper_arm"), "jaw_volume": vol("jaw"),
            "tail_segment_volume": vol("tail_segment"), "shroud_volume": vol("shroud"), "propeller_volume": vol("propeller")}


def geometry(p: dict | None = None) -> dict:
    """Derived geometry [m] in the hull frame (origin at the shell's centre, standing): hips, claws, tail, sizes."""
    import lobster

    p = design_params() if p is None else p
    mm = 1e-3
    zc = p["shell_bottom"] + p["shell_height"] / 2
    hips = {}
    for name, (x, y, z, side) in lobster.SikarianLobster.hips(p).items():
        hips[name] = (x * mm, y * mm, (z - zc) * mm)
    fx, fy, fz = lobster.SikarianLobster.foot_offset(p)
    (j1, _), (j2, _) = lobster.SikarianLobster.tail_frames(p, 0.0, 0.0)[0]
    return {"zc": zc * mm, "hips": hips, "foot": (fx * mm, fy * mm, fz * mm), "foot_r": p["foot_diameter"] / 2 * mm,
            "femur": p["femur_length"] * mm, "tibia": p["tibia_length"] * mm, "splay": math.radians(p["tibia_splay_deg"]),
            "claw": ((p["shell_length"] / 2 - 10.0) * mm, p["claw_y"] * mm, (p["shell_bottom"] + p["claw_height"] - zc) * mm),
            "Lu": p["upper_arm_length"] * mm, "Lp": p["palm_length"] * mm, "Lh": p["hand_length"] * mm, "Lj": p["jaw_length"] * mm,
            "cutter": p["cutter_x"] * mm, "jaw_t": p["jaw_thickness"] * mm, "jaw_d": p["jaw_depth"] * mm,
            "hook": p["hook_length"] * mm, "hook_w": p["hook_width"] * mm, "arm_w": p["arm_width"] * mm,
            "tail0": (j1[0] * mm, 0.0, (j1[2] - zc) * mm), "L1": p["tail_segment_1"] * mm, "L2": p["tail_segment_2"] * mm,
            "tail_r": p["tail_diameter"] / 2 * mm, "shroud_L": p["shroud_length"] * mm,
            "shroud_r": (p["shroud_inner"] / 2 + p["shroud_wall"]) * mm, "prop_D": p["prop_diameter"] * mm,
            "shell": (p["shell_length"] * mm, p["shell_width"] * mm, p["shell_height"] * mm),
            "housing": (p["housing_diameter"] / 2 * mm, p["housing_length"] * mm)}


def _servo_count():
    return {"leg": 12, "arm": 8, "jaw": 2, "tail": 4}


BALLAST_DENSITY = 7850.0                  # kg/m³: steel trim plates (a variant with a target mass carries what is missing)


def budget(p: dict | None = None, cad: dict | None = None, variant=None) -> dict:
    """Mass [kg] and displaced volume [m³] of every part, the foam that trims the robot to ``NET_FRACTION``, and
    the totals: {"parts": {name: (kg, m³)}, "mass", "volume", "net_kg" (mass − ρV), "foam_volume", ...}. A variant
    with a ``target_mass`` gets steel ballast plates so the total is exactly that (``ballast_kg``; 0 and
    ``over_target_kg`` > 0 when the parts alone are heavier)."""
    v = variant_of(variant)
    p = dict(v.params) if p is None else p
    cad = cad_numbers(p, variant=v) if cad is None else cad
    PARTS_KG, PARTS_VOLUME, SERVO_KEYS = v.parts_kg, v.parts_volume, v.servo_keys
    rho = WATER["density"]
    g = geometry(p)
    mm3 = 1e-9
    r_h, L_h = g["housing"]
    parts = {
        "shell (ASA, CAD)": (cad["shell_volume"] * mm3 * MATERIALS["ASA shell"], cad["shell_volume"] * mm3),
        "pressure housing (Al 6082, CAD; displaces its envelope)": (cad["housing_volume"] * mm3 * MATERIALS["Al 6082 housing"], math.pi * r_h ** 2 * L_h),
        "legs 6x (PA12-CF, CAD)": (6 * cad["leg_volume"] * mm3 * MATERIALS["PA12-CF legs and arms"], 6 * cad["leg_volume"] * mm3),
        "jaws 4x (tool steel, CAD)": (4 * cad["jaw_volume"] * mm3 * MATERIALS["tool steel jaws"], 4 * cad["jaw_volume"] * mm3),
        "arms 2x (PA12-CF)": (2 * 0.04, 2 * 0.04 / MATERIALS["PA12-CF legs and arms"]),
        "shroud + propeller (PA12, CAD)": ((cad["shroud_volume"] + cad["propeller_volume"]) * mm3 * MATERIALS["PA12 shroud and propeller"],
                                           (cad["shroud_volume"] + cad["propeller_volume"]) * mm3),
    }
    for kind, n in _servo_count().items():
        a = act.get(SERVO_KEYS[kind])
        parts[f"{kind} servos {n}x ({a.key})"] = (n * a.mass_g / 1000.0, n * a.mass_g / 1000.0 / SERVO_DENSITY)
    for k, m in PARTS_KG.items():
        parts[k] = (m, PARTS_VOLUME.get(k, 0.0))
    m0 = sum(m for m, _ in parts.values())
    v0 = sum(vol for _, vol in parts.values())
    rf = MATERIALS["syntactic foam (10 m class)"]
    f = NET_FRACTION
    mb, over = 0.0, 0.0
    if v.target_mass:
        # total M = m0 + mb + rf Vf = target and M − ρ (v0 + mb/ρb + Vf) = f M: two equations for the ballast mb and
        # the foam Vf
        T = v.target_mass
        mb = (T - m0 - rf * ((1 - f) * T / rho - v0)) / (1 - rf / BALLAST_DENSITY)
        if mb < 0:
            over, mb = -mb, 0.0
        parts["ballast plates (steel, trim to the target mass)"] = (mb, mb / BALLAST_DENSITY)
        m0 += mb
        v0 += mb / BALLAST_DENSITY
    # (m0 + rf Vf) − ρ (v0 + Vf) = f (m0 + rf Vf)  →  Vf
    Vf = ((1 - f) * m0 - rho * v0) / (rho - (1 - f) * rf)
    Vf = max(Vf, 0.0)
    parts["syntactic foam trim"] = (rf * Vf, Vf)
    mass = sum(m for m, _ in parts.values())
    volume = sum(vol for _, vol in parts.values())
    L, W, H = g["shell"]
    free = (L - 2 * 0.03) * (W - 2 * r_h - 0.01) * (H - 0.01)               # the shell's corners beside the housing
    return {"parts": parts, "mass": mass, "volume": volume, "net_kg": mass - rho * volume, "foam_volume": Vf,
            "foam_space": free, "wet_weight_N": (mass - rho * volume) * WATER["g"], "ballast_kg": mb, "over_target_kg": over}


# ----------------------------------------------------------------------------------------------- the thruster
def _prop(p: dict) -> boreas.Propeller:
    D, P = p["prop_diameter"] / 1000.0, p["prop_pitch"] / 1000.0
    return boreas.Propeller.from_pitch(f"{p['prop_diameter']:.0f} mm {int(p['prop_blades'])}-blade", D, P,
                                       blades=int(p["prop_blades"]), chord_root_m=D * 0.16, chord_max_m=D * 0.26,
                                       chord_tip_m=D * 0.12)


def thruster_table(p: dict | None = None, rpm=None, speed=None, variant=None) -> dict:
    """Boreas BEMT in water over rpm × axial inflow speed: thrust [N] (open propeller) and torque [N·m] arrays."""
    v = variant_of(variant)
    p = dict(v.params) if p is None else p
    THRUSTER = v.thruster
    rpm = np.linspace(0.0, THRUSTER["rpm_max"], 12) if rpm is None else np.asarray(rpm, dtype=float)
    speed = np.array([0.0, 0.5, 1.0, 1.5]) if speed is None else np.asarray(speed, dtype=float)
    prop = _prop(p)
    T = np.zeros((len(rpm), len(speed)))
    Q = np.zeros_like(T)
    for i, n in enumerate(rpm):
        for j, v in enumerate(speed):
            if n <= 0:
                continue
            op = boreas.solve(prop, THRUSTER["section"], float(n), float(v), WATER["density"], speed_of_sound=1500.0)
            T[i, j], Q[i, j] = max(op.thrust, 0.0), max(op.torque, 0.0)
    return {"rpm": rpm, "speed": speed, "thrust": T, "torque": Q, "prop": prop}


def thrust(table: dict, rpm: float, speed: float) -> tuple:
    """(thrust [N] × DUCT_GAIN, torque [N·m]) at ``rpm`` and axial inflow ``speed`` (bilinear in the table)."""
    r, s = table["rpm"], table["speed"]
    rpm = float(np.clip(abs(rpm), r[0], r[-1]))
    speed = float(np.clip(speed, s[0], s[-1]))
    i = int(np.clip(np.searchsorted(r, rpm) - 1, 0, len(r) - 2))
    j = int(np.clip(np.searchsorted(s, speed) - 1, 0, len(s) - 2))
    u = (rpm - r[i]) / (r[i + 1] - r[i])
    w = (speed - s[j]) / (s[j + 1] - s[j])

    def bil(A):
        return (A[i, j] * (1 - u) * (1 - w) + A[i + 1, j] * u * (1 - w) + A[i, j + 1] * (1 - u) * w + A[i + 1, j + 1] * u * w)
    return DUCT_GAIN * bil(table["thrust"]), bil(table["torque"])


# ----------------------------------------------------------------------------------------------- joints
def leg_joints(leg: str) -> list:
    return [f"{leg}_yaw", f"{leg}_lift"]


def claw_joints(side: str) -> list:
    return [f"{side}_shoulder_yaw", f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist", f"{side}_jaw_upper", f"{side}_jaw_lower"]


def _servo(kind: str, variant=None) -> Servo:
    v = variant_of(variant)
    kp, kd, arm = v.gains[kind]
    return Servo.from_actuator(act.get(v.servo_keys[kind]), kp=kp, kd=kd, armature=arm)


#: Claws stowed: arms raised and folded back over the nose, jaws closed.
STOW = {"shoulder_yaw": 0.0, "shoulder": math.radians(35.0), "elbow": math.radians(95.0), "wrist": math.radians(20.0), "jaw": 0.0}


def nominal_qpos(p: dict | None = None) -> dict:
    q = {}
    for leg in LEGS:
        y, l = leg_joints(leg)
        q[y], q[l] = 0.0, 0.0
    for s in CLAWS:
        sy, sh, el, wr, ju, jl = claw_joints(s)
        q.update({sy: STOW["shoulder_yaw"], sh: STOW["shoulder"], el: STOW["elbow"], wr: STOW["wrist"], ju: 0.0, jl: 0.0})
    for j in TAIL_JOINTS:
        q[j] = 0.0
    return q


# ----------------------------------------------------------------------------------------------- the robot
SHELL_RGBA, LEG_RGBA, JAW_RGBA, TAIL_RGBA = (0.27, 0.25, 0.20, 1.0), (0.16, 0.16, 0.15, 1.0), (0.55, 0.53, 0.48, 1.0), (0.22, 0.21, 0.18, 1.0)


def _leg_link(leg: str, g: dict, m_leg: float, foot_mu: float, v=None) -> Link:
    _, side = LEGS[leg]
    yj, lj = leg_joints(leg)
    fx, fy, fz = g["foot"]
    fy *= side
    s = g["splay"]
    knee = (0.0, side * g["femur"], 0.0)
    tib_c = (0.0, side * (g["femur"] + g["tibia"] / 2 * math.sin(s)), -g["tibia"] / 2 * math.cos(s))
    return Link(f"{leg}_leg", pos=g["hips"][leg],
                joints=[Joint(yj, axis=(0, 0, -side), range=(math.radians(-40), math.radians(40)), tag="hip_yaw",
                              servo=_servo("leg", v), leg=leg),
                        Joint(lj, axis=(side, 0, 0), range=(math.radians(-30), math.radians(60)), tag="hip_pitch",
                              servo=_servo("leg", v), leg=leg)],
                geoms=[Geom(f"{leg}_femur", "capsule", (0.007,), fromto=(0, 0, 0, *knee), mass=m_leg * 0.45, role="link",
                            friction=(0.6, 0.005, 0.0001), rgba=LEG_RGBA, fluidshape="ellipsoid"),
                       Geom(f"{leg}_tibia", "capsule", (0.007,), fromto=(*knee, fx, fy, fz), mass=m_leg * 0.45, role="link",
                            friction=(0.6, 0.005, 0.0001), rgba=LEG_RGBA, fluidshape="ellipsoid"),
                       Geom(f"{leg}_foot", "sphere", (g["foot_r"],), pos=(fx, fy, fz), mass=m_leg * 0.10, role="foot",
                            friction=(foot_mu, 0.005, 0.0001), rgba=(0.08, 0.08, 0.08, 1.0))])


def _claw_link(side: str, g: dict, m_arm: float, m_jaw: float, m_jaw_servo: float, m_servo: float, jaw_mu: float, v=None) -> Link:
    v = variant_of(v)
    NOTCH_TOOTH = v.notch_tooth
    sgn = CLAWS[side]
    syj, shj, elj, wrj, juj, jlj = claw_joints(side)
    jaws = []
    d_mean = g["jaw_d"] * (1 + 0.45) / 2
    for name, axis, zs in ((juj, (0, -1, 0), 1), (jlj, (0, 1, 0), -1)):
        geoms = [Geom(name, "box", (g["Lj"] / 2, g["jaw_t"] / 2 * 1.5, d_mean / 2), pos=(g["Lj"] / 2, 0.0, zs * d_mean / 2),
                      mass=m_jaw, role="link", friction=(jaw_mu, 0.01, 0.0001), rgba=JAW_RGBA)]
        if g["hook"] > 0:
            geoms.append(Geom(name + "_hook", "box", (g["hook_w"] / 2, g["jaw_t"] / 2 * 1.5, g["hook"] / 2),
                              pos=(g["Lj"] - g["hook_w"] / 2, 0.0, -zs * g["hook"] / 2), role="link",
                              friction=(jaw_mu, 0.01, 0.0001), rgba=JAW_RGBA))
        # the cutter notch's outer flank: a tooth standing NOTCH_TOOTH proud of the inner face just outboard of the
        # notch — closing jaws push a round rope outwards (the scissors' push-out), the flank keeps it in the notch
        tx = g["cutter"] + NOTCH_TOOTH["gap"]
        geoms.append(Geom(name + "_tooth", "box", (NOTCH_TOOTH["width"] / 2, g["jaw_t"] / 2 * 1.5, NOTCH_TOOTH["height"] / 2),
                          pos=(tx + NOTCH_TOOTH["width"] / 2, 0.0, -zs * NOTCH_TOOTH["height"] / 2), role="link",
                          friction=(jaw_mu, 0.01, 0.0001), rgba=JAW_RGBA))
        jaws.append(Link(name + "_link", pos=(g["Lh"], 0.0, 0.0),
                         joints=[Joint(name, axis=axis, range=(math.radians(-12), math.radians(60)), tag="jaw",
                                       servo=_servo("jaw", v))], geoms=geoms))
    hand = Link(f"{side}_hand", pos=(g["Lp"], 0.0, 0.0),
                joints=[Joint(wrj, axis=(0, 1, 0), range=(math.radians(-90), math.radians(90)), tag="wrist", servo=_servo("arm", v))],
                geoms=[Geom(f"{side}_hand", "box", (g["Lh"] / 2, g["arm_w"] * 0.35, g["arm_w"] * 0.5), pos=(g["Lh"] / 2, 0, 0),
                            mass=v.parts_kg["palm housings 2x"] / 2, role="link", friction=(0.5, 0.005, 0.0001), rgba=LEG_RGBA,
                            fluidshape="ellipsoid")],
                masses=[PointMass(f"{side}_jaw_drive", m_jaw_servo, (g["Lh"] / 2, 0.0, 0.0))], children=jaws)
    palm = Link(f"{side}_palm", pos=(g["Lu"], 0.0, 0.0),
                joints=[Joint(elj, axis=(0, 1, 0), range=(math.radians(-20), math.radians(150)), tag="elbow", servo=_servo("arm", v))],
                geoms=[Geom(f"{side}_palm", "box", (g["Lp"] / 2, g["arm_w"] * 0.35, g["arm_w"] * 0.45), pos=(g["Lp"] / 2, 0, 0),
                            mass=m_arm * 0.4, role="link", friction=(0.5, 0.005, 0.0001), rgba=LEG_RGBA,
                            fluidshape="ellipsoid")],
                masses=[PointMass(f"{side}_wrist_servo", m_servo, (g["Lp"], 0.0, 0.0))], children=[hand])
    return Link(f"{side}_arm", pos=g["claw"] if sgn > 0 else (g["claw"][0], -g["claw"][1], g["claw"][2]),
                joints=[Joint(syj, axis=(0, 0, 1), range=(math.radians(-60), math.radians(60)), tag="yaw", servo=_servo("arm", v)),
                        Joint(shj, axis=(0, -1, 0), range=(math.radians(-40), math.radians(80)), tag="shoulder", servo=_servo("arm", v))],
                geoms=[Geom(f"{side}_upper_arm", "box", (g["Lu"] / 2, g["arm_w"] * 0.35, g["arm_w"] / 2), pos=(g["Lu"] / 2, 0, 0),
                            mass=m_arm * 0.6, role="link", friction=(0.5, 0.005, 0.0001), rgba=LEG_RGBA, fluidshape="ellipsoid")],
                masses=[PointMass(f"{side}_elbow_servo", m_servo, (g["Lu"], 0.0, 0.0))], children=[palm])


def _tail_link(g: dict, m_seg: float, m_servo: float, m_thruster: float, m_shroud: float, v=None) -> Link:
    q90 = (math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)          # cylinder axis z -> x
    L1, L2, r = g["L1"], g["L2"], g["tail_r"]
    thruster = Link("thruster", pos=(-L2 - g["shroud_L"] / 2, 0.0, 0.0),
                    geoms=[Geom("shroud", "cylinder", (g["shroud_r"], g["shroud_L"] / 2), quat=q90, mass=m_shroud, role="link",
                                friction=(0.5, 0.005, 0.0001), rgba=TAIL_RGBA, fluidshape="ellipsoid"),
                           Geom("prop_disc", "cylinder", (g["prop_D"] / 2, 0.004), quat=q90, role="visual", rgba=(0.6, 0.55, 0.4, 0.6))],
                    masses=[PointMass("thruster motor", m_thruster, (0.01, 0.0, 0.0))])
    seg2 = Link("tail2", pos=(-L1, 0.0, 0.0),
                joints=[Joint("tail2_yaw", axis=(0, 0, 1), range=(math.radians(-50), math.radians(50)), tag="tail", servo=_servo("tail", v)),
                        Joint("tail2_pitch", axis=(0, 1, 0), range=tuple(math.radians(v) for v in TAIL_PITCH_RANGE[2]), tag="tail", servo=_servo("tail", v))],
                geoms=[Geom("tail2", "capsule", (r * 0.9,), fromto=(0, 0, 0, -L2, 0, 0), mass=m_seg, role="link",
                            friction=(0.5, 0.005, 0.0001), rgba=TAIL_RGBA, fluidshape="ellipsoid")],
                children=[thruster])
    return Link("tail1", pos=g["tail0"],
                joints=[Joint("tail1_yaw", axis=(0, 0, 1), range=(math.radians(-50), math.radians(50)), tag="tail", servo=_servo("tail", v)),
                        Joint("tail1_pitch", axis=(0, 1, 0), range=tuple(math.radians(v) for v in TAIL_PITCH_RANGE[1]), tag="tail", servo=_servo("tail", v))],
                geoms=[Geom("tail1", "capsule", (r,), fromto=(0, 0, 0, -L1, 0, 0), mass=m_seg, role="link",
                            friction=(0.5, 0.005, 0.0001), rgba=TAIL_RGBA, fluidshape="ellipsoid")],
                masses=[PointMass("tail2 servos", 2 * m_servo, (-L1, 0.0, 0.0))], children=[seg2])


#: The cutter notch's outer flank in MuJoCo [m]: ``gap`` outboard of the notch centre, ``width`` along the jaw,
#: ``height`` proud of the inner face (the CAD's V notch, ``cutter_depth``, is cut into the inner edge instead).
NOTCH_TOOTH = {"gap": 0.004, "width": 0.004, "height": 0.004}
BATTERY_X = -0.08                         # m: the battery at the rear of the housing (trim: the claws are heavy forward)
BATTERY_Z = -0.035                        # m: low in the housing ...
FOAM_Z = 0.035                            # ... and the foam high in the shell's upper corners: the centre of buoyancy well
                                          # above the centre of gravity, the hydrostatic righting moment (notebook 24 §1)


# ----------------------------------------------------------------------------------------------- variants
#: Mission tuning per variant (the controller's gait and the scenario's missions read these; notebooks 24/25).
NEFRI_MISSION = {"stride_deg": 22.0, "lift_deg": 25.0, "period": 1.2, "v_nominal": 0.10,
                 "swim_rpm": 2600.0, "swim_depth": 0.7, "swim_k_z": 0.8, "walk_v": 0.10, "creep_v": 0.03,
                 "notch_x": 0.30, "throat": 0.02, "claw_drop": (-0.04, 0.0, -0.09), "jaw_open_deg": 40.0,
                 "duration": {"swim": 45.0, "cut_and_enter": 55.0, "current": 24.0},
                 "press_rpm": 2400.0, "press_deg": -60.0, "press_x": -0.08, "walk_rpm": 2400.0, "walk_deg": -45.0,
                 "walk_x": -0.065, "current_walk_v": 0.08}
ORNATUS_MISSION = {"stride_deg": 36.0, "lift_deg": 25.0, "period": 1.3, "v_nominal": 0.12,
                   "swim_rpm": 2000.0, "swim_depth": 0.8, "swim_k_z": 0.8, "walk_v": 0.12, "creep_v": 0.04,
                   "notch_x": 0.38, "throat": 0.015, "claw_drop": (-0.05, 0.0, -0.11), "jaw_open_deg": 40.0,
                   "duration": {"swim": 45.0, "cut_and_enter": 70.0, "current": 24.0},
                   "press_rpm": 1800.0, "press_deg": -60.0, "press_x": -0.10, "walk_rpm": 1800.0, "walk_deg": -45.0,
                   "walk_x": -0.08, "current_walk_v": 0.08}


@dataclass
class Variant:
    """One member of the Lobster family: its design parameters (``lobster.py``, mm), stored CAD numbers, parts list,
    servos and gains, thruster, trim positions and mission tuning. ``variant_of(x)`` resolves a name, a Variant or
    None (Nefri)."""

    name: str
    display: str
    params: dict
    cad: dict
    parts_kg: dict
    parts_volume: dict
    servo_keys: dict
    gains: dict
    thruster: dict
    battery_key: str
    battery_x: float = BATTERY_X
    battery_z: float = BATTERY_Z
    foam_z: float = FOAM_Z
    notch_tooth: dict = field(default_factory=lambda: dict(NOTCH_TOOTH))
    target_mass: float | None = None
    payload_x: float | None = None            # a "payload" part sits here (hull frame); None: with the rest of the hull mass
    mission: dict = field(default_factory=lambda: dict(NEFRI_MISSION))
    notebook: str = "24"


NEFRI_V = Variant("nefri", "Sikarian Lobster Nefri", NEFRI, CAD, PARTS_KG, PARTS_VOLUME, SERVO_KEYS, GAINS, THRUSTER,
                  "battery 4S3P Li-ion 21700, 207 Wh (in the housing)")

#: Ornatus (v2, the big member, notebook 25): 15 kg, cuts a Ø12 mm PVC power cable (3×2.5 mm² Cu, ~2.5 kN), the same
#: Ø600 pipe — so the legs are narrower than a pure scale-up (the foot span must stay inside the pipe's 0.52 m silt floor).
ORNATUS_PARAMS = dict(NEFRI, **{
    "shell_length": 410.0, "shell_width": 260.0, "shell_height": 170.0, "shell_thickness": 3.0, "shell_chamfer": 36.0,
    "shell_bottom": 90.0, "housing_diameter": 140.0, "housing_wall": 3.0, "housing_length": 280.0, "cap_thickness": 10.0,
    "hip_spacing": 120.0, "hip_height": 40.0, "coxa_length": 26.0, "femur_length": 45.0, "tibia_length": 135.0,
    "tibia_splay_deg": 4.0, "leg_width": 20.0, "leg_thickness": 10.0, "foot_diameter": 26.0, "claw_y": 70.0,
    "claw_height": 90.0, "upper_arm_length": 115.0, "palm_length": 58.0, "hand_length": 28.0, "arm_width": 28.0,
    "jaw_length": 90.0, "jaw_thickness": 8.0, "jaw_depth": 24.0, "jaw_pin_diameter": 8.0, "cutter_x": 18.0, "cutter_depth": 5.5,
    "hook_width": 10.0, "tail_height": 90.0, "tail_segment_1": 110.0, "tail_segment_2": 90.0, "tail_diameter": 64.0,
    "shroud_length": 76.0, "shroud_inner": 106.0, "shroud_wall": 6.0, "prop_diameter": 100.0, "prop_pitch": 92.0,
    "prop_hub": 26.0, "mast_height": 115.0, "mast_diameter": 9.0})
#: CAD of Ornatus (notebook 25 §1; ``cad_numbers(ORNATUS_PARAMS, recompute=True)`` rebuilds them).
ORNATUS_CAD = {'shell_volume': 1044272.8, 'shell_com_z': 177.8, 'housing_volume': 643586.7, 'leg_volume': 65881.5, 'upper_arm_volume': 98117.4, 'jaw_volume': 15679.5, 'tail_segment_volume': 394545.8, 'shroud_volume': 198342.2, 'propeller_volume': 13571.8}
ORNATUS_PARTS_KG = {
    "battery 4S6P Li-ion 21700, 414 Wh (in the housing)": 1.60, "computer, IMU, depth sensor (in the housing)": 0.15,
    "acoustic modem and radio relay (in the housing)": 0.10, "power electronics, thruster ESC (in the housing)": 0.15,
    "O-rings, penetrators, cable glands": 0.18, "cameras 3x low-light (sealed)": 0.09, "imaging sonar (sealed)": 0.40,
    "hydrophone array and antenna masts": 0.12, "wiring outside the housing": 0.12,
    "thruster motor, sealed BLDC": 0.35, "tail segments 2x (PA12 tube, foam-filled)": 0.30,
    "palm housings 2x": 0.12, "foot balls 6x (rubber)": 0.06,
    "payload bay (sample carousel / sensor head, in the shell)": 1.00}
ORNATUS_PARTS_VOLUME = {
    "cameras 3x low-light (sealed)": 45e-6, "imaging sonar (sealed)": 240e-6, "hydrophone array and antenna masts": 60e-6,
    "thruster motor, sealed BLDC": 110e-6, "tail segments 2x (PA12 tube, foam-filled)": 2 * 300e-6,
    "palm housings 2x": 2 * 50e-6, "foot balls 6x (rubber)": 6 * 9.2e-6, "wiring outside the housing": 70e-6,
    "payload bay (sample carousel / sensor head, in the shell)": 800e-6}
ORNATUS_SERVO_KEYS = {"leg": "sealed servo 3 Nm", "arm": "sealed servo 8 Nm", "jaw": "sealed jaw screw 4 kN", "tail": "sealed servo 8 Nm"}
ORNATUS_GAINS = {"leg": (20.0, 0.35, 0.005), "arm": (40.0, 0.6, 0.008), "jaw": (120.0, 1.5, 0.02), "tail": (30.0, 0.5, 0.008)}
ORNATUS_THRUSTER = {
    "motor": boreas.Motor("sealed BLDC 180 kV", kv_rpm_per_volt=180.0, resistance_ohm=0.08, no_load_current_a=0.6,
                          max_current_a=25.0, mass_kg=0.35, source="assumed: underwater thruster motor class, 4S, 300 W"),
    "battery": boreas.Battery("4S6P Li-ion 21700", cells=4, capacity_ah=28.8, cell_voltage_nominal=3.6, usable_fraction=0.85, mass_kg=1.60),
    "section": THRUSTER["section"], "rpm_max": 2800.0, "spool_s": 0.2}
ORNATUS = Variant("ornatus", "Sikarian Lobster Ornatus", ORNATUS_PARAMS, ORNATUS_CAD, ORNATUS_PARTS_KG, ORNATUS_PARTS_VOLUME,
                  ORNATUS_SERVO_KEYS, ORNATUS_GAINS, ORNATUS_THRUSTER, "battery 4S6P Li-ion 21700, 414 Wh (in the housing)",
                  battery_x=-0.11, battery_z=-0.045, foam_z=0.045, notch_tooth={"gap": 0.005, "width": 0.005, "height": 0.005},
                  target_mass=15.0, payload_x=-0.14, mission=dict(ORNATUS_MISSION), notebook="25")
VARIANTS = {"nefri": NEFRI_V, "ornatus": ORNATUS}


def variant_of(x=None) -> Variant:
    """A ``Variant`` from its name, itself, or None (Nefri)."""
    if x is None:
        return NEFRI_V
    if isinstance(x, Variant):
        return x
    try:
        return VARIANTS[str(x).lower()]
    except KeyError:
        raise KeyError(f"unknown Lobster variant {x!r}; known: {list(VARIANTS)}") from None



def lobster(overrides: dict | None = None, *, variant=None, cad: dict | None = None, foot_mu: float = 0.7, jaw_mu: float = 0.6,
            name: str | None = None, trim: bool = True) -> Robot:
    """The Lobster as a Chiron ``Robot`` (masses from ``budget``), **trimmed**: the foam's position along x is solved
    so the centre of buoyancy sits above the centre of gravity in the standing pose (``robot.trim``: the foam x and
    the couple before trimming). ``variant``: "nefri" (default) or "ornatus" (or a ``Variant``) — ``robot.variant``.
    ``robot.volumes``: {body: (m³, centre of buoyancy in the body frame)} for the Water hook; ``robot.lobster``: the
    geometry; ``robot.budget``."""
    v = variant_of(variant)
    name = name or v.display
    if not trim:
        return _lobster(v, overrides, cad, foot_mu, jaw_mu, name, 0.0)
    r0 = _lobster(v, overrides, cad, foot_mu, jaw_mu, name, 0.0)
    M0, up = _pitch_couple(r0)
    foam_x = -M0 / up
    robot = _lobster(v, overrides, cad, foot_mu, jaw_mu, name, foam_x)
    robot.trim = {"foam_x_m": foam_x, "couple_untrimmed_Nm": M0, "couple_trimmed_Nm": _pitch_couple(robot)[0]}
    return robot


def _pitch_couple(robot) -> tuple:
    """(Σ buoyancy·x − Σ weight·x about the hull origin [N·m] in the standing pose, the foam's net lift [N])."""
    import mujoco
    from vegeta.chiron import Flat, SimOptions

    m = mujoco.MjModel.from_xml_string(robot.to_mjcf(Flat(), SimOptions()))
    d = mujoco.MjData(m)
    for jn, v in robot.nominal_qpos.items():
        d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)]] = v
    d.qpos[3:7] = (1, 0, 0, 0)
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    rho_g = WATER["density"] * WATER["g"]
    MB = 0.0
    for name, (V, cb) in robot.volumes.items():
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        MB += rho_g * V * (d.xpos[b] + d.xmat[b].reshape(3, 3) @ np.asarray(cb))[0]
    root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hull")
    MW = float(m.body_subtreemass[root]) * WATER["g"] * d.subtree_com[root][0]
    foam_V = robot.budget["parts"]["syntactic foam trim"][1]
    up = (rho_g * foam_V - robot.budget["parts"]["syntactic foam trim"][0] * WATER["g"])
    return MB - MW, up


def _lobster(v, overrides, cad, foot_mu, jaw_mu, name, foam_x) -> Robot:
    p = design_params(overrides, v)
    cad = cad_numbers(p, variant=v) if cad is None else cad
    g = geometry(p)
    b = budget(p, cad, v)
    parts = b["parts"]
    PARTS_KG, PARTS_VOLUME, SERVO_KEYS = v.parts_kg, v.parts_volume, v.servo_keys
    BATTERY_X, BATTERY_Z, FOAM_Z = v.battery_x, v.battery_z, v.foam_z
    m_servo = act.get(SERVO_KEYS["leg"]).mass_g / 1000.0               # legs
    m_arm_servo = act.get(SERVO_KEYS["arm"]).mass_g / 1000.0           # arms
    m_tail_servo = act.get(SERVO_KEYS["tail"]).mass_g / 1000.0
    m_jaw_servo = act.get(SERVO_KEYS["jaw"]).mass_g / 1000.0
    m_leg = parts["legs 6x (PA12-CF, CAD)"][0] / 6 + PARTS_KG["foot balls 6x (rubber)"] / 6
    m_jaw = parts["jaws 4x (tool steel, CAD)"][0] / 4
    m_arm = parts["arms 2x (PA12-CF)"][0] / 2
    m_seg = PARTS_KG["tail segments 2x (PA12 tube, foam-filled)"] / 2
    m_shroud = parts["shroud + propeller (PA12, CAD)"][0]
    legs = [_leg_link(leg, g, m_leg, foot_mu, v) for leg in LEGS]
    claws = [_claw_link(s, g, m_arm, m_jaw, m_jaw_servo, m_arm_servo, jaw_mu, v) for s in CLAWS]
    tail = _tail_link(g, m_seg, m_tail_servo, PARTS_KG["thruster motor, sealed BLDC"], m_shroud, v)
    L, W, H = g["shell"]
    r_h, L_h = g["housing"]
    zc = g["zc"]
    # hull masses: everything not on a moving link
    on_links = (6 * m_leg + 2 * (m_arm + 2 * m_jaw + PARTS_KG["palm housings 2x"] / 2 + m_jaw_servo + 2 * m_arm_servo)
                + 2 * m_seg + 2 * m_tail_servo + PARTS_KG["thruster motor, sealed BLDC"] + m_shroud)
    m_hull = b["mass"] - on_links
    shell_m = parts["shell (ASA, CAD)"][0]
    housing_m = parts["pressure housing (Al 6082, CAD; displaces its envelope)"][0]
    foam_m = parts["syntactic foam trim"][0]
    ballast_m = b.get("ballast_kg", 0.0)
    inside = sum(m for k, m in PARTS_KG.items() if "(in the housing)" in k)
    hips = [PointMass(f"{leg}_hip_servos", 2 * m_servo, g["hips"][leg]) for leg in LEGS]
    claw_servos = [PointMass(f"{s}_shoulder_servos", 2 * m_arm_servo, (g["claw"][0] - 0.02, CLAWS[s] * g["claw"][1], g["claw"][2])) for s in CLAWS]
    tail_servos = PointMass("tail1 servos", 2 * m_tail_servo, (g["tail0"][0] + 0.02, 0.0, g["tail0"][2]))
    extra = [PointMass("ballast plates", ballast_m, (BATTERY_X, 0.0, -H / 2 + 0.01))] if ballast_m > 0 else []
    payload_m = sum(m for k, m in PARTS_KG.items() if "payload" in k) if v.payload_x is not None else 0.0
    if payload_m > 0:
        extra.append(PointMass("payload bay", payload_m, (v.payload_x, 0.0, -0.02)))
    rest = (m_hull - shell_m - housing_m - foam_m - ballast_m - payload_m - inside - sum(pm.mass for pm in hips + claw_servos)
            - tail_servos.mass)
    hull = Link("hull", log=True,
                geoms=[Geom("shell", "box", (L / 2, W / 2, H / 2), mass=shell_m, role="body", friction=(0.5, 0.005, 0.0001),
                            rgba=SHELL_RGBA, fluidshape="ellipsoid"),
                       Geom("nose", "box", (0.012, W * 0.32, H * 0.12), pos=(L / 2 + 0.006, 0.0, H * 0.05), role="visual",
                            rgba=(0.1, 0.1, 0.12, 1.0)),
                       Geom("mast_a", "cylinder", (0.0035, 0.045), pos=(-L * 0.15, W * 0.2, H / 2 + 0.045), role="visual", rgba=LEG_RGBA),
                       Geom("mast_b", "cylinder", (0.0035, 0.036), pos=(-L * 0.05, -W * 0.2, H / 2 + 0.036), role="visual", rgba=LEG_RGBA)],
                masses=[PointMass("pressure housing", housing_m, (0.0, 0.0, 0.0)),
                        PointMass("battery", PARTS_KG[v.battery_key], (BATTERY_X, 0.0, BATTERY_Z)),
                        PointMass("electronics", inside - PARTS_KG[v.battery_key], (0.03, 0.0, 0.0)),
                        PointMass("syntactic foam", foam_m, (foam_x, 0.0, FOAM_Z)),
                        PointMass("sensors, glands, wiring, payload", max(rest, 0.0), (L * 0.2, 0.0, 0.0)), tail_servos] + hips + claw_servos + extra,
                children=legs + claws + [tail])
    feet = [FootSpec(leg, f"{leg}_foot", leg_joints(leg), "hull") for leg in LEGS]
    robot = Robot(name, hull, feet=feet, nominal_qpos=nominal_qpos(p),
                  notes=f"{name}: 6 two-joint legs, 2 pincer arms, a 2-knuckle tail with a shrouded {p['prop_diameter']:.0f} mm "
                        f"thruster; {b['mass']:.2f} kg, sinks with {b['net_kg']:.2f} kg in fresh water",
                  sources={"geometry": "designs/lobster.py", "masses": "notebook 24 §1 budget", "actuators": "designs/actuators.py",
                           "assumptions": "lobster_robot.ASSUMPTIONS"})
    robot.validate()
    # buoyancy: volumes per body and their centres (body frame); the hull's is what the links do not carry
    v_seg = 2 * PARTS_VOLUME["tail segments 2x (PA12 tube, foam-filled)"] / 2 / 2
    vols = {}
    for leg in LEGS:
        _, side = LEGS[leg]
        vols[f"{leg}_leg"] = (cad["leg_volume"] * 1e-9 + PARTS_VOLUME["foot balls 6x (rubber)"] / 6,
                              (0.0, side * (g["femur"] * 0.7), g["foot"][2] * 0.4))
    for s in CLAWS:
        vols[f"{s}_arm"] = (parts["arms 2x (PA12-CF)"][1] / 2 * 0.6 + m_arm_servo / SERVO_DENSITY, (g["Lu"] * 0.6, 0.0, 0.0))
        vols[f"{s}_palm"] = (parts["arms 2x (PA12-CF)"][1] / 2 * 0.4 + m_arm_servo / SERVO_DENSITY, (g["Lp"] * 0.6, 0.0, 0.0))
        vols[f"{s}_hand"] = (PARTS_VOLUME["palm housings 2x"] / 2 + m_jaw_servo / SERVO_DENSITY, (g["Lh"] / 2, 0.0, 0.0))
        for j in claw_joints(s)[4:]:
            vols[j + "_link"] = (m_jaw / MATERIALS["tool steel jaws"], (g["Lj"] / 2, 0.0, 0.0))
    vols["tail1"] = (v_seg + 2 * m_tail_servo / SERVO_DENSITY, (-g["L1"] / 2, 0.0, 0.0))
    vols["tail2"] = (v_seg, (-g["L2"] / 2, 0.0, 0.0))
    vols["thruster"] = (parts["shroud + propeller (PA12, CAD)"][1] + PARTS_VOLUME["thruster motor, sealed BLDC"], (0.0, 0.0, 0.0))
    v_hull = b["volume"] - sum(V for V, _ in vols.values())
    cb_hull = (parts["syntactic foam trim"][1] * FOAM_Z + parts["shell (ASA, CAD)"][1] * (cad["shell_com_z"] / 1000.0 - zc)) / v_hull
    cbx_hull = parts["syntactic foam trim"][1] * foam_x / v_hull
    vols["hull"] = (v_hull, (cbx_hull, 0.0, cb_hull))
    robot.volumes = vols
    robot.variant = v
    robot.foam_x = foam_x
    robot.params = p
    robot.lobster = g
    robot.budget = b
    return robot


class Water:
    """Scene hook: buoyancy on every body (ρ g V up at its centre of buoyancy) and the thruster: thrust along the
    ``thruster`` body's +x, from the BEMT table at the commanded rpm (``lab.thruster_rpm``, first-order spool-up) and
    the axial inflow, and the propeller's reaction torque. ``lab.thrust`` keeps the last thrust [N]."""

    def __init__(self, robot: Robot, table: dict | None = None, rho: float = WATER["density"], g: float = WATER["g"]):
        self.volumes = dict(robot.volumes)
        self.variant = getattr(robot, "variant", NEFRI_V)
        self.table = thruster_table(robot.params, variant=self.variant) if table is None else table
        self.rho, self.g = rho, g

    def reset(self, lab):
        m = lab.model
        self.bodies = [(name, lab._body_id(name), V, np.asarray(cb, dtype=float)) for name, (V, cb) in self.volumes.items()]
        self.ipos = {b: m.body_ipos[b].copy() for _, b, _, _ in self.bodies}
        self.thr = lab._body_id("thruster")
        self.rpm = 0.0
        lab.thruster_rpm = 0.0
        lab.thrust = 0.0

    def __call__(self, lab):
        d = lab.data
        up = np.array([0.0, 0.0, self.rho * self.g])
        forces = {}
        for name, b, V, cb in self.bodies:
            R = d.xmat[b].reshape(3, 3)
            F = up * V
            r = R @ (cb - self.ipos[b])
            forces[name] = [F, np.cross(r, F)]
        # thruster
        thr = self.variant.thruster
        target = float(np.clip(getattr(lab, "thruster_rpm", 0.0), 0.0, thr["rpm_max"]))
        self.rpm += (target - self.rpm) * min(1.0, lab.control_dt / thr["spool_s"])
        R = d.xmat[self.thr].reshape(3, 3)
        axis = R[:, 0]
        v = np.zeros(6)
        import mujoco
        mujoco.mj_objectVelocity(lab.model, d, mujoco.mjtObj.mjOBJ_BODY, self.thr, v, 0)
        inflow = max(0.0, float(np.dot(v[3:], axis)))               # moving along the thrust: the water comes in
        T, Q = thrust(self.table, self.rpm, inflow)
        lab.thrust = T
        F = axis * T
        forces["thruster"][0] = forces["thruster"][0] + F
        forces["thruster"][1] = forces["thruster"][1] - axis * Q
        for name, (F, tq) in forces.items():
            lab.body_force(name, F, tq)


def lobster_lab(terrain=None, robot: Robot | None = None, variant=None, **kwargs) -> ChironLab:
    """``ChironLab(lobster(variant=...), terrain, **LAB_OPTIONS)`` with the ``Water`` hook (``lab.water``)."""
    robot = robot or lobster(variant=variant)
    opts = dict(LAB_OPTIONS)
    opts.update(kwargs)
    lab = ChironLab(robot, terrain, **opts)
    lab.water = Water(robot)
    lab.add_hook(lab.water)
    return lab
