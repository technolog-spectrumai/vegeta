"""Onager Manus (notebook 21) as a Chiron robot: the Sentinel's chassis (``onager_robot.onager``) without the
sensor turret, plus two manipulator arms on the front of the roof, each with a pincer that cuts wire and grips.

Per arm (``L`` left, ``R`` right; the right arm mirrors the left in y), all hinges actuated by catalogue modules
(``actuators.py``):

=================  =====================  ============================================================================
joint              actuator               convention
=================  =====================  ============================================================================
``<s>_yaw``        harmonic 150 Nm        about the pedestal's vertical axis, + turns the arm to the left
``<s>_shoulder``   cycloidal 400 Nm       about y; the upper arm's elevation above horizontal is ``−q``
``<s>_elbow``      harmonic 150 Nm        about y, relative; the forearm's elevation is ``−(q_sh + q_el)``
``<s>_wrist``      harmonic 60 Nm         about y, relative; the pincer's elevation is ``−(q_sh + q_el + q_wr)``
``<s>_jaw_upper``  jaw screw 6 kN         about the jaw pin; + opens the upper jaw (closed: its inner face on the
``<s>_jaw_lower``  (rotary equivalent)    pincer axis); the lower jaw mirrors it (+ opens downwards)
=================  =====================  ============================================================================

The two jaws are one ball-screw drive in the machine; here each is a hinge with the drive's full torque, so the
squeeze on an object between them is ``τ / x`` (x the contact's distance from the pin): 9 kN in the cutter notch
40 mm out, 1.6 kN at the tips. The jaws do not collide with each other (Chiron's robots have no self-collision):
closing past 0 only squeezes what is between them. ``arm_ik`` turns a pincer pose (its pin position in the hull
frame and its elevation) into the four arm angles.

    from . import onager_manus_robot as omr
    robot = omr.manus()                      # 458 kg
    lab = omr.manus_lab(terrain, props=..., welds=...)

Promoted from ``notebooks/designs/onager_manus_robot.py`` as it was proven there; the notebook copy may move on.
"""
from __future__ import annotations

import math

from vegeta.chiron import ChironLab, Geom, Joint, Link, PointMass, Robot, Servo

from . import actuators as act
from . import onager_robot as orb

__all__ = ["MANUS", "CAD", "PARTS_KG", "ARM_ACTUATORS", "ARM_GAINS", "JAW_ARMATURE", "ASSUMPTIONS", "SIDES",
           "LAB_OPTIONS", "design_params", "cad_numbers", "arm_geometry", "arm_joints", "jaw_geoms", "jaw_parts", "arm_masses",
           "mass_budget", "arm_ik", "arm_fk", "manus", "manus_lab", "nominal_qpos", "STOW"]

#: Manus parameters beyond the Sentinel's (onager_manus.py defaults, mm and degrees).
MANUS = {
    "arm_x": 800.0, "arm_y": 280.0, "pedestal_height": 120.0, "pedestal_diameter": 180.0,
    "upper_arm_length": 600.0, "upper_arm_width": 60.0, "upper_arm_depth": 80.0,
    "forearm_length": 550.0, "forearm_width": 50.0, "forearm_depth": 70.0, "arm_wall": 4.0,
    "arm_boss_diameter": 110.0, "arm_pin_diameter": 30.0, "palm_length": 100.0, "jaw_length": 220.0,
    "jaw_thickness": 14.0, "jaw_depth": 45.0, "jaw_pin_diameter": 16.0, "notch_x": 40.0, "notch_depth": 6.0,
    "hook_length": 40.0, "hook_width": 25.0,
    "arm_shoulder_deg": 50.0, "arm_elbow_deg": 40.0, "jaw_open_deg": 20.0,
}
CHASSIS = {"turret_length": 0.0, "turret_height": 0.0}      # the Sentinel parameters the Manus changes

#: CAD measurements of the default Manus (notebook 21 §1; mm², mm³). ``cad_numbers(recompute=True)`` rebuilds them.
CAD = {
    "hull_surface_area": 6431301.0, "upper_arm_volume": 1624214.0, "forearm_volume": 1058820.0, "jaw_volume": 127929.0,
}
MATERIALS = {"Al 6082-T6 arm tubes": 2.70e-6, "tool steel jaws": 7.85e-6}

#: The listed parts of the Manus (kg): the Sentinel's, with lighter sensors (no turret: mast head, lidar, two wrist
#: cameras) and the arms' own small parts.
PARTS_KG = dict(orb.PARTS_KG)
PARTS_KG.pop("sensors: E/O-IR head, lidar, acoustic, mast")
PARTS_KG["sensors: mast head, lidar, wrist cameras"] = 14.0
ARM_PARTS_KG = {"palm (wrist housing, screw mount)": 1.5, "pedestal": 2.0, "arm cabling and covers": 1.0}

ARM_ACTUATORS = {"yaw": "harmonic 150 Nm, brake", "shoulder": "cycloidal 400 Nm, brake",
                 "elbow": "harmonic 150 Nm, brake", "wrist": "harmonic 60 Nm, brake", "jaw": "jaw screw 6 kN"}
#: Position-loop gains (kp N·m/rad, kd N·m·s/rad) and reflected inertia (kg·m²) per joint (ASSUMPTIONS).
ARM_GAINS = {"yaw": (1500.0, 120.0, 0.3), "shoulder": (3000.0, 250.0, 0.6), "elbow": (1500.0, 100.0, 0.3),
             "wrist": (600.0, 25.0, 0.1), "jaw": (600.0, 40.0, 0.57)}
JAW_ARMATURE = 0.57
SIDES = {"L": 1, "R": -1}

ASSUMPTIONS = {
    "arm_gains": "kp so that the loaded arm sags < 2° under the log without feed-forward; kd for ζ ≈ 0.5–0.7 with "
                 "the arm's inertia about each joint (shoulder ≈ 15 kg·m² outstretched); the controller adds the "
                 "MuJoCo gravity torque (qfrc_bias) as feed-forward",
    "armature": "reflected rotor inertia: rotor ≈ 0.4–1 kg·cm² through the gear ratio² (1:100–1:120 strain wave, 1:60 "
                "cycloid); the jaw: a 1 kg·cm² rotor through the screw (5 mm lead) on the 60 mm lever = 0.57 kg·m²",
    "jaw_model": "two hinges with the drive's full torque each (the machine has one screw and a linkage); the inner "
                 "faces are boxes, the V notch is not modelled: the wire's squeeze is τ / x at its distance x from "
                 "the pin; the hooked tips are boxes (their mass is in the jaws' CAD mass); the jaws do not collide "
                 "with each other",
    "arm_links": "rectangular Al tubes as boxes of their outer size with the CAD mass at their middle; the joint "
                 "modules as point masses at their joints; jaws as boxes of the mean finger depth",
    "sensors": "no turret: 14 kg of sensors (mast head 25 %, mast 15 %, lidar and wrist cameras 60 % on the roof)",
}

LAB_OPTIONS = dict(orb.LAB_OPTIONS, timestep=0.0005, control_dt=0.002, log_dt=0.02)


def design_params(overrides: dict | None = None) -> dict:
    """The Sentinel's parameters with the Manus changes, plus the arm parameters (``MANUS``) [mm, deg]."""
    over = dict(overrides or {})
    arm = {k: over.pop(k) for k in list(over) if k in MANUS}
    p = orb.design_params({**CHASSIS, **over})
    q = dict(MANUS)
    q.update({k: float(v) for k, v in arm.items()})
    return {**p, **q}


def cad_numbers(p: dict | None = None, *, recompute: bool = False) -> dict:
    """Hull area and arm-part volumes (mm², mm³); the stored ``CAD`` for the default design, else rebuilt with
    ``onager_manus.OnagerManus`` (CadQuery)."""
    p = design_params() if p is None else dict(p)
    sentinel = orb.cad_numbers(orb.design_params({k: p[k] for k in orb.DESIGN}), recompute=recompute)
    if not recompute and p == design_params():
        return {**sentinel, **CAD}
    from . import onager_manus

    d = onager_manus.OnagerManus()
    over = {k: v for k, v in p.items() if k in {prm.name for prm in d.parameters}}
    hull = d.generate(part="hull", **over).measure()
    return {**sentinel, "hull_surface_area": hull["surface_area"],
            "upper_arm_volume": d.generate(part="upper_arm", **over).measure()["volume"],
            "forearm_volume": d.generate(part="forearm", **over).measure()["volume"],
            "jaw_volume": d.generate(part="jaw", **over).measure()["volume"]}


def arm_geometry(p: dict | None = None) -> dict:
    """Arm lengths [m] and the left shoulder point in the hull frame (origin at the hull centre)."""
    p = design_params() if p is None else p
    mm = 1e-3
    return {"x": p["arm_x"] * mm, "y": p["arm_y"] * mm, "z": (p["hull_height"] / 2 + p["pedestal_height"]) * mm,
            "Lu": p["upper_arm_length"] * mm, "Lf": p["forearm_length"] * mm, "Lp": p["palm_length"] * mm,
            "Lj": p["jaw_length"] * mm, "notch": p["notch_x"] * mm, "jaw_t": p["jaw_thickness"] * mm,
            "hook": p["hook_length"] * mm, "hook_w": p["hook_width"] * mm,
            "jaw_d": p["jaw_depth"] * mm, "ped_h": p["pedestal_height"] * mm, "ped_r": p["pedestal_diameter"] / 2 * mm}


def arm_joints(side: str) -> list:
    """``[yaw, shoulder, elbow, wrist, jaw_upper, jaw_lower]`` joint names of arm ``side`` ('L' or 'R')."""
    return [f"{side}_{j}" for j in ("yaw", "shoulder", "elbow", "wrist", "jaw_upper", "jaw_lower")]


def jaw_geoms(side: str) -> list:
    """The two fingers (their inner faces cut and squeeze)."""
    return [f"{side}_jaw_upper", f"{side}_jaw_lower"]


def jaw_parts(side: str) -> list:
    """Every jaw geom: the fingers and their hooked tips (what grips)."""
    return jaw_geoms(side) + [f"{side}_jaw_upper_hook", f"{side}_jaw_lower_hook"]


def arm_masses(p: dict | None = None, cad: dict | None = None) -> dict:
    """One arm's parts [kg] (CAD volumes × densities, catalogue modules, listed parts)."""
    p = design_params() if p is None else p
    cad = cad_numbers(p) if cad is None else cad
    out = {"upper arm (Al 6082, CAD)": cad["upper_arm_volume"] * MATERIALS["Al 6082-T6 arm tubes"],
           "forearm (Al 6082, CAD)": cad["forearm_volume"] * MATERIALS["Al 6082-T6 arm tubes"],
           "jaws 2x (tool steel, CAD)": 2 * cad["jaw_volume"] * MATERIALS["tool steel jaws"]}
    for j, key in ARM_ACTUATORS.items():
        out[f"{j} drive ({key})"] = act.get(key).mass_g / 1000.0
    out.update(ARM_PARTS_KG)
    return out


def mass_budget(p: dict | None = None, cad: dict | None = None) -> dict:
    """The whole Manus [kg]: the chassis (``onager_robot.mass_budget`` with the Manus parts) and two arms."""
    p = design_params() if p is None else p
    cad = cad_numbers(p) if cad is None else cad
    chassis = orb.mass_budget({k: p[k] for k in orb.DESIGN}, cad, PARTS_KG)
    arm = arm_masses(p, cad)
    chassis["arms 2x (see arm_masses)"] = 2 * sum(arm.values())
    return chassis


# ----------------------------------------------------------------------------------------------- kinematics
def arm_ik(target, elevation: float, side: str = "L", p: dict | None = None):
    """Arm angles (yaw, shoulder, elbow, wrist) [rad] that put the jaw pin at ``target`` (x, y, z) [m, hull frame]
    with the pincer pointing at ``elevation`` [rad] above horizontal (−π/2 = straight down); elbow up. None when out
    of reach."""
    g = arm_geometry(p)
    sx, sy, sz = g["x"], SIDES[side] * g["y"], g["z"]
    dx, dy, dz = target[0] - sx, target[1] - sy, target[2] - sz
    yaw = math.atan2(dy, dx)
    r = math.hypot(dx, dy)
    wr, wz = r - g["Lp"] * math.cos(elevation), dz - g["Lp"] * math.sin(elevation)
    d = math.hypot(wr, wz)
    Lu, Lf = g["Lu"], g["Lf"]
    if d > Lu + Lf - 1e-9 or d < abs(Lu - Lf) + 1e-9:
        return None
    beta = math.acos(max(-1.0, min(1.0, (Lu * Lu + d * d - Lf * Lf) / (2 * Lu * d))))
    phi1 = math.atan2(wz, wr) + beta                                   # upper arm elevation (elbow up)
    ex, ez = Lu * math.cos(phi1), Lu * math.sin(phi1)
    phi2 = math.atan2(wz - ez, wr - ex)                                # forearm elevation
    return yaw, -phi1, phi1 - phi2, phi2 - elevation


def arm_fk(q, side: str = "L", p: dict | None = None) -> tuple:
    """Jaw pin position (x, y, z) [m, hull frame] and pincer elevation [rad] for arm angles (yaw, sh, el, wr)."""
    g = arm_geometry(p)
    yaw, sh, el, wr = q[:4]
    phi1, phi2, phi3 = -sh, -(sh + el), -(sh + el + wr)
    r = g["Lu"] * math.cos(phi1) + g["Lf"] * math.cos(phi2) + g["Lp"] * math.cos(phi3)
    z = g["Lu"] * math.sin(phi1) + g["Lf"] * math.sin(phi2) + g["Lp"] * math.sin(phi3)
    return (g["x"] + r * math.cos(yaw), SIDES[side] * g["y"] + r * math.sin(yaw), g["z"] + z), phi3


#: Stowed arms: folded back over the roof, pincers pointing forward-down, jaws closed.
STOW = {"yaw": 0.0, "shoulder": math.radians(-70.0), "elbow": math.radians(150.0), "wrist": math.radians(-60.0),
        "jaw": 0.0}


def nominal_qpos(p: dict | None = None) -> dict:
    """Sentinel standing pose for the legs and wheels, arms stowed."""
    q = orb.nominal_qpos({k: v for k, v in (p or design_params()).items() if k in orb.DESIGN})
    for side in SIDES:
        y, s, e, w, ju, jl = arm_joints(side)
        q.update({y: STOW["yaw"], s: STOW["shoulder"], e: STOW["elbow"], w: STOW["wrist"], ju: STOW["jaw"],
                  jl: STOW["jaw"]})
    return q


# ----------------------------------------------------------------------------------------------- the robot
def _servo(joint: str) -> Servo:
    kp, kd, arm = ARM_GAINS[joint]
    return Servo.from_actuator(act.get(ARM_ACTUATORS[joint]), kp=kp, kd=kd, armature=arm)


def _arm(side: str, p: dict, cad: dict, jaw_mu: float, yaw_range_deg: tuple = (-100.0, 100.0)) -> tuple:
    """One arm as a Link tree under the hull, and (its mass, Σ m x in the stowed pose) for the battery placement.
    ``yaw_range_deg``: the pedestal's travel (the Manus ±100°; None = unlimited: the Sweeper's slip-ring pedestals
    swing the arms back to the basket)."""
    g = arm_geometry(p)
    sgn = SIDES[side]
    m = arm_masses(p, cad)
    yaw_j, sh_j, el_j, wr_j, ju_j, jl_j = arm_joints(side)
    mm = 1e-3
    arm_rgba, jaw_rgba = (0.30, 0.31, 0.27, 1.0), (0.55, 0.56, 0.58, 1.0)
    fr = (0.5, 0.005, 0.0001)
    jaw_fr = (jaw_mu, 0.01, 0.0001)
    # jaws: boxes along +x from the pin, inner face on the pincer axis (z = 0), depth the mean of the taper
    d_mean = g["jaw_d"] * (1 + 0.45) / 2
    m_jaw = m["jaws 2x (tool steel, CAD)"] / 2
    jaws = []
    for name, axis, zsign in ((ju_j, (0, -1, 0), 1), (jl_j, (0, 1, 0), -1)):
        jaws.append(Link(name + "_link", pos=(g["Lp"], 0.0, 0.0),
                         joints=[Joint(name, axis=axis, range=(math.radians(-12), math.radians(60)), tag="jaw",
                                       servo=_servo("jaw"))],
                         geoms=[Geom(name, "box", (g["Lj"] / 2, g["jaw_t"] / 2 * 1.5, d_mean / 2),
                                     pos=(g["Lj"] / 2, 0.0, zsign * d_mean / 2), mass=m_jaw, role="link",
                                     friction=jaw_fr, rgba=jaw_rgba)]
                         + ([Geom(name + "_hook", "box", (g["hook_w"] / 2, g["jaw_t"] / 2 * 1.5, g["hook"] / 2),
                                  pos=(g["Lj"] - g["hook_w"] / 2, 0.0, -zsign * g["hook"] / 2), role="link",
                                  friction=jaw_fr, rgba=jaw_rgba)] if g["hook"] > 0 else [])))
    palm = Link(f"{side}_palm", pos=(g["Lf"], 0.0, 0.0),
                joints=[Joint(wr_j, axis=(0, 1, 0), range=(math.radians(-115), math.radians(115)), tag="wrist",
                              servo=_servo("wrist"))],
                geoms=[Geom(f"{side}_palm", "box", (g["Lp"] / 2, g["jaw_t"] * 1.6, g["jaw_d"] * 0.8),
                            pos=(g["Lp"] / 2, 0.0, 0.0), mass=ARM_PARTS_KG["palm (wrist housing, screw mount)"],
                            role="link", friction=fr, rgba=arm_rgba)],
                masses=[PointMass(f"{side}_jaw_drive", m["jaw drive (jaw screw 6 kN)"], (g["Lp"] / 2, 0.0, 0.0))],
                children=jaws)
    fore = Link(f"{side}_forearm", pos=(g["Lu"], 0.0, 0.0),
                joints=[Joint(el_j, axis=(0, 1, 0), range=(math.radians(-10), math.radians(165)), tag="elbow",
                              servo=_servo("elbow"))],
                geoms=[Geom(f"{side}_forearm", "box", (g["Lf"] / 2, p["forearm_width"] / 2 * mm, p["forearm_depth"] / 2 * mm),
                            pos=(g["Lf"] / 2, 0.0, 0.0), mass=m["forearm (Al 6082, CAD)"], role="link", friction=fr,
                            rgba=arm_rgba)],
                masses=[PointMass(f"{side}_wrist_drive", m["wrist drive (harmonic 60 Nm, brake)"], (g["Lf"], 0.0, 0.0))],
                children=[palm])
    upper = Link(f"{side}_upper_arm",
                 joints=[Joint(sh_j, axis=(0, 1, 0), range=(math.radians(-110), math.radians(60)), tag="shoulder",
                               servo=_servo("shoulder"))],
                 geoms=[Geom(f"{side}_upper_arm", "box", (g["Lu"] / 2, p["upper_arm_width"] / 2 * mm, p["upper_arm_depth"] / 2 * mm),
                             pos=(g["Lu"] / 2, 0.0, 0.0), mass=m["upper arm (Al 6082, CAD)"], role="link", friction=fr,
                             rgba=arm_rgba)],
                 masses=[PointMass(f"{side}_elbow_drive", m["elbow drive (harmonic 150 Nm, brake)"], (g["Lu"], 0.0, 0.0)),
                         PointMass(f"{side}_cabling", m["arm cabling and covers"], (g["Lu"] / 2, 0.0, 0.0))],
                 children=[fore])
    base = Link(f"{side}_arm_base", pos=(g["x"], sgn * g["y"], g["z"]),
                joints=[Joint(yaw_j, axis=(0, 0, 1), range=None if yaw_range_deg is None else tuple(math.radians(v) for v in yaw_range_deg), tag="yaw",
                              servo=_servo("yaw"))],
                geoms=[Geom(f"{side}_pedestal", "cylinder", (g["ped_r"], g["ped_h"] / 2), pos=(0.0, 0.0, -g["ped_h"] / 2),
                            mass=m["pedestal"], role="visual", rgba=arm_rgba)],
                masses=[PointMass(f"{side}_shoulder_drive", m["shoulder drive (cycloidal 400 Nm, brake)"], (0.0, 0.0, 0.0))],
                children=[upper])
    # the yaw module sits in the hull under the pedestal
    yaw_mass = PointMass(f"{side}_yaw_drive", m["yaw drive (harmonic 150 Nm, brake)"], (g["x"], sgn * g["y"], g["z"] - g["ped_h"]))
    # first moment along x in the stowed pose (the battery placement balances the standing robot with arms stowed)
    (px, _, _), _ = arm_fk((0.0, STOW["shoulder"], STOW["elbow"], STOW["wrist"]), side, p)
    total = sum(m.values())
    moment = total * (g["x"] + px) / 2                                 # arm mass ~ midway between shoulder and pincer
    return base, yaw_mass, total, moment


def manus(overrides: dict | None = None, *, jaw_mu: float = 0.6, cad: dict | None = None,
          name: str = "Onager Manus", **chassis_kw) -> Robot:
    """Onager Manus as a Chiron ``Robot``: ``onager_robot.onager`` with the Manus parts and two arms.

    ``overrides``: Sentinel or arm parameters [mm, deg]; ``jaw_mu``: friction of the serrated jaws (on wood ~0.6);
    ``chassis_kw``: the chassis options of ``onager_robot.onager`` (gains, friction, ranges). Feet: the four wheels.
    """
    p = design_params(overrides)
    cad = cad_numbers(p) if cad is None else cad
    arms, hull_masses, m_arms, moment = [], [], 0.0, 0.0
    for side in SIDES:
        base, yaw_mass, m_arm, mom = _arm(side, p, cad, jaw_mu)
        arms.append(base)
        hull_masses.append(yaw_mass)
        m_arms += m_arm
        moment += mom
    robot = orb.onager({k: p[k] for k in orb.DESIGN if p[k] != orb.DESIGN[k]}, cad=cad, name=name, parts_kg=PARTS_KG,
                       extra_children=arms, extra_mass_kg=m_arms, extra_moment_kgm=moment,
                       extra_hull_masses=hull_masses,
                       notes=f"Onager Manus: the Sentinel chassis without the turret and two 4-joint arms with "
                             f"cutting pincers ({', '.join(f'{k}: {v}' for k, v in ARM_ACTUATORS.items())})",
                       sources={"geometry": "designs/onager_manus.py", "masses": "notebook 21 §1 mass budget",
                                "actuators": "designs/actuators.py", "assumptions": "onager_manus_robot.ASSUMPTIONS"},
                       **chassis_kw)
    robot.nominal_qpos = nominal_qpos(p)
    robot.validate()
    robot.params = p
    robot.arm = arm_geometry(p)
    return robot


def manus_lab(terrain=None, robot: Robot | None = None, **kwargs) -> ChironLab:
    """``ChironLab(manus(), terrain, **LAB_OPTIONS)``; ``kwargs`` override (``props``, ``welds``, ``log_geoms``...)."""
    opts = dict(LAB_OPTIONS)
    opts.update(kwargs)
    return ChironLab(robot or manus(), terrain, **opts)
