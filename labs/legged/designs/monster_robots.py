"""The monsters (``monsters.py``) as Chiron robots, for export: the same numbers as the CAD (``monster_specs``),
shapes simplified to boxes, capsules and spheres. Design stage: no controllers, no propulsion data yet (the
wyvern's rotors are passive spin joints; Boreas maps come when it goes to Piccolo).

    import monster_robots as mr
    robot = mr.wyvern()          # also wolf, centaur, leech, leshy, wraith, ghoul
"""
from __future__ import annotations

import math

from vegeta.chiron import Geom, Joint, Link, PointMass

import body_kit as bk
import humanoid_robot as hr
import monster_specs as ms
import zoo_robots as zr
import zoo_specs as zs

__all__ = ["wolf", "wyvern", "centaur", "leech", "leshy", "wraith", "ghoul"]

mm = 1e-3
X, Y, Z = (1, 0, 0), (0, 1, 0), (0, 0, 1)


def wolf():
    return zr.quadruped(ms.WOLF, "Fenrir", "qdd 60 Nm", payload_kg=6.0,
                        notes="Fenrir: four-legged wolf, 16 actuated joints (design stage).")


def leshy():
    return zr.quadruped(ms.LESHY, "Leshy", "qdd 60 Nm", payload_kg=6.0,
                        notes="Leshy: deer-like forest spirit with huge antlers, 16 actuated joints (design stage).")


def leech():
    return zr.snake(ms.LEECH, "Hirudo", "smart servo 25 Nm",
                    notes="Hirudo: giant leech, 10 flattened two-axis segments (design stage).")


def ghoul():
    """The ghoul: Talia at the ghoul's numbers with a pincer on each hand (``fantasy_humanoid_robots.add_pincers``)."""
    import fantasy_humanoid_robots as fhr

    keys = set(hr.P) & set(ms.GHOUL)
    r = hr.talia("Ghoul", {k: ms.GHOUL[k] for k in keys},
                 notes="Ghoul: thin, long-armed humanoid variant of Talia with pincers, 21 actuated joints "
                       "(design stage).")
    return fhr.add_pincers(r, ms.GHOUL["forearm_length"])


def _centaur_torso(p, bc):
    key = "harmonic 60 Nm, brake"
    sv, m = bk.servo(key), zr._kg(key)
    waist, top, shoulders = ms.centaur_torso(p)
    w = tuple(c * mm for c in waist)
    Ht, Wt, Dt = p["torso_height"] * mm, p["torso_width"] * mm, p["torso_depth"] * mm
    arms = []
    for side, s in zip(("L", "R"), shoulders):
        el = (s[0] + 60, s[1], s[2] - p["upper_arm_length"])
        wr = (el[0] + p["forearm_length"] * 0.85, el[1], el[2] - p["forearm_length"] * 0.5)
        r = p["arm_radius"] * mm
        arm = bk.chain_through(f"{side}_arm", bk.scaled([s, el, wr]), [[Y, X], Y], [r, r * 0.85], origin=w,
                               servo=sv, actuator_kg=m, names=["shoulder", "elbow"])
        arm.children[0].geoms.append(bk.box(f"{side}_gripper", (0.06, 0.015, 0.045),
                                            tuple((wr[k] - el[k]) * mm for k in range(3)), role="link", rgba=bk.DARK))
        arms.append(arm)
    hz = (top + 40 + p["head_height"] / 2) * mm - w[2]
    torso = Link("torso", pos=tuple(w[k] - bc[k] for k in range(3)),
                 joints=[Joint("waist_yaw", axis=Z, range=(-1.2, 1.2), tag="waist_yaw", servo=sv),
                         Joint("waist_pitch", axis=Y, range=(-0.5, 0.8), tag="waist_pitch", servo=sv)],
                 geoms=[Geom("chest", "box", (Dt / 2 * 0.8, Wt / 2 * 0.8, Ht / 2), pos=(0, 0, Ht / 2),
                             mass=zr._egg_mass(Ht, Wt, Dt), role="body", rgba=bk.GREY),
                        bk.box("head", (p["head_length"] * mm / 2 * 0.8, p["head_width"] * mm / 2 * 0.8,
                                        p["head_height"] * mm / 2 * 0.8), (0.02, 0, hz))],
                 masses=[PointMass("waist_actuators", 2 * m), PointMass("computer_cameras", 1.5, (0, 0, hz))],
                 children=arms)
    return [torso]


def centaur():
    return zr.quadruped(ms.CENTAUR, "Chiron-C", "qdd 60 Nm", _centaur_torso, payload_kg=12.0, with_head=False,
                        notes="Chiron-C: centaur, four legs and a humanoid torso with two arms, 24 actuated joints "
                              "(design stage).")


def wyvern(p=None):
    """Fuselage root; six rotors (passive spin joints), two talon manipulators (two-axis shoulder, elbow, wrist)."""
    p = p or ms.WYVERN
    zc = p["cruise_height"] * mm
    origin = (0.0, 0.0, zc)
    L, W, H = p["body_length"] * mm, p["body_width"] * mm, p["body_height"] * mm
    children = []
    for k, c in enumerate(ms.rotor_centres(p)):
        c = tuple(v * mm for v in c)
        rel = (c[0], c[1], c[2] - zc)
        children.append(Link(f"rotor{k}", pos=(rel[0], rel[1], rel[2] + 0.053),
                             joints=[Joint(f"rotor{k}_spin", axis=Z, tag="rotor")],
                             geoms=[Geom(f"rotor{k}", "cylinder", (p["rotor_radius"] * mm, 0.003), mass=0.03,
                                         role="visual", rgba=bk.DARK)]))
    key = "smart servo 12 Nm"
    for side, sy in (("L", 1), ("R", -1)):
        sh, el, wr = bk.scaled(ms.wyvern_arm_points(p, sy))
        r = p["manip_radius"] * mm
        arm = bk.chain_through(f"{side}_talon", [sh, el, wr, (wr[0] + 0.1, wr[1], wr[2] - 0.09)], [[X, Y], Y, Y],
                               [r, r * 0.8, r * 0.4], origin=origin, servo=bk.servo(key), actuator_kg=zr._kg(key),
                               names=["shoulder", "elbow", "wrist"])
        children.append(arm)
    arm_geoms = []
    for k, c in enumerate(ms.rotor_centres(p)):
        c = tuple(v * mm for v in c)
        arm_geoms.append(bk.capsule(f"rotor_arm{k}", p["rotor_arm_radius"] * mm, (c[0] * 0.15, c[1] * 0.15, -0.03),
                                    (c[0], c[1], c[2] - zc), rho=1600.0, role="body"))
        arm_geoms.append(Geom(f"motor{k}", "cylinder", (p["motor_radius"] * mm, 0.03), pos=(c[0], c[1], c[2] - zc + 0.025),
                              mass=0.25, role="body", rgba=bk.DARK))
    a = math.radians(p["neck_angle_deg"])
    n0 = (L * 0.4, 0, H * 0.15)
    hc = (n0[0] + p["neck_length"] * mm * math.cos(a), 0, n0[2] + p["neck_length"] * mm * math.sin(a))
    body = Link("fuselage", pos=origin, log=True, group="body",
                geoms=[Geom("fuselage", "box", (L / 2 * 0.85, W / 2 * 0.85, H / 2 * 0.85), mass=zr._egg_mass(L, W, H),
                            role="body", rgba=bk.GREY),
                       bk.capsule("neck", p["head_height"] * mm * 0.3, n0, hc, rho=bk.DENSITY["foam"] * 3, role="body"),
                       bk.box("head", (p["head_length"] * mm / 2 * 0.8, p["head_width"] * mm / 2 * 0.8,
                                       p["head_height"] * mm / 2 * 0.8), hc, rho=bk.DENSITY["foam"] * 3),
                       bk.capsule("tail", p["body_height"] * mm * 0.1, (-L * 0.45, 0, 0),
                                  (-L * 0.45 - p["tail_length"] * mm, 0, -0.1), rho=bk.DENSITY["foam"] * 3, role="body")]
                      + arm_geoms,
                masses=[PointMass("battery_12S_16Ah", 4.0), PointMass("flight_computer", 0.4)], children=children)
    return bk.robot("Wyvern", body, "Wyvern: hexacopter dragon, 6 rotors (passive spin joints) and two 4-joint talon "
                                    "manipulators, 8 actuated joints (design stage).",
                    {"geometry": "designs/monster_specs.py", "actuators": f"ogma.actuators: {key}",
                     "propulsion": "none yet: rotors carry geometry only"})


def wraith(p=None):
    """Envelope root (fabric mass only; the lift is not modelled), the gondola fixed under it, ``arms`` arms."""
    p = p or ms.WRAITH
    zc = p["float_height"] * mm
    origin = (0.0, 0.0, zc)
    L, D = p["envelope_length"] * mm, p["envelope_diameter"] * mm
    area = math.pi * D * L * 0.85                                          # ≈ a prolate spheroid's surface
    zg = (p["float_height"] - p["envelope_diameter"] / 2 - p["suspension_length"] - p["gondola_height"] / 2) * mm
    key = "smart servo 6 Nm"
    arms = []
    for i in range(int(p["arms"])):
        pts = bk.scaled(ms.wraith_arm_points(p, i))
        n = len(pts) - 1
        r = p["arm_radius"] * mm
        arms.append(bk.chain_through(f"arm{i}", pts, [[Y, X]] * n, [r * (1 - 0.8 * k / n) for k in range(n)],
                                     origin=origin, servo=bk.servo(key), actuator_kg=zr._kg(key),
                                     names=[f"s{k}" for k in range(n)], rho=bk.DENSITY["foam"] * 4))
    Lg, Wg, Hg = p["gondola_length"] * mm, p["gondola_width"] * mm, p["gondola_height"] * mm
    root = Link("envelope", pos=origin, log=True, group="body",
                geoms=[Geom("envelope", "ellipsoid", (L / 2, D / 2, D / 2), mass=area * 0.2, role="body",
                            rgba=(0.85, 0.87, 0.9, 0.6)),
                       Geom("gondola", "box", (Lg / 2 * 0.85, Wg / 2 * 0.85, Hg / 2 * 0.85), pos=(0, 0, zg - zc),
                            mass=zr._egg_mass(Lg, Wg, Hg), role="body", rgba=bk.GREY)],
                masses=[PointMass("battery_and_electronics", 2.5, (0, 0, zg - zc))], children=arms)
    return bk.robot("Wraith", root, f"Wraith: balloon with a gondola and {int(p['arms'])} hanging two-axis arms, "
                                    f"{int(p['arms'] * p['arm_segments'] * 2)} actuated joints; buoyancy not modelled "
                                    "(design stage).",
                    {"geometry": "designs/monster_specs.py", "actuators": f"ogma.actuators: {key}",
                     "masses": "envelope: 0.2 kg/m² fabric on the spheroid's area"})
