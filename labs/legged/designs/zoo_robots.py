"""The zoomorphic designs as Chiron robots, for export (``machines.py`` → ``tools/export.py``): skeletons, masses and
actuated joints from the same numbers as the CAD (``zoo_specs``), shapes simplified to boxes, capsules and spheres
(the CAD rides along as the visual). Design stage: no controllers, nothing tuned for simulation.

    import zoo_robots as zr
    robot = zr.dog()            # also bear, deer, elephant, snake, bat, octopus
"""
from __future__ import annotations

import math

from vegeta.chiron import Geom, Link, PointMass

import body_kit as bk
import zoo_specs as zs

__all__ = ["quadruped", "dog", "bear", "deer", "elephant", "snake", "bat", "octopus", "ACTUATORS"]

mm = 1e-3
Y, X, Z = (0, 1, 0), (1, 0, 0), (0, 0, 1)

#: the catalogue actuator per machine (legs, then the trunk/arms where there are any)
ACTUATORS = {"dog": ("qdd 24 Nm",), "bear": ("qdd 60 Nm",), "deer": ("qdd 24 Nm",),
             "elephant": ("cycloidal 400 Nm, brake", "harmonic 60 Nm, brake"), "snake": ("smart servo 25 Nm",),
             "bat": ("smart servo 6 Nm", "mini 32 g"), "octopus": ("smart servo 12 Nm",)}


def _kg(key):
    from ogma import actuators as act
    return act.get(key).mass_g / 1000.0


def _egg_mass(L, W, H, rho=bk.DENSITY["shell"]):
    """The lofted egg ≈ an ellipsoid [m, kg]."""
    return math.pi / 6 * L * W * H * rho


def quadruped(p, name, key, extra_links=None, notes="", payload_kg=0.0, with_head=True):
    """A four-legged robot from a ``zoo_specs`` quadruped dict: a trunk (egg torso as a box, neck and head fixed
    on it), four legs through ``zoo_specs.leg_points`` with a two-axis hip (roll, pitch), a knee and, for
    three-segment legs, an ankle; feet ``FL``, ``FR``, ``RL``, ``RR``. ``extra_links(p, trunk_origin_m)``: more
    child links (trunks, arms) on the trunk; ``with_head`` False: no neck and head on the trunk (a centaur)."""
    sv, m_act = bk.servo(key), _kg(key)
    bc = tuple(c * mm for c in zs.body_centre(p))
    L, W, H = p["body_length"] * mm, p["body_width"] * mm, p["body_height"] * mm
    n0, hc = zs.head_centre(p)
    n0, hc = [tuple(c * mm - bc[k] for k, c in enumerate(q)) for q in (n0, hc)]
    torso = Geom("torso", "box", (L / 2 * 0.85, W / 2 * 0.85, H / 2 * 0.85), mass=_egg_mass(L, W, H), role="body",
                 rgba=bk.GREY)
    head = bk.box("head", (p["head_length"] * mm / 2 * 0.8, p["head_width"] * mm / 2 * 0.8,
                           p["head_height"] * mm / 2 * 0.8), hc)
    neck = bk.capsule("neck", p["neck_radius"] * mm, n0, hc, rho=bk.DENSITY["shell"], role="body", rgba=bk.GREY)
    legs = []
    for leg, front, sy in (("FL", True, 1), ("FR", True, -1), ("RL", False, 1), ("RR", False, -1)):
        pts = bk.scaled(zs.leg_points(p, front, sy))
        names = ["hip", "knee", "ankle"][: len(pts) - 1]
        axes = [[X, Y]] + [Y] * (len(pts) - 2)
        r = p["leg_radius"] * mm
        legs.append(bk.chain_through(leg, pts, axes, [r, r * 0.8, r * 0.7], origin=bc, servo=sv,
                                     foot_r=p["foot_radius"] * mm, actuator_kg=m_act, names=names))
    children = legs + (extra_links(p, bc) if extra_links else [])
    masses = [PointMass("battery_and_electronics", payload_kg)] if payload_kg else []
    trunk = Link("trunk", pos=bc, log=True, group="body", geoms=[torso, neck, head] if with_head else [torso],
                 masses=masses,
                 children=children)
    return bk.robot(name, trunk, notes, {"geometry": "designs/zoo_specs.py", "actuators": f"ogma.actuators: {key}",
                                         "masses": "body_kit.DENSITY × shape volumes + listed payload"})


def dog():
    return quadruped(zs.DOG, "Laika", ACTUATORS["dog"][0], payload_kg=2.5,
                     notes="Laika: dog-like quadruped, 16 actuated joints (design stage).")


def bear():
    return quadruped(zs.BEAR, "Ursa", ACTUATORS["bear"][0], payload_kg=8.0,
                     notes="Ursa: bear-like quadruped, plantigrade 2-segment legs, 12 actuated joints (design stage).")


def deer():
    return quadruped(zs.DEER, "Cerva", ACTUATORS["deer"][0], payload_kg=3.0,
                     notes="Cerva: deer-like quadruped, long 3-segment legs, 16 actuated joints (design stage).")


def _elephant_trunk(p, bc):
    _, hc = zs.head_centre(p)
    start = (hc[0] + p["head_length"] * 0.4, 0, hc[2] - p["head_height"] * 0.1)
    pts = bk.scaled(zs.arm_points(start, int(p["trunk_segments"]), p["trunk_segment_length"], -70, -8))
    n = len(pts) - 1
    r = p["head_width"] * mm * 0.14
    return [bk.chain_through("trunk", pts, [[Y, Z]] * n, [r * (1 - 0.06 * i) for i in range(n)], origin=bc,
                             servo=bk.servo(ACTUATORS["elephant"][1]), actuator_kg=_kg(ACTUATORS["elephant"][1]),
                             names=[f"s{i}" for i in range(n)], rho=bk.DENSITY["limb"])]


def elephant():
    return quadruped(zs.ELEPHANT, "Gaja", ACTUATORS["elephant"][0], _elephant_trunk, payload_kg=60.0,
                     notes="Gaja: elephant robot, pillar legs and an 8-segment two-axis trunk, 28 actuated joints "
                           "(design stage).")


def snake(p=None, name="Naga", key=None, notes="Naga: snake robot, 16 two-axis segments (design stage)."):
    """The head is the root; the segments trail behind it (``zoo_specs.SNAKE``, wave pose as in the CAD)."""
    p = p or zs.SNAKE
    key = key or ACTUATORS["snake"][0]
    pts = bk.scaled(zs.slither_points(p))
    n = len(pts) - 1
    r0 = p["body_radius"] * mm
    radii = [r0 + (p["tail_radius"] * mm - r0) * i / max(n - 1, 1) for i in range(n)]
    origin = pts[0]
    body = bk.chain_through("seg", pts, [[Z, Y]] * n, radii, origin=origin, servo=bk.servo(key), actuator_kg=_kg(key),
                            names=[f"{i:02d}" for i in range(n)])
    Lh, Wh, Hh = p["head_length"] * mm, p["head_width"] * mm, p["head_height"] * mm
    head = Link("head", pos=origin, log=True, group="body",
                geoms=[bk.box("head", (Lh / 2 * 0.8, Wh / 2 * 0.8, Hh / 2 * 0.8), (Lh * 0.4, 0, 0))],
                masses=[PointMass("battery_and_electronics", 1.0)], children=[body])
    return bk.robot(name, head, notes, {"geometry": "designs/zoo_specs.py", "actuators": f"ogma.actuators: {key}"})


def bat(p=None):
    """Body root; per wing a two-axis shoulder, elbow, wrist and three finger joints (the fold), two legs."""
    p = p or zs.BAT
    big, small = ACTUATORS["bat"]
    z = p["stand_height"] * mm
    origin = (0.0, 0.0, z)
    children = []
    for side, sy in (("L", 1), ("R", -1)):
        sh, el, wr, tips = (bk.scaled([q])[0] if not isinstance(q, list) else bk.scaled(q)
                            for q in zs.bat_wing_points(p, sy))
        r = p["bone_radius"] * mm
        wing = bk.chain_through(f"{side}_wing", [sh, el, wr], [[X, Z], Z], [r * 1.4, r * 1.2], origin=origin,
                                servo=bk.servo(big), actuator_kg=_kg(big), names=["shoulder", "elbow"])
        elbow_link = wing.children[0]
        for i, t in enumerate(tips):
            elbow_link.children.append(bk.chain_through(f"{side}_finger{i}", [wr, t], [Z], [r * 0.8], origin=el,
                                                        servo=bk.servo(small), actuator_kg=_kg(small),
                                                        names=["wrist"]))
        hip = (-p["body_length"] * 0.3 * mm, sy * p["body_width"] * 0.3 * mm, z - p["body_height"] * 0.3 * mm)
        foot = (-p["body_length"] * 0.35 * mm, sy * p["body_width"] * 0.35 * mm, 8.0 * mm)
        children += [wing, bk.chain_through(side, [hip, foot], [Y], [r], origin=origin, servo=bk.servo(small),
                                            actuator_kg=_kg(small), foot_r=8.0 * mm, names=["hip"])]
    L, W, H = p["body_length"] * mm, p["body_width"] * mm, p["body_height"] * mm
    body = Link("body", pos=origin, log=True, group="body",
                geoms=[Geom("body", "box", (L / 2 * 0.85, W / 2 * 0.85, H / 2 * 0.85), mass=_egg_mass(L, W, H),
                            role="body", rgba=bk.GREY)],
                masses=[PointMass("battery_and_electronics", 0.35)], children=children)
    return bk.robot("Noctis", body, "Noctis: bat robot with folding wings, 14 actuated joints (design stage).",
                    {"geometry": "designs/zoo_specs.py (bat_wing_points)", "actuators": f"ogma.actuators: {big}, {small}"})


def octopus(p=None, name="Kraken", root_geoms=None, root_masses=None, notes=None, key=None):
    """Head root; ``arms`` arms of ``arm_segments`` two-axis segments each."""
    p = p or zs.OCTOPUS
    key = key or ACTUATORS["octopus"][0]
    rh = p["head_radius"] * mm
    zc = (p["arm_radius"] + 40.0 + p["head_radius"] * 0.8) * mm
    origin = (0.0, 0.0, zc)
    arms = []
    for i in range(int(p["arms"])):
        pts = bk.scaled(zs.octopus_arm_points(p, i))
        n = len(pts) - 1
        r = p["arm_radius"] * mm
        arms.append(bk.chain_through(f"arm{i}", pts, [[Z, Y]] * n, [r * (1 - 0.8 * k / n) for k in range(n)],
                                     origin=origin, servo=bk.servo(key), actuator_kg=_kg(key),
                                     names=[f"s{k}" for k in range(n)]))
    geoms = root_geoms if root_geoms is not None else [bk.sphere("head", rh, rho=bk.DENSITY["shell"], rgba=bk.GREY)]
    head = Link("head", pos=origin, log=True, group="body", geoms=geoms,
                masses=root_masses if root_masses is not None else [PointMass("battery_and_electronics", 3.0)],
                children=arms)
    n_j = int(p["arms"]) * int(p["arm_segments"]) * 2
    return bk.robot(name, head, notes or f"Kraken: octopus robot, 8 arms × 8 two-axis segments, {n_j} actuated "
                                         "joints (design stage).",
                    {"geometry": "designs/zoo_specs.py (octopus_arm_points)", "actuators": f"ogma.actuators: {key}"})
