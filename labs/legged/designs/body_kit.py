"""Small building blocks for the design-stage robots (humanoid, zoomorphic and monster designs): Chiron links whose
masses come from their shapes' volumes × a density, actuated joints with a catalogue servo, and a ``Robot`` with
named feet. Enough for a Yax export (``tools/export.py``: skeleton, masses, joints, the CAD looks); none of these
robots has been tuned for simulation yet. Units: SI (m, kg, rad); frames: x forward, y left, z up.

    import body_kit as bk
    leg = bk.chain("FL", (0.2, 0.1, 0.0), [("hip", (0, 1, 0), 0.25, 0.03), ("knee", (0, 1, 0), 0.25, 0.02)],
                   servo=bk.servo("qdd 24 Nm"), foot_r=0.025)
"""
from __future__ import annotations

import math

from vegeta.chiron import FootSpec, Geom, Joint, Link, PointMass, Robot, Servo

from ogma import actuators as act

__all__ = ["DENSITY", "servo", "box_mass", "capsule_mass", "sphere_mass", "box", "capsule", "sphere", "chain",
           "feet_of", "robot"]

#: kg/m³ of the printed and machined parts (shells are solid-equivalent densities: a printed part at ~40 % infill).
DENSITY = {"shell": 450.0, "limb": 600.0, "foam": 60.0, "rubber": 1100.0}

GREY, DARK, BLACK = (0.81, 0.84, 0.87, 1.0), (0.25, 0.26, 0.28, 1.0), (0.1, 0.1, 0.1, 1.0)


def servo(key: str = "qdd 24 Nm", kp: float = 100.0, kd: float = 1.0) -> Servo:
    """A catalogue actuator (``ogma.actuators``) as a Chiron position servo."""
    return Servo.from_actuator(act.get(key), kp=kp, kd=kd)


def box_mass(hx, hy, hz, rho):
    return 8 * hx * hy * hz * rho


def capsule_mass(r, length, rho):
    return (math.pi * r * r * length + 4.0 / 3.0 * math.pi * r ** 3) * rho


def sphere_mass(r, rho):
    return 4.0 / 3.0 * math.pi * r ** 3 * rho


def box(name, half, pos=(0.0, 0.0, 0.0), rho=DENSITY["shell"], role="body", rgba=GREY, **kw) -> Geom:
    return Geom(name, "box", tuple(half), pos=tuple(pos), mass=box_mass(*half, rho), role=role, rgba=rgba, **kw)


def capsule(name, r, a, b, rho=DENSITY["limb"], role="link", rgba=DARK, **kw) -> Geom:
    length = math.dist(a, b)
    return Geom(name, "capsule", (r,), fromto=tuple(a) + tuple(b), mass=capsule_mass(r, length, rho), role=role,
                rgba=rgba, **kw)


def sphere(name, r, pos=(0.0, 0.0, 0.0), rho=DENSITY["rubber"], role="body", rgba=BLACK, **kw) -> Geom:
    return Geom(name, "sphere", (r,), pos=tuple(pos), mass=sphere_mass(r, rho), role=role, rgba=rgba, **kw)


def chain(prefix: str, pos, segments, *, servo: Servo | None = None, foot_r: float | None = None,
          direction=(0.0, 0.0, -1.0), rho=DENSITY["limb"], actuator_kg: float = 0.0, joint_range=(-1.6, 1.6),
          tip_geom: Geom | None = None) -> Link:
    """A serial limb at ``pos`` in its parent: ``segments`` = [(joint, axis, length [m], radius [m]), ...], each a
    hinge at the segment's start and a capsule along ``direction``. ``foot_r`` puts a foot sphere (role 'foot',
    named ``prefix``) at the tip; ``tip_geom`` adds another tip shape. Each joint carries ``actuator_kg`` as a
    point mass at its axis. Joint names: ``f'{prefix}_{joint}'``, ``leg`` tag ``prefix``."""
    n = math.sqrt(sum(c * c for c in direction)) or 1.0
    d = tuple(c / n for c in direction)
    child = None
    links = []
    for jname, axis, length, r in segments:
        tip = tuple(c * length for c in d)
        geoms = [capsule(f"{prefix}_{jname}_seg", r, (0, 0, 0), tip, rho=rho)]
        links.append(Link(f"{prefix}_{jname}_link", pos=(0, 0, 0),
                          joints=[Joint(f"{prefix}_{jname}", axis=tuple(axis), range=joint_range, tag=jname,
                                        servo=servo, leg=prefix)],
                          geoms=geoms,
                          masses=[PointMass(f"{prefix}_{jname}_actuator", actuator_kg)] if actuator_kg else []))
    # place each link at the tip of the previous one and nest them
    for i in range(len(links) - 1, -1, -1):
        lk = links[i]
        if i == len(links) - 1:
            length = segments[i][2]
            tip = tuple(c * length for c in d)
            if foot_r:
                lk.geoms.append(Geom(prefix, "sphere", (foot_r,), pos=tip, mass=sphere_mass(foot_r, DENSITY["rubber"]),
                                     role="foot", friction=(0.8, 0.005, 0.0001), rgba=BLACK))
            if tip_geom is not None:
                lk.geoms.append(tip_geom)
        if child is not None:
            prev_len = segments[i][2]
            child.pos = tuple(c * prev_len for c in d)
            lk.children.append(child)
        child = lk
    child.pos = tuple(pos)
    return child


def feet_of(root: Link, body: str) -> list:
    """FootSpecs for every role-'foot' geom in the tree, its leg's joints being the joints tagged with its name."""
    joints, feet = {}, []
    for lk, _ in root.walk():
        for j in lk.joints:
            if j.leg:
                joints.setdefault(j.leg, []).append(j.name)
    for lk, _ in root.walk():
        for g in lk.geoms:
            if g.role == "foot":
                feet.append(FootSpec(g.name, g.name, joints.get(g.name, []), body))
    return feet


def robot(name: str, root: Link, notes: str, sources: dict) -> Robot:
    return Robot(name, root, feet=feet_of(root, root.name), notes=notes, sources=sources)
