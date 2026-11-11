"""The Pekari family's manipulators (Pekari-2M and Pumba, ``docs/pekari_pumba.md``): one arm description shared by
both platforms — kinematics, statics, the rated load against reach, the workspace and the Chiron (MuJoCo) Link tree.

An arm (``ArmSpec``) is a serial chain on a pedestal on the hull roof, in the Onager Manus conventions
(``onager_manus_robot``): x forward, y left, z up; the shoulder point is the shoulder-pitch axis above the pedestal.

==================  ==========  ===============================================================================
joint               axis        convention
==================  ==========  ===============================================================================
``yaw``             z           the pedestal; + turns the arm to the left
``shoulder``        y           the upper arm's elevation is ``−q``
``elbow``           y           relative; the forearm's elevation ``−(q_sh + q_el)``
``wrist_pitch``     y           relative; the tool axis's elevation ``−(q_sh + q_el + q_wp)``
``wrist_yaw``       z (tool)    optional (the 7-DOF arm): turns the tool about the wrist's own up axis
``wrist_roll``      x (tool)    rolls the gripper about the tool axis
``grip``            y (jaw)     two hinged jaws, symmetric; + closes (mirrored joints ``grip_a`` / ``grip_b``)
==================  ==========  ===============================================================================

The tool centre point (TCP) is ``tool`` metres out along the tool axis from the wrist-pitch axis (between the jaw
tips). ``dof`` counts the arm joints without the gripper: 5 (yaw, shoulder, elbow, wrist pitch, wrist roll — the
light arm), 6 or 7 (adds wrist yaw; the heavy arm). The gripper is interchangeable at the wrist-roll flange
(``tool_flange``): its mass is the gripper's, the rest of the arm does not change.

Statics are planar (in the arm's vertical plane after the yaw): each joint about y holds the moment of everything
outboard of it. ``rated_load`` is the largest load at the TCP that every pitch joint holds with the arm
outstretched horizontally at a given reach (the worst case for every pitch joint at that reach) — at the module's
stall torque over a safety factor (``peak``: lifting) or at its rated (continuous, thermal) torque (``hold``).

    import pekari_arms as pa
    arm = pa.PEKARI_ARM
    q = pa.ik(arm, (0.6, 0.0, 0.1), elevation=-math.pi / 2)
    pa.rated_load(arm, 0.6)               # kg at 0.6 m horizontal reach from the shoulder
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from ogma import actuators as act

__all__ = ["G", "ArmSpec", "PEKARI_ARM", "PUMBA_MAIN_ARM", "CALYDON_MAIN_ARM", "CALYDON_AUX_ARM", "JOINTS", "joint_names", "arm_dof",
           "masses", "total_mass", "fk", "ik", "pose", "mass_points", "gravity_torques", "rated_load", "load_curve",
           "workspace", "reach", "arm_power", "links"]

G = 9.81
SF_PEAK = 1.5                            # stall torque / required torque when lifting (dynamic margin)
JOINTS = ("yaw", "shoulder", "elbow", "wrist_pitch", "wrist_yaw", "wrist_roll", "grip")
AL_RHO = 2700.0                          # Al 6082-T6 tubes [kg/m³]


@dataclass(frozen=True)
class ArmSpec:
    """One manipulator: lengths [m], tube sections [m], drives (``ogma.actuators`` keys), listed parts [kg],
    joint ranges [deg] in ``q`` (the conventions above), servo gains (kp, kd, armature) and the named poses."""

    name: str
    upper: float                          # shoulder pitch to elbow
    fore: float                           # elbow to wrist pitch
    tool: float                           # wrist pitch to the TCP (wrist modules + flange + gripper)
    upper_section: tuple                  # (width y, depth z) of the rectangular tube
    fore_section: tuple
    wall: float                           # tube wall
    pedestal: tuple                       # (height of the shoulder axis above the roof, radius)
    drives: dict                          # joint -> actuator key (``wrist_yaw`` present = a 7-DOF arm)
    gripper: str                          # the default tool on the flange
    gripper_kg: float
    jaw_length: float                     # jaw pivot to tip
    parts_kg: dict                        # brackets, cabling, covers, the flange
    ranges: dict                          # joint -> (lo, hi) deg
    gains: dict                           # joint -> (kp, kd, armature): kp so the rated load sags < ~2 deg without feed-forward
    poses: dict                           # name -> {joint: deg}; 'folded' and 'deployed' at least
    rated_payload_kg: float               # the design value stated for the arm at ``rated_reach`` (checked by rated_load)
    rated_reach: float                    # [m] horizontal, shoulder axis to TCP
    assumptions: dict = field(default_factory=dict)


def joint_names(arm: ArmSpec, prefix: str = "") -> list:
    """The arm's joints in chain order (``grip`` last), with ``prefix`` (``"L_"``, ``"main_"``)."""
    return [prefix + j for j in JOINTS if j in arm.drives]


def arm_dof(arm: ArmSpec) -> int:
    """Degrees of freedom without the gripper."""
    return sum(1 for j in arm.drives if j != "grip")


# ----------------------------------------------------------------------------------------------- masses
def _tube_kg(L: float, section: tuple, wall: float) -> float:
    w, d = section
    return 2 * (w + d) * wall * L * AL_RHO


def masses(arm: ArmSpec) -> dict:
    """The arm's parts [kg]: the two tubes (thin-walled Al 6082, from the lengths), each drive (catalogue), the
    gripper and the listed parts."""
    out = {f"upper arm tube ({arm.upper_section[0] * 1e3:.0f}x{arm.upper_section[1] * 1e3:.0f}x{arm.wall * 1e3:g} mm Al 6082)":
           _tube_kg(arm.upper, arm.upper_section, arm.wall),
           f"forearm tube ({arm.fore_section[0] * 1e3:.0f}x{arm.fore_section[1] * 1e3:.0f}x{arm.wall * 1e3:g} mm Al 6082)":
           _tube_kg(arm.fore, arm.fore_section, arm.wall)}
    for j in JOINTS:
        if j in arm.drives:
            out[f"{j} drive ({arm.drives[j]})"] = act.get(arm.drives[j]).mass_g / 1000.0
    out[f"gripper ({arm.gripper})"] = arm.gripper_kg
    out.update(arm.parts_kg)
    return out


def total_mass(arm: ArmSpec) -> float:
    return sum(masses(arm).values())


def _drive_kg(arm, j):
    return act.get(arm.drives[j]).mass_g / 1000.0 if j in arm.drives else 0.0


# ----------------------------------------------------------------------------------------------- kinematics
def pose(arm: ArmSpec, name: str) -> dict:
    """A named pose as joint angles [rad] (missing joints 0)."""
    return {j: math.radians(arm.poses[name].get(j, 0.0)) for j in JOINTS if j in arm.drives}


def _q(arm, q):
    if isinstance(q, dict):
        return {j: float(q.get(j, 0.0)) for j in JOINTS}
    return dict(zip([j for j in JOINTS if j in arm.drives], [float(v) for v in q])) | {j: 0.0 for j in JOINTS if j not in arm.drives}


def fk(arm: ArmSpec, q, shoulder=(0.0, 0.0, 0.0)) -> dict:
    """Points [m] of the chain for joint angles ``q`` (dict or sequence in ``joint_names`` order) with the shoulder
    at ``shoulder`` (x, y, z): ``elbow``, ``wrist``, ``tcp`` and the tool's ``elevation`` [rad] (wrist yaw is taken
    as turning the tool in the yawed vertical plane's normal direction: it moves the TCP sideways)."""
    q = _q(arm, q)
    sx, sy, sz = shoulder
    yaw = q["yaw"]
    e1 = -q["shoulder"]
    e2 = -(q["shoulder"] + q["elbow"])
    e3 = -(q["shoulder"] + q["elbow"] + q["wrist_pitch"])
    c, s = math.cos(yaw), math.sin(yaw)

    def at(r, z, lateral=0.0):
        return (sx + r * c - lateral * s, sy + r * s + lateral * c, sz + z)

    r1, z1 = arm.upper * math.cos(e1), arm.upper * math.sin(e1)
    r2, z2 = r1 + arm.fore * math.cos(e2), z1 + arm.fore * math.sin(e2)
    wy = q["wrist_yaw"]
    t_along = arm.tool * math.cos(wy)
    r3, z3 = r2 + t_along * math.cos(e3), z2 + t_along * math.sin(e3)
    return {"shoulder": (sx, sy, sz), "elbow": at(r1, z1), "wrist": at(r2, z2), "tcp": at(r3, z3, arm.tool * math.sin(wy)),
            "elevation": e3, "r_tcp": r3, "z_tcp": z3}


def ik(arm: ArmSpec, target, elevation: float, shoulder=(0.0, 0.0, 0.0), roll: float = 0.0, elbow_up: bool = True):
    """Joint angles (dict, rad) that put the TCP at ``target`` (x, y, z) with the tool axis at ``elevation`` [rad]
    above horizontal (−π/2 = straight down) and wrist yaw 0; None when out of reach or outside a joint's range."""
    dx, dy, dz = (target[0] - shoulder[0], target[1] - shoulder[1], target[2] - shoulder[2])
    yaw = math.atan2(dy, dx)
    r = math.hypot(dx, dy)
    wr, wz = r - arm.tool * math.cos(elevation), dz - arm.tool * math.sin(elevation)
    d = math.hypot(wr, wz)
    L1, L2 = arm.upper, arm.fore
    if d > L1 + L2 - 1e-9 or d < abs(L1 - L2) + 1e-9:
        return None
    beta = math.acos(max(-1.0, min(1.0, (L1 * L1 + d * d - L2 * L2) / (2 * L1 * d))))
    e1 = math.atan2(wz, wr) + (beta if elbow_up else -beta)
    ex, ez = L1 * math.cos(e1), L1 * math.sin(e1)
    e2 = math.atan2(wz - ez, wr - ex)
    q = {"yaw": yaw, "shoulder": -e1, "elbow": e1 - e2, "wrist_pitch": e2 - elevation}
    if "wrist_yaw" in arm.drives:
        q["wrist_yaw"] = 0.0
    q["wrist_roll"] = roll
    for j, v in q.items():
        lo, hi = arm.ranges[j]
        if not math.radians(lo) - 1e-9 <= v <= math.radians(hi) + 1e-9:
            return None
    return q


def reach(arm: ArmSpec) -> float:
    """Largest horizontal distance [m] from the shoulder axis to the TCP (outstretched, level)."""
    return arm.upper + arm.fore + arm.tool


# ----------------------------------------------------------------------------------------------- statics
def mass_points(arm: ArmSpec, q, shoulder=(0.0, 0.0, 0.0), load_kg: float = 0.0) -> list:
    """Every mass of the arm (and the load at the TCP) at pose ``q``: [(name, kg, x, y, z)] [m]. The yaw drive
    sits under the pedestal in the hull (at the shoulder's x, y, a pedestal height below)."""
    q = _q(arm, q)
    m = masses(arm)
    pts = fk(arm, q, shoulder)
    sx, sy, sz = shoulder

    def lerp(a, b, t):
        return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))

    sh, el, wr, tcp = pts["shoulder"], pts["elbow"], pts["wrist"], pts["tcp"]
    upper_kg = next(v for k, v in m.items() if k.startswith("upper arm tube"))
    fore_kg = next(v for k, v in m.items() if k.startswith("forearm tube"))
    listed = sum(arm.parts_kg.values())
    out = [("yaw drive", _drive_kg(arm, "yaw"), sx, sy, sz - arm.pedestal[0]),
           ("shoulder drive", _drive_kg(arm, "shoulder"), *sh),
           ("upper arm tube", upper_kg, *lerp(sh, el, 0.5)),
           ("elbow drive", _drive_kg(arm, "elbow"), *el),
           ("forearm tube", fore_kg, *lerp(el, wr, 0.5)),
           ("wrist pitch drive", _drive_kg(arm, "wrist_pitch"), *wr),
           ("wrist roll and yaw drives", _drive_kg(arm, "wrist_roll") + _drive_kg(arm, "wrist_yaw"), *lerp(wr, tcp, 0.3)),
           ("gripper", arm.gripper_kg + _drive_kg(arm, "grip"), *lerp(wr, tcp, 0.65)),
           ("brackets, cabling, covers", listed, *lerp(sh, wr, 0.5))]
    if load_kg:
        out.append(("load", load_kg, *tcp))
    return out


def gravity_torques(arm: ArmSpec, q, load_kg: float = 0.0, extra_force_N=(0.0, 0.0)) -> dict:
    """Static torque [N m] each pitch joint holds at pose ``q`` with ``load_kg`` at the TCP (and an extra force
    (horizontal along the arm, vertical) [N] at the TCP, e.g. a pull): magnitude of the moment of everything
    outboard of the joint about its axis."""
    pts = mass_points(arm, q, load_kg=load_kg)
    p = fk(arm, q)
    names = [n for n, *_ in pts]
    yaw = _q(arm, q)["yaw"]
    c, s = math.cos(yaw), math.sin(yaw)

    def rad(x, y):                       # distance along the arm's plane
        return x * c + y * s

    outboard = {"shoulder": [n for n in names if n not in ("yaw drive", "shoulder drive")],
                "elbow": ["forearm tube", "wrist pitch drive", "wrist roll and yaw drives", "gripper", "load"],
                "wrist_pitch": ["wrist roll and yaw drives", "gripper", "load"]}
    # the listed parts ride half on the upper arm and half on the forearm: take them as outboard of the elbow by half
    out = {}
    for j, key in (("shoulder", "shoulder"), ("elbow", "elbow"), ("wrist_pitch", "wrist")):
        pivot = p[key]
        rp = rad(pivot[0], pivot[1])
        tau = 0.0
        for n, kg, x, y, z in pts:
            if n not in outboard[j] and not (n == "brackets, cabling, covers" and j == "elbow"):
                continue
            w = kg * G * (0.5 if (n == "brackets, cabling, covers" and j == "elbow") else 1.0)
            tau += w * (rad(x, y) - rp)
        fh, fv = extra_force_N
        tcp = p["tcp"]
        tau += -fv * (rad(tcp[0], tcp[1]) - rp) + fh * (tcp[2] - pivot[2])
        out[j] = abs(tau)
    return out


def _limits(arm: ArmSpec, mode: str) -> dict:
    out = {}
    for j in ("shoulder", "elbow", "wrist_pitch"):
        a = act.get(arm.drives[j])
        out[j] = a.stall_Nm / SF_PEAK if mode == "peak" else a.rated_Nm
    return out


def _outstretched(arm: ArmSpec, r: float) -> dict | None:
    """The level pose with the TCP at horizontal distance ``r`` from the shoulder and the shoulder's height, tool
    level (None when ``r`` is out of reach)."""
    return ik(arm, (r, 0.0, 0.0), 0.0) or (pose_level(arm) if r >= reach(arm) - 1e-9 else None)


def pose_level(arm: ArmSpec) -> dict:
    return {"yaw": 0.0, "shoulder": 0.0, "elbow": 0.0, "wrist_pitch": 0.0, "wrist_yaw": 0.0, "wrist_roll": 0.0}


def rated_load(arm: ArmSpec, r: float | None = None, mode: str = "peak") -> float:
    """Largest load [kg] at the TCP that every pitch joint holds with the arm outstretched to horizontal reach ``r``
    [m] at shoulder height (default: full reach; 0 beyond it), at stall / ``SF_PEAK`` (``peak``) or the rated torque (``hold``)."""
    if r is not None and r > reach(arm) + 1e-9:
        return 0.0
    r = reach(arm) if r is None else r
    q = _outstretched(arm, r)
    if q is None:
        return 0.0
    lim = _limits(arm, mode)
    t0 = gravity_torques(arm, q, 0.0)
    t1 = gravity_torques(arm, q, 1.0)
    best = math.inf
    for j in lim:
        per_kg = t1[j] - t0[j]
        if per_kg <= 1e-12:
            continue
        best = min(best, (lim[j] - t0[j]) / per_kg)
    return max(0.0, best)


def load_curve(arm: ArmSpec, n: int = 9, mode: str = "peak") -> list:
    """[(reach m, kg)] from 40 % to 100 % of full reach."""
    R = reach(arm)
    return [(R * f, rated_load(arm, R * f, mode)) for f in [0.4 + 0.6 * i / (n - 1) for i in range(n)]]


def workspace(arm: ArmSpec, elevation: float = -math.pi / 2, n: int = 41, r_max: float | None = None,
              z_span=(-1.5, 1.5)) -> list:
    """Reachable (r, z) points [m] relative to the shoulder in the arm's plane with the tool at ``elevation``
    (default pointing down: grasping from above), on an n x n grid; joint ranges respected."""
    R = reach(arm) if r_max is None else r_max
    zlo, zhi = z_span
    pts = []
    for i in range(n):
        r = R * i / (n - 1)
        for k in range(n):
            z = zlo + (zhi - zlo) * k / (n - 1)
            if ik(arm, (r, 0.0, z), elevation) is not None:
                pts.append((r, z))
    return pts


def arm_power(arm: ArmSpec, duty: float = 0.3, load_kg: float | None = None) -> float:
    """Average electrical power [W] of the arm working at ``duty`` (fraction of the time moving): the pitch drives
    at the torque that holds the rated payload outstretched, at a third of their no-load speed, through 70 %
    motor+gear efficiency; the brakes hold for free the rest of the time; 10 W of controllers always."""
    load = arm.rated_payload_kg if load_kg is None else load_kg
    q = _outstretched(arm, arm.rated_reach) or pose_level(arm)
    tau = gravity_torques(arm, q, load)
    mech = sum(tau[j] * act.get(arm.drives[j]).no_load_rpm * 2 * math.pi / 60 / 3 for j in tau)
    return duty * mech / 0.7 + 10.0


# ----------------------------------------------------------------------------------------------- Chiron
def links(arm: ArmSpec, prefix: str, shoulder: tuple, *, jaw_mu: float = 0.7, rgba=(0.95, 0.55, 0.05, 1.0),
          tube_role: str = "link"):
    """The arm as a Chiron Link tree under the hull (``shoulder`` in the hull frame, m), the yaw drive as a hull
    PointMass, and the arm's moving mass. Joints ``{prefix}yaw`` … ``{prefix}grip_a`` / ``{prefix}grip_b``."""
    from vegeta.chiron import Geom, Joint, Link, PointMass, Servo

    m = masses(arm)
    fr = (0.5, 0.005, 0.0001)
    jaw_fr = (jaw_mu, 0.01, 0.0001)
    dark = (0.22, 0.23, 0.25, 1.0)
    steel = (0.6, 0.62, 0.64, 1.0)

    def servo(j):
        kp, kd, armature = arm.gains[j]
        a = act.get(arm.drives[j])
        return replace(Servo.from_actuator(a, kp=kp, kd=kd, armature=armature), four_quadrant=a.self_locking)

    def rng(j):
        lo, hi = arm.ranges[j]
        return (math.radians(lo), math.radians(hi))

    upper_kg = next(v for k, v in m.items() if k.startswith("upper arm tube"))
    fore_kg = next(v for k, v in m.items() if k.startswith("forearm tube"))
    listed = sum(arm.parts_kg.values())
    uw, ud = arm.upper_section
    fw, fd = arm.fore_section
    # the tool: wrist roll (and yaw) modules near the wrist, the gripper's palm, two jaws
    t_mod = 0.35 * arm.tool
    palm_len = max(arm.tool - arm.jaw_length - t_mod, 0.02)
    jaw_kg = 0.15 * arm.gripper_kg
    palm_kg = arm.gripper_kg - 2 * jaw_kg
    jaw_w, jaw_d = max(0.6 * fw, 0.02), max(0.25 * fd, 0.012)
    jaws = []
    for side, sgn, axis in (("a", 1, (0, 1, 0)), ("b", -1, (0, -1, 0))):     # + turns each tip inwards: closes
        jn = f"{prefix}grip_{side}"
        jaws.append(Link(jn + "_link", pos=(palm_len, 0.0, sgn * 0.5 * fd * 0.6),
                         joints=[Joint(jn, axis=axis, range=rng("grip"), tag="jaw", servo=servo("grip"))],
                         geoms=[Geom(jn, "box", (arm.jaw_length / 2, jaw_w / 2, jaw_d / 2),
                                     pos=(arm.jaw_length / 2, 0.0, -sgn * jaw_d / 2), mass=jaw_kg, role="link",
                                     friction=jaw_fr, rgba=steel)]))
    hand = Link(f"{prefix}hand", pos=(t_mod, 0.0, 0.0),
                joints=[Joint(f"{prefix}wrist_roll", axis=(1, 0, 0), range=rng("wrist_roll"), tag="wrist_roll",
                              servo=servo("wrist_roll"))],
                geoms=[Geom(f"{prefix}palm", "box", (palm_len / 2, fw * 0.6, fd * 0.6), pos=(palm_len / 2, 0.0, 0.0),
                            mass=max(palm_kg, 0.05), role="link", friction=fr, rgba=dark)],
                children=jaws)
    wrist_mod_kg = _drive_kg(arm, "wrist_roll") + _drive_kg(arm, "wrist_yaw") + _drive_kg(arm, "grip")
    wrist_children = [hand]
    if "wrist_yaw" in arm.drives:
        wrist_children = [Link(f"{prefix}wrist_yaw_link", pos=(0.0, 0.0, 0.0),
                               joints=[Joint(f"{prefix}wrist_yaw", axis=(0, 0, 1), range=rng("wrist_yaw"), tag="wrist_yaw",
                                             servo=servo("wrist_yaw"))],
                               geoms=[Geom(f"{prefix}wrist_yaw_housing", "cylinder", (fw * 0.55, fd * 0.45),
                                           pos=(t_mod * 0.3, 0.0, 0.0), mass=_drive_kg(arm, "wrist_yaw"), role="visual", rgba=dark)],
                               children=[hand])]
        wrist_mod_kg -= _drive_kg(arm, "wrist_yaw")
    wrist = Link(f"{prefix}wrist", pos=(arm.fore, 0.0, 0.0),
                 joints=[Joint(f"{prefix}wrist_pitch", axis=(0, 1, 0), range=rng("wrist_pitch"), tag="wrist",
                               servo=servo("wrist_pitch"))],
                 geoms=[Geom(f"{prefix}wrist_housing", "box", (t_mod / 2, fw * 0.55, fd * 0.55), pos=(t_mod / 2, 0.0, 0.0),
                             mass=max(wrist_mod_kg, 0.05), role="link", friction=fr, rgba=dark)],
                 children=wrist_children)
    fore = Link(f"{prefix}forearm", pos=(arm.upper, 0.0, 0.0),
                joints=[Joint(f"{prefix}elbow", axis=(0, 1, 0), range=rng("elbow"), tag="elbow", servo=servo("elbow"))],
                geoms=[Geom(f"{prefix}forearm", "box", (arm.fore / 2, fw / 2, fd / 2), pos=(arm.fore / 2, 0.0, 0.0),
                            mass=fore_kg, role=tube_role, friction=fr, rgba=rgba)],
                masses=[PointMass(f"{prefix}wrist_pitch_drive", _drive_kg(arm, "wrist_pitch"), (arm.fore, 0.0, 0.0)),
                        PointMass(f"{prefix}fore_cabling", listed / 2, (arm.fore / 2, 0.0, 0.0))],
                children=[wrist])
    upper = Link(f"{prefix}upper_arm",
                 joints=[Joint(f"{prefix}shoulder", axis=(0, 1, 0), range=rng("shoulder"), tag="shoulder",
                               servo=servo("shoulder"))],
                 geoms=[Geom(f"{prefix}upper_arm", "box", (arm.upper / 2, uw / 2, ud / 2), pos=(arm.upper / 2, 0.0, 0.0),
                             mass=upper_kg, role=tube_role, friction=fr, rgba=rgba)],
                 masses=[PointMass(f"{prefix}elbow_drive", _drive_kg(arm, "elbow"), (arm.upper, 0.0, 0.0)),
                         PointMass(f"{prefix}upper_cabling", listed / 2, (arm.upper / 2, 0.0, 0.0))],
                 children=[fore])
    ped_h, ped_r = arm.pedestal
    base = Link(f"{prefix}arm_base", pos=tuple(shoulder),
                joints=[Joint(f"{prefix}yaw", axis=(0, 0, 1), range=rng("yaw"), tag="yaw", servo=servo("yaw"))],
                geoms=[Geom(f"{prefix}pedestal", "cylinder", (ped_r, ped_h / 2), pos=(0.0, 0.0, -ped_h / 2),
                            role="visual", rgba=rgba)],
                masses=[PointMass(f"{prefix}shoulder_drive", _drive_kg(arm, "shoulder"), (0.0, 0.0, 0.0))],
                children=[upper])
    yaw_mass = PointMass(f"{prefix}yaw_drive", _drive_kg(arm, "yaw"), (shoulder[0], shoulder[1], shoulder[2] - ped_h))
    moving = total_mass(arm) - _drive_kg(arm, "yaw")
    return base, yaw_mass, moving


def qpos(arm: ArmSpec, prefix: str, q: dict, grip: float = 0.0) -> dict:
    """Chiron joint positions for pose ``q`` (rad) with the jaws at ``grip`` [rad]."""
    out = {f"{prefix}{j}": float(q.get(j, 0.0)) for j in JOINTS if j in arm.drives and j != "grip"}
    out[f"{prefix}grip_a"] = grip
    out[f"{prefix}grip_b"] = grip
    return out


# ----------------------------------------------------------------------------------------------- the arms
_COMMON_RANGES = {"yaw": (-170.0, 170.0), "shoulder": (-180.0, 30.0), "elbow": (-10.0, 178.0),
                  "wrist_pitch": (-120.0, 120.0), "wrist_yaw": (-90.0, 90.0), "wrist_roll": (-180.0, 180.0),
                  "grip": (-15.0, 45.0)}

#: Pekari-2M's arm (two of them): a light 5-DOF arm on strain-wave modules, 0.97 m reach, 5 kg at 0.5 m.
PEKARI_ARM = ArmSpec(
    name="Pekari-2M light arm",
    upper=0.42, fore=0.38, tool=0.17, upper_section=(0.05, 0.06), fore_section=(0.045, 0.05), wall=0.002,
    pedestal=(0.07, 0.05),
    drives={"yaw": "harmonic 30 Nm, brake", "shoulder": "harmonic 60 Nm, brake", "elbow": "harmonic 30 Nm, brake",
            "wrist_pitch": "harmonic 30 Nm, brake", "wrist_roll": "harmonic 12 Nm, brake", "grip": "smart servo 12 Nm"},
    gripper="parallel-jaw gripper, 80 mm stroke, quick-change flange", gripper_kg=0.55, jaw_length=0.09,
    parts_kg={"brackets, cabling, covers, tool flange": 0.7},
    ranges=dict(_COMMON_RANGES),
    gains={"yaw": (600.0, 15.0, 0.05), "shoulder": (2000.0, 40.0, 0.1), "elbow": (1000.0, 20.0, 0.05),
           "wrist_pitch": (400.0, 6.0, 0.02), "wrist_roll": (100.0, 2.0, 0.01), "grip": (15.0, 0.4, 0.001)},
    poses={"folded": {"shoulder": -172.0, "elbow": 174.0, "wrist_pitch": 20.0},
           "deployed": {"shoulder": -35.0, "elbow": 80.0, "wrist_pitch": 45.0},
           "reach": {"shoulder": 0.0, "elbow": 0.0, "wrist_pitch": 0.0}},
    rated_payload_kg=5.0, rated_reach=0.50,
    assumptions={"tubes": "2 mm Al 6082 rectangular tubes; mass from the tube lengths (no CAD weighing yet)",
                 "drives": "strain-wave cobot modules with brakes (actuator catalogue class values, no product chosen)",
                 "gripper": "a bought parallel gripper on an ISO 9409-1-40 flange; other tools (hook, suction, probe) "
                            "on the same flange, same mass budget"},
)

#: Pumba's main arm: the 6-DOF layout of Calydon's heavy arm one size down — 1.58 m reach, ~45 kg, 30 kg at 1.0 m.
PUMBA_MAIN_ARM = ArmSpec(
    name="Pumba main arm",
    upper=0.70, fore=0.62, tool=0.26, upper_section=(0.09, 0.12), fore_section=(0.075, 0.09), wall=0.004,
    pedestal=(0.12, 0.10),
    drives={"yaw": "cycloidal 400 Nm, brake", "shoulder": "cycloidal 800 Nm, brake", "elbow": "cycloidal 400 Nm, brake",
            "wrist_pitch": "harmonic 150 Nm, brake", "wrist_yaw": "harmonic 60 Nm, brake",
            "wrist_roll": "harmonic 60 Nm, brake", "grip": "jaw screw 6 kN"},
    gripper="two-jaw grab (debris, drums, boxes), quick coupler", gripper_kg=5.0, jaw_length=0.16,
    parts_kg={"brackets, cabling, covers, quick coupler": 3.5},
    ranges={**_COMMON_RANGES, "grip": (-55.0, 45.0)},          # the grab opens 55 deg a jaw: ~0.3 m at the tips
    gains={"yaw": (10000.0, 400.0, 0.5), "shoulder": (40000.0, 1500.0, 1.5), "elbow": (20000.0, 800.0, 0.8),
           "wrist_pitch": (6000.0, 150.0, 0.2), "wrist_yaw": (1500.0, 40.0, 0.1), "wrist_roll": (1500.0, 40.0, 0.1),
           "grip": (200.0, 8.0, 0.1)},          # a firm, not crushing, squeeze: ~35 N m, ~300 N at the jaws
    poses={"folded": {"shoulder": -173.0, "elbow": 176.0, "wrist_pitch": 15.0},
           "deployed": {"shoulder": -35.0, "elbow": 75.0, "wrist_pitch": 50.0},
           "reach": {"shoulder": 0.0, "elbow": 0.0, "wrist_pitch": 0.0}},
    rated_payload_kg=30.0, rated_reach=1.00,
    assumptions={"tubes": "4 mm Al 6082 box sections; deflection and stresses open (as Calydon's)",
                 "drives": "light-industrial cycloidal and strain-wave modules with brakes (catalogue class values)",
                 "rated_payload": "30 kg at 1.0 m (peak, SF 1.5); heavier lifts are Calydon's"},
)

#: Calydon's main arm: a 6-DOF heavy arm (wrist yaw added), 1.9 m reach.
CALYDON_MAIN_ARM = ArmSpec(
    name="Calydon heavy arm",
    upper=0.85, fore=0.75, tool=0.32, upper_section=(0.12, 0.16), fore_section=(0.10, 0.12), wall=0.005,
    pedestal=(0.14, 0.13),
    drives={"yaw": "cycloidal 800 Nm, brake", "shoulder": "cycloidal 1600 Nm, brake", "elbow": "cycloidal 800 Nm, brake",
            "wrist_pitch": "cycloidal 400 Nm, brake", "wrist_yaw": "harmonic 150 Nm, brake",
            "wrist_roll": "harmonic 60 Nm, brake", "grip": "jaw screw 6 kN"},
    gripper="two-jaw heavy grab (debris, beams, drums), quick coupler", gripper_kg=9.0, jaw_length=0.20,
    parts_kg={"brackets, cabling, covers, quick coupler": 6.0},
    ranges=dict(_COMMON_RANGES),
    gains={"yaw": (20000.0, 800.0, 1.0), "shoulder": (80000.0, 3000.0, 3.0), "elbow": (40000.0, 1500.0, 1.5),
           "wrist_pitch": (15000.0, 300.0, 0.3), "wrist_yaw": (3000.0, 60.0, 0.1), "wrist_roll": (3000.0, 60.0, 0.1),
           "grip": (1500.0, 40.0, 0.1)},
    poses={"folded": {"shoulder": -173.0, "elbow": 176.0, "wrist_pitch": 15.0},
           "deployed": {"shoulder": -35.0, "elbow": 75.0, "wrist_pitch": 50.0},
           "reach": {"shoulder": 0.0, "elbow": 0.0, "wrist_pitch": 0.0}},
    rated_payload_kg=60.0, rated_reach=1.20,
    assumptions={"tubes": "5 mm Al 6082 box sections; a steel weldment would be ~2.5x heavier for the same stiffness "
                          "class — the deflection check is open",
                 "drives": "light-industrial cycloidal joint modules with brakes (catalogue class values)",
                 "rated_payload": "60 kg at 1.2 m (peak, SF 1.5) is the design value; rated_load gives the curve"},
)

#: Calydon's second arm: the light arm's layout, stronger (Pekari's arm scaled to 1.2 m).
CALYDON_AUX_ARM = ArmSpec(
    name="Calydon auxiliary arm",
    upper=0.55, fore=0.48, tool=0.20, upper_section=(0.07, 0.09), fore_section=(0.06, 0.07), wall=0.003,
    pedestal=(0.10, 0.08),
    drives={"yaw": "harmonic 60 Nm, brake", "shoulder": "harmonic 150 Nm, brake", "elbow": "harmonic 60 Nm, brake",
            "wrist_pitch": "harmonic 60 Nm, brake", "wrist_roll": "harmonic 12 Nm, brake", "grip": "smart servo 25 Nm"},
    gripper="parallel-jaw gripper, 120 mm stroke, quick-change flange", gripper_kg=1.2, jaw_length=0.11,
    parts_kg={"brackets, cabling, covers, tool flange": 1.5},
    ranges=dict(_COMMON_RANGES),
    gains={"yaw": (3000.0, 80.0, 0.1), "shoulder": (8000.0, 250.0, 0.3), "elbow": (4000.0, 120.0, 0.1),
           "wrist_pitch": (1500.0, 30.0, 0.02), "wrist_roll": (300.0, 6.0, 0.01), "grip": (40.0, 1.0, 0.002)},
    poses={"folded": {"shoulder": -172.0, "elbow": 174.0, "wrist_pitch": 20.0},
           "deployed": {"shoulder": -35.0, "elbow": 80.0, "wrist_pitch": 45.0},
           "reach": {"shoulder": 0.0, "elbow": 0.0, "wrist_pitch": 0.0}},
    rated_payload_kg=8.0, rated_reach=0.70,
    assumptions={"role": "holds, steadies and cuts while the main arm lifts (two-handed handling); same modules "
                         "family as Pekari-2M's arm, one size up"},
)
