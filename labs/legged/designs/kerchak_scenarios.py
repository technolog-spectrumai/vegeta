"""Belisario's scenarios, each a scene (Chiron props) and a pre-programmed sequence of motions to start from - the
reference a learned policy is measured against and may be seeded with:

* **wall** — a climbing wall with rungs: the approach on four, the mount (rearing up to the wall, the hands to the
  first rungs, the feet onto the lowest), the climb (``climb_cycle``: the four limbs moving one at a time to the
  next rungs, then the pelvis up), the top.
* **tree** — the same climb on a trunk, the hands round the trunk, the feet on the branches.
* **hull** — a ship's side: the climb up the sloping plate on its ribs, over the rail, onto the deck.
* **rope** — the rope hanging from a beam: the hands alternating up the rope, the feet clamping it below.
* **window** — the wall with an opening: up onto the feet, the hands to the sill, the pelvis up and over it, down on four beyond.
* **obstacles** — bars across the path: a crouch, a leap (the pelvis on a parabola, the limbs tucked), the landing on four.
* **door** — ``kerchak_tasks.door_sequence``.

Every climb is the same machinery: ``climb_cycle`` moves the contacts between holds given by a scene's ``holds(k)``
(the hold positions for cycle ``k``) with the pelvis riding a frame pitched nose-up to face the wall
(``CLIMB_PITCH``), solved by IK in the world frame by frame (``kerchak_tasks.contact_motion``). A sequence is a
list of segments: ``("gait", seconds, lane)`` for the knuckle walk, or ``("motion", Motion)``; ``sequence(name,
params)`` builds a scenario's. ``sims/space_kerchak_scenes.py`` plays them.
"""
from __future__ import annotations

import math

import numpy as np
from vegeta.chiron import Geom, Joint, Link, Prop

import kerchak_motions as gm
import kerchak_tasks as gt
import humanoid_motions as hum
import motion_kit as mk
import space_unit_robots as sr
import space_unit_specs as su

__all__ = ["SCENARIOS", "sequence", "props", "climb_cycle", "mount", "bar", "point", "hand_targets", "foot_targets", "CLIMB_PITCH", "WALL", "TREE", "HULL", "ROPE", "WINDOW", "OBSTACLES"]

GLASS = (0.62, 0.82, 0.9, 1.0)
GREY, WOOD, BARK, STEEL, ROPE_C = (0.45, 0.45, 0.47, 1.0), (0.55, 0.42, 0.28, 1.0), (0.4, 0.3, 0.2, 1.0), (0.5, 0.52, 0.56, 1.0), (0.75, 0.65, 0.45, 1.0)
CLIMB_PITCH = -0.15                                                       #: the pelvis on a wall [rad]: the trunk along it, the shoulders leaning out a little
CLIMB_BACK, CLIMB_UP = 0.45, 0.55                                         #: the pelvis off the surface, and above the feet's holds [m] (Kerchak's 0.7 m legs)


def climb_up():
    """The pelvis above the feet's holds on a climb: higher for the upright frame's longer legs."""
    return 0.66 * gm.scale() if gm.STANCE["upright"] else CLIMB_UP      # standing: the trailing leg must still reach its rung as the pelvis rises


def climb_back():
    return 0.5 * gm.scale() if gm.STANCE["upright"] else CLIMB_BACK      # standing, well out: the legs nearly straight, the knees behind the rungs


def climb_ref(kin, q_quad):
    """The pose a climb's IK is regularised to: Kerchak's quad pose; standing, a crouch with the knees well bent (a
    straight knee is the wrong branch of the leg's IK, and the regularisation would pull towards it)."""
    if not gm.STANCE["upright"]:
        return q_quad
    q = np.asarray(q_quad, dtype=float).copy()
    for side in ("L", "R"):
        q[kin.index[f"{side}_hip_pitch"]], q[kin.index[f"{side}_knee"]], q[kin.index[f"{side}_ankle_pitch"]] = -0.9, 1.5, -0.6
    return q


def foot_tip():
    """A foot on a rung: toes up this far from the horizontal (the standing frame's ankle flexes less with its shin)."""
    return 0.4 if gm.STANCE["upright"] else FOOT_TIP


def walk_speed():
    """The approach's speed: Kerchak's knuckle walk, or the humanoid's walk (``sims/space_kerchak_scenes`` plays the same gaits)."""
    return 1.0 if gm.STANCE["upright"] else 0.25


def stop_off():
    """The walk stops this far short of the surface's holds: the upright frame closer, its arms shorter for its height."""
    return 0.6 * gm.scale() if gm.STANCE["upright"] else 0.75
LEDGE_WRIST = (-0.55, 0.0, 0.835)                                         #: a ledge taken from above: the wrist above and behind the edge, the fist closing down over it
CLIMB_SPINE = -0.1                                                        #: the spine's pitch modules on a wall: a little extended, the trunk along it
WALL = {"x": 2.6, "width": 1.6, "height": 4.5, "rung_spacing": 0.35, "first_rung": 0.2, "rung_out": 0.08, "rung_radius": 0.025, "cycles": 6}
TREE = {"x": 2.8, "radius": 0.3, "height": 6.0, "first_branch": 0.35, "branch_spacing": 0.45, "branch_length": 1.2, "branch_radius": 0.07, "cycles": 6}
HULL = {"x": 2.4, "slope_deg": 62.0, "height": 2.4, "rib_spacing": 0.4, "rib_out": 0.06, "cycles": 5}
ROPE = {"x": 2.2, "length": 4.0, "links": 12, "radius": 0.025, "top_height": 5.0, "cycles": 4}
WINDOW = {"x": 3.0, "sill": 0.9, "width": 1.2, "height": 1.3, "wall_thickness": 0.25}
OBSTACLES = {"x": 2.6, "count": 2, "height": 0.35, "spacing": 2.2, "bar_radius": 0.05}


SLIPPERY = (0.15, 0.005, 0.0001)                                          #: a smooth wall's friction: nothing to push a foot against


def _box(name, half, pos, rgba=GREY, mass=100.0, quat=(1.0, 0.0, 0.0, 0.0), friction=None):
    return Geom(name, "box", tuple(half), pos=tuple(pos), quat=tuple(quat), mass=mass, role="link", rgba=rgba, **({} if friction is None else {"friction": friction}))


def _cap(name, r, a, b, rgba=BARK, mass=20.0):
    return Geom(name, "capsule", (r,), fromto=tuple(a) + tuple(b), mass=mass, role="link", rgba=rgba)


# ---- the scenes
def wall_prop(p):
    x, w, h, sp, z0, out, rr = p["x"], p["width"], p["height"], p["rung_spacing"], p["first_rung"], p["rung_out"], p["rung_radius"]
    geoms = [_box("wall_face", (0.1, w / 2, h / 2), (x + 0.1, 0.0, h / 2), mass=2000.0)]
    z, k = z0, 0
    while z < h - 0.2:
        geoms.append(_cap(f"rung_{k}", rr, (x - out, -w / 2 + 0.1, z), (x - out, w / 2 - 0.1, z), rgba=STEEL, mass=3.0))
        z += sp
        k += 1
    geoms.append(_box("wall_top", (0.6, w / 2 + 0.2, 0.06), (x + 0.6, 0.0, h - 0.06), mass=500.0))
    return Prop(Link("wall", pos=(0.0, 0.0, 0.0), geoms=geoms), free=False)


def tree_prop(p):
    x, r, h, bl, br = p["x"], p["radius"], p["height"], p["branch_length"], p["branch_radius"]
    geoms = [_cap("trunk", r, (x, 0.0, 0.0), (x, 0.0, h), rgba=BARK, mass=800.0)]
    z, k = p["first_branch"], 0
    while z < h - 0.5:                                                     # a branch each side at every level: the feet's steps
        for side, tag in ((1.0, "l"), (-1.0, "r")):
            geoms.append(_cap(f"branch_{k}{tag}", br, (x, side * r * 0.8, z), (x, side * (r + bl), z + 0.12), rgba=BARK, mass=15.0))
        z += p["branch_spacing"]
        k += 1
    return Prop(Link("tree", pos=(0.0, 0.0, 0.0), geoms=geoms), free=False)


def hull_prop(p):
    x, slope, height = p["x"], math.radians(p["slope_deg"]), p["height"]
    length = height / math.sin(slope)
    a = -slope                                                             # the plate's long axis up the slope (it stands on the ground)
    quat = (math.cos(a / 2), 0.0, math.sin(a / 2), 0.0)
    cx, cz = x + 0.5 * length * math.cos(slope), 0.5 * height
    geoms = [_box("hull_plate", (length / 2, 2.5, 0.06), (cx, 0.0, cz), rgba=STEEL, mass=3000.0, quat=quat)]
    s, k = 0.3, 0
    while s < length - 0.2:                                                # the ribs: bars across the plate, hand- and foot-holds
        px, pz = x + s * math.cos(slope) - p["rib_out"] * math.sin(slope), s * math.sin(slope) + p["rib_out"] * math.cos(slope)
        geoms.append(_cap(f"rib_{k}", 0.03, (px, -2.0, pz), (px, 2.0, pz), rgba=STEEL, mass=5.0))
        s += p["rib_spacing"]
        k += 1
    top_x = x + length * math.cos(slope)
    geoms.append(_cap("hull_rail", 0.05, (top_x, -2.5, height + 0.1), (top_x, 2.5, height + 0.1), rgba=STEEL, mass=50.0))
    geoms.append(_box("deck", (1.5, 2.5, 0.06), (top_x + 1.5, 0.0, height - 0.06), rgba=STEEL, mass=3000.0))
    return Prop(Link("hull", pos=(0.0, 0.0, 0.0), geoms=geoms), free=False)


def rope_prop(p):
    x, length, links, r, top = p["x"], p["length"], int(p["links"]), p["radius"], p["top_height"]
    seg = length / links
    beam = Link("rope_beam", pos=(x, 0.0, top), geoms=[_cap("beam", 0.06, (0.0, -1.5, 0.0), (0.0, 1.5, 0.0), rgba=STEEL, mass=100.0)])
    parent = beam
    for k in range(links):
        link = Link(f"rope_{k}", pos=(0.0, 0.0, 0.0 if k == 0 else -seg),
                    joints=[Joint(f"rope_{k}_x", axis=(1, 0, 0), range=(-1.2, 1.2), damping=0.3, stiffness=0.5),
                            Joint(f"rope_{k}_y", axis=(0, 1, 0), range=(-1.2, 1.2), damping=0.3, stiffness=0.5)],
                    geoms=[_cap(f"rope_seg_{k}", r, (0.0, 0.0, 0.0), (0.0, 0.0, -seg), rgba=ROPE_C, mass=0.8)])
        parent.children.append(link)
        parent = link
    return Prop(beam, free=False, solref=(-2000.0, -40.0))


def window_prop(p):
    x, sill, w, h, t = p["x"], p["sill"], p["width"], p["height"], p["wall_thickness"]
    wall_h, wall_w = 3.0, 3.0
    frame = Link("window_wall", pos=(x, 0.0, 0.0),                       # a smooth, solid wall: nothing for a foot
                 geoms=[_box("wall_below", (t / 2, wall_w, sill / 2), (0.0, 0.0, sill / 2), mass=200.0, friction=SLIPPERY),
                        _box("wall_above", (t / 2, wall_w, (wall_h - sill - h) / 2), (0.0, 0.0, sill + h + (wall_h - sill - h) / 2), mass=200.0, friction=SLIPPERY),
                        _box("wall_left", (t / 2, (wall_w - w / 2) / 2, h / 2), (0.0, w / 2 + (wall_w - w / 2) / 2, sill + h / 2), mass=200.0, friction=SLIPPERY),
                        _box("wall_right", (t / 2, (wall_w - w / 2) / 2, h / 2), (0.0, -w / 2 - (wall_w - w / 2) / 2, sill + h / 2), mass=200.0, friction=SLIPPERY),
                        _box("sill_plate", (t / 2 + 0.04, w / 2, 0.03), (0.0, 0.0, sill + 0.03), rgba=WOOD, mass=20.0)]
                 + ([_box("glass", (0.004, w / 2 - 0.01, h / 2 - 0.01), (0.0, 0.0, sill + h / 2), rgba=GLASS, mass=5.0)] if p.get("glass") else [])
                 + _table_with_banana(p.get("table_x", 3.4), p.get("table_y", 0.0)))
    return Prop(frame, free=False)


HAMMER = {"handle": 0.32, "handle_r": 0.018, "head": (0.055, 0.025, 0.03), "rest_z": 0.028}
HAMMER_LYING = (0.70710678, -0.70710678, 0.0, 0.0)                        #: the handle along the world's y, flat on the ground


def hammer_prop(p):
    """A hammer, loose on the ground: its frame at the handle's root, the handle up its z, the head across its x
    at the top; lying with the handle along y, its middle at (``hammer_x``, ``hammer_y``)."""
    H = HAMMER
    hx, hy = p["hammer_x"], p["hammer_y"]
    steel, wood = (0.35, 0.36, 0.4, 1.0), (0.6, 0.45, 0.28, 1.0)
    link = Link("hammer", pos=(hx, hy - H["handle"] / 2, H["rest_z"]), quat=HAMMER_LYING,
                geoms=[_cap("hammer_handle", H["handle_r"], (0.0, 0.0, 0.0), (0.0, 0.0, H["handle"]), rgba=wood, mass=0.6),
                       _box("hammer_head", H["head"], (0.0, 0.0, H["handle"] - 0.01), rgba=steel, mass=2.5)])
    return Prop(link, free=True)


def _table_with_banana(tx, ty, top=0.75, depth=0.8, width=1.2):
    """A table in the room (``tx`` beyond the wall, ``ty`` across) with a banana on it."""
    geoms = [_box("table_top", (depth / 2, width / 2, 0.02), (tx, ty, top - 0.02), rgba=WOOD, mass=15.0)]
    for i, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        geoms.append(_box(f"table_leg_{i}", (0.03, 0.03, (top - 0.04) / 2), (tx + sx * (depth / 2 - 0.05), ty + sy * (width / 2 - 0.05), (top - 0.04) / 2), rgba=WOOD, mass=3.0))
    yellow = (0.95, 0.82, 0.2, 1.0)
    bx, by, bz = tx - 0.15, ty + 0.1, top + 0.03                           # the banana: three capsules on a gentle curve, lying on the top
    pts = [(bx - 0.1, by - 0.03, bz), (bx - 0.03, by + 0.0, bz + 0.005), (bx + 0.04, by + 0.0, bz + 0.005), (bx + 0.11, by - 0.03, bz)]
    for i in range(3):
        geoms.append(_cap(f"banana_{i}", 0.022 if i == 1 else 0.018, pts[i], pts[i + 1], rgba=yellow, mass=0.2))
    return geoms


def obstacles_prop(p):
    n, h, sp, br, x0 = int(p["count"]), p["height"], p["spacing"], p["bar_radius"], p["x"]
    geoms = []
    for k in range(n):
        x = x0 + k * sp
        geoms.append(_cap(f"bar_{k}", br, (x, -1.5, h), (x, 1.5, h), rgba=STEEL, mass=20.0))
        for sy in (-1.5, 1.5):
            geoms.append(_box(f"post_{k}_{'l' if sy > 0 else 'r'}", (0.04, 0.04, h / 2), (x, sy, h / 2), rgba=STEEL, mass=10.0))
    return Prop(Link("obstacles", pos=(0.0, 0.0, 0.0), geoms=geoms), free=False)


def params_for(name, params=None):
    """A scenario's parameters: the defaults, the stance's adjustments (the upright frame strikes at arm's length, so
    its hammer lies further from the wall), the caller's overrides."""
    p = dict(DEFAULTS[name])
    if gm.STANCE["upright"] and name == "window_hammer":
        p["hammer_x"] = p["x"] - 0.75 * gm.scale()
    p.update(params or {})
    return p


COMPOSITE = {}                                                            #: {scenario: fn(params) -> props}: scenes made of several props (the gibbon's course)


def props(name, params=None):
    """The scene's props for a scenario (the door's from ``kerchak_tasks``)."""
    p = params_for(name, params)
    if name in COMPOSITE:
        return COMPOSITE[name](p)
    if name == "door":
        return (gt.door_prop(),)
    if name == "window_hammer":
        return (window_prop(p), hammer_prop(p))
    return (BUILDERS[name](p),)


# ---- the climb
def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / max(np.linalg.norm(v), 1e-9)


def _world_pts(kin, q, base, pitch, pts):
    """[(body, local)] → world points with the pelvis at ``base`` pitched ``pitch``."""
    R, b = gm.base_rot(pitch), np.asarray(base, dtype=float)
    kin.set(q)
    return [R @ kin.world(body, local) + b for body, local in pts]


def _contacts(kin, q, base, pitch):
    return {g: _world_pts(kin, q, base, pitch, plist) for g, plist in gm._contact_points(kin).items()}


# ---- holds: a bar the hand closes on and the foot stands on, or a plain contact point
CURL_CLOSED, CURL_OPEN = 2.0, -0.3        #: the finger module closed on a bar (a fist), and open to take one [rad]
GRIP = sr.hand_grip(CURL_CLOSED)          #: a bar's axis in the hand's frame once the fingers have closed on it [m]: in front of the palm's face, inside the fist


def set_hand(P):
    """The grip from a frame's own hand (``space_unit_specs`` proportions): Kerchak's, the humanoid's or the gibbon's."""
    global GRIP
    GRIP = sr.hand_grip(CURL_CLOSED, P=P)
GRIP_Y = 0.06                             #: a second point along the hand's y (the bar's axis) that fixes the hand's roll
BAR_R = 0.025                             #: the bar the grip is sized for; a thicker hold moves the palm out to its surface
SOLE_ON_BAR = 0.08                        #: the sole's point that rests on a bar: the ball of the foot, 2 cm of toe past the bar [m] (on four)


def sole_on_bar():
    """Where along the sole a bar sits: standing, further back under the foot, the ankle and the shin clear of the rung above."""
    return 0.11 if gm.STANCE["upright"] else SOLE_ON_BAR
SOLE_HALF = 0.015                         #: (Kerchak's early value) the contact points above the sole's bottom face; ``sole_drop`` reads the frame's own [m]


def sole_drop(kin):
    """The foot's contact points (``humanoid_motions.sole_points``) are on the sole's bottom face: a hair above a hold
    puts the foot on it, not in it [m]."""
    return 0.004
FOOT_TIP = 0.5                            #: the foot on a bar: toes up, into the surface [rad]
SWING_FREE = 0.8                          #: through this much of a step onto a bar only the ball of the foot is placed, the foot's tilt free
ORDER = ["L", "R", "L_hand", "R_hand"]    #: a climb cycle's order: the feet step up first (the legs push), then the hands reach
IK_REG, IK_ITERS = 3e-3, 900              #: the climbs' IK: well damped (a grip is four points per hand; a lighter damping jumps between solutions)


def bar(centre, axis=(0.0, 1.0, 0.0), n=(-1.0, 0.0, 0.0), r=BAR_R, wrist_dir=None, tip=None):
    """A hold on a bar at ``centre`` along ``axis``, of radius ``r``, on a surface whose outward normal is ``n``.
    ``wrist_dir``: where the wrist sits from the grip (unit; default a rung's: below and behind). ``tip``: a foot on
    it points its toes up this far from the horizontal."""
    return {"kind": "bar", "c": np.asarray(centre, dtype=float), "axis": _unit(axis), "n": _unit(n), "r": float(r), "sign": 1.0, "phi": 0.0,
            "wrist_dir": None if wrist_dir is None else _unit(wrist_dir), "tip": float(foot_tip() if tip is None else tip)}


def point(centre, n=(-1.0, 0.0, 0.0), pitch=None, over=False):
    """A plain contact at ``centre``: the knuckle on it, or the sole's three points flat on the surface there (the
    foot's long axis along the surface's up in the climbing frame ``pitch``). ``over``: a foot steps onto it over an
    edge (a sill's top), by the lift-over path."""
    return {"kind": "point", "c": np.asarray(centre, dtype=float), "n": _unit(n), "pitch": CLIMB_PITCH if pitch is None else pitch, "over": over}


def _up_along(n):
    z = np.array([0.0, 0.0, 1.0])
    t = z - (z @ n) * n
    return _unit(t) if np.linalg.norm(t) > 1e-6 else np.array([1.0, 0.0, 0.0])


WRIST_ANGLE = 0.6                         #: on a rung the wrist hangs below the grip, this far back from straight below [rad]


def _rot(axis, a):
    K = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(a) * K + (1.0 - math.cos(a)) * K @ K


def hand_frame_on(hold, side="L"):
    """The hand's rotation (columns x, y, z) closed on a bar, y along the bar's axis. A rung (a horizontal bar on a
    surface): the wrist hangs ``WRIST_ANGLE`` back from straight below the grip, the palm's face up under the bar,
    the fingers hooked over its far side - the frame follows from where the wrist is. A vertical bar (a rope, a
    trunk's flank): the palm's face to it turned ``phi`` about the bar, the fingers round it, the wrist out to the
    hand's own side (``sign`` swaps the sides)."""
    axis, n, t = hold["axis"], hold["n"], _up_along(hold["n"])
    if abs(axis @ np.array([0.0, 0.0, 1.0])) < 0.5:
        y = axis
        wd = hold.get("wrist_dir")
        e1 = _unit(t * math.cos(WRIST_ANGLE) - n * math.sin(WRIST_ANGLE)) if wd is None else -wd   # the wrist to the grip: up and into the surface
        e1 = _unit(e1 - (e1 @ y) * y)
        gamma = math.atan2(GRIP[2], GRIP[0])                                     # the grip's bearing in the hand's xz plane
        best = None
        for g in (gamma, -gamma):
            x = _rot(y, g) @ e1
            R = np.stack([x, y, np.cross(x, y)], axis=1)
            score = float((R @ GRIP) @ e1)
            if best is None or score > best[0]:
                best = (score, R)
        return best[1]
    y = axis * hold["sign"] * (-1.0 if side == "L" else 1.0)
    x = _rot(y, hold["phi"] * (1.0 if side == "L" else -1.0)) @ (-n)
    x = _unit(x - (x @ y) * y)
    return np.stack([x, y, np.cross(x, y)], axis=1)


def hand_targets(kin, side, hold, centre=None):
    """The hand on a hold. A bar: three points - the grip centre on the bar's axis (the palm's end against the bar,
    the fingers round it), the wrist back along the approach, a point along the hand's y on the axis (the roll); the
    fingers close in ``extra``. A point: the knuckle on it."""
    c = hold["c"] if centre is None else np.asarray(centre, dtype=float)
    if hold["kind"] == "point":
        body, local = gm._contact_points(kin)[f"{side}_knuckle"][0]
        return [(body, local, c)]
    R = hand_frame_on(hold, side)
    c = c - max(hold["r"] - BAR_R, 0.0) * R[:, 0]                             # a thick hold: the palm's face out at its surface
    hand = kin.body(f"{side}_hand")
    return [(hand, l, c + R @ (l - GRIP)) for l in (GRIP, np.zeros(3), GRIP + np.array([0.0, GRIP_Y, 0.0]))]


def foot_targets(kin, side, hold, centre=None, tip=None):
    """The foot on a hold. A bar: the sole's three points with the sole's ``SOLE_ON_BAR`` point on the bar's outer
    top, the foot pointing into the surface, toes up ``tip`` (default the hold's) from the horizontal. A point: the
    sole flat on the surface."""
    c = hold["c"] if centre is None else np.asarray(centre, dtype=float)
    tip = hold.get("tip", FOOT_TIP) if tip is None else tip
    pts = gm._contact_points(kin)[side]
    drop = sole_drop(kin)
    if hold["kind"] == "point":
        up = gm.base_rot(hold["pitch"]) @ np.array([0.0, 0.0, 1.0])
        return [(body, local, c + _sole_offset(k, hold["pitch"]) + drop * up) for k, (body, local) in enumerate(pts)]
    n, t = hold["n"], _up_along(hold["n"])
    h = -n - (-n @ np.array([0.0, 0.0, 1.0])) * np.array([0.0, 0.0, 1.0])   # into the surface, horizontally
    f = math.cos(tip) * _unit(h) + math.sin(tip) * np.array([0.0, 0.0, 1.0])
    up = np.cross(f, hold["axis"])
    up = up if up @ np.array([0.0, 0.0, 1.0]) >= 0 else -up
    top = c + hold["r"] * _unit(n + t) + drop * up                             # the sole's bottom face on the bar's top
    return [(body, local, top + (o[0] - sole_on_bar()) * f + o[1] * hold["axis"]) for (body, local), o in zip(pts, hum.SOLE_POINTS)]


def _curl(u, a, b, start=CURL_CLOSED):
    """The finger module through a swing window (a, b): ``start`` before it, opening over its first fifth, closing
    again over its last quarter, closed after."""
    if u <= a:
        return start
    if u >= b:
        return CURL_CLOSED
    f = (u - a) / (b - a)
    if f < 0.2:
        return start + (CURL_OPEN - start) * mk.smooth(f / 0.2)
    if f > 0.75:
        return CURL_OPEN + (CURL_CLOSED - CURL_OPEN) * mk.smooth((f - 0.75) / 0.25)
    return CURL_OPEN


def _fingers_extra(kin, windows, holds, q_from, from_start=False):
    """``extra`` for a contact motion: the fingers closed on bar holds, opening through the hands' swing windows
    (from the curl they came with when ``from_start``); a hand on a point hold keeps its curl."""
    def fn(u, q):
        for side in ("L", "R"):
            g = f"{side}_hand"
            if holds[g]["kind"] == "bar":
                start = su.KNUCKLE_CURL if from_start else CURL_CLOSED
                sr.set_curl(q, kin, side, _curl(u, *windows[g], start=start) if g in windows else CURL_CLOSED)
            else:
                for j in sr.finger_joints(side):
                    q[kin.index[j]] = q_from[kin.index[j]]
        return q
    return fn


def _extras(*fns):
    def fn(u, q):
        for f in fns:
            q = f(u, q)
        return q
    return fn


def foot_swing(h0, h1, f, lift=0.07, clear=0.06):
    """A foot's path from hold ``h0`` to ``h1`` at ``f`` ∈ [0, 1]: a solid foot cannot come up through a bar, so it
    lifts out and up to above and outside the next bar, comes in over it, and sets down on it (a point hold: a plain arc)."""
    c0, c1 = h0["c"], h1["c"]
    if h1["kind"] != "bar" and not h1.get("over"):
        return (1 - f) * c0 + f * c1 + math.sin(math.pi * f) * (0.1 * h0["n"] + np.array([0.0, 0.0, 0.1]))
    n, t = h1["n"], _up_along(h1["n"])
    Q = c1 + clear * t + lift * n                                          # above and outside the next bar
    R = c1 + clear * t                                                     # above it
    if f < 0.6:
        s_ = mk.smooth(f / 0.6)
        return (1 - s_) * c0 + s_ * Q + 0.05 * math.sin(math.pi * s_) * n
    if f < 0.85:
        s_ = mk.smooth((f - 0.6) / 0.25)
        return (1 - s_) * Q + s_ * R
    s_ = mk.smooth((f - 0.85) / 0.15)
    return (1 - s_) * R + s_ * c1


def climb_cycle(kin, q_from, holds, base_from, base_to, *, pitch=CLIMB_PITCH, duration=3.2, lift=None, name="climb", tail_fn=None):
    """One cycle up: the contacts move one at a time (``ORDER``: left foot, right foot, left hand, right hand) from
    ``holds[0]`` to ``holds[1]`` ({contact: hold}) on an arc out from the surface - the fingers opening through the
    swing and closing on the bar - while the pelvis rides from ``base_from`` to ``base_to`` (xyz) in the climbing frame."""
    if lift is None:
        sloped = abs(float(holds[1]["L"]["n"] @ np.array([0.0, 0.0, 1.0]))) > 0.3
        lift = 0.25 * gm.scale() if gm.STANCE["upright"] else (0.1 if sloped else 0.12)   # the foot swings well out: the knee clear of the rung above, the shin off a sloped plate
    order = ["L_hand", "R_hand", "L", "R"] if gm.STANCE["upright"] else ORDER   # standing: the hands reach up first, then the legs push
    if tail_fn is None:
        windows = {g: (0.05 + 0.2 * k, 0.05 + 0.2 * (k + 1)) for k, g in enumerate(order)}
    else:                                                                  # a tail too (``tail_fn(u, base)``): the limbs in less of the cycle, the tail moves last
        windows = {g: (0.03 + 0.17 * k, 0.03 + 0.17 * (k + 1)) for k, g in enumerate(order)}

    def targets(u):
        out = [] if tail_fn is None else list(tail_fn(u, base(u)))
        for g in order:
            a, b = windows[g]
            h0, h1 = holds[0][g], holds[1][g]
            if u <= a:
                c, h = h0["c"], h0
            elif u >= b:
                c, h = h1["c"], h1
            elif g.endswith("hand"):
                f = mk.smooth((u - a) / (b - a))
                c, h = (1 - f) * h0["c"] + f * h1["c"] + math.sin(math.pi * f) * lift * h0["n"], h1
            else:
                f = (u - a) / (b - a)
                c, h = foot_swing(h0, h1, f), h1
                if f < SWING_FREE and h1["kind"] == "bar":                  # lifted: the ball of the foot alone, the tilt free (the ankle's range)
                    out.append(foot_targets(kin, g[0], h, c)[0])
                    continue
            out += hand_targets(kin, g[0], h, c) if g.endswith("hand") else foot_targets(kin, g[0], h, c)
        return out

    def base(u):
        if gm.is_gibbon():                                                 # the gibbon's short legs: up only once the trailing foot is off its rung
            f = mk.smooth(max(0.0, min(1.0, (u - 0.62) / 0.33)))
        else:
            f = mk.smooth(u) if not gm.STANCE["upright"] else mk.smooth(max(0.0, min(1.0, (u - 0.45) / 0.45)))   # the pelvis rides up through the cycle; standing, as the legs step
        b = (1 - f) * np.asarray(base_from) + f * np.asarray(base_to)
        return b[0], b[1], b[2], pitch

    extra = _extras(_spine_extra(kin, CLIMB_SPINE), _fingers_extra(kin, windows, holds[1], q_from))
    return gt.contact_motion(kin, name, duration, targets, base, q_init=q_from, extra=extra, reg=IK_REG, iters=IK_ITERS,
                             source="kerchak_scenarios.climb_cycle", q_ref=climb_ref(kin, q_from))


def _spine_extra(kin, value, q_from=None, blend=False):
    """``extra`` for a contact motion: the spine's pitch modules held at ``value`` (or eased there from ``q_from``)."""
    js = ("spine_lumbar_pitch", "spine_thoracic_pitch")

    def fn(u, q):
        for j in js:
            v0 = q_from[kin.index[j]] if (blend and q_from is not None) else value
            q[kin.index[j]] = (1 - mk.smooth(u)) * v0 + mk.smooth(u) * value if blend else value
        return q
    return fn


def _sole_offset(k, pitch):
    """The sole's three points round the foot's hold: the foot's long axis along the wall's up in the climbing frame."""
    o = np.asarray(hum.SOLE_POINTS[k])
    R = gm.base_rot(pitch)
    return R @ np.array([o[0], o[1], 0.0])


def mount(kin, q_quad, base_x, holds, base_climb, *, pitch=CLIMB_PITCH, duration=3.5, name="mount", tail_fn=None):
    """From the quad pose at the foot of the surface up onto it: the pelvis rises and pitches to the climbing frame
    where it stands while the hands go from the ground to their first holds (the fingers opening on the way, closing
    on the bars; the feet planted), then the feet step up to theirs one after the other as the pelvis moves in."""
    b0 = np.array([base_x, 0.0, gm.STANCE["h"]])
    start = _contacts(kin, q_quad, b0, gm.STANCE["pitch"])
    pts = gm._contact_points(kin)
    hands = {}
    for side in ("L", "R"):
        end = hand_targets(kin, side, holds[f"{side}_hand"])
        hands[side] = (_world_pts(kin, q_quad, b0, gm.STANCE["pitch"], [(b, l) for b, l, _ in end]), end)
    bc = np.asarray(base_climb, dtype=float)
    sloped = abs(float(holds["L"]["n"] @ np.array([0.0, 0.0, 1.0]))) > 0.3
    # a vertical surface: the pelvis rises on the planted feet while the hands reach, moves in as the feet step up;
    # a slope (its holds ahead): the pelvis moves in low while the hands reach, the feet step, then it rises
    hands_end = 0.5 if not sloped else 0.35
    foot_windows = {"L": (0.5, 0.75), "R": (0.75, 1.0)} if not sloped else {"L": (0.35, 0.55), "R": (0.55, 0.75)}
    reach = 0.7 * hands_end                                                # the hands' grip points on their arcs until here, then the whole frame settles on the bar

    def targets(u):
        out = [] if tail_fn is None else list(tail_fn(u, base(u)))
        fh = mk.smooth(min(1.0, u / reach))                                  # the hands first
        for side in ("L", "R"):
            s, end = hands[side]
            if u < reach or end[0] is end[-1]:
                h_ = holds[f"{side}_hand"]                                   # up, and out from the surface: clear of the holds below the one it takes
                up_, out_ = (0.08, 0.18) if gm.STANCE["upright"] else (0.3, 0.12)   # standing: out round the hold, not up over it and back down through it
                arc = np.array([0.0, 0.0, up_ * math.sin(math.pi * fh)]) + (out_ * math.sin(math.pi * fh) * h_["n"] if h_["kind"] == "bar" else 0.0)
                out.append((end[0][0], end[0][1], (1 - fh) * s[0] + fh * end[0][2] + arc))
            else:
                out += end
            a, b_ = foot_windows[side]                                       # then the feet, one after the other
            ff = 0.0 if u <= a else 1.0 if u >= b_ else (u - a) / (b_ - a)
            h = holds[side]
            if ff <= 0.0:
                out += [(b, l, sk) for (b, l), sk in zip(pts[side], start[side])]
            elif ff >= 1.0:
                out += foot_targets(kin, side, h)
            else:
                ground = point(np.mean(start[side], axis=0), pitch=0.0)         # the foot's hold on the ground
                c = foot_swing(ground, h, ff)
                goal = foot_targets(kin, side, h, c)
                if ff < SWING_FREE and h["kind"] == "bar":
                    out.append(goal[0])                                          # lifted: the ball of the foot alone
                else:
                    out += goal
        return out

    def base(u):
        f_a = mk.smooth(min(1.0, u / hands_end))                             # the hands' phase
        f_b = mk.smooth(max(0.0, (u - foot_windows["L"][0]) / (1.0 - foot_windows["L"][0])))   # the feet's phase
        f_c = mk.smooth(max(0.0, (u - foot_windows["R"][0]) / (1.0 - foot_windows["R"][0])))   # after the feet
        if sloped and gm.STANCE["upright"]:                                # standing: a squat forward to reach the holds, then up as the feet step
            mid = b0 + gm.scale() * np.array([0.15, 0.0, -0.2])
            bz = b0 + f_a * (mid - b0) + f_c * (bc - mid)
            return bz[0], bz[1], bz[2], gm.STANCE["pitch"] + f_a * (pitch - gm.STANCE["pitch"])
        if sloped:
            f_xy, f_z = 0.8 * f_a + 0.2 * f_c, f_c
            z = (1 - f_z) * b0[2] + f_z * bc[2]
        else:                                                              # up on the planted feet as far as the legs stand, the rest as they step
            f_xy = f_a if gm.STANCE["upright"] else f_b                      # standing, the pelvis comes in over the feet as the hands reach
            z_stand = min(bc[2], hum.base_height(kin) - 0.03)
            z = b0[2] + f_a * (z_stand - b0[2]) + f_b * (bc[2] - z_stand)
        xy = (1 - f_xy) * b0[:2] + f_xy * bc[:2]
        return xy[0], xy[1], z, gm.STANCE["pitch"] + f_a * (pitch - gm.STANCE["pitch"])

    windows = {"L_hand": (0.0, hands_end), "R_hand": (0.0, hands_end)}
    extra = _extras(_spine_extra(kin, CLIMB_SPINE, q_quad, blend=True), _fingers_extra(kin, windows, holds, q_quad, from_start=True))
    return gt.contact_motion(kin, name, duration, targets, base, q_init=q_quad, extra=extra, reg=IK_REG, iters=IK_ITERS,
                             source="kerchak_scenarios.mount", q_ref=climb_ref(kin, q_quad))


VERTICAL_TRIALS = tuple((s, phi) for s in (1.0, -1.0) for phi in (0.0, 0.4, -0.4, 0.8, -0.8, 1.2, -1.2, 1.5, -1.5))


def _pick_frames(kin, q_quad, holds, base, pitch, trials):
    """{hand: (sign, phi)} for the hands' frames on vertical bars: each hand's candidates solved statically at its
    hold (the feet on theirs), the one with the smallest IK error kept."""
    limbs = [j for j in kin.actuated if "spine" not in j and j != "waist_yaw" and "finger" not in j and "thumb" not in j]
    R, b = gm.base_rot(pitch), np.asarray(base, dtype=float)
    q = _spine_extra(kin, CLIMB_SPINE)(1.0, np.asarray(q_quad, dtype=float).copy())
    feet = foot_targets(kin, "L", holds["L"]) + foot_targets(kin, "R", holds["R"])
    to_base = lambda ts: [(bd, l, R.T @ (np.asarray(w) - b)) for bd, l, w in ts]
    q_feet, _ = kin.ik(to_base(feet), q_init=climb_ref(kin, q), q_ref=climb_ref(kin, q_quad), joints=limbs, reg=3e-4, iters=400)
    out = {}
    for side in ("L", "R"):
        g, best = f"{side}_hand", None
        h = dict(holds[g])
        for s, phi in trials:
            h["sign"], h["phi"] = s, phi
            _, err = kin.ik(to_base(hand_targets(kin, side, h) + feet), q_init=q_feet, q_ref=q_quad, joints=limbs, reg=3e-4, iters=400)
            if best is None or err < best[0]:
                best = (err, s, phi)
        out[g] = best[1:]
    return out


def _climb_sequence(kin, q_quad, hold, base, cycles, *, stop_x, pitch=CLIMB_PITCH, v=None, trials=((1.0, 0.0),), tail=None, duration=None):
    """The approach to ``stop_x``, the mount and ``cycles`` climb cycles: ``hold(k)`` → {contact: hold} and
    ``base(k)`` → the pelvis's xyz for cycle ``k``. ``trials``: (sign, phi) candidates for each hand's frame on a
    vertical bar (``_pick_frames``)."""
    v = walk_speed() if v is None else v
    segs = [("gait", max(stop_x, 0.0) / v, 0.0)]
    frames = _pick_frames(kin, q_quad, hold(0), base(0), pitch, trials) if len(trials) > 1 else {"L_hand": trials[0], "R_hand": trials[0]}

    def framed(h):
        for g in ("L_hand", "R_hand"):
            h[g]["sign"], h[g]["phi"] = frames[g]
        return h

    m = mount(kin, q_quad, stop_x, framed(hold(0)), base(0), pitch=pitch, tail_fn=None if tail is None else tail(q_quad, None))
    m.params = {**(m.params or {}), "frames": {g: list(v) for g, v in frames.items()}}
    segs.append(("motion", m))
    q = gt._last(kin, m, q_quad)
    for k in range(cycles):
        hk, hk1 = framed(hold(k)), framed(hold(k + 1))
        c = climb_cycle(kin, q, (hk, hk1), base(k), base(k + 1), pitch=pitch, name=f"climb_{k + 1}",
                        tail_fn=None if tail is None else tail(q, k), **({} if duration is None else {"duration": duration}))
        segs.append(("motion", c))
        q = gt._last(kin, c, q)
    return segs


def scenario_wall(kin, q_quad, p):
    """The rungs: the hands close on a rung four above the feet's, every limb up one rung a cycle."""
    xb, r, sp = p["x"] - p["rung_out"], p["rung_radius"], p["rung_spacing"]
    zf, zh = p["first_rung"], p["first_rung"] + (3 if gm.STANCE["upright"] else 4) * sp   # the hands this many rungs above the feet: the shorter arms, fewer
    hold = lambda k: {"L_hand": bar((xb, 0.36, zh + k * sp), r=r), "R_hand": bar((xb, -0.36, zh + k * sp), r=r),
                      "L": bar((xb, 0.2, zf + k * sp), r=r), "R": bar((xb, -0.2, zf + k * sp), r=r)}
    base = lambda k: np.array([xb - climb_back(), 0.0, zf + k * sp + climb_up()])
    return _climb_sequence(kin, q_quad, hold, base, int(p["cycles"]), stop_x=xb - stop_off())


def scenario_tree(kin, q_quad, p):
    """The trunk: the hands round its flanks (a vertical hold each side, the fingers hooked round the bark), the feet
    on the branches, a branch a cycle."""
    x, r, sp, br = p["x"], p["radius"], p["branch_spacing"], p["branch_radius"]
    z0 = p["first_branch"]
    ang = math.radians(35.0)                                               # the hands' hold on the flanks: this far round from the near face

    foot_w = float(kin.model.geom_size[kin.model.geom("L").id][1])        # the foot's half width: the feet on the branches clear of the trunk
    y_foot = r + foot_w + 0.02

    def hold(k):
        zf, zh = z0 + k * sp, z0 + k * sp + 0.95 * gm.scale()
        out = {}
        for side, sy in (("L", 1.0), ("R", -1.0)):
            n = np.array([-math.cos(ang), sy * math.sin(ang), 0.0])
            out[f"{side}_hand"] = bar((x, 0.0, zh) + r * n, axis=(0.0, 0.0, 1.0), n=n, r=BAR_R)
            out[side] = bar((x, sy * y_foot, zf + 0.01), r=br, tip=0.0)   # flat on the thick branch: toes up would sink the heel into it
        return out

    back = 0.38 * gm.scale() if gm.STANCE["upright"] else climb_back()    # standing: the chest near the bark (the feet are beside the trunk, not ahead of it)
    up = 0.5 * gm.scale() if gm.STANCE["upright"] else climb_up()          # and the pelvis low enough that the legs reach the branches
    base = lambda k: np.array([x - r - back, 0.0, z0 + k * sp + up])
    stop_x = x - r - stop_off() - (0.0 if gm.STANCE["upright"] else 0.1)   # on four, the head is well ahead of the pelvis
    return _climb_sequence(kin, q_quad, hold, base, int(p["cycles"]), stop_x=stop_x, trials=VERTICAL_TRIALS)


def scenario_hull(kin, q_quad, p):
    """The ship's side: the ribs across the plate are the rungs, the hands two ribs above the feet."""
    slope = math.radians(p["slope_deg"])
    pitch = math.pi / 2 - slope + CLIMB_PITCH                              # the trunk along the plate
    up_ = gm.STANCE["upright"]
    back, tip = (0.6 * gm.scale(), slope - 0.4) if up_ else (0.4, slope - 0.2)   # the pelvis well off the plate, the feet along it (a static search against the solid plate)
    up, hand_ribs = (0.7 * gm.scale(), 3) if up_ else (0.6, 2)                  # the pelvis this far up the plate from the feet, the hands this many ribs above them
    n = np.array([-math.sin(slope), 0.0, math.cos(slope)])                 # the plate's normal, outward
    s0, rung = 0.3, p["rib_spacing"]
    rib = lambda s: np.array([p["x"] + s * math.cos(slope), 0.0, s * math.sin(slope)]) + p["rib_out"] * n

    def hold(k):
        sh, sf = s0 + (k + hand_ribs) * rung, s0 + k * rung
        return {"L_hand": bar(rib(sh) + [0.0, 0.36, 0.0], n=n, r=0.03), "R_hand": bar(rib(sh) + [0.0, -0.36, 0.0], n=n, r=0.03),
                "L": bar(rib(sf) + [0.0, 0.2, 0.0], n=n, r=0.03, tip=tip), "R": bar(rib(sf) + [0.0, -0.2, 0.0], n=n, r=0.03, tip=tip)}

    base = lambda k: rib(s0 + k * rung) + up * _unit(np.array([math.cos(slope), 0.0, math.sin(slope)])) + back * n
    stop_x = p["x"] - (0.4 * gm.scale() if up_ else stop_off())                         # standing: close under the plate, the first ribs a step up
    return _climb_sequence(kin, q_quad, hold, base, int(p["cycles"]), stop_x=stop_x, pitch=pitch)


def scenario_rope(kin, q_quad, p):
    """The rope: a vertical bar the hands close on one above the other, the feet clamping it below."""
    x, rung, rr = p["x"], 0.3, p["radius"]

    def hold(k):
        zf = 0.35 + k * rung
        zh = zf + up + (0.3 if gm.STANCE["upright"] else 0.45) * gm.scale()  # about the shoulders' height (standing, the hands further apart: the lower one does not reach across the chest)
        dz = 0.18 if gm.STANCE["upright"] else 0.15
        return {"L_hand": bar((x, 0.0, zh + dz), axis=(0.0, 0.0, 1.0), r=rr), "R_hand": bar((x, 0.0, zh - dz), axis=(0.0, 0.0, 1.0), r=rr),
                "L": point((x - 0.05, 0.1, zf), pitch=0.0), "R": point((x - 0.05, -0.1, zf), pitch=0.0)}   # the feet pinch the rope, soles level

    back = 0.35 * gm.scale() if gm.STANCE["upright"] else climb_back()     # standing frame: it hangs nearer the rope
    stop = x - stop_off() - (0.0 if gm.STANCE["upright"] else 0.1)         # on four, the head is well ahead of the pelvis
    up = 0.68 * gm.scale() if gm.STANCE["upright"] else climb_up()         # and lower above its feet (they pinch the rope straight below)
    base = lambda k: np.array([x - back, 0.0, 0.35 + k * rung + up])
    return _climb_sequence(kin, q_quad, hold, base, int(p["cycles"]), stop_x=stop, trials=VERTICAL_TRIALS)


def scenario_window(kin, q_quad, p):
    """Through the opening: the hands close on the sill's edge, a press-up on the arms lifts the body above the
    sill with the feet hanging (the wall is smooth: nothing for a foot), each foot comes up over the edge onto the
    sill, a crouch on it, then a leap off - the hands open, the pelvis on a parabola to the ground beyond, the
    limbs tucked and then reaching for the knuckle stance (the legs are shorter than the sill is high: there is no
    reaching the ground from it, so the frame drops as a gorilla would) - and the walk up to the table."""
    x = p["x"]
    stop_x = x - (0.7 * gm.scale() if gm.STANCE["upright"] else 0.95)
    return [("gait", stop_x / walk_speed(), 0.0)] + _window_entry(kin, q_quad, p, stop_x)


def _window_entry(kin, q_quad, p, stop_x):
    """From the quad pose at ``stop_x``: the mount onto the sill, the leap in, the walk to the table."""
    if gt.SOLID is not None:
        gt.SOLID.ignore = {"glass"}                                        # the pane is gone (broken, or never there)
    try:
        return _window_entry_upright(kin, q_quad, p, stop_x) if gm.STANCE["upright"] else _window_entry_quad(kin, q_quad, p, stop_x)
    finally:
        if gt.SOLID is not None:
            gt.SOLID.ignore = set()


def _window_entry_quad(kin, q_quad, p, stop_x):
    x, sill, t = p["x"], p["sill"], p["wall_thickness"]
    v = walk_speed()
    top = sill + 0.06
    edge = lambda sy: bar((x - t / 2 - 0.05, sy * 0.36, top - 0.03), r=0.03, wrist_dir=LEDGE_WRIST)   # the ledge's outer edge, taken from above
    holds = {"L_hand": edge(1.0), "R_hand": edge(-1.0), "L": point((x, 0.2, top), pitch=0.0, over=True), "R": point((x, -0.2, top), pitch=0.0, over=True)}
    crouch = np.array([x - 0.3, 0.0, top + 0.3])                          # a deep squat on the sill, the pelvis back from the feet (the knees' fold): the legs folded, the head under the lintel (the opening is 1.3 m tall)
    duck = CLIMB_PITCH                                                     # the trunk as on the climb: tilting it forward would tilt the hips out of their range
    press = np.array([x - 0.55, 0.0, top + 0.35])                          # pressed up on the arms, the body above the sill, the feet hanging
    b0 = np.array([stop_x, 0.0, gm.STANCE["h"]])
    start = _contacts(kin, q_quad, b0, gm.STANCE["pitch"])
    pts = gm._contact_points(kin)
    hands = {}
    for side in ("L", "R"):
        end = hand_targets(kin, side, holds[f"{side}_hand"])
        hands[side] = (_world_pts(kin, q_quad, b0, gm.STANCE["pitch"], [(bd, l) for bd, l, _ in end]), end)
    A, B, C = (0.0, 0.3), (0.3, 0.55), {"L": (0.55, 0.72), "R": (0.72, 0.9)}

    def hanging(side, bz):                                                 # a foot dangling under the pelvis, sole level
        return point(np.array([bz[0] + 0.1, 0.2 if side == "L" else -0.2, bz[2] - (0.85 if gm.STANCE["upright"] else 0.62)]), pitch=0.0)

    def t_up(u):                                                           # the hands to the edge; the press; the feet over the edge
        out = []
        fh = mk.smooth(min(1.0, u / 0.22))
        for side in ("L", "R"):
            s0, end = hands[side]
            if u < 0.22:
                out.append((end[0][0], end[0][1], (1 - fh) * s0[0] + fh * end[0][2] + np.array([0.0, 0.0, 0.3 * math.sin(math.pi * fh)])))
            else:
                out += end
            bz = np.array(b_up(u)[:3])
            hang = hanging(side, bz)
            a, b_ = C[side]
            if u <= B[0]:
                out += [(bd, l, sk) for (bd, l), sk in zip(pts[side], start[side])]
            elif u < a:                                                    # lifting off the ground as the body presses up
                fp = mk.smooth(min(1.0, (u - B[0]) / 0.12))
                goal = foot_targets(kin, side, hang)
                out += [(bd, l, (1 - fp) * sk + fp * g) for (bd, l, g), sk in zip(goal, start[side])]
            elif u >= b_:
                out += foot_targets(kin, side, holds[side])
            else:
                ff = (u - a) / (b_ - a)
                c = foot_swing(hang, holds[side], ff, lift=0.12, clear=0.12)
                tilt = point(holds[side]["c"], pitch=0.3 * (1.0 - mk.smooth(ff)), over=True)   # the toes a little down while lifted, level as it sets down
                out += foot_targets(kin, side, tilt, c)
        return out

    def b_up(u):
        fa = mk.smooth(min(1.0, u / A[1]))                                   # up on the feet, pitched to the climbing frame
        fb = mk.smooth(max(0.0, min(1.0, (u - B[0]) / (B[1] - B[0]))))       # the press-up
        fd = mk.smooth(max(0.0, (u - 0.85) / 0.15))                          # the crouch as the feet come onto the sill
        z_stand = hum.base_height(kin) - 0.03
        p1 = np.array([stop_x, 0.0, z_stand])
        bz = b0 + fa * (p1 - b0) + fb * (press - p1) + fd * (crouch - press)
        return bz[0], bz[1], bz[2], gm.STANCE["pitch"] + fa * (CLIMB_PITCH - gm.STANCE["pitch"]) + fd * (duck - CLIMB_PITCH)

    windows = {"L_hand": (0.0, 0.3), "R_hand": (0.0, 0.3)}
    extra = _extras(_spine_extra(kin, CLIMB_SPINE, q_quad, blend=True), _fingers_extra(kin, windows, holds, q_quad, from_start=True))
    up = gt.contact_motion(kin, "onto_the_sill", 6.0, t_up, b_up, q_init=q_quad, extra=extra, reg=IK_REG, iters=IK_ITERS,
                           source="kerchak_scenarios.window: the hands on the edge, a press-up, the feet over the edge onto the sill")
    q1 = gt._last(kin, up, q_quad)
    tuck = q_quad.copy()
    for side in ("L", "R"):
        tuck[kin.index[f"{side}_knee"]] = min(q_quad[kin.index[f"{side}_knee"]] + 0.8, kin.hi[kin.index[f"{side}_knee"]])
        tuck[kin.index[f"{side}_hip_pitch"]] = max(q_quad[kin.index[f"{side}_hip_pitch"]] - 0.6, kin.lo[kin.index[f"{side}_hip_pitch"]])
        tuck[kin.index[f"{side}_shoulder_pitch"]] = q_quad[kin.index[f"{side}_shoulder_pitch"]] - 1.1   # the arms up and forward, clear of the sill
        tuck[kin.index[f"{side}_elbow"]] = q_quad[kin.index[f"{side}_elbow"]] - 0.9
    release = q1.copy()
    for side in ("L", "R"):
        sr.set_curl(release, kin, side, CURL_OPEN)
    x_land = x + 1.3

    x_clear = x + t / 2 + 0.2                                              # the pelvis past the inner face before it drops: the feet stay up over the sill

    def base(u):                                                           # a dive off the sill: no rise under the lintel; the arms come off the edge first
        f = mk.smooth(max(0.0, (u - 0.2) / 0.8))
        xb = crouch[0] + f * (x_land - crouch[0])
        fz = mk.smooth(max(0.0, min(1.0, (xb - x_clear) / (x_land - x_clear))))
        flight = math.sin(math.pi * fz)
        return xb, 0.0, (1 - fz) * crouch[2] + fz * gm.STANCE["h"] + 0.05 * flight, duck + fz * (gm.STANCE["pitch"] - duck)

    keys = [(0.0, q1), (0.15, release), (0.45, tuck), (0.75, tuck), (1.0, q_quad)]
    leap = _joint_track(kin, "leap_off_the_sill", 1.6, keys, base, source="kerchak_scenarios.window: the leap off the sill onto four")
    table_x = x + p.get("table_x", 3.4)
    walk = max((table_x - 0.4 - 0.55 - x_land) / v, 1.0)                  # up to the table: the knuckles a hand's length short of its edge
    return [("motion", up), ("motion", leap), ("gait", walk, 0.0)]


SILL_CROUCH = (-0.15, 0.45, 0.45)                                          #: the upright frame crouched on the sill: the pelvis behind the wall's middle, above the sill's top, pitched forward [m, m, rad]


def _window_entry_upright(kin, q_quad, p, stop_x):
    """Standing at ``stop_x``: the hands take the jambs (the fingers round the frame's outer corners), a spring off
    the feet and a pull on the arms hoist the body to a crouch in the opening (the wall is smooth: the feet hang
    until they come over the edge onto the sill, one after the other), the head under the lintel, a hop down inside,
    the walk to the table. (The hip's range does not allow a foot stepped up onto a sill at the hip's height.)"""
    x, sill, t = p["x"], p["sill"], p["wall_thickness"]
    v = walk_speed()
    top = sill + 0.06
    S = gm.scale()
    jamb = lambda sy: bar((x - t / 2 + 0.04, sy * (p["width"] / 2 - 0.02), top + 0.7 * S), axis=(0.0, 0.0, 1.0), n=(0.0, -sy, 0.0), r=0.03)   # the jamb's outer corner, the fingers round it
    feet = {"L": point((x, 0.2, top), pitch=0.0, over=True), "R": point((x, -0.2, top), pitch=0.0, over=True)}
    jambs = {"L_hand": jamb(1.0), "R_hand": jamb(-1.0)}
    crouch = np.array([x + SILL_CROUCH[0] * S, 0.0, top + SILL_CROUCH[1] * S])
    hoist = np.array([x - t / 2 - 0.3 * S, 0.0, top + 0.6 * S])             # pulled up high outside the wall's face (no lintel there), the legs hanging clear of the sill's lip
    duck = SILL_CROUCH[2]
    ready = np.array([stop_x + 0.15 * S, 0.0, gm.STANCE["h"] - 0.12 * S])          # set for the spring, the hands on the jambs
    ready_pitch = 0.2
    b0 = np.array([stop_x, 0.0, gm.STANCE["h"]])
    start = _contacts(kin, q_quad, b0, gm.STANCE["pitch"])
    pts = gm._contact_points(kin)
    frames = _pick_frames(kin, q_quad, {**jambs, **feet}, crouch, duck, VERTICAL_TRIALS)
    for g in jambs:
        jambs[g]["sign"], jambs[g]["phi"] = frames[g]
    hands = {}
    for side in ("L", "R"):
        end = hand_targets(kin, side, jambs[f"{side}_hand"])
        hands[side] = (_world_pts(kin, q_quad, b0, gm.STANCE["pitch"], [(bd, l) for bd, l, _ in end]), end)
    A, B, C, D = (0.0, 0.22), (0.22, 0.48), {"L": (0.48, 0.64), "R": (0.64, 0.8)}, (0.8, 1.0)   # the hands to the jambs; the hoist; the feet over the edge; in to the crouch

    def hanging(side, bz):                                                 # a foot dangling behind the wall's face, the toes down
        return point(np.array([bz[0] - 0.3 * S, 0.2 if side == "L" else -0.2, bz[2] - 0.55 * S]), pitch=0.5)

    def t_up(u):
        out = []
        fh = mk.smooth(min(1.0, u / A[1]))
        bz = np.array(b_up(u)[:3])
        for side in ("L", "R"):
            s0, end = hands[side]
            if u < A[1]:
                out.append((end[0][0], end[0][1], (1 - fh) * s0[0] + fh * end[0][2] + np.array([0.0, 0.0, 0.25 * math.sin(math.pi * fh)])))
            else:
                out += end
            hang = hanging(side, bz)
            a, b_ = C[side]
            if u <= B[0]:
                out += [(bd, l, sk) for (bd, l), sk in zip(pts[side], start[side])]
            elif u < a:                                                    # lifting off the ground as the body springs up: the ball of the foot alone, the ankle free
                fp = mk.smooth(min(1.0, (u - B[0]) / 0.15))
                goal = foot_targets(kin, side, hang)
                out += [(bd, l, (1 - fp) * sk + fp * g) for (bd, l, g), sk in zip(goal, start[side])]
            elif u >= b_:
                out += foot_targets(kin, side, feet[side])
            else:
                ff = (u - a) / (b_ - a)
                c = foot_swing(hang, feet[side], ff, lift=0.25 * S, clear=0.15)   # straight up outside the wall, then in high over the sill
                tilt = point(feet[side]["c"], pitch=0.3 * (1.0 - mk.smooth(ff)), over=True)   # the toes a little down while lifted, level as it sets down
                out += foot_targets(kin, side, tilt, c)
        return out

    def b_up(u):
        fa = mk.smooth(min(1.0, u / A[1]))
        fb = mk.smooth(max(0.0, min(1.0, (u - B[0]) / (B[1] - B[0]))))
        fd = mk.smooth(max(0.0, min(1.0, (u - D[0]) / (D[1] - D[0]))))
        bz = b0 + fa * (ready - b0) + fb * (hoist - ready) + fd * (crouch - hoist)
        return bz[0], bz[1], bz[2], gm.STANCE["pitch"] + fa * (ready_pitch - gm.STANCE["pitch"]) + fb * (duck - ready_pitch)

    windows = {"L_hand": A, "R_hand": A}
    extra = _extras(_spine_extra(kin, 0.15, q_quad, blend=True), _fingers_extra(kin, windows, jambs, q_quad, from_start=True))
    up = gt.contact_motion(kin, "onto_the_sill", 7.0, t_up, b_up, q_init=q_quad, extra=extra, reg=IK_REG, iters=IK_ITERS,
                           source="kerchak_scenarios.window (upright): the hands on the jambs, a spring and a pull up outside the wall, the feet over the edge, in to a crouch in the opening",
                           q_ref=climb_ref(kin, q_quad))
    up.params = {**(up.params or {}), "frames": {g: list(f) for g, f in frames.items()}}
    q1 = gt._last(kin, up, q_quad)
    tuck = q1.copy()                                                       # the crouch's folded legs lifted off the sill, the arms forward
    for side in ("L", "R"):
        tuck[kin.index[f"{side}_knee"]] = min(q1[kin.index[f"{side}_knee"]] + 0.25, kin.hi[kin.index[f"{side}_knee"]])
        tuck[kin.index[f"{side}_hip_pitch"]] = max(q1[kin.index[f"{side}_hip_pitch"]] - 0.2, kin.lo[kin.index[f"{side}_hip_pitch"]])
        tuck[kin.index[f"{side}_shoulder_pitch"]] = q_quad[kin.index[f"{side}_shoulder_pitch"]] - 0.6
        tuck[kin.index[f"{side}_elbow"]] = q_quad[kin.index[f"{side}_elbow"]] - 0.3
        sr.set_curl(tuck, kin, side, CURL_OPEN)
    release = q1.copy()
    for side in ("L", "R"):
        sr.set_curl(release, kin, side, CURL_OPEN)
    x_land = x + t / 2 + 0.75 * S
    x_clear = x + t / 2 + 0.25 * S                                         # the pelvis past the inner face before it drops: the feet stay up over the sill

    def base(u):                                                           # a hop down off the sill, landing standing
        f = mk.smooth(u)
        xb = crouch[0] + f * (x_land - crouch[0])
        fz = mk.smooth(max(0.0, min(1.0, (xb - x_clear) / (x_land - x_clear))))
        flight = math.sin(math.pi * fz)
        return xb, 0.0, (1 - fz) * crouch[2] + fz * gm.STANCE["h"] + 0.04 * flight, duck + fz * (gm.STANCE["pitch"] - duck)

    keys = [(0.0, q1), (0.12, release), (0.3, tuck), (0.62, tuck), (1.0, q_quad)]
    leap = _joint_track(kin, "leap_off_the_sill", 1.6, keys, base, source="kerchak_scenarios.window (upright): the hop down off the sill, the legs folded until past the wall")
    table_x = x + p.get("table_x", 3.4)
    walk = max((table_x - 0.4 - 0.45 - x_land) / v, 1.0)
    return [("motion", up), ("motion", leap), ("gait", walk, 0.0)]


def _slerp(R0, R1, f):
    """The rotation ``f`` of the way from ``R0`` to ``R1``."""
    D = R0.T @ R1
    c = max(-1.0, min(1.0, (np.trace(D) - 1.0) / 2.0))
    ang = math.acos(c)
    if ang < 1e-9:
        return R0.copy()
    w = np.array([D[2, 1] - D[1, 2], D[0, 2] - D[2, 0], D[1, 0] - D[0, 1]]) / (2.0 * math.sin(ang))
    return R0 @ _rot(w, f * ang)


def _hand_at(kin, side, c, R):
    """The hand's three grip points for a grip centre ``c`` and a hand frame ``R``."""
    hand = kin.body(f"{side}_hand")
    return [(hand, l, c + R @ (l - GRIP)) for l in (GRIP, np.zeros(3), GRIP + np.array([0.0, GRIP_Y, 0.0]))]


def _lerp_hold(h0, h1, f):
    """A bar hold between two: the centre, the axis and the wrist's direction eased from ``h0`` to ``h1``."""
    f = mk.smooth(f)
    wd0 = h0["wrist_dir"] if h0["wrist_dir"] is not None else h1["wrist_dir"]
    return bar((1 - f) * h0["c"] + f * h1["c"], axis=(1 - f) * h0["axis"] + f * h1["axis"], n=h1["n"], r=h1["r"],
               wrist_dir=(1 - f) * wd0 + f * h1["wrist_dir"])


def _hand_pose(kin, q, base, pitch, side):
    """(R, p) of the hand in the world for a pose at a base place."""
    kin.set(q)
    Rb, b = gm.base_rot(pitch), np.asarray(base, dtype=float)
    hand = kin.body(f"{side}_hand")
    return Rb @ kin.data.xmat[hand].reshape(3, 3), Rb @ kin.data.xpos[hand] + b


STAND = (0.85, 0.1)                                                       #: the pelvis (height, pitch) up on the feet with a tool in hand (on four); standing: its own height


def stand_base():
    return (gm.STANCE["h"] - 0.03, 0.05) if gm.STANCE["upright"] else STAND


def scenario_window_hammer(kin, q_quad, p):
    """The glass broken first: the walk up to the hammer lying by the wall, the right fist closes on its handle from
    above (on three: the left knuckle and the feet), up onto the feet with the hammer, a wind-up over the shoulder
    and a strike through the pane (the glass goes at the blow), the fist opens and the hammer falls, back down on
    four, then the window's entry (``_window_entry``) through the empty frame to the table."""
    import kerchak_tasks as kt
    x, sill, h, t = p["x"], p["sill"], p["height"], p["wall_thickness"]
    H = HAMMER
    up_ = gm.STANCE["upright"]
    S = gm.scale()
    v, stop_x = (walk_speed(), p["hammer_x"] - 0.3 * S) if up_ else (walk_speed(), x - 0.95)   # standing: a short step from the hammer
    b_quad = (stop_x, gm.STANCE["h"], gm.STANCE["pitch"])
    b_pick = (stop_x, 0.46 * S, 0.8) if up_ else b_quad                        # standing: a deep squat, bending to the ground (the spine bent too)
    PICK_SPINE = 0.6 if up_ else None                                     # the spine's pitch modules bent forward over the squat
    b_stand = (stop_x, stand_base()[0], stand_base()[1])
    planted = gt._planted(kin, q_quad, stop_x, which=("L", "R"))
    pts = gm._contact_points(kin)
    start = _contacts(kin, q_quad, (stop_x, 0.0, gm.STANCE["h"]), gm.STANCE["pitch"])
    on_ground = bar((p["hammer_x"], p["hammer_y"], H["rest_z"]), axis=(0.0, 1.0, 0.0), n=(0.0, 0.0, 1.0), r=H["handle_r"], wrist_dir=(-0.45, 0.0, 0.89))
    # the hammer in hand as a hand frame at a grip centre: carried before the chest, the handle up in a forward fist;
    # raised high beside the shoulder; the blow: the handle at the pane, the wrist above and behind
    carry = (np.array([stop_x + (0.45 * S if up_ else 0.8), -0.45 * (S if up_ else 1.0), stand_base()[0] + (0.25 * S if up_ else 0.4)]), hand_frame_on(bar((0, 0, 0), axis=(0.0, 0.0, 1.0), n=(0.0, 1.0, 0.0)), "R"))
    wind_up = (np.array([stop_x + 0.2 * S, -0.55 * S if up_ else -0.65, stand_base()[0] + (1.0 * S if up_ else 1.2)]), hand_frame_on(bar((0, 0, 0), axis=(-0.2, 0.0, 0.98), n=(0.0, 1.0, 0.0)), "R"))   # raised high beside the shoulder: the blow comes down
    glass_point = np.array([x - t / 2 - 0.01, -0.35 if up_ else -0.15, sill + h * 0.45])   # standing: the blow out to the right, the arm not across the chest
    strike_axis = np.array([0.95, 0.0, -0.31])
    strike = (glass_point - 0.31 * strike_axis, hand_frame_on(bar((0, 0, 0), axis=strike_axis, n=(-1.0, 0.0, 0.0), r=H["handle_r"], wrist_dir=(-0.5, 0.0, 0.87)), "R"))
    ground_frame = (on_ground["c"], hand_frame_on(on_ground, "R"))
    kin.set(kin.q0)
    hang = kin.world(*pts["L_knuckle"][0])                                 # the left hand hanging (the design pose, base-relative)
    grip0 = _world_pts(kin, q_quad, (stop_x, 0.0, gm.STANCE["h"]), gm.STANCE["pitch"], [(hand_targets(kin, "R", on_ground)[0][0], hand_targets(kin, "R", on_ground)[0][1])])[0]

    # 1. the pick: on three, the right hand down onto the handle
    def pick_base(u):
        f = mk.smooth(min(1.0, u / 0.5))
        return b_quad[0], b_quad[1] + f * (b_pick[1] - b_quad[1]), b_quad[2] + f * (b_pick[2] - b_quad[2])

    def t_pick(u):
        end = hand_targets(kin, "R", on_ground)
        if gm.STANCE["upright"]:
            xb, hb, pb = pick_base(u)
            left = gm.base_rot(pb) @ hang + np.array([xb, 0.0, hb])
        else:
            left = start["L_knuckle"][0]
        out = planted + ([] if up_ else [(pts["L_knuckle"][0][0], pts["L_knuckle"][0][1], left)])   # standing, the left arm simply hangs from the shoulder
        if u < 0.6:
            f = mk.smooth(u / 0.6)
            out.append((end[0][0], end[0][1], (1 - f) * grip0 + f * on_ground["c"] + np.array([0.0, 0.0, 0.15 * math.sin(math.pi * f)])))
        else:
            out += end
        return out

    def x_pick(u, q):
        c = su.KNUCKLE_CURL + (CURL_OPEN - su.KNUCKLE_CURL) * mk.smooth(min(1.0, u / 0.4)) if u < 0.75 else CURL_OPEN + (kt.LEVER_CURL - CURL_OPEN) * mk.smooth((u - 0.75) / 0.25)
        q = sr.set_curl(q, kin, "R", c)
        if PICK_SPINE is not None:
            q = _spine_extra(kin, PICK_SPINE, q_quad, blend=True)(min(1.0, u / 0.5), q)
        return q

    pick = gt.contact_motion(kin, "pick_up_the_hammer", 1.6, t_pick, pick_base, q_init=q_quad, extra=x_pick, reg=IK_REG, iters=IK_ITERS,
                             source="kerchak_scenarios.window_hammer: the fist closes on the hammer's handle")
    q1 = gt._last(kin, pick, q_quad)
    Rh, ph = _hand_pose(kin, q1, (stop_x, 0.0, b_pick[1]), b_pick[2], "R")
    # the hammer in the hand: its handle along the hand's y through the grip, its root a half handle back
    R_h2m = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    p_h2m = GRIP + np.array([0.0, -H["handle"] / 2, 0.0])
    R_m, p_m = Rh @ R_h2m, ph + Rh @ p_h2m                               # where the hammer lies: where the hand found it
    pick.params = {**pick.params, "hammer_T": (R_h2m.tolist(), p_h2m.tolist()), "hammer_rest": (p_m.tolist(), R_m.tolist())}

    # 2. up onto the feet, the hammer carried, the left hand hanging
    def t_stand(u):
        f = mk.smooth(u)
        xb, hb, pb = b_stand_fn(u)
        R, b = gm.base_rot(pb), np.array([xb, 0.0, hb])
        left = (1 - f) * start["L_knuckle"][0] + f * (R @ hang + b) + np.array([0.0, 0.0, 0.12 * math.sin(math.pi * f)])
        fs = mk.smooth(u)
        c = (1 - fs) * ground_frame[0] + fs * carry[0] + np.array([0.0, 0.0, 0.1 * math.sin(math.pi * fs)])
        return planted + ([] if up_ else [(pts["L_knuckle"][0][0], pts["L_knuckle"][0][1], left)]) + _hand_at(kin, "R", c, _slerp(ground_frame[1], carry[1], fs))

    def b_stand_fn(u):
        f = mk.smooth(u)
        return stop_x, b_pick[1] + f * (stand_base()[0] - b_pick[1]), b_pick[2] + f * (stand_base()[1] - b_pick[2])

    hold_fist = lambda u, q: sr.set_curl(q, kin, "R", kt.LEVER_CURL)
    x_stand = hold_fist if PICK_SPINE is None else _extras(hold_fist, _spine_extra(kin, q_quad[kin.index["spine_lumbar_pitch"]], q1, blend=True))
    stand = gt.contact_motion(kin, "up_with_the_hammer", 2.0, t_stand, b_stand_fn, q_init=q1, extra=x_stand, reg=IK_REG, iters=IK_ITERS,
                              source="kerchak_scenarios.window_hammer: up on the feet with the hammer")
    q2 = gt._last(kin, stand, q1)

    # 3. the strike: a wind-up over the shoulder, the blow through the pane, back to the carry
    def hold_at(u):
        if u < 0.4:
            f = mk.smooth(u / 0.4)
            return (1 - f) * carry[0] + f * wind_up[0] + np.array([-0.1, -0.25, 0.0]) * math.sin(math.pi * f), _slerp(carry[1], wind_up[1], f)   # out to the side on the way up: past the head, not through it
        if u < 0.65:
            f = mk.smooth((u - 0.4) / 0.25)
            return (1 - f) * wind_up[0] + f * strike[0] + np.array([0.0, 0.0, 0.12]) * math.sin(math.pi * f), _slerp(wind_up[1], strike[1], f)
        if u < 0.75:
            return strike
        f = mk.smooth((u - 0.75) / 0.25)
        return (1 - f) * strike[0] + f * carry[0], _slerp(strike[1], carry[1], f)

    def t_smash(u):
        xb, hb, pb = b_smash(u)
        R, b = gm.base_rot(pb), np.array([xb, 0.0, hb])
        c, Rh = hold_at(u)
        return planted + ([] if up_ else [(pts["L_knuckle"][0][0], pts["L_knuckle"][0][1], R @ hang + b)]) + _hand_at(kin, "R", c, Rh)

    def b_smash(u):
        lunge = 0.08 * math.sin(math.pi * min(1.0, max(0.0, (u - 0.3) / 0.5)))
        return stop_x + lunge, stand_base()[0], stand_base()[1]

    smash = gt.contact_motion(kin, "smash_the_glass", 1.8, t_smash, b_smash, q_init=q2, extra=hold_fist, reg=IK_REG, iters=IK_ITERS,
                              source="kerchak_scenarios.window_hammer: the hammer through the pane")
    smash.params = {**smash.params, "strike_u": 0.65}
    q3 = gt._last(kin, smash, q2)

    # 4. the hammer dropped, back down on four
    R_end, p_end = _hand_pose(kin, q3, (stop_x, 0.0, stand_base()[0]), stand_base()[1], "R")
    kn_r = pts["R_knuckle"][0]
    kin.set(q3)
    Rb3 = gm.base_rot(stand_base()[1])
    r_start = Rb3 @ kin.world(*kn_r) + np.array([stop_x, 0.0, stand_base()[0]])
    l_start = Rb3 @ kin.world(*pts["L_knuckle"][0]) + np.array([stop_x, 0.0, stand_base()[0]])

    def t_drop(u):
        if u < 0.2:
            xb, hb, pb = b_drop(u)
            R, b = gm.base_rot(pb), np.array([xb, 0.0, hb])
            return planted + [(pts["L_knuckle"][0][0], pts["L_knuckle"][0][1], R @ hang + b)] + _hand_at(kin, "R", *carry)
        f = mk.smooth((u - 0.2) / 0.8)
        arc = np.array([0.0, 0.0, 0.15 * math.sin(math.pi * f)])
        return planted + [(pts["L_knuckle"][0][0], pts["L_knuckle"][0][1], (1 - f) * l_start + f * start["L_knuckle"][0] + arc),
                          (kn_r[0], kn_r[1], (1 - f) * r_start + f * start["R_knuckle"][0] + arc)]

    def b_drop(u):
        f = mk.smooth(max(0.0, (u - 0.2) / 0.8))
        return stop_x, stand_base()[0] + f * (gm.STANCE["h"] - stand_base()[0]), stand_base()[1] + f * (gm.STANCE["pitch"] - stand_base()[1])

    arm_joints = [j for j in kin.actuated if j.startswith("R_") and ("shoulder" in j or "elbow" in j or "wrist" in j)]

    def x_drop(u, q):
        c = kt.LEVER_CURL + (CURL_OPEN - kt.LEVER_CURL) * mk.smooth(min(1.0, u / 0.2)) if u < 0.2 else CURL_OPEN + (su.KNUCKLE_CURL - CURL_OPEN) * mk.smooth((u - 0.2) / 0.8)
        q = sr.set_curl(q, kin, "R", c)
        if up_ and u > 0.6:                                                # standing: the arm settles to its hanging rest (the IK's straight-arm minimum avoided)
            f = mk.smooth((u - 0.6) / 0.4)
            for j in arm_joints:
                q[kin.index[j]] = (1 - f) * q[kin.index[j]] + f * q_quad[kin.index[j]]
        return q

    drop = gt.contact_motion(kin, "drop_the_hammer", 2.0, t_drop, b_drop, q_init=q3, extra=x_drop, reg=IK_REG, iters=IK_ITERS,
                             source="kerchak_scenarios.window_hammer: the fist opens, back down on four")
    drop.params = {**drop.params, "release_u": 0.2}
    segs = [("gait", stop_x / v, 0.0), ("motion", pick), ("motion", stand), ("motion", smash), ("motion", drop)]
    if up_:                                                                # standing: the steps up to the window
        sill_x = x - 0.7 * S
        segs.append(("gait", (sill_x - stop_x) / v, 0.0))
        return segs + _window_entry(kin, q_quad, p, sill_x)
    return segs + _window_entry(kin, q_quad, p, stop_x)


def _joint_track(kin, name, duration, keys, base_fn, frames=24, source=""):
    """Joint keyposes [(fraction, {joint: value})] eased in turn, the base on ``base_fn(u)`` → (x, y, h, pitch)."""
    ts = [duration * i / frames for i in range(frames + 1)]
    Q, xs, ys, hs, ps = [], [], [], [], []
    for t in ts:
        u = t / duration
        k = max(i for i, (f, _) in enumerate(keys) if f <= u + 1e-9)
        qa = keys[k][1]
        if k + 1 < len(keys):
            fa, fb = keys[k][0], keys[k + 1][0]
            w = mk.smooth((u - fa) / max(fb - fa, 1e-6))
            q = (1 - w) * qa + w * keys[k + 1][1]
        else:
            q = qa
        Q.append(np.clip(q, kin.lo, kin.hi))
        x, y, h, p = base_fn(u)
        xs.append(round(float(x), 4)); ys.append(round(float(y), 4)); hs.append(round(float(h), 4)); ps.append(round(float(p), 4))
    moving = mk._moving(kin, Q)
    cols = [kin.index[j] for j in moving]
    return mk.Motion(name, "transition", moving, ts, np.asarray(Q)[:, cols], loop=False, period=duration,
                     base={"x_m": xs, "y_m": ys, "height_m": hs, "pitch_rad": ps}, source=source, check={"ik_error_max_m": 0.0})


def scenario_obstacles(kin, q_quad, p):
    """Before each bar: a crouch (the knees and the elbows folding), the leap (the pelvis on a parabola clearing the
    bar, the limbs tucked), the landing on four - joint keyposes on a base track (no contact holds in the air)."""
    v = walk_speed()
    segs, x_prev = [], 0.0
    crouch = q_quad.copy()
    tuck = q_quad.copy()
    for side in ("L", "R"):
        crouch[kin.index[f"{side}_knee"]] = min(q_quad[kin.index[f"{side}_knee"]] + 0.35, kin.hi[kin.index[f"{side}_knee"]])
        crouch[kin.index[f"{side}_elbow"]] = q_quad[kin.index[f"{side}_elbow"]] - 0.3
        tuck[kin.index[f"{side}_knee"]] = min(q_quad[kin.index[f"{side}_knee"]] + 0.8, kin.hi[kin.index[f"{side}_knee"]])
        tuck[kin.index[f"{side}_hip_pitch"]] = max(q_quad[kin.index[f"{side}_hip_pitch"]] - 0.6, kin.lo[kin.index[f"{side}_hip_pitch"]])
        tuck[kin.index[f"{side}_shoulder_pitch"]] = q_quad[kin.index[f"{side}_shoulder_pitch"]] - 0.6
        tuck[kin.index[f"{side}_elbow"]] = q_quad[kin.index[f"{side}_elbow"]] - 0.6
    for k in range(int(p["count"])):
        xb = p["x"] + k * p["spacing"]
        take_off = xb - 0.9
        segs.append(("gait", max(take_off - x_prev, 0.0) / v, 0.0))
        apex = p["height"] + 0.55

        def base(u, take_off=take_off, apex=apex):
            f = mk.smooth(u)
            crouch_dip = 0.12 * math.sin(math.pi * min(1.0, u / 0.25)) if u < 0.25 else 0.0
            flight = math.sin(math.pi * max(0.0, (u - 0.25) / 0.75)) if u > 0.25 else 0.0
            return take_off + f * 1.8, 0.0, gm.STANCE["h"] - crouch_dip + apex * flight, gm.STANCE["pitch"] - 0.25 * flight

        keys = [(0.0, q_quad), (0.25, crouch), (0.45, tuck), (0.8, tuck), (1.0, q_quad)]
        segs.append(("motion", _joint_track(kin, f"jump_{k + 1}", 1.8, keys, base, source="kerchak_scenarios.obstacles: the leap")))
        x_prev = xb + 0.9
    segs.append(("gait", 4.0, 0.0))
    return segs


def scenario_door(kin, q_quad, p):
    v = walk_speed()
    segs = [("gait", gt.stop_x() / v, 0.0)]
    segs += [("motion", m) for m in gt.door_sequence(kin, q_quad)]
    segs.append(("gait", 2.75 / v, 0.0))                                    # through the doorway and on
    return segs


WINDOW_HAMMER = {**WINDOW, "glass": 1, "hammer_x": WINDOW["x"] - 0.4, "hammer_y": -0.45}
DEFAULTS = {"wall": WALL, "tree": TREE, "hull": HULL, "rope": ROPE, "window": WINDOW, "window_hammer": WINDOW_HAMMER, "obstacles": OBSTACLES, "door": gt.DOOR}
BUILDERS = {"wall": wall_prop, "tree": tree_prop, "hull": hull_prop, "rope": rope_prop, "window": window_prop, "window_hammer": window_prop,
            "obstacles": obstacles_prop}
SCENARIOS = {"wall": scenario_wall, "tree": scenario_tree, "hull": scenario_hull, "rope": scenario_rope, "window": scenario_window,
             "window_hammer": scenario_window_hammer, "obstacles": scenario_obstacles, "door": scenario_door}


def sequence(name, kin, q_quad, params=None):
    """A scenario's segments: [("gait", seconds, lane) | ("motion", Motion)]."""
    p = params_for(name, params)
    return SCENARIOS[name](kin, q_quad, p)
