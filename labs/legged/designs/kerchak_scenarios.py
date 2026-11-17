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

GREY, WOOD, BARK, STEEL, ROPE_C = (0.45, 0.45, 0.47, 1.0), (0.55, 0.42, 0.28, 1.0), (0.4, 0.3, 0.2, 1.0), (0.5, 0.52, 0.56, 1.0), (0.75, 0.65, 0.45, 1.0)
CLIMB_PITCH = -0.15                                                       #: the pelvis on a wall [rad]: the trunk along it, the shoulders leaning out a little
CLIMB_BACK, CLIMB_UP = 0.45, 0.55                                         #: the pelvis off the surface, and above the feet's holds [m]
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
    a = -(math.pi / 2 - slope)
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
                 + _table_with_banana(p.get("table_x", 3.4), p.get("table_y", 0.0)))
    return Prop(frame, free=False)


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


def props(name, params=None):
    """The scene's props for a scenario (the door's from ``kerchak_tasks``)."""
    p = {**DEFAULTS[name], **(params or {})}
    if name == "door":
        return (gt.door_prop(),)
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
GRIP_Y = 0.06                             #: a second point along the hand's y (the bar's axis) that fixes the hand's roll
BAR_R = 0.025                             #: the bar the grip is sized for; a thicker hold moves the palm out to its surface
SOLE_ON_BAR = 0.08                        #: the sole's point that rests on a bar: the ball of the foot, 2 cm of toe past the bar [m]
SOLE_HALF = 0.015                         #: the sole's half thickness: the contact points are the sole box's centre plane, its bottom face rests on the hold [m]
FOOT_TIP = 0.5                            #: the foot on a bar: toes up, into the surface [rad]
SWING_FREE = 0.8                          #: through this much of a step onto a bar only the ball of the foot is placed, the foot's tilt free
ORDER = ["L", "R", "L_hand", "R_hand"]    #: a climb cycle's order: the feet step up first (the legs push), then the hands reach
IK_REG, IK_ITERS = 3e-3, 900              #: the climbs' IK: well damped (a grip is four points per hand; a lighter damping jumps between solutions)


def bar(centre, axis=(0.0, 1.0, 0.0), n=(-1.0, 0.0, 0.0), r=BAR_R, wrist_dir=None):
    """A hold on a bar at ``centre`` along ``axis``, of radius ``r``, on a surface whose outward normal is ``n``.
    ``wrist_dir``: where the wrist sits from the grip (unit; default a rung's: below and behind)."""
    return {"kind": "bar", "c": np.asarray(centre, dtype=float), "axis": _unit(axis), "n": _unit(n), "r": float(r), "sign": 1.0, "phi": 0.0,
            "wrist_dir": None if wrist_dir is None else _unit(wrist_dir)}


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


def foot_targets(kin, side, hold, centre=None, tip=FOOT_TIP):
    """The foot on a hold. A bar: the sole's three points with the sole's ``SOLE_ON_BAR`` point on the bar's outer
    top, the foot pointing into the surface, toes up ``tip`` from the horizontal. A point: the sole flat on the surface."""
    c = hold["c"] if centre is None else np.asarray(centre, dtype=float)
    pts = gm._contact_points(kin)[side]
    if hold["kind"] == "point":
        up = gm.base_rot(hold["pitch"]) @ np.array([0.0, 0.0, 1.0])
        return [(body, local, c + _sole_offset(k, hold["pitch"]) + SOLE_HALF * up) for k, (body, local) in enumerate(pts)]
    n, t = hold["n"], _up_along(hold["n"])
    h = -n - (-n @ np.array([0.0, 0.0, 1.0])) * np.array([0.0, 0.0, 1.0])   # into the surface, horizontally
    f = math.cos(tip) * _unit(h) + math.sin(tip) * np.array([0.0, 0.0, 1.0])
    up = np.cross(f, hold["axis"])
    up = up if up @ np.array([0.0, 0.0, 1.0]) >= 0 else -up
    top = c + hold["r"] * _unit(n + t) + SOLE_HALF * up                        # the sole's bottom face on the bar's top
    return [(body, local, top + (o[0] - SOLE_ON_BAR) * f + o[1] * hold["axis"]) for (body, local), o in zip(pts, hum.SOLE_POINTS)]


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


def climb_cycle(kin, q_from, holds, base_from, base_to, *, pitch=CLIMB_PITCH, duration=3.2, lift=0.12, name="climb"):
    """One cycle up: the contacts move one at a time (``ORDER``: left foot, right foot, left hand, right hand) from
    ``holds[0]`` to ``holds[1]`` ({contact: hold}) on an arc out from the surface - the fingers opening through the
    swing and closing on the bar - while the pelvis rides from ``base_from`` to ``base_to`` (xyz) in the climbing frame."""
    windows = {g: (0.05 + 0.2 * k, 0.05 + 0.2 * (k + 1)) for k, g in enumerate(ORDER)}

    def targets(u):
        out = []
        for g in ORDER:
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
        f = mk.smooth(u)                                                   # the pelvis rides up through the whole cycle
        b = (1 - f) * np.asarray(base_from) + f * np.asarray(base_to)
        return b[0], b[1], b[2], pitch

    extra = _extras(_spine_extra(kin, CLIMB_SPINE), _fingers_extra(kin, windows, holds[1], q_from))
    return gt.contact_motion(kin, name, duration, targets, base, q_init=q_from, extra=extra, reg=IK_REG, iters=IK_ITERS,
                             source="kerchak_scenarios.climb_cycle")


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


def mount(kin, q_quad, base_x, holds, base_climb, *, pitch=CLIMB_PITCH, duration=3.5, name="mount"):
    """From the quad pose at the foot of the surface up onto it: the pelvis rises and pitches to the climbing frame
    where it stands while the hands go from the ground to their first holds (the fingers opening on the way, closing
    on the bars; the feet planted), then the feet step up to theirs one after the other as the pelvis moves in."""
    b0 = np.array([base_x, 0.0, gm.PELVIS_H])
    start = _contacts(kin, q_quad, b0, gm.PELVIS_PITCH)
    pts = gm._contact_points(kin)
    hands = {}
    for side in ("L", "R"):
        end = hand_targets(kin, side, holds[f"{side}_hand"])
        hands[side] = (_world_pts(kin, q_quad, b0, gm.PELVIS_PITCH, [(b, l) for b, l, _ in end]), end)
    bc = np.asarray(base_climb, dtype=float)
    sloped = abs(float(holds["L"]["n"] @ np.array([0.0, 0.0, 1.0]))) > 0.3
    # a vertical surface: the pelvis rises on the planted feet while the hands reach, moves in as the feet step up;
    # a slope (its holds ahead): the pelvis moves in low while the hands reach, the feet step, then it rises
    hands_end = 0.5 if not sloped else 0.35
    foot_windows = {"L": (0.5, 0.75), "R": (0.75, 1.0)} if not sloped else {"L": (0.35, 0.55), "R": (0.55, 0.75)}
    reach = 0.7 * hands_end                                                # the hands' grip points on their arcs until here, then the whole frame settles on the bar

    def targets(u):
        out = []
        fh = mk.smooth(min(1.0, u / reach))                                  # the hands first
        for side in ("L", "R"):
            s, end = hands[side]
            if u < reach or end[0] is end[-1]:
                arc = np.array([0.0, 0.0, 0.3 * math.sin(math.pi * fh)])
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
        if sloped:
            f_xy, f_z = 0.8 * f_a + 0.2 * f_c, f_c
            z = (1 - f_z) * b0[2] + f_z * bc[2]
        else:                                                              # up on the planted feet as far as the legs stand, the rest as they step
            f_xy = f_b
            z_stand = min(bc[2], hum.base_height(kin) - 0.03)
            z = b0[2] + f_a * (z_stand - b0[2]) + f_b * (bc[2] - z_stand)
        xy = (1 - f_xy) * b0[:2] + f_xy * bc[:2]
        return xy[0], xy[1], z, gm.PELVIS_PITCH + f_a * (pitch - gm.PELVIS_PITCH)

    windows = {"L_hand": (0.0, hands_end), "R_hand": (0.0, hands_end)}
    extra = _extras(_spine_extra(kin, CLIMB_SPINE, q_quad, blend=True), _fingers_extra(kin, windows, holds, q_quad, from_start=True))
    return gt.contact_motion(kin, name, duration, targets, base, q_init=q_quad, extra=extra, reg=IK_REG, iters=IK_ITERS,
                             source="kerchak_scenarios.mount")


VERTICAL_TRIALS = tuple((s, phi) for s in (1.0, -1.0) for phi in (0.0, 0.4, -0.4, 0.8, -0.8, 1.2, -1.2, 1.5, -1.5))


def _pick_frames(kin, q_quad, holds, base, pitch, trials):
    """{hand: (sign, phi)} for the hands' frames on vertical bars: each hand's candidates solved statically at its
    hold (the feet on theirs), the one with the smallest IK error kept."""
    limbs = [j for j in kin.actuated if "spine" not in j and j != "waist_yaw" and "finger" not in j and "thumb" not in j]
    R, b = gm.base_rot(pitch), np.asarray(base, dtype=float)
    q = _spine_extra(kin, CLIMB_SPINE)(1.0, np.asarray(q_quad, dtype=float).copy())
    feet = foot_targets(kin, "L", holds["L"]) + foot_targets(kin, "R", holds["R"])
    to_base = lambda ts: [(bd, l, R.T @ (np.asarray(w) - b)) for bd, l, w in ts]
    q_feet, _ = kin.ik(to_base(feet), q_init=q, q_ref=q_quad, joints=limbs, reg=3e-4, iters=400)
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


def _climb_sequence(kin, q_quad, hold, base, cycles, *, stop_x, pitch=CLIMB_PITCH, v=0.25, trials=((1.0, 0.0),)):
    """The approach to ``stop_x``, the mount and ``cycles`` climb cycles: ``hold(k)`` → {contact: hold} and
    ``base(k)`` → the pelvis's xyz for cycle ``k``. ``trials``: (sign, phi) candidates for each hand's frame on a
    vertical bar (``_pick_frames``)."""
    segs = [("gait", max(stop_x, 0.0) / v, 0.0)]
    frames = _pick_frames(kin, q_quad, hold(0), base(0), pitch, trials) if len(trials) > 1 else {"L_hand": trials[0], "R_hand": trials[0]}

    def framed(h):
        for g in ("L_hand", "R_hand"):
            h[g]["sign"], h[g]["phi"] = frames[g]
        return h

    m = mount(kin, q_quad, stop_x, framed(hold(0)), base(0), pitch=pitch)
    m.params = {**(m.params or {}), "frames": {g: list(v) for g, v in frames.items()}}
    segs.append(("motion", m))
    q = gt._last(kin, m, q_quad)
    for k in range(cycles):
        hk, hk1 = framed(hold(k)), framed(hold(k + 1))
        c = climb_cycle(kin, q, (hk, hk1), base(k), base(k + 1), pitch=pitch, name=f"climb_{k + 1}")
        segs.append(("motion", c))
        q = gt._last(kin, c, q)
    return segs


def scenario_wall(kin, q_quad, p):
    """The rungs: the hands close on a rung four above the feet's, every limb up one rung a cycle."""
    xb, r, sp = p["x"] - p["rung_out"], p["rung_radius"], p["rung_spacing"]
    zf, zh = p["first_rung"], p["first_rung"] + 4 * sp
    hold = lambda k: {"L_hand": bar((xb, 0.36, zh + k * sp), r=r), "R_hand": bar((xb, -0.36, zh + k * sp), r=r),
                      "L": bar((xb, 0.2, zf + k * sp), r=r), "R": bar((xb, -0.2, zf + k * sp), r=r)}
    base = lambda k: np.array([xb - CLIMB_BACK, 0.0, zf + k * sp + CLIMB_UP])
    return _climb_sequence(kin, q_quad, hold, base, int(p["cycles"]), stop_x=xb - 0.75)


def scenario_tree(kin, q_quad, p):
    """The trunk: the hands round its flanks (a vertical hold each side, the fingers hooked round the bark), the feet
    on the branches, a branch a cycle."""
    x, r, sp, br = p["x"], p["radius"], p["branch_spacing"], p["branch_radius"]
    z0 = p["first_branch"]
    ang = math.radians(35.0)                                               # the hands' hold on the flanks: this far round from the near face

    def hold(k):
        zf, zh = z0 + k * sp, z0 + k * sp + 0.95
        out = {}
        for side, sy in (("L", 1.0), ("R", -1.0)):
            n = np.array([-math.cos(ang), sy * math.sin(ang), 0.0])
            out[f"{side}_hand"] = bar((x, 0.0, zh) + r * n, axis=(0.0, 0.0, 1.0), n=n, r=BAR_R)
            out[side] = bar((x, sy * (r * 0.8 + 0.04), zf + 0.01), r=br)
        return out

    base = lambda k: np.array([x - r - CLIMB_BACK, 0.0, z0 + k * sp + CLIMB_UP])
    return _climb_sequence(kin, q_quad, hold, base, int(p["cycles"]), stop_x=x - r - 0.75, trials=VERTICAL_TRIALS)


def scenario_hull(kin, q_quad, p):
    """The ship's side: the ribs across the plate are the rungs, the hands two ribs above the feet."""
    slope = math.radians(p["slope_deg"])
    pitch = math.pi / 2 - slope + CLIMB_PITCH                              # the trunk along the plate
    n = np.array([-math.sin(slope), 0.0, math.cos(slope)])                 # the plate's normal, outward
    s0, rung = 0.3, p["rib_spacing"]
    rib = lambda s: np.array([p["x"] + s * math.cos(slope), 0.0, s * math.sin(slope)]) + p["rib_out"] * n

    def hold(k):
        sh, sf = s0 + (k + 2) * rung, s0 + k * rung
        return {"L_hand": bar(rib(sh) + [0.0, 0.36, 0.0], n=n, r=0.03), "R_hand": bar(rib(sh) + [0.0, -0.36, 0.0], n=n, r=0.03),
                "L": bar(rib(sf) + [0.0, 0.2, 0.0], n=n, r=0.03), "R": bar(rib(sf) + [0.0, -0.2, 0.0], n=n, r=0.03)}

    base = lambda k: rib(s0 + k * rung) + CLIMB_UP * _unit(np.array([math.cos(slope), 0.0, math.sin(slope)])) + CLIMB_BACK * n
    return _climb_sequence(kin, q_quad, hold, base, int(p["cycles"]), stop_x=p["x"] - 0.75, pitch=pitch)


def scenario_rope(kin, q_quad, p):
    """The rope: a vertical bar the hands close on one above the other, the feet clamping it below."""
    x, rung, rr = p["x"], 0.3, p["radius"]

    def hold(k):
        zf = 0.35 + k * rung
        zh = zf + CLIMB_UP + 0.45                                          # about the shoulders' height
        return {"L_hand": bar((x, 0.0, zh + 0.15), axis=(0.0, 0.0, 1.0), r=rr), "R_hand": bar((x, 0.0, zh - 0.15), axis=(0.0, 0.0, 1.0), r=rr),
                "L": point((x - 0.05, 0.1, zf), pitch=0.0), "R": point((x - 0.05, -0.1, zf), pitch=0.0)}   # the feet pinch the rope, soles level

    base = lambda k: np.array([x - CLIMB_BACK, 0.0, 0.35 + k * rung + CLIMB_UP])
    return _climb_sequence(kin, q_quad, hold, base, int(p["cycles"]), stop_x=x - 0.75, trials=VERTICAL_TRIALS)


def scenario_window(kin, q_quad, p):
    """Through the opening: the hands close on the sill's edge, a press-up on the arms lifts the body above the
    sill with the feet hanging (the wall is smooth: nothing for a foot), each foot comes up over the edge onto the
    sill, a crouch on it, then a leap off - the hands open, the pelvis on a parabola to the ground beyond, the
    limbs tucked and then reaching for the knuckle stance (the legs are shorter than the sill is high: there is no
    reaching the ground from it, so the frame drops as a gorilla would) - and the walk up to the table."""
    x, sill, t = p["x"], p["sill"], p["wall_thickness"]
    v, stop_x = 0.25, x - 0.95
    top = sill + 0.06
    edge = lambda sy: bar((x - t / 2 - 0.05, sy * 0.36, top - 0.03), r=0.03, wrist_dir=LEDGE_WRIST)   # the ledge's outer edge, taken from above
    holds = {"L_hand": edge(1.0), "R_hand": edge(-1.0), "L": point((x, 0.2, top), pitch=0.0, over=True), "R": point((x, -0.2, top), pitch=0.0, over=True)}
    crouch = np.array([x - 0.2, 0.0, top + 0.25])                          # a deep squat on the sill: the legs folded, the head under the lintel (the opening is 1.3 m tall)
    duck = CLIMB_PITCH                                                     # the trunk as on the climb: tilting it forward would tilt the hips out of their range
    press = np.array([x - 0.55, 0.0, top + 0.35])                          # pressed up on the arms, the body above the sill, the feet hanging
    b0 = np.array([stop_x, 0.0, gm.PELVIS_H])
    start = _contacts(kin, q_quad, b0, gm.PELVIS_PITCH)
    pts = gm._contact_points(kin)
    hands = {}
    for side in ("L", "R"):
        end = hand_targets(kin, side, holds[f"{side}_hand"])
        hands[side] = (_world_pts(kin, q_quad, b0, gm.PELVIS_PITCH, [(bd, l) for bd, l, _ in end]), end)
    A, B, C = (0.0, 0.3), (0.3, 0.55), {"L": (0.55, 0.72), "R": (0.72, 0.9)}

    def hanging(side, bz):                                                 # a foot dangling under the pelvis, sole level
        return point(np.array([bz[0] + 0.1, 0.2 if side == "L" else -0.2, bz[2] - 0.62]), pitch=0.0)

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
                c = foot_swing(hang, holds[side], ff, lift=0.12, clear=0.1)
                goal = foot_targets(kin, side, holds[side], c)
                out += [goal[0]] if ff < SWING_FREE else goal
        return out

    def b_up(u):
        fa = mk.smooth(min(1.0, u / A[1]))                                   # up on the feet, pitched to the climbing frame
        fb = mk.smooth(max(0.0, min(1.0, (u - B[0]) / (B[1] - B[0]))))       # the press-up
        fd = mk.smooth(max(0.0, (u - 0.85) / 0.15))                          # the crouch as the feet come onto the sill
        z_stand = hum.base_height(kin) - 0.03
        p1 = np.array([stop_x, 0.0, z_stand])
        bz = b0 + fa * (p1 - b0) + fb * (press - p1) + fd * (crouch - press)
        return bz[0], bz[1], bz[2], gm.PELVIS_PITCH + fa * (CLIMB_PITCH - gm.PELVIS_PITCH) + fd * (duck - CLIMB_PITCH)

    windows = {"L_hand": (0.0, 0.3), "R_hand": (0.0, 0.3)}
    extra = _extras(_spine_extra(kin, CLIMB_SPINE, q_quad, blend=True), _fingers_extra(kin, windows, holds, q_quad, from_start=True))
    up = gt.contact_motion(kin, "onto_the_sill", 6.0, t_up, b_up, q_init=q_quad, extra=extra, reg=IK_REG, iters=IK_ITERS,
                           source="kerchak_scenarios.window: the hands on the edge, a press-up, the feet over the edge onto the sill")
    q1 = gt._last(kin, up, q_quad)
    tuck = q_quad.copy()
    for side in ("L", "R"):
        tuck[kin.index[f"{side}_knee"]] = min(q_quad[kin.index[f"{side}_knee"]] + 0.5, kin.hi[kin.index[f"{side}_knee"]])
        tuck[kin.index[f"{side}_hip_pitch"]] = q_quad[kin.index[f"{side}_hip_pitch"]] - 0.4
        tuck[kin.index[f"{side}_shoulder_pitch"]] = q_quad[kin.index[f"{side}_shoulder_pitch"]] - 0.6
        tuck[kin.index[f"{side}_elbow"]] = q_quad[kin.index[f"{side}_elbow"]] - 0.6
    release = q1.copy()
    for side in ("L", "R"):
        sr.set_curl(release, kin, side, CURL_OPEN)
    x_land = x + 1.3

    def base(u):                                                           # a dive off the sill: no rise under the lintel
        f = mk.smooth(u)
        flight = math.sin(math.pi * u)
        return crouch[0] + f * (x_land - crouch[0]), 0.0, (1 - f) * crouch[2] + f * gm.PELVIS_H + 0.05 * flight, duck + f * (gm.PELVIS_PITCH - duck)

    keys = [(0.0, q1), (0.15, release), (0.45, tuck), (0.75, tuck), (1.0, q_quad)]
    leap = _joint_track(kin, "leap_off_the_sill", 1.6, keys, base, source="kerchak_scenarios.window: the leap off the sill onto four")
    table_x = x + p.get("table_x", 3.4)
    walk = max((table_x - 0.4 - 0.55 - x_land) / v, 1.0)                  # up to the table: the knuckles a hand's length short of its edge
    return [("gait", stop_x / v, 0.0), ("motion", up), ("motion", leap), ("gait", walk, 0.0)]


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
    v = 0.25
    segs, x_prev = [], 0.0
    crouch = q_quad.copy()
    tuck = q_quad.copy()
    for side in ("L", "R"):
        crouch[kin.index[f"{side}_knee"]] = min(q_quad[kin.index[f"{side}_knee"]] + 0.35, kin.hi[kin.index[f"{side}_knee"]])
        crouch[kin.index[f"{side}_elbow"]] = q_quad[kin.index[f"{side}_elbow"]] - 0.3
        tuck[kin.index[f"{side}_knee"]] = min(q_quad[kin.index[f"{side}_knee"]] + 0.5, kin.hi[kin.index[f"{side}_knee"]])
        tuck[kin.index[f"{side}_hip_pitch"]] = q_quad[kin.index[f"{side}_hip_pitch"]] - 0.4
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
            return take_off + f * 1.8, 0.0, gm.PELVIS_H - crouch_dip + apex * flight, gm.PELVIS_PITCH - 0.25 * flight

        keys = [(0.0, q_quad), (0.25, crouch), (0.45, tuck), (0.8, tuck), (1.0, q_quad)]
        segs.append(("motion", _joint_track(kin, f"jump_{k + 1}", 1.8, keys, base, source="kerchak_scenarios.obstacles: the leap")))
        x_prev = xb + 0.9
    segs.append(("gait", 4.0, 0.0))
    return segs


def scenario_door(kin, q_quad, p):
    v = 0.25
    segs = [("gait", gt.STOP_X / v, 0.0)]
    segs += [("motion", m) for m in gt.door_sequence(kin, q_quad)]
    segs.append(("gait", 11.0, 0.0))
    return segs


DEFAULTS = {"wall": WALL, "tree": TREE, "hull": HULL, "rope": ROPE, "window": WINDOW, "obstacles": OBSTACLES, "door": gt.DOOR}
BUILDERS = {"wall": wall_prop, "tree": tree_prop, "hull": hull_prop, "rope": rope_prop, "window": window_prop, "obstacles": obstacles_prop}
SCENARIOS = {"wall": scenario_wall, "tree": scenario_tree, "hull": scenario_hull, "rope": scenario_rope, "window": scenario_window,
             "obstacles": scenario_obstacles, "door": scenario_door}


def sequence(name, kin, q_quad, params=None):
    """A scenario's segments: [("gait", seconds, lane) | ("motion", Motion)]."""
    p = {**DEFAULTS[name], **(params or {})}
    return SCENARIOS[name](kin, q_quad, p)
