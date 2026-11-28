"""Kerchak (Belisario's bare frame: Talia's skeleton on its joint modules, no armour) with a Roman legionary's
scutum on the left fist and a baseball bat in the right: the CAD, a STEP file of it and a drawing from two sides.
    python kerchak_scutum_bat.py [out_dir]
mm, x forward, y left, z up (``Humanoid.build``'s pose: the forearms bent forward by ``elbow_bend_deg``)."""
import math

import numpy as np
import sys
from pathlib import Path

import cadquery as cq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shape_kit as sk
import space_unit_specs as su
import space_units as sun
import space_units_draw3d as d3
from fantasy_humanoids import _add, hand_frame

#: the scutum: a plain curved oblong (glued wood, painted white, no decoration), ~1.06 m x 0.66 m; inside it a grip
#: bar across, standing off the board on two posts, that the left fist closes round
SCUTUM = {"height": 1060.0, "width": 660.0, "radius": 520.0, "thickness": 14.0, "bar_d": 24.0, "standoff": 115.0}
#: a wooden baseball bat: 34 in (864 mm), the barrel 2.6 in (66 mm), the handle ~25 mm, a knob
BAT = {"length": 864.0, "barrel_d": 66.0, "handle_d": 24.0, "knob_d": 40.0, "grip_from_knob": 75.0, "rest_on_arm": 0.45}

WHITE, STEEL, WOOD = "#f2f0ea", "#55595e", "#c8a165"


def _grip(p, sy):
    """Where the ``sy`` fist's curled fingers close: the hollow's centre where the built frame has it."""
    return tuple(sun.hand_point(p, sy, sun.fist_hole(p)[0]))


def _bat_line(p):
    """(the grip point, the bat's axis from the knob to the barrel) in the right fist: along the fist's hollow."""
    g = np.array(_grip(p, -1))
    h = sun.hand_axis(p, -1, sun.fist_hole(p)[1])
    h /= np.linalg.norm(h)
    return g, (h if h[2] >= 0 else -h)                                    # the barrel up


def _seg_dist(a0, a1, b0, b1):
    """(distance, s, t): the closest points of two segments, at fractions s and t along them."""
    u, v, w = a1 - a0, b1 - b0, a0 - b0
    best = (1e9, 0.0, 0.0)
    for s in np.linspace(0, 1, 41):
        pa = a0 + s * u
        t = float(np.clip(np.dot(pa - b0, v) / np.dot(v, v), 0, 1))
        dd = float(np.linalg.norm(pa - (b0 + t * v)))
        if dd < best[0]:
            best = (dd, s, t)
    return best


def bat_pose(p):
    """The right arm and wrist - (flex, abd, rot, elbow) and (pronation, flexion, deviation) [deg], every angle in
    a human's range (``space_units.arm_limits``, ``wrist_limits``) - that hold the bat the way a batter rests it:
    the handle through the fist's hollow, the barrel laid against the outside of the upper arm (touching it, between
    a tenth and two thirds of the way down from the shoulder), the fist out in front of the chest, the bat clear of
    the head and the trunk. A least-squares fit from a batter's rest pose."""
    from scipy.optimize import minimize
    B, d = BAT, p["arm_diameter"]
    L, gk = B["length"], B["grip_from_knob"]
    la, lw = sun.arm_limits(), sun.wrist_limits(-1)
    keys_a, keys_w = ("flex", "abd", "rot", "elbow"), ("pr", "fl", "dv")
    lo = [la[k][0] for k in keys_a] + [lw[k][0] for k in keys_w]
    hi = [la[k][1] for k in keys_a] + [lw[k][1] for k in keys_w]
    Wt, Dt = p["torso_width"], p["torso_depth"]
    head = np.array([0.0, 0.0, sun.Humanoid.total_height(p) - p["head_height"] / 2])
    pw = su.hand_layout(p, 1.0)["palm"][1]

    def pose(x):
        return dict(p, arm_R=tuple(x[:4]), wrist_R=tuple(x[4:]))

    def cost(x):
        q = pose(x)
        g, a = _bat_line(q)
        S, E = sun.shoulder_point(q, -1), sun.elbow_point(q, -1)
        dd, s, t = _seg_dist(g + a * (0.45 * L - gk), g + a * (L - gk - 30), S, E)
        r_arm = d * (0.65 if t < 0.2 else 0.5)                                # the shoulder's joint module is wider than the arm
        c = ((dd - (r_arm + B["barrel_d"] / 2 + 3)) / 10.0) ** 2              # the barrel touching the shoulder or the upper arm
        c += (max(0.0, t - 0.66) * 20) ** 2                                   # on the shoulder or the upper arm
        c += (max(0.0, Dt / 2 + 40 - g[0]) / 20) ** 2                         # the fist out in front of the chest
        c += (max(0.0, abs(g[1]) - (Wt / 2 + d * 1.2)) / 20) ** 2             # and not far out to the side
        out_ = g + a * (pw / 2 + 10)                                         # the bat beyond the fist
        W_ = sun.hand_point(q, -1, (0.0, 0.0, 0.0))
        fd = _seg_dist(out_, g + a * (L - gk), E, W_)[0]                     # clear of its own forearm
        c += (max(0.0, d * 0.42 + B["barrel_d"] / 2 + 5 - fd) / 10) ** 2
        neck = np.array([0.0, 0.0, head[2] - p["head_height"]])
        nd = _seg_dist(g - a * 30, g + a * (L - gk), np.array([0.0, 0.0, S[2] - p["torso_height"] * 0.8]), neck)[0]
        c += (max(0.0, Wt / 2 - 20 - nd) / 10) ** 2                          # clear of the ribs and the neck
        hd = _seg_dist(g, g + a * (L - gk), head, head + np.array([0, 0, 1e-3]))[0]
        c += (max(0.0, p["head_width"] / 2 + B["barrel_d"] / 2 + 60 - hd) / 10) ** 2
        return c + 1e-4 * float(np.sum((np.asarray(x) - x0) ** 2)) / 100
    x0 = np.array([30.0, 15.0, 20.0, 110.0, 0.0, 0.0, 0.0])
    from scipy.optimize import differential_evolution
    g0 = differential_evolution(cost, list(zip(lo, hi)), seed=1, maxiter=60, popsize=12, tol=1e-6, polish=False)
    best = minimize(cost, g0.x, method="Powell", bounds=list(zip(lo, hi)), options={"maxiter": 4000, "xtol": 0.05})
    x = best.x
    q = pose(x)
    g, a = _bat_line(q)
    dd, s, t = _seg_dist(g + a * (0.45 * L - gk), g + a * (L - gk - 30), sun.shoulder_point(q, -1), sun.elbow_point(q, -1))
    return tuple(float(v) for v in x[:4]), tuple(float(v) for v in x[4:]), dd - (d * (0.65 if t < 0.2 else 0.5) + B["barrel_d"] / 2), t


def scutum(p):
    """The scutum in the left fist: the fist closed round the grip bar on the board's inner side, the bar on two
    posts off the board, the board curved round a vertical axis behind it (towards the body), white and plain.
    Pieces: ``board`` (white) and ``grip`` (the bar and its posts, steel)."""
    d = p["arm_diameter"]
    S = SCUTUM
    g = _grip(p, 1)
    H, W, R, t = S["height"], S["width"], S["radius"], S["thickness"]
    lay = su.hand_layout(p, 1.0)
    pts = [q for f in lay["fingers"] for q in su.finger_chain(f, 1.6)] + list(su.thumb_chain(lay["thumb"], 1.6))
    front = max(sun.hand_point(p, 1, q)[0] for q in pts) + max(f["radius"] for f in lay["fingers"])   # the fist's front in the world
    back = max(g[0] + S["standoff"], front + 15)                       # the board's inner face, clear of the fingers round the bar
    x_front = back + t
    ax = (x_front - R, g[1])                                              # the curve's axis, behind the board (the body's side)
    half = W / 2 / R
    ring = cq.Workplane("XY").circle(R).circle(R - t).extrude(H)
    wedge = cq.Workplane("XY").polyline([(0, 0), (2 * R * math.cos(half), -2 * R * math.sin(half)),
                                         (2 * R * math.cos(half), 2 * R * math.sin(half))]).close().extrude(H)
    zc = max(g[2], H / 2 + 60)                                            # its middle at the fist's height, its foot off the ground
    board = ring.intersect(wedge).translate((ax[0], ax[1], zc - H / 2))
    yb = d * 0.8                                                          # the bar's half length: past the fist's width
    bar = sk.rod((g[0], g[1] - yb, g[2]), (g[0], g[1] + yb, g[2]), sun.fist_hole(p)[2])   # filling the fist's hollow   # along the fist's hollow (across: the left wrist neutral)
    for sy in (-1, 1):
        y_ = g[1] + sy * yb
        x_in = ax[0] + math.sqrt((R - t) ** 2 - (y_ - ax[1]) ** 2)        # the board's inner face at the post
        bar = bar.union(sk.rod((g[0], y_, g[2]), (x_in + 3, y_, g[2]), S["bar_d"] * 0.45)).union(sk.ball((g[0], y_, g[2]), S["bar_d"] * 0.55))
    assert board.val().BoundingBox().zmin > 50, "the scutum's foot clear of the ground"
    return {"board": board, "grip": bar}


def bat(p):
    """The bat in the right fist: the handle (as thick as the fist's hollow, so the fingers close on it all round)
    through the hollow, the knob just below the little finger, the barrel resting against the upper arm
    (``bat_pose``). Piece: ``bat``."""
    B = BAT
    g, a = _bat_line(p)
    L, hd = B["length"], 2 * sun.fist_hole(p)[2]
    prof = [(0, 0), (hd / 2 + 8, 0), (hd / 2 + 8, 14), (hd / 2, 24), (hd / 2, L * 0.36),
            (B["barrel_d"] * 0.42, L * 0.58), (B["barrel_d"] / 2, L * 0.76), (B["barrel_d"] / 2, L - 20), (B["barrel_d"] * 0.38, L), (0, L)]
    body = cq.Workplane("XZ").polyline(prof).close().revolve(360, (0, 0, 0), (0, 1, 0))   # along +z from the knob
    axis = np.cross((0.0, 0.0, 1.0), a)
    ang = math.degrees(math.acos(max(-1.0, min(1.0, a[2]))))
    if np.linalg.norm(axis) > 1e-9:
        body = body.rotate((0, 0, 0), tuple(axis / np.linalg.norm(axis)), ang)
    pw = su.hand_layout(p, 1.0)["palm"][1]
    knob = g - a * (pw / 2 + 24)                                          # the knob just past the fist's little-finger side
    return {"bat": body.translate(tuple(knob))}


FRAMES = {"kerchak": sun.SpaceMarine, "kerchak2": sun.Kerchak2}


def build(frame="kerchak"):
    d = FRAMES[frame]()
    p = d.resolve()
    p["arm_R"], p["wrist_R"], gap, t = bat_pose(p)                         # the right arm posed for the bat
    print(f"{frame}: right arm (flex, abd, rot, elbow) {tuple(round(v) for v in p['arm_R'])}, wrist (pron, flex, dev) "
          f"{tuple(round(v) for v in p['wrist_R'])} deg; the barrel {gap:+.0f} mm off the upper arm, {t:.2f} of the way down")
    parts = {"skeleton": d.build(dict(p, part="skeleton"))}
    for name, shape in d.actuator_parts(p).items():
        parts[f"module_{name}"] = shape
    parts.update({f"scutum_{k}": v for k, v in scutum(p).items()})
    parts.update(bat(p))
    return p, parts


COLOURS = {"skeleton": d3.BODY, "scutum_board": WHITE, "scutum_grip": STEEL, "bat": WOOD}


def main(out_dir=".", frame="kerchak"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p, parts = build(frame)
    asm = cq.Assembly(name=f"{frame}_scutum_bat")
    for name, shape in parts.items():
        col = COLOURS.get(name, "#3b3f45")
        asm.add(shape, name=name, color=cq.Color(*(int(col[i:i + 2], 16) / 255 for i in (1, 3, 5))))
    step = out / f"{frame}_scutum_bat.step"
    asm.save(str(step))
    pcs = [(d3._tris(shape, 4.0), COLOURS.get(name, "#3b3f45"), d3.EDGE if name == "scutum_board" else None) for name, shape in parts.items()]
    fig = plt.figure(figsize=(21, 9))
    for k, (elev, azim, t) in enumerate(((14, 30, "front quarter"), (12, -40, "the bat side"), (10, 160, "behind: the grip bar"))):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        d3.add_pieces(ax, pcs)
        ax.set_xlim(-1150, 1150); ax.set_ylim(-1150, 1150); ax.set_zlim(0, 2300)
        ax.set_box_aspect((1, 1, 1), zoom=1.5); ax.view_init(elev=elev, azim=azim); ax.set_axis_off(); ax.set_title(t, fontsize=10)
    fig.suptitle(f"{sun.Humanoid.total_height(p) / 1000:.2f} m {frame} (the bare frame): a plain white legionary's scutum (1.06 m x 0.66 m) held by its inner grip bar, "
                 "and a 34 in ash baseball bat gripped in the right fist, its barrel resting on the upper arm", fontsize=11)
    fig.tight_layout()
    png = out / f"{frame}_scutum_bat.png"
    fig.savefig(png, dpi=120)
    return step, png


if __name__ == "__main__":
    print(*main(*sys.argv[1:]))
