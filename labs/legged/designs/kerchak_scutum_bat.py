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


def _rest(p, B):
    """Where the bat's barrel rests: on the outside of the right upper arm, ``rest_on_arm`` of the way down."""
    c, _, _ = hand_frame(p, -1)
    zs = c[2] + p["upper_arm_length"] + p["forearm_length"] * math.cos(math.radians(p["elbow_bend_deg"]))
    return np.array([0.0, c[1] - (p["arm_diameter"] / 2 + B["barrel_d"] / 2 + 4), zs - p["upper_arm_length"] * B["rest_on_arm"]])


def bat_wrist(p):
    """The right wrist (pronation, flexion, deviation) [deg] within a human's ranges (``space_units.wrist_limits``)
    that lines the fist's hollow up with the bat - from the fist to its rest on the upper arm: a grid over the
    ranges, then a finer one round the best."""
    lim = sun.wrist_limits(-1)
    best = (9.0, (0.0, 0.0, 0.0))

    def score(w):
        q = dict(p, wrist_R=w)
        g = np.array(_grip(q, -1))
        h = sun.hand_axis(q, -1, sun.fist_hole(q)[1])
        a = _rest(q, BAT) - g
        return math.acos(min(1.0, abs(np.dot(h, a)) / (np.linalg.norm(h) * np.linalg.norm(a))))
    for step, span in ((10.0, None), (2.0, 10.0)):
        c0 = best[1]
        rng = [np.arange(lim[k][0], lim[k][1] + 1e-6, step) if span is None else
               np.arange(max(lim[k][0], c0[i] - span), min(lim[k][1], c0[i] + span) + 1e-6, step) for i, k in enumerate(("pr", "fl", "dv"))]
        for pr in rng[0]:
            for fl in rng[1]:
                for dv in rng[2]:
                    s_ = score((pr, fl, dv))
                    if s_ < best[0]:
                        best = (s_, (float(pr), float(fl), float(dv)))
    return best[1], math.degrees(best[0])


def scutum(p):
    """The scutum in the left fist: the fist closed round the grip bar on the board's inner side, the bar on two
    posts off the board, the board curved round a vertical axis behind it (towards the body), white and plain.
    Pieces: ``board`` (white) and ``grip`` (the bar and its posts, steel)."""
    d = p["arm_diameter"]
    S = SCUTUM
    g = _grip(p, 1)
    H, W, R, t = S["height"], S["width"], S["radius"], S["thickness"]
    back = g[0] + S["standoff"]                                           # the board's inner face, clear of the fingers round the bar
    x_front = back + t
    ax = (x_front - R, g[1])                                              # the curve's axis, behind the board (the body's side)
    half = W / 2 / R
    ring = cq.Workplane("XY").circle(R).circle(R - t).extrude(H)
    wedge = cq.Workplane("XY").polyline([(0, 0), (2 * R * math.cos(half), -2 * R * math.sin(half)),
                                         (2 * R * math.cos(half), 2 * R * math.sin(half))]).close().extrude(H)
    board = ring.intersect(wedge).translate((ax[0], ax[1], g[2] - H / 2))  # its middle at the fist's height
    yb = d * 0.8                                                          # the bar's half length: past the fist's width
    bar = sk.rod((g[0], g[1] - yb, g[2]), (g[0], g[1] + yb, g[2]), S["bar_d"] / 2)   # along the fist's hollow (across: the left wrist neutral)
    for sy in (-1, 1):
        y_ = g[1] + sy * yb
        x_in = ax[0] + math.sqrt((R - t) ** 2 - (y_ - ax[1]) ** 2)        # the board's inner face at the post
        bar = bar.union(sk.rod((g[0], y_, g[2]), (x_in + 3, y_, g[2]), S["bar_d"] * 0.45)).union(sk.ball((g[0], y_, g[2]), S["bar_d"] * 0.55))
    assert g[2] - H / 2 > 50, "the scutum's foot clear of the ground"
    return {"board": board, "grip": bar}


def bat(p):
    """The bat in the right fist, held hard just above the knob through the curled fingers' hollow, the wrist
    turned (within a human's ranges, ``bat_wrist``) so the hollow runs at the barrel's rest on the outside of the
    upper arm. Piece: ``bat``."""
    B = BAT
    g = np.array(_grip(p, -1))
    a = _rest(p, B) - g
    a /= np.linalg.norm(a)
    L = B["length"]
    prof = [(0, 0), (B["knob_d"] / 2, 0), (B["knob_d"] / 2, 12), (B["handle_d"] / 2, 24), (B["handle_d"] / 2, L * 0.36),
            (B["barrel_d"] * 0.42, L * 0.58), (B["barrel_d"] / 2, L * 0.76), (B["barrel_d"] / 2, L - 20), (B["barrel_d"] * 0.38, L), (0, L)]
    body = cq.Workplane("XZ").polyline(prof).close().revolve(360, (0, 0, 0), (0, 1, 0))   # along +z from the knob
    axis = np.cross((0.0, 0.0, 1.0), a)
    ang = math.degrees(math.acos(max(-1.0, min(1.0, a[2]))))
    if np.linalg.norm(axis) > 1e-9:
        body = body.rotate((0, 0, 0), tuple(axis / np.linalg.norm(axis)), ang)
    knob = g - a * B["grip_from_knob"]
    return {"bat": body.translate(tuple(knob))}


FRAMES = {"kerchak": sun.SpaceMarine, "kerchak2": sun.Kerchak2}


def build(frame="kerchak"):
    d = FRAMES[frame]()
    p = d.resolve()
    p["wrist_R"], miss = bat_wrist(p)                                     # the right wrist turned for the bat
    print(f"{frame}: right wrist (pronation, flexion, deviation) {p['wrist_R']} deg, the fist {miss:.1f} deg off the bat")
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
