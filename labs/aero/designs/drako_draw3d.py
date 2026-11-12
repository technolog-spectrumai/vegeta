"""Drako in 3D with matplotlib (no CadQuery needed): the same parameters and sizing as ``drako.Drako``, drawn as
tubes, plates, discs and boxes, with the spec check against the goal next to it.

    python drako_draw3d.py [out.png]
"""
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from drako import Drako

GOAL = [  # (label, goal text, sizing key, test)
    ("Payload, maximum", "15 kg", None, None),
    ("Endurance, unloaded", "~40 min", "endurance_empty_min", lambda v: v >= 40),
    ("Endurance, 10 kg", "~25 min", "endurance_design_min", lambda v: v >= 25),
    ("T/W at 15 kg", ">= 1.8", "thrust_to_weight_max", lambda v: v >= 1.8),
    ("One motor out, 15 kg", "T/W > 1.2", "motor_out_T_W_max", lambda v: v > 1.2),
    ("Tilt at 80 km/h, empty", "< 30 deg", "tilt_80kmh_empty_deg", lambda v: v < 30),
    ("Tilt at 40 km/h, 15 kg", "< 30 deg", "tilt_40kmh_max_deg", lambda v: v < 30),
    ("Range, unloaded", "20 km", "range_empty_km_at_60kmh", lambda v: v >= 20),
]


def box(ax, c, s, color, alpha=1.0):
    x, y, z = c
    dx, dy, dz = (v / 2 for v in s)
    v = np.array([[x + a * dx, y + b * dy, z + e * dz] for a in (-1, 1) for b in (-1, 1) for e in (-1, 1)])
    faces = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
    ax.add_collection3d(Poly3DCollection([v[f] for f in faces], facecolor=color, edgecolor="k", lw=0.3, alpha=alpha))


def tube(ax, a, b, r, color, n=16):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = (b - a) / np.linalg.norm(b - a)
    u = np.cross(d, [0, 0, 1] if abs(d[2]) < 0.9 else [1, 0, 0])
    u /= np.linalg.norm(u)
    w = np.cross(d, u)
    t = np.linspace(0, 2 * np.pi, n)
    ring = r * (np.outer(np.cos(t), u) + np.outer(np.sin(t), w))
    X = np.stack([a + ring, b + ring])
    ax.plot_surface(X[..., 0], X[..., 1], X[..., 2], color=color, shade=True, lw=0)


def disc(ax, c, r, color, alpha=1.0, h=0.0):
    t = np.linspace(0, 2 * np.pi, 48)
    pts = [[c[0] + r * math.cos(a), c[1] + r * math.sin(a), c[2]] for a in t]
    ax.add_collection3d(Poly3DCollection([pts], facecolor=color, edgecolor="none", alpha=alpha))
    if h:
        tube(ax, c, (c[0], c[1], c[2] + h), r, color)


def draw(ax, p):
    R, t, gap = p["wheelbase"] / 2, p["plate_thickness"], p["plate_gap"]
    arm_z = t + gap / 2
    rp = p["prop_diameter"] * 25.4 / 2
    # plates (hexagons)
    rh = p["plate_diameter"] / 2
    for z in (t / 2, t * 1.5 + gap):
        hexa = [[rh * math.cos(math.radians(30 + 60 * k)), rh * math.sin(math.radians(30 + 60 * k)), z] for k in range(6)]
        ax.add_collection3d(Poly3DCollection([hexa], facecolor="#222", edgecolor="k"))
    # arms, motors, propellers
    for k in range(6):
        a = math.radians(60 * k)
        c, s = math.cos(a), math.sin(a)
        tube(ax, (0.3 * rh * c, 0.3 * rh * s, arm_z), (R * c, R * s, arm_z), p["arm_diameter"] / 2, "#333")
        mz = arm_z + p["arm_diameter"] / 2
        disc(ax, (R * c, R * s, mz), 50, "#b03030", h=45)
        disc(ax, (R * c, R * s, mz + 60), rp, "#4a90d9" if k else "#e0a030", alpha=0.25)
    # battery and avionics
    top = 2 * t + gap
    box(ax, (0, 0, top + 55), (300, 160, 110), "#2e7d32")
    tube(ax, (-60, 0, top + 110), (-60, 0, top + 180), 4, "#555")                 # GNSS mast
    disc(ax, (-60, 0, top + 180), 35, "#888")
    # landing gear
    h, sy, L = p["leg_height"], p["leg_spread"] / 2, p["skid_length"]
    rl = p["leg_diameter"] / 2
    for side in (1, -1):
        tube(ax, (-L / 2, side * sy, -h + rl), (L / 2, side * sy, -h + rl), rl, "#444")
        for sx in (0.35 * L, -0.35 * L):
            tube(ax, (sx * 0.45, side * rh * 0.6, 0), (sx, side * sy, -h + rl), rl * 0.8, "#444")
    # cargo cradle with latches, and the package
    X, Y, b = p["cradle_x"], p["cradle_y"], p["cradle_bar"]
    zc = -h + p["package_z"] + b
    for (x0, y0), (x1, y1) in [((-X / 2, -Y / 2), (X / 2, -Y / 2)), ((X / 2, -Y / 2), (X / 2, Y / 2)),
                               ((X / 2, Y / 2), (-X / 2, Y / 2)), ((-X / 2, Y / 2), (-X / 2, -Y / 2))]:
        tube(ax, (x0, y0, zc - b / 2), (x1, y1, zc - b / 2), b / 2, "#777")
    for sx in (1, -1):
        for side in (1, -1):
            tube(ax, (sx * 0.3 * X, side * Y / 2, zc), (sx * 0.3 * X, side * Y / 2, 0), b / 2, "#777")
            box(ax, (sx * 0.25 * X, side * (Y / 2 + b + 10), zc - b / 2), (40, 20, 38), "#d32f2f")
            box(ax, (sx * 0.25 * X, side * (Y / 2 - 3), zc - b - 20), (30, 6, 40), "#d32f2f")
    box(ax, (0, 0, -h + p["package_z"] / 2), (p["package_x"], p["package_y"], p["package_z"]), "#c8a165", alpha=0.9)
    lim = R + rp
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-h, -h + 2 * lim * 0.6)
    ax.set_box_aspect((1, 1, 0.6))
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_zlabel("z [mm]")


def main(out="drako_3d.png"):
    p = {q.name: q.default for q in Drako.parameters}
    s = Drako.sizing(p)
    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    draw(ax, p)
    ax.view_init(elev=22, azim=-55)
    ax.set_title("Drako hexacopter, 15 kg cargo, 1650 mm, 6 x 30 in (orange disc: front)")
    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    draw(ax2, p)
    ax2.view_init(elev=0, azim=-90)
    ax2.set_title("side view (package latched, landed)")
    tx = fig.add_subplot(2, 2, 4)
    tx.axis("off")
    rows = [[lbl, goal, "" if key is None else f"{s[key]}",
             "OK" if key is None else ("OK" if ok(s[key]) else "SHORT")] for lbl, goal, key, ok in GOAL]
    rows[0][2] = f"{p['max_payload']:.0f} kg (MTOW {s['mtow_max_kg']} kg)"
    tab = tx.table(cellText=rows, colLabels=["spec", "goal", "Drako", ""], loc="upper center", cellLoc="left")
    tab.auto_set_font_size(False); tab.set_fontsize(10); tab.scale(1, 1.5)
    tx.text(0, -0.02, f"empty {s['empty_kg']} kg (battery {s['battery_kg']} kg, {s['battery_Wh']} Wh 12S semi-solid)   "
                      f"hover {s['hover_power_empty_W']} / {s['hover_power_design_W']} / {s['hover_power_max_W']} W "
                      f"(0 / 10 / 15 kg)\nendurance = hover x 0.70 mission factor, 15 % reserve; "
                      f"comms: 45 km = datalink choice (e.g. 900 MHz / 1.4 GHz mesh + LTE backup)",
            transform=tx.transAxes, fontsize=9, va="top")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    return out


if __name__ == "__main__":
    print(main(*sys.argv[1:]))
