"""Cyclopterus in 3D (matplotlib, from the CadQuery tessellation) with each of its three tools, gripping a patch of
net, and its sizing.

    python cyclopterus_draw3d.py [out.png]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from cyclopterus import REQUIREMENTS, TOOLS, Cyclopterus

COLOURS = {"hull": "#5d8a3a", "thrusters": "#222222", "fingers": "#e67e22", "tool": "#2980b9", "disc": "#8e44ad"}


def _tris(shape, tol=2.5):
    verts, tris = shape.val().tessellate(tol, 0.5)
    v = np.array([(q.x, q.y, q.z) for q in verts])
    return [v[list(t)] for t in tris]


def draw(ax, p, elev=18, azim=-50):
    d = Cyclopterus()
    th = None
    for at, a in d.thruster_layout(p):
        t = d._thruster(p, at, a)
        th = t if th is None else th.union(t)
    parts = {"hull": d._hull(p), "thrusters": th, "fingers": d._fingers(p), "tool": d._tool(p), "disc": d._disc(p)}
    for name, shp in parts.items():
        c = COLOURS[name]
        ax.add_collection3d(Poly3DCollection(_tris(shp), facecolors=c, edgecolors=c, linewidths=0.0, shade=True))
    xn = p["length"] / 2 + p["finger_length"] * 0.9                           # the net, a vertical mesh ahead
    for v in np.linspace(-600, 600, 13):
        ax.plot([xn, xn], [v, v], [-400, 800], color="#7f8c8d", lw=0.6)
    for h in np.linspace(-400, 800, 13):
        ax.plot([xn, xn], [-600, 600], [h, h], color="#7f8c8d", lw=0.6)
    ax.set_xlim(-600, 800); ax.set_ylim(-700, 700); ax.set_zlim(-450, 850)
    ax.set_box_aspect((1.4, 1.4, 1.3))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_zlabel("z [mm]")


def main(out="cyclopterus_3d.png"):
    fig = plt.figure(figsize=(19, 10))
    for k, tool in enumerate(TOOLS):
        ax = fig.add_subplot(2, 3, k + 1, projection="3d")
        draw(ax, Cyclopterus.p({"tool": tool}))
        ax.set_title(f"tool: {tool} - {TOOLS[tool][4][:60]}", fontsize=9)
    s = Cyclopterus.sizing()
    tx = fig.add_subplot(2, 1, 2)
    tx.axis("off")
    per = {t: Cyclopterus.sizing({"tool": t}) for t in TOOLS}
    rows = [
        ["site", REQUIREMENTS["site"]],
        ["mass, size", f"{s['mass_in_air_kg']:.0f}-{max(v['mass_in_air_kg'] for v in per.values()):.0f} kg in air; 0.8 x 0.6 x 0.42 m; +0.3 kg buoyant, lead keel {s['trim_kg']:.1f}-{max(v['trim_kg'] for v in per.values()):.1f} kg"],
        ["holding on the net", "4 soft fingers pinch the twine: " + ", ".join(f"{t} {v['hold_margin']:+.2f}" for t, v in per.items()) + f" (x2 safety, {REQUIREMENTS['current_m_s']:g} m/s)"],
        ["inspection", f"{s['inspect_rate_m2_h']:.0f} m2/h: a 160 m pen ({s['net_area_m2']:.0f} m2) in {s['pen_inspection_h']:.1f} h, {s['pens_per_charge']:.1f} pen per charge"],
        ["tool shifts", ", ".join(f"{t} {v['shift_tool_work_h']:.1f} h" for t, v in per.items()) + " on a 1.2 kWh charge"],
        ["morts", f"up to {s['morts_trip_kg']:.0f} kg of fish a trip = only {s['morts_in_water_kg']:.1f} kg in water"],
        ["price", f"EUR {s['unit_price_eur']:,.0f} with one tool, {s['price_with_all_tools_eur']:,.0f} with all three; dock EUR {s['dock_eur']:,.0f} a site (target < 40k)"],
        ["autonomy, safety", "Jetson Orin NX; ducted guarded thrusters, rounded hull, dim/red light at night"],
    ]
    tab = tx.table(cellText=rows, colLabels=["", "Cyclopterus v0"], loc="center", cellLoc="left", colWidths=[0.16, 0.84])
    tab.auto_set_font_size(False); tab.set_fontsize(10); tab.scale(1, 1.9)
    fig.tight_layout()
    fig.savefig(out, dpi=95)
    return out


if __name__ == "__main__":
    print(main(*sys.argv[1:]))
