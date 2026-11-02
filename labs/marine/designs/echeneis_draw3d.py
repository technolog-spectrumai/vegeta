"""Echeneis in 3D (matplotlib, from the CadQuery tessellation) on a patch of hull, and its sizing.

    python echeneis_draw3d.py [out.png]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from echeneis import REQUIREMENTS, Echeneis

COLOURS = {"chassis": "#f39c12", "tracks": "#2c3e50", "head": "#16a085", "suction": "#7f8c8d", "housing": "#95a5a6",
           "thrusters": "#202020", "tether": "#c0392b"}


def _tris(shape, tol=3.0):
    verts, tris = shape.val().tessellate(tol, 0.5)
    v = np.array([(q.x, q.y, q.z) for q in verts])
    return [v[list(t)] for t in tris]


def draw(ax, p, elev=28, azim=-55):
    d = Echeneis()
    parts = {"chassis": d._chassis(p), "tracks": d._track(p, 1).union(d._track(p, -1)), "head": d._head(p),
             "suction": d._suction(p), "housing": d._housing(p), "thrusters": d._swim_thrusters(p), "tether": d._tether(p)}
    for name, shp in parts.items():
        c = COLOURS[name]
        ax.add_collection3d(Poly3DCollection(_tris(shp), facecolors=c, edgecolors=c, linewidths=0.0, shade=True))
    X, Y = np.meshgrid(np.linspace(-1100, 1300, 2), np.linspace(-900, 900, 2))
    ax.plot_surface(X, Y, np.zeros_like(X) - 1, color="#7b241c", alpha=0.25)             # the hull plate (antifouling red)
    ax.set_xlim(-1100, 1300); ax.set_ylim(-1200, 1200); ax.set_zlim(-200, 1000)
    ax.set_box_aspect((2.4, 2.4, 1.2))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x crawl [mm]"); ax.set_ylabel("y [mm]"); ax.set_zlabel("out of the hull [mm]")


def main(out="echeneis_3d.png"):
    p = Echeneis.p()
    s = Echeneis.sizing()
    fig = plt.figure(figsize=(18, 9.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    draw(ax, p)
    ax.set_title("Echeneis on the hull (red): magnetic tracks, cleaning head with capture hood (green),\n"
                 "suction thruster, swim thrusters, tether + return hose (red)")
    tx = fig.add_subplot(1, 2, 2)
    tx.axis("off")
    g = s["grip"]
    rows = [
        ["ships", REQUIREMENTS["ships"]],
        ["mass", f"{s['mass_in_air_kg']:.0f} kg in air, {s['in_water_kg']:g} kg heavy in water"],
        ["size", f"{(p['length'] + p['head_length']) / 1000:.2f} x {2 * p['brush_diameter'] * 0.95 / 1000 + 0.06:.2f} x {p['height'] / 1000:.2f} m (with the head)"],
        ["grip, vertical side", f"margin {g['vertical side, crawling up']['margin']:+.2f} in {REQUIREMENTS['current_m_s']:g} m/s (x2 safety)"],
        ["grip, flat bottom", f"margin {g['flat bottom, upside down']['margin']:+.2f} upside down"],
        ["composite hulls", f"suction only, inspection at 0.5 m/s: margin {g['composite hull, inspection at 0.5 m/s']['margin']:+.2f}"],
        ["cleaning", f"{s['clean_rate_m2_h']:.0f} m2/h ({2 * p['brush_diameter'] / 1000:.1f} m swath at {p['crawl_speed']:g} m/s)"],
    ] + [[f"  {k}", f"{h['hours']:.0f} h alone, {h['robots_for_window']} robot(s) in 12 h"] for k, h in s["hulls"].items()] + [
        ["capture", f"{s['capture_flow_L_s']:.0f} L/s up a {p['hose_diameter']:.0f} mm hose ({s['hose_head_m']:.1f} m head): pump {s['pump_W'] / 1000:.1f} kW"],
        ["power", f"{s['power_W']['total'] / 1000:.1f} kW down the tether at {p['tether_voltage']:.0f} V DC ({s['tether_current_A']:.1f} A)"],
        ["inspection", "UT thickness + CP probe, 4 cameras, photogrammetry; Jetson AGX Orin"],
    ]
    tab = tx.table(cellText=rows, colLabels=["", "Echeneis v0"], loc="upper center", cellLoc="left", colWidths=[0.28, 0.72])
    tab.auto_set_font_size(False); tab.set_fontsize(10); tab.scale(1, 1.8)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    return out


if __name__ == "__main__":
    print(main(*sys.argv[1:]))
