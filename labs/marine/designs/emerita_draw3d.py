"""Emerita in 3D (matplotlib, from the CadQuery tessellation): cruising over a buried cable, and hovering with the
arm out; and its sizing.

    python emerita_draw3d.py [out.png]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from emerita import REQUIREMENTS, Emerita

COLOURS = {"body": "#f1c40f", "antennae": "#e74c3c", "arm": "#2980b9", "skids": "#2c3e50"}


def _tris(shape, tol=4.0):
    verts, tris = shape.val().tessellate(tol, 0.5)
    v = np.array([(q.x, q.y, q.z) for q in verts])
    return [v[list(t)] for t in tris]


def draw(ax, p, altitude_mm, elev=18, azim=-60):
    d = Emerita()
    for name, shp in {"body": d._body(p), "antennae": d._antennae(p), "arm": d._arm(p), "skids": d._skids(p)}.items():
        c = COLOURS[name]
        ax.add_collection3d(Poly3DCollection(_tris(shp), facecolors=c, edgecolors=c, linewidths=0.0, shade=True))
    zb = -p["diameter"] / 2 - 90 - altitude_mm                               # the seabed under the skids
    X, Y = np.meshgrid(np.linspace(-1800, 1800, 2), np.linspace(-900, 900, 2))
    ax.plot_surface(X, Y, np.full_like(X, zb), color="#d4b483", alpha=0.45)
    xs = np.linspace(-1800, 1800, 50)
    ax.plot(xs, np.zeros_like(xs), np.full_like(xs, zb - 1), "k--", lw=1.5, label="buried cable")
    ax.set_xlim(-1800, 1800); ax.set_ylim(-1800, 1800); ax.set_zlim(-1800, 1800)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_zlabel("z [mm]")


def main(out="emerita_3d.png"):
    p = Emerita.p()
    s = Emerita.sizing()
    fig = plt.figure(figsize=(19, 10))
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    draw(ax, p, REQUIREMENTS["survey_altitude_m"] * 1000 * 0.4)
    ax.set_title("survey: the coil array (red 'antennae') tracks the buried cable, arm stowed")
    ax2 = fig.add_subplot(2, 2, 3, projection="3d")
    draw(ax2, Emerita.p({"arm_pose": "deployed"}), 0.0, elev=6, azim=-90)
    ax2.set_title("at a fault: landed on its skids, the arm out to lift and tag the cable")
    tx = fig.add_subplot(1, 2, 2)
    tx.axis("off")
    hb = s["housings"]["battery"]
    rows = [
        ["cables", REQUIREMENTS["cables"]],
        ["size, mass", f"{p['length'] / 1000:.1f} m x {p['diameter'] / 1000:.2f} m; {s['mass_in_air_kg']:.0f} kg in air, +1 kg buoyant"],
        ["depth", f"{REQUIREMENTS['depth_m']:.0f} m: housings {p['housing_od']:.0f} x {p['housing_wall']:.0f} mm Al, buckling margin {hb['buckling_margin']:+.2f}"],
        ["battery", f"{p['battery_kwh']:g} kWh Li-ion ({s['cells_kg']:.0f} kg of cells)"],
        ["survey", f"{REQUIREMENTS['survey_speed_m_s']:g} m/s at {REQUIREMENTS['survey_altitude_m']:g} m altitude: {s['survey_W']:.0f} W ({s['hotel_W']:.0f} W sensors + {s['propulsion_W']:.0f} W propulsion)"],
        ["range", f"{s['range_km']:.0f} km per dive (keeps {REQUIREMENTS['intervention_h']:g} h of hover work and {100 * REQUIREMENTS['reserve']:.0f} % reserve)"],
    ] + [[f"  at {u:g} m/s", f"{r['Wh_per_km']:.0f} Wh/km, {r['range_km']:.0f} km"] for u, r in s["speeds"].items()] + [
        ["sensors", "3-axis coil array (tone/field), side-scan, multibeam, sub-bottom profiler, cameras"],
        ["jobs", "burial depth, exposures and free spans, fault finding, lift/tag/mark with the arm"],
        ["hover", "2 stern + 4 tunnel thrusters; 5-DOF folding arm in a belly bay; low skids"],
        ["autonomy", "Jetson AGX Orin; INS + DVL, reset by the cable itself and USBL fixes"],
    ]
    tab = tx.table(cellText=rows, colLabels=["", "Emerita v0"], loc="upper center", cellLoc="left", colWidths=[0.18, 0.82])
    tab.auto_set_font_size(False); tab.set_fontsize(10); tab.scale(1, 1.9)
    fig.tight_layout()
    fig.savefig(out, dpi=95)
    return out


if __name__ == "__main__":
    print(main(*sys.argv[1:]))
