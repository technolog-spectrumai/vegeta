"""Squilla in 3D (matplotlib, from the CadQuery tessellation) clamped to a cable with the hold arm and cutting with
the work arm, its sizing, and the vision configurations compared.

    python squilla_draw3d.py [out.png]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from squilla import ASPECTS, DEFAULT_SENSORS, REQUIREMENTS, SENSOR_CONFIGS, Squilla

COLOURS = {"body": "#e67e22", "arms": "#8e44ad", "skids": "#2c3e50", "sensors": "#1abc9c"}


def _tris(shape, tol=4.0):
    verts, tris = shape.val().tessellate(tol, 0.5)
    v = np.array([(q.x, q.y, q.z) for q in verts])
    return [v[list(t)] for t in tris]


def draw(ax, p, elev=16, azim=-55):
    d = Squilla()
    for name, shp in {"body": d._body(p), "arms": d._arms(p), "skids": d._skids(p), "sensors": d._sensors(p)}.items():
        c = COLOURS[name]
        ax.add_collection3d(Poly3DCollection(_tris(shp), facecolors=c, edgecolors=c, linewidths=0.0, shade=True))
    zb = -p["diameter"] / 2 - 100
    X, Y = np.meshgrid(np.linspace(-1500, 2000, 2), np.linspace(-1200, 1200, 2))
    ax.plot_surface(X, Y, np.full_like(X, zb), color="#d4b483", alpha=0.4)
    t = np.linspace(-1500, 2000, 60)                                  # an exposed cable across the work site
    ax.plot(np.full_like(t, 1450.0), t * 0.6, np.full_like(t, zb + 40), color="k", lw=4)
    ax.set_xlim(-1500, 2000); ax.set_ylim(-1750, 1750); ax.set_zlim(-1750, 1750)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_zlabel("z [mm]")


def main(out="squilla_3d.png"):
    p = Squilla.p()
    s = Squilla.sizing()
    cmp_ = Squilla.compare_sensors()
    fig = plt.figure(figsize=(19, 11))
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    draw(ax, p)
    ax.set_title("Squilla landed at an exposed cable: hold arm (short) clamps, work arm's raptorial cutter cuts")
    tx = fig.add_subplot(2, 2, 2)
    tx.axis("off")
    rows = [
        ["role", "Nephro's intervention variant: works at a site, not along a 100 km route"],
        ["size, mass", f"{p['length'] / 1000:.1f} m x {p['diameter'] / 1000:.2f} m; {s['mass_in_air_kg']:.0f} kg in air (vision {DEFAULT_SENSORS[0]})"],
        ["arms", f"6-DOF work arm {p['arm_reach'] / 1000:.1f} m, raptorial cutter {p['cut_force_N'] / 1000:.0f} kN in {p['strike_ms']:.0f} ms; 4-DOF hold arm {p['hold_reach'] / 1000:.1f} m"],
        ["thrust", f"2 stern + 8 tunnel: vertical margin {s['vertical_margin']:+.2f} with {REQUIREMENTS['lift_kg_in_water']:g} kg lifted, surge {s['surge_margin']:+.1f} in {REQUIREMENTS['current_m_s']:g} m/s"],
        ["dive", f"{REQUIREMENTS['transit_km']:g} km transit + {REQUIREMENTS['work_h']:g} h work: {s['dive_Wh'] / 1000:.2f} of {s['usable_Wh'] / 1000:.2f} kWh usable ({p['battery_kwh']:g} kWh)"],
        ["jobs", "cut and clear; grab, tag, retrieve; UT / CP / coating by touch"],
        ["compute", f"Jetson AGX Orin: {s['jetson_tops']:.0f} of 275 TOPS with the default vision"],
    ]
    tab = tx.table(cellText=rows, colLabels=["", "Squilla v0"], loc="center", cellLoc="left", colWidths=[0.16, 0.84])
    tab.auto_set_font_size(False); tab.set_fontsize(10); tab.scale(1, 2.0)
    names = list(SENSOR_CONFIGS)
    a3 = fig.add_subplot(2, 2, 3)
    M = np.array([[cmp_[n][a] for a in ASPECTS] for n in names])
    im = a3.imshow(M, cmap="Greens", vmin=0, vmax=2)
    a3.set_xticks(range(len(ASPECTS))); a3.set_xticklabels(ASPECTS, rotation=20)
    a3.set_yticks(range(len(names))); a3.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(ASPECTS)):
            a3.text(j, i, int(M[i, j]), ha="center", va="center", color="white" if M[i, j] == 2 else "black")
    a3.set_title("vision configurations: what each one sees (0 none, 1 some, 2 good)")
    a4 = fig.add_subplot(2, 2, 4)
    price = [cmp_[n]["price EUR"] / 1000 for n in names]
    score = [cmp_[n]["score"] for n in names]
    a4.scatter(price, score, s=[cmp_[n]["power W"] * 2 for n in names], c=["#e67e22" if n == DEFAULT_SENSORS else "#7f8c8d" for n in names])
    for n, x, y in zip(names, price, score):
        a4.annotate(f"{n.split()[0]}  {cmp_[n]['power W']:.0f} W, battery {100 * cmp_[n]['battery margin']:+.0f} %", (x, y),
                    xytext=(6, 4), textcoords="offset points", fontsize=9)
    a4.set_xlabel("vision price [kEUR]"); a4.set_ylabel("total score (of 10)")
    a4.set_title(f"price against capability (marker size = power); default: {DEFAULT_SENSORS}")
    a4.set_xlim(0, 110); a4.set_ylim(5, 11)
    fig.tight_layout()
    fig.savefig(out, dpi=95)
    return out


if __name__ == "__main__":
    print(main(*sys.argv[1:]))
