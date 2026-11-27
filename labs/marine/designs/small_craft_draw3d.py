"""The four small craft in 3D (matplotlib, from the CadQuery tessellation) at their waterlines, and their speed against
the sea state beside the interceptor Barracuda's.

    python small_craft_draw3d.py [out.png]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import small_craft as sc
from barracuda import Barracuda

COLS = {"fishing_boat": ("#a93226", "#ecf0f1"), "pilot_boat": ("#17202a", "#e67e22"), "yacht": ("#fdfefe", "#7f8c8d"),
        "skiff": ("#d6eaf8", "#566573")}


def draw(ax, name):
    c = sc.CRAFT[name]
    wl = c.get("canoe_draught_m", c["draught_m"]) * 1000
    tris, cols = [], []
    for part, col in zip(("hull", "superstructure"), COLS[name]):
        verts, tr = sc.build(name, part).val().tessellate(10.0, 0.4)
        v = np.array([(q.x, q.y, q.z - wl) for q in verts])
        t = [v[list(k)] for k in tr]
        tris += t
        cols += [col] * len(t)
    ax.add_collection3d(Poly3DCollection(tris, facecolors=cols, edgecolors="#2c3e50", linewidths=0.1, shade=True))
    L = c["length_m"] * 1000
    X, Y = np.meshgrid(np.linspace(-0.1 * L, 1.1 * L, 2), np.linspace(-0.4 * L, 0.4 * L, 2))
    ax.plot_surface(X, Y, np.zeros_like(X), color="#2e86c1", alpha=0.18)
    ax.set_xlim(-0.1 * L, 1.1 * L); ax.set_ylim(-0.6 * L, 0.6 * L); ax.set_zlim(-0.3 * L, 0.9 * L)
    ax.set_box_aspect((1, 1, 1), zoom=1.3); ax.view_init(elev=15, azim=-55); ax.set_axis_off()
    s = sc.sizing(name)
    ax.set_title(f"{name.replace('_', ' ')}: {s['length_m']:g} m, {s['displacement_t']:g} t, {s['max_kn']} kn; "
                 f"RCS ~{s['rcs_m2']['beam']:g} m2; AIS {'yes' if s['ais_typical'] else 'often none'}", fontsize=9)


def main(out="small_craft_3d.png"):
    fig = plt.figure(figsize=(20, 11))
    for k, name in enumerate(sc.CRAFT):
        draw(fig.add_axes([0.0 + 0.25 * k, 0.45, 0.25, 0.5], projection="3d"), name)
    ax = fig.add_axes([0.06, 0.07, 0.55, 0.33])
    for name, col in zip(sc.CRAFT, ("#a93226", "#e67e22", "#7f8c8d", "#2980b9")):
        t = sc.sizing(name)["speed_vs_hs"]
        ax.plot([r["hs_m"] for r in t], [r["kn"] for r in t], "-o", color=col, label=name.replace("_", " "))
    b = Barracuda.sizing()["speed_vs_hs"]
    ax.plot([r["hs_m"] for r in b], [r["sustained_kn"] for r in b], "k--", lw=2, label="Barracuda sustained")
    ax.plot([r["hs_m"] for r in b], [r["sprint_kn"] for r in b], "k:", lw=2, label="Barracuda sprint")
    ax.set_xlabel("Hs [m]"); ax.set_ylabel("speed [kn]"); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    ax.set_title("the contacts' speed against the sea state, and the interceptor's (the skiff: its crew's impacts, 1.5 g)")
    tx = fig.add_axes([0.64, 0.05, 0.35, 0.37]); tx.axis("off")
    rows = []
    for n in sc.CRAFT:
        s = sc.sizing(n)
        short = {"fishing_boat": ("class A", "trawling lights, floodlit deck"), "pilot_boat": ("class A", "white over red, sidelights"),
                 "yacht": ("class B or none", "masthead tricolour, often dim"), "skiff": ("none", "often none")}[n]
        rows.append([n.replace("_", " "), f"{s['crew'][0]}-{s['crew'][1]}", short[0], short[1], f"{s['delta_T_K']:g} K"])
    tab = tx.table(cellText=rows, colLabels=["craft", "crew", "AIS", "lights at night", "IR dT"], loc="center", cellLoc="left",
                   colWidths=[0.15, 0.08, 0.27, 0.42, 0.08])
    tab.auto_set_font_size(False); tab.set_fontsize(8); tab.scale(1, 1.8)
    fig.suptitle("Small craft round the platform at night - the contacts (class values with their basis; no weapons)", fontsize=13)
    fig.savefig(out, dpi=95)
    return out


if __name__ == "__main__":
    print(main(*sys.argv[1:]))
