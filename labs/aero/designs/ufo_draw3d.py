"""The two UFOs in 3D (matplotlib, from the CadQuery tessellation) with the alien Jolly Roger on their flags - the
flag drawn through its exported UVs (``ufo.flag_mesh``), so the render checks the texture's mapping - their turn by
speed, and the time each stays in flak range on its typical passes.

    python ufo_draw3d.py [out.png]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

import ufo
import ufo_flag


def _flag_quads(name, cells=(48, 32)):
    """The flag's faces as a grid of quads coloured from the texture, placed by bilinear interpolation of the mesh's
    corner positions over its UVs (exactly what a renderer does with ``uv``)."""
    m = ufo.flag_mesh(name)
    img = np.asarray(Image.open(ufo_flag.DEFAULT).convert("RGB")) / 255.0
    H, W = img.shape[:2]
    V = np.array(m["vertices"]).reshape(-1, 3)
    UV = np.array(m["uv"]).reshape(-1, 2)
    quads, cols = [], []
    nu, nv = cells
    for face in (0, 1):
        idx = list(range(4 * face, 4 * face + 4))
        corner = {tuple(UV[i]): V[i] for i in idx}
        p = lambda u, v: ((1 - u) * (1 - v) * corner[(0, 0)] + u * (1 - v) * corner[(1, 0)] + u * v * corner[(1, 1)] + (1 - u) * v * corner[(0, 1)])
        for i in range(nu):
            for j in range(nv):
                u0, u1, v0, v1 = i / nu, (i + 1) / nu, j / nv, (j + 1) / nv
                quads.append([p(u0, v0), p(u1, v0), p(u1, v1), p(u0, v1)])
                px = int((u0 + u1) / 2 * (W - 1))
                py = int((1 - (v0 + v1) / 2) * (H - 1))              # the image's rows run down; v runs up
                cols.append(img[py, px])
    return quads, cols


def _shade(tris, cols, light=(0.4, -0.5, 0.75)):
    """Lambert shading by hand (a face's colour times its normal's cosine to the light, with an ambient floor)."""
    L = np.array(light) / np.linalg.norm(light)
    out = []
    for t, c in zip(tris, cols):
        n = np.cross(t[1] - t[0], t[2] - t[0])
        k = np.linalg.norm(n)
        f = 0.45 + 0.55 * abs(n @ L) / k if k > 0 else 0.7
        out.append(tuple(min(1.0, f * x) for x in matplotlib.colors.to_rgb(c)))
    return out


def draw(ax, name, elev, azim):
    tris, cols = [], []
    for part, col in ufo.PART_COLOURS[name].items():
        verts, tr = ufo.build(name, part).val().tessellate(15.0, 0.3)
        v = np.array([(q.x, q.y, q.z) for q in verts]) / 1000
        t = [v[list(k)] for k in tr]
        tris += t
        cols += [col] * len(t)
    q, qc = _flag_quads(name)
    # one collection, so the flag's quads and the craft's faces are depth-sorted together
    ax.add_collection3d(Poly3DCollection(tris + [np.array(x) for x in q], facecolors=_shade(tris, cols) + [tuple(c) for c in qc], edgecolors="none"))
    s = ufo.sizing(name)
    L = max(s["size"][:2])
    zc = s["size"][2] / 2 if name == "saucer" else 0.4
    ax.set_xlim(-L / 2, L / 2); ax.set_ylim(-L / 2, L / 2); ax.set_zlim(zc - L / 2, zc + L / 2)
    ax.set_box_aspect((1, 1, 1), zoom=1.25); ax.view_init(elev=elev, azim=azim); ax.set_axis_off()
    k = s["kinematics"]
    vmax = k["v_max_m_s"] if name == "saucer" else k["v_max_m_s"]["high"]
    ax.set_title(f"{name}: {s['size'][0]:g} x {s['size'][1]:g} x {s['size'][2]:g} m, {s['mass_kg'] / 1000:g} t (class), "
                 f"{vmax:g} m/s max - flies like {'a helicopter' if k['model'] == 'helicopter' else 'a jet'}", fontsize=10, y=0.92)


def main(out="ufo_3d.png"):
    fig = plt.figure(figsize=(20, 12))
    draw(fig.add_axes([0.0, 0.42, 0.33, 0.55], projection="3d"), "saucer", 18, -60)
    draw(fig.add_axes([0.33, 0.42, 0.33, 0.55], projection="3d"), "cigar", 38, -125)
    ax = fig.add_axes([0.70, 0.55, 0.28, 0.36])
    for name, col in (("saucer", "#2e86c1"), ("cigar", "#c0392b")):
        rows = ufo.kinematics(name)["turn_by_speed"]
        for alt, ls in (("low", "-o"), ("high", "--s")):
            r = [x for x in rows if x.get("alt", "low") == alt]
            if r:
                ax.plot([x["v_m_s"] for x in r], [x["rate_deg_s"] for x in r], ls, color=col,
                        label=f"{name}{'' if name == 'saucer' else ' ' + alt}")
    ax.set_xlabel("speed [m/s]"); ax.set_ylabel("best turn rate [deg/s]"); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    ax.set_title("turn by speed: the saucer turns on the spot (40 deg/s), banks 45 deg when fast;\n"
                 "the cigar never below 70 m/s, 2.5 g low / 1.5 g high", fontsize=9)
    tx = fig.add_axes([0.03, 0.03, 0.94, 0.36]); tx.axis("off")
    rows = []
    for name in ufo.CRAFT:
        for p in ufo.flak(name)["passes"]:
            for g, gun in ufo.FLAK.items():
                r0, r1 = p["by_gun"][g]["0m"], p["by_gun"][g]["1000m"]
                rows.append([name, f"{p['what']} ({p['v_m_s']:g} m/s at {p['alt_m']:g} m)", gun["what"], f"{gun['range_m']:g} m",
                             f"{r0['in_range_s']:g} s", f"{r1['in_range_s']:g} s",
                             "-" if r0["track_deg_s"] is None else f"{r0['track_deg_s']:g}", "-" if r0["lead_m"] is None else f"{r0['lead_m']:g} m"])
    tab = tx.table(cellText=rows, colLabels=["craft", "pass", "gun (game assumption)", "range", "in range (overhead)",
                                             "in range (1 km off)", "track deg/s at closest", "lead at closest"],
                   loc="center", cellLoc="left", colWidths=[0.06, 0.24, 0.2, 0.06, 0.11, 0.11, 0.12, 0.1])
    tab.auto_set_font_size(False); tab.set_fontsize(8); tab.scale(1, 1.45)
    fig.suptitle("The alien pirates' UFOs - fictional game craft under the alien Jolly Roger: kinematic limits of the aircraft "
                 "each stands in for, collision primitives, flak figures (no weapons, no AIS)", fontsize=12, y=0.995)
    fig.savefig(out, dpi=90)
    return out


if __name__ == "__main__":
    print(main(*sys.argv[1:]))
