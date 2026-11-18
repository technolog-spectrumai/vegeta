"""GAVILAN's pictures (notebook 38): the 3D render (matplotlib, from the CadQuery tessellation: flying, from above,
from the side, and standing on its fins) and the three-view with its dimensions.

    python gavilan_drawings.py [folder]        # writes gavilan_3d.png and gavilan_three_view.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import gavilan as gv

COLOURS = {"body": ("#8d99a6", 1.0), "wing": ("#c9a66b", 1.0), "tail": ("#6b4f2a", 1.0), "pylons": ("#3b3f45", 1.0),
           "camera": ("#111111", 1.0), "props": ("#7fb2e5", 0.30)}
FILL = {"boom": "#6b4f2a", "body": "#8d99a6", "wing": "#c9a66b", "fins": "#6b4f2a", "pylons": "#3b3f45", "pods": "#3b3f45", "camera": "#111111"}


def _tris(shape, tol=0.8):
    out = []
    for s in shape.solids().vals():
        verts, tris = s.tessellate(tol, 0.3)
        v = np.array([(q.x, q.y, q.z) for q in verts])
        out += [v[list(t)] for t in tris]
    return out


def draw(ax, p, elev, azim, title, standing=False):
    """One 3D view; ``standing``: nose up on the fins (+x turned to −z, the feet on z = 0)."""
    L = p["length"]
    for name, shape in gv.Gavilan().parts(p).items():
        col, alpha = COLOURS[name]
        if standing:
            shape = shape.rotate((0, 0, 0), (0, 1, 0), 90).translate((0, 0, L))      # +x (aft) to −z: the nose up
        ax.add_collection3d(Poly3DCollection(_tris(shape), facecolors=col, edgecolors=col, linewidths=0.0, alpha=alpha, shade=alpha == 1.0))
    h = L * 0.52
    cx = (0, 0, L / 2) if standing else (L / 2, 0, 0)
    ax.set_xlim(cx[0] - h, cx[0] + h); ax.set_ylim(cx[1] - h, cx[1] + h); ax.set_zlim(cx[2] - h, cx[2] + h)
    ax.set_box_aspect((1, 1, 1), zoom=1.55); ax.view_init(elev=elev, azim=azim); ax.set_axis_off(); ax.set_title(title, fontsize=11)


def render(out="gavilan_3d.png", p=None, caption=None, name="GAVILAN"):
    p = gv.resolve(p)
    L = gv.Gavilan.layout(p)
    fig = plt.figure(figsize=(15, 11))
    views = [(24, -128, "flying, from the front quarter", False, {}), (90, -90, "from above", False, {}),
             (0, -90, "from the side", False, {}), (14, -60, "standing on its fins, the camera on the horizon", True, {"camera_tilt_deg": 90.0})]
    for k, (elev, azim, title, standing, over) in enumerate(views):
        draw(fig.add_subplot(2, 2, k + 1, projection="3d"), {**p, **over}, elev, azim, title, standing)
    fig.suptitle(caption or (f"{name}\nfour fixed motors, two above and two below the wing; span {p['span']:.0f} mm, "
                             f"length {p['length']:.0f} mm, wing {L['S_mm2'] / 1e6:.3f} m² (aspect ratio {L['AR']:.1f}), "
                             f"7-inch propellers; the only servo tilts the camera"), fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def three_view(out="gavilan_three_view.png", p=None, name="GAVILAN"):
    p = gv.resolve(p)
    o, L = gv.outline(p), gv.Gavilan.layout(p)
    fig, ax = plt.subplots(1, 3, figsize=(18, 6.4), gridspec_kw={"width_ratios": [o["length"], o["length"], o["width"] * 1.15]})
    for a, view, label in zip(ax, ("top", "side", "front"), ("top (x aft, y right)", "side (x aft, z up)", "front (y right, z up)")):
        for key in ("boom", "fins", "wing", "pylons", "body", "pods", "camera"):
            for poly in o[view].get(key, []):
                a.fill(poly[:, 0], poly[:, 1], color=FILL[key], alpha=0.55 if key in ("wing", "fins") else 0.9, ec="k", lw=0.6)
        for (x0, y0), (x1, y1) in o[view].get("prop_lines", []):
            a.plot([x0, x1], [y0, y1], color="#1f77b4", lw=2)
        for c in o[view].get("props", []):
            a.plot(c[:, 0], c[:, 1], color="#1f77b4", lw=1.0)
        a.set_aspect("equal"); a.grid(alpha=0.25); a.set_title(label, fontsize=10); a.set_xlabel("[m]")
    m = 0.06
    ax[0].annotate("", (0, -o["span"] / 2 - m), (o["length"], -o["span"] / 2 - m), arrowprops=dict(arrowstyle="<->"))
    ax[0].text(o["length"] / 2, -o["span"] / 2 - m - 0.012, f"length {p['length']:.0f} mm", ha="center", va="top", fontsize=9)
    ax[0].annotate("", (-m, -o["span"] / 2), (-m, o["span"] / 2), arrowprops=dict(arrowstyle="<->"))
    ax[0].text(-m - 0.012, 0, f"span {p['span']:.0f} mm", ha="right", va="center", rotation=90, fontsize=9)
    ax[0].set_xlim(-0.14, o["length"] + 0.04); ax[0].set_ylim(-o["span"] / 2 - 0.12, o["span"] / 2 + 0.05)
    ax[1].annotate("", (o["length"] + 0.03, -L["foot_half_width"] / 1e3), (o["length"] + 0.03, L["foot_half_width"] / 1e3), arrowprops=dict(arrowstyle="<->"))
    ax[1].text(o["length"] + 0.04, 0, f"feet {2 * L['foot_half_width']:.0f} mm", rotation=90, va="center", fontsize=9)
    ax[1].set_xlim(-0.05, o["length"] + 0.09)
    ax[2].text(0, -o["height"] / 2 - 0.03, f"propellers {p['prop_diameter']:.0f} mm, axes at y ±{p['motor_y']:.0f}, z ±{p['motor_z']:.0f} mm\n"
               f"tips: {L['tip_to_wing']:.0f} mm to the wing, {L['tip_to_tip']:.0f} mm to each other, {L['tip_to_body']:.0f} mm to the body",
               ha="center", va="top", fontsize=8)
    ax[2].set_ylim(-o["height"] / 2 - 0.11, o["height"] / 2 + 0.03)
    fig.suptitle(f"{name} three-view — wing {L['S_mm2'] / 1e6:.3f} m², aspect ratio {L['AR']:.2f}, root chord {L['c_root']:.0f} mm, "
                 f"tip chord {L['c_tip']:.0f} mm, body ⌀ {p['body_diameter']:.0f} mm", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main(folder="."):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for key in gv.VARIANTS:
        tag = "" if key == "A" else f"_{key.lower()}"
        out += [render(folder / f"gavilan{tag}_3d.png", gv.variant(key), name=gv.VARIANT_NAME[key]),
                three_view(folder / f"gavilan{tag}_three_view.png", gv.variant(key), name=gv.VARIANT_NAME[key])]
    return out


if __name__ == "__main__":
    print(*main(*sys.argv[1:]), sep="\n")
