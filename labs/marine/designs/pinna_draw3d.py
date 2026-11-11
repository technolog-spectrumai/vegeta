"""Pinna in 3D (matplotlib, from the CadQuery tessellation): the autonomous drone carrier with Gannets on its pads,
a Fulmar on the catapult and the Walrus davits aft; Gannet and Fulmar on their own.   python pinna_draw3d.py [out.png]"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from pinna import Pinna

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "aero", "designs"))
from fulmar import Fulmar  # noqa: E402
from gannet import Gannet  # noqa: E402

COLOURS = {"hulls": "#34495e", "deck": "#95a5a6", "house": "#ecf0f1", "flight_deck": "#7f8c8d", "pads": "#f1c40f",
           "catapult": "#c0392b", "net": ("#2c3e50", 0.35), "mast": "#d0d3d4", "davits": "#e67e22", "robots": "#27ae60"}


def _tris(shape, tol=30.0, move=(0, 0, 0), rot=0.0):
    verts, tris = shape.val().tessellate(tol, 0.5)
    v = np.array([(q.x, q.y, q.z) for q in verts])
    if rot:
        c, s = np.cos(rot), np.sin(rot)
        v = v @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    v = v + np.array(move)
    return [v[list(t)] for t in tris]


def _add(ax, tris, col, alpha=1.0):
    """Collect the triangles; ``_flush`` draws them as one collection (matplotlib depth-sorts within a collection)."""
    from matplotlib.colors import to_rgba
    acc = ax.__dict__.setdefault("_acc", ([], []))
    acc[0].extend(tris)
    acc[1].extend([to_rgba(col, alpha)] * len(tris))


def _flush(ax):
    tris, cols = ax.__dict__.pop("_acc")
    ax.add_collection3d(Poly3DCollection(tris, facecolors=cols, edgecolors=cols, linewidths=0.0, shade=True))


def draw_vessel(ax, elev, azim, title):
    d, p = Pinna(), Pinna.p()
    for name, c in COLOURS.items():
        col, alpha = (c, 1.0) if isinstance(c, str) else c
        _add(ax, _tris(d.build({**p, "part": name})), col, alpha)
    z_pad = d._z_deck(p) + 3600 + 190
    g = Gannet().build(Gannet.p())
    for x, y in d._pad_at(p)[:3]:
        _add(ax, _tris(g, 15.0, (x, y, z_pad)), "#1f618d")
    f = Fulmar().build(Fulmar.p())
    _add(ax, _tris(f, 15.0, (1800, 0, z_pad + 900), rot=np.pi), "#f39c12")      # nose forward (+x) on the rail
    _flush(ax)
    ax.set_xlim(-12500, 12500); ax.set_ylim(-9000, 9000); ax.set_zlim(-1500, 11000)
    ax.set_box_aspect((25, 18, 12.5), zoom=1.3); ax.view_init(elev=elev, azim=azim); ax.set_axis_off(); ax.set_title(title, fontsize=10)


def draw_drone(ax, shape, col, lim, title):
    _add(ax, _tris(shape, 6.0), col)
    _flush(ax)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim / 2, lim / 2 + 300)
    ax.set_box_aspect((1, 1, 0.5), zoom=1.3); ax.view_init(elev=25, azim=-60); ax.set_axis_off(); ax.set_title(title, fontsize=10)


def main(out="pinna_3d.png"):
    fig = plt.figure(figsize=(20, 12))
    draw_vessel(fig.add_axes([0.0, 0.38, 0.5, 0.55], projection="3d"), 28, -135, "from aft quarter: the motion-compensated Walrus davits (MRU, AHC winch, luff + slew), the basket gantry, the net, the pads (robots green)")
    draw_vessel(fig.add_axes([0.5, 0.38, 0.5, 0.55], projection="3d"), 32, 35, "from the bow: the autonomy mast, the catapult, the crate loader's rail; no wheelhouse, no crew")
    gs, fs = Gannet.sizing(), Fulmar.sizing()
    draw_drone(fig.add_axes([0.05, 0.0, 0.4, 0.38], projection="3d"), Gannet().build(Gannet.p()), "#1f618d", 1200,
               f"Gannet: {gs['mtow_kg']} kg cargo quad, 5 kg crate, radius {gs['radius_km']} km, hover {gs['hover_W']} W, deck Hs {gs['deck_hs_limit_m']} m")
    draw_drone(fig.add_axes([0.55, 0.0, 0.4, 0.38], projection="3d"), Fulmar().build(Fulmar.p()), "#f39c12", 2000,
               f"Fulmar: {fs['mtow_kg']} kg, 3.6 m fixed-wing, 5 kg, radius {fs['radius_km']} km, {fs['wh_per_km']} Wh/km, catapult + net")
    s = Pinna.sizing()
    fig.suptitle(f"Pinna — unmanned, robot-run drone-carrier catamaran ({s['robots']} robots, no crew): {s['length_m']:.0f} x {s['beam_m']:.0f} m, {s['displacement_t']:.0f} t, "
                 f"{s['service_kn']:.0f} kn ({s['service_kW']} kW), {s['bed_load_kW']:.0f} kW on the bed; 4 pads ({s['vtol_recoveries_per_h']:.0f} recoveries/h), "
                 f"catapult {s['fixed_wing_launches_per_h']:.0f}/h, net {s['fixed_wing_recoveries_per_h']:.0f}/h; packs {s['pack_kg_h']} kg/h; davits Hs {s['davit_hs_limit_m']} m (compensated); €{s['price_eur']/1e6:.2f}M", fontsize=12)
    fig.savefig(out, dpi=90)
    return out


if __name__ == "__main__":
    print(main(*sys.argv[1:]))
