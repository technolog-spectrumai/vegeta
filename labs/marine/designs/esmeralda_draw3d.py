"""Esmeralda in 3D (matplotlib, from the CadQuery tessellation): the guard carrier from the aft quarter (the boat bay,
the stern ramp, Barracuda on its cradle, the side hangars), from the bow quarter (the bow water cannon, the hailer and
searchlights, the EO/IR turret), a plan of the deck, and what changed against Zarco.

    python esmeralda_draw3d.py [out.png]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from esmeralda import Esmeralda
from zarco_draw3d import _add, _flush, _tris

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "aero", "designs"))
from fulmar import Fulmar  # noqa: E402
from gannet import Gannet  # noqa: E402

COLOURS = {"hulls": "#34495e", "deck": "#95a5a6", "house": ("#ecf0f1", 0.55), "flight_deck": ("#7f8c8d", 0.6), "pads": "#f1c40f",
           "catapult": "#c0392b", "net": ("#2c3e50", 0.35), "mast": "#d0d3d4", "robots": "#27ae60", "ramp": "#8e44ad",
           "boat": "#e67e22", "effectors": "#2980b9", "turret": "#16a085"}


def draw_vessel(ax, elev, azim, title):
    d, p = Esmeralda(), Esmeralda.p()
    for name, c in COLOURS.items():
        col, alpha = (c, 1.0) if isinstance(c, str) else c
        _add(ax, _tris(d.build({**p, "part": name}), 30.0), col, alpha)
    z_pad = d._z_deck(p) + 3600 + 190
    g = Gannet().build(Gannet.p())
    for x, y in d._pad_at(p)[:2]:
        _add(ax, _tris(g, 15.0, (x, y, z_pad)), "#1f618d")
    f = Fulmar().build(Fulmar.p())
    _add(ax, _tris(f, 15.0, (1800, 0, z_pad + 900), rot=np.pi), "#f39c12")
    _flush(ax)
    X, Y = np.meshgrid(np.linspace(-14000, 14000, 2), np.linspace(-8000, 8000, 2))
    ax.plot_surface(X, Y, np.zeros_like(X), color="#2e86c1", alpha=0.15)
    ax.set_xlim(-12500, 12500); ax.set_ylim(-9000, 9000); ax.set_zlim(-1500, 11000)
    ax.set_box_aspect((25, 18, 12.5), zoom=1.3); ax.view_init(elev=elev, azim=azim); ax.set_axis_off(); ax.set_title(title, fontsize=10)


def plan(ax):
    """The deck's plan at the hangar's level: the boat bay, the side bays with their drones, the mast and the bow."""
    s, p = Esmeralda.sizing(), Esmeralda.p()
    lay = s["layout"]
    ax.add_patch(plt.Rectangle((-12, -5), 24, 10, fc="#d5dbdb", ec="k"))
    ax.add_patch(plt.Rectangle((lay["bay_x0_m"], -p["bay_width"] / 2000), p["bay_length"] / 1000, p["bay_width"] / 1000, fc="#d2b4de", ec="#8e44ad"))
    ax.add_patch(plt.Rectangle((lay["boat_x_m"] - 4.5, -1.44), 9.0, 2.88, fc="#e67e22", ec="k", alpha=0.85))
    ax.text(lay["boat_x_m"], 0, "Barracuda\non its cradle", ha="center", va="center", fontsize=8)
    w = s["hangar"]["side_bay_width_m"]
    y0 = p["bay_width"] / 2000 + 0.3
    for sy, lab in ((1, "port: 4 Gannets + payload workshop"), (-1, "starboard: 2 Fulmars + battery room")):
        yb = y0 if sy > 0 else -y0 - w
        ax.add_patch(plt.Rectangle((-9, yb), 12, w, fc="#fdfefe", ec="k"))
        ax.text(-3, yb + w + 0.3 if sy > 0 else yb - 0.3, lab, ha="center", va="center", fontsize=8)
    for k in range(4):
        ax.add_patch(plt.Rectangle((-8.7 + k * 2.1, y0 + 0.35), 1.79, 1.79, fc="#1f618d", alpha=0.7))
    for k in range(2):
        ax.add_patch(plt.Rectangle((-8.7 + k * 4.0, -y0 - 0.2 - 2.1), 3.6, 2.1, fc="#f39c12", alpha=0.7))
    ax.add_patch(plt.Circle((8.2, 0), 1.2, fc="#d0d3d4", ec="k"))
    ax.text(8.2, 0, "mast\nturret\nhailer", ha="center", va="center", fontsize=7)
    ax.add_patch(plt.Circle((10.3, 0), 0.3, fc="#2980b9"))
    ax.text(10.3, -0.8, "bow cannon", ha="center", fontsize=7)
    ax.annotate("", xy=(-12.0, 0), xytext=(-14.5, 0), arrowprops={"arrowstyle": "->"})
    ax.text(-14.4, 0.4, "boat in / out", fontsize=7)
    ax.set_xlim(-15, 13); ax.set_ylim(-6.5, 6.5); ax.set_aspect("equal"); ax.set_xlabel("x [m] from midship"); ax.set_title("plan at the hangar: the boat bay between the side bays")


def main(out="esmeralda_3d.png"):
    s = Esmeralda.sizing()
    fig = plt.figure(figsize=(20, 13))
    draw_vessel(fig.add_axes([0.0, 0.42, 0.5, 0.53], projection="3d"), 22, -140,
                "from the aft quarter: the boat bay and the 8 deg stern ramp, Barracuda on its cradle (mast folded), the side hangars")
    draw_vessel(fig.add_axes([0.5, 0.42, 0.5, 0.53], projection="3d"), 25, 35,
                "from the bow quarter: the bow water cannon, the hailer and the searchlights on the mast, the EO/IR turret")
    plan(fig.add_axes([0.03, 0.04, 0.45, 0.36]))
    tx = fig.add_axes([0.5, 0.02, 0.48, 0.4])
    tx.axis("off")
    c = s["changed_vs_zarco"]
    e, se = s["effectors"], s["sensors"]
    rows = [["mass", f"{c['mass_t'][0]} t", f"{c['mass_t'][1]} t"],
            ["draught / trim", f"{c['draught_m'][0]} m / level", f"{c['draught_m'][1]} m / {abs(c['trim_deg'][1])} deg by the {s['trim']['trim_by']} (fuel {s['level_by_moving_fuel_m']} m fwd levels it)"],
            ["GM transverse", f"{c['gm_t_m'][0]} m", f"{c['gm_t_m'][1]} m"],
            ["aft deck", "4 AHC Walrus davits, basket gantry", f"boat bay + {s['boat']['ramp']['ramp_length_m']} m ramp: launch {s['boat']['launch_s']:.0f} s, recover {s['boat']['recovery_s']:.0f} s, Hs {s['boat']['hs_limit_m']} m"],
            ["amidships", "processing house, packer, chiller", "the hangar's centre is the boat bay"],
            ["hangar", "8 Gannet, 3 Fulmar, 12 chargers", f"4 Gannet, 2 Fulmar (EO/IR), {s['chargers']} chargers"],
            ["station load", f"{c['station_load_kW'][0]} kW, {c['endurance_days'][0]} days", f"{c['station_load_kW'][1]} kW, {c['endurance_days'][1]} days; peak {s['peak_kW']} kW"],
            ["effectors", "horn, lights (VNUC)", f"hailer {e['hailer']['spl_1m_db']:g} dB ({e['hailer']['intelligible_m']['skiff_at_speed']} m to a fast skiff), 2 x {e['searchlights']['lumens'] / 1000:.0f} klm ({e['searchlights']['range_m']} m), "
                                                   f"bow cannon {e['water_cannon']['flow_l_min']:g} L/min ({e['water_cannon']['throw_m']} m)"],
            ["EO/IR turret", "-", f"skiff ID {se['turret']['dri_skiff_km']['night_clear']['identification']} km at night (clear), {se['turret']['dri_skiff_km']['night_haze']['identification']} km haze, {se['turret']['dri_skiff_km']['night_fog']['detection']} km fog"],
            ["flare launcher / sonar", "-", "not fitted (owner's decision) / not worth it here"],
            ["price", f"EUR {c['price_eur'][0] / 1e6:.2f} M", f"EUR {c['price_eur'][1] / 1e6:.2f} M (+ Barracuda: {s['price_with_barracuda_eur'] / 1e6:.2f} M)"]]
    tab = tx.table(cellText=rows, colLabels=["", "Zarco", "Esmeralda"], loc="center", cellLoc="left", colWidths=[0.14, 0.26, 0.6])
    tab.auto_set_font_size(False); tab.set_fontsize(8.5); tab.scale(1, 1.55)
    fig.suptitle("Esmeralda - Zarco's guard-duty variant: hull, propulsion, mast and flight deck kept; the clam line and Walrus davits out; "
                 "a boat bay for Barracuda, effectors and a long-range EO/IR turret in", fontsize=12)
    fig.savefig(out, dpi=95)
    return out


if __name__ == "__main__":
    print(main(*sys.argv[1:]))
