"""The surveillance pod in 3D on Gannet and Fulmar (matplotlib, from the CadQuery tessellation), the skiff's DRI by
altitude and weather, and the numbers.

    python eoir_payload_draw3d.py [out.png]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import eoir_payload as ep
from fulmar import Fulmar
from gannet import Gannet


def _draw(ax, design, p, parts, lim, title, elev=20, azim=-60):
    tris, cols = [], []
    for part, col in parts:
        verts, tr = design.build(p | {"part": part}).val().tessellate(4.0, 0.4)
        v = np.array([(q.x, q.y, q.z) for q in verts])
        t = [v[list(k)] for k in tr]
        tris += t
        cols += [col] * len(t)
    ax.add_collection3d(Poly3DCollection(tris, facecolors=cols, edgecolors=cols, linewidths=0.0, shade=True))
    (x0, x1), (y0, y1), (z0, z1) = lim
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_zlim(z0, z1)
    ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0), zoom=1.25); ax.view_init(elev=elev, azim=azim); ax.set_axis_off()
    ax.set_title(title, fontsize=10)


def main(out="eoir_payload_3d.png"):
    g, f = Gannet.eoir_sizing(), Fulmar.eoir_sizing()
    fig = plt.figure(figsize=(20, 12))
    _draw(fig.add_axes([0.0, 0.47, 0.33, 0.43], projection="3d"), Gannet(), Gannet.p({"payload": "eoir"}),
          [("body", "#1f618d"), ("arms", "#17202a"), ("rotors", "#aab7b8"), ("skids", "#566573"), ("gimbal", "#16a085")],
          ((-950, 950), (-950, 950), (0, 650)), f"Gannet with its pod (crate off): {g['pod']['mass_kg']} kg, {g['pod']['power_W']:.0f} W", elev=12)
    _draw(fig.add_axes([0.33, 0.47, 0.33, 0.43], projection="3d"), Fulmar(), Fulmar.p({"payload": "eoir"}),
          [("aircraft", "#f39c12"), ("gimbal", "#16a085")], ((-700, 1600), (-1850, 1850), (-400, 400)),
          f"Fulmar with its pod (cargo pod off): {f['pod']['mass_kg']} kg, {f['pod']['power_W']:.0f} W", elev=18, azim=-130)
    ax = fig.add_axes([0.70, 0.55, 0.28, 0.38])
    for drone, s, col in (("Gannet", g, "#1f618d"), ("Fulmar", f, "#f39c12")):
        for alt, ls in zip(s["dri_km"]["skiff_6m"], ("-", "--")):
            r = s["dri_km"]["skiff_6m"][alt]
            for k, (task, mk) in enumerate((("detection", "o"), ("recognition", "s"), ("identification", "^"))):
                ys = [r[c][task] for c in ("night_clear", "night_haze", "night_fog")]
                ax.plot([0, 1, 2], ys, ls, marker=mk, color=col, alpha=0.9 - 0.2 * k, label=f"{drone} {alt} {task}" if True else None)
    ax.set_xticks([0, 1, 2], ["clear", "haze", "fog (150 m layer)"]); ax.set_yscale("log"); ax.set_ylabel("ground range [km]")
    ax.set_title("6 m skiff at night (IR): DRI by weather and altitude", fontsize=10); ax.grid(alpha=0.3); ax.legend(fontsize=6.5, ncol=2)
    tx = fig.add_axes([0.03, 0.03, 0.94, 0.42]); tx.axis("off")
    gs, fs = Gannet.sizing(), Fulmar.sizing()
    gd, fd = g["dri_km"]["skiff_6m"], f["dri_km"]["skiff_6m"]
    rows = [["pod", "HD day 30x + uncooled LWIR 640 (12.4 deg) + LRF 1.5 km, AIS, 2 W video link", "HD day 30x + cooled MWIR 640 (2.5-25 deg) + LRF 5 km, AIS, 5 W video link"],
            ["mass, power, price", f"{g['pod']['mass_kg']} kg, {g['pod']['power_W']:.0f} W, EUR {g['pod']['price_eur'] / 1000:.1f}k (crate 5 kg)",
             f"{f['pod']['mass_kg']} kg, {f['pod']['power_W']:.0f} W, EUR {f['pod']['price_eur'] / 1000:.1f}k (cargo 5 kg)"],
            ["endurance", f"hover {g['endurance_hover_min']} min (cargo {gs['endurance_hover_min']}), loiter {g['endurance_loiter_min']} min at {g['loiter_m_s']} m/s; "
                          f"{g['on_station_min_at_5km']} min on station 5 km out", f"loiter {f['endurance_loiter_min']} min at {f['loiter_m_s']} m/s (cargo cruise {fs['endurance_min']}); "
                                                                                       f"{f['on_station_min_at_30km']} min on station 30 km out"],
            ["range", f"{g['range_km']} km at {Gannet.p()['cruise']:g} m/s", f"{f['range_km']} km at {Fulmar.p()['cruise']:g} m/s"],
            ["skiff, day clear (D/R/I km)", "/".join(str(gd['120m']['day_clear'][k]) for k in ('detection', 'recognition', 'identification')) + " from 120 m",
             "/".join(str(fd['300m']['day_clear'][k]) for k in ('detection', 'recognition', 'identification')) + " from 300 m"],
            ["skiff, night clear", "/".join(str(gd['120m']['night_clear'][k]) for k in ('detection', 'recognition', 'identification')),
             "/".join(str(fd['300m']['night_clear'][k]) for k in ('detection', 'recognition', 'identification'))],
            ["skiff, fog (night)", "/".join(str(gd['300m']['night_fog'][k]) for k in ('detection', 'recognition', 'identification')) + " from 300 m",
             "/".join(str(fd['900m']['night_fog'][k]) for k in ('detection', 'recognition', 'identification')) + " from 900 m"],
            ["datalink", f"{g['datalink']['120m']['range_km']} km ({g['datalink']['120m']['limited_by']}), {g['datalink']['120m']['capacity_mbps']:g} Mbit/s",
             f"{f['datalink']['300m']['range_km']} km ({f['datalink']['300m']['limited_by']}), {f['datalink']['300m']['capacity_mbps']:g} Mbit/s"],
            ["limits (unchanged)", f"wind {g['wind_limit_m_s']} m/s, deck Hs {g['deck_hs_limit_m']} m, light-moderate rain",
             f"wind {f['wind_limit_m_s']} m/s, deck (net) Hs {f['deck_hs_limit_m']} m, light-moderate rain"],
            ["radar option", "-", f"does not fit with the gimbal ({f['radar_option']['with_eoir_kg']} kg > 5 kg); a radar-only pod ({f['radar_option']['radar_only_pod_kg']} kg) fits"]]
    tab = tx.table(cellText=rows, colLabels=["", "Gannet + pod", "Fulmar + pod"], loc="center", cellLoc="left", colWidths=[0.14, 0.43, 0.43])
    tab.auto_set_font_size(False); tab.set_fontsize(9); tab.scale(1, 1.7)
    fig.suptitle("The surveillance pod for the guard carrier's drones: a swap-in for the crate (class values, assumed)", fontsize=13, y=0.985)
    fig.savefig(out, dpi=95)
    return out


if __name__ == "__main__":
    print(main(*sys.argv[1:]))
