"""Barracuda in 3D (matplotlib, from the CadQuery tessellation): the interceptor from the bow quarter and from astern
(waterjets, the recovery eye is at the bow), its speed against the sea state, and its numbers.

    python barracuda_draw3d.py [out.png]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from barracuda import DRAUGHT_CAD_MM, REQUIREMENTS, Barracuda

COLOURS = {"hull": "#7f8c8d", "collar": "#e67e22", "jets": "#2c3e50", "console": "#ecf0f1", "mast": "#bdc3c7",
           "cannon": "#c0392b", "effectors": "#2980b9"}


def _tris(shape, tol=12.0):
    verts, tris = shape.val().tessellate(tol, 0.4)
    v = np.array([(q.x, q.y, q.z - DRAUGHT_CAD_MM) for q in verts])
    return [v[list(t)] for t in tris]


def draw(ax, elev, azim):
    d, p = Barracuda(), Barracuda.p()
    tris, cols = [], []
    for name, c in COLOURS.items():
        t = _tris(d.build(p | {"part": name}))
        tris += t
        cols += [c] * len(t)
    ax.add_collection3d(Poly3DCollection(tris, facecolors=cols, edgecolors=cols, linewidths=0.0, shade=True))
    X, Y = np.meshgrid(np.linspace(-1000, 9500, 2), np.linspace(-2500, 2500, 2))
    ax.plot_surface(X, Y, np.zeros_like(X), color="#2e86c1", alpha=0.18)
    ax.set_xlim(-1000, 9500); ax.set_ylim(-2500, 2500); ax.set_zlim(-800, 5200)
    ax.set_box_aspect((10.5, 5.0, 6.0), zoom=1.35)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def main(out="barracuda_3d.png"):
    s = Barracuda.sizing()
    p = Barracuda.p()
    fig = plt.figure(figsize=(19, 12))
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    draw(ax, 18, -40)
    ax.set_title("from the bow quarter: foam collar, console, folding mast (radar, EO/IR gimbal, antennas), cannon, hailer + light + strobe")
    ax = fig.add_subplot(2, 2, 2, projection="3d")
    draw(ax, 14, -150)
    ax.set_title("from astern: twin waterjets with reversing buckets; the bow eye the carrier's ramp hook takes")

    ax = fig.add_subplot(2, 2, 3)
    t = s["speed_vs_hs"]
    hs = [r["hs_m"] for r in t]
    ax.plot(hs, [r["power_kn"] for r in t], "k:", label="power only (added resistance)")
    ax.plot(hs, [r["sprint_kn"] for r in t], "-o", color="#c0392b", label=f"sprint (impacts <= {p['g_sprint']:g} g)")
    ax.plot(hs, [r["sustained_kn"] for r in t], "-o", color="#2c3e50", label=f"sustained (impacts <= {p['g_sustained']:g} g)")
    ax.plot(hs, [r["at_crewed_1_5g_kn"] for r in t], "--", color="#7f8c8d", label="if it carried a crew (1.5 g)")
    ax.axhline(REQUIREMENTS["sprint_kn"], color="#c0392b", lw=0.6, ls="--")
    ax.plot([REQUIREMENTS["cruise_hs_m"]], [REQUIREMENTS["cruise_kn"]], "s", color="#27ae60", ms=10, label="brief: 25 kn in Hs 1.0 m")
    ax.axvspan(1.25, 2.5, color="#f5b041", alpha=0.15, label="sea state 4")
    ax.set_xlabel("significant wave height Hs [m]"); ax.set_ylabel("speed [kn]"); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    ax.set_title("speed against the sea state (Savitsky calm water, Savitsky & Brown impacts; half load)")

    tx = fig.add_subplot(2, 2, 4)
    tx.axis("off")
    e, m, se, c = s["effectors"], s["manoeuvres"], s["sensors"], s["carrier"]
    dri = se["eoir"]["dri_skiff_km"]
    rows = [
        ["size, mass", f"{s['length_overall_m']} x {s['beam_overall_m']} m, draught {s['draught_m']} m; {s['mass_full_kg'] / 1000:.2f} t full ({s['fuel_L']} L diesel)"],
        ["propulsion", f"2 x {s['engine_kW']:.0f} kW diesels on 2 waterjets; sprint {s['sprint_kn_full_load']} kn, cruise {s['cruise_kn']:g} kn, loiter {s['loiter_kn']:g} kn on one engine"],
        ["endurance", f"{s['range_nm_at_cruise']} nm at {s['cruise_kn']:g} kn (Hs 1 m), {s['loiter_h']} h at {s['loiter_kn']:g} kn, 10 % reserve"],
        ["manoeuvres", f"0-30 kn {m['accel_0_30kn_s']} s; crash stop {m['crash_stop_m']} m from {m['crash_stop_from_kn']} kn; turn {m['turning_circle']['30kn']['diameter_m']} m at 30 kn"],
        ["EO/IR (skiff)", f"day ID {dri['day_clear']['identification']} km, night ID {dri['night_clear']['identification']} km, fog {dri['night_fog']['detection']} km (detect)"],
        ["radar (skiff)", f"{se['radar']['skiff_km_by_hs'][0.5]} km in Hs 0.5 m, {se['radar']['skiff_km_by_hs'][2.5]} km in Hs 2.5 m"],
        ["ladder", f"hailer {e['hailer']['spl_1m_db']:g} dB: {e['hailer']['intelligible_m']['skiff_at_speed']} m to a skiff at speed; light {e['searchlight']['range_m']} m; strobe {e['strobe']['dazzle_m']} m"],
        ["water cannon", f"{e['water_cannon']['flow_l_min']:g} L/min at {e['water_cannon']['pressure_bar']:g} bar: {e['water_cannon']['throw_m']} m throw, {e['water_cannon']['shaft_kW']:g} kW PTO, sea water"],
        ["blocking", f"foam collar absorbs {s['blocking']['collar_energy_kJ']} kJ: parallel contact under {s['blocking']['safe_closing_kn']} kn closing"],
        ["carrier ramp", f"launch {c['launch_s']:.0f} s, recovery {c['recovery_s']:.0f} s, to Hs {c['hs_limit_m']} m; system {c['system_mass_kg'] / 1000:.1f} t"],
        ["price", f"EUR {s['price_eur'] / 1e6:.2f} M"],
    ]
    tab = tx.table(cellText=rows, colLabels=["", "Barracuda v0 (class values, assumed)"], loc="center", cellLoc="left", colWidths=[0.15, 0.85])
    tab.auto_set_font_size(False); tab.set_fontsize(9.5); tab.scale(1, 1.6)
    fig.suptitle("Barracuda - the unmanned interceptor of the guard carrier: hail, shadow, lights and loudhailer, block, water cannon", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == "__main__":
    main(*sys.argv[1:])
