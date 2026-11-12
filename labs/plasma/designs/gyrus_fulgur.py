"""Gyrus in Fulgur (``docs/hall.md``, H1 + H3): the performance table (xenon and krypton, the two references for
scale), the magnetised checks, the magnetic circuit by reluctance and solved on FiPy, the field map and the
radial field along the channel.    python gyrus_fulgur.py [out.png]"""
from __future__ import annotations

import sys

import numpy as np

from vegeta.fulgur.hall import (BHT200, SPT100, CircuitSection, HallPoint, design_coils, magnetised, point_for_power,
                                reluctance_circuit, scale_channel, tune_ampere_turns)

from gyrus import Gyrus


def performance(p=None) -> list:
    p = p or {q.name: q.default for q in Gyrus.parameters}
    rows = []
    for name, pt in (("SPT-100 (ref: 83 mN, 1600 s)", HallPoint(SPT100["v_d"], SPT100["mdot_a"])),
                     ("BHT-200 (ref: 12.8 mN, 1390 s)", HallPoint(BHT200["v_d"], BHT200["mdot_a"])),
                     ("Gyrus-300 xenon", point_for_power(p["power"], p["v_d"])),
                     ("Gyrus-300 krypton", point_for_power(p["power"], p["v_d"], "krypton")),
                     ("Gyrus-300 xenon at 200 V", point_for_power(p["power"], 200.0)),
                     ("Gyrus-300 xenon at 350 V", point_for_power(p["power"], 350.0))):
        rows.append({"case": name, **pt.summary()})
    return rows


def circuit(p=None, lc_fine: float = 0.4e-3) -> dict:
    p = p or {q.name: q.default for q in Gyrus.parameters}
    sec = Gyrus.section(p)
    ch = sec.ch
    rc = reluctance_circuit(ch)
    sol = tune_ampere_turns(sec, ch.b_exit, rc["ampere_turns"], lc_fine=lc_fine)
    coils = design_coils(sol["ampere_turns"], inner_radius=(sec.core_r + sec.coil_in_w / 2), outer_radius=ch.r_out + sec.coil_out_w / 2)
    return {"channel": ch, "section": sec, "reluctance": rc, "solve": sol, "coils": coils, "magnetised": magnetised(ch, p["v_d"])}


def main(out="gyrus_fulgur.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, BLUE, ORANGE, GREY = "#1f2328", "#6e7781", "#2f6db3", "#c8711a", "#9aa4ad"
    c = circuit()
    sol, ch, sec = c["solve"], c["channel"], c["section"]
    fig = plt.figure(figsize=(16, 6.5))
    ax = fig.add_subplot(1, 2, 1)
    B = (np.hypot(sol["B_r"], sol["B_z"]) * 1e4).reshape(sol["shape"])
    sc = ax.pcolormesh(sol["r_edges"] * 1e3, sol["z_edges"] * 1e3, np.minimum(B, 1.5 * ch.b_exit * 1e4), cmap="Blues", shading="flat")
    fig.colorbar(sc, ax=ax, shrink=0.8, label="|B| [G] (capped at 1.5× the target)")
    for name, r1, r2, z1, z2, kind in sec.regions():
        col = {"iron": INK, "coil": ORANGE, "channel": GREY}[kind]
        ax.add_patch(plt.Rectangle((r1 * 1e3, z1 * 1e3), (r2 - r1) * 1e3, (z2 - z1) * 1e3, fill=False, ec=col, lw=1.2))
    ax.set_xlim(0, 60); ax.set_ylim(-12, 30); ax.set_aspect("equal")
    ax.set_xlabel("r [mm]"); ax.set_ylabel("z [mm] (the plume leaves +z)")
    ax.set_title(f"the magnetic circuit solved: {sol['ampere_turns']:.0f} At per coil (iron black, coils orange, channel grey)", loc="left", fontsize=10, color=INK)
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(sol["z_line"] * 1e3, np.abs(sol["B_r_line"]) * 1e4, color=BLUE, lw=2)
    ax.axhline(ch.b_exit * 1e4, color=ORANGE, ls=":", lw=1); ax.text(0.3, ch.b_exit * 1e4 + 8, f"target at the exit {ch.b_exit*1e4:.0f} G", color=ORANGE, fontsize=9)
    ax.axvline(ch.L * 1e3, color=GREY, ls="--", lw=1); ax.text(ch.L * 1e3 + 0.2, 20, "exit plane", color=MUTED, fontsize=9, rotation=90)
    ax.set_xlabel("z along the channel midline [mm] (anode at 0)"); ax.set_ylabel("radial field |B_r| [G]")
    ax.set_title(f"the radial field: {sol['B_r_anode']*1e4:.0f} G at the anode, {sol['B_r_exit']*1e4:.0f} G at the exit, peak {np.abs(sol['B_r_line']).max()*1e4:.0f} G", loc="left", fontsize=10, color=INK)
    ax.grid(color="#e6e8eb")
    m = c["magnetised"]
    fig.suptitle(f"Gyrus-300: channel Ø{ch.d*1e3:.0f} × {ch.h*1e3:.1f} × {ch.L*1e3:.1f} mm; electrons' Larmor radius {m['r_larmor_e_mm']:.2f} mm ≪ width, ions' {m['r_larmor_i_mm']/1e3:.1f} m ≫ length, Hall parameter {m['hall_parameter']:.0f}; "
                 f"coils {c['coils']['inner'].turns} turns at {c['coils']['inner'].current:g} A, {c['coils']['power_W']:.1f} W, {c['coils']['mass_kg']*1e3:.0f} g", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    return out, c


if __name__ == "__main__":
    for row in performance():
        print(f"{row['case']:34s} {row['power_W']:6.0f} W  {row['mdot_a_mg_s']:.2f} mg/s  {row['I_d']:.2f} A  {row['thrust_mN']:5.1f} mN  {row['isp_s']:5.0f} s  eta {row['efficiency']:.2f}  {row['mN_per_kW']:.0f} mN/kW")
    out, c = main(*sys.argv[1:2])
    print("reluctance At", round(c["reluctance"]["ampere_turns"]), "solved At", round(c["solve"]["ampere_turns"]), "B exit/anode G", round(c["solve"]["B_r_exit"]*1e4), round(c["solve"]["B_r_anode"]*1e4))
    print(out)
