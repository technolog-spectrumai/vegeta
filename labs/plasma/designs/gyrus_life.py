"""Gyrus's heat, wear and test (``docs/hall.md``, H5): the power budget, the lumps' temperatures in vacuum, the
channel's erosion and life, the permanent-magnet trade, the test stand and the test plan — from the Gyrus geometry
and the H1/H3 numbers.    python gyrus_life.py [out.png]"""
from __future__ import annotations

import math
import sys

from vegeta.fulgur.hall import point_for_power, reluctance_circuit, design_coils
from vegeta.fulgur.hall_life import TEST_PLAN, ThermalBody, erosion, facility, magnet_trade, power_budget, radiator_for, temperatures, total_impulse

from gyrus import Gyrus


def analysis(p=None) -> dict:
    p = p or {q.name: q.default for q in Gyrus.parameters}
    sec = Gyrus.section(p)
    ch = sec.ch
    pt = point_for_power(p["power"], p["v_d"])
    regs = {n: (r1, r2, z1, z2) for n, r1, r2, z1, z2, _ in sec.regions()}
    r_out = regs["yoke"][1]
    L_body = regs["pole_out"][3] - regs["plate"][2]
    body_area = 2 * math.pi * r_out * L_body + math.pi * r_out ** 2                 # the screen's outside and the back
    front_area = math.pi * (regs["pole_out"][1] ** 2 - regs["pole_in"][1] ** 2)       # the front face round the channel
    rc = reluctance_circuit(ch)
    coils = design_coils(293.0, inner_radius=sec.core_r + sec.coil_in_w / 2, outer_radius=ch.r_out + sec.coil_out_w / 2)
    tb = ThermalBody(ch, body_area, front_area, coils["power_W"])
    temps = temperatures(pt, tb)
    ero = erosion(ch, wall_thickness=p["wall"] * 1e-3)
    return {"point": pt, "budget": power_budget(pt), "thermal": tb, "temperatures": temps, "erosion": ero,
            "impulse": total_impulse(pt, ero["life_h"]), "magnets": magnet_trade(293.0, rc["gap_m"], ch.b_exit, coils["mass_kg"], coils["power_W"]),
            "test_stand": facility(pt), "plan": TEST_PLAN, "areas": {"body_m2": body_area, "front_m2": front_area},
            "radiator": radiator_for(pt, tb, 190.0)}


def main(out="gyrus_life.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, BLUE, ORANGE, GREY = "#1f2328", "#6e7781", "#2f6db3", "#c8711a", "#9aa4ad"
    a = analysis()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    b = a["budget"]
    ax = axes[0]
    names = ["beam", "walls", "anode", "radiation_plume", "cathode_coupling"]
    ax.barh(names, [b[n] for n in names], color=[BLUE, ORANGE, ORANGE, GREY, GREY])
    for i, n in enumerate(names):
        ax.text(b[n] + 2, i, f"{b[n]:.0f} W", va="center", fontsize=9, color=INK)
    ax.set_xlabel("W"); ax.set_title(f"where the {b['discharge']:.0f} W go", loc="left", fontsize=10, color=INK); ax.grid(axis="x", color="#e6e8eb"); ax.set_axisbelow(True)
    t = a["temperatures"]
    ax = axes[1]
    lumps = [("body (iron)", t["body_C"], t["SmCo_limit_C"], "SmCo limit"), ("channel (BN)", t["channel_C"], t["BN_limit_C"], "BN limit"),
             ("anode", t["anode_C"], 800.0, "stainless"), ("coils", t["coils_C"], t["coil_limit_C"], "wire limit")]
    ax.bar(range(4), [l[1] for l in lumps], color=[BLUE if l[1] < l[2] else ORANGE for l in lumps], width=0.6)
    for i, l in enumerate(lumps):
        ax.plot([i - 0.3, i + 0.3], [l[2], l[2]], color=ORANGE, lw=1.5); ax.text(i, l[2] + 10, l[3], ha="center", fontsize=8, color=ORANGE)
        ax.text(i, l[1] + 10, f"{l[1]:.0f} °C", ha="center", fontsize=9, color=INK)
    rad = a["radiator"]
    ax.set_xticks(range(4), [l[0] for l in lumps]); ax.set_ylabel("°C")
    ax.set_title(f"steady temperatures radiating in vacuum, bare; a Ø{rad['disc_diameter_m']*1e3:.0f} mm radiator brings the body to {rad['body_C']:.0f} °C", loc="left", fontsize=9, color=INK)
    ax.grid(axis="y", color="#e6e8eb"); ax.set_axisbelow(True)
    e, imp, ts = a["erosion"], a["impulse"], a["test_stand"]
    ax = axes[2]
    ax.axis("off")
    txt = (f"EROSION AT THE EXIT\n  ions to the wall {e['ion_flux']:.1e} /m²s at {e['wall_energy_eV']:.0f} eV, yield {e['yield']:.3f}\n"
           f"  {e['rate_mm_per_kh']:.2f} mm per 1000 h → the 2 mm wall lasts ~{e['life_h']:.0f} h\n"
           f"  total impulse {imp['total_impulse_Ns']/1e3:.0f} kN·s on {imp['propellant_kg']:.1f} kg of xenon\n\n"
           f"MAGNETS INSTEAD OF COILS ({a['magnets']['material']})\n  {a['magnets']['magnet_length_mm']:.1f} mm of magnet in the loop; saves {a['magnets']['saved_coil_mass_kg']*1e3:.0f} g and {a['magnets']['saved_power_W']:.1f} W\n"
           f"  but no field scan in test; body must stay under {a['magnets']['t_max_C']:.0f} °C: bare {t['body_C']:.0f}, with the radiator {a['radiator']['body_C']:.0f}\n\n"
           f"TEST STAND\n  xenon throughput {ts['throughput_mbar_l_s']:.2f} mbar·l/s → {ts['pumping_speed_l_s']/1e3:.0f} 000 l/s to hold 2e-5 mbar\n"
           f"  {ts['chamber']}\n  thrust balance {ts['thrust_balance']['range_mN']:.0f} mN range, {ts['thrust_balance']['resolution_mN']:.1f} mN resolution\n"
           f"  discharge supply {ts['supplies']['discharge']}")
    ax.text(0.0, 1.0, txt, va="top", ha="left", fontsize=9, family="monospace", color=INK, transform=ax.transAxes)
    fig.suptitle("Gyrus-300 — heat, wear and test (budgets, not measurements)", color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    return out, a


if __name__ == "__main__":
    out, a = main(*sys.argv[1:2])
    print({k: round(v, 1) for k, v in a["budget"].items()})
    print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in a["temperatures"].items()})
    print("radiator:", {k: round(v, 4) for k, v in a["radiator"].items()})
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in a["erosion"].items()})
    print(a["magnets"]); print(a["test_stand"]["throughput_mbar_l_s"], a["test_stand"]["pumping_speed_l_s"])
    for step in a["plan"]:
        print(" ", step)
    print(out)
