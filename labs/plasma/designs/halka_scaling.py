"""Halka scaled (``docs/plasma.md``, P4): the 1-D model calibrated on 2-D solves, swept over diameter, offset,
stages and voltage at a spark margin; the masses of what must fly; the hover wall and the winged route; the
demonstrator decision by the rule of round 1 (attempt; scale up if too weak; none if it cannot work).

    python halka_scaling.py [out.png]      # the calibration solves take minutes; seconds under vegeta.mock
"""
from __future__ import annotations

import math
import sys

import numpy as np

from vegeta.fulgur import Air, DRY, climate
from vegeta.fulgur.breakdown import propagation_voltage
from vegeta.fulgur.scaling import BUILDS, Calibration, Family, at_margin, estimate, flight_masses, thruster_mass, winged
from vegeta.fulgur.thruster import PRESETS

import halka_fulgur as hf

G = 9.81
GRID = dict(pitches=(0.011, 0.014, 0.018), dzs=(0.008, 0.012, 0.016))


def calibration_points(lc: float = 2.5, preset: str = "mark4c") -> list:
    """2-D solves over the pitch x offset grid, each at 0.85 of its gap's spark voltage; the 4C's radii scaled
    about its outer emitter so only the pitch changes."""
    from dataclasses import replace
    pts = []
    base = PRESETS[preset]
    for pitch in GRID["pitches"]:
        k = pitch / 0.014
        rings = tuple(replace(r, radius=0.082 + (r.radius - 0.082) * k) for r in base.rings)
        name = f"{preset}_cal{pitch*1e3:.0f}"
        PRESETS[name] = replace(base, name=name, rings=rings)
        try:
            for dz in GRID["dzs"]:
                V = 0.85 * propagation_voltage(math.hypot(pitch, dz), DRY)
                r = hf.solve_section(name, V, dz=dz, lc=lc)
                pts.append({"pitch": pitch, "dz": dz, "V": V, "I_2d": r["I"], "thrust_2d": r["thrust"], "onset_2d": r["onset"],
                            "edge_r": r["edge_r"], "length": sum(2 * math.pi * rr.radius for rr in rings if rr.role == "emitter"),
                            "converged": r["converged"]})
        finally:
            PRESETS.pop(name, None)
    return pts


def calibrate(lc: float = 2.5) -> Calibration:
    return Calibration.fit(calibration_points(lc))


def sweep(cal: Calibration, diameters=(0.2, 0.3, 0.5, 0.8, 1.2), dzs=(0.008, 0.012, 0.016, 0.020), stages=(1, 2, 3),
          air: Air = DRY, margin: float = 1.2, build: str = "flight", minutes: float = 2.0) -> list:
    rows = []
    for D in diameters:
        for dz in dzs:
            for n in stages:
                f = Family(D, dz=dz, stages=n)
                e = at_margin(f, cal, air, margin)
                m = flight_masses(f, e, build=build, minutes=minutes)
                w = winged(f, e, m)
                rows.append({"diameter": D, "dz": dz, "stages": n, **{k: e[k] for k in ("V", "I", "thrust", "power", "thrust_per_watt", "limited_by")},
                             "mass_thruster": m["thruster"], "mass_supply": m["supply"], "mass_battery": m["battery"], "mass_total": m["total"],
                             "T/W": m["thrust_to_weight"], "winged_margin": w["margin"], "thrust_per_kg": e["thrust"] / m["thruster"] if m["thruster"] else 0.0})
    return rows


def decision(rows: list, cal: Calibration) -> dict:
    """The rule of round 1 applied: hover needs T/W >= 1 with the supply and battery aboard; a winged demonstrator
    needs thrust >= weight / (L/D). The smallest thruster that does each, or none."""
    hover = [r for r in rows if r["T/W"] >= 1.0]
    wing = [r for r in rows if r["winged_margin"] >= 1.0]
    best_tw = max(rows, key=lambda r: r["T/W"])
    pick = lambda rs: min(rs, key=lambda r: r["mass_total"]) if rs else None
    return {"hover": pick(hover), "best_thrust_to_weight": best_tw, "winged": pick(wing), "calibration": cal,
            "verdict": ("hover demonstrator" if hover else "winged demonstrator" if wing else "no demonstrator: scale up or change the build")}


def main(out="halka_scaling.png", lc: float = 2.5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, BLUE, ORANGE, GREY = "#1f2328", "#6e7781", "#2f6db3", "#c8711a", "#9aa4ad"
    cal = calibrate(lc)
    rows = sweep(cal)
    dec = decision(rows, cal)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    # 1. thrust vs diameter per stage count at dz 12 mm
    ax = axes[0, 0]
    for n, c in zip((1, 2, 3), (GREY, BLUE, INK)):
        rs = [r for r in rows if r["dz"] == 0.012 and r["stages"] == n]
        ax.plot([r["diameter"] for r in rs], [r["thrust"] / G * 1e3 for r in rs], "o-", color=c, lw=2, label=f"{n} stage{'s' if n > 1 else ''}")
        ax.plot([r["diameter"] for r in rs], [r["power"] for r in rs], "s--", color=c, lw=1, alpha=0.6)
    ax.set_xlabel("diameter [m]"); ax.set_ylabel("thrust [gf] (solid), corona power [W] (dashed)")
    ax.set_title("thrust and power at dz 12 mm, spark margin 1.2, dry air", loc="left", fontsize=10, color=INK); ax.legend(frameon=False); ax.grid(color="#e6e8eb")
    # 2. thrust-to-weight vs diameter (flight build, 2 min battery) per dz, 3 stages
    ax = axes[0, 1]
    for dz, c in zip((0.008, 0.012, 0.016, 0.020), (GREY, BLUE, ORANGE, INK)):
        rs = [r for r in rows if r["dz"] == dz and r["stages"] == 3]
        ax.plot([r["diameter"] for r in rs], [r["T/W"] for r in rs], "o-", color=c, lw=2, label=f"dz {dz*1e3:.0f} mm")
    ax.axhline(1.0, color=ORANGE, lw=1, ls=":"); ax.text(0.21, 1.03, "hover needs 1", color=ORANGE, fontsize=8)
    ax.axhline(0.1, color=BLUE, lw=1, ls=":"); ax.text(0.21, 0.105, "a glider at L/D 10 needs 0.1", color=BLUE, fontsize=8)
    ax.set_yscale("log"); ax.set_xlabel("diameter [m]"); ax.set_ylabel("thrust / weight of thruster + supply + 2 min battery")
    ax.set_title("the wall: 3 stages, flight build (carbon frame, aluminium grounds)", loc="left", fontsize=10, color=INK); ax.legend(frameon=False); ax.grid(color="#e6e8eb", which="both")
    # 3. mass breakdown at 0.5 m, 3 stages, dz 12
    ax = axes[1, 0]
    r = next(r for r in rows if r["diameter"] == 0.5 and r["dz"] == 0.012 and r["stages"] == 3)
    parts = [("thruster", r["mass_thruster"]), ("HV supply", r["mass_supply"]), ("battery (2 min)", r["mass_battery"])]
    ax.barh([p[0] for p in parts], [p[1] * 1e3 for p in parts], color=[BLUE, ORANGE, GREY])
    for i, p in enumerate(parts):
        ax.text(p[1] * 1e3 + 5, i, f"{p[1]*1e3:.0f} g", va="center", fontsize=9, color=INK)
    ax.set_xlabel("mass [g]"); ax.set_title(f"what must fly: Ø0.5 m, 3 stages, dz 12 mm — {r['thrust']/G*1e3:.0f} gf at {r['V']/1e3:.0f} kV, {r['power']:.0f} W", loc="left", fontsize=10, color=INK)
    ax.grid(axis="x", color="#e6e8eb"); ax.set_axisbelow(True)
    # 4. humidity: the 0.5 m, 3-stage thruster through Warsaw's year
    ax = axes[1, 1]
    conds = climate.conditions()
    keys = list(climate.WARSAW) + ["humid_day", "indoor"]
    f = Family(0.5, dz=0.012, stages=3)
    th = [at_margin(f, cal, conds[k])["thrust"] / G * 1e3 for k in keys]
    lim = [at_margin(f, cal, conds[k])["limited_by"] for k in keys]
    ax.bar(range(len(keys)), th, color=[BLUE if l == "propagation" else ORANGE for l in lim], width=0.72)
    ax.set_xticks(range(len(keys)), [k.replace("humid_day", "rainy\nday") for k in keys], fontsize=8)
    ax.set_ylabel("thrust at spark margin 1.2 [gf]"); ax.set_title("the same thruster through Warsaw's year (orange: wet-surface limited)", loc="left", fontsize=10, color=INK)
    ax.grid(axis="y", color="#e6e8eb"); ax.set_axisbelow(True)
    fig.suptitle(f"Halka scaled — 1-D calibrated on {len(cal.points)} 2-D solves (k_current {cal.k_current:.2f}, k_onset {cal.k_onset:.2f}, spread {cal.spread:.0%}); verdict: {dec['verdict']}", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    return out, rows, dec


if __name__ == "__main__":
    out, rows, dec = main(*sys.argv[1:2])
    cal = dec["calibration"]
    print(f"calibration: k_current={cal.k_current:.3f} k_onset={cal.k_onset:.3f} spread={cal.spread:.2f} on {len(cal.points)} solves")
    for r in sorted(rows, key=lambda r: -r["T/W"])[:6]:
        print(f"D={r['diameter']:.1f} m dz={r['dz']*1e3:.0f} mm x{r['stages']}: {r['V']/1e3:.0f} kV {r['thrust']/G*1e3:.0f} gf {r['power']:.0f} W "
              f"T/P={r['thrust_per_watt']*1e3:.1f} mN/W mass={r['mass_total']*1e3:.0f} g T/W={r['T/W']:.2f} winged margin={r['winged_margin']:.2f}")
    print("verdict:", dec["verdict"])
    if dec["winged"]:
        w = dec["winged"]; print("smallest winged:", f"D={w['diameter']} dz={w['dz']} stages={w['stages']} mass={w['mass_total']*1e3:.0f} g thrust={w['thrust']/G*1e3:.0f} gf power={w['power']:.0f} W")
    print(out)
