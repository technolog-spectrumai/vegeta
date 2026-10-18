"""Notebook 48's figures: the welding speed against power per material and thickness (the model's and the capped one),
the capacity (parts per hour, arc-on against practical), the euro per metre, the stray beam against distance against the
exposure limits, and the capacity's sensitivity. Static Matplotlib."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import kimla as km
import svarog_commercial as sc

COL = km.COL
MAT_LS = {"carbon steel": "-", "stainless": "--", "aluminium": ":"}
MAT_COL = {"carbon steel": "tab:blue", "stainless": "tab:green", "aluminium": "tab:red"}


def _ax(ax, w=8, h=4.5):
    return ax if ax is not None else plt.subplots(figsize=(w, h))[1]


def speed_view(table, joint="butt", ax=None):
    """m/min against power: the model's speed (thin) and the capped speed (thick), per material (colour) and thickness (marker)."""
    ax = _ax(ax, 9, 5)
    marks = {1.0: "o", 2.0: "s", 3.0: "^"}
    sub = table[table["joint"] == joint]
    for (mat, t), g in sub.groupby(["material", "t [mm]"]):
        fam = next(f for f in MAT_COL if f.split()[0] in mat.lower() or (f == "stainless" and "stainless" in mat.lower()))
        g = g.assign(P=[km.SCENARIOS[k].power_W / 1e3 for k in g["scenario"]]).sort_values("P")
        ax.plot(g["P"], g["v model [m/min]"], ls=MAT_LS[fam], marker=marks[t], color=MAT_COL[fam], lw=0.8, alpha=0.5)
        ax.plot(g["P"], g["v [m/min]"], ls=MAT_LS[fam], marker=marks[t], color=MAT_COL[fam], lw=2, label=f"{fam} {t:g} mm")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("laser power [kW]"); ax.set_ylabel("welding speed [m/min]")
    ax.set_title(f"{joint} joints: the model's speed (thin) and the speed after the humping / feed cap (thick) — unvalidated", fontsize=9)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7, ncol=3)
    return ax


def capacity_view(econ, job="assembly_1m", joint="butt", ax=None):
    """Parts per hour: arc-on (hollow) against practical (filled), per scenario and material at 2 mm."""
    ax = _ax(ax, 9, 4.5)
    sub = econ[(econ["job"] == job) & (econ["joint"] == joint) & (econ["t [mm]"] == 2.0)]
    scen = list(km.SCENARIOS)
    x = np.arange(len(scen))
    w = 0.25
    for i, fam in enumerate(MAT_COL):
        g = sub[sub["material"].str.lower().str.contains(fam.split()[0])].set_index("scenario").reindex(scen)
        ax.bar(x + (i - 1) * w, g["parts/h arc-on"], w, color=MAT_COL[fam], alpha=0.25, edgecolor=MAT_COL[fam])
        ax.bar(x + (i - 1) * w, g["parts/h practical"], w, color=MAT_COL[fam], label=f"{fam} (practical; faint: arc-on only)")
    ax.set_xticks(x); ax.set_xticklabels(scen); ax.set_ylabel("parts per hour"); ax.set_yscale("log")
    ax.set_title(f"{km.JOBS[job]['note']} — 2 mm {joint}: arc-on against practical throughput", fontsize=9); ax.grid(alpha=0.3, axis="y", which="both"); ax.legend(fontsize=7)
    return ax


def cost_per_metre_view(econ, joint="butt", ax=None):
    """€/metre practical (the 1 m assembly) against power per material and thickness."""
    ax = _ax(ax, 9, 4.5)
    marks = {1.0: "o", 2.0: "s", 3.0: "^"}
    sub = econ[(econ["joint"] == joint) & (econ["job"] == "assembly_1m")]
    for (mat, t), g in sub.groupby(["material", "t [mm]"]):
        fam = next(f for f in MAT_COL if f.split()[0] in mat.lower())
        g = g.assign(P=[km.SCENARIOS[k].power_W / 1e3 for k in g["scenario"]]).sort_values("P")
        ax.plot(g["P"], g["€/m practical"], ls=MAT_LS[fam], marker=marks[t], color=MAT_COL[fam], label=f"{fam} {t:g} mm")
    ax.set_xscale("log"); ax.set_xlabel("laser power [kW]"); ax.set_ylabel("€ per metre of weld (practical, with operator)")
    ax.set_title("the 1 m assembly: euro per metre against power — the overheads, not the power, set it", fontsize=9); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7, ncol=3)
    return ax


def exposure_view(ax=None, exposure_s=100.0):
    """The beam past the focus and the aluminium reflection against distance per scenario, with the eye's and the
    skin's exposure limits for the stated duration (no aversion response assumed)."""
    ax = _ax(ax, 9, 5)
    d = np.linspace(2.0, 250.0, 400)
    for k, s in km.SCENARIOS.items():
        o = km.optics(s)
        r = np.maximum(o["cone_half_angle"] * d, o["spot_radius"])
        ax.plot(d, o["delivered_W"] / (np.pi * r ** 2), color=COL[k], lw=2, label=f"{k} past the focus")
        ax.plot(d, km._a("reflectivity_aluminium").nominal * o["delivered_W"] / (np.pi * r ** 2), color=COL[k], lw=1, ls=":", label=f"{k} reflected off aluminium")
    for val, name, c in ((float(sc.mpe_eye_1070(exposure_s)), f"MPE eye, {exposure_s:g} s", "k"), (float(sc.mpe_skin_1070(exposure_s)), f"MPE skin, {exposure_s:g} s", "brown")):
        ax.axhline(val, color=c, lw=0.8, ls="--"); ax.text(180, val * 1.3, name, fontsize=7, color=c)
    for dd in km.SAFETY_DISTANCES:
        ax.axvline(dd, color="grey", lw=0.5, ls=":")
    ax.set_yscale("log"); ax.set_xlabel("distance [m]"); ax.set_ylabel("mean irradiance [W/m²]")
    ax.set_title("accidental exposure: the diverging beam and a reflection against distance (IEC 60825-1 limits; no blink assumed)", fontsize=9)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=6, ncol=2)
    return ax


def sensitivity_view(sens, scenario="6 kW", ax=None):
    """A tornado of the practical capacity for one scenario: each assumption's ends against the base."""
    ax = _ax(ax, 8, 4)
    sub = sens[sens["scenario"] == scenario]
    labels, lo, hi = [], [], []
    for key, g in sub.groupby("assumption"):
        labels.append(key); lo.append(g["change [%]"].min()); hi.append(g["change [%]"].max())
    y = np.arange(len(labels))
    ax.barh(y, lo, color="tab:red", alpha=0.7); ax.barh(y, hi, color="tab:green", alpha=0.7)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8); ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("change of the practical parts/h [%]"); ax.set_title(f"{scenario}: what moves the capacity (2 mm carbon steel butt, the 1 m assembly)", fontsize=9); ax.grid(alpha=0.3, axis="x")
    return ax
