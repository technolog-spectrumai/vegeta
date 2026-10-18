"""Drawings for Svarog-industrial (notebooks 48 and 49): the penetration curves and process windows, the stray beams
against distance, the guard's timeline against the exposure classes, a barrier's temperature field."""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

COL = {"mild steel": "#1f77b4", "aluminium": "#ff7f0e", "T1": "#2a7f3f", "T2": "#d4a017", "T3": "#c62828"}


def penetration_view(weld: dict, thicknesses_mm: dict, ax=None):
    """Depth against speed (m/min) per material with the thicknesses as horizontal lines."""
    ax = ax or plt.gca()
    for name, w in weld.items():
        v = w["curve"]["speed"] * 60
        ax.plot(v, w["curve"]["depth"] * 1e3, "o-", color=COL.get(name, "k"), lw=2, label=f"{name} (keyhole from {w['keyhole']['P_threshold_W']:.0f} W)")
    for name, ts in thicknesses_mm.items():
        for t in ts:
            ax.axhline(t, color=COL.get(name, "k"), lw=0.5, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("welding speed [m/min]")
    ax.set_ylabel("penetration [mm]")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    ax.set_title("keyhole penetration at the workpiece power (energy balance, η_melt 0.4)")
    return ax


def window_view(weld: dict, ax=None):
    """The fastest full-penetration speed per thickness and material, as bars."""
    ax = ax or plt.gca()
    width = 0.35
    for i, (name, w) in enumerate(weld.items()):
        rows = w["window"]
        x = np.arange(len(rows)) + i * width
        ax.bar(x, [r["v_full_penetration"] * 60 for r in rows], width, color=COL.get(name, "k"), alpha=0.5, label=f"{name}: full penetration")
        ax.bar(x, [r["v_with_margin"] * 60 for r in rows], width, color=COL.get(name, "k"), label=f"{name}: with 20 % depth in hand")
        for xx, r in zip(x, rows):
            ax.text(xx, r["v_full_penetration"] * 60, f"{r['thickness'] * 1e3:g} mm", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("speed [m/min]")
    ax.set_xticks([])
    ax.set_yscale("log")
    ax.grid(alpha=0.3, axis="y", which="both")
    ax.legend(fontsize=8)
    ax.set_title("the process window: how fast each thickness welds through")
    return ax


def stray_view(beams, head, fibre, power, ax=None, distances=None):
    """The mean irradiance of each stray-beam case against the distance from the head."""
    ax = ax or plt.gca()
    d = np.linspace(0.5, 30, 200) if distances is None else distances
    coll = head.collimated_radius(fibre) + fibre.core_radius / head.f_collimator * d
    div = head.cone_half_angle(fibre) * d
    ax.semilogy(d, power / (math.pi * coll ** 2) / 1e4, lw=2, label="raw collimated beam")
    ax.semilogy(d, power / (math.pi * div ** 2) / 1e4, lw=2, label="diverging past the focus")
    ax.semilogy(d, 0.2 * power / (math.pi * div ** 2) / 1e4, lw=2, ls="--", label="keyhole back-reflection (20 %)")
    ax.axvline(20, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("distance from the head [m]")
    ax.set_ylabel("mean irradiance [W/cm²]")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    ax.set_title("the stray beam against distance")
    return ax


def guard_timeline(rows, exposure: dict, ax=None, beam_name: str = "raw collimated beam"):
    """Per barrier, how long it holds the worst beam (log time) against the exposure classes."""
    ax = ax or plt.gca()
    rs = [r for r in rows if r["beam"] == beam_name]
    names = [r["barrier"] for r in rs]
    holds = np.array([min(r["holds_s"], 1e6) if r["holds_s"] > 0 else 0.05 for r in rs])
    colours = ["#c62828" if not r["T1"] else ("#d4a017" if not r["T2"] else ("#7cb342" if not r["T3"] else "#2a7f3f")) for r in rs]
    y = np.arange(len(rs))
    ax.barh(y, holds, color=colours)
    for yy, r, hld in zip(y, rs, holds):
        label = "transparent" if r["holds_s"] == 0 else ("holds (steady)" if math.isinf(r["holds_s"]) else f"{r['holds_s']:.0f} s")
        ax.text(hld * 1.15, yy, f"{label} — {r['source'].replace('simulation: ', '')}", va="center", fontsize=7)
    for cls, t in exposure.items():
        ax.axvline(t, color=COL[cls], lw=1.2, ls="--")
        ax.text(t, len(rs) - 0.4, f" {cls}: {t:g} s", color=COL[cls], fontsize=8, rotation=90, va="top")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(0.03, 3e6)
    ax.set_xlabel("time the barrier holds [s]")
    ax.set_title(f"the guard against the {beam_name} at 20 m: red fails T1, amber T2, light green T3, green holds")
    ax.grid(alpha=0.3, axis="x", which="both")
    return ax


def field_view(sim, ax=None, title: str = ""):
    """A barrier's final temperature field (r–z), with the removed cells blank."""
    ax = ax or plt.gca()
    r, z = sim.r * 1e3, sim.z * 1e3
    im = ax.pcolormesh(r, z, sim.T_field, shading="nearest", cmap="inferno")
    ax.invert_yaxis()
    ax.set_xlabel("r [mm]")
    ax.set_ylabel("depth [mm]")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="T [K]")
    return ax


def transient_view(sim, ax=None, title: str = ""):
    ax = ax or plt.gca()
    ax.plot(sim.t, sim.T_top, lw=2, label="centre, exposed face")
    ax.plot(sim.t, sim.T_bottom, lw=2, ls="--", label="centre, back face")
    ax.axhline(sim.solid.T_melt, color="grey", lw=0.8, ls=":", label="melting / softening")
    ax2 = ax.twinx()
    ax2.plot(sim.t, (sim.melt_depth + sim.ablated_depth) * 1e3, color="#c62828", lw=1, label="melt + ablation front [mm]")
    ax2.set_ylabel("front depth [mm]")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("T [K]")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return ax
