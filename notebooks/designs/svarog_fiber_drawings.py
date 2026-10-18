"""Drawings for Svarog-fiber (notebook 47): the power evolution along the fibre, the length and coupler sweeps, the
bend loss against the coil radius, the power curves. matplotlib only here."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

COL = {"pump": "#ff7f0e", "fwd": "#1f77b4", "bwd": "#9ecae1", "inv": "#7a4fb3", "heat": "#d62728"}


def evolution_view(sol, ax=None, title: str = "", seed_label: str = "signal →"):
    """Pump and signal powers along z, the inversion on a second axis."""
    ax = ax or plt.gca()
    ax.plot(sol.z, sol.Pp, color=COL["pump"], lw=2, label="pump")
    ax.plot(sol.z, sol.Ps_fwd, color=COL["fwd"], lw=2, label=seed_label)
    if np.any(sol.Ps_bwd > 0):
        ax.plot(sol.z, sol.Ps_bwd, color=COL["bwd"], lw=2, ls="--", label="← signal")
    ax.set_xlabel("z along the fibre [m]")
    ax.set_ylabel("power [W]")
    ax2 = ax.twinx()
    ax2.plot(sol.z, sol.inversion * 100, color=COL["inv"], lw=1, ls=":", label="inversion N₂/N")
    ax2.plot(sol.z, sol.heat, color=COL["heat"], lw=1, ls="-.", label="heat [W/m]")
    ax2.set_ylabel("inversion [%], heat [W/m]")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return ax


def length_view(choice: dict, ax=None, title: str = "", key: str = "output_W"):
    ax = ax or plt.gca()
    L = [r["length"] for r in choice["rows"]]
    ax.plot(L, [r[key] for r in choice["rows"]], "o-", color=COL["fwd"], label="output")
    ax.plot(L, [r["unabsorbed_W"] for r in choice["rows"]], "s--", color=COL["pump"], label="unabsorbed pump")
    ax.axvline(choice["chosen_length"], color="grey", ls="--", lw=0.8, label=f"chosen {choice['chosen_length']:g} m")
    ax.set_xlabel("active fibre length [m]")
    ax.set_ylabel("[W]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title(title)
    return ax


def coil_view(coils: dict[str, dict], ax=None, budgets: dict | None = None):
    """Bend loss per metre against the coil radius for several fibres (log scale)."""
    ax = ax or plt.gca()
    for i, (name, c) in enumerate(coils.items()):
        ax.semilogy(c["radii"] * 1e3, np.maximum(c["db_per_m"], 1e-6), lw=2, color=plt.cm.tab10(i), label=f"{name} (V {c['V']:.2f})")
    for name, r in (budgets or {}).items():
        ax.axvline(r * 1e3, color="grey", ls=":", lw=0.8)
        ax.text(r * 1e3, 1e-5, f" {name}: R ≥ {r * 1e3:.0f} mm", fontsize=8, rotation=90, va="bottom")
    ax.set_ylim(1e-6, 1e3)
    ax.set_xlabel("coil radius [mm]")
    ax.set_ylabel("LP01 bend loss [dB/m]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    ax.set_title("bend loss of the fundamental mode (Marcuse)")
    return ax


def power_view(curves: dict[str, dict], ax=None, pump_design: float | None = None):
    ax = ax or plt.gca()
    for i, (name, c) in enumerate(curves.items()):
        ax.plot(c["pump_W"], c["output_W"], lw=2, color=plt.cm.tab10(i), label=f"{name}: slope {c['slope']:.2f}, threshold {c['threshold_W']:.2f} W")
    if pump_design is not None:
        ax.axvline(pump_design, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("launched pump [W]")
    ax.set_ylabel("output [W]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax
