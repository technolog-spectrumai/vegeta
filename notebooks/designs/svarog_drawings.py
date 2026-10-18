"""Drawings for Svarog (notebook 46): the cavity with its mode envelope, a beam train, the stability window, the
power curve, the pump inside the rod. matplotlib only here; the design module stays free of it."""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from vegeta import hyperion as hy

COL = {"t": "#1f77b4", "s": "#d62728", "pump": "#ff7f0e", "rod": "#c7b4e8", "glass": "#bcd6f0", "mirror": "#333333"}


def _envelope(ax, trace, colour, label, scale=1e6):
    ax.plot(trace.z * 1e3, trace.w * scale, color=colour, lw=1.4, label=label)
    ax.plot(trace.z * 1e3, -trace.w * scale, color=colour, lw=1.4)


def _label_of(e) -> str | None:
    """The rod is split in two around its thermal lens: label the first half as the rod, the rest not at all."""
    if not e.name or isinstance(e, hy.FreeSpace):
        return None
    if e.name.endswith(" a"):
        return e.name[:-2]
    if e.name.endswith(" b") or isinstance(e, hy.ThermalLens):
        return None
    return e.name


def elements_on(ax, train: hy.Train, height: float, scale: float = 1e3):
    """Draw the train's elements as symbols at their positions (mm): mirrors as thick lines or arcs (a concave mirror
    curves towards the beam), rods and glass as boxes, lenses as double arrows, thermal lenses as dashed lines."""
    span = max(train.length * scale, 1e-9)
    for e, z in zip(train.elements, train.positions()):
        zz = z * scale
        if isinstance(e, hy.FlatMirror):
            ax.plot([zz, zz], [-height, height], color=COL["mirror"], lw=4)
        elif isinstance(e, hy.CurvedMirror):
            sag = 0.04 * span * (1 if e.R > 0 else -1) * (1 if zz > 0 else -1)      # in axis units: a concave mirror bows away from the beam
            y = np.linspace(-height, height, 40)
            ax.plot(zz + sag * (y / height) ** 2, y, color=COL["mirror"], lw=4)
        elif isinstance(e, (hy.Slab, hy.ThickLens)):
            ax.add_patch(plt.Rectangle((zz, -height * 0.8), e.length * scale, 1.6 * height, color=COL["rod"] if "rod" in e.name else COL["glass"], alpha=0.6))
        elif isinstance(e, hy.ThermalLens):
            ax.plot([zz, zz], [-0.7 * height, 0.7 * height], color="#7a4fb3", lw=1.2, ls="--")
        elif isinstance(e, hy.ThinLens):
            ax.annotate("", (zz, height), (zz, -height), arrowprops=dict(arrowstyle="<->", color="#2a7f3f", lw=2))
        label = _label_of(e)
        if label:
            ax.text(zz + (e.length * scale) / 2, height * 1.05, label, ha="center", va="bottom", fontsize=8)


def cavity_view(res: hy.Resonator, ax=None, title: str | None = None):
    """The cavity's elements and the mode envelope (µm) in both planes along the axis (mm)."""
    ax = ax or plt.gca()
    modes = res.modes(samples=60)
    wmax = 0.0
    for p, m in modes.items():
        if m.stable:
            _envelope(ax, m.trace, COL[p], f"{'tangential' if p == 't' else 'sagittal'} w (w0 {m.w0 * 1e6:.0f} µm)")
            wmax = max(wmax, float(m.trace.w.max()))
    if wmax == 0.0:
        wmax = 500e-6
        ax.text(0.5, 0.5, "unstable", transform=ax.transAxes, ha="center", color="red", fontsize=14)
    elements_on(ax, res.forward() if res.ring else hy.Train(res.train.elements[1:]), wmax * 1e6 * 1.1)
    hr = res.train.elements[0]
    if isinstance(hr, (hy.FlatMirror, hy.CurvedMirror)):
        ax.plot([0, 0], [-wmax * 1e6 * 1.1, wmax * 1e6 * 1.1], color=COL["mirror"], lw=4)
        ax.text(-0.02 * res.train.length * 1e3, wmax * 1e6 * 1.05, hr.name, ha="right", va="bottom", fontsize=8)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel("z along the axis [mm]")
    ax.set_ylabel("mode radius [µm]")
    ax.set_title(title or res.name)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(-wmax * 1e6 * 1.5, wmax * 1e6 * 1.6)
    return ax


def train_view(trace: hy.BeamTrace, ax=None, title: str = "", scale_w: float = 1e3, unit: str = "mm"):
    """A beam through a train: the envelope (in ``unit``) and the elements."""
    ax = ax or plt.gca()
    _envelope(ax, trace, COL["t"], "w(z)", scale=scale_w)
    elements_on(ax, trace.train, float(trace.w.max()) * scale_w * 1.1)
    for wst in trace.waists:
        ax.plot(wst["z"] * 1e3, 0, "k|", ms=12)
        ax.text(wst["z"] * 1e3, -float(trace.w.max()) * scale_w * 1.3, f"waist {wst['w0'] * 1e6:.0f} µm", ha="center", fontsize=8)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel("z [mm]")
    ax.set_ylabel(f"beam radius [{unit}]")
    ax.set_title(title)
    ax.set_ylim(-float(trace.w.max()) * scale_w * 1.5, float(trace.w.max()) * scale_w * 1.6)
    return ax


def stability_view(window: dict, ax=None, title: str = "", rod_name: str = "rod a"):
    """The spot on the rod, the HR and the OC against the pump power in both planes; the stability window shaded."""
    ax = ax or plt.gca()
    for p, key in (("t", "sweep_t"), ("s", "sweep_s")):
        sw = window[key]
        ls = "-" if p == "t" else "--"
        ax.plot(sw["values"], sw["spots"][rod_name] * 1e6, color=COL[p], ls=ls, label=f"rod ({p})")
        ax.plot(sw["values"], sw["spots"]["OC"] * 1e6, color="#2a7f3f", ls=ls, label=f"OC ({p})")
        ax.plot(sw["values"], sw["spots"]["HR"] * 1e6, color="#7a4fb3", ls=ls, label=f"HR ({p})")
    ok = window["sweep_t"]["stable"] & window["sweep_s"]["stable"]
    p = window["sweep_t"]["values"]
    ax.fill_between(p, 0, 1, where=ok, transform=ax.get_xaxis_transform(), color="#2a7f3f", alpha=0.08, label="stable (both planes)")
    ax.set_xlabel("pump power [W]")
    ax.set_ylabel("spot radius [µm]")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    return ax


def power_view(curves: dict[str, dict], ax=None, pump_design: float | None = None):
    """Output against pump for several revisions (or couplers); the mode radius on a second axis."""
    ax = ax or plt.gca()
    ax2 = ax.twinx()
    for i, (name, c) in enumerate(curves.items()):
        colour = plt.cm.tab10(i)
        ax.plot(c["pump_W"], c["output_W"], color=colour, lw=2, label=f"{name}: output")
        ax2.plot(c["pump_W"], c["w_rod"] * 1e6, color=colour, lw=1, ls=":", label=f"{name}: mode in rod")
    if pump_design is not None:
        ax.axvline(pump_design, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("pump power [W]")
    ax.set_ylabel("output [W]")
    ax2.set_ylabel("mode radius in the rod [µm] (dotted)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    return ax


def pump_view(pr: dict, rod: hy.Rod, ax=None):
    """The pump radius along the rod with the absorbed-power density, and the effective radius."""
    ax = ax or plt.gca()
    z = pr["z"] * 1e3
    ax.plot(z, pr["w"] * 1e6, color=COL["pump"], lw=2, label="pump radius w_p(z)")
    ax.axhline(pr["w_effective"] * 1e6, color=COL["pump"], ls="--", lw=1, label=f"effective {pr['w_effective'] * 1e6:.0f} µm")
    ax.set_xlabel("depth in the rod [mm]")
    ax.set_ylabel("radius [µm]")
    ax2 = ax.twinx()
    a = rod.medium.alpha_pump
    ax2.fill_between(z, 0, a * np.exp(-a * pr["z"]) / 1e2, color="#999", alpha=0.25, label="absorbed per cm (of the incident)")
    ax2.set_ylabel("pump absorbed [1/cm]")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("the pump inside the rod")
    return ax
