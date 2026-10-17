"""Drawings for Svarog-combining (notebook 47): the channel layout, near- and far-field maps, 3-D surfaces, cross
sections, side-by-side comparisons of the methods, the two-beam fringes, the phase and loss sensitivities. NumPy and
Matplotlib only; static (no animation). The physics is in ``svarog_combining`` / ``hyperion.combining``."""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from vegeta.hyperion import combining as cb

COL = {"incoherent": "#1f77b4", "coherent-filled": "#2a7f3f", "coherent-tiled": "#8c564b", "spectral": "#d62728", "one Gaussian": "k"}


def layout_view(chans: list[cb.Channel], ax=None, title: str = ""):
    """The channels at the combining plane: circles of radius w0, colour by wavelength."""
    ax = ax or plt.gca()
    lams = sorted({c.wavelength for c in chans})
    for c in chans:
        col = plt.cm.viridis(lams.index(c.wavelength) / max(len(lams) - 1, 1))
        ax.add_patch(plt.Circle((c.x * 1e6, c.y * 1e6), c.w0 * 1e6, fill=False, color=col, lw=1.5))
        ax.annotate(f"{c.name}\n{c.wavelength * 1e9:.0f} nm" if len(lams) > 1 else c.name, (c.x * 1e6, c.y * 1e6), fontsize=7, ha="center", va="center")
    r = max(math.hypot(c.x, c.y) + 2 * c.w0 for c in chans) * 1e6
    ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_aspect("equal")
    ax.set_xlabel("x [µm]"); ax.set_ylabel("y [µm]"); ax.set_title(title or f"{len(chans)} channels"); ax.grid(alpha=0.3)
    return ax


def field_map(res: cb.Combined, far: bool = False, ax=None, title: str = "", log: bool = False, zoom: float | None = None):
    """The near-field (W/cm²) or far-field (W/sr) intensity as a 2-D map; ``zoom`` the half-width shown (µm or mrad)."""
    ax = ax or plt.gca()
    if far:
        X, Y, I = res.theta_x * 1e3, res.theta_y * 1e3, res.I_far
        unit, lab = "W/sr", "θ [mrad]"
    else:
        X, Y = res.grid.xy(); X, Y, I = X * 1e6, Y * 1e6, res.I_near / 1e4
        unit, lab = "W/cm²", "[µm]"
    Ip = I if not log else np.log10(np.maximum(I, I.max() * 1e-6))
    m = ax.pcolormesh(X, Y, Ip, shading="auto", cmap="inferno")
    plt.colorbar(m, ax=ax, label=(f"log10 {unit}" if log else unit))
    if zoom:
        ax.set_xlim(-zoom, zoom); ax.set_ylim(-zoom, zoom)
    ax.set_aspect("equal"); ax.set_xlabel("x " + lab); ax.set_ylabel("y " + lab)
    ax.set_title(title or f"{res.method}: {'far' if far else 'near'} field")
    return ax


def surface_view(res: cb.Combined, far: bool = True, ax=None, title: str = "", zoom: float | None = None, stride: int = 2):
    """The intensity as a 3-D surface (matplotlib ``plot_surface``); ``ax`` must be a 3-D axes when given."""
    if ax is None:
        ax = plt.figure(figsize=(7, 5)).add_subplot(projection="3d")
    if far:
        X, Y, I = res.theta_x * 1e3, res.theta_y * 1e3, res.I_far
        lab, zl = "θ [mrad]", "I [W/sr]"
    else:
        X, Y = res.grid.xy(); X, Y, I = X * 1e6, Y * 1e6, res.I_near / 1e4
        lab, zl = "[µm]", "I [W/cm²]"
    if zoom:
        n = I.shape[0] // 2
        k = int(np.searchsorted(X[0], zoom)) - n
        X, Y, I = X[n - k:n + k, n - k:n + k], Y[n - k:n + k, n - k:n + k], I[n - k:n + k, n - k:n + k]
    ax.plot_surface(X[::stride, ::stride], Y[::stride, ::stride], I[::stride, ::stride], cmap="inferno", linewidth=0, antialiased=False)
    ax.set_xlabel("x " + lab); ax.set_ylabel("y " + lab); ax.set_zlabel(zl)
    ax.set_title(title or f"{res.method}: {'far' if far else 'near'} field")
    return ax


def cross_sections(results: dict, far: bool = True, ax=None, title: str = "", normalise: bool = False, zoom: float | None = None):
    """The intensity along x through the origin for {label: Combined}."""
    ax = ax or plt.gca()
    for lab, r in results.items():
        x, I = r.cross_section("x", far=far)
        x = x * (1e3 if far else 1e6)
        I = I / I.max() if normalise and I.max() > 0 else (I if far else I / 1e4)
        key = lab.split(" ")[0] if lab.split(" ")[0] in COL else None
        ax.plot(x, I, lw=1.5, label=lab, color=COL.get(key) if key else None)
    if zoom:
        ax.set_xlim(-zoom, zoom)
    ax.set_xlabel("θx [mrad]" if far else "x [µm]"); ax.set_ylabel("normalised" if normalise else ("I [W/sr]" if far else "I [W/cm²]"))
    ax.grid(alpha=0.3); ax.legend(fontsize=8); ax.set_title(title or ("far field" if far else "near field"))
    return ax


def side_by_side(results: dict, far: bool = True, zoom: float | None = None, log: bool = False, title: str = ""):
    """One row of maps for {label: Combined}, the same colour scale when not log."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4))
    axes = np.atleast_1d(axes)
    vmax = max((r.I_far if far else r.I_near / 1e4).max() for r in results.values())
    for ax, (lab, r) in zip(axes, results.items()):
        field_map(r, far=far, ax=ax, title=lab, log=log, zoom=zoom)
        if not log:
            ax.collections[0].set_clim(0, vmax)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def method_panel(res_by_method: dict, n: int, zoom_far: float | None = None, zoom_near: float | None = None):
    """For one channel count: the near and the far field of every method, two rows."""
    ms = list(res_by_method)
    fig, axes = plt.subplots(2, len(ms), figsize=(4.2 * len(ms), 8))
    for j, m in enumerate(ms):
        r = res_by_method[m][n]
        field_map(r, far=False, ax=axes[0, j], title=f"{m}, N = {n}: near", zoom=zoom_near)
        field_map(r, far=True, ax=axes[1, j], title=f"{m}, N = {n}: far (log)", log=True, zoom=zoom_far)
    fig.tight_layout()
    return fig


def metrics_view(rows: list[dict], key: str, ax=None, log: bool = True, title: str = ""):
    """One metric against N per method, from ``metrics_table`` rows; the one-Gaussian row as a dashed line."""
    ax = ax or plt.gca()
    by = {}
    for r in rows:
        by.setdefault(r["method"], []).append((r["N"], r[key]))
    for m, pts in by.items():
        if m == "one Gaussian":
            ax.axhline(pts[0][1], color="k", ls="--", lw=1, label="one Gaussian of the same power")
        else:
            pts.sort(); ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=COL.get(m), label=m)
    if log:
        ax.set_yscale("log")
    ax.set_xscale("log", base=2); ax.set_xlabel("channels N"); ax.set_ylabel(key); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    ax.set_title(title or key)
    return ax


def phase_view(ps_by_n: dict, ax=None, title: str = ""):
    """The on-axis gain against the rms phase error per channel count, with the closed form."""
    ax = ax or plt.gca()
    for k, ps in ps_by_n.items():
        line, = ax.plot(ps["sigma_phase"], ps["axis_gain"], "o-", label=f"N = {k}: on-axis gain")
        ax.plot(ps["sigma_phase"], ps["closed_form"], ":", color=line.get_color(), label=f"N = {k}: 1/N + (1 − 1/N) e^(−σ²)")
        ax.fill_between(ps["sigma_phase"], ps["efficiency"] - ps["efficiency_std"], ps["efficiency"] + ps["efficiency_std"], color=line.get_color(), alpha=0.12)
    ax.set_xlabel("rms phase error σ [rad]"); ax.set_ylabel("gain / efficiency"); ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    ax.set_title(title or "coherent combining against phase error (bands: the central-lobe efficiency's spread)")
    return ax


def loss_view(ls: dict, key: str = "brightness [W/(cm²·sr)]", ax=None, title: str = ""):
    ax = ax or plt.gca()
    for m, rows in ls.items():
        ax.plot([r["loss"] * 100 for r in rows], [r[key] for r in rows], "o-", color=COL.get(m), label=m)
    ax.set_xlabel("loss per channel [%]"); ax.set_ylabel(key); ax.grid(alpha=0.3); ax.legend(fontsize=8); ax.set_title(title or key + " against the channels' losses")
    return ax


def interference_view(pairs: dict, zoom: float | None = None):
    """{label: Combined} of two-beam cases: far-field maps on top, the x cross sections below."""
    fig, axes = plt.subplots(2, len(pairs), figsize=(4.2 * len(pairs), 7.5))
    axes = np.atleast_2d(axes) if len(pairs) > 1 else np.array([[axes[0]], [axes[1]]])
    for j, (lab, r) in enumerate(pairs.items()):
        field_map(r, far=True, ax=axes[0, j], title=lab, zoom=zoom)
        x, I = r.cross_section("x", far=True)
        axes[1, j].plot(x * 1e3, I, lw=1.2, color="#8c564b")
        if zoom:
            axes[1, j].set_xlim(-zoom, zoom)
        axes[1, j].set_xlabel("θx [mrad]"); axes[1, j].set_ylabel("I [W/sr]"); axes[1, j].grid(alpha=0.3)
    fig.tight_layout()
    return fig
