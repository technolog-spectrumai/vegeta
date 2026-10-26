"""Basic plots (matplotlib, the ``plot`` extra): velocity profiles and convergence histories."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np


def _plt():
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt


def velocity_profile(coordinate: np.ndarray, numerical: np.ndarray, reference: np.ndarray | None = None, *,
                     xlabel: str = "position [lattice units]", ylabel: str = "velocity [lattice units]",
                     title: str = "", path: str | Path | None = None, reference_label: str = "reference"):
    """Numerical profile (markers) against an optional reference curve; saved to ``path`` when given."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.plot(coordinate, numerical, "o", ms=3, label="Mancha")
    if reference is not None:
        ax.plot(coordinate, reference, "-", lw=1.2, label=reference_label)
        ax.legend()
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return fig


def convergence(resolutions: Sequence[float], errors: Sequence[float], *, order: float | None = None,
                title: str = "", path: str | Path | None = None):
    """Error against resolution on log axes, with a reference slope of ``order`` when given."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5, 3.6))
    n = np.asarray(resolutions, dtype=float)
    e = np.asarray(errors, dtype=float)
    ax.loglog(n, e, "o-", label="error")
    if order is not None:
        ax.loglog(n, e[0] * (n / n[0]) ** (-order), "--", label=f"order {order:g}")
    ax.set(xlabel="resolution [cells]", ylabel="error", title=title)
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return fig


def history(steps: Sequence[int], values: Sequence[float], *, ylabel: str, title: str = "", log: bool = False,
            path: str | Path | None = None):
    """A monitored quantity (kinetic energy, mass drift, residual) against the time step."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5, 3.6))
    (ax.semilogy if log else ax.plot)(steps, values)
    ax.set(xlabel="step", ylabel=ylabel, title=title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return fig
