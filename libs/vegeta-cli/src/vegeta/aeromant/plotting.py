"""Matplotlib helpers for Aeromant cases. They return figures and never call ``plt.show()``."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .results import find_coefficient_files, read_coefficients, read_solver_log


def plot_coefficients(case: str | Path, names=("Cd", "Cl", "Cm"), ax=None):
    """Force coefficient histories against iteration."""
    import matplotlib.pyplot as plt

    hist = read_coefficients(find_coefficient_files(Path(case)))
    if ax is None:
        _, ax = plt.subplots()
    for n in names:
        if n in hist:
            ax.plot(hist.iterations, hist[n], label=n)
    ax.set_xlabel("iteration")
    ax.set_ylabel("coefficient")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax.figure


def plot_residuals(case: str | Path, log_name: str = "log.solver", ax=None):
    """Initial residuals per field (log scale)."""
    import matplotlib.pyplot as plt

    sl = read_solver_log(Path(case) / log_name)
    if ax is None:
        _, ax = plt.subplots()
    for name, r in sl.residuals.items():
        ax.semilogy(np.arange(1, len(r) + 1), r, label=name)
    ax.set_xlabel("iteration")
    ax.set_ylabel("initial residual")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    return ax.figure
