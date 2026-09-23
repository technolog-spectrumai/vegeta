"""Matplotlib helpers for sliced G-code. They return figures and never call ``plt.show()``."""
from __future__ import annotations

from pathlib import Path


def _info(obj):
    from .gcode import GcodeInfo, read_gcode

    if isinstance(obj, GcodeInfo):
        return obj
    if hasattr(obj, "artifacts"):
        return read_gcode(obj.artifacts["gcode"])
    return read_gcode(Path(obj))


def plot_layers(gcode, ax=None):
    """Filament extruded per layer against layer height (where the material goes)."""
    import matplotlib.pyplot as plt

    info = _info(gcode)
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(info.layer_extrusion_mm, info.layer_z, "-")
    ax.set_xlabel("filament per layer [mm]")
    ax.set_ylabel("Z [mm]")
    ax.grid(True, alpha=0.3)
    return ax.figure
