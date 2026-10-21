"""What the aero studies repeated (``from aero_kit import plain``): helpers that need nothing from a notebook."""
from __future__ import annotations

import numpy as np


def plain(x):
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def lower_skins(step):
    from vegeta import talos
    info = talos.inspect_step(step, units="mm-N-MPa")
    skins = [s for s in info.surfaces if s.kind == "BSpline surface" and s.area > 5e4]
    return [min((s for s in skins if (s.centroid[1] > 0) == right), key=lambda s: s.centroid[2]).tag
            for right in (False, True)]
