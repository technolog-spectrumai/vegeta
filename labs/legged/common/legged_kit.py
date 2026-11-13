"""What the legged studies repeated (``from legged_kit import pin_stress``)."""
from __future__ import annotations

import math

import numpy as np


def pin_stress(F, d): return math.sqrt(3) * F / (2 * math.pi * d**2 / 4)


def posed(mesh, R, centre, zc):
    """A copy of a segment's mesh rotated by ``R`` and moved to ``centre`` (the part frame has its hips at ``zc``)."""
    m_ = mesh.copy(); m_.points = (m_.points - np.array([0, 0, zc])) @ R.T + centre
    return m_
