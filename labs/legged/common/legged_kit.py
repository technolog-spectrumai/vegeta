"""What the legged studies repeated (``from legged_kit import pin_stress``)."""
from __future__ import annotations

import math


def pin_stress(F, d): return math.sqrt(3) * F / (2 * math.pi * d**2 / 4)
