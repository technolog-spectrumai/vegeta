"""The D3Q19 velocity set: 19 discrete velocities, their weights and opposites (lattice units, cs² = 1/3)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _d3q19_velocities() -> np.ndarray:
    """Rest, the 6 face neighbours, then the 12 edge neighbours; each moving velocity is followed by its opposite."""
    c = [(0, 0, 0)]
    for axis in range(3):
        e = [0, 0, 0]
        e[axis] = 1
        c += [tuple(e), tuple(-v for v in e)]
    for a, b in ((0, 1), (0, 2), (1, 2)):
        for sa, sb in ((1, 1), (-1, -1), (1, -1), (-1, 1)):
            e = [0, 0, 0]
            e[a], e[b] = sa, sb
            c.append(tuple(e))
    return np.array(c, dtype=np.int64)


@dataclass(frozen=True)
class D3Q19:
    """The D3Q19 lattice.

    ``c[i]`` is the i-th velocity (integer lattice units), ``w[i]`` its weight and ``opposite[i]`` the index of
    ``-c[i]``. Ordering: 0 rest, 1–6 faces (+x, −x, +y, −y, +z, −z), 7–18 edges, opposites adjacent.
    """

    name: str = field(default="D3Q19", init=False)
    d: int = field(default=3, init=False)
    q: int = field(default=19, init=False)
    cs2: float = field(default=1.0 / 3.0, init=False)

    @property
    def c(self) -> np.ndarray:
        return _C.copy()

    @property
    def w(self) -> np.ndarray:
        return _W.copy()

    @property
    def opposite(self) -> np.ndarray:
        return _OPP.copy()


_C = _d3q19_velocities()
_W = np.where((_C != 0).sum(axis=1) == 0, 1.0 / 3.0, np.where((_C != 0).sum(axis=1) == 1, 1.0 / 18.0, 1.0 / 36.0))
_OPP = np.array([int(np.flatnonzero((_C == -ci).all(axis=1))[0]) for ci in _C], dtype=np.int64)
