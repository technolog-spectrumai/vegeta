"""Viscous sponge layers along faces of the box.

Within ``width`` cells of a listed face the viscosity rises smoothly to ``ν + viscosity`` at the face,
``ν_s(d) = viscosity · ((width − d)/width)²`` with ``d`` the distance from the halfway wall plane. Two uses:

* stability where a face condition is stiff — the far-field velocity faces of a rotating frame carry large
  tangential speeds at the box edges, where bounce-back with a wall velocity went unstable at low viscosity (a
  6-cell, 0.05 sponge cured it; 0.02 did not);
* damping outgoing disturbances before they reach a reflecting face (an absorbing layer for acoustics; a smooth
  profile reflects less than a step).

The flow inside the sponge is not physical; keep bodies and probes outside it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..errors import InvalidParameterError
from .conditions import FACES


@dataclass(frozen=True)
class Sponge:
    width: float
    viscosity: float
    faces: tuple[str, ...] = ("y-", "y+", "z-", "z+")

    def __post_init__(self) -> None:
        if not (self.width > 0 and self.viscosity > 0):
            raise InvalidParameterError("a sponge needs a positive width and viscosity")
        unknown = set(self.faces) - set(FACES)
        if unknown:
            raise InvalidParameterError(f"unknown faces {sorted(unknown)}")
        object.__setattr__(self, "faces", tuple(self.faces))

    def flags(self) -> tuple[bool, ...]:
        return tuple(f in self.faces for f in FACES)

    def field(self, shape: tuple[int, int, int]) -> np.ndarray:
        """The extra viscosity at every node, ``[nx, ny, nz]``."""
        out = np.zeros(shape)
        for k, on in enumerate(self.flags()):
            if not on:
                continue
            axis, high = divmod(k, 2)
            i = np.arange(shape[axis], dtype=np.float64)
            d = (shape[axis] - 0.5 - i) if high else (i + 0.5)
            prof = np.where(d < self.width, self.viscosity * ((self.width - d) / self.width) ** 2, 0.0)
            bshape = [1, 1, 1]
            bshape[axis] = -1
            out = np.maximum(out, prof.reshape(bshape))
        return out

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"faces": list(self.faces)}

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "Sponge | None":
        return None if d is None else cls(d["width"], d["viscosity"], tuple(d["faces"]))
