"""Large-eddy simulation: the WALE eddy viscosity (Nicoud & Ducros 1999).

    ν_t = (C_w Δ)² (S^d:S^d)^{3/2} / [(S:S)^{5/2} + (S^d:S^d)^{5/4}]

with ``g_ij = ∂u_i/∂x_j``, ``S = (g + gᵀ)/2`` and ``S^d = (g² + g²ᵀ)/2 − tr(g²) I/3``. The filter width Δ is the
cell (1 in lattice units). WALE vanishes in pure shear and scales as y³ near a wall, so it leaves laminar boundary
layers alone — the reason to prefer it over Smagorinsky for propellers at moderate Reynolds numbers. It does
respond to rotation (by design: it detects vortical structures), so in a rotating frame it is evaluated with the
**lab-frame** velocity gradient, ``g + ∇(Ω×r)``: the far field, at rest in the lab and a rigid rotation in the
frame, then gets no eddy viscosity.

Lattice implementation (both backends, identically): the velocity gradient is a central difference of the
velocity of the *previous* step (so the step stays one pass over memory); where a neighbour is a solid voxel or
lies beyond a non-periodic face, the node's own velocity stands in for it (a one-sided difference of half
weight). The relaxation time becomes ``τ = 3 (ν + ν_t) + ½`` node by node.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .errors import InvalidParameterError

EPS = 1e-30


@dataclass(frozen=True)
class WALE:
    """WALE eddy viscosity with constant ``c_w`` (0.325 is OpenFOAM's default; Nicoud & Ducros quote 0.55–0.6
    for decaying isotropic turbulence)."""

    c_w: float = 0.325
    kind: str = "wale"

    def __post_init__(self) -> None:
        if not (self.c_w > 0):
            raise InvalidParameterError(f"c_w must be positive, got {self.c_w!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "WALE | None":
        return None if d is None else cls(c_w=d["c_w"])


def wale_viscosity(g: np.ndarray, c_w: float) -> np.ndarray:
    """WALE ν_t from the velocity gradient ``g[3, 3, ...]`` (``g[i, j] = ∂u_i/∂x_j``), Δ = 1."""
    s = 0.5 * (g + g.swapaxes(0, 1))
    g2 = np.einsum("ik...,kj...->ij...", g, g)
    sd = 0.5 * (g2 + g2.swapaxes(0, 1))
    tr = (g2[0, 0] + g2[1, 1] + g2[2, 2]) / 3.0
    for i in range(3):
        sd[i, i] -= tr
    ss = (s * s).sum(axis=(0, 1))
    sdsd = (sd * sd).sum(axis=(0, 1))
    num = sdsd ** 1.5
    den = ss ** 2.5 + sdsd ** 1.25
    return c_w * c_w * num / (den + EPS)


def to_lab_gradient(g: np.ndarray, omega: float) -> np.ndarray:
    """Add the gradient of the frame velocity Ω×r = (0, −ω z, ω y) to a frame-relative gradient (in place)."""
    g[1, 2] -= omega
    g[2, 1] += omega
    return g


def velocity_gradient(u: np.ndarray, solid: np.ndarray | None, periodic: tuple[bool, bool, bool]) -> np.ndarray:
    """Central differences ``g[i, j] = ∂u_i/∂x_j`` with the boundary rule of this module."""
    shape = u.shape[1:]
    g = np.empty((3, 3) + shape, dtype=u.dtype)
    for j in range(3):
        up = np.roll(u, -1, axis=1 + j)            # u(x + e_j)
        dn = np.roll(u, 1, axis=1 + j)             # u(x − e_j)
        if solid is not None:
            sp = np.roll(solid, -1, axis=j)
            sm = np.roll(solid, 1, axis=j)
            up = np.where(sp, u, up)
            dn = np.where(sm, u, dn)
        if not periodic[j]:
            last = [slice(None)] * 4
            last[1 + j] = -1
            up[tuple(last)] = u[tuple(last)]
            first = [slice(None)] * 4
            first[1 + j] = 0
            dn[tuple(first)] = u[tuple(first)]
        g[:, j] = 0.5 * (up - dn)
    return g
