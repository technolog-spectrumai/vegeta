"""BGK / single-relaxation-time collision: ``f* = f − (f − f^eq) / tau``, with Guo forcing for a body force.

Guo et al. (2002): with a force density ``F`` the equilibrium uses ``u = (Σ f c + F/2) / ρ`` and the source
``S_i = (1 − 1/2tau) w_i [3 (c_i − u) + 9 (c_i·u) c_i]·F`` is added. Moments of the post-collision populations
then carry ``ρu + F/2`` — the backends subtract ``F/2`` when reporting the velocity.
"""
from __future__ import annotations

import math
import warnings
from typing import Any, ClassVar

import numpy as np

from ..errors import InvalidParameterError
from .base import CollisionModel
from .equilibrium import equilibrium

CS2 = 1.0 / 3.0
TAU_MARGINAL = 0.51   # below this BGK is close to its stability limit even at small velocities
TAU_HIGH = 2.0        # above this the wall location of bounce-back drifts with tau (a viscosity-dependent slip)


def tau_from_viscosity(nu: float) -> float:
    """Relaxation time for a lattice viscosity: ``nu = cs² (tau − 1/2)``."""
    return nu / CS2 + 0.5


def viscosity_from_tau(tau: float) -> float:
    return CS2 * (tau - 0.5)


class BGK(CollisionModel):
    """Single relaxation time ``tau`` (lattice units). ``tau <= 1/2`` (zero or negative viscosity) is refused."""

    kind: ClassVar[str] = "bgk"

    def __init__(self, tau: float):
        tau = float(tau)
        if not math.isfinite(tau) or tau <= 0.5:
            raise InvalidParameterError(f"BGK needs tau > 1/2 (a positive viscosity), got tau = {tau!r}")
        if tau < TAU_MARGINAL:
            warnings.warn(f"tau = {tau:.4g} is close to the BGK stability limit 1/2; expect instability unless "
                          "velocities are very small", stacklevel=2)
        elif tau > TAU_HIGH:
            warnings.warn(f"tau = {tau:.4g} > {TAU_HIGH}: halfway bounce-back walls are then displaced by a "
                          "tau-dependent amount (BGK artefact); TRT fixes this", stacklevel=2)
        self.tau = tau

    @classmethod
    def from_viscosity(cls, nu: float) -> "BGK":
        if not math.isfinite(nu) or nu <= 0:
            raise InvalidParameterError(f"the lattice viscosity must be positive, got {nu!r}")
        return cls(tau_from_viscosity(nu))

    @property
    def omega(self) -> float:
        return 1.0 / self.tau

    @property
    def viscosity(self) -> float:
        return viscosity_from_tau(self.tau)

    def collide(self, f: np.ndarray, rho: np.ndarray, u: np.ndarray, c: np.ndarray, w: np.ndarray,
                force: np.ndarray | None = None) -> np.ndarray:
        feq = equilibrium(rho, u, c, w).astype(f.dtype, copy=False)
        out = f - (f - feq) * f.dtype.type(self.omega)
        if force is not None and np.any(force):
            out += guo_source(u, c, w, force, self.omega).astype(f.dtype, copy=False)
        return out

    def parameters(self) -> dict[str, Any]:
        return {"kind": self.kind, "tau": self.tau}

    def __repr__(self) -> str:
        return f"BGK(tau={self.tau!r})"


def guo_source(u: np.ndarray, c: np.ndarray, w: np.ndarray, force: np.ndarray, omega: float) -> np.ndarray:
    """Guo's forcing term ``(1 − ω/2) w_i [3 (c_i − u) + 9 (c_i·u) c_i]·F`` for a uniform force ``F[3]``."""
    force = np.asarray(force, dtype=u.dtype)
    cf = c.astype(u.dtype) @ force                                    # c_i·F, shape (q,)
    uf = np.tensordot(force, u, axes=(0, 0))                          # u·F
    cu = np.tensordot(c.astype(u.dtype), u, axes=(1, 0))              # c_i·u
    shape = (-1,) + (1,) * (u.ndim - 1)
    return (1.0 - 0.5 * omega) * w.reshape(shape) * (3.0 * (cf.reshape(shape) - uf) + 9.0 * cu * cf.reshape(shape))
