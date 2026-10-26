"""Regularized BGK (Latt & Chopard 2006): relax only the hydrodynamic part of the non-equilibrium.

Before relaxing, ``f − f^eq`` is replaced by its projection on the first two Hermite polynomials,

    f_i^neq ≈ w_i [3 c_i·J + 9/2 (c_i c_i − I/3) : Π],   J = Σ c_i (f_i − f_i^eq),   Π = Σ c_i c_i (f_i − f_i^eq),

and ``f* = f^eq + (1 − ω) f^neq + S`` (Guo's source ``S``). The projection removes the non-hydrodynamic
(ghost) modes that make BGK unstable as τ → ½, the regime of LES at high Reynolds numbers; the viscosity is the
same as BGK's. The first-order term matters only with a force: Guo's half-force velocity leaves ``J = −F/2``,
and dropping it would make the momentum wrong by ``F(1 − ω)/2`` (seen as an O(1) error in forced Poiseuille
flow before it was kept). With halfway bounce-back the wall slip differs from BGK's (smaller in the Poiseuille
tests at τ = 0.6–1.2) and is still τ-dependent.
"""
from __future__ import annotations

from typing import ClassVar

import numpy as np

from .bgk import BGK, guo_source
from .equilibrium import equilibrium


class RegularizedBGK(BGK):
    """Regularized single relaxation time; parameters and validation as :class:`BGK`."""

    kind: ClassVar[str] = "rbgk"

    def collide(self, f: np.ndarray, rho: np.ndarray, u: np.ndarray, c: np.ndarray, w: np.ndarray,
                force: np.ndarray | None = None, nu_t: np.ndarray | None = None) -> np.ndarray:
        feq = equilibrium(rho, u, c, w).astype(f.dtype, copy=False)
        cf = c.astype(f.dtype)
        neq = f - feq
        j1 = np.tensordot(cf.T, neq, axes=(1, 0))                                   # [3, ...]
        pi = np.einsum("ia,ib,i...->ab...", cf, cf, neq)                            # [3, 3, ...]
        q = np.einsum("ia,ib->iab", cf, cf) - np.eye(3, dtype=f.dtype) / 3.0
        shape = (-1,) + (1,) * (f.ndim - 1)
        rec = w.astype(f.dtype).reshape(shape) * (3.0 * np.tensordot(cf, j1, axes=(1, 0))
                                                  + 4.5 * np.einsum("iab,ab...->i...", q, pi))
        omega = self.omega if nu_t is None else 1.0 / (3.0 * (self.viscosity + nu_t) + 0.5)
        out = feq + (1.0 - np.asarray(omega, dtype=f.dtype)) * rec
        if force is not None and np.any(force):
            out += guo_source(u, c, w, force, omega).astype(f.dtype, copy=False)
        return out

    def __repr__(self) -> str:
        return f"RegularizedBGK(tau={self.tau!r})"
