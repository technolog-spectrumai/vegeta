"""Second-order equilibrium and the macroscopic moments (NumPy; the reference both backends follow)."""
from __future__ import annotations

import numpy as np


def moments(f: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Density ``rho = Σ f_i`` and velocity ``u = Σ f_i c_i / rho`` of populations ``f[q, ...]``."""
    rho = f.sum(axis=0)
    j = np.tensordot(c.T.astype(f.dtype), f, axes=(1, 0))
    return rho, j / rho


def equilibrium(rho: np.ndarray | float, u: np.ndarray, c: np.ndarray, w: np.ndarray) -> np.ndarray:
    """``f_i^eq = w_i ρ [1 + 3 c_i·u + 9/2 (c_i·u)² − 3/2 u·u]`` (cs² = 1/3), shape ``(q,) + u.shape[1:]``.

    ``u`` has its components first: ``u[3, ...]``.
    """
    u = np.asarray(u)
    dtype = u.dtype if np.issubdtype(u.dtype, np.floating) else np.float64
    cu = np.tensordot(c.astype(dtype), u, axes=(1, 0))
    usq = (u * u).sum(axis=0)
    wb = w.astype(dtype).reshape((-1,) + (1,) * (u.ndim - 1))
    return wb * np.asarray(rho, dtype=dtype) * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)
