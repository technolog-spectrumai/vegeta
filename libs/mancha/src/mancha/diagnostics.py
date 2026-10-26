"""Scalar diagnostics on host fields: mass, kinetic energy, errors, convergence order, momentum-exchange force."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def total_mass(rho: np.ndarray, solid: np.ndarray | None = None) -> float:
    """Σ ρ over fluid nodes (lattice units, cell volume 1)."""
    return float(rho.sum(dtype=np.float64) if solid is None else rho[~solid].sum(dtype=np.float64))


def kinetic_energy(rho: np.ndarray, u: np.ndarray, solid: np.ndarray | None = None) -> float:
    """Σ ½ ρ |u|² over fluid nodes."""
    e = 0.5 * rho * (u * u).sum(axis=0)
    return float(e.sum(dtype=np.float64) if solid is None else e[~solid].sum(dtype=np.float64))


def l2_relative_error(numerical: np.ndarray, reference: np.ndarray) -> float:
    """‖num − ref‖₂ / ‖ref‖₂ (raises if the reference is identically zero)."""
    num = np.linalg.norm(np.asarray(numerical, dtype=np.float64) - reference)
    den = np.linalg.norm(np.asarray(reference, dtype=np.float64))
    if den == 0:
        raise ValueError("relative error against a zero reference")
    return float(num / den)


def linf_error(numerical: np.ndarray, reference: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(numerical, dtype=np.float64) - reference)))


def observed_order(resolutions: Sequence[float], errors: Sequence[float]) -> float:
    """Least-squares slope of log(error) against log(1/resolution): the observed order of convergence."""
    h = 1.0 / np.asarray(resolutions, dtype=np.float64)
    slope = np.polyfit(np.log(h), np.log(np.asarray(errors, dtype=np.float64)), 1)[0]
    return float(slope)


def momentum_exchange(f: np.ndarray, solid: np.ndarray, c: np.ndarray, periodic=(True, True, True),
                      part: np.ndarray | None = None, origin=(0.0, 0.0, 0.0), w: np.ndarray | None = None,
                      reference_density: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Force and moment about ``origin`` on solid voxels (all of them, or the subset ``part``) from the
    post-collision populations ``f*`` (lattice units).

    Momentum exchange for halfway bounce-back on a body at rest (in the simulation's frame): every link from a
    fluid node ``x`` to a solid node ``x + c_i`` carries ``2 c_i (f*_i(x) − w_i ρ_ref)`` into the body per step,
    applied at the wall point ``x + c_i/2``. Subtracting the reference state makes the pressure part gauge
    (``ρ_ref``: the simulation's reference density): on a closed body it cancels anyway, but on a *part* of a
    body (a blade on a hub, a body standing on a wall) the unwetted contact face would otherwise leave the
    absolute pressure ``cs² ρ_ref`` acting. Links that would leave the box through a non-periodic face are not
    counted.
    """
    shape = solid.shape
    target_mask = solid if part is None else (np.asarray(part, dtype=bool) & solid)
    fluid = ~solid
    grids = np.meshgrid(*(np.arange(n, dtype=np.float64) for n in shape), indexing="ij")
    force = np.zeros(3, dtype=np.float64)
    moment = np.zeros(3, dtype=np.float64)
    for i, ci in enumerate(c):
        if not ci.any():
            continue
        target = np.roll(target_mask, shift=tuple(-ci), axis=(0, 1, 2))    # the part at x + c_i
        valid = np.ones(shape, dtype=bool)
        for a in range(3):
            if ci[a] != 0 and not periodic[a]:
                sl = [slice(None)] * 3
                sl[a] = -1 if ci[a] > 0 else 0
                valid[tuple(sl)] = False
        links = target & fluid & valid
        if not links.any():
            continue
        ref = 0.0 if w is None else float(w[i]) * reference_density
        fl = 2.0 * (f[i][links].astype(np.float64) - ref)          # momentum per link along c_i
        force += ci * fl.sum()
        r = np.stack([g[links] + 0.5 * ci[a] - origin[a] for a, g in enumerate(grids)])
        moment += np.cross(r.T, np.outer(fl, ci)).sum(axis=0)
    return force, moment


def momentum_exchange_force(f: np.ndarray, solid: np.ndarray, c: np.ndarray, periodic=(True, True, True)
                            ) -> np.ndarray:
    """The force of :func:`momentum_exchange` alone (absolute pressure, no reference subtracted)."""
    return momentum_exchange(f, solid, c, periodic)[0]
