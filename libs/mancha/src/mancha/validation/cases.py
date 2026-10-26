"""Validation cases. Each builds a :class:`~mancha.Simulation`, runs it and returns its metrics as a dict.

The automated tests (``tests/``) call them on small grids with loose, regression-style tolerances; the
benchmark scripts (``benchmarks/``) call them on finer grids and in refinement series for the scientific record.
Everything is in lattice units.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..boundaries import sphere as sphere_mask
from ..diagnostics import l2_relative_error, linf_error
from ..simulation import Simulation
from . import references as ref


def _steady(sim: Simulation, max_steps: int, interval: int, tol: float) -> tuple[int, float]:
    """Run until the relative velocity change over ``interval`` steps falls below ``tol``."""
    u_old = sim.velocity()
    change = math.inf
    while sim.steps_done < max_steps:
        sim.run(interval)
        u = sim.velocity()
        change = float(np.linalg.norm(u - u_old) / max(np.linalg.norm(u), 1e-30))
        u_old = u
        if change < tol:
            break
    return sim.steps_done, change


def uniform(shape=(16, 16, 16), velocity=(0.0, 0.0, 0.0), viscosity: float = 0.1, steps: int = 200,
            backend: str = "numpy", **kw: Any) -> dict[str, Any]:
    """Uniform density and velocity on a periodic box: an exact steady solution that must be preserved."""
    sim = Simulation(shape=shape, viscosity=viscosity, velocity=velocity, backend=backend, **kw)
    m0 = sim.mass()
    sim.run(steps)
    u, rho = sim.velocity(), sim.density()
    target = np.asarray(velocity, dtype=float).reshape(3, 1, 1, 1)
    return {"velocity_error_max": float(np.abs(u - target).max()), "density_error_max": float(np.abs(rho - 1).max()),
            "mass_drift_rel": (sim.mass() - m0) / m0, "steps": steps}


def taylor_green(n: int = 32, viscosity: float = 0.02, u0: float = 0.04, steps: int | None = None,
                 nz: int = 1, backend: str = "numpy", **kw: Any) -> dict[str, Any]:
    """2D Taylor–Green vortex (z-invariant, periodic box ``n × n × nz``), an exact Navier–Stokes solution.

    ``u = U sin(kx) cos(ky)``, ``v = −U cos(kx) sin(ky)`` with ``k = 2π/n``, pressure initialised
    consistently. Compares the energy decay rate with ``2ν(kx² + ky²)`` and the final field with the decayed
    analytical one.
    """
    k = 2.0 * math.pi / n
    rate = ref.taylor_green_decay_rate(viscosity, k, k)
    steps = steps if steps is not None else int(round(math.log(2.0) / rate))      # one energy half-life
    x = np.arange(n, dtype=np.float64)
    X, Y = np.meshgrid(x, x, indexing="ij")
    ux = (u0 * np.sin(k * X) * np.cos(k * Y))[:, :, None].repeat(nz, axis=2)
    uy = (-u0 * np.cos(k * X) * np.sin(k * Y))[:, :, None].repeat(nz, axis=2)
    p = (u0 * u0 / 4.0) * (np.cos(2 * k * X) + np.cos(2 * k * Y))
    rho = (1.0 + 3.0 * p)[:, :, None].repeat(nz, axis=2)
    sim = Simulation(shape=(n, n, nz), viscosity=viscosity, backend=backend, **kw)
    sim.initialize(density=rho, velocity=np.stack([ux, uy, np.zeros_like(ux)]))
    m0, e0 = sim.mass(), sim.kinetic_energy()
    sim.run(steps)
    e1 = sim.kinetic_energy()
    measured = -math.log(e1 / e0) / steps
    decay = math.exp(-0.5 * rate * steps)
    u = sim.velocity()
    exact = np.stack([ux * decay, uy * decay, np.zeros_like(ux)])
    return {"n": n, "steps": steps, "decay_rate": measured, "decay_rate_exact": rate,
            "decay_rate_error_rel": measured / rate - 1.0,
            "velocity_l2_error_rel": l2_relative_error(u, exact),
            "mass_drift_rel": (sim.mass() - m0) / m0, "simulation": sim}


def poiseuille(ny: int = 16, viscosity: float = 1 / 6, u_max: float = 0.02, nx: int = 4, nz: int = 1,
               max_steps: int = 40000, tol: float = 1e-9, backend: str = "numpy", **kw: Any) -> dict[str, Any]:
    """Body-force-driven channel between walls at ``y−``/``y+`` (halfway: at ``y = −½`` and ``ny − ½``),
    periodic in x and z. Compares the steady profile with the parabola ``F/(2ν) y'(h − y')``, ``h = ny``."""
    h = float(ny)
    force = 8.0 * viscosity * u_max / h ** 2
    sim = Simulation(shape=(nx, ny, nz), viscosity=viscosity, force=(force, 0.0, 0.0), backend=backend, **kw)
    sim.set_boundary("y-", "wall").set_boundary("y+", "wall")
    m0 = sim.mass()
    steps, change = _steady(sim, max_steps, 500, tol)
    y = np.arange(ny) + 0.5
    exact = ref.poiseuille_force_profile(y, h, force, viscosity)
    profile = sim.velocity()[0].mean(axis=(0, 2))
    return {"ny": ny, "steps": steps, "residual": change, "l2_error_rel": l2_relative_error(profile, exact),
            "linf_error_rel": linf_error(profile, exact) / u_max, "profile": profile, "exact": exact, "y": y,
            "mass_drift_rel": (sim.mass() - m0) / m0, "simulation": sim}


def channel(nx: int = 64, ny: int = 16, nz: int = 1, viscosity: float = 0.1, u_in: float = 0.02,
            max_steps: int = 40000, tol: float = 1e-8, backend: str = "numpy", **kw: Any) -> dict[str, Any]:
    """Velocity inlet (``x−``, uniform ``u_in``) to pressure outlet (``x+``, ρ = 1) between walls at ``y±``,
    periodic in z. Checks the developed profile near the outlet against the parabola of the same mean velocity,
    and the flux along the channel."""
    sim = Simulation(shape=(nx, ny, nz), viscosity=viscosity, backend=backend, **kw)
    sim.set_boundary("x-", "velocity", velocity=(u_in, 0, 0)).set_boundary("x+", "pressure", density=1.0)
    sim.set_boundary("y-", "wall").set_boundary("y+", "wall")
    steps, change = _steady(sim, max_steps, 500, tol)
    rho, u = sim.density(), sim.velocity()
    flux = (rho * u[0]).sum(axis=(1, 2))                   # mass flux through each x plane
    station = int(0.75 * nx)
    profile = u[0, station].mean(axis=1)
    mean = profile.mean()
    y = np.arange(ny) + 0.5
    exact = ref.poiseuille_mean_profile(y, float(ny), mean)
    return {"steps": steps, "residual": change, "profile": profile, "exact": exact, "station": station,
            "profile_l2_error_rel": l2_relative_error(profile, exact), "peak_ratio": profile.max() / mean,
            "flux_inlet": float(flux[0]), "flux_outlet": float(flux[-1]),
            "flux_variation_rel": float((flux.max() - flux.min()) / flux.mean()),
            "inflow_mean_over_u_in": float(u[0, 0].mean() / u_in), "simulation": sim}


def cavity(n: int = 64, reynolds: float = 100.0, u_lid: float = 0.05, nz: int = 1, max_steps: int = 200000,
           tol: float = 1e-7, backend: str = "numpy", **kw: Any) -> dict[str, Any]:
    """Lid-driven cavity, walls at ``x±`` and ``y−``, lid ``y+`` moving in +x; ``z`` periodic with ``nz``
    layers, so the flow is two-dimensional and comparable with Ghia et al. (1982). Positions are normalised by
    the cavity side ``n`` with the halfway walls at 0 and 1 (node ``i`` at ``(i + ½)/n``)."""
    viscosity = u_lid * n / reynolds
    sim = Simulation(shape=(n, n, nz), viscosity=viscosity, backend=backend, **kw)
    for face in ("x-", "x+", "y-"):
        sim.set_boundary(face, "wall")
    sim.set_boundary("y+", "wall", velocity=(u_lid, 0, 0))
    steps, change = _steady(sim, max_steps, 1000, tol)
    u = sim.velocity().mean(axis=3) / u_lid                # average over z: [3, n, n]
    s = (np.arange(n) + 0.5) / n
    s_full = np.concatenate([[0.0], s, [1.0]])
    mid = (n - 1) / 2.0                                    # centreline between the two middle nodes for even n
    lo, hi = int(math.floor(mid)), int(math.ceil(mid))
    u_centre = 0.5 * (u[0, lo, :] + u[0, hi, :])           # u along y at x = 0.5
    v_centre = 0.5 * (u[1, :, lo] + u[1, :, hi])           # v along x at y = 0.5
    u_at = np.interp(ref.GHIA_RE100_U[:, 0], s_full, np.concatenate([[0.0], u_centre, [1.0]]))
    v_at = np.interp(ref.GHIA_RE100_V[:, 0], s_full, np.concatenate([[0.0], v_centre, [0.0]]))
    psi = np.cumsum(u[0], axis=1) / n                      # stream function ψ = ∫ u dy (ψ = 0 on the bottom wall)
    i, j = np.unravel_index(np.argmin(psi), psi.shape)     # primary vortex: the extremum of ψ
    out = {"n": n, "reynolds": reynolds, "steps": steps, "residual": change,
           "u_centreline": (s, u_centre), "v_centreline": (s, v_centre),
           "vortex_centre": (float(s[i]), float(s[j])), "mass": sim.mass(), "simulation": sim}
    if abs(reynolds - 100.0) < 1e-9:
        out["ghia_u_max_abs_error"] = float(np.abs(u_at - ref.GHIA_RE100_U[:, 1]).max())
        out["ghia_v_max_abs_error"] = float(np.abs(v_at - ref.GHIA_RE100_V[:, 1]).max())
        cx, cy = ref.GHIA_RE100_PRIMARY_VORTEX
        out["vortex_centre_error"] = float(math.hypot(s[i] - cx, s[j] - cy))
    return out


def sphere(diameter: float = 10.0, reynolds: float = 20.0, u_in: float = 0.04, size=(8.0, 4.0, 4.0),
           max_steps: int = 100000, tol: float = 1e-7, backend: str = "numpy", **kw: Any) -> dict[str, Any]:
    """Uniform inflow past a voxelised sphere: velocity inlet ``x−``, pressure outlet ``x+``, periodic laterally
    (a square array of spheres). ``size`` is the box in diameters; the sphere sits at 1/4 of the length.

    Reports the drag coefficient (momentum exchange) next to the unconfined Schiller–Naumann value, the
    blockage ratio and the recirculation length behind the sphere. The periodic lateral images and the voxel
    staircase both raise the drag; this is a characterisation, not a validated drag prediction.
    """
    nx, ny, nz = (int(round(s * diameter)) for s in size)
    viscosity = u_in * diameter / reynolds
    centre = (nx / 4.0, (ny - 1) / 2.0, (nz - 1) / 2.0)
    mask = sphere_mask((nx, ny, nz), centre, diameter / 2.0)
    sim = Simulation(shape=(nx, ny, nz), viscosity=viscosity, velocity=(u_in, 0, 0), backend=backend, **kw)
    sim.set_boundary("x-", "velocity", velocity=(u_in, 0, 0)).set_boundary("x+", "pressure", density=1.0)
    sim.set_solid(mask)
    steps, change = _steady(sim, max_steps, 500, tol)
    force = sim.solid_force()
    area = math.pi * diameter ** 2 / 4.0
    cd = 2.0 * force[0] / (1.0 * u_in ** 2 * area)
    u = sim.velocity()
    jc, kc = int(round(centre[1])), int(round(centre[2]))
    line = u[0, :, jc, kc]
    back = int(math.floor(centre[0] + diameter / 2.0)) + 1
    behind = line[back:]
    reverse = np.flatnonzero(behind < 0)
    wake = 0.0 if reverse.size == 0 else (reverse.max() + 1) / diameter
    return {"reynolds": reynolds, "diameter": diameter, "shape": (nx, ny, nz), "steps": steps, "residual": change,
            "cd": float(cd), "cd_schiller_naumann": ref.schiller_naumann_cd(reynolds),
            "blockage_ratio": area / (ny * nz), "solid_voxels": int(mask.sum()),
            "voxel_volume_ratio": float(mask.sum() / (math.pi * diameter ** 3 / 6.0)),
            "lateral_force": force[1:].tolist(), "wake_length_over_d": float(wake), "simulation": sim}
