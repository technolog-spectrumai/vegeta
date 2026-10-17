"""The plate's heat problems on FEniCSx (dolfinx ≥ 0.8) — the tested finite-element route for what ``thermal.py`` does
with its own finite volumes: the steady state of a plate under a beam (``steady_plate_fenicsx``) and the transient with
melting and ablation (``heat_plate_fenicsx``). Both take the same arguments and return the same shapes as the numpy
functions, so ``thermal.steady_plate(..., backend="fenicsx")`` and ``thermal.heat_plate(..., backend="fenicsx")`` are
drop-in; ``backend="auto"`` takes FEniCSx when it is installed and falls back to numpy otherwise, saying so in the
result's ``note``/``backend``.

Axisymmetric r–z on a rectangle (r in [0, r_max], z in [−d, 0], the exposed face at z = 0), P1 Lagrange elements,
the weak form weighted by r. Steady: Newton on the nonlinear radiation (dolfinx's ``NonlinearProblem``). Transient:
implicit Euler, the apparent-heat-capacity form of the enthalpy method (the latent heats spread over ±``band`` K),
Picard on the capacity and the linearised radiation, ablation by cell deactivation once a cell held at the failure
point has taken its latent heat of removal, the beam's source moved to the next live cell. Linear solves by PETSc
(CG + hypre BoomerAMG). Installation: docs/fenics.md."""
from __future__ import annotations

import math

import numpy as np

from .thermal import SIGMA_SB, T_ROOM, HeatingResult, Solid, SOLIDS, irradiance

try:
    import ufl
    from dolfinx import fem, mesh
    from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
    from dolfinx.nls.petsc import NewtonSolver
    from mpi4py import MPI
    from petsc4py import PETSc
    HAVE_DOLFINX = True
except ImportError:
    HAVE_DOLFINX = False


def available() -> bool:
    return HAVE_DOLFINX


def _solid(s):
    return s if isinstance(s, Solid) else SOLIDS[str(s).lower()]


def _setup(s: Solid, thickness, radius, r_max, nr, nz):
    comm = MPI.COMM_WORLD
    msh = mesh.create_rectangle(comm, [np.array([0.0, -thickness]), np.array([r_max, 0.0])], [nr, nz], cell_type=mesh.CellType.quadrilateral)
    V = fem.functionspace(msh, ("Lagrange", 1))
    fdim = msh.topology.dim - 1
    top = mesh.locate_entities_boundary(msh, fdim, lambda p: np.isclose(p[1], 0.0))
    bottom = mesh.locate_entities_boundary(msh, fdim, lambda p: np.isclose(p[1], -thickness))
    edge = mesh.locate_entities_boundary(msh, fdim, lambda p: np.isclose(p[0], r_max))
    values = np.concatenate([np.full(len(top), 1, np.int32), np.full(len(bottom), 2, np.int32), np.full(len(edge), 3, np.int32)])
    tags = mesh.meshtags(msh, fdim, np.concatenate([top, bottom, edge]), values)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=tags)
    return msh, V, ds, edge


def _beam_flux(s: Solid, power, radius, shape, msh):
    """The absorbed irradiance on the exposed face as a UFL expression of r."""
    x = ufl.SpatialCoordinate(msh)
    r = x[0]
    if shape == "gaussian":
        return s.absorptivity * 2 * power / (math.pi * radius ** 2) * ufl.exp(-2 * (r / radius) ** 2)
    return s.absorptivity * power / (math.pi * radius ** 2) * ufl.conditional(ufl.le(r, radius), 1.0, 0.0)


def steady_plate_fenicsx(solid, thickness: float, power: float, radius: float, shape: str = "gaussian", h: float = 10.0, T0: float = T_ROOM,
                         nr: int = 32, nz: int = 8, r_max: float | None = None, tol: float = 1e-6, **_) -> dict:
    """The steady conduction with convection and radiation at both faces (nonlinear, Newton), the far edge at T0."""
    if not HAVE_DOLFINX:
        raise ImportError("dolfinx is not installed (docs/fenics.md)")
    s = _solid(solid)
    if r_max is None:
        r_max = max(8.0 * radius, 12.0 * thickness, 0.02)
    msh, V, ds, edge = _setup(s, thickness, radius, r_max, nr, nz)
    x = ufl.SpatialCoordinate(msh)
    r = x[0]
    T = fem.Function(V, name="T")
    T.x.array[:] = T0
    v = ufl.TestFunction(V)
    q = _beam_flux(s, power, radius, shape, msh)
    loss = lambda Tf: h * (Tf - T0) + s.emissivity * SIGMA_SB * (Tf ** 4 - T0 ** 4)
    F = s.k * ufl.dot(ufl.grad(T), ufl.grad(v)) * r * ufl.dx - q * v * r * ds(1) + loss(T) * v * r * (ds(1) + ds(2))
    bc = fem.dirichletbc(PETSc.ScalarType(T0), fem.locate_dofs_topological(V, msh.topology.dim - 1, edge), V)
    problem = NonlinearProblem(F, T, bcs=[bc])
    solver = NewtonSolver(msh.comm, problem)
    solver.rtol, solver.max_it = tol, 60
    ksp = solver.krylov_solver
    ksp.setType("cg")
    ksp.getPC().setType("hypre")
    its, converged = solver.solve(T)
    vals = T.x.array
    coords = V.tabulate_dof_coordinates()
    on_axis = np.isclose(coords[:, 0], 0.0)
    damage_T = s.T_melt if s.failure == "melt" else s.T_fail
    return {"T_max": float(vals.max()), "T_top_centre": float(vals[on_axis & np.isclose(coords[:, 1], 0.0)].max()),
            "T_bottom_centre": float(vals[on_axis & np.isclose(coords[:, 1], -thickness)].max()), "damage_T": damage_T,
            "reaches_damage": bool(vals.max() >= damage_T), "iterations": int(its), "converged": bool(converged), "T_field": T, "mesh": msh,
            "transmitted_fraction": s.transmitted(thickness), "backend": "fenicsx"}


def heat_plate_fenicsx(solid, thickness: float, power: float, radius: float, t_max: float, shape: str = "gaussian", h: float = 10.0, T0: float = T_ROOM,
                       nr: int = 32, nz: int = 8, r_max: float | None = None, stop_at_failure: bool = True, dt: float | None = None, band: float = 15.0,
                       picard: int = 6, record_every: float | None = None, **_) -> HeatingResult:
    """The transient with the apparent-heat-capacity enthalpy method and ablation by cell deactivation; returns the
    same ``HeatingResult`` as the numpy solver (T_field as a (nz, nr) array sampled at the cells' centres)."""
    if not HAVE_DOLFINX:
        raise ImportError("dolfinx is not installed (docs/fenics.md)")
    s = _solid(solid)
    if r_max is None:
        r_max = max(8.0 * radius, 12.0 * thickness, 0.02)
    msh, V, ds, edge = _setup(s, thickness, radius, r_max, nr, nz)
    D0 = fem.functionspace(msh, ("DG", 0))
    x = ufl.SpatialCoordinate(msh)
    r = x[0]
    dz, dr = thickness / nz, r_max / nr
    q_face = s.absorptivity * irradiance(power, radius, (np.arange(nr) + 0.5) * dr, shape)      # per column, W/m²
    if dt is None:
        dt = min(0.02 * s.enthalpy_to_fail(T0) * dz / max(float(q_face.max()), 1e-9), t_max / 200.0)
    # the cells: their column and depth order, for the live-surface source
    ncell = msh.topology.index_map(msh.topology.dim).size_local
    conn = msh.topology.connectivity(msh.topology.dim, 0)
    mids = np.array([msh.geometry.x[conn.links(c)].mean(axis=0) for c in range(ncell)])
    col = np.clip((mids[:, 0] / dr).astype(int), 0, nr - 1)
    layer = np.clip((-mids[:, 1] / dz).astype(int), 0, nz - 1)                                 # 0 at the exposed face
    alive = np.ones(ncell, bool)
    removal = np.zeros(ncell)
    T_n = fem.Function(V); T_n.x.array[:] = T0
    T_k = fem.Function(V); T_k.x.array[:] = T0
    src = fem.Function(D0); k_eff = fem.Function(D0); c_eff = fem.Function(D0)
    dtc = fem.Constant(msh, PETSc.ScalarType(dt))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    bc = fem.dirichletbc(PETSc.ScalarType(T0), fem.locate_dofs_topological(V, msh.topology.dim - 1, edge), V)
    T_cell = fem.Function(D0)
    expr = fem.Expression(T_k, D0.element.interpolation_points())

    def capacity(Tf):
        rho_c = s.rho * s.c
        melt = (s.rho * s.L_melt / (2 * band)) * ufl.conditional(ufl.lt(abs(Tf - s.T_melt), band), 1.0, 0.0) if s.L_melt > 0 else 0.0
        fail = (s.rho * s.L_fail / (2 * band)) * ufl.conditional(ufl.lt(abs(Tf - s.T_fail), band), 1.0, 0.0)
        return rho_c + melt + fail

    rows = {"t": [], "T_top": [], "T_bottom": [], "T_max": [], "melt": [], "abl": []}
    rec_dt = record_every or max(dt, t_max / 400.0)
    t, next_rec = 0.0, 0.0
    damage, failure, mode = math.nan, math.nan, ""
    damage_T = s.T_melt if s.failure == "melt" else s.T_fail
    while t < t_max:
        src_arr = np.zeros(ncell)
        for j in range(nr):
            cells = np.where((col == j) & alive)[0]
            if len(cells):
                src_arr[cells[np.argmin(layer[cells])]] = q_face[j] / dz
        src.x.array[:] = src_arr
        k_eff.x.array[:] = np.where(alive, s.k, 1e-6 * s.k)
        c_eff.x.array[:] = np.where(alive, 1.0, 1e-6)
        for _ in range(picard):
            cap = c_eff * capacity(T_k)
            h_rad = s.emissivity * SIGMA_SB * (T_k ** 2 + T0 ** 2) * (T_k + T0)
            a = (cap * u * v / dtc + k_eff * ufl.dot(ufl.grad(u), ufl.grad(v))) * r * ufl.dx + (h + h_rad) * u * v * r * (ds(1) + ds(2))
            L = (cap * T_n * v / dtc + src * v) * r * ufl.dx + (h + h_rad) * T0 * v * r * (ds(1) + ds(2))
            T_new = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "cg", "pc_type": "hypre"}).solve()
            change = float(np.abs(T_new.x.array - T_k.x.array).max())
            T_k.x.array[:] = T_new.x.array
            if change < 0.5:
                break
        T_cell.interpolate(expr)
        Tc = T_cell.x.array
        hot = alive & (Tc >= s.T_fail - band)
        removal[hot] += src_arr[hot] * dt
        gone = hot & (removal >= s.rho * s.L_fail)
        alive[gone] = False
        T_n.x.array[:] = T_k.x.array
        t += dt
        centre = np.where(col == 0)[0]
        live_c = centre[alive[centre]]
        abl = float(np.sum(~alive[centre]) * dz)
        melt = float(np.sum((Tc[live_c] >= s.T_melt - band)) * dz) if len(live_c) else 0.0
        T_top = float(Tc[live_c].max()) if len(live_c) else math.nan
        if math.isnan(damage) and T_top >= damage_T - band:
            damage = t
        if math.isnan(failure):
            if s.failure == "ablate" and len(live_c) == 0:
                failure, mode = t, "ablated through at the centre"
            elif s.failure == "melt" and (abl + melt >= thickness - 1e-12 or len(live_c) == 0):
                failure, mode = t, "melt-through at the centre"
        if t >= next_rec or not math.isnan(failure):
            rows["t"].append(t); rows["T_top"].append(T_top); rows["T_bottom"].append(float(Tc[centre[layer[centre] == nz - 1]].max()) if len(centre) else math.nan)
            rows["T_max"].append(float(Tc[alive].max()) if alive.any() else T0); rows["melt"].append(melt); rows["abl"].append(abl)
            next_rec = t + rec_dt
        if not math.isnan(failure) and stop_at_failure:
            break
    field = np.full((nz, nr), np.nan)
    for c in range(ncell):
        if alive[c]:
            field[layer[c], col[c]] = Tc[c]
    res = HeatingResult(s, thickness, power, radius, shape, np.array(rows["t"]), np.array(rows["T_top"]), np.array(rows["T_bottom"]), np.array(rows["T_max"]),
                        np.array(rows["melt"]), np.array(rows["abl"]), damage, failure, mode, False, t, s.transmitted(thickness), field,
                        (np.arange(nr) + 0.5) * dr, (np.arange(nz) + 0.5) * dz, note="backend fenicsx")
    return res
