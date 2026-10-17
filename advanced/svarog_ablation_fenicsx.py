"""The guard's ablation as a field problem in FEniCSx (dolfinx ≥ 0.8): a barrier under the stray beam, axisymmetric
r–z, transient heat conduction with the apparent-heat-capacity form of the enthalpy method (melting and the
decomposition/vaporisation plateau), convection and radiation at the faces, and ablation by cell deactivation — a cell
whose temperature has sat at the failure point long enough to take its latent heat of removal is switched off (its
conductivity and capacity go to a residual) and the beam's source moves down to the next live cell. Writes XDMF (the
temperature, the live indicator, the deposited enthalpy) for PyVista and notebook 50, and a JSON of the surface-damage
and burn-through times for the comparison with ``hyperion.thermal.heat_plate``.

Not run in the repository's cloud sessions (dolfinx is a conda-forge / Docker install, see docs/fenics.md). Run:

    python advanced/svarog_ablation_fenicsx.py --material black_pc --thickness 10e-3 --power 933 --radius 17e-3 --t-max 40 --out advanced/runs/pc10

The material numbers are Hyperion's (`vegeta.hyperion.thermal.SOLIDS`), so both solvers see the same inputs."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

try:
    import ufl
    from dolfinx import fem, io, mesh
    from dolfinx.fem.petsc import LinearProblem
    from mpi4py import MPI
    from petsc4py import PETSc
    HAVE_DOLFINX = True
except ImportError:                                            # the script imports without dolfinx so its argument parsing and the material table can be tested
    HAVE_DOLFINX = False

from vegeta.hyperion import thermal as th

SIGMA = th.SIGMA_SB


ALIASES = {"black_pc": "black polycarbonate", "clear_pc": "clear polycarbonate", "laser_pc": "laser-safety polycarbonate", "black_pe": "black polyethylene",
           "brick": "fired clay brick", "mild_steel": "mild steel", "stainless": "stainless 304", "aluminium": "aluminium"}


def material(name: str) -> th.Solid:
    return th.solid(ALIASES.get(name, name.replace("_", " ")))


def build_mesh(comm, thickness: float, r_max: float, nr: int, nz: int):
    """A rectangle in (r, z): r in [0, r_max], z in [−thickness, 0] with the exposed face at z = 0."""
    return mesh.create_rectangle(comm, [np.array([0.0, -thickness]), np.array([r_max, 0.0])], [nr, nz], cell_type=mesh.CellType.quadrilateral)


def apparent_capacity(s: th.Solid, T, band: float = 15.0):
    """ρ c_app(T): the heat capacity with the latent heats spread over ±band K around T_melt and T_fail (UFL expression)."""
    rho_c = s.rho * s.c
    melt = s.rho * s.L_melt / (2 * band) * ufl.conditional(ufl.lt(abs(T - s.T_melt), band), 1.0, 0.0) if s.L_melt > 0 else 0.0
    fail = s.rho * s.L_fail / (2 * band) * ufl.conditional(ufl.lt(abs(T - s.T_fail), band), 1.0, 0.0)
    return rho_c + melt + fail


def run(args) -> dict:
    if not HAVE_DOLFINX:
        raise SystemExit("dolfinx is not installed: see docs/fenics.md (conda-forge: fenics-dolfinx, or the dolfinx/dolfinx Docker image)")
    s = material(args.material)
    comm = MPI.COMM_WORLD
    r_max = args.r_max or max(6 * args.radius, 12 * args.thickness, 0.05)
    msh = build_mesh(comm, args.thickness, r_max, args.nr, args.nz)
    V = fem.functionspace(msh, ("Lagrange", 1))
    D0 = fem.functionspace(msh, ("DG", 0))
    x = ufl.SpatialCoordinate(msh)
    r = x[0]
    T0 = args.T0
    T_n = fem.Function(V, name="T"); T_n.x.array[:] = T0          # the previous step
    T_k = fem.Function(V); T_k.x.array[:] = T0                    # the Picard iterate
    alive = fem.Function(D0, name="alive"); alive.x.array[:] = 1.0
    removed_energy = fem.Function(D0, name="removal_enthalpy"); removed_energy.x.array[:] = 0.0   # J/m³ taken past T_fail, per cell
    surface_z = fem.Function(D0, name="surface_depth")             # the live surface's depth per column, for the source
    # the beam as a volumetric source in the first live layer of each column: q_in(r) / dz over the layer
    dz = args.thickness / args.nz
    q_in = s.absorptivity * (args.power / (math.pi * args.radius ** 2)) if args.shape == "tophat" else None
    cell_mid = fem.Function(D0)                                    # the cells' mid-depths, to pick the first live layer
    cells_x = msh.geometry.x
    conn = msh.topology.connectivity(msh.topology.dim, 0)
    mids = np.array([cells_x[conn.links(c)].mean(axis=0) for c in range(msh.topology.index_map(msh.topology.dim).size_local)])
    cell_mid.x.array[:] = mids[:, 1]
    cell_r = mids[:, 0]
    col_of = np.round((cell_r - 0.5 * r_max / args.nr) / (r_max / args.nr)).astype(int)      # the column index of each cell
    src = fem.Function(D0, name="source")
    k_eff = fem.Function(D0); c_scale = fem.Function(D0)
    h_conv = fem.Constant(msh, PETSc.ScalarType(args.h))
    dt = fem.Constant(msh, PETSc.ScalarType(args.dt))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    n = ufl.FacetNormal(msh)
    # face markers: the exposed face z = 0 (1), the back face z = −d (2), the far edge r = r_max (3)
    fdim = msh.topology.dim - 1
    facets = {1: mesh.locate_entities_boundary(msh, fdim, lambda p: np.isclose(p[1], 0.0)),
              2: mesh.locate_entities_boundary(msh, fdim, lambda p: np.isclose(p[1], -args.thickness)),
              3: mesh.locate_entities_boundary(msh, fdim, lambda p: np.isclose(p[0], r_max))}
    tags = mesh.meshtags(msh, fdim, np.concatenate(list(facets.values())), np.concatenate([np.full(len(f), m, dtype=np.int32) for m, f in facets.items()]))
    ds = ufl.Measure("ds", domain=msh, subdomain_data=tags)
    dx = ufl.dx(domain=msh)
    edge_dofs = fem.locate_dofs_topological(V, fdim, facets[3])
    bc = fem.dirichletbc(PETSc.ScalarType(T0), edge_dofs, V)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    xdmf = io.XDMFFile(comm, str(out / "ablation.xdmf"), "w"); xdmf.write_mesh(msh)
    log = {"t": [], "T_top_centre": [], "ablated_depth_centre": [], "live_fraction": []}
    t, step, t_damage, t_fail = 0.0, 0, math.nan, math.nan
    T_live_fail = s.T_fail
    while t < args.t_max:
        # the source layer: the first live cell of each column
        alive_arr = alive.x.array
        src_arr = np.zeros_like(alive_arr)
        for col in range(args.nr):
            cells = np.where((col_of == col) & (alive_arr > 0.5))[0]
            if len(cells) == 0:
                continue
            top = cells[np.argmax(cell_mid.x.array[cells])]        # the highest live cell (z nearest 0)
            rr = cell_r[top]
            qi = q_in if (args.shape == "tophat" and rr <= args.radius) else (s.absorptivity * 2 * args.power / (math.pi * args.radius ** 2) * math.exp(-2 * (rr / args.radius) ** 2) if args.shape == "gaussian" else 0.0)
            src_arr[top] = (qi or 0.0) / dz
        src.x.array[:] = src_arr
        k_eff.x.array[:] = np.where(alive_arr > 0.5, s.k, 1e-6 * s.k)
        c_scale.x.array[:] = np.where(alive_arr > 0.5, 1.0, 1e-6)
        # implicit Euler with Picard on the apparent capacity and the radiation (linearised about T_k); weighted by r for axisymmetry
        for _ in range(args.picard):
            cap = c_scale * apparent_capacity(s, T_k)
            h_rad = s.emissivity * SIGMA * (T_k ** 2 + T0 ** 2) * (T_k + T0)
            a = (cap * u * v / dt + k_eff * ufl.dot(ufl.grad(u), ufl.grad(v))) * r * dx + (h_conv + h_rad) * u * v * r * (ds(1) + ds(2))
            L = (cap * T_n * v / dt + src * v) * r * dx + (h_conv + h_rad) * T0 * v * r * (ds(1) + ds(2))
            problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "cg", "pc_type": "hypre"})
            T_new = problem.solve()
            change = float(np.abs(T_new.x.array - T_k.x.array).max())
            T_k.x.array[:] = T_new.x.array
            if change < 0.5:
                break
        # removal: cells held at T_fail accumulate the energy that would have raised them; past L_fail they leave
        T_cell = fem.Function(D0); T_cell.interpolate(fem.Expression(T_k, D0.element.interpolation_points))
        hot = (T_cell.x.array >= T_live_fail - 20.0) & (alive_arr > 0.5)
        removed_energy.x.array[hot] += (src_arr[hot] * args.dt)     # the beam's energy into a cell that can no longer heat up
        gone = hot & (removed_energy.x.array >= s.rho * s.L_fail)
        alive.x.array[gone] = 0.0
        T_n.x.array[:] = T_k.x.array
        t += args.dt; step += 1
        # the centre column's state
        centre = np.where(col_of == 0)[0]
        abl = float(np.sum(alive.x.array[centre] < 0.5) * dz)
        top_live = centre[alive.x.array[centre] > 0.5]
        T_top = float(T_cell.x.array[top_live].max()) if len(top_live) else math.nan
        damage_T = s.T_melt if s.failure == "melt" else s.T_fail
        if math.isnan(t_damage) and T_top >= damage_T:
            t_damage = t
        if math.isnan(t_fail) and (abl >= args.thickness - 1e-9 or len(top_live) == 0):
            t_fail = t
        log["t"].append(t); log["T_top_centre"].append(T_top); log["ablated_depth_centre"].append(abl); log["live_fraction"].append(float(alive.x.array.mean()))
        if step % args.write_every == 0 or not math.isnan(t_fail):
            T_n.name = "T"; xdmf.write_function(T_n, t); xdmf.write_function(alive, t); xdmf.write_function(removed_energy, t)
        if not math.isnan(t_fail):
            break
    xdmf.close()
    result = {"material": s.name, "thickness": args.thickness, "power": args.power, "radius": args.radius, "shape": args.shape, "dt": args.dt, "nr": args.nr, "nz": args.nz,
              "surface_damage_time": t_damage, "failure_time": t_fail, "t_end": t, "log": log}
    (out / "result.json").write_text(json.dumps(result, indent=1))
    return result


def hyperion_reference(args) -> dict:
    """The same case in Hyperion's cell model, for the comparison (runs anywhere)."""
    s = material(args.material)
    res = th.heat_plate(s, args.thickness, args.power, args.radius, args.t_max, shape=args.shape, h=args.h, nr=args.nr, nz=args.nz,
                        r_max=args.r_max or max(6 * args.radius, 12 * args.thickness, 0.05), stop_at_failure=True)
    return res.to_dict()


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="the guard's ablation in FEniCSx, against hyperion.thermal")
    ap.add_argument("--material", default="black_pc", help="a name from hyperion.thermal.SOLIDS, spaces as underscores")
    ap.add_argument("--thickness", type=float, default=10e-3)
    ap.add_argument("--power", type=float, default=933.0)
    ap.add_argument("--radius", type=float, default=17e-3)
    ap.add_argument("--shape", default="tophat", choices=("tophat", "gaussian"))
    ap.add_argument("--t-max", type=float, default=40.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--nr", type=int, default=60)
    ap.add_argument("--nz", type=int, default=20)
    ap.add_argument("--r-max", type=float, default=None)
    ap.add_argument("--h", type=float, default=10.0)
    ap.add_argument("--T0", type=float, default=th.T_ROOM)
    ap.add_argument("--picard", type=int, default=6)
    ap.add_argument("--write-every", type=int, default=10)
    ap.add_argument("--out", default="advanced/runs/default")
    ap.add_argument("--reference-only", action="store_true", help="only Hyperion's cell model (no dolfinx needed)")
    return ap


if __name__ == "__main__":
    a = parser().parse_args()
    t0 = time.time()
    if a.reference_only:
        print(json.dumps(hyperion_reference(a), indent=1))
    else:
        res = run(a)
        print(json.dumps({k: v for k, v in res.items() if k != "log"}, indent=1), f"\n{time.time() - t0:.0f} s")
