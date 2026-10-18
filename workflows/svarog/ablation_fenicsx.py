#!/usr/bin/env python
"""svarog/ablation (FEniCSx) — the target and the shields under the beams of notebook 46, as transient field problems

    .venv/bin/python workflows/svarog/ablation_fenicsx.py --variant 20m --chapter target
    .venv/bin/python workflows/svarog/ablation_fenicsx.py --variant 50m --chapter shield --cases aluminium
    .venv/bin/python workflows/svarog/ablation_fenicsx.py --resume workflows/runs/svarog/<chapter>/<run id>

A thin workflow script (workflows/README.md): explicit inputs, the cases under a budget, one run directory. The cases
are ``svarog_remote.ablation_cases(variant, chapter)`` — the standoff is the variant (20m / 50m / 100m); the beam each
case sees (power, 1/e² radius) comes from the head of notebook 46 and the Gaussian budget at that standoff. Each case
is an axisymmetric r–z slab under a Gaussian beam: transient conduction with the apparent-heat-capacity enthalpy
method (melting and the removal plateau — boiling for the metals, decomposition for the plastic), convection and
radiation at the faces, implicit Euler with Picard iterations, and ablation by cell deactivation once a cell held at
the removal temperature has taken its latent heat. It writes XDMF (T, the live indicator) and ``result.json`` per case
(melt-onset, removal-onset and burn-through times, the ablated depth log); ``summary.json`` collects them.

dolfinx is a conda-forge / Docker install (docs/fenics.md on branch dev_svarog_advanced): the chapter's need says so
and the notebook shows NOT RUN until the runs exist. The notebook launches the missing chapters detached; the runs
are read when it is next executed.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "workflows", ROOT / "notebooks" / "designs", ROOT / "vegeta-yanus" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import runkit  # noqa: E402
import library  # noqa: E402
from vegeta import yanus  # noqa: E402
import svarog_remote as sr  # noqa: E402

try:
    import ufl
    from dolfinx import fem, io, mesh
    from dolfinx.fem.petsc import LinearProblem
    from mpi4py import MPI
    from petsc4py import PETSc
    HAVE_DOLFINX = True
except ImportError:                                            # the script imports without dolfinx: --list and the case table work anywhere
    HAVE_DOLFINX = False

BOOK, KIND, WORKFLOW = "svarog", "ablation", "svarog_ablation"
SIGMA_SB = 5.670374419e-8
T0 = 293.0
H_CONV = 10.0


def cases(variant: str, campaign: str) -> dict:
    """``{case name: case spec}`` — the explicit list of cases of the campaign (never widened by the script)."""
    return sr.ablation_cases(variant, campaign)


def apparent_capacity(s: sr.AblationSolid, T, band: float = 15.0):
    """ρ c_app(T): the heat capacity with the latent heats spread over ±band K around T_melt and T_fail (UFL)."""
    rho_c = s.rho * s.c
    melt = s.rho * s.L_melt / (2 * band) * ufl.conditional(ufl.lt(abs(T - s.T_melt), band), 1.0, 0.0) if s.L_melt > 0 else 0.0
    fail = s.rho * s.L_fail / (2 * band) * ufl.conditional(ufl.lt(abs(T - s.T_fail), band), 1.0, 0.0)
    return rho_c + melt + fail


def solve(name: str, spec: dict, case_dir: Path, cancel) -> dict:
    """One case: the r–z slab under the beam until burn-through or ``t_max``; ``{"status": "solved"|"FAILED", …}``."""
    if not HAVE_DOLFINX:
        return {"status": "FAILED", "reason": "dolfinx is not installed (conda-forge fenics-dolfinx, or the dolfinx/dolfinx Docker image)"}
    s = sr.ABLATION_SOLIDS[spec["solid"]]
    d, P, w = spec["thickness"], spec["power"], spec["radius"]
    nr, nz, dt_s, t_max = spec["nr"], spec["nz"], spec["dt"], spec["t_max"]
    r_max = max(6 * w, 12 * d, 0.02)
    comm = MPI.COMM_WORLD
    msh = mesh.create_rectangle(comm, [np.array([0.0, -d]), np.array([r_max, 0.0])], [nr, nz], cell_type=mesh.CellType.quadrilateral)
    V = fem.functionspace(msh, ("Lagrange", 1))
    D0 = fem.functionspace(msh, ("DG", 0))
    x = ufl.SpatialCoordinate(msh)
    r = x[0]
    T_n = fem.Function(V, name="T"); T_n.x.array[:] = T0
    T_k = fem.Function(V); T_k.x.array[:] = T0
    alive = fem.Function(D0, name="alive"); alive.x.array[:] = 1.0
    removed = np.zeros(alive.x.array.shape)
    dz = d / nz
    conn = msh.topology.connectivity(msh.topology.dim, 0)
    n_cells = msh.topology.index_map(msh.topology.dim).size_local
    mids = np.array([msh.geometry.x[conn.links(c)].mean(axis=0) for c in range(n_cells)])
    cell_r, cell_z = mids[:, 0], mids[:, 1]
    col_of = np.round((cell_r - 0.5 * r_max / nr) / (r_max / nr)).astype(int)
    src = fem.Function(D0, name="source")
    k_eff = fem.Function(D0); c_scale = fem.Function(D0)
    h_conv = fem.Constant(msh, PETSc.ScalarType(H_CONV))
    dt = fem.Constant(msh, PETSc.ScalarType(dt_s))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    fdim = msh.topology.dim - 1
    facets = {1: mesh.locate_entities_boundary(msh, fdim, lambda p: np.isclose(p[1], 0.0)),
              2: mesh.locate_entities_boundary(msh, fdim, lambda p: np.isclose(p[1], -d)),
              3: mesh.locate_entities_boundary(msh, fdim, lambda p: np.isclose(p[0], r_max))}
    tags = mesh.meshtags(msh, fdim, np.concatenate(list(facets.values())), np.concatenate([np.full(len(f), m, dtype=np.int32) for m, f in facets.items()]))
    ds = ufl.Measure("ds", domain=msh, subdomain_data=tags)
    dx = ufl.dx(domain=msh)
    bc = fem.dirichletbc(PETSc.ScalarType(T0), fem.locate_dofs_topological(V, fdim, facets[3]), V)
    xdmf = io.XDMFFile(comm, str(case_dir / "ablation.xdmf"), "w"); xdmf.write_mesh(msh)
    log = {"t": [], "T_top_centre": [], "ablated_depth_centre": [], "live_fraction": []}
    t, step, t_melt, t_removal, t_through = 0.0, 0, math.nan, math.nan, math.nan
    I0 = s.absorptivity * 2 * P / (math.pi * w ** 2)
    while t < t_max and not cancel.is_set():
        alive_arr = alive.x.array
        src_arr = np.zeros_like(alive_arr)
        for col in range(nr):
            cells = np.where((col_of == col) & (alive_arr > 0.5))[0]
            if len(cells) == 0:
                continue
            top = cells[np.argmax(cell_z[cells])]
            src_arr[top] = I0 * math.exp(-2 * (cell_r[top] / w) ** 2) / dz
        src.x.array[:] = src_arr
        k_eff.x.array[:] = np.where(alive_arr > 0.5, s.k, 1e-6 * s.k)
        c_scale.x.array[:] = np.where(alive_arr > 0.5, 1.0, 1e-6)
        for _ in range(6):
            cap = c_scale * apparent_capacity(s, T_k)
            h_rad = s.emissivity * SIGMA_SB * (T_k ** 2 + T0 ** 2) * (T_k + T0)
            a = (cap * u * v / dt + k_eff * ufl.dot(ufl.grad(u), ufl.grad(v))) * r * dx + (h_conv + h_rad) * u * v * r * (ds(1) + ds(2))
            Lf = (cap * T_n * v / dt + src * v) * r * dx + (h_conv + h_rad) * T0 * v * r * (ds(1) + ds(2))
            T_new = LinearProblem(a, Lf, bcs=[bc], petsc_options={"ksp_type": "cg", "pc_type": "hypre"}).solve()
            change = float(np.abs(T_new.x.array - T_k.x.array).max())
            T_k.x.array[:] = T_new.x.array
            if change < 0.5:
                break
        T_cell = fem.Function(D0); T_cell.interpolate(fem.Expression(T_k, D0.element.interpolation_points))
        hot = (T_cell.x.array >= s.T_fail - 20.0) & (alive_arr > 0.5)
        removed[hot] += src_arr[hot] * dt_s
        alive.x.array[hot & (removed >= s.rho * s.L_fail)] = 0.0
        T_n.x.array[:] = T_k.x.array
        t += dt_s; step += 1
        centre = np.where(col_of == 0)[0]
        abl = float(np.sum(alive.x.array[centre] < 0.5) * dz)
        top_live = centre[alive.x.array[centre] > 0.5]
        T_top = float(T_cell.x.array[top_live].max()) if len(top_live) else math.nan
        if math.isnan(t_melt) and T_top >= s.T_melt:
            t_melt = t
        if math.isnan(t_removal) and abl > 0:
            t_removal = t
        if math.isnan(t_through) and (abl >= d - 1e-9 or len(top_live) == 0):
            t_through = t
        log["t"].append(t); log["T_top_centre"].append(T_top); log["ablated_depth_centre"].append(abl); log["live_fraction"].append(float(alive.x.array.mean()))
        if step % 20 == 0 or not math.isnan(t_through):
            xdmf.write_function(T_n, t); xdmf.write_function(alive, t)
        if not math.isnan(t_through):
            break
    xdmf.close()
    result = {"status": "solved", "solid": s.name, "thickness": d, "power": P, "radius": w, "I0_abs_W_m2": I0, "dt": dt_s, "nr": nr, "nz": nz,
              "melt_onset_s": t_melt, "removal_onset_s": t_removal, "burn_through_s": t_through, "t_end": t, "ablated_depth_end_m": abl, "T_top_end_K": T_top,
              "cancelled": bool(cancel.is_set()), "log": log}
    (case_dir / "result.json").write_text(json.dumps(result, indent=1))
    metrics = {k: v for k, v in result.items() if k not in ("log", "status", "cancelled")}
    return {"status": "solved" if not cancel.is_set() else "cancelled", "process": "ok", "converged": True, "metrics": metrics,
            "artifacts": {"result": case_dir / "result.json", "xdmf": case_dir / "ablation.xdmf"}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", required=True, choices=sr.ABLATION_VARIANTS)
    camps = [c["campaign"] for c in library.BOOKS[BOOK]["chapters"].values() if c["kind"] == KIND]
    ap.add_argument("--chapter", "--campaign", dest="chapter", default=camps[0], choices=tuple(camps))
    runkit.add_common_arguments(ap)
    args = ap.parse_args(argv)
    chapter = library.chapter_of(BOOK, KIND, args.chapter)
    specs = cases(args.variant, args.chapter)
    if getattr(args, "cases", None):
        specs = {k: v for k, v in specs.items() if k in runkit.select_cases(list(specs), args.cases)}
    if getattr(args, "list", False):
        for n, s in specs.items():
            print(f"{n:28s} {s['note']}: P = {s['power']:.0f} W, w = {s['radius']*1e3:.2f} mm, {s['thickness']*1e3:g} mm, {s['t_max']:g} s")
        return 0
    gate = yanus.check_chapter(f"{BOOK}/{chapter}", variant=args.variant, ignore=args.ignore_prerequisites)
    config = {"variant": args.variant, "campaign": args.chapter}
    if args.resume:
        run = runkit.Run.open(args.resume, config=config, heartbeat_s=args.heartbeat, echo=not args.quiet)
    else:
        out = Path(args.out) if args.out else library.run_dir(BOOK, chapter, args.variant)
        run = runkit.Run.create(out, workflow=WORKFLOW, campaign=args.chapter, config=config, design=sr.ablation_fingerprint(args.variant),
                                planned_cases=list(specs), heartbeat_s=args.heartbeat, echo=not args.quiet)
    run.manifest.update(book=BOOK, chapter=chapter, kind=KIND, prerequisites_ignored=gate["missing"] if args.ignore_prerequisites else [],
                        yanus={"needs": gate["needs"]})
    runkit.write_json_atomic(run.root / "manifest.json", run.manifest)
    if args.prepare_only:
        (run.root / "cases.json").write_text(json.dumps(specs, indent=1))
        code = run.finish({"cases": {}, "prepared": list(specs)}, state="prepared")
        run.close()
        return code
    records = {}
    for name, spec in specs.items():
        if run.cancel.is_set():
            break
        records[name] = run.run_case(name, runkit.sha256_json({"case": name, **spec}),
                                     lambda d, cancel, n=name, s=spec: solve(n, s, d, cancel), rerun_failed=args.rerun_failed)
    state = "cancelled" if run.cancel.is_set() else "completed"
    code = run.finish({"variant": args.variant, "campaign": args.chapter, "specs": specs, "cases": {n: r.to_dict() for n, r in records.items()}}, state=state)
    print(f"run directory: {run.root}")
    run.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
