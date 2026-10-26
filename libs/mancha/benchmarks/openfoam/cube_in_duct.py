"""Mancha against OpenFOAM: steady laminar flow past a wall-mounted cube in a square duct.

    python benchmarks/openfoam/cube_in_duct.py [--n 8 12] [--reynolds 40] [--backend numpy|triton]
                                               [--openfoam-bashrc /usr/share/openfoam/etc/bashrc]
                                               [--procs 4] [--out results/openfoam]

The case (lengths in cube heights h): a duct 12 h long, 3 h high (y) and 3 h wide (z), walls on all four sides,
uniform inflow at x = 0, fixed pressure at x = 12 h. A cube of side h sits on the floor, its upstream face at
x = 3 h, centred in z. Re = U_b h / ν with U_b the bulk (mean) velocity.

Every surface is grid-aligned, so both codes represent the geometry exactly: OpenFOAM meshes it with uniform
hexahedra (blockMesh, ``n`` cells per h), and Mancha uses ``n`` lattice cells per h with halfway bounce-back
walls. OpenFOAM's cell centres then coincide with Mancha's nodes, and the fields compare point by point with no
interpolation.

Mancha runs first. Its delivered bulk velocity (the uniform bounce-back inlet loses a little flux on the edges
where it meets the walls) and its viscosity define the Reynolds number; OpenFOAM (simpleFoam, laminar,
second-order schemes) then runs at that Reynolds number with the bulk velocity as its uniform inlet value, in
units h = 1, U_b = 1. Compared: drag and lift coefficients of the cube (OpenFOAM: pressure + viscous stress
integrated over the cube faces from the converged fields — function objects fail in some OpenFOAM builds; Mancha:
momentum exchange), the recirculation length behind the cube, the pressure drop along the duct, centreplane
velocity profiles, and the whole-field velocity difference.

Neither code is the truth: OpenFOAM's result carries its own discretisation error, which its finer
``--reference-n`` meshes show. The JSON written records both, and ``docs/openfoam.md`` discusses them.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mancha import Simulation, __version__  # noqa: E402
from mancha.boundaries import box  # noqa: E402

LENGTH, HEIGHT, WIDTH = 12, 3, 3          # duct, in cube heights
CUBE_X0 = 3                               # upstream face of the cube
U_LATTICE = 0.05                          # Mancha's inlet speed (Ma ≈ 0.09)
STATIONS = (2.5, 3.5, 4.5, 6.0, 8.0)      # x / h of the centreplane profiles (upstream, on top, wake)


# --------------------------------------------------------------------------------------------- Mancha
def run_mancha(n: int, reynolds: float, backend: str, tol: float, max_steps: int) -> dict[str, Any]:
    nx, ny, nz = LENGTH * n, HEIGHT * n, WIDTH * n
    viscosity = U_LATTICE * n / reynolds              # nominal; the bulk velocity is measured below
    sim = Simulation(shape=(nx, ny, nz), viscosity=viscosity, velocity=(U_LATTICE, 0, 0), backend=backend)
    sim.set_boundary("x-", "velocity", velocity=(U_LATTICE, 0, 0)).set_boundary("x+", "pressure", density=1.0)
    for face in ("y-", "y+", "z-", "z+"):
        sim.set_boundary(face, "wall")
    sim.set_solid(box(sim.shape, (CUBE_X0 * n, 0, n), ((CUBE_X0 + 1) * n, n, 2 * n)))
    t0 = time.perf_counter()
    u_old = sim.velocity()
    change = math.inf
    while sim.steps_done < max_steps:
        sim.run(500)
        u = sim.velocity()
        change = float(np.linalg.norm(u - u_old) / np.linalg.norm(u))
        u_old = u
        if change < tol:
            break
    rho, u = sim.density(), sim.velocity()
    flux = (rho * u[0]).sum(axis=(1, 2))                        # mass flux through each x plane
    bulk = float(flux[-1] / (ny * nz))                          # ρ ≈ 1: the bulk velocity at the outlet
    force = sim.solid_force()      # gauge pressure (reference density 1 = the outlet): comparable with OpenFOAM
    q = 0.5 * bulk ** 2 * n ** 2                                # dynamic pressure × frontal area h²
    return {"n": n, "steps": sim.steps_done, "residual": change, "wall_time_s": time.perf_counter() - t0,
            "bulk_velocity": bulk, "reynolds": bulk * n / viscosity, "viscosity": viscosity,
            "cd": force[0] / q, "cl": force[1] / q, "cz": force[2] / q,
            "flux_variation_rel": float((flux.max() - flux.min()) / flux.mean()),
            "u": u / bulk, "p": (rho - 1.0) / 3.0 / bulk ** 2, "solid": sim.solid}


# ------------------------------------------------------------------------------------------- OpenFOAM
HEADER = """FoamFile {{ version 2.0; format ascii; class {cls}; object {obj}; }}
"""


def _vertex(i: int, j: int, k: int, xs, ys, zs) -> int:
    return (k * len(ys) + j) * len(xs) + i


def block_mesh_dict(n: int) -> str:
    """Multi-block hex mesh: the duct split by the cube's planes, the cube's block left out."""
    xs, ys, zs = [0, CUBE_X0, CUBE_X0 + 1, LENGTH], [0, 1, HEIGHT], [0, 1, 2, WIDTH]
    verts = [f"({x} {y} {z})" for z in zs for y in ys for x in xs]
    blocks, faces = [], {"inlet": [], "outlet": [], "cube": []}
    for k in range(3):
        for j in range(2):
            for i in range(3):
                if (i, j, k) == (1, 0, 1):
                    continue
                v = [_vertex(i + a, j + b, k + c, xs, ys, zs) for a, b, c in
                     ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))]
                cells = [(xs[i + 1] - xs[i]) * n, (ys[j + 1] - ys[j]) * n, (zs[k + 1] - zs[k]) * n]
                blocks.append(f"hex ({' '.join(map(str, v))}) ({cells[0]} {cells[1]} {cells[2]}) simpleGrading (1 1 1)")
                face = {"-x": (v[0], v[4], v[7], v[3]), "+x": (v[1], v[2], v[6], v[5]),
                        "-y": (v[0], v[1], v[5], v[4]), "+y": (v[3], v[7], v[6], v[2]),
                        "-z": (v[0], v[3], v[2], v[1]), "+z": (v[4], v[5], v[6], v[7])}
                if i == 0:
                    faces["inlet"].append(face["-x"])
                if i == 2:
                    faces["outlet"].append(face["+x"])
                if (i, j, k) == (0, 0, 1):
                    faces["cube"].append(face["+x"])
                if (i, j, k) == (2, 0, 1):
                    faces["cube"].append(face["-x"])
                if (i, j, k) == (1, 1, 1):
                    faces["cube"].append(face["-y"])
                if (i, j, k) == (1, 0, 0):
                    faces["cube"].append(face["+z"])
                if (i, j, k) == (1, 0, 2):
                    faces["cube"].append(face["-z"])

    def patch(name: str, kind: str) -> str:
        fs = " ".join(f"({' '.join(map(str, f))})" for f in faces[name])
        return f"{name} {{ type {kind}; faces ({fs}); }}"

    return (HEADER.format(cls="dictionary", obj="blockMeshDict") + "convertToMeters 1;\n"
            f"vertices ({' '.join(verts)});\n"
            f"blocks ({' '.join(blocks)});\n"
            "edges ();\n"
            f"boundary ({patch('inlet', 'patch')} {patch('outlet', 'patch')} {patch('cube', 'wall')});\n"
            "defaultPatch { name walls; type wall; }\n"
            "mergePatchPairs ();\n")


def write_case(case: Path, n: int, reynolds: float, iterations: int, procs: int) -> None:
    if case.exists():
        shutil.rmtree(case)
    for d in ("0", "constant", "system"):
        (case / d).mkdir(parents=True)
    files = {
        "system/blockMeshDict": block_mesh_dict(n),
        "system/controlDict": HEADER.format(cls="dictionary", obj="controlDict") + f"""
application simpleFoam; startFrom startTime; startTime 0; stopAt endTime; endTime {iterations}; deltaT 1;
writeControl runTime; writeInterval {iterations}; purgeWrite 1; writeFormat ascii; writePrecision 10;
writeCompression off; timeFormat general; timePrecision 8; runTimeModifiable false;
""",
        "system/fvSchemes": HEADER.format(cls="dictionary", obj="fvSchemes") + """
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; div(phi,U) bounded Gauss linear; div((nuEff*dev2(T(grad(U))))) Gauss linear; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
""",
        "system/fvSolution": HEADER.format(cls="dictionary", obj="fvSolution") + """
solvers
{
    p { solver GAMG; smoother GaussSeidel; tolerance 1e-9; relTol 0.01; }
    U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-10; relTol 0.05; }
}
SIMPLE { consistent yes; nNonOrthogonalCorrectors 0; residualControl { p 1e-7; U 1e-8; } }
relaxationFactors { equations { U 0.9; } fields { p 1; } }
""",
        "system/decomposeParDict": HEADER.format(cls="dictionary", obj="decomposeParDict")
        + f"numberOfSubdomains {procs}; method simple; simpleCoeffs {{ n ({procs} 1 1); delta 0.001; }}\n",
        "constant/transportProperties": HEADER.format(cls="dictionary", obj="transportProperties")
        + f"transportModel Newtonian; nu {1.0 / reynolds!r};\n",
        "constant/turbulenceProperties": HEADER.format(cls="dictionary", obj="turbulenceProperties")
        + "simulationType laminar;\n",
        "0/U": HEADER.format(cls="volVectorField", obj="U") + """
dimensions [0 1 -1 0 0 0 0]; internalField uniform (1 0 0);
boundaryField
{
    inlet { type fixedValue; value uniform (1 0 0); }
    outlet { type inletOutlet; inletValue uniform (0 0 0); value uniform (1 0 0); }
    "(walls|cube)" { type noSlip; }
}
""",
        "0/p": HEADER.format(cls="volScalarField", obj="p") + """
dimensions [0 2 -2 0 0 0 0]; internalField uniform 0;
boundaryField
{
    inlet { type zeroGradient; }
    outlet { type fixedValue; value uniform 0; }
    "(walls|cube)" { type zeroGradient; }
}
""",
    }
    for name, text in files.items():
        (case / name).write_text(text)


def foam(case: Path, bashrc: str, command: str, log: str) -> None:
    script = f"source {bashrc} >/dev/null 2>&1; cd {case} && {command} > {log} 2>&1"
    r = subprocess.run(["bash", "-c", script])
    if r.returncode != 0:
        tail = (case / log).read_text()[-2000:] if (case / log).exists() else ""
        raise RuntimeError(f"OpenFOAM command failed: {command}\n{tail}")


def read_field(path: Path) -> np.ndarray:
    """The internalField of an ASCII OpenFOAM field file (scalar or vector)."""
    text = path.read_text()
    m = re.search(r"internalField\s+nonuniform\s+List<(\w+)>\s*(\d+)\s*\(", text)
    if not m:
        raise ValueError(f"no nonuniform internalField in {path}")
    count = int(m.group(2))
    body = text[m.end():]
    if m.group(1) == "vector":
        vals = np.array(re.findall(r"\(([^()]+)\)", body[: body.index("\n)\n")]), dtype=object)
        arr = np.array([list(map(float, v.split())) for v in vals[:count]])
    else:
        arr = np.array(body[: body.index("\n)\n")].split(), dtype=float)
    assert len(arr) == count, (path, len(arr), count)
    return arr


def run_openfoam(n: int, reynolds: float, bashrc: str, work: Path, iterations: int, procs: int) -> dict[str, Any]:
    case = work / f"cube_n{n}"
    write_case(case, n, reynolds, iterations, procs)
    t0 = time.perf_counter()
    foam(case, bashrc, "blockMesh", "log.blockMesh")
    foam(case, bashrc, "checkMesh", "log.checkMesh")
    if procs > 1:
        foam(case, bashrc, "decomposePar -force", "log.decomposePar")
        foam(case, bashrc, f"mpirun --allow-run-as-root --oversubscribe -np {procs} simpleFoam -parallel",
             "log.simpleFoam")
        foam(case, bashrc, "reconstructPar -latestTime", "log.reconstructPar")
    else:
        foam(case, bashrc, "simpleFoam", "log.simpleFoam")
    wall = time.perf_counter() - t0
    times = sorted((d for d in case.iterdir() if re.fullmatch(r"\d+(\.\d+)?", d.name) and d.name != "0"),
                   key=lambda d: float(d.name))
    last = times[-1]
    centres, u, p = cell_centres(case / "constant" / "polyMesh"), read_field(last / "U"), read_field(last / "p")
    log = (case / "log.simpleFoam").read_text()
    converged = "SIMPLE solution converged" in log
    iters = re.findall(r"^Time = (\d+)", log, re.M)
    residuals = {f: float(v[-1]) for f in ("Ux", "p")
                 if (v := re.findall(rf"Solving for {f}, Initial residual = ([-+.\deE]+)", log))}
    # place the cell values on Mancha's node grid: centres are at (i + 1/2)/n
    idx = np.floor(centres * n).astype(int)
    shape = (LENGTH * n, HEIGHT * n, WIDTH * n)
    U = np.full((3,) + shape, np.nan)
    P = np.full(shape, np.nan)
    U[:, idx[:, 0], idx[:, 1], idx[:, 2]] = u.T
    P[idx[:, 0], idx[:, 1], idx[:, 2]] = p
    total = cube_force(U, P, n, 1.0 / reynolds)
    q = 0.5                                            # ½ ρ U_b² h² with every factor 1
    return {"n": n, "cells": len(p), "iterations": int(iters[-1]) if iters else None, "converged": converged,
            "final_initial_residuals": residuals, "wall_time_s": wall, "cd": total[0] / q, "cl": total[1] / q,
            "cz": total[2] / q, "u": U, "p": P, "case": str(case)}


def _read_list(path: Path, item: str) -> list[str]:
    text = path.read_text()
    m = re.search(r"^(\d+)\s*\n\(", text, re.M)
    count = int(m.group(1))
    body = text[m.end():]
    items = re.findall(item, body[: body.index("\n)")])
    assert len(items) == count, (path, len(items), count)
    return items


def cell_centres(mesh: Path) -> np.ndarray:
    """Cell centres of a hexahedral mesh from ``polyMesh`` (mean of the face centres of each cell; exact for the
    uniform boxes used here). Computed directly because some OpenFOAM builds cannot run function objects."""
    points = np.array([v.split() for v in _read_list(mesh / "points", r"\(([^()]+)\)")], dtype=float)
    faces = np.array([v.split() for v in _read_list(mesh / "faces", r"4\(([^()]+)\)")], dtype=int)
    owner = np.array(_read_list(mesh / "owner", r"-?\d+"), dtype=int)
    neighbour = np.array(_read_list(mesh / "neighbour", r"-?\d+"), dtype=int)
    fc = points[faces].mean(axis=1)
    ncells = owner.max() + 1
    acc = np.zeros((ncells, 3))
    cnt = np.zeros(ncells)
    np.add.at(acc, owner, fc)
    np.add.at(cnt, owner, 1)
    np.add.at(acc, neighbour, fc[: len(neighbour)])
    np.add.at(cnt, neighbour, 1)
    return acc / cnt[:, None]


def cube_force(U: np.ndarray, P: np.ndarray, n: int, nu: float) -> np.ndarray:
    """Force on the cube from the cell values next to its five faces, as OpenFOAM's ``forces`` evaluates it on
    this mesh: pressure (zero-gradient, so the face value is the cell value) plus the wall shear
    ``ν u_t / (Δ/2)`` from the tangential cell velocity (no slip, orthogonal faces). Unit density; face area
    ``1/n²``."""
    x0, x1, k0, k1 = CUBE_X0 * n, (CUBE_X0 + 1) * n, n, 2 * n
    area, half = 1.0 / n ** 2, 0.5 / n
    force = np.zeros(3)
    for cells, normal in (((x0 - 1, slice(0, n), slice(k0, k1)), (1, 0, 0)),     # upstream face, n into the cube
                          ((x1, slice(0, n), slice(k0, k1)), (-1, 0, 0)),        # downstream face
                          ((slice(x0, x1), n, slice(k0, k1)), (0, -1, 0)),        # top
                          ((slice(x0, x1), slice(0, n), k0 - 1), (0, 0, 1)),      # side at z = 1
                          ((slice(x0, x1), slice(0, n), k1), (0, 0, -1))):        # side at z = 2
        nrm = np.asarray(normal, dtype=float)
        p = P[cells].sum()
        u = U[(slice(None),) + cells].reshape(3, -1).sum(axis=1)
        u_t = u - (u @ nrm) * nrm                 # the normal gradient of the normal velocity vanishes at a wall
        force += area * (p * nrm + nu * u_t / half)
    return force


# ------------------------------------------------------------------------------------------- comparison
def recirculation_length(u: np.ndarray, n: int) -> float:
    """Reattachment distance (in h) behind the cube on the floor line through its centre (first node above the
    floor): where the last stretch of reversed flow ends, measured from the cube's downstream face. Taking the last
    reversal rather than the first forward node skips the small counter-rotating vortex in the corner at the
    cube's base, which fine grids resolve."""
    line = u[0, (CUBE_X0 + 1) * n:, 0, int(1.5 * n)]
    rev = np.flatnonzero(line < 0)
    if rev.size == 0:
        return 0.0
    k = int(rev[-1])                                   # last reversed node
    if k + 1 >= line.size:
        return float(line.size / n)                    # reversed all the way to the outlet
    frac = line[k] / (line[k] - line[k + 1])           # linear zero crossing between nodes k and k+1
    return float((k + 0.5 + frac) / n)                 # node j sits (j + ½)/n behind the face


def compare(m: dict[str, Any], f: dict[str, Any]) -> dict[str, Any]:
    n = m["n"]
    fluid = ~m["solid"]
    um, uf = m["u"], f["u"]
    ok = fluid & np.isfinite(uf[0])
    full = np.sqrt(((um - uf) ** 2).sum(axis=0))
    diff = full[ok]
    beyond = ok.copy()
    beyond[: n] = False            # the inlet corners differ by construction (wall precedence at Mancha's inlet edges)
    where = np.unravel_index(np.argmax(np.where(ok, full, -1.0)), full.shape)
    # pressure drop between the planes 1 h after the inlet and 1 h before the outlet (plane averages)
    i0, i1 = n, (LENGTH - 1) * n
    dp_m = float(m["p"][i0].mean() - m["p"][i1].mean())
    dp_f = float(np.nanmean(f["p"][i0]) - np.nanmean(f["p"][i1]))
    profiles = {}
    for xs in STATIONS:
        i = int(xs * n)
        profiles[str(xs)] = {"y": ((np.arange(HEIGHT * n) + 0.5) / n).tolist(),
                             "mancha": um[0, i, :, int(1.5 * n)].tolist(),
                             "openfoam": uf[0, i, :, int(1.5 * n)].tolist()}
    return {"cd": {"mancha": m["cd"], "openfoam": f["cd"], "rel_diff": m["cd"] / f["cd"] - 1},
            "cl": {"mancha": m["cl"], "openfoam": f["cl"]},
            "recirculation_length": {"mancha": recirculation_length(um, n), "openfoam": recirculation_length(uf, n)},
            "pressure_drop": {"mancha": dp_m, "openfoam": dp_f, "rel_diff": dp_m / dp_f - 1},
            "velocity_diff_rms": float(np.sqrt((diff ** 2).mean())), "velocity_diff_max": float(diff.max()),
            "velocity_diff_max_at_h": [float((i + 0.5) / n) for i in where],
            "velocity_diff_rms_beyond_1h": float(np.sqrt((full[beyond] ** 2).mean())),
            "velocity_diff_max_beyond_1h": float(full[beyond].max()),
            "profiles": profiles}


def plot(results: list[dict[str, Any]], out: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    best = results[-1]["comparison"]["profiles"]
    fig, axes = plt.subplots(1, len(STATIONS), figsize=(3 * len(STATIONS), 3.6), sharey=True)
    for ax, xs in zip(axes, STATIONS):
        prof = best[str(xs)]
        ax.plot(prof["openfoam"], prof["y"], "-", label="OpenFOAM")
        ax.plot(prof["mancha"], prof["y"], "o", ms=3, label="Mancha")
        ax.axhline(1.0, color="0.7", lw=0.6) if CUBE_X0 <= xs <= CUBE_X0 + 1 else None
        ax.set(title=f"x = {xs} h", xlabel="u / U_b")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("y / h")
    axes[0].legend()
    fig.suptitle(f"Cube in a duct, centreplane, n = {results[-1]['n']} cells per h")
    fig.tight_layout()
    fig.savefig(out / "cube_profiles.png", dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, nargs="+", default=[8, 12], help="cells per cube height")
    ap.add_argument("--reynolds", type=float, default=40.0)
    ap.add_argument("--backend", default="numpy")
    ap.add_argument("--openfoam-bashrc", default=os.environ.get("MANCHA_OPENFOAM_BASHRC",
                                                                "/usr/share/openfoam/etc/bashrc"))
    ap.add_argument("--procs", type=int, default=1, help="MPI ranks for simpleFoam")
    ap.add_argument("--iterations", type=int, default=4000, help="simpleFoam iteration limit")
    ap.add_argument("--tol", type=float, default=1e-6, help="Mancha steady-state tolerance")
    ap.add_argument("--reference-n", type=int, nargs="*", default=[16, 24],
                    help="extra OpenFOAM-only resolutions: its own grid convergence")
    ap.add_argument("--out", default="results/openfoam")
    args = ap.parse_args(argv)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for n in args.n:
        cache = out / f"mancha_n{n}_re{args.reynolds:g}_{args.backend}.npz"
        if cache.exists():
            with np.load(cache, allow_pickle=False) as d:
                m = json.loads(str(d["meta"])) | {"u": d["u"], "p": d["p"], "solid": d["solid"]}
        else:
            m = run_mancha(n, args.reynolds, args.backend, args.tol, 400000)
            meta = {k: v for k, v in m.items() if k not in ("u", "p", "solid")}
            np.savez_compressed(cache, u=m["u"], p=m["p"], solid=m["solid"], meta=np.array(json.dumps(meta)))
        print(f"n={n}: Mancha {m['steps']} steps, {m['wall_time_s']:.0f} s, Re = {m['reynolds']:.2f}, "
              f"Cd = {m['cd']:.4f}", flush=True)
        f = run_openfoam(n, m["reynolds"], args.openfoam_bashrc, out / "cases", args.iterations, args.procs)
        print(f"n={n}: OpenFOAM {f['iterations']} iterations (converged: {f['converged']}), {f['wall_time_s']:.0f} s, "
              f"Cd = {f['cd']:.4f}", flush=True)
        c = compare(m, f)
        row = {"n": n, "reynolds": m["reynolds"],
               "mancha": {k: v for k, v in m.items() if k not in ("u", "p", "solid")},
               "openfoam": {k: v for k, v in f.items() if k not in ("u", "p")}, "comparison": c}
        results.append(row)
        print(f"   Cd diff {c['cd']['rel_diff']:+.2%}, recirculation {c['recirculation_length']}, "
              f"Δp diff {c['pressure_drop']['rel_diff']:+.2%}, |Δu| rms {c['velocity_diff_rms']:.4f} "
              f"max {c['velocity_diff_max']:.4f}; beyond x = h: rms {c['velocity_diff_rms_beyond_1h']:.4f} "
              f"max {c['velocity_diff_max_beyond_1h']:.4f}", flush=True)
        previous = json.loads((out / "cube_in_duct.json").read_text()) if (out / "cube_in_duct.json").exists() else {}
        (out / "cube_in_duct.json").write_text(json.dumps(
            {"openfoam_reference": previous.get("openfoam_reference", []), "mancha_version": __version__,
             "backend": args.backend, "case": "wall-mounted cube in a duct",
             "geometry_h": {"length": LENGTH, "height": HEIGHT, "width": WIDTH, "cube_x0": CUBE_X0},
             "openfoam": "simpleFoam (laminar, Gauss linear, SIMPLEC), " + args.openfoam_bashrc,
             "results": results}, indent=1))
    reference = []
    re_ref = results[-1]["reynolds"] if results else args.reynolds
    for n in args.reference_n:
        f = run_openfoam(n, re_ref, args.openfoam_bashrc, out / "cases", args.iterations, args.procs)
        reference.append({"n": n, "reynolds": re_ref, "cells": f["cells"], "iterations": f["iterations"],
                          "converged": f["converged"], "wall_time_s": f["wall_time_s"], "cd": f["cd"], "cl": f["cl"],
                          "recirculation_length": recirculation_length(f["u"], n),
                          "pressure_drop": float(np.nanmean(f["p"][n]) - np.nanmean(f["p"][(LENGTH - 1) * n]))})
        print(f"OpenFOAM reference n={n}: {reference[-1]}", flush=True)
        record = json.loads((out / "cube_in_duct.json").read_text()) if (out / "cube_in_duct.json").exists() else {}
        record["openfoam_reference"] = [r for r in record.get("openfoam_reference", []) if r["n"] not in
                                        {x["n"] for x in reference}] + reference
        (out / "cube_in_duct.json").write_text(json.dumps(record, indent=1))
    plot(results, out)


if __name__ == "__main__":
    main()
