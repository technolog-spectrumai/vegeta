"""The rotor of rotor_fwh.py in OpenFOAM (simpleFoam with MRF over the whole domain — a single rotating frame
in absolute-velocity form), on the identical voxel geometry, written in the same cache format so
farfield_100m.py can propagate both. (SRFSimpleFoam in the packaged v1912 solves only the pressure equation —
no momentum predictor output, converged after 50 iterations — so it is not used.)

    python benchmarks/acoustics/openfoam_rotor.py [--diameter 16] [--tip-speed 0.04] [--advance 0.5]
                                                  [--viscosity 0.005] [--box 3.0] [--procs 4]
                                                  [--openfoam-bashrc /usr/share/openfoam/etc/bashrc]
                                                  [--out results/acoustics/openfoam]

The mesh is a uniform hexahedral box whose cell centres are Mancha's nodes (lattice units: cell size 1), with the
blade and hub voxels removed (``topoSet`` + ``subsetMesh``): the same staircase body. OpenFOAM solves the
same steady problem in the same rotating frame (laminar, same ν, rotation rate and inflow): the lab-frame
velocity U along x on the inlet and the lateral far field (non-rotating patches), fixed pressure 0 at the outlet
(Mancha: ρ = 1, the same gauge reference), no slip on the body, which MRF turns with the frame.
Incompressible, so ρ = 1; Mancha is weakly compressible (tip
Mach ≈ 0.07). Mancha's lateral sponge has no counterpart here.

Blade loads are integrated from the converged fields face by face (pressure + wall shear ν u_t/(Δ/2), as the
cube comparison does — function objects fail in the packaged OpenFOAM v1912): per-face forces at the face
centres feed FW-H's loading term, exactly as Mancha's per-link forces do.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "openfoam"))

from cube_in_duct import HEADER, cell_centres, foam, read_field  # noqa: E402
from rotor_fwh import rotor_mask  # noqa: E402

from mancha.diagnostics import solid_faces  # noqa: E402


def setup(args) -> dict:
    """The same rotor and box as rotor_fwh.run_mancha."""
    D = args.diameter
    R = D / 2
    nx, ny = int(4.5 * D), int(args.box * D) // 2 * 2
    centre = ((ny - 1) / 2, (ny - 1) / 2)
    x0 = 1.5 * D
    omega = args.tip_speed / R
    U = args.advance * omega * D / (2 * np.pi)
    mask, blades = rotor_mask((nx, ny, ny), centre, R, x0, 0.2 * D, 2.0, 0.8 * D)
    return {"D": D, "shape": (nx, ny, ny), "centre": centre, "x0": x0, "omega": omega, "U": U, "mask": mask,
            "blades": blades}


def write_case(case: Path, s: dict, nu: float, iterations: int, procs: int) -> None:
    if case.exists():
        shutil.rmtree(case)
    for d in ("0", "constant", "system"):
        (case / d).mkdir(parents=True)
    nx, ny, nz = s["shape"]
    lo, hi = (-0.5, -0.5, -0.5), (nx - 0.5, ny - 0.5, nz - 0.5)
    v = [(lo[0], lo[1], lo[2]), (hi[0], lo[1], lo[2]), (hi[0], hi[1], lo[2]), (lo[0], hi[1], lo[2]),
         (lo[0], lo[1], hi[2]), (hi[0], lo[1], hi[2]), (hi[0], hi[1], hi[2]), (lo[0], hi[1], hi[2])]
    verts = " ".join(f"({a} {b} {c})" for a, b, c in v)
    files = {
        "system/blockMeshDict": HEADER.format(cls="dictionary", obj="blockMeshDict") + f"""
convertToMeters 1; vertices ({verts});
blocks (hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1));
edges ();
boundary
(
    inlet {{ type patch; faces ((0 4 7 3)); }}
    outlet {{ type patch; faces ((1 2 6 5)); }}
    farfield {{ type patch; faces ((0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7)); }}
    walls {{ type wall; faces (); }}
);
mergePatchPairs ();
""",
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
SIMPLE { momentumPredictor yes; consistent yes; nNonOrthogonalCorrectors 0; residualControl { p 1e-6; U 1e-7; } }
relaxationFactors { equations { U 0.8; } fields { p 1; } }
""",
        "system/decomposeParDict": HEADER.format(cls="dictionary", obj="decomposeParDict")
        + f"numberOfSubdomains {procs}; method simple; simpleCoeffs {{ n ({procs} 1 1); delta 0.001; }}\n",
        "system/topoSetDict": HEADER.format(cls="dictionary", obj="topoSetDict") + "actions\n(\n    { name fluid; "
        "type cellSet; action new; source labelToCell; sourceInfo { value ("
        + " ".join(map(str, np.flatnonzero(~s["mask"].transpose(2, 1, 0).reshape(-1))))
        + "); } }\n);\n",
        "system/topoSetDict.zone": HEADER.format(cls="dictionary", obj="topoSetDict") + """actions
(
    { name all; type cellSet; action new; source boxToCell; sourceInfo { box (-1e9 -1e9 -1e9) (1e9 1e9 1e9); } }
    { name rotor; type cellZoneSet; action new; source setToCellZone; sourceInfo { set all; } }
);
""",
        "constant/transportProperties": HEADER.format(cls="dictionary", obj="transportProperties")
        + f"transportModel Newtonian; nu {nu!r};\n",
        "constant/turbulenceProperties": HEADER.format(cls="dictionary", obj="turbulenceProperties")
        + "simulationType laminar;\n",
        "constant/MRFProperties": HEADER.format(cls="dictionary", obj="MRFProperties") + f"""
rotor {{ cellZone rotor; active yes; nonRotatingPatches (inlet outlet farfield); origin (0 {s['centre'][0]!r}
  {s['centre'][1]!r}); axis (1 0 0); omega {s['omega']!r}; }}
""",
        "0/U": HEADER.format(cls="volVectorField", obj="U") + f"""
dimensions [0 1 -1 0 0 0 0]; internalField uniform ({s['U']!r} 0 0);
boundaryField
{{
    "(inlet|farfield)" {{ type fixedValue; value uniform ({s['U']!r} 0 0); }}
    outlet {{ type inletOutlet; inletValue uniform (0 0 0); value uniform ({s['U']!r} 0 0); }}
    walls {{ type fixedValue; value uniform (0 0 0); }}
}}
""",
        "0/p": HEADER.format(cls="volScalarField", obj="p") + """
dimensions [0 2 -2 0 0 0 0]; internalField uniform 0;
boundaryField
{
    "(inlet|farfield|walls)" { type zeroGradient; }
    outlet { type fixedValue; value uniform 0; }
}
""",
    }
    for name, text in files.items():
        (case / name).write_text(text)


def run(args) -> Path:
    s = setup(args)
    out = Path(args.out)
    case = out / f"case_D{args.diameter}"
    write_case(case, s, args.viscosity, args.iterations, args.procs)
    t0 = time.perf_counter()
    rc = args.openfoam_bashrc
    foam(case, rc, "blockMesh", "log.blockMesh")
    foam(case, rc, "topoSet", "log.topoSet")
    foam(case, rc, "subsetMesh fluid -patch walls -overwrite", "log.subsetMesh")
    foam(case, rc, "topoSet -dict system/topoSetDict.zone", "log.topoSetZone")
    foam(case, rc, "checkMesh", "log.checkMesh")
    if args.procs > 1:
        foam(case, rc, "decomposePar -force", "log.decomposePar")
        foam(case, rc, f"mpirun --allow-run-as-root --oversubscribe -np {args.procs} simpleFoam -parallel",
             "log.simpleFoam")
        foam(case, rc, "reconstructPar -latestTime", "log.reconstructPar")
    else:
        foam(case, rc, "simpleFoam", "log.simpleFoam")
    wall = time.perf_counter() - t0
    log = (case / "log.simpleFoam").read_text()
    converged = "SIMPLE solution converged" in log
    times = sorted((d for d in case.iterdir() if d.name.replace(".", "").isdigit() and d.name != "0"),
                   key=lambda d: float(d.name))
    last = times[-1]
    cc = cell_centres(case / "constant" / "polyMesh")
    uabs, p = read_field(last / "U"), read_field(last / "p")
    cy0, cz0 = s["centre"]
    # MRF solves for the absolute velocity; the relative one is u − Ω×r (Ω×r = (0, −ω z', ω y'))
    urel = uabs - np.stack([np.zeros(len(cc)), -s["omega"] * (cc[:, 2] - cz0), s["omega"] * (cc[:, 1] - cy0)], 1)
    idx = np.rint(cc).astype(int)
    shape = s["shape"]
    U = np.zeros((3,) + shape)
    P = np.zeros(shape)
    U[:, idx[:, 0], idx[:, 1], idx[:, 2]] = urel.T
    P[idx[:, 0], idx[:, 1], idx[:, 2]] = p
    nfluid = int((~s["mask"]).sum())
    if len(p) != nfluid:
        raise RuntimeError(f"{len(p)} OpenFOAM cells but {nfluid} fluid voxels")

    # per-face loads on the blades (pressure + wall shear), the OpenFOAM counterpart of Mancha's link forces
    blades, solid = s["blades"], s["mask"]
    pos, force = [], []
    for axis in range(3):
        for sign in (1, -1):
            e = np.zeros(3)
            e[axis] = sign                                   # from the fluid cell into the blade voxel
            nb_blade = np.roll(blades, -sign, axis=axis)
            sl = [slice(None)] * 3
            sl[axis] = -1 if sign > 0 else 0
            nb_blade[tuple(sl)] = False
            cells = np.argwhere(nb_blade & ~solid)
            if not len(cells):
                continue
            pc = P[cells[:, 0], cells[:, 1], cells[:, 2]]
            uc = U[:, cells[:, 0], cells[:, 1], cells[:, 2]].T
            ut = uc - np.outer(uc @ e, e)
            pos.append(cells + 0.5 * e)
            force.append(pc[:, None] * e[None, :] + args.viscosity * ut / 0.5)
    pos, force = np.concatenate(pos), np.concatenate(force)
    cy, cz = s["centre"]
    r = pos - np.array([0.0, cy, cz])
    moment = np.cross(r, force).sum(0)
    fc, fn = solid_faces(solid)
    meta = {"solver": "OpenFOAM simpleFoam + MRF over the whole domain (laminar)", "D": s["D"],
            "omega": s["omega"], "U": s["U"],
            "centre": list(s["centre"]), "x0": s["x0"], "shape": list(shape), "thrust": float(-force[:, 0].sum()),
            "torque": float(-moment[0]), "viscosity": args.viscosity, "tip_speed": args.tip_speed,
            "advance": args.advance, "converged": converged, "iterations": int(float(last.name)),
            "wall_time_s": wall, "cells": nfluid}
    cache = out / f"openfoam_D{args.diameter}.npz"
    np.savez_compressed(cache, meta=np.array(json.dumps(meta)), pressure=P, density=np.ones(shape), velocity=U,
                        link_pos=pos, link_force=force, face_pos=fc, face_normal=fn, mask=solid, blades=blades)
    print(f"OpenFOAM D={s['D']}: {meta['iterations']} iterations (converged: {converged}), {wall:.0f} s, "
          f"T = {meta['thrust']:.5g}, Q = {meta['torque']:.5g} -> {cache}")
    return cache


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--diameter", type=int, default=16)
    ap.add_argument("--tip-speed", type=float, default=0.04)
    ap.add_argument("--advance", type=float, default=0.5)
    ap.add_argument("--viscosity", type=float, default=0.005)
    ap.add_argument("--box", type=float, default=3.0)
    ap.add_argument("--iterations", type=int, default=3000)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--openfoam-bashrc", default="/usr/share/openfoam/etc/bashrc")
    ap.add_argument("--out", default="results/acoustics/openfoam")
    args = ap.parse_args(argv)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
