"""The jet: what the corona's force does to the air (``docs/plasma.md``, P3).

Two levels. The **integral model** here: the thrust is a momentum flux through the exit, so the mean exit
velocity is ``sqrt(T / (rho A_open))`` with ``A_open`` the exit area less the rings' blockage; the convergence
angle of the jet is the force field's own (the inward radial force on the ions inside each ground ring over the
axial force), and the focal distance follows from it; the velocity at the focal point is the mean one times the
area ratio, capped (a free jet entrains and spreads: the cap is the estimate the CFD replaces).

The **CFD level**: ``force_zones`` bins the solved force density ``rho_q E`` into annular cell zones and
``write_openfoam_sources`` writes them as OpenFOAM ``fvOptions`` momentum sources (``vectorSemiImplicitSource``,
``volumeMode absolute``: the zone's axial force in newtons) with the ``topoSetDict`` that makes the zones and a
``run.sh`` with the step order — into a case Aeromant prepared (``labs/plasma/designs/halka_cfd.py``). The zones
carry the axial force only: the radial force cancels between the inner and outer grounds and a uniform vector per
annulus could not represent it anyway; what it does to the jet's convergence is in the integral model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .gas import Air, DRY


@dataclass(frozen=True)
class JetEstimate:
    thrust: float               # N
    exit_area: float            # m^2, open
    v_mean: float               # m/s at the exit
    half_angle: float           # rad, the jet's convergence
    focal_distance: float       # m downstream of the grounds
    v_focal: float              # m/s, the estimate at the focal point
    mass_flow: float            # kg/s


def jet_estimate(thrust: float, exit_area: float, blockage: float = 0.15, half_angle: float = 0.25,
                 r_mean: float = 0.05, air: Air = DRY, gain_cap: float = 1.6) -> JetEstimate:
    """The integral jet: ``blockage`` the rings' and spokes' share of the exit area, ``half_angle`` the jet's
    convergence (from ``convergence_angle`` when a solve is at hand), ``r_mean`` the rings' mean radius."""
    a_open = exit_area * (1.0 - blockage)
    v = math.sqrt(max(thrust, 0.0) / (air.density * a_open))
    L = r_mean / math.tan(half_angle) if half_angle > 1e-3 else float("inf")
    gain = min(gain_cap, 1.0 + 0.6 * math.tan(half_angle) * 4.0)        # ~1.4 at 0.25 rad (the P1 estimate)
    return JetEstimate(thrust, a_open, v, half_angle, L, gain * v, air.density * a_open * v)


def convergence_angle(problem, r_split: float) -> float:
    """The jet's half-angle from the force field: the radial force (either sign: each emitter pushes air both
    inward and outward, and the shear between the opposed streams is what folds the jet) over the axial force,
    as a magnitude — ``r_split`` (the emitters' mean radius) weights the outer half, which is what converges."""
    E = np.asarray(problem._E_cell())
    rho = np.asarray(problem.rho.value)
    vol = np.asarray(problem._volumes)
    r = np.asarray(problem.mesh.cellCenters[0])
    f_z = float(np.sum(rho * vol * E[1]))
    f_r = float(np.sum(np.abs(rho * vol * E[0])[r > r_split])) + 0.5 * float(np.sum(np.abs(rho * vol * E[0])[r <= r_split]))
    return math.atan2(f_r, max(f_z, 1e-12))


def force_zones(problem, r_edges, z_edges) -> list[dict]:
    """Bin the solved force density into annular zones [r1, r2] x [z1, z2]: each zone's axial force [N] and
    volume [m^3]; zones with no force are left out."""
    E = np.asarray(problem._E_cell())
    rho = np.asarray(problem.rho.value)
    vol = np.asarray(problem._volumes)
    c = np.asarray(problem.mesh.cellCenters)
    fz = rho * vol * E[1]
    out = []
    for r1, r2 in zip(r_edges[:-1], r_edges[1:]):
        for z1, z2 in zip(z_edges[:-1], z_edges[1:]):
            m = (c[0] >= r1) & (c[0] < r2) & (c[1] >= z1) & (c[1] < z2)
            F = float(fz[m].sum())
            if abs(F) > 1e-9:
                out.append({"r1": r1, "r2": r2, "z1": z1, "z2": z2, "force_z": F,
                            "volume": math.pi * (r2 * r2 - r1 * r1) * (z2 - z1)})
    return out


def write_openfoam_sources(case_dir, zones: list[dict], *, axis=(1.0, 0.0, 0.0), origin=(0.0, 0.0, 0.0),
                           solver: str = "simpleFoam", pipeline: tuple = ("blockMesh", "surfaceFeatureExtract", "snappyHexMesh -overwrite",
                                                                            "checkMesh", "restore0Dir", "topoSet")) -> dict:
    """Write ``system/topoSetDict`` (one annular cell set per zone, about ``axis`` through ``origin``: the thrust
    axis in the case's frame, +x in Aeromant's external templates), ``system/fvOptions`` (the momentum sources)
    and ``run.sh`` (the pipeline with ``topoSet`` before the solver) into ``case_dir``. Returns the paths."""
    case_dir = Path(case_dir)
    (case_dir / "system").mkdir(parents=True, exist_ok=True)
    ax = np.asarray(axis, float) / np.linalg.norm(axis)
    o = np.asarray(origin, float)
    head = lambda cls, obj, loc: ("FoamFile\n{\n    version     2.0;\n    format      ascii;\n"
                                  f"    class       {cls};\n    location    \"{loc}\";\n    object      {obj};\n}}\n\n")
    acts = []
    for k, z in enumerate(zones):
        p1, p2 = o + ax * z["z1"], o + ax * z["z2"]
        acts.append(f"""    {{
        name    ehd{k};
        type    cellSet;
        action  new;
        source  cylinderAnnulusToCell;
        p1      ({p1[0]:.6g} {p1[1]:.6g} {p1[2]:.6g});
        p2      ({p2[0]:.6g} {p2[1]:.6g} {p2[2]:.6g});
        outerRadius {z['r2']:.6g};
        innerRadius {z['r1']:.6g};
    }}
    {{
        name    ehd{k};
        type    cellZoneSet;
        action  new;
        source  setToCellZone;
        set     ehd{k};
    }}""")
    topo = ("// Fulgur: the corona's force on the air as annular cell zones (vegeta.fulgur.jet.write_openfoam_sources)\n"
            + head("dictionary", "topoSetDict", "system") + "actions\n(\n" + "\n".join(acts) + "\n);\n")
    srcs = []
    for k, z in enumerate(zones):
        F = ax * z["force_z"]
        srcs.append(f"""ehd{k}
{{
    type            vectorSemiImplicitSource;
    selectionMode   cellZone;
    cellZone        ehd{k};
    volumeMode      absolute;           // the whole zone's force [N]
    sources
    {{
        U           (({F[0]:.6g} {F[1]:.6g} {F[2]:.6g}) 0);
    }}
}}
""")
    fvo = ("// Fulgur: the EHD body force rho_q E of the solved corona, binned into annular zones (axial part)\n"
           + head("dictionary", "fvOptions", "system") + "\n".join(srcs))
    total = sum(z["force_z"] for z in zones)
    sh = "#!/usr/bin/env bash\n# Halka's jet: the Aeromant case with Fulgur's EHD sources; topoSet makes the zones before the solver\nset -e\n" \
         + "\n".join(pipeline) + f"\n{solver}\n# total EHD force in the zones: {total:.4g} N along ({ax[0]:g} {ax[1]:g} {ax[2]:g})\n"
    paths = {"topoSetDict": case_dir / "system" / "topoSetDict", "fvOptions": case_dir / "system" / "fvOptions", "run": case_dir / "run.sh"}
    paths["topoSetDict"].write_text(topo)
    paths["fvOptions"].write_text(fvo)
    paths["run"].write_text(sh)
    paths["run"].chmod(0o755)
    return {**paths, "zones": len(zones), "force_z": total}
