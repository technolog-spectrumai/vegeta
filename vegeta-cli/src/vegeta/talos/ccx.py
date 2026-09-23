"""CalculiX input deck writer and runner."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from ._process import run_command
from .loads import Acceleration, Displacement, FixedSupport, Force, PointMass, Pressure
from .mesh import MeshData, consistent_nodal_forces, element_faces


def nset(region: str) -> str:
    return f"N_{region.upper()}"


def surf(region: str) -> str:
    return f"S_{region.upper()}"


def _chunks(ids: Iterable[int], n: int = 16):
    ids = list(ids)
    for i in range(0, len(ids), n):
        yield ", ".join(str(int(x)) for x in ids[i:i + n])


def write_inp(path: Path, mesh: MeshData, material, supports, loads, masses=(), *, modes: int | None = None) -> dict:
    """Write a CalculiX deck: linear static (default) or, with ``modes``, a ``*FREQUENCY`` step for the
    first ``modes`` eigenmodes (loads are then ignored). Returns bookkeeping (applied loads, sets)."""
    cmap = mesh.coord_map()
    lines = ["*HEADING", "Talos linear static analysis" if modes is None else f"Talos modal analysis ({modes} modes)",
             "*NODE, NSET=NALL"]
    lines += [f"{int(n)}, {x:.12g}, {y:.12g}, {z:.12g}" for n, (x, y, z) in zip(mesh.node_ids, mesh.coords)]
    lines.append(f"*ELEMENT, TYPE={mesh.element_type}, ELSET=EALL")
    for eid, row in zip(mesh.element_ids, mesh.connectivity):
        ids = [str(int(eid))] + [str(int(n)) for n in row]
        # CalculiX allows at most 16 entries per line; continuation lines end with a comma
        if len(ids) > 16:
            lines.append(", ".join(ids[:16]) + ",")
            lines.append(", ".join(ids[16:]))
        else:
            lines.append(", ".join(ids))

    used_regions = {s.region for s in supports} | {l.region for l in loads if hasattr(l, "region")} | {m.region for m in masses}
    for name in sorted(used_regions):
        lines.append(f"*NSET, NSET={nset(name)}")
        lines += list(_chunks(np.unique(mesh.region_nodes[name])))
    for name in sorted({l.region for l in loads if isinstance(l, Pressure)}):
        lines.append(f"*SURFACE, NAME={surf(name)}, TYPE=ELEMENT")
        lines += [f"{e}, S{f}" for e, f in element_faces(mesh, mesh.region_triangles[name])]

    lines += ["*MATERIAL, NAME=MAT", "*ELASTIC", f"{material.youngs_modulus:.12g}, {material.poissons_ratio:.12g}"]
    if material.density is not None:
        lines += ["*DENSITY", f"{material.density:.12g}"]
    lines += ["*SOLID SECTION, ELSET=EALL, MATERIAL=MAT"]
    next_eid = int(mesh.element_ids.max()) + 1
    for i, m in enumerate(masses):
        nodes = np.unique(mesh.region_nodes[m.region])
        lines.append(f"*ELEMENT, TYPE=MASS, ELSET=MASS_{i}")
        lines += [f"{next_eid + k}, {int(n)}" for k, n in enumerate(nodes)]
        next_eid += len(nodes)
        lines += [f"*MASS, ELSET=MASS_{i}", f"{m.mass / len(nodes):.12g}"]
    if modes is not None:
        lines += ["*STEP", "*FREQUENCY", f"{int(modes)}"]
    else:
        lines += ["*STEP", "*STATIC"]

    lines.append("*BOUNDARY")
    for s in supports:
        if isinstance(s, FixedSupport):
            lines.append(f"{nset(s.region)}, 1, 3, 0.0")
        elif isinstance(s, Displacement):
            lines += [f"{nset(s.region)}, {dof}, {dof}, {val:.12g}" for dof, val in s.dofs()]

    applied = np.zeros(3)
    cloads, dloads = [], []
    for load in (loads if modes is None else []):
        if isinstance(load, Force):
            forces = consistent_nodal_forces(mesh.region_triangles[load.region], cmap, np.array(load.vector))
            for n, f in sorted(forces.items()):
                cloads += [f"{n}, {d + 1}, {f[d]:.12g}" for d in range(3) if f[d] != 0.0]
            applied += np.array(load.vector)
        elif isinstance(load, Pressure):
            dloads.append(f"{surf(load.region)}, P, {load.value:.12g}")
        elif isinstance(load, Acceleration):
            vec = np.array(load.vector, dtype=float)
            mag = float(np.linalg.norm(vec))
            d = vec / mag
            dloads.append(f"EALL, GRAV, {mag:.12g}, {d[0]:.12g}, {d[1]:.12g}, {d[2]:.12g}")
    if cloads:
        lines += ["*CLOAD"] + cloads
    if dloads:
        lines += ["*DLOAD"] + dloads

    if modes is not None:
        lines += ["*NODE FILE", "U", "*END STEP", ""]
    else:
        lines += ["*NODE FILE", "U", "*EL FILE", "S"]
        for s in supports:
            lines += [f"*NODE PRINT, NSET={nset(s.region)}, TOTALS=YES", "RF"]
        lines += ["*END STEP", ""]
    path.write_text("\n".join(lines))
    return {"applied_force_total": applied.tolist(), "n_cload_lines": len(cloads)}


def ccx_version(executable: str = "ccx") -> str | None:
    import subprocess

    try:
        out = subprocess.run([executable, "-v"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        if "Version" in line:
            return line.strip()
    return out.strip() or None


def run_ccx(workdir: Path, jobname: str, executable: str = "ccx", threads: int = 1, timeout=None,
            on_output=None, cancel=None):
    return run_command(
        [executable, "-i", jobname], workdir, env={"OMP_NUM_THREADS": str(threads), "CCX_NPROC_EQUATION_SOLVER": str(threads)},
        timeout=timeout, log_name="ccx", on_output=on_output, cancel=cancel,
    )
