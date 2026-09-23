"""Structural model: explicit geometry, units, material, regions, supports, loads and mesh settings."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from ._process import describe_failure, utc_now
from ._progress import resolve_progress
from .ccx import ccx_version, nset, run_ccx, write_inp
from .frd import read_dat_eigen, read_dat_reactions, read_frd
from .loads import Acceleration, Displacement, FixedSupport, Force, PointMass, Pressure
from .materials import Material
from .mesh import MeshSettings, generate_mesh, read_mesh
from .regions import Surfaces, SurfacesInBox, SurfacesOnPlane
from .result import Result
from .units import get_units

MESH_FILE = "mesh.msh"
MESH_SUMMARY = "mesh_summary.json"
JOB = "model"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class StructuralModel:
    """A linear static analysis definition. Nothing is inferred: every input is explicit.

    ``mesh()`` and ``solve()`` are separate calls; ``solve()`` requires an up-to-date mesh in the
    same working directory and never meshes on its own.
    """

    geometry: str | Path
    units: str
    material: Material
    regions: Sequence[Surfaces | SurfacesInBox | SurfacesOnPlane]
    supports: Sequence[FixedSupport | Displacement]
    loads: Sequence[Force | Pressure | Acceleration]
    mesh_settings: MeshSettings
    name: str = "talos_model"
    notes: str = ""
    masses: Sequence[PointMass] = ()   # lumped masses (motors, batteries): inertia in modal analysis
    _unit_system: object = field(init=False, repr=False)

    def __post_init__(self):
        self.geometry = Path(self.geometry)
        self._unit_system = get_units(self.units)
        if not isinstance(self.material, Material):
            raise ValueError("material must be a talos.Material")
        if self.material.units is not None and self.material.units != self.units:
            raise ValueError(f"material {self.material.name!r} is tagged {self.material.units!r} "
                             f"but the model uses {self.units!r}")
        names = [r.name for r in self.regions]
        upper = [n.upper() for n in names]
        if len(set(upper)) != len(upper):
            raise ValueError(f"region names must be unique (case-insensitive): {names}")
        if not self.supports:
            raise ValueError("no supports defined; a static analysis needs explicit supports")
        if not self.loads and not self.masses:
            raise ValueError("no loads defined (a modal-only model needs at least its masses)")
        for item in list(self.supports) + [l for l in self.loads if hasattr(l, "region")] + list(self.masses):
            if item.region not in names:
                raise ValueError(f"{type(item).__name__} refers to unknown region {item.region!r}; defined: {names}")
        if any(isinstance(l, Acceleration) for l in self.loads) and self.material.density is None:
            raise ValueError("an Acceleration load needs material density; none was given")
        if not isinstance(self.mesh_settings, MeshSettings):
            raise ValueError("mesh_settings must be a talos.MeshSettings")

    # -- description ------------------------------------------------------------------------
    def config(self) -> dict:
        def tagged(obj):
            return {"type": type(obj).__name__, **asdict(obj)}

        return {
            "name": self.name,
            "geometry": str(self.geometry),
            "units": self.units,
            "material": asdict(self.material),
            "regions": [tagged(r) for r in self.regions],
            "supports": [tagged(s) for s in self.supports],
            "loads": [tagged(l) for l in self.loads],
            "masses": [tagged(m) for m in self.masses],
            "mesh": asdict(self.mesh_settings),
            "notes": self.notes,
        }

    def _mesh_key(self) -> str:
        cfg = {
            "geometry_sha256": _sha256(self.geometry) if self.geometry.is_file() else None,
            "units": self.units,
            "regions": self.config()["regions"],
            "mesh": asdict(self.mesh_settings),
        }
        return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()

    # -- meshing ----------------------------------------------------------------------------
    def mesh(self, workdir: str | Path, progress=False) -> Result:
        """Mesh the geometry with Gmsh into ``workdir/mesh.msh``."""
        t0 = time.monotonic()
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        res = Result(kind="talos.mesh", metadata={"model": self.config(), "started_at": utc_now()})
        cb, close = resolve_progress(progress, "talos mesh")
        msh, log = workdir / MESH_FILE, workdir / "gmsh.log"
        try:
            stats = generate_mesh(self.geometry, self._unit_system, self.regions, self.mesh_settings, msh, log, cb)
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            res.fail(f"meshing failed: {exc}")
            stats = None
        except Exception as exc:  # gmsh raises plain Exception on meshing errors
            res.fail(f"Gmsh error: {exc}")
            stats = None
        finally:
            cb("done", 1.0)
            close()
        if log.is_file():
            res.artifacts["gmsh_log"] = log
        if stats is not None:
            res.artifacts["mesh"] = msh
            res.metrics = {k: v for k, v in stats.items() if k not in ("region_surfaces", "bbox", "gmsh_version")}
            res.metadata.update(region_surfaces=stats["region_surfaces"], bbox=stats["bbox"],
                                tool_versions={"gmsh": stats["gmsh_version"]}, mesh_key=self._mesh_key())
            if stats["min_quality_sicn"] is not None and stats["min_quality_sicn"] <= 0:
                res.fail("mesh contains inverted elements (minSICN <= 0)")
            elif stats["min_quality_sicn"] is not None and stats["min_quality_sicn"] < 0.1:
                res.messages.append(f"low element quality: minSICN = {stats['min_quality_sicn']:.3f}")
        res.duration_s = time.monotonic() - t0
        res.artifacts["mesh_summary"] = res.save_json(workdir / MESH_SUMMARY)
        return res

    # -- solving ----------------------------------------------------------------------------
    def solve(self, workdir: str | Path, *, executable: str = "ccx", threads: int = 1,
              timeout: float | None = None, progress=False, cancel: threading.Event | None = None) -> Result:
        """Write ``model.inp`` from the existing mesh, run CalculiX and summarise results."""
        t0 = time.monotonic()
        workdir = Path(workdir)
        res = Result(kind="talos.solve", metadata={"model": self.config(), "started_at": utc_now(),
                                                   "units": asdict(self._unit_system)})
        done = lambda: (setattr(res, "duration_s", time.monotonic() - t0), res.save_json(workdir / "summary.json"))

        summary = workdir / MESH_SUMMARY
        if not (workdir / MESH_FILE).is_file() or not summary.is_file():
            res.fail(f"no mesh in {workdir}; run model.mesh(workdir) first")
            workdir.mkdir(parents=True, exist_ok=True)
            done()
            return res
        mesh_info = json.loads(summary.read_text())
        if mesh_info.get("status") != "success":
            res.fail("the mesh in this directory failed; re-run model.mesh(workdir)")
            done()
            return res
        if mesh_info.get("metadata", {}).get("mesh_key") != self._mesh_key():
            res.fail("the mesh is out of date (geometry, regions or mesh settings changed); re-run model.mesh(workdir)")
            done()
            return res
        res.artifacts["mesh"] = workdir / MESH_FILE

        cb, close = resolve_progress(progress, "talos solve")
        try:
            cb("write deck", 0.05)
            mesh = read_mesh(workdir / MESH_FILE)
            inp = workdir / f"{JOB}.inp"
            if not self.loads:
                raise ValueError("no loads defined; use solve_modes() for a modal analysis")
            book = write_inp(inp, mesh, self.material, self.supports, self.loads, self.masses)
            res.artifacts["inp"] = inp
            res.metadata["tool_versions"] = {"ccx": ccx_version(executable),
                                             **mesh_info.get("metadata", {}).get("tool_versions", {})}

            def on_line(line: str):
                low = line.lower()
                if "factoring" in low:
                    cb("CalculiX: factoring", 0.3)
                elif "solving" in low:
                    cb("CalculiX: solving", 0.6)
                elif "job finished" in low:
                    cb("CalculiX: finished", 0.95)

            cb("CalculiX", 0.1)
            rec = run_ccx(workdir, JOB, executable, threads, timeout, on_line, cancel)
            res.execution.append(rec)
            if rec.log_file:
                res.artifacts["ccx_log"] = Path(rec.log_file)
            for ext in ("frd", "dat", "sta", "cvg"):
                if (workdir / f"{JOB}.{ext}").is_file():
                    res.artifacts[ext] = workdir / f"{JOB}.{ext}"
            if rec.error == "cancelled":
                res.fail("CalculiX run cancelled by user", status="cancelled")
            elif not rec.ok:
                res.fail(describe_failure(rec))
            elif "*ERROR" in rec.stdout or "*ERROR" in rec.stderr:
                res.fail(describe_failure(rec) + "\nCalculiX reported *ERROR")
            elif "frd" not in res.artifacts:
                res.fail("CalculiX finished but wrote no .frd results")
            else:
                self._summarise(res, workdir, book, mesh)
        except Exception as exc:  # parsing or deck-writing problems become a failed result
            res.fail(f"{type(exc).__name__}: {exc}")
        finally:
            cb("done", 1.0)
            close()
        done()
        res.artifacts["summary"] = workdir / "summary.json"
        res.save_json(res.artifacts["summary"])
        return res

    def solve_modes(self, workdir: str | Path, n_modes: int = 10, *, executable: str = "ccx", threads: int = 1,
                    timeout: float | None = None, progress=False, cancel: threading.Event | None = None) -> Result:
        """Natural frequencies and mode shapes (CalculiX ``*FREQUENCY``) of the supported structure with
        its point masses; loads are ignored. Needs the material density. Results: ``modes.frd``/``.dat``."""
        t0 = time.monotonic()
        workdir = Path(workdir)
        job = "modes"
        res = Result(kind="talos.modes", metadata={"model": self.config(), "started_at": utc_now(), "n_modes": n_modes,
                                                   "units": asdict(self._unit_system)})
        done = lambda: (setattr(res, "duration_s", time.monotonic() - t0), res.save_json(workdir / "modes_summary.json"))
        if self.material.density is None:
            res.fail("a modal analysis needs the material density; none was given")
            workdir.mkdir(parents=True, exist_ok=True)
            done()
            return res
        summary = workdir / MESH_SUMMARY
        if not (workdir / MESH_FILE).is_file() or not summary.is_file():
            res.fail(f"no mesh in {workdir}; run model.mesh(workdir) first")
            workdir.mkdir(parents=True, exist_ok=True)
            done()
            return res
        mesh_info = json.loads(summary.read_text())
        if mesh_info.get("status") != "success" or mesh_info.get("metadata", {}).get("mesh_key") != self._mesh_key():
            res.fail("the mesh is missing, failed or out of date; re-run model.mesh(workdir)")
            done()
            return res
        res.artifacts["mesh"] = workdir / MESH_FILE
        cb, close = resolve_progress(progress, "talos modes")
        try:
            cb("write deck", 0.05)
            mesh = read_mesh(workdir / MESH_FILE)
            inp = workdir / f"{job}.inp"
            write_inp(inp, mesh, self.material, self.supports, self.loads, self.masses, modes=n_modes)
            res.artifacts["inp"] = inp
            res.metadata["tool_versions"] = {"ccx": ccx_version(executable),
                                             **mesh_info.get("metadata", {}).get("tool_versions", {})}
            cb("CalculiX eigenvalues", 0.1)
            rec = run_ccx(workdir, job, executable, threads, timeout,
                          lambda line: cb("CalculiX: solving", 0.5) if "eigen" in line.lower() else None, cancel)
            res.execution.append(rec)
            if rec.log_file:
                res.artifacts["ccx_log"] = Path(rec.log_file)
            for ext in ("frd", "dat"):
                if (workdir / f"{job}.{ext}").is_file():
                    res.artifacts[ext] = workdir / f"{job}.{ext}"
            if rec.error == "cancelled":
                res.fail("CalculiX run cancelled by user", status="cancelled")
            elif not rec.ok or "*ERROR" in rec.stdout or "*ERROR" in rec.stderr:
                res.fail(describe_failure(rec))
            elif "dat" not in res.artifacts:
                res.fail("CalculiX finished but wrote no .dat file")
            else:
                eig = read_dat_eigen(res.artifacts["dat"])
                if not eig["frequencies_hz"]:
                    res.fail("no eigenvalues found in modes.dat")
                else:
                    res.metrics = {"n_nodes": int(len(mesh.node_ids)), "n_elements": int(len(mesh.element_ids)),
                                   "n_modes": len(eig["frequencies_hz"]), "frequencies_hz": eig["frequencies_hz"],
                                   "first_frequency_hz": eig["frequencies_hz"][0],
                                   "effective_modal_mass": eig["effective_modal_mass"] or None,
                                   "total_effective_mass": eig["total_effective_mass"],
                                   "point_mass_total": float(sum(m.mass for m in self.masses))}
                    res.messages.append("frequencies are undamped and linear (no stress stiffening, no pre-load); "
                                        "printed parts are anisotropic and often softer than the datasheet modulus")
        except Exception as exc:
            res.fail(f"{type(exc).__name__}: {exc}")
        finally:
            cb("done", 1.0)
            close()
        done()
        res.artifacts["summary"] = workdir / "modes_summary.json"
        res.save_json(res.artifacts["summary"])
        return res

    def _summarise(self, res: Result, workdir: Path, book: dict, mesh) -> None:
        fr = read_frd(workdir / f"{JOB}.frd")
        if "DISP" not in fr.fields:
            res.fail("no displacement field in .frd")
            return
        u = fr.displacement_magnitude
        i = int(np.nanargmax(u))
        m = res.metrics
        m["n_nodes"] = int(len(mesh.node_ids))
        m["n_elements"] = int(len(mesh.element_ids))
        m["element_type"] = mesh.element_type
        m["max_displacement"] = float(u[i])
        m["max_displacement_node"] = int(fr.node_ids[i])
        m["max_displacement_location"] = fr.coords[i].tolist()
        m["displacement_min"] = np.nanmin(fr.displacement, axis=0).tolist()
        m["displacement_max"] = np.nanmax(fr.displacement, axis=0).tolist()
        if "STRESS" in fr.fields:
            vm = fr.von_mises
            j = int(np.nanargmax(vm))
            m["max_von_mises"] = float(vm[j])
            m["max_von_mises_location"] = fr.coords[j].tolist()
        else:
            m["max_von_mises"] = None
            res.messages.append("no stress field in .frd")
        reactions = read_dat_reactions(workdir / f"{JOB}.dat") if "dat" in res.artifacts else {}
        by_support = {}
        for s in self.supports:
            by_support[s.region] = reactions.get(nset(s.region))
        m["reactions"] = by_support
        known = [np.array(v) for v in by_support.values() if v is not None]
        m["reaction_total"] = np.sum(known, axis=0).tolist() if known else None
        m["applied_force_total"] = book["applied_force_total"]
        supported = set()
        for sup in self.supports:
            supported.update(int(n) for n in mesh.region_nodes.get(sup.region, []))
        loaded_on_support = [
            l.region for l in self.loads
            if hasattr(l, "region") and supported.intersection(int(n) for n in mesh.region_nodes.get(l.region, []))
        ]
        if loaded_on_support or any(isinstance(l, Acceleration) for l in self.loads):
            res.messages.append(
                "reactions are CalculiX RF values (internal nodal forces); load applied directly on supported "
                "nodes (body-force share of supported nodes, or loads on regions "
                f"{sorted(set(loaded_on_support))}) is not included in them"
            )
        ys = self.material.yield_strength
        if ys is not None and m.get("max_von_mises"):
            m["safety_factor_yield"] = ys / m["max_von_mises"]
        else:
            m["safety_factor_yield"] = None
            if ys is None:
                res.messages.append("safety factor not computed: no yield_strength given for the material")
        res.messages.append(
            "peak von Mises is a nodal (extrapolated, averaged) value; it can be mesh-dependent at "
            "supports, point-like loads and sharp re-entrant corners"
        )
        res.metadata["units"]["note"] = (
            f"lengths {self._unit_system.length}, forces {self._unit_system.force}, "
            f"stresses {self._unit_system.stress}"
        )
