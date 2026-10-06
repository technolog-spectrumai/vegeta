"""CFD cases built from templates: prepare, run explicitly chosen steps, extract results."""
from __future__ import annotations

import hashlib
import math
import json
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ._process import describe_failure, run_command, utc_now
from ._progress import resolve_progress
from .environment import OpenFOAMEnvironment
from .result import CommandRecord, Result
from . import cache as _cache
from .results import find_coefficient_files, read_checkmesh, read_coefficients, read_solver_log, read_surface_field_values
from .stl import LENGTH_TO_METRES, read_stl, write_stl_ascii
from .templates import GAS_CONSTANT, Step, TemplateSpec, get_template

CASE_INFO = "aeromant_case.json"
_PLACEHOLDER = re.compile(r"\{\{([A-Z0-9_]+)\}\}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_speed(p: dict, L: float) -> float:
    """The speed the Reynolds number uses: the free stream, a rotor's tip speed, or a suction duct's velocity."""
    if "velocity" in p:
        return p["velocity"]
    if "rpm" in p:
        return p["rpm"] * math.pi / 60 * L
    return p["flow_rate"] / p["duct_inner"] ** 2


class CFDCase:
    """One CFD case directory made from a template, an STL and explicit parameters.

    Nothing runs on construction. ``prepare()`` writes the case; ``run()`` executes the template
    pipeline (or chosen steps); ``results()`` reads what exists. All files stay in ``workdir``.
    """

    def __init__(self, template: str | TemplateSpec, geometry: str | Path, parameters: dict[str, Any],
                 workdir: str | Path, geometry_units: str, environment: OpenFOAMEnvironment | None = None,
                 static_geometry: str | Path | None = None, surfaces: dict[str, str | Path] | None = None):
        self.template = get_template(template)
        self.geometry = Path(geometry)
        if self.template.static_geometry and static_geometry is None:
            raise ValueError(f"template {self.template.name!r} needs static_geometry (the standing body's STL, same units)")
        if static_geometry is not None and not self.template.static_geometry:
            raise ValueError(f"template {self.template.name!r} has no standing body; static_geometry is for rotor_mrf_installed")
        self.static_geometry = Path(static_geometry) if static_geometry is not None else None
        surfaces = dict(surfaces or {})
        missing, unknown = set(self.template.surfaces) - set(surfaces), set(surfaces) - set(self.template.surfaces)
        if missing:
            raise ValueError(f"template {self.template.name!r} needs the surface STL(s) {sorted(missing)} "
                             f"(surfaces={{name: path}}, same units as the geometry)")
        if unknown:
            raise ValueError(f"template {self.template.name!r} has no surface(s) {sorted(unknown)}; "
                             f"it takes {list(self.template.surfaces) or 'none'}")
        self.surfaces = {k: Path(surfaces[k]) for k in self.template.surfaces}
        if geometry_units not in LENGTH_TO_METRES:
            raise ValueError(f"geometry_units must be one of {sorted(LENGTH_TO_METRES)} (STL has no units)")
        self.geometry_units = geometry_units
        self.parameters = self.template.resolve(dict(parameters))
        self.user_parameters = dict(parameters)
        self.workdir = Path(workdir)
        self.environment = environment or OpenFOAMEnvironment()
        detected = self.environment.flavor()
        self.flavor = detected or self.template.flavors[0].name
        self.files = self.template.flavor(self.flavor)  # case files + pipeline for this OpenFOAM fork

    # -- description ------------------------------------------------------------------------
    def config(self) -> dict:
        return {
            "template": self.template.name,
            "openfoam_flavor": self.flavor,
            "openfoam_version": self.environment.version(),
            "geometry": str(self.geometry),
            "geometry_units": self.geometry_units,
            **({"static_geometry": str(self.static_geometry)} if self.static_geometry else {}),
            **({"surfaces": {k: str(v) for k, v in self.surfaces.items()}} if self.surfaces else {}),
            "parameters": self.parameters,
            "user_parameters": self.user_parameters,
            "environment": self.environment.describe(),
        }

    def _key(self) -> str:
        cfg = dict(self.config(), geometry_sha256=_sha256(self.geometry) if self.geometry.is_file() else None)
        if self.static_geometry is not None:
            cfg["static_geometry_sha256"] = _sha256(self.static_geometry) if self.static_geometry.is_file() else None
        for k, v in self.surfaces.items():
            cfg[f"surface_{k}_sha256"] = _sha256(v) if v.is_file() else None
        cfg.pop("environment")
        cfg.pop("openfoam_version")
        return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()

    @property
    def is_prepared(self) -> bool:
        info = self.workdir / CASE_INFO
        if not info.is_file():
            return False
        return json.loads(info.read_text()).get("key") == self._key()

    # -- prepare ----------------------------------------------------------------------------
    def prepare(self, overwrite: bool = False) -> Result:
        """Copy the template, insert the scaled STL and the explicit values."""
        t0 = time.monotonic()
        res = Result(kind="aeromant.prepare", metadata={"case": self.config(), "started_at": utc_now()})
        try:
            if self.workdir.exists() and any(self.workdir.iterdir()):
                if not overwrite:
                    raise ValueError(f"{self.workdir} is not empty; choose a new directory or prepare(overwrite=True)")
                shutil.rmtree(self.workdir)
            surface = read_stl(self.geometry)
            scale = LENGTH_TO_METRES[self.geometry_units]
            body = surface.scaled(scale)
            bmin, bmax = body.bbox
            extent = float((bmax - bmin).max())
            L = self.parameters.get("reference_length") or self.parameters["diameter"]   # rotor templates: D
            u_ref = reference_speed(self.parameters, L)
            if extent > self.template.max_body_extent * L:
                raise ValueError(
                    f"body size {extent:.4g} m is {extent / L:.1f} x reference_length; template "
                    f"{self.template.name!r} is designed for bodies up to {self.template.max_body_extent} x L_ref "
                    f"— check geometry_units and reference_length"
                )
            derive_p = dict(self.parameters)
            if self.static_geometry is not None:
                static = read_stl(self.static_geometry).scaled(scale)
                derive_p["_static_bbox"] = static.bbox
            extra = {k: read_stl(v).scaled(scale) for k, v in self.surfaces.items()}
            if extra:
                derive_p["_surface_bboxes"] = {k: s.bbox for k, s in extra.items()}
                derive_p["_surface_areas"] = {k: s.area for k, s in extra.items()}
            values = self.template.derive(derive_p, bmin, bmax)
            shutil.copytree(self.template.case_dir(self.flavor), self.workdir)
            geometry_dir = self.workdir / self.files.geometry_dir
            geometry_dir.mkdir(parents=True, exist_ok=True)
            (geometry_dir / "README").unlink(missing_ok=True)
            inputs = self.workdir / "inputs"
            inputs.mkdir()
            shutil.copy2(self.geometry, inputs / self.geometry.name)
            write_stl_ascii(body, geometry_dir / "body.stl", "body")
            if self.static_geometry is not None:
                shutil.copy2(self.static_geometry, inputs / ("static_" + self.static_geometry.name))
                write_stl_ascii(static, geometry_dir / "static.stl", "static")
            for k, surf in extra.items():
                shutil.copy2(self.surfaces[k], inputs / (f"{k}_" + self.surfaces[k].name))
                write_stl_ascii(surf, geometry_dir / f"{k}.stl", k)
            self._render(values)
            info = {"key": self._key(), "config": self.config(), "derived": values, "created_at": utc_now(),
                    "body_bbox_m": [bmin.tolist(), bmax.tolist()], "body_area_m2": body.area,
                    "geometry_sha256": _sha256(self.geometry),
                    **({"surface_areas_m2": {k: s.area for k, s in extra.items()},
                        "surface_bboxes_m": {k: [s.bbox[0].tolist(), s.bbox[1].tolist()] for k, s in extra.items()}}
                       if extra else {})}
            (self.workdir / CASE_INFO).write_text(json.dumps(info, indent=2, default=str))
            res.metrics = {
                "body_bbox_min_m": bmin.tolist(), "body_bbox_max_m": bmax.tolist(),
                "body_surface_area_m2": body.area, "n_triangles": int(len(body.triangles)),
                "reynolds_number": u_ref * L / self.parameters["kinematic_viscosity"],
                "background_cells": ([int(values["NX"]), int(values["NY"]), int(values["NZ"])] if "NX" in values
                                     else [int(values["NX0"]) + int(values["NX1"]) + int(values["NX2"]),
                                           int(values["NY0"]) + int(values["NY1"]) + int(values["NY2"]), int(values["NZ"])]),
            }
            if body.volume <= 0 and not self.surfaces:
                res.messages.append("STL encloses no positive volume (open or inverted surface?); snappyHexMesh may fail")
            res.artifacts.update(case=self.workdir, case_info=self.workdir / CASE_INFO,
                                 body_stl=geometry_dir / "body.stl")
            if self.static_geometry is not None:
                res.metrics.update(static_bbox_min_m=static.bbox[0].tolist(), static_bbox_max_m=static.bbox[1].tolist(),
                                   static_surface_area_m2=static.area)
                res.artifacts["static_stl"] = geometry_dir / "static.stl"
                if static.volume <= 0:
                    res.messages.append("static STL encloses no positive volume (open or inverted surface?)")
            for k, surf in extra.items():
                res.metrics[f"{k}_area_m2"] = surf.area
                res.artifacts[f"{k}_stl"] = geometry_dir / f"{k}.stl"
        except (ValueError, FileNotFoundError, OSError) as exc:
            res.fail(str(exc))
        if res.ok and self.environment.flavor() is None:
            res.messages.append(
                f"OpenFOAM flavor could not be detected (WM_PROJECT_VERSION unknown); using the {self.flavor} "
                f"case files. Use OpenFOAMEnvironment(bashrc=...), .conda(...) or .detect() to be explicit"
            )
        res.duration_s = time.monotonic() - t0
        return res

    def _render(self, values: dict[str, str]) -> None:
        for path in sorted(p for p in self.workdir.rglob("*") if p.is_file()):
            if path.parent.name in ("triSurface", "geometry", "inputs"):
                continue
            text = path.read_text()
            names = set(_PLACEHOLDER.findall(text))
            if not names:
                continue
            missing = sorted(names - set(values))
            if missing:
                raise ValueError(f"template file {path.relative_to(self.workdir)} uses unknown placeholders {missing}")
            path.write_text(_PLACEHOLDER.sub(lambda m: values[m.group(1)], text))

    # -- run --------------------------------------------------------------------------------
    def pipeline(self, processors: int = 1) -> list[Step]:
        """The steps ``run`` executes. With ``processors`` > 1 the mesh is still built serially, then the case is
        decomposed (``decomposePar``), the solver runs under ``mpirun -np N ... -parallel`` and the last time is
        reconstructed (``reconstructPar -latestTime``) so that results and plots read the case as usual."""
        steps = list(self.files.pipeline)
        if processors <= 1:
            return steps
        out = []
        for s in steps:
            if s.name == "solver":
                out.append(Step("decompose", ("decomposePar", "-force"), description=f"split the case into {processors} subdomains"))
                out.append(Step("solver", ("mpirun", "-np", str(processors), *s.argv, "-parallel"),
                                description=f"{s.description} on {processors} processors"))
                out.append(Step("reconstruct", ("reconstructPar", "-latestTime"), description="gather the last time step"))
            else:
                out.append(s)
        return out

    def _set_subdomains(self, processors: int) -> None:
        f = self.workdir / "system" / "decomposeParDict"
        text = f.read_text()
        new, n = re.subn(r"numberOfSubdomains\s+\d+\s*;", f"numberOfSubdomains {processors};", text)
        if n != 1:
            raise ValueError(f"{f} has no numberOfSubdomains entry to set")
        f.write_text(new)

    def run(self, steps: Sequence[str] | None = None, *, progress=False, cancel: threading.Event | None = None,
            timeout: float | None = None, processors: int = 1, cache: str | None = None) -> Result:
        """Run the template pipeline, or only ``steps`` (names from ``pipeline(processors)``), in order.

        ``processors`` > 1 runs the solver in parallel with MPI (see ``pipeline``); meshing stays serial.
        Stops at the first failing step. ``checkMesh`` failures are reported but do not stop the run.

        ``cache``: an entry name in the notebook's cache (``vegeta.cache``), for a whole run (no ``steps``). When the
        entry exists it is loaded (its force and residual logs copied in the cache; the case directory itself is not)
        and nothing runs; else the finished result is saved there. Nothing checks whether the case changed: delete
        the entry when it does.
        """
        if cache is not None and steps is not None:
            raise ValueError("cache applies to a whole run: give either steps or cache")
        return _cache.cached(cache, lambda: self._run(steps, progress=progress, cancel=cancel, timeout=timeout,
                                                      processors=processors),
                             Result, CommandRecord, label="aeromant", kinds=("aeromant.run", "aeromant.results"))

    def _run(self, steps: Sequence[str] | None = None, *, progress=False, cancel: threading.Event | None = None,
             timeout: float | None = None, processors: int = 1) -> Result:
        t0 = time.monotonic()
        res = Result(kind="aeromant.run", metadata={"case": self.config(), "started_at": utc_now(), "processors": processors})
        if not self.is_prepared:
            res.fail(f"case in {self.workdir} is not prepared for this configuration; call prepare() first")
            res.duration_s = time.monotonic() - t0
            return res
        if processors < 1:
            res.fail("processors must be >= 1")
            return res
        pipeline = self.pipeline(processors)
        known = [s.name for s in pipeline]
        names = list(steps) if steps is not None else known
        unknown = [s for s in names if s not in known]
        if unknown:
            res.fail(f"unknown step(s) {unknown}; template steps: {known}")
            return res
        by_name = {s.name: s for s in pipeline}
        if processors > 1:
            self._set_subdomains(processors)
        env = dict(self.environment.env or {})
        if processors > 1 and hasattr(os, "geteuid") and os.geteuid() == 0:   # OpenMPI refuses root without these
            env.update(OMPI_ALLOW_RUN_AS_ROOT="1", OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1")
        res.metadata["steps"] = names
        cb, close = resolve_progress(progress, "aeromant")
        iterations = self.parameters.get("iterations", 1)
        try:
            for k, name in enumerate(names):
                step = by_name[name]
                base = k / len(names)
                cb(name, base)
                if step.kind == "internal":
                    rec = self._internal(step)
                else:
                    on_line = None
                    if name == "solver":
                        def on_line(line, base=base, share=1 / len(names)):
                            m = re.match(r"^Time = (\d+)", line)
                            if m:
                                cb(name, base + share * min(1.0, int(m.group(1)) / iterations), f"iteration {m.group(1)}")
                    rec = run_command(self.environment.command(list(step.argv)), self.workdir, env=env or None,
                                      timeout=timeout, log_path=self.workdir / f"log.{name}", on_output=on_line,
                                      cancel=cancel)
                res.execution.append(rec)
                if rec.log_file:
                    res.artifacts[f"log_{name}"] = Path(rec.log_file)
                if rec.error == "cancelled":
                    res.fail(f"{name} cancelled by user", status="cancelled")
                    break
                if rec.error and "not found" in rec.error and step.kind == "openfoam":
                    res.fail(f"step {name} failed: {rec.error} — is this an {self.flavor} installation "
                             f"(version {self.environment.version()})? Aeromant selected the {self.flavor} case files")
                    break
                if name == "checkMesh" and rec.error is None and rec.log_file:
                    mc = read_checkmesh(rec.log_file)
                    res.metrics.update(mesh_cells=mc.cells, mesh_ok=mc.ok, mesh_failed_checks=mc.failed_checks)
                    if not mc.ok:
                        res.messages.append(f"checkMesh reports {mc.failed_checks} failed check(s): "
                                            + "; ".join(mc.messages[:5]))
                    continue
                if not rec.ok or "FOAM FATAL" in rec.stdout or "FOAM FATAL" in rec.stderr:
                    res.fail(f"step {name} failed: " + describe_failure(rec))
                    if 'IOstream "sha1"' in rec.stdout + rec.stderr:
                        res.messages.append(
                            "this is a known defect of the Ubuntu 'openfoam' 1912 package (function objects such as "
                            "forceCoeffs cannot start); use an official openfoam.com/.org release or conda-forge "
                            "'openfoam' (see docs/installation.md)"
                        )
                    break
        finally:
            cb("done", 1.0)
            close()
        res.artifacts["case"] = self.workdir
        if res.ok and "solver" in names:
            summary = self.results()
            res.metrics.update(summary.metrics)
            res.messages.extend(summary.messages)
            res.artifacts.update(summary.artifacts)
            if not summary.ok:
                res.fail("solver ran but results are incomplete")
        res.metadata["tool_versions"] = {"openfoam": _version_from_logs(self.workdir)}
        res.duration_s = time.monotonic() - t0
        res.artifacts["summary"] = res.save_json(self.workdir / "summary.json")
        return res

    def _internal(self, step) -> CommandRecord:
        t0 = time.monotonic()
        started = utc_now()
        src, dst = self.workdir / step.argv[1], self.workdir / step.argv[2]
        try:
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            err, rc = None, 0
        except OSError as exc:
            err, rc = str(exc), None
        return CommandRecord(command=["(aeromant)", *step.argv], cwd=str(self.workdir), returncode=rc,
                             duration_s=time.monotonic() - t0, started_at=started, error=err)

    # -- results ----------------------------------------------------------------------------
    def results(self, average_window: int = 50) -> Result:
        """Read existing results; never runs anything."""
        return read_case_results(self.workdir, average_window)

    @property
    def key(self) -> str:
        """The case's identity: template, parameters and the STL files' hashes (not the OpenFOAM installation)."""
        return self._key()

    @property
    def is_solved(self) -> bool:
        """Prepared for exactly this configuration and its results read back complete."""
        return self.is_prepared and self.results().ok

    def ensure(self, *, run: bool = True, processors: int = 1, progress=False, cancel: threading.Event | None = None,
               timeout: float | None = None, cache: str | None = None) -> Result:
        """The case's results, solving only when needed.

        Solved already with these inputs (same key, results complete) -> read back, nothing runs. Otherwise the
        directory is prepared again (a stale or half-run case is replaced) and the whole pipeline runs. With
        ``run=False`` nothing is prepared or run: an unsolved case comes back as a failed result saying NOT RUN.
        ``metadata["reused"]`` says which happened. ``cache``: as in ``run`` (an existing entry is returned before
        anything else is looked at; a result solved or read back here is saved as the entry).
        """
        return _cache.cached(cache, lambda: self._ensure(run=run, processors=processors, progress=progress, cancel=cancel,
                                                         timeout=timeout),
                             Result, CommandRecord, label="aeromant", kinds=("aeromant.run", "aeromant.results"))

    def _ensure(self, *, run: bool = True, processors: int = 1, progress=False, cancel: threading.Event | None = None,
                timeout: float | None = None) -> Result:
        if self.is_prepared:
            done = self.results()
            if done.ok:
                done.metadata["reused"] = True
                return done
        if not run:
            res = Result(kind="aeromant.run", metadata={"case": self.config(), "reused": False, "not_run": True})
            return res.fail(f"NOT RUN: {self.template.name} case in {self.workdir} (run=False)")
        prep = self.prepare(overwrite=True)
        if not prep.ok:
            prep.metadata["reused"] = False
            return prep
        res = self.run(progress=progress, processors=processors, cancel=cancel, timeout=timeout)
        res.metadata["reused"] = False
        return res


def open_case(workdir: str | Path) -> dict:
    """Case information written by ``prepare()``."""
    info = Path(workdir) / CASE_INFO
    if not info.is_file():
        raise FileNotFoundError(f"{workdir} is not an Aeromant case (no {CASE_INFO})")
    return json.loads(info.read_text())


def _version_from_logs(case: Path) -> str | None:
    from .results import read_version

    for log in ("log.blockMesh", "log.solver"):
        p = Path(case) / log
        if p.is_file():
            v = read_version(p.read_text(errors="replace"))
            if v:
                return v
    return None


def read_case_results(workdir: str | Path, average_window: int = 50) -> Result:
    """Summarise an existing case directory: coefficients, convergence, mesh check."""
    workdir = Path(workdir)
    res = Result(kind="aeromant.results", artifacts={"case": workdir})
    try:
        info = open_case(workdir)
    except FileNotFoundError as exc:
        return res.fail(str(exc))
    p = info["config"]["parameters"]
    res.metadata = {"case": info["config"]}
    m = res.metrics
    if "velocity" in p:
        m["reynolds_number"] = p["velocity"] * p["reference_length"] / p["kinematic_viscosity"]
    elif "rpm" in p:  # rotor: Reynolds number of the tip speed and the diameter
        omega = p["rpm"] * 2 * math.pi / 60
        m["tip_speed_m_s"] = omega * p["diameter"] / 2
        m["reynolds_number"] = m["tip_speed_m_s"] * p["diameter"] / p["kinematic_viscosity"]
    else:  # suction: the duct velocity and the hood width
        m["suction_velocity_m_s"] = reference_speed(p, p["reference_length"])
        m["reynolds_number"] = m["suction_velocity_m_s"] * p["reference_length"] / p["kinematic_viscosity"]
    if (workdir / "log.checkMesh").is_file():
        mc = read_checkmesh(workdir / "log.checkMesh")
        m.update(mesh_cells=mc.cells, mesh_ok=mc.ok, mesh_failed_checks=mc.failed_checks)
    log = workdir / "log.solver"
    if log.is_file():
        sl = read_solver_log(log)
        m.update(iterations=sl.iterations, converged=sl.converged,
                 final_residuals={k: float(v[-1]) for k, v in sl.residuals.items() if len(v)})
        res.artifacts["solver_log"] = log
        if not sl.completed:
            res.messages.append("solver log has no 'End': the run did not finish")
        if not sl.converged:
            res.messages.append(f"residual target not reached in {sl.iterations} iterations; "
                                "check coefficient histories before trusting the values")
    if info["config"].get("template") == "compressor_mrf":
        return _compressor_results(res, workdir, p, info, average_window)
    if str(info["config"].get("template", "")).startswith("rotor_"):
        return _rotor_results(res, workdir, p, average_window)
    if info["config"].get("template") == "suction_hood":
        return _suction_results(res, workdir, p, average_window)
    files = find_coefficient_files(workdir)
    if not files:
        for c in ("Cd", "Cl", "Cm"):
            m[c] = None
        return res.fail("no forceCoeffs output found (solver NOT RUN or failed)")
    hist = read_coefficients(files)
    res.artifacts["force_coefficients"] = files[-1]
    density = p["density"] if "density" in p else p["pressure"] / (GAS_CONSTANT * p["temperature"])
    q = 0.5 * density * p["velocity"] ** 2
    n = max(1, min(average_window, len(hist.data)))
    for c in ("Cd", "Cl", "Cm"):
        if c in hist:
            series = hist[c]
            m[c] = float(series[-1])
            m[f"{c}_mean_last{n}"] = float(np.mean(series[-n:]))
            m[f"{c}_std_last{n}"] = float(np.std(series[-n:]))
        else:
            m[c] = None
            res.messages.append(f"{c} not present in forceCoeffs output")
    if m.get("Cd") is not None:
        m["drag_force_N"] = m["Cd"] * q * p["reference_area"]
    if m.get("Cl") is not None:
        m["lift_force_N"] = m["Cl"] * q * p["reference_area"]
    if info["config"].get("template") == "jet_external":
        _jet_results(res, workdir, p, average_window)
    return res


def _jet_results(res: Result, workdir: Path, p: dict, average_window: int) -> None:
    """The engine's flows through the intake and nozzle faces (a check of the boundary conditions), the free-stream
    Mach number, and the jet's temperature and exhaust fraction along its axis (``postProcessing/plumeLine``)."""
    from .results import read_sample_set

    m = res.metrics
    a0 = math.sqrt(1.4 * GAS_CONSTANT * p["temperature"])
    m["mach_number"] = p["velocity"] / a0
    n = average_window
    for name, key, set_value in (("intakeFlow", "intake_mass_flow_kg_s", p["intake_mass_flow"]),
                                 ("exhaustFlow", "exhaust_mass_flow_kg_s", p["exhaust_mass_flow"])):
        try:
            rows = read_surface_field_values(workdir, name)
        except FileNotFoundError:
            m[key] = None
            continue
        flow = abs(float(np.mean(rows[-min(n, len(rows)):, 1])))
        m[key] = flow
        if abs(flow / set_value - 1) > 0.02:
            res.messages.append(f"{name}: {flow:.4g} kg/s through the face, {set_value:.4g} set: check the face STL")
    try:
        x, cols = read_sample_set(workdir, "plumeLine")
    except FileNotFoundError:
        m["plume"] = None
        res.messages.append("no plume samples (postProcessing/plumeLine): solver not run, or the sets function failed")
        return
    T, tracer = cols.get("T"), cols.get("exhaust")
    x0 = float(x[0])
    plume = {"distance_m": (x - x0).tolist()}
    if T is not None:
        plume["temperature_K"] = T.tolist()
        plume["excess_temperature_K"] = (T - p["temperature"]).tolist()
    if tracer is not None:
        plume["exhaust_fraction"] = tracer.tolist()
    m["plume"] = plume
    for d in (0.5, 1.0, 2.0, 5.0):
        if x[-1] - x0 >= d:
            if T is not None:
                m[f"jet_excess_T_at_{d:g}m_K"] = float(np.interp(x0 + d, x, T) - p["temperature"])
            if tracer is not None:
                m[f"exhaust_fraction_at_{d:g}m"] = float(np.interp(x0 + d, x, tracer))


def _compressor_results(res: Result, workdir: Path, p: dict, info: dict, average_window: int) -> Result:
    """Mass flow, total-to-total pressure ratio, isentropic efficiency and shaft power of a ``compressor_mrf`` case.

    Totals from the mass-averaged static pressure, temperature and speed at each face (``inletState``/``outletState``)
    with the isentropic relations; the shaft power from the impeller torque (``forces``). The efficiency from the
    temperatures and from the torque should agree within a few percent on a converged case."""
    from .results import find_force_files, read_force_history

    m = res.metrics
    g, cp = 1.4, 1005.0
    n = average_window
    try:
        flow_in = read_surface_field_values(workdir, "inletFlow")
        flow_out = read_surface_field_values(workdir, "outletFlow")
        s_in = read_surface_field_values(workdir, "inletState")
        s_out = read_surface_field_values(workdir, "outletState")
    except FileNotFoundError as exc:
        m.update(mass_flow_kg_s=None, pressure_ratio_tt=None, efficiency_tt=None)
        return res.fail(f"no surfaceFieldValue output found (solver NOT RUN or failed): {exc}")

    def last(a, col):
        k = max(1, min(n, len(a)))
        return float(np.mean(a[-k:, col]))

    mdot = abs(last(flow_in, 1))
    m["mass_flow_kg_s"] = mdot
    m["mass_flow_outlet_kg_s"] = abs(last(flow_out, 1))
    totals = {}
    for side, s in (("inlet", s_in), ("outlet", s_out)):
        ps, ts, u = last(s, 1), last(s, 2), last(s, 3)
        t0 = ts + u * u / (2 * cp)
        totals[side] = (ps * (t0 / ts) ** (g / (g - 1)), t0, ps, ts, u)
        m[f"{side}_static_pressure_Pa"], m[f"{side}_static_temperature_K"], m[f"{side}_speed_m_s"] = ps, ts, u
        m[f"{side}_total_pressure_Pa"], m[f"{side}_total_temperature_K"] = totals[side][0], t0
    pr = totals["outlet"][0] / totals["inlet"][0]
    tr = totals["outlet"][1] / totals["inlet"][1]
    m["pressure_ratio_tt"] = pr
    m["temperature_ratio_tt"] = tr
    m["efficiency_tt"] = (pr ** ((g - 1) / g) - 1) / (tr - 1) if tr > 1 else None
    omega = p["rotation"] * p["rpm"] * 2 * math.pi / 60
    m["tip_speed_m_s"] = abs(omega) * p["diameter"] / 2
    q_files = find_force_files(workdir, "moment.dat", "forces")
    if q_files:
        Q = read_force_history(q_files)
        k = max(1, min(n, len(Q)))
        torque = -float(np.mean(Q[-k:, 1])) * p["rotation"]
        power = torque * abs(omega)
        m.update(torque_Nm=torque, shaft_power_W=power)
        if mdot > 0 and power > 0:
            m["work_per_kg_J"] = power / mdot
            m["work_coefficient"] = power / mdot / m["tip_speed_m_s"] ** 2   # slip x power input of the cycle model
            ideal = cp * totals["inlet"][1] * (pr ** ((g - 1) / g) - 1)
            m["efficiency_from_torque"] = ideal * mdot / power
    tin = totals["inlet"][1]
    m["corrected_mass_flow_kg_s"] = mdot * math.sqrt(tin / 288.15) / (totals["inlet"][0] / 101325.0)
    if abs(m["mass_flow_outlet_kg_s"] / max(mdot, 1e-12) - 1) > 0.02:
        res.messages.append(f"inlet and outlet mass flows differ by {m['mass_flow_outlet_kg_s'] / mdot - 1:+.1%}: not converged")
    res.messages.append(f"averaged over the last {n} iterations; totals from mass-averaged static values (the outlet's swirl "
                        "counts in the total pressure: put the outlet face far out in the vaneless diffuser)")
    return res


def _suction_results(res: Result, workdir: Path, p: dict, average_window: int) -> Result:
    """The depression at the duct, the flow drawn and the air power of a ``suction_hood`` case, from its
    surfaceFieldValue function objects (``postProcessing/<name>/<time>/surfaceFieldValue.dat``)."""
    from .results import read_surface_field_values

    m = res.metrics
    rho, Q_set = p["density"], p["flow_rate"]
    try:
        p_duct = read_surface_field_values(workdir, "suctionPressure")
        phi = read_surface_field_values(workdir, "suctionFlow")
        p_hood = read_surface_field_values(workdir, "hoodPressure")
    except FileNotFoundError as exc:
        m.update(suction_pressure_Pa=None, flow_rate_m3_s=None, air_power_W=None)
        return res.fail(f"no surfaceFieldValue output found (solver NOT RUN or failed): {exc}")
    n = max(1, min(average_window, len(p_duct)))
    dp = -rho * float(np.mean(p_duct[-n:, 1]))                  # kinematic p at the duct -> the depression there
    Q = float(np.mean(phi[-n:, 1]))                             # phi is the outward flux: positive leaving the domain
    m.update(suction_pressure_Pa=-dp, fan_static_pressure_Pa=dp, flow_rate_m3_s=Q, flow_rate_set_m3_s=Q_set,
             hood_wall_pressure_Pa=rho * float(np.mean(p_hood[-n:, 1])), air_power_W=Q * dp,
             suction_pressure_std_Pa=rho * float(np.std(p_duct[-n:, 1])), averaging_window=n)
    if Q_set > 0 and abs(Q / Q_set - 1) > 0.02:
        res.messages.append(f"the flow through the suction patch ({Q:.4g} m3/s) differs from flow_rate ({Q_set:.4g}): "
                            "the suction square is not the duct's inner section (check duct_center, duct_inner, duct_wall)")
    res.messages.append(f"pressures averaged over the last {n} iterations; the fan must supply the depression at the duct plus "
                        "the losses downstream of it (not modelled)")
    return res


def _rotor_results(res: Result, workdir: Path, p: dict, average_window: int) -> Result:
    """Thrust, torque and power of a rotor template from the forces function object."""
    from .results import find_force_files, read_force_history

    m = res.metrics
    f_files, q_files = find_force_files(workdir, "force.dat", "forces"), find_force_files(workdir, "moment.dat", "forces")
    if not f_files or not q_files:
        m.update(thrust_N=None, torque_Nm=None, power_W=None)
        return res.fail("no forces output found (solver NOT RUN or failed)")
    F, Q = read_force_history(f_files), read_force_history(q_files)
    res.artifacts.update(forces=f_files[-1], moments=q_files[-1])
    n = max(1, min(average_window, len(F)))
    omega = p["rotation"] * p["rpm"] * 2 * math.pi / 60
    fx, mx = F[-n:, 1], Q[-n:, 1]
    thrust = -float(np.mean(fx))                        # the rotor pushes fluid toward +x, so its thrust is along -x
    torque = -float(np.mean(mx)) * p["rotation"]        # fluid torque on the rotor opposes the rotation
    power = torque * abs(omega)
    nrev, D, rho = p["rpm"] / 60, p["diameter"], p["density"]
    m.update(thrust_N=thrust, thrust_std_lastN=float(np.std(fx)), torque_Nm=torque, torque_std_lastN=float(np.std(mx)),
             power_W=power, omega_rad_s=omega, averaging_window=n,
             force_N=[-float(x) for x in np.mean(F[-n:, 1:4], axis=0)],
             moment_Nm=[-float(x) for x in np.mean(Q[-n:, 1:4], axis=0)],
             ct=thrust / (rho * nrev ** 2 * D ** 4), cp=power / (rho * nrev ** 3 * D ** 5) if power else None)
    if "airspeed" in p:
        m["efficiency"] = thrust * p["airspeed"] / power if power > 0 else None
        m["advance_ratio"] = p["airspeed"] / (nrev * D)
    else:
        area = math.pi * (D / 2) ** 2
        ideal = thrust * math.sqrt(max(thrust, 0.0) / (2 * rho * area))
        m["figure_of_merit"] = ideal / power if power > 0 else None
    s_files = find_force_files(workdir, "force.dat", "staticForces")
    if s_files:                                          # rotor_mrf_installed: the standing body's own forces
        S = read_force_history(s_files)
        fb = np.mean(S[-n:, 1:4], axis=0)
        m.update(body_force_N=[float(x) for x in fb], body_drag_N=float(fb[0]), net_thrust_N=thrust - float(fb[0]))
        res.artifacts["body_forces"] = s_files[-1]
        if "airspeed" in p and power > 0:
            m["net_efficiency"] = m["net_thrust_N"] * p["airspeed"] / power
    if thrust < 0:
        res.messages.append("negative thrust: the blades are handed against the sense of rotation; rerun with the "
                            "other 'rotation' value (or check that the rotor axis is +x)")
    if torque < 0:
        res.messages.append("negative torque (fluid drives the rotor): check the sense of rotation")
    res.messages.append(f"forces averaged over the last {n} iterations; steady MRF result, compare with blade element theory")
    return res
