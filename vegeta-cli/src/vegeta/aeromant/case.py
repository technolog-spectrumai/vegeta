"""CFD cases built from templates: prepare, run explicitly chosen steps, extract results."""
from __future__ import annotations

import hashlib
import json
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
from .results import find_coefficient_files, read_checkmesh, read_coefficients, read_solver_log
from .stl import LENGTH_TO_METRES, read_stl, write_stl_ascii
from .templates import TemplateSpec, get_template

CASE_INFO = "aeromant_case.json"
_PLACEHOLDER = re.compile(r"\{\{([A-Z0-9_]+)\}\}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CFDCase:
    """One CFD case directory made from a template, an STL and explicit parameters.

    Nothing runs on construction. ``prepare()`` writes the case; ``run()`` executes the template
    pipeline (or chosen steps); ``results()`` reads what exists. All files stay in ``workdir``.
    """

    def __init__(self, template: str | TemplateSpec, geometry: str | Path, parameters: dict[str, Any],
                 workdir: str | Path, geometry_units: str, environment: OpenFOAMEnvironment | None = None):
        self.template = get_template(template)
        self.geometry = Path(geometry)
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
            "parameters": self.parameters,
            "user_parameters": self.user_parameters,
            "environment": self.environment.describe(),
        }

    def _key(self) -> str:
        cfg = dict(self.config(), geometry_sha256=_sha256(self.geometry) if self.geometry.is_file() else None)
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
            L = self.parameters["reference_length"]
            if extent > self.template.max_body_extent * L:
                raise ValueError(
                    f"body size {extent:.4g} m is {extent / L:.1f} x reference_length; template "
                    f"{self.template.name!r} is designed for bodies up to {self.template.max_body_extent} x L_ref "
                    f"— check geometry_units and reference_length"
                )
            values = self.template.derive(self.parameters, bmin, bmax)
            shutil.copytree(self.template.case_dir(self.flavor), self.workdir)
            geometry_dir = self.workdir / self.files.geometry_dir
            geometry_dir.mkdir(parents=True, exist_ok=True)
            (geometry_dir / "README").unlink(missing_ok=True)
            inputs = self.workdir / "inputs"
            inputs.mkdir()
            shutil.copy2(self.geometry, inputs / self.geometry.name)
            write_stl_ascii(body, geometry_dir / "body.stl", "body")
            self._render(values)
            info = {"key": self._key(), "config": self.config(), "derived": values, "created_at": utc_now(),
                    "body_bbox_m": [bmin.tolist(), bmax.tolist()], "body_area_m2": body.area,
                    "geometry_sha256": _sha256(self.geometry)}
            (self.workdir / CASE_INFO).write_text(json.dumps(info, indent=2, default=str))
            res.metrics = {
                "body_bbox_min_m": bmin.tolist(), "body_bbox_max_m": bmax.tolist(),
                "body_surface_area_m2": body.area, "n_triangles": int(len(body.triangles)),
                "reynolds_number": self.parameters["velocity"] * L / self.parameters["kinematic_viscosity"],
                "background_cells": [int(values["NX"]), int(values["NY"]), int(values["NZ"])],
            }
            if body.volume <= 0:
                res.messages.append("STL encloses no positive volume (open or inverted surface?); snappyHexMesh may fail")
            res.artifacts.update(case=self.workdir, case_info=self.workdir / CASE_INFO,
                                 body_stl=geometry_dir / "body.stl")
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
    def run(self, steps: Sequence[str] | None = None, *, progress=False, cancel: threading.Event | None = None,
            timeout: float | None = None) -> Result:
        """Run the template pipeline, or only ``steps`` (names from ``template.step_names``), in order.

        Stops at the first failing step. ``checkMesh`` failures are reported but do not stop the run.
        """
        t0 = time.monotonic()
        res = Result(kind="aeromant.run", metadata={"case": self.config(), "started_at": utc_now()})
        if not self.is_prepared:
            res.fail(f"case in {self.workdir} is not prepared for this configuration; call prepare() first")
            res.duration_s = time.monotonic() - t0
            return res
        names = list(steps) if steps is not None else self.template.step_names(self.flavor)
        unknown = [s for s in names if s not in self.template.step_names(self.flavor)]
        if unknown:
            res.fail(f"unknown step(s) {unknown}; template steps: {self.template.step_names(self.flavor)}")
            return res
        by_name = {s.name: s for s in self.files.pipeline}
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
                    rec = run_command(self.environment.command(list(step.argv)), self.workdir, env=self.environment.env,
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
    m["reynolds_number"] = p["velocity"] * p["reference_length"] / p["kinematic_viscosity"]
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
    files = find_coefficient_files(workdir)
    if not files:
        for c in ("Cd", "Cl", "Cm"):
            m[c] = None
        return res.fail("no forceCoeffs output found (solver NOT RUN or failed)")
    hist = read_coefficients(files)
    res.artifacts["force_coefficients"] = files[-1]
    q = 0.5 * p["density"] * p["velocity"] ** 2
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
    return res
