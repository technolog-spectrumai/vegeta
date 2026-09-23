"""Revisions: immutable design states (parameters + source), plus geometry and evaluations added later."""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Callable, Sequence

from ._io import append_jsonl, environment, read_jsonl, utc_now, write_once
from .evaluation import Evaluation, factory_record
from .workspace import check_name

LABELS = ("preferred", "rejected", "reference", "unclassified")


def create_revision(ws, design, parent, overrides: dict, note: str):
    dd = design.load()
    base = dict(parent.params) if parent is not None else {}
    unknown = sorted(set(overrides) - {p.name for p in dd.params})
    if unknown:
        raise ValueError(f"unknown parameter(s) {unknown} for design {design.name!r}")
    params = dd.resolve(**{**base, **overrides})         # validates types and ranges
    identity = dd.source_identity()
    reference = parent.params if parent is not None else dd.params.defaults
    changes = {k: v for k, v in params.items() if reference.get(k) != v}
    rid, rdir = ws._allocate_revision_dir()
    try:
        snapshot = None
        if identity.get("source_file") and Path(identity["source_file"]).is_file():
            (rdir / "source").mkdir()
            snap = rdir / "source" / Path(identity["source_file"]).name
            shutil.copy2(identity["source_file"], snap)
            snapshot = os.path.relpath(snap, rdir)
        write_once(rdir / "revision.json", {
            "id": rid, "design": design.name, "parent": parent.id if parent is not None else None,
            "parameters": params, "changes": changes, "note": note,
            "source": {"spec": design.source, "build_sha256": identity.get("source_sha256"),
                       "qualname": identity.get("qualname"), "snapshot": snapshot},
            "units": dd.units, "created_at": utc_now(), "environment": environment(),
        })
    except Exception:
        shutil.rmtree(rdir, ignore_errors=True)
        raise
    return ws.revision(rid)


class Revision:
    """One immutable design state. Geometry and evaluations are added by explicit calls only."""

    def __init__(self, workspace, rid: str, record: dict):
        self.workspace = workspace
        self.id = rid
        self.record = record
        self.dir = workspace.root / "revisions" / rid

    # -- identity ---------------------------------------------------------------------------
    @property
    def design_name(self) -> str:
        return self.record["design"]

    @property
    def design(self):
        return self.workspace.design(self.design_name)

    @property
    def params(self) -> dict:
        return dict(self.record["parameters"])

    @property
    def parent(self) -> str | None:
        return self.record["parent"]

    @property
    def changes(self) -> dict:
        return dict(self.record.get("changes", {}))

    # -- geometry ---------------------------------------------------------------------------
    @property
    def geometry_dir(self) -> Path:
        return self.dir / "geometry"

    def _geometry_file(self, suffix: str) -> Path | None:
        hits = sorted(self.geometry_dir.glob(f"*.{suffix}")) if self.geometry_dir.is_dir() else []
        return hits[0] if hits else None

    @property
    def step(self) -> Path | None:
        return self._geometry_file("step")

    @property
    def stl(self) -> Path | None:
        return self._geometry_file("stl")

    @property
    def is_generated(self) -> bool:
        return self.step is not None and self.stl is not None

    def geometry_summary(self) -> dict | None:
        p = self.geometry_dir / "summary.json"
        if not p.is_file():
            return None
        import json

        return json.loads(p.read_text())

    def generate(self, stl_tolerance: float = 0.01, stl_angular_tolerance: float = 0.1):
        """Build and export this revision's geometry (STEP + STL) once. Returns the Dedalus result."""
        from vegeta.dedalus import Result

        dd = self.design.load()
        res = Result(kind="dedalus.generate")
        if self.is_generated:
            return res.fail(f"{self.id} is already generated; revisions are immutable (branch to change it)")
        sha = dd.source_identity().get("source_sha256")
        if sha != self.record["source"]["build_sha256"]:
            return res.fail(
                f"the design source changed since {self.id} was created (recorded build sha256 "
                f"{str(self.record['source']['build_sha256'])[:12]}, now {str(sha)[:12]}); "
                f"create a new revision with branch() to use the current source"
            )
        if self.geometry_dir.exists():  # a previous failed attempt: keep it, out of the way
            n = 1
            while (self.dir / f"geometry.failed-{n}").exists():
                n += 1
            self.geometry_dir.rename(self.dir / f"geometry.failed-{n}")
        res = dd.run(self.geometry_dir, formats=("step", "stl"), stl_tolerance=stl_tolerance,
                     stl_angular_tolerance=stl_angular_tolerance, **self.params)
        append_jsonl(self.dir / "annotations.jsonl", {"type": "generate", "status": res.status,
                                                      "at": utc_now(), "messages": res.messages})
        return res

    # -- evaluations ------------------------------------------------------------------------
    def _eval_dir(self, kind: str, name: str) -> Path:
        return self.dir / "evaluations" / f"{kind}-{name}"

    def evaluations(self) -> list[Evaluation]:
        root = self.dir / "evaluations"
        if not root.is_dir():
            return []
        return [Evaluation.load(p) for p in sorted(root.iterdir()) if (p / "evaluation.json").is_file()]

    def evaluation(self, kind: str, name: str) -> Evaluation | None:
        d = self._eval_dir(kind, name)
        return Evaluation.load(d) if (d / "evaluation.json").is_file() else None

    def _precheck(self, kind: str, name: str, needs: str) -> Evaluation | None:
        check_name("evaluation", name)
        if not self.is_generated:
            return Evaluation(kind, name, "failed", messages=[
                f"{self.id} has no geometry; run generate() first (nothing is generated automatically)"])
        if self._eval_dir(kind, name).exists():
            return Evaluation(kind, name, "failed", messages=[
                f"{kind} evaluation {name!r} already exists on {self.id}; results are immutable, use a new name"])
        return None

    def _record(self, kind, name, t0, started, status, metrics, messages, tool_results, config, factory,
                extra=None) -> Evaluation:
        edir = self._eval_dir(kind, name)
        tool_versions, commands, summaries = {}, [], []
        for r in tool_results:
            tool_versions.update(r.metadata.get("tool_versions", {}) or {})
            commands += [c.to_dict() for c in r.execution]
            if "summary" in r.artifacts:
                summaries.append(os.path.relpath(r.artifacts["summary"], edir))
            elif "mesh_summary" in r.artifacts:
                summaries.append(os.path.relpath(r.artifacts["mesh_summary"], edir))
        rec = {"kind": kind, "name": name, "revision": self.id, "status": status, "metrics": metrics,
               "messages": messages, "tool_summaries": summaries, "config": config, "factory": factory,
               "tool_versions": tool_versions, "commands": commands, "started_at": started,
               "finished_at": utc_now(), "duration_s": time.monotonic() - t0, "environment": environment()}
        rec.update(extra or {})
        write_once(edir / "evaluation.json", rec)
        return Evaluation(kind, name, status, metrics, messages, edir, rec, tool_results)

    def run_fea(self, name: str, model_factory: Callable, *, threads: int = 1, timeout: float | None = None,
                executable: str = "ccx", progress=False) -> Evaluation:
        """Mesh and solve ``model_factory(self)`` (a ``talos.StructuralModel`` on this revision's STEP)."""
        from vegeta.talos import StructuralModel

        refused = self._precheck("fea", name, "step")
        if refused:
            return refused
        model = model_factory(self)
        if not isinstance(model, StructuralModel):
            raise ValueError("model_factory must return a talos.StructuralModel")
        if Path(model.geometry).resolve() != self.step.resolve():
            raise ValueError(f"the model must use this revision's STEP ({self.step}), not {model.geometry}")
        t0, started = time.monotonic(), utc_now()
        edir = self._eval_dir("fea", name)
        mesh = model.mesh(edir, progress=progress)
        results = [mesh]
        if mesh.ok:
            results.append(model.solve(edir, executable=executable, threads=threads, timeout=timeout,
                                       progress=progress))
        last = results[-1]
        metrics = dict(last.metrics) if last.ok else {k: v for k, v in mesh.metrics.items()}
        return self._record("fea", name, t0, started, last.status, metrics,
                            [m for r in results for m in r.messages], results, model.config(),
                            factory_record(model_factory))

    def run_cfd(self, name: str, case_factory: Callable, *, steps: Sequence[str] | None = None,
                timeout: float | None = None, progress=False) -> Evaluation:
        """Prepare and run ``case_factory(self, workdir)`` (an ``aeromant.CFDCase`` on this revision's STL)."""
        from vegeta.aeromant import CFDCase

        refused = self._precheck("cfd", name, "stl")
        if refused:
            return refused
        edir = self._eval_dir("cfd", name)
        case = case_factory(self, edir / "case")
        if not isinstance(case, CFDCase):
            raise ValueError("case_factory must return an aeromant.CFDCase")
        if case.geometry.resolve() != self.stl.resolve():
            raise ValueError(f"the case must use this revision's STL ({self.stl}), not {case.geometry}")
        if case.workdir.resolve() != (edir / "case").resolve():
            raise ValueError("the case must use the workdir passed to case_factory")
        t0, started = time.monotonic(), utc_now()
        prep = case.prepare()
        results = [prep]
        if prep.ok:
            results.append(case.run(steps=steps, timeout=timeout, progress=progress))
        last = results[-1]
        return self._record("cfd", name, t0, started, last.status, dict(last.metrics),
                            [m for r in results for m in r.messages], results, case.config(),
                            factory_record(case_factory), {"steps": list(steps) if steps else None})

    def run_print(self, name: str, settings, orientation, *, executable="prusa-slicer",
                  timeout: float | None = None) -> Evaluation:
        """Slice this revision's STL with explicit settings and an engineer-chosen orientation."""
        from vegeta.mellonia import Orientation, PrintSettings, slice_stl

        if not isinstance(settings, PrintSettings) or not isinstance(orientation, Orientation):
            raise ValueError("run_print needs mellonia.PrintSettings and mellonia.Orientation")
        refused = self._precheck("print", name, "stl")
        if refused:
            return refused
        t0, started = time.monotonic(), utc_now()
        res = slice_stl(self.stl, settings, orientation, self._eval_dir("print", name), executable=executable,
                        timeout=timeout)
        config = {"settings_name": settings.name, "printer": settings.printer, "filament": settings.filament,
                  "print": settings.print, "orientation": res.metadata.get("orientation")}
        return self._record("print", name, t0, started, res.status, dict(res.metrics), list(res.messages),
                            [res], config, None)

    # -- history ----------------------------------------------------------------------------
    def branch(self, note: str = "", **parameters):
        """New revision from this one with some parameters changed (and the design's current source)."""
        from .revision import create_revision

        return create_revision(self.workspace, self.design, parent=self, overrides=parameters, note=note)

    def label(self, label: str, note: str = "") -> None:
        if label not in LABELS:
            raise ValueError(f"label must be one of {LABELS}")
        append_jsonl(self.dir / "annotations.jsonl", {"type": "label", "label": label, "note": note, "at": utc_now()})

    def note(self, text: str) -> None:
        append_jsonl(self.dir / "annotations.jsonl", {"type": "note", "note": text, "at": utc_now()})

    def annotations(self) -> list[dict]:
        return read_jsonl(self.dir / "annotations.jsonl")

    @property
    def current_label(self) -> str:
        labels = [a["label"] for a in self.annotations() if a.get("type") == "label"]
        return labels[-1] if labels else "unclassified"

    def __repr__(self) -> str:
        ch = ", ".join(f"{k}={v}" for k, v in self.changes.items())
        return f"<Revision {self.id} {self.design_name}({ch}) parent={self.parent} label={self.current_label}>"
