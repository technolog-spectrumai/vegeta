"""The design copilot: iterate on one Dedalus design file with an AI proposer, under the engineer's control.

Proposed code never runs in this process: the current design and every proposal are built and measured by
Fidia's sandboxed runner (``sandbox.execute``), so the design file must follow Fidia's contract (one ``Design``
class; imports from cadquery, math, numpy and ``vegeta.dedalus`` only)."""
from __future__ import annotations

import json
import shutil
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from vegeta.dedalus.loading import load_design

from .proposals import Proposal, Validation
from .sandbox import ExecResult, Sandbox, execute


class Proposer(Protocol):
    name: str

    def propose(self, context: dict, instruction: str, history: list[dict]) -> tuple[dict, dict]: ...

    def describe(self) -> dict: ...


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DesignSession:
    """Iterate on ``design_spec`` (``path/file.py:Name``) with a proposer.

    Every ``ask()`` produces a Proposal that Vegeta builds and measures in the sandbox (a subprocess with a
    timeout, a memory cap and no API keys); the design file changes only in ``accept()``. All proposals and decisions are appended to
    ``<design dir>/<file>.ai.jsonl`` and the previous file content is kept as ``<file>.<n>.bak``.
    """

    def __init__(self, design_spec: str, proposer: Proposer, *, parameters: dict | None = None,
                 notes: str = "", log_dir: str | Path | None = None, sandbox: Sandbox | None = None):
        target, sep, attr = design_spec.partition(":")
        self.path = Path(target).resolve()
        if not self.path.is_file() or self.path.suffix != ".py":
            raise ValueError("design_spec must be 'path/to/design.py:Name' (a Python file you own)")
        self.attr = attr
        self.proposer = proposer
        self.parameters: dict[str, Any] = dict(parameters or {})
        self.notes = notes
        self.log = (Path(log_dir) if log_dir else self.path.parent) / f"{self.path.stem}.ai.jsonl"
        self.history: list[dict] = []
        self.proposals: dict[str, Proposal] = {}
        self.sandbox = sandbox or Sandbox()
        self._n = 0

    # -- state ------------------------------------------------------------------------------
    def _spec(self, path: Path | None = None) -> str:
        p = path or self.path
        return f"{p}:{self.attr}" if self.attr else str(p)

    def design(self, path: Path | None = None):
        """The design loaded in this process (the engineer's accepted file; proposals never go through here)."""
        return load_design(self._spec(path))

    def _run(self, source: str, *, strict: dict | None = None, optional: dict | None = None,
             keep_step: Path | None = None) -> ExecResult:
        """Build ``source`` in the sandbox (a throw-away directory); optionally keep its STEP file."""
        with tempfile.TemporaryDirectory(prefix="fidia-copilot-") as tmp:
            ex = execute(source, Path(tmp) / "run", sandbox=self.sandbox, parameters=strict, optional_parameters=optional)
            if keep_step is not None and ex.ok:
                shutil.copyfile(Path(tmp) / "run" / "model.step", keep_step)
        return ex

    def context(self) -> dict:
        source = self.path.read_text()
        ex = self._run(source, optional=self.parameters)
        if not ex.parameter_specs:
            raise RuntimeError(f"the current design does not load ({ex.status}): {ex.error}")
        values = ex.parameters or {p["name"]: p["default"] for p in ex.parameter_specs}
        params = [{"name": p["name"], "value": values.get(p["name"], p["default"]), "units": p["units"], "min": p["min"],
                   "max": p["max"], "description": p["description"]} for p in ex.parameter_specs]
        # the current design may itself be broken; say so instead of hiding it
        measurements = ex.measurements if ex.ok else {"error": f"current design does not build: {ex.error}"}
        return {"path": str(self.path), "source": source, "parameters": params,
                "measurements": measurements, "notes": self.notes}

    # -- propose / validate -----------------------------------------------------------------
    def ask(self, instruction: str) -> Proposal:
        """Ask for a change. Returns a validated Proposal; the design file is untouched."""
        context = self.context()
        data, usage = self.proposer.propose(context, instruction, self.history)
        self._n += 1
        pid = f"p{self._n}"
        prop = Proposal(
            id=pid, kind=data["kind"], summary=data["summary"], rationale=data["rationale"],
            instruction=instruction, source=data.get("source"), parameters=dict(data.get("parameters") or {}),
            expected_effects=list(data.get("expected_effects") or []), risks=list(data.get("risks") or []),
            source_before=context["source"], usage=usage, model=self.proposer.describe().get("model"),
            created_at=utc_now(),
        )
        if prop.kind == "source" and not prop.source:
            prop.kind = "answer"
            prop.risks.append("the proposal was marked as a source change but contained no source")
        prop.validation = self.validate(prop, context)
        self.proposals[pid] = prop
        self.history.append({"user": instruction, "assistant": json.dumps(data)})
        self._record("proposal", prop)
        return prop

    def validate(self, prop: Proposal, context: dict | None = None) -> Validation:
        """Build the proposal in the sandbox and compare measurements (the proposed code runs only there)."""
        context = context or self.context()
        before = {p["name"]: p["value"] for p in context["parameters"]}
        v = Validation(ok=True, parameters_before=before, measurements_before=context.get("measurements"))
        if prop.kind == "answer":
            v.ok = True
            v.messages.append("no change proposed")
            v.parameters_after = before
            return v
        ex = self._run(prop.source if prop.kind == "source" else context["source"], strict=prop.parameters,
                       optional=self.parameters)
        names = [p["name"] for p in ex.parameter_specs]
        if names:
            v.parameters_added = [n for n in names if n not in before]
            v.parameters_removed = [n for n in before if n not in names]
        if not ex.ok:
            v.ok = False
            v.messages.append(f"{ex.status}: {ex.error}")
            if ex.traceback:
                v.messages.append(ex.traceback.strip().splitlines()[-1])
            return v
        v.parameters_after = ex.parameters
        v.measurements_after = ex.measurements
        if not (ex.measurements or {}).get("valid"):
            v.ok = False
            v.messages.append("proposed geometry is not a valid solid")
        return v

    # -- decisions --------------------------------------------------------------------------
    def accept(self, proposal: Proposal | str, note: str = "") -> Path:
        """Write the proposal to the design file (backup kept) and adopt its parameters."""
        prop = self.proposals[proposal] if isinstance(proposal, str) else proposal
        if not prop.ok:
            raise ValueError(f"proposal {prop.id} did not validate; it cannot be accepted")
        if prop.kind == "source":
            n = 1
            while (bak := self.path.with_name(f"{self.path.name}.{n}.bak")).exists():
                n += 1
            shutil.copy2(self.path, bak)
            self.path.write_text(prop.source)
        self.parameters.update(prop.parameters)
        prop.status = "accepted"
        self._record("accept", prop, note=note)
        return self.path

    def reject(self, proposal: Proposal | str, reason: str = "") -> None:
        prop = self.proposals[proposal] if isinstance(proposal, str) else proposal
        prop.status = "rejected"
        self.history.append({"user": f"(rejected proposal {prop.id}: {reason})", "assistant": '{"kind": "answer"}'})
        self._record("reject", prop, note=reason)

    def geometry(self, proposal: Proposal | str):
        """The proposed geometry (for display) without accepting anything: built in the sandbox, read back as STEP."""
        from vegeta.dedalus.geometry import Geometry

        prop = self.proposals[proposal] if isinstance(proposal, str) else proposal
        with tempfile.TemporaryDirectory(prefix="fidia-copilot-") as tmp:
            step = Path(tmp) / "proposal.step"
            ex = self._run(prop.source if prop.kind == "source" else self.path.read_text(), strict=prop.parameters,
                           optional=self.parameters, keep_step=step)
            if not ex.ok:
                raise RuntimeError(f"proposal {prop.id} does not build: {ex.status}: {ex.error}")
            geometry = Geometry.from_step(step)
        geometry.parameters = dict(ex.parameters)
        geometry.name = f"proposal {prop.id}"
        return geometry

    def _record(self, event: str, prop: Proposal, note: str = "") -> None:
        rec = {"event": event, "at": utc_now(), "note": note, "proposer": self.proposer.describe(),
               "proposal": prop.to_dict()}
        with open(self.log, "a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
