"""The design copilot: iterate on one Dedalus design file with an AI proposer, under the engineer's control."""
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


class Proposer(Protocol):
    name: str

    def propose(self, context: dict, instruction: str, history: list[dict]) -> tuple[dict, dict]: ...

    def describe(self) -> dict: ...


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DesignSession:
    """Iterate on ``design_spec`` (``path/file.py:Name``) with a proposer.

    Every ``ask()`` produces a Proposal that Vegeta builds and measures in a temporary copy; the design
    file changes only in ``accept()``. All proposals and decisions are appended to
    ``<design dir>/<file>.ai.jsonl`` and the previous file content is kept as ``<file>.<n>.bak``.
    """

    def __init__(self, design_spec: str, proposer: Proposer, *, parameters: dict | None = None,
                 notes: str = "", log_dir: str | Path | None = None):
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
        self._n = 0

    # -- state ------------------------------------------------------------------------------
    def _spec(self, path: Path | None = None) -> str:
        p = path or self.path
        return f"{p}:{self.attr}" if self.attr else str(p)

    def design(self, path: Path | None = None):
        return load_design(self._spec(path))

    def context(self) -> dict:
        dd = self.design()
        values = dd.resolve(**self.parameters)
        params = []
        for p in dd.params:
            row = {"name": p.name, "value": values[p.name], "units": p.units, "min": p.min, "max": p.max,
                   "description": p.description}
            params.append(row)
        measurements = None
        try:
            measurements = dd.generate(**self.parameters).measure()
        except Exception as exc:  # the current design may itself be broken; say so instead of hiding it
            measurements = {"error": f"current design does not build: {exc!r}"}
        return {"path": str(self.path), "source": self.path.read_text(), "parameters": params,
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
        """Build the proposal in a temporary copy and compare measurements. Runs the proposed code."""
        context = context or self.context()
        before = {p["name"]: p["value"] for p in context["parameters"]}
        v = Validation(ok=True, parameters_before=before, measurements_before=context.get("measurements"))
        if prop.kind == "answer":
            v.ok = True
            v.messages.append("no change proposed")
            v.parameters_after = before
            return v
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / self.path.name
                path.write_text(prop.source if prop.kind == "source" else context["source"])
                dd = self.design(path)
                names = [p.name for p in dd.params]
                v.parameters_added = [n for n in names if n not in before]
                v.parameters_removed = [n for n in before if n not in names]
                kept = {k: val for k, val in self.parameters.items() if k in names}
                values = dd.resolve(**{**kept, **prop.parameters})
                v.parameters_after = values
                v.measurements_after = dd.generate(**values).measure()
                if not v.measurements_after.get("valid"):
                    v.ok = False
                    v.messages.append("proposed geometry is not a valid solid")
        except Exception as exc:
            v.ok = False
            v.messages.append(f"{type(exc).__name__}: {exc}")
            v.messages.append(traceback.format_exc().strip().splitlines()[-1])
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
        """The proposed geometry (for display) without accepting anything."""
        prop = self.proposals[proposal] if isinstance(proposal, str) else proposal
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / self.path.name
            path.write_text(prop.source if prop.kind == "source" else self.path.read_text())
            dd = self.design(path)
            return dd.generate(**dd.resolve(**{**self.parameters, **prop.parameters}))

    def _record(self, event: str, prop: Proposal, note: str = "") -> None:
        rec = {"event": event, "at": utc_now(), "note": note, "proposer": self.proposer.describe(),
               "proposal": prop.to_dict()}
        with open(self.log, "a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
