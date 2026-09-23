"""Evaluation records: one analysis (FEA, CFD, print) run on one revision's geometry."""
from __future__ import annotations

import html
import inspect
from pathlib import Path
from typing import Any

from ._io import read_json

KINDS = ("fea", "cfd", "print")


class Evaluation:
    """A recorded (or refused) analysis. ``tool_results`` holds the tools' own result objects when the
    evaluation was just run in this Python session; the record on disk is ``evaluation.json``."""

    def __init__(self, kind: str, name: str, status: str, metrics: dict | None = None,
                 messages: list[str] | None = None, directory: Path | None = None, record: dict | None = None,
                 tool_results: list | None = None):
        self.kind = kind
        self.name = name
        self.status = status
        self.metrics = metrics or {}
        self.messages = messages or []
        self.directory = directory
        self.record = record or {}
        self.tool_results = tool_results or []

    @property
    def ok(self) -> bool:
        return self.status == "success"

    @property
    def recorded(self) -> bool:
        return self.directory is not None and (self.directory / "evaluation.json").is_file()

    @classmethod
    def load(cls, directory: Path) -> "Evaluation":
        rec = read_json(directory / "evaluation.json")
        return cls(rec["kind"], rec["name"], rec["status"], rec.get("metrics"), rec.get("messages"),
                   directory, rec)

    def __repr__(self) -> str:
        return f"<Evaluation {self.kind}:{self.name} {self.status}>"

    def __str__(self) -> str:
        lines = [f"{self.kind}:{self.name} {self.status.upper()}"]
        lines += [f"  {k:<28} {v}" for k, v in self.metrics.items() if not isinstance(v, (dict, list))]
        lines += [f"  - {m}" for m in self.messages]
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        colour = {"success": "#2e7d32", "failed": "#c62828"}.get(self.status, "#ef6c00")
        rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{html.escape(str(v))}</td></tr>"
                       for k, v in self.metrics.items() if not isinstance(v, (dict, list)))
        msgs = "".join(f"<li>{html.escape(m)}</li>" for m in self.messages)
        return (f"<div><b>{self.kind}:{html.escape(self.name)}</b> "
                f"<span style='color:{colour};font-weight:bold'>{self.status.upper()}</span>"
                f"<table>{rows}</table><ul>{msgs}</ul></div>")


def factory_record(fn: Any) -> dict:
    """Identity and source text of the Python callable that built an analysis configuration."""
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        src = None
    return {"module": getattr(fn, "__module__", None), "qualname": getattr(fn, "__qualname__", None), "source": src}
