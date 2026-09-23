"""Result objects shared in *shape* (not code) by all Vegeta engineering packages.

This module is intentionally copied into each package; see docs/result-shape.md.
"""
from __future__ import annotations

import html
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SUCCESS = "success"
FAILED = "failed"
CANCELLED = "cancelled"


@dataclass
class CommandRecord:
    """Everything needed to understand and reproduce one external command."""

    command: list[str]
    cwd: str
    returncode: int | None
    duration_s: float
    stdout: str = ""
    stderr: str = ""
    started_at: str = ""
    error: str | None = None
    log_file: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.returncode == 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResultError(RuntimeError):
    """Raised by ``Result.raise_for_status`` for failed results."""


@dataclass
class Result:
    kind: str
    status: str = SUCCESS
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Path] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    execution: list[CommandRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == SUCCESS

    def fail(self, message: str, status: str = FAILED) -> "Result":
        self.status = status
        self.messages.append(message)
        return self

    def raise_for_status(self) -> "Result":
        if not self.ok:
            raise ResultError(f"{self.kind} {self.status}: " + "; ".join(self.messages))
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "status": self.status,
            "metrics": _jsonable(self.metrics),
            "artifacts": {k: str(v) for k, v in self.artifacts.items()},
            "messages": list(self.messages),
            "duration_s": self.duration_s,
            "execution": [c.to_dict() for c in self.execution],
            "metadata": _jsonable(self.metadata),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())
        return path

    def summary_lines(self) -> list[str]:
        lines = [f"{self.kind}: {self.status.upper()} ({self.duration_s:.2f} s)"]
        for k, v in self.metrics.items():
            lines.append(f"  {k:<32} {_fmt(v)}")
        if self.artifacts:
            lines.append("  artifacts:")
            lines += [f"    {k:<30} {v}" for k, v in self.artifacts.items()]
        if self.messages:
            lines.append("  messages:")
            lines += [f"    - {m}" for m in self.messages]
        return lines

    def __str__(self) -> str:
        return "\n".join(self.summary_lines())

    def _repr_html_(self) -> str:
        colour = {"success": "#2e7d32", "failed": "#c62828"}.get(self.status, "#ef6c00")
        rows = "".join(
            f"<tr><td>{html.escape(k)}</td><td>{html.escape(_fmt(v))}</td></tr>" for k, v in self.metrics.items()
        )
        arts = "".join(
            f"<tr><td>{html.escape(k)}</td><td><code>{html.escape(str(v))}</code></td></tr>"
            for k, v in self.artifacts.items()
        )
        msgs = "".join(f"<li>{html.escape(m)}</li>" for m in self.messages)
        return (
            f"<div><b>{html.escape(self.kind)}</b> "
            f"<span style='color:{colour};font-weight:bold'>{self.status.upper()}</span> "
            f"({self.duration_s:.2f} s)"
            + (f"<table><tr><th>metric</th><th>value</th></tr>{rows}</table>" if rows else "")
            + (f"<table><tr><th>artifact</th><th>path</th></tr>{arts}</table>" if arts else "")
            + (f"<ul>{msgs}</ul>" if msgs else "")
            + "</div>"
        )


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _jsonable(obj: Any) -> Any:
    """Convert numpy scalars/arrays, paths and tuples into JSON-friendly values."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "tolist"):  # numpy arrays and scalars
        return _jsonable(obj.tolist())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)
