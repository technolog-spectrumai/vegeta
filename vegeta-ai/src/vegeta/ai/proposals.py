"""Typed proposals: what the assistant may suggest. Data only — nothing here executes anything."""
from __future__ import annotations

import difflib
from dataclasses import asdict, dataclass, field
from typing import Any

PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["source", "parameters", "answer"],
                 "description": "source: a new version of the design file; parameters: only parameter values "
                                "change; answer: no change, just an explanation or a question"},
        "summary": {"type": "string", "description": "one sentence: what changes"},
        "rationale": {"type": "string", "description": "why, in engineering terms"},
        "source": {"type": ["string", "null"], "description": "complete new content of the design file "
                                                              "(kind=source), else null"},
        "parameters": {"type": "object", "additionalProperties": {"type": ["number", "string", "boolean"]},
                       "description": "parameter values to use (may be empty)"},
        "expected_effects": {"type": "array", "items": {"type": "string"}},
        "risks": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["kind", "summary", "rationale", "source", "parameters", "expected_effects", "risks"],
    "additionalProperties": False,
}


@dataclass
class Validation:
    """What Vegeta measured when it built the proposal (never the model's opinion)."""

    ok: bool
    messages: list[str] = field(default_factory=list)
    parameters_before: dict[str, Any] = field(default_factory=dict)
    parameters_after: dict[str, Any] = field(default_factory=dict)
    parameters_added: list[str] = field(default_factory=list)
    parameters_removed: list[str] = field(default_factory=list)
    measurements_before: dict[str, Any] | None = None
    measurements_after: dict[str, Any] | None = None

    def changes(self) -> dict[str, tuple[Any, Any]]:
        out = {}
        if self.measurements_before and self.measurements_after:
            for k in ("volume", "surface_area", "dimensions", "n_faces", "n_solids"):
                b, a = self.measurements_before.get(k), self.measurements_after.get(k)
                if b != a:
                    out[k] = (b, a)
        return out


@dataclass
class Proposal:
    id: str
    kind: str
    summary: str
    rationale: str
    instruction: str
    source: str | None
    parameters: dict[str, Any]
    expected_effects: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    source_before: str | None = None
    validation: Validation | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    model: str | None = None
    created_at: str = ""
    status: str = "proposed"  # proposed | accepted | rejected

    @property
    def diff(self) -> str:
        if self.source is None or self.source_before is None:
            return ""
        return "".join(difflib.unified_diff(self.source_before.splitlines(True), self.source.splitlines(True),
                                            "current", f"proposal {self.id}"))

    @property
    def ok(self) -> bool:
        return self.validation is not None and self.validation.ok

    def to_dict(self) -> dict:
        d = asdict(self)
        d["diff"] = self.diff
        return d

    def __str__(self) -> str:
        lines = [f"[{self.id}] {self.kind}: {self.summary}", f"  why: {self.rationale}"]
        if self.parameters:
            lines.append(f"  parameters: {self.parameters}")
        for e in self.expected_effects:
            lines.append(f"  expect: {e}")
        for r in self.risks:
            lines.append(f"  risk:   {r}")
        if self.validation:
            v = self.validation
            lines.append(f"  validation: {'OK' if v.ok else 'FAILED'}" + ("" if v.ok else " — " + "; ".join(v.messages)))
            for k, (b, a) in v.changes().items():
                lines.append(f"    {k}: {b} -> {a}")
            if v.parameters_added or v.parameters_removed:
                lines.append(f"    parameters added {v.parameters_added}, removed {v.parameters_removed}")
        if self.diff:
            lines.append("  diff:")
            lines += ["    " + l.rstrip("\n") for l in self.diff.splitlines()[:80]]
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        import html

        colour = "#2e7d32" if self.ok else ("#c62828" if self.validation else "#ef6c00")
        body = html.escape(str(self))
        return f"<pre style='border-left:4px solid {colour};padding-left:8px'>{body}</pre>"
