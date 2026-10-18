"""runs — reading workflow run directories through runkit's **file contract** (schema 1): ``manifest.json``,
``status.json``, ``summary.json`` / ``summary.partial.json``. Nothing of runkit is imported."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA = 1


def read_json(p: Path) -> dict | None:
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def _alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


@dataclass
class RunView:
    root: Path
    manifest: dict | None = None
    status: dict | None = None
    summary: dict | None = None
    partial: dict | None = None
    state: str | None = None
    problem: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def variant(self) -> str | None:
        return ((self.manifest or {}).get("config") or {}).get("variant")

    @property
    def workflow(self) -> str | None:
        return (self.manifest or {}).get("workflow")

    @property
    def created_at(self) -> str:
        return (self.manifest or {}).get("created_at") or ""

    @property
    def run_id(self) -> str:
        return (self.manifest or {}).get("run_id") or self.root.name

    def provenance(self) -> str:
        m = self.manifest or {}
        prov = m.get("provenance") or {}
        if prov.get("imported_from"):
            return f"imported from {Path(prov['imported_from']).name} ({prov.get('notebook')}, {prov.get('date')})"
        tv = m.get("tool_versions") or {}
        return f"run {self.run_id} on {m.get('host', '?')}, {self.created_at[:10]} (git {tv.get('git')}{' dirty' if tv.get('git_dirty') else ''})"


def read_run(root) -> RunView:
    """The run directory as it is on disk; ``state`` follows runkit, with a ``running`` run whose process is gone
    read as ``failed`` (it did not finish)."""
    root = Path(root)
    v = RunView(root)
    if not root.is_dir():
        v.problem = f"not found: {root}"
        return v
    v.manifest = read_json(root / "manifest.json")
    if not isinstance(v.manifest, dict):
        v.manifest = None
        v.problem = f"{root}: manifest.json is not readable JSON" if (root / "manifest.json").exists() else f"not found: {root} has no manifest.json"
        return v
    if v.manifest.get("schema") != SCHEMA:
        v.problem = f"run directory schema {v.manifest.get('schema')!r}, this Yanus reads schema {SCHEMA}"
        return v
    v.status = read_json(root / "status.json")
    if not isinstance(v.status, dict):
        v.status = {}
    v.summary = read_json(root / "summary.json")
    if not isinstance(v.summary, dict):
        v.summary = None
    v.partial = read_json(root / "summary.partial.json")
    v.state = v.status.get("state")
    if v.state == "running" and not _alive(v.status.get("pid")):
        v.state = "failed"
        v.extra["dead"] = True
    return v


def runs_of(runs_root, book: str, chapter: str | None = None, *, variant: str | None = None) -> list[RunView]:
    """The runs of a book (and chapter) under ``<runs_root>/<book>/<chapter>/<run id>/``, newest first."""
    base = Path(runs_root) / book
    out = []
    if not base.is_dir():
        return out
    for ch in sorted(base.iterdir()):
        if not ch.is_dir() or (chapter and ch.name != chapter):
            continue
        for d in sorted(ch.iterdir()):
            if d.is_dir() and (d / "manifest.json").exists():
                r = read_run(d)
                r.extra["chapter"] = ch.name
                if variant and r.variant != variant and not r.problem:     # a broken run stays listed, with its problem
                    continue
                out.append(r)
    return sorted(out, key=lambda r: r.created_at, reverse=True)
