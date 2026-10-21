"""A workspace: one directory holding designs, revisions and evaluations as plain files."""
from __future__ import annotations

import os
import re
from pathlib import Path

from ._io import environment, read_json, utc_now, write_once

SCHEMA = 1
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_RID = re.compile(r"^r(\d+)$")


def check_name(kind: str, name: str) -> str:
    if not _NAME.match(name or ""):
        raise ValueError(f"{kind} name {name!r} must start with a letter and use letters, digits, '_' or '-'")
    return name


class Workspace:
    """Open an existing workspace directory (see :meth:`create`)."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        meta = self.root / "workspace.json"
        if not meta.is_file():
            raise FileNotFoundError(f"{self.root} is not a Vegeta workspace (no workspace.json); use Workspace.create")
        self.record = read_json(meta)
        if self.record.get("schema") != SCHEMA:
            raise ValueError(f"unsupported workspace schema {self.record.get('schema')!r} (expected {SCHEMA})")

    @classmethod
    def create(cls, root: str | Path, name: str | None = None, description: str = "") -> "Workspace":
        root = Path(root).resolve()
        if (root / "workspace.json").exists():
            raise FileExistsError(f"{root} already contains a workspace; use Workspace.open")
        root.mkdir(parents=True, exist_ok=True)
        for sub in ("designs", "revisions"):
            (root / sub).mkdir(exist_ok=True)
        write_once(root / "workspace.json", {
            "schema": SCHEMA, "name": name or root.name, "description": description,
            "created_at": utc_now(), "environment": environment(),
        })
        return cls(root)

    @classmethod
    def open(cls, root: str | Path) -> "Workspace":
        return cls(root)

    @property
    def name(self) -> str:
        return self.record["name"]

    # -- designs ----------------------------------------------------------------------------
    def add_design(self, name: str, source: str, description: str = ""):
        """Register a Dedalus design by spec: ``"package.module:Name"`` or ``"path/file.py:Name"``.

        File paths are stored relative to the workspace so the workspace can move.
        """
        from .design import Design

        check_name("design", name)
        target, sep, attr = source.partition(":")
        if target.endswith(".py"):
            p = Path(target).resolve()
            if not p.is_file():
                raise FileNotFoundError(f"design file not found: {target}")
            source = os.path.relpath(p, self.root) + (sep + attr if sep else "")
        design = Design(self, name, {"name": name, "source": source, "description": description,
                                     "created_at": utc_now()})
        dd = design.load()  # validate now: must load and expose parameters
        design.record["parameters"] = dd.params.table()
        design.record["units"] = dd.units
        if (self.root / "designs" / name / "design.json").exists():
            raise FileExistsError(f"design {name!r} already exists in this workspace")
        write_once(self.root / "designs" / name / "design.json", design.record)
        return design

    def design(self, name: str):
        from .design import Design

        path = self.root / "designs" / name / "design.json"
        if not path.is_file():
            raise KeyError(f"no design {name!r}; known: {[d.name for d in self.designs()]}")
        return Design(self, name, read_json(path))

    def designs(self) -> list:
        return [self.design(p.name) for p in sorted((self.root / "designs").iterdir()) if (p / "design.json").is_file()]

    # -- revisions --------------------------------------------------------------------------
    def revision(self, rid: str):
        from .revision import Revision

        path = self.root / "revisions" / rid / "revision.json"
        if not path.is_file():
            raise KeyError(f"no revision {rid!r} in {self.root}")
        return Revision(self, rid, read_json(path))

    def revisions(self, design: str | None = None) -> list:
        out = []
        for p in (self.root / "revisions").iterdir():
            m = _RID.match(p.name)
            if m and (p / "revision.json").is_file():
                out.append((int(m.group(1)), p.name))
        revs = [self.revision(rid) for _, rid in sorted(out)]
        return [r for r in revs if design is None or r.design_name == design]

    def _allocate_revision_dir(self) -> tuple[str, Path]:
        existing = [int(m.group(1)) for p in (self.root / "revisions").iterdir() if (m := _RID.match(p.name))]
        n = max(existing, default=0) + 1
        while True:
            rid = f"r{n}"
            path = self.root / "revisions" / rid
            try:
                path.mkdir()
                return rid, path
            except FileExistsError:
                n += 1

    # -- overview ---------------------------------------------------------------------------
    def status(self, design: str | None = None):
        from .status import status_table

        return status_table(self, design)

    def __repr__(self) -> str:
        return f"<Workspace {self.name} at {self.root}>"
