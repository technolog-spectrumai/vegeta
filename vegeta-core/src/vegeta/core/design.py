"""Designs registered in a workspace (a Dedalus design referenced by its source spec)."""
from __future__ import annotations

from pathlib import Path


class Design:
    def __init__(self, workspace, name: str, record: dict):
        self.workspace = workspace
        self.name = name
        self.record = record

    @property
    def source(self) -> str:
        return self.record["source"]

    def resolved_source(self) -> str:
        """Source spec with file paths resolved against the workspace root."""
        target, sep, attr = self.source.partition(":")
        if target.endswith(".py"):
            target = str((self.workspace.root / target).resolve())
        return target + (sep + attr if sep else "")

    def load(self):
        """The Dedalus design object, loaded the same way ``vegeta dedalus`` loads it."""
        from vegeta.dedalus.loading import load_design

        return load_design(self.resolved_source())

    def new_revision(self, note: str = "", **parameters):
        """Record a new revision from the design defaults plus ``parameters``. Nothing is generated."""
        from .revision import create_revision

        return create_revision(self.workspace, self, parent=None, overrides=parameters, note=note)

    def revisions(self) -> list:
        return self.workspace.revisions(design=self.name)

    def __repr__(self) -> str:
        return f"<Design {self.name} ({self.source})>"
