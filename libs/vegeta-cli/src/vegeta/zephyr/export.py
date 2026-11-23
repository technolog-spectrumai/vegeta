"""Polars to and from one JSON file (the hand-off to a wing design or a notebook)."""
from __future__ import annotations

import json
from pathlib import Path

from .polar import Polar


def save_polars(path: str | Path, polars, notes: str = "") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"kind": "zephyr.polars", "notes": notes, "polars": [p.to_dict() for p in polars]}
    path.write_text(json.dumps(doc, indent=1))
    return path


def load_polars(path: str | Path) -> list[Polar]:
    doc = json.loads(Path(path).read_text())
    if doc.get("kind") != "zephyr.polars":
        raise ValueError(f"{path} is not a zephyr polars file")
    return [Polar.from_dict(d) for d in doc["polars"]]
