"""A design to and from one JSON file (the hand-off to a notebook, a build, another tool): trains, resonators, beams,
lasers and plain numbers under named keys."""
from __future__ import annotations

import json
from pathlib import Path

from .beam import GaussianBeam
from .elements import Train
from .resonator import Resonator

KIND = "hyperion.design"


def _encode(v):
    if isinstance(v, (Train, Resonator, GaussianBeam)):
        return {"_type": type(v).__name__, **v.to_dict()}
    if hasattr(v, "to_dict"):
        return {"_type": type(v).__name__, **v.to_dict()}
    if hasattr(v, "tolist"):
        return v.tolist()
    if isinstance(v, dict):
        return {str(k): _encode(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_encode(x) for x in v]
    if isinstance(v, float) and v != v:
        return None
    if isinstance(v, float) and v in (float("inf"), float("-inf")):
        return "inf" if v > 0 else "-inf"
    return v


def _decode(v):
    if isinstance(v, dict):
        t = v.get("_type")
        d = {k: _decode(x) for k, x in v.items() if k != "_type"}
        if t == "Train":
            return Train.from_dict(d)
        if t == "Resonator":
            return Resonator.from_dict(d)
        if t == "GaussianBeam":
            return GaussianBeam(**{k: d[k] for k in ("wavelength", "w0", "z0", "m2", "n", "name")})
        return d
    if isinstance(v, list):
        return [_decode(x) for x in v]
    if v == "inf":
        return float("inf")
    if v == "-inf":
        return float("-inf")
    return v


def save_design(path: str | Path, design: dict, notes: str = "") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"kind": KIND, "notes": notes, "design": _encode(design)}, indent=1))
    return path


def load_design(path: str | Path) -> dict:
    doc = json.loads(Path(path).read_text())
    if doc.get("kind") != KIND:
        raise ValueError(f"{path} is not a hyperion design file")
    return _decode(doc["design"])
