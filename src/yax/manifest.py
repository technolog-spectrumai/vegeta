"""The manifest ``yax.json``: its identity and its validation. Specification: docs/spec.md."""
from __future__ import annotations

import hashlib
import json
import math

VERSION = 1
NAME = "yax.json"
BODY = "body.stl"
LENGTH_UNITS = ("mm", "m")
AXES = ("+x", "-x", "+y", "-y", "+z", "-z")


def canonical(value) -> str:
    """Sorted keys, no whitespace, UTF-8 as is, no NaN — the canonical JSON form also used by Porunga."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def identity(producer: dict) -> str:
    """SHA-256 of what was designed: the producer's tool, design class and parameters (not its version or revision,
    so the same design exported by two revisions of the tool has the same identity)."""
    what = {k: producer.get(k) for k in ("tool", "design_class", "parameters")}
    return hashlib.sha256(canonical(what).encode("utf-8")).hexdigest()


def _vec(v, n=3):
    return isinstance(v, list) and len(v) == n and all(isinstance(x, (int, float)) and math.isfinite(x) for x in v)


def validate(m: dict) -> list:
    """Problems with a manifest, as sentences; ``[]`` when it is valid. Unknown fields are allowed and kept."""
    p = []
    if m.get("yax") != VERSION:
        p.append(f"yax: expected {VERSION}, got {m.get('yax')!r}")
    if not isinstance(m.get("name"), str) or not m.get("name"):
        p.append("name: a non-empty string is required")
    units = m.get("units", {})
    if units.get("length") not in LENGTH_UNITS:
        p.append(f"units.length: one of {LENGTH_UNITS} is required")
    if units.get("mass", "kg") != "kg":
        p.append("units.mass: must be kg")
    frame = m.get("frame", {})
    up, fwd = frame.get("up"), frame.get("forward")
    if up not in AXES or fwd not in AXES or up[1] == fwd[1]:
        p.append(f"frame: up and forward must be two perpendicular axes of {AXES}, got {up!r}, {fwd!r}")
    files = m.get("files", {})
    if BODY not in files:
        p.append(f"files: {BODY} is required")
    for name, digest in files.items():
        if name.startswith("/") or ".." in name.split("/"):
            p.append(f"files: {name!r} must be a relative path inside the package")
        if not (isinstance(digest, str) and len(digest) == 64):
            p.append(f"files: {name!r} needs a SHA-256 hex digest")
    g = m.get("geometry", {})
    if not (isinstance(g.get("bbox"), list) and len(g["bbox"]) == 2 and all(_vec(c) for c in g["bbox"])):
        p.append("geometry.bbox: [[xmin, ymin, zmin], [xmax, ymax, zmax]] is required")
    if not (isinstance(g.get("volume"), (int, float)) and g["volume"] > 0):
        p.append("geometry.volume: a positive number is required")
    if not _vec(g.get("com")):
        p.append("geometry.com: [x, y, z] is required")
    mass = m.get("mass")
    if mass is not None:
        if not (isinstance(mass.get("kg"), (int, float)) and mass["kg"] > 0):
            p.append("mass.kg: a positive number is required")
        if not _vec(mass.get("com")):
            p.append("mass.com: [x, y, z] is required")
        inertia = mass.get("inertia")
        if not (isinstance(inertia, list) and len(inertia) == 3 and all(_vec(r) for r in inertia)):
            p.append("mass.inertia: a 3x3 matrix (kg m^2, about mass.com) is required")
        elif any(abs(inertia[i][j] - inertia[j][i]) > 1e-12 * max(1.0, abs(inertia[i][j])) for i in range(3)
                 for j in range(3)):
            p.append("mass.inertia: must be symmetric")
        if mass.get("source") not in ("density", "given"):
            p.append("mass.source: 'density' or 'given'")
    for part in m.get("parts", []):
        if part.get("file") not in files:
            p.append(f"parts: {part.get('name')!r} names a file not listed in files")
    producer = m.get("producer")
    if producer is not None and m.get("identity") != identity(producer):
        p.append("identity: does not match the producer's tool, design_class and parameters")
    return p
