"""STL files for the ARIGO CFD cases, named by their geometry.

The first scripts cached each STL by aircraft and angle only, so a changed design (the goshawk's longer tail)
silently reused the old STL — and its CFD was the old aircraft's. Now each export is compared with the cached file:
identical bytes (CadQuery's export is deterministic) keep the old name, so solved cases stay valid; any difference
gets a name with the geometry's hash, and the case folder carries the same tag, so OpenFOAM solves the new shape.
"""
from __future__ import annotations

import filecmp
import hashlib
from pathlib import Path


def stl(folder, stem: str, alpha_deg: float, build) -> tuple[Path, str]:
    """``build(path)`` writes the STL. Returns (the STL to use, the tag for the case's name: "" or "_g<hash>")."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    old = folder / f"{stem}_a{alpha_deg:+.0f}.stl"
    tmp = folder / f".{stem}_a{alpha_deg:+.0f}.tmp.stl"
    build(tmp)
    if old.exists() and filecmp.cmp(old, tmp, shallow=False):
        tmp.unlink()
        return old, ""
    h = hashlib.sha1(tmp.read_bytes()).hexdigest()[:8]
    new = folder / f"{stem}_g{h}_a{alpha_deg:+.0f}.stl"
    if new.exists():
        tmp.unlink()
    else:
        tmp.replace(new)
    return new, f"_g{h}"
