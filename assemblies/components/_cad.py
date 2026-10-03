"""CAD files that stay the same while their parameters do.

STEP files carry a time stamp: writing a new one for unchanged parameters changes its hash, and with it every FEA mesh
key made from it, so those models would be solved again. ``export_kept`` writes the files once per parameter set and
keeps them (``<basename>.params.json`` holds the parameters' digest and a few numbers of the part).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from ..vida import digest


def export_kept(make: Callable[[], object], params: dict, folder: Path, basename: str, *, formats=("step", "stl"),
                stl_tolerance: float = 0.02) -> tuple[dict[str, Path], dict]:
    """``make()`` builds the Dedalus geometry (called only when the files are missing or the parameters changed).
    Returns ``({format: path}, info)`` with ``info`` the part's volume, surface area and dimensions [mm]."""
    folder = Path(folder)
    stamp = folder / f"{basename}.params.json"
    want = digest({"params": params, "formats": list(formats), "stl_tolerance": stl_tolerance})
    paths = {f: folder / f"{basename}.{f}" for f in formats}
    if stamp.is_file():
        kept = json.loads(stamp.read_text())
        if kept.get("params") == want and all(p.is_file() for p in paths.values()):
            return paths, kept["info"]
    folder.mkdir(parents=True, exist_ok=True)
    g = make()
    res = g.export(folder, formats=tuple(formats), basename=basename, stl_tolerance=stl_tolerance)
    res.raise_for_status()
    info = {"volume_mm3": float(g.volume), "surface_area_mm2": float(g.surface_area),
            "dimensions_mm": [float(x) for x in g.dimensions], "valid": bool(g.shape.isValid())}
    stamp.write_text(json.dumps({"params": want, "info": info}) + "\n")
    return {f: Path(res.artifacts[f]) for f in formats}, info
