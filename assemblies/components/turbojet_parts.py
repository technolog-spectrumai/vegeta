"""The microjet's three parts as STEP files (impeller, turbine wheel, engine outside) with their masses and sizes."""
from __future__ import annotations

import json
from pathlib import Path

from ..vida import Assembly, digest
from . import turbojet as tj

PARTS = ("impeller", "turbine", "engine")


def assembly(P: dict) -> Assembly:
    return Assembly("parts", "turbojet_parts", params={"geometry": P})


def make(node: Assembly, out: Path) -> dict[str, Path]:
    """The STEP files in ``out/<part>/`` and, on ``node``, the masses and each part's size and validity.

    A STEP made from the same parameters is kept (``<part>.params.json`` holds their digest and the part's size): STEP
    files carry a time stamp, so a new file would change every FEA mesh key made from it and solve those again."""
    out = Path(out)
    P = node.params["geometry"]
    paths, sizes, made = {}, {}, False
    for part in PARTS:
        folder = out / part
        stamp = folder / f"{part}.params.json"
        want = digest({"geometry": P, "part": part})
        steps = sorted(folder.glob("*.step")) if folder.is_dir() else []
        kept = json.loads(stamp.read_text()) if stamp.is_file() else {}
        if len(steps) == 1 and kept.get("params") == want:
            paths[part], sizes[part] = steps[0], kept["size"]
            continue
        for old in steps:
            old.unlink()
        g = tj.Turbojet().generate(**dict(P, part=part))
        res = g.export(folder, formats=("step",), basename=part)
        res.raise_for_status()
        sizes[part] = {"valid": bool(g.shape.isValid()), "solids": len(g.shape.Solids()),
                       "dimensions_mm": [float(x) for x in g.dimensions]}
        stamp.write_text(json.dumps({"params": want, "size": sizes[part]}) + "\n")
        made = True
        paths[part] = Path(res.artifacts["step"])
    for part, path in paths.items():
        node.attach(f"{part}.step", path, "geometry")
    if made or not node.results:
        node.record(size=sizes, masses_kg=tj.masses(P))
    return paths
