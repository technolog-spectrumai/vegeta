"""The CAD as MuJoCo meshes: each part of Pinna, Gannet, Fulmar and Walrus tessellated from its CadQuery design and
written as a binary STL (mm, as the CAD; the MJCF scales by 1e-3), cached under ``labs/marine/sims/output`` (ignored by
git) by a hash of the design's parameters and source, so the CAD is built once per change.

    paths = meshes.export_all()          # {"pinna_hulls": Path(...stl), ..., "gannet_body": ..., "walrus": ...}
    meshes.bbox("fulmar")                # the tessellation's bounding box (mm)"""
from __future__ import annotations

import hashlib
import inspect
import os
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CACHE = Path(os.environ.get("PINNA_LAUNCH_CACHE", ROOT / "labs" / "marine" / "sims" / "output" / "pinna_launch" / "meshes"))
for _d in (ROOT / "labs" / "marine" / "designs", ROOT / "labs" / "aero" / "designs"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

#: Pinna's parts drawn (the davits are drawn by the MJCF, one of them moves) and their colours.
PINNA_PARTS = {"hulls": (0.20, 0.29, 0.37, 1), "deck": (0.58, 0.65, 0.65, 1), "house": (0.92, 0.94, 0.95, 1),
               "flight_deck": (0.45, 0.50, 0.52, 1), "pads": (0.95, 0.77, 0.06, 1), "catapult": (0.75, 0.22, 0.17, 1),
               "net": (0.17, 0.24, 0.31, 0.35), "mast": (0.82, 0.83, 0.84, 1), "robots": (0.15, 0.68, 0.38, 1)}
GANNET_PARTS = {"body": (0.12, 0.38, 0.55, 1), "arms": (0.15, 0.15, 0.17, 1), "rotors": (0.75, 0.80, 0.85, 0.35),
                "skids": (0.2, 0.2, 0.22, 1), "crate": (0.85, 0.55, 0.25, 1)}
FULMAR_PARTS = {"aircraft": (0.95, 0.61, 0.07, 1)}
WALRUS_PARTS = {"tracks": (0.15, 0.15, 0.15, 1), "frame": (0.85, 0.85, 0.80, 1), "housing": (0.70, 0.72, 0.75, 1),
                "gantry": (0.30, 0.30, 0.35, 1), "basket": (0.20, 0.55, 0.30, 1), "thrusters": (0.95, 0.75, 0.10, 1),
                "lights": (0.95, 0.95, 0.70, 1), "arms": (0.55, 0.35, 0.20, 1)}
SQUILLA_PARTS = {"body": (0.95, 0.80, 0.15, 1), "arms": (0.25, 0.25, 0.28, 1), "skids": (0.2, 0.2, 0.22, 1), "sensors": (0.1, 0.1, 0.12, 1)}
JACKET_PARTS = {"structure": (0.80, 0.62, 0.20, 1), "anodes": (0.75, 0.75, 0.78, 1), "conductors": (0.5, 0.5, 0.52, 1),
                "risers": (0.25, 0.25, 0.28, 1), "topside": (0.86, 0.86, 0.84, 1)}
TOL = {"pinna": 30.0, "gannet": 6.0, "fulmar": 6.0, "walrus": 4.0, "squilla": 5.0, "jacket": 120.0}


def _designs():
    from fulmar import Fulmar
    from gannet import Gannet
    from oil_platform import Jacket
    from pinna import Pinna
    from squilla import Squilla
    from walrus import Walrus
    return {"pinna": (Pinna, PINNA_PARTS), "gannet": (Gannet, GANNET_PARTS), "fulmar": (Fulmar, FULMAR_PARTS), "walrus": (Walrus, WALRUS_PARTS),
            "squilla": (Squilla, SQUILLA_PARTS), "jacket": (Jacket, JACKET_PARTS)}


def write_stl(path: Path, verts: np.ndarray, tris: np.ndarray) -> Path:
    """A binary STL of the triangles (normals from the winding)."""
    v = verts[tris]
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(tris), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = n, v
    with open(path, "wb") as f:
        f.write(b"pinna_launch mesh".ljust(80, b" "))
        f.write(struct.pack("<I", len(tris)))
        f.write(rec.tobytes())
    return path


def tessellate(shape, tol: float):
    verts, tris = shape.val().tessellate(tol, 0.5)
    return np.array([(q.x, q.y, q.z) for q in verts], dtype=float), np.array(tris, dtype=np.int64).reshape(-1, 3)


def _key(cls) -> str:
    src = inspect.getsource(sys.modules[cls.__module__])
    return hashlib.sha256((src + repr(sorted(cls.p().items()))).encode()).hexdigest()[:12]


def export(design: str, cache: Path = CACHE) -> dict:
    """The parts of one design as STL files: {f"{design}_{part}": path}."""
    cls, parts = _designs()[design]
    d = Path(cache) / f"{design}_{_key(cls)}"
    out = {f"{design}_{part}": d / f"{part}.stl" for part in parts}
    if all(p.exists() for p in out.values()):
        return out
    d.mkdir(parents=True, exist_ok=True)
    obj, p = cls(), cls.p()
    for part, path in out.items():
        v, t = tessellate(obj.build({**p, "part": path.stem}), TOL[design])
        write_stl(path, v, t)
    return out


def export_all(cache: Path = CACHE) -> dict:
    out = {}
    for name in ("pinna", "gannet", "fulmar", "walrus"):
        out.update(export(name, cache))
    return out


def colours() -> dict:
    return {f"{d}_{k}": c for d, (_, parts) in _designs().items() for k, c in parts.items()}


def read_stl(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        f.seek(80)
        n = struct.unpack("<I", f.read(4))[0]
        rec = np.frombuffer(f.read(), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")], count=n)
    return rec["v"].reshape(-1, 3)


def bbox(design: str) -> tuple:
    v = np.vstack([read_stl(p) for p in export(design).values()])
    return v.min(axis=0), v.max(axis=0)


if __name__ == "__main__":
    for k, v in export_all().items():
        print(k, v, v.stat().st_size // 1024, "kB")
    for d in ("pinna", "gannet", "fulmar", "walrus"):
        print(d, *bbox(d))
