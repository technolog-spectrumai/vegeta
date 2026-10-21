"""Minimal STL reading/writing (ASCII and binary) with numpy."""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LENGTH_TO_METRES = {"m": 1.0, "mm": 1e-3, "cm": 1e-2, "in": 0.0254}


@dataclass
class Surface:
    triangles: np.ndarray  # (M, 3, 3)
    name: str = "body"

    @property
    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        pts = self.triangles.reshape(-1, 3)
        return pts.min(axis=0), pts.max(axis=0)

    @property
    def area(self) -> float:
        t = self.triangles
        return float(0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum())

    @property
    def volume(self) -> float:
        """Signed enclosed volume (meaningful for closed, consistently oriented surfaces)."""
        t = self.triangles
        return float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.0)

    def scaled(self, factor: float) -> "Surface":
        return Surface(self.triangles * factor, self.name)

    def normals(self) -> np.ndarray:
        t = self.triangles
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        norm = np.linalg.norm(n, axis=1, keepdims=True)
        return np.divide(n, norm, out=np.zeros_like(n), where=norm > 0)


def read_stl(path: str | Path) -> Surface:
    path = Path(path)
    data = path.read_bytes()
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if len(data) == 84 + 50 * n:
            rec = np.frombuffer(data[84:], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=n)
            return Surface(rec["v"].astype(float), path.stem)
    text = data.decode("ascii", errors="replace")
    if not text.lstrip().lower().startswith("solid"):
        raise ValueError(f"{path}: not a valid ASCII or binary STL")
    verts = [list(map(float, line.split()[1:4])) for line in text.splitlines() if line.strip().startswith("vertex")]
    if not verts or len(verts) % 3:
        raise ValueError(f"{path}: malformed ASCII STL ({len(verts)} vertices)")
    return Surface(np.array(verts).reshape(-1, 3, 3), path.stem)


def write_stl_ascii(surface: Surface, path: str | Path, solid_name: str | None = None) -> Path:
    path = Path(path)
    name = solid_name or surface.name
    lines = [f"solid {name}"]
    for n, tri in zip(surface.normals(), surface.triangles):
        lines.append(f"  facet normal {n[0]:.9g} {n[1]:.9g} {n[2]:.9g}")
        lines.append("    outer loop")
        lines += [f"      vertex {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}" for v in tri]
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append(f"endsolid {name}")
    path.write_text("\n".join(lines) + "\n")
    return path
