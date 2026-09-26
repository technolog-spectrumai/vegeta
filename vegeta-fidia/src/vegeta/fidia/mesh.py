"""Parts as triangle meshes: what the runner wrote (``result.json`` + ``parts.npz``), loaded without pickles."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Part:
    """One named, coloured part: vertices in mm (Z up), triangles, and the runner's B-rep measurements."""

    name: str
    color: tuple[float, float, float, float]
    vertices: np.ndarray
    triangles: np.ndarray
    brep: dict = field(default_factory=dict)
    source_name: str = ""
    color_given: bool = True

    @property
    def rgba8(self) -> list[int]:
        return [int(round(255 * min(max(c, 0.0), 1.0))) for c in self.color]

    @property
    def bounds(self) -> np.ndarray:
        return np.array([self.vertices.min(0), self.vertices.max(0)]) if len(self.vertices) else np.zeros((2, 3))

    def to_trimesh(self, merge: bool = True):
        """A trimesh with seam vertices merged (CadQuery repeats them per face; without merging nothing is closed)."""
        import trimesh

        m = trimesh.Trimesh(self.vertices, self.triangles, process=merge)
        if merge:
            m.merge_vertices()
        return m


def load_parts(outdir: str | Path) -> list[Part]:
    """Parts from a runner output directory (``result.json`` + ``parts.npz``, loaded with ``allow_pickle=False``)."""
    out = Path(outdir)
    meta = json.loads((out / "result.json").read_text())
    if meta.get("status") != "ok":
        raise ValueError(f"the run in {out} did not succeed: {meta.get('status')}: {meta.get('error')}")
    with np.load(out / "parts.npz", allow_pickle=False) as arrays:
        return [Part(name=p["name"], color=tuple(p["color"]), vertices=np.asarray(arrays[f"v{p['index']}"], dtype=float),
                     triangles=np.asarray(arrays[f"t{p['index']}"], dtype=np.int64), brep=p.get("brep", {}),
                     source_name=p.get("source_name", p["name"]), color_given=p.get("color_given", True))
                for p in meta["parts"]]


def bounds(parts: list[Part]) -> np.ndarray:
    """[[xmin, ymin, zmin], [xmax, ymax, zmax]] over all parts, in mm."""
    b = np.array([p.bounds for p in parts if len(p.vertices)])
    return np.array([b[:, 0].min(0), b[:, 1].max(0)]) if len(b) else np.zeros((2, 3))
