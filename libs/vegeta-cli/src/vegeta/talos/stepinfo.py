"""Inspect any STEP file so the engineer can choose regions explicitly."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from ._gmsh import gmsh_session, import_step
from .units import get_units


@dataclass
class SurfaceInfo:
    tag: int
    kind: str
    area: float
    centroid: tuple
    bbox_min: tuple
    bbox_max: tuple
    normal: tuple | None  # outward-ish normal at the parametric centre, planar surfaces only


@dataclass
class StepInfo:
    path: str
    units: str
    volumes: list[dict]
    surfaces: list[SurfaceInfo]
    bbox: list[float]

    def to_dict(self):
        return {"path": self.path, "units": self.units, "volumes": self.volumes, "bbox": self.bbox,
                "surfaces": [asdict(s) for s in self.surfaces]}

    def table(self) -> str:
        head = f"{'tag':>4}  {'type':<10} {'area':>11}  {'centroid':<30} normal"
        rows = [head]
        for s in self.surfaces:
            c = ", ".join(f"{v:.4g}" for v in s.centroid)
            n = "" if s.normal is None else ", ".join(f"{v:+.3f}" for v in s.normal)
            rows.append(f"{s.tag:>4}  {s.kind:<10} {s.area:>11.5g}  ({c:<28}) {n}")
        return "\n".join(rows)

    def to_dataframe(self):
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise ImportError("to_dataframe needs pandas: pip install 'vegeta-cli[pandas]'") from exc
        return pd.DataFrame([asdict(s) for s in self.surfaces])

    def _repr_html_(self):
        return "<pre>" + self.table() + "</pre>"


def inspect_step(path: str | Path, units: str) -> StepInfo:
    """List volumes and surfaces (Gmsh tags, area, centroid, bbox, normal) of a STEP file."""
    u = get_units(units)
    with gmsh_session() as gmsh:
        info = import_step(gmsh, path, u)
        occ, model = gmsh.model.occ, gmsh.model
        vols = []
        for t in info["volumes"]:
            vols.append({"tag": t, "volume": occ.getMass(3, t), "center_of_mass": list(occ.getCenterOfMass(3, t))})
        surfs = []
        for _, t in model.getEntities(2):
            bb = model.getBoundingBox(2, t)
            kind = model.getType(2, t)
            normal = None
            if kind == "Plane":
                lo, hi = model.getParametrizationBounds(2, t)
                uv = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2]
                normal = tuple(float(v) for v in model.getNormal(t, uv))
            surfs.append(SurfaceInfo(
                tag=t, kind=kind, area=occ.getMass(2, t), centroid=tuple(occ.getCenterOfMass(2, t)),
                bbox_min=tuple(bb[:3]), bbox_max=tuple(bb[3:]), normal=normal,
            ))
    return StepInfo(path=str(path), units=units, volumes=vols, surfaces=surfs, bbox=info["bbox"])
