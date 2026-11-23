"""Generated geometry: measurements, export and inspection."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .result import Result


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _cq():
    import cadquery as cq

    return cq


class Geometry:
    """A CadQuery shape plus the parameters and source that produced it.

    Lengths are in ``units`` (CadQuery works in millimetres by default).
    """

    def __init__(self, shape, *, parameters=None, source=None, units="mm", name="geometry", origin=None):
        self.shape = shape
        self.parameters: dict[str, Any] = dict(parameters or {})
        self.source: dict[str, Any] = dict(source or {})
        self.units = units
        self.name = name
        self.origin = origin  # e.g. the STEP file a geometry was loaded from
        self.created_at = utc_now()
        self._measurements: dict[str, Any] | None = None
        self._messages: list[str] = []

    # -- construction -----------------------------------------------------------------------
    @classmethod
    def from_cadquery(cls, obj, **kwargs) -> "Geometry":
        """Accept a ``cq.Workplane``, ``cq.Shape`` or ``cq.Assembly``."""
        cq = _cq()
        if isinstance(obj, cq.Workplane):
            vals = [v for v in obj.vals() if isinstance(v, cq.Shape)]
            if not vals:
                raise ValueError("Workplane contains no shapes")
            shape = vals[0] if len(vals) == 1 else cq.Compound.makeCompound(vals)
        elif isinstance(obj, cq.Assembly):
            shape = obj.toCompound()
        elif isinstance(obj, cq.Shape):
            shape = obj
        else:
            raise TypeError(f"expected a CadQuery Workplane, Shape or Assembly, got {type(obj).__name__}")
        return cls(shape, **kwargs)

    @classmethod
    def from_step(cls, path: str | Path, units: str = "mm") -> "Geometry":
        cq = _cq()
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        wp = cq.importers.importStep(str(path))
        g = cls.from_cadquery(wp, units=units, name=path.stem, origin=str(path))
        return g

    # -- measurements -----------------------------------------------------------------------
    def measure(self) -> dict[str, Any]:
        """Geometry measurements; quantities that are not reliable are ``None`` (see ``messages``)."""
        if self._measurements is not None:
            return dict(self._measurements)
        s = self.shape
        msgs: list[str] = []
        solids = s.Solids()
        valid = bool(s.isValid())
        bb = s.BoundingBox()
        m: dict[str, Any] = {
            "units": self.units,
            "valid": valid,
            "n_solids": len(solids),
            "n_faces": len(s.Faces()),
            "n_edges": len(s.Edges()),
            "bbox_min": [bb.xmin, bb.ymin, bb.zmin],
            "bbox_max": [bb.xmax, bb.ymax, bb.zmax],
            "dimensions": [bb.xlen, bb.ylen, bb.zlen],
            "surface_area": float(s.Area()),
            "volume": None,
            "center_of_mass": None,
        }
        if not valid:
            msgs.append("shape is not valid according to OpenCascade; volume and center of mass not reported")
        elif not solids:
            msgs.append("shape contains no closed solids; volume and center of mass not reported")
        else:
            m["volume"] = float(sum(sol.Volume() for sol in solids))
            com = _cq().Shape.centerOfMass(s)
            m["center_of_mass"] = [com.x, com.y, com.z]
            if m["volume"] <= 0:
                msgs.append("non-positive solid volume; check face orientation")
                m["volume"] = None
                m["center_of_mass"] = None
        self._measurements = m
        self._messages = msgs
        return dict(m)

    @property
    def messages(self) -> list[str]:
        self.measure()
        return list(self._messages)

    @property
    def volume(self):
        return self.measure()["volume"]

    @property
    def surface_area(self):
        return self.measure()["surface_area"]

    @property
    def dimensions(self):
        return self.measure()["dimensions"]

    @property
    def bounding_box(self):
        m = self.measure()
        return m["bbox_min"], m["bbox_max"]

    # -- tessellation -----------------------------------------------------------------------
    def tessellate(self, tolerance: float = 0.1, angular_tolerance: float = 0.2):
        """Return ``(vertices (N,3), triangles (M,3))`` numpy arrays."""
        verts, tris = self.shape.tessellate(tolerance, angular_tolerance)
        v = np.array([[p.x, p.y, p.z] for p in verts], dtype=float).reshape(-1, 3)
        t = np.array(tris, dtype=int).reshape(-1, 3)
        return v, t

    # -- export -----------------------------------------------------------------------------
    def export_step(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _cq().exporters.export(self.shape, str(path), exportType="STEP")
        return path

    def export_stl(self, path: str | Path, tolerance: float = 0.01, angular_tolerance: float = 0.1) -> Path:
        """Write a binary STL; ``tolerance`` is the linear deflection in model units."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _cq().exporters.export(
            self.shape, str(path), exportType="STL", tolerance=tolerance, angularTolerance=angular_tolerance
        )
        return path

    def export(
        self,
        outdir: str | Path,
        formats: Iterable[str] = ("step", "stl"),
        *,
        basename: str | None = None,
        stl_tolerance: float = 0.01,
        stl_angular_tolerance: float = 0.1,
    ) -> Result:
        """Export to ``outdir`` and return a result with measurements and artifact paths."""
        t0 = time.monotonic()
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        base = basename or self.name
        res = Result(kind="dedalus.generate", metadata=self.metadata())
        res.metadata["stl_tolerance"] = stl_tolerance
        res.metadata["stl_angular_tolerance"] = stl_angular_tolerance
        res.metrics = self.measure()
        res.messages.extend(self.messages)
        for fmt in formats:
            fmt = fmt.lower()
            try:
                if fmt == "step":
                    res.artifacts["step"] = self.export_step(outdir / f"{base}.step")
                elif fmt == "stl":
                    res.artifacts["stl"] = self.export_stl(
                        outdir / f"{base}.stl", stl_tolerance, stl_angular_tolerance
                    )
                else:
                    res.fail(f"unknown export format {fmt!r} (supported: step, stl)")
            except Exception as exc:  # OCC exporters raise a variety of errors
                res.fail(f"{fmt.upper()} export failed: {exc!r}")
        for key, p in res.artifacts.items():
            if not Path(p).is_file() or Path(p).stat().st_size == 0:
                res.fail(f"{key} export produced no data at {p}")
        res.duration_s = time.monotonic() - t0
        res.artifacts["summary"] = outdir / "summary.json"
        res.save_json(res.artifacts["summary"])
        return res

    def metadata(self) -> dict[str, Any]:
        try:
            import cadquery

            cq_version = cadquery.__version__
        except Exception:  # pragma: no cover
            cq_version = None
        return {
            "name": self.name,
            "units": self.units,
            "parameters": dict(self.parameters),
            "source": dict(self.source),
            "origin": self.origin,
            "created_at": self.created_at,
            "tool_versions": {"cadquery": cq_version},
        }

    # -- inspection -------------------------------------------------------------------------
    def show(self, **kwargs):
        """Open CadQuery's own viewer (``cadquery.vis.show``)."""
        from cadquery.vis import show

        return show(self.shape, **kwargs)

    def display(self):
        """Interactive 3D view in Jupyter using CadQuery's built-in notebook renderer."""
        from cadquery.occ_impl.jupyter_tools import display

        return display(self.shape)

    def _repr_javascript_(self):
        return self.shape._repr_javascript_()

    def __repr__(self) -> str:
        d = self.dimensions
        return f"<Geometry {self.name} {d[0]:.4g} x {d[1]:.4g} x {d[2]:.4g} {self.units}>"


def load_step(path: str | Path, units: str = "mm") -> Geometry:
    """Load any STEP file (from any CAD program) for measurement and inspection."""
    return Geometry.from_step(path, units=units)
