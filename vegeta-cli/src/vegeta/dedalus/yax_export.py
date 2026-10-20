"""Export a geometry as a Yax design package (Ozaru's ``yax``): ``body.stl`` + ``yax.json``, optional STEP and parts.

The package is what Piccolo, Nomios and Ketos read — they never import Vegeta. Lengths stay in the geometry's units
(mm), Z up, X forward; mass properties come exactly from OpenCascade, the inertia in kg m^2 about the centre of mass.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

FRAME = {"up": "+z", "forward": "+x"}
_TO_M = {"mm": 1e-3, "m": 1.0}


def mass_properties(geometry) -> dict[str, Any]:
    """Volume, centre of mass and the inertia tensor about it for unit density (length^5), from OpenCascade."""
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(geometry.shape.wrapped, props)
    c, m = props.CentreOfMass(), props.MatrixOfInertia()
    return {"volume": props.Mass(), "com": [c.X(), c.Y(), c.Z()],
            "inertia": [[m.Value(i, j) for j in (1, 2, 3)] for i in (1, 2, 3)]}


def _mass(props, units, mass_kg, density_kg_m3):
    scale = _TO_M[units]
    kg = mass_kg if mass_kg is not None else density_kg_m3 * props["volume"] * scale ** 3
    per_volume = kg / props["volume"] * scale ** 2
    return {"kg": kg, "com": props["com"], "inertia": [[v * per_volume for v in row] for row in props["inertia"]],
            "source": "given" if mass_kg is not None else "density"}


def _revision() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True,
                             timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def export_yax(geometry, out: str | Path, *, name: str | None = None, mass_kg: float | None = None,
               density_kg_m3: float | None = None, parts: dict | None = None, step: bool = False,
               stl_tolerance: float = 0.01):
    """Write ``geometry`` as a Yax package at ``out`` and return the loaded ``yax.Design``.

    ``mass_kg`` (a total, spread uniformly) or ``density_kg_m3`` adds the ``mass`` section; ``parts`` maps names to
    further geometries in the same frame (``parts/<name>.stl``, with their own mass when a density is given)."""
    import yax

    if mass_kg is not None and density_kg_m3 is not None:
        raise ValueError("give mass_kg or density_kg_m3, not both")
    if geometry.units not in _TO_M:
        raise ValueError(f"Yax lengths are mm or m, not {geometry.units!r}")
    m = geometry.measure()
    if m["volume"] is None:
        raise ValueError(f"{geometry.name}: no closed solid to export ({'; '.join(geometry.messages)})")
    props = mass_properties(geometry)
    src = geometry.source
    manifest: dict[str, Any] = {
        "name": name or geometry.name,
        "producer": {"tool": "vegeta.dedalus", "version": _version(), "revision": _revision(),
                     "design_class": f"{src.get('module')}:{src.get('qualname')}" if src else geometry.name,
                     "parameters": geometry.parameters},
        "units": {"length": geometry.units, "mass": "kg"},
        "frame": dict(FRAME),
        "geometry": {"bbox": [m["bbox_min"], m["bbox_max"]], "volume": props["volume"], "com": props["com"]},
    }
    if mass_kg is not None or density_kg_m3 is not None:
        manifest["mass"] = _mass(props, geometry.units, mass_kg, density_kg_m3)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        files = {"body.stl": geometry.export_stl(tmp / "body.stl", tolerance=stl_tolerance)}
        if step:
            files["body.step"] = geometry.export_step(tmp / "body.step")
        if parts:
            manifest["parts"] = []
            for part_name, part in parts.items():
                rel = f"parts/{part_name}.stl"
                files[rel] = part.export_stl(tmp / rel, tolerance=stl_tolerance)
                pp = mass_properties(part)
                entry = {"name": part_name, "file": rel, "com": pp["com"]}
                if density_kg_m3 is not None:
                    entry["kg"] = density_kg_m3 * pp["volume"] * _TO_M[part.units] ** 3
                manifest["parts"].append(entry)
        return yax.write(out, manifest, files)


def _version() -> str:
    from . import __version__

    return __version__
