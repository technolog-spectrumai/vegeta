"""og.cad — Ogma's CAD strategy: parts ↔ CadQuery/OpenCascade solids.

A strategy only translates; the physics stays in the laws. CadQuery is imported lazily (the optional ``cad`` extra),
so ``import ogma`` never needs it.

- ``mass_properties(shape)``: volume, centre of mass and the inertia tensor about it for unit density, exactly from
  OpenCascade (any CadQuery Workplane/Shape or OCP TopoDS_Shape). Moved from Vegeta's ``dedalus.yax_export``.
- ``mass_section(props, units, mass_kg=None, density_kg_m3=None)``: those, scaled to a mass — the ``mass`` section of
  a Yax manifest (lengths in the CAD's units, inertia in kg m²).
- ``properties(shape, units="mm", mass_kg=… | density_kg_m3=…)``: the same as an ``og.mass.MassProperties`` in SI.
- ``solid(shape, pos=(0, 0, 0), units="mm")``: an ``og.mass`` solid shape (``Box``, ``Cylinder``, ``Sphere``) as a
  CadQuery Workplane centred at ``pos`` [m], built in the CAD's units — so a budget and its CAD agree by construction.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from . import mass as om

TO_M = {"mm": 1e-3, "m": 1.0}


def _topods(shape):
    """The OCP TopoDS_Shape inside a CadQuery Workplane / Shape (or the shape itself)."""
    if hasattr(shape, "val") and callable(shape.val):              # a Workplane
        shape = shape.val()
    return getattr(shape, "wrapped", shape)


def mass_properties(shape) -> dict[str, Any]:
    """Volume, centre of mass and the inertia tensor about it for unit density (length^5), from OpenCascade."""
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(_topods(shape), props)
    c, m = props.CentreOfMass(), props.MatrixOfInertia()
    return {"volume": props.Mass(), "com": [c.X(), c.Y(), c.Z()],
            "inertia": [[m.Value(i, j) for j in (1, 2, 3)] for i in (1, 2, 3)]}


def mass_section(props: dict, units: str, mass_kg: float | None = None, density_kg_m3: float | None = None) -> dict:
    """``mass_properties`` scaled to ``mass_kg`` (spread uniformly) or ``density_kg_m3``: kg, the centre of mass in
    the CAD's units, the inertia about it in kg m², and where the mass came from."""
    scale = TO_M[units]
    kg = mass_kg if mass_kg is not None else density_kg_m3 * props["volume"] * scale ** 3
    per_volume = kg / props["volume"] * scale ** 2
    return {"kg": kg, "com": props["com"], "inertia": [[v * per_volume for v in row] for row in props["inertia"]],
            "source": "given" if mass_kg is not None else "density"}


def properties(shape, units: str = "mm", *, mass_kg: float | None = None, density_kg_m3: float | None = None) -> om.MassProperties:
    """The solid's mass, centre of mass [m] and inertia about it [kg m²] as an ``og.mass.MassProperties``."""
    if (mass_kg is None) == (density_kg_m3 is None):
        raise ValueError("give mass_kg or density_kg_m3")
    s = mass_section(mass_properties(shape), units, mass_kg, density_kg_m3)
    return om.MassProperties(float(s["kg"]), np.asarray(s["com"], float) * TO_M[units], np.asarray(s["inertia"], float))


def solid(shape, pos=(0.0, 0.0, 0.0), units: str = "mm"):
    """An ``og.mass`` solid shape as a CadQuery Workplane centred at ``pos`` [m] (in the CAD's ``units``)."""
    import cadquery as cq

    k = 1.0 / TO_M[units]
    x, y, z = (float(v) * k for v in pos)
    if isinstance(shape, om.Box):
        return cq.Workplane("XY").box(shape.sx * k, shape.sy * k, shape.sz * k).translate((x, y, z))
    if isinstance(shape, om.Sphere):
        return cq.Workplane("XY").sphere(shape.radius * k).translate((x, y, z))
    if isinstance(shape, om.Cylinder):
        plane = {"x": "YZ", "y": "XZ", "z": "XY"}[shape.axis]
        h = shape.length * k
        return cq.Workplane(plane).circle(shape.radius * k).extrude(h / 2, both=True).translate((x, y, z))
    raise NotImplementedError(f"og.cad.solid: no CAD solid for {type(shape).__name__} (thin shapes and ellipsoids "
                              "are budget idealisations; build them in the design's CAD)")


__all__ = ["mass_properties", "mass_section", "properties", "solid", "TO_M"]
