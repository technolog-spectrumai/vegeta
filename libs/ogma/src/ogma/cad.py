"""og.cad — Ogma's CAD strategy: parts ↔ CadQuery/OpenCascade solids.

A strategy only translates; the physics stays in the laws. CadQuery is imported lazily (the optional ``cad`` extra),
so ``import ogma`` never needs it.

- ``mass_properties(shape)``: volume, centre of mass and the inertia tensor about it for unit density, exactly from
  OpenCascade (any CadQuery Workplane/Shape or OCP TopoDS_Shape). Moved from Vegeta's ``dedalus.yax_export``.
- ``mass_section(props, units, mass_kg=None, density_kg_m3=None)``: those, scaled to a mass — the ``mass`` section of
  a Yax manifest (lengths in the CAD's units, inertia in kg m²).
- ``properties(shape, units="mm", mass_kg=… | density_kg_m3=…)``: the same as an ``og.mass.MassProperties`` in SI.
- ``solid(shape, pos=(0, 0, 0), units="mm")``: an ``og.mass`` solid shape (``Box``, ``Sphere``, ``Cylinder``,
  ``Capsule``, ``Ellipsoid``) as a CadQuery Workplane centred at ``pos`` [m], built in the CAD's units — so a budget and
  its CAD agree by construction. The thin shapes (``Rod``, ``Plate``, ``Shell``) are budget idealisations: with
  ``thin=`` [m] they get a visual stand-in (a thin cylinder, a thin plate, an ellipsoid skin), never a mass.
- ``assembly(budget, units="mm")``: a CadQuery ``Assembly`` of the budget's shaped items, named and placed, coloured
  by group; ``assembly_properties(budget)``: the budget's mass properties with every solid item's inertia taken from
  its CAD solid — equal to ``Budget.properties()`` (the check that the CAD and the budget are the same machine).
- ``export(shape_or_assembly, path)``: STEP or STL by the suffix.
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


def mass_properties(shape, precision: float | None = None) -> dict[str, Any]:
    """Volume, centre of mass and the inertia tensor about it for unit density (length^5), from OpenCascade.
    ``precision`` (relative, e.g. 1e-9) switches OpenCascade to adaptive integration — needed on spline surfaces
    (an ellipsoid: 0.4 % off with the default fixed rule); None keeps the default (what Dedalus always used)."""
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    props = GProp_GProps()
    if precision is None:
        BRepGProp.VolumeProperties_s(_topods(shape), props)
    else:
        BRepGProp.VolumeProperties_s(_topods(shape), props, precision, True)
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


def properties(shape, units: str = "mm", *, mass_kg: float | None = None, density_kg_m3: float | None = None,
               precision: float | None = 1e-9) -> om.MassProperties:
    """The solid's mass, centre of mass [m] and inertia about it [kg m²] as an ``og.mass.MassProperties``
    (adaptive integration to ``precision`` by default)."""
    if (mass_kg is None) == (density_kg_m3 is None):
        raise ValueError("give mass_kg or density_kg_m3")
    s = mass_section(mass_properties(shape, precision), units, mass_kg, density_kg_m3)
    return om.MassProperties(float(s["kg"]), np.asarray(s["com"], float) * TO_M[units], np.asarray(s["inertia"], float))


def _ellipsoid(a, b, c):
    """A solid ellipsoid of semi-axes a, b, c: a unit sphere stretched by a general transformation."""
    import cadquery as cq
    from OCP.BRepBuilderAPI import BRepBuilderAPI_GTransform
    from OCP.gp import gp_GTrsf, gp_Mat, gp_XYZ

    g = gp_GTrsf()
    g.SetVectorialPart(gp_Mat(a, 0, 0, 0, b, 0, 0, 0, c))
    g.SetTranslationPart(gp_XYZ(0, 0, 0))
    sphere = cq.Solid.makeSphere(1.0, angleDegrees1=-90, angleDegrees2=90)
    return cq.Workplane("XY").add(cq.Shape.cast(BRepBuilderAPI_GTransform(sphere.wrapped, g, True).Shape()))


def solid(shape, pos=(0.0, 0.0, 0.0), units: str = "mm", *, thin: float | None = None):
    """An ``og.mass`` shape as a CadQuery Workplane centred at ``pos`` [m] (in the CAD's ``units``); thin shapes need
    ``thin`` [m] (their stand-in's thickness) and are for looking at, not for mass."""
    import cadquery as cq

    k = 1.0 / TO_M[units]
    x, y, z = (float(v) * k for v in pos)
    plane = {"x": "YZ", "y": "XZ", "z": "XY"}
    if isinstance(shape, om.Box):
        w = cq.Workplane("XY").box(shape.sx * k, shape.sy * k, shape.sz * k)
    elif isinstance(shape, om.Sphere):
        w = cq.Workplane("XY").sphere(shape.radius * k)
    elif isinstance(shape, om.Cylinder):
        w = cq.Workplane(plane[shape.axis]).circle(shape.radius * k).extrude(shape.length * k / 2, both=True)
    elif isinstance(shape, om.Capsule):
        r, h = shape.radius * k, shape.length * k
        w = cq.Workplane(plane[shape.axis]).circle(r).extrude(h / 2, both=True)
        n = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[shape.axis]
        for sgn in (1, -1):
            w = w.union(cq.Workplane("XY").sphere(r).translate(tuple(sgn * h / 2 * c for c in n)))
    elif isinstance(shape, om.Ellipsoid):
        w = _ellipsoid(shape.a * k, shape.b * k, shape.c * k)
    elif thin is None:
        raise NotImplementedError(f"og.cad.solid: {type(shape).__name__} is a budget idealisation (a thin shape): pass "
                                  "thin=<thickness [m]> for a visual stand-in, or build it in the design's CAD")
    elif isinstance(shape, om.Rod):
        w = cq.Workplane(plane[shape.axis]).circle(thin * k / 2).extrude(shape.length * k / 2, both=True)
    elif isinstance(shape, om.Plate):
        size = {shape.axes[0]: shape.a * k, shape.axes[1]: shape.b * k}
        w = cq.Workplane("XY").box(size.get("x", thin * k), size.get("y", thin * k), size.get("z", thin * k))
    elif isinstance(shape, om.Shell):
        outer = _ellipsoid(shape.a * k, shape.b * k, shape.c * k)
        t = thin * k
        w = outer.cut(_ellipsoid(max(shape.a * k - t, t), max(shape.b * k - t, t), max(shape.c * k - t, t)))
    else:
        raise NotImplementedError(f"og.cad.solid: no CAD for {type(shape).__name__}")
    return w.translate((x, y, z))


_SOLIDS = (om.Box, om.Sphere, om.Cylinder, om.Capsule, om.Ellipsoid)
_PALETTE = [(0.55, 0.60, 0.70), (0.80, 0.55, 0.30), (0.40, 0.65, 0.45), (0.70, 0.40, 0.45), (0.50, 0.45, 0.70),
            (0.75, 0.70, 0.35)]


def assembly(budget: om.Budget, units: str = "mm", *, thin: float | None = 0.002, name: str = "budget"):
    """The budget's shaped items as a CadQuery ``Assembly`` (one part per item, named after it, coloured by group);
    thin shapes as ``thin`` [m] stand-ins (``None`` leaves them out); point masses have no geometry."""
    import cadquery as cq

    asm = cq.Assembly(name=name)
    groups = {}
    for i, it in enumerate(budget.items):
        if it.shape is None or (thin is None and not isinstance(it.shape, _SOLIDS)):
            continue
        rgb = _PALETTE[groups.setdefault(it.group, len(groups)) % len(_PALETTE)]
        asm.add(solid(it.shape, it.pos, units, thin=thin), name=f"{i:02d} {it.name}", color=cq.Color(*rgb))
    return asm


def assembly_properties(budget: om.Budget) -> om.MassProperties:
    """The budget's mass properties with each solid item's own inertia from its CAD solid (OpenCascade); thin shapes
    keep their idealised inertia, point masses none. Equal to ``budget.properties()`` when CAD and budget agree."""
    total = None
    for it in budget.items:
        if isinstance(it.shape, _SOLIDS):
            mp = properties(solid(it.shape, it.pos, "m"), "m", mass_kg=it.kg)
        else:
            own = it.shape.inertia(it.kg) if it.shape is not None else np.zeros((3, 3))
            mp = om.MassProperties(float(it.kg), np.asarray(it.pos, float), own)
        total = mp if total is None else total + mp
    return total


def export(obj, path) -> str:
    """Write a Workplane, Shape or Assembly as STEP (``.step``/``.stp``) or STL (``.stl``); returns the path."""
    import cadquery as cq
    from pathlib import Path

    path = Path(path)
    suffix = path.suffix.lower()
    if isinstance(obj, cq.Assembly):
        obj = obj.toCompound()
    if suffix in (".step", ".stp"):
        cq.exporters.export(obj, str(path), exportType="STEP")
    elif suffix == ".stl":
        cq.exporters.export(obj, str(path), exportType="STL")
    else:
        raise ValueError(f"og.cad.export: .step or .stl, not {suffix!r}")
    return str(path)


__all__ = ["mass_properties", "mass_section", "properties", "solid", "assembly", "assembly_properties", "export", "TO_M"]
