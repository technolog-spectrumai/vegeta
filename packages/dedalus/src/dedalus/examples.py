"""Example parametric designs (millimetres). They are examples, not engineering recommendations."""
from __future__ import annotations

import math

import cadquery as cq

from .design import Design
from .parameters import Parameter


class CantileverBeam(Design):
    """Rectangular prismatic beam along +X starting at x=0 (fixed end) to x=length (free end)."""

    parameters = [
        Parameter("length", 200.0, "mm", min=1, description="beam length along X"),
        Parameter("width", 20.0, "mm", min=0.1, description="section width along Y"),
        Parameter("height", 10.0, "mm", min=0.1, description="section height along Z"),
    ]

    def build(self, p):
        return cq.Workplane("XY").box(p["length"], p["width"], p["height"], centered=(False, True, True))


class Bracket(Design):
    """Flat plate with mounting holes and rounded vertical edges."""

    parameters = [
        Parameter("length", 80.0, "mm", min=10),
        Parameter("width", 40.0, "mm", min=10),
        Parameter("thickness", 6.0, "mm", min=1),
        Parameter("hole_diameter", 6.5, "mm", min=0.5),
        Parameter("hole_margin", 10.0, "mm", min=1),
        Parameter("fillet", 4.0, "mm", min=0),
    ]

    def build(self, p):
        if p["hole_diameter"] >= 2 * p["hole_margin"]:
            raise ValueError("hole_diameter must be smaller than 2 * hole_margin")
        body = cq.Workplane("XY").box(p["length"], p["width"], p["thickness"])
        if p["fillet"] > 0:
            body = body.edges("|Z").fillet(p["fillet"])
        dx = p["length"] / 2 - p["hole_margin"]
        dy = p["width"] / 2 - p["hole_margin"]
        return (
            body.faces(">Z").workplane()
            .pushPoints([(dx, dy), (-dx, dy), (dx, -dy), (-dx, -dy)])
            .hole(p["hole_diameter"])
        )


class StreamlinedBody(Design):
    """Axisymmetric body along +X: elliptic nose, cylindrical middle, tapered tail."""

    parameters = [
        Parameter("length", 100.0, "mm", min=1),
        Parameter("diameter", 20.0, "mm", min=0.1),
        Parameter("nose_fraction", 0.25, min=0.05, max=0.6),
        Parameter("tail_fraction", 0.4, min=0.05, max=0.8),
        Parameter("tail_diameter_ratio", 0.2, min=0.0, max=1.0),
    ]

    def build(self, p):
        L, r = p["length"], p["diameter"] / 2
        ln, lt = L * p["nose_fraction"], L * p["tail_fraction"]
        if ln + lt > L:
            raise ValueError("nose_fraction + tail_fraction must not exceed 1")
        rt = r * p["tail_diameter_ratio"]
        # half-profile in the XY plane (y = radius), revolved about the X axis
        nose = [(ln * (1 - math.cos(a)), r * math.sin(a)) for a in (i * math.pi / 16 for i in range(9))]
        prof = (
            cq.Workplane("XY")
            .moveTo(0, 0)
            .spline(nose[1:], tangents=[(0, 1), (1, 0)], includeCurrent=True)
            .lineTo(L - lt, r)
            .lineTo(L, rt)
        )
        if rt > 0:
            prof = prof.lineTo(L, 0)
        return prof.close().revolve(360, (0, 0, 0), (1, 0, 0))


class Cube(Design):
    """Axis-aligned cube resting on z=0, centred in XY."""

    parameters = [Parameter("size", 20.0, "mm", min=0.1)]

    def build(self, p):
        s = p["size"]
        return cq.Workplane("XY").box(s, s, s, centered=(True, True, False))


__all__ = ["CantileverBeam", "Bracket", "StreamlinedBody", "Cube"]
