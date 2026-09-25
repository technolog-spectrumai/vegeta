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


class Propeller(Design):
    """Constant-geometric-pitch propeller: hub cylinder plus ``blades`` lofted NACA-4 sections along the
    radius (chord grows to ``chord_max`` at 35 % of the blade then tapers to the tip). Rotation axis Z,
    blade 1 along +X. The planform formula matches ``vegeta.boreas.Propeller.from_pitch``. ``skew_deg`` sweeps the
    sections back about the axis, linearly from 0 at the root to the tip value (a skewed, quieter propeller)."""

    parameters = [
        Parameter("diameter", 127.0, "mm", min=20, description="tip-to-tip"),
        Parameter("pitch", 109.0, "mm", min=1, description="geometric pitch (advance per turn)"),
        Parameter("blades", 2, min=1, max=8),
        Parameter("hub_diameter", 14.0, "mm", min=2),
        Parameter("hub_height", 8.0, "mm", min=1),
        Parameter("bore", 5.0, "mm", min=0, description="shaft hole (0 = none)"),
        Parameter("chord_root", 10.0, "mm", min=1),
        Parameter("chord_max", 16.0, "mm", min=1),
        Parameter("chord_tip", 5.0, "mm", min=0.5),
        Parameter("thickness", 0.10, "", min=0.04, max=0.3, description="section thickness / chord"),
        Parameter("camber", 0.04, "", min=0.0, max=0.12, description="section camber / chord"),
        Parameter("stations", 10, min=4, max=30),
        Parameter("skew_deg", 0.0, "deg", min=0.0, max=60.0, description="tip skew, swept back against the rotation (linear from the root)"),
    ]

    @staticmethod
    def _section(chord, t, m, n=16):
        """Closed NACA-4 style section (chord along +x, camber up +y) as a point list, cut at 98 %."""
        xs = [0.98 * 0.5 * (1 - math.cos(math.pi * i / n)) for i in range(n + 1)]
        up, lo = [], []
        for x in xs:
            yt = 5 * t * (0.2969 * math.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
            yc = m / 0.16 * (0.8 * x - x**2) if x < 0.4 else m / 0.36 * (0.2 + 0.8 * x - x**2)
            up.append((chord * x, chord * (yc + yt)))
            lo.append((chord * x, chord * (yc - yt)))
        return up[::-1] + lo[1:]

    def build(self, p):
        R, r0 = p["diameter"] / 2, p["hub_diameter"] / 2
        if p["bore"] >= p["hub_diameter"]:
            raise ValueError("bore must be smaller than hub_diameter")
        if r0 >= R:
            raise ValueError("hub_diameter must be smaller than diameter")
        n = p["stations"]
        hub = cq.Workplane("XY").circle(r0).extrude(p["hub_height"]).translate((0, 0, -p["hub_height"] / 2))
        blade = None
        for i in range(n):
            x = i / (n - 1)
            r = r0 * 0.9 + (R - r0 * 0.9) * x
            if x < 0.35:
                c = p["chord_root"] + (p["chord_max"] - p["chord_root"]) * x / 0.35
            else:
                c = p["chord_max"] + (p["chord_tip"] - p["chord_max"]) * ((x - 0.35) / 0.65) ** 1.5
            beta = math.degrees(math.atan(p["pitch"] / (2 * math.pi * r)))
            pts = [(-0.3 * c + u, -v) for u, v in self._section(c, p["thickness"], p["camber"])]
            # section drawn in the YZ plane at radius r, pitched by beta: a right-hand propeller about +Z (it pushes
            # fluid toward -Z when turning by the right-hand rule; rotated z->x it matches aeromant's rotor templates)
            wp = cq.Workplane("YZ", origin=(r, 0, 0)).polyline([(-u, v) for u, v in pts]).close()
            wire = wp.wires().val().rotate((r, 0, 0), (r + 1, 0, 0), -beta)
            if p["skew_deg"]:
                wire = wire.rotate((0, 0, 0), (0, 0, 1), -p["skew_deg"] * x)            # swept back: the tip trails the root
            blade = cq.Workplane("XY").add(wire) if blade is None else blade.add(wire)
        blade = blade.toPending().loft(ruled=False)
        prop = hub
        for k in range(p["blades"]):
            prop = prop.union(blade.rotate((0, 0, 0), (0, 0, 1), 360 * k / p["blades"]))
        if p["bore"] > 0:
            prop = prop.cut(cq.Workplane("XY").circle(p["bore"] / 2).extrude(50).translate((0, 0, -25)))
        return prop


class Cube(Design):
    """Axis-aligned cube resting on z=0, centred in XY."""

    parameters = [Parameter("size", 20.0, "mm", min=0.1)]

    def build(self, p):
        s = p["size"]
        return cq.Workplane("XY").box(s, s, s, centered=(True, True, False))


__all__ = ["CantileverBeam", "Bracket", "Propeller", "StreamlinedBody", "Cube"]
