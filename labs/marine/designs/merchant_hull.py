"""Merchant hull: a generic 120 m steel cargo ship's hull (a handy-size bulker / feeder class) for Echensis's
hull-cleaning jobs in Ketos — the ship that comes alongside the farm's quay. Not a real ship: a parametric hull of
``stations`` ruled sections, each a quarter superellipse mirrored about the centreline.

The section at station s (s = -1 the stern, +1 the stem; x = s L/2) has half-breadth b(s) = B/2 w(s) and goes from
the keel z_k(s) up to the deck (z = D - T); its exponent n(s) sets its fullness (n = 10 midship: a box with a ~2 m
bilge; n = 2.5 at the ends: V-ish). The parallel midbody runs from s = -0.45 to s = +0.30; the bow tapers to a 0.4 m
stem; the stern narrows to a 0.55 B transom with the keel rising 5 m over the last 10 % (the run to the propeller).
With the defaults the measured block coefficient (``sizing()['cb_measured']``) is about 0.78.

Layout: x forward, y left, z up; z = 0 the waterline, x = 0 midship, y = 0 the centreline (metres in the
parameters, millimetres in the CAD like every lab design).

``part``: ``hull`` (the only part).
"""
from __future__ import annotations

import math

try:
    import cadquery as cq
except ImportError:
    cq = None

from vegeta.dedalus import Design, Parameter

RHO_SW = 1025.0


class MerchantHull(Design):
    parameters = [
        Parameter("part", "hull", choices=("hull",), description="what to build"),
        Parameter("length", 120.0, "m", min=20, description="length between perpendiculars"),
        Parameter("beam", 20.0, "m", min=4),
        Parameter("draft", 7.5, "m", min=1, description="design draft (keel to waterline)"),
        Parameter("depth", 11.0, "m", min=2, description="keel to main deck"),
        Parameter("cb", 0.78, "", min=0.4, max=0.9, description="block coefficient (used in the wetted area; the shape gives ~0.78)"),
        Parameter("stations", 24, min=6, max=60, description="sections along the hull"),
        Parameter("points", 10, min=4, max=30, description="points on each quarter section"),
    ]

    @staticmethod
    def p(over=None):
        return {q.name: q.default for q in MerchantHull.parameters} | (over or {})

    @staticmethod
    def width(s):
        """w(s): half-breadth over B/2 at station s (-1 stern .. +1 stem)."""
        if s > 0.30:
            return max(0.02, 1 - ((s - 0.30) / 0.70) ** 2.0)
        if s < -0.45:
            return 1 - 0.45 * ((-0.45 - s) / 0.55) ** 1.6
        return 1.0

    @staticmethod
    def keel(s, T):
        """z of the keel [m] at station s: flat at -T, rising at the stern (the run) and the forefoot."""
        if s < -0.8:
            return -T + 5.0 * ((-0.8 - s) / 0.2) ** 1.5 * T / 7.5
        if s > 0.92:
            return -T + 0.35 * T * ((s - 0.92) / 0.08) ** 2
        return -T

    @staticmethod
    def fullness(s):
        """n(s): the superellipse exponent of the section (10 midship, 2.5 at the ends)."""
        w = MerchantHull.width(s)
        return 2.5 + 7.5 * w ** 3

    def section(self, p, s):
        """The closed section at station s as (y, z) points [m], keel to deck, both sides."""
        B, T, D = p["beam"], p["draft"], p["depth"]
        b = B / 2 * self.width(s)
        zk, top = self.keel(s, T), D - T
        H = top - zk
        n = self.fullness(s)
        k = int(p["points"])
        half = []
        for i in range(k + 1):
            th = (math.pi / 2) * i / k                       # 0 at the deck edge, pi/2 at the keel
            c, sn = math.cos(th), math.sin(th)
            y = b * (c ** (2 / n) if c > 0 else 0.0)
            z = zk + H * (1 - (sn ** (2 / n) if sn > 0 else 0.0))
            half.append((y, z))
        # deck edge (b, top) .. keel (0, zk), then up the other side
        pts = half + [(-y, z) for y, z in reversed(half[:-1])]
        return pts

    @classmethod
    def sizing(cls, over=None):
        """Displacement and wetted area (Denny-Mumford: S = 1.7 L T + Cb L B), and the shape's own block coefficient
        (the CAD's volume below z = 0 over L B T) where CadQuery is."""
        p = cls.p(over)
        L, B, T = p["length"], p["beam"], p["draft"]
        out = {"wetted_area_m2": 1.7 * L * T + p["cb"] * L * B, "displacement_t": p["cb"] * L * B * T * RHO_SW / 1000}
        if cq is not None:
            hull = cls().build(p)
            below = cq.Workplane("XY").box(L * 2000, B * 2000, T * 2000, centered=(True, True, False)).translate((0, 0, -T * 2000))
            v = hull.intersect(below).val().Volume() / 1e9
            out["cb_measured"] = v / (L * B * T)
        return out

    def build(self, p):
        if cq is None:
            raise ImportError("MerchantHull's CAD needs CadQuery")
        n = int(p["stations"])
        wires = []
        for i in range(n + 1):
            s = -1 + 2 * i / n
            x = s * p["length"] / 2 * 1000
            pts = [cq.Vector(x, y * 1000, z * 1000) for y, z in self.section(p, s)]
            wires.append(cq.Wire.makePolygon(pts, close=True))
        solid = cq.Solid.makeLoft(wires, True)
        return cq.Workplane("XY").add(solid)


if __name__ == "__main__":
    print(MerchantHull.sizing())
