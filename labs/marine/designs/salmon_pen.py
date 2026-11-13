"""Salmon pen collar: the floating ring of a fjord salmon farm (Ketos's pens). Two HDPE floater pipes side by side,
a walkway between them, handrail stanchions and a rail on the inside, and the sinker tube that keeps the net's bottom
ring down (hung below on chains; drawn as a ring at ``sinker_depth``). The net itself is not CAD: Ketos simulates it
(panels with fouling and holes).

Layout: z = 0 the waterline, the pipes half under it; the centre of the ring at x = y = 0.

``part``: ``collar`` (pipes + walkway + rail), ``sinker``.
"""
from __future__ import annotations

import math

try:
    import cadquery as cq
except ImportError:
    cq = None

from vegeta.dedalus import Design, Parameter

RHO_SW, G = 1025.0, 9.81


class SalmonPen(Design):
    parameters = [
        Parameter("part", "collar", choices=("collar", "sinker"), description="what to build"),
        Parameter("circumference", 160.0, "m", min=40, description="the collar's inner circumference"),
        Parameter("pipe_d", 500.0, "mm", min=200, description="each floater pipe's outer diameter"),
        Parameter("pipe_gap", 900.0, "mm", min=300, description="centre to centre of the two pipes (the walkway)"),
        Parameter("rail_h", 1000.0, "mm", min=500),
        Parameter("stanchions", 48, min=8, max=200),
        Parameter("net_depth", 25.0, "m", min=5, description="the net's wall, collar to sinker"),
        Parameter("cone_depth", 8.0, "m", min=0, description="the bottom cone below the wall"),
        Parameter("sinker_depth", 26.0, "m", min=5),
    ]

    @staticmethod
    def p(over=None):
        return {q.name: q.default for q in SalmonPen.parameters} | (over or {})

    @classmethod
    def sizing(cls, over=None):
        """Net areas and volume, the collar's reserve buoyancy (two pipes)."""
        p = cls.p(over)
        R = p["circumference"] / (2 * math.pi)
        wall = p["circumference"] * p["net_depth"]
        cone = math.pi * R * math.hypot(R, p["cone_depth"])
        volume = math.pi * R * R * p["net_depth"] + math.pi * R * R * p["cone_depth"] / 3
        pipe_len = 2 * p["circumference"] + 2 * math.pi * p["pipe_gap"] / 1000
        area = math.pi * (p["pipe_d"] / 2000) ** 2
        buoy = RHO_SW * area * pipe_len * G
        return {"radius_m": R, "net_wall_m2": wall, "net_cone_m2": cone, "net_area_m2": wall + cone,
                "volume_m3": volume, "fish_at_25kg_m3_t": volume * 25 / 1000, "collar_buoyancy_kN": buoy / 1000}

    def build(self, p):
        if cq is None:
            raise ImportError("SalmonPen's CAD needs CadQuery")
        R = p["circumference"] / (2 * math.pi) * 1000                     # mm, the inner pipe's centre line
        r = p["pipe_d"] / 2
        if p["part"] == "sinker":
            return cq.Workplane("XY").add(cq.Solid.makeTorus(R, 150, pnt=cq.Vector(0, 0, -p["sinker_depth"] * 1000)))
        out = cq.Workplane("XY")
        for k in (0, 1):                                                   # two floater pipes, half submerged
            out = out.add(cq.Solid.makeTorus(R + k * p["pipe_gap"], r, pnt=cq.Vector(0, 0, 0)))
        n = int(p["stanchions"])
        for i in range(n):                                                 # stanchions on the inner pipe
            a = 2 * math.pi * i / n
            x, y = R * math.cos(a), R * math.sin(a)
            out = out.add(cq.Solid.makeCylinder(40, p["rail_h"], cq.Vector(x, y, r * 0.5)))
        out = out.add(cq.Solid.makeTorus(R, 40, pnt=cq.Vector(0, 0, r * 0.5 + p["rail_h"])))
        return out.combine()


if __name__ == "__main__":
    s = SalmonPen.sizing()
    print(f"R {s['radius_m']:.1f} m, net {s['net_area_m2']:.0f} m^2 (wall {s['net_wall_m2']:.0f}, cone {s['net_cone_m2']:.0f}), "
          f"volume {s['volume_m3']:.0f} m^3 ({s['fish_at_25kg_m3_t']:.0f} t at 25 kg/m^3), collar {s['collar_buoyancy_kN']:.0f} kN")
