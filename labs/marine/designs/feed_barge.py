"""Feed barge: the moored concrete barge a fjord salmon farm is run from (Ketos's base). A box hull, the feed silos
on deck, the control house with the crew room, and the dock for the resident robots (Cyclopterus's inductive docks)
on its side, at ``dock_depth``. The feed is blown to the pens through floating pipes (not CAD).

Layout: x along the barge, z = 0 the waterline, the hull's centre at x = y = 0.

``part``: ``barge`` (all), ``hull``, ``silos``, ``house``, ``dock``.
"""
from __future__ import annotations

try:
    import cadquery as cq
except ImportError:
    cq = None

from vegeta.dedalus import Design, Parameter

RHO_SW = 1025.0


class FeedBarge(Design):
    parameters = [
        Parameter("part", "barge", choices=("barge", "hull", "silos", "house", "dock"), description="what to build"),
        Parameter("length", 36.0, "m", min=10),
        Parameter("beam", 14.0, "m", min=5),
        Parameter("depth", 4.0, "m", min=1, description="keel to deck"),
        Parameter("draft", 2.4, "m", min=0.5),
        Parameter("silos", 6, min=1, max=16),
        Parameter("silo_d", 4.0, "m", min=1),
        Parameter("silo_h", 7.0, "m", min=1),
        Parameter("dock_depth", 3.0, "m", min=0.5, description="the robots' dock below the waterline"),
        Parameter("docks", 3, min=1, max=8),
    ]

    @staticmethod
    def p(over=None):
        return {q.name: q.default for q in FeedBarge.parameters} | (over or {})

    @classmethod
    def sizing(cls, over=None):
        p = cls.p(over)
        disp = p["length"] * p["beam"] * p["draft"] * RHO_SW / 1000          # t
        silo_m3 = p["silos"] * 3.1416 * (p["silo_d"] / 2) ** 2 * p["silo_h"]
        return {"displacement_t": disp, "feed_t": silo_m3 * 0.65, "dock_positions": int(p["docks"])}

    def dock_positions(self, p):
        """Where each robot docks (m, barge axes): along the port side, under water."""
        n = int(p["docks"])
        return [(-p["length"] / 2 + p["length"] * (i + 1) / (n + 1), p["beam"] / 2 + 0.6, -p["dock_depth"]) for i in range(n)]

    def build(self, p):
        if cq is None:
            raise ImportError("FeedBarge's CAD needs CadQuery")
        mm = 1000.0
        L, B, D, T = p["length"] * mm, p["beam"] * mm, p["depth"] * mm, p["draft"] * mm
        hull = cq.Workplane("XY").box(L, B, D, centered=(True, True, False)).translate((0, 0, -T)).edges("|Z").fillet(800)
        n = int(p["silos"])
        silos = cq.Workplane("XY")
        for i in range(n):
            row, col = i % 2, i // 2
            x = -L / 2 + 3000 + col * (p["silo_d"] * mm + 600) + p["silo_d"] * mm / 2
            y = (row - 0.5) * (p["silo_d"] * mm + 600)
            silos = silos.add(cq.Solid.makeCylinder(p["silo_d"] * mm / 2, p["silo_h"] * mm, cq.Vector(x, y, D - T)))
            silos = silos.add(cq.Solid.makeCone(p["silo_d"] * mm / 2, 300, 1200, cq.Vector(x, y, D - T + p["silo_h"] * mm)))
        silos = silos.combine()
        house = cq.Workplane("XY").box(10000, B - 2000, 5000, centered=(True, True, False)).translate((L / 2 - 6500, 0, D - T))
        house = house.union(cq.Workplane("XY").box(6000, B - 4000, 2500, centered=(True, True, False)).translate((L / 2 - 6500, 0, D - T + 5000)))
        dock = cq.Workplane("XY")
        for x, y, z in self.dock_positions(p):
            dock = dock.add(cq.Solid.makeCylinder(500, 200, cq.Vector(x * mm, y * mm, z * mm)))
            dock = dock.add(cq.Solid.makeBox(300, 700, abs(z) * mm + 200, cq.Vector(x * mm - 150, B / 2 - 100, z * mm)))
        dock = dock.combine()
        parts = {"hull": hull, "silos": silos, "house": house, "dock": dock}
        if p["part"] in parts:
            return parts[p["part"]]
        return hull.union(silos).union(house).union(dock)


if __name__ == "__main__":
    print(FeedBarge.sizing())
