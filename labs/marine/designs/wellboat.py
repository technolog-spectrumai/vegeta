"""Wellboat: the 75 m live-fish carrier that comes alongside a salmon farm's pens in Ketos — it pumps the fish aboard
into its wells for delice treatment (freshwater bath, or thermal: ~34 C water for ~30 s) or for the trip to the
slaughterhouse. Modern Norwegian wellboats of this size carry 2 500-4 000 m^3 in two to four wells.

Quick CAD: a box hull with a lofted bow (the keel rising 3 m to the stem, a 1.5 m sheer at the bow), the
superstructure aft (accommodation, the bridge on top), two well hatches on the midship deck and a knuckle crane
between them.

Numbers (``sizing``): displacement = Cb L B T rho with Cb 0.70 at the 6 m draft (5 170 t); wells two boxes of
15 x 13 x 7 m inside the hull (2 x 1 365 m^3); 6 m/s (~12 kn) service speed.

Layout: x forward, y left, z up; z = 0 the waterline, x = 0 midship (the hull's middle), y = 0 the centreline.

``part``: ``boat`` (all), ``hull``, ``house``, ``wells`` (the hatches), ``crane``.
"""
from __future__ import annotations

try:
    import cadquery as cq
except ImportError:
    cq = None

from vegeta.dedalus import Design, Parameter

RHO_SW = 1025.0


class Wellboat(Design):
    parameters = [
        Parameter("part", "boat", choices=("boat", "hull", "house", "wells", "crane"), description="what to build"),
        Parameter("length", 75.0, "m", min=20),
        Parameter("beam", 16.0, "m", min=5),
        Parameter("depth", 8.0, "m", min=2, description="keel to main deck"),
        Parameter("draft", 6.0, "m", min=1, description="loaded, wells full"),
        Parameter("cb", 0.70, "", min=0.4, max=0.9, description="block coefficient at the draft"),
        Parameter("bow_length", 12.0, "m", min=2, description="the lofted bow ahead of the box"),
        Parameter("wells", 2, min=1, max=4),
        Parameter("well_length", 15.0, "m", min=2),
        Parameter("well_width", 13.0, "m", min=2),
        Parameter("well_depth", 7.0, "m", min=1),
        Parameter("max_speed", 6.0, "m/s", min=1, description="service speed (~12 kn)"),
        Parameter("price_eur", 55e6, "EUR", min=0, description="new-build class price (assumed)"),
    ]

    @staticmethod
    def p(over=None):
        return {q.name: q.default for q in Wellboat.parameters} | (over or {})

    @classmethod
    def sizing(cls, over=None):
        p = cls.p(over)
        disp_t = p["cb"] * p["length"] * p["beam"] * p["draft"] * RHO_SW / 1000
        wells = int(p["wells"]) * p["well_length"] * p["well_width"] * p["well_depth"]
        return {"displacement_t": disp_t, "mass_kg": disp_t * 1000, "wells_m3": wells}

    def well_positions(self, p):
        """x of each well's centre [m]: side by side ahead of the house, 4 m apart."""
        n = int(p["wells"])
        x0 = -p["length"] / 2 + 18.0 + p["well_length"] / 2
        return [x0 + i * (p["well_length"] + 4.0) for i in range(n)]

    def build(self, p):
        if cq is None:
            raise ImportError("Wellboat's CAD needs CadQuery")
        mm = 1000.0
        L, B, D, T, Lb = p["length"] * mm, p["beam"] * mm, p["depth"] * mm, p["draft"] * mm, p["bow_length"] * mm
        x_bow = L / 2 - Lb
        box = cq.Workplane("XY").box(L - Lb, B, D, centered=(False, True, False)).translate((-L / 2, 0, -T))
        box = box.edges("|X and <Z").fillet(1500)
        aft = cq.Wire.makePolygon([cq.Vector(x_bow, -B / 2, -T), cq.Vector(x_bow, B / 2, -T), cq.Vector(x_bow, B / 2, D - T),
                                   cq.Vector(x_bow, -B / 2, D - T)], close=True)
        mid = cq.Wire.makePolygon([cq.Vector(x_bow + Lb * 0.6, -B * 0.32, -T + 1500), cq.Vector(x_bow + Lb * 0.6, B * 0.32, -T + 1500),
                                   cq.Vector(x_bow + Lb * 0.6, B * 0.36, D - T + 800), cq.Vector(x_bow + Lb * 0.6, -B * 0.36, D - T + 800)],
                                  close=True)
        stem = cq.Wire.makePolygon([cq.Vector(L / 2, -300, -T + 3000), cq.Vector(L / 2, 300, -T + 3000),
                                    cq.Vector(L / 2, 300, D - T + 1500), cq.Vector(L / 2, -300, D - T + 1500)], close=True)
        bow = cq.Workplane("XY").add(cq.Solid.makeLoft([aft, mid, stem], True))
        hull = box.union(bow)
        deck = D - T
        house = cq.Workplane("XY").box(14000, B - 1000, 7000, centered=(False, True, False)).translate((-L / 2 + 1500, 0, deck))
        house = house.union(cq.Workplane("XY").box(8000, B - 3000, 3000, centered=(False, True, False))
                            .translate((-L / 2 + 5500, 0, deck + 7000)))
        house = house.union(cq.Workplane("XY").circle(400).extrude(5000).translate((-L / 2 + 4000, 0, deck + 7000)))   # mast
        wells = cq.Workplane("XY")
        for x in self.well_positions(p):
            wells = wells.add(cq.Solid.makeBox(p["well_length"] * mm - 3000, p["well_width"] * mm - 3000, 900,
                                               cq.Vector(x * mm - (p["well_length"] * mm - 3000) / 2, -(p["well_width"] * mm - 3000) / 2, deck)))
        wells = wells.combine()
        xs = self.well_positions(p)
        xc = (xs[0] + xs[1]) / 2 * mm if len(xs) > 1 else xs[0] * mm + p["well_length"] * mm / 2 + 2000
        crane = cq.Workplane("XY").add(cq.Solid.makeCylinder(900, 6000, cq.Vector(xc, B / 2 - 2000, deck)))
        crane = crane.union(cq.Workplane("XY").add(cq.Solid.makeCylinder(450, 16000, cq.Vector(xc, B / 2 - 2000, deck + 5500),
                                                                         cq.Vector(0.55, -0.55, 0.62))))
        parts = {"hull": hull, "house": house, "wells": wells, "crane": crane}
        if p["part"] in parts:
            return parts[p["part"]]
        return hull.union(house).union(wells).union(crane)


if __name__ == "__main__":
    print(Wellboat.sizing())
