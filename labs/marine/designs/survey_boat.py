import math
import cadquery as cq
from vegeta.dedalus import Design, Parameter


class SurveyBoat(Design):
    """Small unmanned surface vessel: a hard-chine V-bottom hull lofted through stations, a flat
    transom, a deck, and a transom bracket for the motor pod. Bow at +X, waterline plane z = 0 is the
    keel at midship... no: the keel is at z = 0, up is +Z, the deck at z = depth. ``part`` selects the
    whole boat, the outer hull solid (for hydrostatics), the hull shell (for structure) or the bracket."""

    parameters = [
        Parameter("part", "boat", choices=("boat", "hull_solid", "hull_shell", "bracket"), description="what to build"),
        Parameter("length", 1000.0, "mm", min=200),
        Parameter("beam", 300.0, "mm", min=50, description="maximum beam at the deck"),
        Parameter("depth", 170.0, "mm", min=30, description="keel to deck at midship"),
        Parameter("deadrise_deg", 16.0, "deg", min=0, max=40, description="V-bottom angle at midship"),
        Parameter("bow_rise", 60.0, "mm", min=0, description="keel rise at the bow"),
        Parameter("transom_beam_ratio", 0.85, "", min=0.3, max=1.0),
        Parameter("shell", 3.0, "mm", min=1, description="hull wall thickness"),
        Parameter("bracket_height", 120.0, "mm", min=20, description="motor pod bracket below the transom"),
        Parameter("bracket_width", 60.0, "mm", min=10),
        Parameter("bracket_thickness", 8.0, "mm", min=2),
        Parameter("pod_diameter", 45.0, "mm", min=10),
        Parameter("stations", 8, min=4, max=20),
    ]

    def _station(self, p, x, inset=0.0):
        """Closed section polygon (y, z) at longitudinal position x (0 = transom, length = bow tip);
        ``inset`` moves the section inward (the inner surface of the shell) and above the deck."""
        L, B, D = p["length"], p["beam"], p["depth"]
        s = x / L                                                   # 0 transom .. 1 bow
        # beam distribution: widest at 45 % from the transom, finite at the transom, to a point at the bow
        half = 0.5 * B * max(0.03, (p["transom_beam_ratio"] + (1 - p["transom_beam_ratio"]) * math.sin(math.pi * min(s / 0.9, 1.0)) ** 0.7) * (1 - s ** 6))
        keel_z = p["bow_rise"] * max(0.0, (s - 0.55) / 0.45) ** 2
        chine_z = keel_z + half * math.tan(math.radians(p["deadrise_deg"])) * (1 + 0.8 * s)   # deadrise grows toward the bow
        chine_z = min(chine_z, D - 20)
        if inset:
            half = max(half - inset, 1.0)
            keel_z += inset / math.cos(math.radians(p["deadrise_deg"]))
            chine_z += inset * 0.9
            D = D + 10.0                                                  # open top
        return [(-half, D), (-half * 0.92, chine_z), (0.0, keel_z), (half * 0.92, chine_z), (half, D)]

    def _hull_solid(self, p, inset=0.0):
        n = p["stations"]
        wires = []
        for i in range(n):
            x = p["length"] * (i / (n - 1)) ** 0.85                      # denser stations toward the bow
            if inset:
                x = min(x, p["length"] - 2.0 * inset)                     # the inner surface stops short of the stem
            pts = [cq.Vector(x, y, z) for y, z in self._station(p, x, inset)]
            wires.append(cq.Wire.makePolygon(pts, close=True))
        return cq.Workplane("XY").add(cq.Solid.makeLoft(wires, ruled=True))

    def _bracket(self, p):
        h, w, t = p["bracket_height"], p["bracket_width"], p["bracket_thickness"]
        plate = cq.Workplane("YZ").rect(w, h + 40).extrude(t).translate((-t, 0, -h / 2 + 20))       # bolted to the transom
        pod_z = -h
        pod = cq.Workplane("YZ").center(0, pod_z).circle(p["pod_diameter"] / 2).extrude(-90).translate((-t, 0, 0))
        strut = cq.Workplane("YZ").rect(w * 0.5, h).extrude(t).translate((-t - 30, 0, -h / 2))
        holes = cq.Workplane("YZ").pushPoints([(-w / 2 + 8, 30), (w / 2 - 8, 30), (-w / 2 + 8, -h + 20), (w / 2 - 8, -h + 20)]).circle(3.2).extrude(3 * t, both=True).translate((-t, 0, 0))
        return plate.union(strut).union(pod).cut(holes)

    def build(self, p):
        if p["part"] == "bracket":
            return self._bracket(p)
        solid = self._hull_solid(p)
        if p["part"] == "hull_solid":
            return solid
        shell = solid.cut(self._hull_solid(p, inset=p["shell"]))
        if p["part"] == "hull_shell":
            return shell
        deck = (cq.Workplane("XY").workplane(offset=p["depth"] - p["shell"]).center(p["length"] / 2, 0).rect(p["length"], p["beam"])
                .extrude(p["shell"]).intersect(solid))
        bracket = self._bracket(p).translate((0, 0, p["depth"] * 0.55))
        return shell.union(deck).union(bracket)
