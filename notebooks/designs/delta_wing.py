"""A delta-wing (triangular flying-wing) delivery drone for the notebook ``16_delta_wing_delivery``.

Flow along -X: the nose is at +X, the tail at x = 0; the span is along Y; up is +Z. The wing is a swept delta
with biconvex sections, blended into a central body that holds the nose payload bay; twin fins stand on the
wing near the trailing edge. ``propulsion`` picks the tail end: a pusher-propeller boss (the propeller plane
just behind it) or a turbojet nozzle duct. ``part`` selects the aircraft, the wing alone, the payload bay
(a hollow box for the medicine) or one fin. Two sizes are the presets ``SIZE_2M`` and ``SIZE_3M``.
"""
import math

import cadquery as cq
from vegeta.dedalus import Design, Parameter

SIZE_2M = dict(span=2000.0, root_chord=1150.0, body_width=220.0, body_height=150.0, bay_length=320.0,
               fin_height=260.0, fin_chord=200.0)
SIZE_3M = dict(span=3000.0, root_chord=1650.0, body_width=300.0, body_height=200.0, bay_length=450.0,
               fin_height=360.0, fin_chord=280.0)


class DeltaWing(Design):
    """Delta flying wing with a central body, nose payload bay, twin fins; pusher propeller or jet at the tail."""

    parameters = [
        Parameter("part", "aircraft", choices=("aircraft", "wing", "half_wing", "payload_bay", "fin"),
                  description="what to build (half_wing: the +Y half, root face on y = 0, for a symmetric FEA)"),
        Parameter("propulsion", "pusher", choices=("pusher", "jet"), description="tail end: pusher boss or jet nozzle"),
        Parameter("span", 2000.0, "mm", min=500),
        Parameter("root_chord", 1150.0, "mm", min=200, description="chord at the centreline (the delta's length)"),
        Parameter("tip_chord", 60.0, "mm", min=10, description="small tip chord (a cropped delta)"),
        Parameter("thickness", 0.08, "", min=0.04, max=0.16, description="section thickness / chord (biconvex)"),
        Parameter("body_width", 220.0, "mm", min=60),
        Parameter("body_height", 150.0, "mm", min=40, description="body depth at the thickest point"),
        Parameter("body_length_ratio", 0.92, "", min=0.5, max=1.0, description="body length / root chord"),
        Parameter("bay_length", 320.0, "mm", min=50, description="payload bay length inside the nose"),
        Parameter("bay_wall", 3.0, "mm", min=1, description="payload bay wall thickness"),
        Parameter("fin_height", 260.0, "mm", min=50),
        Parameter("fin_chord", 200.0, "mm", min=30),
        Parameter("fin_thickness", 8.0, "mm", min=2),
        Parameter("fin_position", 0.32, "", min=0.15, max=0.48, description="fin from the centreline / half span"),
        Parameter("fin_cant_deg", 15.0, "deg", min=0, max=45, description="fins leaning outwards"),
        Parameter("boss_diameter", 60.0, "mm", min=10, description="pusher: spinner boss diameter"),
        Parameter("nozzle_diameter", 90.0, "mm", min=20, description="jet: nozzle exit diameter"),
        Parameter("angle_of_attack_deg", 0.0, "deg", min=-10, max=25, description="rotate nose-up about Y"),
    ]

    # -- pieces ---------------------------------------------------------------------------------------------------
    @staticmethod
    def _biconvex(wp, chord, t_ratio, x0, y0):
        """A biconvex section (circular arcs top and bottom) in a workplane, leading edge at x0 (+X), chord along -X."""
        t = t_ratio * chord
        pts_u = [(x0 - chord * s, y0 + 2 * t * s * (1 - s)) for s in [i / 12 for i in range(13)]]
        pts_l = [(x, y0 - (y - y0)) for x, y in pts_u]
        return (wp.moveTo(*pts_u[0]).spline(pts_u[1:], includeCurrent=True)
                .spline(pts_l[::-1][1:], includeCurrent=True).close())

    def _wing(self, p, half=False):
        """A cropped delta: root section at the centreline lofted to a tip section at the half span."""
        c0, ct, b2 = p["root_chord"], p["tip_chord"], p["span"] / 2
        # the trailing edge is straight at x = 0; the leading edge sweeps back from the nose (x = c0) to the tip
        wp = cq.Workplane("XZ")                                 # XZ normal is -Y: offset -b2 moves to +Y
        wp = self._biconvex(wp, c0, p["thickness"], c0, 0.0).workplane(offset=-b2)
        wp = self._biconvex(wp, ct, p["thickness"], ct, 0.0)
        half_wing = wp.loft(combine=True, ruled=True)
        return half_wing if half else half_wing.union(half_wing.mirror("XZ"))

    def _body(self, p):
        """A blended central body: an ellipsoidal spindle from the nose back to the tail, on the wing's centreline."""
        L = p["body_length_ratio"] * p["root_chord"]
        w, h = p["body_width"], p["body_height"]
        x0 = p["root_chord"]                                    # nose
        prof = [(x0 - L * s, 0.5 * math.sqrt(max(1e-6, 1 - (2 * s - 1) ** 2 * 0.85))) for s in [i / 16 for i in range(17)]]
        body = (cq.Workplane("XY").moveTo(prof[0][0], 0).spline([(x, y * w) for x, y in prof[1:]], includeCurrent=True)
                .lineTo(prof[-1][0], 0).close().revolve(360, (0, 0, 0), (1, 0, 0)))
        return cq.Workplane("XY").add(body.val().transformGeometry(          # squash the spindle to the body height
            cq.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, h / w, 0]])))

    def _fin(self, p, y=0.0, cant=0.0):
        """A swept fin standing on the wing near the trailing edge, canted outwards."""
        c, hgt, t = p["fin_chord"], p["fin_height"], p["fin_thickness"]
        fin = (cq.Workplane("XZ").polyline([(c, 0), (0, 0), (0.15 * c, hgt), (0.7 * c, hgt)]).close()
               .extrude(t / 2, both=True))
        return fin.rotate((0, 0, 0), (1, 0, 0), -cant if y >= 0 else cant).translate((0, y, -0.02 * p["root_chord"]))

    def _payload_bay(self, p):
        """A hollow box inside the nose: outer walls from the body section, open at the top hatch (a lid is separate)."""
        L, w, h, t = p["bay_length"], 0.7 * p["body_width"], 0.7 * p["body_height"], p["bay_wall"]
        x1 = p["root_chord"] - 0.12 * p["root_chord"]           # bay starts a little behind the nose tip
        outer = (cq.Workplane("XY").box(L, w, h, centered=(False, True, True)).edges("|X").fillet(min(12.0, 0.2 * h))
                 .translate((x1 - L, 0, 0)))
        inner = cq.Workplane("XY").box(L - 2 * t, w - 2 * t, h, centered=(False, True, True)).translate((x1 - L + t, 0, t))
        return outer.cut(inner)                                  # open at the top: the hatch

    def _tail(self, p):
        w, h = p["body_width"], p["body_height"]
        if p["propulsion"] == "pusher":
            d = p["boss_diameter"]
            return (cq.Workplane("YZ").circle(d / 2).extrude(-0.9 * d).faces("<X").edges().fillet(d * 0.3)
                    .translate((0.02 * p["root_chord"], 0, 0)))
        dn = p["nozzle_diameter"]
        duct = (cq.Workplane("YZ").workplane(offset=0.18 * p["root_chord"]).circle(dn / 2 + 8).extrude(-0.2 * p["root_chord"])
                .cut(cq.Workplane("YZ").workplane(offset=0.2 * p["root_chord"]).circle(dn / 2).extrude(-0.25 * p["root_chord"])))
        return duct

    # -- build ----------------------------------------------------------------------------------------------------
    def build(self, p):
        if p["part"] == "fin":
            return self._fin(p)
        if p["part"] == "payload_bay":
            return self._payload_bay(p)
        if p["part"] == "half_wing":
            return self._wing(p, half=True)
        wing = self._wing(p)
        if p["part"] == "wing":
            return wing
        yf = p["fin_position"] * p["span"] / 2
        aircraft = (wing.union(self._body(p)).union(self._fin(p, yf, p["fin_cant_deg"]))
                    .union(self._fin(p, -yf, p["fin_cant_deg"])).union(self._tail(p)))
        if p["angle_of_attack_deg"]:
            aircraft = aircraft.rotate((0, 0, 0), (0, 1, 0), p["angle_of_attack_deg"])
        return aircraft
