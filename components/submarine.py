import math
import cadquery as cq
from vegeta.dedalus import Design, Parameter


class Submarine(Design):
    """Small autonomous underwater vehicle: a Myring-type body of revolution (elliptic nose, cylindrical
    parallel middle body, power-law tail), a sail, four cruciform stern fins and a stern cone for the
    propeller. Axis +x from the tail (x = 0) to the nose; ``part`` selects the outer body (CFD, hydrostatics),
    the cylindrical pressure hull as a closed shell (FEA under external pressure), or the whole vehicle."""

    parameters = [
        Parameter("part", "vehicle", choices=("vehicle", "body", "pressure_hull"), description="what to build"),
        Parameter("length", 1200.0, "mm", min=200, description="overall length"),
        Parameter("diameter", 180.0, "mm", min=40, description="hull diameter"),
        Parameter("nose_length", 220.0, "mm", min=20),
        Parameter("tail_length", 380.0, "mm", min=20),
        Parameter("tail_exponent", 2.2, "", min=1.0, max=4.0, description="Myring tail power (2 = parabolic)"),
        Parameter("sail_length", 160.0, "mm", min=0, description="0 = no sail"),
        Parameter("sail_height", 90.0, "mm", min=10),
        Parameter("sail_width", 50.0, "mm", min=5),
        Parameter("sail_position", 0.55, "", min=0.2, max=0.8, description="sail centre as a fraction of the length from the tail"),
        Parameter("fin_span", 90.0, "mm", min=10, description="fin span beyond the hull surface"),
        Parameter("fin_chord", 100.0, "mm", min=10),
        Parameter("fin_thickness", 8.0, "mm", min=2),
        Parameter("fin_position", 120.0, "mm", min=10, description="fin trailing edge from the tail tip"),
        Parameter("shell", 6.0, "mm", min=1, description="pressure hull wall thickness"),
        Parameter("pressure_hull_length", 500.0, "mm", min=50, description="cylindrical pressure hull length"),
        Parameter("end_cap_depth", 60.0, "mm", min=5, description="depth of the domed end caps"),
        Parameter("stations", 24, min=8, max=60),
    ]

    def _radius(self, p, x):
        """Hull radius at x (0 = tail tip, length = nose tip)."""
        L, R, ln, lt, n = p["length"], p["diameter"] / 2, p["nose_length"], p["tail_length"], p["tail_exponent"]
        if x >= L - ln:                                   # elliptic nose
            s = (L - x) / ln
            return R * math.sqrt(max(0.0, 1 - (1 - s) ** 2))
        if x <= lt:                                       # power-law tail to a small stern cone radius
            s = x / lt
            return R * (0.1 + 0.9 * s ** (1 / n))
        return R

    def _body(self, p):
        L, ln, lt = p["length"], p["nose_length"], p["tail_length"]
        if ln + lt >= L:
            raise ValueError("nose_length + tail_length must be smaller than length")
        n = p["stations"]
        xs = sorted(set([lt, L - ln] + [L * i / (n - 1) for i in range(1, n - 1)] +
                        [lt * (i / 8) ** 1.5 for i in range(1, 8)] + [L - ln + ln * i / 8 for i in range(1, 8)]))
        pts = [(x, self._radius(p, x)) for x in xs if 0 < x < L]
        prof = (cq.Workplane("XY").moveTo(0, 0).lineTo(0, self._radius(p, 0.0))
                .spline(pts + [(L, 0.0)], includeCurrent=True).close())
        return prof.revolve(360, (0, 0, 0), (1, 0, 0))

    def _pressure_hull(self, p):
        """Closed cylindrical vessel with ellipsoidal end caps, as a shell of thickness ``shell``."""
        R = p["diameter"] / 2
        lp, t, d = p["pressure_hull_length"], p["shell"], p["end_cap_depth"]
        ri = R - 12.0
        if t >= min(ri, d) / 2:
            raise ValueError("shell too thick for the pressure hull radius / cap depth")

        def vessel(r, dc, x0, x1):
            # ellipsoidal caps truncated by a small flat (a penetrator plate): a distinct face to hold the vessel in an FEA
            f0 = math.asin(0.12)                                                      # flat radius = 0.12 r
            phis = [f0 + (math.pi / 2 - f0) * i / 8 for i in range(1, 9)]              # cap angle: 0 = tip, pi/2 = cylinder
            left = [(x0 - dc * math.cos(f), r * math.sin(f)) for f in phis]            # ends at (x0, r)
            right = [(x1 + dc * math.cos(f), r * math.sin(f)) for f in phis[::-1][1:] + [f0]]  # from the cylinder to the right flat
            prof = (cq.Workplane("XY").moveTo(x0 - dc * math.cos(f0), 0).lineTo(x0 - dc * math.cos(f0), r * math.sin(f0))
                    .spline(left, includeCurrent=True).lineTo(x1, r).spline(right, includeCurrent=True)
                    .lineTo(x1 + dc * math.cos(f0), 0).close())
            return prof.revolve(360, (0, 0, 0), (1, 0, 0))

        outer = vessel(ri, d, 0.0, lp)
        inner = vessel(ri - t, d - t, 0.0, lp)
        return outer.cut(inner)

    def build(self, p):
        R = p["diameter"] / 2
        if p["part"] == "pressure_hull":
            return self._pressure_hull(p)
        body = self._body(p)
        if p["part"] == "body":
            return body
        L = p["length"]
        vehicle = body
        if p["sail_length"] > 0:
            xc = p["sail_position"] * L
            sail = (cq.Workplane("XY").workplane(offset=0)
                    .center(xc, 0).ellipse(p["sail_length"] / 2, p["sail_width"] / 2).extrude(R + p["sail_height"]))
            vehicle = vehicle.union(sail)
        xf = p["fin_position"]
        for k in range(4):
            ang = 90 * k
            fin = (cq.Workplane("XY").polyline([(xf, 0), (xf + p["fin_chord"], 0), (xf + p["fin_chord"] * 0.75, R + p["fin_span"]),
                                                (xf + p["fin_chord"] * 0.35, R + p["fin_span"])]).close()
                   .extrude(p["fin_thickness"] / 2, both=True))
            vehicle = vehicle.union(fin.rotate((0, 0, 0), (1, 0, 0), ang))
        return vehicle
