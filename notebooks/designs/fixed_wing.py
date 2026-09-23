import math
import cadquery as cq
from vegeta.dedalus import Design, Parameter


def naca4(camber, camber_pos, thickness, n=40, te_cut=0.97):
    """NACA 4-digit section as (upper, lower) point lists from the trailing edge to the leading edge,
    chord 1, truncated at ``te_cut`` so the trailing edge has a small finite thickness."""
    m, p, t = camber, camber_pos, thickness
    xs = [te_cut * 0.5 * (1 - math.cos(math.pi * i / n)) for i in range(n + 1)]   # cosine spacing, LE dense
    upper, lower = [], []
    for x in xs:
        yt = 5 * t * (0.2969 * math.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
        if p > 0 and x < p:
            yc, dyc = m / p**2 * (2 * p * x - x**2), 2 * m / p**2 * (p - x)
        elif p > 0:
            yc, dyc = m / (1 - p)**2 * (1 - 2 * p + 2 * p * x - x**2), 2 * m / (1 - p)**2 * (p - x)
        else:
            yc, dyc = 0.0, 0.0
        th = math.atan(dyc)
        upper.append((x - yt * math.sin(th), yc + yt * math.cos(th)))
        lower.append((x + yt * math.sin(th), yc - yt * math.cos(th)))
    return upper[::-1], lower[::-1]          # both start at the trailing edge, end at the leading edge


class FixedWing(Design):
    """Twin-motor fixed-wing drone: NACA-section tapered wing with dihedral, streamlined fuselage,
    two wing-mounted nacelles (mirror symmetric) and a flat-plate tail. Wing root leading edge at the
    origin, chord along +X, span along Y, up is +Z. ``part`` selects what is built."""

    parameters = [
        Parameter("part", "aircraft", choices=("aircraft", "wing", "nacelle"), description="what to build"),
        Parameter("span", 1000.0, "mm", min=200),
        Parameter("root_chord", 200.0, "mm", min=50),
        Parameter("taper", 0.7, "", min=0.3, max=1.0, description="tip chord / root chord"),
        Parameter("dihedral_deg", 3.0, "deg", min=0, max=15),
        Parameter("camber", 0.02, "", min=0, max=0.09, description="NACA max camber (2 -> 0.02)"),
        Parameter("camber_pos", 0.4, "", min=0.1, max=0.9, description="NACA camber position (4 -> 0.4)"),
        Parameter("thickness", 0.12, "", min=0.06, max=0.25, description="NACA thickness (12 -> 0.12)"),
        Parameter("fuselage_length", 620.0, "mm", min=100),
        Parameter("fuselage_diameter", 70.0, "mm", min=20),
        Parameter("nose_length", 160.0, "mm", min=20, description="fuselage ahead of the wing leading edge"),
        Parameter("nacelle_y", 300.0, "mm", min=50, description="nacelle centre from the symmetry plane"),
        Parameter("nacelle_diameter", 34.0, "mm", min=10),
        Parameter("nacelle_length", 90.0, "mm", min=20),
        Parameter("nacelle_forward", 45.0, "mm", min=5, description="nacelle ahead of the leading edge"),
        Parameter("tail_span", 360.0, "mm", min=50),
        Parameter("tail_chord", 110.0, "mm", min=20),
        Parameter("fin_height", 130.0, "mm", min=20),
        Parameter("tail_thickness", 5.0, "mm", min=1),
        Parameter("angle_of_attack_deg", 0.0, "deg", min=-10, max=20, description="rotate the aircraft nose-up about Y"),
    ]

    def _sections(self, p):
        up, lo = naca4(p["camber"], p["camber_pos"], p["thickness"])

        def section(wp, c, dx, dz):
            pts_u = [(dx + x * c, dz + y * c) for x, y in up]
            pts_l = [(dx + x * c, dz + y * c) for x, y in lo]
            return (wp.moveTo(*pts_u[0]).spline(pts_u[1:], includeCurrent=True)
                    .spline(pts_l[::-1][1:], includeCurrent=True).close())
        return section

    def _wing(self, p):
        """Constant-chord centre section (inside the fuselage) plus two tapered outer panels with dihedral.
        The centre section has its own faces, so it can be selected as the clamped region in an FEA."""
        c0, c1 = p["root_chord"], p["root_chord"] * p["taper"]
        yc, b2 = p["fuselage_diameter"] / 2 + 5.0, p["span"] / 2
        if b2 <= yc + 10:
            raise ValueError("span too small for the fuselage diameter")
        section = self._sections(p)
        sweep = (c0 - c1) / 4                                  # straight quarter-chord line
        dz = (b2 - yc) * math.tan(math.radians(p["dihedral_deg"]))
        # XZ workplane: normal is -Y, so positive offsets go toward -Y; build the -Y panel and mirror it
        centre = section(cq.Workplane("XZ").workplane(offset=-yc), c0, 0.0, 0.0).extrude(2 * yc)
        panel = (section(cq.Workplane("XZ").workplane(offset=yc), c0, 0.0, 0.0)
                 .workplane(offset=b2 - yc).moveTo(0, 0))
        panel = section(panel, c1, sweep, dz).loft(combine=True, ruled=True)
        return centre.union(panel).union(panel.mirror("XZ"))

    def _nacelle(self, p, y):
        d, L, f = p["nacelle_diameter"], p["nacelle_length"], p["nacelle_forward"]
        body = (cq.Workplane("YZ").workplane(offset=-f).center(y, 0.0).circle(d / 2).extrude(L)
                .faces(">X").edges().fillet(d * 0.3).faces("<X").edges().chamfer(1.5))
        return body

    def build(self, p):
        if p["part"] == "nacelle":
            return self._nacelle(p, 0.0)
        wing = self._wing(p)
        wing = wing.union(self._nacelle(p, -p["nacelle_y"])).union(self._nacelle(p, p["nacelle_y"]))
        if p["part"] == "wing":
            return wing
        # fuselage: elliptic nose, cylinder, tapered tail; axis at z = -D/2 so the wing sits on top
        D, Lf, ln = p["fuselage_diameter"], p["fuselage_length"], p["nose_length"]
        r = D / 2
        lt = 0.45 * Lf
        x0 = -ln
        nose = [(x0 + ln * 0.5 * (1 - math.cos(a)), r * math.sin(a)) for a in (i * math.pi / 2 / 8 for i in range(9))]
        prof = (cq.Workplane("XY").moveTo(x0, 0).spline(nose[1:], includeCurrent=True)
                .lineTo(x0 + Lf - lt, r).lineTo(x0 + Lf, 0.12 * r).lineTo(x0 + Lf, 0).close())
        fuselage = prof.revolve(360, (0, 0, 0), (1, 0, 0)).translate((0, 0, -r))
        # tail: flat plates at the end of the fuselage
        xt = x0 + Lf - p["tail_chord"]
        tt = p["tail_thickness"]
        hstab = cq.Workplane("XY").box(p["tail_chord"], p["tail_span"], tt, centered=(False, True, True)).translate((xt, 0, -r))
        fin = cq.Workplane("XY").box(p["tail_chord"], tt, p["fin_height"], centered=(False, True, False)).translate((xt, 0, -r))
        aircraft = fuselage.union(wing).union(hstab).union(fin)
        if p["angle_of_attack_deg"]:
            aircraft = aircraft.rotate((0, 0, 0), (0, 1, 0), p["angle_of_attack_deg"])    # right-hand rule about +Y: tail down, nose up
        return aircraft
