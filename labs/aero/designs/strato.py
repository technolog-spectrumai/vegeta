"""Strato — a solar-electric, helium-filled stratospheric airship (HAPS), the CAD (Stage V8, ``todo.md``).

The hull profile and every part's size come from ``strato_geometry`` (metres, fractions of the hull), so the
15 m × 3.5 m demonstrator and the 30–80 m ships are one drawing at different scales. Parameters are in metres (SI);
the solid is built in millimetres, Dedalus's CAD unit, so STEP files open at the right size.

Frame: hull axis along +X from the nose tip (+X aft), Z up, Y to starboard. ``part`` selects what is built:

- ``airship``     everything, one solid (pictures, CFD surface, mass properties of the shapes)
- ``hull``        the helium volume (the envelope's inside; the envelope itself is a 0.1–0.3 mm laminate)
- ``fins``        the four X fins
- ``gondola``     the payload bay under the hull with its two pylons
- ``battery``     the battery block inside the gondola (forward part)
- ``solar``       the conformal solar array on top of the hull (drawn 0.004·D thick to be visible)
- ``propellers``  the stern pusher, the two vectoring propellers, their nacelles and outriggers
- ``structure``   the nose-cone battens and the bottom keel
- ``tether``      confluence lines from four hull patches to the tether point below, and a stub of the tether
  (``with_tether``; Strato-Micro's aerostat)

The parts are placed where ``strato_geometry.Layout`` puts them; they are not sized by physics yet.
"""
from __future__ import annotations

import math

import cadquery as cq
from vegeta.dedalus import Design, Parameter

import strato_geometry as sg

PARTS = ("airship", "hull", "fins", "gondola", "battery", "solar", "propellers", "structure", "tether")
OPTIONAL = ("solar", "propellers", "structure", "tether")   # parts a variant can leave out (``with_*`` parameters)
MM = 1000.0
_N_PROFILE = 61                   # spline points along the hull profile
_TAIL_CUT = 0.04                  # the tail is closed where r = _TAIL_CUT·R (the stern hub covers the point)


class StratoAirship(Design):
    name = "strato"
    parameters = [
        Parameter("part", "airship", choices=PARTS, description="what to build"),
        Parameter("length", 15.0, "m", min=5.0, max=300.0, description="hull length"),
        Parameter("diameter", 3.5, "m", min=0.5, max=80.0, description="maximum hull diameter"),
        Parameter("x_max", 0.40, "", min=0.2, max=0.6, description="station of the maximum diameter / length"),
        Parameter("n_nose", 2.0, "", min=1.2, max=4.0, description="nose super-ellipse exponent"),
        Parameter("n_tail", 2.0, "", min=1.2, max=4.0, description="tail exponent"),
        Parameter("solar_half_angle_deg", 55.0, "deg", min=5.0, max=90.0, description="array half-width about the top"),
        Parameter("solar_x0", 0.18, "", min=0.0, max=1.0, description="array start / length"),
        Parameter("solar_x1", 0.72, "", min=0.0, max=1.0, description="array end / length"),
        Parameter("fin_span", 0.55, "", min=0.1, max=2.0, description="fin span / diameter"),
        Parameter("fin_root_chord", 0.14, "", min=0.02, max=0.4, description="fin root chord / length"),
        Parameter("with_solar", True, "", description="the solar array (Strato-Micro has none)"),
        Parameter("with_propellers", True, "", description="propulsion (a tethered aerostat has none)"),
        Parameter("with_structure", True, "", description="nose battens and keel"),
        Parameter("with_tether", False, "", description="confluence lines and the tether stub (a tethered aerostat)"),
    ]

    # ---------------------------------------------------------------- model
    def model(self, p) -> sg.Strato:
        lay = sg.Layout(solar_half_angle_deg=p["solar_half_angle_deg"], solar_x0=p["solar_x0"],
                        solar_x1=p["solar_x1"], fin_span=p["fin_span"], fin_root_chord=p["fin_root_chord"],
                        fin_tip_chord=0.5 * p["fin_root_chord"])
        return sg.Strato(sg.Hull(p["length"], p["diameter"], p["x_max"], p["n_nose"], p["n_tail"]), lay)

    # ---------------------------------------------------------------- bodies of revolution (mm)
    @staticmethod
    def _profile_points(hull: sg.Hull, x0: float, x1: float, offset: float = 0.0, n: int = _N_PROFILE):
        """(x, r) in mm along the hull between stations ``x0`` and ``x1`` [m], ``offset`` [m] outside the skin."""
        import numpy as np
        u = 0.5 * (1 - np.cos(np.linspace(0.0, math.pi, n)))
        x = x0 + (x1 - x0) * u
        r = hull.r(x) + offset
        return [(float(a * MM), float(b * MM)) for a, b in zip(x, r)]

    def _tail_end(self, hull: sg.Hull) -> float:
        """Station [m] where the tail radius falls to _TAIL_CUT·R."""
        xm = hull.x_max * hull.length
        return xm + (hull.length - xm) * (1 - _TAIL_CUT) ** (1 / hull.n_tail)

    def _hull(self, hull: sg.Hull):
        pts = self._profile_points(hull, 0.0, self._tail_end(hull))
        wp = cq.Workplane("XY").moveTo(0.0, 0.0).spline(pts[1:], includeCurrent=True)
        wp = wp.lineTo(pts[-1][0], 0.0).close()
        return wp.revolve(360, (0, 0, 0), (1, 0, 0))

    def _band(self, hull: sg.Hull, x0: float, x1: float, thickness: float, centre_deg: float, half_deg: float,
              inset: float = 0.0):
        """A conformal band on the hull between stations ``x0`` and ``x1`` [m], ``thickness`` [m] proud of the skin
        (from ``inset`` below it), ±``half_deg`` about ``centre_deg`` (0 = top, 90 = starboard)."""
        outer = self._profile_points(hull, x0, x1, thickness)
        inner = self._profile_points(hull, x0, x1, -inset)[::-1]
        wp = cq.Workplane("XY").moveTo(*inner[-1]).spline(outer[1:], includeCurrent=True)
        wp = wp.lineTo(*inner[0]).spline(inner[1:], includeCurrent=True).close()
        # revolving about +X sweeps +Y towards +Z; start at the band's first edge measured from +Y
        band = wp.revolve(2 * half_deg, (0, 0, 0), (1, 0, 0))
        return band.rotate((0, 0, 0), (1, 0, 0), 90.0 - half_deg - centre_deg)

    # ---------------------------------------------------------------- appendages (mm)
    def _fin(self, m: sg.Strato, roll_deg: float):
        h, lay = m.hull, m.layout
        L, D = h.length, h.diameter
        c_root, c_tip = lay.fin_root_chord * L, lay.fin_tip_chord * L
        x_te = lay.fin_trailing_edge * L
        x_le = x_te - c_root
        span = lay.fin_span * D
        r_root = float(h.r([x_le + 0.5 * c_root])[0])
        z0 = 0.4 * r_root                                   # the root runs into the hull (the union trims it)
        z_tip = r_root + span
        x_le_tip = x_le + (z_tip - r_root) * math.tan(math.radians(lay.fin_sweep_deg))
        t = lay.fin_thickness * c_root
        poly = [(x_le * MM, z0 * MM), (x_te * MM, z0 * MM), ((x_le_tip + c_tip) * MM, z_tip * MM), (x_le_tip * MM, z_tip * MM)]
        fin = cq.Workplane("XZ").polyline(poly).close().extrude(t * MM / 2, both=True)
        try:
            fin = fin.edges("|Y").fillet(0.2 * t * MM)
        except Exception:                                   # a fillet that OCC cannot place is cosmetic only
            pass
        return fin.rotate((0, 0, 0), (1, 0, 0), roll_deg)

    def _gondola(self, m: sg.Strato, battery: bool = False):
        h, lay = m.hull, m.layout
        L, D = h.length, h.diameter
        xc = h.centre_of_buoyancy
        lg, wg, hg = lay.gondola_length * L, lay.gondola_width * D, lay.gondola_height * D
        r_c = float(h.r([xc])[0])
        z_top = -(r_c + lay.gondola_drop * D)
        if battery:
            lb = lay.battery_fraction * lg
            w = 0.1 * hg
            return (cq.Workplane("XY").box((lb - 2 * w) * MM, (wg - 2 * w) * MM, (hg - 2 * w) * MM)
                    .translate(((xc - 0.5 * lg + 0.5 * lb) * MM, 0, (z_top - 0.5 * hg) * MM)))
        body = (cq.Workplane("XY").box(lg * MM, wg * MM, hg * MM).edges().fillet(0.3 * min(wg, hg) * MM)
                .translate((xc * MM, 0, (z_top - 0.5 * hg) * MM)))
        rod = 0.025 * D
        for dx in (-0.3 * lg, 0.3 * lg):
            body = body.union(cq.Workplane("XY").circle(rod * MM).extrude((lay.gondola_drop * D + 0.15 * r_c) * MM)
                              .translate(((xc + dx) * MM, 0, z_top * MM)))
        return body

    @staticmethod
    def _rotor(diameter: float, blades: int = 2, pitch_deg: float = 20.0):
        """A propeller in the YZ plane at the origin (spins about X), mm: hub and flat elliptic blades at pitch."""
        R = 0.5 * diameter * MM
        hub = cq.Workplane("YZ").circle(0.12 * R).extrude(0.18 * R, both=True)
        chord, t = 0.16 * R, 0.025 * R
        rotor = hub
        for k in range(blades):
            blade = (cq.Workplane("XY").ellipse(0.5 * t, 0.5 * chord).extrude(0.95 * R)   # along +Z
                     .rotate((0, 0, 0), (0, 0, 1), pitch_deg)
                     .rotate((0, 0, 0), (1, 0, 0), 360.0 * k / blades))
            rotor = rotor.union(blade)
        return rotor

    def _propellers(self, m: sg.Strato):
        h, lay = m.hull, m.layout
        L, D = h.length, h.diameter
        x_end = self._tail_end(h)
        r_end = float(h.r([x_end])[0])
        d_stern = lay.stern_prop_diameter * D
        # stern: a hub cone closing the tail, a short shaft, the pusher
        cone = (cq.Workplane("YZ").workplane(offset=(x_end - 0.02 * L) * MM).circle(1.4 * r_end * MM)
                .workplane(offset=0.05 * L * MM).circle(0.4 * r_end * MM).loft())
        out = cone.union(self._rotor(d_stern, blades=3).translate(((x_end + 0.03 * L) * MM, 0, 0)))   # hub on the cone tip
        # vectoring propellers: outriggers at the side, nacelles, tractor rotors
        xs = lay.side_prop_station * L
        r_s = float(h.r([xs])[0])
        reach = r_s + lay.side_prop_outboard * D
        d_side = lay.side_prop_diameter * D
        nac_r, nac_l = 0.08 * d_side, 0.35 * d_side
        for side in (1, -1)[: lay.side_prop_count]:
            arm = (cq.Workplane("XZ").circle(0.02 * D * MM).extrude(-(reach - 0.6 * r_s) * MM)
                   .translate((xs * MM, 0.6 * r_s * MM, 0)))
            if side < 0:
                arm = arm.mirror("XZ")
            nacelle = (cq.Workplane("YZ").circle(nac_r * MM).extrude(nac_l * MM)
                       .translate(((xs - 0.5 * nac_l) * MM, side * reach * MM, 0)))
            rotor = self._rotor(d_side).translate(((xs - 0.5 * nac_l - 0.02 * d_side) * MM, side * reach * MM, 0))
            out = out.union(arm).union(nacelle).union(rotor)
        return out

    def _structure(self, m: sg.Strato):
        h, lay = m.hull, m.layout
        L, D = h.length, h.diameter
        t = 0.004 * D
        out = None
        for k in range(lay.batten_count):
            b = self._band(h, 0.004 * L, lay.batten_length * L, t, 360.0 * k / lay.batten_count, 1.2, inset=t)
            out = b if out is None else out.union(b)
        keel = self._band(h, lay.keel_x0 * L, lay.keel_x1 * L, 2 * t, 180.0, 4.0, inset=2 * t)
        return out.union(keel)

    def _solar(self, m: sg.Strato):
        h, lay = m.hull, m.layout
        t = 0.004 * h.diameter                             # drawn thickness (14 mm on the 15 m ship) so it shows; the
                                                           # real thin-film array is ~1 mm and its mass is budgeted, not drawn
        return self._band(h, lay.solar_x0 * h.length, lay.solar_x1 * h.length, t, 0.0,
                          lay.solar_half_angle_deg, inset=0.5 * t)

    # ---------------------------------------------------------------- build
    def build(self, p):
        if p["solar_x1"] <= p["solar_x0"]:
            raise ValueError("solar_x1 must be aft of solar_x0")
        m = self.model(p)                                  # raises ValueError for an impossible hull
        part = p["part"]
        builders = {"hull": lambda: self._hull(m.hull),
                    "fins": lambda: self._fins(m),
                    "gondola": lambda: self._gondola(m),
                    "battery": lambda: self._gondola(m, battery=True),
                    "solar": lambda: self._solar(m),
                    "propellers": lambda: self._propellers(m),
                    "structure": lambda: self._structure(m),
                    "tether": lambda: self._tether(m)}
        if part != "airship":
            if part in OPTIONAL and not p[f"with_{part}"]:
                raise ValueError(f"part {part!r} is switched off (with_{part}=False)")
            return builders[part]()
        out = builders["hull"]()
        for name in ("fins", "gondola", "solar", "propellers", "structure", "tether"):
            if name in OPTIONAL and not p[f"with_{name}"]:
                continue
            out = out.union(builders[name]())
        return out

    def _tether(self, m: sg.Strato):
        """Four confluence lines from patches at 30 % and 60 % of the length, ±40° below the side, to the tether
        point 0.7 D under the centre of buoyancy, and 0.5 D of tether below it (drawn 0.6 % of D thick)."""
        h = m.hull
        D, xc = h.diameter, h.centre_of_buoyancy
        conf = cq.Vector(xc * MM, 0.0, -(h.radius + 0.7 * D) * MM)
        rod = 0.003 * D * MM
        out = None
        for xf in (0.30, 0.60):
            x = xf * h.length
            r = float(h.r([x])[0])
            for side in (1, -1):
                a = math.radians(40.0)
                top = cq.Vector(x * MM, side * r * math.sin(a) * MM, -r * math.cos(a) * MM)
                v = conf - top
                line = cq.Workplane("XY").add(cq.Solid.makeCylinder(rod, v.Length, top, v.normalized()))
                out = line if out is None else out.union(line)
        stub = cq.Workplane("XY").add(cq.Solid.makeCylinder(1.5 * rod, 0.5 * D * MM, conf, cq.Vector(0, 0, -1)))
        ring = cq.Workplane("XY").add(cq.Solid.makeSphere(4 * rod, conf))
        return out.union(stub).union(ring)

    def _fins(self, m: sg.Strato):
        lay = m.layout
        out = None
        for k in range(lay.fin_count):
            f = self._fin(m, lay.fin_roll_deg + 360.0 * k / lay.fin_count)
            out = f if out is None else out.union(f)
        return out


def parts(**overrides) -> dict:
    """Every part as its own ``Geometry`` (the coloured assembly views and the per-part STEP files)."""
    d = StratoAirship()
    p = d.resolve(**overrides)
    return {name: d.generate(part=name, **overrides) for name in PARTS
            if name != "airship" and (name not in OPTIONAL or p[f"with_{name}"])}


def for_length(length: float, fineness: float = 15.0 / 3.5) -> dict:
    """The parameter overrides of the demonstrator's drawing scaled to ``length`` [m]."""
    return {"length": float(length), "diameter": float(length / fineness)}
