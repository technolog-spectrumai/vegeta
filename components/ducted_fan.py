"""Electric ducted fan (EDF) — the rotor and its standing parts: the geometry (from ``notebooks/designs/ducted_fan.py``,
notebook 25b).

A small rotor of many blades running in a duct, a row of stator vanes behind it that takes the swirl out, a centre
body (the motor housing) and a convergent nozzle.

- ``EDFRotor``   the fan rotor in the CAD frame of ``dedalus.examples.Propeller`` (axis Z, blade 1 along +X, units mm):
  the Propeller's hub and blades (the same constant-pitch planform), the blade tips trimmed to the cylinder
  ``r = diameter/2`` and an elliptic spinner on the hub's upstream face. Turned +90 deg about +Y (Z -> X, X -> -Z) it
  stands in the CFD frame below with the flow along +X, so UPSTREAM in this CAD frame is -Z.
- ``EDFHousing`` the standing parts in the CFD frame of Aeromant's rotor templates (as ``air_propeller.PropPod``):
  the axis is +x through the origin = the rotor centre, the flow comes along +x, units mm. The duct, the centre body
  and the radial stator vanes between them, as one solid.
- ``rotor_trailing_edge`` where the rotor's blades end downstream (the housing's layout needs it).

Angles about the axis: ``phi`` from +y towards +z, stator vane k at ``phi = 360 k / stator_vanes``, so vane 0 stands
along +y.

Geometry only: ``housing_geometry`` (the areas and lengths the performance and noise models take) and ``outline`` (the
polygons for the particle movies) stay in ``notebooks/designs/ducted_fan.py``.
"""
import math

import cadquery as cq
import numpy as np
from vegeta.dedalus import Design, Parameter
from vegeta.dedalus.examples import Propeller

# The rotor parameters the housing needs too (for the shroud, the centre body and the rotor's trailing-edge plane):
# pass the rotor's values under these same names to EDFHousing.
SHARED = ("diameter", "hub_diameter", "hub_height", "pitch", "chord_root", "chord_max", "chord_tip", "thickness",
          "camber", "stations")

_ROTOR = [
    Parameter("diameter", 90.0, "mm", min=20, description="rotor tip diameter (the blade tips are trimmed to it)"),
    Parameter("hub_diameter", 40.0, "mm", min=2, description="hub (and centre body) diameter; hub/tip 0.45"),
    Parameter("hub_height", 14.0, "mm", min=1, description="the hub's axial length, centred on the rotor plane"),
    Parameter("pitch", 160.0, "mm", min=1, description="constant geometric pitch (advance per turn)"),
    Parameter("chord_root", 14.0, "mm", min=1),
    Parameter("chord_max", 15.0, "mm", min=1, description="at 35 % of the blade (Propeller's planform)"),
    Parameter("chord_tip", 12.0, "mm", min=0.5),
    Parameter("thickness", 0.08, "", min=0.04, max=0.3, description="blade section thickness / chord"),
    Parameter("camber", 0.04, "", min=0.0, max=0.12, description="blade section camber / chord"),
    Parameter("stations", 6, min=4, max=30, description="lofted blade sections from root to tip"),
]


def _naca00(t, n=24):
    """Closed symmetric NACA 4-digit section, unit chord, leading edge at 0, as (x, y) points: cut at 99 % (a finite
    trailing edge) and stretched back to the unit chord. As ``air_propeller._naca00``, kept here so the design files
    stay independent."""
    xs = [0.99 * 0.5 * (1 - math.cos(math.pi * i / n)) for i in range(n + 1)]
    yt = [5 * t * (0.2969 * math.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1015 * x ** 4) for x in xs]
    up = [(x / 0.99, y) for x, y in zip(xs, yt)]
    lo = [(x / 0.99, -y) for x, y in zip(xs, yt)]
    return up[::-1] + lo[1:]


def rotor_trailing_edge(p) -> dict:
    """Where the rotor's blades end downstream, from the same station geometry ``dedalus.examples.Propeller`` lofts:
    station ``i`` at ``x = i / (stations - 1)``, radius ``r = 0.9 r_hub + (R - 0.9 r_hub) x``, chord (root -> max at
    35 % -> tip, ``^1.5`` taper), pitch angle ``beta = atan(pitch / (2 pi r))``; the section (NACA 4-digit, pivoted at
    30 % chord) turned by ``beta`` puts a chord point ``(Y, Z)`` at the axial position ``z = -Y sin(beta) + Z cos(beta)``
    (Y = 0.3 c - u along the chord, Z = -v across). Returns the most downstream point over all stations (``te_x``, mm
    behind the rotor plane: the trailing-edge plane) and over the stations at or inside the hub radius (``root_te_x``:
    the blade root's stub behind the hub face — the root station sits at 0.9 r_hub, inside the hub radially, but its
    pitched section is longer axially than the hub is), plus the per-station values. The steepest (root) station
    reaches furthest back and the loft between stations does not bulge past it: for the context EDF the CAD's bounding
    box agrees to 0.01 mm."""
    R, r0, n = p["diameter"] / 2, p["hub_diameter"] / 2, int(p["stations"])
    r_st, te = [], []
    for i in range(n):
        x = i / (n - 1)
        r = r0 * 0.9 + (R - r0 * 0.9) * x
        if x < 0.35:
            c = p["chord_root"] + (p["chord_max"] - p["chord_root"]) * x / 0.35
        else:
            c = p["chord_max"] + (p["chord_tip"] - p["chord_max"]) * ((x - 0.35) / 0.65) ** 1.5
        beta = math.atan(p["pitch"] / (2 * math.pi * r))
        sec = np.array(Propeller._section(c, p["thickness"], p["camber"]))
        Y, Z = 0.3 * c - sec[:, 0], -sec[:, 1]
        r_st.append(r)
        te.append(float(np.max(-Y * math.sin(beta) + Z * math.cos(beta))))
    r_st, te = np.array(r_st), np.array(te)
    return {"te_x": float(te.max()), "root_te_x": float(te[r_st <= r0].max()), "station_r": r_st, "station_te_x": te}


class EDFRotor(Design):
    """The fan rotor: ``dedalus.examples.Propeller`` (hub cylinder, ``blades`` lofted NACA-4 sections of constant
    geometric pitch, a shaft bore) with the blade tips trimmed to the cylinder ``r = diameter/2`` — the Propeller's tip
    is a flat section whose corners reach ``sqrt(R^2 + (0.68 c cos(beta))^2)``, about R + 0.6 mm here, more than a
    0.5 mm tip clearance; trimmed, the clearance to the shroud is the same along the whole tip chord — and an elliptic
    spinner (``r = r_hub sin(a)``, ``z = -hub_height/2 - spinner_length cos(a)``, ``a`` 0 -> 90 deg: blunt at the nose,
    tangent to the hub at its base) on the hub's upstream (-Z) face; with a spinner the bore is blind, open on the
    downstream face. One closed solid. Building 12 blades takes ~7 s."""

    parameters = [Parameter("blades", 12, min=1, max=16)] + _ROTOR + [
        Parameter("bore", 4.0, "mm", min=0, description="shaft hole from the downstream face (0 = none)"),
        Parameter("skew_deg", 0.0, "deg", min=0.0, max=60.0, description="tip skew (Propeller's)"),
        Parameter("spinner_length", 22.0, "mm", min=0, description="the nose cone ahead of the hub (0 = none)"),
    ]

    def build(self, p):
        prop = Propeller()
        kw = {k: p[k] for k in prop.params.defaults}
        R, r0, h = p["diameter"] / 2, p["hub_diameter"] / 2, p["hub_height"]
        # the Propeller with one blade and no bore, less its hub cylinder = blade 1 alone (with the root stub behind
        # the hub face), its tip trimmed to r = R; the copies about Z are where Propeller puts blades 2..B. One fuse of
        # hub, blades and spinner takes about half the time of the Propeller's blade-by-blade unions, same solid.
        one = prop.build(prop.resolve(**dict(kw, blades=1, bore=0.0))).val()
        hub = cq.Solid.makeCylinder(r0, h, cq.Vector(0, 0, -h / 2))
        span = 2 * h + 4 * p["pitch"] + 50                              # longer than the rotor axially
        blade = one.cut(hub).intersect(cq.Solid.makeCylinder(R, span, cq.Vector(0, 0, -span / 2)))
        n = int(p["blades"])
        parts = [blade.rotate((0, 0, 0), (0, 0, 1), 360.0 * k / n) for k in range(n)]
        if p["bore"] > 0:
            hub = hub.cut(cq.Solid.makeCylinder(p["bore"] / 2, h + 2, cq.Vector(0, 0, -h / 2 - 1)))
        L = p["spinner_length"]
        if L > 0:
            zb = -h / 2
            nose = [(r0 * math.sin(a), zb - L * math.cos(a)) for a in (i * math.pi / 16 for i in range(9))]
            # the half profile (r, z) in the XZ plane (local y = global +Z), revolved about the Z axis
            spinner = (cq.Workplane("XZ").moveTo(0, zb - L)
                       .spline(nose[1:], tangents=[(1, 0), (0, 1)], includeCurrent=True)
                       .lineTo(0, zb).close().revolve(360, (0, 0, 0), (0, 1, 0)))
            parts.append(spinner.val())
        return hub.fuse(*parts).clean()


class EDFHousing(Design):
    """The standing parts of the EDF in the CFD frame (axis +x, rotor centre at the origin, flow along +x, mm).

    - duct: a body of revolution, ``t = wall_thickness`` thick. Its inner (flow) surface: the shroud cylinder
      ``R_s = diameter/2 + tip_clearance`` over the rotor and the stators; ahead of it the bell-mouth, an arc of
      ``lip_radius`` tangent to the shroud; from ``tail_gap`` behind the stators' trailing edge a convergent nozzle to
      the exit at ``x = duct_length`` (below). Its outside: the nacelle cylinder ``R_s + t`` from the lip back to the
      nozzle, then the nozzle wall offset by ``t`` (``r + t sqrt(1 + r'^2)``: ``t`` across the wall), flat at the exit.
      The lip is a thick, rounded lip, convex all round: a nose circle of ``lip_nose_radius`` tangent to the nacelle
      cylinder and (inside it) to the bell-mouth arc joins the two. With the bell arc's centre at ``(x_s, R_s + lip_radius)``
      and the nose's at ``(x_s - dx, R_s + t - lip_nose_radius)``, tangency (centres ``lip_radius - lip_nose_radius``
      apart) gives ``dx = sqrt((t - 2 lip_nose_radius)(2 lip_radius - t))``; the highlight (the lip's foremost point)
      is at ``x = -lip_length``, ``r = R_s + t - lip_nose_radius``. Nothing stands proud of the nacelle (its largest
      radius is ``R_s + t`` unless the nozzle widens) and the wall bounds the bell-mouth's flare: the bell arc rises
      from the shroud only until it meets the nose (1.1 mm, the highlight 1.75 mm above the shroud, for the context EDF).
    - nozzle, area-ruled: the annulus between the duct and the centre body falls as
      ``A(s) = A_e + (A_0 - A_e)(1 + cos(pi s))/2`` (s 0 -> 1 from the nozzle's start to the exit, level at both ends)
      from the shroud's ``A_0 = pi (R_s^2 - r_hub^2)`` to ``A_e = exit_area_ratio pi (R^2 - r_hub^2)`` (R the rotor tip
      radius: the exit annulus over the fan annulus), and the duct's radius follows from it and the tail cone's radius
      ``r``: ``R_d = sqrt(A/pi + r^2)``, ``R_d' = (A'/pi + 2 r r') / (2 R_d)``. The flow area thus falls monotonically and
      the exit is the nozzle's throat (for ``A_e < A_0``, i.e. exit_area_ratio below 1.045 here; above that the nozzle
      diffuses); where the tail cone still narrows at the exit the duct wall converges there too (11 deg here).
    - centre body (the motor housing): diameter ``hub_diameter``, a flat face ``axial_gap`` behind the rotor's
      downstream end — the hub face at ``hub_height/2`` or, where it reaches further, the blades' root stub
      (``rotor_trailing_edge``'s ``root_te_x``: the root section, pitched, is longer than the hub; 7.77 against 7 mm
      here) — so ``axial_gap`` is the free gap between the turning and the standing parts; cylindrical past the stators
      to the nozzle's start, then a pointed tail cone ``r = r_hub (1 - s^2)`` (s 0 -> 1 over ``tail_length``: level at its
      start; ``r r' -> 0`` at its tip, so the duct's radius stays smooth where the cone ends inside the nozzle).
    - stator: ``stator_vanes`` radial vanes, symmetric NACA 00xx (``stator_thickness``) of ``stator_chord``, leading edge
      ``stator_gap`` behind the rotor's trailing-edge plane (``rotor_trailing_edge``), set at ``stagger_deg`` about
      their leading edge (positive turns the trailing edge towards +phi, the rotor's turning sense — the way the swirl
      it meets goes). Each is its section as a prism along the vane's radial direction, from the axis out past the
      duct wall, cut to the annulus ``r_hub - 1 <= r <= R_s + 0.4 t``: at any stagger both ends follow the curved
      surfaces, 1 mm into the centre body and ``0.4 t`` into the duct wall, so the three parts are one solid (the
      staggered section, seen along the axis, must stay narrower than ``r_hub - 1`` on either side of the vane's line).

    The rotor's blade parameters (``SHARED``) place the trailing-edge plane and the body's face; pass the rotor's
    values."""

    parameters = [
        _ROTOR[0], Parameter("tip_clearance", 0.8, "mm", min=0, description="radial gap, blade tips to shroud"),
        *_ROTOR[1:],
        Parameter("axial_gap", 1.0, "mm", min=0.1,
                  description="free gap from the rotor's downstream end (hub face or blade-root stub) to the centre body"),
        Parameter("lip_length", 31.5, "mm", min=1, description="from the rotor plane forward to the lip's highlight (~0.35 D)"),
        Parameter("duct_length", 67.5, "mm", min=1, description="from the rotor plane back to the nozzle exit (~0.75 D)"),
        Parameter("wall_thickness", 2.5, "mm", min=0.3),
        Parameter("lip_radius", 6.0, "mm", min=0.5, description="the bell-mouth's radius of curvature (> wall_thickness/2)"),
        Parameter("lip_nose_radius", 0.75, "mm", min=0.05, description="the lip's leading-edge radius (< wall_thickness/2)"),
        Parameter("exit_area_ratio", 0.9, "", min=0.3, max=1.5, description="nozzle exit annulus / fan annulus"),
        Parameter("stator_vanes", 7, min=1, max=31),
        Parameter("stator_chord", 16.0, "mm", min=1),
        Parameter("stator_thickness", 0.08, "", min=0.03, max=0.3, description="thickness / chord (NACA 00xx)"),
        Parameter("stator_gap", 14.0, "mm", min=0, description="rotor trailing-edge plane to the stators' leading edge"),
        Parameter("stagger_deg", 0.0, "deg", min=-60, max=60, description="vane chord against the axis (0 = aligned)"),
        Parameter("tail_gap", 4.0, "mm", min=0.5, description="stators' trailing edge to the tail cone and nozzle start"),
        Parameter("tail_length", 36.0, "mm", min=1, description="the centre body's pointed tail cone"),
    ]

    @staticmethod
    def layout(p) -> dict:
        """Axial stations, radii (mm) and areas (mm^2) of the housing, checked for consistency."""
        R, r_b = p["diameter"] / 2, p["hub_diameter"] / 2
        if r_b >= R:
            raise ValueError("hub_diameter must be smaller than diameter")
        t, rho, rn = p["wall_thickness"], p["lip_radius"], p["lip_nose_radius"]
        if not rn < t / 2 < rho:
            raise ValueError("the lip needs lip_nose_radius < wall_thickness/2 < lip_radius")
        R_s = R + p["tip_clearance"]
        te = rotor_trailing_edge(p)
        x_l = -p["lip_length"]
        dx = math.sqrt((t - 2 * rn) * (2 * rho - t))                               # nose centre ahead of the bell's
        x_s = x_l + rn + dx                                                         # the bell arc meets the shroud
        if x_s > -p["hub_height"] / 2:
            raise ValueError("lip_length is too short: the shroud must be cylindrical over the rotor (from -hub_height/2)")
        x_b0 = max(p["hub_height"] / 2, te["root_te_x"]) + p["axial_gap"]          # behind hub face and root stub
        s_le = te["te_x"] + p["stator_gap"]
        st = math.radians(p["stagger_deg"])
        s_te = s_le + p["stator_chord"] * math.cos(st)
        if x_b0 >= s_le:
            raise ValueError("the centre body must start ahead of the stators' leading edge")
        if np.abs(EDFHousing._vane_xz(p, s_le)[:, 1]).max() >= r_b - 1.0:
            raise ValueError("the staggered stator section is wider across the axis than the centre body (r_hub - 1)")
        x_t0 = s_te + p["tail_gap"]
        x_e = p["duct_length"]
        if x_t0 >= x_e - 1.0:
            raise ValueError(f"the stators and tail_gap reach x = {x_t0:.1f} mm: duct_length must leave room for the nozzle")
        fan = math.pi * (R ** 2 - r_b ** 2)
        A_0, A_e = math.pi * (R_s ** 2 - r_b ** 2), p["exit_area_ratio"] * fan
        r_t = EDFHousing.tail_radius(p, x_e, x_t0)
        R_e = math.sqrt(A_e / math.pi + r_t ** 2)
        return {"R": R, "R_s": R_s, "r_b": r_b, "x_l": x_l, "x_s": x_s, "lip_dx": dx, "x_b0": x_b0, "s_le": s_le,
                "s_te": s_te, "x_t0": x_t0, "x_t1": x_t0 + p["tail_length"], "x_e": x_e, "R_e": R_e, "r_t": r_t,
                "fan": fan, "A_0": A_0, "A_e": A_e, "te": te}

    @staticmethod
    def _tail(p, x, x_t0):
        """The centre body's radius ``r_hub (1 - s^2)`` and slope ``dr/dx = -2 r_hub s / tail_length`` at ``x``
        (arrays): ``r_hub`` ahead of the tail cone (``x < x_t0``), 0 past its tip."""
        L, r_b = p["tail_length"], p["hub_diameter"] / 2
        s = np.clip((np.asarray(x, float) - x_t0) / L, 0.0, 1.0)
        return r_b * (1 - s ** 2), np.where(s < 1.0, -2 * r_b * s / L, 0.0)

    @staticmethod
    def tail_radius(p, x, x_t0):
        """The centre body's radius at ``x`` on the tail cone (``x >= x_t0``); 0 past its tip."""
        return float(EDFHousing._tail(p, x, x_t0)[0])

    @staticmethod
    def nozzle_radius(p, x, L=None):
        """The duct's inner radius ``R_d`` and slope ``dR_d/dx`` over the nozzle (``x_t0 <= x <= x_e``, arrays) from the
        area law of the class docstring: ``R_d = sqrt(A/pi + r^2)``, ``R_d' = (A'/pi + 2 r r') / (2 R_d)``."""
        L = EDFHousing.layout(p) if L is None else L
        x = np.asarray(x, float)
        ln = L["x_e"] - L["x_t0"]
        s = np.clip((x - L["x_t0"]) / ln, 0.0, 1.0)
        A = L["A_e"] + (L["A_0"] - L["A_e"]) * (1 + np.cos(np.pi * s)) / 2
        dA = -(L["A_0"] - L["A_e"]) * np.pi / (2 * ln) * np.sin(np.pi * s)
        r, dr = EDFHousing._tail(p, x, L["x_t0"])
        R_d = np.sqrt(A / np.pi + r ** 2)
        return R_d, (dA / np.pi + 2 * r * dr) / (2 * R_d)

    @staticmethod
    def duct_profile(p, n_arc: int = 16, n_nozzle: int = 24) -> dict:
        """The duct's meridional section (x, r) in mm: ``inner`` (the flow side: highlight -> the nose's lower half ->
        the bell arc -> shroud -> nozzle -> exit), ``outer`` (exit -> the nozzle's outer wall -> the nacelle cylinder ->
        the nose's top), ``nose`` (the nose's upper half, top -> highlight) and ``closed`` (all three as one polygon)."""
        L = EDFHousing.layout(p)
        t, rho, rn = p["wall_thickness"], p["lip_radius"], p["lip_nose_radius"]
        R_s, x_s, x0, x_e = L["R_s"], L["x_s"], L["x_t0"], L["x_e"]
        cx, cr = x_s - L["lip_dx"], R_s + t - rn                                   # the nose circle's centre
        phi = math.atan2(L["lip_dx"], rho - t + rn)                                # the bell arc's angle, shroud -> nose
        m = max(4, n_arc // 2)
        a = np.linspace(0.5 * math.pi, math.pi, m + 1)                             # top -> highlight
        nose = np.column_stack([cx + rn * np.cos(a), cr + rn * np.sin(a)])
        a = np.linspace(math.pi, 1.5 * math.pi - phi, m + 1)                       # highlight -> the tangent point
        nose_in = np.column_stack([cx + rn * np.cos(a), cr + rn * np.sin(a)])
        f = np.linspace(phi, 0.0, n_arc + 1)                                       # tangent point -> shroud
        bell = np.column_stack([x_s - rho * np.sin(f), R_s + rho - rho * np.cos(f)])
        xn = x0 + (x_e - x0) * np.linspace(0, 1, n_nozzle + 1)
        R_d, dR = EDFHousing.nozzle_radius(p, xn, L)
        inner = np.vstack([nose_in, bell[1:], np.column_stack([xn, R_d])])
        outer = np.vstack([np.column_stack([xn, R_d + t * np.sqrt(1 + dR ** 2)])[::-1], [[cx, R_s + t]]])
        closed = np.vstack([inner, outer, nose[1:-1]])
        return {"inner": inner, "outer": outer, "nose": nose, "closed": closed}

    @staticmethod
    def body_profile(p, n_tail: int = 16) -> np.ndarray:
        """The centre body's half profile (x, r) in mm, from its front face on the axis to the tail cone's tip on the
        axis; the exit plane is one of the tail's points when the cone reaches it, so the CAD has the exact exit radius."""
        L = EDFHousing.layout(p)
        x_t0, x_t1, x_e = L["x_t0"], L["x_t1"], L["x_e"]
        xs = np.linspace(x_t0, x_t1, n_tail + 1)
        if x_t0 < x_e < x_t1:
            xs = np.unique(np.append(xs, x_e))
        tail = np.column_stack([xs, EDFHousing._tail(p, xs, x_t0)[0]])
        return np.vstack([[(L["x_b0"], 0.0), (L["x_b0"], L["r_b"])], tail])

    @staticmethod
    def _vane_xz(p, x_le) -> np.ndarray:
        c, st = p["stator_chord"], math.radians(p["stagger_deg"])
        sec = np.array(_naca00(p["stator_thickness"])) * c
        return np.column_stack([x_le + sec[:, 0] * math.cos(st) - sec[:, 1] * math.sin(st),
                                sec[:, 0] * math.sin(st) + sec[:, 1] * math.cos(st)])

    @staticmethod
    def vane_section(p) -> np.ndarray:
        """Vane 0's section (x, z) in mm (it stands along +y): NACA 00xx, leading edge at ``stator_le_x``, staggered
        about it."""
        return EDFHousing._vane_xz(p, EDFHousing.layout(p)["s_le"])

    def build(self, p):
        L = self.layout(p)
        duct = cq.Workplane("XY").polyline([tuple(q) for q in self.duct_profile(p)["closed"]]).close() \
            .revolve(360, (0, 0, 0), (1, 0, 0))
        body = cq.Workplane("XY").polyline([tuple(q) for q in self.body_profile(p)]).close() \
            .revolve(360, (0, 0, 0), (1, 0, 0))
        r0 = L["r_b"] - 1.0                                                        # into the centre body
        r1 = L["R_s"] + 0.4 * p["wall_thickness"]                                  # into the duct wall
        sec = self.vane_section(p)
        # the section in the XZ plane (local (x, y) -> global (X, Z)) as a prism along +y from the axis past r1, cut to
        # the annulus r0 <= r <= r1: its ends follow the cylinders whatever the stagger
        prism = cq.Workplane("XZ").polyline([tuple(q) for q in sec]).close().extrude(-(r1 + 1.0)).val()
        xa, la = sec[:, 0].min() - 1.0, np.ptp(sec[:, 0]) + 2.0
        ring = cq.Solid.makeCylinder(r1, la, cq.Vector(xa, 0, 0), cq.Vector(1, 0, 0)).cut(
            cq.Solid.makeCylinder(r0, la, cq.Vector(xa, 0, 0), cq.Vector(1, 0, 0)))
        vane = prism.intersect(ring)
        n = int(p["stator_vanes"])
        vanes = [vane.rotate((0, 0, 0), (1, 0, 0), 360.0 * k / n) for k in range(n)]   # +phi: +y towards +z
        return body.val().fuse(duct.val(), *vanes).clean()                        # one boolean: half the time of n + 1


__all__ = ["EDFRotor", "EDFHousing", "SHARED", "rotor_trailing_edge"]
