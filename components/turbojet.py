"""A model-aircraft turbojet in CAD: the geometry (from ``notebooks/designs/turbojet.py``, notebook 28) — the radial
compressor impeller, the axial turbine wheel and the engine's outside.

Frame: the engine axis is +X, the flow runs along +X (intake at small x), units mm. The impeller's inducer plane is
``x = 0``.

- **Impeller** (``part="impeller"``): a hub of revolution (spinner, the hub line curving from the inducer hub radius
  out to ``d2/2``, the backplate) and ``blades`` full radial-element blades.
- **Turbine wheel** (``part="turbine"``): a cast disc (bore, hub, web, rim) with ``turbine_blades`` twisted blades.
- **Engine** (``part="engine"``): the casing's outside with the intake bell-mouth and the nozzle with its tail cone.

Geometry only: the compressor's CFD passage (``passage_profile``, ``compressor_surfaces``), the masses and the sizing
from the Boreas cycle (``sized``) stay in ``notebooks/designs/turbojet.py``.
"""
import math

import cadquery as cq
import numpy as np
from vegeta.dedalus import Design, Parameter


class Turbojet(Design):
    parameters = [
        Parameter("part", "impeller", choices=("impeller", "turbine", "engine"), description="what to build"),
        Parameter("impeller_diameter", 68.0, "mm", min=20, max=200, description="compressor tip (exducer) diameter d2"),
        Parameter("inducer_tip_ratio", 0.62, "", min=0.4, max=0.8, description="inducer tip diameter / d2"),
        Parameter("inducer_hub_ratio", 0.30, "", min=0.15, max=0.5, description="inducer hub diameter / d2"),
        Parameter("axial_length_ratio", 0.30, "", min=0.15, max=0.6, description="impeller axial length / d2"),
        Parameter("exit_width_ratio", 0.075, "", min=0.03, max=0.15, description="blade height at the exit / d2"),
        Parameter("blades", 6, "", min=3, max=20, description="full impeller blades"),
        Parameter("blade_thickness", 0.8, "mm", min=0.2, max=5),
        Parameter("inducer_angle_deg", 58.0, "deg", min=20, max=75,
                  description="blade angle from the axis at the inducer tip (atan(U1 / c_axial) at the design point)"),
        Parameter("tip_clearance", 0.3, "mm", min=0.05, max=2),
        Parameter("backplate", 3.0, "mm", min=0.5, max=10, description="impeller backplate thickness"),
        Parameter("diffuser_ratio", 1.45, "", min=1.1, max=2.0, description="vaneless diffuser outer diameter / d2"),
        Parameter("inlet_length_ratio", 0.8, "", min=0.3, max=2.0, description="straight inlet duct ahead of the inducer / d2"),
        Parameter("spinner_ratio", 0.35, "", min=0.0, max=1.0, description="spinner length ahead of the inducer / d2"),
        Parameter("turbine_tip_ratio", 0.95, "", min=0.6, max=1.2, description="turbine tip diameter / d2"),
        Parameter("turbine_hub_ratio", 0.68, "", min=0.4, max=0.85, description="turbine rim (blade root) / tip diameter"),
        Parameter("turbine_blades", 23, "", min=7, max=60),
        Parameter("turbine_chord", 9.0, "mm", min=2, max=40),
        Parameter("turbine_blade_t", 0.10, "", min=0.03, max=0.25, description="turbine blade thickness / chord"),
        Parameter("turbine_stagger_deg", 35.0, "deg", min=0, max=70, description="blade stagger at mid-span"),
        Parameter("turbine_twist_deg", 12.0, "deg", min=0, max=40, description="stagger change hub to tip"),
        Parameter("bore_diameter", 8.0, "mm", min=2, max=40, description="impeller and turbine bore (the shaft)"),
        Parameter("disc_width", 12.0, "mm", min=3, max=60, description="turbine disc width at the hub"),
        Parameter("web_width", 5.0, "mm", min=1, max=40, description="turbine disc width at mid-radius"),
        Parameter("outer_diameter", 112.0, "mm", min=30, max=400, description="engine casing diameter (datasheet)"),
        Parameter("length", 300.0, "mm", min=50, max=1500, description="engine length, intake lip to nozzle exit (datasheet)"),
        Parameter("nozzle_area", 792.0, "mm^2", min=50, description="nozzle exit annulus area (the cycle's A8)"),
        Parameter("tail_cone_ratio", 0.45, "", min=0.0, max=0.8, description="tail cone diameter / nozzle exit diameter"),
    ]

    # ------------------------------------------------------------------------------------- meridional lines
    @staticmethod
    def stations(p) -> dict:
        d2 = p["impeller_diameter"]
        r2, r1t, r1h = d2 / 2, p["inducer_tip_ratio"] * d2 / 2, p["inducer_hub_ratio"] * d2 / 2
        lax, b2 = p["axial_length_ratio"] * d2, p["exit_width_ratio"] * d2
        if b2 >= lax or r1t <= r1h or r1t >= r2:
            raise ValueError("impeller proportions: need exit width < axial length and inducer hub < tip < d2/2")
        return {"r2": r2, "r1t": r1t, "r1h": r1h, "lax": lax, "b2": b2, "r3": p["diffuser_ratio"] * r2,
                "x_in": -p["inlet_length_ratio"] * d2, "x_spinner": -p["spinner_ratio"] * d2,
                "x_back": lax + p["backplate"]}

    @staticmethod
    def hub_line(p, n=24) -> np.ndarray:
        """(x, r) of the hub from the inducer (x = 0, r1h) to the exit (lax, r2): a quarter ellipse."""
        s = Turbojet.stations(p)
        t = np.linspace(0, math.pi / 2, n)
        return np.column_stack([s["lax"] * np.sin(t), s["r2"] - (s["r2"] - s["r1h"]) * np.cos(t)])

    @staticmethod
    def shroud_line(p, n=24, clearance=0.0) -> np.ndarray:
        """(x, r) of the blade tips from the inducer tip (0, r1t) to the exit (lax - b2, r2); ``clearance`` offsets it
        outward along its normal (the casing)."""
        s = Turbojet.stations(p)
        t = np.linspace(0, math.pi / 2, n)
        a, b = s["lax"] - s["b2"], s["r2"] - s["r1t"]
        x, r = a * np.sin(t), s["r2"] - b * np.cos(t)
        if clearance:
            nx, nr = -b * np.sin(t), a * np.cos(t)                    # outward normal of the ellipse arc
            k = clearance / np.hypot(nx, nr)
            x, r = x + nx * k, r + nr * k
        return np.column_stack([x, r])

    # ------------------------------------------------------------------------------------- impeller
    @staticmethod
    def blade_angle(p, s, r, r_tip):
        """Wrap angle (rad) of the camber line at the meridional fraction ``s`` on a line at inducer radius ``r``:
        the inducer meets the flow (``tan beta = U / c``, so ``tan beta`` grows with the radius), the blade turns
        to radial by mid-passage."""
        tan_tip = math.tan(math.radians(p["inducer_angle_deg"]))
        tan_b = tan_tip * r / r_tip
        s_e = 0.55
        return tan_b * np.where(s < s_e, s - s * s / (2 * s_e), s_e / 2)

    def _hub(self, p):
        s = self.stations(p)
        hub = self.hub_line(p)
        prof = []
        if p["spinner_ratio"] > 0:
            ls, t = -s["x_spinner"], np.linspace(0, math.pi / 2, 10)
            prof += [(-ls * math.cos(a), s["r1h"] * math.sin(a)) for a in t]
        else:
            prof += [(0.0, 0.0)]
        prof += [tuple(q) for q in hub[1:]]
        prof += [(s["x_back"], s["r2"]), (s["x_back"], 0.0)]
        if prof[0][1] > 0:
            prof = [(prof[0][0], 0.0)] + prof
        return cq.Workplane("XY").polyline(prof).close().revolve(360, (0, 0, 0), (1, 0, 0))

    def _blade(self, p, k):
        s = self.stations(p)
        n = 16
        hub, tip = self.hub_line(p, n), self.shroud_line(p, n)
        sf = np.linspace(0, 1, n)
        half = p["blade_thickness"] / 2
        th_h = self.blade_angle(p, sf, s["r1h"], s["r1t"]) * s["lax"] / s["r1h"] * s["r1h"] / s["r1t"]
        th_t = self.blade_angle(p, sf, s["r1t"], s["r1t"]) * (s["lax"] - s["b2"]) / s["r1t"]
        base = 2 * math.pi * k / p["blades"]

        def point(x, r, th):
            a = base + th
            return np.array([x, r * math.cos(a), r * math.sin(a)])

        roots = [point(hub[i, 0], hub[i, 1] - 0.6, th_h[i]) for i in range(n)]   # the root reaches into the hub
        tips = [point(tip[i, 0], tip[i, 1], th_t[i]) for i in range(n)]
        wires = []
        for i in range(n):
            j0, j1 = max(i - 1, 0), min(i + 1, n - 1)
            along = 0.5 * (roots[j1] + tips[j1]) - 0.5 * (roots[j0] + tips[j0])
            span = tips[i] - roots[i]
            nrm = np.cross(span, along)
            nrm = nrm / np.linalg.norm(nrm) * half
            # a flat parallelogram (planar: the loft's end caps must be planar faces)
            c = [roots[i] - nrm, tips[i] - nrm, tips[i] + nrm, roots[i] + nrm]
            wires.append(cq.Wire.makePolygon([cq.Vector(*q) for q in c], close=True))
        return cq.Solid.makeLoft(wires, True)

    def impeller(self, p):
        hub = self._hub(p).val()
        blades = [self._blade(p, k) for k in range(int(p["blades"]))]
        wheel = hub.fuse(*blades).clean()
        s = self.stations(p)
        rb = p["bore_diameter"] / 2
        if rb >= 0.8 * s["r1h"]:
            raise ValueError("bore_diameter too large for the impeller's inducer hub")
        x0 = s["x_spinner"] if p["spinner_ratio"] > 0 else 0.0
        # the shaft's bore, from the spinner's base (the nut sits in the spinner) through the backplate
        x_from = x0 + 0.5 * (0.0 - x0)
        bore = cq.Solid.makeCylinder(rb, s["x_back"] - x_from + 1.0, cq.Vector(x_from, 0, 0), cq.Vector(1, 0, 0))
        return cq.Workplane("XY").add(wheel.cut(bore).clean())

    # ------------------------------------------------------------------------------------- turbine
    @staticmethod
    def turbine_stations(p) -> dict:
        rt = p["turbine_tip_ratio"] * p["impeller_diameter"] / 2
        rr = p["turbine_hub_ratio"] * rt
        return {"r_tip": rt, "r_rim": rr, "r_bore": p["bore_diameter"] / 2, "rim_width": p["turbine_chord"] * 1.15,
                "blade_height": rt - rr}

    def turbine(self, p):
        t = self.turbine_stations(p)
        rb, rr, wh, ww, wr = t["r_bore"], t["r_rim"], p["disc_width"], p["web_width"], t["rim_width"]
        if rb >= 0.4 * rr:
            raise ValueError("bore_diameter too large for the turbine disc")
        rh = rb + 0.35 * (rr - rb)                                   # the hub's outer radius
        prof = [(-wh / 2, rb), (wh / 2, rb), (wh / 2, rh), (ww / 2, rh + 0.25 * (rr - rh)), (ww / 2, rr - 0.25 * (rr - rh)),
                (wr / 2, rr - 1.0), (wr / 2, rr), (-wr / 2, rr), (-wr / 2, rr - 1.0), (-ww / 2, rr - 0.25 * (rr - rh)),
                (-ww / 2, rh + 0.25 * (rr - rh)), (-wh / 2, rh)]
        disc = cq.Workplane("XY").polyline(prof).close().revolve(360, (0, 0, 0), (1, 0, 0)).val()
        blades = [self._turbine_blade(p, k) for k in range(int(p["turbine_blades"]))]
        return cq.Workplane("XY").add(disc.fuse(*blades).clean())

    def _turbine_section(self, p, chord, stagger_deg, r, phi):
        """A cambered-plate section (circular-arc camber, elliptic thickness) at radius ``r`` in the blade's local
        (axial, tangential) plane, staggered; returned as a closed wire at angle ``phi`` about the axis."""
        n = 14
        u = 0.5 * (1 - np.cos(np.linspace(0, math.pi, n)))
        camber = 0.06 * chord * 4 * u * (1 - u)
        thick = p["turbine_blade_t"] * chord * np.sqrt(np.clip(u * (1 - u), 0, None)) * 2
        x = (u - 0.5) * chord
        upper = np.column_stack([x, camber + thick / 2])
        lower = np.column_stack([x, camber - thick / 2])
        pts = np.vstack([upper, lower[::-1][1:-1]])
        g = math.radians(stagger_deg)
        rot = np.array([[math.cos(g), -math.sin(g)], [math.sin(g), math.cos(g)]])
        pts = pts @ rot.T
        er = np.array([0.0, math.cos(phi), math.sin(phi)])
        et = np.array([0.0, -math.sin(phi), math.cos(phi)])
        # in the plane tangent to the cylinder at r (planar, for the loft's end caps)
        vs = [cq.Vector(*(np.array([ax, 0.0, 0.0]) + r * er + tg * et)) for ax, tg in pts]
        return cq.Wire.makePolygon(vs, close=True)

    def _turbine_blade(self, p, k):
        t = self.turbine_stations(p)
        phi = 2 * math.pi * k / p["turbine_blades"]
        st, tw, c = p["turbine_stagger_deg"], p["turbine_twist_deg"], p["turbine_chord"]
        root = self._turbine_section(p, c, st - tw / 2, t["r_rim"] - 0.8, phi)
        tip = self._turbine_section(p, 0.8 * c, st + tw / 2, t["r_tip"], phi)
        return cq.Solid.makeLoft([root, tip], True)

    # ------------------------------------------------------------------------------------- engine outside
    @staticmethod
    def engine_profile(p) -> np.ndarray:
        """Half profile (x, r) of the engine's outside and its tail cone: bell-mouth lip, casing, the nozzle converging
        to the exit annulus around the tail cone."""
        R, L = p["outer_diameter"] / 2, p["length"]
        re = math.sqrt(p["nozzle_area"] / (math.pi * (1 - p["tail_cone_ratio"] ** 2)))
        if re >= R:
            raise ValueError("the nozzle exit is wider than the casing")
        lip = 0.08 * L
        t = np.linspace(0, math.pi / 2, 8)
        pts = [(lip * (1 - math.sin(a)), R - 0.12 * R * math.cos(a)) for a in t[::-1]]
        pts = pts[::-1]
        pts += [(0.75 * L, R), (L, re + 1.5)]
        return np.array(pts), re

    def engine(self, p):
        prof, re = self.engine_profile(p)
        rc = p["tail_cone_ratio"] * re
        R = p["outer_diameter"] / 2
        outer = [tuple(q) for q in prof] + [(p["length"], re), (0.9 * p["length"], 0.0) if rc == 0 else (p["length"], rc)]
        if rc > 0:
            outer += [(p["length"] + 0.6 * re, 0.0)]
        outer = [(prof[0][0], 0.0)] + outer
        body = cq.Workplane("XY").polyline(outer).close().revolve(360, (0, 0, 0), (1, 0, 0))
        # the intake opening (a short recess to the compressor face)
        r_in = 0.62 * p["impeller_diameter"] / 2 + 1.0
        body = body.cut(cq.Workplane("YZ").circle(r_in).extrude(0.06 * p["length"]).translate((-1.0, 0, 0)))
        del R
        return body

    def build(self, p):
        if p["part"] == "turbine":
            return self.turbine(p)
        if p["part"] == "engine":
            return self.engine(p)
        return self.impeller(p)


__all__ = ["Turbojet"]
