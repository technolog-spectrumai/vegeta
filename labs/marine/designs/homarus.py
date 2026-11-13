"""Homarus: a work-class ROV for the oil field (Ketos Oil), named after *Homarus gammarus*, the European lobster:
heavy, armoured, two strong claws. It is the classic electro-hydraulic work-class vehicle: an open aluminium frame
under a syntactic-foam buoyancy block, a 110 kW (150 hp) hydraulic power unit fed through the umbilical, seven
hydraulic thrusters (four horizontal vectored at 45 deg, three vertical), a 7-function position-controlled
manipulator on the right and a 5-function rate grabber on the left, a tooling skid under the frame.

It never carries a battery: it hangs from a tether management system (TMS, the top-hat garage in
``oil_platform.py``) on the vessel's lift umbilical, and swims out of it on a 250 m excursion tether. The numbers
are class values for 3000 m work-class ROVs (assumed; no vendor's vehicle):

* 3.3 m x 1.9 m x 1.9 m, ~4.2 t in air, ~+20 kg buoyant (trimmed with the tooling skid on);
* thrust: 4 x 3.2 kN horizontal (vectored), 3 x 3.0 kN vertical; ~3 kn forward, ~1.8 kn lateral;
* HPU 110 kW shaft; hotel (cameras, lights, sonar, survey sensors, the valve pack's pilot) 3.5 kW;
* rated 3000 m; the oil field is 110 m.

Layout: x forward, y left, z up; the vehicle's middle height at z = 0 in the export.

``part``: ``vehicle`` (all), ``frame``, ``foam``, ``thrusters``, ``arms``, ``skid``.
"""
from __future__ import annotations

import math

try:
    import cadquery as cq
except ImportError:
    cq = None

from vegeta.dedalus import Design, Parameter

RHO_SW = 1025.0
G = 9.81

#: Tooling skids and tools (mass kg in air, power W while working, price EUR, what): class values (assumed).
TOOLS = {
    "camera": (0.0, 400.0, 0.0, "the vehicle's own: 4 HD cameras, 2 pan-tilt, LED lights, a multibeam imaging sonar"),
    "cp": (6.0, 50.0, 12000.0, "CP probe on the manipulator: contact potential (mV vs Ag/AgCl) on a node or an anode"),
    "cleaning": (180.0, 15000.0, 90000.0, "hydraulic brush and water-jet skid: marine growth off a weld for CVI"),
    "acfm": (25.0, 300.0, 120000.0, "ACFM probe (alternating current field measurement): crack depth and length at a weld"),
    "fmd": (40.0, 200.0, 80000.0, "flooded member detection (gamma or UT through the wall)"),
    "anode": (120.0, 2000.0, 30000.0, "anode change-out: clamp-on retrofit anode carried in the skid basket"),
}


class Homarus(Design):
    parameters = [
        Parameter("part", "vehicle", choices=("vehicle", "frame", "foam", "thrusters", "arms", "skid")),
        Parameter("length", 3300.0, "mm"),
        Parameter("width", 1900.0, "mm"),
        Parameter("height", 1900.0, "mm"),
        Parameter("foam_height", 650.0, "mm"),
        Parameter("bar", 120.0, "mm", description="frame section"),
        Parameter("hpu_kw", 110.0, "kW"),
        Parameter("hotel_kw", 3.5, "kW"),
        Parameter("h_thrust_kn", 3.2, "kN", description="each horizontal thruster"),
        Parameter("v_thrust_kn", 3.0, "kN", description="each vertical thruster"),
        Parameter("mass_kg", 4200.0, "kg"),
        Parameter("net_buoyancy_kg", 20.0, "kg"),
        Parameter("tether_m", 250.0, "m", description="excursion tether from the TMS"),
        Parameter("rating_m", 3000.0, "m"),
    ]

    @staticmethod
    def p(over=None):
        return {q.name: q.default for q in Homarus.parameters} | (over or {})

    @staticmethod
    def thruster_layout(p):
        """[(at mm, axis)]: four horizontal at 45 deg in the corners, three vertical (two fore, one aft)."""
        L, W = p["length"], p["width"]
        hx, hy, hz = L / 2 - 350, W / 2 - 250, -p["height"] / 2 + 650
        s = math.sqrt(0.5)
        hor = [((hx, hy, hz), (s, -s, 0.0)), ((hx, -hy, hz), (s, s, 0.0)),
               ((-hx, hy, hz), (s, s, 0.0)), ((-hx, -hy, hz), (s, -s, 0.0))]
        vz = p["height"] / 2 - p["foam_height"] / 2
        ver = [((L / 4, W / 4, vz), (0.0, 0.0, 1.0)), ((L / 4, -W / 4, vz), (0.0, 0.0, 1.0)), ((-L / 4, 0.0, vz), (0.0, 0.0, 1.0))]
        return hor + ver

    @classmethod
    def sizing(cls, over=None):
        p = cls.p(over)
        th = cls.thruster_layout(p)
        fx = sum(abs(a[0]) for _, a in th[:4]) * p["h_thrust_kn"] * 1000
        fy = sum(abs(a[1]) for _, a in th[:4]) * p["h_thrust_kn"] * 1000
        fz = 3 * p["v_thrust_kn"] * 1000
        L, W, H = p["length"] / 1000, p["width"] / 1000, p["height"] / 1000
        cd = (0.9, 1.0, 1.1)
        v = {"surge": math.sqrt(2 * fx / (RHO_SW * cd[0] * W * H)), "sway": math.sqrt(2 * fy / (RHO_SW * cd[1] * L * H)),
             "heave": math.sqrt(2 * fz / (RHO_SW * cd[2] * L * W))}
        disp_L = (p["mass_kg"] + p["net_buoyancy_kg"]) / RHO_SW * 1000
        # the thrusters' hydraulic power at full thrust: the HPU shared by the seven (~0.75 of shaft to thrust work)
        th_W = p["hpu_kw"] * 1000 * 0.75 / 4
        return {"thrust_N": {"surge": fx, "sway": fy, "heave": fz}, "v_max_m_s": v, "displacement_L": disp_L,
                "thruster_W": th_W, "hotel_W": p["hotel_kw"] * 1000, "price_eur": 3.2e6}

    # --- CAD ---------------------------------------------------------------------------------------------------
    def _frame(self, p):
        L, W, H, b = p["length"], p["width"], p["height"], p["bar"]
        z0, z1 = -H / 2, H / 2 - p["foam_height"]
        solids = []
        for y in (W / 2 - b / 2, -W / 2 + b / 2):
            for z in (z0 + b / 2, z1 - b / 2):
                solids.append(cq.Solid.makeBox(L, b, b, cq.Vector(-L / 2, y - b / 2, z - b / 2)))
            for x in (L / 2 - b / 2, 0.0, -L / 2 + b / 2):
                solids.append(cq.Solid.makeBox(b, b, z1 - z0, cq.Vector(x - b / 2, y - b / 2, z0)))
        for x in (L / 2 - b / 2, -L / 2 + b / 2):
            for z in (z0 + b / 2, z1 - b / 2):
                solids.append(cq.Solid.makeBox(b, W, b, cq.Vector(x - b / 2, -W / 2, z - b / 2)))
        # the HPU and the valve packs: two grey cans low in the frame
        solids.append(cq.Solid.makeCylinder(320, 1400, cq.Vector(-700, 0, z0 + 450), cq.Vector(1, 0, 0)))
        solids.append(cq.Solid.makeBox(600, 900, 450, cq.Vector(700, -450, z0 + 200)))
        return cq.Workplane("XY").newObject([cq.Compound.makeCompound(solids)])

    def _foam(self, p):
        L, W, H, fh = p["length"], p["width"], p["height"], p["foam_height"]
        foam = cq.Workplane("XY").box(L, W, fh).translate((0, 0, H / 2 - fh / 2)).edges("|Y").fillet(180)
        holes = [(at[0], at[1]) for at, a in self.thruster_layout(p) if a[2] == 1.0]
        for x, y in holes:
            foam = foam.cut(cq.Workplane("XY").add(cq.Solid.makeCylinder(260, fh + 20, cq.Vector(x, y, H / 2 - fh - 10))))
        return foam

    def _thrusters(self, p):
        solids = []
        for (x, y, z), a in self.thruster_layout(p):
            ax = cq.Vector(*a)
            base = cq.Vector(x, y, z) - ax * 300
            solids.append(cq.Solid.makeCylinder(230, 600, base, ax))
        return cq.Workplane("XY").newObject([cq.Compound.makeCompound(solids)])

    def _arms(self, p):
        L, W, H = p["length"], p["width"], p["height"]
        z = -H / 2 + 600
        solids = []
        for y, reach in ((-W / 2 + 350, 1500.0), (W / 2 - 350, 1200.0)):   # the 7-function right, the grabber left
            sh = cq.Vector(L / 2 - 100, y, z)
            solids.append(cq.Solid.makeSphere(140, sh))
            up = cq.Vector(0.55, 0, 0.35).normalized()
            solids.append(cq.Solid.makeCylinder(90, reach * 0.5, sh, up))
            el = sh + up * (reach * 0.5)
            fo = cq.Vector(0.8, 0, -0.6).normalized()
            solids.append(cq.Solid.makeCylinder(75, reach * 0.5, el, fo))
            wr = el + fo * (reach * 0.5)
            solids.append(cq.Solid.makeBox(140, 60, 220, wr + cq.Vector(0, 40, -110)))
            solids.append(cq.Solid.makeBox(140, 60, 220, wr + cq.Vector(0, -100, -110)))
        return cq.Workplane("XY").newObject([cq.Compound.makeCompound(solids)])

    def _skid(self, p):
        L, W, H = p["length"], p["width"], p["height"]
        return cq.Workplane("XY").box(L * 0.8, W * 0.9, 300).translate((0, 0, -H / 2 - 150)).edges("|Z").fillet(60)

    def build(self, p):
        if cq is None:
            raise ImportError("Homarus's CAD needs CadQuery")
        parts = {"frame": self._frame, "foam": self._foam, "thrusters": self._thrusters, "arms": self._arms, "skid": self._skid}
        if p["part"] in parts:
            return parts[p["part"]](p)
        return cq.Workplane("XY").newObject([cq.Compound.makeCompound([f(p).val() for f in parts.values()])])


if __name__ == "__main__":
    import json
    print(json.dumps(Homarus.sizing(), indent=1))
