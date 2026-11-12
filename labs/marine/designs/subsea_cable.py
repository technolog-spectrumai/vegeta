"""The machines and things Ketos Cable works with: subsea power and telecom cables (an offshore wind farm's export
cable, an interconnector) - surveyed along their route, buried, inspected, located at a fault and repaired. None is
a real installation or a vendor's machine; the numbers are class values for North Sea cable work (assumed).

* ``JetTrencher`` - a free-flying jetting ROV trencher (``Jetter``-class): an open steel frame under foam, two
  jetting swords that fluidise the seabed under a cable lying on it so the cable sinks to the trench's bottom, two
  jet pumps (2 x 300 kW) and 8 thrusters on a 1 MW umbilical, tracked skids (it sits on the seabed while it
  trenches, its weight on the tracks). ``trench_rate(soil, depth)`` is its speed by soil: jetting fluidises sand and
  silt easily, soft clay slowly, stiff clay barely, rock not at all (the specific energy per m3 of trench).
* ``CableVessel`` - a ~130 m cable-lay and repair vessel (DP2): a 7000 t cable carousel below deck, a tensioner and
  the stern chute (the overboarding sheave) aft, the jointing container on deck, a 250 t crane and the trencher's
  LARS (an A-frame) on the starboard side.
* ``CableSegment`` - a 10 m length of a 3-core 220 kV HVAC export cable (armoured, 0.24 m, 90 kg/m in air): the mesh
  the viewer strings along the route, and the cable's properties (minimum bend radius, armour).
* ``CableJoint`` - a rigid universal repair joint (the "omega" repair needs two of them and a bight of new cable),
  its length, mass and the time to make one in the jointing container.

Layout: x forward, y left, z up; z = 0 the waterline (the vessel), the middle height (the trencher) or the cable's
axis (the segment, the joint). Metres in the parameters, millimetres in the CAD.
"""
from __future__ import annotations

import math

try:
    import cadquery as cq
except ImportError:
    cq = None

from vegeta.dedalus import Design, Parameter

MM = 1000.0
RHO_SW = 1025.0
G = 9.81

#: Seabed soils and what a jetting sword needs to fluidise them: specific energy (kJ per m3 of trench) and the
#: deepest trench it reaches there (m). Class values (assumed): loose to dense sand, silt, soft/firm clay, stiff clay.
SOILS = {
    "sand": {"specific_kJ_m3": 2500.0, "max_depth_m": 3.0, "what": "loose to medium-dense sand: jetting's best ground"},
    "silt": {"specific_kJ_m3": 1800.0, "max_depth_m": 3.0, "what": "silt and muddy sand: fluidises, settles slowly"},
    "soft_clay": {"specific_kJ_m3": 6000.0, "max_depth_m": 2.0, "what": "soft to firm clay (< 40 kPa): slow"},
    "stiff_clay": {"specific_kJ_m3": 15000.0, "max_depth_m": 0.8, "what": "stiff clay and till: barely, a plough's ground"},
    "rock": {"specific_kJ_m3": None, "max_depth_m": 0.0, "what": "chalk, boulders, bedrock: no jetting, rock placement instead"},
}

#: The cable: a 3-core 220 kV HVAC export cable with single-wire armour (class values, assumed).
CABLE = {"diameter_m": 0.24, "kg_per_m_air": 90.0, "kg_per_m_water": 44.0, "mbr_m": 4.0, "armour": "single wire, galvanised steel",
         "cores": 3, "kv": 220, "fibre": True, "tension_max_kN": 120.0}


class JetTrencher(Design):
    parameters = [
        Parameter("part", "vehicle", choices=("vehicle", "frame", "foam", "thrusters", "swords", "tracks")),
        Parameter("length", 8.0, "m"),
        Parameter("width", 5.0, "m"),
        Parameter("height", 3.6, "m"),
        Parameter("foam_height", 1.0, "m"),
        Parameter("mass_kg", 16000.0, "kg", description="in air"),
        Parameter("jet_kw", 600.0, "kW", description="two jet pumps"),
        Parameter("jet_eff", 0.45, "", description="pump power that fluidises soil (nozzle and pipe losses, assumed)"),
        Parameter("thruster_kw", 300.0, "kW", description="8 electric thrusters, all"),
        Parameter("thrust_kn", 6.0, "kN", description="each thruster"),
        Parameter("hotel_kw", 15.0, "kW"),
        Parameter("trench_width", 0.5, "m"),
        Parameter("sword_depth", 3.0, "m", description="the deepest trench the swords reach"),
        Parameter("max_speed_m_h", 600.0, "m/h", description="the cable's handling limit while trenching"),
        Parameter("rating_m", 1000.0, "m"),
        Parameter("umbilical_m", 1200.0, "m"),
        Parameter("price_eur", 9.0e6, "EUR"),
        Parameter("day_rate_eur", 30000.0, "EUR/day", description="the spread on top of the vessel (assumed)"),
    ]

    @staticmethod
    def p(over=None):
        return {q.name: q.default for q in JetTrencher.parameters} | (over or {})

    @staticmethod
    def thruster_layout(p):
        """[(at m, axis)]: four horizontal vectored at 45 deg, four vertical in the foam."""
        L, W = p["length"], p["width"]
        hx, hy, hz = L / 2 - 0.8, W / 2 - 0.6, 0.2
        s = math.sqrt(0.5)
        hor = [((hx, hy, hz), (s, -s, 0.0)), ((hx, -hy, hz), (s, s, 0.0)), ((-hx, hy, hz), (s, s, 0.0)), ((-hx, -hy, hz), (s, -s, 0.0))]
        vz = p["height"] / 2 - p["foam_height"] / 2
        ver = [((sx * L / 4, sy * W / 4, vz), (0.0, 0.0, 1.0)) for sx in (1, -1) for sy in (1, -1)]
        return hor + ver

    @classmethod
    def trench_rate(cls, soil, depth_m, over=None):
        """Trenching speed (m/h) in ``soil`` for a trench ``depth_m`` deep: the jet power that fluidises soil over
        the specific energy of a trench width x depth, capped by the cable's handling limit; 0 if out of reach."""
        p = cls.p(over)
        s = SOILS[soil]
        if s["specific_kJ_m3"] is None or depth_m > min(s["max_depth_m"], p["sword_depth"]) + 1e-9:
            return 0.0
        v = p["jet_kw"] * p["jet_eff"] / (s["specific_kJ_m3"] * p["trench_width"] * max(depth_m, 0.3))   # m/s
        return min(v * 3600.0, p["max_speed_m_h"])

    @classmethod
    def sizing(cls, over=None):
        p = cls.p(over)
        th = cls.thruster_layout(p)
        fx = sum(abs(a[0]) for _, a in th[:4]) * p["thrust_kn"] * 1000
        fz = 4 * p["thrust_kn"] * 1000
        W, H = p["width"], p["height"]
        v = math.sqrt(2 * fx / (RHO_SW * 1.0 * W * H))
        disp_L = p["mass_kg"] / RHO_SW * 1000 * 0.82          # trimmed heavy: ~18 % of its weight on the tracks
        on_tracks_N = (p["mass_kg"] - disp_L * RHO_SW / 1000) * G
        rates = {s: {d: round(cls.trench_rate(s, d, over), 1) for d in (1.0, 1.5, 2.0, 3.0)} for s in SOILS}
        return {"thrust_N": {"surge": fx, "heave": fz}, "v_max_m_s": v, "displacement_L": disp_L,
                "on_tracks_N": on_tracks_N, "power_W": (p["jet_kw"] + p["thruster_kw"] + p["hotel_kw"]) * 1000,
                "thruster_W": p["thruster_kw"] * 1000 / 8, "trench_m_h": rates, "price_eur": p["price_eur"]}

    # --- CAD (mm) ------------------------------------------------------------------------------------------------
    def _frame(self, p):
        L, W, H, b = p["length"] * MM, p["width"] * MM, p["height"] * MM, 200.0
        z0, z1 = -H / 2 + 600, H / 2 - p["foam_height"] * MM
        s = []
        for y in (W / 2 - b / 2, -W / 2 + b / 2):
            for z in (z0 + b / 2, z1 - b / 2):
                s.append(cq.Solid.makeBox(L, b, b, cq.Vector(-L / 2, y - b / 2, z - b / 2)))
            for x in (L / 2 - b / 2, 0.0, -L / 2 + b / 2):
                s.append(cq.Solid.makeBox(b, b, z1 - z0, cq.Vector(x - b / 2, y - b / 2, z0)))
        for x in (L / 2 - b / 2, -L / 2 + b / 2):
            for z in (z0 + b / 2, z1 - b / 2):
                s.append(cq.Solid.makeBox(b, W, b, cq.Vector(x - b / 2, -W / 2, z - b / 2)))
        for y in (900.0, -900.0):                                        # the two jet pumps
            s.append(cq.Solid.makeCylinder(450, 2200, cq.Vector(-1600, y, z0 + 700), cq.Vector(1, 0, 0)))
        s.append(cq.Solid.makeBox(1200, 1200, 900, cq.Vector(1600, -600, z0 + 250)))   # the HV transformer can
        return cq.Workplane("XY").newObject([cq.Compound.makeCompound(s)])

    def _foam(self, p):
        L, W, H, fh = p["length"] * MM, p["width"] * MM, p["height"] * MM, p["foam_height"] * MM
        foam = cq.Workplane("XY").box(L, W, fh).translate((0, 0, H / 2 - fh / 2)).edges("|Y").fillet(250)
        for (x, y, _), a in self.thruster_layout(p):
            if a[2] == 1.0:
                foam = foam.cut(cq.Workplane("XY").add(cq.Solid.makeCylinder(380, fh + 20, cq.Vector(x * MM, y * MM, H / 2 - fh - 10))))
        return foam

    def _thrusters(self, p):
        s = []
        for (x, y, z), a in self.thruster_layout(p):
            ax = cq.Vector(*a)
            s.append(cq.Solid.makeCylinder(320, 800, cq.Vector(x * MM, y * MM, z * MM) - ax * 400, ax))
        return cq.Workplane("XY").newObject([cq.Compound.makeCompound(s)])

    def _swords(self, p):
        """Two jetting swords either side of the cable, hinged at the frame's back, down into the seabed."""
        H, d = p["height"] * MM, p["sword_depth"] * MM
        s = []
        for y in (350.0, -350.0):
            s.append(cq.Solid.makeBox(500, 160, d + 600, cq.Vector(-1200, y - 80, -H / 2 - d)))
            s.append(cq.Solid.makeCylinder(120, 1600, cq.Vector(-1000, y, -H / 2 + 500), cq.Vector(1, 0, 0)))
        s.append(cq.Solid.makeBox(900, 300, 250, cq.Vector(2800, -150, -H / 2 + 50)))    # the cable's depressor guide
        return cq.Workplane("XY").newObject([cq.Compound.makeCompound(s)])

    def _tracks(self, p):
        L, W, H = p["length"] * MM, p["width"] * MM, p["height"] * MM
        s = []
        for y in (W / 2 - 500, -W / 2 + 500):
            box = cq.Workplane("XY").box(L * 0.85, 700, 600).translate((0, y, -H / 2 + 300)).edges("|Y").fillet(280)
            s.append(box.val())
        return cq.Workplane("XY").newObject([cq.Compound.makeCompound(s)])

    def build(self, p):
        if cq is None:
            raise ImportError("the trencher's CAD needs CadQuery")
        parts = {"frame": self._frame, "foam": self._foam, "thrusters": self._thrusters, "swords": self._swords, "tracks": self._tracks}
        if p["part"] in parts:
            return parts[p["part"]](p)
        return cq.Workplane("XY").newObject([cq.Compound.makeCompound([f(p).val() for f in parts.values()])])


class CableVessel(Design):
    parameters = [
        Parameter("part", "vessel", choices=("vessel", "hull", "house", "deck")),
        Parameter("length", 130.0, "m"),
        Parameter("beam", 28.0, "m"),
        Parameter("draft", 7.0, "m"),
        Parameter("depth", 10.0, "m"),
        Parameter("carousel_t", 7000.0, "t", description="cable on the carousel"),
        Parameter("carousel_d", 26.0, "m"),
        Parameter("lay_speed_m_h", 400.0, "m/h", description="laying, surface-laid"),
        Parameter("crane_t", 250.0, "t"),
        Parameter("lars_x", 10.0, "m", description="the trencher's A-frame along the ship (from midships)"),
        Parameter("dp_wind_m_s", 20.0, "m/s"),
        Parameter("dp_current_m_s", 1.5, "m/s"),
        Parameter("lars_hs_m", 2.5, "m", description="launch and recovery of the trencher up to this Hs"),
        Parameter("joint_hs_m", 2.0, "m", description="jointing on deck needs the vessel on station up to this Hs"),
        Parameter("day_rate_eur", 120000.0, "EUR/day", description="vessel, crew and jointers (assumed)"),
    ]

    @staticmethod
    def p(over=None):
        return {q.name: q.default for q in CableVessel.parameters} | (over or {})

    @staticmethod
    def overboard(p):
        """The stern chute (x aft, on the centre line) and the trencher's LARS (starboard): m, the vessel's frame."""
        return {"chute": (-p["length"] / 2 - 1.0, 0.0, 0.0), "lars": (p["lars_x"], -(p["beam"] / 2 + 4.0), 0.0)}

    @classmethod
    def sizing(cls, over=None):
        p = cls.p(over)
        km = p["carousel_t"] * 1000 / CABLE["kg_per_m_air"] / 1000
        return {"carousel_km": km, "lay_days_per_100km": 100000 / p["lay_speed_m_h"] / 24, "day_rate_eur": p["day_rate_eur"]}

    def build(self, p):
        if cq is None:
            raise ImportError("the vessel's CAD needs CadQuery")
        from merchant_hull import MerchantHull
        L, B, D, T = p["length"], p["beam"], p["depth"], p["draft"]
        hull = MerchantHull().build(MerchantHull.p({"length": L, "beam": B, "draft": T, "depth": D}))
        dz = (D - T) * MM
        comp = lambda s: cq.Workplane("XY").newObject([cq.Compound.makeCompound(s)])
        house = comp([cq.Solid.makeBox(24 * MM, (B - 2) * MM, 14 * MM, cq.Vector(28 * MM, -(B / 2 - 1) * MM, dz)),
                      cq.Solid.makeBox(12 * MM, (B - 6) * MM, 4 * MM, cq.Vector(34 * MM, -(B / 2 - 3) * MM, dz + 14 * MM))])
        R = p["carousel_d"] / 2
        a = [cq.Solid.makeCylinder(R * MM, 3 * MM, cq.Vector(-8 * MM, 0, dz)),                       # the carousel's rim
             cq.Solid.makeBox(10 * MM, 6 * MM, 3 * MM, cq.Vector((-L / 2 + 8) * MM, -3 * MM, dz)),     # the tensioner
             cq.Solid.makeCylinder(3 * MM, 3 * MM, cq.Vector((-L / 2 + 1) * MM, -1.5 * MM, dz + 1 * MM), cq.Vector(0, 1, 0)),  # chute
             cq.Solid.makeBox(12 * MM, 3 * MM, 3 * MM, cq.Vector(12 * MM, (B / 2 - 4) * MM, dz)),      # the jointing container
             cq.Solid.makeCylinder(1.2 * MM, 30 * MM, cq.Vector(20 * MM, (B / 2 - 3) * MM, dz), cq.Vector(-0.6, 0, 0.8))]  # crane
        x, y0, y1, top = p["lars_x"] * MM, -(B / 2 - 1) * MM, -(B / 2 + 4) * MM, dz + 12 * MM
        for dx in (-3 * MM, 3 * MM):
            va, vb = cq.Vector(x + dx, y0, dz), cq.Vector(x + dx, y1, top)
            a.append(cq.Solid.makeCylinder(450, (vb - va).Length, va, (vb - va).normalized()))
        a.append(cq.Solid.makeCylinder(500, 6.6 * MM, cq.Vector(x - 3.3 * MM, y1, top), cq.Vector(1, 0, 0)))
        deck = comp(a)
        parts = {"hull": hull, "house": house, "deck": deck}
        if p["part"] in parts:
            return parts[p["part"]]
        return comp([hull.val(), house.val(), deck.val()])


class CableSegment(Design):
    parameters = [Parameter("length", 10.0, "m"), Parameter("diameter", CABLE["diameter_m"], "m")]

    @staticmethod
    def p(over=None):
        return {q.name: q.default for q in CableSegment.parameters} | (over or {})

    def build(self, p):
        if cq is None:
            raise ImportError("the cable's CAD needs CadQuery")
        L = p["length"] * MM
        return cq.Workplane("YZ").circle(p["diameter"] * MM / 2).extrude(L).translate((-L / 2, 0, 0))


class CableJoint(Design):
    """A universal rigid repair joint: two bend restrictors either side of a steel casing over the cores' splices."""
    parameters = [
        Parameter("length", 6.0, "m", description="the casing"),
        Parameter("diameter", 0.9, "m"),
        Parameter("restrictor", 3.0, "m", description="each bend restrictor"),
        Parameter("mass_kg", 2500.0, "kg"),
        Parameter("joint_h", 96.0, "h", description="making one joint in the jointing container (3-core HVAC, assumed)"),
        Parameter("price_eur", 350000.0, "EUR"),
    ]

    @staticmethod
    def p(over=None):
        return {q.name: q.default for q in CableJoint.parameters} | (over or {})

    def build(self, p):
        if cq is None:
            raise ImportError("the joint's CAD needs CadQuery")
        L, R, r = p["length"] * MM, p["diameter"] * MM / 2, p["restrictor"] * MM
        body = cq.Workplane("YZ").circle(R).extrude(L * 0.7).translate((-L * 0.35, 0, 0))
        for sx in (1, -1):
            cone = cq.Solid.makeCone(R * 0.8, R * 0.45, L * 0.15, cq.Vector(sx * L * 0.35, 0, 0), cq.Vector(sx, 0, 0))
            rs = cq.Solid.makeCone(R * 0.45, CABLE["diameter_m"] * MM / 2 + 20, r, cq.Vector(sx * L * 0.5, 0, 0), cq.Vector(sx, 0, 0))
            body = body.union(cq.Workplane("XY").add(cone)).union(cq.Workplane("XY").add(rs))
        return body


if __name__ == "__main__":
    import json
    print(json.dumps({"trencher": JetTrencher.sizing(), "vessel": CableVessel.sizing()}, indent=1, default=str))
