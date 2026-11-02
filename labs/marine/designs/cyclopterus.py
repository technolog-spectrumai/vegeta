"""Cyclopterus (Cyclopterus lumpus, the lumpfish: the cleaner fish salmon farms keep in their pens, which clings by a
belly disc): a small resident robot for salmon pens. It lives docked on the feed barge, swims out on shifts, grips the
net to work and comes back to charge. One vehicle, three tools on a front tool port: a net-repair clip tool, a
dead-fish (mortality) grabber with a basket, and a net brush. Scampi's and Echensis's family: the same autonomy stack
(on a Jetson Orin NX), cameras and water model, cut down to a price.

Requirements (``REQUIREMENTS``, from the design questions): all three jobs by swappable tool; swim + grip the net (a
suction disc cannot hold on a mesh — water flows through it — so four soft fingers pinch the twine; the belly disc is
the dock's latch); resident on the feed barge; under EUR 40k a unit.

Layout (x forward, y left, z up; z = 0 the underside): a rounded, fish-safe hull (no exposed propellers: six ducted
thrusters with guards, four vectored horizontal and two vertical); the electronics and battery in one 50 m housing;
the camera and lights in the nose; the tool port under the nose; four soft net fingers on the sides of the nose;
the docking disc under the belly (inductive charging and the latch).

``part``: ``vehicle`` (with ``tool``), ``hull``, ``housing``, ``thruster``, ``fingers``, ``tool_body``, ``disc``.
"""
from __future__ import annotations

import math

try:
    import cadquery as cq
except ImportError:
    cq = None
from vegeta.dedalus import Design, Parameter

RHO_SW, G = 1025.0, 9.81
FISH_RHO = 1070.0                   # salmon: a dead fish is ~4 % heavy in seawater
REQUIREMENTS = {
    "site": "salmon pens: 120-160 m circumference, 20-40 m deep nets; currents up to 0.5 m/s in the pen",
    "jobs": ["net inspection + small repair (clip tool)", "dead-fish removal (grabber + basket)", "net brushing"],
    "mobility": "swim + grip the net with soft fingers (no suction on a mesh)",
    "base": "resident: docked on the feed barge, inductive charging, shifts day and night",
    "price_eur": 40000.0,
    "depth_m": 50.0,
    "current_m_s": 0.5,
    "fish_safety": "ducted, guarded thrusters; no sharp edges; slow near fish; dim/red light at night",
}
#: Bought parts: (mass kg, volume L, average power W, price EUR) — class values, not quotes.
PARTS = {
    "Jetson Orin NX 16GB + carrier": (0.3, 0.0, 15.0, 900.0),
    "6 ducted thrusters with guards (T200 class)": (6 * 0.4, 6 * 0.15, 0.0, 6 * 260.0),
    "camera module (stereo) + 4 dimmable lights": (1.5, 0.6, 60.0, 2400.0),
    "IMU + depth sensor": (0.3, 0.1, 3.0, 800.0),
    "mechanical scanning sonar (net distance)": (0.6, 0.3, 5.0, 2800.0),
    "acoustic positioning (USBL at the dock)": (0.4, 0.2, 2.0, 4500.0),
    "4 soft net fingers + drives": (2.0, 1.2, 15.0, 2500.0),
    "LiFePO4 battery 1.2 kWh, 24 V + BMS": (11.0, 6.0, 0.0, 1800.0),
    "electronics housing (Al, 50 m)": (3.0, 9.0, 0.0, 1500.0),
    "hull, frame, foam, fasteners": (5.0, 13.0, 0.0, 3500.0),
    "docking disc (inductive coil + latch plate)": (1.0, 0.6, 0.0, 800.0),
    "assembly, test, cabling": (1.0, 0.3, 0.0, 4000.0),
}
TOOLS = {   # mass kg, volume L, power W while working, price EUR, what
    "clip": (2.0, 1.2, 40.0, 3000.0, "closes holes up to ~0.3 m with polymer clips from a 60-clip magazine"),
    "morts": (3.0, 4.0, 40.0, 2200.0, "grabber + 60 L basket at the net's bottom: up to ~8 fish (5 kg) a trip"),
    "brush": (2.5, 1.5, 150.0, 1600.0, "rotary brush, 0.5 m: spot-cleans the net between the farm's big cleaners"),
}
DOCK_EUR = 6500.0                    # on the barge: inductive charger, latch, the USBL head, Wi-Fi/4G link (one per site)


class Cyclopterus(Design):
    parameters = [
        Parameter("part", "vehicle", choices=("vehicle", "hull", "housing", "thruster", "fingers", "tool_body", "disc"),
                  description="what to build"),
        Parameter("tool", "clip", choices=tuple(TOOLS), description="the tool on the port"),
        Parameter("length", 800.0, "mm", min=300),
        Parameter("width", 600.0, "mm", min=200),
        Parameter("height", 420.0, "mm", min=150),
        Parameter("duct", 100.0, "mm", min=50, description="thruster duct with guard, outer"),
        Parameter("finger_length", 160.0, "mm", min=40),
        Parameter("finger_pinch_N", 60.0, "N", min=0, description="each finger's pinch on the twine (assumed)"),
        Parameter("finger_mu", 0.5, "", min=0.05, description="soft finger on wet nylon/HDPE twine (assumed)"),
        Parameter("camera_swath_m", 1.5, "m", min=0.2, description="net seen per lane at the working distance"),
        Parameter("inspect_speed", 0.4, "m/s", min=0.05),
        Parameter("battery_kwh", 1.2, "kWh", min=0.2),
    ]

    @staticmethod
    def p(over=None):
        return {q.name: q.default for q in Cyclopterus.parameters} | (over or {})

    @classmethod
    def sizing(cls, over=None):
        """Mass and buoyancy with a tool, holding on the net in the current, inspection time per pen, a mortality
        trip, the shift on one charge, the unit's and the site's price."""
        p = cls.p(over)
        tm, tv, tw, tc, _ = TOOLS[p["tool"]]
        m = sum(a for a, _, _, _ in PARTS.values()) + tm
        V = (sum(b for _, b, _, _ in PARTS.values()) + tv) / 1000
        net_up = 0.3                                                       # kg positive: it floats up if it dies
        trim_kg = RHO_SW * V - m - net_up                                  # >0: ballast to add, <0: foam to add
        # holding on the net: drag of the hull broadside in the current, the tool's reaction (brushing), the morts
        L, W, H = p["length"] / 1000, p["width"] / 1000, p["height"] / 1000
        drag = 0.5 * RHO_SW * 0.8 * (L * H) * REQUIREMENTS["current_m_s"] ** 2
        tool_F = {"clip": 20.0, "morts": 15.0, "brush": 60.0}[p["tool"]]
        hold = 4 * p["finger_pinch_N"] * p["finger_mu"] * 2               # two contact faces per pinch
        hold_margin = hold / (2.0 * (drag + tool_F)) - 1
        # inspection: lanes round the pen; a 160 m x 30 m net plus the bottom cone
        net_area = 160.0 * 30.0 + math.pi * (160 / (2 * math.pi)) ** 2 * 1.2
        rate = p["camera_swath_m"] * p["inspect_speed"] * 3600 * 0.8
        # power: hotel + swimming at the inspection speed (CdA ~ 0.8 x frontal), the tool
        hotel = sum(w for _, _, w, _ in PARTS.values())
        Dsw = 0.5 * RHO_SW * 0.8 * (W * H) * (p["inspect_speed"] + 0.25) ** 2
        swim_W = Dsw * (p["inspect_speed"] + 0.25) / 0.25 + 40.0
        usable = 0.9 * p["battery_kwh"] * 1000
        shift_inspect_h = usable / (hotel + swim_W)
        work_W = hotel + 0.3 * swim_W + tw
        shift_work_h = usable / work_W
        morts_kg = 8 * 5.0
        morts_in_water = morts_kg * (1 - RHO_SW / FISH_RHO)
        price = sum(c for _, _, _, c in PARTS.values()) + tc
        return {"mass_in_air_kg": m, "displacement_L": V * 1000, "trim_kg": trim_kg, "drag_N": drag, "hold_N": hold,
                "hold_margin": hold_margin, "net_area_m2": net_area, "inspect_rate_m2_h": rate,
                "pen_inspection_h": net_area / rate, "hotel_W": hotel, "swim_W": swim_W,
                "shift_inspection_h": shift_inspect_h, "shift_tool_work_h": shift_work_h,
                "pens_per_charge": shift_inspect_h / (net_area / rate),
                "morts_trip_kg": morts_kg, "morts_in_water_kg": morts_in_water,
                "unit_price_eur": price, "price_with_all_tools_eur": sum(c for _, _, _, c in PARTS.values()) + sum(t[3] for t in TOOLS.values()),
                "dock_eur": DOCK_EUR}

    # ------------------------------------------------------------------------------------------- CAD
    def _hull(self, p):
        L, W, H = p["length"], p["width"], p["height"]
        body = cq.Workplane("XY").box(L, W, H).edges().fillet(min(W, H) * 0.42).translate((0, 0, H / 2))
        nose = cq.Workplane("XY").sphere(H * 0.40).translate((L / 2 - H * 0.25, 0, H * 0.55))
        fin = cq.Workplane("XZ").polyline([(0, 0), (-120, 0), (-160, 90), (-40, 40)]).close().extrude(10, both=True) \
            .translate((-L / 2 + 40, 0, H * 0.75))                                   # a dorsal crest, the lumpfish's
        return body.union(nose).union(fin)

    def _housing(self, p):
        return cq.Workplane("YZ").center(0, p["height"] * 0.45).circle(85).extrude(p["length"] * 0.28, both=True)

    def thruster_layout(self, p):
        L, W, H = p["length"], p["width"], p["height"]
        out = []
        for sx in (1, -1):
            for sy in (1, -1):
                out.append(((sx * L * 0.30, sy * (W / 2 + p["duct"] * 0.35), H * 0.45),
                            (math.cos(math.radians(30)) * sx, sy * math.sin(math.radians(30)), 0.0)))
        for sy in (1, -1):
            out.append(((0.0, sy * W * 0.30, H - 10), (0.0, 0.0, 1.0)))
        return out

    def _thruster(self, p, at=(0, 0, 0), axis=(1, 0, 0)):
        d = p["duct"]
        a = cq.Vector(*axis).normalized()
        base = cq.Vector(*at) - a * 50
        duct = cq.Solid.makeCylinder(d / 2, 100, base, a).cut(cq.Solid.makeCylinder(d / 2 - 6, 100, base, a))
        out = cq.Workplane("XY").add(duct)
        for k in range(4):                                                  # the guard bars across both ends
            ang = math.radians(45 * k)
            u = a.cross(cq.Vector(0, 0, 1) if abs(a.z) < 0.9 else cq.Vector(1, 0, 0)).normalized()
            w = a.cross(u)
            dvec = (u * math.cos(ang) + w * math.sin(ang)) * (d - 12)
            for end in (base, base + a * 100):
                c = end - dvec * 0.5
                out = out.add(cq.Solid.makeCylinder(2.0, dvec.Length, c, dvec))
        return cq.Workplane("XY").add(out.vals()).combine()

    def _fingers(self, p):
        L, W, H, fl = p["length"], p["width"], p["height"], p["finger_length"]
        out = None
        for sy in (1, -1):
            for sz in (0.35, 0.75):
                a = cq.Vector(L / 2 - 60, sy * W * 0.32, H * sz)
                for k, d in enumerate((cq.Vector(fl * 0.7, sy * fl * 0.3, 0), cq.Vector(fl * 0.4, sy * fl * 0.05, -fl * 0.25))):
                    seg = cq.Workplane("XY").add(cq.Solid.makeCylinder(11 - 3 * k, d.Length, a, d))
                    out = seg if out is None else out.union(seg)
                    a = a + d
        return out

    def _tool(self, p):
        L, H = p["length"], p["height"]
        x = L / 2 + 40
        if p["tool"] == "clip":
            body = cq.Workplane("XY").box(160, 90, 80).translate((x, 0, 60))
            return body.union(cq.Workplane("XY").box(60, 30, 40).translate((x + 100, 0, 50)))
        if p["tool"] == "brush":
            body = cq.Workplane("XY").box(120, 90, 80).translate((x, 0, 60))
            return body.union(cq.Workplane("YZ").circle(120).extrude(30).translate((x + 60, 0, 60)))
        basket = (cq.Workplane("XY").box(420, 420, 340).translate((0, 0, -170 - 5))
                  .cut(cq.Workplane("XY").box(400, 400, 330).translate((0, 0, -170 + 5))))   # an open-top basket under
        jaw = cq.Workplane("XY").box(140, 80, 60).translate((x, 0, 50))
        return basket.union(jaw)

    def _disc(self, p):
        return cq.Workplane("XY").circle(110).extrude(15).translate((-p["length"] * 0.1, 0, -5))

    def build(self, p):
        if cq is None:
            raise ImportError("Cyclopterus's CAD needs CadQuery")
        part = p["part"]
        single = {"hull": self._hull, "housing": self._housing, "fingers": self._fingers, "tool_body": self._tool,
                  "disc": self._disc}
        if part in single:
            return single[part](p)
        if part == "thruster":
            return self._thruster(p)
        body = self._hull(p).union(self._disc(p)).union(self._fingers(p)).union(self._tool(p))
        for at, ax in self.thruster_layout(p):
            body = body.union(self._thruster(p, at, ax))
        return body


if __name__ == "__main__":
    for tool in TOOLS:
        s = Cyclopterus.sizing({"tool": tool})
        print(f"[{tool}] mass {s['mass_in_air_kg']:.1f} kg, displacement {s['displacement_L']:.1f} L, trim {s['trim_kg']:+.1f} kg, "
              f"hold {s['hold_N']:.0f} N vs drag {s['drag_N']:.0f} N (margin {s['hold_margin']:+.2f}), "
              f"tool work {s['shift_tool_work_h']:.1f} h, price EUR {s['unit_price_eur']:,.0f}")
    s = Cyclopterus.sizing()
    print(f"net {s['net_area_m2']:.0f} m^2, inspect {s['inspect_rate_m2_h']:.0f} m^2/h -> {s['pen_inspection_h']:.1f} h a pen; "
          f"shift inspecting {s['shift_inspection_h']:.1f} h ({s['pens_per_charge']:.1f} pens a charge); hotel {s['hotel_W']:.0f} W, swim {s['swim_W']:.0f} W")
    print(f"morts trip {s['morts_trip_kg']:.0f} kg of fish = {s['morts_in_water_kg']:.1f} kg in water; "
          f"all tools EUR {s['price_with_all_tools_eur']:,.0f}, dock EUR {s['dock_eur']:,.0f} a site")
