"""Echeneis (Echeneis naucrates, the live sharksucker remora): a tethered hull crawler for large merchant ships — it
swims from the quay to the hull, holds on with magnetic tracks and a suction thruster, cleans with fouling capture
and inspects on the same pass. Scampi's sibling (``scampi.py``): the same Jetson autonomy, cameras, UT/CP probe and
water physics, a different vehicle.

Requirements (``REQUIREMENTS``, from the design questions): magnets + thrust suction (steel first, aluminium/composite
at lower grip); cleaning with capture + inspection; bulkers, tankers and container ships (flat steel sides, 10-25 m
draft); power, data and the fouling return hose down a tether from the quay or a boat.

Layout (x forward = the crawl direction, y left, z out of the hull; z = 0 the hull): a flat chassis between two
magnetic tracks; the cleaning head at the front (two counter-rotating brush discs under a hood whose suction draws the
fouling into the return hose); the suction thruster in the middle (pulls the chassis onto the hull); four small swim
thrusters for the transit and hull-to-hull hops; the electronics housing, the cameras and the UT/CP probe at the rear;
the tether and hose at the back, off the hull.

``part``: ``vehicle``, ``chassis``, ``track`` (one), ``head`` (brushes and hood), ``suction`` (the thruster duct),
``housing``, ``tether``.
"""
from __future__ import annotations

import math

try:
    import cadquery as cq
except ImportError:
    cq = None
from vegeta.dedalus import Design, Parameter

RHO_SW, G, NU_SW = 1025.0, 9.81, 1.05e-6
SAFETY_GRIP = 2.0                   # grip over the worst load (wet, fouled paint, gusts of current)
REQUIREMENTS = {
    "ships": "bulkers, tankers, container ships: flat steel sides, 10-25 m draft",
    "adhesion": "magnetic tracks + suction thruster; steel first, aluminium/composite at lower grip",
    "jobs": ["hull cleaning with capture of the fouling", "UT thickness, CP/anode, photogrammetry inspection"],
    "power": "tether from the quay or a boat: power, fibre, the fouling return hose",
    "depth_m": 30.0,
    "current_m_s": 1.5,             # ~3 kn in a port or at anchor
    "hull_area_m2": {"Supramax bulker 190 m": 7500.0, "Aframax tanker 245 m": 11000.0, "14k TEU container 366 m": 20000.0},
    "clean_window_h": 12.0,         # a port call's working window
    "capture": "fouling drawn into a hood and pumped ashore through a filter (in-water cleaning with capture)",
}
PARTS = {   # mass in air [kg], volume [L], power [W]
    "Jetson AGX Orin + carrier": (0.9, 0.0, 40.0),
    "cameras (4) + lights (4)": (3.0, 1.2, 150.0),
    "UT thickness probe + CP probe": (1.0, 0.4, 10.0),
    "INS + depth + hull-relative odometry": (1.5, 0.6, 15.0),
    "4 swim thrusters (T200 class)": (4 * 0.35, 4 * 0.15, 0.0),
    "track drives (2 x geared BLDC, sealed)": (2 * 3.5, 2 * 1.2, 0.0),
    "brush drives (2 x BLDC)": (2 * 2.5, 2 * 0.9, 0.0),
}


class Echeneis(Design):
    parameters = [
        Parameter("part", "vehicle", choices=("vehicle", "chassis", "track", "head", "suction", "housing", "tether"),
                  description="what to build"),
        Parameter("length", 1000.0, "mm", min=400, description="chassis length (without the head)"),
        Parameter("width", 800.0, "mm", min=300),
        Parameter("height", 260.0, "mm", min=100, description="hull to the top of the chassis"),
        Parameter("track_width", 120.0, "mm", min=40),
        Parameter("track_gap", 6.0, "mm", min=1, description="magnet face to the hull (paint + fouling + rubber)"),
        Parameter("magnet_N_per_m", 3500.0, "N/m", min=0, description="track pull per metre of contact at the gap (NdFeB blocks, steel hull; assumed)"),
        Parameter("suction_duct", 200.0, "mm", min=50, description="suction thruster duct (pulls the chassis onto the hull)"),
        Parameter("suction_N", 400.0, "N", min=0, description="suction thruster pull (T500 class at the duct; assumed)"),
        Parameter("brush_diameter", 500.0, "mm", min=100, description="each of the two brush discs"),
        Parameter("head_length", 560.0, "mm", min=100, description="the cleaning head ahead of the chassis"),
        Parameter("crawl_speed", 0.25, "m/s", min=0.01, description="cleaning speed along the hull"),
        Parameter("hood_inlet_speed", 0.3, "m/s", min=0.05, description="water speed into the hood's rim (holds the loosened fouling)"),
        Parameter("rim_gap", 15.0, "mm", min=3, description="gap between the hood's skirt and the hull"),
        Parameter("hose_diameter", 100.0, "mm", min=20, description="return hose to the quay's filter"),
        Parameter("hose_length", 60.0, "m", min=5),
        Parameter("tether_voltage", 400.0, "V", min=48, description="DC power down the tether"),
        Parameter("housing_od", 160.0, "mm", min=60),
        Parameter("housing_wall", 4.0, "mm", min=1, description="Al 6061-T6 at 30 m (x 1.5)"),
        Parameter("housing_length", 500.0, "mm", min=100),
    ]

    # ------------------------------------------------------------------------------------------- sizing
    @staticmethod
    def p(over=None):
        return {q.name: q.default for q in Echeneis.parameters} | (over or {})

    @classmethod
    def masses(cls, p):
        """Mass in air and displaced volume of the chassis, tracks, head, housing and parts [kg, L]."""
        L, W, H = p["length"] / 1000, p["width"] / 1000, p["height"] / 1000
        items = {
            "chassis + hood (HDPE, Al brackets)": (18.0, 17.0),
            "magnetic tracks (NdFeB blocks, steel links, rubber)": (2 * 9.0, 2 * 1.6),
            "suction thruster (T500 class) + duct": (2.0, 1.0),
            "brush discs + hood rim": (4.0, 3.0),
            "electronics housing": (5.0 + 2.0, math.pi * (p["housing_od"] / 2000) ** 2 * p["housing_length"] / 1000 * 1000),
            "foam (syntactic 300 m class)": (0.0, 0.0),
        }
        items.update({k: (m, v) for k, (m, v, _) in PARTS.items()})
        m = sum(a for a, _ in items.values())
        V = sum(b for _, b in items.values()) / 1000
        return items, m, V

    @classmethod
    def sizing(cls, over=None):
        """Grip on a vertical side and under the flat bottom in the current, the cleaning rate against the hulls,
        the capture flow and pump, tether power."""
        p = cls.p(over)
        items, m_dry, V_dry = cls.masses(p)
        net_down = 2.0                                                     # kg heavy in water: it sinks if it lets go
        V_f = max(0.0, (m_dry - net_down - RHO_SW * V_dry) / (RHO_SW - 420.0))
        m_air = m_dry + 420.0 * V_f
        w_water = net_down * G
        # loads along the hull: current drag on the chassis (side area while crawling across), the brushes' reaction
        L, W, H = p["length"] / 1000, p["width"] / 1000, p["height"] / 1000
        drag = 0.5 * RHO_SW * 1.1 * (W * H) * REQUIREMENTS["current_m_s"] ** 2
        lift = 0.5 * RHO_SW * 0.3 * (L * W) * REQUIREMENTS["current_m_s"] ** 2   # the current pulls it off the hull
        brush_F = 120.0                                                    # the discs' net scrubbing reaction [N] (assumed)
        contact = 2 * L * 0.9                                              # track length on the hull [m]
        f_mag = p["magnet_N_per_m"] * contact
        mu = 0.45                                                          # rubber on wet painted steel
        cases = {
            "vertical side, crawling up": (f_mag + p["suction_N"] - lift, w_water + drag + brush_F),
            "flat bottom, upside down": (f_mag + p["suction_N"] - lift - w_water, drag + brush_F),
        }
        # composite / aluminium hulls: suction only — inspection without brushing, in a slack current (0.5 m/s)
        k2 = (0.5 / REQUIREMENTS["current_m_s"]) ** 2
        cases["composite hull, inspection at 0.5 m/s"] = (p["suction_N"] - lift * k2, w_water + drag * k2)
        grip = {k: {"normal_N": n, "tangential_N": t, "margin": (mu * n) / (SAFETY_GRIP * t) - 1} for k, (n, t) in cases.items()}
        # cleaning rate: the two discs' swath at the crawl speed, 85 % efficient lanes (overlap, turns)
        swath = 2 * p["brush_diameter"] / 1000 * 0.95
        rate = swath * p["crawl_speed"] * 3600 * 0.85                      # m^2/h
        hulls = {k: {"hours": a / rate, "robots_for_window": math.ceil(a / rate / REQUIREMENTS["clean_window_h"])}
                 for k, a in REQUIREMENTS["hull_area_m2"].items()}
        # capture: the hood's rim draws water in at hood_inlet_speed; the flow goes up the hose to the quay's filter
        rim = 2 * (swath + p["head_length"] / 1000) * p["rim_gap"] / 1000
        Q = rim * p["hood_inlet_speed"]
        D = p["hose_diameter"] / 1000
        v = Q / (math.pi * D * D / 4)
        Re = v * D / NU_SW
        f = 0.25 / math.log10(5e-5 / (3.7 * D) + 5.74 / Re ** 0.9) ** 2    # Swamee-Jain, smooth hose
        head = f * p["hose_length"] / D * v * v / (2 * G) + 1.5 * v * v / (2 * G) + 3.0   # + filter (3 m, assumed)
        pump_W = RHO_SW * G * Q * head / 0.5
        hotel = sum(w for _, _, w in PARTS.values())
        drive_W = 2 * 0.5 * (cases["vertical side, crawling up"][1]) * p["crawl_speed"] / 0.5 + 60.0
        brush_W = 2 * 350.0                                                # two discs on fouled paint (assumed)
        suction_W = 900.0                                                  # T500 class at ~400 N bollard pull
        total = hotel + drive_W + brush_W + suction_W + pump_W
        return {"mass_in_air_kg": m_air, "dry_mass_kg": m_dry, "foam_L": V_f * 1000, "in_water_kg": net_down,
                "items": items, "drag_N": drag, "lift_N": lift, "magnet_N": f_mag, "grip": grip,
                "clean_rate_m2_h": rate, "hulls": hulls,
                "capture_flow_L_s": Q * 1000, "hose_speed_m_s": v, "hose_head_m": head, "pump_W": pump_W,
                "power_W": {"hotel": hotel, "tracks": drive_W, "brushes": brush_W, "suction": suction_W, "capture pump": pump_W,
                            "total": total},
                "tether_current_A": total / p["tether_voltage"]}

    # ------------------------------------------------------------------------------------------- CAD
    def _box(self, lx, ly, lz, c):
        return cq.Workplane("XY").box(lx, ly, lz).translate(c)

    def _chassis(self, p):
        L, W, H, tw = p["length"], p["width"], p["height"], p["track_width"]
        body = (cq.Workplane("XY").box(L, W - 2 * tw - 20, H - 60).edges("|X").fillet(30)
                .translate((0, 0, 60 + (H - 60) / 2)))
        return body.cut(cq.Workplane("XY").circle(p["suction_duct"] / 2).extrude(H + 10))

    def _track(self, p, side=1):
        L, W, tw = p["length"], p["width"], p["track_width"]
        y = side * (W / 2 - tw / 2)
        r = 75.0
        belt = (cq.Workplane("XZ").center(0, r + p["track_gap"]).slot2D(L, 2 * r).extrude(tw / 2, both=True)
                .translate((0, y, 0)))
        return belt.cut(cq.Workplane("XZ").center(0, r + p["track_gap"]).slot2D(L - 40, 2 * r - 40).extrude(tw, both=True).translate((0, y, 0)))

    def _head(self, p):
        L, W = p["length"], p["width"]
        hl, d = p["head_length"], p["brush_diameter"]
        x = L / 2 + hl / 2
        hw = 2 * d * 0.95 + 60                                             # the hood covers both discs
        hood = (cq.Workplane("XY").box(hl, hw, 110).edges("|Z").fillet(40).translate((x, 0, 30 + 55))
                .cut(cq.Workplane("XY").box(hl - 20, hw - 20, 100).translate((x, 0, 30 + 45))))
        out = hood
        for s in (1, -1):
            disc = cq.Workplane("XY").center(x, s * d / 2 * 0.95).circle(d / 2).extrude(30).translate((0, 0, 8))
            out = out.union(disc)
        port = cq.Workplane("XY").center(x, 0).circle(p["hose_diameter"] / 2 + 8).extrude(120).translate((0, 0, 130))
        return out.union(port)

    def _suction(self, p):
        d = p["suction_duct"]
        return (cq.Workplane("XY").circle(d / 2 + 8).circle(d / 2).extrude(p["height"] - 20).translate((0, 0, 20)))

    def _housing(self, p):
        r = p["housing_od"] / 2
        return cq.Workplane("YZ").center(0, p["height"] + r - 10).circle(r).extrude(p["housing_length"] / 2, both=True) \
            .translate((-p["length"] * 0.18, 0, 0))

    def _swim_thrusters(self, p):
        L, W = p["length"], p["width"]
        out = None
        for sx in (1, -1):
            for sy in (1, -1):
                t = (cq.Workplane("YZ").circle(50).circle(44).extrude(45, both=True)
                     .translate((sx * L * 0.3, sy * (W / 2 - 160), p["height"] + 30)))
                out = t if out is None else out.union(t)
        return out

    def _tether(self, p):
        a = cq.Vector(-p["length"] / 2 + 60, 0, p["height"] - 20)
        b = a + cq.Vector(-600, 0, 450)
        return cq.Workplane("XY").add(cq.Solid.makeCylinder(25, (b - a).Length, a, b - a))

    def build(self, p):
        if cq is None:
            raise ImportError("Echeneis's CAD needs CadQuery")
        part = p["part"]
        single = {"chassis": self._chassis, "head": self._head, "suction": self._suction, "housing": self._housing,
                  "tether": self._tether}
        if part in single:
            return single[part](p)
        if part == "track":
            return self._track(p)
        body = self._chassis(p).union(self._track(p, 1)).union(self._track(p, -1)).union(self._head(p))
        body = body.union(self._suction(p)).union(self._housing(p)).union(self._swim_thrusters(p)).union(self._tether(p))
        return body


if __name__ == "__main__":
    s = Echeneis.sizing()
    print(f"mass in air {s['mass_in_air_kg']:.1f} kg (dry {s['dry_mass_kg']:.1f}), foam {s['foam_L']:.1f} L, {s['in_water_kg']:g} kg heavy in water")
    print(f"current drag {s['drag_N']:.0f} N, lift-off {s['lift_N']:.0f} N, magnets {s['magnet_N']:.0f} N")
    for k, g in s["grip"].items():
        print(f"  {k:30s} normal {g['normal_N']:6.0f} N, along the hull {g['tangential_N']:5.0f} N, margin {g['margin']:+.2f}")
    print(f"cleaning {s['clean_rate_m2_h']:.0f} m^2/h")
    for k, h in s["hulls"].items():
        print(f"  {k:28s} {h['hours']:5.1f} h -> {h['robots_for_window']} robot(s) for a 12 h window")
    print(f"capture {s['capture_flow_L_s']:.1f} L/s, hose {s['hose_speed_m_s']:.2f} m/s, head {s['hose_head_m']:.1f} m, pump {s['pump_W']:.0f} W")
    print("power", {k: round(v) for k, v in s["power_W"].items()}, f"tether {s['tether_current_A']:.1f} A")
