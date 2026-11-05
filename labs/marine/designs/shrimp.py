"""Shrimp: the smallest of the family (Scampi, Echensis, Cyclopterus, Nephro, Squilla: ``scampi.md`` and on) — a
backpack-held craft for aquaculture work. One person carries it to the pen, drops it in by hand, flies it from a
tablet on a 50 m tether, and packs it back. Jobs: net inspection (holes, biofouling, mooring lines), mortality
spotting and counting, feeding and behaviour observation, water sampling, and the daily oxygen profile (the probe
pod is standard). The brain is on the surface: the family's net-following and hole/mort detection run on a Jetson
Orin Nano in the tether box, the vehicle keeps only a microcontroller (thrusters, IMU, depth, the video encoder) and
an auto-surface reflex (round 2 of ``shrimp.md``).

Requirements (``REQUIREMENTS``): the whole kit in one backpack under 10 kg; the vehicle under 5 kg and hand-launched;
50 m depth (nets are 20–40 m); works in the pen's current; fish-safe (ducted, guarded thrusters, no sharp edges);
under EUR 6k a kit.

Layout (x forward, y left, z up; z = 0 the underside): a rounded shell round one Ø90 mm electronics tube (the
camera dome at its front end, the tether port at its back), six small ducted thrusters (four vectored horizontal at
45°, two vertical), two lights beside the dome, a tool port under the nose, a carry handle on top, two skids.

``part``: ``vehicle`` (with ``tool``), ``shell``, ``housing``, ``thrusters``, ``lights``, ``tool_body``, ``handle``, ``skids``.
"""
from __future__ import annotations

import math

try:
    import cadquery as cq
except ImportError:
    cq = None
from vegeta.dedalus import Design, Parameter

RHO_SW, G = 1025.0, 9.81
REQUIREMENTS = {
    "site": "salmon and trout pens; also mussel and seaweed lines: 20-40 m deep nets, currents to 0.5 m/s",
    "jobs": ["net inspection (holes, fouling, mooring)", "mortality spotting and counting", "feeding / behaviour observation",
             "water sampling and probing at depth"],
    "carry": "one backpack, under 10 kg all in; the vehicle under 5 kg, hand-launched from a walkway or boat",
    "depth_m": 50.0,
    "current_m_s": 0.5,
    "tether_m": 50.0,
    "price_eur": 6000.0,
    "fish_safety": "ducted, guarded thrusters; no sharp edges; slow near fish; red light at night",
}
#: Bought parts: (mass kg, volume L, average power W, price EUR) — class values, not quotes.
PARTS = {
    "vehicle MCU (thrusters, IMU, depth) + video encoder": (0.05, 0.02, 3.0, 60.0),
    "6 micro ducted thrusters with guards (60 mm)": (6 * 0.14, 6 * 0.06, 0.0, 6 * 110.0),
    "camera (4K, low light) + 2 dimmable white/red lights + strobe": (0.27, 0.16, 12.0, 380.0),
    "IMU + depth + leak sensors": (0.05, 0.02, 1.0, 150.0),
    "tether interface (Ethernet over 2-wire)": (0.05, 0.02, 2.0, 120.0),
    "Li-ion battery 100 Wh, 14.4 V + BMS (swappable)": (0.60, 0.35, 0.0, 120.0),
    "electronics tube Ø90 x 250 (Al / acrylic, 50 m) + dome": (0.90, 1.60, 0.0, 380.0),
    "shell, frame, foam, handle, skids (printed PA12 + foam)": (1.20, 1.90, 0.0, 260.0),
    "assembly, cabling, test": (0.20, 0.05, 0.0, 500.0),
}
#: What stays on the surface: (mass kg, price EUR).
SURFACE = {
    "tether 50 m, 3 mm, neutral + hand reel": (1.60, 650.0),
    "tether box with the Jetson Orin Nano (the brain) + rugged tablet + gamepad": (1.40, 1250.0),
    "backpack, spare battery, charger": (1.40, 220.0),
}
TOOLS = {   # mass kg, volume L, power W while working, price EUR, what
    "probe": (0.30, 0.20, 2.0, 650.0, "the standard pod: dissolved oxygen, temperature, salinity, logged with depth (the daily O2 profile)"),
    "sampler": (0.35, 0.30, 5.0, 180.0, "a 250 mL syringe sampler, triggered at depth: water for lab analysis"),
    "tags": (0.25, 0.20, 3.0, 120.0, "a magazine of six numbered clip-on tags: marks a hole or a damaged line for the divers; nothing is repaired"),
    "none": (0.0, 0.0, 0.0, 0.0, "camera only"),
}
PROFILE = {"depth_m": 40.0, "vertical_m_s": 0.5, "dwell_s": 10.0}        # the oxygen profile dive: down the net's depth and back
TETHER_D, TETHER_CD = 0.003, 1.2


class Shrimp(Design):
    parameters = [
        Parameter("part", "vehicle", choices=("vehicle", "shell", "housing", "thrusters", "lights", "tool_body", "handle", "skids"),
                  description="what to build"),
        Parameter("tool", "probe", choices=tuple(TOOLS), description="the tool on the port"),
        Parameter("length", 360.0, "mm", min=200),
        Parameter("width", 230.0, "mm", min=120),
        Parameter("height", 130.0, "mm", min=80),
        Parameter("tube_d", 90.0, "mm", min=60),
        Parameter("duct", 60.0, "mm", min=30, description="thruster duct with guard, outer"),
        Parameter("thrust_each_N", 6.0, "N", min=1, description="each thruster's bollard thrust"),
        Parameter("thruster_W", 60.0, "W", min=5, description="each thruster at full thrust"),
        Parameter("camera_swath_m", 1.0, "m", min=0.2, description="net seen per lane at the working distance"),
        Parameter("inspect_speed", 0.3, "m/s", min=0.05),
        Parameter("battery_wh", 100.0, "Wh", min=20),
        Parameter("tether_exposed_m", 25.0, "m", min=1, description="tether in the current between the surface and the craft"),
    ]

    @staticmethod
    def p(over=None):
        return {q.name: q.default for q in Shrimp.parameters} | (over or {})

    @classmethod
    def sizing(cls, over=None):
        """Mass and buoyancy with the tool, the backpack, speed and station-keeping against the current with the
        tether's drag, endurance, the inspection rate, the price."""
        p = cls.p(over)
        tm, tv, tw, tc, _ = TOOLS[p["tool"]]
        m = sum(a for a, _, _, _ in PARTS.values()) + tm
        V = (sum(b for _, b, _, _ in PARTS.values()) + tv) / 1000
        net_up = 0.05                                                      # kg positive: it floats up if it dies
        trim_kg = RHO_SW * V - m - net_up                                  # >0: ballast to add, <0: foam to add
        backpack = m + sum(a for a, _ in SURFACE.values())
        L, W, H = p["length"] / 1000, p["width"] / 1000, p["height"] / 1000
        cda_fwd = 0.7 * W * H                                              # a rounded shell, frontal
        cda_side = 0.9 * L * H
        T_fwd = 4 * p["thrust_each_N"] * math.cos(math.radians(45))        # four vectored, all pulling forward
        T_side = T_fwd
        v_max = math.sqrt(T_fwd / (0.5 * RHO_SW * cda_fwd))
        # station-keeping in the current: the shell's drag plus the tether's (a cylinder across the flow, Cd 1.2)
        u = REQUIREMENTS["current_m_s"]
        tether_drag = 0.5 * RHO_SW * u * u * TETHER_CD * TETHER_D * p["tether_exposed_m"]
        hull_drag = 0.5 * RHO_SW * u * u * cda_fwd
        hold_margin = T_fwd / (hull_drag + tether_drag) - 1
        # the current it can hold at the set tether length, and the tether length it can afford at 0.5 m/s
        u_max = math.sqrt(T_fwd / (0.5 * RHO_SW * (cda_fwd + TETHER_CD * TETHER_D * p["tether_exposed_m"])))
        tether_max = (T_fwd / (0.5 * RHO_SW * u * u) - cda_fwd) / (TETHER_CD * TETHER_D)
        # endurance: hotel + swimming at the inspection speed (plus a 0.2 m/s current), the tool
        hotel = sum(w for _, _, w, _ in PARTS.values())
        v_s = p["inspect_speed"] + 0.2
        D = 0.5 * RHO_SW * cda_fwd * v_s * v_s
        swim_W = D * v_s / 0.30 + 6 * 2.0                                  # propulsive efficiency 0.3, the ESCs idling
        usable = 0.9 * p["battery_wh"]
        dive_h = usable / (hotel + swim_W + 0.3 * tw)
        # inspection: lanes round the pen; a 160 m x 30 m net plus the bottom cone
        net_area = 160.0 * 30.0 + math.pi * (160 / (2 * math.pi)) ** 2 * 1.2
        rate = p["camera_swath_m"] * p["inspect_speed"] * 3600 * 0.8
        price = sum(c for _, _, _, c in PARTS.values()) + sum(c for _, c in SURFACE.values()) + tc
        profile_min = (2 * PROFILE["depth_m"] / PROFILE["vertical_m_s"] + PROFILE["dwell_s"] * (PROFILE["depth_m"] / 5)) / 60
        return {"mass_in_air_kg": m, "displacement_L": V * 1000, "trim_kg": trim_kg, "backpack_kg": backpack,
                "v_max_m_s": v_max, "thrust_fwd_N": T_fwd, "hull_drag_N": hull_drag, "tether_drag_N": tether_drag,
                "hold_margin": hold_margin, "current_max_m_s": u_max, "tether_max_at_0.5_m": tether_max,
                "hotel_W": hotel, "swim_W": swim_W, "dive_h": dive_h, "net_area_m2": net_area, "inspect_rate_m2_h": rate,
                "pen_inspection_h": net_area / rate, "dives_per_pen": (net_area / rate) / dive_h, "profile_min": profile_min,
                "kit_price_eur": price, "price_all_tools_eur": price - tc + sum(t[3] for t in TOOLS.values())}

    # ------------------------------------------------------------------------------------------- CAD
    def _shell(self, p):
        L, W, H = p["length"], p["width"], p["height"]
        body = cq.Workplane("XY").box(L, W, H).edges().fillet(min(W, H) * 0.38).translate((0, 0, H / 2))
        # the tube's bore through the shell (the dome and the port show at both ends)
        return body.cut(cq.Workplane("YZ").center(0, H * 0.55).circle(p["tube_d"] / 2 + 1.0).extrude(L, both=True))

    def _housing(self, p):
        L, H, d = p["length"], p["height"], p["tube_d"]
        tube = cq.Workplane("YZ").center(0, H * 0.55).circle(d / 2).extrude(L * 0.36, both=True)
        dome = cq.Workplane("XY").sphere(d / 2).translate((L * 0.36, 0, H * 0.55))
        port = cq.Workplane("YZ").workplane(offset=-L * 0.36 - 12).center(0, H * 0.55).circle(d / 2 - 8).extrude(12)
        return tube.union(dome).union(port)

    def thruster_layout(self, p):
        L, W, H = p["length"], p["width"], p["height"]
        c = math.cos(math.radians(45))
        out = []
        for sx in (1, -1):
            for sy in (1, -1):
                out.append(((sx * L * 0.30, sy * (W / 2 + p["duct"] * 0.35), H * 0.5), (sx * c, sy * c, 0)))
        for sx in (1, -1):
            out.append(((sx * L * 0.12, 0, H + p["duct"] * 0.1), (0, 0, 1)))
        return out

    def _thruster(self, p, at, axis):
        d = p["duct"]
        ax = cq.Vector(*axis)
        duct = cq.Workplane("XY").add(cq.Solid.makeCylinder(d / 2, d * 0.6, cq.Vector(*at) - ax * d * 0.3, ax))
        bore = cq.Workplane("XY").add(cq.Solid.makeCylinder(d / 2 - 5, d * 0.6, cq.Vector(*at) - ax * d * 0.3, ax))
        hub = cq.Workplane("XY").add(cq.Solid.makeCylinder(d * 0.18, d * 0.5, cq.Vector(*at) - ax * d * 0.25, ax))
        return duct.cut(bore).union(hub)

    def _thrusters(self, p):
        out = None
        for at, a in self.thruster_layout(p):
            t = self._thruster(p, at, a)
            out = t if out is None else out.union(t)
        return out

    def _lights(self, p):
        L, W, H = p["length"], p["width"], p["height"]
        out = None
        for sy in (1, -1):
            l = cq.Workplane("YZ").workplane(offset=L * 0.30).center(sy * (p["tube_d"] / 2 + 22), H * 0.55).circle(12).extrude(25)
            out = l if out is None else out.union(l)
        return out

    def _tool(self, p):
        L, H = p["length"], p["height"]
        if p["tool"] == "none":
            return cq.Workplane("XY").box(30, 30, 8).translate((L * 0.25, 0, 4))                     # the port's blank
        if p["tool"] == "sampler":
            return cq.Workplane("YZ").workplane(offset=L * 0.05).center(0, -14).circle(14).extrude(140)
        if p["tool"] == "probe":
            return cq.Workplane("XY").box(70, 40, 24).edges("|X").fillet(6).translate((L * 0.22, 0, -12))
        return cq.Workplane("XY").box(70, 30, 18).edges("|Z").fillet(5).translate((L * 0.28, 0, -9))    # the tag magazine

    def _handle(self, p):
        L, H = p["length"], p["height"]
        return (cq.Workplane("XZ").polyline([(-70, 0), (-70, 36), (70, 36), (70, 0), (50, 0), (50, 22), (-50, 22), (-50, 0)])
                .close().extrude(10, both=True).translate((-L * 0.05, 0, H)))

    def _skids(self, p):
        L, W = p["length"], p["width"]
        out = None
        for sy in (1, -1):
            s = cq.Workplane("XY").box(L * 0.7, 12, 10).edges("|Y").fillet(4).translate((0, sy * W * 0.3, -5))
            out = s if out is None else out.union(s)
        return out

    def build(self, p):
        parts = {"shell": self._shell, "housing": self._housing, "thrusters": self._thrusters, "lights": self._lights,
                 "tool_body": self._tool, "handle": self._handle, "skids": self._skids}
        if p["part"] in parts:
            return parts[p["part"]](p)
        out = None
        for name in parts:
            s = parts[name](p)
            out = s if out is None else out.union(s)
        return out


if __name__ == "__main__":
    for tool in TOOLS:
        s = Shrimp.sizing({"tool": tool})
        print(f"{tool:8s} {s['mass_in_air_kg']:.2f} kg, trim {s['trim_kg']:+.2f} kg, backpack {s['backpack_kg']:.1f} kg, v_max {s['v_max_m_s']:.2f} m/s, "
              f"hold margin {s['hold_margin']:+.2f} at 25 m tether, dive {s['dive_h']:.1f} h, kit EUR {s['kit_price_eur']:.0f}")
    s = Shrimp.sizing()
    print(f"current max at 25 m tether {s['current_max_m_s']:.2f} m/s; tether max at 0.5 m/s {s['tether_max_at_0.5_m']:.0f} m; "
          f"pen {s['pen_inspection_h']:.1f} h = {s['dives_per_pen']:.1f} dives; an O2 profile {s['profile_min']:.0f} min; all tools EUR {s['price_all_tools_eur']:.0f}")
