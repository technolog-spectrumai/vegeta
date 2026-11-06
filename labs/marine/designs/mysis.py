"""Mysis (the mysid shrimp, which live in swarms): a small portable submarine for maritime maintenance that works in
groups. One person carries a unit in its case; a group of them — four to six — goes onto one ship's hull, a pier or a
mooring field under one surface node (on the workboat or a buoy: GPS, the USBL head, the acoustic modem, two charging
bays) and one supervisor. Untethered: the units swim to the hull, land belly-first on a magnetic roller pad (the
press for the brush comes free on steel; the vertical thrusters only land and lift off), and groom it with the belly
brush lane by lane, each lane its own; they share the hull map and the lanes over the modem.

Jobs: hull grooming (light fouling, early, no capture — the heavy and the capture jobs stay with Echensis), hull
and appendage inspection (photo, anodes, CP by a contact probe, thruster tunnels, sea-chest grates), intake grate
clearing, mooring chain and pile inspection. Family: Scampi's autonomy stack on a Jetson Orin NX; the hull as its own
ruler (weld seams, plate edges, draft marks) with four altimeters for distance and attitude to it.

Requirements (``REQUIREMENTS``): a unit under 20 kg and one-person portable in a case; 60 m depth; a group of up to
six on one node; a ship's hull in one 12 h shift; under EUR 20k a unit and EUR 15k a node.

Layout (x forward, y left, z up; z = 0 the brush's face, the hull side): a rounded shell round one Ø130 × 400 mm
tube; the brush head under the belly inside the magnetic roller pad (its magnets on a servo clutch: lifted 10 mm
to let go); four ducted thrusters — two horizontal in differential (surge, yaw) and two vertical (heave, and the
roll a belly pad needs to meet a vertical ship side); no sway: the pad's friction holds the lane. Six (the ROV
reflex: four vectored plus two vertical) was the first layout; four is what the job needs, and a unit's redundancy
matters less in a group than its price. Four altimeters at the corners; the camera dome in the nose, lights beside
it; the USBL transponder and the modem on top; two carry handles.

``part``: ``vehicle``, ``shell``, ``housing``, ``thrusters``, ``brush``, ``altimeters``, ``lights``, ``top``, ``handles``.
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
    "sites": "merchant ships at berth or anchor (flat steel sides, 8-15 m draft), piers and piles, mooring fields",
    "jobs": ["hull grooming (light fouling, no capture)", "hull + appendage inspection (photo, anodes, CP probe, tunnels, grates)",
             "intake grate clearing", "mooring chain and pile inspection"],
    "carry": "a unit under 20 kg, one person with a case (two handles); a group of up to six units on one node",
    "depth_m": 60.0,
    "current_m_s": 0.5,
    "unit_price_eur": 20000.0,
    "node_price_eur": 15000.0,
    "group_max": 6,
}
#: Bought parts per unit: (mass kg, volume L, average power W, price EUR) — class values, not quotes.
PARTS = {
    "Jetson Orin NX 16GB + carrier": (0.30, 0.0, 15.0, 900.0),
    "ducted thrusters with guards (T200 class), each": (0.35, 0.15, 0.0, 300.0),      # times the thruster count
    "camera (stereo) + 4 lights": (1.20, 0.50, 40.0, 1500.0),
    "4 altimeters (hull distance, attitude)": (0.60, 0.25, 4.0, 1000.0),
    "IMU + depth + leak": (0.30, 0.10, 3.0, 800.0),
    "USBL transponder + acoustic modem": (1.00, 0.50, 5.0, 4500.0),
    "CP contact probe + anode gauge": (0.30, 0.10, 1.0, 600.0),
    "LiFePO4 battery 500 Wh, 24 V + BMS": (3.50, 2.60, 0.0, 900.0),
    "tube Ø130 x 400 (Al, 60 m) + dome": (2.80, 5.30, 0.0, 1400.0),
    "shell, frame, foam, handles": (2.20, 6.50, 0.0, 2000.0),
    "magnetic roller pad (NdFeB blocks on 3 rollers, 80 N on steel) + servo clutch": (0.65, 0.18, 1.0, 360.0),
    "assembly, cabling, test": (0.50, 0.10, 0.0, 2000.0),
}
BRUSH = {"mass": 2.8, "volume": 1.4, "power_W": 200.0, "width_m": 0.35, "press_N": 40.0, "price": 1400.0}
PAD_PRELOAD_N = 80.0                                            # the magnetic pad's pull on steel, engaged
PAD_RELEASED_N = 5.0                                            # with the clutch lifting the blocks 10 mm
PAD_MU = 0.3                                                    # the rollers' lateral grip on a wet, slimy hull (assumed)
NODE = {   # on the workboat or a buoy: (mass kg, price EUR)
    "USBL head + acoustic modem (surface)": (6.0, 7000.0),
    "GPS + 4G/Wi-Fi + the group's computer": (3.0, 2500.0),
    "two charging bays (cradles, 48 V chargers)": (12.0, 3000.0),
    "cases, spares, cabling": (8.0, 1500.0),
}
MODEM_BPS, REPORT_BYTES, REPORT_HZ = 5000.0, 100.0, 0.5        # the acoustic link's budget (dead reckoning between reports)
USBL_HZ = 1.0                                                   # one fix a second, round-robin over the group


class Mysis(Design):
    parameters = [
        Parameter("part", "vehicle", choices=("vehicle", "shell", "housing", "thrusters", "brush", "altimeters", "lights", "top", "handles"),
                  description="what to build"),
        Parameter("length", 620.0, "mm", min=300),
        Parameter("width", 360.0, "mm", min=200),
        Parameter("height", 240.0, "mm", min=120),
        Parameter("tube_d", 130.0, "mm", min=80),
        Parameter("duct", 100.0, "mm", min=50, description="thruster duct with guard, outer"),
        Parameter("thrusters", 4, "", min=3, max=6, description="4: two horizontal differential + two vertical; 6: four vectored + two vertical"),
        Parameter("thrust_each_N", 30.0, "N", min=5, description="each thruster's bollard thrust (T200 class at 24 V)"),
        Parameter("thruster_W", 350.0, "W", min=20),
        Parameter("groom_speed", 0.4, "m/s", min=0.05, description="along the lane while brushing"),
        Parameter("survey_speed", 0.6, "m/s", min=0.1),
        Parameter("battery_wh", 500.0, "Wh", min=100),
        Parameter("group", 6, "", min=1, max=6, description="units on the node"),
        Parameter("ship_hull_m2", 9000.0, "m2", min=100, description="the wetted side-and-bottom area to groom (a 180 m bulker)"),
        Parameter("lane_overlap", 0.15, "", min=0, max=0.5),
    ]

    @staticmethod
    def p(over=None):
        return {q.name: q.default for q in Mysis.parameters} | (over or {})

    @classmethod
    def sizing(cls, over=None):
        """The unit: mass, trim, speed, pressing and holding, power and endurance, cleaning rate, price. The group:
        lanes, the shift on one hull, the acoustic link and positioning budget, the rotation through the node's bays."""
        p = cls.p(over)
        n_t = int(p["thrusters"])
        each = PARTS["ducted thrusters with guards (T200 class), each"]
        extra = n_t - 1                                                       # the parts list counts one thruster
        m = sum(a for a, _, _, _ in PARTS.values()) + extra * each[0] + BRUSH["mass"]
        V = (sum(b for _, b, _, _ in PARTS.values()) + extra * each[1] + BRUSH["volume"]) / 1000
        net_up = 0.2
        trim_kg = RHO_SW * V - m - net_up
        L, W, H = p["length"] / 1000, p["width"] / 1000, p["height"] / 1000
        cda_fwd = 0.7 * W * H
        n_h = int(p["thrusters"]) - 2
        T_fwd = n_h * p["thrust_each_N"] * (math.cos(math.radians(45)) if n_h == 4 else 1.0)
        T_sway = 4 * p["thrust_each_N"] * math.cos(math.radians(45)) if n_h == 4 else 0.0
        T_press = 2 * p["thrust_each_N"]
        v_max = math.sqrt(T_fwd / (0.5 * RHO_SW * cda_fwd))
        u = REQUIREMENTS["current_m_s"]
        hull_drag = 0.5 * RHO_SW * u * u * (0.9 * L * H)                      # broadside to a current along the ship
        # on the hull: the magnetic pad gives the brush's press for nothing (steel); the vertical pair lands and lifts
        # off (it must beat the pad: the lift-off margin); friction from the preload helps hold the lane
        hold_margin = (T_sway + PAD_MU * PAD_PRELOAD_N) / hull_drag - 1       # the cross-current: sway thrust or the pad's grip
        press_margin = PAD_PRELOAD_N / BRUSH["press_N"] - 1
        liftoff_margin = T_press / PAD_RELEASED_N - 1                          # the clutch lets go first
        # power: hotel + brush + driving the lane against drag and the pad's rolling friction
        hotel = sum(w for _, _, w, _ in PARTS.values())
        k_T = p["thruster_W"] / p["thrust_each_N"] ** 1.5                     # W per N^1.5 (propeller: P ~ T^1.5)
        press_W = 0.0
        drive_N = 0.5 * RHO_SW * cda_fwd * p["groom_speed"] ** 2 + 0.08 * PAD_PRELOAD_N + 0.2 * BRUSH["press_N"]   # hard rollers roll at ~0.08
        per = (lambda N: N / 4 / math.cos(math.radians(45))) if n_h == 4 else (lambda N: N / 2)
        drive_W = n_h * k_T * per(drive_N) ** 1.5
        groom_W = hotel + BRUSH["power_W"] + press_W + drive_W
        survey_N = 0.5 * RHO_SW * cda_fwd * p["survey_speed"] ** 2
        survey_W = hotel + n_h * k_T * per(survey_N) ** 1.5
        usable = 0.9 * p["battery_wh"]
        groom_h, survey_h = usable / groom_W, usable / survey_W
        rate = BRUSH["width_m"] * (1 - p["lane_overlap"]) * p["groom_speed"] * 3600 * 0.75   # m2/h, 75 % lane efficiency
        unit_price = sum(c for _, _, _, c in PARTS.values()) + extra * each[3] + BRUSH["price"]
        # the group
        n = int(p["group"])
        bays = 2
        working = max(n - bays, 1) if n > bays else n                           # the bays keep two charging while the rest work
        charge_h = p["battery_wh"] / 400.0                                       # 400 W chargers
        duty = min(1.0, groom_h / (groom_h + charge_h)) if n > bays else 1.0     # each unit's share of time on the hull
        group_rate = min(n * duty, working) * rate if n > bays else n * rate * groom_h / (groom_h + charge_h)
        shift_h = p["ship_hull_m2"] / group_rate
        lanes = math.ceil(p["ship_hull_m2"] / (BRUSH["width_m"] * (1 - p["lane_overlap"])) / 180.0)   # 180 m lanes
        link_bps = n * REPORT_BYTES * 8 * REPORT_HZ
        link_margin = MODEM_BPS / link_bps - 1
        fix_period_s = n / USBL_HZ
        drift_m = 0.02 * p["groom_speed"] * fix_period_s                        # 2 % odometry between fixes
        node_price = sum(c for _, c in NODE.values())
        return {"mass_in_air_kg": m, "displacement_L": V * 1000, "trim_kg": trim_kg, "case_kg": m + 7.0,
                "v_max_m_s": v_max, "thrust_fwd_N": T_fwd, "press_N": PAD_PRELOAD_N, "press_margin": press_margin, "hold_margin": hold_margin,
                "liftoff_margin": liftoff_margin, "press_W": press_W,
                "hotel_W": hotel, "groom_W": groom_W, "survey_W": survey_W, "groom_h": groom_h, "survey_h": survey_h,
                "rate_m2_h": rate, "unit_price_eur": unit_price,
                "group": n, "group_rate_m2_h": group_rate, "shift_h": shift_h, "lanes": lanes, "duty": duty,
                "link_bps": link_bps, "link_margin": link_margin, "fix_period_s": fix_period_s, "drift_between_fixes_m": drift_m,
                "node_price_eur": node_price, "fleet_price_eur": n * unit_price + node_price}

    # ------------------------------------------------------------------------------------------- CAD
    def _shell(self, p):
        L, W, H = p["length"], p["width"], p["height"]
        body = cq.Workplane("XY").box(L, W, H).edges().fillet(min(W, H) * 0.35).translate((0, 0, H / 2 + 40))
        return body.cut(cq.Workplane("YZ").center(0, H * 0.55 + 40).circle(p["tube_d"] / 2 + 1).extrude(L, both=True))

    def _housing(self, p):
        L, H, d = p["length"], p["height"], p["tube_d"]
        z = H * 0.55 + 40
        tube = cq.Workplane("YZ").center(0, z).circle(d / 2).extrude(L * 0.33, both=True)
        dome = cq.Workplane("XY").sphere(d / 2).translate((L * 0.33, 0, z))
        return tube.union(dome)

    def thruster_layout(self, p):
        L, W, H = p["length"], p["width"], p["height"]
        c = math.cos(math.radians(45))
        out = []
        if int(p["thrusters"]) >= 6:
            for sx in (1, -1):
                for sy in (1, -1):
                    out.append(((sx * L * 0.30, sy * (W / 2 + p["duct"] * 0.3), H * 0.5 + 40), (sx * c, sy * c, 0)))
        else:
            for sy in (1, -1):                                                 # two along the sides, pushing forward
                out.append(((-L * 0.22, sy * (W / 2 + p["duct"] * 0.3), H * 0.5 + 40), (1, 0, 0)))
        for sx in (1, -1):
            out.append(((sx * L * 0.14, 0, H + 40 + p["duct"] * 0.05), (0, 0, 1)))
        return out

    def _thruster(self, p, at, axis):
        d = p["duct"]
        ax = cq.Vector(*axis)
        duct = cq.Workplane("XY").add(cq.Solid.makeCylinder(d / 2, d * 0.6, cq.Vector(*at) - ax * d * 0.3, ax))
        bore = cq.Workplane("XY").add(cq.Solid.makeCylinder(d / 2 - 6, d * 0.6, cq.Vector(*at) - ax * d * 0.3, ax))
        hub = cq.Workplane("XY").add(cq.Solid.makeCylinder(d * 0.18, d * 0.5, cq.Vector(*at) - ax * d * 0.25, ax))
        return duct.cut(bore).union(hub)

    def _thrusters(self, p):
        out = None
        for at, a in self.thruster_layout(p):
            t = self._thruster(p, at, a)
            out = t if out is None else out.union(t)
        return out

    def _brush(self, p):
        r = BRUSH["width_m"] * 1000 / 2
        head = cq.Workplane("XY").circle(r + 10).extrude(40).translate((p["length"] * 0.05, 0, 0))     # the housing
        disc = cq.Workplane("XY").circle(r).extrude(12).translate((p["length"] * 0.05, 0, -12))          # the brush face
        return head.union(disc)

    def _altimeters(self, p):
        L, W = p["length"], p["width"]
        out = None
        for sx in (1, -1):
            for sy in (1, -1):
                a = cq.Workplane("XY").center(sx * L * 0.42, sy * W * 0.38).circle(16).extrude(45).translate((0, 0, 15))
                out = a if out is None else out.union(a)
        return out

    def _lights(self, p):
        L, H = p["length"], p["height"]
        out = None
        for sy in (1, -1):
            l = cq.Workplane("YZ").workplane(offset=L * 0.30).center(sy * (p["tube_d"] / 2 + 30), H * 0.55 + 40).circle(16).extrude(30)
            out = l if out is None else out.union(l)
        return out

    def _top(self, p):
        L, H = p["length"], p["height"]
        usbl = cq.Workplane("XY").circle(22).extrude(50).translate((-L * 0.30, 0, H + 40))
        modem = cq.Workplane("XY").circle(18).extrude(40).translate((-L * 0.38, 60, H + 40))
        return usbl.union(modem)

    def _handles(self, p):
        L, W, H = p["length"], p["width"], p["height"]
        out = None
        for sy in (1, -1):
            h = (cq.Workplane("XZ").polyline([(-70, 0), (-70, 30), (70, 30), (70, 0), (52, 0), (52, 18), (-52, 18), (-52, 0)])
                 .close().extrude(8, both=True).translate((0, sy * W * 0.32, H + 40)))
            out = h if out is None else out.union(h)
        return out

    def build(self, p):
        parts = {"shell": self._shell, "housing": self._housing, "thrusters": self._thrusters, "brush": self._brush,
                 "altimeters": self._altimeters, "lights": self._lights, "top": self._top, "handles": self._handles}
        if p["part"] in parts:
            return parts[p["part"]](p)
        out = None
        for name in parts:
            s = parts[name](p)
            out = s if out is None else out.union(s)
        return out


if __name__ == "__main__":
    s = Mysis.sizing()
    print(f"unit ({Mysis.p()['thrusters']} thrusters): {s['mass_in_air_kg']:.1f} kg (case {s['case_kg']:.0f} kg), trim {s['trim_kg']:+.2f} kg, {s['v_max_m_s']:.2f} m/s, pad {s['press_N']:.0f} N "
          f"(press margin {s['press_margin']:+.2f}, lift-off {s['liftoff_margin']:+.2f}), hold {s['hold_margin']:+.2f}; groom {s['groom_W']:.0f} W -> {s['groom_h']:.1f} h, survey {s['survey_h']:.1f} h; "
          f"{s['rate_m2_h']:.0f} m2/h; EUR {s['unit_price_eur']:.0f}")
    for n in (1, 2, 4, 6):
        g = Mysis.sizing({"group": n})
        print(f"group of {n}: {g['group_rate_m2_h']:.0f} m2/h, a {Mysis.p()['ship_hull_m2']:.0f} m2 hull in {g['shift_h']:.1f} h ({g['lanes']} lanes), duty {g['duty']:.2f}, "
              f"link {g['link_bps']:.0f} bps (margin {g['link_margin']:+.1f}), a fix every {g['fix_period_s']:.0f} s (drift {g['drift_between_fixes_m']:.2f} m), fleet EUR {g['fleet_price_eur']:.0f}")
