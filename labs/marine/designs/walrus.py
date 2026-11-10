"""Walrus (it finds clams with its whiskers and digs them out with its tusks and flippers): a subsurface drone
that picks clams. Not a dredge: a tracked seabed crawler that sees the clams' shows (siphon holes, shells) with its
camera, loosens each with a short water-jet pulse, lifts it with a soft gripper on a belly gantry, sizes it, and
keeps the legal ones in a basket — undersize and the wrong species go back into the sand behind it. Two simple
arms on the front (one joint each, a flat paddle) dig: they scrape the top few centimetres aside where the clams are
deep or the sand is packed, so the jet and the gripper reach them; they also clear weed and stones from the lane. Tethered to the
harvest boat (power, fibre, video), it **swims itself to the bed**: a GPS fix at the surface, a dive, then dead
reckoning on the DVL with the boat's USBL fixes to the waypoint, 150 m of tether paying out behind it; on the bed it
crawls in lanes with two pickers side by side; a full basket is
released on its own float line and hauled by the crew while the crawler keeps working on the spare; the two
vertical thrusters only bring the crawler up at the end of the shift. The family: Scampi's autonomy stack (``scampi.md``), Mysis's lane
map (``mysis.md``) on the seabed instead of a hull.

Jobs: selective harvest of farmed and wild clam beds (Manila, carpet-shell, razor by jet), bed surveys (density and
size by lane: the stock estimate), reseeding (the same gantry drops spat from a hopper).

Requirements (``REQUIREMENTS``): under 45 kg and lifted by a davit (two people at a pinch); 30 m depth; beds of sand, mud and
gravel; harvest at a rate that beats a diver and leaves the bed intact; under EUR 30k.

Layout (x forward, y left, z up; z = 0 the track's bottom): two tracks with wide grousers; a frame with the
electronics tube on top; the camera and lights looking down ahead of the gantry; the belly gantry (y travel across
the swath, z plunge) carrying the jet nozzle and the soft gripper; the basket at the back over the sizing chute;
two vertical thrusters; the tether at the rear.

``part``: ``vehicle``, ``tracks``, ``frame``, ``housing``, ``gantry``, ``basket``, ``thrusters``, ``lights``.
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
    "beds": "farmed and wild clam beds, sand / mud / gravel, 2-30 m; Manila and carpet-shell clams, razor clams by jet",
    "jobs": ["selective harvest (legal size, right species)", "bed survey: density and size by lane", "reseeding from a spat hopper"],
    "carry": "under 45 kg: a davit on the harvest boat, two people at a pinch",
    "transit": "swims itself to the bed: GPS fix, dive, DVL dead reckoning + USBL fixes, within the tether's 150 m",
    "depth_m": 30.0,
    "price_eur": 30000.0,
    "bed_impact": "no dredging: one jet pulse per clam, the sand settles behind the tracks; undersize clams go back",
}
#: Bought parts: (mass kg, volume L, average power W, price EUR) — class values, not quotes.
PARTS = {
    "Jetson Orin NX 16GB + carrier": (0.30, 0.0, 15.0, 900.0),
    "2 track modules (wide grousers, 48 V motors)": (7.0, 3.0, 120.0, 2600.0),
    "camera (stereo, down-looking) + 4 lights": (1.20, 0.50, 40.0, 1500.0),
    "belly gantry (y travel, z plunge) with 2 soft grippers": (4.5, 2.0, 80.0, 3600.0),
    "jet pump + nozzle (pulsed)": (1.8, 0.8, 60.0, 900.0),
    "sizing chute + return flap": (0.8, 0.6, 5.0, 400.0),
    "2 digging arms (one joint, a paddle, 48 V servo each)": (2.4, 1.0, 40.0, 900.0),
    "2 baskets 30 L on float lines (release + spare)": (3.0, 0.0, 0.0, 500.0),
    "2 vertical + 2 horizontal thrusters (T200 class)": (1.4, 0.6, 0.0, 1200.0),
    "USBL transponder (the boat's head fixes it)": (0.5, 0.3, 3.0, 2500.0),
    "IMU + depth + altimeter + DVL-lite (odometry)": (0.9, 0.3, 8.0, 3200.0),
    "tether 150 m (48 V power + fibre, neutral) + reel + topside": (12.0, 0.0, 0.0, 4500.0),
    "electronics tube (Al, 30 m)": (2.5, 4.0, 0.0, 900.0),
    "frame, skids, foam, cabling": (6.5, 17.5, 0.0, 2700.0),
    "assembly, test": (0.5, 0.1, 0.0, 2500.0),
}
CLAM = {"mass_kg": 0.03, "legal_fraction": 0.6, "price_eur_per_kg": 6.0}       # a Manila clam; 60 % of a bed above size


class Walrus(Design):
    parameters = [
        Parameter("part", "vehicle", choices=("vehicle", "tracks", "frame", "housing", "gantry", "basket", "thrusters", "lights", "arms"),
                  description="what to build"),
        Parameter("length", 900.0, "mm", min=400),
        Parameter("width", 700.0, "mm", min=300),
        Parameter("height", 380.0, "mm", min=150),
        Parameter("track_w", 120.0, "mm", min=50),
        Parameter("swath", 0.5, "m", min=0.2, description="the gantry's reach across the track, the lane's width"),
        Parameter("crawl_speed", 0.15, "m/s", min=0.02),
        Parameter("pick_s", 6.0, "s", min=1, description="see, jet, grip, lift, size, drop: one clam, one picker"),
        Parameter("dig_depth", 60.0, "mm", min=0, description="what the paddles scrape aside ahead of the gantry"),
        Parameter("dig_fraction", 0.3, "", min=0, max=1, description="the share of the lane the arms must dig first"),
        Parameter("pickers", 2, "", min=1, max=4, description="grippers on the gantry working side by side"),
        Parameter("shifts", 2, "", min=1, max=3, description="8 h shifts a day: the crawler does not care about daylight"),
        Parameter("bed_density", 80.0, "per m2", min=1, description="clams of all sizes per m2 of bed"),
        Parameter("basket_kg", 20.0, "kg", min=2),
        Parameter("boat_power_W", 1000.0, "W", min=200, description="what the boat's tether supply gives"),
        Parameter("transit_m", 120.0, "m", min=10, description="the swim from the boat to the bed (under the tether's reach)"),
    ]

    @staticmethod
    def p(over=None):
        return {q.name: q.default for q in Walrus.parameters} | (over or {})

    @classmethod
    def sizing(cls, over=None):
        """Mass and trim (it must sit on the bed: negative), the harvest rate (pick-limited or crawl-limited), basket
        trips, a day's take and its value against a diver, the bed's impact, the power from the boat, the price."""
        p = cls.p(over)
        m = sum(a for a, _, _, _ in PARTS.values())
        m_wet = m - PARTS["tether 150 m (48 V power + fibre, neutral) + reel + topside"][0]        # the tether stays on the boat
        V = sum(b for _, b, _, _ in PARTS.values()) / 1000
        sink_kg = m_wet - RHO_SW * V                                                                 # >0: it sits on the bed
        lift_N = 2 * 30.0                                                                            # the vertical pair
        lift_margin = lift_N / max(sink_kg * G, 1e-6) - 1                                            # up with an empty basket: the full ones go up on their lines
        # the swim: the horizontal pair against the hull's drag, the tether's drag behind it, a 0.5 m/s current to hold
        L, W, H = p["length"] / 1000, p["width"] / 1000, p["height"] / 1000
        cda = 0.9 * W * H                                                                            # a boxy crawler
        T_fwd = 2 * 30.0
        v_swim = math.sqrt(T_fwd / (0.5 * RHO_SW * (cda + 1.2 * 0.006 * p["transit_m"] * 0.5)))      # half the tether broadside
        transit_min = (p["transit_m"] / max(v_swim, 0.05) + 2 * REQUIREMENTS["depth_m"] / 0.3) / 60
        u = 0.5
        hold_current = T_fwd / (0.5 * RHO_SW * u * u * cda) - 1
        nav_error_m = 0.01 * p["transit_m"] + 0.02 * p["transit_m"] * 0.3                           # USBL 1 % of range, DVL 2 % over the stretches between fixes
        # harvest: clams seen per second along the lane against the picking cycle
        legal_per_m2 = p["bed_density"] * CLAM["legal_fraction"]
        seen_per_s = legal_per_m2 * p["swath"] * p["crawl_speed"]
        pick_per_s = p["pickers"] / p["pick_s"]
        rate_per_s = min(seen_per_s, pick_per_s)
        limited_by = "picking" if pick_per_s < seen_per_s else "crawling"
        speed_eff = min(p["crawl_speed"], pick_per_s / (legal_per_m2 * p["swath"]))               # the crawl slows to the picker
        # digging: the paddles move the top layer of the dug share of the lane; a 2 kg load per stroke, a 1.5 s stroke
        dig_m3_h = p["dig_fraction"] * p["swath"] * speed_eff * p["dig_depth"] / 1000 * 3600
        dig_strokes_h = dig_m3_h * 1600 / 2.0
        dig_busy = dig_strokes_h * 1.5 / 3600                                                     # the arms' duty
        rate_kg_h = rate_per_s * 3600 * CLAM["mass_kg"]
        basket_h = p["basket_kg"] / rate_kg_h
        trip_min = 2.0                                                                               # release the full basket, dock the spare: the crawler barely stops
        day_h = 8.0 * p["shifts"]
        day_kg = day_h / (basket_h + trip_min / 60) * p["basket_kg"]
        diver_kg_day = 150.0                                                                        # a good hand-harvest diver's day
        hotel = sum(w for _, _, w, _ in PARTS.values())
        power_margin = p["boat_power_W"] / hotel - 1
        price = sum(c for _, _, _, c in PARTS.values())
        payback_days = price / (day_kg * CLAM["price_eur_per_kg"]) if day_kg else float("inf")
        return {"mass_in_air_kg": m, "mass_wet_vehicle_kg": m_wet, "sink_kg": sink_kg, "lift_margin": lift_margin,
                "seen_per_s": seen_per_s, "rate_per_h": rate_per_s * 3600, "limited_by": limited_by, "speed_eff_m_s": speed_eff,
                "rate_kg_h": rate_kg_h, "basket_h": basket_h, "trip_min": trip_min, "day_kg": day_kg,
                "vs_diver": day_kg / diver_kg_day, "bed_m2_per_day": speed_eff * p["swath"] * 3600 * day_h * (basket_h / (basket_h + trip_min / 60)),
                "dig_m3_h": dig_m3_h, "dig_duty": dig_busy,
                "v_swim_m_s": v_swim, "transit_min": transit_min, "hold_current_margin": hold_current, "nav_error_m": nav_error_m,
                "power_W": hotel, "power_margin": power_margin, "price_eur": price, "payback_days": payback_days}

    # ------------------------------------------------------------------------------------------- CAD
    def _tracks(self, p):
        L, W, tw = p["length"], p["width"], p["track_w"]
        out = None
        for sy in (1, -1):
            t = cq.Workplane("XY").box(L * 0.95, tw, 140).edges("|Y").fillet(60).translate((0, sy * (W / 2 - tw / 2), 70))
            out = t if out is None else out.union(t)
        return out

    def _frame(self, p):
        L, W, tw = p["length"], p["width"], p["track_w"]
        deck = cq.Workplane("XY").box(L * 0.8, W - 2 * tw - 20, 20).translate((0, 0, 160))
        for sx in (1, -1):
            deck = deck.union(cq.Workplane("XY").box(L * 0.8, 30, 150).translate((0, 0, 230)).translate((0, sx * (W / 2 - tw - 25), 0)))
        return deck

    def _housing(self, p):
        L = p["length"]
        return cq.Workplane("YZ").center(0, 300).circle(80).extrude(L * 0.3, both=True)

    def _gantry(self, p):
        L, W, tw = p["length"], p["width"], p["track_w"]
        rail = cq.Workplane("XY").box(60, W - 2 * tw - 40, 30).translate((L * 0.1, 0, 150))
        carriage = cq.Workplane("XY").box(70, 70, 40).translate((L * 0.1, 80, 115))
        plunge = cq.Workplane("XY").circle(12).extrude(110).translate((L * 0.1, 80, 5))
        gripper = cq.Workplane("XY").box(40, 50, 25).translate((L * 0.1, 80, 12))
        nozzle = cq.Workplane("XY").circle(6).extrude(90).translate((L * 0.1 + 40, 80, 20))
        return rail.union(carriage).union(plunge).union(gripper).union(nozzle)

    def _basket(self, p):
        L, W, tw = p["length"], p["width"], p["track_w"]
        b = cq.Workplane("XY").box(220, W - 2 * tw - 60, 200).translate((-L * 0.28, 0, 270))
        return b.cut(cq.Workplane("XY").box(200, W - 2 * tw - 80, 190).translate((-L * 0.28, 0, 280)))

    def _thrusters(self, p):
        L, W, tw = p["length"], p["width"], p["track_w"]
        out = None
        for sx in (1, -1):                                                                       # the vertical pair
            d = 100
            duct = cq.Workplane("XY").circle(d / 2).circle(d / 2 - 6).extrude(60).translate((sx * L * 0.35, 0, 360))
            hub = cq.Workplane("XY").circle(18).extrude(50).translate((sx * L * 0.35, 0, 365))
            t = duct.union(hub)
            out = t if out is None else out.union(t)
        for sy in (1, -1):                                                                       # the horizontal pair, pushing forward, outboard of the tracks' rear
            d = 100
            duct = cq.Workplane("YZ").workplane(offset=-L * 0.45).center(sy * (W / 2 - tw / 2), 230).circle(d / 2).circle(d / 2 - 6).extrude(60)
            hub = cq.Workplane("YZ").workplane(offset=-L * 0.45).center(sy * (W / 2 - tw / 2), 230).circle(18).extrude(50)
            out = out.union(duct.union(hub))
        return out

    def _lights(self, p):
        L, W, tw = p["length"], p["width"], p["track_w"]
        out = None
        for sy in (1, -1):
            l = cq.Workplane("XY").circle(16).extrude(30).translate((L * 0.35, sy * 120, 170))
            out = l if out is None else out.union(l)
        cam = cq.Workplane("XY").box(90, 60, 50).translate((L * 0.35, 0, 195))
        return out.union(cam)

    def _arms(self, p):
        """Two one-joint arms on the front corners: a shoulder block, an upper link angled down, a flat paddle."""
        L, W, tw = p["length"], p["width"], p["track_w"]
        out = None
        for sy in (1, -1):
            x0, y0 = L * 0.46, sy * (W / 2 - tw - 40)
            shoulder = cq.Workplane("XY").box(60, 60, 60).translate((x0, y0, 150))
            link = (cq.Workplane("XY").box(200, 36, 24).translate((100, 0, 0))
                    .rotate((0, 0, 0), (0, 1, 0), 35).translate((x0 + 20, y0, 150)))
            paddle = (cq.Workplane("XY").box(90, 110, 8).translate((45, 0, 0))
                      .rotate((0, 0, 0), (0, 1, 0), 35).translate((x0 + 20 + 200 * math.cos(math.radians(35)), y0, 150 - 200 * math.sin(math.radians(35)))))
            a = shoulder.union(link).union(paddle)
            out = a if out is None else out.union(a)
        return out

    def build(self, p):
        parts = {"tracks": self._tracks, "frame": self._frame, "housing": self._housing, "gantry": self._gantry,
                 "basket": self._basket, "thrusters": self._thrusters, "lights": self._lights, "arms": self._arms}
        if p["part"] in parts:
            return parts[p["part"]](p)
        out = None
        for name in parts:
            s = parts[name](p)
            out = s if out is None else out.union(s)
        return out


if __name__ == "__main__":
    s = Walrus.sizing()
    print(f"{s['mass_in_air_kg']:.1f} kg in air (vehicle {s['mass_wet_vehicle_kg']:.1f}), sits {s['sink_kg']:+.1f} kg on the bed, lift margin {s['lift_margin']:+.2f}; "
          f"sees {s['seen_per_s']*60:.0f} legal clams/min, picks {s['rate_per_h']:.0f}/h ({s['limited_by']}-limited, crawls {s['speed_eff_m_s']:.2f} m/s): "
          f"{s['rate_kg_h']:.1f} kg/h, a basket in {s['basket_h']*60:.0f} min, {s['day_kg']:.0f} kg a day of {Walrus.p()['shifts']} shifts ({s['vs_diver']:.1f}x a diver), "
          f"swims {s['v_swim_m_s']:.2f} m/s, {Walrus.p()['transit_m']:.0f} m to the bed in {s['transit_min']:.0f} min (nav error ~{s['nav_error_m']:.1f} m, holds 0.5 m/s with {s['hold_current_margin']:+.1f}); "
          f"{s['bed_m2_per_day']:.0f} m2 of bed, the arms dig {s['dig_m3_h']:.2f} m3/h at {s['dig_duty']:.0%} duty; {s['power_W']:.0f} W from the boat (margin {s['power_margin']:+.1f}); EUR {s['price_eur']:.0f}, payback {s['payback_days']:.0f} days")
    for d in (20, 80, 200):
        s = Walrus.sizing({"bed_density": d})
        print(f"  bed {d}/m2: {s['rate_per_h']:.0f}/h ({s['limited_by']}), {s['day_kg']:.0f} kg/day")
