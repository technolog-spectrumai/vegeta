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
    "8 ducted thrusters: 4 vectored 140 mm + 4 vertical 120 mm props, ~1300-1500 rpm": (3.5, 1.6, 0.0, 2760.0),
    "USBL transponder (the boat's head fixes it)": (0.5, 0.3, 3.0, 2500.0),
    "IMU + depth + altimeter + DVL-lite (odometry)": (0.9, 0.3, 8.0, 3200.0),
    "tether 150 m (48 V power + fibre, neutral) + reel + topside": (12.0, 0.0, 0.0, 4500.0),
    "electronics tube (Al, 30 m)": (2.5, 4.0, 0.0, 900.0),
    "frame, skids, foam, cabling": (6.5, 17.5, 0.0, 2700.0),
    "assembly, test": (0.5, 0.1, 0.0, 2500.0),
}
#: The thrusters (round 2: the owner's brief - move in every direction, big slow props). v0 had four T200-class
#: units (76 mm props at ~5000 rpm, 30 N each): two vertical, two horizontal pushing forward - no sway, and small fast
#: props waste power. Three candidate layouts are compared in ``compare_layouts`` (walrus.md has the table), all
#: ducted, sized at the same disc loading (``DISC_N_M2``: 30 N on a 160 mm disc) so a bigger prop gives more thrust
#: for the same efficiency, and checked for fit inside the 900 x 700 x 380 envelope:
#:   A - 4 vectored + canted: one 160 mm duct per deck corner, 45 deg in plan, canted 35 deg up (surge, sway, heave,
#:       yaw; roll and pitch only passive);
#:   B - 4 vectored horizontal (140 mm, corners) + 2 vertical (160 mm, on the centreline fore and aft): + pitch;
#:   C - 4 vectored horizontal (140 mm, corners) + 4 vertical (120 mm over the tracks): all six degrees of freedom.
#: Actuator-disc (momentum) theory: the ideal bollard power for thrust T through a disc of area A is
#: T^1.5 / sqrt(2 rho A); the real one divides by the figure of merit - 0.55 for a small fast open prop, 0.65 with a
#: duct round a larger slow one (the duct factor) - and by the motor and driver's 0.8. rpm from T = Ct rho n^2 D^4.
THRUSTER_V0 = {"n": 4, "diameter_m": 0.076, "max_N": 30.0, "fm": 0.55, "layout": "v0: 2 vertical + 2 horizontal (forward), T200 class"}
DISC_N_M2 = 30.0 / (math.pi * 0.16 ** 2 / 4)          # ~1490 N/m2
FM_DUCTED, MOTOR_EFF, CT = 0.65, 0.8, 0.12
LAYOUT = "C"                                           # the one chosen (``compare_layouts``): full 6-DOF control


def unit(D, fm=FM_DUCTED):
    """A ducted thruster of prop diameter D at the design disc loading: max thrust, its rpm and bollard power."""
    A = math.pi * D * D / 4
    T = DISC_N_M2 * A
    return {"diameter_m": D, "max_N": T, "fm": fm, "duct_od_m": D + 0.02, "mass_kg": 0.2 + 14.0 * D * D, "price_eur": 150 + 1500 * D}


def prop(t, T=None):
    """Power (W, electrical) for thrust T (default the max), rpm, N/W."""
    D, T = t["diameter_m"], (t["max_N"] if T is None else abs(T))
    A = math.pi * D * D / 4
    ideal = T ** 1.5 / math.sqrt(2 * RHO_SW * A)
    P = ideal / (t["fm"] * MOTOR_EFF)
    rpm = 60 * math.sqrt(T / (CT * RHO_SW * D ** 4))
    return {"rpm": rpm, "ideal_W": ideal, "power_W": P, "N_per_W": T / P if P else 0.0, "diameter_m": D, "max_N": t["max_N"]}


def thruster_layout(p, name=None):
    """The thrusters of a layout: [{"at": (x, y, z) m body frame, z up from the track bottom, "axis", unit...}]."""
    name = name or LAYOUT
    L, W, H = p["length"] / 1000, p["width"] / 1000, p["height"] / 1000
    h = math.sqrt(0.5)
    out = []
    def corners(D, cant):
        u = unit(D)
        r = u["duct_od_m"] / 2
        c = math.radians(cant)
        for sx, sy, hx, hy in ((1, 1, 1, -1), (1, -1, 1, 1), (-1, 1, -1, -1), (-1, -1, -1, 1)):
            out.append(u | {"at": (sx * (L / 2 - r - 0.01), sy * (W / 2 - r - 0.01), H - r - 0.02),
                            "axis": (math.cos(c) * hx * h, math.cos(c) * hy * h, math.sin(c))})
    if name == "A":
        corners(0.16, 35.0)
    elif name == "B":
        corners(0.14, 0.0)
        u = unit(0.16)
        for sx in (1, -1):
            out.append(u | {"at": (sx * 0.20, 0.0, H - 0.03), "axis": (0.0, 0.0, 1.0)})
    else:
        corners(0.14, 0.0)
        u = unit(0.12)
        for sx in (1, -1):
            for sy in (1, -1):
                out.append(u | {"at": (sx * 0.17, sy * (W / 2 - u["duct_od_m"] / 2 - 0.005), H - 0.03), "axis": (0.0, 0.0, 1.0)})
    return out


def allocation(p, name=None):
    """The 6 x n matrix from thrusts to (surge, sway, heave, roll, pitch, yaw) about the middle of the envelope."""
    zc = p["height"] / 2000
    cols = []
    for t in thruster_layout(p, name):
        x, y, z = t["at"]
        ax, ay, az = t["axis"]
        rz = z - zc
        cols.append([ax, ay, az, y * az - rz * ay, rz * ax - x * az, x * ay - y * ax])
    return [list(r) for r in zip(*cols)]


def compare_layouts(p=None):
    """Each candidate: count, diameters, mass, fit, the rank of its allocation (of 6), the weakest pure force over
    every direction (N; no moments where it can hold them), the power to lift 30 N straight up and to push 60 N
    ahead (least-squares allocation), and the thrust per watt at full."""
    import numpy as np
    p = p or Walrus.p()
    L, W, H = p["length"] / 1000, p["width"] / 1000, p["height"] / 1000
    rng = np.random.default_rng(0)
    dirs = rng.standard_normal((400, 3))
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    out = {}
    for name in ("A", "B", "C"):
        th = thruster_layout(p, name)
        B = np.array(allocation(p, name))
        rank = int(np.linalg.matrix_rank(B, tol=1e-6))
        Tm = np.array([t["max_N"] for t in th])
        held = []                                                  # the moments it can hold besides the forces
        for k in (5, 3, 4):
            if np.linalg.matrix_rank(B[[0, 1, 2] + held + [k]], tol=1e-6) > np.linalg.matrix_rank(B[[0, 1, 2] + held], tol=1e-6):
                held.append(k)
        # the moments it can hold stay zero; the force direction is d
        rows = [0, 1, 2] + held
        Bp = np.linalg.pinv(B[rows])
        def force(d):
            w = np.concatenate([d, np.zeros(len(held))])
            t = Bp @ w
            res = np.linalg.norm(B[[0, 1, 2]] @ t - d)
            return 0.0 if res > 1e-3 else float(np.min(Tm / np.maximum(np.abs(t), 1e-12)))
        fmin = min(force(d) for d in dirs)
        def power(d, F):
            w = np.concatenate([np.array(d, float) * F, np.zeros(len(held))])
            t = Bp @ w
            return sum(prop(u, ti)["power_W"] for u, ti in zip(th, t))
        def ext(t, k):                                             # the duct's half extent along body axis k
            a, r, hl = abs(t["axis"][k]), t["duct_od_m"] / 2, 0.035
            return a * hl + math.sqrt(max(0.0, 1 - a * a)) * r
        fits = all(abs(t["at"][0]) + ext(t, 0) <= L / 2 + 1e-9 and abs(t["at"][1]) + ext(t, 1) <= W / 2 + 1e-9
                   and t["at"][2] + ext(t, 2) <= H + 0.02 for t in th)
        out[name] = {"n": len(th), "diameters_mm": sorted({round(t["diameter_m"] * 1000) for t in th}),
                     "mass_kg": sum(t["mass_kg"] for t in th), "price_eur": sum(t["price_eur"] for t in th), "fits": fits,
                     "rank": rank, "dof": ["surge", "sway", "heave"] + [("roll", "pitch", "yaw")[k - 3] for k in held],
                     "min_force_N": fmin, "lift30_W": power((0, 0, 1), 30.0), "surge60_W": power((1, 0, 0), 60.0),
                     "N_per_W": prop(th[0])["N_per_W"], "rpm": [round(prop(t)["rpm"]) for t in th][::4 if len(th) > 4 else 1]}
    return out


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
        lay = thruster_layout(p)
        lift_N = sum(t["max_N"] * abs(t["axis"][2]) for t in lay)
        lift_margin = lift_N / max(sink_kg * G, 1e-6) - 1                                            # up with an empty basket: the full ones go up on their lines
        # the swim: the horizontal pair against the hull's drag, the tether's drag behind it, a 0.5 m/s current to hold
        L, W, H = p["length"] / 1000, p["width"] / 1000, p["height"] / 1000
        cda = 0.9 * W * H                                                                            # a boxy crawler
        T_fwd = sum(t["max_N"] * abs(t["axis"][0]) for t in lay)
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
        pr = prop(lay[0])
        payback_days = price / (day_kg * CLAM["price_eur_per_kg"]) if day_kg else float("inf")
        return {"mass_in_air_kg": m, "mass_wet_vehicle_kg": m_wet, "sink_kg": sink_kg, "lift_margin": lift_margin,
                "seen_per_s": seen_per_s, "rate_per_h": rate_per_s * 3600, "limited_by": limited_by, "speed_eff_m_s": speed_eff,
                "rate_kg_h": rate_kg_h, "basket_h": basket_h, "trip_min": trip_min, "day_kg": day_kg,
                "vs_diver": day_kg / diver_kg_day, "bed_m2_per_day": speed_eff * p["swath"] * 3600 * day_h * (basket_h / (basket_h + trip_min / 60)),
                "dig_m3_h": dig_m3_h, "dig_duty": dig_busy,
                "v_swim_m_s": v_swim, "transit_min": transit_min, "hold_current_margin": hold_current, "nav_error_m": nav_error_m,
                "power_W": hotel, "power_margin": power_margin,
                "thrusters": len(lay), "thrust_N": lay[0]["max_N"], "prop_rpm": pr["rpm"], "thruster_W": pr["power_W"], "thrust_per_W": pr["N_per_W"],
                "swim_W": sum(prop(t)["power_W"] for t in lay if abs(t["axis"][2]) < 0.5),
                "lift_N": lift_N, "surge_N": T_fwd, "price_eur": price, "payback_days": payback_days}

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
        """Four ducted props, one per deck corner, each along its vectored, canted axis."""
        out = None
        for t in thruster_layout(p):
            (x, y, z), (ax, ay, az) = t["at"], t["axis"]
            D, od = t["diameter_m"] * 1000, t["duct_od_m"] * 1000
            duct = cq.Workplane("XY").circle(od / 2).circle(D / 2 + 2).extrude(70).translate((0, 0, -35))
            hub = cq.Workplane("XY").circle(22).extrude(60).translate((0, 0, -30))
            blades = cq.Workplane("XY").box(D - 4, 18, 4).union(cq.Workplane("XY").box(18, D - 4, 4))
            th = duct.union(hub).union(blades)
            # z axis -> (ax, ay, az): tilt about the horizontal perpendicular
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, az))))
            if tilt > 1e-6:
                th = th.rotate((0, 0, 0), (-ay, ax, 0), tilt)
            th = th.translate((x * 1000, y * 1000, z * 1000))
            out = th if out is None else out.union(th)
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
          f"{s['thrusters']} thrusters ({LAYOUT}) {Walrus.p() and thruster_layout(Walrus.p())[0]['diameter_m']*1000:.0f} mm at {s['prop_rpm']:.0f} rpm, {s['thrust_N']:.0f} N for {s['thruster_W']:.0f} W ({s['thrust_per_W']:.2f} N/W; v0 {prop(THRUSTER_V0)['N_per_W']:.2f}), lift {s['lift_N']:.0f} N, surge {s['surge_N']:.0f} N; swims {s['v_swim_m_s']:.2f} m/s, {Walrus.p()['transit_m']:.0f} m to the bed in {s['transit_min']:.0f} min (nav error ~{s['nav_error_m']:.1f} m, holds 0.5 m/s with {s['hold_current_margin']:+.1f}); "
          f"{s['bed_m2_per_day']:.0f} m2 of bed, the arms dig {s['dig_m3_h']:.2f} m3/h at {s['dig_duty']:.0%} duty; {s['power_W']:.0f} W from the boat (margin {s['power_margin']:+.1f}); EUR {s['price_eur']:.0f}, payback {s['payback_days']:.0f} days")
    for k, c in compare_layouts().items():
        print(f"  layout {k}: {c['n']} x {c['diameters_mm']} mm, {c['mass_kg']:.1f} kg, fits {c['fits']}, rank {c['rank']}/6 {c['dof']}, "
              f"weakest {c['min_force_N']:.0f} N, lift 30 N {c['lift30_W']:.0f} W, surge 60 N {c['surge60_W']:.0f} W, {c['N_per_W']:.2f} N/W")
    for d in (20, 80, 200):
        s = Walrus.sizing({"bed_density": d})
        print(f"  bed {d}/m2: {s['rate_per_h']:.0f}/h ({s['limited_by']}), {s['day_kg']:.0f} kg/day")
