"""Barracuda (after *Sphyraena barracuda*: long, fast, a patient shadower that closes in a burst): the unmanned
interceptor boat of the guard carrier Esmeralda (``esmeralda.py``). It finds, shadows and blocks small fast craft
near an offshore platform at night and climbs the escalation ladder - hail, shadow, lights and loudhailer, block,
water cannon - under a remote operator. It does not fight.

What it is (x forward from the transom, y left, z up from the keel in the CAD; the card from the waterline):

* an 8.6 m **deep-V aluminium hull with a closed-cell foam collar** (a RIB's collar that cannot be punctured: it is
  the fender it blocks with), 22 deg deadrise, the survey boat's lofted hull (``survey_boat.py``) at 8 m;
* **twin high-speed diesels on twin waterjets** (no propeller near people or lines when it blocks; reversing
  buckets for the crash stop; one engine and one jet for the loiter); the starboard engine's front PTO drives the
  water cannon's pump;
* a **folding sensor mast**: navigation radar, a gimballed day / cooled-MWIR camera with a laser rangefinder, AIS,
  VHF, the radios and the satcom; a microphone array; the hailer, a searchlight and the strobe on a pan-tilt head
  slaved to the gimbal; the **water-cannon monitor** on the foredeck;
* a **bow recovery eye** (the carrier's capture hook takes it) and keel guides for the carrier's stern ramp.

Every number comes from ``sizing``: the mass budget closes with the engines, the jets and the fuel it needs; the
calm-water resistance is Savitsky's planing method (the hull's equilibrium trim) with a class curve below the hump;
the motion in waves is Savitsky and Brown's impact acceleration; the sensors and effectors are ``sensing.py``.
Class values and assumptions are marked where they are set; no vendor's boat or equipment.
"""
from __future__ import annotations

import math

try:
    import cadquery as cq
except ImportError:
    cq = None
from vegeta.dedalus import Design, Parameter

import sensing

RHO_SW = 1025.0
RHO_AIR = 1.225
NU_SW = 1.19e-6
G = 9.81
KN = 0.5144
NM = 1852.0
DIESEL_KG_L = 0.84
FT = 0.3048

#: What it must do (the brief, task 1 of the sea-game machines).
REQUIREMENTS = {"sprint_kn": 35.0, "cruise_kn": 25.0, "cruise_hs_m": 1.0, "loiter_kn": 8.0, "loiter_h": 10.0,
                "range_nm": 150.0, "max_hs_m": 2.5, "reserve": 0.10, "length_m": (8.0, 9.0)}

#: Diesel ratings to choose from (kW per engine at the crank; high-speed marine diesels, class steps).
ENGINE_STEPS = (220.0, 260.0, 300.0, 340.0, 380.0, 440.0)
#: Machinery per kW (class values, assumed): engine with gearbox and cooling (kg/kW, EUR/kW), waterjet dry and its
#: entrained water (kg/kW), waterjet price (EUR/kW).
ENGINE_KG_KW, ENGINE_EUR_KW = 1.75, 190.0
JET_KG_KW, JET_WATER_KG_KW, JET_EUR_KW = 0.33, 0.20, 110.0

#: Fixed parts (mass kg, power W, price EUR), class values (assumed). The hull's mass comes from its plating.
SENSORS = {
    "navigation radar (X-band, solid state, 4 ft open array)": (28.0, 150.0, 18000.0),
    "EO/IR gimbal: HD day zoom + cooled MWIR 640 + laser rangefinder": (26.0, 250.0, 185000.0),
    "AIS receiver (class B transponder, silent mode)": (2.0, 10.0, 1500.0),
    "VHF marine radio with DSC (25 W)": (3.0, 30.0, 1500.0),
    "microphone array (8 mics, beam-forming, on the mast)": (4.0, 15.0, 6000.0),
    "MANET radio (4.9 GHz, 5 W, video) + UHF command link": (8.0, 60.0, 28000.0),
    "LEO broadband satcom + L-band backup terminal": (14.0, 90.0, 16000.0),
    "autonomy computer (COLREGs, tracking, effectors' interlocks)": (18.0, 300.0, 120000.0),
    "RTK GNSS + MEMS-FOG INS": (6.0, 30.0, 22000.0),
}
EFFECTORS_KIT = {
    "long-range acoustic hailer (150 dB class)": (30.0, 400.0, 40000.0),
    "LED searchlight on the pan-tilt head (gimbal-slaved)": (18.0, 600.0, 9000.0),
    "white LED dazzler strobe (non-laser)": (7.0, 300.0, 6000.0),
    "water cannon: PTO pump, sea chest, monitor, piping": (175.0, 0.0, 38000.0),
    "navigation lights, horn, blue flashing light": (6.0, 60.0, 1500.0),
}
OUTFIT = {
    "foam collar (closed-cell PE, PU skin) + rubber D-fender": None,      # sized in ``sizing`` from its section
    "enclosed electronics console + folding mast": (190.0, 0.0, 30000.0),
    "fuel system (tank, lines, filters, auto-coupling)": None,            # 12 % of the fuel it holds
    "24 V LiFePO4 bank 4 kWh + 2 alternators + wiring": (130.0, 0.0, 9000.0),
    "bow recovery eye, keel guides, towing and lifting points": (80.0, 0.0, 6000.0),
    "steering and bucket hydraulics, controls": (60.0, 0.0, 12000.0),
    "fire suppression, bilge pumps, outfit": (70.0, 0.0, 6000.0),
}
WEIGHT_MARGIN = 0.08
DRAUGHT_CAD_MM = 630.0               # the CAD's waterline above the keel: the sizing's full-load draught (a test holds them together)


class Barracuda(Design):
    parameters = [
        Parameter("part", "boat", choices=("boat", "hull", "collar", "jets", "console", "mast", "cannon", "effectors"),
                  description="what to build"),
        Parameter("length", 8400.0, "mm", min=6000, description="the aluminium hull (the collar adds 0.58 m)"),
        Parameter("beam", 2400.0, "mm", min=1500, description="the hull's deck beam inside the collar"),
        Parameter("chine_beam", 2000.0, "mm", min=1200, description="the planing bottom's beam (Savitsky's b): narrow for the impacts"),
        Parameter("depth", 1350.0, "mm", min=800, description="keel to the gunwale at midship"),
        Parameter("deadrise_deg", 24.0, "deg", min=10, max=30, description="at the transom: a deep V"),
        Parameter("collar_d", 480.0, "mm", min=300, description="the foam collar's diameter"),
        Parameter("lcg_frac", 0.38, "", min=0.25, max=0.5, description="the centre of gravity from the transom, of the waterline length"),
        Parameter("mast_h", 4.0, "m", min=2.5, description="the gimbal and radar above the waterline, mast up"),
        Parameter("sprint_kn", 38.0, "kn", min=30, description="the design sprint at full load in calm water (the brief: >= 35)"),
        Parameter("service_margin", 1.20, "", min=1.0, description="real hull vs Savitsky: spray, appendages' interaction, fouling, engine derating (assumed)"),
        Parameter("g_sustained", 2.0, "g", min=0.5, description="the average of the 1/10 highest impacts at the CG for sustained running: no crew (crewed practice is 1.0-1.5 g), equipment shock-mounted, the hull's fatigue sets it (assumed)"),
        Parameter("g_sprint", 3.0, "g", min=0.5, description="the same for a sprint of minutes (the hull's structural design acceleration is higher, assumed)"),
        Parameter("lateral_g", 0.5, "g", min=0.2, description="sustained turn (a deep V banks; above it chine-walk and tripping risk)"),
        Parameter("reverse_frac", 0.45, "", min=0.2, max=0.8, description="the reversing buckets' thrust, of the forward bollard thrust"),
        Parameter("bollard_factor", 1.0, "", min=0.5, description="waterjet bollard thrust over its thrust at 20 kn (class)"),
        Parameter("engine_ramp_s", 5.0, "s", min=0.5, description="idle to full power: turbo lag and the controller's throttle rate (assumed)"),
        Parameter("hailer_db", 150.0, "dB", description="at 1 m, 2 kHz"),
        Parameter("searchlight_lm", 14000.0, "lm", description="LED searchlight"),
        Parameter("searchlight_deg", 4.0, "deg", description="its full beam"),
        Parameter("cannon_l_min", 1500.0, "L/min"),
        Parameter("cannon_bar", 10.0, "bar"),
    ]

    @staticmethod
    def p(over=None):
        return {q.name: q.default for q in Barracuda.parameters} | (over or {})

    # ----------------------------------------------------------------------------- hydrodynamics
    @staticmethod
    def lwl(p):
        return 0.9 * p["length"] / 1000

    @classmethod
    def savitsky(cls, p, mass_kg, v):
        """Savitsky's planing equilibrium for a prismatic hull with the thrust through the CG: the trim at which the
        centre of pressure falls on the LCG; the wetted length, the drag (pressure + friction). ``v`` m/s."""
        b, beta = p["chine_beam"] / 1000, p["deadrise_deg"]
        W = mass_kg * G
        cv = v / math.sqrt(G * b)
        clb = W / (0.5 * RHO_SW * v * v * b * b)
        cl0 = clb
        for _ in range(60):
            cl0 = clb + 0.0065 * beta * cl0 ** 0.6
        lcg = p["lcg_frac"] * cls.lwl(p)

        def lam_for(tau):
            lo, hi = 0.05, 12.0
            for _ in range(60):
                lam = 0.5 * (lo + hi)
                f = tau ** 1.1 * (0.012 * lam ** 0.5 + 0.0055 * lam ** 2.5 / cv ** 2)
                lo, hi = (lam, hi) if f < cl0 else (lo, lam)
            return lam

        def lp(tau):
            lam = lam_for(tau)
            return b * lam * (0.75 - 1 / (5.21 * cv ** 2 / lam ** 2 + 2.39)), lam

        lo, hi = 0.5, 15.0                                   # lp falls as the trim rises
        for _ in range(60):
            tau = 0.5 * (lo + hi)
            lo, hi = (tau, hi) if lp(tau)[0] > lcg else (lo, tau)
        tau = 0.5 * (lo + hi)
        lam = lp(tau)[1]
        t = math.radians(tau)
        cl = 0.012 * lam ** 0.5 * tau ** 1.1
        vm = v * math.sqrt(max(1 - (cl - 0.0065 * beta * cl ** 0.6) / (lam * math.cos(t)), 0.1))
        re = vm * lam * b / NU_SW
        cf = 0.075 / (math.log10(re) - 2) ** 2 + 0.0004
        df = 0.5 * RHO_SW * vm * vm * lam * b * b * cf / math.cos(math.radians(beta))
        D = W * math.tan(t) + df / math.cos(t)
        return {"trim_deg": tau, "lambda": lam, "wetted_length_m": lam * b, "cv": cv, "drag_N": D, "friction_N": df,
                "valid": 0.6 <= cv <= 13 and lam <= 4.0 and 2.0 <= tau <= 15.0}

    @classmethod
    def resistance(cls, p, mass_kg, v_kn, hs_m=0.0):
        """Total resistance (N) at ``v_kn``: Savitsky above 16 kn; below it a class curve of resistance over weight
        (displacement mode to 8 kn, the hump at 13 kn: R/W 0.045 and 0.16, assumed) joined to Savitsky at 16 kn;
        the service margin; added resistance in waves (+0.35 Hs/b, assumed); the air drag of the topsides."""
        W = mass_kg * G
        if v_kn >= 16.0:
            R = cls.savitsky(p, mass_kg, v_kn * KN)["drag_N"]
        else:
            r16 = cls.savitsky(p, mass_kg, 16 * KN)["drag_N"] / W
            pts = ((0.0, 0.0), (8.0, 0.045), (13.0, 0.16), (16.0, r16))
            if v_kn <= 8.0:
                rw = 0.045 * (v_kn / 8.0) ** 2
            else:
                rw = next(r0 + (r1 - r0) * (v_kn - v0) / (v1 - v0) for (v0, r0), (v1, r1) in zip(pts, pts[1:]) if v0 <= v_kn <= v1)
            R = rw * W
        R *= p["service_margin"] * (1 + 0.35 * hs_m / (p["chine_beam"] / 1000))
        a_front = (p["beam"] + p["collar_d"]) / 1000 * 1.9
        return R + 0.5 * RHO_AIR * 0.7 * a_front * (v_kn * KN) ** 2

    @staticmethod
    def jet_eta(v_kn):
        """Waterjet overall propulsive efficiency (effective power over shaft power), rising with speed: 0.43 at 8 kn,
        0.58 at 25, 0.62 at 35 (class curve, assumed)."""
        return 0.30 + 0.37 * (1 - math.exp(-max(v_kn, 0.0) / 18.0))

    @classmethod
    def shaft_kW(cls, p, mass_kg, v_kn, hs_m=0.0, hotel_W=0.0):
        """The engines' total shaft power at ``v_kn`` (kW), the hotel load through the alternators (85 %)."""
        if v_kn <= 0:
            return hotel_W / 0.85 / 1000
        return cls.resistance(p, mass_kg, v_kn, hs_m) * v_kn * KN / cls.jet_eta(v_kn) / 1000 + hotel_W / 0.85 / 1000

    @staticmethod
    def sfc(load):
        """High-speed diesel specific fuel consumption (g/kWh) by load fraction: 205 at full, ~270 at 10 % (class)."""
        return 205.0 + 95.0 * (1 - min(max(load, 0.05), 1.0)) ** 3

    @classmethod
    def fuel_l_h(cls, p, mass_kg, v_kn, engine_kW, hs_m=0.0, hotel_W=0.0, engines=None):
        """Litres an hour at ``v_kn``: on one engine below 12 kn (the other stopped), on both above."""
        P = cls.shaft_kW(p, mass_kg, v_kn, hs_m, hotel_W)
        n = engines or (1 if v_kn < 12 else 2)
        load = P / (n * engine_kW)
        return P * cls.sfc(load) / 1000 / DIESEL_KG_L, load

    @classmethod
    def max_speed(cls, p, mass_kg, P_kW, hs_m=0.0, hotel_W=0.0):
        """The speed (kn) the engines' power holds (calm water when hs = 0)."""
        lo, hi = 0.0, 70.0
        for _ in range(50):
            v = 0.5 * (lo + hi)
            lo, hi = (v, hi) if cls.shaft_kW(p, mass_kg, v, hs_m, hotel_W) < P_kW else (lo, v)
        return lo

    @classmethod
    def impact_g(cls, p, mass_kg, v_kn, hs_m):
        """Savitsky and Brown (1976): the average of the 1/10 highest vertical impact accelerations at the CG (g),
        from the significant wave height over the beam, the trim, the deadrise, the speed-length ratio and the beam
        loading. Valid to H/b 0.7 and V/sqrt(L) 6 (knots, feet); above, an extrapolation."""
        b = p["chine_beam"] / 1000
        L = cls.lwl(p)
        tau = min(max(cls.savitsky(p, mass_kg, max(v_kn, 16) * KN)["trim_deg"], 3.0), 7.0)
        cdel = mass_kg / (RHO_SW * b ** 3)
        vl = v_kn / math.sqrt(L / FT)
        return 0.0104 * (hs_m / b + 0.084) * (tau / 4) * (5 / 3 - p["deadrise_deg"] / 30) * vl ** 2 * (L / b) / cdel

    @classmethod
    def slam_speed(cls, p, mass_kg, hs_m, g_limit):
        if hs_m <= 0:
            return 99.0
        lo, hi = 0.0, 70.0
        for _ in range(50):
            v = 0.5 * (lo + hi)
            lo, hi = (v, hi) if cls.impact_g(p, mass_kg, v, hs_m) < g_limit else (lo, v)
        return lo

    # ----------------------------------------------------------------------------- the boat
    @classmethod
    def hull_mass(cls, p):
        """Aluminium (5083) plating with frames and stringers (x 1.35): bottom 6 mm, sides 5 mm, deck 4 mm."""
        L, B, D = p["length"] / 1000, p["beam"] / 1000, p["depth"] / 1000
        bottom = 0.85 * L * p["chine_beam"] / 1000 / math.cos(math.radians(p["deadrise_deg"]))
        sides = 2 * 0.9 * L * D * 0.75
        deck = 0.8 * L * B
        return 2700 * 1.35 * (bottom * 0.006 + sides * 0.005 + deck * 0.004 + 1.2 * D * 0.006)

    @classmethod
    def collar_mass(cls, p):
        """Closed-cell PE foam (35 kg/m3) with a 4 kg/m PU skin and D-fender, round the hull but the transom."""
        r = p["collar_d"] / 2000
        run = 2 * 0.95 * p["length"] / 1000 + 0.8
        return run * (math.pi * r * r * 35.0 + 4.0), run

    @classmethod
    def mass_budget(cls, p, engine_kW, fuel_L):
        fixed = {k: v[0] for k, v in {**SENSORS, **EFFECTORS_KIT}.items()}
        fixed |= {k: v[0] for k, v in OUTFIT.items() if v}
        collar, _ = cls.collar_mass(p)
        items = {"hull (aluminium, framed)": cls.hull_mass(p), "foam collar (closed-cell PE, PU skin) + rubber D-fender": collar,
                 f"2 diesels {engine_kW:.0f} kW + gearboxes": 2 * engine_kW * ENGINE_KG_KW,
                 "2 waterjets (dry)": 2 * engine_kW * JET_KG_KW, **fixed,
                 "fuel system (tank, lines, filters, auto-coupling)": 0.12 * fuel_L * DIESEL_KG_L}
        light = sum(items.values()) * (1 + WEIGHT_MARGIN)
        load = {"fuel": fuel_L * DIESEL_KG_L, "entrained jet water": 2 * engine_kW * JET_WATER_KG_KW}
        return items, light, load

    @classmethod
    def hotel_W(cls):
        return sum(v[1] for v in SENSORS.values()) + sum(v[1] for k, v in EFFECTORS_KIT.items() if "water cannon" not in k)

    @classmethod
    def sizing(cls, over=None):
        """Close the boat: pick the smallest engine pair that drives the full-load boat at the design sprint, size
        the tank for the worst of the range, the loiter and a guard sortie (+10 %), and repeat until the mass
        settles. Then the speed against the sea state, the fuel per speed band, the manoeuvres, the sensors and the
        effectors, the carrier's ramp, the price."""
        p = cls.p(over)
        R = REQUIREMENTS
        hotel = cls.hotel_W()
        engine, fuel_L = ENGINE_STEPS[0], 600.0
        for _ in range(30):
            items, light, load = cls.mass_budget(p, engine, fuel_L)
            m_full = light + sum(load.values())
            need = cls.shaft_kW(p, m_full, p["sprint_kn"], 0.0, hotel)
            engine = next((e for e in ENGINE_STEPS if 2 * e * 0.95 >= need), ENGINE_STEPS[-1])
            m_half = light + load["entrained jet water"] + 0.5 * load["fuel"]
            l_cruise, _ = cls.fuel_l_h(p, m_half, R["cruise_kn"], engine, R["cruise_hs_m"], hotel)
            l_loiter, _ = cls.fuel_l_h(p, m_half, R["loiter_kn"], engine, 0.5, hotel)
            l_sprint, _ = cls.fuel_l_h(p, m_half, p["sprint_kn"], engine, 0.0, hotel)
            need_range = R["range_nm"] / R["cruise_kn"] * l_cruise
            need_loiter = R["loiter_h"] * l_loiter
            sortie = need_loiter + 2 * 30 / R["cruise_kn"] * l_cruise + 0.5 * l_sprint   # 10 h on watch, 30 nm out and back, 30 min of sprints
            new_fuel = max(need_range, need_loiter, sortie) * (1 + R["reserve"])
            if abs(new_fuel - fuel_L) < 1:
                break
            fuel_L = new_fuel
        fuel_L = math.ceil(fuel_L / 10) * 10
        items, light, load = cls.mass_budget(p, engine, fuel_L)
        m_full = light + sum(load.values())
        m_half = light + load["entrained jet water"] + 0.5 * load["fuel"]
        P = 2 * engine * 0.95                                   # 95 % of the rating, continuously

        # draught and freeboard: a deep V's displaced volume at rest (prismatic section, Cb 0.45 over the waterline)
        b, beta = p["chine_beam"] / 1000, math.radians(p["deadrise_deg"])
        vol = m_full / RHO_SW
        lwl = cls.lwl(p)
        T_v = b / 2 * math.tan(beta)                            # the V's own height
        area = vol / (lwl * 0.62)                               # midship section area needed
        if area <= b * T_v / 2:
            draught = math.sqrt(2 * area * math.tan(beta))
        else:
            draught = T_v + (area - b * T_v / 2) / b
        draught_jets = draught + 0.05

        # speed against sea state: power (added resistance) and the impacts, sustained and in a sprint
        table = []
        for hs in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5):
            v_pow = cls.max_speed(p, m_half, P, hs, hotel)
            v_sus = cls.slam_speed(p, m_half, hs, p["g_sustained"])
            v_spr = cls.slam_speed(p, m_half, hs, p["g_sprint"])
            sus, spr = min(v_pow, v_sus), min(v_pow, v_spr)
            crewed = min(v_pow, cls.slam_speed(p, m_half, hs, 1.5))
            table.append({"hs_m": hs, "sustained_kn": round(sus, 1), "sprint_kn": round(spr, 1), "at_crewed_1_5g_kn": round(crewed, 1),
                          "power_kn": round(v_pow, 1), "sustained_limited_by": "propulsion" if v_pow <= v_sus else "slamming",
                          "sprint_limited_by": "propulsion" if v_pow <= v_spr else "slamming",
                          "g_at_sprint": round(cls.impact_g(p, m_half, spr, hs), 2) if hs else 0.0,
                          "outside_savitsky_brown": hs > 0 and (hs / b > 0.7 or sus / math.sqrt(lwl / FT) > 6)})
        cruise_at_hs1 = next(r["sustained_kn"] for r in table if r["hs_m"] == R["cruise_hs_m"])

        # fuel per speed band (half load, the sea state each band is used in)
        bands = {}
        for name, v, hs in (("drift (engines off, hotel on the battery and an alternator run)", 0.0, 0.5),
                            ("loiter", R["loiter_kn"], 0.5), ("transit below the hump", 12.0, 0.5),
                            ("cruise", R["cruise_kn"], R["cruise_hs_m"]), ("sprint", p["sprint_kn"], 0.0)):
            if v == 0:
                duty = hotel / 0.85 / 6000.0                       # two 3 kW alternators on one idling engine, run to recharge the bank
                lh = duty * (2.0 + 6.0 / 0.85 * 0.30 / DIESEL_KG_L)   # idle 2 L/h + the alternators' load at 300 g/kWh (class)
                bands[name] = {"kn": 0.0, "kW": round(hotel / 1000, 2), "l_h": round(lh, 1), "load": 0.0,
                               "hours_on_tank": round(fuel_L / (1 + R["reserve"]) / lh)}
                continue
            lh, load_f = cls.fuel_l_h(p, m_half, v, engine, hs, hotel)
            bands[name] = {"kn": v, "hs_m": hs, "kW": round(cls.shaft_kW(p, m_half, v, hs, hotel)), "l_h": round(lh, 1),
                           "load": round(load_f, 2), "nm_per_l": round(v / lh, 3),
                           "hours_on_tank": round(fuel_L / (1 + R["reserve"]) / lh, 1),
                           "range_nm": round(fuel_L / (1 + R["reserve"]) / lh * v)}
        cannon = sensing.water_cannon(p["cannon_l_min"], p["cannon_bar"])
        lh_cannon = cannon["shaft_kW"] * cls.sfc(0.5) / 1000 / DIESEL_KG_L
        bands["water cannon (on top of the speed's fuel)"] = {"kW": cannon["shaft_kW"], "l_h": round(lh_cannon, 1)}

        man = cls.manoeuvres(p, m_half, P, engine)
        man["station_keeping"]["pace_target_up_to_kn_by_hs"] = {r["hs_m"]: round(max(r["sustained_kn"] - 2, 0), 1) for r in table}
        man["station_keeping"]["block_target_up_to_kn_by_hs"] = {r["hs_m"]: round(max(r["sprint_kn"] - 5, 0), 1) for r in table}
        sens = cls.sensor_ranges(p)
        eff = cls.effector_numbers(p)
        ramp = cls.carrier_ramp(p, m_full)
        bump = cls.blocking(p, m_full)

        price_items = {k: v[2] for k, v in {**SENSORS, **EFFECTORS_KIT}.items()}
        price_items |= {k: v[2] for k, v in OUTFIT.items() if v}
        price_items |= {"hull, collar, build and trials": 165000.0, "2 diesels + gearboxes": 2 * engine * ENGINE_EUR_KW,
                        "2 waterjets with reversing buckets": 2 * engine * JET_EUR_KW, "integration, certification (class, flag)": 90000.0}
        sprint_calm = cls.max_speed(p, m_full, P, 0.0, hotel)
        return {"length_overall_m": round((p["length"] + p["collar_d"] * 1.2) / 1000, 2),
                "beam_overall_m": round((p["beam"] + p["collar_d"]) / 1000, 2), "hull_length_m": p["length"] / 1000,
                "lwl_m": round(lwl, 2), "depth_m": p["depth"] / 1000, "draught_m": round(draught_jets, 2),
                "freeboard_m": round(p["depth"] / 1000 - draught, 2), "air_draught_mast_up_m": round(p["mast_h"] + 0.6, 2),
                "air_draught_mast_folded_m": round(p["depth"] / 1000 - draught + 1.25, 2),
                "items": {k: round(v, 1) for k, v in items.items()}, "lightship_kg": round(light),
                "fuel_L": fuel_L, "fuel_kg": round(fuel_L * DIESEL_KG_L), "mass_full_kg": round(m_full), "mass_half_kg": round(m_half),
                "engine_kW": engine, "engines": 2, "installed_kW": 2 * engine,
                "sprint_kn_full_load": round(sprint_calm, 1), "sprint_kn_half_load": round(cls.max_speed(p, m_half, P, 0.0, hotel), 1),
                "cruise_kn": R["cruise_kn"], "loiter_kn": R["loiter_kn"], "cruise_kn_at_hs1": cruise_at_hs1,
                "speed_vs_hs": table, "fuel_bands": bands,
                "range_nm_at_cruise": bands["cruise"]["range_nm"], "loiter_h": bands["loiter"]["hours_on_tank"],
                "trim_at_sprint_deg": round(cls.savitsky(p, m_half, sprint_calm * KN)["trim_deg"], 1),
                "savitsky_valid_at_sprint": cls.savitsky(p, m_half, sprint_calm * KN)["valid"],
                "manoeuvres": man, "sensors": sens, "effectors": eff | {"water_cannon": cannon | {"fuel_l_h": round(lh_cannon, 1),
                                                                                                   "drive": "starboard engine's front PTO, clutched; sea water from its own sea chest and strainer"}},
                "blocking": bump, "carrier": ramp, "hotel_W": round(hotel),
                "price": price_items, "price_eur": round(sum(price_items.values()), -3),
                "meets": {"sprint_35kn": sprint_calm >= R["sprint_kn"], "cruise_25kn_hs1": cruise_at_hs1 >= R["cruise_kn"],
                          "range_150nm": bands["cruise"]["range_nm"] >= R["range_nm"], "loiter_10h": bands["loiter"]["hours_on_tank"] >= R["loiter_h"],
                          "length_8_9m": R["length_m"][0] <= (p["length"] + p["collar_d"] * 1.2) / 1000 <= R["length_m"][1]}}

    @classmethod
    def manoeuvres(cls, p, mass_kg, P_kW, engine_kW):
        """Turning circle by speed (the lateral acceleration's limit, never under 2.2 lengths: the jets' vectoring at
        low speed), acceleration 0 -> 30 kn and the crash stop from the sprint (time integration: jet thrust
        min(bollard, eta P / v), the engines' ramp, resistance; the virtual mass +5 %)."""
        L = (p["length"] + p["collar_d"] * 1.2) / 1000
        m = mass_kg * 1.05
        v20 = 20 * KN
        t_bollard = p["bollard_factor"] * cls.jet_eta(20) * P_kW * 1000 / v20

        def thrust(v, t):
            ramp = min(1.0, 0.15 + 0.85 * t / p["engine_ramp_s"])
            return min(t_bollard, cls.jet_eta(v / KN) * P_kW * 1000 * ramp / max(v, 0.5))

        v, t, x, dt = 0.0, 0.0, 0.0, 0.02
        while v < 30 * KN and t < 120:
            a = (thrust(v, t) - cls.resistance(p, mass_kg, v / KN)) / m
            v, x, t = v + a * dt, x + v * dt, t + dt
        t_accel, x_accel = t, x
        v0 = cls.max_speed(p, mass_kg, P_kW, 0.0, cls.hotel_W()) * KN
        v, t, x = v0, 0.0, 0.0
        rev = p["reverse_frac"] * t_bollard
        while v > 0.05 and t < 60:
            ramp = min(1.0, t / 1.5)                           # the buckets drop in 1.5 s
            a = -(cls.resistance(p, mass_kg, v / KN) + rev * ramp) / m
            v, x, t = v + a * dt, x + v * dt, t + dt
        turn = {}
        for kn in (5, 10, 20, 30):
            vv = kn * KN
            d = max(2.2 * L, 2 * vv * vv / (p["lateral_g"] * G))
            turn[f"{kn}kn"] = {"diameter_m": round(d), "half_turn_s": round(math.pi * d / 2 / vv, 1)}
        return {"turning_circle": turn, "lateral_g": p["lateral_g"], "accel_0_30kn_s": round(t_accel, 1),
                "accel_0_30kn_m": round(x_accel), "bollard_thrust_kN": round(t_bollard / 1000, 1),
                "crash_stop_from_kn": round(v0 / KN, 1), "crash_stop_s": round(t, 1), "crash_stop_m": round(x),
                "crash_stop_lengths": round(x / L, 1), "reverse_thrust_kN": round(rev / 1000, 1),
                "station_keeping": {"offset_hold_m": 2.0, "relative_fix_m": 0.5,
                                    "how": "the target's track from the radar, the gimbal's laser range and bearing (+-0.5 m at 50 m) and our RTK fix; "
                                           "speed and heading loops on both jets; to get ahead of a target to block it needs >= 5 kn over its speed"}}

    @classmethod
    def sensor_ranges(cls, p):
        """The sensors' ranges at the mast's height: the gimbal's DRI for a 6 m skiff and a 20 m fishing boat, the
        radar's detection by sea state, AIS, VHF, the microphones."""
        h = p["mast_h"]
        gimbal = {"channels": [{"band": "visible", "pixels_h": 1920, "hfov_narrow_deg": 2.0, "hfov_wide_deg": 60.0, "zoom": 30},
                               {"band": "ir", "pixels_h": 640, "hfov_narrow_deg": 2.2, "hfov_wide_deg": 25.0, "zoom": 11}]}
        jitter = 25e-6                                            # the gimbal's stabilisation residual on a planing boat (assumed)
        sk, fb = sensing.TARGETS["skiff_6m"], sensing.TARGETS["fishing_20m"]
        radar = {"what": "X-band solid-state navigation radar, 4 ft array, Doppler target separation", "ref_range_km": 9.0,
                 "ref_rcs_m2": 10.0, "antenna_m": h + 0.3,
                 "skiff_km_by_hs": {hs: round(sensing.radar_range(9.0, 10.0, sk["rcs_m2"], h + 0.3, sk["height_m"], hs), 1) for hs in (0.5, 1.0, 2.0, 2.5)},
                 "fishing_km_by_hs": {hs: round(sensing.radar_range(9.0, 10.0, fb["rcs_m2"], h + 0.3, fb["height_m"], hs), 1) for hs in (0.5, 1.0, 2.0, 2.5)},
                 "note": "ref range: 10 m2 at Pd 0.5 in calm sea (class value, assumed); rain shortens it further (not modelled)"}
        ais = {"receive_class_a_km": round(sensing.horizon_km(h + 0.5, 15.0, "radar"), 1),
               "receive_class_b_km": round(min(sensing.horizon_km(h + 0.5, 3.0, "radar"), 15.0), 1),
               "note": "VHF line of sight (4/3 earth); class B at 2 W also power-limited to ~15 km (class value)"}
        mics = {"outboard_at_speed_m": round(sensing.sound_range(105.0, 55.0 - 10.0, 0.003)),
                "outboard_idle_m": round(sensing.sound_range(85.0, 55.0 - 10.0, 0.003)),
                "with_own_engines_running_m": round(sensing.sound_range(105.0, 85.0 - 10.0, 0.003)),
                "note": "8-mic array, 10 dB array gain; a bearing to an outboard when we drift or loiter on one engine; "
                        "listens for the reply to the hailer. Own engines at speed drown it (assumed noise levels)"}
        return {"mast_m": h, "eoir": gimbal | {"lrf_km": 10.0, "jitter_urad": jitter * 1e6,
                                               "dri_skiff_km": sensing.eoir_dri(gimbal, sk, h, jitter),
                                               "dri_fishing_km": sensing.eoir_dri(gimbal, fb, h, jitter)},
                "radar": radar, "ais": ais, "vhf_km": round(sensing.horizon_km(h + 0.5, 3.0, "radar"), 1), "microphones": mics}

    @classmethod
    def effector_numbers(cls, p):
        return {"hailer": sensing.hailer(p["hailer_db"]) | {"what": "directional (30 deg at 2 kHz), pre-recorded and live voice, on the pan-tilt head"},
                "searchlight": sensing.searchlight(p["searchlight_lm"], p["searchlight_deg"]) | {"haze_range_m": sensing.searchlight(p["searchlight_lm"], p["searchlight_deg"], 5.0)["range_m"]},
                "strobe": sensing.strobe(2.0e6, 0.005, 6.0) | {"what": "white LED, 6 Hz; disorients, does not blind (non-laser; to be checked against IEC 62471 at its closest use)"},
                "vhf_hail": "DSC call and voice on ch 16 and the working channel; an AIS safety message"}

    @classmethod
    def blocking(cls, p, mass_kg, target_kg=1500.0):
        """The collar as the fender it blocks with: the energy it absorbs (closed-cell PE: ~60 kPa mean crush stress
        to 45 % strain over a 1.5 m contact, assumed) against the collision's relative kinetic energy (the reduced
        mass of the two craft, a skiff of 1.5 t); the closing speed it takes without the aluminium touching."""
        r = p["collar_d"] / 2000
        crush = 0.45 * 2 * r
        E = 60e3 * 1.5 * (2 * r * 0.6) * crush
        mu = mass_kg * target_kg / (mass_kg + target_kg)
        v_safe = math.sqrt(2 * E / mu)
        return {"collar_energy_kJ": round(E / 1000, 1), "collar_crush_m": round(crush, 2), "safe_closing_m_s": round(v_safe, 2),
                "safe_closing_kn": round(v_safe / KN, 1), "against_kg": target_kg,
                "how": "blocks by parallel contact at matched speed, collar to hull, closing under the safe speed; never bow-on"}

    @classmethod
    def carrier_ramp(cls, p, mass_kg):
        """What the carrier needs: a stern ramp between its demihulls with a capture hook and a haul-in winch, the
        boat on a cradle at its top. The winch's pull up the ramp (8 deg, rollers mu 0.1), the cycle times (class),
        and the sea-state limit: the boat's heave against the ramp's end (the carrier's heave and pitch,
        ``zarco.SEA_RESPONSE``, 80 % correlated with the boat's) within the ramp's tolerance of 1.0 m (assumed)."""
        from zarco import SEA_RESPONSE as S
        winch_N = 15000.0
        ramp_deg, mu = 8.0, 0.1
        pull = mass_kg * G * (math.sin(math.radians(ramp_deg)) + mu * math.cos(math.radians(ramp_deg)))
        lever = 11.0                                              # the ramp's end from the carrier's motion centre (m)

        def rel(hs):
            f = S["design_factor"] * hs
            boat = 0.5 * hs * S["design_factor"]
            ramp = math.hypot(S["heave_m"] * f, lever * math.sin(math.radians(S["pitch_deg"] * f)))
            return math.sqrt(max(boat ** 2 + ramp ** 2 - 2 * 0.8 * boat * ramp, 0.0))

        hs = 0.0
        while rel(hs + 0.05) <= 1.0 and hs < 5:
            hs += 0.05
        L = (p["length"] + p["collar_d"] * 1.2) / 1000
        return {"system": "stern ramp between the demihulls: a capture hook for the bow eye, a haul-in winch, a cradle with keel guides at its top",
                "ramp_deg": ramp_deg, "ramp_length_m": round(L + 2.0, 1), "ramp_width_m": round((p["beam"] + p["collar_d"]) / 1000 + 0.5, 2),
                "clear_height_m": round(p["depth"] / 1000 + 1.25 + 0.3, 2),
                "winch_kN": winch_N / 1000, "haul_pull_kN": round(pull / 1000, 1), "max_boat_kg": round(winch_N / (pull / mass_kg)),
                "system_mass_kg": 2900.0, "system_price_eur": 260000.0, "power_kW": 22.0,
                "launch_s": 90.0, "recovery_s": 240.0, "refuel_l_min": 100.0,
                "hs_limit_m": round(hs, 2), "relative_motion_tolerance_m": 1.0,
                "cycle": {"launch": "release the lashings 20 s, slack the winch and slide down the wet ramp 40 s, jets on and away 30 s",
                          "recovery": "line up astern at 4 kn 60 s, drive into the ramp's mouth, the hook takes the bow eye 30 s, haul in 90 s, "
                                      "cradle, lash, refuel and shore power couple 60 s"}}

    # ----------------------------------------------------------------------------- CAD (mm, keel at z = 0, transom x = 0)
    def _hull(self, p):
        from survey_boat import SurveyBoat
        hp = {q.name: q.default for q in SurveyBoat.parameters} | {"length": p["length"], "beam": p["beam"], "depth": p["depth"], "deadrise_deg": p["deadrise_deg"],
                               "bow_rise": 450.0, "transom_beam_ratio": 0.92, "stations": 12}
        return SurveyBoat()._hull_solid(hp)

    def _collar(self, p):
        """The foam collar: a tube along each gunwale converging at the bow, its top at the gunwale."""
        L, B, D, r = p["length"], p["beam"], p["depth"], p["collar_d"] / 2
        z = D - r * 0.6
        pts = [(-150, B / 2 + r * 0.2), (L * 0.45, B / 2 + r * 0.35), (L * 0.8, B * 0.32), (L + r * 0.4, 0.0)]
        out = None
        for s in (1, -1):
            for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
                a, b = cq.Vector(x0, s * y0, z + (x0 / L) * 120), cq.Vector(x1, s * y1, z + (x1 / L) * 120)
                seg = cq.Workplane("XY").add(cq.Solid.makeCylinder(r, (b - a).Length, a, b - a)).union(
                    cq.Workplane("XY").add(cq.Solid.makeSphere(r, b)))
                out = seg if out is None else out.union(seg)
        return out

    def _jets(self, p):
        """Twin waterjet nozzles with their reversing buckets behind the transom."""
        out = None
        for y in (420, -420):
            nz = cq.Workplane("YZ").center(y, 330).circle(170).extrude(-450)
            bucket = cq.Workplane("YZ").center(y, 330).rect(420, 420).extrude(-80).translate((-470, 0, 0))
            j = nz.union(bucket)
            out = j if out is None else out.union(j)
        return out

    def _console(self, p):
        """The enclosed electronics console amidships and the engine hatches aft."""
        D = p["depth"]
        con = cq.Workplane("XY").box(2000, 1500, 1150).translate((p["length"] * 0.5, 0, D + 575)).edges("|Z").fillet(200)
        hatch = cq.Workplane("XY").box(2000, 1900, 300).translate((1300, 0, D + 150))
        return con.union(hatch)

    def _mast(self, p):
        """The folding mast on the console: radar array, the EO/IR gimbal ball, the antennas, the bow eye."""
        D = p["depth"]
        x0 = p["length"] * 0.5 - 200
        top = p["mast_h"] * 1000 + DRAUGHT_CAD_MM
        pole = cq.Workplane("XY").add(cq.Solid.makeCylinder(70, top - (D + 1150), cq.Vector(x0, 0, D + 1150)))
        radar = cq.Workplane("XY").box(250, 1220, 160).translate((x0, 0, top + 120))
        ball = cq.Workplane("XY").add(cq.Solid.makeSphere(190, cq.Vector(x0 + 300, 0, top - 200)))
        arm = cq.Workplane("XY").box(400, 80, 80).translate((x0 + 150, 0, top - 40))
        ants = cq.Workplane("XY").newObject([cq.Compound.makeCompound([cq.Solid.makeCylinder(25, 900, cq.Vector(x0 - 150, y, top)) for y in (300, -300)])])
        dome = cq.Workplane("XY").add(cq.Solid.makeSphere(170, cq.Vector(x0 - 250, 0, top - 300)))
        eye = cq.Workplane("XY").add(cq.Solid.makeTorus(110, 30, cq.Vector(p["length"] + 120, 0, D - 150), cq.Vector(0, 1, 0)))
        return pole.union(radar).union(ball).union(arm).union(ants).union(dome).union(eye)

    def _cannon(self, p):
        """The water-cannon monitor on the foredeck: a pedestal, the swivel, the barrel at its elevation."""
        D = p["depth"]
        x = p["length"] * 0.76
        ped = cq.Workplane("XY").add(cq.Solid.makeCylinder(110, 450, cq.Vector(x, 0, D)))
        d = cq.Vector(math.cos(math.radians(20)), 0, math.sin(math.radians(20)))
        barrel = cq.Workplane("XY").add(cq.Solid.makeCylinder(45, 1000, cq.Vector(x, 0, D + 500), d))
        swivel = cq.Workplane("XY").add(cq.Solid.makeSphere(130, cq.Vector(x, 0, D + 500)))
        return ped.union(swivel).union(barrel)

    def _effectors(self, p):
        """The pan-tilt head on the console's roof: the hailer's flat panel, the searchlight, the strobe."""
        D = p["depth"]
        x = p["length"] * 0.5 + 650
        z = D + 1150
        post = cq.Workplane("XY").add(cq.Solid.makeCylinder(60, 250, cq.Vector(x, 0, z)))
        hailer = cq.Workplane("XY").box(140, 600, 420).translate((x + 60, 0, z + 480))
        light = cq.Workplane("XY").add(cq.Solid.makeCylinder(130, 300, cq.Vector(x - 60, 420, z + 430), cq.Vector(1, 0, 0)))
        strobe = cq.Workplane("XY").box(150, 220, 150).translate((x, -420, z + 420))
        return post.union(hailer).union(light).union(strobe)

    def build(self, p):
        if cq is None:
            raise ImportError("Barracuda's CAD needs CadQuery")
        parts = {"hull": self._hull, "collar": self._collar, "jets": self._jets, "console": self._console, "mast": self._mast,
                 "cannon": self._cannon, "effectors": self._effectors}
        if p["part"] in parts:
            return parts[p["part"]](p)
        return cq.Workplane("XY").newObject([cq.Compound.makeCompound([f(p).val() for f in parts.values()])])


if __name__ == "__main__":
    import json
    s = Barracuda.sizing()
    print(json.dumps({k: v for k, v in s.items() if k not in ("items", "price")}, indent=1, default=str))
