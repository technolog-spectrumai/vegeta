"""WASP — ÉMERILLON made very small (the speed quadcopter of notebook 37, scaled down).

The question: for a speed comparable to ÉMERILLON's (ER1 261 km/h, ER2 383 km/h), how much smaller can the machine
be — at 250 g if possible, under 800 g in any case?

How it is answered: ÉMERILLON's own model (``emerillon_flight``: the airframe sized from its parts, the drag build-up,
the Mach-aware blade-element propeller, the motor, the sagging pack, the level-flight balance) run over a catalogue of
**small** parts. Nothing in the physics is new; what changes is in ``small()``:

- the catalogue (``MOTORS``, ``PROPELLERS``, ``ESCS``, ``BATTERIES``, ``FIXED``): small parts that shops list — T-Motor and
  EMAX motors from 1404 to 2207, APC's Speed 400 electric propellers from 4.1 to 5.5 inches, all-in-one boards, GNB
  and Tattu packs from 4S 300 mAh to 6S 850 mAh, nano avionics. **Every part is bought**; the airframe is printed, as
  ÉMERILLON's is. No own propeller: a row without a listed reverse-rotation version (``pair`` False) is kept in the
  table to show what it would give, and is never chosen;
- the printed airframe's small-scale constants: a 0.8 mm skin, 0.9 mm pods, a shorter stack bay, smaller fins, thinner arms,
  2 g of bumpers and 8 g of power wiring (ÉMERILLON: 1.4 mm, 50 mm, 55 mm fins, 8 mm arms, 8 g, 32 g);
- the blade section at the small propellers' Reynolds number (about 7e4 against 3e5): more profile drag, less lift.

What the model says about size and speed, before any numbers: a propeller's speed limit is a tip speed, so its pitch
speed at that limit is (pitch / diameter) x the limit / pi — **the same at every size**. A small machine reaches
ÉMERILLON's speed only with ÉMERILLON's pitch-to-diameter ratio (ER1: 1.0, ER2: 1.43). The small propellers sold in both rotations stop at 1.0:
ER1's speed is within reach of bought parts, ER2's is not.
"""
from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass

import emerillon_flight as ef
import emerillon_market as mk
from emerillon_market import Part
from ogma import mass as og_mass

PRICED_ON = "2026-11-24"


def _p(key, name, group, price, mass_g, url, kind="partly", note="", **data):
    return Part(key, name, group, price, mass_g, shop=url.split("/")[2] if url else "", url=url, kind=kind, note=note, data=data)


# Every row is a product a shop lists (``url``), read through a web search on PRICED_ON: search snippets, not order
# confirmations (emerillon_market's caveats apply: stock and prices move). ``kind``: ``sourced`` — mass and the
# electrical numbers from the listing; ``partly`` — the product and its price are listed, the numbers named in ``note``
# are assumed.
# kv, r_ohm, i0_a, max_a, max_cells, d_mm x h_mm
MOTORS = [
    _p("tm-f1404-4600", "T-Motor F1404 4600KV", "motor", 16.89, 9.34, "https://www.racedayquads.com/products/t-motor-f1404-4600kv-micro-motor", "sourced",
       "9.34 g with cable, 138 mOhm, 0.6 A idle at 10 V, 20 A peak (60 s), 316 W, 3-4S, 17.9 x 16.6 mm",
       kv=4600.0, r_ohm=0.138, i0_a=0.6, max_a=20.0, max_cells=4, d_mm=17.9, h_mm=16.6),
    _p("tm-f1507-3800", "T-Motor F1507 3800KV", "motor", 19.90, 15.0, "https://pyrodrone.com/products/t-motor-f1507-3800kv-motor", "partly",
       "15 g with cable, 81 mOhm, 23 A peak (60 s), 372 W, 3-4S from T-Motor's table; idle current, can size and price assumed",
       kv=3800.0, r_ohm=0.081, i0_a=0.9, max_a=23.0, max_cells=4, d_mm=19.6, h_mm=17.5),
    _p("emax-2004-3000", "EMAX ECO II 2004 3000KV", "motor", 13.99, 16.4, "https://www.racedayquads.com/products/emax-eco-ii-2004-3000kv-motor", "partly",
       "16.4 g without wire, 80 mOhm, 1.1 A idle at 10 V, 32 A peak, 3-6S, 24.7 x 19.2 mm; price assumed",
       kv=3000.0, r_ohm=0.080, i0_a=1.1, max_a=32.0, max_cells=6, d_mm=24.7, h_mm=19.2),
    _p("emax-2207-1900", "EMAX ECO II 2207 1900KV", "motor", 15.99, 33.4, "https://www.racedayquads.com/products/emax-eco-ii-series-2207-1900kv-motor", "partly",
       "33.4 g without wire, 59 mOhm, 1.4 A idle at 10 V, 45 A peak, 3-6S; can size and price assumed",
       kv=1900.0, r_ohm=0.059, i0_a=1.4, max_a=45.0, max_cells=6, d_mm=27.9, h_mm=19.5),
]
# d_in x p_in, blades, rpm_limit_k (APC: 190,000 / diameter[in] for the Speed 400 electric series), chord fractions,
# tip thickness; ``pair``: a reverse-rotation version is listed (a quadcopter needs two of each hand)
_APC = dict(blades=2, rpm_limit_k=190000.0, chord=(0.075, 0.105, 0.045), tc_tip=0.06)
PROPELLERS = [
    _p("apc41x41e", "APC 4.1x4.1E + 4.1x4.1EP (Speed 400 electric)", "propeller", 2.55, 3.1, "https://motionrc.com/products/apc-4-1x4-1-speed-400-electric-propeller-lp04141e", "sourced",
       "3.1 g; the reverse (EP) is listed at BuddyRC and ReadyMadeRC", d_in=4.1, p_in=4.1, pair=True, **_APC),
    _p("apc475x475e", "APC 4.75x4.75E + 4.75x4.75EP", "propeller", 2.55, 3.96, "https://motionrc.com/products/apc-4-75x4-75-speed-400-electric-propeller-lp04747e", "sourced",
       "3.96 g; the reverse is LP04747EP (stock varies)", d_in=4.75, p_in=4.75, pair=True, **_APC),
    _p("apc5x5e", "APC 5x5E + 5x5EP", "propeller", 2.59, 4.5, "https://www.readymaderc.com/products/details/apc-5x5e-prop", "partly",
       "the reverse is LPB05050EP at Motion RC; mass scaled from the 4.75's", d_in=5.0, p_in=5.0, pair=True, **_APC),
    _p("apc475x55e", "APC 4.75x5.5E (no reverse version found)", "propeller", 2.59, 4.1, "https://motionrc.com/products/apc-4-75x5-5-speed-400-electric-propeller-lp04755e", "partly",
       "mass scaled; NO reverse-rotation listing found: not usable on a quadcopter until one is", d_in=4.75, p_in=5.5, pair=False, **_APC),
    _p("apc525x625e", "APC 5.25x6.25E (no reverse version found)", "propeller", 2.59, 5.0, "https://motionrc.com/products/apc-5-25x6-25-speed-400-electric-propeller-lp053625e", "partly",
       "mass scaled; NO reverse-rotation listing found", d_in=5.25, p_in=6.25, pair=False, **_APC),
    _p("apc55x65e", "APC 5.5x6.5E (no reverse version found)", "propeller", 2.59, 5.5, "https://www.motionrc.com/products/apc-5-5x6-5-speed-400-electric-propeller-lp05565e", "partly",
       "mass scaled; NO reverse-rotation listing found", d_in=5.5, p_in=6.5, pair=False, **_APC),
]
ESCS = [
    _p("geprc-taker-f411-12", "GEPRC TAKER F411 12A AIO (ELRS built in), 2-4S", "esc", 55.99, 7.0, "https://www.getfpv.com/geprc-taker-aio-f411-fc-12a-2-4s-esc-built-in-elrs-2-4ghz.html", "partly",
       "7 g, 12 A continuous, 2-4S, the receiver on board; price assumed", max_a=12.0, max_cells=4, count=1, with_fc=True, with_rx=True),
    _p("geprc-f411-35", "GEPRC GEP-F411-35A AIO, 2-6S", "esc", 62.99, 8.8, "https://www.racedayquads.com/products/geprc-f411-2-6s-aio-toothpick-whoop-flight-controller-w-35a-4in1-esc", "partly",
       "8.8 g, 35 A continuous (40 A burst), 2-6S; price assumed", max_a=35.0, max_cells=6, count=1, with_fc=True, with_rx=False),
    _p("speedybee-f405-aio-40", "SpeedyBee F405 AIO 40A, 3-6S", "esc", 67.49, 13.6, "https://www.racedayquads.com/products/speedybee-aio-f4-fc-40a-3-6s-4-in-1-bluejay-esc-25x25", "sourced",
       "13.6 g with its CNC case, 40 A continuous (45 A for 10 s), 3-6S", max_a=45.0, max_cells=6, count=1, with_fc=True, with_rx=False),
]
BATTERIES = [
    _p("gnb-4s-300", "GNB 4S 300 mAh 80C HV", "battery", 11.99, 36.0, "https://racedayquads.com/products/gaoneng-gnb-15-2v-4s-300mah-80c-lihv-micro-battery-xt30", "partly",
       "36 g, 53 x 17 x 24 mm; price assumed", cells=4, ah=0.30, c_rating=80.0, size_mm=(53.0, 24.0, 17.0)),
    _p("gnb-4s-450", "GNB 4S 450 mAh 80C", "battery", 12.99, 52.1, "https://newbeedrone.com/a/p/products/gnb-450mah-14-8v-4s-80c-xt30-lipo-battery", "partly",
       "52.1 g measured (Oscar Liang), 64 x 26.5 x 17.5 mm; price assumed", cells=4, ah=0.45, c_rating=80.0, size_mm=(64.0, 26.5, 17.5)),
    _p("tattu-r-4s-550", "Tattu R-Line 4S 550 mAh 95C", "battery", 14.99, 60.0, "https://alofthobbies.com/products/tattu-r-line-4s-550mah-95c-xt30", "partly",
       "60 g (+-10), 66 x 19 x 28 mm; price assumed", cells=4, ah=0.55, c_rating=95.0, size_mm=(66.0, 28.0, 19.0)),
    _p("tattu-r-4s-850", "Tattu R-Line 4S 850 mAh 95C", "battery", 17.99, 100.0, "https://www.getfpv.com/tattu-r-line-850mah-4s-95c-lipo-battery-xt-30.html", "partly",
       "95 to 104 g between listings, 59 x 30 x 30 mm; price assumed", cells=4, ah=0.85, c_rating=95.0, size_mm=(59.0, 30.0, 30.0)),
    _p("gnb-6s-650", "GNB 6S 650 mAh 120C HV", "battery", 22.99, 100.0, "https://pyrodrone.com/products/gaoneng-gnb-650mah-22-8v-6s-120c-hv-lipo-battery-xt30", "partly",
       "100 g, 63 x 30 x 31 mm (another shop lists 122 g for the 80C); price assumed", cells=6, ah=0.65, c_rating=120.0, size_mm=(63.0, 31.0, 30.0)),
    _p("tattu-r-6s-650", "Tattu R-Line 6S 650 mAh 95C", "battery", 24.99, 115.0, "https://www.adorama.com/ta6506s1pt30.html", "partly",
       "115 g, 62 x 31 x 36 mm; price assumed", cells=6, ah=0.65, c_rating=95.0, size_mm=(62.0, 36.0, 31.0)),
    _p("gnb-6s-850", "GNB 6S 850 mAh 120C HV", "battery", 26.99, 130.0, "https://pyrodrone.com/products/gaoneng-gnb-850mah-22-8v-6s-hv-120c-lipo-battery-xt30", "partly",
       "130 g, 64 x 30 x 40 mm; price assumed", cells=6, ah=0.85, c_rating=120.0, size_mm=(64.0, 40.0, 30.0)),
]
FIXED = [
    Part("rp1-v2-w", "RadioMaster RP1 V2 ExpressLRS receiver", "radio", 18.99, 1.5, shop="RadioMaster", url="https://www.radiomasterrc.com/products/rp1-expresslrs-2-4ghz-nano-receiver", kind="partly", note="as emerillon_market; left out when the board has a receiver"),
    _p("hglrc-m100-mini", "HGLRC M100 Mini GPS", "gps", 16.99, 2.6, "https://www.getfpv.com/hglrc-m100-mini-gps.html", "partly", "2.6 g, 18.5 mm square; price assumed"),
    _p("caddx-ant", "Caddx Ant nano camera", "video", 15.99, 2.0, "https://caddxfpv.com/products/caddx-ant", "partly", "mass and price assumed (about 2 g)"),
    _p("tbs-unify-nano", "TBS Unify Pro32 Nano 5G8 video transmitter", "video", 29.95, 1.5, "https://www.team-blacksheep.com/products/prod:unifypro32_nano", "partly", "mass and price assumed (about 1.5 g with its lead)"),
    Part("hw-w", "wires, XT30, screws, antenna", "hardware", 10.0, 9.0, kind="assumed", note="an allowance"),
]
mk.CATALOGUE.update({p.key: p for p in MOTORS + PROPELLERS + ESCS + BATTERIES + FIXED})

TARGET_KMH = {"ER1": 261.0, "ER2": 383.0}
SMALL = dict(WALL=0.0008, STACK_BAY=0.034, FIN_SPAN=0.028, FIN_CHORD=0.034, FIN_T=0.002,
             SECTION=dict(cl_alpha=2 * math.pi * 0.9, alpha0_deg=-2.0, cl_max=1.0, cd0=0.020, k=0.040))   # blade at Re ~ 7e4 (HAND)
ARM_T_MIN, POD_WALL, BUMPERS_G, WIRING_G = 0.005, 0.0009, 2.0, 8.0


@dataclass(frozen=True)
class Config(ef.Config):
    """A WASP build: as ``emerillon_flight.Config``, the cells in series read from the pack (4S or 6S)."""
    motor: str = "tm-f1404-4600"
    propeller: str = "apc41x41e"
    esc: str = "geprc-f411-35"
    battery: str = "gnb-4s-450"

    @property
    def cells(self) -> int:
        return mk.part(self.battery).cells * self.n_series


_geometry, _printed = ef.geometry, ef.printed_masses


def geometry(c) -> dict:
    """ÉMERILLON's geometry with the small arm: 5 mm thick at least (8 mm there)."""
    g = _geometry(c)
    g["arm_t"] = max(ARM_T_MIN, 0.30 * g["pod_d"])
    g["arm_c"] = g["arm_t"] / ef.ARM_TC
    g["s_arms"] = 4 * 2.04 * g["arm_c"] * g["arm_l"]
    g["x_prop"] = g["x_arm"] - 0.5 * g["arm_c"] - ef.PROP_GAP
    return g


def printed_masses(c) -> dict:
    """ÉMERILLON's printed-mass rules with the small pod's wall (0.9 mm; 1.6 mm there) and bumpers."""
    g = geometry(c)
    pa6 = mk.part("pa6-cf")
    k = ef.PRINT_FACTOR / 1.3
    arm = 4 * 0.68 * g["arm_c"] * g["arm_t"] * g["arm_l"] * 0.55 * pa6.density * k
    pods = 4 * math.pi * g["pod_d"] * 0.75 * g["pod_l"] * POD_WALL * pa6.density * k
    rings = 2 * math.pi * g["d"] * 0.010 * 0.003 * pa6.density
    return {"petg-hf": _printed(c)["petg-hf"], "pa6-cf": round((arm + pods + rings) * 1e3, 1), "tpu": BUMPERS_G}


def mass_budget(c) -> og_mass.Budget:
    """``emerillon_flight.mass_budget`` with the small build's wiring allowance (8 g a pack-less harness, 4 g a pack)."""
    g = geometry(c)
    motor, prop, esc, bat = (mk.part(k) for k in (c.motor, c.propeller, c.esc, c.battery))
    b, I = og_mass.Budget(), og_mass.Item
    for i, a in enumerate((45, 135, 225, 315)):
        y, z = g["r_arm"] * math.cos(math.radians(a)), g["r_arm"] * math.sin(math.radians(a))
        b.add(I(f"motor {i + 1}", motor.mass_g / 1e3, (g["x_arm"], y, z), group="propulsion"),
              I(f"propeller {i + 1}", prop.mass_g / 1e3, (g["x_prop"], y, z), group="propulsion"))
    x_bay = g["nose"]
    b.add(I("all-in-one board", esc.mass_g / 1e3, (x_bay + 0.015, 0, 0), group="propulsion"))
    L = bat.size_mm[0] / 1e3 + 0.006
    for k in range(c.packs):
        b.add(I(f"pack {k + 1}", bat.mass_g / 1e3, (x_bay + ef.STACK_BAY + (k + 0.5) * L, 0, 0), group="battery"))
    for q in FIXED:
        if q.group == "radio" and getattr(esc, "with_rx", False):
            continue
        b.add(I(q.name, q.mass_g / 1e3, (0.6 * g["nose"] if q.group in ("video", "gps") else x_bay + 0.015, 0, 0), group="avionics"))
    pm = printed_masses(c)
    b.add(I("skins (PETG)", pm["petg-hf"] / 1e3, (0.52 * g["length"], 0, 0), group="airframe"),
          I("arms, pods, rings (PA6-CF)", pm["pa6-cf"] / 1e3, (g["x_arm"], 0, 0), group="airframe"),
          I("bumpers (TPU)", pm["tpu"] / 1e3, (g["length"], 0, 0), group="airframe"),
          I("power wiring", (WIRING_G + 4.0 * (c.packs - 1)) / 1e3, (x_bay + 0.02, 0, 0), group="propulsion"))
    return b


@contextlib.contextmanager
def small():
    """ÉMERILLON's model at WASP's scale: its constants, catalogue of fixed parts, geometry and mass rules swapped for
    the small ones, and put back afterwards. Every WASP number is computed inside this."""
    keep = {k: getattr(ef, k) for k in SMALL}
    keep_fn = ef.geometry, ef.printed_masses, ef.mass_budget, mk.FIXED
    for k, v in SMALL.items():
        setattr(ef, k, v)
    ef.geometry, ef.printed_masses, ef.mass_budget, mk.FIXED = geometry, printed_masses, mass_budget, FIXED
    try:
        yield
    finally:
        for k, v in keep.items():
            setattr(ef, k, v)
        ef.geometry, ef.printed_masses, ef.mass_budget, mk.FIXED = keep_fn


def candidates() -> list:
    out = []
    for m in MOTORS:
        for p in PROPELLERS:
            for b in BATTERIES:
                if b.cells > m.max_cells:
                    continue
                esc = min((e for e in ESCS if e.max_cells >= b.cells and e.max_a >= m.max_a), key=lambda e: e.mass_g, default=None)
                if esc is None:
                    esc = max((e for e in ESCS if e.max_cells >= b.cells), key=lambda e: e.max_a)
                out.append(Config(m.key, p.key, esc.key, b.key, 1, 1))
    return out


def size(c) -> dict:
    """Overall size [m]: length, the width over the propeller tips, the propeller-axis square."""
    g = ef.geometry(c)
    return {"length": g["length"], "body_d": g["d"], "across_tips": 2 * (g["r_arm"] / math.sqrt(2)) + g["prop_d"], "prop_d": g["prop_d"],
            "frontal_cm2": g["frontal"] * 1e4}


def trade(configs=None) -> list:
    with small():
        rows = []
        for c in configs or candidates():
            r = ef.evaluate(c)
            r.update(size(c))
            rows.append(r)
    return rows


def buyable(r) -> bool:
    """Every part of the build is sold in the form a quadcopter needs (the propeller in both rotations)."""
    return bool(mk.part(r["config"].propeller).pair)


def lightest(rows, kmh: float, pairs_only: bool = True):
    ok = [r for r in rows if r["feasible"] and r["within_rpm_limit"] and r["kmh"] >= kmh and (buyable(r) or not pairs_only)]
    return min(ok, key=lambda r: r["mass_g"], default=None)


def fastest(rows, max_g: float, pairs_only: bool = True):
    ok = [r for r in rows if r["feasible"] and r["within_rpm_limit"] and r["mass_g"] <= max_g and (buyable(r) or not pairs_only)]
    return max(ok, key=lambda r: r["kmh"], default=None)


# ------------------------------------------------------------------------------------------------------------ WASP-1
WASP1 = Config("tm-f1404-4600", "apc41x41e", "geprc-f411-35", "gnb-4s-300", 1, 1)         # under 250 g
WASP1_FAST = Config("tm-f1507-3800", "apc41x41e", "geprc-f411-35", "tattu-r-4s-550", 1, 1)  # the lightest at ER1's speed
VARIANTS = {"WASP-1": WASP1, "WASP-1F": WASP1_FAST}


def card(c) -> dict:
    """One build: the evaluation, the size, the mass by group, the bill of materials (inside ``small``)."""
    with small():
        r = ef.evaluate(c)
        r.update(size(c))
        b = ef.mass_budget(c)
        groups = {}
        for it in b.items:
            groups[it.group] = groups.get(it.group, 0.0) + it.kg * 1e3
        esc = mk.part(c.esc)
        parts = [(4, mk.part(c.motor)), (4, mk.part(c.propeller)), (1, esc), (1, mk.part(c.battery))] + \
                [(1, q) for q in FIXED if not (q.group == "radio" and getattr(esc, "with_rx", False))]
        r.update({"mass_groups_g": {k: round(v, 1) for k, v in groups.items()}, "printed_g": ef.printed_masses(c),
                  "cad": ef.cad_parameters(c), "geometry": ef.geometry(c),
                  "bom": [{"qty": n, "part": q.name, "g_each": q.mass_g, "usd_each": q.price_usd, "kind": q.kind, "url": q.url, "note": q.note} for n, q in parts]})
    return r


def cad_parameters(c) -> dict:
    """``emerillon.Emerillon``'s parameters [mm] for a WASP build: ÉMERILLON's CAD at WASP's size, the small fins,
    skin and tail end."""
    with small():
        q = ef.cad_parameters(c)
    return {**q, "fin_span": SMALL["FIN_SPAN"] * 1e3, "fin_chord": SMALL["FIN_CHORD"] * 1e3, "fin_thickness": SMALL["FIN_T"] * 1e3,
            "wall": SMALL["WALL"] * 1e3, "tail_end_diameter": 7.0, "pad_thickness": 2.5}


def cad(c, **over):
    """The build as Dedalus geometry (``part``: aircraft, arm, nose; ``angle_of_attack_deg`` pitches it for the CFD)."""
    import emerillon
    return emerillon.Emerillon().generate(**cad_parameters(c), **over)


def vehicle(c):
    """What the solver scripts ask of a build: mass, CG and inertia (the budget), the geometry, the top speed and its
    rpm, the static thrust of one motor, the motor unit's mass (inside ``small``)."""
    from types import SimpleNamespace
    with small():
        b = ef.mass_budget(c)
        pr = b.properties()
        top = ef.top_speed(c, b.mass)
        motor, prop = mk.part(c.motor), mk.part(c.propeller)
        return SimpleNamespace(config=c, label=c.label(), mass=b.mass, x_cg=float(pr.com[0]) if hasattr(pr, "com") else float(pr.cg[0]),
                               inertia=[float(pr.inertia[i][i]) for i in range(3)], geo=ef.geometry(c), V_top=float(top["V"]), rpm_top=float(top["rpm"]),
                               thrust_top=float(top["thrust"]), static_thrust=float(ef.drive(c, 0.5)["thrust"]) / 4,
                               motor_unit_kg=(motor.mass_g + prop.mass_g) / 1e3, budget=b)
