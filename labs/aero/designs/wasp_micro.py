"""WASP-M — a WASP smaller still, to carry a micro-load at a quadcopter's speed.

WASP-1 (``wasp.py``) is ÉMERILLON at 225 g, built for speed alone. WASP-M is the same bullet-with-four-motors a size
down, with a **payload bay** behind the nose: a few grams (a sensor tag, a sample vial, a memory card, a dose) carried
fast and far for its size. "A quadcopter's speed" is taken as 100 km/h in cruise and at least 120 km/h flat out — what a
5-inch freestyle quadcopter of ten times the mass does.

The method is ``wasp``'s: ÉMERILLON's model over a catalogue of parts **shops list**, here the smallest classes —
1103 to 1404 motors, 2.5 and 3 inch propellers sold in both rotations, 2S to 4S packs of 300 to 450 mAh, an
all-in-one board with the receiver on it. ``micro(payload_g)`` is ``wasp.small()`` with the printed airframe's
constants one step smaller (a 0.6 mm skin, a 4 mm arm), the blade section at a Reynolds number of about 3e4, the
payload in the mass budget and its bay in the body's length (the payload at 1 g/cm3, HAND).

What decides a build here is not the record run but the delivery: ``cruise`` gives the electrical power in level
flight at a speed (the throttle that balances the drag and the weight) and the range one charge gives with 20 %
kept and half a minute of hover at each end.
"""
from __future__ import annotations

import contextlib
import math

import emerillon_flight as ef
import emerillon_market as mk
import wasp
from wasp import Config, _p
from ogma import mass as og_mass

MOTORS = [
    _p("hm-ex1103-8000", "Happymodel EX1103 8000KV", "motor", 11.99, 3.7, "https://rcdrone.top/products/happymodel-ex1103-brushless-motor", "partly",
       "3.7 g, 2-3S, 13.5 x 16 mm from the listings; resistance, idle and peak current assumed",
       kv=8000.0, r_ohm=0.42, i0_a=0.35, max_a=5.0, max_cells=3, d_mm=13.5, h_mm=12.0),
    _p("tm-f1204-5000", "T-Motor F1204 5000KV", "motor", 15.90, 6.9, "https://www.getfpv.com/t-motor-f1204-5000kv-6500kv-motor.html", "sourced",
       "6.9 g with cable, 240 mOhm, 6 A peak (60 s), 72 W, 2-3S, 15.5 x 15.6 mm (T-Motor's table); idle current assumed",
       kv=5000.0, r_ohm=0.240, i0_a=0.35, max_a=6.0, max_cells=3, d_mm=15.5, h_mm=12.0),
    _p("tm-f1204-6500", "T-Motor F1204 6500KV", "motor", 15.90, 6.9, "https://www.racedayquads.com/products/t-motor-f1204-6500kv-motor", "sourced",
       "6.9 g with cable, 140 mOhm, 7 A peak (60 s), 60 W, 2-3S; idle current assumed",
       kv=6500.0, r_ohm=0.140, i0_a=0.45, max_a=7.0, max_cells=3, d_mm=15.5, h_mm=12.0),
    _p("tm-f1303-5000", "T-Motor F1303 5000KV", "motor", 15.90, 6.1, "https://www.getfpv.com/tmotor-f1303-5000kv-brushless-motor-1pc.html", "sourced",
       "6.1 g with wires, 280 mOhm, 0.32 A idle, 7.45 A peak (60 s), 87 W, 2-3S, 16.5 x 14.55 mm",
       kv=5000.0, r_ohm=0.280, i0_a=0.35, max_a=7.45, max_cells=3, d_mm=16.5, h_mm=12.5),
    mk.part("tm-f1404-4600"),
]
# quadcopter propellers, sold as clockwise + counter-clockwise sets. The makers publish no speed limit: 190,000 / D is
# assumed (2.5 inch: 76,000 rpm), as for the other small propellers here
_Q = dict(rpm_limit_k=190000.0, tc_tip=0.08, pair=True)
PROPELLERS = [
    _p("gf-3018-2", "Gemfan Hurricane 3018 two-blade (CW + CCW)", "propeller", 0.50, 0.85, "https://www.getfpv.com/gemfan-hurricane-3018-durable-2-blade-propeller-1-5-2mm-shaft-set-of-8.html", "partly",
       "0.85 g, sold in sets of 8; the speed limit and blade planform assumed", d_in=3.0, p_in=1.8, blades=2, chord=(0.08, 0.11, 0.05), **_Q),
    _p("gf-2520-3", "Gemfan Hurricane 2520 three-blade (CW + CCW)", "propeller", 0.45, 1.0, "https://pyrodrone.com/products/gemfan-hurricane-2-5-2520-3-pc-durable-tri-blade-prop-1-5mm-shaft-4ccw-4cw-choose-color", "partly",
       "1.0 g; the speed limit and blade planform assumed", d_in=2.5, p_in=2.0, blades=3, chord=(0.10, 0.14, 0.07), **_Q),
    _p("gf-2540-3", "Gemfan Flash 2540 three-blade (CW + CCW)", "propeller", 0.45, 1.0, "https://www.getfpv.com/gemfan-flash-2540-durable-3-blade-white-set-of-8.html", "partly",
       "1 g, sets of 8; the speed limit and blade planform assumed", d_in=2.5, p_in=4.0, blades=3, chord=(0.10, 0.14, 0.07), **_Q),
]
ESCS = [
    mk.part("geprc-taker-f411-12"),
    _p("geprc-taker-f411-12-2s", "GEPRC TAKER F411 12A AIO (ELRS built in), 1-2S", "esc", 49.99, 4.2, "https://pyrodrone.com/products/geprc-taker-f411-aio-flight-controller-built-in-elrs-2-4g-receiver-and-1-2s-12a-esc-25x25mm", "partly",
       "4.2 g, 12 A, 1-2S, the receiver on board; price assumed", max_a=12.0, max_cells=2, count=1, with_fc=True, with_rx=True),
]
BATTERIES = [
    _p("gnb-2s-300", "GNB 2S 300 mAh 80C HV", "battery", 7.99, 19.0, "https://racedayquads.com/products/gaoneng-gnb-3-8v-2s-300mah-80c-lihv-whoop-micro-battery-w-cabled-connector-xt30", "partly",
       "19 g (+-2), 52 x 17 x 12 mm; price assumed", cells=2, ah=0.30, c_rating=80.0, size_mm=(52.0, 17.0, 12.0)),
    _p("gnb-3s-300", "GNB 3S 300 mAh 80C HV", "battery", 9.99, 28.0, "https://www.racedayquads.com/products/1x-gaoneng-gnb-11-4v-3s-300mah-80c-lihv-micro-battery-xt30", "partly",
       "28 g (+-2), 53 x 17 x 18 mm; price assumed", cells=3, ah=0.30, c_rating=80.0, size_mm=(53.0, 18.0, 17.0)),
    _p("gnb-3s-450", "GNB 3S 450 mAh 80C HV", "battery", 11.99, 42.0, "https://pyrodrone.com/collections/0-500mah/products/gnb-3s-11-4v-hv-450mah-lipo-battery-80c-xt30-lihv", "partly",
       "42 g (+-2), 66 x 18 x 21 mm (one listing); price assumed", cells=3, ah=0.45, c_rating=80.0, size_mm=(66.0, 21.0, 18.0)),
    mk.part("gnb-4s-300"),
]
FIXED = [q for q in wasp.FIXED if q.group != "hardware"] + [mk.Part("hw-m", "wires, XT30, screws, antenna", "hardware", 8.0, 5.0, kind="assumed", note="an allowance")]
mk.CATALOGUE.update({p.key: p for p in MOTORS + PROPELLERS + ESCS + BATTERIES + FIXED})

CRUISE_KMH, TOP_KMH_MIN = 100.0, 120.0
MICRO = dict(WALL=0.0006, FIN_SPAN=0.020, FIN_CHORD=0.026, FIN_T=0.0015,
             SECTION=dict(cl_alpha=2 * math.pi * 0.85, alpha0_deg=-2.0, cl_max=0.9, cd0=0.026, k=0.045))   # blade at Re ~ 3e4 (HAND)
BAY_D = 0.024                 # the payload bay's bore [m]
HOVER_END_S = 30.0            # hover at each end of a delivery


@contextlib.contextmanager
def micro(payload_g: float = 10.0, constants: bool = True):
    """``wasp.small()`` one size down, with ``payload_g`` aboard and its bay in the body. ``constants`` False keeps
    ``wasp``'s own airframe constants and fixed parts (a WASP-1 build carrying a payload: ``wasp_family``)."""
    bay = payload_g * 1e-6 / (0.7 * math.pi * BAY_D ** 2 / 4)                        # 1 g/cm3, 70 % of the bore used
    keep = dict(SMALL=dict(wasp.SMALL), ARM_T_MIN=wasp.ARM_T_MIN, POD_WALL=wasp.POD_WALL, BUMPERS_G=wasp.BUMPERS_G, WIRING_G=wasp.WIRING_G,
                FIXED=wasp.FIXED, mass_budget=wasp.mass_budget)
    base_budget = wasp.mass_budget

    def mass_budget(c):
        b = base_budget(c)
        if payload_g:
            g = wasp.geometry(c)
            b.add(og_mass.Item("payload", payload_g / 1e3, (g["nose"] + 0.5 * bay, 0, 0), group="payload"))
        return b

    if constants:
        wasp.SMALL.update(MICRO, STACK_BAY=0.028 + bay)
        wasp.ARM_T_MIN, wasp.POD_WALL, wasp.BUMPERS_G, wasp.WIRING_G, wasp.FIXED = 0.004, 0.0008, 1.0, 4.0, FIXED
    else:
        wasp.SMALL.update(STACK_BAY=wasp.SMALL["STACK_BAY"] + bay)
    wasp.mass_budget = mass_budget
    try:
        with wasp.small():
            yield
    finally:
        wasp.SMALL.clear(); wasp.SMALL.update(keep["SMALL"])
        wasp.ARM_T_MIN, wasp.POD_WALL, wasp.BUMPERS_G, wasp.WIRING_G = keep["ARM_T_MIN"], keep["POD_WALL"], keep["BUMPERS_G"], keep["WIRING_G"]
        wasp.FIXED, wasp.mass_budget = keep["FIXED"], keep["mass_budget"]


def cruise(c, V: float, mass: float) -> dict | None:
    """Level flight at ``V``: the throttle at which the thrust is what the drag and the weight ask, and the power.
    None when full throttle does not reach it. Call inside ``micro``."""
    need = lambda u: ef.drive(c, V, throttle=u)
    q = 0.5 * ef.RHO * V * V

    def excess(u):
        d = need(u)
        db = ef.drag_buildup(c, V, slip=min(ef._slip(d), 4.0))
        return d["thrust"] - ef.trim(c, V, mass, db["CdA"], ef.RHO)[1], d
    if excess(1.0)[0] < 0:
        return None
    lo, hi = 0.05, 1.0
    for _ in range(24):
        u = 0.5 * (lo + hi)
        lo, hi = (lo, u) if excess(u)[0] > 0 else (u, hi)
    e, d = excess(hi)
    return {"V": V, "throttle": hi, "electric_w": d["electric_w"], "thrust": d["thrust"], "rpm": d["rpm"], "current": d["current"]}


def delivery(c, payload_g: float, kmh: float = CRUISE_KMH, constants: bool = True) -> dict:
    """One build with a payload: the evaluation, the cruise point and the range (out, or out and back)."""
    with micro(payload_g, constants):
        r = ef.evaluate(c)
        r.update(wasp.size(c))
        m = r["mass_g"] / 1e3
        bat = mk.part(c.battery)
        wh = bat.cells * 3.7 * bat.ah
        cr = cruise(c, kmh / 3.6, m)
        static = ef.drive(c, 0.5)
        hover_w = static["electric_w"] * (m * ef.G / max(static["thrust"], 1e-6)) ** 1.5
        usable = ef.USABLE * wh - 2 * HOVER_END_S * hover_w / 3600.0
        b = ef.mass_budget(c)
        groups = {}
        for it in b.items:
            groups[it.group] = round(groups.get(it.group, 0.0) + it.kg * 1e3, 1)
    fails = [f for f in r["fails"] if f != "energy"]                       # the record run's two passes are not this machine's task
    if cr is None:
        fails.append("cruise speed")
    elif usable <= 0:
        fails.append("hover energy")
    r.update({"payload_g": payload_g, "fails": fails, "feasible": not fails, "wh": wh, "hover_w": hover_w, "mass_groups_g": groups,
              "cruise": cr, "cruise_kmh": kmh, "range_km": None if cr is None else max(usable, 0.0) / cr["electric_w"] * kmh,
              "minutes": None if cr is None else 60 * max(usable, 0.0) / cr["electric_w"]})
    return r


def candidates() -> list:
    out = []
    for m in MOTORS:
        for p in PROPELLERS:
            for b in BATTERIES:
                if b.cells > m.max_cells:
                    continue
                esc = min((e for e in ESCS if e.max_cells >= b.cells), key=lambda e: e.mass_g)
                out.append(Config(m.key, p.key, esc.key, b.key, 1, 1))
    return out


def trade(payload_g: float = 10.0) -> list:
    return [delivery(c, payload_g) for c in candidates()]


def pick(rows, top_min: float = TOP_KMH_MIN):
    """The lightest feasible build inside the propeller's speed limit that reaches ``top_min`` flat out."""
    ok = [r for r in rows if r["feasible"] and r["within_rpm_limit"] and r["kmh"] >= top_min]
    return min(ok, key=lambda r: r["mass_g"], default=None)


# ----------------------------------------------------------------------------------------------------------- WASP-M
WASP_M = Config("tm-f1204-6500", "gf-2540-3", "geprc-taker-f411-12-2s", "gnb-2s-300", 1, 1)       # the small one
WASP_M2 = Config("tm-f1204-6500", "gf-2540-3", "geprc-taker-f411-12", "gnb-3s-450", 1, 1)         # Cotesia (first WASP-ML, then WASP-M2): the one that goes furthest, the kept build
WASP_M_LONG = WASP_M2                                                                             # its first name (Cotesia)
VARIANTS = {"WASP-M": WASP_M, "Cotesia": WASP_M2}
PAYLOAD_G = 10.0


def cad_parameters(c, payload_g: float = PAYLOAD_G) -> dict:
    """``emerillon.Emerillon``'s parameters [mm] for a build at this size (true size: ``cad`` builds at twice it)."""
    with micro(payload_g):
        q = ef.cad_parameters(c)
    return {**q, "fin_span": MICRO["FIN_SPAN"] * 1e3, "fin_chord": MICRO["FIN_CHORD"] * 1e3, "fin_thickness": MICRO["FIN_T"] * 1e3,
            "wall": MICRO["WALL"] * 1e3, "tail_end_diameter": 5.0, "pad_thickness": 2.0, "prop_gap": 6.0}


def cad(c, payload_g: float = PAYLOAD_G, **over):
    """The build as Dedalus geometry: ÉMERILLON's CAD at this size. Its parameters have minimums above the smallest
    WASP's sizes (a 30 mm body), so it is built at twice the size and scaled by one half."""
    import emerillon
    from vegeta import dedalus
    q = cad_parameters(c, payload_g)
    big = emerillon.Emerillon().generate(**{k: 2.0 * v for k, v in q.items()}, **over)
    return dedalus.Geometry.from_cadquery(big.shape.scale(0.5), name="wasp_m")
