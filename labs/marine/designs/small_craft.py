"""The small craft round an offshore platform at night: the **contacts** a simulator shows to the guard carrier
(Esmeralda) - a fishing boat, a pilot boat, a sailing yacht and a skiff. Not robots: plain craft, kept simple but
physically honest, each number a **typical class value with its basis** (``BASIS``). No weapons on any craft.

What is computed, not typed in:

* the **displacement** from the hull's length, beam, draught and block coefficient;
* the **top speed**: for the displacement craft (fishing boat, yacht) the least of the installed power's and the
  hull speed's (a Froude number on the waterline); for the planing craft (pilot boat, skiff) the speed the power
  holds on Savitsky's planing resistance (``barracuda.Barracuda.resistance``: the same method as the interceptor);
* the **speed against the sea state**: the displacement craft lose speed with the wave height over their length
  (class); the planing craft are limited by the power with the added resistance or by the impacts their crew takes
  (Savitsky and Brown at the crew's limit: 1.25 g a pilot boat's seated crew, 1.5 g a skiff's raid crew who accept
  more; assumed);
* the **turning circle** (the least of a lateral-acceleration limit and a number of lengths) and the
  **acceleration** (power over the virtual mass, from rest to cruise).

Radar cross-sections are orders of magnitude by aspect (broadside, bow, stern); the visual and IR notes and the AIS
carriage are what such craft typically have. x forward, y left, z up; the CAD's keel at z = 0, the card's waterline.
"""
from __future__ import annotations

import math

try:
    import cadquery as cq
except ImportError:
    cq = None

RHO_SW = 1025.0
G = 9.81
KN = 0.5144
FT = 0.3048

#: The four craft (class values; the basis of each in ``BASIS``).
CRAFT = {
    "fishing_boat": {
        "what": "a 20 m stern trawler / longliner, steel, wheelhouse forward, gantry aft, gear in the water when fishing",
        "length_m": 20.0, "beam_m": 6.2, "draught_m": 2.6, "depth_m": 3.6, "cb": 0.50, "deadrise_deg": 12.0,
        "planing": False, "power_kW": 300.0, "eta": 0.55, "cruise_kn": 8.0, "fishing_kn": 3.0, "froude_max": 0.37,
        "turn_lengths": 3.5, "lateral_g": 0.10, "loss_per_hs": 0.30,
        "rcs_m2": {"beam": 300.0, "bow": 60.0, "stern": 120.0},
        "lights": "trawling: green over white all-round, sidelights, sternlight, deck floodlights; masthead when steaming",
        "ir": "warm exhaust stack and wheelhouse, a large hull: a big IR target (~6 K over the sea)", "delta_T_K": 6.0,
        "ais": "class A (EU: fishing vessels >= 15 m must carry it)", "ais_typical": True, "crew": [3, 8],
        "behaviour": "slow, long dwell on the grounds, gear astern (it cannot manoeuvre freely while trawling)"},
    "pilot_boat": {
        "what": "a 16 m pilot boat, aluminium, twin diesels on propellers, a wheelhouse amidships, fendered all round",
        "length_m": 16.0, "beam_m": 5.0, "draught_m": 1.4, "canoe_draught_m": 0.95, "depth_m": 2.4, "cb": 0.36, "deadrise_deg": 18.0, "chine_beam_m": 4.2,
        "planing": True, "power_kW": 2 * 450.0, "eta": 0.60, "cruise_kn": 20.0, "lcg_frac": 0.38, "crew_g": 1.25,
        "turn_lengths": 3.0, "lateral_g": 0.35,
        "rcs_m2": {"beam": 30.0, "bow": 8.0, "stern": 12.0},
        "lights": "white over red all-round (pilot on duty), masthead, sidelights, sternlight; 'PILOT' on the wheelhouse",
        "ir": "two diesel exhausts at the transom, warm wheelhouse (~5 K over the sea)", "delta_T_K": 5.0,
        "ais": "class A, always on", "ais_typical": True, "crew": [2, 3],
        "behaviour": "fast, direct between the pilot station and the ship it serves; talks on VHF"},
    "yacht": {
        "what": "a 12 m sailing yacht (sloop, fin keel), GRP, a 40 kW auxiliary diesel",
        "length_m": 12.0, "beam_m": 3.9, "draught_m": 2.0, "canoe_draught_m": 0.7, "depth_m": 1.7, "cb": 0.38, "deadrise_deg": 10.0,
        "planing": False, "power_kW": 40.0, "eta": 0.50, "cruise_kn": 6.0, "sail_kn": [4.0, 8.0], "froude_max": 0.40,
        "turn_lengths": 3.0, "lateral_g": 0.08, "loss_per_hs": 0.30,
        "rcs_m2": {"beam": 3.0, "bow": 1.0, "stern": 1.5, "with_reflector": 10.0},
        "lights": "tricolour at the masthead (or sidelights and sternlight); often dim",
        "ir": "a cold hull under sail; the diesel's exhaust only under engine (~2 K over the sea)", "delta_T_K": 2.0,
        "ais": "class B, or none", "ais_typical": False, "crew": [1, 6],
        "behaviour": "slow, its course set by the wind; often without AIS; may not answer VHF"},
    "skiff": {
        "what": "a 7 m open fibreglass skiff, two outboards, low freeboard, no superstructure",
        "length_m": 7.0, "beam_m": 2.0, "draught_m": 0.35, "depth_m": 0.9, "cb": 0.40, "deadrise_deg": 16.0, "chine_beam_m": 1.7,
        "planing": True, "power_kW": 2 * 75.0, "eta": 0.55, "cruise_kn": 25.0, "lcg_frac": 0.36, "crew_g": 1.5,
        "load_kg": {"hull": 650.0, "2 outboards": 2 * 160.0, "fuel 200 L": 170.0, "crew of 6 + kit": 6 * 85.0},
        "light_load_kg": {"hull": 650.0, "2 outboards": 2 * 160.0, "fuel 100 L": 85.0, "crew of 2": 2 * 85.0},
        "turn_lengths": 2.5, "lateral_g": 0.45,
        "rcs_m2": {"beam": 1.0, "bow": 0.3, "stern": 0.5},
        "lights": "often none", "ir": "the outboards' cowls and exhausts, the crew (~4 K over the sea); the hull is cold", "delta_T_K": 4.0,
        "ais": "none", "ais_typical": False, "crew": [2, 8],
        "behaviour": "fast and low, often no lights and no AIS; drifts or idles, then runs"},
}
#: Where each class value comes from (no vendor's craft; typical of the type).
BASIS = {
    "fishing_boat": "dimensions, power and speeds typical of 18-24 m North Sea trawlers and longliners; RCS of a steel "
                    "craft that size (tens to hundreds of m2); AIS per EU Regulation 2002/59 as amended (fishing >= 15 m)",
    "pilot_boat": "15-17 m aluminium pilot boats with twin 400-600 kW diesels, 20-25 kn; RCS of an aluminium "
                  "superstructure; AIS class A by duty",
    "yacht": "11-13 m cruising sloops with a 30-40 kW auxiliary; hull speed ~1.34 sqrt(LWL ft); RCS of a GRP hull with "
             "an aluminium mast (a radar reflector raises it to ~10 m2)",
    "skiff": "6-8 m open fibreglass skiffs with one or two 40-100 kW outboards, 25-40 kn light; RCS of a GRP hull with "
             "outboards and people (~1 m2); no lights, no AIS (as reported of raiding skiffs)",
}


def _barracuda_p(c: dict) -> dict:
    """The craft's hull in the parameters ``barracuda.Barracuda``'s planing methods read (no collar)."""
    from barracuda import Barracuda
    return Barracuda.p() | {"length": c["length_m"] * 1000 * 0.92 / 0.9, "beam": c["beam_m"] * 1000, "chine_beam": c["chine_beam_m"] * 1000,
                            "deadrise_deg": c["deadrise_deg"], "lcg_frac": c["lcg_frac"], "collar_d": 0.0, "service_margin": 1.15}


def displacement_t(c: dict) -> float:
    if "load_kg" in c:
        return sum(c["load_kg"].values()) / 1000
    T = c.get("canoe_draught_m", c["draught_m"])
    return 0.95 * c["length_m"] * c["beam_m"] * T * c["cb"] * RHO_SW / 1000


def sizing(name: str) -> dict:
    c = CRAFT[name]
    disp = displacement_t(c)
    m = disp * 1000
    lwl = 0.92 * c["length_m"]
    if c["planing"]:
        from barracuda import Barracuda
        bp = _barracuda_p(c)

        def p_needed(v, hs=0.0):
            return Barracuda.resistance(bp, m, v, hs) * v * KN / c["eta"] / 1000

        def v_power(hs=0.0):
            lo, hi = 0.0, 70.0
            for _ in range(50):
                v = (lo + hi) / 2
                lo, hi = (v, hi) if p_needed(v, hs) < 0.95 * c["power_kW"] else (lo, v)
            return lo
        v_max = v_power()
        if "light_load_kg" in c:
            m_full, m = m, sum(c["light_load_kg"].values())
            v_light = v_power()
            m = m_full
        table = []
        for hs in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5):
            vp = v_power(hs)
            vs = Barracuda.slam_speed(bp, m, hs, c["crew_g"]) if hs else 99.0
            table.append({"hs_m": hs, "kn": round(min(vp, vs), 1), "limited_by": "propulsion" if vp <= vs else "slamming (crew)"})
        trim = Barracuda.savitsky(bp, m, v_max * KN)
    else:
        v_hull = c["froude_max"] * math.sqrt(G * lwl) / KN
        Ct = 0.0045                                                      # total resistance coefficient below hull speed (class)
        wet = lwl * (c["beam_m"] + 2 * c.get("canoe_draught_m", c["draught_m"])) * 0.8
        v_pw = (2 * c["power_kW"] * 1000 * c["eta"] / (RHO_SW * wet * Ct)) ** (1 / 3) / KN
        v_max = min(v_hull, v_pw)
        table = [{"hs_m": hs, "kn": round(v_max * max(0.4, 1 - c["loss_per_hs"] * hs / math.sqrt(c["length_m"])), 1),
                  "limited_by": "hull speed" if v_hull <= v_pw else "power"} for hs in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)]
        trim = None
    cruise = min(c["cruise_kn"], v_max)
    turn = {}
    for kn in sorted({5.0, round(cruise), round(v_max)}):
        v = kn * KN
        turn[f"{kn:.0f}kn"] = round(max(c["turn_lengths"] * c["length_m"], 2 * v * v / (c["lateral_g"] * G)))
    thrust0 = c["power_kW"] * 1000 * c["eta"] / max(0.3 * cruise * KN, 1.0)        # thrust at low speed (class)
    a0 = min(thrust0 / (1.1 * m), 0.35 * G)
    t_cruise = cruise * KN / (0.5 * a0)                                              # thrust falls as the speed rises (half on average)
    return {"name": name, "what": c["what"], "length_m": c["length_m"], "beam_m": c["beam_m"], "draught_m": c["draught_m"],
            "displacement_t": round(disp, 2), "power_kW": c["power_kW"], "max_kn": round(v_max, 1), "cruise_kn": round(cruise, 1),
            "speed_vs_hs": table, **({"max_kn_light": round(v_light, 1)} if c["planing"] and "light_load_kg" in c else {}), "turning_circle_m": turn, "accel_0_cruise_s": round(t_cruise, 1),
            "trim_at_max_deg": round(trim["trim_deg"], 1) if trim else None,
            "rcs_m2": c["rcs_m2"], "lights": c["lights"], "ir": c["ir"], "delta_T_K": c["delta_T_K"], "ais": c["ais"], "ais_typical": c["ais_typical"],
            "crew": c["crew"], "behaviour": c["behaviour"], "basis": BASIS[name],
            **({k: c[k] for k in ("fishing_kn", "sail_kn") if k in c})}


# ----------------------------------------------------------------------------------------------------------- CAD
def _hull(c):
    from survey_boat import SurveyBoat
    hp = {q.name: q.default for q in SurveyBoat.parameters} | {
        "length": c["length_m"] * 1000, "beam": c["beam_m"] * 1000, "depth": c["depth_m"] * 1000, "deadrise_deg": c["deadrise_deg"],
        "bow_rise": c["depth_m"] * 250, "transom_beam_ratio": 0.85 if c["planing"] else 0.7, "stations": 10}
    return SurveyBoat()._hull_solid(hp)


def _box(x0, x1, y, z0, z1):
    return cq.Workplane("XY").box(x1 - x0, 2 * y, z1 - z0).translate(((x0 + x1) / 2, 0, (z0 + z1) / 2))


def superstructure(name: str):
    """The craft's superstructure, mast and machinery above the deck (mm, the hull's frame: transom x = 0, keel z = 0)."""
    c = CRAFT[name]
    L, B, D = c["length_m"] * 1000, c["beam_m"] * 1000, c["depth_m"] * 1000
    if name == "fishing_boat":
        house = _box(L * 0.62, L * 0.85, B * 0.32, D, D + 2600)
        mast = cq.Workplane("XY").add(cq.Solid.makeCylinder(120, 5500, cq.Vector(L * 0.72, 0, D + 2600)))
        gantry = _box(300, 700, B * 0.45, D, D + 4200).cut(_box(250, 750, B * 0.38, D, D + 3700))
        drum = cq.Workplane("XY").add(cq.Solid.makeCylinder(700, B * 0.6, cq.Vector(L * 0.25, -B * 0.3, D + 700), cq.Vector(0, 1, 0)))
        return house.union(mast).union(gantry).union(drum)
    if name == "pilot_boat":
        house = _box(L * 0.40, L * 0.72, B * 0.38, D, D + 2200)
        mast = cq.Workplane("XY").add(cq.Solid.makeCylinder(80, 3000, cq.Vector(L * 0.55, 0, D + 2200)))
        return house.union(mast)
    if name == "yacht":
        house = _box(L * 0.35, L * 0.62, B * 0.3, D, D + 700)
        mast = cq.Workplane("XY").add(cq.Solid.makeCylinder(90, 16000, cq.Vector(L * 0.6, 0, D)))
        keel = _box(L * 0.45, L * 0.6, 60, -(c["draught_m"] - c["canoe_draught_m"]) * 1000, 100)
        return house.union(mast).union(keel)
    if name == "skiff":
        console = _box(L * 0.45, L * 0.55, 300, D - 300, D + 600)
        outs = None
        for y in (350, -350):
            o = _box(-450, 50, 170, D - 900, D + 400).union(_box(-300, -100, 60, D - 1300, D - 900)).translate((0, y, 0))
            outs = o if outs is None else outs.union(o)
        return console.union(outs)
    raise KeyError(name)


def build(name: str, part: str = "craft"):
    if cq is None:
        raise ImportError("the craft's CAD needs CadQuery")
    c = CRAFT[name]
    if part == "hull":
        return _hull(c)
    if part == "superstructure":
        return superstructure(name)
    return _hull(c).union(superstructure(name))


if __name__ == "__main__":
    import json
    for n in CRAFT:
        s = sizing(n)
        print(json.dumps({k: v for k, v in s.items() if k not in ("what", "basis", "behaviour", "lights", "ir")}))
