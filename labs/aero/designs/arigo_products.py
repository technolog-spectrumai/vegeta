"""ARIGO product lines (notebook 39, the product round). ARIGO-1 hybrid has graduated to **ARIGO-1H**, the family's
baseline airframe. Four lines start from the ARIGO family; their names are mission identities, not airframes. Each is
scored on its own KPIs (``reports/arigo_catalog/KPIS.md``, mirrored in ``sims/arigo_charts.MISSIONS`` and ``GATES``):

- **BUTEO** (buzzard, a watchful perch hunter) — point defence. A site-based platform ready at a fixed perch or launch station. Priorities: readiness,
  weather tolerance, reliable deployment, straightforward recovery.
- **ARPELLA** (harrier, sustained patrol over open terrain) — area defence. A patrol platform with a required mission radius of up to 10 km. Priorities: endurance,
  efficient coverage, dependable communications, return reserves.
- **ASTOR** (goshawk, a powerful, agile pursuit hunter) — fast response. Arrival time before patrol time. Priorities: quick deployment, a reliable prepared
  launch, rapid turnaround, predictable recovery.
- **SPARFELL** — the portable field platform (Breton for sparrowhawk). Backpack-carried and hand-launched by one
  person. Priorities: low mass, compact packing, quick assembly, minimal equipment, rugged handling.

A **spec** is an airframe (ARIGO-1H, scaled or with another span or chord, or ARIGO-20) plus a pack, a motor and
kits (how it is launched, how it comes back, what it carries). ``PRODUCTS[...]["base"]`` is each product's starting
spec; ``explore(product)`` the design space around it that ``sims/arigo_explore.sh`` sweeps. Airframes other than
ARIGO-1H and ARIGO-20 have no CFD: they are compared on the lattice model, with ARIGO-1H on the lattice beside them
(``ARIGO-1H lattice``) so the comparison is like with like; the winners go to CFD next.

Kit masses and times are HAND estimates (marked).
"""
from __future__ import annotations

import itertools
import math

import arigo1_model as am

# ------------------------------------------------------------------------------------------------ kits (HAND)
# name: (mass items [(item, g, source)], features for the mission metrics)
KITS = {
    "perch": ([("perch hook + spring push-off latch", 25.0, "HAND (+-8)")], {"launch": "perch", "perch_push_m_s": 8.0}),
    "charging_perch": ([("charging foot pads + ideal-diode bridge", 15.0, "HAND (+-5)")], {"charging": True}),
    "rail": ([("rail shoe + launch hook", 10.0, "HAND (+-4)")], {"launch": "rail"}),
    "parachute": ([("parachute + canopy bay + release servo", 85.0, "HAND (+-20)")], {"recovery": "parachute"}),
    "two_piece": ([("two-piece wing joiner + plugs", 30.0, "HAND (+-10)")], {"two_piece": True, "assembly_s": 60.0}),
    "quick_joiner": ([("magnetic quick-release wing joiner, blind connector", 40.0, "HAND (+-10)")],
                     {"two_piece": True, "assembly_s": 30.0}),
    "lite_avionics": ([("lite avionics: 60 g vision module (not 125), 20 g camera, lighter wiring", -90.0,
                        "HAND (+-25); less vision compute: a shorter detection range and slower tracking")],
                      {"lite_avionics": True}),
    "turning_station": ([], {"turning_station": True}),      # ground equipment: the perch / rail weathervanes into the wind
    "rugged_skin": ([("EPP belly and leading-edge armour", 45.0, "HAND (+-15)")], {"rugged": True}),
    "relay_radio": ([("long-range radio + diversity antennas", 25.0, "HAND (+-8)")], {"relay_radio": True}),
}
PARACHUTE_SINK_M_S = 5.0           # HAND: descent under a ~0.8 m2 canopy
PARACHUTE_MISS_M = 15.0            # HAND: the drift left after aiming upwind in a 6 m/s wind
TURNAROUND_S = {"belly": 120.0, "parachute": 360.0}     # HAND: pack swap + check (+ canopy repack)
ASSEMBLY_S = 20.0                  # HAND: one-piece: the pack in, power up, self-test
MOTOR_G_PER_W = 0.12               # HAND: motor + ESC mass per watt of rating around 800 W (2814-class ~155 g)

PRODUCTS = {
    "buteo": {"title": "BUTEO", "mission": "buteo", "role": "Point defence",
             "what": "site-based platform ready at a fixed perch or launch station",
             "priorities": "readiness, weather tolerance, reliable deployment, straightforward recovery",
             "base": {"kits": ("perch", "charging_perch")}},
    "arpella": {"title": "ARPELLA", "mission": "arpella", "role": "Area defence",
                "what": "patrol platform with a required mission radius of up to 10 km",
                "priorities": "endurance, efficient coverage, dependable communications, return reserves",
                "base": {"kits": ("relay_radio",), "launch": "hand"}},
    "astor": {"title": "ASTOR", "mission": "astor", "role": "Fast response",
              "what": "rapid-response platform: arrival time before patrol time",
              "priorities": "quick deployment, reliable prepared launch, rapid turnaround, predictable recovery",
              "base": {"kits": ("rail", "parachute")}},
    "sparfell": {"title": "SPARFELL", "mission": "sparfell", "role": "Portable field platform",
                 "what": "backpack-carried, hand-launched by one person (Breton: sparrowhawk)",
                 "priorities": "low mass, compact packing, quick assembly, minimal equipment, rugged handling",
                 "base": {"kits": ("two_piece",), "pack": "Li-ion 3S1P P42A", "launch": "hand"}},
}

# the design space around each product's base (sims/arigo_explore.sh sweeps it)
SPACE = {
    "buteo": {"span": (950.0, 1050.0, 1150.0), "chord": (280.0, 300.0, 330.0),
             "kits": [("perch", "charging_perch"), ("perch", "charging_perch", "turning_station"),
                      ("perch", "charging_perch", "turning_station", "parachute"), ("rail", "charging_perch", "turning_station", "parachute")]},
    "arpella": {"span": (950.0, 1100.0, 1250.0), "chord": (260.0, 300.0),
                "pack": ("Li-ion 3S2P P42A", "Li-ion 3S3P P42A", "Li-ion 3S4P P42A")},
    "astor": {"span": (800.0, 950.0), "chord": (260.0, 300.0), "motor_w": (800.0, 1200.0),
              "pack": ("Li-ion 3S2P P42A", "Li-ion 3S3P P42A"),
              "kits": [("rail", "parachute"), ("rail", "turning_station", "parachute")]},
    "sparfell": {"scale": (0.7, 0.8, 0.9, 1.0), "pack": ("Li-ion 3S1P P42A", "Li-ion 3S2P P42A"),
                 "kits": [("two_piece",), ("quick_joiner", "lite_avionics"), ("quick_joiner", "lite_avionics", "rugged_skin")]},
}


# ------------------------------------------------------------------------------------------------ airframes
def airframe(scale: float = 1.0, span: float | None = None, chord: float | None = None) -> str:
    """ARIGO-1H's geometry, scaled (lengths x ``scale``; the bay's diameter not below 62 mm: the avionics), with
    another span or root chord; the tail grown with the wing (tail volume kept: linear dims x sqrt(S c / S0 c0),
    HAND); the wing's quarter chord kept; the CG by the static margin (12 % MAC), as on the lattice baseline.
    Registered in ``arigo1.VARIANTS``; returns its name."""
    import arigo1 as a1
    name = f"1H_s{scale:.2f}_b{span or 0:.0f}_c{chord or 0:.0f}"
    if name in a1.VARIANTS:
        return name
    d = a1.Arigo1().resolve(**a1.VARIANTS["hybrid"])
    L = ("span", "root_chord", "wing_x", "fuselage_length", "nose_length", "tail_cone", "htail_span", "htail_root_chord",
         "fin_height", "fin_root_chord", "prop_diameter", "camera_diameter", "wing_z")
    q = {k: d[k] * scale for k in L}
    q["fuselage_diameter"] = max(62.0, d["fuselage_diameter"] * scale)
    q["tail_end_diameter"] = max(22.0, d["tail_end_diameter"] * scale)
    b0, c0 = q["span"], q["root_chord"]
    if span:
        q["span"] = span
    if chord:
        q["wing_x"] += (c0 - chord) / 4.0                          # the quarter chord stays put
        q["root_chord"] = chord
    k = math.sqrt((q["span"] * q["root_chord"] ** 2) / (b0 * c0 ** 2))   # S c ~ b c^2 for a fixed taper
    for t in ("htail_span", "htail_root_chord", "fin_height", "fin_root_chord"):
        q[t] *= k
    for t in ("taper", "le_sweep_deg", "dihedral_deg", "washout_deg"):
        q[t] = d[t]
    q["cg_x"] = 0.0
    a1.VARIANTS[name] = q
    return name


def lattice_baseline() -> str:
    """ARIGO-1H on the lattice (no CFD; CG by the static margin): the like-for-like reference for the sweeps."""
    return airframe(1.0)


# ------------------------------------------------------------------------------------------------ specs
def spec_key(product: str, s: dict) -> tuple[str, str]:
    """(run key, display name) of a spec. Keys start with the airframe family (the run files filter on it)."""
    fr = s.get("frame", "1H")
    parts, nm = [], []
    if fr == "arigo-20":
        fam, nm = "arigo-20", ["ARIGO-20"]
    else:
        fam = "arigo-1"
        sc, b, c = s.get("scale", 1.0), s.get("span"), s.get("chord")
        nm = ["ARIGO-1H" + (" lattice" if s.get("lattice") else "")]
        if sc != 1.0:
            nm.append(f"x{sc:.2f}"); parts.append(f"s{sc:.2f}")
        if b:
            nm.append(f"b{b:.0f}"); parts.append(f"b{b:.0f}")
        if c:
            nm.append(f"c{c:.0f}"); parts.append(f"c{c:.0f}")
    if s.get("pack"):
        p = s["pack"].replace("Li-ion ", "").replace(" P42A", "").replace(" mAh LiPo", "")
        nm.append(p); parts.append(p.replace(" ", ""))
    if s.get("motor_w"):
        nm.append(f"{s['motor_w']:.0f} W"); parts.append(f"m{s['motor_w']:.0f}")
    kits = s.get("kits", ())
    if kits:
        nm.append("+" + "+".join(kits)); parts.append("-".join(k[:5] for k in kits))
    key = f"{fam}_{product}" + ("_" + "_".join(parts) if parts else "") + ("_lat" if s.get("lattice") else "")
    return key, " ".join(nm)


def build(product: str, s: dict):
    """(coefficients fn, sim factory) for a spec."""
    import arigo1_mujoco as amj
    import arigo20_model as m20
    import arigo20_mujoco as m20j
    if s.get("frame") == "arigo-20":
        base, simf = m20.get, (lambda bandit=False: m20j.Sim20(bandit=bandit))
    else:
        geo = (s.get("scale", 1.0) != 1.0) or s.get("span") or s.get("chord") or s.get("lattice")
        v = airframe(s.get("scale", 1.0), s.get("span"), s.get("chord")) if geo else "hybrid"
        src = "model" if geo else "auto"
        base = (lambda v=v, src=src: am.coefficients(v, "pusher", src))
        simf = (lambda bandit=False, v=v: amj.Sim(v, "pusher", bandit=bandit))
    feats = {"product": product, "frame": s.get("frame", "1H"), "variant_spec": {k: v for k, v in s.items() if k != "kits"}}
    if s.get("launch"):
        feats["launch"] = s["launch"]

    def Cf():
        return equip(base(), s.get("kits", ()), s.get("pack"), s.get("motor_w"), **feats)

    def sf(bandit=False):
        return resim(simf(bandit), Cf(), bandit)
    return Cf, sf


def equip(C: dict, kits=(), pack: str | None = None, motor_w: float | None = None, **features) -> dict:
    """A copy of the coefficients with the kits' items (and a 10 % growth margin on them), another pack, another motor
    rating; mass, inertia and thrust updated; the kits' features in ``C["equipment"]``."""
    import arigo_mass
    C = dict(C)
    mb = {**C["mass_budget"], "items": list(C["mass_budget"]["items"])}
    items, feat = [], {}
    for k in kits:
        it, f = KITS[k]
        items += it
        feat.update(f)
    if motor_w:
        items.append((f"motor + ESC for {motor_w:.0f} W (not 800 W)", MOTOR_G_PER_W * (motor_w - 800.0), "HAND: 0.12 g per W"))
    add = 0.0
    for n, g, src in items:
        mb["items"].append({"item": n, "g": round(g, 1), "source": src})
        add += g
    pos = sum(g for _, g, _ in items if g > 0)
    if pos:
        mb["items"].append({"item": "growth margin on the kit (10 %)", "g": round(0.1 * pos, 1), "source": "HAND"})
        add += 0.1 * pos
    kind = arigo_mass.kind_of(C.get("propulsion", ""))
    if pack:
        g, wh, amps, src = arigo_mass.packs(kind)[pack]
        mb["pack"] = {"name": pack, "g": g, "Wh_nominal": round(wh, 1), "max_A": amps, "source": src}
    mb["empty_g"] = round(mb["empty_g"] + add, 1)
    mb["total_g"] = round(mb["empty_g"] + mb["pack"]["g"], 1)
    mb["pack_share"] = round(mb["pack"]["g"] / mb["total_g"], 3)
    m = mb["total_g"] / 1000.0
    C["inertia"] = tuple(i * m / C["mass"] for i in C["inertia"])
    C["mass"], C["mass_budget"] = m, mb
    if C.get("propulsion") != "edf":                            # the fans keep their maps; the propellers are re-selected
        try:
            import arigo_propulsion as ap
            C["thrust"] = ap.thrust_table(ap.select(C["propulsion"], round(m, 3), round(C["CD0"], 4), round(C["S"], 4),
                                                    power_w=motor_w))
        except Exception:                                       # noqa: BLE001  keep the model usable without the library
            C["thrust"] = am.propulsion_table(C["propulsion"], m, C["CD0"], C["S"])
    C["equipment"] = {**(C.get("equipment") or {}), **feat, **features, "kits": list(kits), "pack": mb["pack"]["name"],
                      "motor_w": motor_w}
    return C


def resim(sim, C, bandit=False):
    """Rebuild a simulator around other coefficients (the kit's mass and inertia)."""
    import mujoco
    import arigo1_mujoco as amj
    sim.C = C
    sim.m = mujoco.MjModel.from_xml_string(amj.mjcf(C, None, bandit))
    sim.d = mujoco.MjData(sim.m)
    sim.bid = mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_BODY, "arigo")
    return sim


def _row(product, s):
    import arigo1_mujoco as amj
    import arigo20_model as m20
    import arigo20_mujoco as m20j
    key, name = spec_key(product, s)
    Cf, sf = build(product, s)
    a20 = s.get("frame") == "arigo-20"
    return key, name, Cf, sf, (m20j.manoeuvre if a20 else amj.manoeuvre), (m20.CRUISE if a20 else 18.0)


def candidates():
    """The stability run's product cards: each product's base spec on ARIGO-1H, and ARIGO-20 for ASTOR. Same shape
    as ``arigo_perf.configurations``."""
    for pk, pr in PRODUCTS.items():
        yield _row(pk, dict(pr["base"]))
    yield _row("astor", {"frame": "arigo-20", "kits": ("rail", "parachute"), "pack": "6S 6000 mAh LiPo"})


def explore(product: str):
    """Every spec in the product's design space (``SPACE``), the lattice baseline first; ASTOR also on ARIGO-20."""
    pr, sp = PRODUCTS[product], SPACE[product]
    base = dict(pr["base"])
    yield {**base, "lattice": True}
    axes = [k for k in ("scale", "span", "chord", "pack", "motor_w", "kits") if k in sp]
    for combo in itertools.product(*(sp[a] for a in axes)):
        s = {**base, **dict(zip(axes, combo))}
        if s.get("scale", 1.0) == 1.0 and not s.get("span") and not s.get("chord"):
            s["lattice"] = True                                   # ARIGO-1H's own geometry: on the lattice, like the others
        yield s
    if product == "astor":
        for pk in ("6S 4000 mAh LiPo", "6S 5000 mAh LiPo", "6S 6000 mAh LiPo"):
            yield {"frame": "arigo-20", "kits": ("rail", "parachute"), "pack": pk}


def explore_rows(products=None):
    """(product, key, display name, coefficients fn, sim factory, manoeuvre, cruise) for the sweep, de-duplicated."""
    seen = set()
    for pk in products or PRODUCTS:
        for s in explore(pk):
            r = _row(pk, s)
            if r[0] in seen:
                continue
            seen.add(r[0])
            yield (pk, *r)


def product_scores(cards: dict, mission_kpi: dict) -> dict:
    """Per product: its candidates on the product's mission score, best first."""
    out = {}
    for pk, pr in PRODUCTS.items():
        rows = []
        for k, c in cards.items():
            eq = c.get("equipment") or {}
            if eq.get("product") != pk or k not in mission_kpi:
                continue
            rows.append({"key": k, "name": c["aircraft"], "score": mission_kpi[k].get(pr["mission"]),
                         "mass_kg": (c.get("summary") or {}).get("mass_kg")})
        rows = sorted((r for r in rows if r["score"] is not None), key=lambda r: -r["score"])
        out[pk] = {"title": pr["title"], "role": pr["role"], "what": pr["what"], "priorities": pr["priorities"],
                   "candidates": rows, "design": rows[0]["name"] if rows else None}
    return out
