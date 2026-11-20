"""ARIGO products (notebook 39, the product round): a product is a client's job, scored on its own mission KPIs; the
airframes (ARIGO-0, -1, -2, -20, and the small ARIGO-1 backpack wing) are candidates for it, each with a kit — how it
is launched, how it comes back, which pack, what extra it carries. The run scores every candidate; the best candidate
of each product is that product's current design, and its score is the number a redesign has to beat.

Products (the client's view):

1. Perch Sentry — point defence of infrastructure. Waits on a mast or a roof edge, motor off; drops from the perch on
   the site sensor's cue; recovers by a belly landing near the site. Launch: a perch with a spring push-off (no
   person needed).
2. Area Guard — area defence up to 10 km out (pipelines, perimeters, a convoy's route). Launched by hand or from a
   vehicle; goes out fast, chases, comes back.
3. Dash Interceptor — fast targets (fixed-wing one-way drones, 30-50 m/s). Lives in a launcher at the site; rail
   launch; recovers under a parachute.
4. Backpack — carried by one defender in a backpack, assembled in a minute, thrown by hand at a quadcopter that is
   already close. Light and short-ranged.

Kit items are HAND estimates (marked); everything else is the airframe's own model.
"""
from __future__ import annotations

import arigo1_model as am

# ------------------------------------------------------------------------------------------------ kits (HAND)
PERCH_KIT = [("perch hook + release servo", 25.0, "HAND (+-8)")]
PERCH_PUSH_M_S = 8.0               # HAND: a spring push-off on the perch (vs 4 m/s by the aircraft itself)
PARACHUTE_KIT = [("parachute + canopy bay + release servo", 85.0, "HAND (+-20)")]
PARACHUTE_SINK_M_S = 5.0           # HAND: descent under a ~0.8 m2 canopy at ~2.3 kg
PARACHUTE_MISS_M = 15.0            # HAND: the drift left after aiming upwind of the site in a 6 m/s wind
TWO_PIECE_KIT = [("two-piece wing joiner + plugs", 30.0, "HAND (+-10)")]
ASSEMBLY_S = 60.0                  # HAND: wing halves on, pack in, power up, self-test


PRODUCTS = {
    "perch_sentry": {"title": "Perch Sentry", "mission": "point_defence",
                     "weights": {"point_defence": 0.60, "recovery": 0.25, "area_defence": 0.15},
                     "what": "point defence of infrastructure from a perch; no person needed to launch"},
    "area_guard": {"title": "Area Guard", "mission": "area_defence",
                   "weights": {"area_defence": 0.55, "recovery": 0.20, "field": 0.15, "point_defence": 0.10},
                   "what": "area defence up to 10 km out; hand or vehicle launch"},
    "dash_interceptor": {"title": "Dash Interceptor", "mission": "fast_intercept",
                         "weights": {"fast_intercept": 0.65, "recovery": 0.20, "point_defence": 0.15},
                         "what": "fast targets (30-50 m/s); a launcher at the site; parachute recovery"},
    "backpack": {"title": "Backpack", "mission": "backpack",
                 "weights": {"backpack": 0.65, "recovery": 0.20, "field": 0.15},
                 "what": "carried by one defender; assembled in a minute; thrown at a close quadcopter"},
}


def equip(C: dict, kit=(), pack: str | None = None, **features) -> dict:
    """A copy of the coefficients with kit items added (and the growth margin on them), another pack, the mass,
    inertia and thrust table updated; ``features`` stored in ``C["equipment"]`` for the mission metrics."""
    import arigo_mass
    C = dict(C)
    mb = {**C["mass_budget"], "items": list(C["mass_budget"]["items"])}
    add = 0.0
    for n, g, s in kit:
        mb["items"].append({"item": n, "g": g, "source": s})
        add += g
    if add:
        mb["items"].append({"item": "growth margin on the kit (10 %)", "g": round(0.1 * add, 1), "source": "HAND"})
        add *= 1.1
    k = arigo_mass.kind_of(C.get("propulsion", ""))
    if pack:
        g, wh, amps, src = arigo_mass.packs(k)[pack]
        mb["pack"] = {"name": pack, "g": g, "Wh_nominal": round(wh, 1), "max_A": amps, "source": src}
    mb["empty_g"] = round(mb["empty_g"] + add, 1)
    mb["total_g"] = round(mb["empty_g"] + mb["pack"]["g"], 1)
    mb["pack_share"] = round(mb["pack"]["g"] / mb["total_g"], 3)
    m = mb["total_g"] / 1000.0
    C["inertia"] = tuple(i * m / C["mass"] for i in C["inertia"])
    C["mass"], C["mass_budget"] = m, mb
    if C.get("propulsion") != "edf":                          # the fans keep their maps; the propellers are re-selected
        C["thrust"] = am.propulsion_table(C["propulsion"], m, C["CD0"], C["S"])
    C["equipment"] = {**(C.get("equipment") or {}), **features,
                      "kit": [n for n, _, _ in kit], "pack": mb["pack"]["name"]}
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


def candidates():
    """(key, display name, coefficients fn, sim factory, manoeuvre fn, cruise speed) for every product candidate —
    the same shape as ``arigo_perf.configurations``. Keys start with the airframe's family (the run files filter on it)."""
    import arigo1_mujoco as amj
    import arigo2_model as m2
    import arigo2_mujoco as m2j
    import arigo20_model as m20
    import arigo20_mujoco as m20j

    def a1(w, prop):
        return lambda: am.coefficients(w, prop)

    def mk(base_fn, sim_cls_fn, product, kit=(), pack=None, **feat):
        def Cf():
            return equip(base_fn(), kit, pack, product=product, **feat)

        def sf(bandit=False):
            return resim(sim_cls_fn(bandit), Cf(), bandit)
        return Cf, sf

    S1 = lambda w, p: (lambda bandit=False: amj.Sim(w, p, bandit=bandit))
    S2 = lambda w: (lambda bandit=False: m2j.Sim2(w, bandit=bandit))
    S20 = lambda bandit=False: m20j.Sim20(bandit=bandit)
    perch = dict(launch="perch", perch_push_m_s=PERCH_PUSH_M_S, recovery="belly")
    rows = [
        # 1. Perch Sentry
        ("arigo-1_hybrid_perch", "ARIGO-1 hybrid, perch kit", a1("hybrid", "pusher"), S1("hybrid", "pusher"), "perch_sentry", PERCH_KIT, None, perch),
        ("arigo-0_hybrid_perch", "ARIGO-0 hybrid, perch kit", a1("hybrid", "tractor_lowcam"), S1("hybrid", "tractor_lowcam"), "perch_sentry", PERCH_KIT, None, perch),
        ("arigo-2_hybrid_perch", "ARIGO-2 hybrid, perch kit", lambda: m2.coefficients("hybrid"), S2("hybrid"), "perch_sentry", PERCH_KIT, None, perch),
        ("arigo-1_zero_perch", "ARIGO-1 zero, perch kit", a1("zero", "pusher"), S1("zero", "pusher"), "perch_sentry", PERCH_KIT, None, perch),
        # 2. Area Guard (hand launch; the 3S3P pack for the radius)
        ("arigo-1_hybrid_area", "ARIGO-1 hybrid, 3S3P", a1("hybrid", "pusher"), S1("hybrid", "pusher"), "area_guard", (), "Li-ion 3S3P P42A", dict(launch="hand", recovery="belly")),
        ("arigo-1_zero_area", "ARIGO-1 zero, 3S3P", a1("zero", "pusher"), S1("zero", "pusher"), "area_guard", (), "Li-ion 3S3P P42A", dict(launch="hand", recovery="belly")),
        ("arigo-2_hybrid_area", "ARIGO-2 hybrid, 3S3P", lambda: m2.coefficients("hybrid"), S2("hybrid"), "area_guard", (), "Li-ion 3S3P P42A", dict(launch="hand", recovery="belly")),
        # 3. Dash Interceptor: ARIGO-20 site (rail launcher + parachute + bigger packs) and the bare ARIGO-20 for reference
        ("arigo-20_site", "ARIGO-20 site, 6S 5000", m20.get, S20, "dash_interceptor", PARACHUTE_KIT, "6S 5000 mAh LiPo", dict(launch="rail", recovery="parachute")),
        ("arigo-20_site6", "ARIGO-20 site, 6S 6000", m20.get, S20, "dash_interceptor", PARACHUTE_KIT, "6S 6000 mAh LiPo", dict(launch="rail", recovery="parachute")),
        ("arigo-20_bare", "ARIGO-20 bare, catapult", m20.get, S20, "dash_interceptor", (), None, dict(launch="rail", recovery="belly")),
        ("arigo-1_hybrid_dash", "ARIGO-1 hybrid, rail + parachute", a1("hybrid", "pusher"), S1("hybrid", "pusher"), "dash_interceptor", PARACHUTE_KIT, None, dict(launch="rail", recovery="parachute")),
        # 4. Backpack: the small two-piece wing (ARIGO-1 "backpack" variant) and lighter ARIGO-1s
        ("arigo-1_backpack", "ARIGO-1 backpack, 3S1P", a1("backpack", "pusher"), S1("backpack", "pusher"), "backpack", TWO_PIECE_KIT, "Li-ion 3S1P P42A", dict(launch="hand", recovery="belly", two_piece=True)),
        ("arigo-1_backpack2", "ARIGO-1 backpack, 3S2P", a1("backpack", "pusher"), S1("backpack", "pusher"), "backpack", TWO_PIECE_KIT, None, dict(launch="hand", recovery="belly", two_piece=True)),
        ("arigo-1_hybrid_backpack", "ARIGO-1 hybrid two-piece, 3S1P", a1("hybrid", "pusher"), S1("hybrid", "pusher"), "backpack", TWO_PIECE_KIT, "Li-ion 3S1P P42A", dict(launch="hand", recovery="belly", two_piece=True)),
    ]
    for key, name, base, simf, product, kit, pack, feat in rows:
        Cf, sf = mk(base, simf, product, kit, pack, **feat)
        V = m20.CRUISE if key.startswith("arigo-20") else 18.0
        man = m20j.manoeuvre if key.startswith("arigo-20") else (m2j.manoeuvre if key.startswith("arigo-2_") else amj.manoeuvre)
        yield key, name, Cf, sf, man, V


def product_scores(cards: dict, mission_kpi: dict) -> dict:
    """Per product: every candidate's product score (its mission scores under the product's weights), best first."""
    out = {}
    for pk, pr in PRODUCTS.items():
        rows = []
        for k, c in cards.items():
            eq = (c.get("equipment") or {})
            if eq.get("product") != pk or k not in mission_kpi:
                continue
            m = mission_kpi[k]
            sc = sum(w * (m.get(mm) or 0.0) for mm, w in pr["weights"].items())
            rows.append({"key": k, "name": c["aircraft"], "score": round(sc, 1), **{mm: m.get(mm) for mm in pr["weights"]}})
        rows.sort(key=lambda r: -r["score"])
        out[pk] = {"title": pr["title"], "what": pr["what"], "weights": pr["weights"], "candidates": rows,
                   "design": rows[0]["name"] if rows else None}
    return out
