"""ARIGO — a bigger, more manoeuvrable hawk than GAVILAN, with a longer chase, that can drop its net on a bandit in the
air or on the ground (notebook 39). This module explores the paths before one is chosen.

GAVILAN (notebook 38) is a sparrowhawk: it waits on a charging perch, makes a short pursuit (11 min of cruise on one
pack) and throws a net from 3-5 m. ARIGO keeps the perch, the opposed latex slingshot (one net forward, one aft: the
aft one points straight down when it hangs nose-up, which is how it drops a net on a bandit below it — hovering, or
landed) and the charging feet, and asks for more: a longer chase and harder manoeuvring, at a somewhat larger size.

The paths, all on GAVILAN's models (``gavilan_flight``: the body-axis point mass, ÉMERILLON's drive and catalogue, the
vortex lattice, the build-ups), all carrying the same slingshot, charging parts and propeller guards:

- **X** — GAVILAN-X's four identical swept panels, scaled (``scale``), on one or two packs, on either of the
  catalogue's 7-inch propellers the drive model knows well (``hq7x4x3``: thrust at low speed; ``apc7x7e``: holds its
  thrust to higher speeds);
- **A** — GAVILAN-A's short broad wing on posts with a long tail and four fins (the best chaser of the first study),
  scaled, one or two packs;
- **H** — GAVILAN-H, the fixed wing that flies past its stall on a blown tail (two motors), one or two packs.

The motors are the catalogue's ECO II 2807 1300 KV throughout (the trade in ``emerillon_flight`` and the first study
found nothing better at these speeds); the propellers stay at 7 inches because the catalogue's maps stop there.

``candidates`` builds every path's members as ``Machine``s; ``evaluate`` gives each the hunt's numbers; ``trade``
runs them all; ``pareto`` keeps the ones nothing beats on both manoeuvre and reach.
"""
from __future__ import annotations

import dataclasses
import functools
import math

import numpy as np

import emerillon_flight as ef
import gavilan as gv
import gavilan_flight as gf
import gavilan_perch as gpe

# Larger propellers than the catalogue lists, for the larger airframe: ASSUMED rows (HQProp makes these sizes; mass,
# speed limit and planform are taken from the catalogue's 7x4x3 scaled, nothing sourced). Added to the catalogue in this
# process only (emerillon_market's file is not changed).
EXTRA_PROPELLERS = {
    "hq8x4.5x3": dict(name="HQProp 8x4.5x3 (assumed row)", mass_g=14.0, price=4.0, d_in=8.0, p_in=4.5),
    "hq9x5x3": dict(name="HQProp 9x5x3 (assumed row)", mass_g=19.0, price=5.0, d_in=9.0, p_in=5.0),
}


def _register_propellers():
    import emerillon_market as mk
    for key, r in EXTRA_PROPELLERS.items():
        if key not in mk.CATALOGUE:
            mk.CATALOGUE[key] = mk.Part(key, r["name"], "propeller", r["price"], r["mass_g"], (), "", "", "assumed",
                                        "scaled from the 7x4x3 row: mass, the 190,000/D speed limit and the planform are assumptions",
                                        data=dict(d_in=r["d_in"], p_in=r["p_in"], blades=3, rpm_limit_k=190000.0, chord=(0.085, 0.115, 0.055), tc_tip=0.07))


_register_propellers()

PACK_BAY = 84.5            # mm of body bay per extra 6S 1700 mAh pack in line (gavilan.VARIANTS['J'] against 'A')
PURSUIT_S = 120.0          # the hard part of a hunt at the far end: two minutes at 2 g (the job asks for 5 at Gavilan's level; here a share of it)
RESERVE = 0.10             # of the usable pack, kept for the landing on the perch
SHOT_RANGE = 4.0           # the throw, from the middle of the net's 3-5 m


def catcher_kg() -> float:
    return gf.sling_kg() + gpe.onboard_kg()


def x_params(scale: float, packs: int) -> dict:
    """GAVILAN-X's parameters scaled: the panels (radius, chords), the motors out along the panels in proportion, the
    body one pack-bay longer per extra pack, the panels moved back a little with it, the slingshot under the belly."""
    panel_x = 215.0 + 50.0 * (packs - 1)
    r_m = 200.0 * scale
    return {"panel_radius": 340.0 * scale, "panel_root_chord": 220.0 * scale, "panel_tip_chord": 110.0 * scale, "panel_x": panel_x,
            "length": panel_x + 365.0 * scale, "bay_length": 134.5 + PACK_BAY * (packs - 1),
            "motor_y": r_m / math.sqrt(2.0), "motor_z": r_m / math.sqrt(2.0), "slingshot": True, "talons": False}


def a_params(scale: float, packs: int) -> dict:
    p = {"span": 700.0 * scale, "root_chord": 230.0 * scale, "motor_y": 200.0 * scale, "wing_x": 235.0, "bay_length": 134.5 + PACK_BAY * (packs - 1),
         "length": 780.0 * scale + PACK_BAY * (packs - 1)}
    return p


HIGH_VOLTAGE = ("v3120-700", "apd-120f3")    # the catalogue's 3120 on 12S: two packs in series, an 120 A ESC per motor


def _config(prop: str, packs: int, hv: bool = False) -> ef.Config:
    """The drive: ECO 2807s on 6S (``packs`` in parallel), or (``hv``) T-Motor 3120s on 12S — packs in series pairs
    (``packs`` 2: 12S1P, 4: 12S2P)."""
    if hv:
        return ef.Config(HIGH_VOLTAGE[0], prop, HIGH_VOLTAGE[1], gf.CONFIG_J.battery, 2, max(packs // 2, 1))
    return dataclasses.replace(gf.CONFIG_J, propeller=prop, n_parallel=packs)


@functools.lru_cache(maxsize=None)
def capped_drive_map(c: ef.Config, n_motors: int = 4, v_max: float = 80.0, n_v: int = 17) -> dict:
    """``gavilan_flight.drive_map`` with the throttle held where the propeller reaches its maker's speed limit
    (``rpm_limit_k / diameter``): on 12S the 3120s would spin the 7x4x3 past it (29,000 rpm against 27,100)."""
    import emerillon_market as mk
    p = mk.part(c.propeller)
    limit = p.rpm_limit_k / p.d_in
    V = np.linspace(0.0, v_max, n_v)
    T, P = np.zeros((n_v, len(gf.U_GRID))), np.zeros((n_v, len(gf.U_GRID)))
    for i, v in enumerate(V):
        vv = max(float(v), 0.5)
        lo, hi = 0.05, 1.0
        if ef.drive(c, vv)["rpm"] > limit:
            for _ in range(14):
                u = 0.5 * (lo + hi)
                lo, hi = (u, hi) if ef.drive(c, vv, throttle=u)["rpm"] <= limit else (lo, u)
            cap = lo
        else:
            cap = 1.0
        for j, u in enumerate(gf.U_GRID):
            d = ef.drive(c, vv, throttle=u * cap)
            T[i, j], P[i, j] = max(d["thrust"], 0.0) * n_motors / 4, d["electric_w"] * n_motors / 4
    T = np.maximum.accumulate(T, axis=1) + np.arange(len(gf.U_GRID)) * 1e-9
    return {"V": V, "T": T, "P": P, "n": n_motors, "rpm_limit": limit}


def _esc_extra_kg(c: ef.Config) -> float:
    """The mass tables carry one ESC item; a drive with an ESC per motor carries three more."""
    import emerillon_market as mk
    e = mk.part(c.esc)
    return (e.count - 1) * e.mass_g / 1e3


def x_member(scale: float, packs: int, prop: str = "hq7x4x3", hv: bool = False) -> gf.Machine:
    import emerillon_market as mk
    q = {**x_params(scale, packs), "prop_diameter": round(mk.part(prop).d_in * 25.4, 1)}
    c = _config(prop, packs, hv)
    m = gf.gavilan_x(q, c)
    mass = gf.x_mass_budget(c, q, talons=False).mass + catcher_kg() + _esc_extra_kg(c)
    if hv:                                                            # the propeller kept under its speed limit
        t_max, power = gf._drive_fns(capped_drive_map(c), gf.AVIONICS_W)
        m = dataclasses.replace(m, t_max=t_max, power=power)
    tag = f"12S{max(packs // 2, 1)}P 3120s" if hv else f"{packs} pack{'s' if packs > 1 else ''}"
    return dataclasses.replace(m, name=f"Arigo-X {scale:.2f}x, {tag}, {prop}", mass=mass,
                               cda0=m.cda0 - gf.TALONS["cda_m2"] + gf.SLING_CDA, payload_kg=catcher_kg(), gun="slingshot",
                               notes={**m.notes, "path": "X12" if hv else "X", "scale": scale, "packs": packs, "propeller": prop, "params": q})


def a_member(scale: float, packs: int, prop: str = "hq7x4x3", hv: bool = False) -> gf.Machine:
    p = a_params(scale, 2 if hv else packs)
    c = _config(prop, packs, hv)
    m = gf.gavilan("A", emitters=0, config=c, p=p)
    if hv:
        t_max, power = gf._drive_fns(capped_drive_map(c), gf.AVIONICS_W)
        m = dataclasses.replace(m, t_max=t_max, power=power)
    extra = catcher_kg() + gf.guards_kg() + _esc_extra_kg(c)
    m = gf.with_payload(m, extra, gf.SLING_CDA + gf.GUARDS["cda_m2"], "", gun="slingshot")
    tag = "12S 3120s" if hv else f"{packs} pack{'s' if packs > 1 else ''}"
    return dataclasses.replace(m, name=f"Arigo-A {scale:.2f}x, {tag}, {prop}",
                               notes={**m.notes, "path": "A12" if hv else "A", "scale": scale, "packs": 2 if hv else packs, "propeller": prop, "params": p})


def h_member(packs: int) -> gf.Machine:
    c = dataclasses.replace(gf.CONFIG_H, n_parallel=packs)
    m = gf.gavilan("F", emitters=0, p=gf.H_PARAMS, config=c)
    m = dataclasses.replace(m, theta_limit=math.radians(90.0), stops=True, roll_rate=math.radians(300.0),
                            mass=gf.h_mass_budget(False).mass + (packs - 1) * 0.246 + catcher_kg() + gf.guards_kg() * 0.5)
    m = dataclasses.replace(m, cda0=m.cda0 + gf.SLING_CDA, payload_kg=catcher_kg(), gun="slingshot",
                            name=f"Arigo-H (blown tail), {packs} pack{'s' if packs > 1 else ''}",
                            notes={**m.notes, "path": "H", "scale": 1.0, "packs": packs, "propeller": c.propeller})
    return m


def candidates_round2() -> dict:
    """The second round: the 12S X-wing with four packs (12S2P) for the reach, at two sizes; and the 12S members of
    the first round again, now with the propeller held under its speed limit."""
    out = {f"X12 {s:.2f} 12S2P hq7x4x3": (lambda s=s: x_member(s, 4, "hq7x4x3", hv=True)) for s in (1.15, 1.3)}
    out["X12 1.15 12S hq7x4x3"] = lambda: x_member(1.15, 2, "hq7x4x3", hv=True)
    out["A12 1.15 12S hq7x4x3"] = lambda: a_member(1.15, 2, "hq7x4x3", hv=True)
    return out


def candidates_round3() -> dict:
    """The third round: the larger X (1.3x) on larger propellers (ASSUMED rows) — 8 and 9 inches on the 12S 3120s,
    one and two pairs of packs, and 8 inches on the 6S ECOs on two packs."""
    out = {}
    for prop in ("hq8x4.5x3", "hq9x5x3"):
        for packs in (2, 4):
            out[f"X12 1.30 12S{packs // 2}P {prop}"] = lambda prop=prop, packs=packs: x_member(1.3, packs, prop, hv=True)
    out["X 1.30 2p hq8x4.5x3"] = lambda: x_member(1.3, 2, "hq8x4.5x3")
    return out


def candidates() -> dict:
    out = {}
    for s in (1.0, 1.15, 1.3):
        for n in (1, 2):
            out[f"X {s:.2f} {n}p hq7x4x3"] = lambda s=s, n=n: x_member(s, n)
    out["X 1.15 2p apc7x7e"] = lambda: x_member(1.15, 2, "apc7x7e")
    for s in (1.15, 1.3):
        for prop in ("hq7x4x3", "apc7x7e"):
            out[f"X12 {s:.2f} 12S {prop}"] = lambda s=s, prop=prop: x_member(s, 2, prop, hv=True)
    for s in (1.0, 1.15):
        for n in (1, 2):
            out[f"A {s:.2f} {n}p hq7x4x3"] = lambda s=s, n=n: a_member(s, n)
    out["A12 1.15 12S hq7x4x3"] = lambda: a_member(1.15, 2, "hq7x4x3", hv=True)
    return out              # H (the blown-tail fixed wing) was weighed in notebook 38: it stops in 36 m, not 14; left out here


def chase_radius_km(m: gf.Machine) -> float:
    """How far out a bandit can be met: out and back at the best-range speed, ``PURSUIT_S`` at 2 g at the far end,
    ``RESERVE`` kept."""
    en = gf.endurance(m)
    usable = gf.USABLE * m.battery_wh * (1 - RESERVE)
    pursuit = m.power(2.0 * m.weight, 0.0) * PURSUIT_S / 3600.0
    wh_km = en["cruise_w"] / en["cruise_v"] / 3.6
    return max(usable - pursuit, 0.0) / (2.0 * wh_km)


def evaluate(key: str, make) -> dict:
    m = make()
    gf._SPRINTERS[key] = make
    v_shot = gf.fastest_safe_shot(SHOT_RANGE, m)
    keep = gf.CONTACT_SPEED
    gf.CONTACT_SPEED = max(v_shot, 0.5)
    try:
        foul = {d: gf.intercept(key, d - SHOT_RANGE)["time_s"] for d in (100.0, 300.0, 600.0)}
    finally:
        gf.CONTACT_SPEED = keep
    en, br = gf.endurance(m), gf.brake(m)
    t15, t25 = gf.turn(m, 15.0), gf.turn(m, 25.0)
    return {"key": key, "aircraft": m.name, "path": m.notes.get("path"), "scale": m.notes.get("scale"), "packs": m.notes.get("packs"),
            "propeller": m.notes.get("propeller"), "mass_kg": m.mass, "width_m": m.span, "thrust_over_weight": m.t_max(0.0) / m.weight,
            "thrust_over_weight_with_bandit": m.t_max(0.0) / ((m.mass + gf.BANDIT_KG) * gf.G),
            "turn_15_deg_s": t15["rate_inst_deg_s"], "turn_25_deg_s": t25["rate_inst_deg_s"], "sustained_25_deg_s": t25["rate_sus_deg_s"],
            "stop_from_25_m": br["distance_m"], "reverse_15_s": gf.reversal(m)["time_s"], "top_speed_m_s": gf.top_speed(m),
            "cruise_min": en["cruise_min"], "cruise_m_s": en["cruise_v"], "hover_min": en["hover_min"] if m.hovers else None,
            "chase_radius_km": chase_radius_km(m), "safe_shot_m_s": v_shot,
            **{f"perch_to_fouled_{d:.0f}_m_s": t + float(np.interp(SHOT_RANGE, list(gf.SLING_NET_FLIGHT_S), list(gf.SLING_NET_FLIGHT_S.values())))
               for d, t in foul.items()}}


def trade(progress=print) -> list:
    rows = []
    for key, make in candidates().items():
        r = evaluate(key, make)
        rows.append(r)
        if progress:
            progress(f"{key}: {r['mass_kg']:.2f} kg, T/W {r['thrust_over_weight']:.1f}, turn {r['turn_15_deg_s']:.0f}/{r['turn_25_deg_s']:.0f} deg/s, "
                     f"stop {r['stop_from_25_m']:.1f} m, cruise {r['cruise_min']:.0f} min, radius {r['chase_radius_km']:.2f} km, "
                     f"300 m in {r['perch_to_fouled_300_m_s']:.1f} s")
    return rows


def manoeuvre_index(r: dict, base: dict) -> float:
    """One number for 'more manoeuvrable', against GAVILAN (``base``): the mean of the turn rates at 15 and 25 m/s and
    the inverse of the stopping distance, each over GAVILAN's (1 = Gavilan)."""
    return (r["turn_15_deg_s"] / base["turn_15_deg_s"] + r["turn_25_deg_s"] / base["turn_25_deg_s"] + base["stop_from_25_m"] / r["stop_from_25_m"]) / 3.0


def pareto(rows: list, base: dict) -> list:
    """The members nothing beats on both the manoeuvre index and the chase radius."""
    pts = [(manoeuvre_index(r, base), r["chase_radius_km"], r) for r in rows]
    return [r for mi, cr, r in pts if not any(m2 >= mi and c2 >= cr and (m2 > mi or c2 > cr) for m2, c2, _ in pts)]
