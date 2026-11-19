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
import math

import numpy as np

import emerillon_flight as ef
import gavilan as gv
import gavilan_flight as gf
import gavilan_perch as gpe

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


def _esc_extra_kg(c: ef.Config) -> float:
    """The mass tables carry one ESC item; a drive with an ESC per motor carries three more."""
    import emerillon_market as mk
    e = mk.part(c.esc)
    return (e.count - 1) * e.mass_g / 1e3


def x_member(scale: float, packs: int, prop: str = "hq7x4x3", hv: bool = False) -> gf.Machine:
    q = x_params(scale, packs)
    c = _config(prop, packs, hv)
    m = gf.gavilan_x(q, c)
    mass = gf.x_mass_budget(c, q, talons=False).mass + catcher_kg() + _esc_extra_kg(c)
    tag = f"12S{max(packs // 2, 1)}P 3120s" if hv else f"{packs} pack{'s' if packs > 1 else ''}"
    return dataclasses.replace(m, name=f"Arigo-X {scale:.2f}x, {tag}, {prop}", mass=mass,
                               cda0=m.cda0 - gf.TALONS["cda_m2"] + gf.SLING_CDA, payload_kg=catcher_kg(), gun="slingshot",
                               notes={**m.notes, "path": "X12" if hv else "X", "scale": scale, "packs": packs, "propeller": prop, "params": q})


def a_member(scale: float, packs: int, prop: str = "hq7x4x3", hv: bool = False) -> gf.Machine:
    p = a_params(scale, 2 if hv else packs)
    c = _config(prop, packs, hv)
    m = gf.gavilan("A", emitters=0, config=c, p=p)
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
    """The second round: the 12S X-wing with four packs (12S2P) for the reach, at two sizes."""
    return {f"X12 {s:.2f} 12S2P hq7x4x3": (lambda s=s: x_member(s, 4, "hq7x4x3", hv=True)) for s in (1.15, 1.3)}


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
    for n in (1, 2):
        out[f"H 1.00 {n}p hq7x4x3"] = lambda n=n: h_member(n)
    return out


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
