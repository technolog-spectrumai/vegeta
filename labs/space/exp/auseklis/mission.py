"""Per-removal Δv ledgers, propellant use, duration and 5-year throughput for each variant and debris class.

Disposal modes: ``decay5`` — hand the debris over on an orbit whose circular-equivalent decay time is <= 5 years (ESA
Zero Debris policy; conservative because an elliptical orbit decays faster than a circular one of equal a), and
``reentry`` — a perigee low enough (60 km) for a controlled/near-immediate reentry. The satellite then returns alone to
550 km. Debris and atmosphere parameters are ASSUMED; replace the density table before quoting lifetimes."""
from __future__ import annotations

import math
from dataclasses import dataclass

from . import physics as ph
from .provenance import assumed
from .variants import ALT_KM, LIFE_YEARS, Variant, budget


@dataclass(frozen=True)
class Debris:
    name: str
    mass: float                  # kg
    cda_over_m: float            # m^2/kg (Cd * A / m), tumbling-averaged
    cooperative: bool
    tumble_deg_s: float
    radius: float = 0.7          # m, equivalent cylinder
    length: float = 2.0          # m


DEBRIS = {
    100: Debris("100 kg (small stage/satellite)", 100, float(assumed(0.022, "m2/kg", "2.2 x ~1.0 m2 / 100 kg", "B_100")), True, 0.5, 0.4, 1.2),
    250: Debris("250 kg (upper stage)", 250, float(assumed(0.030, "m2/kg", "2.2 x ~3.4 m2 / 250 kg", "B_250")), False, 2.0, 0.6, 2.8),
    500: Debris("500 kg (large stage)", 500, float(assumed(0.030, "m2/kg", "2.2 x ~6.8 m2 / 500 kg", "B_500")), False, 3.0, 0.8, 3.6),
}
PHASING_ANGLE_DEG = assumed(90.0, "deg", "mean in-plane phase to close to the target", "phasing_angle")
PHASING_DAYS = assumed(10.0, "d", "drift time allowed for phasing", "phasing_days")
ALT_CHANGE_KM = assumed(20.0, "km", "mean altitude difference between servicer and target", "alt_change")
RDV_DV = assumed(12.0, "m/s", "far/near rendezvous, inspection fly-around, abort margin", "rdv_dv")
OPS_DAYS = assumed(8.0, "d", "inspection + approach + capture + stabilisation + checkout", "ops_days")
CONTINGENCY = assumed(0.25, "-", "schedule contingency", "contingency")
SUN_DUTY = assumed(0.65, "-", "fraction of time the Hall thruster can run (power, eclipse, pointing)", "sun_duty")
RESERVE = assumed(0.10, "-", "propellant kept back at end of life (disposal of the servicer itself)", "prop_reserve")


def disposal_altitude_km(d: Debris, years: float = LIFE_YEARS, activity: str = "mod") -> float:
    """Highest circular altitude at which ``d`` decays within ``years`` (bisection on ``physics.lifetime_days``)."""
    lo, hi = 250.0, ALT_KM
    if ph.lifetime_days(hi, d.cda_over_m, activity) / 365.25 <= years:
        return hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if ph.lifetime_days(mid, d.cda_over_m, activity) / 365.25 <= years:
            lo = mid
        else:
            hi = mid
    return lo


def _dv_tow(v: Variant, mode: str, d: Debris, activity: str = "mod"):
    r0 = ph.radius(ALT_KM)
    if mode == "reentry":
        if v.propulsion.kind == "hall_electric":
            return None                                  # a low-thrust spiral cannot make a controlled reentry burn
        return ph.perigee_lowering_dv(r0, ph.radius(60.0)), 60.0
    alt_d = disposal_altitude_km(d, activity=activity)
    if v.propulsion.kind == "hall_electric":
        return ph.spiral_dv(r0, ph.radius(alt_d)), alt_d
    r_p = 2 * ph.radius(alt_d) - r0                      # same semi-major axis as the circular hand-over orbit
    return ph.perigee_lowering_dv(r0, r_p), alt_d


def removal(v: Variant, d: Debris, mode: str = "decay5", m_sat: float | None = None, activity: str = "mod") -> dict:
    """Ledger of one removal starting from satellite mass ``m_sat`` (default: variant filled to 400 kg wet)."""
    p = v.propulsion
    m = budget(v, wet_target=400.0)["wet"] if m_sat is None else m_sat
    if v.cooperative_only and not d.cooperative:
        return {"feasible": False, "why": "uncooperative target: variant has no grapple-free capture", "variant": v.key}
    tow = _dv_tow(v, mode, d, activity)
    if tow is None:
        return {"feasible": False, "why": "controlled reentry needs an impulsive burn: not possible with low thrust", "variant": v.key}
    dv_tow, alt_d = tow
    r0 = ph.radius(ALT_KM)
    ph_leg = ph.phasing_dv(r0, math.radians(float(PHASING_ANGLE_DEG)), n_orbits=int(float(PHASING_DAYS) * 86400 / ph.period(r0)))["dv"]
    legs = [("phasing", ph_leg, m, "main"), ("altitude change", ph.hohmann(r0, ph.radius(ALT_KM + float(ALT_CHANGE_KM)))["dv"], m, "main"),
            ("rendezvous + inspection", float(RDV_DV), m, "rcs"),
            ("tow to hand-over (pair)", dv_tow, m + d.mass, "main"), ("return to 550 km (alone)", dv_tow, m, "main")]
    # propellant per leg; the main system and the RCS are drawn separately (variant C), otherwise one tank
    use = {"main": 0.0, "rcs": 0.0}
    t_leg, m_now, rows = [], m, []
    for name, dv, m_leg, system in legs:
        mass = m_leg if "pair" in name else m_now
        if "pair" in name:
            mass = m_now + d.mass
        isp = float(p.rcs_isp) if (system == "rcs" and p.rcs_dv) else float(p.isp)
        key = "rcs" if (system == "rcs" and p.rcs_dv) else "main"
        prop = ph.propellant_mass(mass, dv, isp)
        use[key] += prop
        m_now -= prop
        thrust = float(p.thrust) if key == "main" else 10.0
        burn = ph.burn_time(mass, dv, isp, thrust)
        dur = burn / float(SUN_DUTY) if (key == "main" and p.kind == "hall_electric") else burn
        rows.append({"leg": name, "dv": dv, "mass_kg": mass, "propellant_kg": prop, "burn_h": burn / 3600, "duration_d": dur / 86400})
    days = (max(float(PHASING_DAYS), rows[0]["duration_d"]) + rows[1]["duration_d"] + float(OPS_DAYS)
            + rows[3]["duration_d"] + rows[4]["duration_d"]) * (1 + float(CONTINGENCY))
    return {"feasible": True, "variant": v.key, "mode": mode, "debris": d.name, "handover_alt_km": alt_d, "legs": rows,
            "dv_total": sum(r["dv"] for r in rows), "propellant_main": use["main"], "propellant_rcs": use["rcs"],
            "propellant": use["main"] + use["rcs"], "days": days, "sat_mass_after": m_now}


def campaign(v: Variant, d: Debris, mode: str = "decay5", years: float = LIFE_YEARS, wet: float = 400.0, activity: str = "mod") -> dict:
    """Repeat removals of ``d`` until propellant (less the reserve) or time runs out."""
    b = budget(v, wet_target=wet)
    main_left, rcs_left = b["propellant"] * (1 - float(RESERVE)), b["rcs_propellant"] * (1 - float(RESERVE))
    m, t, n = b["wet"], 0.0, 0
    first = removal(v, d, mode, m, activity)
    if not first["feasible"]:
        return {"removals": 0, "limiting": first["why"], "variant": v.key}
    while True:
        r = removal(v, d, mode, m, activity)
        if r["propellant_main"] > main_left or r["propellant_rcs"] > rcs_left:
            limiting = "propellant"
            break
        if t + r["days"] > years * 365.25:
            limiting = "time"
            break
        main_left -= r["propellant_main"]
        rcs_left -= r["propellant_rcs"]
        m -= r["propellant"]
        t += r["days"]
        n += 1
        if n > 100:
            limiting = "cap"
            break
    return {"variant": v.key, "mode": mode, "debris": d.name, "removals": n, "limiting": limiting, "days_used": t,
            "first": first, "removed_mass_kg": n * d.mass}
