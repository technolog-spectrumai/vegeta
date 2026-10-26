"""Strato — the feasibility gate: one design and mission in, one verdict out, with every reason (Stage V8.12).

``evaluate`` configures the design point (``strato_design``), runs the day/night cycle (``strato_sim``) to get the
gas superheat, re-configures with it (the superpressure and the envelope depend on it), and measures station-keeping
availability, gas life and energy closure. Its ``problems`` are physical impossibilities (the ship does not float,
cannot be trimmed, the envelope cannot be made strong enough, the battery cannot carry the payload through the night
even in calm air): the design is rejected. Its ``warnings`` are requirements a real design could still meet another
way (static instability needs active control; a vented design's valve). Nothing is clipped silently.

Objectives (for the optimiser): payload [kg] and availability [0–1] and endurance [days] up; system mass [kg],
drag at the design wind [N] and daily energy [Wh] down.
"""
from __future__ import annotations

import math
from dataclasses import asdict

import strato_aero as sa
import strato_design as sd
import strato_sim as ss
import strato_wind as sw


def evaluate(d: sd.Design, m: sd.Mission = sd.Mission(), *, passes: int = 2, n_days: int = 1000,
             detail: bool = False, superheat_day_floor: float = 0.0) -> dict:
    """The verdict on ``d`` flying mission ``m`` (``detail`` keeps the configuration and the day cycle).
    ``superheat_day_floor`` [K] sizes the envelope for a hotter month than this one (a year-round design)."""
    try:
        c = sd.configure(d, m)
    except ValueError as e:
        return {"feasible": False, "problems": [f"invalid design: {e}"], "warnings": [], "design": asdict(d)}
    # the thermal cycle at the month's median wind: a station-keeping ship flies at about the wind speed, and the
    # forced convection that brings is what limits the superheat
    v_thermal = sw.speed_percentile(d.altitude, m.month, m.lat, 50.0, 4000)
    cyc = None
    for _ in range(passes):
        sh = ss.Ship(c)
        cyc = ss.day_cycle(c, min(v_thermal, sh.v_max), ship=sh)
        c = sd.configure(d, m, superheat_day=max(cyc["superheat_max"], superheat_day_floor, 0.0),
                         superheat_night=min(cyc["superheat_min"], 0.0))
    sh = ss.Ship(c)
    cyc = ss.day_cycle(c, min(v_thermal, sh.v_max), ship=sh)
    av = ss.availability(c, cyc, n_days=n_days, ship=sh)
    problems, warnings = list(c.problems), list(c.warnings)
    if not av["calm_closes"]:
        problems.append("energy does not close even in calm air: the array cannot carry avionics and payload through "
                        f"the day and the battery through the night in month {m.month}")
    gas_life = ss.gas_life_days(c) if c.surplus_lift >= 0 else 0.0
    battery_hours = sh.usable_wh / max(sh.loads, 1e-9)
    endurance = (gas_life if d.gas_system == "sealed" else math.nan) if av["calm_closes"] else battery_hours / 24.0
    wind = av["wind_p90"]
    drag = sa.drag(c.ship, wind, d.altitude).total
    out = {
        "feasible": not problems, "problems": problems, "warnings": warnings, "design": asdict(d),
        "mission": asdict(m), "length": d.length, "diameter": c.ship.hull.diameter, "volume": c.ship.hull.volume,
        "envelope_area": c.ship.hull.surface_area, "solar_area": c.ship.solar_area, "payload": d.payload_mass,
        "payload_power": c.payload_power, "system_mass": c.system_mass, "masses": c.masses, "gas_mass": c.gas_mass,
        "gross_lift": c.gross_lift, "net_lift": c.net_lift, "surplus_lift": c.surplus_lift,
        "max_float_altitude": sd.max_float_altitude(c), "plies": c.plies, "dp_min": c.dp_min, "dp_max": c.dp_max,
        "hoop_line_load": c.hoop_line_load, "bending_moment": c.bending_moment,
        "superheat_day": c.superheat_day, "superheat_night": c.superheat_night, "cg": list(map(float, c.cg)),
        "x_cb": c.x_cb, "stability_margin": sa.stability(c.ship, float(c.cg[0]))["margin"],
        "v_max": av["v_max"], "v_energy": av["v_energy"], "availability": av["availability"],
        "wind_p50": av["wind_p50"], "wind_p90": av["wind_p90"], "wind_p95": av["wind_p95"],
        "energy_solar_wh": cyc["energy_solar_wh"], "energy_loads_wh": cyc["energy_load_wh"] - cyc["p_prop_need"] * 24.0,
        "energy_at_p90_wh": cyc["energy_load_wh"] - cyc["p_prop_need"] * 24.0 + float(sh.power_for(min(wind, sh.v_max))) * 24.0,
        "drag_p90": drag, "cdv": sa.drag(c.ship, max(wind, 1.0), d.altitude).cdv,
        "gas_life_days": gas_life, "endurance_days": endurance, "battery_hours": battery_hours,
    }
    if detail:
        out["configuration"] = c
        out["cycle"] = cyc
        out["ship"] = sh
    return out


def objectives(r: dict) -> dict:
    """The optimiser's view of a verdict: values to maximise are negated (everything minimised)."""
    return {"-payload": -r.get("payload", 0.0), "-availability": -r.get("availability", 0.0),
            "-endurance": -min(r.get("endurance_days", 0.0) or 0.0, 3650.0), "mass": r.get("system_mass", math.inf),
            "drag": r.get("drag_p90", math.inf), "energy": r.get("energy_at_p90_wh", math.inf)}
