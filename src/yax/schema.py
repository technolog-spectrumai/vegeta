"""The engineering schemas: what each part-data block holds, in which units and frames, and what each target format
loses of it (docs/engineering.md is the contract; this module is its oracle).

A block is JSON under a documented key — on the machine (``data[key]``), a link (``link.data[key]``) or a joint
(``joint.data[key]``). Fields are named with their unit as a suffix (``stall_torque_nm``, ``no_load_speed_rpm``,
``density_kg_m3``); a dimensionless field has none and is listed as such. Every block may carry ``source`` (where
the numbers come from) and ``validity`` (the ranges they hold in); readers keep fields they do not know. Yax
validates shape, types, units, references and frames; it never fills a field a producer left out, and it never
computes a physical quantity from another.

``problems(m)`` lists what is wrong with a machine's blocks; ``report_losses(m, fmt, rep)`` says what a target
format cannot carry of them. Tags classify the blocks (``yax.tags``); the schemas say what is in them.
"""
from __future__ import annotations

import math

from .hub import Machine, Report

#: Unit suffixes a numeric field may end with (SI, plus the ones engineers quote: rpm, deg, °C).
UNITS = ("_nm", "_n", "_kg", "_kgm2", "_kg_m3", "_m", "_m2", "_m3", "_m_s", "_rad", "_rad_s", "_rad_s2", "_rpm", "_deg", "_v",
         "_a", "_ohm", "_w", "_s", "_hz", "_pa", "_gpa", "_mpa", "_pa_sqrt_m", "_c", "_k", "_rpm_per_v", "_nm_per_a",
         "_v_s_per_rad", "_j", "_wh", "_ah", "_per_k", "_n_m", "_n_per_m", "_nm_per_rad", "_1_k")

NUM, STR, LIST, DICT, BOOL = "number", "string", "list", "dict", "bool"


def _unit_ok(name: str) -> bool:
    return any(name.endswith(u) for u in UNITS)


# ----------------------------------------------------------------------------------------------------- the blocks
#: field → (type, required). Numeric fields are named with their unit; the dimensionless ones are listed in DIMLESS.
MOTOR = {"name": (STR, False), "kind": (STR, False), "stall_torque_nm": (NUM, False), "rated_torque_nm": (NUM, False),
         "no_load_speed_rpm": (NUM, False), "no_load_speed_rad_s": (NUM, False), "rated_speed_rpm": (NUM, False),
         "rated_speed_rad_s": (NUM, False), "voltage_v": (NUM, False), "stall_current_a": (NUM, False),
         "no_load_current_a": (NUM, False), "rated_current_a": (NUM, False), "max_current_a": (NUM, False),
         "resistance_ohm": (NUM, False), "inductance_h": (NUM, False), "kv_rpm_per_v": (NUM, False), "kt_nm_per_a": (NUM, False),
         "ke_v_s_per_rad": (NUM, False), "rotor_inertia_kgm2": (NUM, False), "mass_kg": (NUM, False),
         "electrical_time_constant_s": (NUM, False), "mechanical_time_constant_s": (NUM, False), "response_time_s": (NUM, False),
         "torque_speed": (DICT, False), "limits": (DICT, False), "transmission": (DICT, False), "control": (DICT, False),
         "source": (STR, False), "validity": (DICT, False)}
TRANSMISSION = {"type": (STR, False), "gear_ratio": (NUM, False), "efficiency": (NUM, False), "backlash_rad": (NUM, False),
                "stiffness_nm_per_rad": (NUM, False), "max_torque_nm": (NUM, False), "max_speed_rpm": (NUM, False),
                "max_speed_rad_s": (NUM, False), "reflected_inertia_kgm2": (NUM, False), "self_locking": (BOOL, False),
                "stages": (LIST, False), "source": (STR, False)}
LIMITS = {"max_torque_nm": (NUM, False), "max_speed_rpm": (NUM, False), "max_speed_rad_s": (NUM, False), "max_power_w": (NUM, False),
          "max_current_a": (NUM, False), "max_voltage_v": (NUM, False), "max_temperature_c": (NUM, False),
          "continuous_torque_nm": (NUM, False), "duty_cycle": (NUM, False)}
CURVE = {"speed_rpm": (LIST, False), "speed_rad_s": (LIST, False), "torque_nm": (LIST, False), "current_a": (LIST, False),
         "power_w": (LIST, False), "efficiency": (LIST, False), "voltage_v": (NUM, False), "source": (STR, False)}
DIMLESS = {"gear_ratio", "efficiency", "duty_cycle", "kF", "kM", "spin", "blades", "solidity", "ct", "cp", "cq", "eta",
           "CT", "CP", "CQ", "J", "safety_factor", "nu", "poisson", "margin", "residual", "unconverged_points",
           "rho"}                                                              # rho: the propulsion grid's density [kg/m³], kept for the maps Boreas wrote

#: block key → (where, schema, system tags)
BLOCKS = {
    "motor": ("joint", MOTOR, ("00motor",)),
    "motors": ("machine", {"*": (DICT, False)}, ("00motor",)),
    "propulsion": ("machine", {"rotors": (LIST, False), "propellers": (DICT, False), "motors": (DICT, False), "reduced": (DICT, False),
                               "battery": (DICT, False), "source": (STR, False), "validity": (DICT, False)}, ("00prop",)),
    "ogma": ("machine", {"rotors": (LIST, False), "source": (STR, False)}, ("00prop", "00motor")),
}
PROPELLER = {"name": (STR, False), "blades": (NUM, False), "diameter_m": (NUM, True), "hub_radius_m": (NUM, False), "pitch_m": (NUM, False),
             "stations": (LIST, False), "mass_kg": (NUM, False), "rotor_mass_kg": (NUM, False), "inertia_kgm2": (NUM, False),
             "coefficients": (DICT, False), "grid": (DICT, False), "geometry_ref": (STR, False), "spin": (NUM, False),
             "source": (STR, False), "validity": (DICT, False)}
COEFFICIENTS = {"J": (LIST, True), "CT": (LIST, True), "CP": (LIST, False), "CQ": (LIST, False), "eta": (LIST, False),
                "rho_ref_kg_m3": (NUM, False), "rpm_band": (LIST, False), "source": (STR, False)}
GRID = {"rpm": (LIST, True), "airspeed_m_s": (LIST, True), "rho": (NUM, False), "rho_kg_m3": (NUM, False), "thrust_n": (LIST, True),
        "torque_nm": (LIST, False), "power_w": (LIST, False), "current_a": (LIST, False), "voltage_v": (LIST, False),
        "efficiency": (LIST, False), "unconverged_points": (NUM, False), "power_limit": (DICT, False)}


# ------------------------------------------------------------------------------------------------------- checking
def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _check(d, schema: dict, where: str, p: list, *, nested: dict | None = None):
    """``d`` against ``schema``: types, required fields, unit suffixes on numbers; ``nested`` maps a field to the
    schema of its dictionary (checked in turn)."""
    if not isinstance(d, dict):
        p.append(f"{where}: a dictionary is required, got {type(d).__name__}")
        return
    star = schema.get("*")
    for k, (typ, req) in schema.items():
        if k == "*":
            continue
        if req and k not in d:
            p.append(f"{where}: {k} is required")
    for k, v in d.items():
        typ = schema.get(k, star)[0] if (k in schema or star) else None
        if typ is None:
            if _num(v) and not _unit_ok(k) and k not in DIMLESS:
                p.append(f"{where}: {k}: a number needs a unit suffix ({', '.join(UNITS[:8])}, …) or a place in the dimensionless list")
            continue
        ok = {NUM: _num, STR: lambda x: isinstance(x, str), LIST: lambda x: isinstance(x, (list, tuple)),
              DICT: lambda x: isinstance(x, dict), BOOL: lambda x: isinstance(x, bool)}[typ]
        if not ok(v):
            p.append(f"{where}: {k} must be a {typ}, got {type(v).__name__}")
            continue
        if typ == NUM and not _unit_ok(k) and k not in DIMLESS:
            p.append(f"{where}: {k}: a number needs a unit suffix")
        if nested and k in nested and isinstance(v, dict):
            _check(v, nested[k], f"{where}/{k}", p, nested=nested.get("__" + k))


def _same_length(d: dict, keys, where: str, p: list):
    lens = {k: len(d[k]) for k in keys if k in d and isinstance(d[k], (list, tuple))}
    if len(set(lens.values())) > 1:
        p.append(f"{where}: {', '.join(keys)} must have one length, got {lens}")
    for k in keys:
        if k in d and isinstance(d[k], (list, tuple)) and not all(_num(x) for x in d[k]):
            p.append(f"{where}: {k} must hold finite numbers")


def check_motor(d, where: str, p: list, catalogue: dict | None = None):
    if isinstance(d, str):                                                     # a name in the machine's motors catalogue
        if catalogue is None or d not in catalogue:
            p.append(f"{where}: motor {d!r} is not in data['motors']")
        return
    _check(d, MOTOR, where, p, nested={"transmission": TRANSMISSION, "limits": LIMITS, "torque_speed": CURVE})
    if not isinstance(d, dict):
        return
    if "no_load_speed_rpm" in d and "no_load_speed_rad_s" in d:
        p.append(f"{where}: give no_load_speed_rpm or no_load_speed_rad_s, not both")
    for k in ("stall_torque_nm", "rated_torque_nm", "no_load_speed_rpm", "no_load_speed_rad_s", "resistance_ohm", "rotor_inertia_kgm2"):
        if k in d and _num(d[k]) and d[k] < 0:
            p.append(f"{where}: {k} must be ≥ 0")
    ts = d.get("torque_speed")
    if isinstance(ts, dict):
        if not (("speed_rpm" in ts) ^ ("speed_rad_s" in ts)):
            p.append(f"{where}/torque_speed: speed_rpm or speed_rad_s is required (one of them)")
        _same_length(ts, ("speed_rpm", "speed_rad_s", "torque_nm", "current_a", "power_w", "efficiency"), f"{where}/torque_speed", p)
    tr = d.get("transmission")
    if isinstance(tr, dict):
        if _num(tr.get("efficiency")) and not 0 < tr["efficiency"] <= 1:
            p.append(f"{where}/transmission: efficiency must be in (0, 1]")
        if _num(tr.get("gear_ratio")) and tr["gear_ratio"] <= 0:
            p.append(f"{where}/transmission: gear_ratio must be > 0")


def check_propulsion(d, where: str, p: list, m: Machine):
    _check(d, BLOCKS["propulsion"][1], where, p)
    if not isinstance(d, dict):
        return
    props, mots = d.get("propellers") or {}, d.get("motors") or {}
    for name, pe in props.items():
        w = f"{where}/propellers/{name}"
        _check(pe, PROPELLER, w, p, nested={"coefficients": COEFFICIENTS, "grid": GRID})
        if not isinstance(pe, dict):
            continue
        co = pe.get("coefficients")
        if isinstance(co, dict):
            _same_length(co, ("J", "CT", "CP", "CQ", "eta"), f"{w}/coefficients", p)
            J = co.get("J")
            if isinstance(J, (list, tuple)) and len(J) > 1 and any(b <= a for a, b in zip(J, J[1:])):
                p.append(f"{w}/coefficients: J must be increasing")
        g = pe.get("grid")
        if isinstance(g, dict):
            for key in ("thrust_n", "torque_nm", "power_w", "current_a", "voltage_v"):
                tab = g.get(key)
                if isinstance(tab, (list, tuple)) and "rpm" in g and "airspeed_m_s" in g:
                    if len(tab) != len(g["rpm"]) or any(len(r) != len(g["airspeed_m_s"]) for r in tab):
                        p.append(f"{w}/grid: {key} must be rpm-major, {len(g['rpm'])} × {len(g['airspeed_m_s'])}")
        if "geometry_ref" in pe and not any(lk.name == pe["geometry_ref"] or any(s.name == pe["geometry_ref"] for s in lk.shapes) for lk in m.links):
            p.append(f"{w}: geometry_ref {pe['geometry_ref']!r} names no link or shape")
    for name, mo in mots.items():
        check_motor(mo, f"{where}/motors/{name}", p)
    names = {lk.name for lk in m.links}
    for i, r in enumerate(d.get("rotors") or []):
        w = f"{where}/rotors/{i}"
        if not isinstance(r, dict):
            p.append(f"{w}: a dictionary is required")
            continue
        if r.get("propeller") not in props:
            p.append(f"{w}: propeller {r.get('propeller')!r} is not in the block")
        if r.get("motor") not in mots:
            p.append(f"{w}: motor {r.get('motor')!r} is not in the block")
        if "link" in r and r["link"] not in names:
            p.append(f"{w}: link {r['link']!r} is not a link of the machine")
        for key in ("pos", "axis"):
            if key in r and not (isinstance(r[key], (list, tuple)) and len(r[key]) == 3 and all(_num(x) for x in r[key])):
                p.append(f"{w}: {key} must be three numbers (machine frame)")
        if "spin" in r and r["spin"] not in (1, -1):
            p.append(f"{w}: spin must be +1 or -1")


def problems(m: Machine) -> list:
    """What is wrong with the machine's engineering blocks: shapes, types, units, references, frames. ``[]`` when
    every known block is well formed. Tag syntax is ``yax.tags``'; the YX2 profile is ``yax.formats.yx2``'s."""
    p = []
    catalogue = m.data.get("motors")
    if catalogue is not None:
        if not isinstance(catalogue, dict):
            p.append("data:motors: a dictionary name → motor is required")
        else:
            for name, mo in catalogue.items():
                check_motor(mo, f"data:motors/{name}", p)
    for j in m.joints:
        if "motor" in j.data:
            check_motor(j.data["motor"], f"joint:{j.name}/data:motor", p, catalogue if isinstance(catalogue, dict) else None)
    if "propulsion" in m.data:
        check_propulsion(m.data["propulsion"], "data:propulsion", p, m)
    for extra in _EXTRA_CHECKS:
        extra(m, p)
    return p


_EXTRA_CHECKS = []                                                             # later stages register their checks here


# --------------------------------------------------------------------------------------------------------- losses
#: What a format keeps of each block; everything else of the block is a loss to report.
HOLDS = {"jsbsim": {"propulsion": "the propellers' CT/CP against J and the motors' power as a JSBSim propeller and electric engine"},
         "mjcf": {"motor": "nothing of the motor: the PD drive written is not the motor model (only the joint's kp, kv, force range)"},
         "urdf": {"motor": "nothing of the motor: the drive's max force as the effort limit, not the motor model"},
         "sdf": {"motor": "nothing of the motor: a Gazebo plugin would carry it; the joint's limits and damping are written"}}


def report_losses(m: Machine, fmt: str, rep: Report) -> None:
    """Say, explicitly, what ``fmt`` cannot carry of the engineering blocks (called by ``yax.save`` for every format
    but USD and YX2; the generic 'machine data has no place' lines stay for keys no schema knows)."""
    if fmt in ("usd", "yx2"):
        return
    holds = HOLDS.get(fmt, {})
    for j in m.joints:
        if "motor" in j.data:
            mo = j.data["motor"]
            what = "a catalogue motor" if isinstance(mo, str) else ", ".join(k for k in ("stall_torque_nm", "no_load_speed_rpm", "no_load_speed_rad_s",
                                                                                     "torque_speed", "transmission", "limits") if k in mo) or "its fields"
            rep.drop(f"joint {j.name}: its motor model ({what}) — {fmt} keeps {holds.get('motor', 'nothing of it')}")
    if "motors" in m.data and fmt not in ("usd", "yx2"):
        rep.drop(f"data:motors: the motor catalogue ({len(m.data['motors'])} motors) — {fmt} has no place for it")
    if "propulsion" in m.data:
        d = m.data["propulsion"]
        if fmt == "jsbsim":
            kept = holds["propulsion"]
            lost = [k for k in ("grid", "stations", "validity") if any(k in (pe or {}) for pe in (d.get("propellers") or {}).values())]
            if lost:
                rep.drop(f"data:propulsion: {fmt} keeps {kept}; the propellers' {', '.join(lost)} do not reach it")
        else:
            rep.drop(f"data:propulsion: the propellers' maps ({len(d.get('propellers') or {})} propellers, {len(d.get('rotors') or [])} rotors) — "
                     f"{fmt} has no place for them" + ("; its reduced constants reach a motor plugin through yax sync" if fmt == "sdf" and d.get("reduced") else ""))
