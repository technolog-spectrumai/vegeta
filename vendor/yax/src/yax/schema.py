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


MATERIAL = {"name": (STR, False), "class": (STR, False), "density_kg_m3": (NUM, False), "youngs_modulus_pa": (NUM, False),
            "poisson_ratio": (NUM, False), "shear_modulus_pa": (NUM, False), "yield_strength_pa": (NUM, False),
            "ultimate_strength_pa": (NUM, False), "compressive_strength_pa": (NUM, False), "shear_strength_pa": (NUM, False),
            "fracture_toughness_pa_sqrt_m": (NUM, False), "elongation_at_break": (NUM, False), "hardness": (NUM, False),
            "anisotropy": (STR, False), "orthotropic": (DICT, False), "composite": (DICT, False), "failure": (DICT, False),
            "thermal": (DICT, False), "source": (STR, False), "validity": (DICT, False)}
ORTHOTROPIC = {"E1_pa": (NUM, False), "E2_pa": (NUM, False), "E3_pa": (NUM, False), "G12_pa": (NUM, False), "G13_pa": (NUM, False),
               "G23_pa": (NUM, False), "nu12": (NUM, False), "nu13": (NUM, False), "nu23": (NUM, False),
               "Xt_pa": (NUM, False), "Xc_pa": (NUM, False), "Yt_pa": (NUM, False), "Yc_pa": (NUM, False), "S_pa": (NUM, False)}
COMPOSITE = {"fibre": (STR, False), "matrix": (STR, False), "fibre_volume_fraction": (NUM, False), "layup": (LIST, False),
             "ply_thickness_m": (NUM, False), "reference_axis": (LIST, False), "frame": (STR, False), "source": (STR, False)}
FAILURE = {"criterion": (STR, True), "parameters": (DICT, False), "calibrated": (BOOL, False), "source": (STR, False), "validity": (DICT, False)}
STRENGTH = {"material": (STR, False), "allowable_stress_pa": (NUM, False), "allowable_strain": (NUM, False), "safety_factor": (NUM, False),
            "limit_load_n": (NUM, False), "ultimate_load_n": (NUM, False), "limit_moment_nm": (NUM, False), "ultimate_moment_nm": (NUM, False),
            "fatigue": (DICT, False), "analysis": (DICT, False), "source": (STR, False), "validity": (DICT, False)}
ANALYSIS = {"tool": (STR, False), "kind": (STR, False), "case": (STR, False), "max_stress_pa": (NUM, False), "max_displacement_m": (NUM, False),
            "margin": (NUM, False), "date": (STR, False), "source": (STR, False)}
CONNECTION = {"name": (STR, True), "type": (STR, True), "links": (LIST, True), "frame": (STR, False), "pos": (LIST, False), "quat": (LIST, False),
              "axis": (LIST, False), "geometry": (DICT, False), "fasteners": (DICT, False), "material": (STR, False),
              "limits": (DICT, False), "detachable": (BOOL, False), "source": (STR, False), "validity": (DICT, False)}
CONNECTION_TYPES = ("bolt", "screw", "rivet", "weld", "bond", "pin", "press", "snap", "clamp", "magnet", "detachable", "other")
CONN_GEOMETRY = {"diameter_m": (NUM, False), "length_m": (NUM, False), "area_m2": (NUM, False), "width_m": (NUM, False),
                 "thickness_m": (NUM, False), "pattern": (LIST, False), "count": (NUM, False), "shape": (STR, False)}
FASTENERS = {"count": (NUM, False), "size": (STR, False), "grade": (STR, False), "material": (STR, False), "preload_n": (NUM, False),
             "torque_nm": (NUM, False), "pitch_m": (NUM, False), "source": (STR, False)}
CONN_LIMITS = {"tension_n": (NUM, False), "compression_n": (NUM, False), "shear_n": (NUM, False), "moment_nm": (NUM, False),
               "torque_nm": (NUM, False), "energy_j": (NUM, False), "safety_factor": (NUM, False), "basis": (STR, False)}
DIMLESS |= {"poisson_ratio", "nu12", "nu13", "nu23", "elongation_at_break", "hardness", "fibre_volume_fraction", "allowable_strain", "count"}
BLOCKS.update({
    "materials": ("machine", {"*": (DICT, False)}, ("00mat",)),
    "material": ("link", MATERIAL, ("00mat",)),
    "strength": ("link|machine", STRENGTH, ("00str",)),
    "connections": ("machine", {"*": (DICT, False)}, ("00conn",)),
})


def check_material(d, where: str, p: list):
    _check(d, MATERIAL, where, p, nested={"orthotropic": ORTHOTROPIC, "composite": COMPOSITE, "failure": FAILURE})
    if not isinstance(d, dict):
        return
    if _num(d.get("poisson_ratio")) and not -1 < d["poisson_ratio"] < 0.5:
        p.append(f"{where}: poisson_ratio must be in (-1, 0.5)")
    for k in ("density_kg_m3", "youngs_modulus_pa", "shear_modulus_pa", "yield_strength_pa", "ultimate_strength_pa", "fracture_toughness_pa_sqrt_m"):
        if _num(d.get(k)) and d[k] <= 0:
            p.append(f"{where}: {k} must be > 0")
    if d.get("anisotropy") not in (None, "isotropic", "orthotropic", "transversely isotropic", "anisotropic"):
        p.append(f"{where}: anisotropy must be isotropic, orthotropic, transversely isotropic or anisotropic")
    comp = d.get("composite")
    if isinstance(comp, dict) and isinstance(comp.get("layup"), (list, tuple)):
        for i, ply in enumerate(comp["layup"]):
            if not isinstance(ply, dict) or not _num(ply.get("angle_deg")):
                p.append(f"{where}/composite/layup/{i}: a ply is {{'angle_deg', 'thickness_m'?, 'material'?}}")


def check_strength(d, where: str, p: list, materials: dict):
    _check(d, STRENGTH, where, p, nested={"analysis": ANALYSIS})
    if isinstance(d, dict):
        if "material" in d and d["material"] not in materials:
            p.append(f"{where}: material {d['material']!r} is not in data['materials']")
        if _num(d.get("safety_factor")) and d["safety_factor"] <= 0:
            p.append(f"{where}: safety_factor must be > 0")


def check_connection(name, d, where: str, p: list, m: Machine, materials: dict):
    _check(d, CONNECTION, where, p, nested={"geometry": CONN_GEOMETRY, "fasteners": FASTENERS, "limits": CONN_LIMITS})
    if not isinstance(d, dict):
        return
    if d.get("name") != name:
        p.append(f"{where}: name must be {name!r} (its key)")
    if d.get("type") not in CONNECTION_TYPES:
        p.append(f"{where}: type must be one of {CONNECTION_TYPES}")
    names = {lk.name for lk in m.links}
    links = d.get("links")
    if isinstance(links, (list, tuple)):
        if len(links) != 2 or not all(isinstance(x, str) for x in links):
            p.append(f"{where}: links must name two links")
        else:
            for x in links:
                if x not in names:
                    p.append(f"{where}: link {x!r} is not a link of the machine")
    fr = d.get("frame", "machine")
    if fr != "machine" and not (fr.startswith("link:") and fr[5:] in names):
        p.append(f"{where}: frame must be 'machine' or 'link:<name>' of a link")
    for key, n in (("pos", 3), ("quat", 4), ("axis", 3)):
        if key in d and not (isinstance(d[key], (list, tuple)) and len(d[key]) == n and all(_num(x) for x in d[key])):
            p.append(f"{where}: {key} must be {n} numbers")
    if "material" in d and d["material"] not in materials:
        p.append(f"{where}: material {d['material']!r} is not in data['materials']")
    fa = d.get("fasteners")
    if isinstance(fa, dict) and "material" in fa and fa["material"] not in materials and not isinstance(fa["material"], str):
        p.append(f"{where}/fasteners: material must be a name")


def _stage2(m: Machine, p: list):
    materials = m.data.get("materials")
    if materials is not None and not isinstance(materials, dict):
        p.append("data:materials: a dictionary name → material is required")
        materials = {}
    materials = materials or {}
    for name, d in materials.items():
        check_material(d, f"data:materials/{name}", p)
        if isinstance(d, dict) and "name" in d and d["name"] != name:
            p.append(f"data:materials/{name}: name must be {name!r} (its key)")
    for lk in m.links:
        if "material" in lk.data:
            mat = lk.data["material"]
            if isinstance(mat, str):
                if mat not in materials:
                    p.append(f"link:{lk.name}/data:material: material {mat!r} is not in data['materials']")
            else:
                check_material(mat, f"link:{lk.name}/data:material", p)
        if "strength" in lk.data:
            check_strength(lk.data["strength"], f"link:{lk.name}/data:strength", p, materials)
    if "strength" in m.data:
        check_strength(m.data["strength"], "data:strength", p, materials)
    conns = m.data.get("connections")
    if conns is not None:
        if not isinstance(conns, dict):
            p.append("data:connections: a dictionary name → connection is required")
        else:
            for name, d in conns.items():
                check_connection(name, d, f"data:connections/{name}", p, m, materials)


AERO = {"name": (STR, False), "reference": (DICT, False), "frame": (STR, False), "reference_point_m": (LIST, False),
        "center_of_pressure_m": (LIST, False), "axes": (DICT, False), "coefficients": (DICT, False), "derivatives": (DICT, False),
        "tables": (LIST, False), "method": (STR, False), "source": (STR, False), "validity": (DICT, False), "assumptions": (LIST, False)}
AERO_REFERENCE = {"area_m2": (NUM, False), "length_m": (NUM, False), "span_m": (NUM, False), "chord_m": (NUM, False),
                  "diameter_m": (NUM, False), "source": (STR, False)}
AERO_METHODS = ("analytic", "cfd", "experiment", "manufacturer", "assumed", "fitted")
AERO_INPUTS = ("alpha_deg", "beta_deg", "re", "mach", "v_m_s", "rpm", "flap_deg", "J")
AERO_COEFFICIENTS = ("CD", "CL", "CY", "Cl", "Cm", "Cn", "CX", "CZ", "CA", "CN", "CD0", "CL0", "Cm0", "k", "e", "CLalpha", "CLmax")
DIMLESS |= set(AERO_COEFFICIENTS) | {"re", "mach", "alpha0", "oswald"}
BLOCKS.update({"aerodynamics": ("machine", AERO, ("00aero",)), "aero": ("link", AERO, ("00aero",))})


def check_aero(d, where: str, p: list, m: Machine):
    _check(d, AERO, where, p, nested={"reference": AERO_REFERENCE})
    if not isinstance(d, dict):
        return
    names = {lk.name for lk in m.links}
    fr = d.get("frame", "machine")
    if fr != "machine" and not (isinstance(fr, str) and fr.startswith("link:") and fr[5:] in names):
        p.append(f"{where}: frame must be 'machine' or 'link:<name>' of a link")
    for key in ("reference_point_m", "center_of_pressure_m"):
        if key in d and not (isinstance(d[key], (list, tuple)) and len(d[key]) == 3 and all(_num(x) for x in d[key])):
            p.append(f"{where}: {key} must be three numbers in {fr}")
    if "method" in d and d["method"] not in AERO_METHODS:
        p.append(f"{where}: method must be one of {AERO_METHODS}")
    co = d.get("coefficients")
    if isinstance(co, dict):
        for k, v in co.items():
            if not _num(v):
                p.append(f"{where}/coefficients: {k} must be a number (a table goes in 'tables')")
    de = d.get("derivatives")
    if isinstance(de, dict):
        for k, v in de.items():
            if not _num(v):
                p.append(f"{where}/derivatives: {k} must be a number")
    for i, t in enumerate(d.get("tables") or []):
        w = f"{where}/tables/{i}"
        if not isinstance(t, dict) or not isinstance(t.get("inputs"), dict) or not isinstance(t.get("values"), dict):
            p.append(f"{w}: a table is {{'name'?, 'inputs': {{axis: [...]}}, 'values': {{coefficient: nested lists}}, 'source'?, 'validity'?}}")
            continue
        axes = list(t["inputs"])
        bad = [a for a in axes if a not in AERO_INPUTS]
        if bad:
            p.append(f"{w}: inputs {bad} are not among {AERO_INPUTS}")
        shape = []
        for a in axes:
            v = t["inputs"][a]
            if not (isinstance(v, (list, tuple)) and len(v) > 0 and all(_num(x) for x in v)):
                p.append(f"{w}/inputs/{a}: a list of numbers is required")
                break
            if len(v) > 1 and any(b <= c for c, b in zip(v, v[1:])):
                p.append(f"{w}/inputs/{a}: must be increasing")
            shape.append(len(v))
        else:
            for k, arr in t["values"].items():
                if _shape(arr) != shape:
                    p.append(f"{w}/values/{k}: shape must be {shape} ({' × '.join(axes)}), got {_shape(arr)}")


def _shape(arr) -> list:
    out = []
    while isinstance(arr, (list, tuple)):
        out.append(len(arr))
        if not arr:
            break
        arr = arr[0]
    return out


def _stage3(m: Machine, p: list):
    if "aerodynamics" in m.data:
        check_aero(m.data["aerodynamics"], "data:aerodynamics", p, m)
    for lk in m.links:
        if "aero" in lk.data:
            check_aero(lk.data["aero"], f"link:{lk.name}/data:aero", p, m)


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


_EXTRA_CHECKS = [_stage2, _stage3]                                                      # the later stages' checks


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
    for key, what in (("materials", "the materials catalogue"), ("connections", "the connections"), ("strength", "the machine's strength data")):
        if key in m.data:
            n = len(m.data[key]) if isinstance(m.data[key], dict) else 1
            rep.drop(f"data:{key}: {what} ({n} entries) — {fmt} has no place for it")
    for lk in m.links:
        for key, what in (("material", "its material"), ("strength", "its strength limits")):
            if key in lk.data:
                rep.drop(f"link {lk.name}: {what} — {fmt} has no place for it")
    def _aero_what(d):
        n_t = len(d.get("tables") or [])
        n_c = len(d.get("coefficients") or {}) + len(d.get("derivatives") or {})
        return f"{n_c} coefficients" + (f", {n_t} tables" if n_t else "")
    if "aerodynamics" in m.data and fmt != "jsbsim":
        rep.drop(f"data:aerodynamics: the aerodynamics ({_aero_what(m.data['aerodynamics'])}) — {fmt} has no place for it"
                 + ("; a table is not a constant: none was substituted" if (m.data["aerodynamics"].get("tables")) else ""))
    if "aerodynamics" in m.data and fmt == "jsbsim":
        rep.drop(f"data:aerodynamics: the aerodynamics block ({_aero_what(m.data['aerodynamics'])}) is not written as JSBSim aerodynamics "
                 "(only an aircraft's own verbatim sections are); the block stays in the YX2")
    for lk in m.links:
        if "aero" in lk.data:
            rep.drop(f"link {lk.name}: its aerodynamics ({_aero_what(lk.data['aero'])}) — {fmt} has no place for it")
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
