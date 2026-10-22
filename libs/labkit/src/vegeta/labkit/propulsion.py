"""Propeller maps as Yax part data: the ``propulsion`` block (vendor/yax, docs/converter.md "Propulsion") built from
what the labs compute — Boreas BEMT maps, Ogma propulsion maps and electric drives — so a machine's rotors leave Vegeta
with their geometry, their coefficients against the advance ratio, the dimensional grid, the motor and the reduced
two-constant model. Yax keeps the whole block in USD, writes the coefficients as a JSBSim propeller, reports the
reduction for SDF (a Gazebo motor plugin takes ``reduced`` through ``yax sync``) and the loss everywhere else.

Layers, deepest first — a consumer takes the deepest one its engine can use::

    rotors      name, link, pos and axis (machine frame), spin (+1: drag torque on the frame along +axis), propeller, motor
    propellers  blades, diameter, hub radius, stations, mass, inertia about the shaft, coefficients {J, CT, CP}, grid
    motors      power_w (what a JSBSim electric engine takes), kv_rpm_per_v, resistance_ohm, no_load_current_a, max_current_a, voltage_v, esc_efficiency
    reduced     per rotor: kF (T = kF ω²), kM (Q = kM T), w_max, tau_up, tau_down, taken_at, residual

A **grid** is ``{"rpm": [...], "airspeed_m_s": [...], "rho": ρ, "thrust_n": [[...]], "torque_nm": [[...]],
"power_w": [[...]]}`` with the 2-D tables rpm-major (``[i_rpm][j_airspeed]``), as Boreas' ``performance_map`` makes it;
``grid_from_map`` and ``grid_from_drive`` turn Ogma's airspeed-major tables around. Coefficients: CT = T / ρ n² D⁴,
CP = P / ρ n³ D⁵, J = V / n D (n in rev/s), one value per distinct J (averaged over rpm: Boreas has no Reynolds
effect, so the collapse is exact at J = 0 and a small blur elsewhere).
"""
from __future__ import annotations

import math

import numpy as np

KEY = "propulsion"


# ---------------------------------------------------------------------------------------------------------- grids
def grid_from_boreas(prop, airfoil, rpms, airspeeds, rho: float = 1.225) -> dict:
    """Boreas BEMT over ``rpms`` × ``airspeeds`` (``vegeta.boreas.export.performance_map``) as a grid."""
    from vegeta.boreas.export import performance_map
    pm = performance_map(prop, airfoil, rpms, airspeeds, rho)
    return {"rpm": pm["rpm"], "airspeed_m_s": pm["airspeed_m_s"], "rho": pm["rho"], "thrust_n": pm["thrust_n"],
            "torque_nm": pm["torque_nm"], "power_w": pm["power_w"], "unconverged_points": pm["unconverged_points"]}


def grid_from_map(pm, rho: float = 1.225) -> dict:
    """An Ogma ``PropulsionMap`` (airspeed-major: thrust, shaft power, torque, current, voltage over V × rpm) as a grid
    (rpm-major), the motor's current and voltage riding along."""
    return {"rpm": np.asarray(pm.rpm, float).tolist(), "airspeed_m_s": np.asarray(pm.V, float).tolist(), "rho": float(rho),
            "thrust_n": np.asarray(pm.thrust, float).T.tolist(), "torque_nm": np.asarray(pm.torque, float).T.tolist(),
            "power_w": np.asarray(pm.shaft_power, float).T.tolist(), "current_a": np.asarray(pm.current, float).T.tolist(),
            "voltage_v": np.asarray(pm.voltage, float).T.tolist()}


def grid_from_drive(dr) -> dict:
    """An Ogma ``ElectricDrive`` (signed thrust and torque at sea level over V × rpm) as a grid at sea-level density:
    the propeller alone, the brake region included (negative thrust and torque are what the windmill gives)."""
    from ogma.propulsion import RHO0
    V, rpm = np.asarray(dr.V, float), np.asarray(dr.rpm, float)
    T, Q = np.asarray(dr.thrust0, float), np.asarray(dr.torque0, float)
    P = Q * (rpm * 2 * math.pi / 60)[None, :]
    return {"rpm": rpm.tolist(), "airspeed_m_s": V.tolist(), "rho": float(RHO0), "thrust_n": T.T.tolist(), "torque_nm": Q.T.tolist(),
            "power_w": P.T.tolist()}


def at(grid: dict, key: str, rpm: float, airspeed: float = 0.0) -> float:
    """Bilinear interpolation of a grid table at (rpm, airspeed), clipped to the grid."""
    r, v = np.asarray(grid["rpm"], float), np.asarray(grid["airspeed_m_s"], float)
    tab = np.asarray(grid[key], float)
    rows = np.array([np.interp(airspeed, v, tab[i]) for i in range(len(r))])
    return float(np.interp(rpm, r, rows))


def rpm_for_thrust(grid: dict, thrust: float, airspeed: float = 0.0) -> float:
    """The rpm at which the grid gives ``thrust`` at ``airspeed`` (the thrust column must rise with rpm)."""
    r, v = np.asarray(grid["rpm"], float), np.asarray(grid["airspeed_m_s"], float)
    tab = np.asarray(grid["thrust_n"], float)
    col = np.array([np.interp(airspeed, v, tab[i]) for i in range(len(r))])
    order = np.argsort(col)
    return float(np.interp(thrust, col[order], r[order]))


# --------------------------------------------------------------------------------------------------- coefficients
def coefficients(grid: dict, diameter_m: float) -> dict:
    """``{"J", "CT", "CP"}`` of a grid: one value per distinct advance ratio, averaged over the rpm that share it."""
    D, rho = float(diameter_m), float(grid["rho"])
    rpm, V = np.asarray(grid["rpm"], float), np.asarray(grid["airspeed_m_s"], float)
    T, P = np.asarray(grid["thrust_n"], float), np.asarray(grid["power_w"], float)
    n = (rpm / 60.0)[:, None]
    Jg = V[None, :] / (n * D)
    CTg, CPg = T / (rho * n ** 2 * D ** 4), P / (rho * n ** 3 * D ** 5)
    J, inv = np.unique(np.round(Jg.ravel(), 12), return_inverse=True)
    cnt = np.bincount(inv)
    return {"J": J.tolist(), "CT": (np.bincount(inv, CTg.ravel()) / cnt).tolist(), "CP": (np.bincount(inv, CPg.ravel()) / cnt).tolist()}


# ------------------------------------------------------------------------------------------------------- entries
def propeller_entry(prop, grid: dict, *, inertia_kgm2: float | None = None, source: str = "") -> dict:
    """A ``propellers`` entry from a Boreas ``Propeller`` (its geometry) and a grid (its map and coefficients).
    ``inertia_kgm2`` is about the shaft; by default the blade pair as a thin rod, m R² / 3."""
    d = prop.describe()
    R = float(prop.radius)
    inertia = float(prop.mass_kg) * R * R / 3 if inertia_kgm2 is None else float(inertia_kgm2)
    return {"name": d["name"], "blades": int(prop.blades), "diameter_m": float(prop.diameter), "hub_radius_m": float(prop.hub_radius),
            "stations": d["stations"], "mass_kg": float(prop.mass_kg), "rotor_mass_kg": float(prop.rotor_mass_kg), "inertia_kgm2": inertia,
            "coefficients": coefficients(grid, prop.diameter), "grid": grid, "source": source or d.get("notes", "")}


def motor_entry(name: str, power_w: float, *, kv: float | None = None, R_ohm: float | None = None, I0_A: float | None = None,
                I_max_A: float | None = None, battery_v: float | None = None, esc_eff: float | None = None, source: str = "") -> dict:
    """A ``motors`` entry: ``power_w`` (the shaft power a JSBSim electric engine delivers at full throttle) and the
    DC-motor constants a consumer modelling the electric chain needs."""
    out = {"name": name, "kind": "bldc", "limits": {"max_power_w": float(power_w)}, "power_w": float(power_w)}
    for k, v in (("kv_rpm_per_v", kv), ("resistance_ohm", R_ohm), ("no_load_current_a", I0_A), ("max_current_a", I_max_A),
                 ("voltage_v", battery_v), ("esc_efficiency", esc_eff)):                 # the contract's names (vendor/yax docs/engineering.md)
        if v is not None:
            out[k] = float(v)
    if source:
        out["source"] = source
    return out


def max_shaft_power(grid: dict, *, battery_v: float | None = None, i_max: float | None = None,
                    voltage_for=None) -> tuple[float, float]:
    """The largest shaft power [W] the grid reaches at V = 0 within the motor's voltage and current, and its rpm.
    The voltage and current come from the grid's own columns (``grid_from_map``) or from ``voltage_for(rpm, torque)
    → (V, I)`` (a Boreas ``Motor``); with neither, the grid's maximum."""
    rpm = np.asarray(grid["rpm"], float)
    P = np.asarray(grid["power_w"], float)[:, 0]
    Q = np.asarray(grid["torque_nm"], float)[:, 0]
    ok = np.ones(len(rpm), bool)
    if "voltage_v" in grid:
        Vn, I = np.asarray(grid["voltage_v"], float)[:, 0], np.asarray(grid["current_a"], float)[:, 0]
    elif voltage_for is not None:
        Vn, I = (np.array(x, float) for x in zip(*(voltage_for(r, q) for r, q in zip(rpm, Q))))
    else:
        Vn = I = None
    if Vn is not None:
        if battery_v is not None:
            ok &= Vn <= battery_v
        if i_max is not None:
            ok &= I <= i_max
    if not ok.any():
        raise ValueError("no grid point within the motor's voltage and current at V = 0")
    i = int(np.flatnonzero(ok)[np.argmax(P[ok])])
    return float(P[i]), float(rpm[i])


def reduced(grid: dict, *, rpm: float, taken_at: str, tau_up: float | None = None, tau_down: float | None = None,
            check: tuple[float, float] | None = None, kF: float | None = None, kM: float | None = None) -> dict:
    """The two-constant model at ``rpm`` (V = 0): kF = T / ω², kM = Q / T, ω_max = ω — or the given ``kF``/``kM`` — with
    the time constants and, when ``check = (rpm, airspeed)`` is given, the residual |kF ω² − T| / T of the model
    against the grid there (how far the constant model is from the map at that point)."""
    w = rpm * 2 * math.pi / 60
    T, Q = at(grid, "thrust_n", rpm), at(grid, "torque_nm", rpm)
    out = {"kF": float(T / w ** 2) if kF is None else float(kF), "kM": float(Q / T) if kM is None else float(kM), "w_max": float(w),
           "taken_at": taken_at}
    for k, v in (("tau_up", tau_up), ("tau_down", tau_down)):
        if v is not None:
            out[k] = float(v)
    if check is not None:
        rc, vc = check
        wc = rc * 2 * math.pi / 60
        Tc = at(grid, "thrust_n", rc, vc)
        out["residual"] = float(abs(out["kF"] * wc * wc - Tc) / Tc) if Tc else None
        out["residual_at"] = f"{rc:g} rpm, {vc:g} m/s"
    return out


def rotor(name: str, pos, axis, spin: int, propeller: str, motor: str, link: str | None = None) -> dict:
    a = np.asarray(axis, float)
    out = {"name": name, "pos": [float(x) for x in pos], "axis": (a / np.linalg.norm(a)).tolist(), "spin": int(spin),
           "propeller": propeller, "motor": motor}
    if link:
        out["link"] = link
    return out


def block(rotors: list, propellers: list, motors: list, *, reduced: dict | None = None, battery: dict | None = None,
          source: str = "") -> dict:
    """The block, propellers and motors keyed by name; checks every rotor names a propeller and a motor it holds."""
    props, mots = {p["name"]: p for p in propellers}, {m["name"]: m for m in motors}
    for r in rotors:
        if r["propeller"] not in props or r["motor"] not in mots:
            raise KeyError(f"rotor {r['name']}: propeller {r['propeller']!r} or motor {r['motor']!r} not in the block")
    out = {"rotors": rotors, "propellers": props, "motors": mots}
    if reduced:
        out["reduced"] = reduced
    if battery:
        out["battery"] = battery
    if source:
        out["source"] = source
    return out


def in_machine_frame(m, pos_body, axis_body) -> tuple[list, list]:
    """A rotor's position and axis given in the root link's frame, in the Yax machine frame (the root at q = 0)."""
    from yax import hub
    root = m.roots()[0]
    Rr = hub.R(root.quat)
    return (np.asarray(root.pos, float) + Rr @ np.asarray(pos_body, float)).tolist(), (Rr @ np.asarray(axis_body, float)).tolist()
