"""Propeller maps and properties as part data: the ``propulsion`` block (docs/converter.md, "Propulsion").

A design tool (Vegeta's Boreas) computes a propeller's map; the block carries it in the machine's data, layered so a
consumer takes the deepest layer its engine can use::

    {"rotors": [{"name": "rotor_fl", "link": "rotor_fl", "pos": [x, y, z], "axis": [0, 0, 1], "spin": 1,
                 "propeller": "apc_5x4p3", "motor": "2306_2400kv"}],
     "propellers": {"apc_5x4p3": {"blades": 2, "diameter_m": 0.127, "mass_kg": 0.004, "inertia_kgm2": 1e-6,
                                  "coefficients": {"J": [...], "CT": [...], "CP": [...]},
                                  "grid": {"rpm": [...], "airspeed_m_s": [...], "rho": 1.225,
                                           "thrust_n": [[...]], "torque_nm": [[...]], "power_w": [[...]]}}},
     "motors": {"2306_2400kv": {"power_w": 300.0, "kv": 2400, ...}},
     "reduced": {"rotor_fl": {"kF": ..., "kM": ..., "w_max": ..., "tau_up": ..., "tau_down": ..., "taken_at": "hover"}}}

``coefficients`` are the propeller's thrust and power coefficients against the advance ratio (CT = T / ρ n² D⁴,
CP = P / ρ n³ D⁵, J = V / n D with n in rev/s): the density-invariant form flight simulators take. ``grid`` is the
dimensional rpm × airspeed map as the design tool made it; when there are no coefficients, ``coefficients()``
collapses the grid onto J (one value per distinct J, averaged) and says so. ``reduced`` holds the constants of the
two-constant motor model (T = kF ω², Q = kM T) that Gazebo and PX4 take, with where they were taken.
"""
from __future__ import annotations

import math

import numpy as np

KEY = "propulsion"


def block(m) -> dict | None:
    return m.data.get(KEY) or None


def rotors(blk: dict) -> list:
    """The rotors with their names (``name``, else ``link``, else ``rotor<i>``)."""
    out = []
    for i, r in enumerate(blk.get("rotors", [])):
        r = dict(r)
        r["name"] = r.get("name") or r.get("link") or f"rotor{i}"
        out.append(r)
    return out


def coefficients(prop: dict) -> tuple[dict, str | None]:
    """``{"J", "CT", "CP"}`` of a propeller entry and a note when they were collapsed from its grid (None when the
    entry gives them). Raises ``ValueError`` when the entry has neither."""
    c = prop.get("coefficients")
    if c:
        J, CT, CP = (np.asarray(c[k], float) for k in ("J", "CT", "CP"))
        if not (len(J) == len(CT) == len(CP) >= 2) or np.any(np.diff(J) <= 0):
            raise ValueError("propulsion coefficients: J must be increasing, with CT and CP of the same length")
        return {"J": J.tolist(), "CT": CT.tolist(), "CP": CP.tolist()}, None
    g = prop.get("grid")
    if not g:
        raise ValueError(f"propeller {prop.get('name', '')!r}: neither coefficients nor a grid")
    D, rho = float(prop["diameter_m"]), float(g.get("rho", 1.225))
    rpm, V = np.asarray(g["rpm"], float), np.asarray(g["airspeed_m_s"], float)
    T = np.asarray(g["thrust_n"], float)
    if "power_w" in g:
        P = np.asarray(g["power_w"], float)
    else:
        P = np.asarray(g["torque_nm"], float) * (rpm * 2 * math.pi / 60)[:, None]
    n = (rpm / 60.0)[:, None]
    Jg = V[None, :] / (n * D)
    CTg, CPg = T / (rho * n ** 2 * D ** 4), P / (rho * n ** 3 * D ** 5)
    J, inv = np.unique(np.round(Jg.ravel(), 12), return_inverse=True)
    CT = np.bincount(inv, CTg.ravel()) / np.bincount(inv)
    CP = np.bincount(inv, CPg.ravel()) / np.bincount(inv)
    note = (f"its {len(rpm)} × {len(V)} rpm × airspeed grid collapsed to CT(J), CP(J) ({len(J)} points, "
            f"rpm {rpm.min():g}–{rpm.max():g}, ρ = {rho:g})")
    return {"J": J.tolist(), "CT": CT.tolist(), "CP": CP.tolist()}, note


def motor_power(blk: dict, r: dict, prop: dict) -> tuple[float, str | None]:
    """The rotor's motor power [W] (``motors[<name>].power_w``), else the map's maximum shaft power with a note."""
    mot = blk.get("motors", {}).get(r.get("motor") or "", {})
    if mot.get("power_w"):
        return float(mot["power_w"]), None
    g = prop.get("grid") or {}
    if "power_w" in g:
        P = float(np.max(g["power_w"]))
    elif "torque_nm" in g:
        P = float(np.max(np.asarray(g["torque_nm"], float) * (np.asarray(g["rpm"], float) * 2 * math.pi / 60)[:, None]))
    else:
        raise ValueError(f"rotor {r['name']}: no motor power (motors[...].power_w) and no grid to take it from")
    return P, f"rotor {r['name']}: its motor power assumed as the map's maximum shaft power ({P:g} W)"


def report_reduction(m, rep, target: str = "SDF") -> None:
    """Say, per rotor, how the propeller map reaches a format with no propeller element: through its reduced
    constants (a motor plugin carries them), or not at all."""
    blk = block(m)
    if not blk:
        return
    red = blk.get("reduced", {})
    rs = rotors(blk)
    for r in rs:
        k = red.get(r["name"])
        if k:
            where = f" at {k['taken_at']}" if k.get("taken_at") else ""
            res = f", residual {k['residual']:.3g}" if isinstance(k.get("residual"), (int, float)) else ""
            rep.drop(f"propulsion: rotor {r['name']}: its propeller map reduced to kF = {k.get('kF', 0):.4g}, "
                     f"kM = {k.get('kM', 0):.4g}{where}{res} ({target} has no propeller element; a motor plugin "
                     f"carries the constants — map them with yax sync)")
        else:
            rep.drop(f"propulsion: rotor {r['name']}: its propeller map ({target} has no propeller element and no "
                     f"reduced constants were given: add propulsion.reduced[{r['name']!r}] for a motor plugin)")
    if not rs:
        names = ", ".join(blk.get("propellers", {})) or "—"
        rep.drop(f"propulsion: propellers {names} ({target} has no propeller element)")
