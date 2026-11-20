"""ASTOR's flight model: ARIGO-1's machinery (``arigo1_model.build``: the vortex lattice + hand estimates) on ASTOR's
wing and cruciform tail (``astor``), with its own mass budget, pack and propeller.

Sources, marked in the output:

- **lattice** (``ogma.aero.vlm``): CL0, CL_alpha, the neutral point, the elevator's power;
- **CFD** (``data/astor_cfd_screening.yaml``, when ``sims/astor_cfd.sh`` has run): CL, CD, Cm over alpha replace the
  lattice's and the hand polar's; the CG is moved in the table if it has moved since;
- **propulsor library** (``propulsor_maps``): the propeller's full-throttle thrust over speed at the motor's rating;
- **hand** (HAND, with margins): CD0 0.028 +-0.006 until the CFD (a slim 70 mm body and a thin 9 % wing: below
  ARIGO-1's 0.032), the thin section's maximum lift (0.92 of the family's value, +-0.05), the damping and lateral
  derivatives, the inertias, the masses in ``astor.ITEMS``.
"""
from __future__ import annotations

import math
from functools import lru_cache

import arigo1_model as am
import astor as A

CD0 = 0.028                        # HAND until the CFD
THIN_CLMAX = 0.92                  # HAND: a 9 % section with 1.2 % camber against the family's 12 % / 2 %
CRUISE = 30.0                      # transit and the stability tests
DASH = 65.0                        # the design dash
PROP_SIZES_IN = (9.0,)             # the airframe is drawn for a 9-inch folding propeller


def propulsor(mass_kg: float, power_w: float = A.MOTOR_W) -> dict:
    """The propeller from the library: among the 2-blade 9-inch entries, the most thrust at the dash, with a static
    thrust of at least the weight (the rail gives the speed, the propeller must hold the climb-out)."""
    import arigo_propulsion as ap
    import propulsor_maps as pm
    best = None
    for e in pm.load()["entries"]:
        if e["kind"] != "propeller" or e.get("blades") != 2 or e["diameter_in"] not in PROP_SIZES_IN:
            continue
        u = ap._unit(e["id"], power_w, "pusher")
        T0, Td = u.full(0.0)[0], u.full(DASH)[0]
        if T0 < mass_kg * am.G:
            continue
        if best is None or Td > best[0]:
            best = (Td, e["id"], T0, u)
    if best is None:
        raise ValueError("no 9-inch library propeller holds the weight statically at this power")
    Td, eid, T0, u = best
    return {"entry": eid, "layout": "pusher", "units": 1, "power_w_per_unit": power_w, "static_thrust_N": round(T0, 1),
            "thrust_at_dash_N": round(Td, 1), "dash_speed_m_s": DASH, "cruise_speed_m_s": CRUISE,
            "cruise_power_W": None,
            "why": f"most thrust at {DASH:.0f} m/s among the library's 9-inch 2-blade propellers with static thrust >= weight, at {power_w:.0f} W"}


@lru_cache(maxsize=None)
def coefficients(pack: str | None = None, power_w: float = A.MOTOR_W) -> dict:
    import arigo_propulsion as ap
    q = A.variant()
    p = A.resolve(q)
    L = A.Astor.layout(p)
    mb = A.budget(p, L, pack)
    m = mb["total_g"] / 1000.0
    sel = propulsor(m, power_w)
    thrust = ap.thrust_table(sel, V=[float(v) for v in range(0, 92, 2)])
    C = am.build(q, p, L, "astor", "pusher", mass=m, CD0=CD0, thrust=thrust)
    C["CL_MAX"] *= THIN_CLMAX
    C["alpha_stall"] = (C["CL_MAX"] - C["CL0"]) / C["CLa"]
    C.update({"aircraft": "ASTOR", "mass_budget": mb, "pack_kind": "astor", "aero_source": "model (lattice + hand)",
              "equipment": {"product": "astor", "launch": "rail", "recovery": "parachute", "turning_station": True,
                            "kits": ["rail", "parachute (in the airframe's own budget)"], "pack": mb["pack"]["name"],
                            "motor_w": power_w}})
    t = cfd_table(C)
    if t is not None:
        am.calibrate(C, t, CRUISE)
        C["aero_source"] = f"cfd (screening RANS, {t.get('file', 'astor')})"
    return C


def get(pack: str | None = None, power_w: float = A.MOTOR_W) -> dict:
    return dict(coefficients(pack, power_w))


def cfd_table(C):
    """ASTOR's CFD rows (``data/astor_cfd_screening.yaml``), if computed for this wing (S matches); the moment moved
    to the current CG if it differs."""
    import yaml
    f = am.DATA / "astor_cfd_screening.yaml"
    if not f.exists():
        return None
    d = yaml.safe_load(f.read_text()) or {}
    ref = d.get("reference", {})
    if abs(ref.get("S", 0) - C["S"]) > 1e-4:
        return None
    rows = am.clean_rows(d.get("rows", []))
    if len(rows) < 6:
        return None
    t = am.table(rows)
    dx = C["x_cg_m"] * 1e3 - ref.get("cg_mm", [0])[0]
    t = am.transfer(t, dx / 1000.0, C["c"]) if abs(dx) > 1.0 else t
    return {**t, "file": f.name}


def top_speed(C=None) -> float:
    """Level top speed [m/s] at full thrust (bisection on thrust - drag with the lift at that speed)."""
    C = C or get()
    W = C["mass"] * am.G

    def excess(V):
        qS = 0.5 * am.RHO * V * V * C["S"]
        return am.max_thrust(C, V) - qS * (C["CD0"] + C["k"] * (W / qS) ** 2)
    lo, hi = 20.0, 89.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if excess(mid) > 0 else (lo, mid)
    return lo


def requirement(C=None) -> dict:
    """The sizing requirement: a tail chase of a 50 m/s target."""
    C = C or get()
    Vmax = top_speed(C)
    need = A.CHASE_MARGIN * A.TARGET_M_S
    mb = C["mass_budget"]
    amps = C["thrust"]["propulsor"]["power_w_per_unit"] / (6 * 3.5)          # 6S under load ~3.5 V per cell (HAND)
    return {"target_m_s": A.TARGET_M_S, "top_speed_needed_m_s": need, "top_speed_m_s": round(Vmax, 1), "met": bool(Vmax >= need),
            "closing_speed_m_s": round(Vmax - A.TARGET_M_S, 1), "dynamic_pressure_at_top_Pa": round(0.5 * am.RHO * Vmax ** 2),
            "pack_current_A": round(amps), "pack_limit_A": mb["pack"]["max_A"], "pack_current_ok": bool(amps <= mb["pack"]["max_A"]),
            "stall_nominal_m_s": round(math.sqrt(C["mass"] * am.G / (0.5 * am.RHO * C["S"] * C["CL_MAX"])), 1),
            "stall_in_the_sim_m_s": _sim_stall(C),
            "stall_note": "nominal: the design CL_max; in the sim: the lift model's own maximum (before the CFD the hand "
                          "stall blend gives ~0.7 of the nominal: conservative; the CFD table replaces both)"}


def _sim_stall(C) -> float:
    import arigo_perf as P
    return round(math.sqrt(C["mass"] * am.G / (0.5 * am.RHO * C["S"] * P.polar(C)[1])), 1)
