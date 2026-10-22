"""Kestrel-4Q-2 against 4Q-1, and the case for 4Q-3 (notebook 42).

Notebook 40 handed over two aircraft: **4Q-1** (round struts, the X, elevons as effectors) and **4Q-2** (faired posts,
no X, the reflexed section, elevons) — 4Q-2 flies twice as far on the battery and balloons on the pitch-up from the
cruise, because a clean aircraft cannot throw its speed away. This module rebuilds both from notebook 40's recipe
(``vehicles``), flies the **remedies** for the back transition (``REMEDIES`` / ``remedy_matrix``: the crow-brake
elevons, the pitch bulge past the vertical, reverse thrust on bidirectional drives, a descent into the transition,
the level deceleration to 13 m/s first — ``decelerate_level`` gives its time, distance and energy), and compares the
aircraft as flying machines (``performance``: battery life and range against speed, the thrust-limited maximum speed,
the climb — specific excess power, best rate, time to 100 m — the dive speeds at idle, the acceleration, the turn at
the corner speed and the sustained turn, the rotors' and elevons' authority in the horizontal plane; ``pursuit`` puts
the numbers a chaser lives by side by side). ``propose_4q3`` writes the third aircraft from what the two before it
proved: 4Q-2's airframe with bidirectional drives, wider elevons (the crow brake and the roll), and the autopilot's
back-transition logic — and evaluates it in the same tables.
"""
from __future__ import annotations

import math

import numpy as np

import kestrel
import kestrel_control as kc
import kestrel_flight as kf
import kestrel_systems as ks
import kestrel_transition as kt

G, RHO = ks.G, ks.RHO0
N_STRUCTURAL = 4.0
REVERSE_N = 3.0                  # N per rotor: what a 9x6 gives in reverse at low speed, roughly a third of its forward thrust
AIRBRAKE_DEG = 20.0
OVERSHOOT_DEG = 20.0
DESCENT_ENTRY = 2.0              # m/s
V_SLOW = 13.0                    # the slow-entry rule


# ------------------------------------------------------------------------------------------------- the aircraft
def section_numbers():
    """Notebook 40 § 14's finalist on Zephyr (NACA 23112, k₂/k₁ 0.03) when its table is on file: the section drag over the
    0012's, cl_max, α₀, the elevon's τ; else the 0012's with τ 0.40."""
    try:
        import kestrel_section as ksec
        t = ksec.table()
        pick, base = ksec.key("NACA 23112", 0.03), "NACA 0012"
        sp, b = t[pick], t[base]
        return {"pick": pick, "cd_ratio": sp["cruise"]["cd_design"] / b["cruise"]["cd_design"], "cl_max": 0.9 * sp["cruise"]["cl_max"],
                "alpha0": math.radians(sp["cruise"]["alpha0_deg"]), "tau": float(b["elevon"]["tau"]), "have_zephyr": True}
    except Exception:
        return {"pick": None, "cd_ratio": 1.0, "cl_max": None, "alpha0": None, "tau": 0.40, "have_zephyr": False}


def params(which: str = "4Q-2", p=None) -> dict:
    p = kestrel.resolve(p)
    if which == "4Q-1":
        return p
    if which == "4Q-2":
        return kestrel.resolve(dict(p, elevons=True, strut_fairing=True, bracing="posts"))
    if which == "4Q-3":
        return kestrel.resolve(dict(p, elevons=True, strut_fairing=True, bracing="posts", elevon_chord=0.32, elevon_inner_y=250.0))
    raise ValueError(which)


def polar_for(which: str, p=None, aero=None, sec=None):
    """Notebook 40's polar recipe: 4Q-1 on the round-strut build-up; 4Q-2 / 4Q-3 on the faired build-up with the
    finalist section's drag ratio on the wings' share, its cl_max and α₀, and the sleeves' broadside area in cd90."""
    pp = params(which, p)
    a = aero or kf.aero(pp)
    if which == "4Q-1":
        return kf.polar(pp, a=a)
    sec = sec or section_numbers()
    d = kf.drag_buildup(pp, faired=True)
    pol = kf.polar(pp, a=a, d=d)
    L = kestrel.layout(pp)
    broadside = (2 * 0.029 * L["post_length"] + 4 * 0.023 * L["diagonal_length"]) - (2 * 0.008 * L["post_length"] + 4 * 0.006 * L["diagonal_length"])
    cd90 = pol.cd90 + 1.2 * 0.5 * broadside / L["S_ref"]
    if sec["have_zephyr"]:
        return kf.Polar(cl_alpha=pol.cl_alpha, cd0=d["Cd0"] - d["Cd0_wings"] + d["Cd0_wings"] * sec["cd_ratio"], k=pol.k, cl_max=sec["cl_max"], cd90=cd90,
                        alpha0=sec["alpha0"], source=f"{sec['pick']} (Zephyr)")
    return kf.Polar(**{**pol.__dict__, "cd90": cd90})


def vehicles(p=None, *, which=("4Q-1", "4Q-2")) -> dict:
    """``kestrel_transition.Vehicle`` for each aircraft, as notebook 40 built them (the elevon τ from Zephyr)."""
    p = kestrel.resolve(p)
    a = kf.aero(p)
    sec = section_numbers()
    out = {}
    for w in which:
        pp = params(w, p)
        v = kt.make_vehicle(pp, aero=a, pol=polar_for(w, p, a, sec), name=f"Kestrel-{w}")
        v.elevon_tau = sec["tau"]
        out[w] = v
    return out


# ------------------------------------------------------------------------------------------------- the remedies
REMEDIES = {
    "as is (entered at 16 m/s)": {},
    "slow entry at 13 m/s": {"V_entry": V_SLOW},
    f"crow-brake elevons ±{AIRBRAKE_DEG:g}°": {"airbrake_deg": AIRBRAKE_DEG},
    f"pitch {OVERSHOOT_DEG:g}° past the vertical": {"overshoot_deg": OVERSHOOT_DEG},
    f"reverse thrust {REVERSE_N:g} N per rotor": {"reverse_N": REVERSE_N},
    f"descend into it at {DESCENT_ENTRY:g} m/s": {"descent_entry": DESCENT_ENTRY},
    "crow brake + reverse thrust": {"airbrake_deg": AIRBRAKE_DEG, "reverse_N": REVERSE_N},
}
COLS = ["height loss [m]", "height gain [m]", "pitch error max [deg]", "time stalled, free stream [s]", "α blown max while stalled [deg]",
        "moment shortfall [s]", "least pitch margin", "peak rotor thrust / T_max", "elevon max [deg]", "roll demand / authority", "energy [Wh]",
        "hover reached at [s]", "pass"]


def remedy_matrix(veh, *, remedies: dict | None = None, durations=(1.5, 3.0, 5.0), headwinds=(0.0, 6.0), gust_up=0.0, progress=False):
    """The back transition for every remedy × schedule × headwind (vegeta.aeromant.transitions.remedy_matrix with
    Kestrel's remedies): the metrics table and the episodes."""
    return kt.remedy_matrix(veh, remedies or REMEDIES, kind="back", durations=durations, headwinds=headwinds, gust_up=gust_up, progress=progress)


remedy_verdict = kt.remedy_verdict


def decelerate_level(veh, V0: float = 16.0, V1: float = V_SLOW, *, airbrake_deg: float = 0.0, reverse_N: float = 0.0, dt: float = 0.01) -> dict:
    """The deceleration in level flight before the pitch-up: the rotors at idle (or in reverse), the crow brake if
    asked, the lift held at the weight (the trim angle rises as the speed falls, so the induced drag with it);
    the time, the distance and the energy from ``V0`` to ``V1``."""
    L, pol, d = veh.L, veh.pol, veh.drive
    S, W, m = L["S_ref"], veh.weight, veh.mass
    y_e = veh.elevon_y_in if veh.elevon_y_in is not None else L["motor_y"] - L["R_prop"]
    S_e4 = 4 * max(L["span"] / 2 - y_e, 0.0) * L["chord"]
    T = -reverse_N if reverse_N > 0 else d.T_min
    V, t, x, E = V0, 0.0, 0.0, 0.0
    while V > V1 and t < 60.0:
        q = 0.5 * RHO * V * V
        CL = W / (q * S)
        if CL > pol.cl_max:
            break
        D = q * S * (pol.cd0 + pol.k * CL ** 2) + q * S_e4 * 1.2 * math.sin(math.radians(airbrake_deg)) ** 2
        a = -(D - 4 * T) / m
        P = 4 * float(d.power(abs(T), V)) + 13.0
        V += a * dt; x += V * dt; t += dt; E += P * dt
    return {"from [m/s]": V0, "to [m/s]": V1, "time [s]": t, "distance [m]": x, "energy [Wh]": E / 3600, "reached": V <= V1 + 1e-6,
            "mean deceleration [m/s²]": (V0 - V1) / t if t > 0 else np.nan}


# ------------------------------------------------------------------------------------------------- performance
def drag(veh, V: float, n: float = 1.0) -> float:
    S, pol = veh.L["S_ref"], veh.pol
    q = 0.5 * RHO * V * V
    CL = n * veh.weight / (q * S)
    return q * S * (pol.cd0 + pol.k * CL ** 2)


def stall_speed(veh, n: float = 1.0) -> float:
    return math.sqrt(2 * n * veh.weight / (RHO * veh.L["S_ref"] * veh.pol.cl_max))


def performance(veh, *, speeds=None, avionics_W: float = 13.0) -> dict:
    """The flying machine in numbers: endurance and range against speed, the thrust-limited maximum speed, the climb
    (specific excess power P_s, the best rate and its speed, the steepest climb, the time to 100 m), the dive (the
    steady speeds at idle thrust at 30°, 60° and straight down), the level acceleration (13 → 25 m/s and to 90 % of
    V max), the turn at the corner speed (the lift limit meets the 4 g structural limit) and the sustained turn at the
    cruise, and the manoeuvrability in the horizontal plane from ``kestrel_control.cruise_manoeuvrability``."""
    import pandas as pd
    d, W, m, S = veh.drive, veh.weight, veh.mass, veh.L["S_ref"]
    Vs = stall_speed(veh)
    speeds = np.arange(max(9.0, math.ceil(Vs)), 36.5, 0.5) if speeds is None else np.asarray(speeds, float)
    rows = {}
    Ewh = ks.COMPONENTS["battery"]["cells"] * 3.7 * ks.COMPONENTS["battery"]["capacity_ah"] * ks.COMPONENTS["battery"]["usable"]
    for V in speeds:
        D = drag(veh, V)
        T_av = 4 * float(d.t_max(V))
        P_el = 4 * float(d.power(D / 4, V)) + avionics_W if D / 4 <= float(d.t_max(V)) else np.nan
        rows[float(V)] = {"drag [N]": D, "thrust available [N]": T_av, "excess thrust [N]": T_av - D, "P_s (climb rate) [m/s]": (T_av - D) * V / W,
                          "climb angle [deg]": math.degrees(math.asin(max(min((T_av - D) / W, 1.0), -1.0))), "electrical, level [W]": P_el,
                          "endurance [min]": Ewh / P_el * 60 if np.isfinite(P_el) else np.nan, "range [km]": Ewh / P_el * V * 3.6 if np.isfinite(P_el) else np.nan,
                          "n available (lift)": min(0.5 * RHO * V * V * S * veh.pol.cl_max / W, N_STRUCTURAL)}
    tab = pd.DataFrame(rows).T
    tab.index.name = "V [m/s]"
    feas = tab[tab["excess thrust [N]"] > 0]
    V_max = float(feas.index.max()) if len(feas) else np.nan
    if len(feas) and V_max < speeds[-1]:
        i = list(tab.index).index(V_max)
        V2 = tab.index[i + 1]
        e1, e2 = tab.loc[V_max, "excess thrust [N]"], tab.loc[V2, "excess thrust [N]"]
        V_max = float(V_max + (V2 - V_max) * e1 / (e1 - e2))
    best = feas["P_s (climb rate) [m/s]"].idxmax() if len(feas) else np.nan
    # the time to climb 100 m at the best rate, the acceleration in level flight
    def accel_time(Va, Vb):
        t, V, x = 0.0, Va, 0.0
        while V < Vb and t < 120.0:
            a = (4 * float(d.t_max(V)) - drag(veh, V)) / m
            if a <= 0.01:
                return np.nan, np.nan
            V += a * 0.01; x += V * 0.01; t += 0.01
        return t, x
    t_25, x_25 = accel_time(13.0, 25.0)
    t_90, x_90 = accel_time(13.0, 0.9 * V_max) if np.isfinite(V_max) else (np.nan, np.nan)
    # the dive at idle thrust: D(V) = W sin γ + 4 T_min
    def dive_speed(gamma_deg):
        target = W * math.sin(math.radians(gamma_deg)) + 4 * d.T_min
        lo, hi = Vs, 120.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if drag(veh, mid) < target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
    V_corner = Vs * math.sqrt(N_STRUCTURAL)
    n_c = N_STRUCTURAL
    turn_rate_c = math.degrees(G * math.sqrt(n_c ** 2 - 1) / V_corner)
    radius_c = V_corner ** 2 / (G * math.sqrt(n_c ** 2 - 1))
    # the sustained turn at 16 m/s: the load factor whose drag the available thrust can hold
    V16 = 16.0
    T16 = 4 * float(d.t_max(V16))
    lo, hi = 1.0, min(N_STRUCTURAL, 0.5 * RHO * V16 ** 2 * S * veh.pol.cl_max / W)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if drag(veh, V16, mid) < T16:
            lo = mid
        else:
            hi = mid
    n_s = lo
    manv = kc.cruise_manoeuvrability(veh, speeds=(13.0, 16.0, 20.0, 25.0), elevon_tau=veh.elevon_tau)
    summary = {"stall speed [m/s]": Vs, "V max, thrust = drag [m/s]": V_max, "best endurance [min]": float(feas["endurance [min]"].max()) if len(feas) else np.nan,
               "at [m/s]": float(feas["endurance [min]"].idxmax()) if len(feas) else np.nan, "best range [km]": float(feas["range [km]"].max()) if len(feas) else np.nan,
               "at range speed [m/s]": float(feas["range [km]"].idxmax()) if len(feas) else np.nan, "endurance at 16 m/s [min]": float(tab.loc[16.0, "endurance [min]"]) if 16.0 in tab.index else np.nan,
               "best climb rate [m/s]": float(feas["P_s (climb rate) [m/s]"].max()) if len(feas) else np.nan, "at climb speed [m/s]": float(best) if np.isfinite(best) else np.nan,
               "steepest climb [deg]": float(feas["climb angle [deg]"].max()) if len(feas) else np.nan, "time to climb 100 m [s]": 100.0 / float(feas["P_s (climb rate) [m/s]"].max()) if len(feas) else np.nan,
               "vertical climb (hover thrust margin) [m/s²]": (4 * float(d.t_max(0.0)) - W) / m,
               "accelerate 13 → 25 m/s [s]": t_25, "distance for it [m]": x_25, "accelerate 13 → 0.9 V max [s]": t_90,
               "dive at 30° (idle) [m/s]": dive_speed(30.0), "dive at 60° (idle) [m/s]": dive_speed(60.0), "terminal dive (idle) [m/s]": dive_speed(90.0),
               "corner speed [m/s]": V_corner, "turn rate at the corner (4 g) [deg/s]": turn_rate_c, "turn radius at the corner [m]": radius_c,
               "time for 180° at the corner [s]": 180.0 / turn_rate_c, "sustained load factor at 16 m/s": n_s,
               "sustained turn rate at 16 m/s [deg/s]": math.degrees(G * math.sqrt(max(n_s ** 2 - 1, 0.0)) / V16),
               "pitch at 16 m/s, total [rad/s²]": float(manv.loc[16.0, "pitch: total [rad/s²]"]), "steady roll rate at 16 m/s [deg/s]": float(manv.loc[16.0, "steady roll rate [deg/s]"]),
               "yaw at 16 m/s, rotors [rad/s²]": float(manv.loc[16.0, "yaw: rotors [rad/s²]"]), "mass [kg]": m, "Cd0": veh.pol.cd0}
    return {"table": tab, "summary": summary, "manoeuvrability": manv}


def pursuit(perfs: dict) -> "pd.DataFrame":
    """The numbers a chaser lives by, side by side: speed, acceleration, climb, dive, the turn, and what it costs in
    endurance — one column per aircraft, from ``performance``."""
    import pandas as pd
    keys = ["mass [kg]", "Cd0", "stall speed [m/s]", "V max, thrust = drag [m/s]", "accelerate 13 → 25 m/s [s]", "accelerate 13 → 0.9 V max [s]",
            "best climb rate [m/s]", "at climb speed [m/s]", "steepest climb [deg]", "time to climb 100 m [s]", "vertical climb (hover thrust margin) [m/s²]",
            "dive at 30° (idle) [m/s]", "dive at 60° (idle) [m/s]", "terminal dive (idle) [m/s]", "corner speed [m/s]", "turn rate at the corner (4 g) [deg/s]",
            "turn radius at the corner [m]", "time for 180° at the corner [s]", "sustained load factor at 16 m/s", "sustained turn rate at 16 m/s [deg/s]",
            "pitch at 16 m/s, total [rad/s²]", "steady roll rate at 16 m/s [deg/s]", "yaw at 16 m/s, rotors [rad/s²]",
            "endurance at 16 m/s [min]", "best endurance [min]", "at [m/s]", "best range [km]", "at range speed [m/s]"]
    return pd.DataFrame({k: {kk: v["summary"].get(kk) for kk in keys} for k, v in perfs.items()})


# ------------------------------------------------------------------------------------------------- Kestrel-4Q-3
def propose_4q3() -> dict:
    """What the third aircraft is, and why — each line a change over 4Q-2 and the result that asks for it."""
    return {
        "airframe": "4Q-2's: faired posts, no X diagonals, elevons, the reflexed NACA 23112 (k₂/k₁ 0.03); the payload bay",
        "drives": f"bidirectional ESCs (reverse thrust, ~{REVERSE_N:g} N per rotor at low speed): the brake the clean aircraft lacks — the back transition "
                  f"from the cruise speed without the balloon, and a dive brake",
        "elevons": "32 % chord from 250 mm to the tips (4Q-2: 28 % from 275 mm): more roll and pitch in cruise, and travel to spare for the crow brake "
                   "with ±20° still left for control",
        "autopilot": "the back transition's logic: brake first (reverse thrust, the crow brake), pitch up when the speed is at 1.2 × the stall; "
                     "the slow-entry rule stays as the fallback without reverse thrust",
        "what it keeps": "the four fixed rotors, the zero stagger, the 280 mm gap (the pitch lever through the stall), the diagonal pairing",
        "what it costs": "the ESCs (reverse-capable firmware; the same hardware), ~10 g of elevon and horn, the braking energy (a few hundred joules per transition)",
        "what it buys": "the endurance of 4Q-2 (40 min at 16 m/s, 57 best) with the transitions of 4Q-1 (any entry speed); the pursuit numbers of the clean "
                        "airframe — the same V max (pitch-speed limited), the better acceleration and climb (less drag), the 4 g corner turn",
        "open": "the propeller for speed (a 9x7.5 raises the pitch speed and V max, costs hover thrust: prop_lab's calibrated drive table decides); "
                "reverse thrust measured, not assumed (prop_lab: a negative-rpm case); the roll–yaw in 6 DOF",
    }
