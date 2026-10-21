"""The KPI table: K4TQ-F2, F3, G2, G3 against the 15-inch FPV quad (1 kg, 2 kg) — mass, payload, the R3 radius with the
payload, the mission-averaged speed, the top speed held for 3 s, the still-air range, the hover time to an empty pack,
and the 0–100 km/h acceleration horizontally (from the hover) and vertically (straight up).

Definitions (all on the Stage 1 models; ESTIMATE where noted):
- **average speed**: the R3 mission's distance over its whole time — 2 × 10 km (or the radius it can fly, if less)
  over take-off / landing hovers (2 min in all), K4TQ's four 15 s transitions and the two legs in the 8 m/s wind, each at
  the speed that uses least energy per ground km. "Without payload" flies the same mission empty;
- **top speed for 3 s**: the drive at its 180 s peak current (the motors' published rating; a 3 s burst is within it —
  no thermal model: the 3 s figure equals the 180 s one), four motors, level;
- **hover time**: the usable pack (to 3.0 V/cell, ageing and cold allowance) over the hover power;
- **0–100 km/h horizontal** (ESTIMATE): from the hover, at full current; K4TQ's wings tilt at most 90° in
  ``k4tq_sizing.TILT_TIME_S`` and lift up to 90 % of CL max, the thrust carries the rest of the weight; the FPV quad tilts at
  once (up to 65°); the thrust at the propeller's axial inflow;
- **0–100 km/h up** (ESTIMATE): straight up from the hover, full current, the broadside drag; "max …" where the climb
  rate never reaches 100 km/h.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import k4tq
import k4tq_fast as kf
import k4tq_quads as kq
import k4tq_scaled as ks
import k4tq_sizing as sz

G, RHO = k4tq.G, k4tq.RHO
V100 = 100 / 3.6
DESIGNS = ("F2", "F3", "G2", "G3")
FPV_PAYLOADS = (1.0, 2.0)


# ================================================================================================= the mission's average speed
def _avg_k4(v: dict, R_km: float) -> float:
    c2 = sz.cruise(v, n_running=2, speeds=kf.SPEEDS)
    m = sz.mission(v, radius_km=R_km, cr=c2, P_transition=sz.hover(v)["electrical [W]"])
    t = sum(leg["time [min]"] for leg in m["legs"].values()) * 60
    return 2 * R_km * 1000 / t


def _avg_fpv(f: dict, R_km: float) -> float:
    tab = f["table"][f["table"]["T each / max"] < 0.8]
    w = sz.MISSION["wind_m_s"]
    t = sz.MISSION["hover_total_s"]
    for sgn in (-1, 1):
        gs = tab.index.values + sgn * w
        e = np.where(gs > 1.0, tab["electrical [W]"].values / gs, np.inf)
        t += R_km * 1000 / gs[int(np.argmin(e))]
    return 2 * R_km * 1000 / t


# ================================================================================================= acceleration
def _accel_k4(v: dict, *, up: bool) -> tuple[float, float]:
    """(seconds to 100 km/h or nan, the speed reached [m/s]) — K4TQ from the hover."""
    W, m = v["W"], v["mass"]["mass_kg"]
    drv, L = v["drive"], v["L"]
    S, CLm = L["S_ref"], 0.9 * v["stall"]["CL max (aircraft)"]
    d = sz.drag_buildup(L)
    k = sz.k_total(v)
    p = v["p"]
    V, t, dt = 0.0, 0.0, 0.02
    if up:
        CdA = p["fuselage_length"] * p["fuselage_width"] + 0.05
        while t < 30.0:
            a = (4 * sz.max_thrust(drv, V) - W * sz.DOWNLOAD - 0.5 * RHO * V * V * CdA) / m
            if a <= 1e-3:
                break
            V += a * dt
            t += dt
            if V >= V100:
                return t, V
        return math.nan, V
    tilt = math.pi / 2
    rate = (math.pi / 2) / sz.TILT_TIME_S
    while t < 30.0:
        q = 0.5 * RHO * V * V
        Lw = min(W, q * S * CLm)
        T = 4 * sz.max_thrust(drv, V * math.cos(tilt))
        Tv = max(W * sz.DOWNLOAD - Lw, 0.0)
        need = math.asin(min(Tv / T, 1.0)) if T > 0 else math.pi / 2    # the lowest tilt that still holds the weight
        tilt = max(need, tilt - rate * dt)
        CL = Lw / (q * S) if q > 0 else 0.0
        D = q * S * (d["Cd0"] + k * CL * CL)
        a = (T * math.cos(tilt) - D) / m
        if a <= 1e-3 and V > 1.0:
            break
        V += max(a, 0.0) * dt
        t += dt
        if V >= V100:
            return t, V
    return math.nan, V


def _accel_fpv(payload: float, *, up: bool) -> tuple[float, float]:
    f = kq.fpv15(payload)
    m = f["mass with the pack [kg]"]
    W = m * G
    drv = sz.drive({"prop": "15x8", "motor": "MN5006-450", "battery": f["pack"]}, V=tuple(range(0, 50, 2)))
    CdA = kq.FPV15_CDA + (0.010 if payload > 0 else 0.0)
    V, t, dt = 0.0, 0.0, 0.02
    while t < 30.0:
        if up:
            a = (4 * sz.max_thrust(drv, V) - W - 0.5 * RHO * V * V * (CdA + 0.02)) / m
        else:
            D = 0.5 * RHO * V * V * CdA
            best = -math.inf
            for tl in np.radians(np.arange(5.0, 65.1, 2.5)):                    # the disc tilt that accelerates most, holding the height
                T = 4 * sz.max_thrust(drv, V * math.sin(tl))
                if T * math.cos(tl) >= W:
                    best = max(best, (T * math.sin(tl) - D) / m)
            a = best
        if a <= 1e-3:
            break
        V += a * dt
        t += dt
        if V >= V100:
            return t, V
    return math.nan, V


def _fmt_accel(r: tuple[float, float]) -> str:
    t, V = r
    return f"{t:.1f} s" if math.isfinite(t) else f"— (max {3.6 * V:.0f} km/h)"


# ================================================================================================= the table
def _k4_row(d: str) -> dict:
    v, v0 = sz.vehicle(d), sz.vehicle(d, payload_kg=0.0)
    p = v["p"]
    pl = p["payload_kg"]
    R = ks.radius(v)
    R0 = ks.radius(v0)
    usable = v["battery"]["usable_wh"]
    b2 = sz.best_speeds(sz.cruise(v, n_running=2, speeds=kf.SPEEDS))
    b20 = sz.best_speeds(sz.cruise(v0, n_running=2, speeds=kf.SPEEDS))
    return {"mass with payload [kg]": v["mass"]["mass_kg"], "mass with the pack, no payload [kg]": v0["mass"]["mass_kg"], "payload [kg]": pl,
            "R3 radius with the payload [km]": R,
            "average speed on R3, with payload [km/h]": 3.6 * _avg_k4(v, min(10.0, R)),
            "average speed on R3, without payload [km/h]": 3.6 * _avg_k4(v0, min(10.0, R0)),
            "top speed (3 s), with payload [km/h]": 3.6 * kf.top_speed(v), "top speed (3 s), without payload [km/h]": 3.6 * kf.top_speed(v0),
            "still-air range, with payload [km]": b2["range [km]"], "still-air range, without payload [km]": b20["range [km]"],
            "hover to an empty pack, with payload [min]": usable / sz.hover(v)["electrical [W]"] * 60,
            "hover to an empty pack, without payload [min]": usable / sz.hover(v0)["electrical [W]"] * 60,
            "0–100 km/h horizontal, with payload": _fmt_accel(_accel_k4(v, up=False)),
            "0–100 km/h horizontal, without payload": _fmt_accel(_accel_k4(v0, up=False)),
            "0–100 km/h up, with payload": _fmt_accel(_accel_k4(v, up=True)), "0–100 km/h up, without payload": _fmt_accel(_accel_k4(v0, up=True))}


def _fpv_row(pl: float) -> dict:
    f, f0 = kq.fpv15(pl), kq.fpv15(0.0)
    return {"mass with payload [kg]": f["mass with the pack [kg]"], "mass with the pack, no payload [kg]": f0["mass with the pack [kg]"], "payload [kg]": pl,
            "R3 radius with the payload [km]": f["R3 radius [km]"],
            "average speed on R3, with payload [km/h]": 3.6 * _avg_fpv(f, min(10.0, f["R3 radius [km]"])),
            "average speed on R3, without payload [km/h]": 3.6 * _avg_fpv(f0, min(10.0, f0["R3 radius [km]"])),
            "top speed (3 s), with payload [km/h]": 3.6 * f["max speed [m/s]"], "top speed (3 s), without payload [km/h]": 3.6 * f0["max speed [m/s]"],
            "still-air range, with payload [km]": f["still-air range [km]"], "still-air range, without payload [km]": f0["still-air range [km]"],
            "hover to an empty pack, with payload [min]": f["hover minutes"], "hover to an empty pack, without payload [min]": f0["hover minutes"],
            "0–100 km/h horizontal, with payload": _fmt_accel(_accel_fpv(pl, up=False)), "0–100 km/h horizontal, without payload": _fmt_accel(_accel_fpv(0.0, up=False)),
            "0–100 km/h up, with payload": _fmt_accel(_accel_fpv(pl, up=True)), "0–100 km/h up, without payload": _fmt_accel(_accel_fpv(0.0, up=True))}


def table(designs=DESIGNS, fpv_payloads=FPV_PAYLOADS) -> pd.DataFrame:
    rows = {}
    for d in designs:
        nick = k4tq.NICKNAME.get(d, "")
        rows[f"K4TQ-{d}" + (f" ({nick})" if nick else "")] = _k4_row(d)
    for pl in fpv_payloads:
        rows[f"15\" FPV quad, {pl:g} kg"] = _fpv_row(pl)
    df = pd.DataFrame(rows)
    df.attrs["label"] = sz.label("CALC") + " / ESTIMATE (acceleration)"
    return df
