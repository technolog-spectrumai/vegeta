"""Maritime KPIs: K4TQ-F3, G3, H3, J3 against the 15-inch FPV quad — the sea's wind, the time on station, reaching a person
in the water, the deck landing, ditching, rescue gear, salt.

Definitions (Stage 1 models; ESTIMATE where noted — no new simulation):
- **radius in wind**: R3 (2 min hover, 20 % reserve) in a 12 and a 15 m/s wind (the open sea's moderate and fresh breeze);
- **time on station**: at 10 km out (8 m/s wind): the usable pack less the reserve, the hovers and the two legs, flown at the
  best-endurance speed (K4TQ on two motors; the quad at its own);
- **to a person 2 km away**: take-off hover 15 s + transition 15 s (K4TQ) + 2 km at the top speed in still air;
- **crosswind hover limit** (ESTIMATE): the side wind the hover holds with the thrust tilted 20° (K4TQ: the fuselage side,
  the fin, the nacelles; wings edge-on — into a headwind it partly converts and flies wing-borne instead) / 35° (the quad);
- **deck margin**: the thrust left over the weight on a sagged pack, in g — the climb-away from a heaving deck;
- **flotation** (ESTIMATE): the sealed fuselage's displacement over the take-off mass (≥ 1: floats until recovered);
- **rescue gear**: auto-inflating lifejackets (~1.0 kg each) and a SOLAS lifebuoy (≥ 2.5 kg) within the payload;
- **salt**: the motors' sealing (the T-Motor Navigator MN-S line is sold as waterproof; the Antigravity MN5006 is not).
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

G, RHO, RHO_W = k4tq.G, k4tq.RHO, 1025.0
DESIGNS = ("F3", "G3", "H3", "J3")
WINDS = (12.0, 15.0)
LIFEJACKET_KG, LIFEBUOY_KG = 1.0, 2.5


def _radius_k4(v: dict, wind: float) -> float:
    c2 = sz.cruise(v, n_running=2, speeds=kf.SPEEDS)
    P = sz.hover(v)["electrical [W]"]
    lo, hi = 0.0, 100.0
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if sz.mission(v, radius_km=mid, wind=wind, cr=c2, P_transition=P)["closes"] else (lo, mid)
    return lo


def _radius_fpv(f: dict, wind: float) -> float:
    tab = f["table"][f["table"]["T each / max"] < 0.8]
    usable = k4tq.battery({"battery": f["pack"]})["usable_wh"] * (1 - sz.MISSION["reserve_fraction"])
    fixed = sz.MISSION["hover_total_s"] * f["hover [W]"] / 3600 + 2 * f["mass with the pack [kg]"] * G * sz.MISSION["cruise_altitude_m"] / 0.45 / 3600
    e_km = 0.0
    for sgn in (-1, 1):
        gs = tab.index.values + sgn * wind
        e = np.where(gs > 1.0, tab["electrical [W]"].values / gs / 3.6, np.inf)
        e_km += float(e.min())
    return max((usable - fixed) / e_km, 0.0) if math.isfinite(e_km) else 0.0


def _station_k4(v: dict, R_km: float = 10.0) -> float:
    c2 = sz.cruise(v, n_running=2, speeds=kf.SPEEDS)
    m = sz.mission(v, radius_km=R_km, cr=c2, P_transition=sz.hover(v)["electrical [W]"])
    left = v["battery"]["usable_wh"] * (1 - sz.MISSION["reserve_fraction"]) - m["energy used [Wh]"]
    P_loiter = float(c2[c2["feasible"]]["electrical [W]"].min())
    return max(left / P_loiter * 60, 0.0)


def _station_fpv(f: dict, R_km: float = 10.0) -> float:
    if f["R3 radius [km]"] < R_km:
        return 0.0
    tab = f["table"][f["table"]["T each / max"] < 0.8]
    w = sz.MISSION["wind_m_s"]
    usable = k4tq.battery({"battery": f["pack"]})["usable_wh"] * (1 - sz.MISSION["reserve_fraction"])
    used = sz.MISSION["hover_total_s"] * f["hover [W]"] / 3600
    for sgn in (-1, 1):
        gs = tab.index.values + sgn * w
        used += float(np.where(gs > 1.0, tab["electrical [W]"].values / gs / 3.6, np.inf).min()) * R_km
    return max((usable - used) / float(tab["electrical [W]"].min()) * 60, 0.0)


def _crosswind_k4(v: dict, tilt_deg: float = 20.0) -> float:
    p, L = v["p"], v["L"]
    sc = p.get("scale", 1.0)
    side = (p["fuselage_length"] * p["fuselage_height"] * 0.8 + (p.get("fin_area") or 0.03 * sc ** 2)
            + 4 * 0.045 * sc * 0.15 * sc + 4 * 0.12 * L["wings"]["front"]["mac"] * (L["wings"]["front"]["semi_span"] + L["wings"]["rear"]["semi_span"]) / 2)
    return math.sqrt(2 * v["W"] * math.tan(math.radians(tilt_deg)) / (RHO * 1.1 * side))


def _flotation_k4(v: dict) -> float:
    p = v["p"]
    vol = math.pi / 4 * p["fuselage_width"] * p["fuselage_height"] * p["fuselage_length"] * 0.75
    return vol * RHO_W / v["mass"]["mass_kg"]


def _gear(payload: float) -> str:
    lj = int(payload // LIFEJACKET_KG)
    return f"{lj} lifejacket{'s' if lj != 1 else ''}" + (" or 1 lifebuoy" if payload >= LIFEBUOY_KG else "")


def table(designs=DESIGNS, fpv_payloads=(1.0, 2.0)) -> pd.DataFrame:
    rows = {}
    for d in designs:
        v = sz.vehicle(d)
        e = kf.evaluate(v)
        p = v["p"]
        top = e["top speed [m/s]"]
        mo = k4tq.CATALOGUE["motor"][p["motor"]]["name"]
        rows[f"K4TQ-{d} ({k4tq.NICKNAME.get(d, '')})"] = {
            "take-off mass [kg]": e["take-off mass [kg]"], "payload [kg]": p["payload_kg"],
            **{f"R3 radius in {w:g} m/s wind [km]": _radius_k4(v, w) for w in WINDS},
            "time on station 10 km out [min]": _station_k4(v), "to a person 2 km away [s]": 30.0 + 2000.0 / top,
            "ground speed into 15 m/s wind [km/h]": 3.6 * (top - 15.0),
            "crosswind hover limit (ESTIMATE) [m/s]": _crosswind_k4(v), "deck margin (T/W − 1, sagged) [g]": e["T/W sagged"] - 1.0,
            "flotation (displacement / mass, ESTIMATE)": _flotation_k4(v), "rescue gear in the payload": _gear(p["payload_kg"]),
            "motors sealed (salt / spray)": "yes (Navigator MN-S, sold waterproof)" if "Navigator" in mo else "no"}
    for pl in fpv_payloads:
        f = kq.fpv15(pl)
        W = f["mass with the pack [kg]"] * G
        rows[f"15\" FPV quad, {pl:g} kg"] = {
            "take-off mass [kg]": f["mass with the pack [kg]"], "payload [kg]": pl,
            **{f"R3 radius in {w:g} m/s wind [km]": _radius_fpv(f, w) for w in WINDS},
            "time on station 10 km out [min]": _station_fpv(f), "to a person 2 km away [s]": 5.0 + 2000.0 / f["max speed [m/s]"],
            "ground speed into 15 m/s wind [km/h]": 3.6 * (f["max speed [m/s]"] - 15.0),
            "crosswind hover limit (ESTIMATE) [m/s]": math.sqrt(2 * W * math.tan(math.radians(35.0)) / (RHO * 1.1 * 0.06)),
            "deck margin (T/W − 1, sagged) [g]": f["T/W"] * (3.3 / sz.V_BATT_LOADED) ** 2 - 1.0,
            "flotation (displacement / mass, ESTIMATE)": 0.0, "rescue gear in the payload": _gear(pl), "motors sealed (salt / spray)": "no (MN5006 Antigravity)"}
    df = pd.DataFrame(rows)
    df.attrs["label"] = sz.label("CALC") + " / ESTIMATE"
    return df
