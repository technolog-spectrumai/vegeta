"""K4TQ's heavy-lift autonomous designs grown from K4TQ-G3 "Ginna 3": K4TQ-H3 "Hanna 3" (3.5 kg payload) and K4TQ-J3
"Joanna 3" (4.5 kg payload). Not chasers: the top speed is free; what counts is the payload, the hover (T/W ≥ 1.6 on the
sagged pack, the most loaded motor ≤ 75 %, and the hover time reported), R3 (10 km radius, 2 min hover, 20 % reserve,
8 m/s wind) and **Ginna 3's range kept** (still-air ≥ ``RANGE_KEEP`` of it).

The sweep (``sweep``): Freddy autonomous (F2) scaled by s, off-the-shelf T-Motor Navigator 12S motors and Aeronaut CAM /
T-Motor folding / APC rigid propellers, 12S packs of Molicel P45B; the stall at Freddy's (≤ 18 m/s), the smallest root hardware that
holds; the pick is the lightest design that passes. ESTIMATE / CALC (MOCK BEMT under the mocks); no CAD of these scales.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import k4tq
import k4tq_fast as kf
import k4tq_scaled as ks
import k4tq_sizing as sz

PAYLOAD = {"H3": 3.5, "J3": 4.5}
NAMES = {"H3": "Hanna 3", "J3": "Joanna 3"}
SCALES = {"H3": (1.40, 1.44, 1.48, 1.52), "J3": (1.50, 1.54, 1.58, 1.62)}
MOTORS = ("MN605S-170", "MN801S-150")
PROPS = ("CAM18x13", "APC20x10E", "APC22x10E", "APC22x12E")     # the folding CAM 18x13 and the bigger rigid APCs (folding ones of 20"+ are low pitch: they cannot cruise)
PACKS = ("12S3P-P45B", "12S4P-P45B", "12S5P-P45B", "12S6P-P45B")
RANGE_KEEP = 0.90


def over(s: float, *, motor: str, prop: str, battery: str, payload_kg: float, root_set: str = "root", spar="auto") -> dict:
    return k4tq.scaled("F2", s, motor=motor, prop=prop, battery=battery, esc="esc_hv", root_set=root_set, spar_tube_mm=spar, payload_kg=payload_kg,
                       autonomy=True, v_map_max=48, v_max_design=40.0, made_calibration=k4tq.MADE_CALIBRATION, cad_mass=None)


def build(s: float, **kw) -> dict:
    return sz.vehicle("F2", **over(s, **kw))


def ginna3_range() -> float:
    return sz.best_speeds(sz.cruise(sz.vehicle("G3"), n_running=2, speeds=kf.SPEEDS))["range [km]"]


def sweep(letter: str, *, log=print) -> pd.DataFrame:
    pay = PAYLOAD[letter]
    r_min = RANGE_KEEP * ginna3_range()
    rows = []
    for s in SCALES[letter]:
        for mo in MOTORS:
            for pr in PROPS:
                for pk in PACKS:
                    kw = dict(motor=mo, prop=pr, battery=pk, payload_kg=pay)
                    try:
                        v = build(s, **kw)
                        if sz.stall_speed(v) > ks.STALL_MAX + 0.05:
                            rows.append({"scale": s, "motor": mo, "prop": pr, "pack": pk, "ok": False, "note": "stall above 18 m/s"})
                            continue
                        for rs in ("root", "root_M", "root_L"):
                            vr = build(s, **kw, root_set=rs, spar=v["p"]["spar_tube_mm"])
                            if ks.root_ok(vr)[0]:
                                break
                        e = kf.evaluate(vr)
                    except Exception as ex:                                   # noqa: BLE001 — a drive outside its map
                        rows.append({"scale": s, "motor": mo, "prop": pr, "pack": pk, "ok": False, "note": f"{type(ex).__name__}: {str(ex)[:40]}"})
                        continue
                    hover_min = vr["battery"]["usable_wh"] / e["hover [W]"] * 60
                    ok = e["hover ok"] and e["R3 closes"] and e["root ok"] and e["still-air range [km]"] >= r_min and e["stall [m/s]"] <= ks.STALL_MAX + 0.05
                    rows.append({"scale": s, "motor": mo, "prop": pr, "pack": pk, "root set": rs, **e, "hover minutes": hover_min, "ok": ok, "note": ""})
                    log(f"  {letter} s{s:.2f} {mo} {pr} {pk} {rs}: {e['take-off mass [kg]']:.2f} kg, top {e['top speed [m/s]']:.0f} m/s, "
                        f"range {e['still-air range [km]']:.0f} km, hover {hover_min:.1f} min, T/W {e['T/W sagged']:.2f}, ok {ok}")
    df = pd.DataFrame(rows)
    df.attrs["label"] = sz.label("CALC")
    return df


def pick(df: pd.DataFrame) -> pd.Series:
    """The lightest passing design (then the longest hover)."""
    ok = df[df["ok"] == True]                                                 # noqa: E712 — the CSV round trip
    if not len(ok):
        raise ValueError("no candidate passes")
    return ok.sort_values(["take-off mass [kg]", "hover minutes"], ascending=[True, False]).iloc[0]


def comparison(designs=("G3", "H3", "J3")) -> pd.DataFrame:
    rows = {}
    for d in designs:
        v = sz.vehicle(d)
        e = kf.evaluate(v)
        p = v["p"]
        rows[f"K4TQ-{d} ({k4tq.NICKNAME.get(d, '')})"] = {
            "scale": p.get("scale", 1.0), "fuselage × span [m]": f"{p['fuselage_length']:.2f} × {p['span']:.2f}", "motor / propeller": f"{p['motor']} / {p['prop']}",
            "pack": p["battery"], "payload [kg]": p["payload_kg"], "take-off mass [kg]": e["take-off mass [kg]"],
            "mass with the pack, no payload [kg]": e["take-off mass [kg]"] - p["payload_kg"], "top speed [km/h]": 3.6 * e["top speed [m/s]"],
            "still-air range [km]": e["still-air range [km]"], "R3 radius [km]": ks.radius(v), "hover [W]": e["hover [W]"],
            "hover to an empty pack [min]": v["battery"]["usable_wh"] / e["hover [W]"] * 60, "T/W sagged": e["T/W sagged"], "stall [m/s]": e["stall [m/s]"],
            "actuator reserve": e["actuator reserve"], "bought parts [EUR]": k4tq.price(v["items"], p)["total bought"]}
    df = pd.DataFrame(rows).T
    df.attrs["label"] = sz.label("CALC")
    return df
