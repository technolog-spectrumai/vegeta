"""K4TQ scaled up to a take-off mass limit: K4TQ-B "Bertha" (≤ 24 kg: under the 25 kg C-class line with a 1 kg safety
margin) and K4TQ-G "Ginna" (≤ 12.5 kg).

The airframe is K4TQ-F1 "Freddy" with every length × s (``k4tq.scaled``: the stations stay fractions of the fuselage,
the cargo cartridge grows with it); the hardware goes up a class — 12S Li-ion packs of the same cells, 12S FOC ESCs,
the large carbon propellers on folding hubs, the smallest wing-root hardware (``CATALOGUE["root"]``, ``"root_M"``,
``"root_L"``) whose actuator, bearings and locking pin hold — and the spar comes from the root loads. Bertha has no CAD: the
made parts' estimates are corrected by the factor K4TQ-F1's CAD found on its own (``calibration``).

The sizing (``size``): for each scale, the propulsion that passes the hover rules (T/W ≥ 1.6 on the sagged pack, the
most loaded motor ≤ 75 %) with the least cruise power; then for each 12S pack the payload that fills the mass limit,
cut back until the R3 mission (10 km radius, 2 min hover, 20 % reserve, 8 m/s wind) closes; the design is the scale
and pack carrying the most payload. ESTIMATE / CALC (MOCK BEMT under the mocks) as in Stage 1; no CAD, no FEA.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import k4tq
import k4tq_sizing as sz

BASE_DESIGN = "F1"
LIMITS = {"B": 24.0, "G": 12.5}                    # take-off mass limits [kg]
NAMES = {"B": "Bertha", "G": "Ginna"}
SCALES = {"B": (1.45, 1.5, 1.55, 1.6, 1.7), "G": (1.2, 1.25, 1.3, 1.35, 1.4)}
PROPS = {"B": ("24x15", "26x16", "28x17", "30x18"), "G": ("20x10", "22x11", "24x12", "24x15")}
MOTORS = {"B": ("MN1005-90", "U12II-100", "U12II-120", "U10II-100"), "G": ("MN601S-180", "MN701S-135", "U8II-170", "U8L-150", "MN801S-150")}
STALL_MAX = 18.0                                   # m/s: Freddy's 17.7 — the same transition corridor and wing-borne speeds
PACKS = tuple(f"12S{n}P-P45B" for n in range(2, 9))
SPEEDS = np.arange(14.0, 45.0, 1.0)
TW_MIN, MOTOR_MAX = 1.6, 0.75
CALIBRATED = ("fuselage shell", "wing panel", "nacelle", "fin", "wing root")


def calibration() -> dict:
    """{item prefix: CAD / estimate} of K4TQ-F1 for the made parts (and the roots' made metal)."""
    p = k4tq.params(BASE_DESIGN)
    cad = p.get("cad_mass") or {}
    est = k4tq.mass_table(k4tq.layout(k4tq.params(BASE_DESIGN, cad_mass=None)))
    out = {}
    for pre, kg in cad.items():
        if pre.startswith(CALIBRATED):
            e = [i["mass_kg"] for i in est if i["item"].startswith(pre)]
            if e:
                out[pre] = kg / (sum(e) / len(e))
    return out


def over(s: float, *, motor: str, prop: str, battery: str, payload_kg: float, root_set: str = "root_L", spar="auto", cal: dict | None = None) -> dict:
    return k4tq.scaled(BASE_DESIGN, s, motor=motor, prop=prop, battery=battery, esc="esc_hv", root_set=root_set, spar_tube_mm=spar,
                       payload_kg=payload_kg, made_calibration=cal if cal is not None else calibration(), cad_mass=None)


def build(s: float, **kw) -> dict:
    return sz.vehicle(BASE_DESIGN, **over(s, **kw))


def _fill(s, limit, kw):
    """The vehicle with the payload that brings the take-off mass to ``limit`` (linear in the payload; a second step
    for the spar the heavier aircraft takes)."""
    v = build(s, **{**kw, "payload_kg": 0.0})
    pay = 0.0
    for _ in range(3):
        pay = max(pay + limit - v["mass"]["mass_kg"], 0.0)
        v = build(s, **{**kw, "payload_kg": pay})
        if v["mass"]["mass_kg"] <= limit + 1e-3:
            break
    return v, pay


def hover_ok(v: dict) -> tuple[bool, dict]:
    h = sz.hover(v)
    return (h["T/W (sagged pack, 3.3 V/cell)"] >= TW_MIN and h["most loaded motor / its maximum"] <= MOTOR_MAX), h


def root_ok(v: dict) -> tuple[bool, dict]:
    r = sz.root_loads(v)
    return (r["actuator reserve"] >= 1.0 and r["bearing static reserve"] >= 1.0 and r["locking pin shear stress, double shear [MPa]"] <= 600.0), r


def propulsion(s: float, limit: float, battery: str, cal: dict, props=PROPS["B"], motors=MOTORS["B"]) -> pd.DataFrame:
    """Every motor × propeller whose disc fits the scaled span: the hover rules at the limit mass and the cruise power."""
    rows = []
    D_ref = 16.0 * s
    for pr in props:
        d = k4tq.CATALOGUE["propeller"][pr]["d_in"]
        if not (D_ref - 3.5 <= d <= D_ref + 4.5):
            continue
        for mo in motors:
            kw = dict(motor=mo, prop=pr, battery=battery, cal=cal, root_set="root_M")
            try:
                v, pay = _fill(s, limit, kw)
            except Exception as e:                                      # a layout the checks reject (discs into the fuselage …)
                rows.append({"prop": pr, "motor": mo, "ok": False, "note": str(e)[:60]})
                continue
            ok, h = hover_ok(v)
            c = sz.cruise(v, n_running=2, speeds=SPEEDS)
            b = sz.best_speeds(c)
            rows.append({"prop": pr, "motor": mo, "ok": ok and math.isfinite(h["electrical [W]"]), "payload at the limit [kg]": pay,
                         "T/W sagged": h["T/W (sagged pack, 3.3 V/cell)"], "most loaded motor": h["most loaded motor / its maximum"],
                         "hover [W]": h["electrical [W]"], "cruise [W]": sz._at(c, b["best range V [m/s]"], "electrical [W]"),
                         "range [km]": b["range [km]"], "note": ""})
    return pd.DataFrame(rows)


def radius(v: dict) -> float:
    c = sz.cruise(v, n_running=2, speeds=SPEEDS)
    P = sz.hover(v)["electrical [W]"]
    if not math.isfinite(P):
        return 0.0
    lo, hi = 0.0, 80.0
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        if sz.mission(v, radius_km=mid, cr=c, P_transition=P)["closes"]:
            lo = mid
        else:
            hi = mid
    return lo


def closes(v: dict) -> tuple[bool, dict]:
    c = sz.cruise(v, n_running=2, speeds=SPEEDS)
    m = sz.mission(v, cr=c, P_transition=sz.hover(v)["electrical [W]"])
    return bool(m["closes"]), m


def size(letter: str, *, scales=None, packs=PACKS, log=print) -> dict:
    """The scale, propulsion, pack and payload of K4TQ-``letter`` at ``LIMITS[letter]``: the most payload on the R3
    mission within the mass limit."""
    limit = LIMITS[letter]
    cal = calibration()
    rows, best = [], None
    for s in scales or SCALES[letter]:
        mid = packs[len(packs) // 2]
        pt = propulsion(s, limit, mid, cal, PROPS[letter], MOTORS[letter])
        good = pt[pt["ok"] & pt["cruise [W]"].notna()] if len(pt) else pt
        if not len(good):
            log(f"  s {s:.2f}: no propulsion passes the hover rules")
            rows.append({"scale": s, "note": "no propulsion passes the hover rules"})
            continue
        pick = good.sort_values("cruise [W]").iloc[0]
        for pk in packs:
            kw = dict(motor=pick["motor"], prop=pick["prop"], battery=pk, cal=cal)
            v, pay = _fill(s, limit, kw)
            for rs in ("root", "root_M", "root_L"):                     # the smallest root hardware that holds
                if root_ok(build(s, **{**kw, "payload_kg": pay, "root_set": rs}))[0]:
                    break
            kw["root_set"] = rs
            v, pay = _fill(s, limit, kw)
            for _ in range(14):                                         # cut the payload until the hover rules and the mission pass
                okh, h = hover_ok(v)
                okc, m = closes(v)
                if (okh and okc) or pay <= 0.0:
                    break
                pay = max(pay - max(0.1, 0.12 * pay), 0.0)
                v = build(s, **{**kw, "payload_kg": pay})
            ok = okh and okc and v["mass"]["mass_kg"] <= limit + 1e-3
            row = {"scale": s, "motor": pick["motor"], "prop": pick["prop"], "pack": pk, "root set": rs, "spar": v["p"]["spar_tube_mm"],
                   "payload [kg]": pay if ok else 0.0, "take-off mass [kg]": v["mass"]["mass_kg"], "R3 closes": okc,
                   "R3 margin [Wh]": m["margin [Wh]"], "hover ok": okh, "T/W sagged": h["T/W (sagged pack, 3.3 V/cell)"],
                   "most loaded motor": h["most loaded motor / its maximum"], "stall [m/s]": sz.stall_speed(v)}
            row["meets the stall cap"] = row["stall [m/s]"] <= STALL_MAX
            rows.append(row)
            log(f"  s {s:.2f} {pick['motor']} {pick['prop']} {pk} {rs}: payload {row['payload [kg]']:.2f} kg, mass {row['take-off mass [kg]']:.2f} kg, "
                f"stall {row['stall [m/s]']:.1f}, closes {okc}, hover {okh}")
            if ok and row["meets the stall cap"] and (best is None or row["payload [kg]"] > best[0]["payload [kg]"]):
                best = (row, kw | {"payload_kg": pay, "spar": v["p"]["spar_tube_mm"]}, s)
    df = pd.DataFrame(rows)
    df.attrs["label"] = sz.label("CALC")
    return {"letter": letter, "limit [kg]": limit, "table": df, "best": best[0] if best else None,
            "variant": variant_dict(letter, best[2], best[1]) if best else None}


def variant_dict(letter: str, s: float, kw: dict) -> dict:
    """The ``k4tq.VARIANTS`` entry of the sized design (the payload as sized; the spar fixed)."""
    kw = dict(kw)
    cal = kw.pop("cal")
    spar = kw.pop("spar")
    o = over(s, **kw, spar=spar, cal=cal)
    o.pop("cad_mass", None)
    title = f"K4TQ-{letter} \"{NAMES[letter]}\": K4TQ-F1 × {s:g} to ≤ {LIMITS[letter]:g} kg, {kw['motor']} + {kw['prop']}, {kw['battery']}"
    return {**o, "variant": letter, "title": title, "spar_tube_mm": tuple(spar), "made_calibration": {k: round(v, 3) for k, v in cal.items()}}


def summary(designs=("B", "G"), reference=("F1",)) -> pd.DataFrame:
    """The sized designs against Freddy: masses, payload, mission, hover, cruise, stall, the root and the spar."""
    rows = {}
    for d in list(reference) + list(designs):
        v = sz.vehicle(d)
        p = v["p"]
        h = sz.hover(v)
        c = sz.cruise(v, n_running=2, speeds=SPEEDS)
        b = sz.best_speeds(c)
        m = sz.mission(v, cr=c, P_transition=h["electrical [W]"])
        r = sz.root_loads(v)
        nick = k4tq.NICKNAME.get(d, "")
        rows[f"K4TQ-{d}" + (f" ({nick})" if nick else "")] = {
            "scale": p.get("scale", 1.0), "fuselage × span [m]": f"{p['fuselage_length']:.2f} × {p['span']:.2f}", "width over the discs [m]": v["L"]["width_overall"],
            "motor / propeller": f"{p['motor']} / {p['prop']}", "pack": p["battery"], "take-off mass [kg]": v["mass"]["mass_kg"], "payload [kg]": p["payload_kg"],
            "cartridge [l]": float(np.prod(p["cargo_box"])) * 1e3 * 0.42, "empty (no payload, no pack) [kg]": v["mass"]["mass_kg"] - p["payload_kg"] - v["battery"]["mass_kg"],
            "tilt-wing tax [kg]": k4tq.tilt_wing_tax(v["items"]), "wing loading [kg/m²]": v["mass"]["mass_kg"] / v["L"]["S_ref"],
            "stall speed [m/s]": sz.stall_speed(v), "best-range V [m/s]": b["best range V [m/s]"], "best L/D": b["L/D max"],
            "cruise power [W]": sz._at(c, b["best range V [m/s]"], "electrical [W]"), "hover [W]": h["electrical [W]"],
            "T/W sagged": h["T/W (sagged pack, 3.3 V/cell)"], "still-air range [km]": b["range [km]"], "endurance [min]": b["endurance [min]"],
            "R3 closes": m["closes"], "R3 margin [Wh]": m["margin [Wh]"], "R3 radius [km]": radius(v), "max level V [m/s]": b["max level V [m/s]"],
            "spar": f"Ø{p['spar_tube_mm'][0]}×{p['spar_tube_mm'][1]}" if isinstance(p.get("spar_tube_mm"), tuple) else "Ø16×14",
            "root hardware": p.get("root_set", "root"), "actuator reserve": r["actuator reserve"], "bearing reserve": r["bearing static reserve"],
            "payload fraction": p["payload_kg"] / v["mass"]["mass_kg"], "bought parts [EUR]": k4tq.price(v["items"], p)["total bought"],
        }
    df = pd.DataFrame(rows).T
    df.attrs["label"] = sz.label("CALC")
    return df


def payload_limits(variant: str, *, hi: float = 12.0) -> dict:
    """What limits the payload of a design (its pack and airframe as they are): the hover rules (T/W ≥ 1.6 sagged, the
    most loaded motor ≤ 75 %), the R3 mission, the mass limit (B, G) — each the largest payload that passes it alone."""
    def bisect(test, top):
        lo, up = 0.0, top
        if not test(lo):
            return 0.0
        for _ in range(12):
            mid = 0.5 * (lo + up)
            lo, up = (mid, up) if test(mid) else (lo, mid)
        return lo
    v0 = sz.vehicle(variant, payload_kg=0.0)
    m0 = v0["mass"]["mass_kg"]
    out = {"empty with the pack [kg]": m0, "design payload [kg]": k4tq.params(variant)["payload_kg"]}
    out["hover rules [kg]"] = bisect(lambda x: hover_ok(sz.vehicle(variant, payload_kg=x))[0], hi)
    out["R3 mission [kg]"] = bisect(lambda x: closes(sz.vehicle(variant, payload_kg=x))[0], hi)
    lim = LIMITS.get(variant)
    out["mass limit [kg]"] = max(lim - m0, 0.0) if lim else math.nan
    vals = {k: x for k, x in out.items() if k.endswith("[kg]") and k not in ("empty with the pack [kg]", "design payload [kg]") and math.isfinite(x)}
    out["max payload [kg]"] = min(vals.values())
    out["limited by"] = min(vals, key=vals.get).replace(" [kg]", "")
    return out
