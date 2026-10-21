"""K4TQ's fast autonomous designs: K4TQ-F3 "Freddy 3" and K4TQ-G3 "Ginna 3" — Freddy (1.2 m) and Ginna (Freddy × 1.35,
≤ 12.5 kg) with the Jetson (K4TQ-A2's companion and perception) and a bigger drive, sized to **match or exceed the top
speed of the 15-inch FPV quad carrying a payload** (``k4tq_quads.fpv15``: the same catalogue and drive model).

Why the drive: Freddy's 28 m/s is the propeller's pitch speed (16x10 at KV300 on 6S ≈ 27 m/s), not its power — the
level drag at 35 m/s is only ≈ 0.5 kW of thrust power. So the sweep (``sweep``) runs motors × propellers (higher pitch)
× packs (6S, 8S, 12S of the same Molicel P45B cells) and keeps what reaches ``target_speed()`` with four motors, passes
the hover rules (T/W ≥ 1.6 on the sagged pack, most loaded motor ≤ 75 %), closes R3 (10 km radius, 2 min hover, 20 %
reserve, 8 m/s wind) and keeps the root's actuator, bearings and pin within their ratings (the actuator sized for the
top speed + 10 %). Freddy 3 keeps the 1 kg payload and takes the lightest such design; Ginna 3 fills the 12.5 kg with
payload (cut until the hover and R3 pass) and takes the most payload.

ESTIMATE / CALC (MOCK BEMT under the mocks); the new parts are "to verify".
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import k4tq
import k4tq_quads as kq
import k4tq_scaled as ks
import k4tq_sizing as sz

SPEEDS = np.arange(14.0, 49.0, 1.0)
FPV_PAYLOADS = (1.0, 2.0)
ESC_FOR = {6: "esc", 8: "esc_8s", 12: "esc_hv"}
CANDIDATES = {
    "F3": {"motors": ("MN501S-300", "MN505S-380", "MN5006-450", "MN601S-180"), "props": ("16x10", "16x12", "16x13", "17x12"),
           "packs": ("6S3P-P45B", "8S2P-P45B", "8S3P-P45B", "12S2P-P45B")},
    "G3": {"motors": ("U8II-170", "U8L-150", "MN801S-150", "MN701S-135", "MN601S-180"), "props": ("20x10", "20x13", "20x14", "22x14"),
           "packs": ("12S2P-P45B", "12S3P-P45B", "12S4P-P45B")},
}
BASE = {"F3": ("F2", 1.0), "G3": ("F2", 1.35)}          # (the airframe's variant, scale) — Freddy autonomous; Ginna's scale
NAMES = {"F3": "Freddy 3", "G3": "Ginna 3"}


def target_speed() -> float:
    """The 15-inch FPV quad's top level speed with a payload (the fastest of its payload cases) [m/s]."""
    return max(kq.fpv15(pl)["max speed [m/s]"] for pl in FPV_PAYLOADS)


def over(letter: str, *, motor: str, prop: str, battery: str, payload_kg: float, root_set: str = "root", spar="auto", v_top: float = 35.0) -> dict:
    base, s = BASE[letter]
    cells = k4tq.CATALOGUE["battery"][battery]["cells_s"]
    common = dict(motor=motor, prop=prop, battery=battery, esc=ESC_FOR[cells], root_set=root_set, spar_tube_mm=spar, payload_kg=payload_kg,
                  autonomy=True, v_map_max=48, v_max_design=1.1 * v_top)
    if s == 1.0:
        return {**k4tq.VARIANTS[base], **common}                    # Freddy's geometry: its CAD masses stay current
    return k4tq.scaled(base, s, made_calibration=k4tq.MADE_CALIBRATION, cad_mass=None, **common)


def build(letter: str, **kw) -> dict:
    v = sz.vehicle(BASE[letter][0], **over(letter, **kw))
    v["variant"] = v["p"]["variant"] = letter
    return v


def top_speed(v: dict) -> float:
    return sz.best_speeds(sz.cruise(v, n_running=4, speeds=SPEEDS))["max level V [m/s]"]


def evaluate(v: dict) -> dict:
    okh, h = ks.hover_ok(v)
    c2 = sz.cruise(v, n_running=2, speeds=SPEEDS)
    b2 = sz.best_speeds(c2)
    m = sz.mission(v, cr=c2, P_transition=h["electrical [W]"])
    okr, r = ks.root_ok(v)
    return {"take-off mass [kg]": v["mass"]["mass_kg"], "payload [kg]": v["p"]["payload_kg"], "top speed [m/s]": top_speed(v),
            "hover ok": okh, "T/W sagged": h["T/W (sagged pack, 3.3 V/cell)"], "most loaded motor": h["most loaded motor / its maximum"],
            "hover [W]": h["electrical [W]"], "R3 closes": bool(m["closes"]), "R3 margin [Wh]": m["margin [Wh]"],
            "still-air range [km]": b2["range [km]"], "cruise power [W]": sz._at(c2, b2["best range V [m/s]"], "electrical [W]"),
            "root ok": okr, "actuator reserve": r["actuator reserve"], "stall [m/s]": sz.stall_speed(v), "spar": v["p"].get("spar_tube_mm", (16, 14))}


def sweep(letter: str, *, log=print) -> pd.DataFrame:
    """Every candidate motor × propeller × pack: F3 at its 1 kg payload; G3 with the payload that fills 12.5 kg, cut until
    the hover rules and R3 pass."""
    V_t = target_speed()
    c = CANDIDATES[letter]
    rows = []
    for mo in c["motors"]:
        for pr in c["props"]:
            for pk in c["packs"]:
                kw = dict(motor=mo, prop=pr, battery=pk, v_top=V_t)
                try:
                    if letter == "F3":
                        v = build(letter, **kw, payload_kg=1.0)
                    else:
                        v, pay = ks._fill_with(lambda **k: build(letter, **kw, **k), ks.LIMITS["G"])
                    if top_speed(v) < V_t - 1e-6:                       # too slow: no need for the rest
                        rows.append({"motor": mo, "prop": pr, "pack": pk, "top speed [m/s]": top_speed(v), "take-off mass [kg]": v["mass"]["mass_kg"],
                                     "meets the target": False, "note": "below the target speed"})
                        log(f"  {letter} {mo} {pr} {pk}: top {top_speed(v):.1f} m/s — too slow")
                        continue
                    for rs in ("root", "root_M", "root_L"):
                        if ks.root_ok(build(letter, **kw, payload_kg=v["p"]["payload_kg"], root_set=rs))[0]:
                            break
                    kw["root_set"] = rs
                    if letter == "F3":
                        v = build(letter, **kw, payload_kg=1.0)
                        e = evaluate(v)
                    else:
                        v, pay = ks._fill_with(lambda **k: build(letter, **kw, **k), ks.LIMITS["G"])
                        for _ in range(14):
                            e = evaluate(v)
                            if (e["hover ok"] and e["R3 closes"]) or pay <= 0.0:
                                break
                            pay = max(pay - max(0.1, 0.12 * pay), 0.0)
                            v = build(letter, **kw, payload_kg=pay)
                        e = evaluate(v)
                except Exception as ex:                                 # noqa: BLE001 — a drive that cannot reach the map's corners
                    rows.append({"motor": mo, "prop": pr, "pack": pk, "meets the target": False, "note": f"{type(ex).__name__}: {str(ex)[:50]}"})
                    continue
                ok = e["top speed [m/s]"] >= V_t - 1e-6 and e["hover ok"] and e["R3 closes"] and e["root ok"] and e["payload [kg]"] > 0
                if letter == "G3":
                    ok = ok and e["take-off mass [kg]"] <= ks.LIMITS["G"] + 1e-3 and e["stall [m/s]"] <= ks.STALL_MAX + 0.1
                rows.append({"motor": mo, "prop": pr, "pack": pk, "root set": rs, **e, "meets the target": ok, "note": ""})
                log(f"  {letter} {mo} {pr} {pk} {rs}: top {e['top speed [m/s]']:.1f} m/s, mass {e['take-off mass [kg]']:.2f} kg, payload {e['payload [kg]']:.2f}, "
                    f"hover {e['hover ok']}, R3 {e['R3 closes']} ({e['R3 margin [Wh]']:.0f} Wh), ok {ok}")
    df = pd.DataFrame(rows)
    df.attrs["label"] = sz.label("CALC")
    df.attrs["target speed [m/s]"] = V_t
    return df


SPEED_MARGIN = 1.05                                             # the top speed ≥ 1.05 × the FPV's: the model's 1 m/s grid and its uncertainty
MARGIN_MIN = 20.0                                               # Wh beyond R3's 20 % reserve: the wind or a cold pack must not end the mission
PARENT_MOTOR = {"F3": "MN501S-300", "G3": "U8II-170"}         # Freddy's and Ginna's: "bigger motors" = heavier (more copper, more power) than these


def bigger(letter: str, motor: str) -> bool:
    m = k4tq.CATALOGUE["motor"]
    return m[motor]["mass_kg"] > m[PARENT_MOTOR[letter]]["mass_kg"] + 1e-9


def pick(letter: str, df: pd.DataFrame) -> pd.Series:
    """The design among the candidates that meet the target with a motor bigger than the parent's, ``MARGIN_MIN`` on R3
    and ``SPEED_MARGIN`` on the speed: F3 the lightest (then the most R3 margin), G3 the most payload."""
    ok = df[df["meets the target"] & df["motor"].map(lambda mo: bigger(letter, mo)) & (df["R3 margin [Wh]"] >= MARGIN_MIN)
            & (df["top speed [m/s]"] >= SPEED_MARGIN * df.attrs.get("target speed [m/s]", target_speed()))]
    if not len(ok):
        raise ValueError(f"no {letter} candidate reaches {df.attrs.get('target speed [m/s]', 0):.1f} m/s with the rules")
    if letter == "F3":                                                  # the lightest; then the most R3 margin
        return ok.sort_values(["take-off mass [kg]", "R3 margin [Wh]"], ascending=[True, False]).iloc[0]
    return ok.sort_values(["payload [kg]", "R3 margin [Wh]"], ascending=[False, False]).iloc[0]


def comparison(designs=("F2", "F3", "G", "G3"), fpv_payloads=(0.0, 1.0, 2.0)) -> pd.DataFrame:
    """The fast designs against their slower parents and the 15-inch FPV quad: mass with the pack, payload, top speed,
    range, hover."""
    rows = {}
    for d in designs:
        v = sz.vehicle(d)
        e = evaluate(v)
        p = v["p"]
        nick = k4tq.NICKNAME.get(d, "")
        rows[f"K4TQ-{d}" + (f" ({nick})" if nick else "")] = {
            "Jetson": bool(p.get("autonomy")), "motor / propeller": f"{p['motor']} / {p['prop']}", "pack": p["battery"],
            "mass with the pack, no payload [kg]": e["take-off mass [kg]"] - p["payload_kg"], "payload [kg]": p["payload_kg"],
            "take-off mass [kg]": e["take-off mass [kg]"], "top speed [km/h]": 3.6 * e["top speed [m/s]"],
            "still-air range [km]": e["still-air range [km]"], "R3 radius [km]": ks.radius(v), "R3 margin [Wh]": e["R3 margin [Wh]"],
            "hover [W]": e["hover [W]"], "T/W sagged": e["T/W sagged"], "stall [m/s]": e["stall [m/s]"], "actuator reserve": e["actuator reserve"],
            "bought parts [EUR]": k4tq.price(v["items"], p)["total bought"]}
    for pl in fpv_payloads:
        f = kq.fpv15(pl)
        rows[f"15\" FPV quad, {pl:g} kg"] = {
            "Jetson": False, "motor / propeller": "MN5006-450 / 15x8", "pack": f["pack"], "mass with the pack, no payload [kg]": f["mass with the pack [kg]"] - pl,
            "payload [kg]": pl, "take-off mass [kg]": f["mass with the pack [kg]"], "top speed [km/h]": 3.6 * f["max speed [m/s]"],
            "still-air range [km]": f["still-air range [km]"], "R3 radius [km]": f["R3 radius [km]"], "R3 margin [Wh]": math.nan,
            "hover [W]": f["hover [W]"], "T/W sagged": f["T/W"] * (3.3 / sz.V_BATT_LOADED) ** 2, "stall [m/s]": 0.0, "actuator reserve": math.nan,
            "bought parts [EUR]": math.nan}
    df = pd.DataFrame(rows).T
    df.attrs["label"] = sz.label("CALC")
    return df


def figure(designs=("F2", "F3"), path=None, fpv_payloads=(1.0, 2.0)):
    """Electrical power against airspeed (K4TQ with four motors running — the top speed — and two — the cruise) and the
    15-inch FPV quad's with its payloads; the target speed marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    path = Path(path or sz.DATA / "figures" / f"k4tq_fast_{'_'.join(designs)}.png")
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5), dpi=110)
    for d in designs:
        v = sz.vehicle(d)
        lab = f"K4TQ-{d}" + (f" ({k4tq.NICKNAME[d]})" if k4tq.NICKNAME.get(d) else "")
        for n, ls in ((4, "-"), (2, "--")):
            c = sz.cruise(v, n_running=n, speeds=SPEEDS)
            c = c[c["feasible"]]
            ax[0].plot(c.index * 3.6, c["electrical [W]"], ls, label=f"{lab}, {n} motors")
            if n == 2:
                ax[1].plot(c.index * 3.6, c["range [km]"], ls, label=lab)
    for pl in fpv_payloads:
        f = kq.fpv15(pl)
        t = f["table"]
        ax[0].plot(t.index * 3.6, t["electrical [W]"], ":", label=f"15\" FPV, {pl:g} kg")
        ax[1].plot(t.index * 3.6, t["range [km]"], ":", label=f"15\" FPV, {pl:g} kg")
    V_t = target_speed() * 3.6
    for a in ax:
        a.axvline(V_t, color="k", lw=0.8)
        a.legend(fontsize=7)
    ax[0].text(V_t + 1, ax[0].get_ylim()[1] * 0.9, f"FPV top speed with payload {V_t:.0f} km/h", fontsize=8)
    ax[0].set(xlabel="airspeed [km/h]", ylabel="electrical power [W]", title="power in level flight (each line ends at the top speed)")
    ax[1].set(xlabel="airspeed [km/h]", ylabel="still-air range [km]", title="range on the pack")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
