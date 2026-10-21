"""K4TQ against the quadcopters: K4TQ-A1 "Cathy" and K4TQ-F1 "Freddy" (or A2 / F2, any variant) against

- **Q-13**: a quadcopter on 13" multirotor propellers (13x5.5) on the compact X (the discs 60 mm apart);
- **Q-17**: a quadcopter on 17" propellers (17x5.8) with its motors spread to K4TQ's plan footprint (span over the
  hover discs × the fuselage length) — the biggest discs the same parking space takes;
- **Q-17 heavy**: Q-17 spending K4TQ's tilt-wing tax on cells — the largest 6S·nP pack that keeps its take-off mass at
  K4TQ's (the fair "same mass" quad: does a quad with a bigger battery catch up?).

Both carry what K4TQ carries (the same payload, cargo cartridge with its temperature logging, avionics — the Jetson when
the reference design has it —, parachute, pack and growth margin; ``k4tq_sizing.quad_reference``); each gets the
catalogue motor that gives it the longest R3 radius with T/W ≥ ``TW_MIN`` (KV300 on a 13" propeller would be unfair).

Everything is CALC on the Stage 1 models (the BEMT drive maps — coarsened under the mocks —, the quad's Glauert forward
flight); the hover agility and the turns are ESTIMATE (rigid body, point-mass inertia for the quads).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

import k4tq
import k4tq_compare as kc
import k4tq_envelope as ke
import k4tq_sizing as sz

G, RHO = k4tq.G, k4tq.RHO
DESIGNS = ("A1", "F1")
QUADS = {"Q-13": {"prop": "13x5.5", "footprint": False}, "Q-17": {"prop": "17x5.8", "footprint": True},
         "Q-17 heavy": {"prop": "17x5.8", "footprint": True, "same_mass": True}}
TW_MIN = 1.6
DATA = kc.DATA


def footprint(veh: dict) -> tuple[float, float]:
    """K4TQ's plan box in the hover [m]: the span over the discs × the fuselage length."""
    return float(veh["L"]["width_overall"]), float(veh["p"]["fuselage_length"])


def _radius(q: dict, k4: dict) -> float:
    return sz.quad_mission_radius(q, k4)


def quad(name: str, prop_key: str, k4: dict, *, in_footprint: bool = False, motor_key: str | None = None) -> dict:
    """One quadcopter reference built on K4TQ's vehicle ``k4``; the motor swept over the catalogue unless given."""
    R = k4tq.CATALOGUE["propeller"][prop_key]["d_in"] * 0.0254 / 2
    arm = None
    if in_footprint:
        bw, bl = footprint(k4)
        arm = math.hypot(bw / 2 - R, bl / 2 - R)
    motors = [motor_key] if motor_key else list(k4tq.CATALOGUE["motor"])
    best = None
    for mk in motors:
        q = sz.quad_reference(name, prop_key, k4=k4, motor_key=mk, arm_len=arm)
        q["mission radius [km]"] = _radius(q, k4)
        ok = q["T/W"] >= TW_MIN
        key = (ok, q["mission radius [km]"])
        if best is None or key > best[0]:
            best = (key, q)
    q = best[1]
    a = q["arm [m]"]
    if in_footprint:
        q["plan box [m]"] = footprint(k4)
    else:
        s = 2 * (a / math.sqrt(2) + R)
        q["plan box [m]"] = (s, s)
    q["disc gap [m]"] = a * math.sqrt(2) - 2 * R if not in_footprint else 2 * (footprint(k4)[1] / 2 - R) - 2 * R
    return q


def quad_envelope(q: dict) -> dict:
    """The quad's climb, top speed, turn and hover agility (ESTIMATE where noted)."""
    W = q["mass [kg]"] * G
    drv = q["drive"]
    CdA_up = 0.15 * 0.45 + 0.05                                       # the pod and the legs broadside to a vertical climb
    lo, hi = 0.0, 25.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if 4 * sz.max_thrust(drv, mid) - W - 0.5 * RHO * mid * mid * CdA_up > 0:
            lo = mid
        else:
            hi = mid
    T3 = (W + 0.5 * RHO * 9.0 * CdA_up) / 4
    dp = sz.drive_point(drv, 3.0, T3)
    aw = q["hover [W]"] - 4 * sz.drive_point(drv, 0.0, W / 4)["electrical_W"]
    P3 = 4 * dp["electrical_W"] + aw if dp["feasible"] else math.inf
    # turns: the bank the thrust margin allows at the best-range speed and at 15 m/s (the disc thrust ≤ 80 % of its maximum)
    Tmax = sz.max_thrust(drv, 0.0)
    rows = {}
    for V in (q["best range V [m/s]"], 15.0):
        if not math.isfinite(V):
            continue
        D = 0.5 * RHO * V * V * q["CdA [m²]"]
        Tav = 0.8 * 4 * Tmax
        n = min(math.sqrt(max(Tav ** 2 - D ** 2, 0.0)) / W, sz.N_LIMIT)
        rows[V] = (math.degrees(G * math.sqrt(max(n * n - 1, 0.0)) / V), V * V / (G * math.sqrt(max(n * n - 1, 1e-9))))
    # hover agility: point masses at the motors (motor + propeller + ESC + arm) and a 0.45 × 0.15 m pod for the rest
    cat = k4tq.CATALOGUE
    m_tip = cat["motor"][q["motor"]]["mass_kg"] + cat["propeller"][q["propeller"]]["mass_kg"] + 0.028 + 0.073 * q["arm [m]"] / 2
    bw, bl = q["plan box [m]"]
    R = cat["propeller"][q["propeller"]]["d_in"] * 0.0254 / 2
    hx, hy = bl / 2 - R, bw / 2 - R                                    # the motors' half-spacing (along, across)
    m_body = q["mass [kg]"] - 4 * m_tip
    Ib = m_body * (0.45 ** 2 + 0.15 ** 2) / 12
    Ixx, Iyy, Izz = 4 * m_tip * hy ** 2 + Ib, 4 * m_tip * hx ** 2 + Ib, 4 * m_tip * (hx ** 2 + hy ** 2) + Ib
    T0 = W / 4
    dT = min(Tmax - T0, T0)
    hp = sz.drive_point(drv, 0.0, T0)
    Q0 = hp["shaft_W"] / (hp["rpm"] * 2 * math.pi / 60)              # the reaction torque per rotor at the hover
    dQ = Q0 * dT / T0 * 1.5                                            # Q ∝ T^1.5 at the margin
    # forward climb: the power left between 80 % of the maximum thrust and level flight, through the motor efficiency
    P80 = 4 * sz.drive_point(drv, 0.0, 0.8 * Tmax)["electrical_W"] + aw
    fwd = q["table"][q["table"]["T each / max"] < 0.8]
    roc = float(((P80 - fwd["electrical [W]"]) * q["eta motor"] * 0.8 / W).max()) if len(fwd) else math.nan     # 0.8: the disc's induced losses on the extra thrust
    return {"max vertical climb [m/s]": lo, "best forward climb (80 %) [m/s]": roc, "power climbing at 3 m/s [W]": P3,
            "sustained turn at best-range V [deg/s]": rows.get(q["best range V [m/s]"], (math.nan,))[0],
            "sustained turn at 15 m/s [deg/s]": rows.get(15.0, (math.nan,))[0],
            "roll acceleration [deg/s²]": math.degrees(4 * dT * hy / Ixx), "pitch acceleration [deg/s²]": math.degrees(4 * dT * hx / Iyy),
            "yaw acceleration [deg/s²]": math.degrees(4 * dQ / Izz), "label": "ESTIMATE"}


def with_pack(k4: dict, n_p: int) -> dict:
    """``k4`` with a 6S·``n_p``P pack of the same cells (mass, energy, price ∝ P) for the quad references."""
    base_key = k4["p"]["battery"]
    b0 = k4tq.CATALOGUE["battery"][base_key]
    p0 = int(base_key.split("S")[1].split("P")[0])
    key = base_key.replace(f"{p0}P", f"{n_p}P")
    f = n_p / p0
    if key not in k4tq.CATALOGUE["battery"]:
        k4tq.CATALOGUE["battery"][key] = {**b0, "mass_kg": b0["mass_kg"] * f, "capacity_ah": b0["capacity_ah"] * f, "price_eur": b0["price_eur"] * f,
                                          "max_A": b0.get("max_A", 0.0) * f, "source": f"{base_key} scaled to {n_p}P (quad reference)"}
    bat = {**k4["battery"], "key": key, "cells_p": n_p, "mass_kg": k4["battery"]["mass_kg"] * f, "capacity_ah": k4["battery"]["capacity_ah"] * f,
           "energy_wh": k4["battery"]["energy_wh"] * f, "usable_wh": k4["battery"]["usable_wh"] * f}
    return {**k4, "p": {**k4["p"], "battery": key}, "battery": bat}


def quad_same_mass(name: str, prop_key: str, k4: dict, *, in_footprint: bool = True) -> dict:
    """The largest pack (6S·nP) with which the quad's take-off mass stays at or under K4TQ's, its motor re-chosen."""
    target = k4["mass"]["mass_kg"]
    p0 = int(k4["p"]["battery"].split("S")[1].split("P")[0])
    best = None
    for n in range(p0, p0 + 12):
        kb = with_pack(k4, n)
        q = quad(name, prop_key, kb, in_footprint=in_footprint)
        if q["mass [kg]"] > target + 1e-6:
            break
        q["pack"], q["k4"] = kb["p"]["battery"], kb
        best = q
    return best


def k4_radius(variant: str, payload_kg: float = 1.0) -> float:
    """The R3 radius [km] (8 m/s wind, 2 min hover, 20 % reserve) with ``payload_kg``."""
    v = sz.vehicle(variant, payload_kg=payload_kg)
    cr = sz.cruise(v, n_running=2)
    P = sz.hover(v)["electrical [W]"]
    lo, hi = 0.0, 80.0
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        if sz.mission(v, radius_km=mid, cr=cr, P_transition=P)["closes"]:
            lo = mid
        else:
            hi = mid
    return lo


def study(designs=DESIGNS) -> dict:
    """The K4TQ designs and the two quads built on the first design (they carry its avionics)."""
    out = {"k4tq": {}, "quads": {}}
    for v in designs:
        e = kc.evaluate(v, corridor=False, extended=False)
        veh = e["veh"]
        e["radius 1 kg [km]"] = k4_radius(v, 1.0)
        e["radius 0.5 kg [km]"] = k4_radius(v, 0.5)
        e["vertical climb"] = ke.vertical_climb(veh)
        cl = ke.climb(veh)
        e["climb"] = cl
        e["turns"] = ke.turns(veh)
        e["hover agility"] = ke.hover_agility(veh)
        e["price"] = k4tq.price(veh["items"], veh["p"])
        out["k4tq"][v] = e
    k4 = out["k4tq"][designs[0]]["veh"]
    k4_half = sz.vehicle(designs[0], payload_kg=0.5)
    for name, spec in QUADS.items():
        if spec.get("same_mass"):
            q = quad_same_mass(name, spec["prop"], k4, in_footprint=spec["footprint"])
            half = with_pack(k4_half, int(q["pack"].split("S")[1].split("P")[0]))
        else:
            q = quad(name, spec["prop"], k4, in_footprint=spec["footprint"])
            q["pack"], half = k4["p"]["battery"], k4_half
        q["envelope"] = quad_envelope(q)
        q5 = quad(name, spec["prop"], half, in_footprint=spec["footprint"], motor_key=q["motor"])
        q["radius 0.5 kg [km]"] = q5["mission radius [km]"]
        out["quads"][name] = q
    return out


def table(res: dict) -> pd.DataFrame:
    rows = {}
    for v, e in res["k4tq"].items():
        veh, h, b2, b4 = e["veh"], e["hover"], e["best2"], e["best4"]
        cl8 = e["climb"][e["climb"]["thrust"] == "80%"]
        ha = e["hover agility"]
        bw, bl = footprint(veh)
        rows[kc.name(v)] = {
            "propeller / motor": f"{veh['p']['prop']} / {veh['p']['motor']}", "pack": veh["p"]["battery"], "plan box (hover) [m]": f"{bw:.2f} × {bl:.2f}",
            "take-off mass [kg]": veh["mass"]["mass_kg"], "hover [W]": h["electrical [W]"], "hover minutes (whole pack)": h["hover minutes on the usable pack"],
            "T/W": h["T/W (loaded pack)"], "best-range V [m/s]": b2["best range V [m/s]"],
            "cruise power [W]": sz._at(e["cruise2"], b2["best range V [m/s]"], "electrical [W]"), "still-air range [km]": b2["range [km]"],
            "endurance (cruise) [min]": b2["endurance [min]"], "R3 radius, 1 kg [km]": e["radius 1 kg [km]"], "R3 radius, 0.5 kg [km]": e["radius 0.5 kg [km]"],
            "max level speed [m/s]": b4["max level V [m/s]"], "max vertical climb [m/s]": e["vertical climb"]["max vertical climb [m/s]"],
            "best forward climb (80 %) [m/s]": float(cl8["rate of climb [m/s]"].max()),
            "sustained turn (best) [deg/s]": float(e["turns"]["sustained rate [deg/s]"].max()),
            "hover roll / pitch / yaw [deg/s²]": "{:.0f} / {:.0f} / {:.0f}".format(*[ha[k] for k in ha if "acceleration" in k]),
        }
    for k, q in res["quads"].items():
        en = q["envelope"]
        bw, bl = q["plan box [m]"]
        roc = en["best forward climb (80 %) [m/s]"]
        rows[f"{k} ({q['propeller']} / {q['motor']}, {q['pack']})"] = {
            "propeller / motor": f"{q['propeller']} / {q['motor']}", "pack": q["pack"], "plan box (hover) [m]": f"{bw:.2f} × {bl:.2f}",
            "take-off mass [kg]": q["mass [kg]"], "hover [W]": q["hover [W]"], "hover minutes (whole pack)": q["hover minutes"], "T/W": q["T/W"],
            "best-range V [m/s]": q["best range V [m/s]"], "cruise power [W]": sz._at(q["table"], q["best range V [m/s]"], "electrical [W]"),
            "still-air range [km]": q["range [km]"], "endurance (cruise) [min]": q["endurance [min]"],
            "R3 radius, 1 kg [km]": q["mission radius [km]"], "R3 radius, 0.5 kg [km]": q["radius 0.5 kg [km]"],
            "max level speed [m/s]": q["max speed (T ≤ 80 % max) [m/s]"], "max vertical climb [m/s]": en["max vertical climb [m/s]"],
            "best forward climb (80 %) [m/s]": roc, "sustained turn (best) [deg/s]": en["sustained turn at 15 m/s [deg/s]"],
            "hover roll / pitch / yaw [deg/s²]": f"{en['roll acceleration [deg/s²]']:.0f} / {en['pitch acceleration [deg/s²]']:.0f} / {en['yaw acceleration [deg/s²]']:.0f}",
        }
    df = pd.DataFrame(rows).T
    df.attrs["label"] = sz.label("CALC")
    return df


def figure(res: dict, path: Path) -> Path:
    """Power and range against speed for every aircraft, and the headline bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5), dpi=110)
    for v, e in res["k4tq"].items():
        c = e["cruise2"][e["cruise2"]["feasible"]]
        ax[0].plot(c.index, c["electrical [W]"], "o-", ms=3, label=kc.name(v))
        ax[1].plot(c.index, c["range [km]"] if "range [km]" in c else e["veh"]["battery"]["usable_wh"] / c["electrical [W]"] * c.index * 3.6, "o-", ms=3, label=kc.name(v))
        ax[0].axhline(e["hover"]["electrical [W]"], ls=":", lw=0.8, color=ax[0].lines[-1].get_color())
    for k, q in res["quads"].items():
        t = q["table"][q["table"]["T each / max"] < 0.8]
        ax[0].plot(t.index, t["electrical [W]"], "s--", ms=3, label=f"{k} ({q['propeller']})")
        ax[1].plot(t.index, t["range [km]"], "s--", ms=3, label=f"{k} ({q['propeller']})")
    ax[0].set(xlabel="airspeed [m/s]", ylabel="electrical power [W]", title="power against speed (K4TQ: hover dotted, cruise on two motors)")
    ax[1].set(xlabel="airspeed [m/s]", ylabel="still-air range [km]", title="still-air range on the same pack")
    df = table(res)
    cols = ["R3 radius, 1 kg [km]", "R3 radius, 0.5 kg [km]", "hover minutes (whole pack)", "max level speed [m/s]"]
    x = np.arange(len(cols))
    n = len(df)
    for i, (k, r) in enumerate(df.iterrows()):
        ax[2].bar(x + (i - (n - 1) / 2) * 0.8 / n, [float(r[c]) for c in cols], width=0.8 / n, label=k.split(" (")[0])
    ax[2].set_xticks(x, ["R3 radius 1 kg", "R3 radius 0.5 kg", "hover min", "max speed"], rotation=10)
    ax[2].set(title="headline")
    for a in ax:
        if a.get_legend_handles_labels()[0]:
            a.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def verdict(df: pd.DataFrame) -> str:
    """Plain-language reading: range, the same-mass quad, hover, speed and agility."""
    k4 = [i for i in df.index if i.startswith("K4TQ")]
    qs = [i for i in df.index if i.startswith("Q-")]
    f = lambda i, c: float(df.loc[i, c])                                                         # noqa: E731
    short = lambda i: i.split(" (")[0]                                                           # noqa: E731
    bk = max(k4, key=lambda i: f(i, "R3 radius, 1 kg [km]"))
    light = [i for i in qs if "heavy" not in i]
    heavy = [i for i in qs if "heavy" in i]
    rk = f(bk, "R3 radius, 1 kg [km]")
    lines = [f"- **Same pack**: {short(bk)} flies the R3 mission (1 kg, 8 m/s wind) to {rk:.1f} km radius; "
             + ", ".join(f"{short(i)} to {f(i, 'R3 radius, 1 kg [km]'):.1f} km" for i in light)
             + f" — on the same {df.loc[bk, 'pack']} only K4TQ closes the 10 km requirement (still-air range "
             f"{f(bk, 'still-air range [km]'):.0f} km against {max(f(i, 'still-air range [km]') for i in light):.0f})."]
    for i in heavy:
        r = f(i, "R3 radius, 1 kg [km]")
        lines.append(f"- **Same take-off mass**: {short(i)} puts K4TQ's tilt-wing tax into cells ({df.loc[i, 'pack']}, {f(i, 'take-off mass [kg]'):.2f} kg) "
                     f"and reaches {r:.1f} km — {'it closes R3 too' if r >= 10 else 'still short of R3'}, {100 * (1 - r / rk):.0f} % under K4TQ, "
                     f"with {df.loc[i, 'pack'].split('S')[1].split('P')[0]}P against 3P: more than twice the cells to buy, charge and ship as dangerous goods, "
                     f"and T/W {f(i, 'T/W'):.2f} (forward climb {f(i, 'best forward climb (80 %) [m/s]'):.1f} m/s, turn {f(i, 'sustained turn (best) [deg/s]'):.0f} deg/s).")
    hq = max(qs, key=lambda i: f(i, "hover minutes (whole pack)"))
    lines.append(f"- **Hover**: the quads win — {short(hq)} {f(hq, 'hover minutes (whole pack)'):.0f} min, Q-17 on the same pack "
                 f"{f(light[-1], 'hover minutes (whole pack)'):.0f} min, against {max(f(i, 'hover minutes (whole pack)') for i in k4):.0f} min for K4TQ "
                 "(bigger, flatter-pitched discs, no wings in the downwash, no tilt-wing tax).")
    lines.append(f"- **Agility**: the light quads turn and climb harder at low speed ({max(f(i, 'sustained turn (best) [deg/s]') for i in light):.0f} against "
                 f"{max(f(i, 'sustained turn (best) [deg/s]') for i in k4):.0f} deg/s) and roll/pitch faster in the hover; K4TQ yaws two to three times as hard in the hover "
                 "(the wings' differential tilt against the rotors' reaction torque) and is the steadier platform in the cruise.")
    lines.append(f"- **Speed**: K4TQ {max(f(i, 'max level speed [m/s]') for i in k4):.0f} m/s at its drive's limit (16x10 on KV300); the quads reach the "
                 "model's 30 m/s sweep end — optimistic: the edgewise propeller losses and the body's lift at 30° disc tilt are not modelled. "
                 "At the speeds that matter for range the quads burn 2–3× K4TQ's power.")
    return "\n".join(lines)


# ================================================================================================= a 15-inch FPV quad
FPV15 = {   # a long-range / heavy 15" FPV build in K4TQ's parts (estimates): carbon X frame 600–700 mm, 6S Li-ion
    "frame (15\" carbon X, arms + plates + standoffs)": 0.55, "4-in-1 ESC 60 A": 0.06, "FC + VTX + camera + RX + GNSS": 0.15,
    "wiring, XT90, straps, antenna mounts": 0.08,
}
FPV15_CDA = 0.045           # m²: the frame, the pack slung under it, the motors and the props' hubs (ESTIMATE; a payload adds ~0.01)


def fpv15(payload_kg: float = 0.0, *, prop: str = "15x8", motor: str = "MN5006-450", battery: str = "6S2P-P45B", tilt_max_deg: float = 65.0) -> dict:
    """A 15-inch FPV quadcopter built from the same catalogue and drive model: the mass with the pack, the hover, the
    top speed (full throttle, the discs tilted until the thrust's horizontal part meets the drag; the thrust at the
    axial inflow V·sin(tilt), current-limited) — ESTIMATE / CALC."""
    cat = k4tq.CATALOGUE
    m = sum(FPV15.values()) + 4 * (cat["motor"][motor]["mass_kg"] + cat["propeller"][prop]["mass_kg"]) + cat["battery"][battery]["mass_kg"] + payload_kg
    W = m * G
    drv = sz.drive({"prop": prop, "motor": motor, "battery": battery}, V=tuple(range(0, 50, 2)))
    CdA = FPV15_CDA + (0.010 if payload_kg > 0 else 0.0)
    v_max = 0.0
    for V in np.arange(1.0, 48.01, 0.5):
        D = 0.5 * RHO * V * V * CdA
        a = math.atan2(D, W)
        if math.degrees(a) > tilt_max_deg:
            break
        if math.hypot(W, D) / 4 <= sz.max_thrust(drv, V * math.sin(a)):
            v_max = float(V)
        else:
            break
    hp = sz.drive_point(drv, 0.0, W / 4)
    bat = k4tq.battery({"battery": battery})
    aw = 12.0                                                            # FC, video, link, GNSS
    # forward flight as k4tq_sizing.quad_reference: Glauert's induced power, the hover's profile power grown with μ²
    from ogma import rotors as og_rotors
    R_ = cat["propeller"][prop]["d_in"] * 0.0254 / 2
    A = math.pi * R_ ** 2
    eta = hp["shaft_W"] / hp["electrical_W"]
    P0 = hp["shaft_W"] - 1.15 * (W / 4) * og_rotors.hover_induced_velocity(W / 4, A)
    omega_R = hp["rpm"] * 2 * math.pi / 60 * R_
    T_max0 = sz.max_thrust(drv, 0.0)
    rows = []
    for V in np.arange(0.0, v_max + 0.01, 1.0):
        D = 0.5 * RHO * V * V * CdA
        T = math.hypot(W, D) / 4
        a = math.atan2(D, W)
        vi = og_rotors.induced_velocity(T, V, math.pi / 2 - a, A)
        mu = V * math.cos(a) / omega_R
        P_el = 4 * (1.15 * T * (vi + V * math.sin(a)) + P0 * (1 + 4.65 * mu ** 2)) / eta + aw
        rows.append({"V [m/s]": V, "electrical [W]": P_el, "endurance [min]": bat["usable_wh"] / P_el * 60, "range [km]": bat["usable_wh"] / P_el * V * 3.6,
                     "T each / max": T / T_max0})
    tab = pd.DataFrame(rows).set_index("V [m/s]")
    P_h = 4 * hp["electrical_W"] + aw
    q = {"table": tab, "hover [W]": P_h, "mass [kg]": m}
    ok = tab[tab["T each / max"] < 0.8]
    return {"mass with the pack [kg]": m, "payload [kg]": payload_kg, "pack": battery, "T/W": 4 * T_max0 / W,
            "hover [W]": P_h, "max speed [m/s]": v_max, "hover minutes": bat["usable_wh"] / P_h * 60,
            "best range V [m/s]": float(ok["range [km]"].idxmax()), "still-air range [km]": float(ok["range [km]"].max()),
            "R3 radius [km]": sz.quad_mission_radius(q, {"battery": bat}), "table": tab, "label": sz.label("CALC")}


def speed_mass_table(designs=("A1", "F1", "G", "B"), fpv_payloads=(0.0, 1.0, 2.0)) -> pd.DataFrame:
    """Mass with the pack (empty of payload, and at take-off), the top level speed and the range: the K4TQ designs (four
    motors for the top speed, two for the cruise; the sweep to 44 m/s) against the 15-inch FPV quad. Range: still air
    at the best-range speed on the usable pack, and the R3 radius (2 min hover, 20 % reserve, 8 m/s wind)."""
    import k4tq_scaled as ks
    rows = {}
    for d in designs:
        v = sz.vehicle(d)
        c4 = sz.cruise(v, n_running=4, speeds=np.arange(14.0, 45.0, 1.0))
        b4 = sz.best_speeds(c4)
        b2 = sz.best_speeds(sz.cruise(v, n_running=2, speeds=np.arange(14.0, 45.0, 1.0)))
        p = v["p"]
        nick = k4tq.NICKNAME.get(d, "")
        rows[f"K4TQ-{d}" + (f" ({nick})" if nick else "")] = {
            "mass with the pack, no payload [kg]": v["mass"]["mass_kg"] - p["payload_kg"], "payload [kg]": p["payload_kg"],
            "take-off mass [kg]": v["mass"]["mass_kg"], "pack": p["battery"], "max level speed [m/s]": b4["max level V [m/s]"],
            "max level speed [km/h]": 3.6 * b4["max level V [m/s]"], "cruise (best range) [km/h]": 3.6 * b2["best range V [m/s]"],
            "still-air range [km]": b2["range [km]"], "R3 radius [km]": ks.radius(v), "hover minutes": sz.hover(v)["hover minutes on the usable pack"]}
    for pl in fpv_payloads:
        f = fpv15(pl)
        rows[f"15\" FPV quad, {pl:g} kg payload"] = {
            "mass with the pack, no payload [kg]": f["mass with the pack [kg]"] - pl, "payload [kg]": pl, "take-off mass [kg]": f["mass with the pack [kg]"],
            "pack": f["pack"], "max level speed [m/s]": f["max speed [m/s]"], "max level speed [km/h]": 3.6 * f["max speed [m/s]"],
            "cruise (best range) [km/h]": 3.6 * f["best range V [m/s]"], "still-air range [km]": f["still-air range [km]"],
            "R3 radius [km]": f["R3 radius [km]"], "hover minutes": f["hover minutes"]}
    df = pd.DataFrame(rows).T
    df.attrs["label"] = sz.label("CALC")
    return df
