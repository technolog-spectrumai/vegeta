"""RAPTORS first-cut sizing: the configurations that could serve each product line (BUTEO, ARPELLA, ASTOR, SPARFELL),
compared on one transparent model before any of them gets CAD, CFD or a 6-DOF simulation.

Every number here is an ESTIMATE (EST in KPIS.md): a parabolic polar (CD = CD0 + CL^2 / (pi e AR)), a mass build-up
from areal and linear densities (the ARIGO mass budget's, ``arigo_mass``), momentum theory for hover, efficiencies by
hand. Where a configuration already has a better model it is used as an anchor: ARIGO-1H (screening CFD, the run-1
sweep), GAVILAN-X (``data/gavilan_flight_card.json``). The point is the comparison between configurations for a job,
not the absolute values; each line's chosen configuration then gets its own CAD, coefficients and simulation.

    python labs/aero/designs/raptors_sizing.py          # the table, and data/raptors_sizing.yaml
"""
from __future__ import annotations

import math
from pathlib import Path

RHO, G = 1.225, 9.80665
ETA_M = 0.85            # HAND: motor x ESC
ETA_P = 0.70            # HAND: propeller, cruise to dash
ETA_FAN = 0.55          # HAND: ducted fan, overall at dash
FM = 0.65               # HAND: hover figure of merit (small rotors)
P_AV = {"std": 15.0, "lite": 10.0}      # HAND: avionics power [W]
AV_G = {"std": 295.0, "lite": 205.0}    # flight controller, vision computer, camera, radio, wiring, skid (arigo_mass)
WING_KG_M2, TAIL_KG_M2, FUSE_KG_M = 1.55, 0.90, 0.28     # arigo_mass (HAND +-20 %)
EPP_WING_KG_M2 = 1.25   # HAND: an EPP flying wing with a carbon spar (+-20 %)
LIION_P = (225.0, 45.4)  # g, Wh per parallel string of 3S Molicel P42A (arigo_mass)
LIPO_G_PER_WH = 6.76     # g per Wh, 6S LiPo class (150 g / Ah at 22.2 V; arigo_mass HAND)
HAND_THROW = 12.0
WIND, GUST = 12.0, 18.0  # the B6 threshold

# ---------------------------------------------------------------------------------------------- the candidates
# kind: fixed / tailsitter / liftcruise; tail: share of the wing area in the tail (0: tailless)
C = {
    # BUTEO: point defence, on station, unattended
    "B1 fixed-wing perch drop": dict(line="BUTEO", kind="fixed", b=1.05, AR=4.5, tail=0.40, fus=0.72, CD0=0.033, CLmax=1.0,
        P=800, pack=("liion", 2), av="std", kits=40, pieces=1,
        note="ARIGO-1H class; perch with a spring and a turning station (run 1); recovery: belly landing in a field, someone carries it back"),
    "B2 VTOL tailsitter (X-wing)": dict(line="BUTEO", kind="tailsitter", anchor="gavilan",
        note="GAVILAN-X (notebook 38): stands on a charging perch, launches and lands vertically on it"),
    "B3 VTOL lift + cruise": dict(line="BUTEO", kind="liftcruise", b=1.05, AR=5.5, tail=0.35, fus=0.72, CD0=0.045, CLmax=1.0,
        P=800, pack=("liion", 2), av="std", kits=40, lift=dict(n=4, D=0.203, P_each=350, g_each=95.0), pieces=1,
        note="a fixed wing with four lift rotors on two booms: takes off from and lands on a charging pad; wing level in the hover"),
    # ARPELLA: area patrol, 10 km radius
    "A1 ARIGO-1H 3S4P (run 1 best)": dict(line="ARPELLA", kind="fixed", b=1.25, AR=5.0, tail=0.40, fus=0.72, CD0=0.033, CLmax=1.0,
        P=800, pack=("liion", 4), av="std", kits=25, pieces=1, note="run 1's best ARPELLA"),
    "A2 high-AR patroller": dict(line="ARPELLA", kind="fixed", b=1.80, AR=10.0, tail=0.25, fus=0.85, CD0=0.026, CLmax=1.15,
        P=500, pack=("liion", 4), av="std", kits=25, pieces=2,
        note="a long, slender pusher wing (two-piece), V or T tail, gimbal camera; quiet loiter, bungee or hand launch"),
    "A3 motor-glider": dict(line="ARPELLA", kind="fixed", b=2.40, AR=14.0, tail=0.20, fus=1.0, CD0=0.022, CLmax=1.25,
        P=400, pack=("liion", 4), av="std", kits=25, pieces=3, note="sailplane wing: the longest patrol, a bungee launch, wind-sensitive"),
    # ASTOR: fast response from a launcher
    "S1 ARIGO-20 (twin EDF)": dict(line="ASTOR", kind="fixed", b=0.95, AR=4.3, tail=0.45, fus=0.82, CD0=0.035, CLmax=1.0,
        P=2400, fan=True, pack=("lipo", 133), av="std", kits=95, pieces=1, note="ARIGO-20 with a 6S 6000 pack, rail + parachute"),
    "S2 propeller dart": dict(line="ASTOR", kind="fixed", b=0.80, AR=4.0, tail=0.35, fus=0.72, CD0=0.026, CLmax=0.95,
        P=1400, pack=("liion", 3), av="std", kits=95, pieces=1,
        note="short swept wing, slim fuselage, a 1.4 kW pusher on a folding prop; rail or tube, parachute (run 1: the 800 mm ARIGO-1H)"),
    "S3 tube-launched folding dart": dict(line="ASTOR", kind="fixed", b=0.80, AR=5.0, tail=0.30, fus=0.70, CD0=0.024, CLmax=0.9,
        P=1400, pack=("liion", 3), av="std", kits=135, pieces=1, folds=True,
        note="S2 with wings and fins that fold along the body in a 120 mm tube: a sealed canister, launched by gas; +40 g of hinges"),
    # SPARFELL: one person, a backpack
    "P1 ARIGO-1H lite 3S1P (run 1 best)": dict(line="SPARFELL", kind="fixed", b=0.95, AR=4.3, tail=0.40, fus=0.72, CD0=0.033, CLmax=1.0,
        P=800, pack=("liion", 1), av="lite", kits=40, pieces=2, note="run 1's best SPARFELL"),
    "P2 compact flying wing": dict(line="SPARFELL", kind="fixed", b=0.80, AR=4.0, tail=0.0, fus=0.0, CD0=0.022, CLmax=0.85,
        P=350, pack=("liion", 1), av="lite", kits=30, pieces=2, epp=True,
        note="tailless EPP wing in two halves, a pod for the avionics, a small folding pusher; nothing sticks out to break"),
    "P3 small conventional (folding)": dict(line="SPARFELL", kind="fixed", b=0.80, AR=5.5, tail=0.30, fus=0.50, CD0=0.030, CLmax=1.0,
        P=350, pack=("liion", 1), av="lite", kits=40, pieces=2, note="a scaled-down ARIGO with a 500 mm fuselage"),
}


def _mass(c):
    S = c["b"] ** 2 / c["AR"]
    wing = (EPP_WING_KG_M2 if c.get("epp") else WING_KG_M2) * S * 1000
    tail = TAIL_KG_M2 * c["tail"] * S * 1000
    fus = FUSE_KG_M * c["fus"] * 1000
    prop = (40.0 + 0.12 * c["P"]) + (0 if c.get("fan") else 30.0) + (110.0 if c.get("fan") else 0.0)   # motors/ESC, prop or fan units
    lift = 0.0
    if c.get("lift"):
        L = c["lift"]
        lift = L["n"] * L["g_each"] + 70.0                         # rotors, ESCs, motors + two booms
    servos = 48.0
    items = wing + tail + fus + prop + lift + servos + AV_G[c["av"]] + c["kits"]
    items *= 1.10                                                    # growth margin
    if c["pack"][0] == "liion":
        pg, wh = LIION_P[0] * c["pack"][1], LIION_P[1] * c["pack"][1]
    else:
        wh = c["pack"][1]
        pg = LIPO_G_PER_WH * wh
    return S, (items + pg) / 1000.0, pg / 1000.0, wh


def size(name, c) -> dict:
    if c["kind"] == "tailsitter" and c.get("anchor") == "gavilan":
        return _gavilan(name, c)
    S, m, mp, wh = _mass(c)
    W = m * G
    k = 1.0 / (math.pi * 0.8 * c["AR"])
    D = lambda V: 0.5 * RHO * V * V * S * (c["CD0"] + k * (W / (0.5 * RHO * V * V * S)) ** 2)
    Vs = math.sqrt(W / (0.5 * RHO * S * c["CLmax"]))
    eta = ETA_FAN if c.get("fan") else ETA_P
    Pa = c["P"] * ETA_M * eta
    Vs_ = [Vs * 1.2 + 0.1 * i for i in range(800)]
    Vmax = max(V for V in Vs_ if D(V) * V <= Pa) if any(D(V) * V <= Pa for V in Vs_) else Vs
    roc = max((Pa - D(V) * V) / W for V in Vs_ if V < Vmax)
    Pav = P_AV[c["av"]]
    Preq = [(V, D(V) * V / (ETA_P * ETA_M) + Pav) for V in Vs_ if V < Vmax]
    Ve, Pe = min(Preq, key=lambda x: x[1])
    Vr, Pr = min(Preq, key=lambda x: x[1] / x[0])
    E = wh * 0.85 * 3600.0
    n_i = min(0.5 * RHO * 20 ** 2 * S * c["CLmax"] / W, 6.0)
    out = {"line": c["line"], "kind": c["kind"], "note": c["note"], "span_m": c["b"], "S_m2": round(S, 3), "AR": c["AR"],
           "mass_kg": round(m, 2), "pack_kg": round(mp, 2), "pack_Wh": round(wh), "wing_loading_kg_m2": round(m / S, 1),
           "stall_m_s": round(Vs, 1), "max_speed_m_s": round(Vmax, 1), "climb_m_s": round(roc, 1),
           "endurance_min": round(E / Pe / 60.0, 0), "endurance_speed_m_s": round(Ve, 1), "loiter_W": round(Pe, 0),
           "range_km": round(E / Pr * Vr / 1000.0, 0), "full_power_min": round(E / (c["P"] + Pav) / 60.0, 1),
           "throw_margin": round(HAND_THROW / Vs, 2), "turn_20_m_s_deg_s": round(math.degrees(G * math.sqrt(n_i ** 2 - 1) / 20.0), 0),
           "packed_mm": round(max(c["fus"] * 1000, c["b"] * 1000 / c["pieces"] + (20 if c["pieces"] > 1 else 0)) if not c.get("folds") else 720, 0),
           "max_dyn_pressure_Pa": round(0.5 * RHO * Vmax ** 2, 0)}
    if c.get("lift"):
        L = c["lift"]
        A = L["n"] * math.pi * (L["D"] / 2) ** 2
        Ph = W ** 1.5 / math.sqrt(2 * RHO * A) / FM / ETA_M + Pav
        Tmax = L["n"] * (L["P_each"] * ETA_M * FM * math.sqrt(2 * RHO * A / L["n"])) ** (2 / 3)    # momentum: T^1.5 ~ P
        side = 0.5 * RHO * GUST ** 2 * (c["fus"] * 0.08 + 0.15 * c["tail"] * S) * 1.2   # fuselage + fin side, gust
        out.update(_hover(Ph, E, W, Tmax, side))
    return out


def _hover(Ph, E, W, Tmax, side):
    tw = Tmax / W
    tilt_max = math.degrees(math.acos(min(1.0, 1.0 / tw)))
    tilt = math.degrees(math.atan(side / W))
    return {"hover_W": round(Ph, 0), "hover_min": round(E / Ph / 60.0, 1), "hover_T_W": round(tw, 2),
            "gust_tilt_deg": round(tilt, 0), "tilt_available_deg": round(tilt_max, 0),
            "holds_18_m_s_gust_in_hover": bool(tilt < 0.8 * tilt_max)}


def _gavilan(name, c):
    import json
    gc = json.loads((Path(__file__).resolve().parent / "data" / "gavilan_flight_card.json").read_text())
    W = gc["mass_kg"] * G
    S_side = 0.48 * 0.22 + 0.3 * 0.05 * 4                        # HAND: the X panels' area seen side-on in the hover (worst axis)
    side = 0.5 * RHO * GUST ** 2 * S_side * 1.2
    E = 1.7 * 22.2 * 0.85 * 3600.0                               # its 6S 1700 mAh pack (the mass table), usable
    Ph = E / (gc["endurance_min"]["hover"] * 60.0)
    out = {"line": c["line"], "kind": "tailsitter", "note": c["note"], "span_m": gc["span_m"], "mass_kg": gc["mass_kg"],
           "stall_m_s": 0.0, "max_speed_m_s": gc["max_speed_m_s"], "climb_m_s": gc["climb_rate_m_s"],
           "endurance_min": gc["endurance_min"]["cruise"], "range_km": round(gc["endurance_min"]["cruise"] * 60 * 17 / 1000, 0),
           "full_power_min": gc["endurance_min"]["full_power"], "throw_margin": None,
           "turn_20_m_s_deg_s": 153.0, "packed_mm": 580, "hover_to_25_m_s_s": gc["hover_to_25_m_s"]["time_s"],
           "source": "GAVILAN-X flight card (reduced point-mass models, notebook 38)"}
    out.update(_hover(Ph, E, W, gc["max_acceleration_g"] * W + W, side))
    return out


def table() -> dict:
    return {n: size(n, c) for n, c in C.items()}


if __name__ == "__main__":
    import yaml
    T = table()
    cols = ["mass_kg", "span_m", "stall_m_s", "max_speed_m_s", "climb_m_s", "endurance_min", "range_km", "full_power_min",
            "throw_margin", "turn_20_m_s_deg_s", "packed_mm", "hover_min", "gust_tilt_deg", "tilt_available_deg"]
    print(f"{'candidate':36s}" + "".join(f"{x[:10]:>11s}" for x in cols))
    for n, r in T.items():
        print(f"{n:36s}" + "".join(f"{str(r.get(x, '')):>11s}" for x in cols))
    (Path(__file__).resolve().parent / "data" / "raptors_sizing.yaml").write_text(yaml.safe_dump(T, sort_keys=False, width=140))
