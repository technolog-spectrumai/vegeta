"""K4TQ Stage 1 — engineering design and feasibility (``notebooks/k4_plan.md`` steps 1.4–1.16).

First-principles sizing on the parameters of ``k4tq`` (the single source of truth). Every analysis returns plain
dicts / DataFrames with a ``label`` saying what kind of number it is:

- ESTIMATE — a hand formula or a correlation (drag build-up, stall rule, structure beam rules, Raymer's Cnβ);
- CALC — a numerical model run here (the lattice ``og.aero.vlm``, Boreas BEMT drive maps, the transition corridor);
- MOCK — anything a mocked solver produced (``vegeta.mock``: the BEMT is coarsened, Zephyr's polar is thin-airfoil).

Nothing here is a solver result (no CFD, no FEA): Stage 3 / 4 prepare those runs. What each analysis reuses:
``og.aero`` (vlm, Polar, cf_mixed, viterna_cd90), ``og.rotors`` (Glauert inflow, momentum hover), ``og.mass``
(through ``k4tq.mass_table``), ``og.propulsion.PropulsionMap`` filled from ``vegeta.boreas`` (BEMT + motor), the
tandem method of ``kestrel_tandem.aero`` (slopes against one wing alone, the front share, the stall rule).

Run ``python k4tq_sizing.py`` (mocks on, as R12 says) to write ``data/k4tq/phase1.md``, ``phase1_spec.json`` and
``bom_phase1.csv``.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from ogma import aero as og_aero
from ogma import rotors as og_rotors
from ogma.propulsion import PropulsionMap

import k4tq

G, RHO = k4tq.G, k4tq.RHO
MISSION = k4tq.MISSION
NU = 1.46e-5
DATA = Path(__file__).resolve().parent / "data" / "k4tq"

# ------------------------------------------------------------------------------------------------ fixed assumptions
# Section constants recorded from NeuralFoil (vegeta.zephyr.analyze, Re 2.0e5, n_crit 6, 2026-10-21), because under the
# mocks (R12) Zephyr returns the thin-airfoil stand-in; ``section_check`` reruns them when the mocks are off.
SECTIONS = {
    "NACA 2412": {"cl_max": 1.23, "alpha_cl_max_deg": 15.0, "cd0": 0.0095, "cm0": -0.050, "alpha0_deg": -2.1, "ld_max": 64.5},
    "NACA 4412": {"cl_max": 1.34, "alpha_cl_max_deg": 18.0, "cd0": 0.0102, "cm0": -0.098, "alpha0_deg": -4.1, "ld_max": 76.4},
    "NACA 2415": {"cl_max": 1.23, "alpha_cl_max_deg": 16.0, "cd0": 0.0105, "cm0": -0.043, "alpha0_deg": -2.1, "ld_max": 63.1},
    "NACA 4415": {"cl_max": 1.32, "alpha_cl_max_deg": 14.0, "cd0": 0.0113, "cm0": -0.088, "alpha0_deg": -4.0, "ld_max": 71.1},
}
SECTION_SOURCE = "NeuralFoil via vegeta.zephyr, Re 2.0e5, n_crit 6 (recorded 2026-10-21)"
BUILT_UP = 0.90                   # the built-up wing's cl_max against the section's (as kestrel_fast)
STATIC_MARGIN = 0.12              # of the MAC, the CG ahead of the neutral point (an actively stabilised UAV; Kestrel-Q4T used 10 % of the stagger ≈ 0.3 MAC)
CRUISE_MIN_OVER_STALL = 1.25      # wing-borne cruise at or above 1.25 × the stall speed (gust margin); below it the wings tilt up
OSWALD_VISCOUS = 0.90             # e = 0.9 × the lattice's
INTERFERENCE = 1.10               # the build-up's junction allowance (as kestrel_flight)
ROOT_GAP_CDA = 0.0004             # m² per wing root: the gap and the cartridge fairing of a rotating joint (ESTIMATE)
FOLDED_BLADE_CDA = 0.0010         # m² per stopped propeller, blades folded along the nacelle (ESTIMATE)
AVIONICS_W = 19.0                 # flight stack, telemetry, lights 12 W; the temperature logger and cartridge fan 3 W; servos 4 W — the baseline's;
                                  # every analysis takes the vehicle's own load from k4tq.avionics_power(p) (K4TQ-A2 adds the Jetson companion and perception)
V_BATT_LOADED = 3.5               # V per cell, mid-discharge under load (Li-ion); sag case 3.3 V
DOWNLOAD = 1.03                   # hover thrust over weight: the edge-on wings and the nacelles in the slipstream
HOVER_FM_DRIVE = None             # (from the BEMT map)
N_LIMIT = 3.8                     # limit load factor (manoeuvre + alleviated gust, small UAV practice)
ULTIMATE = 1.5                    # ultimate / limit
PRINTED_FACTOR = 2.0              # additional factor on printed polymer parts
SPAR_F1_MIN = 12.0                # Hz: the spar-alone first bending with the motor (the skins add stiffness; Stage 4 modal FEA decides) kept clear of the 30 Hz wing-pitch mode
TILT_TIME_S = 2.0                 # a full 0→90° conversion of the wings (the transition schedule is slower; this is the actuator's sizing rate)


def mocked() -> bool:
    try:
        from vegeta import mock
        return bool(mock.installed())
    except Exception:
        return False


def label(kind: str) -> str:
    """CALC becomes MOCK when the mocks are installed (the numerical models that run a mocked tool)."""
    return "MOCK" if (kind == "CALC" and mocked()) else kind


# ================================================================================================= sections
def section_check(name: str = "NACA 2412", Re: float = 2.0e5) -> dict:
    """The section polar from Zephyr (NeuralFoil) — the recorded constants against it. Under the mocks Zephyr returns
    the thin-airfoil stand-in, so the row is labelled MOCK and the recorded constants are what the sizing uses."""
    from vegeta import zephyr
    s = zephyr.naca(name.split()[-1])
    pol = zephyr.analyze(s, np.arange(-4.0, 20.0, 1.0), Re=Re, n_crit=6)
    i = int(np.argmax(pol.cl))
    a0 = float(np.interp(0.0, pol.cl, pol.alpha_deg))
    return {"section": name, "Re": Re, "cl_max": float(pol.cl[i]), "alpha_cl_max_deg": float(pol.alpha_deg[i]), "alpha0_deg": a0,
            "cd0": float(pol.cd[int(np.argmin(np.abs(pol.alpha_deg)))]), "recorded": SECTIONS.get(name), "source": pol.source,
            "label": "MOCK" if mocked() else "CALC"}


def section(p: dict, which: str = "front") -> dict:
    name = k4tq.section_of(p, which)
    return {"name": name, **SECTIONS[name], "source": SECTION_SOURCE}


def sections(p: dict) -> dict:
    return {w: section(p, w) for w in ("front", "rear")}


def _tc(name: str) -> float:
    """t/c from a NACA designation's last two digits (0.12 otherwise)."""
    d = name.split()[-1]
    return int(d[-2:]) / 100 if d.isdigit() else 0.12


# ================================================================================================= drive maps
def _boreas_parts(p: dict, prop_key: str | None = None, motor_key: str | None = None):
    from vegeta import boreas
    pr = k4tq.CATALOGUE["propeller"][prop_key or p["prop"]]
    mo = k4tq.CATALOGUE["motor"][motor_key or p["motor"]]
    D, P = boreas.inches(pr["d_in"], pr["p_in"])
    s = pr["d_in"] / 10.0                                  # the generic 10-inch planform of the Kestrel props, scaled
    prop = boreas.Propeller.from_pitch(f"{pr['d_in']}x{pr['p_in']}", D, P, 2, chord_root_m=0.018 * s, chord_max_m=0.026 * s, chord_tip_m=0.010 * s,
                                       mass_kg=pr["mass_kg"], rotor_mass_kg=pr["mass_kg"] + 0.05)
    motor = boreas.Motor(mo["name"], kv_rpm_per_volt=mo["kv"], resistance_ohm=mo["R_ohm"], no_load_current_a=mo["I0_A"], max_current_a=mo["I_max_A"],
                         mass_kg=mo["mass_kg"])
    return prop, boreas.Airfoil(), motor


def drive(p: dict, *, prop_key: str | None = None, motor_key: str | None = None, cells: int | None = None, v_cell: float = V_BATT_LOADED,
          V=tuple(range(0, 34, 2))) -> PropulsionMap:
    """The drive (one propeller on one motor) as an ``og.propulsion.PropulsionMap`` over airspeed × rpm from Boreas's
    BEMT and the first-order motor, at the loaded pack voltage (``cells`` in series: the pack's); cached in
    ``data/k4tq/`` per components and mode."""
    prop_key, motor_key = prop_key or p["prop"], motor_key or p["motor"]
    cells = cells or k4tq.CATALOGUE["battery"][p["battery"]]["cells_s"]
    if p.get("scale", 1.0) > 1.0:                                        # K4TQ-B flies faster: the map to 44 m/s
        V = tuple(range(0, 46, 2))
    return _drive(prop_key, motor_key, cells, v_cell, tuple(V), mocked())


@lru_cache(maxsize=32)
def _drive(prop_key, motor_key, cells, v_cell, V, mock_flag) -> PropulsionMap:
    from vegeta import boreas
    key = {"prop": prop_key, "motor": motor_key, "cells": cells, "v_cell": v_cell, "V": list(V), "mock": mock_flag,
           "prop_row": k4tq.CATALOGUE["propeller"][prop_key], "motor_row": k4tq.CATALOGUE["motor"][motor_key]}
    f = DATA / f"drive_{motor_key}_{prop_key}_{cells}S_{v_cell:g}V{'_mock' if mock_flag else ''}.json"
    if f.exists():
        doc = json.loads(f.read_text())
        if doc.get("key") == json.loads(json.dumps(key)):
            return PropulsionMap.from_json(doc["map"])
    prop, airfoil, motor = _boreas_parts({}, prop_key, motor_key)
    vb = cells * v_cell
    rpm = np.linspace(800.0, motor.kv_rpm_per_volt * vb * 0.98, 28)
    shape = (len(V), len(rpm))
    T, Q, Ps, I, U = (np.zeros(shape) for _ in range(5))
    for i, v in enumerate(V):
        for j, n in enumerate(rpm):
            op = boreas.solve(prop, airfoil, float(n), float(v))
            T[i, j], Q[i, j], Ps[i, j] = op.thrust, op.torque, op.power
            U[i, j], I[i, j] = motor.voltage_for(float(n), max(op.torque, 0.0))
    m = PropulsionMap(np.asarray(V, float), rpm, T, Ps, Q, I, U, vb, 0.96, motor.name,
                      {"prop": prop_key, "source": "Boreas BEMT (generic section) + first-order motor" + (" — MOCK (coarse BEMT)" if mock_flag else "")})
    DATA.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"key": key, "map": m.to_json()}))
    return m


def max_thrust(m: PropulsionMap, V: float, I_max: float | None = None) -> float:
    """The drive's maximum thrust at axial speed ``V``: full throttle, or the throttle where the motor reaches its
    current limit ``I_max`` (the map's motor row in ``k4tq.CATALOGUE``)."""
    I_max = I_max or _imax(m)
    full = m.at(V, 1.0)
    if full["current"] <= I_max:
        return full["thrust"]
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if m.at(V, mid)["current"] > I_max:
            hi = mid
        else:
            lo = mid
    return m.at(V, lo)["thrust"]


def _imax(m: PropulsionMap) -> float:
    for row in k4tq.CATALOGUE["motor"].values():
        if row["name"] == m.motor_name:
            return row["I_max_A"]
    return math.inf


def drive_point(m: PropulsionMap, V: float, T: float) -> dict:
    """Electrical power, current, rpm and throttle of one drive for thrust ``T`` at axial speed ``V`` (inf if beyond its
    current-limited maximum)."""
    tmax = max_thrust(m, V)
    if T > tmax + 1e-9:
        return {"feasible": False, "electrical_W": math.inf, "T_max_N": tmax}
    th = m.throttle_for_thrust(V, max(T, 0.0))
    pt = m.at(V, th)
    return {"feasible": True, "electrical_W": pt["electrical"], "current_A": pt["current"], "rpm": pt["rpm"], "throttle": th, "T_max_N": tmax,
            "shaft_W": pt["shaft_power"]}


# ================================================================================================= tandem aerodynamics
def _quads(L: dict, tilt: dict | None = None) -> tuple[list, list]:
    """Four trapezoid quads (front L/R, rear L/R) from the centreline to the tips (the fuselage carries over), and
    their incidences (the decalage on the front pair)."""
    quads, inc = [], []
    for which in ("front", "rear"):
        w = L["wings"][which]
        pc = w["pivot_chord"]
        xr_le = w["x_pivot"] - pc * w["root_chord"]
        xt_le = w["x_pivot"] - pc * w["tip_chord"]
        y1, z = w["y1"], w["z"]
        R = [[xr_le, 0.0, z], [xt_le, y1, z], [xt_le + w["tip_chord"], y1, z], [xr_le + w["root_chord"], 0.0, z]]
        Lq = [[xt_le, -y1, z], [xr_le, 0.0, z], [xr_le + w["root_chord"], 0.0, z], [xt_le + w["tip_chord"], -y1, z]]
        quads += [np.array(Lq), np.array(R)]
        d = L["p"]["decalage_deg"] if which == "front" else 0.0
        inc += [d, d]
    return quads, inc


def tandem(L: dict, *, n_span: int = 14, n_chord: int = 3) -> dict:
    """The lattice on the two tilt wings at cruise tilt (``og.aero.vlm``): the pair's lift slope on S_ref, the
    induced-drag factor, the neutral point, the front share of the lift, each wing's slope against one wing alone
    (the rear one in the front's downwash), CL at α = 0 from the decalage. As ``kestrel_tandem.aero``."""
    qs, inc = _quads(L)
    S, c = L["S_ref"], L["c_ref"]
    x_ref = L["wings"]["front"]["x_ac_mac"]
    kw = dict(S_ref=S, c_ref=c, x_ref=x_ref, n_span=n_span, n_chord=n_chord)
    r0 = og_aero.vlm(qs, alpha_deg=0.0, incidence_deg=inc, **kw)
    r1 = og_aero.vlm(qs, alpha_deg=4.0, incidence_deg=inc, **kw)
    da = math.radians(4.0)
    CLa = (r1["CL"] - r0["CL"]) / da
    Cma = (r1["Cm"] - r0["Cm"]) / da
    k = (r1["CDi"] - r0["CDi"]) / (r1["CL"] ** 2 - r0["CL"] ** 2)
    x_np = x_ref - Cma / CLa * c
    lat = r1["lattice"]
    front = lat["quad"] < 2
    dLf = r1["F"][front, 2].sum() - r0["F"][front, 2].sum()
    dLr = r1["F"][~front, 2].sum() - r0["F"][~front, 2].sum()
    share4 = float(r1["F"][front, 2].sum() / r1["F"][:, 2].sum())
    one1 = og_aero.vlm(qs[:2], alpha_deg=4.0, S_ref=L["wings"]["front"]["S_pair"], c_ref=c, x_ref=x_ref, n_span=n_span, n_chord=n_chord)
    one0 = og_aero.vlm(qs[:2], alpha_deg=0.0, S_ref=L["wings"]["front"]["S_pair"], c_ref=c, x_ref=x_ref, n_span=n_span, n_chord=n_chord)
    CLa_one = (one1["CL"] - one0["CL"]) / da
    Sf, Sr = L["wings"]["front"]["S_pair"], L["wings"]["rear"]["S_pair"]
    slope_f = dLf / Sf / da / CLa_one
    slope_r = dLr / Sr / da / CLa_one
    AR = L["AR_total"]
    return {"CLa": CLa, "CLa_one_pair": CLa_one, "k": k, "e": 1 / (math.pi * AR * k), "k_one_pair": one1["CDi"] / one1["CL"] ** 2, "x_np": x_np,
            "Cma": Cma, "CL0": r0["CL"], "Cm0": r0["Cm"], "front_share_alpha4": share4, "slope_front": slope_f, "slope_rear": slope_r,
            "label": "CALC", "method": "og.aero.vlm, 4 quads, the decalage on the front pair"}


def cg_target(L: dict, a: dict) -> float:
    return a["x_np"] - STATIC_MARGIN * L["c_ref"]


def stall(L: dict, a: dict, x_cg: float, sec: dict) -> dict:
    """``sec``: one section for both wings, or {"front": …, "rear": …} — each wing at its own built-up cl_max."""
    secs = sec if "front" in sec else {"front": sec, "rear": sec}
    """The tandem's stall rule (as ``kestrel_tandem.stall``, unequal areas): at trim each pair carries the share that
    balances the moments about the CG (lift at the MAC quarter chords); its lift coefficient is share × CL × S / S_pair;
    the pair with the higher one reaches the built-up cl_max first. The aircraft's CL max follows."""
    wf, wr = L["wings"]["front"], L["wings"]["rear"]
    xf, xr = wf["x_ac_mac"], wr["x_ac_mac"]
    share_f = float(np.clip((xr - x_cg) / (xr - xf), 0.05, 0.95))
    S = L["S_ref"]
    load_f = share_f * S / wf["S_pair"]                 # cl_front / CL
    load_r = (1 - share_f) * S / wr["S_pair"]
    cl_f, cl_r = BUILT_UP * secs["front"]["cl_max"], BUILT_UP * secs["rear"]["cl_max"]
    first = "front" if cl_f / load_f <= cl_r / load_r else "rear"
    CLmax = min(cl_f / load_f, cl_r / load_r)
    cl_wing = cl_f if first == "front" else cl_r
    return {"front share at trim": share_f, "cl front / CL": load_f, "cl rear / CL": load_r, "stalls first": first, "wing cl_max (built-up)": cl_wing,
            "front cl_max (built-up)": cl_f, "rear cl_max (built-up)": cl_r,
            "CL max (aircraft)": CLmax, "rear cl at the front's stall": CLmax * load_r, "nose-down stall": first == "front",
            "static margin [m]": a["x_np"] - x_cg, "static margin [MAC]": (a["x_np"] - x_cg) / L["c_ref"], "label": "ESTIMATE"}


# ================================================================================================= drag
def _ff_wing(t: float = 0.12, xt: float = 0.3) -> float:
    return 1 + 0.6 / xt * t + 100 * t ** 4


def drag_buildup(L: dict, *, V: float = 20.0, n_running: int = 4) -> dict:
    """Zero-lift drag areas [m²] in cruise (tilt 0): four wing panels, the fuselage, four nacelles, the legs, the fin,
    the root gaps of the rotating joints, the stopped propellers (when ``n_running`` < 4); the 10 % junction allowance."""
    p = L["p"]
    rows = {}
    for which in ("front", "rear"):
        w = L["wings"][which]
        rows[f"wings {which}"] = og_aero.cf_mixed(V * w["mac"] / NU) * _ff_wing(_tc(k4tq.section_of(p, which))) * 2.04 * w["S_pair"]
    Lf, bw, bh = L["fuselage_length"], p["fuselage_width"], p["fuselage_height"]
    d = math.sqrt(bw * bh)
    fr = Lf / d
    S_wet = math.pi * math.sqrt((bw ** 2 + bh ** 2) / 2) * Lf * 0.78
    rows["fuselage"] = og_aero.cf_mixed(V * Lf / NU) * (1 + 60 / fr ** 3 + fr / 400) * S_wet
    sc = p.get("scale", 1.0)
    dn, ln = 0.045 * sc, 0.15 * sc
    rows["nacelles (4)"] = 4 * (og_aero.cf_mixed(V * ln / NU) * 1.25 * math.pi * dn * ln + 0.05 * math.pi * dn ** 2 / 4)
    leg_h = -p["leg_bottom_z"] - bh / 2
    rows["legs (4, faired)"] = 4 * 0.12 * 0.010 * sc * leg_h
    if p["fin"]:
        S_fin = p["fin_area"] or 0.030 * sc ** 2
        rows["fin"] = og_aero.cf_mixed(V * math.sqrt(S_fin) / NU) * 1.15 * 2.04 * S_fin
    rows["root gaps (4)"] = 4 * ROOT_GAP_CDA * sc ** 2
    if n_running < 4:
        rows[f"stopped propellers ({4 - n_running}, folded)"] = (4 - n_running) * FOLDED_BLADE_CDA * sc ** 2
    total = sum(rows.values()) * INTERFERENCE
    return {"parts_m2": rows, "CdA_m2": total, "Cd0": total / L["S_ref"], "S_wet_fuselage": S_wet, "label": "ESTIMATE"}


# ================================================================================================= the vehicle
def vehicle(variant: str = k4tq.MAIN, **over) -> dict:
    """Layout, the lattice, the mass table with the battery placed for the stability's CG target (two passes: the fin
    area from the directional stability changes the mass), the stall rule, the polar."""
    p = k4tq.params(variant, **over)
    if p.get("spar_tube_mm") == "auto":                                # K4TQ-B: the spar from the root loads (Ø16×14 first, then what they choose)
        tube = (16, 14)
        for _ in range(4):
            v = vehicle(variant, **{**over, "spar_tube_mm": tube})
            od, idd = (int(x) for x in root_loads(v)["spar chosen"].lstrip("Ø").split("×"))
            if (od, idd) == tuple(tube):
                return v
            tube = (od, idd)
        return v
    L = k4tq.layout(p)
    a = tandem(L)
    sec = sections(p)
    if p["fin"] and not p["fin_area"]:
        fin0 = directional(L, None, a, fin_area=0.0)
        p = dict(p, fin_area=fin0["fin area needed [m²]"])
        L = k4tq.layout(p)
    target = cg_target(L, a)
    if "cargo_station" not in over:
        p = pack(p, target)
        L = k4tq.layout(p)
    items = k4tq.mass_table(L, x_cg_target=target)
    ci = k4tq.cg_inertia(items)
    if p["fin"] and not over.get("fin_area"):                          # second pass: the fin for the real CG
        need = directional(L, {"mass": ci}, a, fin_area=0.0)["fin area needed [m²]"]
        if abs(need - p["fin_area"]) > 0.05 * p["fin_area"]:
            p = dict(p, fin_area=need)
            L = k4tq.layout(p)
            items = k4tq.mass_table(L, x_cg_target=target)
            ci = k4tq.cg_inertia(items)
    st = stall(L, a, ci["x_cg"], sec)
    d = drag_buildup(L)
    W = ci["mass_kg"] * G
    return {"variant": variant, "p": p, "L": L, "aero": a, "section": sec, "items": items, "mass": ci, "W": W, "stall": st, "drag": d,
            "x_cg_target": target, "battery": k4tq.battery(p), "drive": drive(p)}


def pack(p: dict, target: float, gap: float = 0.005) -> dict:
    """Stage 2 packing along the fuselage floor. The pack is split into two parallel halves (also a redundancy: one
    failed half leaves half the energy): half A at the aft end of the nose bay, just ahead of the front wing station's
    frame; the avionics in the tail bay behind the rear station (they balance the heavier nose); half B just behind the front station, then the cargo cartridge, then the
    parachute (Ø70, its line near the CG) — that group slides as one until the CG is on the target (the CG is linear in
    its station), bounded by the front station's servo and the rear station's frame. Returns the parameters with
    ``battery_x``, ``cargo_station``, ``parachute_station`` and ``avionics_station`` set."""
    Lf = p["fuselage_length"]
    sc = p.get("scale", 1.0)
    cx = p["cargo_box"][0]
    hx = k4tq.battery_half_box(p["battery"], k4tq.across_of(p))[0]
    L = k4tq.layout(p)
    x_f, x_r = L["wings"]["front"]["x_pivot"], L["wings"]["rear"]["x_pivot"]
    xa = x_f - 0.032 * sc - gap - hx / 2
    lo = x_f + 0.050 * sc + hx / 2                                  # half B behind the front servo
    span = hx / 2 + gap + cx + 0.040 * sc + 0.035 * sc              # half B's centre to the parachute's aft face
    hi = x_r - 0.035 * sc - span

    def stations(sb):
        xc = sb + hx / 2 + gap + cx / 2
        x_tail = x_r + (0.050 + 0.055) * sc                                       # the tail bay behind the rear station
        if p.get("autonomy"):                                               # K4TQ-A2: the companion first (the wider section), the avionics behind it
            return dict(p, battery_x=(xa / Lf, sb / Lf), cargo_station=xc / Lf, parachute_station=(xc + cx / 2 + 0.040 * sc) / Lf,
                        companion_station=x_tail / Lf, avionics_station=(x_tail + 0.097 * sc) / Lf)
        return dict(p, battery_x=(xa / Lf, sb / Lf), cargo_station=xc / Lf, parachute_station=(xc + cx / 2 + 0.040 * sc) / Lf, avionics_station=x_tail / Lf)
    sb = lo
    for _ in range(3):
        q = stations(sb)
        it = k4tq.mass_table(k4tq.layout(q))
        M = sum(i["mass_kg"] for i in it)
        Mx = sum(i["mass_kg"] * i["x"] for i in it)
        mg = sum(i["mass_kg"] for i in it if i["group"] in ("cargo", "payload", "safety") or i["item"].startswith("battery B"))
        sb = float(np.clip(sb + (target * M - Mx) / mg, lo, hi))
    q = stations(sb)
    q["packing_clipped"] = not (lo + 1e-6 < sb < hi - 1e-6)
    return q


def k_total(veh: dict) -> float:
    return veh["aero"]["k"] / OSWALD_VISCOUS + 0.008                       # + the sections' profile drag rise with cl


PROFILE_ROUGHNESS = 1.10           # a built-up wing's profile drag over the clean section's (NeuralFoil, n_crit 6)


def profile_model(veh: dict):
    """CD(V, CL, build-up) with each wing's profile drag from its NeuralFoil polar (``k4tq_sections``, cached) at its own
    lift coefficient (the tandem's split) and Reynolds number, × ``PROFILE_ROUGHNESS``; the rest of the build-up (fuselage,
    nacelles, legs, fin, root gaps) and the lattice's induced drag as before. None when a polar is missing (the
    build-up's flat-plate wings then)."""
    try:
        import k4tq_sections as ksec
        cache = ksec.load_cache()
        L, st = veh["L"], veh["stall"]
        loads = {"front": st["cl front / CL"], "rear": st["cl rear / CL"]}
        names = {w: k4tq.section_of(veh["p"], w) for w in loads}
        for w in loads:                                            # one probe: the cache answers (or NeuralFoil, mocks off)
            ksec.polar(names[w], 3e5, cache=cache)
    except Exception:
        return None
    k_ind = veh["aero"]["k"] / OSWALD_VISCOUS

    def CD(V, CL, d):
        cd = 0.0
        for w in ("front", "rear"):
            wg = L["wings"][w]
            pol = ksec.polar(names[w], V * wg["mac"] / NU, cache=cache)
            cl = loads[w] * CL
            s = pol.attached()
            cdw = float(np.interp(cl, pol.cl[s], pol.cd[s])) if cl <= pol.cl[s][-1] else float(pol.cd[s][-1]) * 2.0
            cd += PROFILE_ROUGHNESS * cdw * wg["S_pair"] / L["S_ref"]
        wings = sum(v for k, v in d["parts_m2"].items() if k.startswith("wings")) * INTERFERENCE
        return (d["CdA_m2"] - wings) / L["S_ref"] + cd + k_ind * CL ** 2
    return CD


def stall_speed(veh: dict, n: float = 1.0) -> float:
    return math.sqrt(2 * n * veh["W"] / (RHO * veh["L"]["S_ref"] * veh["stall"]["CL max (aircraft)"]))


# ================================================================================================= cruise
def cruise(veh: dict, *, speeds=np.arange(10.0, 33.0, 1.0), n_running: int = 4) -> pd.DataFrame:
    """Level flight at each speed: CL, L/D, drag, thrust per running motor, electrical power (the drive map at the
    axial speed), endurance and still-air range on the usable pack. ``n_running`` = 2: the rear pair stopped, its
    blades folded (their drag added)."""
    L, W, m = veh["L"], veh["W"], veh["drive"]
    d = drag_buildup(L, n_running=n_running)
    S, kk = L["S_ref"], k_total(veh)
    prof = profile_model(veh)
    Ewh = veh["battery"]["usable_wh"]
    CLmax = veh["stall"]["CL max (aircraft)"]
    aw = k4tq.avionics_power(veh["p"])
    rows = []
    for V in speeds:
        q = 0.5 * RHO * V * V
        CL = W / (q * S)
        if CL > CLmax / CRUISE_MIN_OVER_STALL ** 2:
            continue
        if prof:
            CD = prof(V, CL, d)
        else:
            CD = d["Cd0"] + kk * CL ** 2
        D = q * S * CD
        T = D / n_running
        dp = drive_point(m, V, T)
        if not dp["feasible"]:
            rows.append({"V [m/s]": V, "CL": CL, "L/D": CL / CD, "drag [N]": D, "thrust each [N]": T, "feasible": False})
            continue
        P = n_running * dp["electrical_W"] + aw
        rows.append({"V [m/s]": V, "CL": CL, "L/D": CL / CD, "drag [N]": D, "thrust each [N]": T, "rpm": dp["rpm"], "throttle": dp["throttle"],
                     "propulsive eff.": D * V / (P - aw), "electrical [W]": P, "endurance [min]": Ewh / P * 60, "range [km]": Ewh / P * V * 3.6,
                     "feasible": True})
    cols = ["V [m/s]", "CL", "L/D", "drag [N]", "thrust each [N]", "rpm", "throttle", "propulsive eff.", "electrical [W]", "endurance [min]", "range [km]", "feasible"]
    df = pd.DataFrame(rows, columns=cols).set_index("V [m/s]")
    df["feasible"] = df["feasible"].fillna(False).astype(bool)
    df.attrs["label"] = label("CALC")
    df.attrs["n_running"] = n_running
    return df


def best_speeds(df: pd.DataFrame) -> dict:
    ok = df[df["feasible"]]
    if not len(ok):
        return {"best endurance V [m/s]": math.nan, "endurance [min]": 0.0, "best range V [m/s]": math.nan, "range [km]": 0.0, "max level V [m/s]": math.nan,
                "L/D max": math.nan}
    return {"best endurance V [m/s]": float(ok["endurance [min]"].idxmax()), "endurance [min]": float(ok["endurance [min]"].max()),
            "best range V [m/s]": float(ok["range [km]"].idxmax()), "range [km]": float(ok["range [km]"].max()),
            "max level V [m/s]": float(ok.index.max()), "L/D max": float(ok["L/D"].max())}


# ================================================================================================= hover
def hover(veh: dict) -> dict:
    """The hover: thrust split front/rear for the CG (hover thrust stations), ideal and BEMT power, figure of merit,
    T/W at the loaded and sagged pack, the most loaded motor's fraction of its static maximum, hover minutes."""
    L, W, m = veh["L"], veh["W"], veh["drive"]
    xf, xr = L["thrust_x"]["front"], L["thrust_x"]["rear"]
    x_cg = veh["mass"]["x_cg"]
    T_total = W * DOWNLOAD
    share_f = (xr - x_cg) / (xr - xf)
    Tf, Tr = share_f * T_total / 2, (1 - share_f) * T_total / 2
    A = L["prop"]["A"]
    pf, pr = drive_point(m, 0.0, Tf), drive_point(m, 0.0, Tr)
    aw = k4tq.avionics_power(veh["p"])
    P = 2 * (pf["electrical_W"] + pr["electrical_W"]) + aw
    P_ideal = 2 * (og_rotors.ideal_hover_power(Tf, A) + og_rotors.ideal_hover_power(Tr, A))
    P_shaft = 2 * (pf.get("shaft_W", math.nan) + pr.get("shaft_W", math.nan))
    T_max = max_thrust(m, 0.0)
    sag = (3.3 / V_BATT_LOADED) ** 2
    return {"weight [N]": W, "front share of the thrust": share_f, "thrust front each [N]": Tf, "thrust rear each [N]": Tr,
            "disc loading [N/m²]": T_total / (4 * A), "induced velocity [m/s]": og_rotors.hover_induced_velocity(T_total / 4, A),
            "ideal power [W]": P_ideal, "shaft power [W]": P_shaft, "figure of merit": P_ideal / P_shaft, "electrical [W]": P,
            "rpm front": pf.get("rpm"), "current per front motor [A]": pf.get("current_A"), "pack current [A]": (P - aw) / m.battery_v, "avionics and payload electronics [W]": aw,
            "T/W (loaded pack)": 4 * T_max / W, "T/W (sagged pack, 3.3 V/cell)": 4 * T_max * sag / W,
            "most loaded motor / its maximum": max(Tf, Tr) / T_max, "hover minutes on the usable pack": veh["battery"]["usable_wh"] / P * 60,
            "motor out": "four rotors: a motor out in the hover leaves no yaw/roll balance — the parachute is the mitigation (SORA)",
            "label": label("CALC")}


# ================================================================================================= propeller selection
def prop_sweep(veh: dict, props=("13x8", "14x8.5", "15x8", "15x10", "16x10", "17x10"), V_cruise: float = 23.0, V_dash: float = 28.0) -> pd.DataFrame:
    """Each propeller on the motor: hover power at this vehicle's hover thrust, cruise power at V_cruise (4 and 2
    motors), thrust left at V_dash against the drag there, static T/W; the hover-against-cruise compromise."""
    L, W, p = veh["L"], veh["W"], veh["p"]
    d4, d2 = drag_buildup(L, n_running=4), drag_buildup(L, n_running=2)
    kk = k_total(veh)
    rows = []
    for key in props:
        m = drive(p, prop_key=key)
        th = W * DOWNLOAD / 4
        h = drive_point(m, 0.0, th)
        hv = hover(dict(veh, drive=m))
        row = {"propeller": key, "hover W": hv["electrical [W]"], "most loaded motor in the hover": hv["most loaded motor / its maximum"],
               "static T/W": 4 * max_thrust(m, 0.0) / W}
        for nr, dd in ((4, d4), (2, d2)):
            q = 0.5 * RHO * V_cruise ** 2
            CL = W / (q * L["S_ref"])
            D = q * L["S_ref"] * (dd["Cd0"] + kk * CL ** 2)
            c = drive_point(m, V_cruise, D / nr)
            row[f"cruise W at {V_cruise:g} m/s ({nr} motors)"] = nr * c["electrical_W"] if c["feasible"] else math.inf
        q = 0.5 * RHO * V_dash ** 2
        Dd = q * L["S_ref"] * (d4["Cd0"] + kk * (W / (q * L["S_ref"])) ** 2)
        row[f"thrust at {V_dash:g} m/s / drag"] = 4 * max_thrust(m, V_dash) / Dd
        disc = 2 * k4tq.CATALOGUE["propeller"][key]["d_in"] * 0.0254 / 2
        row["fits (hover disc separation ≥ 50 mm)"] = L["stagger"] - disc >= 0.05
        rows.append(row)
    df = pd.DataFrame(rows).set_index("propeller")
    df.attrs["label"] = label("CALC")
    return df


# ================================================================================================= transition corridor
def _wing_polar(veh: dict, which: str) -> og_aero.Polar:
    w = veh["L"]["wings"][which]
    sec = veh["section"][which]
    AR = w["AR_pair"]
    a = 2 * math.pi * 0.9 * AR / (AR + 2)
    return og_aero.Polar(cl_alpha=a, cd0=sec["cd0"] + 0.004, k=1 / (math.pi * AR * 0.85), cl_max=BUILT_UP * sec["cl_max"], cd90=og_aero.viterna_cd90(AR),
                         alpha0=math.radians(sec["alpha0_deg"]), source=f"{sec['name']} built-up, finite AR {AR:.1f}")


def blown_fraction(L: dict, which: str) -> float:
    """The share of a panel's span inside its propeller's (contracted, 0.9 R) slipstream."""
    w, p, R = L["wings"][which], L["p"], 0.9 * L["prop"]["R"]
    lo, hi = max(p["motor_y"] - R, w["y0"]), min(p["motor_y"] + R, w["y1"])
    return max(hi - lo, 0.0) / w["semi_span"]


def _forces(veh: dict, V: float, gamma: float, tilts: dict, T_pair: dict) -> dict:
    """Body-frame (X forward, Z up, fuselage level) forces and the pitching moment about the CG of the four wings and
    the four rotors at each pair's tilt and thrust; the slipstream on each pair's blown span from Glauert's inflow
    (``og.rotors.induced_velocity``), developed to 1.5 vᵢ at the wing."""
    L = veh["L"]
    x_cg, z_cg = veh["mass"]["x_cg"], veh["mass"]["z_cg"]
    A = L["prop"]["A"]
    w_inf = -V * np.array([math.cos(gamma), math.sin(gamma)])        # the air past the aircraft (relative)
    out = {"X": 0.0, "Z": 0.0, "M": 0.0, "parts": {}}
    q_inf = 0.5 * RHO * V * V
    for which in ("front", "rear"):
        w = L["wings"][which]
        tilt = tilts[which]
        t = np.array([math.cos(tilt), math.sin(tilt)])              # the chord / thrust direction (forward and up)
        pol = _wing_polar(veh, which)
        T_rot = T_pair[which] / 2
        vi = og_rotors.induced_velocity(T_rot, V, tilt - gamma, A) if T_rot > 0 else 0.0
        ws = 1.5 * vi
        fb = blown_fraction(L, which)
        Fx = Fz = 0.0
        stalled_span = 0.0
        alphas = {}
        # the rear pair flies in the front pair's downwash: ε = (1 − the lattice's rear slope ratio) × the front's cl / its slope
        eps = (1.0 - veh["aero"]["slope_rear"]) * out.get("cl_front", 0.0) / pol.cl_alpha if which == "rear" else 0.0
        for part, frac, wloc in (("blown", fb, w_inf - ws * t), ("free", 1 - fb, w_inf)):
            if frac <= 0:
                continue
            spd = float(np.hypot(*wloc))
            if spd < 1e-6:
                continue
            a_dir = -wloc / spd                                         # where the air comes from (forward-ish)
            if eps:
                ce, se = math.cos(eps), math.sin(eps)                   # the downwash turns the incoming flow downward (from above)
                a_dir = np.array([ce * a_dir[0] - se * a_dir[1], se * a_dir[0] + ce * a_dir[1]])
            alpha = math.atan2(a_dir[0] * t[1] - a_dir[1] * t[0], a_dir[0] * t[0] + a_dir[1] * t[1])
            cl, cd, cm = pol.coefficients(alpha)
            if which == "front" and part == "free":
                out["cl_front"] = float(cl)
            q = 0.5 * RHO * spd * spd
            S = frac * w["S_pair"]
            lift_dir = np.array([-a_dir[1], a_dir[0]])
            F = q * S * (float(cl) * lift_dir - float(cd) * a_dir)
            Fx += F[0]
            Fz += F[1]
            alphas[part] = math.degrees(alpha)
            if abs(alpha) > pol.alpha_stall:
                stalled_span += frac
            out["M"] += q * S * w["mac"] * (float(cm) + veh["section"][which]["cm0"] * (1 - min(abs(alpha) / pol.alpha_stall, 1.0)))
        # the wing force acts near the pivot (the AC is 0.05 c ahead of it along the chord: second order)
        x_w, z_w = w["x_pivot"], w["z"]
        out["M"] += -(x_w - x_cg) * Fz - (z_w - z_cg) * Fx
        # the rotors: thrust along t at the hubs (the hub sits ahead of the pivot along the chord by the arm)
        arm = L["poses"][(which, 1)]["cruise"]["arm"]
        x_h, z_h = w["x_pivot"] - arm * t[0], w["z"] + arm * t[1]
        FT = T_pair[which] * t
        out["M"] += -(x_h - x_cg) * FT[1] - (z_h - z_cg) * FT[0]
        out["X"] += Fx + FT[0]
        out["Z"] += Fz + FT[1]
        out["parts"][which] = {"alpha blown [deg]": alphas.get("blown"), "alpha free [deg]": alphas.get("free"), "stalled span": stalled_span,
                               "slipstream [m/s]": ws, "wing lift [N]": Fz, "wing drag [N]": -Fx}
    CdA_fus = veh["drag"]["parts_m2"]["fuselage"] * INTERFERENCE + veh["drag"]["parts_m2"]["legs (4, faired)"]
    out["X"] += -q_inf * CdA_fus * math.cos(gamma)
    out["Z"] += -q_inf * CdA_fus * math.sin(gamma)
    return out


def _tilts(tilt: float, delta: float) -> dict:
    return {"front": tilt, "rear": float(np.clip(tilt - delta, -0.35, math.pi / 2 + 0.2))}


def _solve_thrust(veh, V, gamma, tilts, share_f, W):
    """The total thrust that balances Z at these tilts (bisection), the split ``share_f`` front."""
    lo, hi = 0.0, 3.0 * W
    for _ in range(28):
        mid = 0.5 * (lo + hi)
        f = _forces(veh, V, gamma, tilts, {"front": share_f * mid, "rear": (1 - share_f) * mid})
        if f["Z"] - W > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _balance(veh, V, g, share_f, delta, W):
    """Every tilt that balances X and Z at this split and differential tilt: a 3° scan of the X residual from 0 to 90°,
    each sign change refined; returns [(tilt, T)]."""
    def x_res(tl):
        tt = _tilts(tl, delta)
        T = _solve_thrust(veh, V, g, tt, share_f, W)
        return _forces(veh, V, g, tt, {"front": share_f * T, "rear": (1 - share_f) * T})["X"], T
    grid = np.radians(np.arange(0.0, 90.01, 5.0))
    vals = [x_res(float(x)) for x in grid]
    roots = []
    for (a, (xa, _)), (b, (xb, _)) in zip(zip(grid, vals), zip(grid[1:], vals[1:])):
        if xa > 0 >= xb:
            lo, hi = float(a), float(b)
            for _ in range(12):
                mid = 0.5 * (lo + hi)
                if x_res(mid)[0] > 0:
                    lo = mid
                else:
                    hi = mid
            tl = 0.5 * (lo + hi)
            roots.append((tl, x_res(tl)[1]))
    return roots


def _power(veh, V, g, tilts, share_f, T):
    m = veh["drive"]
    pts = []
    for which, s in (("front", share_f), ("rear", 1 - share_f)):
        Va = max(V * math.cos(tilts[which] - g), 0.0)
        pts.append(drive_point(m, Va, s * T / 2))
    P = 2 * (pts[0]["electrical_W"] + pts[1]["electrical_W"]) + k4tq.avionics_power(veh["p"])
    return P, pts


def _power_smooth(veh, V, g, tilts, share_f, T) -> float:
    """Electrical power with a steep linear penalty beyond a motor's current-limited maximum (a smooth objective for
    the optimiser; the reported power is ``_power``'s)."""
    m = veh["drive"]
    P = k4tq.avionics_power(veh["p"])
    for which, s in (("front", share_f), ("rear", 1 - share_f)):
        Va = max(V * math.cos(tilts[which] - g), 0.0)
        t = max(s * T / 2, 0.0)
        tmax = max_thrust(m, Va)
        tc = min(t, tmax)
        P += 2 * (m.electrical_for_thrust(Va, tc) if tc > 1e-6 else 0.0)
        if t > tmax:
            P += 2 * 400.0 * (t - tmax)
    return P


def _hover_point(veh, g, W) -> dict:
    tt = _tilts(math.pi / 2, 0.0)
    T = _solve_thrust(veh, 0.0, g, tt, 0.6, W)
    m0 = _forces(veh, 0.0, g, tt, {"front": 0.0, "rear": T})["M"]
    m1 = _forces(veh, 0.0, g, tt, {"front": T, "rear": 0.0})["M"]
    s = float(np.clip(-m0 / (m1 - m0), 0.0, 1.0))
    return {"x": np.array([math.pi / 2, 0.0, _solve_thrust(veh, 0.0, g, tt, s, W), s])}


def corridor_point(veh: dict, V: float, gamma_deg: float = 0.0, *, x0=None) -> dict:
    """One speed of the corridor (fuselage level, steady): the least-power state that balances X, Z and the pitching
    moment — unknowns the tilt, the rear pair's differential tilt δ (rear = tilt − δ: the all-moving wings are the
    elevator), the total thrust and the front share of it (SLSQP from ``x0``, the previous speed's state, and from the
    balances of a tilt scan when that fails). Power, stall state and the motors' margins at the solution."""
    from scipy.optimize import minimize
    W = veh["W"]
    g = math.radians(gamma_deg)
    if V < 0.05 and x0 is None:
        x0 = _hover_point(veh, g, W)["x"]                                 # the cambered wings in their own slipstream pull sideways-in-chord: trimmed by the tilt
    if True:
        cref = veh["L"]["c_ref"]

        def f(x):
            return _forces(veh, V, g, _tilts(x[0], x[1]), {"front": x[3] * x[2], "rear": (1 - x[3]) * x[2]})
        cons = [{"type": "eq", "fun": lambda x: f(x)["X"] / W}, {"type": "eq", "fun": lambda x: (f(x)["Z"] - W) / W},
                {"type": "eq", "fun": lambda x: f(x)["M"] / (W * cref)}]
        obj = lambda x: _power_smooth(veh, V, g, _tilts(x[0], x[1]), x[3], x[2]) / 1000.0          # noqa: E731
        bounds = [(-0.05, math.pi / 2 + 0.2), (math.radians(-12), math.radians(12)), (0.0, 3.0 * W), (0.0, 1.0)]
        starts = [np.asarray(x0, float)] if x0 is not None else []
        best = None

        def run(xs):
            r = minimize(obj, xs, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 200, "ftol": 1e-7})
            ff = f(r.x)
            res = max(abs(ff["X"]) / W, abs(ff["Z"] - W) / W, abs(ff["M"]) / (W * cref))
            return r.x, res, obj(r.x)
        for xs in starts:
            xr, res, P = run(xs)
            if res < 2e-3:
                best = (P, xr, res)
        if (best is None and V >= 0.05) or V >= 0.8 * stall_speed(veh):                 # fresh starts from the scan's balances (and near the stall always: the wing-borne branch)
            for tl, T in _balance(veh, V, g, 0.6, 0.0, W) or [(math.radians(45.0), W)]:
                for s0 in (0.7,):
                    xr, res, P = run(np.array([tl, 0.0, T, s0]))
                    if res < 2e-3 and (best is None or P < best[0]):
                        best = (P, xr, res)
        ok = best is not None
        x = best[1] if ok else (np.asarray(x0, float) if x0 is not None else np.array([math.radians(45), 0.0, W, 0.6]))
    tilt, delta, T, share_f = (float(v) for v in x)
    tt = _tilts(tilt, delta)
    fo = _forces(veh, V, g, tt, {"front": share_f * T, "rear": (1 - share_f) * T})
    P, pts = _power(veh, V, g, tt, share_f, T)
    return {"V [m/s]": V, "gamma [deg]": gamma_deg, "tilt [deg]": math.degrees(tilt), "rear differential [deg]": math.degrees(delta),
            "thrust total [N]": T, "front share": share_f, "alpha front blown": fo["parts"]["front"]["alpha blown [deg]"],
            "alpha front free": fo["parts"]["front"]["alpha free [deg]"], "alpha rear free": fo["parts"]["rear"]["alpha free [deg]"],
            "stalled span front": fo["parts"]["front"]["stalled span"], "stalled span rear": fo["parts"]["rear"]["stalled span"],
            "wing lift [N]": fo["parts"]["front"]["wing lift [N]"] + fo["parts"]["rear"]["wing lift [N]"], "trim residual [N·m]": fo["M"],
            "X residual [N]": fo["X"], "electrical [W]": P, "front motor / its max": share_f * T / 2 / pts[0]["T_max_N"],
            "rear motor / its max": (1 - share_f) * T / 2 / pts[1]["T_max_N"],
            "feasible": bool(ok and pts[0]["feasible"] and pts[1]["feasible"]), "note": "" if ok else "no balanced state found", "_x": [tilt, delta, T, share_f]}


def corridor(veh: dict, *, speeds=np.arange(0.0, 24.0, 1.0), gamma_deg: float = 0.0) -> pd.DataFrame:
    """The corridor by continuation: each speed starts from the previous one's state."""
    rows, x0 = [], None
    for V in speeds:
        r = corridor_point(veh, float(V), gamma_deg, x0=x0)
        x0 = r.pop("_x")
        rows.append(r)
    df = pd.DataFrame(rows).set_index("V [m/s]")
    df.attrs["label"] = label("CALC")
    return df


def corridor_summary(df: pd.DataFrame, veh: dict) -> dict:
    ok = df[df["feasible"]]
    wing_borne = ok[ok["tilt [deg]"] < 8.0]
    stalled = ok[(ok["stalled span front"] > 0.01) | (ok["stalled span rear"] > 0.01)]
    return {"continuous from hover to cruise": bool(df["feasible"].all()), "peak power [W]": float(ok["electrical [W]"].max()),
            "speed at the peak [m/s]": float(ok["electrical [W]"].idxmax()), "least power [W]": float(ok["electrical [W]"].min()),
            "most loaded motor (max over V)": float(ok[["front motor / its max", "rear motor / its max"]].max().max()),
            "wing-borne (tilt < 8°) from [m/s]": float(wing_borne.index.min()) if len(wing_borne) else math.nan,
            "free span stalled between [m/s]": (float(stalled.index.min()), float(stalled.index.max())) if len(stalled) else None,
            "front share range": (float(ok["front share"].min()), float(ok["front share"].max())),
            "rear differential tilt range [deg]": (float(ok["rear differential [deg]"].min()), float(ok["rear differential [deg]"].max())),
            "stall speed (wings alone) [m/s]": stall_speed(veh)}


# ================================================================================================= structure: wing root
def root_loads(veh: dict) -> dict:
    """The wing root's loads (front panel, the more loaded): bending in the manoeuvre/gust and in the hover with one
    motor at its maximum, the shear, the actuator torque budget, the bearing and fold-latch forces; the spar tube
    chosen from ``k4tq.TUBES_MM`` by strength (ultimate), deflection and the first bending mode."""
    L, W, p = veh["L"], veh["W"], veh["p"]
    w = L["wings"]["front"]
    b = w["semi_span"]
    share_f = veh["stall"]["front share at trim"]
    cat = k4tq.CATALOGUE
    sc = p.get("scale", 1.0)
    root = cat[p.get("root_set", "root")]
    m_tip = cat["motor"][p["motor"]]["mass_kg"] + L["prop"]["mass_kg"] + cat[p.get("esc", "esc")]["mass_kg"] + k4tq.NACELLE_KG * sc ** 2
    m_panel = k4tq.WING_AREAL_KG_M2 * w["S_panel"]
    r_mot = p["motor_y"] - w["y_hinge"]
    T_max = max_thrust(veh["drive"], 0.0)
    # bending at the fold hinge (the outboard of it is the cantilever)
    lift = N_LIMIT * W * share_f / 2
    M_cruise = lift * (w["y0"] - w["y_hinge"] + 4 * b / (3 * math.pi)) - N_LIMIT * G * (m_panel * (r_mot * 0.55) + m_tip * r_mot)
    M_hover = T_max * r_mot - G * (m_panel * r_mot * 0.55 + m_tip * r_mot)          # one motor at its maximum (in the chord plane)
    M_limit = max(M_cruise, M_hover)
    case = "cruise manoeuvre/gust" if M_cruise >= M_hover else "hover, motor at maximum"
    M_ult = ULTIMATE * M_limit
    shear_ult = ULTIMATE * max(lift - N_LIMIT * G * (m_panel + m_tip), T_max)
    # actuator torque about the tilt axis (limit values)
    sec = veh["section"]["front"]
    c = w["mac"]
    q_max = 0.5 * RHO * 30.0 ** 2
    aero_cruise = q_max * w["S_panel"] * c * (abs(sec["cm0"]) + 0.6 * abs(w["pivot_chord"] - 0.25))
    q_tr = 0.5 * RHO * 14.0 ** 2
    aero_transition = q_tr * w["S_panel"] * c * 1.2 * 0.15                         # CN 1.2 with the centre of pressure at 0.40 c
    mass_offset = N_LIMIT * G * (m_panel + m_tip) * 0.010 * sc                        # the panel + nacelle CG within 10 mm of the axis
    thrust_offset = T_max * 0.003 * sc                                                 # thrust line within 3 mm of the axis
    I_p = L["prop"]["mass_kg"] * L["prop"]["D"] ** 2 / 12 + 0.6e-4                  # propeller + motor bell
    omega = veh["drive"].at(0.0, 1.0)["rpm"] * 2 * math.pi / 60
    gyro = I_p * omega * 1.0                                                       # 1 rad/s aircraft yaw rate in cruise (H ⟂ tilt axis)
    I_theta = m_tip * (w["pivot_chord"] * w["c_at"](p["motor_y"]) + p["prop_ahead_of_le"]) ** 2 + m_panel * c ** 2 / 12
    accel = I_theta * (math.pi / 2) / (0.5 * TILT_TIME_S) ** 2 * 2
    handling = 20.0 * 0.06                                                        # a person holds a tip: 20 N at 60 mm from the axis
    budget = {"aerodynamic, cruise at 30 m/s": aero_cruise, "aerodynamic, transition (separated)": aero_transition, "mass offset at n_limit": mass_offset,
              "thrust-line offset": thrust_offset, "gyroscopic (1 rad/s yaw)": gyro, "tilt acceleration": accel}
    T_flight = max(aero_cruise, aero_transition) + mass_offset + thrust_offset + gyro + accel
    T_design = max(ULTIMATE * T_flight, handling)
    servo = root["tilt servo"]
    servo_Nm, ratio = servo["torque_Nm"], p["z_gear"] / p["z_pinion"]
    bearing = next(v for k, v in root.items() if k.startswith("bearings"))
    C0 = bearing["C0_N"]
    pin_r = next(v for k, v in root.items() if k.startswith("fold knuckles"))["pin_d_m"] / 2
    # the stub's two 6804 bearings (``bearing_spacing`` apart) take the root moment as a couple
    s_b = p["bearing_spacing"]
    F_bearing = M_ult / s_b + shear_ult / 2
    # fold hinge (pin along the chord, outboard of the skin): the lift moment (about the chord) is held by the locking
    # pin against the hinge pin (``lock_couple`` apart); the thrust's moment (about the thickness axis) by the outer
    # lugs (``knuckle_spread`` apart along the chord)
    M_thrust = ULTIMATE * T_max * r_mot
    F_latch = M_ult / p["lock_couple"]
    F_lug = M_thrust / p["knuckle_spread"]
    # spar tube
    E, Gm, sa, rho_c = (k4tq.CARBON_TUBE[k] for k in ("E", "G", "sigma_allow", "rho"))
    tubes = []
    for od_mm, id_mm in k4tq.TUBES_MM:
        od, idd = od_mm / 1e3, id_mm / 1e3
        I = math.pi / 64 * (od ** 4 - idd ** 4)
        J = 2 * I
        A = math.pi / 4 * (od ** 2 - idd ** 2)
        sigma = M_ult * od / 2 / I
        defl = (lift * b ** 3 / (8 * E * I)) * 0.85
        k_b = 3 * E * I / r_mot ** 3
        f1 = math.sqrt(k_b / (m_tip + 0.24 * (m_panel + A * rho_c * b))) / (2 * math.pi)
        GJ = Gm * J
        tubes.append({"tube": f"Ø{od_mm}×{id_mm}", "reserve (ultimate)": sa / sigma, "tip deflection at n_limit [% semi-span]": 100 * defl / b,
                      "first bending with the motor [Hz]": f1, "GJ [N·m²]": GJ, "mass per wing [g]": A * rho_c * (b + 0.03) * 1e3})
    tdf = pd.DataFrame(tubes).set_index("tube")
    ok = tdf[(tdf["reserve (ultimate)"] >= 1.5) & (tdf["tip deflection at n_limit [% semi-span]"] <= 4.0) & (tdf["first bending with the motor [Hz]"] >= SPAR_F1_MIN)]
    chosen = ok.index[0] if len(ok) else tdf.index[-1]
    # tilt (pitch) mode of the wing on its drive and divergence of the all-moving wing
    k_drive = 120.0                                                                # N·m/rad at the spar: servo + 2:1 gears with the backlash taken up (requirement)
    f_pitch = math.sqrt(k_drive / I_theta) / (2 * math.pi)
    e_ac = (w["pivot_chord"] - 0.25) * c
    a_w = 2 * math.pi * 0.9 * w["AR_pair"] / (w["AR_pair"] + 2)
    q_div = k_drive / (e_ac * w["S_panel"] * a_w) if e_ac > 0 else math.inf
    rpm_h = drive_point(veh["drive"], 0.0, W * DOWNLOAD / 4)["rpm"]
    return {"case governing the bending": case, "root moment, cruise limit [N·m]": M_cruise, "root moment, hover motor-max limit [N·m]": M_hover,
            "root moment ultimate [N·m]": M_ult, "root shear ultimate [N]": shear_ult, "actuator torque budget (limit) [N·m]": budget,
            "actuator design torque [N·m]": T_design, "ground handling torque [N·m]": handling,
            "actuator available at the spar [N·m]": servo_Nm * ratio, "actuator reserve": servo_Nm * ratio / T_design, "actuator": servo["source"] + f", {ratio:g}:1",
            "bearing radial (ultimate) [N]": F_bearing, "bearing static rating C0 [N]": C0, "bearing static reserve": C0 / F_bearing,
            "fold lug force from the thrust (ultimate) [N]": F_lug, "locking pin force from the lift (ultimate) [N]": F_latch,
            "locking pin shear stress, double shear [MPa]": F_latch / (2 * math.pi * pin_r ** 2) / 1e6,
            "fold latch": "the locking pin carries the lift moment as a couple with the hinge pin: a structural pin in double shear, not a detent",
            "spar tubes": tdf, "spar chosen": chosen, "drive stiffness required [N·m/rad]": k_drive, "wing pitch mode on the drive [Hz]": f_pitch,
            "hover 1P [Hz]": rpm_h / 60, "divergence speed of the all-moving wing [m/s]": math.sqrt(2 * q_div / RHO) if math.isfinite(q_div) else math.inf,
            "label": "ESTIMATE"}


def materials() -> pd.DataFrame:
    rows = [
        ("spar", "carbon tube, pultruded / roll-wrapped", "E 80 GPa, σ_allow 400 MPa (compression with knock-down)", f"ultimate {ULTIMATE}"),
        ("wing skins", "glass 80–110 g/m² + carbon UD caps on XPS foam", "areal 0.70 kg/m²", f"ultimate {ULTIMATE}"),
        ("root housing", "7075-T6 machined", "σ_y 500 MPa", f"ultimate {ULTIMATE}, fitting factor 1.15"),
        ("fold hinge pin, locking pin", "hardened steel (1.4542 / 17-4PH)", "τ_allow 400 MPa", f"ultimate {ULTIMATE}"),
        ("bearings", "6804-2RS deep groove", "C 4.0 kN, C0 2.47 kN", "static reserve ≥ 2 on ultimate"),
        ("printed brackets, covers", "PA-CF / PETG-CF (Mellonia)", "σ ~ 45–60 MPa XY, ~50 % in Z", f"ultimate {ULTIMATE} × printed {PRINTED_FACTOR}"),
        ("fuselage shell", "glass/carbon on a foam or honeycomb core", "areal 0.60 kg/m²", f"ultimate {ULTIMATE}"),
    ]
    df = pd.DataFrame(rows, columns=["part", "material", "allowable", "factor"]).set_index("part")
    df.attrs["label"] = "ESTIMATE"
    return df


# ================================================================================================= directional stability
def directional(L: dict, veh: dict | None, a: dict, *, fin_area: float | None = None, V: float = 20.0, target: float = 0.05) -> dict:
    """Cnβ [1/rad] on S_ref·b: the fuselage (Raymer), the four propellers' normal force ahead of the CG (De Young
    slope), the wings (sweep, small), and the fin; the fin area that reaches ``target``."""
    p = L["p"]
    S, b = L["S_ref"], p["span"]
    Lf, bw, bh = L["fuselage_length"], p["fuselage_width"], p["fuselage_height"]
    vol = math.pi / 4 * bw * bh * Lf * 0.65
    cn_fus = -1.3 * vol / (S * b) * (bh / bw)
    x_cg = veh["mass"]["x_cg"] if veh else L["wings"]["front"]["x_pivot"] + 0.45 * L["stagger"]
    # propellers: De Young's normal-force slope with the K4TQ blade at a light cruise loading
    pr = L["prop"]
    R = pr["R"]
    c75 = 0.020 * pr["d_in"] / 10
    sig = 4 / 3 * 2 * c75 / (math.pi * R)
    beta = math.atan(pr["pitch"] / (2 * math.pi * 0.75 * R))
    q = 0.5 * RHO * V * V
    T_each = 1.5
    tc = T_each / (q * pr["A"])
    cna = 4.25 * sig / (1 + 2 * sig) * math.sin(beta + math.radians(8)) * (1 + 0.5 * (math.sqrt(1 + 8 * tc / math.pi) - 1))
    cn_prop = 0.0
    for (which, side), pose in L["poses"].items():
        xh = pose["cruise"]["hub"][0]
        cn_prop += -cna * pr["A"] * (x_cg - xh) / (S * b)
    cn_wing = 0.0                                                   # the leading edges sweep ~1.5° only (the spar is straight): negligible
    l_v = 0.93 * Lf - x_cg
    A_v = 1.6                                                       # dorsal + ventral, end-plated by the fuselage
    a_v = 2 * math.pi * A_v / (2 + math.sqrt(A_v ** 2 + 4))
    per_m2 = a_v * 0.9 * l_v / (S * b)
    base = cn_fus + cn_prop + cn_wing
    need = max((target - base) / per_m2, 0.0)
    S_fin = need if fin_area is None else fin_area
    return {"Cnβ fuselage": cn_fus, "Cnβ propellers": cn_prop, "Cnβ wings": cn_wing, "Cnβ without a fin": base, "fin arm [m]": l_v,
            "fin lift slope [1/rad]": a_v, "fin area needed [m²]": need, "Cnβ with the fin": base + S_fin * per_m2, "fin area used [m²]": S_fin,
            "target Cnβ [1/rad]": target, "verdict": "unstable without a fin: the fin is needed (R9)" if base < 0 else "stable without a fin",
            "fin": f"split dorsal/ventral, {S_fin * 1e4:.0f} cm² total; the ventral half doubles as the tail skid", "label": "ESTIMATE"}


# ================================================================================================= launch and landing
def launch(veh: dict, *, wind: float = 8.0, roof_speed_up: float = 1.25, gust_factor: float = 1.45, tube_id: float | None = None) -> dict:
    """R6: hand release, the transport tube as the pad, the hospital roof: clearances, thrust margins, the overturning
    of the hover stance in the roof wind (vertical wings broadside) against the cruise stance (wings flat, −5°), the
    downwash on the pad."""
    L, W, p = veh["L"], veh["W"], veh["p"]
    h = hover(veh)
    R = L["prop"]["R"]
    hand = {"disc edge to the grip (centre ±50 mm) [m]": p["motor_y"] - R - 0.05,
            "wing root to the grip (hover stance) [m]": L["wings"]["front"]["y0"] - 0.05,
            "T/W at release": h["T/W (sagged pack, 3.3 V/cell)"],
            "verdict": "feasible as a hand RELEASE in the hover stance (grip handle under the fuselage, arm-gated motors, gloves); "
                       "a hand THROW in cruise is not: stall speed ≈ {:.0f} m/s against a ~6–8 m/s throw of a {:.1f} kg body".format(stall_speed(veh), veh["mass"]["mass_kg"])}
    # tube: the opened half-tube as the pad: the vertical wings' trailing edges reach below the fuselage; inside a
    # half-tube of radius r they hit its walls unless the aircraft stands on top of it
    env = folded_best(L)
    r = (tube_id or env["tube inner diameter needed [m]"]) / 2
    # standing in the open half-tube on its feet: the tube's axis is r above the feet; the hover stance's trailing edges
    # (z_te, from y0 out to the tip) must stay above the wall z_w(y) = axis − √(r² − y²) for every y < r
    w0 = L["wings"]["front"]
    te_z = min(k4tq.wing_point(L, wh, 1, L["wings"][wh]["y0"], 1.0, math.pi / 2)[2] for wh in ("front", "rear"))
    z_axis = p["leg_bottom_z"] + r
    ys = np.linspace(w0["y0"], r, 20) if r > w0["y0"] else np.array([])
    clear = bool(np.all(te_z > z_axis - np.sqrt(np.maximum(r ** 2 - ys ** 2, 0.0)))) if len(ys) else True
    tube = {"tube inner diameter [m]": 2 * r, "tube length [m]": env["length [m]"] + 0.05,
            "hover-stance trailing edge below the axis [m]": z_axis - te_z, "stands inside the open half-tube": clear,
            "verdict": "can take off from inside the open half-tube" if clear else
                       "the closed tube (or its lid) is the pad: the aircraft stands on it on its four legs; inside the open half-tube the vertical wings' "
                       "trailing edges cross the walls, so it is lifted out and set on the tube first",
            "ejection with in-flight unfolding": "rejected: spring ejection, in-flight unfold and tilt-up add failure modes without a civilian benefit"}
    # roof: mean wind × speed-up at the roof edge × gust factor
    Vg = wind * roof_speed_up * gust_factor
    q = 0.5 * RHO * Vg ** 2
    S_w = L["S_ref"]
    feet_x = (L["wings"]["front"]["x_pivot"], L["wings"]["rear"]["x_pivot"])
    base_half = 0.5 * (feet_x[1] - feet_x[0])
    h_cp = 0.0 - p["leg_bottom_z"]                                           # the wings' centre of pressure ~ the pivot height above the feet
    D_hover = q * S_w * 1.17                                                 # broadside flat plates
    M_over_h = D_hover * h_cp
    M_rest = W * base_half
    pol = _wing_polar(veh, "front")
    cl, cd, _ = pol.coefficients(math.radians(-5.0))
    Lift = q * S_w * float(cl)
    D_flat = q * (S_w * float(cd) + veh["drag"]["CdA_m2"])
    M_over_f = D_flat * h_cp
    M_rest_f = (W - Lift) * base_half                                        # a down-force (Lift < 0) adds to the weight
    # the largest wind the hover stance holds with a 1.5 margin
    V_hold_hover = math.sqrt(M_rest / 1.5 / (h_cp * S_w * 1.17 * 0.5 * RHO))
    downwash = 2 * h["induced velocity [m/s]"]
    roof = {"design gust at the roof edge [m/s]": Vg, "hover stance: overturning / restoring": M_over_h / M_rest,
            "cruise stance at −5°: overturning / restoring": M_over_f / M_rest_f, "wind the hover stance holds (margin 1.5) [m/s]": V_hold_hover,
            "downwash at the pad [m/s]": downwash,
            "procedure": "on the roof the wings wait flat at −5° (down-force, little drag); motors arm only at tilt ≥ 80°; nose into the wind; the take-off "
                         "starts at the corridor's tilt for the wind speed (a 'flying start'), so the wings never stand broadside to the gust; keep 3 m clear of the discs",
            "verdict": "feasible with the procedure; the hover stance alone tips over in the design gust" if M_over_h / M_rest > 1 / 1.5 else "feasible"}
    return {"hand": hand, "tube": tube, "roof": roof, "label": "ESTIMATE"}


def folded_best(L: dict) -> dict:
    """The smallest tube over fold angles (70–90°) and the tilt (0, 45, 90°) at which the wings fold; the front folded
    wing must clear the rear root (its tip aft of it at a different height or outboard of it)."""
    best = None
    for tilt in (90.0,):                                        # the wings fold only at 90° (the hinge pin vertical)
        for fold in (70.0, 75.0, 80.0, 85.0, 90.0):
            env = k4tq.folded_envelope(L, fold, tilt)
            # clearance: the folded front wing (chord band) against the rear root fitting at y_hinge..y0, z_rear
            if best is None or env["tube inner diameter needed [m]"] < best["tube inner diameter needed [m]"] - 1e-9:
                best = env
    best = dict(best)
    best["note"] = "first estimate (corner points); Stage 2 checks the folded solids against each other and the tube"
    return best


# ================================================================================================= mission
def mission(veh: dict, *, radius_km: float | None = None, wind: float | None = None, cr: pd.DataFrame | None = None, cor: pd.DataFrame | None = None,
            P_transition: float | None = None) -> dict:
    """R3: two legs (headwind out, tailwind back), each: hover 30 s (take-off/landing incl. the climb/descent), forward
    and back transitions (15 s each at the corridor's mean power), climb to the cruise altitude on the wings, the
    cruise at the speed that minimises energy per ground kilometre; reserve 20 %."""
    R_km = MISSION["radius_km"] if radius_km is None else radius_km
    w = MISSION["wind_m_s"] if wind is None else wind
    cr = cr if cr is not None else cruise(veh, n_running=2)
    h = hover(veh)
    ok = cr[cr["feasible"]]
    if not len(ok):
        return {"legs": {}, "energy used [Wh]": math.inf, "energy with reserve [Wh]": math.inf, "usable pack [Wh]": veh["battery"]["usable_wh"],
                "margin [Wh]": -math.inf, "closes": False, "label": label("CALC"), "note": "no feasible cruise speed"}
    if P_transition is None:
        cor = cor if cor is not None else corridor(veh, speeds=np.arange(0.0, 18.0, 2.0))
        P_transition = float(cor[cor["feasible"]]["electrical [W]"].mean())
    P_tr = P_transition
    E = {}
    hover_s = MISSION["hover_total_s"] / 4
    for leg, sgn in (("out (headwind)", -1), ("back (tailwind)", +1)):
        gs = ok.index.values + sgn * w
        e_per_km = np.where(gs > 1.0, ok["electrical [W]"].values / gs / 3.6, np.inf)          # Wh per km
        i = int(np.argmin(e_per_km))
        V = float(ok.index.values[i])
        climb_wh = veh["W"] * MISSION["cruise_altitude_m"] / 0.45 / 3600
        E[leg] = {"cruise V [m/s]": V, "ground speed [m/s]": V + sgn * w, "cruise [Wh]": float(e_per_km[i] * R_km),
                  "hover [Wh]": 2 * hover_s * h["electrical [W]"] / 3600, "transitions [Wh]": 2 * 15.0 * P_tr / 3600, "climb [Wh]": climb_wh,
                  "time [min]": (R_km * 1000 / (V + sgn * w) + 2 * hover_s + 30) / 60}
    used = sum(sum(v for k, v in e.items() if k.endswith("[Wh]")) for e in E.values())
    usable = veh["battery"]["usable_wh"]
    need = used / (1 - MISSION["reserve_fraction"])
    return {"legs": E, "energy used [Wh]": used, "energy with reserve [Wh]": need, "usable pack [Wh]": usable, "margin [Wh]": usable - need,
            "closes": usable >= need, "label": label("CALC")}


def payload_range(variant: str = k4tq.MAIN, *, payloads=(0.0, 0.25, 0.5, 0.75, 1.0, 1.25), packs=("6S2P-P45B", "6S3P-P45B"), mtow_limit: float | None = None) -> pd.DataFrame:
    """The radius each payload and pack reach with the mission's hover, transitions, wind and reserve (bisection on
    the radius), and whether the take-off mass stays inside the structure's design mass (A with 1 kg and 6S2P)."""
    ref = vehicle(variant)
    m_lim = mtow_limit or ref["mass"]["mass_kg"]
    cor = corridor(ref, speeds=np.arange(0.0, 18.0, 2.0))
    P_ref = float(cor[cor["feasible"]]["electrical [W]"].mean())          # the transition power, scaled with W^1.5 (momentum theory)
    rows = []
    for pack in packs:
        for pl in payloads:
            veh = vehicle(variant, payload_kg=pl, battery=pack)
            cr = cruise(veh, n_running=2)
            P_tr = P_ref * (veh["W"] / ref["W"]) ** 1.5
            lo, hi = 0.0, 60.0
            if not mission(veh, radius_km=0.0, cr=cr, P_transition=P_tr)["closes"]:      # (also when no cruise speed is feasible)
                rad = 0.0
            else:
                for _ in range(22):
                    mid = 0.5 * (lo + hi)
                    if mission(veh, radius_km=mid, cr=cr, P_transition=P_tr)["closes"]:
                        lo = mid
                    else:
                        hi = mid
                rad = lo
            rows.append({"pack": pack, "payload [kg]": pl, "take-off mass [kg]": veh["mass"]["mass_kg"], "within the design mass": veh["mass"]["mass_kg"] <= m_lim + 1e-6,
                         "radius [km]": rad, "T/W": hover(veh)["T/W (loaded pack)"]})
    df = pd.DataFrame(rows)
    df.attrs["label"] = label("CALC")
    df.attrs["design mass [kg]"] = m_lim
    return df


# ================================================================================================= the quadcopter references
def quad_reference(name: str, prop_key: str, *, k4: dict, motor_key: str | None = None, arm_len: float | None = None) -> dict:
    """A quadcopter of the same size for the comparison: the same motors, pack, payload, cargo cartridge, temperature
    logging, avionics, parachute and growth margin; no wings, roots, fin or nacelles — four carbon arms and a short pod.
    Forward flight: the discs tilted for the drag, Glauert's induced power, the profile power from the BEMT hover
    grown with μ², the body's parasite drag (CdA from the build-up of the pod, arms, motors and legs). ``motor_key``
    another motor (its mass replaces K4TQ's), ``arm_len`` the centre-to-motor arm [m] (default the compact X)."""
    p = k4["p"]
    cat = k4tq.CATALOGUE
    pr = cat["propeller"][prop_key]
    D = pr["d_in"] * 0.0254
    R = D / 2
    A = math.pi * R ** 2
    arm_len = arm_len or max(0.55, 0.5 * (D + 0.06) * math.sqrt(2) * 1.0)  # the discs 60 mm apart on the X
    keep = [i for i in k4["items"] if i["group"] not in ("roots", "battery", "margin") and not i["item"].startswith(("wing panel", "spar", "fin", "nacelle", "power wiring", "fuselage", "propeller"))]
    m = sum(i["mass_kg"] for i in keep)
    m += 4 * pr["mass_kg"]
    if motor_key and motor_key != p["motor"]:
        m += 4 * (cat["motor"][motor_key]["mass_kg"] - cat["motor"][p["motor"]]["mass_kg"])
    tube_kg_m = 0.073
    m_arms = 4 * (tube_kg_m * arm_len + 0.025)                            # Ø16 tube + clamps
    m_pod = k4tq.FUSELAGE_AREAL_KG_M2 * (math.pi * 0.15 * 0.45 * 0.85)  # a 0.45 m pod around the cartridge
    m_wiring = 4 * k4tq.WIRING_KG_M * (arm_len + 0.1)
    m += m_arms + m_pod + m_wiring
    m += k4tq.GROWTH_MARGIN * m
    m += cat["battery"][p["battery"]]["mass_kg"]
    W = m * G
    drv = drive(p, prop_key=prop_key, motor_key=motor_key)
    hp = drive_point(drv, 0.0, W / 4)
    aw = k4tq.avionics_power(p)
    P_hover = 4 * hp["electrical_W"] + aw
    if not hp["feasible"]:                                                 # this motor cannot lift it on this propeller
        return {"name": name, "propeller": prop_key, "mass [kg]": m, "hover [W]": math.inf, "hover minutes": 0.0, "CdA [m²]": math.nan, "table": None,
                "best range V [m/s]": math.nan, "range [km]": 0.0, "endurance [min]": 0.0, "max speed (T ≤ 80 % max) [m/s]": 0.0, "footprint [m]": math.nan,
                "T/W": 4 * max_thrust(drv, 0.0) / W, "label": label("CALC"), "note": "cannot hover", "motor": motor_key or p["motor"], "arm [m]": arm_len, "drive": drv}
    eta_motor = hp["shaft_W"] / hp["electrical_W"]
    vh = og_rotors.hover_induced_velocity(W / 4, A)
    P0 = hp["shaft_W"] - 1.15 * (W / 4) * vh                              # profile power per rotor in the hover
    omega_R = hp["rpm"] * 2 * math.pi / 60 * R
    CdA = 0.6 * 0.15 * 0.16 + 4 * 1.0 * 0.016 * arm_len * 0.7 + 4 * 0.8 * 0.042 * 0.03 + 4 * 0.012 * 0.10 + 0.004   # pod, arms, motors, legs, GNSS mast
    usable = k4["battery"]["usable_wh"]
    T_max0 = max_thrust(drv, 0.0)
    rows = []
    for V in np.arange(0.0, 31.0, 1.0):
        Dp = 0.5 * RHO * V * V * CdA
        T = math.hypot(W, Dp) / 4
        a_tilt = math.atan2(Dp, W)
        vi = og_rotors.induced_velocity(T, V, math.pi / 2 - a_tilt, A)    # the disc axis ~ vertical: the flow at ~90° − tilt to it
        mu = V * math.cos(a_tilt) / omega_R
        P_shaft = 4 * (1.15 * T * (vi + V * math.sin(a_tilt)) + P0 * (1 + 4.65 * mu ** 2))
        P_el = P_shaft / eta_motor + aw
        rows.append({"V [m/s]": V, "electrical [W]": P_el, "endurance [min]": usable / P_el * 60, "range [km]": usable / P_el * V * 3.6,
                     "disc tilt [deg]": math.degrees(a_tilt), "T each / max": T / T_max0})
    df = pd.DataFrame(rows).set_index("V [m/s]")
    feas = df[df["T each / max"] < 0.8]
    if not len(feas):
        return {"name": name, "propeller": prop_key, "mass [kg]": m, "hover [W]": P_hover, "hover minutes": usable / P_hover * 60, "CdA [m²]": CdA,
                "table": df, "best range V [m/s]": math.nan, "range [km]": 0.0, "endurance [min]": 0.0, "max speed (T ≤ 80 % max) [m/s]": 0.0,
                "footprint [m]": 2 * (arm_len + R), "T/W": 4 * max_thrust(drv, 0.0) / W, "label": label("CALC"), "note": "cannot hover below 80 % of its thrust"}
    best_r = float(feas["range [km]"].idxmax())
    footprint = 2 * (arm_len + R)
    return {"name": name, "propeller": prop_key, "mass [kg]": m, "hover [W]": P_hover, "hover minutes": usable / P_hover * 60, "CdA [m²]": CdA,
            "table": df, "best range V [m/s]": best_r, "range [km]": float(feas["range [km]"].max()), "endurance [min]": float(feas["endurance [min]"].max()),
            "max speed (T ≤ 80 % max) [m/s]": float(feas.index.max()), "footprint [m]": footprint, "T/W": 4 * max_thrust(drv, 0.0) / W, "label": label("CALC"),
            "motor": motor_key or p["motor"], "arm [m]": arm_len, "drive": drv, "eta motor": eta_motor}


def quad_mission_radius(q: dict, k4: dict) -> float:
    """The quad's radius on the R3 mission (same hover, wind, reserve; no transitions; cruise at the best ground-km
    speed per leg)."""
    if q["table"] is None:
        return 0.0
    df = q["table"][q["table"]["T each / max"] < 0.8]
    if not len(df):
        return 0.0
    w = MISSION["wind_m_s"]
    usable = k4["battery"]["usable_wh"] * (1 - MISSION["reserve_fraction"])
    hover_wh = MISSION["hover_total_s"] * q["hover [W]"] / 3600
    climb_wh = 2 * q["mass [kg]"] * G * MISSION["cruise_altitude_m"] / 0.45 / 3600
    e_km = 0.0
    for sgn in (-1, 1):
        gs = df.index.values + sgn * w
        e = np.where(gs > 1.0, df["electrical [W]"].values / gs / 3.6, np.inf)
        e_km += float(e.min())
    return max((usable - hover_wh - climb_wh) / e_km, 0.0)


# ================================================================================================= the comparison
def evaluate(variant: str) -> dict:
    veh = vehicle(variant)
    c4, c2 = cruise(veh, n_running=4), cruise(veh, n_running=2)
    h = hover(veh)
    cor = corridor(veh)
    mi = mission(veh, cr=c2, cor=cor)
    return {"veh": veh, "cruise4": c4, "cruise2": c2, "best4": best_speeds(c4), "best2": best_speeds(c2), "hover": h, "corridor": cor,
            "corridor_summary": corridor_summary(cor, veh), "mission": mi, "root": root_loads(veh), "directional": directional(veh["L"], veh, veh["aero"], fin_area=veh["p"]["fin_area"]),
            "fold": folded_best(veh["L"]), "price": k4tq.price(veh["items"], veh["p"])}


SELECTION = [
    ("hover: most loaded motor ≤ 75 % of its maximum", lambda e: e["hover"]["most loaded motor / its maximum"] <= 0.75),
    ("hover: T/W ≥ 1.6 on the sagged pack", lambda e: e["hover"]["T/W (sagged pack, 3.3 V/cell)"] >= 1.6),
    ("R3 standard mission closes with 1 kg", lambda e: e["mission"]["closes"] and e["veh"]["p"]["payload_kg"] >= 1.0),
    ("transition corridor continuous", lambda e: e["corridor_summary"]["continuous from hover to cruise"]),
    ("front wing stalls first (nose-down)", lambda e: e["veh"]["stall"]["nose-down stall"]),
    ("no layout clearance violated", lambda e: not k4tq.checks(e["veh"]["L"])),
    ("hover discs ≥ 100 mm apart (rotor interference)", lambda e: e["veh"]["L"]["clearance"]["hover disc separation along x [m]"] >= 0.10),
]


def selection(evals: dict) -> pd.DataFrame:
    """Each configuration against the selection criteria; among those that pass all, the larger R3 margin wins."""
    rows = {}
    for k, e in evals.items():
        r = {name: bool(fn(e)) for name, fn in SELECTION}
        r["passes all"] = all(r.values())
        r["R3 margin [Wh]"] = e["mission"]["margin [Wh]"]
        rows[f"K4TQ-{k}"] = r
    df = pd.DataFrame(rows).T
    ok = df[df["passes all"]]
    df["selected"] = False
    if len(ok):
        df.loc[ok["R3 margin [Wh]"].astype(float).idxmax(), "selected"] = True
    df.attrs["label"] = "ESTIMATE / " + label("CALC")
    return df


def _at(df: pd.DataFrame, V: float, col: str) -> float:
    return float(df.loc[V, col]) if (isinstance(V, (int, float)) and math.isfinite(V) and V in df.index) else math.nan


def comparison(evals: dict, quads: dict) -> pd.DataFrame:
    rows = {}
    for k, e in evals.items():
        v = e["veh"]
        rows[f"K4TQ-{k}"] = {"take-off mass [kg]": v["mass"]["mass_kg"], "payload [kg]": v["p"]["payload_kg"], "span / footprint [m]": v["L"]["width_overall"],
                             "hover [W]": e["hover"]["electrical [W]"], "T/W": e["hover"]["T/W (loaded pack)"],
                             "cruise power at best range [W]": _at(e["cruise2"], e["best2"]["best range V [m/s]"], "electrical [W]"),
                             "best range V [m/s]": e["best2"]["best range V [m/s]"], "still-air range [km]": e["best2"]["range [km]"],
                             "endurance [min]": e["best2"]["endurance [min]"], "max speed [m/s]": e["best4"]["max level V [m/s]"],
                             "R3 mission closes": e["mission"]["closes"], "R3 margin [Wh]": e["mission"]["margin [Wh]"],
                             "stall speed [m/s]": stall_speed(v), "tilt-wing tax [kg]": k4tq.tilt_wing_tax(v["items"]), "bought parts [EUR]": e["price"]["total bought"]}
    for k, q in quads.items():
        rows[k] = {"take-off mass [kg]": q["mass [kg]"], "payload [kg]": 1.0, "span / footprint [m]": q["footprint [m]"], "hover [W]": q["hover [W]"], "T/W": q["T/W"],
                   "cruise power at best range [W]": _at(q["table"], q["best range V [m/s]"], "electrical [W]"), "best range V [m/s]": q["best range V [m/s]"],
                   "still-air range [km]": q["range [km]"], "endurance [min]": q["hover minutes"], "max speed [m/s]": q["max speed (T ≤ 80 % max) [m/s]"],
                   "R3 mission closes": q["mission radius [km]"] >= MISSION["radius_km"], "R3 margin [Wh]": math.nan, "stall speed [m/s]": 0.0,
                   "tilt-wing tax [kg]": 0.0, "bought parts [EUR]": math.nan}
    df = pd.DataFrame(rows).T
    df.attrs["label"] = label("CALC")
    return df


def run_all(variants=("E", "A", "C", "A2")) -> dict:
    evals = {v: evaluate(v) for v in variants}
    k4 = evals[k4tq.MAIN]["veh"]
    q_same = f"Q-{k4['p']['prop'].split('x')[0]} ({k4['p']['prop']}, K4TQ's drive)"
    quads = {q_same: quad_reference("Q-same", k4["p"]["prop"], k4=k4), "Q-best (17x5.8)": quad_reference("Q-best", "17x5.8", k4=k4)}
    for q in quads.values():
        q["mission radius [km]"] = quad_mission_radius(q, k4)
    return {"evals": evals, "quads": quads, "comparison": comparison(evals, quads), "selection": selection(evals), "props": prop_sweep(k4), "payload_range": payload_range(k4tq.MAIN),
            "section": section_check(k4tq.section_of(k4["p"], "front")), "materials": materials(), "launch": launch(k4), "mocked": mocked()}


# ================================================================================================= the report
def _fmt(x, nd=2):
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        if not math.isfinite(x):
            return "—"
        return f"{x:.{nd}f}" if abs(x) < 1000 else f"{x:.0f}"
    if isinstance(x, tuple):
        return "(" + ", ".join(_fmt(v, nd) for v in x) + ")"
    return str(x)


def _kv(d: dict, nd=2) -> str:
    return "\n".join(f"| {k} | {_fmt(v, nd)} |" for k, v in d.items() if not isinstance(v, (dict, pd.DataFrame)))


def _df(df: pd.DataFrame, nd=2) -> str:
    cols = list(df.columns)
    head = "| " + (df.index.name or "") + " | " + " | ".join(map(str, cols)) + " |\n|" + "---|" * (len(cols) + 1)
    body = "\n".join("| " + _fmt(i, nd) + " | " + " | ".join(_fmt(v, nd) for v in r) + " |" for i, r in zip(df.index, df.values))
    return head + "\n" + body


def bom(veh: dict) -> pd.DataFrame:
    """The initial BOM from the mass table and the catalogue (bought parts priced; made parts as material + mass)."""
    rows = []
    for i in veh["items"]:
        make = "buy" if i["group"] in ("propulsion", "avionics", "battery", "cargo", "safety") else "hybrid" if i["group"] == "roots" else "make"
        rows.append({"item": i["item"], "group": i["group"], "mass [g]": round(1e3 * i["mass_kg"], 1), "make/buy": make, "source / material": i["source"]})
    return pd.DataFrame(rows)


def write_report(res: dict, out: Path = DATA) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    ev = res["evals"]
    A = ev[k4tq.MAIN]
    v = A["veh"]
    lab = "MOCK" if res["mocked"] else "CALC"
    md = []
    md.append("# K4TQ — Stage 1: engineering design and feasibility\n")
    md.append(f"Generated by `labs/aero/designs/k4tq_sizing.py` ({'mocks ON (R12): numerical results labelled MOCK' if res['mocked'] else 'mocks off'}). "
              "Labels: ESTIMATE (hand formula / correlation), CALC (numerical model), MOCK (a mocked tool ran). No solver result is in this report.\n")
    md.append("## Verdict\n")
    md.append(verdict_text(res) + "\n")
    md.append("## 1.15 Selection — " + res["selection"].attrs["label"] + "\n")
    bp = res["evals"][k4tq.MAIN]["veh"]["p"]
    md.append(_df(res["selection"].rename_axis("config")) + f"\n\nBaseline: **K4TQ-{k4tq.MAIN}** (larger rear wing, {bp['fuselage_length']:.1f} m fuselage, {bp['prop']} on "
              f"{k4tq.CATALOGUE['motor'][bp['motor']]['name']}, {bp['battery']}; masses: {bp.get('cad_mass_status', 'estimates')}) — the fuselage approved 2026-10-21; "
              "the propulsion and the pack revised by Stage 2.8 (the CAD masses).\n")
    md.append("\n## 1.15 Configurations and the quadcopter references\n")
    md.append(_df(res["comparison"].rename_axis("config")) + "\n")
    md.append("\n## 1.3 Mass budget (A) — ESTIMATE\n")
    gdf = pd.DataFrame({"mass [kg]": k4tq.groups(v["items"])}).rename_axis("group")
    md.append(_df(gdf, 3) + f"\n\nTake-off mass **{v['mass']['mass_kg']:.2f} kg**, CG x = {v['mass']['x_cg']:.3f} m (target {v['x_cg_target']:.3f}), "
              f"tilt-wing tax {k4tq.tilt_wing_tax(v['items']):.2f} kg. Inertia Ixx {v['mass']['Ixx']:.3f}, Iyy {v['mass']['Iyy']:.3f}, Izz {v['mass']['Izz']:.3f} kg·m².\n")
    md.append("\n## 1.4–1.5 Wing, section, tandem interference — CALC (lattice) / ESTIMATE (stall rule)\n")
    a = v["aero"]
    md.append("| quantity | value |\n|---|---|\n" + _kv({"S_ref [m²]": v["L"]["S_ref"], "AR (span²/S)": v["L"]["AR_total"], "wing loading [N/m²]": v["W"] / v["L"]["S_ref"],
                                                   "CLα [1/rad]": a["CLa"], "k": a["k"], "e": a["e"], "front share at α=4°": a["front_share_alpha4"],
                                                   "rear slope / one pair alone (downwash)": a["slope_rear"], "front slope / one pair alone": a["slope_front"],
                                                   "neutral point x [m]": a["x_np"], **{k: x for k, x in v["stall"].items() if k != "label"},
                                                   "stall speed [m/s]": stall_speed(v), "sections (front / rear)": f"{v['section']['front']['name']} / {v['section']['rear']['name']}",
                                                   "section source": v["section"]["front"]["source"]}, 3) + "\n")
    md.append(f"\nSection check through Zephyr: {res['section']['label']} — cl_max {res['section']['cl_max']:.2f} (recorded {res['section']['recorded']['cl_max']:.2f}).\n")
    md.append("\n## 1.6 Cruise — " + lab + "\n")
    md.append("Two motors (rear pair stopped, blades folded):\n\n" + _df(A["cruise2"][A["cruise2"]["feasible"]].drop(columns=["feasible"]).iloc[::2]) + "\n")
    md.append("\n| | 4 motors | 2 motors |\n|---|---|---|\n" + "\n".join(f"| {k} | {_fmt(A['best4'][k])} | {_fmt(A['best2'][k])} |" for k in A["best4"]) + "\n")
    md.append("\n## 1.7 Hover — " + lab + "\n")
    md.append("| quantity | value |\n|---|---|\n" + _kv(A["hover"]) + "\n")
    md.append("\n## 1.8 Propeller selection — " + lab + "\n")
    md.append(_df(res["props"]) + "\n")
    md.append("\n## 1.9 Transition corridor (level flight) — " + lab + "\n")
    cols = ["tilt [deg]", "thrust total [N]", "front share", "alpha front blown", "alpha front free", "stalled span front", "stalled span rear", "electrical [W]",
            "front motor / its max", "rear motor / its max", "feasible"]
    md.append(_df(A["corridor"][cols].iloc[::2]) + "\n")
    md.append("\n| summary | value |\n|---|---|\n" + _kv(A["corridor_summary"]) + "\n")
    md.append("\n## 1.10–1.11 Wing-root loads, spar, actuator — ESTIMATE\n")
    r = A["root"]
    md.append("| quantity | value |\n|---|---|\n" + _kv(r) + "\n\nActuator torque budget (limit, N·m):\n\n| term | N·m |\n|---|---|\n" + _kv(r["actuator torque budget (limit) [N·m]"], 3) + "\n")
    md.append("\nSpar tubes:\n\n" + _df(r["spar tubes"]) + f"\n\nChosen: **{r['spar chosen']}**.\n\nMaterials and factors:\n\n" + _df(res["materials"]) + "\n")
    md.append("\n## 1.12 Directional stability — ESTIMATE\n")
    md.append("| quantity | value |\n|---|---|\n" + _kv(A["directional"], 4) + "\n")
    md.append("\n## 1.13 Launch and landing (R6) — ESTIMATE\n")
    for k in ("hand", "tube", "roof"):
        md.append(f"\n**{k}**\n\n| quantity | value |\n|---|---|\n" + _kv(res["launch"][k]) + "\n")
    md.append("\n## Packing (R5) — ESTIMATE\n")
    md.append("| quantity | value |\n|---|---|\n" + _kv(A["fold"], 3) + "\n")
    md.append("\n## 1.14 Mission (R3) and payload–range (R2) — " + lab + "\n")
    m = A["mission"]
    for leg, e in m["legs"].items():
        md.append(f"\n**{leg}**: " + ", ".join(f"{k} {_fmt(x)}" for k, x in e.items()))
    md.append(f"\n\nUsed {m['energy used [Wh]']:.0f} Wh, with the 20 % reserve {m['energy with reserve [Wh]']:.0f} Wh, usable {m['usable pack [Wh]']:.0f} Wh → "
              f"**{'closes' if m['closes'] else 'does not close'}** (margin {m['margin [Wh]']:.0f} Wh).\n")
    md.append("\n" + _df(res["payload_range"].set_index("pack")) + f"\n\nDesign mass {res['payload_range'].attrs['design mass [kg]']:.2f} kg.\n")
    md.append("\n## Quadcopter references — " + lab + "\n")
    for k, q in res["quads"].items():
        md.append(f"\n**{k}**: mass {q['mass [kg]']:.2f} kg, footprint {q['footprint [m]']:.2f} m, hover {q['hover [W]']:.0f} W ({q['hover minutes']:.1f} min), "
                  f"CdA {q['CdA [m²]']:.3f} m², best range {q['range [km]']:.1f} km at {q['best range V [m/s]']:.0f} m/s, R3 radius {q['mission radius [km]']:.1f} km.")
    md.append("\n\n## Assumptions and uncertainties\n")
    md.append("\n".join(f"- {s}" for s in ASSUMPTIONS) + "\n")
    md.append("\n## Initial BOM\n\nSee `bom_phase1.csv` (masses from the budget, bought parts priced in `k4tq.CATALOGUE`; suppliers in Stage 4). Bought parts: "
              + ", ".join(f"{k} {x:.0f} €" for k, x in A["price"].items()) + ".\n")
    (out / "phase1.md").write_text("\n".join(md))
    bom(v).to_csv(out / "bom_phase1.csv", index=False)
    spec = {"label": lab, "variant": k4tq.MAIN, "params": v["p"], "requirements": k4tq.REQUIREMENTS, "mass": {k: x for k, x in v["mass"].items()},
            "x_cg_target": v["x_cg_target"], "aero": a, "stall": v["stall"], "drag": {"Cd0": v["drag"]["Cd0"], "CdA_m2": v["drag"]["CdA_m2"]},
            "hover": A["hover"], "best_cruise_2motors": A["best2"], "best_cruise_4motors": A["best4"], "corridor_summary": A["corridor_summary"],
            "mission": {k: x for k, x in A["mission"].items()}, "root": {k: x for k, x in A["root"].items() if k != "spar tubes"},
            "directional": A["directional"], "fold": A["fold"], "price": A["price"]}
    (out / "phase1_spec.json").write_text(json.dumps(spec, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    return spec


ASSUMPTIONS = [
    "Masses of the made parts from areal/linear densities (wings 0.70 kg/m², fuselage 0.60 kg/m²); the roots itemised from catalogue parts — Stage 2 CAD replaces them.",
    "Motor model: first-order (Kv, R, I0 constant); the MN4014's I0 at 22 V and its mass with cable are to verify — the no-load loss dominates the cruise.",
    "Propellers: Boreas's generic section on a scaled Kestrel planform (constant pitch) — the real blades differ by ±10 % in thrust and power.",
    "Section constants from NeuralFoil at Re 2e5, n_crit 6; the built-up wing at 0.9 × cl_max.",
    "Tandem lattice: flat wake, no slipstream; the transition corridor: quasi-steady, momentum slipstream at 1.5 vᵢ on the blown span, post-stall flat plate — the free span's stall at intermediate tilt is what the CFD (R10) must check.",
    "Li-ion 6S2P Molicel P45B, usable 85 % × 95 % of nominal; loaded voltage 3.5 V/cell, sagged 3.3 V/cell.",
    "Roof wind: 8 m/s × 1.25 roof-edge speed-up × 1.45 gust factor; quasi-static overturning about the feet.",
    "Quadcopter references: same motors, pack, payload, cargo, avionics, parachute and margin; momentum theory with a μ² profile-power growth and a build-up CdA.",
]


def verdict_text(res: dict) -> str:
    cmp = res["comparison"]
    A = cmp.loc[f"K4TQ-{k4tq.MAIN}"]
    q13, qb = cmp.loc[[k for k in cmp.index if k.startswith("Q-") and "best" not in k][0]], cmp.loc["Q-best (17x5.8)"]
    better_range = A["still-air range [km]"] / max(qb["still-air range [km]"], q13["still-air range [km]"])
    lines = [
        f"- **Feasible on paper (ESTIMATE/{('MOCK' if res['mocked'] else 'CALC')})**: K4TQ-{k4tq.MAIN} takes off at {A['take-off mass [kg]']:.2f} kg with 1 kg of payload, "
        f"hovers on {A['hover [W]']:.0f} W (T/W {A['T/W']:.2f}) and cruises at {A['best range V [m/s]']:.0f} m/s on {A['cruise power at best range [W]']:.0f} W; "
        f"the R3 mission {'closes' if A['R3 mission closes'] else 'does NOT close'} (margin {A['R3 margin [Wh]']:.0f} Wh).",
        f"- **Against the same-size quadcopters**: still-air range {A['still-air range [km]']:.0f} km against {q13['still-air range [km]']:.0f} km ({q13.name}) and "
        f"{qb['still-air range [km]']:.0f} km (Q-best) — ×{better_range:.1f}; top speed {A['max speed [m/s]']:.0f} against {qb['max speed [m/s]']:.0f} m/s; "
        f"but it hovers on {A['hover [W]'] / qb['hover [W]']:.2f}× Q-best's power, carries {A['tilt-wing tax [kg]']:.2f} kg of wings and mechanisms, "
        "and adds four actuators, four hinges and a conversion to the failure modes.",
        "- **Why**: the quad pays for lift with induced power at every speed; K4TQ pays it once in the hover and then flies on its wings at L/D ≈ "
        f"{res['evals'][k4tq.MAIN]['best2']['L/D max']:.0f}. The price is the mass of the wings and roots, a worse hover (heavier, smaller discs than Q-best), "
        "and mechanical complexity. The tilt-wing wins as soon as the mission is mostly cruise; the quad wins for short hops and long hovers.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    from vegeta import mock
    DEBUG = mock.setup(True)                                     # R12: mocks on unless VEGETA_DEBUG says otherwise
    res = run_all()
    spec = write_report(res)
    print(res["comparison"].round(2).to_string())
    print("\nwritten:", DATA / "phase1.md")
