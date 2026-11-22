"""STERNA's flight model (notebook 51): the coefficient set the lab's ``arigo1_model.aero`` / ``trim`` / ``modes``
and the MuJoCo ``Sim`` take, built for STERNA's geometry and mass.

What comes from where (the lab's convention):
- **lattice** (``ogma.aero.vlm`` on ``sterna.planform``): CL0, CLa, the neutral point, Cm0, the elevator's CL and Cm.
  The fuselage and the propeller are not in the lattice: their destabilising share is taken as ``FUSELAGE_NP_SHIFT``;
- **section** (Zephyr): CL_max 2-D, passed in by the notebook (``cl_max_2d``), else a hand value;
- **hand**: CD0, e, the lateral derivatives, damping, inertia, the thrust line. Marked ``HAND`` with a margin;
- **CFD**: ``calibrate(C, rows)`` replaces CL0, CLa, CD0, k and the neutral point with a fit to the notebook's
  Aeromant cases.
"""
from __future__ import annotations

import math

import numpy as np

import arigo1_model as am
import sterna as st
import sterna_mass as sm

CL_MAX_2D = 1.25               # HAND: 3 % camber, 12 % thick at Re 2e5 (+-0.15); the notebook passes Zephyr's value
FUSELAGE_NP_SHIFT = 0.04       # HAND: fuselage + tractor propeller move the neutral point forward by ~4 % MAC (+-2)
A0_DEG = 3.1                   # thin aerofoil: zero-lift angle of 3 % camber at 40 % (-3.1 deg)
T_STATIC_OVER_WEIGHT = 0.7     # the requirement on the motor: static thrust / weight at the mass limit
V0_M_S = 26.0                  # HAND: the speed where full-throttle thrust falls to zero (pitch speed of an 11x7 at ~7500 rpm, +-4)


def _vlm(pf, alpha, inc):
    from ogma.aero import vlm
    return vlm(pf["quads"], alpha_deg=alpha, incidence_deg=inc, S_ref=pf["S_ref"], c_ref=pf["c_ref"], x_ref=0.0, n_span=st.N_SPAN_PER_STRIP)


def alpha_stall(CL0: float, CLa: float, CL_MAX: float) -> float:
    """The stall angle [rad] for ``arigo1_model.aero``: its lift is the linear curve blended onto a flat plate by a
    sigmoid 2 degrees wide about ``alpha_stall``, which rounds the peak well below the linear value at that angle.
    The angle is set so the blended curve's peak is CL_MAX (the stall speed in the simulation is then the stated one)."""
    al = np.radians(np.arange(0.0, 30.0, 0.05))

    def peak(a_s):
        s = 1 / (1 + np.exp(-(al - a_s) / math.radians(2.0)))
        return float(np.max((1 - s) * (CL0 + CLa * al) + s * 0.9 * 2 * np.sin(al) * np.cos(al)))
    a_s = (CL_MAX - CL0) / CLa
    while peak(a_s) < CL_MAX and a_s < math.radians(25.0):
        a_s += math.radians(0.05)
    return float(a_s)


def coefficients(case: str = sm.DEFAULT_CASE, cl_max_2d: float | None = None, **over) -> dict:
    p = st.resolve(over)
    L = st.Sterna.layout(p)
    mb = sm.budget(p, L, case)
    mass = mb["total_g"] / 1000.0
    pf = st.planform(p)
    S, c, b = pf["S_ref"], pf["c_ref"], pf["b"]
    inc = [i + (A0_DEG if w else 0.0) for i, w in zip(pf["incidence_deg"], pf["is_wing"])]
    r0, r4 = _vlm(pf, 0.0, inc), _vlm(pf, 4.0, inc)
    re = _vlm(pf, 0.0, [i + (0.0 if w else 1.0) for i, w in zip(inc, pf["is_wing"])])
    CLa = (r4["CL"] - r0["CL"]) / math.radians(4.0)
    dCm = (r4["Cm"] - r0["Cm"]) / math.radians(4.0)
    x_np_lattice = -dCm / CLa * c
    x_np = x_np_lattice - FUSELAGE_NP_SHIFT * c
    x_cg = mb["x_cg_mm"] / 1000.0
    Cm0 = r0["Cm"] + r0["CL"] * x_cg / c
    CLde = (re["CL"] - r0["CL"]) * am.TAU_E
    Cmde = ((re["Cm"] + re["CL"] * x_cg / c) - (r0["Cm"] + r0["CL"] * x_cg / c)) * am.TAU_E
    AR = b * b / S
    lam = p["tip_chord"] / p["root_chord"]
    e = 1.78 * (1 - 0.045 * AR ** 0.68) - 0.64                     # HAND: Raymer 12.49 (+-0.1)
    l_h, l_v = L["x_ac_h"] / 1e3 - x_cg, L["x_ac_v"] / 1e3 - x_cg
    S_h, S_v = L["S_h_mm2"] / 1e6, L["S_v_mm2"] / 1e6
    AR_h = p["htail_span"] ** 2 / L["S_h_mm2"]
    a_h = 2 * math.pi * AR_h / (2 + (AR_h ** 2 + 4) ** 0.5)
    AR_v = 1.5 * 205.0 ** 2 / L["S_v_mm2"]                         # HAND: the fuselage as a partial end plate (x1.5)
    a_v = 2 * math.pi * AR_v / (2 + (AR_v ** 2 + 4) ** 0.5)
    eta = 0.9                                                      # HAND: the low tail near the wing's wake
    z_v = L["z_ac_v"] / 1e3
    CYb = -eta * a_v * S_v / S
    Cnb = eta * a_v * S_v * l_v / (S * b) - 0.3 * 1.1 * 0.11 ** 2 / (S * b) * 2.0     # the fuselage's share as in arigo1_model
    CLa_w = 2 * math.pi * AR / (2 + (AR * AR + 4) ** 0.5)
    yb, b2 = p["centre_span"] / 2, p["span"] / 2
    gam_eff = math.radians(p["dihedral_deg"]) * (1 - (yb / b2) ** 2)                  # HAND: the outer panels' share of the rolling arm
    Clb = -CLa_w * gam_eff / 6 * (1 + 2 * lam) / (1 + lam) - 0.0004 * 57.3 * 0.5 + CYb * z_v / b   # shoulder wing: ~-0.0002/deg (HAND)
    Clp = -CLa_w / 12 * (1 + 3 * lam) / (1 + lam)
    Cnr = -2 * eta * a_v * S_v * l_v ** 2 / (S * b * b)
    Cmq = -2 * eta * a_h * (S_h * l_h / (S * c)) * l_h / c
    stn = np.array([s[:3] for s in L["stations"]]) / 1000.0
    y = np.linspace(0, b / 2, 400)
    cy = np.interp(y, stn[:, 0], stn[:, 2])
    on = (y >= p["aileron_from"] / 1e3) & (y <= p["aileron_to"] / 1e3)
    Clda = 2 * CLa_w * am.TAU_A * float(np.trapezoid(cy * y * on, y)) / (S * b) * math.pi / 180
    Cndr = -eta * a_v * 0.45 * S_v * l_v / (S * b) * math.pi / 180                    # tau 0.45 for the 30 % rudder (HAND)
    CYdr = eta * a_v * 0.45 * S_v / S * math.pi / 180
    cl2 = CL_MAX_2D if cl_max_2d is None else cl_max_2d
    CL_MAX = 0.9 * cl2                                             # HAND: Raymer 12.15, unswept (+-0.1)
    CD0 = 0.030                                                    # HAND: build-up class value with rails and a window step (+-0.008)
    Lx, Ly = b, L["length"] / 1e3
    I = (mass * (0.22 * Lx) ** 2, mass * (0.26 * Ly) ** 2, mass * ((0.22 * Lx) ** 2 + (0.26 * Ly) ** 2) * 0.95)   # HAND (+-30 %)
    T0 = T_STATIC_OVER_WEIGHT * sm.MTOW_G / 1000.0 * am.G
    return {"variant": "sterna", "propulsion": "tractor", "case": case, "S": S, "c": c, "b": b, "AR": AR, "mass": mass, "inertia": I,
            "x_cg_m": x_cg, "z_cg_m": 0.0, "x_np_m": x_np, "x_np_lattice_m": x_np_lattice, "static_margin": (x_np - x_cg) / c,
            "CL0": r0["CL"], "CLa": CLa, "CL_MAX": CL_MAX, "alpha_stall": alpha_stall(r0["CL"], CLa, CL_MAX), "CD0": CD0, "k": 1 / (math.pi * e * AR),
            "Cm0": Cm0 + r0["CL"] * 0.0, "Cma": CLa * (x_cg - x_np) / c, "Cmq": Cmq, "Cma_dot": 0.35 * Cmq, "CLde": CLde, "Cmde": Cmde,
            "CLq": -Cmq * c / l_h, "CYb": CYb, "Clb": Clb, "Clb_sweep_per_CL": 0.0, "Cnb": Cnb, "Clp": Clp, "Clr_per_CL": 0.25,
            "Cnp_per_CL": -0.125, "Cnr": Cnr, "Clda": Clda, "Cnda": -0.1 * Clda, "Cndr": Cndr, "CYdr": CYdr, "Cldr": CYdr * z_v / b,
            "V_h": L["V_h"], "V_v": L["V_v"], "thrust": {"T0_N": T0, "V0_m_s": V0_M_S, "note": "hand line T0 (1 - V/V0): the motor requirement, not a measured map"},
            "mass_budget": mb,
            "sources": {"vlm": ["CL0", "CLa", "Cm0", "CLde", "Cmde", "x_np_lattice_m"],
                        "hand": ["x_np_m (lattice - fuselage share)", "CL_MAX", "CD0", "k", "Cmq", "CYb", "Clb", "Cnb", "Clp", "Cnr", "Clda", "Cndr", "inertia", "thrust"],
                        "budget": ["mass", "x_cg_m"]}}


def calibrate(C: dict, rows: list) -> dict:
    """The coefficients with CL0, CLa, CD0, k, Cm0 and the neutral point from a fit to the CFD rows
    (``alpha_deg``, ``Cl``, ``Cd``, ``Cm_cg`` about the design CG, attached flow only). The CG stays where the budget
    put it; the static margin is what the CFD says it is."""
    A = np.radians([r["alpha_deg"] for r in rows]); CL = np.array([r["Cl"] for r in rows]); CD = np.array([r["Cd"] for r in rows]); Cm = np.array([r["Cm_cg"] for r in rows])
    CLa, CL0 = np.polyfit(A, CL, 1)
    k, CD0 = np.polyfit(CL ** 2, CD, 1)
    Cma, Cm0 = np.polyfit(A, Cm, 1)
    x_cg_cfd = rows[0].get("x_cg_m", C["x_cg_m"])
    x_np = x_cg_cfd - Cma / CLa * C["c"]
    out = dict(C)
    out.update({"CL0": float(CL0), "CLa": float(CLa), "CD0": float(CD0), "k": float(max(k, 1e-3)), "x_np_m": float(x_np),
                "static_margin": float((x_np - C["x_cg_m"]) / C["c"]), "Cma": float(CLa * (C["x_cg_m"] - x_np) / C["c"]),
                "Cm0": float(Cm0 + CL0 * (C["x_cg_m"] - x_cg_cfd) / C["c"]), "alpha_stall": alpha_stall(float(CL0), float(CLa), C["CL_MAX"]),
                "calibrated_from": f"{len(rows)} CFD cases"})
    return out


def stall_speed(C: dict) -> float:
    return math.sqrt(C["mass"] * am.G / (0.5 * am.RHO * C["S"] * C["CL_MAX"]))


def speeds(C: dict) -> dict:
    """Stall, best glide and minimum-power speeds from the polar CD0 + k CL^2 (level flight, propeller efficiency apart)."""
    W, q1 = C["mass"] * am.G, 0.5 * am.RHO * C["S"]
    cl_ld, cl_mp = math.sqrt(C["CD0"] / C["k"]), math.sqrt(3 * C["CD0"] / C["k"])
    V = lambda cl: math.sqrt(W / (q1 * cl))
    return {"V_stall": V(C["CL_MAX"]), "V_best_glide": V(cl_ld), "LD_max": cl_ld / (2 * C["CD0"]), "V_min_power": V(min(cl_mp, 0.95 * C["CL_MAX"])),
            "CL_best_glide": cl_ld, "CL_min_power": cl_mp}


def power_required(C: dict, V: float) -> float:
    """Aerodynamic power for level flight at V [W] (drag x speed; no propulsive efficiency)."""
    W, q = C["mass"] * am.G, 0.5 * am.RHO * V * V
    cl = W / (q * C["S"])
    return q * C["S"] * (C["CD0"] + C["k"] * cl * cl) * V


# ------------------------------------------------------------------------------------------------------- propulsion
RPM_THROTTLES = tuple(round(0.1 * k, 1) for k in range(1, 11))     # the drive's throttle settings in the rpm map
AVIONICS_W = 12.0            # HAND: autopilot, radios, video downlink, camera, servos at rest (+-5)


def propulsion(modules: int = 1, kv: float = 700.0):
    """The drive as a Boreas system: an 11x7 folding propeller (generic section: Boreas says its static points come
    out about 25 % low against measured propellers), a 35 mm outrunner class motor (HAND resistance and no-load
    current), 4S Li-ion modules in parallel."""
    from vegeta import boreas
    import sterna_mass as sm
    d, pitch = boreas.inches(11, 7)
    prop = boreas.Propeller.from_pitch("11x7 folding", d, pitch, blades=2, chord_root_m=0.018, chord_max_m=0.026, chord_tip_m=0.008,
                                       mass_kg=0.020, rotor_mass_kg=0.040)
    airfoil = boreas.Airfoil(cl_alpha=5.7, alpha0_deg=-2.0, cl_max=1.1, cd0=0.02, k=0.04)
    motor = boreas.Motor(f"35 mm outrunner, {kv:.0f} KV class", kv_rpm_per_volt=kv, resistance_ohm=0.05, no_load_current_a=1.0, max_current_a=40.0)
    battery = boreas.Battery(f"4S{modules}P P42A", cells=4, capacity_ah=sm.CELL_AH * modules)
    return boreas.Propulsion(prop, airfoil, motor, battery)


def rpm_grid(system, speeds) -> dict:
    """The drive's speed and thrust over its own throttle and the airspeed: what turns a thrust at an airspeed back
    into the propeller's rpm (the flight model scales the full-throttle thrust linearly with its throttle; the real
    drive's rpm for that thrust is not linear in it: in cruise the propeller must turn faster than its zero-thrust
    speed, V / pitch)."""
    pts = [[system.at_throttle(t, airspeed=float(V)) for V in speeds] for t in RPM_THROTTLES]
    return {"throttle": list(RPM_THROTTLES), "V": [float(v) for v in speeds],
            "rpm": [[float(q.rpm) for q in row] for row in pts], "thrust_N": [[float(q.aero.thrust) for q in row] for row in pts]}


def thrust_table(system, speeds=(0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0)) -> dict:
    """Full-throttle thrust, current and efficiency over speed, in the form ``arigo1_model.max_thrust`` reads."""
    pts = [system.at_throttle(1.0, airspeed=float(V)) for V in speeds]
    return {"table_V": [float(v) for v in speeds], "table_T": [max(float(q.aero.thrust), 0.0) for q in pts],
            "table_A": [float(q.current) for q in pts], "table_eta_prop": [float(q.aero.efficiency) for q in pts],
            "current_limited": any(q.current_limited for q in pts), "T0_N": float(pts[0].aero.thrust),
            "table_rpm": [float(q.rpm) for q in pts],
            "rpm_grid": rpm_grid(system, speeds),
            "note": "Boreas blade-element map at full throttle (generic section; no installation loss)"}


def cruise_point(C: dict, system, V: float) -> dict:
    """Level flight at V on the polar: the drag, the drive's operating point for that thrust, electrical power with
    the avionics, and the time one charge lasts at that power (80 % of the nominal energy used)."""
    import sterna_mass as sm
    W, q = C["mass"] * am.G, 0.5 * am.RHO * V * V
    cl = W / (q * C["S"])
    D = q * C["S"] * (C["CD0"] + C["k"] * cl * cl)
    pt = system.for_thrust(D, airspeed=V)
    n = C["mass_budget"]["battery"]["modules"]
    P = float(pt.electrical_power) + AVIONICS_W
    return {"V": V, "CL": cl, "L_D": W / D, "drag_N": D, "throttle": float(pt.throttle), "prop_efficiency": float(pt.aero.efficiency),
            "drive_W": float(pt.electrical_power), "total_W": P, "usable_Wh": 0.8 * n * sm.MODULE_WH, "minutes": 60 * 0.8 * n * sm.MODULE_WH / P}
