"""ARIGO-1's 6-DOF flight model: the coefficients, the trim, the linear modes (notebook 39). Shared by the MuJoCo
stability runs (``sims/arigo1_mujoco.py``) and the manoeuvre videos (``sims/arigo1_video.py``).

What comes from where — read before trusting a number:

- **lattice (``ogma.aero.vlm``, wing + stabiliser, washout in the strips)**: CL_alpha, Cm_alpha, the neutral point,
  the elevator's lift and moment per degree (the stabiliser's incidence moved);
- **hand estimates (textbook formulas, marked ``HAND``, margin stated)**: the 2 % camber's zero-lift angle, CL_max
  and the post-stall blend, CD0, the Oswald factor, the lateral derivatives (dihedral and sweep for Cl_beta, the fin
  for Cn_beta / Cn_r / CY_beta, strip theory for Cl_p and the ailerons), the damping Cm_q, the flap effectiveness
  tau, the mass (no budget yet: 2.5 kg, the brief's middle), the inertias (radii of gyration), the thrust curve;
- **to be replaced by CFD (``arigo1_cfd``)**: CL_max, the stall break, CD0 and the moment curve — paste the YAML back.

Frame: body x forward, y right, z down (the flight-mechanics convention); alpha, beta in rad; SI.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

import arigo1 as a1

RHO, G = 1.225, 9.80665
MASS = 2.5                          # HAND: the brief's 2-3 kg, middle; no mass budget yet (+-0.5 kg)
STATIC_MARGIN = 0.12                # the CG put 12 % MAC ahead of the lattice's neutral point (the fuselage's
                                    # destabilising share, about 3-6 % MAC, is NOT in the lattice: the real margin is smaller)
CL_MAX_2D = 1.25                    # HAND: NACA 2412-like at Re 3e5 (NeuralFoil/XFOIL class value, +-0.1)
TAU_E, TAU_A, TAU_R = 0.55, 0.50, 0.55   # HAND: flap effectiveness for 35/25/40 % chord (Raymer fig. 16.7, +-0.1)
DE_MAX, DA_MAX, DR_MAX = 25.0, 20.0, 25.0  # deg, servo travel


def _vlm(pf, alpha, inc):
    from ogma.aero import vlm
    return vlm(pf["quads"], alpha_deg=alpha, incidence_deg=inc, S_ref=pf["S_ref"], c_ref=pf["c_ref"], x_ref=0.0)


@lru_cache(maxsize=None)
def coefficients(variant: str = "hybrid", propulsion: str = "pusher") -> dict:
    """The aircraft's derivatives (per rad unless named per deg) and the reference geometry; the CG placed by the
    static margin. Each entry's source is in ``sources``."""
    q = a1.variant(variant, propulsion)
    p = a1.resolve(q)
    L = a1.Arigo1.layout(p)
    pf = a1.planform(q)
    S, c, b = pf["S_ref"], pf["c_ref"], pf["b"]
    a0 = 2.0                                                       # HAND: 2 % camber at 40 %: zero-lift -2.1 deg (thin aerofoil)
    inc = [i + (a0 if w else 0.0) for i, w in zip(pf["incidence_deg"], pf["is_wing"])]
    r0, r4 = _vlm(pf, 0.0, inc), _vlm(pf, 4.0, inc)
    inc_e = [i + (0.0 if w else 1.0) for i, w in zip(inc, pf["is_wing"])]
    re = _vlm(pf, 0.0, inc_e)
    CLa = (r4["CL"] - r0["CL"]) / math.radians(4.0)
    dCm = (r4["Cm"] - r0["Cm"]) / math.radians(4.0)                # about the nose tip
    x_np = -dCm / CLa * c                                          # [m from the nose tip]
    x_cg = x_np - STATIC_MARGIN * c
    Cma = CLa * (x_cg - x_np) / c                                  # = -SM * CLa
    Cm0 = r0["Cm"] + r0["CL"] * x_cg / c                           # moved to the CG
    CLde = (re["CL"] - r0["CL"]) * TAU_E                           # per deg
    Cmde = ((re["Cm"] + re["CL"] * x_cg / c) - (r0["Cm"] + r0["CL"] * x_cg / c)) * TAU_E
    AR = b * b / S
    lam = p["taper"]
    sweep_qc = math.atan(math.tan(math.radians(p["le_sweep_deg"])) - (1 - lam) / (AR * (1 + lam)))
    e = 1.78 * (1 - 0.045 * AR ** 0.68) - 0.64                     # HAND: Raymer 12.49 (+-0.1)
    l_h, l_v = L["x_ac_h"] / 1e3 - x_cg, L["x_ac_v"] / 1e3 - x_cg
    S_h, S_v = L["S_h_mm2"] / 1e6, L["S_v_mm2"] / 1e6
    a_h = 2 * math.pi * (p["htail_span"] ** 2 / L["S_h_mm2"]) / (2 + (p["htail_span"] ** 2 / L["S_h_mm2"] + 4) ** 0.5)
    AR_v = 2 * p["fin_height"] ** 2 / L["S_v_mm2"]                 # end plate of the stabiliser: twice the geometric (HAND)
    a_v = 2 * math.pi * AR_v / (2 + (AR_v * AR_v + 4) ** 0.5)
    eta = 0.9                                                      # HAND: tail dynamic pressure ratio
    z_v = L["z_ac_v"] / 1e3
    CYb = -eta * a_v * S_v / S                                     # HAND (DATCOM-like, fuselage sidewash left out, +-30 %)
    Cnb = eta * a_v * S_v * l_v / (S * b) - 0.3 * (p["fuselage_length"] / 1e3) * (p["fuselage_diameter"] / 1e3) ** 2 / (S * b) * 2.0
    CLa_w = 2 * math.pi * AR / (2 + (AR * AR / math.cos(sweep_qc) ** 2 * (1 + math.tan(sweep_qc) ** 2) + 4) ** 0.5)
    gam = math.radians(p["dihedral_deg"])
    Clb_dihedral = -CLa_w * gam / 6 * (1 + 2 * lam) / (1 + lam)    # HAND (+-30 %)
    Clb_v = CYb * z_v / b
    Clp = -CLa_w / 12 * (1 + 3 * lam) / (1 + lam)                  # HAND: strip theory
    Cnr = -2 * eta * a_v * S_v * l_v ** 2 / (S * b * b)            # HAND: the fin only (wing's share small, left out)
    Clr0 = 1 / 4.0                                                 # HAND: Clr = CL/4 (strip theory), times CL in use
    Cnp0 = -1 / 8.0                                                # HAND: Cnp = -CL/8
    Cmq = -2 * eta * a_h * (S_h * l_h / (S * c)) * l_h / c         # HAND: the tail's damping (wing's share small)
    Cma_dot = Cmq * 0.35                                           # HAND: downwash lag ~ the tail's share times d(eps)/d(alpha)
    # ailerons by strips on the lattice's stations (HAND): the outer aileron_span of each half
    st = np.array([s[:3] for s in L["stations"]]) / 1000.0
    y = np.linspace(0, b / 2, 400)
    cy = np.interp(y, st[:, 0], st[:, 2])
    on = y >= b / 2 * (1 - p["aileron_span"])
    Clda = 2 * CLa_w * TAU_A * float(np.trapezoid(cy * y * on, y)) / (S * b) * math.pi / 180    # per deg
    Cnda = -0.1 * Clda                                             # HAND: adverse yaw, CL ~ 0.5
    Cndr = -eta * a_v * TAU_R * S_v * l_v / (S * b) * math.pi / 180 * 0.85                # per deg; 0.85: the rudder's span share
    CYdr = eta * a_v * TAU_R * S_v / S * math.pi / 180 * 0.85
    Cldr = CYdr * z_v / b
    CL_MAX = 0.9 * CL_MAX_2D * math.cos(sweep_qc)                  # HAND: Raymer 12.15 (+-0.1)
    CD0 = 0.032 if propulsion != "twin" else 0.036                  # HAND: component build-up class value (+-0.008); twin: two nacelles
    if propulsion == "tractor":
        CD0 += 0.003                                               # the chin turret's strut
    Lx, Ly = p["span"] / 1e3, p["fuselage_length"] / 1e3
    I = (MASS * (0.24 * Lx) ** 2, MASS * (0.30 * Ly) ** 2, MASS * ((0.24 * Lx) ** 2 + (0.30 * Ly) ** 2) * 0.95)  # HAND: radii of gyration (+-30 %)
    return {"variant": variant, "propulsion": propulsion, "S": S, "c": c, "b": b, "AR": AR, "mass": MASS, "inertia": I,
            "x_cg_m": x_cg, "z_cg_m": p["wing_z"] / 1e3 * 0.3, "x_np_m": x_np, "static_margin": STATIC_MARGIN,
            "CL0": r0["CL"], "CLa": CLa, "CL_MAX": CL_MAX, "alpha_stall": (CL_MAX - r0["CL"]) / CLa, "CD0": CD0, "k": 1 / (math.pi * e * AR),
            "Cm0": Cm0, "Cma": Cma, "Cmq": Cmq, "Cma_dot": Cma_dot, "CLde": CLde, "Cmde": Cmde, "CLq": -Cmq * c / l_h,
            "CYb": CYb, "Clb": Clb_dihedral + Clb_v, "Clb_sweep_per_CL": -math.tan(sweep_qc) / (AR * 1.0) * 0.6, "Cnb": Cnb,
            "Clp": Clp, "Clr_per_CL": Clr0, "Cnp_per_CL": Cnp0, "Cnr": Cnr, "Clda": Clda, "Cnda": Cnda, "Cndr": Cndr, "CYdr": CYdr, "Cldr": Cldr,
            "V_h": L["V_h"], "V_v": L["V_v"],
            "thrust": {"T0_N": 28.0, "V0_m_s": 38.0},               # HAND: a 10-11 in prop on a 4S 1000 W class motor, T = T0 (1 - V/V0) (+-25 %)
            "sources": {"vlm": ["CL0", "CLa", "Cma", "Cm0", "CLde", "Cmde", "x_np_m"],
                        "hand": ["CL_MAX", "CD0", "k", "Cmq", "Cma_dot", "CYb", "Clb", "Cnb", "Clp", "Cnr", "Clda", "Cndr", "inertia", "mass", "thrust"]}}


# ----------------------------------------------------------------------------------------------------------- forces
def aero(C: dict, vel_b, omega_b, de=0.0, da=0.0, dr=0.0, throttle=0.5):
    """Force and moment in the body frame (x fwd, y right, z down) about the CG [N, N m]; controls in deg."""
    u, v, w = vel_b
    p, q, r = omega_b
    V = max(math.sqrt(u * u + v * v + w * w), 0.5)
    al = math.atan2(w, u)
    be = math.asin(max(-1.0, min(1.0, v / V)))
    qd = 0.5 * RHO * V * V
    S, c, b = C["S"], C["c"], C["b"]
    # lift: linear to the stall, then a smooth fall onto the flat plate (2 sin a cos a) — HAND
    CL_lin = C["CL0"] + C["CLa"] * al
    CL_fp = 2 * math.sin(al) * math.cos(al)
    s = 1 / (1 + math.exp(-(abs(al) - C["alpha_stall"]) / math.radians(2.0)))
    CL = (1 - s) * CL_lin + s * CL_fp * 0.9
    CL += C["CLde"] * de + C["CLq"] * q * c / (2 * V)
    CD = C["CD0"] + C["k"] * CL_lin ** 2 * (1 - s) + s * (2 * math.sin(al) ** 2 * 0.9 + C["CD0"])
    CY = C["CYb"] * be + C["CYdr"] * dr
    k_ps = C.get("post_stall_cm_factor", 1.0)           # HAND: 1 = the stability slope kept past the stall (tail unstalled,
                                                        # optimistic); 0.35 = mostly lost (pessimistic); CFD decides
    Cm = C["Cm0"] + C["Cma"] * ((1 - s) * al + s * (C["alpha_stall"] + k_ps * (abs(al) - C["alpha_stall"])) * math.copysign(1, al)) \
        + C["Cmq"] * q * c / (2 * V) + C["Cmde"] * de
    CLu = max(CL, 0.0)
    Cl = (C["Clb"] + C["Clb_sweep_per_CL"] * CLu) * be + C["Clp"] * p * b / (2 * V) + C["Clr_per_CL"] * CLu * r * b / (2 * V) \
        + C["Clda"] * da + C["Cldr"] * dr
    Cl *= (1 - 0.6 * s)                                             # HAND: the ailerons and roll damping fade in the stall
    Cn = C["Cnb"] * be + C["Cnp_per_CL"] * CLu * p * b / (2 * V) + C["Cnr"] * r * b / (2 * V) + C["Cnda"] * da + C["Cndr"] * dr
    ca, sa, cb, sb = math.cos(al), math.sin(al), math.cos(be), math.sin(be)
    # wind -> body: drag along -V, lift perpendicular in the x-z plane, side force along y
    Fx = qd * S * (-CD * ca * cb + CL * sa)
    Fy = qd * S * (-CD * sb + CY)
    Fz = qd * S * (-CD * sa * cb - CL * ca)
    T = max(C["thrust"]["T0_N"] * max(min(throttle, 1.0), 0.0) * (1 - u / C["thrust"]["V0_m_s"]), 0.0)
    return np.array([Fx + T, Fy, Fz]), np.array([qd * S * b * Cl, qd * S * c * Cm, qd * S * b * Cn]), {"alpha": al, "beta": be, "V": V, "CL": CL, "stall": s}


# ----------------------------------------------------------------------------------------- rigid body (numpy twin)
def _R(phi, th, psi):
    cf, sf, ct, st, cp, sp = math.cos(phi), math.sin(phi), math.cos(th), math.sin(th), math.cos(psi), math.sin(psi)
    return np.array([[ct * cp, sf * st * cp - cf * sp, cf * st * cp + sf * sp],
                     [ct * sp, sf * st * sp + cf * cp, cf * st * sp - sf * cp],
                     [-st, sf * ct, cf * ct]])                       # body -> NED


def deriv(C, x, u):
    """x = [u v w p q r phi theta psi N E D]; u = (de, da, dr, throttle)."""
    vb, wb, (phi, th, psi) = x[0:3], x[3:6], x[6:9]
    F, M, _ = aero(C, vb, wb, *u)
    m, I = C["mass"], np.array(C["inertia"])
    gb = _R(phi, th, psi).T @ np.array([0, 0, G])
    dv = F / m + gb - np.cross(wb, vb)
    dw = (M - np.cross(wb, I * wb)) / I
    p, q, r = wb
    dang = np.array([p + (q * math.sin(phi) + r * math.cos(phi)) * math.tan(th), q * math.cos(phi) - r * math.sin(phi),
                     (q * math.sin(phi) + r * math.cos(phi)) / math.cos(th)])
    return np.concatenate([dv, dw, dang, _R(phi, th, psi) @ vb])


def trim(C, V: float = 18.0) -> dict:
    """Level flight at V: alpha, elevator, throttle (Newton on the three equations)."""
    x = np.array([math.radians(3.0), 0.0, 0.5])
    for _ in range(40):
        def res(z):
            al, de, th = z
            d = deriv(C, np.array([V * math.cos(al), 0, V * math.sin(al), 0, 0, 0, 0, al, 0, 0, 0, 0]), (de, 0, 0, th))
            return np.array([d[0], d[2], d[4]])
        f = res(x)
        J = np.column_stack([(res(x + dz) - f) / 1e-5 for dz in np.eye(3) * 1e-5])
        x = x - np.linalg.solve(J, f)
        if np.abs(f).max() < 1e-8:
            break
    al, de, th = x
    return {"V": V, "alpha_deg": math.degrees(al), "elevator_deg": de, "throttle": th, "ok": bool(np.abs(f).max() < 1e-5 and 0 <= th <= 1 and abs(de) < DE_MAX)}


def modes(C, V: float = 18.0) -> dict:
    """The linear modes at the level trim (finite differences of ``deriv``): period, damping ratio, time to half
    (or double) amplitude for each."""
    t = trim(C, V)
    al = math.radians(t["alpha_deg"])
    x0 = np.array([V * math.cos(al), 0, V * math.sin(al), 0, 0, 0, 0, al, 0, 0, 0, 0])
    u0 = (t["elevator_deg"], 0, 0, t["throttle"])
    n = 9
    A = np.zeros((n, n))
    f0 = deriv(C, x0, u0)[:n]
    for j in range(n):
        dx = np.zeros(12); dx[j] = 1e-5
        A[:, j] = (deriv(C, x0 + dx, u0)[:n] - f0) / 1e-5
    lon, lat = [0, 2, 4, 7], [1, 3, 5, 6]
    out = {"trim": t}
    for name, idx in (("longitudinal", lon), ("lateral", lat)):
        ev = np.linalg.eigvals(A[np.ix_(idx, idx)])
        rows = []
        for e in sorted(ev, key=lambda z: -abs(z)):
            if e.imag < -1e-9:
                continue
            wn = abs(e)
            rows.append({"real": float(e.real), "imag": float(e.imag), "omega_n": float(wn), "zeta": float(-e.real / wn) if wn else None,
                         "period_s": float(2 * math.pi / e.imag) if e.imag > 1e-9 else None,
                         "t_half_or_double_s": float(math.log(2) / abs(e.real)) if abs(e.real) > 1e-9 else None, "stable": bool(e.real < 0)})
        out[name] = rows
    return out


def name_modes(md: dict) -> dict:
    """Short period / phugoid; roll subsidence / Dutch roll / spiral — by the eigenvalues' shape."""
    lon = md["longitudinal"]; lat = md["lateral"]
    osc = sorted([r for r in lon if r["period_s"]], key=lambda r: -r["omega_n"])
    named = {}
    if osc:
        named["short_period"] = osc[0]
        if len(osc) > 1:
            named["phugoid"] = osc[-1]
    real = sorted([r for r in lat if not r["period_s"]], key=lambda r: r["real"])
    if real:
        named["roll_subsidence"] = real[0]
        if len(real) > 1:
            named["spiral"] = real[-1]
    o = [r for r in lat if r["period_s"]]
    if o:
        named["dutch_roll"] = o[0]
    return named
