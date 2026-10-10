"""FALCO structure — the load cases from the mountain flight conditions (in EAS), the hand checks, the Talos models of
the load-carrying parts and what linear-static FEA cannot tell (notebook 33).

NISUS's method (``nisus_structure``, whose materials, knock-downs, FEA case record and result post-processing this
module reuses) with what the mountains change:

- **the speeds are EAS**: the dynamic pressure that loads the structure is ``½ ρ0 V_EAS²`` at any altitude; the gust's
  load factor (Pratt) is ``Δn = K_g ρ0 V_EAS U_EAS CLα S / 2W`` with the alleviation's mass ratio at the altitude's
  density (thinner air: a heavier aircraft relative to the air, a little less alleviation) — taken at the work
  altitude (4000 m) and at sea level, the larger counts;
- **the gusts are mountain gusts**: 10 m/s EAS at V_C (22 m/s EAS: the penetration speed in a 12 m/s valley wind) and
  5 m/s at V_NE (35 m/s EAS); the manoeuvre limit ``falco_flight.N_STRUCTURAL`` (4.4: the pull-out after a fast descent);
- **crow**: the flaps at 55° at V_FE (their hinge moment on the horn, the tail's load to trim the crow's pitching
  moment), the propeller brake's reverse thrust on the motor mount at V_NE;
- **launch and landing** at 5 kg: a bungee (18 m/s in 0.25 s: 7.3 g) as well as a hand throw; the belly landing at
  2.5 m/s sink on the 70 mm TPU skid (20 mm stroke) — the touchdown at a 1200 m meadow is ~13 m/s TAS;
- **new parts**: the carbon **spar joiner** at the panel joint (the outer panel's lift as bearing loads on its outer
  half, its inner half held in the centre spar).

Units: the Talos models are mm-N-MPa (density t/mm³).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import falco
import falco_flight as ff
import falco_systems as fs
import nisus_structure as nst
from nisus_structure import (ULTIMATE, KNOCKDOWN, PETG, CARBON, BUILD_AXIS, PRINT_SETTINGS, FEACase, talos_material, across_layers, nominal_stress,
                             summary_row)

G, RHO0 = fs.G, fs.RHO0
FOAM_LIGHTENING_SHARE = nst.FOAM_LIGHTENING_SHARE
V_C_EAS = 22.0                      # the penetration speed in a 12 m/s valley wind
U_C, U_D = 10.0, 5.0


# ================================================================================================= loads
def design_mass_kg(p=None, battery_key=fs.DEFAULT_PACK) -> float:
    """The heavy configuration (6S4P) + 8 % growth (the operating limits' maximum take-off mass)."""
    return 1.08 * fs.cg_inertia(fs.mass_table(battery_key, p=p), p)["mass_kg"]


def gust_n(mass_kg, V_eas, U_eas, a, rho_alt=RHO0) -> float:
    """Pratt's gust in EAS: Δn = K_g ρ0 V_e U_e CLα S / 2W, K_g = 0.88 μ / (5.3 + μ) with μ at the altitude's density."""
    W = mass_kg * G
    mu = 2 * W / (rho_alt * a["c_ref"] * a["CLa"] * G * a["S_ref"])
    Kg = 0.88 * mu / (5.3 + mu)
    return 1 + Kg * RHO0 * V_eas * U_eas * a["CLa"] * a["S_ref"] / (2 * W)


def load_cases(p=None, *, V_C=V_C_EAS, V_NE=ff.V_NE_EAS, V_FE=ff.V_FE_EAS, U_C_=U_C, U_NE=U_D, h_work=4000.0, dr: fs.FalcoDrive | None = None, a=None) -> pd.DataFrame:
    """The structural load cases (limit values; x ultimate in the FEA) with their derivation."""
    p = falco.resolve(p)
    L = falco.Falco.layout(p)
    a = a or ff.aero(p)
    m = design_mass_kg(p)
    W = m * G
    n_man = ff.N_STRUCTURAL
    rho_w = fs.atmosphere(h_work)["rho"]
    n_gc = max(gust_n(m, V_C, U_C_, a, RHO0), gust_n(m, V_C, U_C_, a, rho_w))
    n_gd = max(gust_n(m, V_NE, U_NE, a, RHO0), gust_n(m, V_NE, U_NE, a, rho_w))
    q_ne = 0.5 * RHO0 * V_NE ** 2
    q_fe = 0.5 * RHO0 * V_FE ** 2
    n_lift_ne = q_ne * L["S_ref"] * ff.cl_max(p) / W
    n_limit = max(n_man, n_gc, n_gd)
    tail = q_ne * L["S_h"] * ff.TAIL_SECTION["cl_max_2d"]
    fin = q_ne * L["S_v_each"] * ff.TAIL_SECTION["cl_max_2d"]
    cr = ff.crow_increments(p, a=a)
    tail_crow = q_fe * L["S_ref"] * L["mac"] / 1000 * abs(cr["dCm"]) / (L["l_t"] / 1000)
    c_f = p["flap_chord_frac"] * L["chord_joint"] / 1000
    H_flap = q_fe * L["S_flap_each"] * c_f * 0.40 * math.radians(40.0)
    flap_normal = q_fe * L["S_flap_each"] * 1.2                                       # a 55° plain flap's normal force coefficient ~1.2 (assumed)
    dr = dr or fs.drive()
    st = dr.at(0.0, 1.0)
    torque = st["shaft_power"] / (st["rpm"] * math.pi / 30)
    V_ne_sl = V_NE                                                                     # EAS = TAS at sea level: the brake's largest drag
    brake = dr.brake(V_ne_sl, 1.0, RHO0)
    v_l, t_l = 18.0, 0.25
    a_launch = max(v_l / t_l / G, 4.0)
    sink, stroke = 2.5, 0.020
    a_land = sink ** 2 / (2 * stroke) / G + 1.0
    ci = fs.cg_inertia(fs.mass_table(p=p), p)
    l_stub = (L["boom_x1"] - 90.0) / 1000 - ci["x_cg_m"]
    m_eff = 1 / (1 / m + l_stub ** 2 / (ci["Iyy"] * 1.08))
    sink_t, stroke_t = 1.5, 0.012
    F_stub = m_eff * sink_t ** 2 / (2 * stroke_t)
    rows = {
        "design mass [kg]": (m, "6S4P + 8 % (MTOM)"),
        "manoeuvre limit n": (n_man, "falco_flight.N_STRUCTURAL: the pull-out after a fast descent"),
        f"gust n at V_C = {V_C:g} m/s EAS, U = {U_C_:g} m/s": (n_gc, f"Pratt in EAS, μ at sea level and at {h_work:.0f} m: the larger"),
        f"gust n at V_NE = {V_NE:g} m/s EAS, U = {U_NE:g} m/s": (n_gd, "Pratt in EAS"),
        "lift-limited n at V_NE": (n_lift_ne, "CL_max at V_NE: the gust cannot exceed it"),
        "design limit n (wing)": (n_limit, "max of manoeuvre and gusts"),
        "ultimate n (wing)": (ULTIMATE * n_limit, f"x {ULTIMATE}"),
        "tail load at V_NE, limit [N]": (tail, f"q S_h cl_max_t ({ff.TAIL_SECTION['cl_max_2d']}), both booms"),
        "fin side load at V_NE, limit [N]": (fin, "q S_v1 cl_max_t, per fin"),
        "tail load to trim crow at V_FE, limit [N]": (tail_crow, f"ΔCm_crow {cr['dCm']:+.3f} x q S c / l_t"),
        "flap hinge moment in crow at V_FE, limit [N m]": (H_flap, "q S_f c_f Ch_δ δ (Ch_δ 0.4, δ capped at 40°: assumed)"),
        "flap normal force in crow at V_FE, limit [N]": (flap_normal, "q S_f x 1.2 (assumed): on its two hinges"),
        "tail-first touchdown on a stub, limit [N]": (F_stub, f"{sink_t} m/s, {stroke_t * 1000:.0f} mm stroke, effective mass {m_eff:.3f} kg"),
        "launch acceleration [g]": (a_launch, f"a bungee: {v_l:g} m/s in {t_l:g} s (the hand throw is less)"),
        "belly landing [g]": (a_land, f"{sink} m/s sink, {stroke * 1000:.0f} mm stroke (skid + grass) + 1 g"),
        "static thrust [N]": (st["thrust"], "the drive at full throttle, sea level, static"),
        "motor torque [N m]": (torque, "shaft power / ω at full throttle"),
        f"propeller brake at V_NE, sea level [N]": (brake["thrust"], "reverse thrust at the largest braking drag"),
    }
    df = pd.DataFrame({k: {"value": v[0], "basis": v[1]} for k, v in rows.items()}).T
    df.attrs.update(mass=m, W=W, n_limit=n_limit, n_ult=ULTIMATE * n_limit, tail=max(tail, tail_crow), fin=fin, F_stub=F_stub, a_launch=a_launch, a_land=a_land,
                    thrust=st["thrust"], torque=torque, brake=brake["thrust"], V_NE=V_NE, V_FE=V_FE, H_flap=H_flap, flap_normal=flap_normal)
    return df


def schrenk(p=None, n_pts=600) -> tuple:
    """Schrenk's spanwise lift per unit span for unit total lift [1/m] (NISUS's): (y [m], l(y))."""
    p = falco.resolve(p)
    b2 = p["span"] / 2000
    c0, c1 = p["root_chord"] / 1000, p["tip_chord"] / 1000
    y = np.linspace(0, b2, n_pts)
    c = c0 + (c1 - c0) * y / b2
    S = b2 * (c0 + c1)
    l_trap = c / S
    l_ell = 4 / (math.pi * 2 * b2) * np.sqrt(np.maximum(1 - (y / b2) ** 2, 0.0))
    return y, 0.5 * (l_trap + l_ell)


def moment_at(p, total_lift_N, y0_m) -> tuple:
    """The shear [N] and bending moment [N m] of one wing at station y0 under ``total_lift_N`` (both wings)."""
    y, l = schrenk(p)
    k = y >= y0_m
    return total_lift_N * np.trapezoid(l[k], y[k]), total_lift_N * np.trapezoid(l[k] * (y[k] - y0_m), y[k])


def spar_segment_forces(p=None, total_lift_N=1.0) -> list:
    """NISUS's: the lift on each of the 7 outer spar segments of one side, the lift beyond the spar's end added
    moment-equivalently to the last."""
    p = falco.resolve(p)
    L = falco.Falco.layout(p)
    y, l = schrenk(p)
    y_mm = y * 1000
    stations = [L["yc"]] + list(np.linspace(L["yc"], p["spar_half_length"], 8)[1:])
    out = []
    for y0, y1 in zip(stations[:-1], stations[1:]):
        k = (y_mm >= y0) & (y_mm <= y1)
        F = total_lift_N * np.trapezoid(l[k], y[k])
        yb = np.trapezoid(l[k] * y[k], y[k]) / np.trapezoid(l[k], y[k]) * 1000
        out.append({"y0": y0, "y1": y1, "F": F, "y_bar": yb})
    k = y_mm >= p["spar_half_length"]
    F_out = total_lift_N * np.trapezoid(l[k], y[k])
    y_out = np.trapezoid(l[k] * y[k], y[k]) / np.trapezoid(l[k], y[k]) * 1000
    last = out[-1]
    last["F_tip_raw"] = F_out
    last["F"] += F_out * (y_out - L["yc"]) / (last["y_bar"] - L["yc"])
    return out


def wing_hand(p=None, cases=None) -> pd.DataFrame:
    """The wing by hand at the ultimate load (NISUS's checks on FALCO): root bending and the spar's stress and
    deflection, the joiner at the panel joint (bending, bearing in the spar), the torsion at V_NE on the 2 mm balsa
    D-box, the foam and the spar, the spar–foam bond."""
    p = falco.resolve(p)
    L = falco.Falco.layout(p)
    c = cases if cases is not None else load_cases(p)
    W, n_ult = c.attrs["W"], c.attrs["n_ult"]
    total = n_ult * W
    yc = L["yc"] / 1000
    _, M_root = moment_at(p, total, yc)
    do, di = p["spar_od"], p["spar_id"]
    I = math.pi / 64 * (do ** 4 - di ** 4)
    sigma = M_root * 1000 * (do / 2) / I
    E = CARBON["E"]
    y, l = schrenk(p)
    ys = np.linspace(yc, p["spar_half_length"] / 1000, 200)
    M = np.array([total * np.trapezoid(l[y >= s] * (y[y >= s] - s), y[y >= s]) for s in ys])
    curv = M * 1000 / (E * I)
    slope = np.concatenate([[0], np.cumsum(0.5 * (curv[1:] + curv[:-1]) * np.diff(ys * 1000))])
    defl = float(np.trapezoid(slope, ys * 1000))
    # the joiner at the joint
    yj = p["wing_joint_y"] / 1000
    V_j, M_j = moment_at(p, total, yj)
    jo, ji = p["joiner_od"], p["joiner_id"]
    I_j = math.pi / 64 * (jo ** 4 - ji ** 4)
    sig_j = M_j * 1000 * (jo / 2) / I_j
    half = p["joiner_length"] / 2
    R_end = (M_j * 1000 + V_j * 15.0) / (half - 30.0)                  # the two-point bearing of the joiner's outer half in the panel's spar
    bear = (R_end + V_j) / (jo * 30.0)
    # torsion at V_NE
    q_ne = 0.5 * RHO0 * c.attrs["V_NE"] ** 2
    cmac = L["mac"] / 1000
    T = q_ne * L["S_ref"] / 2 * cmac * (abs(ff.AIRFOIL["cm_ac"]) + 0.15 / 3) * ULTIMATE
    c_root = p["root_chord"] / 1000
    t_c = p["thickness"] * c_root
    A_box = 0.30 * c_root * t_c * 0.70
    perim = 2 * 0.31 * c_root + t_c
    t_balsa = 2.0e-3
    G_balsa, G_foam, G_tube = 150e6, 8e6, 5e9
    GJ_box = G_balsa * 4 * A_box ** 2 * t_balsa / perim
    a_e, b_e = c_root / 2, t_c / 2
    GJ_foam = G_foam * math.pi * a_e ** 3 * b_e ** 3 / (a_e ** 2 + b_e ** 2) * (1 - FOAM_LIGHTENING_SHARE)
    J_tube = 2 * math.pi / 64 * ((do / 1000) ** 4 - (di / 1000) ** 4)
    GJ_tube = G_tube * J_tube
    GJ = GJ_box + GJ_foam + GJ_tube
    T_box, T_foam, T_tube = T * GJ_box / GJ, T * GJ_foam / GJ, T * GJ_tube / GJ
    tau_balsa = T_box / (2 * A_box * t_balsa) / 1e6
    tau_foam = 2 * T_foam / (math.pi * a_e * b_e ** 2) / 1e6
    tau_tube = T_tube * (do / 2000) / J_tube / 1e6
    twist = math.degrees(T / ULTIMATE * (L["b2"] / 1000) / (2 * GJ))
    lmax = total * float(l[np.searchsorted(y, yc)])
    tau_bond = lmax / (math.pi * do / 1000 * 2 / 3) / 1e6
    kd = CARBON["strength"] * KNOCKDOWN["carbon tube"]
    rows = {"root bending moment, ultimate [N m]": M_root, "spar stress, ultimate [MPa]": sigma, "spar margin (500 MPa x 0.8 knock-down)": kd / sigma - 1,
            "spar tip deflection at y = 1000 mm, ultimate [mm]": defl,
            "joint: shear / moment at the panel joint, ultimate [N / N m]": f"{V_j:.1f} / {M_j:.2f}",
            "joiner stress at the joint, ultimate [MPa]": sig_j, "joiner margin (500 MPa x 0.8)": kd / sig_j - 1,
            "joiner bearing on the panel's spar [MPa]": bear, "joiner bearing margin (tube crushing 60 MPa assumed x 0.8)": 60 * 0.8 / bear - 1,
            "torque per wing at V_NE, ultimate [N m]": T, "torsional stiffness shares (D-box / foam / spar)": f"{GJ_box / GJ:.2f} / {GJ_foam / GJ:.2f} / {GJ_tube / GJ:.2f}",
            "balsa D-box shear [MPa]": tau_balsa, "balsa D-box margin (shear 1.5 MPa assumed x 0.7)": 1.5 * 0.7 / tau_balsa - 1,
            "foam core shear [MPa]": tau_foam, "foam core margin (XPS shear 0.25 MPa x 0.7)": 0.25 * 0.7 / tau_foam - 1,
            "spar tube torsion shear [MPa]": tau_tube, "spar torsion margin (in-plane shear 40 MPa x 0.8)": 40 * 0.8 / tau_tube - 1,
            "wing tip twist at V_NE, limit [deg]": twist,
            "spar–foam bond shear at the root [MPa]": tau_bond, "spar–foam bond margin (XPS shear 0.25 MPa x 0.7)": 0.25 * 0.7 / tau_bond - 1}
    return pd.DataFrame({"value": rows})


def joints_hand(p=None, cases=None, battery_key=fs.DEFAULT_PACK) -> pd.DataFrame:
    """The joints by hand at the ultimate loads (NISUS's list on FALCO) plus the flap's horn and hinges in crow."""
    p = falco.resolve(p)
    L = falco.Falco.layout(p)
    c = cases if cases is not None else load_cases(p)
    a = c.attrs
    d = falco.Falco()
    fs_ = d.fitting_stations(p)
    rows = {}
    Ls = fs_["socket_length"]
    arm = L["x_ac_tail"] - fs_["x_te"]
    F_tail = ULTIMATE * max(a["tail"] / 2, a["F_stub"])
    R_exit = F_tail * (arm + Ls) / Ls
    tau = R_exit / (math.pi * p["boom_od"] * Ls / 2)
    rows["boom socket: length [mm]"] = Ls
    rows["boom socket: bearing reaction at the exit, ultimate [N]"] = R_exit
    rows["boom socket: epoxy shear [MPa]"] = tau
    rows["boom socket: margin (epoxy on PETG 5 MPa x 0.5)"] = 5.0 * KNOCKDOWN["adhesive"] / tau - 1
    # the boom tube at the socket's exit: the tail's vertical load and the fin's side load together (ultimate), as the FEA
    I_b = math.pi / 64 * (p["boom_od"] ** 4 - p["boom_id"] ** 4)
    F_fin = ULTIMATE * a["fin"]
    sig_b = math.hypot(F_tail, F_fin) * arm * (p["boom_od"] / 2) / I_b
    rows["boom tube: bending at the socket exit, tail + fin ultimate [MPa]"] = sig_b
    rows["boom tube: margin (500 MPa x 0.8)"] = CARBON["strength"] * KNOCKDOWN["carbon tube"] / sig_b - 1
    dx = fs_["x_rear"] - fs_["x_front"]
    R_rear = F_tail * (L["x_ac_tail"] - fs_["x_front"]) / dx
    R_front = F_tail * (L["x_ac_tail"] - fs_["x_rear"]) / dx
    rows["fitting: rear ring on the rear tube, ultimate [N]"] = R_rear
    bear = R_rear / (p["rear_spar_od"] * p["fitting_width"])
    rows["fitting: ring bearing on the tube [MPa]"] = bear
    rows["fitting: bearing margin (PETG 45 MPa x 0.7 x 0.5 for the layers)"] = 45 * 0.7 * 0.5 / bear - 1
    I_r = math.pi / 64 * (p["rear_spar_od"] ** 4 - p["rear_spar_id"] ** 4)
    arm_r = p["boom_y"] - L["yc"]
    sig_r = R_rear * arm_r * (p["rear_spar_od"] / 2) / I_r
    kd = CARBON["strength"] * KNOCKDOWN["carbon tube"]
    rows["rear tube: bending stress at the pod side, ultimate [MPa]"] = sig_r
    rows["rear tube: margin (500 MPa x 0.8)"] = kd / sig_r - 1
    I_s = math.pi / 64 * (p["spar_od"] ** 4 - p["spar_id"] ** 4)
    sig_s = R_front * arm_r * (p["spar_od"] / 2) / I_s
    rows["main spar: bending from the boom (tail case) [MPa]"] = sig_s
    rows["main spar: margin in the tail case (500 MPa x 0.8)"] = kd / sig_s - 1
    F_bolt = ULTIMATE * 0.5 * a["n_limit"] * a["W"] * 0.75 / 2
    rows["wing bolts (2 x M6 nylon PA66): tension each, inverted ultimate [N]"] = F_bolt
    rows["wing bolts: margin (M6 PA66 ~ 900 N x 0.7)"] = 900 * 0.7 / F_bolt - 1
    m_b = fs.pack(battery_key).mass_g / 1000
    F_strap = ULTIMATE * a["a_land"] * G * m_b / 2
    rows["battery straps (two): tension per leg, ultimate [N]"] = F_strap / 2
    rows["battery straps: margin (25 mm strap ~ 250 N x 0.5)"] = 250 * 0.5 / (F_strap / 2) - 1
    F_launch = ULTIMATE * a["a_launch"] * G * m_b
    rows["battery: launch inertia, ultimate [N] (front stop of the tray)"] = F_launch
    rows["battery front stop (PETG lip 80 x 14 x 2.5, bending) margin"] = (45 * 0.7 * 80 * 2.5 ** 2 / 6 / (F_launch * 7.0)) - 1
    m_mot = 0.355 + 0.046
    F_scr = ULTIMATE * (a["thrust"] + m_mot * a["a_land"] * G) / 4
    rows["motor screws (4 x M4, heat-set inserts): force each, ultimate [N]"] = F_scr
    rows["motor screws: margin (M4 insert pull-out 700 N x 0.7)"] = 700 * 0.7 / F_scr - 1
    H = ULTIMATE * a["H_flap"]
    rows["flap horn in crow at V_FE: hinge moment, ultimate [N cm]"] = H * 100
    rows["flap horn: servo margin (6 kg·cm servo)"] = 6.0 * G / 100 / H - 1
    F_h = ULTIMATE * a["flap_normal"] / 3
    rows["flap hinges (3 per flap, CA hinges): force each, ultimate [N]"] = F_h
    rows["flap hinges: margin (CA hinge ~ 60 N pull-out x 0.5)"] = 60 * 0.5 / F_h - 1
    return pd.DataFrame({"value": rows})


# ================================================================================================= Talos models
def export_step(part: str, workdir: Path, p=None) -> Path:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    # keyed by the parameters: a changed design never meshes a stale STEP
    key = hashlib.sha1(json.dumps(falco.resolve(p), sort_keys=True, default=str).encode()).hexdigest()[:10]
    path = workdir / f"falco_{part}_{key}.step"
    if not path.exists():
        falco.Falco().generate(**falco.overrides(p), part=part).export_step(str(path))
    return path


def spar_case(step, p=None, cases=None, element_size=3.0):
    """NISUS's spar case on FALCO: the right half of the main spar tube (straight), the Schrenk lift on its 7 outer
    segments (95 % on the wing), clamped in the saddle. The joiner's double wall at the joint is left out (the tube
    alone: conservative there)."""
    from vegeta import talos
    p = falco.resolve(p)
    L = falco.Falco.layout(p)
    c = cases if cases is not None else load_cases(p)
    segs = spar_segment_forces(p, c.attrs["n_ult"] * c.attrs["W"] * 0.95)
    e = p["spar_od"] / 2 * math.sin(math.radians(p["dihedral_deg"])) + 0.3      # the tube's end circles lean with the dihedral
    regions = [talos.SurfacesInBox("clamp", (-50.0, -2.0, -60.0, 250.0, L["yc"] + e, 100.0))]
    loads = []
    for i, sg in enumerate(segs):
        regions.append(talos.SurfacesInBox(f"seg{i}", (-50.0, sg["y0"] - e, -60.0, 250.0, sg["y1"] + e, 100.0)))
        loads.append(talos.Force(f"seg{i}", fz=sg["F"]))
    case = FEACase("spar_pullup", "spar_fea", f"spar tube (right half), ultimate n = {c.attrs['n_ult']:.2f}, Schrenk lift on 7 segments",
                   talos.StructuralModel(step, "mm-N-MPa", talos_material(CARBON), regions, [talos.FixedSupport("clamp")], loads,
                                         talos.MeshSettings(element_size=element_size, order=2), name="spar_pullup"), None, CARBON["strength"])
    case.exclude = lambda xyz: xyz[:, 1] < L["yc"] + p["spar_od"]
    return case


def joiner_case(step, p=None, cases=None, element_size=2.5):      # 1.5 mm meshes now and then held a sliver element (10 GPa spikes); 3.0 and 2.5 agree at 132 MPa
    """The spar joiner (``spar_joiner_fea``: five pieces): its inner half held in the centre section's spar, the outer
    panel's lift at the joint (shear V and moment M, ultimate) as two bearing forces on its outer half — up on the 30 mm
    piece at its end, down on the 30 mm piece at the joint (a two-point bearing: an idealisation of the slide fit)."""
    from vegeta import talos
    p = falco.resolve(p)
    c = cases if cases is not None else load_cases(p)
    yj, half = p["wing_joint_y"], p["joiner_length"] / 2
    V_j, M_j = moment_at(p, c.attrs["n_ult"] * c.attrs["W"], yj / 1000)
    y1, y2 = yj + 17.0, yj + half - 15.0                       # the two pieces' centres [mm]
    R2 = (M_j * 1000 + V_j * (y1 - yj)) / (y2 - y1)
    R1 = V_j - R2
    r = p["joiner_od"] / 2 + 6.0
    xs, zs = falco.Falco().tube_point(p, yj, p["spar_x_frac"])
    box = lambda ya, yb: (xs - r, ya, zs - r - 10, xs + r, yb, zs + r + 10)
    e = p["joiner_od"] / 2 * math.sin(math.radians(p["dihedral_deg"])) + 0.3
    regions = [talos.SurfacesInBox("inner", box(yj - half - 2.0, yj - 2.0 + e)), talos.SurfacesInBox("near", box(yj + 2.0 - e, yj + 32.0 + e)),
               talos.SurfacesInBox("end", box(yj + half - 30.0 - e, yj + half + 2.0))]
    case = FEACase("spar_joiner", "spar_joiner_fea", f"spar joiner: joint shear {V_j:.0f} N, moment {M_j:.1f} N m (ultimate) as bearing {R2:.0f} N / {R1:.0f} N",
                   talos.StructuralModel(step, "mm-N-MPa", talos_material(CARBON), regions, [talos.FixedSupport("inner")],
                                         [talos.Force("end", fz=R2), talos.Force("near", fz=R1)], talos.MeshSettings(element_size=element_size, order=2), name="spar_joiner"),
                   None, CARBON["strength"])
    case.exclude = lambda xyz: (xyz[:, 1] < yj + 2.0 + p["joiner_od"]) | (xyz[:, 1] > yj + half - 30.0 - 2.0)
    case.hand = {"V_j": V_j, "M_j": M_j, "R_end": R2, "R_near": R1}
    return case


def boom_case(step, p=None, cases=None, element_size=3.0):
    """NISUS's boom case on FALCO (the tail load or the stub touchdown up, the fin's side load)."""
    from vegeta import talos
    p = falco.resolve(p)
    L = falco.Falco.layout(p)
    c = cases if cases is not None else load_cases(p)
    a = c.attrs
    fs_ = falco.Falco().fitting_stations(p)
    y = p["boom_y"]
    r = p["boom_od"] / 2 + 0.5
    F_v = ULTIMATE * max(a["tail"] / 2, a["F_stub"])
    F_s = ULTIMATE * a["fin"]
    x_sl = L["boom_x1"] - p["tail_chord"]
    regions = [talos.SurfacesInBox("socket", (p["boom_x0"] - 0.5, y - r, p["boom_z"] - r, fs_["x_te"] + 0.3, y + r, p["boom_z"] + r)),
               talos.SurfacesInBox("sleeve", (x_sl - 0.3, y - r, p["boom_z"] - r, L["boom_x1"] + 0.5, y + r, p["boom_z"] + r))]
    case = FEACase("boom_tail", "boom_fea", f"boom tube: tail {F_v:.1f} N up and fin {F_s:.1f} N side (ultimate) on the tail sleeve; socket clamped",
                   talos.StructuralModel(step, "mm-N-MPa", talos_material(CARBON), regions, [talos.FixedSupport("socket")],
                                         [talos.Force("sleeve", fy=F_s, fz=F_v)], talos.MeshSettings(element_size=element_size, order=2), name="boom_tail"),
                   None, CARBON["strength"])
    # nominal: the tube between the clamped socket and the loaded sleeve (the load enters the 1 mm wall through the
    # sleeve's surface: local, not the tube's bending)
    case.exclude = lambda xyz: (xyz[:, 0] < fs_["x_te"] + p["boom_od"]) | (xyz[:, 0] > x_sl - p["boom_od"])
    return case


def boom_fitting_case(step, p=None, cases=None, element_size=2.5):
    """NISUS's boom root fitting case on FALCO."""
    from vegeta import talos
    p = falco.resolve(p)
    L = falco.Falco.layout(p)
    c = cases if cases is not None else load_cases(p)
    a = c.attrs
    y = p["boom_y"]
    fs_ = falco.Falco().fitting_stations(p)
    x_te = fs_["x_te"]
    Ls = x_te - p["boom_x0"]
    xm = 0.5 * (p["boom_x0"] + x_te)
    arm = L["x_ac_tail"] - x_te
    F_v = ULTIMATE * max(a["tail"] / 2, a["F_stub"])
    F_s = ULTIMATE * a["fin"]
    k_exit, k_front = (arm + Ls) / Ls, arm / Ls
    r = p["boom_od"] / 2 + 0.3
    w = p["fitting_width"] / 2 + 0.5
    rr = max(p["spar_od"], p["rear_spar_od"]) / 2 + p["fitting_ring_wall"] + 0.5
    regions = [talos.SurfacesInBox("ring_front", (fs_["x_front"] - rr, y - w, fs_["z_front"] - rr, fs_["x_front"] + rr, y + w, fs_["z_front"] + rr)),
               talos.SurfacesInBox("ring_rear", (fs_["x_rear"] - rr, y - w, fs_["z_rear"] - rr, fs_["x_rear"] + rr, y + w, fs_["z_rear"] + rr)),
               talos.SurfacesInBox("bore_front", (p["boom_x0"] - 1.5, y - r, p["boom_z"] - r, xm + 0.3, y + r, p["boom_z"] + r)),
               talos.SurfacesInBox("bore_rear", (xm - 0.3, y - r, p["boom_z"] - r, x_te + 1.5, y + r, p["boom_z"] + r))]
    loads = [talos.Force("bore_rear", fy=F_s * k_exit, fz=F_v * k_exit), talos.Force("bore_front", fy=-F_s * k_front, fz=-F_v * k_front)]
    out = FEACase("boom_fitting", "boom_fitting_fea", f"boom root fitting: boom reactions {F_v * k_exit:.0f} N / {F_v * k_front:.0f} N up, "
                                                       f"{F_s * k_exit:.0f} / {F_s * k_front:.0f} N side (ultimate); ring bores fixed",
                   talos.StructuralModel(step, "mm-N-MPa", talos_material(PETG), regions, [talos.FixedSupport("ring_front"), talos.FixedSupport("ring_rear")],
                                         loads, talos.MeshSettings(element_size=element_size, order=2), name="boom_fitting"),
                  BUILD_AXIS["boom_fitting"], PETG["strength"], PETG["strength_z"])
    r_f, r_r = p["spar_od"] / 2 + 0.1, p["rear_spar_od"] / 2 + 0.1
    # the ring bores are fixed (the carbon tubes): their edges are singular — the nominal stress leaves out 2 mm around them
    # and the socket bore's two ends, where the bearing loads are applied as uniform tractions on a half bore, likewise
    out.exclude = lambda xyz: ((np.hypot(xyz[:, 0] - fs_["x_front"], xyz[:, 2] - fs_["z_front"]) < r_f + 2.0)
                               | (np.hypot(xyz[:, 0] - fs_["x_rear"], xyz[:, 2] - fs_["z_rear"]) < r_r + 2.0)
                               | (xyz[:, 0] < p["boom_x0"] + 3.0) | (xyz[:, 0] > x_te - 3.0))
    return out


def motor_mount_case(step, p=None, cases=None, element_size=1.5, case="flight"):
    """NISUS's motor mount case on FALCO's mount (the 41xx bolt pattern), three load cases: ``flight`` (static thrust,
    torque, the motor's inertia at the limit n), ``brake`` (the propeller brake's reverse thrust at V_NE, the
    regeneration's reverse torque), ``landing`` (the inertia at the belly landing). Ultimate."""
    from vegeta import talos
    p = falco.resolve(p)
    L = falco.Falco.layout(p)
    c = cases if cases is not None else load_cases(p)
    a = c.attrs
    xe, zm = L["x_pod_end"], p["motor_z"]
    t = p["mount_thickness"]
    holes = falco.Falco().motor_holes(p)
    r_out = p["motor_diameter"] / 2 + 2.0
    info = talos.inspect_step(step, units="mm-N-MPa")
    skirt = [sf.tag for sf in info.surfaces if sf.kind.startswith("Cylinder") and xe - 15.0 < sf.centroid[0] < xe
             and abs(sf.area - 2 * math.pi * r_out * 15.0) < 0.05 * 2 * math.pi * r_out * 15.0]
    regions = [talos.Surfaces("skirt", skirt)]
    m_motor = 0.355 + 0.046 + 0.01
    if case == "flight":
        T, Q, n = a["thrust"], a["torque"], a["n_limit"]
    elif case == "brake":
        T, Q, n = a["brake"], -0.5 * a["torque"], a["n_limit"]
    else:
        T, Q, n = 0.0, 0.0, a["a_land"]
    F_in = ULTIMATE * m_motor * n * G
    th = math.radians(p["motor_downthrust_deg"])
    down = np.array([-math.sin(th), 0.0, -math.cos(th)])
    loads = []
    for name, (yy, zz) in holes.items():
        regions.append(talos.SurfacesInBox(name, (xe - 0.5, yy - 3.0, zm + zz - 3.0, xe + t + 0.5, yy + 3.0, zm + zz + 3.0)))
        r = math.hypot(yy, zz)
        tang = np.array([-zz, yy]) / r
        Ft = ULTIMATE * Q / (4 * r / 1000) * tang
        f = np.array([-ULTIMATE * T / 4, Ft[0], Ft[1]]) + F_in / 4 * down
        loads.append(talos.Force(name, fx=float(f[0]), fy=float(f[1]), fz=float(f[2])))
    desc = f"motor mount, {case}: thrust {ULTIMATE * T:.1f} N, torque {ULTIMATE * Q:.2f} N m, inertia {F_in:.1f} N (ultimate); skirt bonded"
    return FEACase(f"motor_mount_{case}", "motor_mount_fea", desc,
                   talos.StructuralModel(step, "mm-N-MPa", talos_material(PETG), regions, [talos.FixedSupport("skirt")], loads,
                                         talos.MeshSettings(element_size=element_size, order=2), name=f"motor_mount_{case}"),
                   BUILD_AXIS["motor_mount"], PETG["strength"], PETG["strength_z"])


def battery_tray_case(step, p=None, cases=None, element_size=2.5, battery_key=fs.DEFAULT_PACK):
    """The battery cradle with the 6S pack's belly-landing inertia on its floor and a 10 g cartwheel on a lip; the two
    ends screwed to the keel frame and the floor's underside resting on it (held: a 2.5 mm floor spanning the 310 mm bay
    between its ends alone fails under a 1.8 kg pack at 17 g — the first model's finding; the cradle is a liner on the
    frame, not a beam). What the case checks: the lips in the cartwheel, the slots. Ultimate."""
    from vegeta import talos
    p = falco.resolve(p)
    c = cases if cases is not None else load_cases(p)
    a = c.attrs
    x0, x1, w, h = falco.Falco.bays(p)["battery bay"]
    z = -p["pod_height"] + p["pod_wall"] + 12.0
    bw = p["battery_tray_width"]
    xa, xb = x0 + 5.0, x1 - 5.0
    regions = [talos.SurfacesOnPlane("end_front", "x", xa), talos.SurfacesOnPlane("end_rear", "x", xb),
               talos.SurfacesInBox("floor_top", (xa - 0.5, -bw / 2 - 0.3, z + 2.0, xb + 0.5, bw / 2 + 0.3, z + 3.0)),
               talos.SurfacesInBox("lip_inner", (xa - 0.5, bw / 2 - 0.5, z + 2.0, xb + 0.5, bw / 2 + 0.5, z + 14.5)),
               talos.SurfacesInBox("underside", (xa - 0.5, -bw / 2 - 3.0, z - 0.5, xb + 0.5, bw / 2 + 3.0, z + 0.5))]
    m = fs.pack(battery_key).mass_g / 1000
    F_down = ULTIMATE * m * a["a_land"] * G
    F_side = ULTIMATE * m * 10.0 * G
    return FEACase("battery_tray", "battery_tray", f"battery tray: {F_down:.0f} N down on the floor, {F_side:.0f} N on a lip (ultimate)",
                   talos.StructuralModel(step, "mm-N-MPa", talos_material(PETG), regions,
                                         [talos.FixedSupport("end_front"), talos.FixedSupport("end_rear"), talos.FixedSupport("underside")],
                                         [talos.Force("floor_top", fz=-F_down), talos.Force("lip_inner", fy=F_side)],
                                         talos.MeshSettings(element_size=element_size, order=2), name="battery_tray"),
                   BUILD_AXIS["battery_tray"], PETG["strength"], PETG["strength_z"])


def models(workdir, p=None, cases=None) -> list:
    """Every FEA case with its STEP exported into ``workdir`` (nothing is meshed or solved here)."""
    workdir = Path(workdir)
    c = cases if cases is not None else load_cases(p)
    steps = {part: export_step(part, workdir / "cad", p) for part in ("spar_fea", "spar_joiner_fea", "boom_fea", "boom_fitting_fea", "motor_mount_fea", "battery_tray")}
    return [spar_case(steps["spar_fea"], p, c), joiner_case(steps["spar_joiner_fea"], p, c), boom_case(steps["boom_fea"], p, c),
            boom_fitting_case(steps["boom_fitting_fea"], p, c),
            motor_mount_case(steps["motor_mount_fea"], p, c, case="flight"), motor_mount_case(steps["motor_mount_fea"], p, c, case="brake"),
            motor_mount_case(steps["motor_mount_fea"], p, c, case="landing"), battery_tray_case(steps["battery_tray"], p, c)]


NOT_ESTABLISHED = list(nst.NOT_ESTABLISHED) + [
    ("flutter at TAS", "the never-exceed speed is an EAS (the structure's dynamic pressure); flutter and divergence depend on the TAS, 25 % higher at "
                       "4000 m: the wing's torsion and the tail on the booms need Talos modes and a flutter estimate before the dives (todo)"),
    ("the crow flaps' separated loads", "the 55° flap's normal force and hinge moment are coefficients assumed for a separated plain flap; buffet on the "
                                        "tail behind the crow wing and behind the braked propeller is not computed"),
    ("cold", "the printed PETG and the epoxy at −10 °C (stiffer, more brittle), the foam's contraction against the carbon tube"),
]


__all__ = ["ULTIMATE", "KNOCKDOWN", "PETG", "CARBON", "BUILD_AXIS", "PRINT_SETTINGS", "talos_material", "design_mass_kg", "gust_n", "load_cases", "schrenk",
           "moment_at", "spar_segment_forces", "wing_hand", "joints_hand", "FEACase", "export_step", "spar_case", "joiner_case", "boom_case", "boom_fitting_case",
           "motor_mount_case", "battery_tray_case", "models", "across_layers", "nominal_stress", "summary_row", "NOT_ESTABLISHED"]
