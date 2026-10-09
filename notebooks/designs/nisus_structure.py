"""NISUS structure — the load cases from the flight conditions, the Talos models of the load-carrying parts, the
printed parts' anisotropy, the joints by hand and what linear-static FEA cannot tell (notebook 31).

Load cases (``load_cases``), each derived from a flight or ground condition with the design mass (the heaviest
variant + 10 %):

- **manoeuvre**: the limit load factor ``N_STRUCTURAL`` of ``nisus_flight`` (the autopilot's bank and pull-up limits);
- **gust**: Pratt's discrete gust at the cruise speed (7.5 m/s) and at the never-exceed speed (3.75 m/s) — the larger of
  the two with the manoeuvre is the **design limit load factor**; ultimate = 1.5 x limit;
- **tail**: the stabiliser at its section's maximum lift at V_NE (full elevator in a pull or a gust), the fins likewise
  (full rudder or a sideslip): the booms' bending and torsion;
- **launch**: the hand throw — 12 m/s in 0.30 s (the simulation's impulse) is 4.1 g along x, taken as 5 g;
- **landing**: a belly landing on the skid at 2.5 m/s sink, stopped by the TPU skid and the grass over 15 mm: 21 g
  vertical on the pod's contents; the tail-first touchdown on the fin stubs (the boom ends) at 1.5 m/s;
- **motor**: the static full-throttle thrust and torque (``nisus_systems.propulsion_map``) with the flight inertia, and
  the landing inertia of the motor and propeller.

Talos models (``models``): the carbon **spar** in the pull-up (its right half, straight; the lift of each span strip on
its own tube segment, a Schrenk distribution; the tube clamped in the pod's wing saddle), the carbon **boom** under the tail load, the carbon
**rear carry-through tube** (the boom fittings' couple), the printed **boom root fitting** (the boom's bearing reactions
on the front and rear halves of its socket, held by its clamp rings on the two tubes),
the printed **motor mount** (thrust, torque and inertia on its four bolt holes; bonded at its skirt) and the printed
**battery tray** (the pack's landing inertia on the floor and a lip). Isotropic linear elastic materials; the printed
parts' anisotropy is checked afterwards on the stress component across the layers (``across_layers``) against the
layer-adhesion strength; the carbon tube's directionality is its axial modulus in bending (the hoop and shear
properties are much lower: the clamp's crushing is a hand check, not FEA).

Units: the Talos models are mm-N-MPa (density t/mm³).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import nisus
import nisus_flight as nf
import nisus_systems as ns

G = 9.81
RHO = 1.225
ULTIMATE = 1.5
FOAM_LIGHTENING_SHARE = 0.25          # the share of the core's torsional stiffness lost to the lightening holes (= ns.FOAM_LIGHTENING, assumed)
KNOCKDOWN = {"PETG printed": 0.70, "carbon tube": 0.80, "adhesive": 0.50, "XPS foam 30": 0.70}   # manufacturing variability (assumed)

#: the materials for Talos (mm-N-MPa), with the strength the margins use
PETG = dict(name="PETG printed (in the layer plane)", E=2000.0, nu=0.38, rho=1.25e-9, strength=45.0, strength_z=27.0,
            source=ns.MATERIALS["PETG printed"]["source"])
CARBON = dict(name="carbon tube (axial)", E=ns.MATERIALS["carbon tube"]["E"], nu=0.30, rho=1.55e-9, strength=500.0,
              source=nisus.CARBON_TUBE["source"])
#: print orientation of each printed part: the build direction (the axis across the layers) in the nisus frame
BUILD_AXIS = {"boom_fitting": "y", "motor_mount": "x", "battery_tray": "z"}
PRINT_SETTINGS = {
    "boom_fitting": "on its side (the ring axes vertical, the socket flat on the bed); 4 perimeters 0.45 mm, 50 % gyroid, 0.2 mm layers: the web's bending (x-z plane) lies in the layers; across the layers (y) only the rings' width",
    "motor_mount": "plate face down (build axis along the thrust line); 100 % infill, 0.15 mm layers: the skirt-plate junction carries the landing moment across the layers",
    "battery_tray": "floor on the bed; 3 perimeters, 30 % infill: the lips bend across the layers",
    "tail_fitting": "sleeve axis vertical; 3 perimeters, 30 % gyroid",
    "skid": "TPU 95A, on its side; 3 perimeters, 15 % infill (it is meant to crush a little)",
}


def talos_material(m: dict):
    from vegeta import talos
    return talos.Material(m["name"], youngs_modulus=m["E"], poissons_ratio=m["nu"], density=m["rho"], yield_strength=m["strength"],
                          source=m["source"])


# ================================================================================================= loads
def design_mass_kg(p=None) -> float:
    """The heaviest variant (Zero) + 10 % growth (the operating limits' maximum take-off mass)."""
    return 1.10 * ns.cg_inertia(ns.mass_table("Zero", p=p))["mass_kg"]


def gust_n(mass_kg, V, U, p=None, rho=RHO) -> float:
    a = nf.aero(p)
    W = mass_kg * G
    mu = 2 * W / (rho * a["c_ref"] * a["CLa"] * G * a["S_ref"])
    Kg = 0.88 * mu / (5.3 + mu)
    return 1 + Kg * rho * V * U * a["CLa"] * a["S_ref"] / (2 * W)


def load_cases(p=None, *, V_C=16.0, V_NE=nf.V_NE, U_C=7.5, U_NE=3.75, pm: ns.PropulsionMap | None = None) -> dict:
    """The structural load cases (limit values; x ultimate in the FEA) with their derivation."""
    p = nisus.resolve(p)
    L = nisus.Nisus.layout(p)
    m = design_mass_kg(p)
    W = m * G
    n_man = nf.N_STRUCTURAL
    n_gc, n_gd = gust_n(m, V_C, U_C, p), gust_n(m, V_NE, U_NE, p)
    q_ne = 0.5 * RHO * V_NE ** 2
    S_h, S_v1 = L["S_h"], L["S_v_each"]
    n_lift_ne = q_ne * L["S_ref"] * nf.cl_max(p) / W
    n_limit = max(n_man, n_gc, n_gd)
    tail = q_ne * S_h * nf.TAIL_SECTION["cl_max_2d"]
    fin = q_ne * S_v1 * nf.TAIL_SECTION["cl_max_2d"]
    pm = pm or ns.propulsion_map()
    st = pm.at(0.0, 1.0)
    torque = st["shaft_power"] / (st["rpm"] * 2 * math.pi / 60)
    v_launch, t_launch = 12.0, 0.30
    a_launch = max(v_launch / t_launch / G, 5.0)
    sink, stroke = 2.5, 0.015
    a_land = sink ** 2 / (2 * stroke) / G + 1.0
    ci = ns.cg_inertia(ns.mass_table("Zero", p=p))
    l_stub = (L["boom_x1"] - 70.0) / 1000 - ci["x_cg_m"]
    m_eff = 1 / (1 / m + l_stub ** 2 / (ci["Iyy"] * 1.1))
    sink_t, stroke_t = 1.5, 0.010
    F_stub = m_eff * sink_t ** 2 / (2 * stroke_t)
    rows = {
        "design mass [kg]": (m, "Zero + 10 % (MTOM)"),
        "manoeuvre limit n": (n_man, "nisus_flight.N_STRUCTURAL"),
        f"gust n at V_C = {V_C:g} m/s, U = {U_C:g} m/s": (n_gc, "Pratt, alleviation from the mass ratio"),
        f"gust n at V_NE = {V_NE:g} m/s, U = {U_NE:g} m/s": (n_gd, "Pratt"),
        "lift-limited n at V_NE": (n_lift_ne, "CL_max at V_NE: the gust cannot exceed it"),
        "design limit n (wing)": (n_limit, "max of manoeuvre and gusts"),
        "ultimate n (wing)": (ULTIMATE * n_limit, f"x {ULTIMATE}"),
        "tail load at V_NE, limit [N]": (tail, f"q S_h cl_max_t ({nf.TAIL_SECTION['cl_max_2d']}), both booms"),
        "fin side load at V_NE, limit [N]": (fin, "q S_v1 cl_max_t, per fin"),
        "tail-first touchdown on a stub, limit [N]": (F_stub, f"{sink_t} m/s, {stroke_t * 1000:.0f} mm stroke, effective mass {m_eff:.3f} kg"),
        "launch acceleration [g]": (a_launch, f"{v_launch:g} m/s in {t_launch:g} s, at least 5 g"),
        "belly landing [g]": (a_land, f"{sink} m/s sink, {stroke * 1000:.0f} mm stroke (skid + grass) + 1 g"),
        "static thrust [N]": (st["thrust"], "propulsion map, full throttle, static"),
        "motor torque [N m]": (torque, "shaft power / ω at full throttle"),
    }
    df = pd.DataFrame({k: {"value": v[0], "basis": v[1]} for k, v in rows.items()}).T
    df.attrs.update(mass=m, W=W, n_limit=n_limit, n_ult=ULTIMATE * n_limit, tail=tail, fin=fin, F_stub=F_stub, a_launch=a_launch, a_land=a_land,
                    thrust=st["thrust"], torque=torque, V_NE=V_NE)
    return df


def schrenk(p=None, n_pts=400) -> tuple:
    """Schrenk's spanwise lift per unit span for unit total lift [1/m]: the mean of the trapezoidal planform and the
    ellipse of the same area; (y [m], l(y))."""
    p = nisus.resolve(p)
    b2 = p["span"] / 2000
    c0, c1 = p["root_chord"] / 1000, p["tip_chord"] / 1000
    y = np.linspace(0, b2, n_pts)
    c = c0 + (c1 - c0) * y / b2
    S = b2 * (c0 + c1)                        # both halves
    l_trap = c / S
    l_ell = 4 / (math.pi * 2 * b2) * np.sqrt(np.maximum(1 - (y / b2) ** 2, 0.0))
    return y, 0.5 * (l_trap + l_ell)


def spar_segment_forces(p=None, total_lift_N=1.0) -> list:
    """The lift on each spar segment of one side outside the centre piece [N] at its y range [mm] for the aircraft's
    ``total_lift_N`` (both wings; ``schrenk`` is normalised to it), the outboard lift
    beyond the spar's end added to the last segment so that the root bending moment is right (moment-equivalent: the
    shear at the last segment is overstated by the same factor — conservative there)."""
    p = nisus.resolve(p)
    L = nisus.Nisus.layout(p)
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
    """The wing by hand at the ultimate load: root bending moment (Schrenk) and the tube's stress and tip deflection
    (cantilever from the saddle), the torsion at V_NE (Cm0 and full aileron) carried by the foam core and its covering
    (shared by stiffness: the 1.5 mm balsa D-box, the foam core, the spar tube), the spar–foam bond shear."""
    p = nisus.resolve(p)
    L = nisus.Nisus.layout(p)
    c = cases if cases is not None else load_cases(p)
    W, n_ult = c.attrs["W"], c.attrs["n_ult"]
    y, l = schrenk(p)                                                 # per unit TOTAL lift (both halves)
    yc = L["yc"] / 1000
    total = n_ult * W
    k = y >= yc
    M_root = total * np.trapezoid(l[k] * (y[k] - yc), y[k])          # about the saddle edge
    do, di = p["spar_od"], p["spar_id"]
    I = math.pi / 64 * (do ** 4 - di ** 4)
    sigma = M_root * 1000 * (do / 2) / I
    # deflection: the tube carries the bending to its end; integrate M/EI twice (the rods beyond 500 mm not counted)
    E = CARBON["E"]
    ys = np.linspace(yc, p["spar_half_length"] / 1000, 200)
    M = np.array([total * np.trapezoid(l[y >= s] * (y[y >= s] - s), y[y >= s]) for s in ys])
    curv = M * 1000 / (E * I)                                         # 1/mm
    slope = np.concatenate([[0], np.cumsum(0.5 * (curv[1:] + curv[:-1]) * np.diff(ys * 1000))])
    defl = float(np.trapezoid(slope, ys * 1000))
    # torsion at V_NE: q (S/2) c (|Cm_ac| + a third of the aileron's ΔCm, the usual allowance at the dive speed) per wing,
    # ultimate; shared by the torsional stiffness of the closed balsa D-box (Bredt–Batho), the solid foam core and the spar
    q_ne = 0.5 * RHO * c.attrs["V_NE"] ** 2
    cmac = L["mac"] / 1000
    T = q_ne * L["S_ref"] / 2 * cmac * (abs(nf.AIRFOIL["cm_ac"]) + 0.15 / 3) * ULTIMATE
    c_root = p["root_chord"] / 1000
    t_c = p["thickness"] * c_root
    A_box = 0.30 * c_root * t_c * 0.70                                # the D-box's enclosed area (LE to the spar)
    perim = 2 * 0.31 * c_root + t_c                                     # its two skins and the spar web
    t_balsa = 1.5e-3
    G_balsa, G_foam, G_tube = 150e6, 8e6, 5e9                            # Pa (assumed: balsa G_LT, XPS, a pultruded tube in shear)
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
    twist = math.degrees(T / ULTIMATE * (L["b2"] / 1000) / (2 * GJ))      # limit torque, distributed: half the root torque's twist
    # spar-foam bond: the lift per length on the outer segment into the bond (perimeter x 2/3 effective)
    lmax = total * float(l[np.searchsorted(y, yc)])
    tau_bond = lmax / (math.pi * do / 1000 * 2 / 3) / 1e6
    rows = {"root bending moment, ultimate [N m]": M_root, "spar stress, ultimate [MPa]": sigma,
            "spar margin (500 MPa x 0.8 knock-down)": CARBON["strength"] * KNOCKDOWN["carbon tube"] / sigma - 1,
            "spar tip deflection at y = 500 mm, ultimate [mm]": defl,
            "spar tip deflection, limit, if E = 35 GPa (Easy Composites datasheet) [mm]": defl / ULTIMATE * E / 35e3,
            "torque per wing at V_NE, ultimate [N m]": T, "torsional stiffness shares (D-box / foam / spar)": f"{GJ_box / GJ:.2f} / {GJ_foam / GJ:.2f} / {GJ_tube / GJ:.2f}",
            "balsa D-box shear [MPa]": tau_balsa, "balsa D-box margin (shear 1.5 MPa assumed x 0.7)": 1.5 * 0.7 / tau_balsa - 1,
            "foam core shear [MPa]": tau_foam, "foam core margin (XPS shear 0.25 MPa x 0.7)": 0.25 * 0.7 / tau_foam - 1,
            "spar tube torsion shear [MPa]": tau_tube, "spar torsion margin (in-plane shear 40 MPa x 0.8)": 40 * 0.8 / tau_tube - 1,
            "wing tip twist at V_NE, limit [deg]": twist,
            "spar–foam bond shear at the root [MPa]": tau_bond, "spar–foam bond margin (XPS shear 0.25 MPa x 0.7)": 0.25 * 0.7 / tau_bond - 1}
    return pd.DataFrame({"value": rows})


def joints_hand(p=None, cases=None) -> pd.DataFrame:
    """The joints by hand at the ultimate loads: the boom bonded in its socket, the fitting bonded under the wing, the
    two nylon wing bolts, the tail sleeve, the battery strap, the motor's M3 screws."""
    p = nisus.resolve(p)
    L = nisus.Nisus.layout(p)
    c = cases if cases is not None else load_cases(p)
    a = c.attrs
    rows = {}
    # boom in its socket: the socket length from boom_x0 to the wing TE at the boom station
    f = (p["boom_y"] - L["yc"]) / (L["b2"] - L["yc"])
    x_te = L["le_sweep_tip"] * f + p["root_chord"] + (p["tip_chord"] - p["root_chord"]) * f
    Ls = x_te - p["boom_x0"]
    arm = L["x_ac_tail"] - x_te
    F_tail = ULTIMATE * max(a["tail"] / 2, a["F_stub"])
    R_exit = F_tail * (arm + Ls) / Ls
    bond_area = math.pi * p["boom_od"] * Ls
    tau = (R_exit) / (bond_area / 2)                                          # half the bond carries the bearing reaction (assumption)
    rows["boom socket: length [mm]"] = Ls
    rows["boom socket: bearing reaction at the exit, ultimate [N]"] = R_exit
    rows["boom socket: epoxy shear [MPa]"] = tau
    rows["boom socket: margin (epoxy on PETG 5 MPa x 0.5)"] = 5.0 * KNOCKDOWN["adhesive"] / tau - 1
    # the boom fitting's two clamp rings: the boom's shear F at the tail's ac reaches the main spar and the rear tube as a couple
    fs = nisus.Nisus().fitting_stations(p)
    dx = fs["x_rear"] - fs["x_front"]
    R_rear = F_tail * (L["x_ac_tail"] - fs["x_front"]) / dx
    R_front = F_tail * (L["x_ac_tail"] - fs["x_rear"]) / dx
    rows["fitting: ring spacing [mm]"] = dx
    rows["fitting: rear ring on the rear tube, ultimate [N]"] = R_rear
    rows["fitting: front ring on the main spar (opposite), ultimate [N]"] = R_front
    bear = R_rear / (p["rear_spar_od"] * p["fitting_width"])
    rows["fitting: ring bearing on the tube [MPa]"] = bear
    rows["fitting: bearing margin (PETG 45 MPa x 0.7 x 0.5 for the layers)"] = 45 * 0.7 * 0.5 / bear - 1
    I_r = math.pi / 64 * (p["rear_spar_od"] ** 4 - p["rear_spar_id"] ** 4)
    arm_r = p["boom_y"] - L["yc"]
    sig_r = R_rear * arm_r * (p["rear_spar_od"] / 2) / I_r
    rows["rear tube: bending stress at the pod side, ultimate [MPa]"] = sig_r
    rows["rear tube: margin (500 MPa x 0.8)"] = CARBON["strength"] * KNOCKDOWN["carbon tube"] / sig_r - 1
    I_s = math.pi / 64 * (p["spar_od"] ** 4 - p["spar_id"] ** 4)
    sig_s = R_front * arm_r * (p["spar_od"] / 2) / I_s
    rows["main spar: bending from the boom (tail case, 1 g lift ignored) [MPa]"] = sig_s
    rows["main spar: margin in the tail case (500 MPa x 0.8)"] = CARBON["strength"] * KNOCKDOWN["carbon tube"] / sig_s - 1
    # wing to pod: two M5 nylon bolts carry the inverted load (negative n = -0.5 n_limit) in tension
    F_bolt = ULTIMATE * 0.5 * a["n_limit"] * a["W"] * 0.75 / 2
    rows["wing bolts (2 x M5 nylon PA66): tension each, inverted ultimate [N]"] = F_bolt
    rows["wing bolts: margin (M5 PA66 ~ 600 N x 0.7)"] = 600 * 0.7 / F_bolt - 1
    # battery strap: 21 g on the pack, the strap in tension around it (two legs) + velcro shear in the 5 g launch
    m_b = 0.19
    F_strap = ULTIMATE * a["a_land"] * G * m_b / 2
    rows["battery strap: tension per leg, ultimate [N]"] = F_strap
    rows["battery strap: margin (20 mm hook-and-loop strap ~ 150 N x 0.5)"] = 150 * 0.5 / F_strap - 1
    F_launch = ULTIMATE * a["a_launch"] * G * m_b
    rows["battery: launch inertia, ultimate [N] (front stop of the tray + velcro)"] = F_launch
    rows["battery velcro shear (2 x 20x60 mm, 0.05 MPa) margin"] = (2 * 20 * 60 * 0.05) * 0.5 / F_launch - 1
    # motor screws: 4 x M3 into the mount (heat-set inserts): thrust + landing inertia
    F_scr = ULTIMATE * (a["thrust"] + 0.085 * a["a_land"] * G) / 4
    rows["motor screws (4 x M3, heat-set inserts): force each, ultimate [N]"] = F_scr
    rows["motor screws: margin (insert pull-out 400 N x 0.7)"] = 400 * 0.7 / F_scr - 1
    return pd.DataFrame({"value": rows})


# ================================================================================================= Talos models
@dataclass
class FEACase:
    name: str
    part: str
    description: str
    model: object
    build_axis: str | None = None
    strength: float = 0.0
    strength_z: float | None = None
    exclude: object = None           # coords (N, 3) -> bool mask of nodes near a support (singular corner) left out of the "nominal" stress
    hand: dict | None = None


def export_step(part: str, workdir: Path, p=None) -> Path:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / f"nisus_{part}.step"
    if not path.exists():
        nisus.Nisus().generate(**nisus.overrides(p), part=part).export_step(str(path))
    return path


def _box(c, h):
    return (c[0] - h[0], c[1] - h[1], c[2] - h[2], c[0] + h[0], c[1] + h[1], c[2] + h[2])


def spar_case(step, p=None, cases=None, element_size=2.0):
    """The right half of the spar tube (``spar_fea``, straight) in the ultimate pull-up: each outer segment carries its
    strip's lift (``spar_segment_forces``, +z; 95 % of the lift on the wing, the rest on the tail and the pod), the
    centre piece clamped in the saddle (the pod's half width). Symmetric load: the half model with the clamp is exact."""
    from vegeta import talos
    p = nisus.resolve(p)
    L = nisus.Nisus.layout(p)
    c = cases if cases is not None else load_cases(p)
    segs = spar_segment_forces(p, c.attrs["n_ult"] * c.attrs["W"] * 0.95)
    regions = [talos.SurfacesInBox("clamp", (-50.0, -0.5, -50.0, 200.0, L["yc"] + 0.3, 80.0))]
    loads = []
    for i, sg in enumerate(segs):
        regions.append(talos.SurfacesInBox(f"seg{i}", (-50.0, sg["y0"] - 0.3, -50.0, 200.0, sg["y1"] + 0.3, 80.0)))
        loads.append(talos.Force(f"seg{i}", fz=sg["F"]))
    case = FEACase("spar_pullup", "spar_fea", f"spar tube (right half), ultimate n = {c.attrs['n_ult']:.2f}, Schrenk lift on 7 segments",
                   talos.StructuralModel(step, "mm-N-MPa", talos_material(CARBON), regions, [talos.FixedSupport("clamp")], loads,
                                         talos.MeshSettings(element_size=element_size, order=2), name="spar_pullup"), None, CARBON["strength"])
    case.exclude = lambda xyz: xyz[:, 1] < L["yc"] + p["spar_od"]
    return case


def boom_case(step, p=None, cases=None, element_size=2.0):
    """One boom tube (``boom_fea``: three pieces with their own faces): the piece in the socket clamped (its bond), the
    tail's share of the ultimate load (or the stub touchdown, whichever is larger) up and the fin's side force on the
    piece under the tail sleeve."""
    from vegeta import talos
    p = nisus.resolve(p)
    L = nisus.Nisus.layout(p)
    c = cases if cases is not None else load_cases(p)
    a = c.attrs
    fs = nisus.Nisus().fitting_stations(p)
    y = p["boom_y"]
    r = p["boom_od"] / 2 + 0.5
    F_v = ULTIMATE * max(a["tail"] / 2, a["F_stub"])
    F_s = ULTIMATE * a["fin"]
    x_sl = L["boom_x1"] - p["tail_chord"]
    regions = [talos.SurfacesInBox("socket", (p["boom_x0"] - 0.5, y - r, p["boom_z"] - r, fs["x_te"] + 0.3, y + r, p["boom_z"] + r)),
               talos.SurfacesInBox("sleeve", (x_sl - 0.3, y - r, p["boom_z"] - r, L["boom_x1"] + 0.5, y + r, p["boom_z"] + r))]
    case = FEACase("boom_tail", "boom_fea", f"boom tube: tail {F_v:.1f} N up and fin {F_s:.1f} N side (ultimate) on the tail sleeve; socket clamped",
                   talos.StructuralModel(step, "mm-N-MPa", talos_material(CARBON), regions, [talos.FixedSupport("socket")],
                                         [talos.Force("sleeve", fy=F_s, fz=F_v)], talos.MeshSettings(element_size=element_size, order=2), name="boom_tail"),
                   None, CARBON["strength"])
    case.exclude = lambda xyz: xyz[:, 0] < fs["x_te"] + p["boom_od"]
    case.hand = {"cantilever [mm]": 0.5 * (x_sl + L["boom_x1"]) - fs["x_te"], "F_v": F_v, "F_s": F_s}
    return case


def boom_fitting_case(step, p=None, cases=None, element_size=2.0):
    """The printed boom root fitting: its two clamp rings held by the tubes (the ring bores fixed: carbon tubes are
    ~60 x stiffer than PETG), the boom's bearing reactions on the socket bore's rear half (R_exit up and sideways) and
    front half (−R_front): a cantilevered tube in a socket of length Ls with the tail force at ``arm`` behind the exit
    gives R_exit = F (arm + Ls)/Ls and R_front = F arm/Ls (two-point bearing, an idealisation of the epoxy-filled
    socket). The boom's torsion from the fin is not applied here (``nisus.boom_check``)."""
    from vegeta import talos
    p = nisus.resolve(p)
    L = nisus.Nisus.layout(p)
    c = cases if cases is not None else load_cases(p)
    a = c.attrs
    y = p["boom_y"]
    fs = nisus.Nisus().fitting_stations(p)
    x_te = fs["x_te"]
    Ls = x_te - p["boom_x0"]
    xm = 0.5 * (p["boom_x0"] + x_te)
    arm = L["x_ac_tail"] - x_te
    F_v = ULTIMATE * max(a["tail"] / 2, a["F_stub"])
    F_s = ULTIMATE * a["fin"]
    k_exit, k_front = (arm + Ls) / Ls, arm / Ls
    r = p["boom_od"] / 2 + 0.3
    w = p["fitting_width"] / 2 + 0.5
    regions = [talos.SurfacesInBox("ring_front", (fs["x_front"] - 5.5, y - w, fs["z_front"] - 5.5, fs["x_front"] + 5.5, y + w, fs["z_front"] + 5.5)),
               talos.SurfacesInBox("ring_rear", (fs["x_rear"] - 5.5, y - w, fs["z_rear"] - 5.5, fs["x_rear"] + 5.5, y + w, fs["z_rear"] + 5.5)),
               talos.SurfacesInBox("bore_front", (p["boom_x0"] - 1.5, y - r, p["boom_z"] - r, xm + 0.3, y + r, p["boom_z"] + r)),
               talos.SurfacesInBox("bore_rear", (xm - 0.3, y - r, p["boom_z"] - r, x_te + 1.5, y + r, p["boom_z"] + r))]
    loads = [talos.Force("bore_rear", fy=F_s * k_exit, fz=F_v * k_exit), talos.Force("bore_front", fy=-F_s * k_front, fz=-F_v * k_front)]
    return FEACase("boom_fitting", "boom_fitting_fea", f"boom root fitting: boom reactions {F_v * k_exit:.0f} N / {F_v * k_front:.0f} N up, "
                                                       f"{F_s * k_exit:.0f} / {F_s * k_front:.0f} N side (ultimate); ring bores fixed",
                   talos.StructuralModel(step, "mm-N-MPa", talos_material(PETG), regions, [talos.FixedSupport("ring_front"), talos.FixedSupport("ring_rear")],
                                         loads, talos.MeshSettings(element_size=element_size, order=2), name="boom_fitting"),
                   BUILD_AXIS["boom_fitting"], PETG["strength"], PETG["strength_z"])


def rear_spar_case(step, p=None, cases=None, element_size=1.5):
    """The right half of the rear carry-through tube (``rear_spar_fea``) in the tail case: the fitting's rear ring
    pushes it on its outer segment (the ultimate R_rear of ``joints_hand``'s couple), clamped in the pod's saddle."""
    from vegeta import talos
    p = nisus.resolve(p)
    L = nisus.Nisus.layout(p)
    c = cases if cases is not None else load_cases(p)
    a = c.attrs
    F = ULTIMATE * max(a["tail"] / 2, a["F_stub"])
    fs = nisus.Nisus().fitting_stations(p)
    R_rear = F * (L["x_ac_tail"] - fs["x_front"]) / (fs["x_rear"] - fs["x_front"])
    st = [L["yc"]] + list(np.linspace(L["yc"], p["rear_spar_half_length"], 3)[1:])
    regions = [talos.SurfacesInBox("clamp", (fs["x_rear"] - 20, -0.5, -40, fs["x_rear"] + 20, L["yc"] + 0.3, 40)),
               talos.SurfacesInBox("ring", (fs["x_rear"] - 20, st[-2] - 0.3, -40, fs["x_rear"] + 20, st[-1] + 0.3, 40))]
    case = FEACase("rear_spar_tail", "rear_spar_fea", f"rear tube (right half): {R_rear:.0f} N at the fitting (tail case, ultimate)",
                   talos.StructuralModel(step, "mm-N-MPa", talos_material(CARBON), regions, [talos.FixedSupport("clamp")], [talos.Force("ring", fz=R_rear)],
                                         talos.MeshSettings(element_size=element_size, order=2), name="rear_spar_tail"), None, CARBON["strength"])
    case.exclude = lambda xyz: xyz[:, 1] < L["yc"] + p["rear_spar_od"]
    case.hand = {"R_rear": R_rear}
    return case


def motor_mount_case(step, p=None, cases=None, element_size=1.2, case="flight"):
    """The printed motor mount (``motor_mount_fea``: untilted, its axis along x; the loads are rotated into its frame
    by the down-thrust), bonded on its skirt's outer face (fixed). ``flight``: the static thrust forward along the
    axis, the motor torque as tangential forces on the four bolt holes, the motor and propeller's inertia at the design
    limit n (down in the aircraft frame); ``landing``: their inertia at the belly-landing deceleration, the thrust off.
    Ultimate (x 1.5)."""
    from vegeta import talos
    p = nisus.resolve(p)
    L = nisus.Nisus.layout(p)
    c = cases if cases is not None else load_cases(p)
    a = c.attrs
    xe, zm = L["x_pod_end"], p["motor_z"]
    t = p["mount_thickness"]
    holes = {"h1": (8.0, 0.0), "h2": (-8.0, 0.0), "h3": (0.0, 9.5), "h4": (0.0, -9.5)}
    r_out = p["motor_diameter"] / 2 + 2.0
    info = talos.inspect_step(step, units="mm-N-MPa")
    skirt = [sf.tag for sf in info.surfaces if sf.kind.startswith("Cylinder") and xe - 12.0 < sf.centroid[0] < xe
             and abs(sf.area - 2 * math.pi * r_out * 12.0) < 0.05 * 2 * math.pi * r_out * 12.0]
    regions = [talos.Surfaces("skirt", skirt)]
    m_motor = 0.069 + 0.012 + 0.004
    if case == "flight":
        T, Q, n = a["thrust"], a["torque"], a["n_limit"]
    else:
        T, Q, n = 0.0, 0.0, a["a_land"]
    F_in = ULTIMATE * m_motor * n * G
    th = math.radians(p["motor_downthrust_deg"])
    # the aircraft's "down" (−z) in the mount's frame (the mount is turned nose-down by th about y): (−sin th, 0, −cos th)
    down = np.array([-math.sin(th), 0.0, -math.cos(th)])
    loads = []
    for name, (yy, zz) in holes.items():
        regions.append(talos.SurfacesInBox(name, (xe - 0.5, yy - 2.0, zm + zz - 2.0, xe + t + 0.5, yy + 2.0, zm + zz + 2.0)))
        r = math.hypot(yy, zz)
        tang = np.array([-zz, yy]) / r
        Ft = ULTIMATE * Q / (4 * r / 1000) * tang
        f = np.array([-ULTIMATE * T / 4, Ft[0], Ft[1]]) + F_in / 4 * down
        loads.append(talos.Force(name, fx=float(f[0]), fy=float(f[1]), fz=float(f[2])))
    desc = f"motor mount, {case}: thrust {ULTIMATE * T:.1f} N, torque {ULTIMATE * Q:.2f} N m, inertia {F_in:.1f} N (ultimate); skirt bonded"
    out = FEACase(f"motor_mount_{case}", "motor_mount_fea", desc,
                  talos.StructuralModel(step, "mm-N-MPa", talos_material(PETG), regions, [talos.FixedSupport("skirt")], loads,
                                        talos.MeshSettings(element_size=element_size, order=2), name=f"motor_mount_{case}"),
                  BUILD_AXIS["motor_mount"], PETG["strength"], PETG["strength_z"])
    return out


def battery_tray_case(step, p=None, cases=None, element_size=1.5, battery_g=190.0):
    """The printed battery tray: screwed to the pod's keel frame at its two ends (the end faces fixed), the pack's
    belly-landing inertia on the floor (down) and a lateral 10 g (a cartwheel) on one lip. Ultimate."""
    from vegeta import talos
    p = nisus.resolve(p)
    c = cases if cases is not None else load_cases(p)
    a = c.attrs
    x0, x1, w, h = nisus.Nisus.bays(p)["battery bay"]
    z = -p["pod_height"] + p["pod_wall"] + 12.0
    xa, xb = x0 + 5.0, x1 - 5.0
    regions = [talos.SurfacesOnPlane("end_front", "x", xa), talos.SurfacesOnPlane("end_rear", "x", xb),
               talos.SurfacesInBox("floor_top", (xa - 0.5, -21.5, z + 1.5, xb + 0.5, 21.5, z + 2.5)),
               talos.SurfacesInBox("lip_inner", (xa - 0.5, 20.5, z + 1.5, xb + 0.5, 21.5, z + 10.5))]
    m = battery_g / 1000
    F_down = ULTIMATE * m * a["a_land"] * G
    F_side = ULTIMATE * m * 10.0 * G
    return FEACase("battery_tray", "battery_tray", f"battery tray: {F_down:.0f} N down on the floor, {F_side:.0f} N on a lip (ultimate)",
                   talos.StructuralModel(step, "mm-N-MPa", talos_material(PETG), regions,
                                         [talos.FixedSupport("end_front"), talos.FixedSupport("end_rear")],
                                         [talos.Force("floor_top", fz=-F_down), talos.Force("lip_inner", fy=F_side)],
                                         talos.MeshSettings(element_size=element_size, order=2), name="battery_tray"),
                   BUILD_AXIS["battery_tray"], PETG["strength"], PETG["strength_z"])


def models(workdir, p=None, cases=None) -> list:
    """Every FEA case with its STEP exported into ``workdir`` (nothing is meshed or solved here)."""
    workdir = Path(workdir)
    c = cases if cases is not None else load_cases(p)
    steps = {part: export_step(part, workdir / "cad", p) for part in ("spar_fea", "rear_spar_fea", "boom_fea", "boom_fitting_fea", "motor_mount_fea", "battery_tray")}
    return [spar_case(steps["spar_fea"], p, c), rear_spar_case(steps["rear_spar_fea"], p, c), boom_case(steps["boom_fea"], p, c),
            boom_fitting_case(steps["boom_fitting_fea"], p, c),
            motor_mount_case(steps["motor_mount_fea"], p, c, case="flight"), motor_mount_case(steps["motor_mount_fea"], p, c, case="landing"),
            battery_tray_case(steps["battery_tray"], p, c)]


def across_layers(result, axis: str | None) -> float:
    """The largest tensile normal stress across the printed layers (the build axis) [MPa] from the .frd field."""
    if axis is None:
        return float("nan")
    from vegeta import talos
    f = talos.read_frd(result.artifacts["frd"])
    i = "xyz".index(axis)
    return float(np.max(f.stress[:, i]))


def nominal_stress(case: FEACase, result) -> tuple:
    """(max von Mises away from the support's singular edge — nodes the case's ``exclude`` mask leaves — or nan when the
    case has none; the 99.5th percentile of the nodal von Mises)."""
    from vegeta import talos
    f = talos.read_frd(result.artifacts["frd"])
    s = f.stress
    vm = np.sqrt(0.5 * ((s[:, 0] - s[:, 1]) ** 2 + (s[:, 1] - s[:, 2]) ** 2 + (s[:, 2] - s[:, 0]) ** 2) + 3 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2))
    keep = ~case.exclude(f.coords) if case.exclude is not None else np.zeros(len(vm), bool)
    nominal = float(np.max(vm[keep])) if keep.any() else float("nan")       # nan: no exclusion zone, or no node outside it
    return nominal, float(np.percentile(vm, 99.5))


def summary_row(case: FEACase, result) -> dict:
    """Stress, deflection and margins of one solved case: von Mises against the in-plane strength x knock-down, and the
    across-layer tension against the layer-adhesion strength x knock-down for printed parts."""
    if result is None or not getattr(result, "ok", False):
        return {"case": case.name, "status": "NOT RUN" if result is None else ("NOT RUN" if "NOT RUN" in " ".join(getattr(result, "messages", [])) else "FAILED"),
                "description": case.description}
    m = result.metrics
    vm, disp = m.get("max_von_mises"), m.get("max_displacement")
    nominal, p995 = nominal_stress(case, result)
    kd = KNOCKDOWN["PETG printed"] if case.strength_z else KNOCKDOWN["carbon tube"]
    gov = nominal if np.isfinite(nominal) else vm
    row = {"case": case.name, "status": "solved", "description": case.description, "peak von Mises [MPa]": vm,
           "nominal von Mises [MPa]": nominal, "99.5th percentile von Mises [MPa]": p995, "max displacement [mm]": disp,
           "allowable [MPa]": case.strength * kd, "margin (governing)": case.strength * kd / gov - 1 if gov else np.nan,
           "governing stress": "nominal (≥ 1 diameter from the support)" if np.isfinite(nominal) else "peak",
           "reaction [N]": np.linalg.norm(m.get("reaction_total", [np.nan] * 3)) if m.get("reaction_total") is not None else np.nan,
           "applied [N]": np.linalg.norm(m.get("applied_force_total", [np.nan] * 3)) if m.get("applied_force_total") is not None else np.nan}
    if case.strength_z:
        sz = across_layers(result, case.build_axis)
        row.update({"build axis": case.build_axis, "across layers [MPa]": sz, "margin (layers)": case.strength_z * kd / sz - 1 if sz > 0 else np.inf})
    return row


NOT_ESTABLISHED = [
    ("buckling", "local buckling of the 10/8 mm tubes (D/t = 10: a thick tube, the material fails first by hand); the foam core's "
                 "and the covering's wrinkling on the wing's compression side; the battery tray's lips — not computed (no eigenvalue buckling in Talos)"),
    ("adhesive failure", "the epoxy bonds (spar–foam, boom socket, fitting–wing) are tied contacts in the FEA or hand checks with an assumed "
                         "strength; peel at the bond ends and the foam's own tensile failure under the fitting need a coupon test"),
    ("nonlinear effects", "large deflection of the wing (geometric nonlinearity) at the ultimate load, PETG's yielding and creep under the motor's "
                          "heat, the TPU skid's and the foam's crushing in a landing (the landing loads are a stroke assumption, not a contact analysis)"),
    ("anisotropy", "the carbon tube is modelled with its axial modulus only; the printed parts are isotropic with a separate check across the "
                   "layers; layer lines at a hole's edge (stress concentration in the layer plane) are not modelled"),
    ("dynamics", "the loads are quasi-static: the landing impact's dynamic overshoot, flutter of the ailerons and the tail on the booms, "
                 "propeller-induced vibration of the booms (their first bending mode against the propeller's 1P / 2P) are not in the static FEA"),
]


def mesh_convergence(step, sizes=(3.0, 2.0, 1.4, 1.0), workdir=None, p=None, cases=None, cache_prefix=None):
    """The boom fitting at several element sizes: peak von Mises and displacement against the size (solved here)."""
    from vegeta import talos
    workdir = Path(workdir)
    rows = []
    for h in sizes:
        case = boom_fitting_case(step, p, cases, element_size=h)
        wd = workdir / f"h{h:g}"
        case.model.mesh(wd)
        r = case.model.solve(wd)
        nodes = None
        try:
            nodes = len(talos.read_frd(r.artifacts["frd"]).node_ids)
        except Exception:
            pass
        rows.append({"element size [mm]": h, "nodes": nodes, "max von Mises [MPa]": r.metrics.get("max_von_mises"),
                     "max displacement [mm]": r.metrics.get("max_displacement"), "across layers [MPa]": across_layers(r, "y") if r.ok else np.nan})
    return pd.DataFrame(rows).set_index("element size [mm]")


__all__ = ["ULTIMATE", "KNOCKDOWN", "PETG", "CARBON", "BUILD_AXIS", "PRINT_SETTINGS", "talos_material", "design_mass_kg", "gust_n", "load_cases",
           "schrenk", "spar_segment_forces", "wing_hand", "joints_hand", "FEACase", "export_step", "spar_case", "boom_case", "boom_fitting_case",
           "rear_spar_case", "motor_mount_case", "battery_tray_case", "models", "across_layers", "nominal_stress", "summary_row", "NOT_ESTABLISHED", "mesh_convergence"]
