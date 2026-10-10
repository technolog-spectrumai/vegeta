"""FALCO's structure: the load cases, the hand checks and the Talos (CalculiX) models.

NISUS+'s module does the wing (``nisus_plus_structure`` through its ``design``/``mass_table``/``drive`` seams: the
gust and manoeuvre cases, Schrenk, the spar and joiner hand checks and FEA cases, the battery tray); this one adds what
the tractor and the single tail change:

* the load cases — NISUS+'s plus the propeller's gyroscopic moment on the firewall, the single tube's torsion from the
  one fin, the parked-propeller landing (the blade-strike case that the parking prevents: named, not designed for);
* the joints by hand — the keel socket's bearing and epoxy, the tube at the socket's exit under the tail's lift and the
  fin's side load together (bending + torsion: the frame study's finding that the single tube's torsion is the sizing
  term, with the roll-wrapped tube's G), the stabiliser's 12/10 spar as a cantilever on the fitting, the fin's rod, the
  firewall's screws under the AT5220's thrust and the landing's inertia, the wing bolts, the straps, the flap horn;
* the Talos cases — the spar and the joiner (NISUS+'s), the tail tube A/B and its modes on the frame study's machinery
  (the saddle blocks, the sound-mesh retry), the keel socket (PETG) under the tube's bearing loads, the firewall
  (flight, brake, landing), the battery tray.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

import falco
import falco_flight as fl
import falco_systems as fsy
import frame_study as fst
import nisus_plus_structure as st
import nisus_plus_systems as fs
from nisus_structure import ULTIMATE, KNOCKDOWN, PETG, CARBON, BUILD_AXIS, PRINT_SETTINGS, FEACase, talos_material, across_layers, nominal_stress, summary_row  # noqa: F401

try:
    from vegeta.boreas import RHO0
except Exception:                                                        # pragma: no cover
    RHO0 = 1.225
G = 9.81
V_C_EAS = st.V_C_EAS
G_TUBE = fst.G_TUBE["roll-wrapped"]                                      # MPa, the roll-wrapped tube's shear modulus (frame study: assumption)
MOTOR_GROUP_KG = 0.450 + 0.117 + 0.030                                   # the AT5220, the 20x13, the spinner


def _d():
    return falco.Falco()


def _mass_table(k, p_):
    return fsy.mass_table(k, p=p_)


def design_mass_kg(p=None, battery_key=fsy.DEFAULT_PACK) -> float:
    """The 8S3P configuration + 8 % growth (the same 24 cells as NISUS+'s 6S4P: the same design mass)."""
    return st.design_mass_kg(falco.resolve(p), battery_key, design=_d(), mass_table=_mass_table)


# ================================================================================================= load cases
def load_cases(p=None, *, dr=None, a=None, **kw) -> pd.DataFrame:
    """NISUS+'s cases on FALCO (its mass, its drive, its tail areas) plus the tractor's: the propeller's gyroscopic
    moment at the limit pitch rate, the tube's torsion from the single fin's side load at its centroid, the
    blade-strike landing (what happens when the parking fails: a blade tip meeting the meadow at the touchdown speed —
    the propeller breaks; the firewall sees the motor's inertia at the strike's deceleration; this is NOT a design case
    but the number that says why the parking must work)."""
    p = falco.resolve(p)
    L = falco.Falco.layout(p)
    dr = dr or fsy.drive()
    a = a or fl.aero(p)
    df = st.load_cases(p, dr=dr, a=a, design=_d(), mass_table=_mass_table, **kw)
    at = df.attrs
    st_ = dr.at(0.0, 1.0)
    prop = fsy.propeller()
    I_prop = prop.rotor_mass * (prop.radius ** 2) / 3 if hasattr(prop, "rotor_mass") else 0.16 * 1.78 * (0.254 ** 2) / 3
    omega = st_["rpm"] * math.pi / 30
    q_pitch = math.radians(60.0)                                         # the pull-out's pitch rate (assumed: 60°/s at full power)
    M_gyro = I_prop * omega * q_pitch
    fin_arm = (0.45 * p["fin_height"] + p["tail_tube_od"] / 2) / 1000     # the plate fin's centre of pressure above the tube's axis
    T_fin = at["fin"] * fin_arm
    v_td, s_blade = 11.0, 0.030                                           # the touchdown speed and the blade's crush stroke (assumed)
    a_strike = v_td ** 2 / (2 * s_blade) / G
    extra = {
        "propeller gyroscopic moment at full power, limit [N m]": (M_gyro, f"I_prop ω q: {I_prop * 1e3:.2f} g m², {st_['rpm']:.0f} rpm, 60°/s pitch rate (assumed)"),
        "tail tube torsion from the fin at V_NE, limit [N m]": (T_fin, f"fin side load x {fin_arm * 1000:.0f} mm (centroid at 45 % of the plate + the tube's radius)"),
        "blade strike if the propeller is NOT parked [g]": (a_strike, f"a vertical blade meets the meadow at {v_td:g} m/s over a {s_blade * 1000:.0f} mm crush: not a design case — the parking's reason"),
        "parked propeller clearance at rest [mm]": (L["prop_ground_margin_parked_resting"], "the spinner over the ground line through the keel's corner and the tail bumper"),
        "blade-down clearance at rest [mm]": (L["prop_ground_margin_resting"], "negative: a vertical blade is in the ground — hence the parking"),
    }
    df2 = pd.DataFrame({k: {"value": v[0], "basis": v[1]} for k, v in extra.items()}).T
    out = pd.concat([df, df2])
    out.attrs.update(at, M_gyro=M_gyro, T_fin=T_fin, a_strike=a_strike, fin_arm=fin_arm)
    return out


# ================================================================================================= hand checks
def wing_hand(p=None, cases=None) -> pd.DataFrame:
    return st.wing_hand(falco.resolve(p), cases, design=_d())


def joints_hand(p=None, cases=None, battery_key=fsy.DEFAULT_PACK) -> pd.DataFrame:
    """The joints by hand at the ultimate loads: the keel socket, the tube at its exit (bending + torsion, the
    roll-wrapped G), the stabiliser's spar and the fin's rod on the end fitting, the wing bolts, the straps, the
    firewall's screws, the flap horn and hinges."""
    p = falco.resolve(p)
    L = falco.Falco.layout(p)
    c = cases if cases is not None else load_cases(p)
    a = c.attrs
    rows = {}
    Ls = p["tail_socket_length"]
    arm = L["x_ac_tail"] - L["x_pod_end"]
    F_tail = ULTIMATE * max(a["tail"], a["F_stub"])
    F_fin = ULTIMATE * a["fin"]
    R_exit = F_tail * (arm + Ls) / Ls
    tau = R_exit / (math.pi * p["tail_tube_od"] * Ls / 2)
    rows["keel socket: length [mm]"] = Ls
    rows["keel socket: bearing reaction at the exit, ultimate [N]"] = R_exit
    rows["keel socket: bearing on the PETG bore [MPa]"] = R_exit / (p["tail_tube_od"] * Ls / 3)
    rows["keel socket: bearing margin (PETG 45 MPa x 0.7 x 0.5 for the layers)"] = 45 * 0.7 * 0.5 / (R_exit / (p["tail_tube_od"] * Ls / 3)) - 1
    rows["keel socket: epoxy shear [MPa]"] = tau
    rows["keel socket: margin (epoxy on PETG 5 MPa x 0.5)"] = 5.0 * KNOCKDOWN["adhesive"] / tau - 1
    do, di = p["tail_tube_od"], p["tail_tube_id"]
    I_b = math.pi / 64 * (do ** 4 - di ** 4)
    J_b = 2 * I_b
    arm_fin = L["x_ac_fin"] - L["x_pod_end"]
    sig_b = math.hypot(F_tail * arm, F_fin * arm_fin) * (do / 2) / I_b
    T_ult = ULTIMATE * a["T_fin"] * 1000
    tau_t = T_ult * (do / 2) / J_b
    sig_vm = math.sqrt(sig_b ** 2 + 3 * tau_t ** 2)
    kd = CARBON["strength"] * KNOCKDOWN["carbon tube"]
    rows["tail tube: bending at the socket exit, tail + fin ultimate [MPa]"] = sig_b
    rows["tail tube: torsion shear from the fin, ultimate [MPa]"] = tau_t
    rows["tail tube: von Mises at the exit [MPa]"] = sig_vm
    rows["tail tube: margin (500 MPa x 0.8)"] = kd / sig_vm - 1
    rows["tail tube: torsion margin (in-plane shear 40 MPa x 0.8, roll-wrapped)"] = 40 * 0.8 / tau_t - 1
    free = (L["x_ac_fin"] - L["x_pod_end"])
    twist = math.degrees(a["T_fin"] * 1000 * free / (G_TUBE * J_b))
    rows["tail tube: fin twist under its limit side load [deg] (G 20 GPa roll-wrapped)"] = twist
    E = CARBON["E"]
    defl = F_tail / ULTIMATE * arm ** 3 / (3 * E * I_b)
    rows["tail tube: tail deflection under the limit tail load [mm]"] = defl
    # the stabiliser's spar: each half a cantilever from the fitting's saddle under half the tail load at 45 % of the half span
    so, si = fsy.STAB_SPAR["od"], fsy.STAB_SPAR["id_"]
    I_s = math.pi / 64 * (so ** 4 - si ** 4)
    half = p["tail_span"] / 2
    M_s = F_tail / 2 * 0.45 * half
    sig_s = M_s * (so / 2) / I_s
    rows["stabiliser spar 12/10: bending at the fitting, ultimate [MPa]"] = sig_s
    rows["stabiliser spar: margin (500 MPa x 0.8)"] = kd / sig_s - 1
    rows["stabiliser spar: tip deflection under the limit load [mm]"] = (F_tail / ULTIMATE / 2) * (0.45 * half) ** 2 * (3 * half - 0.45 * half) / (6 * E * I_s)
    ro, ri = fsy.FIN_ROD["od"], fsy.FIN_ROD["id_"]
    I_f = math.pi / 64 * (ro ** 4 - ri ** 4)
    M_f = F_fin * 0.45 * p["fin_height"]
    sig_f = M_f * (ro / 2) / I_f
    rows["fin rod 8/6: bending at the post, ultimate [MPa]"] = sig_f
    rows["fin rod: margin (500 MPa x 0.8)"] = kd / sig_f - 1
    # the end fitting's saddle on the tube: the tail's lift as a couple on the sleeve's bond
    L_sleeve = p["tail_chord"]
    R_sleeve = F_tail * (0.25 * p["tail_chord"] + 0.5 * L_sleeve) / (0.5 * L_sleeve)
    tau_s = R_sleeve / (math.pi * do * L_sleeve / 2)
    rows["end fitting: sleeve bond shear, ultimate [MPa]"] = tau_s
    rows["end fitting: bond margin (epoxy 5 MPa x 0.5)"] = 5.0 * KNOCKDOWN["adhesive"] / tau_s - 1
    F_bolt = ULTIMATE * 0.5 * a["n_limit"] * a["W"] * 0.75 / 2
    rows["wing bolts (2 x M6 nylon PA66): tension each, inverted ultimate [N]"] = F_bolt
    rows["wing bolts: margin (M6 PA66 ~ 900 N x 0.7)"] = 900 * 0.7 / F_bolt - 1
    m_b = fsy.pack(battery_key).mass_g / 1000
    F_strap = ULTIMATE * a["a_land"] * G * m_b / 2
    rows["battery straps (two): tension per leg, ultimate [N]"] = F_strap / 2
    rows["battery straps: margin (25 mm strap ~ 250 N x 0.5)"] = 250 * 0.5 / (F_strap / 2) - 1
    F_launch = ULTIMATE * a["a_launch"] * G * m_b
    rows["battery: launch inertia, ultimate [N] (front stop of the tray)"] = F_launch
    rows["battery front stop (PETG lip 80 x 14 x 2.5, bending) margin"] = (45 * 0.7 * 80 * 2.5 ** 2 / 6 / (F_launch * 7.0)) - 1
    F_scr = ULTIMATE * (a["thrust"] + MOTOR_GROUP_KG * a["a_land"] * G) / 4
    rows["firewall screws (4 x M4, heat-set inserts): force each, ultimate [N]"] = F_scr
    rows["firewall screws: margin (M4 insert pull-out 700 N x 0.7)"] = 700 * 0.7 / F_scr - 1
    r_bolt = p["motor_bolt_a"] / 2 / 1000
    F_tq = ULTIMATE * (a["torque"] + a["M_gyro"]) / (4 * r_bolt)
    rows["firewall screws: shear from the torque + gyroscopic moment, ultimate [N]"] = F_tq
    rows["firewall screws: shear margin (M4 8.8 ~ 3 kN x 0.5 in PETG bearing)"] = 3000 * 0.5 / F_tq - 1
    H = ULTIMATE * a["H_flap"]
    rows["flap horn in crow at V_FE: hinge moment, ultimate [N cm]"] = H * 100
    rows["flap horn: servo margin (6 kg·cm servo)"] = 6.0 * G / 100 / H - 1
    F_h = ULTIMATE * a["flap_normal"] / 3
    rows["flap hinges (3 per flap, CA hinges): force each, ultimate [N]"] = F_h
    rows["flap hinges: margin (CA hinge ~ 60 N pull-out x 0.5)"] = 60 * 0.5 / F_h - 1
    return pd.DataFrame({"value": rows})


# ================================================================================================= Talos models
def export_step(part: str, workdir: Path, p=None) -> Path:
    return st.export_step(part, workdir, falco.resolve(p), design=_d())


def spar_case(step, p=None, cases=None, element_size=3.0):
    return st.spar_case(step, falco.resolve(p), cases, element_size, design=_d())


def joiner_case(step, p=None, cases=None, element_size=2.5):
    return st.joiner_case(step, falco.resolve(p), cases, element_size, design=_d())


def battery_tray_case(step, p=None, cases=None, element_size=2.5):
    return st.battery_tray_case(step, falco.resolve(p), cases, element_size, design=_d())


def tail_frame(p=None) -> fst.Frame:
    """FALCO's tail as the frame study's ``Frame`` (the single tube variant with FALCO's numbers): the study's CAD, FEA
    cases and tail-motion readers then apply."""
    p = falco.resolve(p)
    L = falco.Falco.layout(p)
    return fst.Frame(name="falco", kind="single", prop_in=p["prop_diameter"] / 25.4, pitch_in=fsy.PROP[1], boom_y=0.0, tube_od=p["tail_tube_od"],
                     tube_id=p["tail_tube_id"], tube_kind="roll-wrapped", x_root=L["x_pod_end"], x_end=p["tail_x_end"], tail_span=p["tail_span"],
                     tail_chord=p["tail_chord"], fin_height=p["fin_height"], fin_chord=p["fin_chord"], n_fins=1, tie_od=fsy.STAB_SPAR["od"],
                     tie_id=fsy.STAB_SPAR["id_"], flap_y0=p["flap_y0"], rear_spar_half_length=p["rear_spar_half_length"], rear_spar_od=p["rear_spar_od"],
                     rear_spar_id=p["rear_spar_id"], fin_ventral=0.0, p={"boom_z": p["boom_z"], "tail_thickness": p["tail_thickness"]})


def _np_over(p) -> dict:
    """The NISUS+ overrides the frame study's helpers need for FALCO's tube (its axis height, the tail plates)."""
    return {"boom_z": p["boom_z"], "tail_thickness": p["tail_thickness"]}


def tail_tube_cases(workdir, p=None, cases=None, element_size: float = 2.5) -> list:
    """The frame study's cases on FALCO's tube: ``A`` the symmetric pull-out (the stabiliser's ultimate lift on both
    halves, the fin's side load at its centroid), ``B`` the asymmetric one (the left half's lift only, the fin's side
    load), the socket fixed; and the modal model with the tail's masses. FALCO's loads (``load_cases``)."""
    from vegeta import talos
    p = falco.resolve(p)
    c = cases if cases is not None else load_cases(p)
    f = tail_frame(p)
    pn = _np_over(p)
    workdir = Path(workdir)
    (workdir / "cad").mkdir(parents=True, exist_ok=True)
    step = workdir / "cad" / "falco_tail_frame.step"
    if not step.exists():
        fst._frame_cad(f, pn).val().exportStep(str(step))
    z = p["boom_z"]
    r = f.tube_od / 2 + 0.5
    x_sock0 = f.x_root - 200.0 - 0.5
    zt = fst._zt(f, pn)
    rt = f.tie_od / 2 + 0.3
    regions = [talos.SurfacesInBox("socket", (x_sock0, -r, z - r, f.x_root + 0.3, r, z + r)),
               talos.SurfacesInBox("tie_left", (f.x_tie - rt, -f.tail_span / 2 - 0.5, zt - rt, f.x_tie + rt, 0.5, zt + rt)),
               talos.SurfacesInBox("tie_right", (f.x_tie - rt, -0.5, zt - rt, f.x_tie + rt, f.tail_span / 2 + 0.5, zt + rt)),
               talos.SurfacesInBox("ends", (f.x_end - 0.3, -r, z - r, f.x_end + 0.3, r, z + r)),
               talos.SurfacesInBox("post_0", (f.x_fin - 8.5, -8.5, z + f.h_fin_ac - 0.3, f.x_fin + 8.5, 8.5, z + f.h_fin_ac + 0.3))]
    F_tail = ULTIMATE * max(c.attrs["tail"], c.attrs["F_stub"])
    F_fin = ULTIMATE * c.attrs["fin"]
    F_half = F_tail / 2
    mesh = talos.MeshSettings(element_size=element_size, order=2)
    mat = talos_material(CARBON)
    m = fst.frame_mass(f, pn)
    masses = [talos.PointMass("tie_left", 0.5 * m["stabiliser [g]"] * 1e-6), talos.PointMass("tie_right", 0.5 * m["stabiliser [g]"] * 1e-6),
              talos.PointMass("ends", fsy.CAD["tail_fitting"] * 1e-3 * PETG["rho"] * fsy.PRINT_FILL["tail_fitting"] * 1e-6 + 2 * 22.0 * 1e-6),
              talos.PointMass("post_0", m["fins [g]"] * 1e-6)]
    out = []
    loads = {"A": [talos.Force("tie_left", fz=F_half), talos.Force("tie_right", fz=F_half), talos.Force("post_0", fy=F_fin)],
             "B": [talos.Force("tie_left", fz=F_half), talos.Force("post_0", fy=F_fin)]}
    desc = {"A": f"symmetric pull-out: stabiliser {F_tail:.0f} N up, fin {F_fin:.0f} N side (ultimate)",
            "B": f"asymmetric: the left half's {F_half:.0f} N only, fin {F_fin:.0f} N side (ultimate)"}
    for key in ("A", "B"):
        case = FEACase(f"tail_tube_{key}", "tail_tube_fea", f"FALCO tail tube 30/28 roll-wrapped: {desc[key]}",
                       talos.StructuralModel(step, "mm-N-MPa", mat, regions, [talos.FixedSupport("socket")], loads[key], mesh, name=f"tail_tube_{key}"),
                       None, CARBON["strength"])
        case.exclude = lambda xyz, f=f: ((xyz[:, 0] < f.x_root + f.tube_od) | (np.abs(xyz[:, 0] - f.x_tie) < 15.0) | (np.abs(xyz[:, 0] - f.x_fin) < 15.0)
                                         | (xyz[:, 2] > p["boom_z"] + f.tube_od / 2 - 0.5))
        case.kind, case.frame = "static", f
        out.append(case)
    modal = FEACase("tail_tube_modes", "tail_tube_fea", "FALCO tail tube: the first modes with the tail's masses (the servos at the fitting included)",
                    talos.StructuralModel(step, "mm-N-MPa", mat, regions, [talos.FixedSupport("socket")], [], mesh, masses=masses, name="tail_tube_modes"),
                    None, CARBON["strength"])
    modal.kind, modal.frame = "modes", f
    out.append(modal)
    return out


def solve_tail_tube(workdir, p=None, cases=None, *, run: bool = True, threads: int = 4, element_size: float = 2.5, retries: int = 2, progress: bool = False) -> tuple:
    """Mesh and solve the tail tube's cases with the frame study's solver (``solve_cases`` without its NISUS+ retry):
    an unsound result (``frame_study._unsound``) rebuilds the three cases at a smaller element size. Returns
    (cases, results)."""
    p = falco.resolve(p)
    cs = tail_tube_cases(workdir, p, cases, element_size)
    res = fst.solve_cases(cs, workdir, run=run, threads=threads, retries=0, p=_np_over(p), progress=progress, label="tail tube FEA")
    n = 0
    while run and retries > 0 and any(fst._unsound(c, res[c.name]) for c in cs):
        n += 1
        print("tail tube: unsound mesh — " + "; ".join(f"{c.name}: {fst._unsound(c, res[c.name])}" for c in cs if fst._unsound(c, res[c.name])))
        element_size *= 0.88
        wd = Path(workdir) / f"retry{n}"
        cs = tail_tube_cases(wd, p, cases, element_size)
        res = fst.solve_cases(cs, wd, run=run, threads=threads, retries=0, p=_np_over(p), progress=progress, label=f"tail tube FEA (retry {n})")
        retries -= 1
    return cs, res


def tail_tube_table(cs: list, res: dict, p=None) -> pd.DataFrame:
    return fst.fea_table(cs, res, _np_over(falco.resolve(p)))


def tail_tube_modes(cs: list, res: dict) -> pd.DataFrame:
    return fst.modes_table(cs, res)


def tail_socket_case(step, p=None, cases=None, element_size=2.0):
    """The keel's PETG socket: the tube's bearing on its bore as the tail's ultimate load (lift and the stub case, the
    fin's side load) reacted at the bore's two ends (the couple of a cantilever in a socket: ``joints_hand``'s R_exit
    down at the exit, R_exit − F up at the front), the block's top and bottom faces bonded to the keel frame."""
    from vegeta import talos
    p = falco.resolve(p)
    L = falco.Falco.layout(p)
    c = cases if cases is not None else load_cases(p)
    a = c.attrs
    Ls = p["tail_socket_length"]
    arm = L["x_ac_tail"] - L["x_pod_end"]
    F_tail = ULTIMATE * max(a["tail"], a["F_stub"])
    F_fin = ULTIMATE * a["fin"]
    arm_fin = L["x_ac_fin"] - L["x_pod_end"]
    R_exit_z = F_tail * (arm + Ls) / Ls
    R_exit_y = F_fin * (arm_fin + Ls) / Ls
    x0, x1 = L["x_tube0"], L["x_pod_end"]
    z, w = p["boom_z"], p["tail_tube_od"] + 8.0
    r = p["tail_tube_od"] / 2 + 0.1
    seg = 25.0
    regions = [talos.SurfacesOnPlane("top", "z", z + w / 2), talos.SurfacesOnPlane("bottom", "z", z - w / 2),
               talos.SurfacesInBox("bore_exit", (x1 - seg - 0.5, -r - 0.5, z - r - 0.5, x1 + 0.5, r + 0.5, z + r + 0.5)),
               talos.SurfacesInBox("bore_front", (x0 - 0.5, -r - 0.5, z - r - 0.5, x0 + seg + 0.5, r + 0.5, z + r + 0.5))]
    loads = [talos.Force("bore_exit", fz=-R_exit_z, fy=R_exit_y), talos.Force("bore_front", fz=R_exit_z - F_tail, fy=-(R_exit_y - F_fin))]
    desc = f"keel socket (PETG): the tube's bearing {R_exit_z:.0f} N down at the exit, {R_exit_z - F_tail:.0f} N up at the front, the fin's {R_exit_y:.0f} N sideways (ultimate)"
    return FEACase("tail_socket", "tail_socket_fea", desc,
                   talos.StructuralModel(step, "mm-N-MPa", talos_material(PETG), regions, [talos.FixedSupport("top"), talos.FixedSupport("bottom")], loads,
                                         talos.MeshSettings(element_size=element_size, order=2), name="tail_socket"),
                   BUILD_AXIS.get("boom_fitting", "z"), PETG["strength"], PETG["strength_z"])


def motor_mount_case(step, p=None, cases=None, element_size=1.5, case="flight"):
    """FALCO's firewall (the 52xx bolt pattern), three load cases: ``flight`` (static thrust, torque + the gyroscopic
    moment, the motor group's inertia at the limit n), ``brake`` (the propeller brake's reverse thrust at V_NE, the
    regeneration's reverse torque), ``landing`` (the inertia at the belly landing). The skirt bonded in the cowl.
    Ultimate."""
    from vegeta import talos
    p = falco.resolve(p)
    L = falco.Falco.layout(p)
    c = cases if cases is not None else load_cases(p)
    a = c.attrs
    xf, zm, t = L["x_firewall"], L["motor_z"], p["mount_thickness"]
    holes = falco.Falco().motor_holes(p)
    prof = falco.Falco.pod_profile(p)
    r_loc = float(np.interp(xf, prof[:, 0], prof[:, 1])) - p["pod_wall"] - 0.3
    info = talos.inspect_step(step, units="mm-N-MPa")
    skirt = [sf.tag for sf in info.surfaces if sf.kind.startswith("Cylinder") and xf + t < sf.centroid[0] < xf + t + 15.0
             and abs(sf.area - 2 * math.pi * (r_loc + 0.3) * 15.0) < 0.05 * 2 * math.pi * (r_loc + 0.3) * 15.0]
    regions = [talos.Surfaces("skirt", skirt)]
    if case == "flight":
        T, Q, n = a["thrust"], a["torque"] + a["M_gyro"], a["n_limit"]
    elif case == "brake":
        T, Q, n = a["brake"], -0.5 * a["torque"], a["n_limit"]
    else:
        T, Q, n = 0.0, 0.0, a["a_land"]
    F_in = ULTIMATE * MOTOR_GROUP_KG * n * G
    th = math.radians(p["motor_downthrust_deg"])
    down = np.array([-math.sin(th), 0.0, -math.cos(th)])
    loads = []
    for name, (yy, zz) in holes.items():
        regions.append(talos.SurfacesInBox(name, (xf - 0.5, yy - 3.0, zm + zz - 3.0, xf + t + 0.5, yy + 3.0, zm + zz + 3.0)))
        r = math.hypot(yy, zz)
        tang = np.array([-zz, yy]) / r
        Ft = ULTIMATE * Q / (4 * r / 1000) * tang
        f = np.array([-ULTIMATE * T / 4, Ft[0], Ft[1]]) + F_in / 4 * down
        loads.append(talos.Force(name, fx=float(f[0]), fy=float(f[1]), fz=float(f[2])))
    desc = f"firewall, {case}: thrust {ULTIMATE * T:.1f} N, torque {ULTIMATE * Q:.2f} N m, inertia {F_in:.1f} N (ultimate); skirt bonded in the cowl"
    return FEACase(f"motor_mount_{case}", "motor_mount_fea", desc,
                   talos.StructuralModel(step, "mm-N-MPa", talos_material(PETG), regions, [talos.FixedSupport("skirt")], loads,
                                         talos.MeshSettings(element_size=element_size, order=2), name=f"motor_mount_{case}"),
                   BUILD_AXIS["motor_mount"], PETG["strength"], PETG["strength_z"])


def models(workdir, p=None, cases=None) -> list:
    """Every FEA case but the tail tube's (those come from ``tail_tube_cases``: the frame study's solver runs them)
    with its STEP exported into ``workdir`` (nothing is meshed or solved here)."""
    workdir = Path(workdir)
    p = falco.resolve(p)
    c = cases if cases is not None else load_cases(p)
    steps = {part: export_step(part, workdir / "cad", p) for part in ("spar_fea", "spar_joiner_fea", "tail_socket_fea", "motor_mount_fea", "battery_tray")}
    return [spar_case(steps["spar_fea"], p, c), joiner_case(steps["spar_joiner_fea"], p, c), tail_socket_case(steps["tail_socket_fea"], p, c),
            motor_mount_case(steps["motor_mount_fea"], p, c, case="flight"), motor_mount_case(steps["motor_mount_fea"], p, c, case="brake"),
            motor_mount_case(steps["motor_mount_fea"], p, c, case="landing"), battery_tray_case(steps["battery_tray"], p, c)]


NOT_ESTABLISHED = [r for r in st.NOT_ESTABLISHED if not str(r[0]).startswith("flutter")] + [
    ("flutter at TAS", "the never-exceed speed is an EAS; flutter depends on the TAS, 25 % higher at 4000 m: the wing's torsion and the tail on the single "
                       "tube (its torsional mode: the FEA's isotropic G makes it 2-3 x too stiff) need a flutter estimate before the dives (todo)"),
    ("the parked propeller", "the ESC's brake stops the blades where they are; the parking nudge and the Hall sensor are a bench item: until proven, a "
                             "landing with a blade down is a propeller strike (the load case names the g, the firewall is not designed for it)"),
    ("the slipstream", "the tractor's wash over the fuselage and the wing root (drag, the tail's dynamic pressure at power) is not in the build-up; "
                       "the CFD rotor-disk case is its check — NOT RUN here"),
    ("the roll-wrapped tube's G", "20 GPa assumed from catalogue ranges: the fin's twist under side load scales with it; measure a sample"),
]


__all__ = ["ULTIMATE", "KNOCKDOWN", "PETG", "CARBON", "BUILD_AXIS", "PRINT_SETTINGS", "FEACase", "talos_material", "design_mass_kg", "load_cases", "wing_hand",
           "joints_hand", "export_step", "spar_case", "joiner_case", "battery_tray_case", "tail_frame", "tail_tube_cases", "solve_tail_tube", "tail_tube_table",
           "tail_tube_modes", "tail_socket_case", "motor_mount_case", "models", "across_layers", "nominal_stress", "summary_row", "NOT_ESTABLISHED", "MOTOR_GROUP_KG"]
