"""The submarine: a 1.2 m AUV with a pressure hull, sail, cruciform fins and a 120 mm propeller (notebook 13).

The tree::

    submarine (submersible)        the drive: top speed, cruise, endurance and range vs speed with the hotel load; the
                                   resistance curve (13 cell 8), the drive at every speed and at 14's survey speed (13 cell
                                   13, 14 cell 4), the dive thrust grid (13 cell 29); the pressure hull's thin-shell theory,
                                   collapse, ring mode and pressure vs depth (13 cells 16, 26, 28); the open-water curves
                                   (14 cell 6) and map; noise and cavitation at the points (13 cell 23); the blade-pass
                                   margins once the blade modes are solved (13 cell 26)
      hull (submarine)             vehicle, body and pressure-hull CAD; buoyancy, mass budget (with x/z, 13 cell 6), the
                                   trim lead, BG
      hull_cfd (vehicle_cfd)       the vehicle nose-upstream at cruise (Aeromant rans_ksst_external)
      pressure_hull_fea (pressure_hull)  the pressure hull at the rated depth (Talos)
      propeller (propeller)        the 120 mm propeller's CAD
      rotor_cfd (rotor)            the propeller at cruise in its wake (Aeromant rotor_mrf)
      blade_fea (blade)            one aluminium blade at top speed; its 4 modes in air and x0.65 in water (13 cell 25)
      scene_cfd (rotor_disk)       the vehicle with its propeller as a rotor disk (scenarios/run_scenario.py sub(), with the
                                   wetted surface, buoyancy centre and rpm of this tree instead of its recorded ones)

Diving, depth cycling and the propeller study of notebook 14 stay in the notebooks (they can read the drive, the dive
thrust grid and the open-water data from here).
"""
from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
from vegeta import aeromant, boreas, chronos, talos

from .. import DATA, RUNS, vida, results
from .._cli import main, parser
from ..components import hull as H, propeller as pr
from ..components._cad import export_kept
from ..components.impeller import environment
from ..components.submarine import Submarine
from ..vida import Assembly
from ._common import add_after, run_cfd as cfd_node, solve_fea, solve_modes

NAME = "submarine"
RHO_W, NU_W, G = 1025.0, 1.05e-6, 9.81
DESIGN_DEPTH_M, V_CRUISE, X_PH, RESERVE = 200.0, 1.5, 420.0, 0.005                      # 13 cells 2, 4, 6
RHO_AL, RHO_GFRP, FAIRING_T = 2.70e-6, 1.8e-6, 2.0
ITEMS = [("battery 7S Li-ion 20 Ah", 6.0, 560.0, -30.0), ("electronics, INS, acoustic modem", 1.5, 780.0, 0.0),
         ("sonar payload (nose, free flooded)", 3.5, 1060.0, -15.0), ("thruster, propeller, stern gear", 1.5, 80.0, 0.0)]
DRIVE = {"propeller": "120 mm 3-blade", "motor": "thruster 100KV", "battery": "7S Li-ion 20 Ah", "wake": 0.85,
         "thrust_deduction": 0.10, "hotel_W": 25.0}                                              # 13 cell 13
AL6061 = dict(name="6061-T6", youngs_modulus=69000.0, poissons_ratio=0.33, density=2.7e-9, yield_strength=240.0,
              source="handbook, machined")
ELEMENT = {"pressure_hull": {"smoke": 12.0, "quick": 9.0, "full": 6.0}, "blade": {"smoke": 4.0, "quick": 3.0, "full": 2.0}}
HULL_CFD = {"smoke": dict(iterations=60, surface_level=3, near_level=2, wake_level=1, cells_per_length=3.0),
            "quick": dict(iterations=400, surface_level=3, near_level=2, wake_level=1, cells_per_length=3.0),
            "full": dict(iterations=400, surface_level=4, near_level=3, wake_level=2, cells_per_length=3.0)}
SCENE_ITERATIONS = {"smoke": 60, "quick": 300, "full": 600}
TEST_FACTOR, P_ATM = 1.5, 101325.0                          # 13 cell 2: test depth = 1.5 x rated depth; cell 28 [Pa]
V_SURVEY = 1.0                                              # 14 cell 2: SPEEDS["survey"] [m/s]
BLADE_MODES, ADDED_MASS_FACTOR, BLADE_DAMPING = 4, 0.65, 0.02   # 13 cells 25-26: in-water / in-air frequency, assumed
CP_MIN, NOISE_DIST, NOISE_ANGLE, NOISE_HARMONICS = -1.0, 1.0, 90.0, 5   # 13 cell 23: section Cp_min, 1 m broadside, dB re 1 uPa
CAV_DEPTHS_M = {"5m": 5.0, "rated_depth": DESIGN_DEPTH_M}   # 13 cell 23: the cavitation number at 5 m and at the rated depth
MAP_RPM, MAP_V = np.arange(200, 2501, 100), np.arange(0.0, 4.01, 0.25)   # the open-water map (submarine.json's "map")


def build(fidelity: str = "full", sub: dict | None = None) -> Assembly:
    p = Submarine().resolve(**dict(sub or {}, part="vehicle"))
    root = Assembly(NAME, "submersible", params={"fidelity": fidelity, "drive": DRIVE, "v_cruise": V_CRUISE,
                                                 "design_depth_m": DESIGN_DEPTH_M})
    root.add(Assembly("hull", "submarine", params={"sub": p, "items": [list(i) for i in ITEMS], "reserve": RESERVE, "x_ph": X_PH}))
    root.add(Assembly("propeller", "propeller", params={"spec": pr.get(DRIVE["propeller"]).to_dict()}))
    return root


def mass_and_trim(mv: dict, mh: dict) -> dict:
    """Buoyancy, the mass budget with the computed trim lead, CG, BG (13 cell 6)."""
    buoyancy_kg = mv["volume_mm3"] * 1e-9 * RHO_W
    CB = np.array(mv["center_of_mass_mm"])
    items = [("pressure hull (6061-T6, from CAD)", mh["volume_mm3"] * RHO_AL, X_PH + mh["center_of_mass_mm"][0], 0.0),
             ("fairing, sail and fins (GFRP 2 mm)", mv["surface_area_mm2"] * FAIRING_T * RHO_GFRP, CB[0], CB[2]), *ITEMS]
    m = np.array([i[1] for i in items])
    M_target = buoyancy_kg * (1 - RESERVE)
    m_lead = M_target - m.sum()
    if m_lead < 0:
        raise ValueError(f"the vehicle is {-m_lead:.2f} kg too heavy for its volume: enlarge the hull or cut the budget")
    x_lead = (CB[0] * M_target - (m * [i[2] for i in items]).sum()) / m_lead
    items.append(("trim lead (computed)", m_lead, x_lead, -60.0))
    m = np.array([i[1] for i in items])
    M = m.sum()
    CG = np.array([(m * [i[2] for i in items]).sum() / M, 0.0, (m * [i[3] for i in items]).sum() / M])
    BG = CB[2] - CG[2]
    return {"buoyancy_kg": buoyancy_kg, "mass_kg": M, "net_buoyancy_N": (buoyancy_kg - M) * G, "CB_mm": CB, "CG_mm": CG,
            "BG_mm": BG, "righting_moment_at_90deg_Nm": M * G * BG / 1000, "trim_lead_kg": m_lead, "trim_lead_x_mm": x_lead,
            "items_kg": {i[0]: i[1] for i in items},
            "mass_budget": {i[0]: {"mass_kg": float(i[1]), "x_mm": float(i[2]), "z_mm": float(i[3])} for i in items}}


def drive(L_m: float, D_m: float, S_wet: float, S_body: float):
    """Top speed, cruise, endurance and range (13 cell 13)."""
    spec = pr.get(DRIVE["propeller"])
    prop, sec = spec.model(), spec.airfoil()
    battery = pr.battery(DRIVE["battery"])
    d = boreas.Propulsion(prop, sec, pr.motor(DRIVE["motor"]), battery, rho=RHO_W)
    WAKE, T_DED, HOTEL_W = DRIVE["wake"], DRIVE["thrust_deduction"], DRIVE["hotel_W"]

    def resistance(V):
        return H.body_of_revolution_resistance(V, L_m, D_m, S_wet, S_body)

    def point_at(V):
        return d.for_thrust(resistance(V)[0] / (1 - T_DED), WAKE * V)

    lo, hi = 0.3, 5.0
    for _ in range(24):
        Vm = 0.5 * (lo + hi)
        lo, hi = (Vm, hi) if d.at_throttle(1.0, WAKE * Vm).thrust * (1 - T_DED) > resistance(Vm)[0] else (lo, Vm)
    V_MAX = lo
    cruise, full = point_at(V_CRUISE), d.at_throttle(1.0, WAKE * V_MAX)
    speeds = np.linspace(0.5, V_MAX, 14)
    pts = [point_at(v) for v in speeds]
    endurance_h = np.array([battery.usable_wh / (pt.electrical_power + HOTEL_W) for pt in pts])
    R_cruise, _, _, Cf_c, k_form = resistance(V_CRUISE)                     # 13 cell 8: the resistance curve, Cf, 1 + k
    Vs = np.linspace(0.3, 4.0, 38)
    R = np.array([resistance(v)[:3] for v in Vs])
    survey = point_at(V_SURVEY)                                             # 14 cell 4's survey point, on this drive
    return prop, sec, battery, {"cruise": cruise, "full": full}, {
        "top_speed_m_s": V_MAX, "resistance_at_cruise_N": resistance(V_CRUISE)[0], "speeds_m_s": speeds,
        "endurance_h": endurance_h, "range_km": endurance_h * speeds * 3.6,
        **speed_sweep(pts), "cruise_throttle": float(cruise.throttle),
        "Re_cruise": V_CRUISE * L_m / NU_W, "Cf_cruise": Cf_c, "form_factor_1pk": 1 + k_form,
        "effective_power_cruise_W": R_cruise * V_CRUISE,
        "resistance_curve": {"speed_m_s": Vs, "total_N": R[:, 0], "bare_body_N": R[:, 1], "appendages_N": R[:, 2],
                             "effective_power_W": R[:, 0] * Vs},
        "drive_points": {k: point_row(v, V) for k, v, V in (("survey", survey, V_SURVEY), ("cruise", cruise, V_CRUISE),
                                                            ("full", full, V_MAX))}}


def speed_sweep(pts) -> dict:
    """13 cell 13's ``pts`` (the drive at each of ``speeds_m_s``) as columns: what the cell plots (the electrical power)
    and the rest of each point."""
    return {"rpm": [float(pt.rpm) for pt in pts], "throttle": [float(pt.throttle) for pt in pts],
            "thrust_N": [float(pt.thrust) for pt in pts], "torque_Nm": [float(pt.aero.torque) for pt in pts],
            "electrical_W": [float(pt.electrical_power) for pt in pts], "current_A": [float(pt.current) for pt in pts],
            "prop_eff": [float(pt.aero.efficiency) for pt in pts], "motor_eff": [float(pt.motor_efficiency) for pt in pts],
            "advance_ratio": [float(pt.aero.advance_ratio) for pt in pts],
            "current_limited": [bool(pt.current_limited) for pt in pts]}


def point_row(v, V: float) -> dict:
    """One operating point as 14 cell 4's table has it (the inflow is ``wake x speed``, 14's ``(1 - W_MEAN) x speed``),
    plus the throttle and the motor efficiency (13 cell 13's table)."""
    return {"boat_speed_m_s": float(V), "inflow_m_s": DRIVE["wake"] * V, "rpm": float(v.rpm), "thrust_N": float(v.thrust),
            "torque_Nm": float(v.aero.torque), "J": float(v.aero.advance_ratio), "prop_efficiency": float(v.aero.efficiency),
            "electrical_W": float(v.electrical_power), "current_A": float(v.current), "current_limited": bool(v.current_limited),
            "throttle": float(v.throttle), "motor_efficiency": float(v.motor_efficiency)}


def propulsion():
    """The drive's propeller, section, battery and ``boreas.Propulsion`` in sea water, as ``drive`` builds them."""
    spec = pr.get(DRIVE["propeller"])
    prop, sec = spec.model(), spec.airfoil()
    battery = pr.battery(DRIVE["battery"])
    return prop, sec, battery, boreas.Propulsion(prop, sec, pr.motor(DRIVE["motor"]), battery, rho=RHO_W)


def dive_thrust(drive, resistance, cruise, V_MAX: float) -> dict:
    """13 cell 29's thrust grid for the dive simulation: the net thrust (after the thrust deduction) at the cruise
    throttle and at full throttle over 0.05 to 1.05 x top speed, with the resistance at the same speeds (the cell's
    ``resistance(max(u, 0.05))``). ``drive`` is the ``boreas.Propulsion`` (``propulsion()``), ``resistance(V)`` the
    body's ``(R, ...)``, ``cruise`` the cruise point. About 17 s (48 drive points)."""
    WAKE, T_DED = DRIVE["wake"], DRIVE["thrust_deduction"]
    u_grid = np.linspace(0.05, V_MAX * 1.05, 24)
    T_grid = {thr: np.array([drive.at_throttle(thr, WAKE * u).thrust for u in u_grid]) * (1 - T_DED) for thr in (cruise.throttle, 1.0)}
    return {"speed_m_s": u_grid, "inflow_ratio": WAKE, "thrust_deduction": T_DED, "cruise_throttle": float(cruise.throttle),
            "net_thrust_N": {"cruise_throttle": T_grid[cruise.throttle], "full_throttle": T_grid[1.0]},
            "resistance_N": np.array([resistance(max(u, 0.05))[0] for u in u_grid])}


def scene_params(S_wet: float, cb_x_m: float, L_m: float, rpm: float, prop, section, iterations: int = 600) -> dict:
    """run_scenario.sub(): the propeller as a rotor disk 40 mm behind the tail tip, the vehicle nose-upstream."""
    return dict(velocity=V_CRUISE, kinematic_viscosity=NU_W, density=RHO_W, reference_area=S_wet, reference_length=L_m / 4,
                center_of_rotation=(-cb_x_m, 0.0, 0.0), iterations=iterations, residual_target=1e-4,
                cells_per_length=3.0, surface_level=4, near_level=3, wake_level=2,
                disk1_center=[0.04, 0.0, 0.0], disk_axis=[-1.0, 0.0, 0.0],
                diameter=prop.diameter, rpm=float(rpm), blades=prop.blades,
                blade=pr.blade_table(prop), polar=pr.polar_table(section), rotation1=1, disk_level=5)


def _point(v) -> dict:
    return {"rpm": float(v.rpm), "thrust_N": float(v.thrust), "current_A": float(v.current), "electrical_W": float(v.electrical_power),
            "prop_eff": float(v.aero.efficiency), "motor_eff": float(v.motor_efficiency), "current_limited": bool(v.current_limited)}


# ----------------------------------------------------------------------------------------------- notebook data (13, 14)
def pressure_hull_theory(p: dict, max_von_mises_MPa: float | None = None) -> dict:
    """The pressure hull without a solver: 13 cell 16's thin-shell theory and collapse table (yield and Windenburg-Trilling
    elastic buckling of the cylinder; the caps and plate joints are not covered), cell 26's ring (breathing) mode and
    cell 28's pressure-vs-depth table. ``max_von_mises_MPa`` is the FEA's peak at the rated depth (cell 28's hotspot per
    MPa, linear); without it the hotspot and SF_yield columns are None."""
    AL = talos.Material(**AL6061)
    P_DESIGN = RHO_W * G * DESIGN_DEPTH_M / 1e6                                       # cell 15
    R_I, T_SH, D_CAP, L_PH = p["diameter"] / 2 - 12.0, p["shell"], p["end_cap_depth"], p["pressure_hull_length"]  # noqa: F841
    r_m = R_I - T_SH / 2                                                              # cell 16
    hoop = P_DESIGN * r_m / T_SH; axial = hoop / 2; vm_theory = math.sqrt(hoop**2 + axial**2 - hoop * axial)  # noqa: E702
    dr_theory = P_DESIGN * r_m**2 / (AL.youngs_modulus * T_SH) * (1 - AL.poissons_ratio / 2)
    D_m = 2 * r_m
    p_yield = AL.yield_strength * T_SH / r_m
    p_buckle = 2.42 * AL.youngs_modulus * (T_SH / D_m) ** 2.5 / ((1 - AL.poissons_ratio**2) ** 0.75
                                                                * (L_PH / D_m - 0.45 * math.sqrt(T_SH / D_m)))
    collapse = {"design depth": P_DESIGN, "test depth": TEST_FACTOR * P_DESIGN, "yield (p r / t = σ_y)": p_yield,
                "elastic buckling (Windenburg–Trilling)": p_buckle}
    f_ring = math.sqrt(AL.youngs_modulus * 1e6 / (AL.density * 1e12 * (1 - AL.poissons_ratio**2))) / (2 * math.pi * r_m / 1000)  # cell 26
    HOTSPOT_PER_MPA = max_von_mises_MPa / P_DESIGN if max_von_mises_MPa is not None else None                  # cell 28
    table_depths = [0, 50, 100, 150, DESIGN_DEPTH_M, TEST_FACTOR * DESIGN_DEPTH_M]
    pressure_table = [{"depth_m": d, "gauge_MPa": RHO_W * G * d / 1e6, "absolute_bar": (P_ATM + RHO_W * G * d) / 1e5,
                       "hull_hotspot_MPa": HOTSPOT_PER_MPA * RHO_W * G * d / 1e6 if HOTSPOT_PER_MPA is not None else None,
                       "SF_yield": (AL.yield_strength / (HOTSPOT_PER_MPA * RHO_W * G * d / 1e6) if d else float("inf"))
                       if HOTSPOT_PER_MPA is not None else None,
                       "SF_buckling": p_buckle / (RHO_W * G * d / 1e6) if d else float("inf")} for d in table_depths]
    return {"design_depth_m": DESIGN_DEPTH_M, "design_pressure_MPa": P_DESIGN, "test_factor": TEST_FACTOR,
            "test_depth_m": TEST_FACTOR * DESIGN_DEPTH_M, "material": dict(AL6061), "mean_radius_mm": r_m,
            "theory": {"hoop_MPa": hoop, "axial_MPa": axial, "von_mises_MPa": vm_theory, "radial_displacement_mm": dr_theory},
            "collapse": {k: {"pressure_MPa": v, "depth_m": v * 1e6 / (RHO_W * G), "factor on design": v / P_DESIGN}
                         for k, v in collapse.items()},
            "p_yield_MPa": p_yield, "p_buckle_MPa": p_buckle,
            "collapse_depth_m": {"yield": p_yield * 1e6 / (RHO_W * G), "elastic_buckling": p_buckle * 1e6 / (RHO_W * G)},
            "ring_mode_hz": f_ring, "hotspot_MPa_per_MPa": HOTSPOT_PER_MPA, "pressure_vs_depth": pressure_table}


def blade_modes(f_air) -> dict:
    """13 cell 25: the blade's modes in air and, times the assumed added-mass factor 0.65, in water."""
    f_air = [float(f) for f in f_air]
    f_water = [f * ADDED_MASS_FACTOR for f in f_air]
    return {"modes_hz_air": f_air, "added_mass_factor": ADDED_MASS_FACTOR, "modes_hz_water_assumed": f_water}


def blade_margins(prop, pts: dict, f_water) -> dict:
    """13 cell 26's table (the Campbell plot left out): blade-pass at each point, the nearest wet blade mode, the margin
    to it and the amplification at 2 % damping."""
    blade_structure = chronos.Structure(tuple(f_water), damping_ratio=BLADE_DAMPING, source="Talos modal x added-mass factor 0.65")
    rows = {k: {"BPF_hz": prop.blades * pt.rpm / 60,
                "nearest_blade_mode_hz": blade_structure.nearest_mode(prop.blades * pt.rpm / 60),
                "margin": blade_structure.margin(prop.blades * pt.rpm / 60),
                "amplification": float(blade_structure.amplification(prop.blades * pt.rpm / 60)[0])}
            for k, pt in pts.items()}
    return {"modes_hz_water_assumed": [float(f) for f in f_water], "damping_ratio": BLADE_DAMPING,
            "points": {k: {c: float(v) for c, v in r.items()} for k, r in rows.items()}}


def open_water(prop, sec, rpm: float) -> dict:
    """14 cell 6 (the plot left out): K_T, K_Q and the open-water efficiency against the advance ratio J at ``rpm`` (the
    cruise rpm), the inflow from 0.05 to 1.3 x n D, in sea water (Boreas BEMT). 14 does it for its own blade section;
    this is the drive's."""
    Js, KT, KQ, ETA = [], [], [], []
    n_c = rpm / 60
    Vs = np.linspace(0.05, 1.3, 30) * n_c * prop.diameter
    for v in Vs:
        op = boreas.solve(prop, sec, rpm, v, RHO_W)
        Js.append(op.advance_ratio); KT.append(op.thrust / (RHO_W * n_c**2 * prop.diameter**4))       # noqa: E702
        KQ.append(op.torque / (RHO_W * n_c**2 * prop.diameter**5)); ETA.append(op.efficiency)         # noqa: E702
    return {"rpm": float(rpm), "rho": RHO_W, "inflow_m_s": [float(v) for v in Vs], "J": [float(j) for j in Js],
            "KT": [float(k) for k in KT], "KQ": [float(k) for k in KQ], "eta0": [float(e) for e in ETA]}


def noise_rows(prop, pts: dict, V_MAX: float) -> tuple[dict, dict, dict]:
    """13 cell 23 (the plot left out): at cruise and top speed, Gutin's tones and the broadband allowance at 1 m broadside
    in dB re 1 uPa (``propeller.noise``, the cell's lines), their total and the cavitation number at 0.7 R at 5 m and at
    the rated depth, in the cell's columns. Also each point's tone spectrum (the cell plots cruise's) and both whole
    cavitation checks (with ``rpm_at_inception``)."""
    WAKE, speed = DRIVE["wake"], {"cruise": V_CRUISE, "full": V_MAX}
    rotor = pr.noise(prop, pts, medium=boreas.SEA_WATER, distance=NOISE_DIST, angle_deg=NOISE_ANGLE, harmonics=NOISE_HARMONICS)
    rows, tones, cav = {}, {}, {}
    for name, pt in pts.items():
        n, V = rotor[name], speed[name]
        shallow = boreas.cavitation(prop, pt.rpm, WAKE * V, CAV_DEPTHS_M["5m"], boreas.SEA_WATER, cp_min=CP_MIN)
        deep = boreas.cavitation(prop, pt.rpm, WAKE * V, CAV_DEPTHS_M["rated_depth"], boreas.SEA_WATER, cp_min=CP_MIN)
        rows[name] = {"rpm": n["rpm"], "BPF_hz": n["BPF_hz"], "tonal_dB_re_1uPa_1m": n["tonal_dB"], "broadband_dB": n["broadband_dB"],
                      "total_dB": n["one_rotor_dB"], "sigma_at_5m": shallow["cavitation_number"], "cavitates_at_5m": shallow["cavitates"],
                      "sigma_at_rated_depth": deep["cavitation_number"], "cavitates_at_rated_depth": deep["cavitates"]}
        tones[name] = {"frequency_hz": n["frequency_hz"], "spl_db": n["spl_db"]}
        cav[name] = {"5m": shallow, "rated_depth": deep}
    return rows, tones, cav


def run(*, fidelity: str = "full", sub: dict | None = None, run_cfd: bool = True, run_fea: bool = True, processors: int = 1,
        jobs: int = 1, threads: int = 1, out: Path | None = None, vida_path: Path | None = None, include: str = "results",
        export: bool = True, export_path: Path | None = None, force: bool = False, redo=(), progress: bool = True) -> Assembly:
    out = Path(out or RUNS / NAME).resolve()
    vida_path = Path(vida_path or DATA / f"{NAME}.vida")
    if force and out.exists():
        shutil.rmtree(out)
    prior = vida.load(vida_path) if vida_path.is_file() and not force else None
    root = build(fidelity, sub)
    root.reuse(prior)
    for path in redo:
        if path in ("hull", "propeller"):
            root.child(path).forget()

    hn = root.child("hull")
    p = hn.params["sub"]
    sd = Submarine()
    files, info = {}, {}
    for part, fmts, tol in (("vehicle", ("step", "stl"), 0.2), ("body", ("step",), 0.2), ("pressure_hull", ("step",), 0.2)):
        q = dict(p, part=part)
        files[part], info[part] = export_kept(lambda q=q: sd.generate(**q), q, out / "cad" / part, part, formats=fmts, stl_tolerance=tol)
        hn.attach(f"{part}.step", files[part]["step"], "geometry")
    if not hn.results:
        hn.record(vehicle_volume_L=info["vehicle"]["volume_mm3"] * 1e-6, wetted_surface_m2=info["vehicle"]["surface_area_mm2"] * 1e-6,
                  body_surface_m2=info["body"]["surface_area_mm2"] * 1e-6, **mass_and_trim(info["vehicle"], info["pressure_hull"]))
    if "mass_budget" not in hn.results:                           # 13 cell 6: the items with their positions (a saved hull)
        hn.record(mass_budget=mass_and_trim(info["vehicle"], info["pressure_hull"])["mass_budget"])
    L_M, D_M = p["length"] / 1000, p["diameter"] / 1000
    S_WET, S_BODY, CB = hn.results["wetted_surface_m2"], hn.results["body_surface_m2"], np.asarray(hn.results["CB_mm"])

    # the vehicle nose-upstream in CFD (13 cell 10)
    hull_cfd_values = dict(velocity=V_CRUISE, kinematic_viscosity=NU_W, density=RHO_W, reference_area=S_WET, reference_length=L_M / 4,
                           center_of_rotation=(-CB[0] / 1000, 0.0, 0.0), residual_target=1e-4, **HULL_CFD[fidelity])
    hc = add_after(root, Assembly("hull_cfd", "vehicle_cfd", params={"sub": p, "cfd": hull_cfd_values}), prior, redo)
    nose = None
    if not hc.results.get("complete"):
        nose, _ = export_kept(lambda: H.nose_upstream(sd.generate(**dict(p, part="vehicle"))), {"sub": p},
                              out / "cad" / "nose_upstream", "vehicle_nose_upstream", formats=("stl",), stl_tolerance=0.2)
        hc.attach("vehicle_nose_upstream.stl", nose["stl"], "geometry")
        case = aeromant.CFDCase("rans_ksst_external", nose["stl"], hull_cfd_values, workdir=out / "hull_cfd" / "cruise",
                                geometry_units="mm", environment=environment(run_cfd))
        cfd_node(hc, case, ("drag_force_N", "Cd", "converged", "mesh_cells"), run=run_cfd, processors=processors, progress=progress)

    # the pressure hull at the rated depth (13 cell 15)
    P_DESIGN = RHO_W * G * DESIGN_DEPTH_M / 1e6
    pf = add_after(root, Assembly("pressure_hull_fea", "pressure_hull", params={"sub": p, "material": AL6061, "pressure_MPa": P_DESIGN,
                                                                                "element_mm": ELEMENT["pressure_hull"][fidelity]}),
                   prior, redo)
    if not pf.results.get("complete"):
        step = files["pressure_hull"]["step"]
        tinfo = talos.inspect_step(step, "mm-N-MPa")
        R_I, T_SH, D_CAP, L_PH = p["diameter"] / 2 - 12.0, p["shell"], p["end_cap_depth"], p["pressure_hull_length"]
        x_flat = D_CAP * math.cos(math.asin(0.12))
        wet = [s.tag for s in tinfo.surfaces if s.kind != "Plane" and s.bbox_max[1] > R_I - T_SH / 2]
        regions = [talos.Surfaces("wet", wet), talos.SurfacesOnPlane("stern_plate", "x", -x_flat, tol=0.05),
                   talos.SurfacesOnPlane("bow_plate", "x", L_PH + x_flat, tol=0.05)]
        model = talos.StructuralModel(step, "mm-N-MPa", talos.Material(**AL6061), regions, [talos.FixedSupport("stern_plate")],
                                      [talos.Pressure("wet", P_DESIGN)], talos.MeshSettings(element_size=pf.params["element_mm"]),
                                      name="design_depth")
        solve_fea(pf, {"design_depth": model}, out / "pressure_hull_fea", run=run_fea, threads=threads, progress=progress)

    # the drive, the propeller, its CFD at cruise and one blade at top speed (13 cells 13, 18, 25)
    prop, sec, battery, pts, perf = drive(L_M, D_M, S_WET, S_BODY)
    ps = pr.get(DRIVE["propeller"])
    pn = root.child("propeller")
    pfiles, bfiles = pr.cad_files(ps, out / "propeller"), pr.cad_files(ps, out / "blade", blades=1)
    pn.attach("propeller.step", pfiles["step"], "geometry")
    if not pn.results:
        pn.record(describe=prop.describe())
    rc = add_after(root, Assembly("rotor_cfd", "rotor", params=pr.rotor_params(ps, pts["cruise"].rpm, airspeed=DRIVE["wake"] * V_CRUISE,
                                                                               medium="sea water", fidelity=fidelity)), prior, redo)
    if not rc.results.get("complete"):
        case = pr.rotor_case(pfiles["stl_axis_x"], rc.params, out / "rotor_cfd" / "cruise", environment(run_cfd))
        cfd_node(rc, case, ("thrust_N", "torque_Nm", "power_W", "efficiency", "converged", "mesh_cells"), run=run_cfd,
                 processors=processors, progress=progress)
    loads = pr.blade_loads(prop, pts["full"])
    bl = add_after(root, Assembly("blade_fea", "blade", params={"spec": ps.to_dict(), "loads": loads, "material": AL6061,
                                                                "element_mm": ELEMENT["blade"][fidelity]}), prior, redo)
    if not bl.results.get("complete"):
        model = pr.blade_model(ps, bfiles["step"], loads, element_mm=bl.params["element_mm"], material=AL6061,
                               name="blade_top_speed", root_gap_mm=2.0)
        solve_fea(bl, {"blade_top_speed": model}, out / "blade_fea", run=run_fea, threads=threads, progress=progress)
    if bl.results.get("complete") and "modes_hz_air" not in bl.results:   # 13 cell 25: the same model and mesh, 4 modes
        if run_fea:
            bm = pr.blade_model(ps, bfiles["step"], loads, element_mm=bl.params["element_mm"], material=AL6061,
                                name="blade_top_speed", root_gap_mm=2.0)
            mdir = out / "blade_fea" / "modal"
            mr = solve_modes(bm, mdir, n_modes=BLADE_MODES, mesh_from=out / "blade_fea" / "blade_top_speed", threads=threads,
                             progress=progress)
            if mr.ok:
                bl.attach("modal", mdir, "mesh")
                bl.record(**blade_modes(mr.metrics["frequencies_hz"]))
            else:
                bl.not_run(f"modes: {mr.messages[-1] if mr.messages else mr.status}")
        else:
            bl.not_run("modes: run_fea=False")

    # the vehicle with its propeller as a rotor disk (run_scenario sub(), this tree's numbers)
    sp = scene_params(S_WET, CB[0] / 1000, L_M, pts["cruise"].rpm, prop, sec, SCENE_ITERATIONS[fidelity])
    sc = add_after(root, Assembly("scene_cfd", "rotor_disk", params={"sub": p, "cfd": sp}), prior, redo)
    if not sc.results.get("complete"):
        if nose is None:
            nose, _ = export_kept(lambda: H.nose_upstream(sd.generate(**dict(p, part="vehicle"))), {"sub": p},
                                  out / "cad" / "nose_upstream", "vehicle_nose_upstream", formats=("stl",), stl_tolerance=0.2)
        case = aeromant.CFDCase("hull_rotor_disk", nose["stl"], sp, workdir=out / "scene_cfd" / "cruise", geometry_units="mm",
                                environment=environment(run_cfd))
        cfd_node(sc, case, ("drag_force_N", "converged", "mesh_cells"), run=run_cfd, processors=processors, progress=progress)

    root.record(points={k: _point(v) for k, v in pts.items()}, performance=perf)
    dt = root.results.get("dive_thrust")                          # 13 cell 29 (17 s): kept while the tree and the drive are the same
    if dt is None or not np.isclose(dt["cruise_throttle"], float(pts["cruise"].throttle), rtol=1e-12, atol=0) \
            or not np.isclose(dt["speed_m_s"][-1], 1.05 * perf["top_speed_m_s"], rtol=1e-12, atol=0):
        root.record(dive_thrust=dive_thrust(propulsion()[3], lambda V: H.body_of_revolution_resistance(V, L_M, D_M, S_WET, S_BODY),
                                            pts["cruise"], perf["top_speed_m_s"]))
    grid = boreas.performance_map(prop, sec, MAP_RPM, MAP_V, RHO_W)                    # about 3 s
    noise, tones, cav = noise_rows(prop, pts, perf["top_speed_m_s"])                    # 13 cell 23
    stress = ((pf.results.get("stress") or {}).get("design_depth") or {}).get("max_von_mises_MPa")
    f_water = bl.results.get("modes_hz_water_assumed")
    root.record(pressure_hull=pressure_hull_theory(p, stress),                         # 13 cells 16, 26, 28
                open_water=open_water(prop, sec, pts["cruise"].rpm), open_water_map=grid,   # 14 cell 6
                cavitation={"depths_m": dict(CAV_DEPTHS_M), "cp_min": CP_MIN, "radial_station": 0.7,
                            "inflow_m_s": {"cruise": DRIVE["wake"] * V_CRUISE, "full": DRIVE["wake"] * perf["top_speed_m_s"]},
                            "points": cav},
                noise={"distance_m": NOISE_DIST, "angle_deg": NOISE_ANGLE, "harmonics": NOISE_HARMONICS,
                       "reference": "dB re 1 uPa", "points": noise, "tones": tones},
                blade_structure=blade_margins(prop, pts, f_water) if f_water else None,   # 13 cell 26
                drive_wake={"inflow_ratio": DRIVE["wake"], "wake_fraction": round(1 - DRIVE["wake"], 12),
                            "meaning": "the propeller's inflow is inflow_ratio x vehicle speed (drive.wake); "
                                       "wake_fraction = 1 - inflow_ratio (notebook 14's W_MEAN)"})
    root.meta["workflow"] = f"assemblies.workflows.{NAME}"
    root.meta["saved_to"] = str(root.save(vida_path, include=include))
    if export:                                                    # plain data for notebooks (assemblies.results)
        root.meta["results_to"] = str(results.write(root, vida_path))
    if export:
        path = Path(export_path or DATA / f"{NAME}.json")
        boreas.export(path, prop, sec, motor=pr.motor(DRIVE["motor"]), battery=battery, points=pts, map=grid, rho=RHO_W,
                      notes=f"submarine from assemblies.workflows.{NAME} ({fidelity}); wake {DRIVE['wake']}").raise_for_status()
        root.meta["exported_to"] = str(path)
    return root


def _parser():
    return parser("The submarine: hull, trim, CFD, pressure-hull FEA, drive, rotor CFD, blade FEA, scene")


if __name__ == "__main__":
    raise SystemExit(main(run, _parser(), DATA / f"{NAME}.vida"))
