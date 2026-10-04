"""The Onager series' sizing outside CAD, FEA and MuJoCo, as notebooks 20-23 compute it.

* the Sentinel (20): wheel mode, walking mode, the stand-up and the thermal duty (cells 10, 11, 12, 14, 15, 26, 27),
  the corner's quarter car over three terrains and the rock strike it gives (cells 17, 22), the stub axle and knee
  pin by hand (cell 23);
* what the standing ChironLab episode gives every machine (20 cell 8, 21 / 22 / 23 cell 6): CG, corner loads, rear
  share, sag;
* the Atlas (21): the lift drive (cell 12), and on the standing CG the load chart, the stance table and the ride with
  a pallet (cells 8, 10, 18);
* the Manus (22): the arm torques holding the log, the heaviest log, the cutting envelope, the shoulder servo's mode
  (cells 8, 10, 12, 14);
* the Sweeper (23): power and endurance, the arms' reach and torques, the pick-up table, the unbalance forcing
  (cells 8, 10, 15, 21).

Lifted as they were: each cell's code in a function, its globals as arguments, the inputs the cells set as the module
constants below (the export names of each notebook's last cell). The cells' own copies of the quarter car, the
resistance, the hub traction, the wheel-leg torques and the pins are the ones already lifted to ``road_wheel`` and
``leg`` (tested against those cells), cell function -> shared helper:

* 20 cell 10 ``resistance`` -> ``rw.rolling_resistance``, ``tractive`` -> ``rw.hub_tractive``;
* 20 cell 14 ``wheel_peak`` -> ``leg.foot_peak``, ``leg_torques`` -> ``leg.two_link_torques``;
* 20 cell 17 and 21 cell 18 ``iso8608_profile``, ``add_rocks``, ``add_drop``, ``quarter_car`` -> the ``rw`` functions of
  the same names;
* 20 cell 23 ``pin_bending``, ``pin_shear`` -> ``leg.pin_bending``, ``leg.pin_shear``.

``tests/test_data_onager.py`` runs those cells as they are and compares. Plots, prints, the FEA, the CFD and the MuJoCo
runs stay where they were.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import actuators as act, leg, road_wheel as rw
from . import onager_atlas_robot as oar, onager_atlas_scenario as oas, onager_manus_robot as omr
from . import onager_manus_scenario as oms, onager_robot as orb, onager_sweeper_cfd as cfd
from . import onager_sweeper_controller as swc, onager_sweeper_robot as osr, onager_sweeper_scenario as oss

G = 9.81

# ------------------------------------------------------------------------------------------------ the inputs
#: Notebook 20's inputs to the drive: the datasheet speed (cell 2), resistance and battery (cells 10, 12), the gaits and
#: the stance (cells 14, 15), the stand-up's crouch (cell 26).
SENTINEL_DRIVE = {
    "V_SPEC_KMH": 38, "RHO_AIR": 1.2, "C_D": 1.0, "A_FRONT": 1.9,
    "C_RR": {"asphalt": 0.015, "gravel": 0.03, "soft soil": 0.08}, "MU_TYRE": 0.8, "GRADES_PCT": [0, 10, 20, 30],
    "E_BATT_WH": 6000.0, "USABLE": 0.9, "ETA_DRIVE": 0.85,
    "P_HOTEL": {"on station (sensors, computer, radio)": 220.0, "driving (adds the mast stowed, lidar spinning)": 350.0},
    "BETA": {"walk": 0.75, "amble": 0.6}, "STANCE": 0.50, "SWING_SPEED_FRACTION": 0.6, "CROUCH": 0.20,
}
#: Notebook 20 cell 17: the tyre and the three terrains (ISO 8608 class, Gd(n0), speed [m/s], length [m], rocks
#: (height, width, spacing) [m], drop (at, depth) [m], seed).
K_TYRE_20 = 150e3
TERRAINS = {
    "asphalt road": {"class": "A/B", "gd": 32e-6, "speed": 10.0, "length": 150.0, "rocks": None, "drop": None, "seed": 1},
    "gravel track": {"class": "C", "gd": 256e-6, "speed": 6.0, "length": 150.0, "rocks": (0.05, 0.25, 8.0), "drop": None, "seed": 2},
    "rocky field": {"class": "E", "gd": 4096e-6, "speed": 2.5, "length": 120.0, "rocks": (0.10, 0.30, 3.0), "drop": (90.0, 0.15), "seed": 3},
}
PIN_YIELD_MPA = 650.0                    # 20 cell 22: STEEL_PIN, 42CrMo4
ROCK_STRIKE = "rock strike, rocky field (quarter car)"
STANDING = "standing, rear wheel"
#: Notebook 21's inputs: the rated load (cell 2), the tipping safety factor (cell 8), the mast friction (cell 12), the
#: tyre and the gravel road of the ride (cell 18: iso8608_profile(60.0, 0.005, 256e-6, 5)).
ATLAS = {"m_load": 200, "SF_TIP": 1.5, "FRICTION": 0.08, "K_TYRE": 150e3, "road": [60.0, 0.005, 256e-6, 5]}
#: Notebook 22's inputs: the wire classes (cell 10), the outboard masses' lever (cell 12), the log and the wire (scenario).
MANUS = {"WIRES": {"fence, mild Ø 2.5 (400 MPa)": (2.5, 400), "barbed, Ø 2.5 (700 MPa)": (2.5, 700),
                   "fence, high-tensile Ø 3.15 (1200 MPa)": (3.15, 1200), "high-tensile Ø 4.0 (1200 MPa)": (4.0, 1200),
                   "rebar Ø 6 (550 MPa)": (6.0, 550)},
         "lever_out": 0.55, "LOG": dict(oms.LOG)}
#: Notebook 23's inputs (cells 8, 21) and the suction it falls back to without OpenFOAM (cell 13).
SWEEPER = {"N_bristle": 100.0, "mu_bristle": 0.5, "dp_downstream": 1500.0, "eta_fan": 0.55, "eta_motor": 0.9,
           "fan_rpm": 4500, "P_elec": 150.0, "E_usable_Wh": 6.0e3 * 0.9, "e_disc": 2.0e-3, "m_imp": 3.0, "grade": 6.3e-3,
           "k_tyre": 290e3, "suction": dict(cfd.SUCTION)}


# ------------------------------------------------------------------------------------------------ the stand (MuJoCo)
def stand_summary(log) -> dict:
    """What the standing episode's log gives (20 cell 8; 21, 22, 23 cell 6): the CG ``com_m`` [x, y, z] (``cg_m`` x and
    z: 21's X_CG, Z_CG), the corner loads, the rear pair's share, the heavier rear wheel (20 cell 22's standing case)
    and the hull's sag below the nominal pose."""
    fz_stand = np.asarray(log["foot_force"])[-1, :, 2]
    com_stand = np.asarray(log["com"])[-1]
    sag = np.asarray(log["body_pos"])[0, 0, 2] - np.asarray(log["body_pos"])[-1, 0, 2]
    REAR_SHARE = fz_stand[2:].sum() / fz_stand.sum()
    return {"com_m": [float(c) for c in com_stand], "cg_m": {"x": float(com_stand[0]), "z": float(com_stand[2])},
            "corner_loads_N": {str(w): float(f) for w, f in zip(log["feet"], fz_stand)}, "rear_share": float(REAR_SHARE),
            "rear_wheel_max_N": float(fz_stand[2:].max()), "sag_m": float(sag)}


# ------------------------------------------------------------------------------------------------ the Sentinel (20)
def sentinel_corner(M_TOTAL: float, budget: dict, geo: dict, K_TYRE: float = K_TYRE_20) -> dict:
    """20 cell 17's corner: the unsprung and sprung masses, the knee lever, the knee servo as the spring and damper at
    the wheel, heave and wheel hop."""
    HUB = act.get(orb.WHEEL_MOTOR)
    L2 = geo["L2"]
    M_UNSPRUNG = orb.PARTS_KG["wheels: tyres + rims 4x"] / 4 + HUB.mass_g / 1000 + budget["lower legs 4x (Al 7075-T6, from CAD)"] / 8
    M_SPRUNG_CORNER = (M_TOTAL - 4 * M_UNSPRUNG) / 4
    LEVER_KNEE = L2 * math.sin(geo["a2"])                   # vertical travel of the wheel per rad of knee angle [m]
    K_SUSP = orb.LEG_KP / LEVER_KNEE**2                      # N/m at the wheel: the knee servo as the spring
    C_SUSP = orb.LEG_KD / LEVER_KNEE**2
    ZETA = C_SUSP / (2 * math.sqrt(K_SUSP * M_SPRUNG_CORNER))
    f_heave = math.sqrt(K_SUSP / M_SPRUNG_CORNER) / (2 * math.pi)
    f_hop = math.sqrt((K_TYRE + K_SUSP) / M_UNSPRUNG) / (2 * math.pi)
    return {"m_unsprung_kg": M_UNSPRUNG, "m_sprung_corner_kg": M_SPRUNG_CORNER, "lever_knee_m": LEVER_KNEE,
            "k_susp_N_m": K_SUSP, "c_susp_N_s_m": C_SUSP, "zeta": ZETA, "k_tyre_N_m": K_TYRE, "f_heave_hz": f_heave,
            "f_hop_hz": f_hop}


def sentinel_drive(M_TOTAL: float, budget: dict, geo: dict, p: dict | None = None, inputs: dict = SENTINEL_DRIVE,
                   REAR_SHARE: float | None = None) -> dict:
    """20 cells 10, 11, 12 (wheel mode), 14, 15 (walking mode), 26 (the stand-up from the crouch, one knee in Chiron's
    servo model) and 27 (thermal duty), in cell 33's export sections ``wheel_mode``, ``walking_mode``, ``stand_up``,
    ``thermal``. ``REAR_SHARE`` (cell 8, the standing ChironLab episode) adds what the standing knee torque gives: the
    torque row 'standing, rear wheel', ``holding_power_W`` and the thermal row 'standing, brakes off (knee)'; without
    it they are left out (``holding_power_W`` None). ``p``: the Sentinel's design (the standing knee angle)."""
    from vegeta.chiron import servo as cservo

    W = M_TOTAL * G
    V_SPEC = inputs["V_SPEC_KMH"]                               # SPEC["speed, wheels [km/h]"]
    # cell 10: resistance against the hub motors' line
    HUB = act.get(orb.WHEEL_MOTOR); R_W = geo["r_wheel"]
    W0_HUB = HUB.no_load_rpm * 2 * math.pi / 60
    RHO_AIR, C_D, A_FRONT = inputs["RHO_AIR"], inputs["C_D"], inputs["A_FRONT"]
    C_RR = inputs["C_RR"]
    MU_TYRE = inputs["MU_TYRE"]
    GRADES_PCT = inputs["GRADES_PCT"]

    def resistance(v, c_rr, grade_pct):
        return rw.rolling_resistance(v, W, c_rr, grade_pct, rho_air=RHO_AIR, cd=C_D, frontal_area_m2=A_FRONT)

    def tractive(v, n_motors=4):
        return rw.hub_tractive(v, stall_Nm=HUB.stall_Nm, wheel_radius_m=R_W, no_load_omega=W0_HUB, n_motors=n_motors)

    F_TRAC_RATED = 4 * HUB.rated_Nm / R_W
    v_grid = np.linspace(0, 14, 400)
    top_speed = {}
    for gr in GRADES_PCT:
        for surf in ("asphalt", "gravel"):
            res = np.array([resistance(v, C_RR[surf], gr) for v in v_grid])
            trac = np.array([tractive(v) for v in v_grid])
            ok = np.where(trac >= res)[0]
            top_speed[(gr, surf)] = v_grid[ok[-1]] if len(ok) else 0.0
    grade_cont = math.degrees(math.asin(min(1.0, (F_TRAC_RATED - C_RR["gravel"] * W) / W)))

    # cell 11: the 0-38 km/h runs
    I_WHEEL = 0.5 * (orb.PARTS_KG["wheels: tyres + rims 4x"] / 4) * R_W**2 + orb.WHEEL_ARMATURE      # tyre + rim as a disc, plus the rotor
    M_EFF = M_TOTAL + 4 * I_WHEEL / R_W**2

    def accel_run(c_rr, grade_pct=0, v_end=None, t_max=30.0, dt=1e-3):
        v_end = v_end or V_SPEC / 3.6 * 0.98
        t, v, s, E = 0.0, 0.0, 0.0, 0.0
        hist = []
        while v < v_end and t < t_max:
            F = min(tractive(v), MU_TYRE * W) - resistance(v, c_rr, grade_pct)
            tau = min(tractive(v), MU_TYRE * W) / 4 * R_W
            i_motor = tau / HUB.stall_Nm * HUB.stall_A                           # DC-motor estimate: current ∝ torque
            P_el = 4 * (tau * v / R_W + i_motor**2 * (HUB.voltage_V / HUB.stall_A))   # mechanical + copper loss
            hist.append((t, v, tau, i_motor, P_el))
            v += F / M_EFF * dt; s += v * dt; E += P_el * dt; t += dt
        h = np.array(hist)
        return {"time_s": t, "distance_m": s, "energy_Wh": E / 3600, "peak_motor_current_A": h[:, 3].max(), "peak_battery_power_kW": h[:, 4].max() / 1000, "hist": h}
    runs = {"asphalt, flat": accel_run(C_RR["asphalt"]), "gravel, flat": accel_run(C_RR["gravel"]), "gravel, 10 % grade": accel_run(C_RR["gravel"], 10)}
    accel = pd.DataFrame({k: {kk: v for kk, v in r.items() if kk != "hist"} for k, r in runs.items()}).T

    # cell 12: range and endurance
    E_BATT_WH, USABLE, ETA_DRIVE = inputs["E_BATT_WH"], inputs["USABLE"], inputs["ETA_DRIVE"]
    P_HOTEL = inputs["P_HOTEL"]

    def range_km(v_kmh, c_rr):
        v = v_kmh / 3.6
        P = resistance(v, c_rr, 0) * v / ETA_DRIVE + P_HOTEL["driving (adds the mast stowed, lidar spinning)"]
        hours = E_BATT_WH * USABLE / P
        return {"power [W]": P, "hours": hours, "range [km]": hours * v_kmh}
    endurance = pd.DataFrame({f"{v} km/h on {surf}": range_km(v, C_RR[surf]) for v in (11, 20, 38) for surf in ("asphalt", "gravel")}).T
    endurance.loc["on station, 0 km/h"] = {"power [W]": P_HOTEL["on station (sensors, computer, radio)"], "hours": E_BATT_WH * USABLE / P_HOTEL["on station (sensors, computer, radio)"], "range [km]": 0.0}
    mixed = 0.3 * endurance.loc["11 km/h on gravel", "power [W]"] + 0.7 * P_HOTEL["on station (sensors, computer, radio)"]
    endurance.loc["mission: 30 % moving at 11 km/h on gravel, 70 % on station"] = {"power [W]": mixed, "hours": E_BATT_WH * USABLE / mixed, "range [km]": E_BATT_WH * USABLE / mixed * 0.3 * 11}

    # cell 14: joint torques over a stance
    LEG = act.get(orb.LEG_ACTUATOR)
    L1, L2, R_WHEEL = geo["L1"], geo["L2"], geo["r_wheel"]
    H_AXLE = geo["h_axle"]
    BETA = inputs["BETA"]

    def wheel_peak(W, beta):
        return leg.foot_peak(W, beta)

    def leg_torques(x_axle, fx, fz, fy=0.0):
        return leg.two_link_torques(x_axle, fx, fz, L1=L1, L2=L2, h_axle=H_AXLE, r_wheel=R_WHEEL)
    STANCE = inputs["STANCE"]
    xs = np.linspace(geo["axle_x"] + STANCE / 2, geo["axle_x"] - STANCE / 2, 41)
    torque_rows = {}
    for (gname, beta), ls in zip(BETA.items(), ("-", "--")):
        Fz = wheel_peak(W, beta) * np.sin(np.pi * np.linspace(0, 1, len(xs)))
        Fx = 0.3 * Fz * np.cos(np.pi * np.linspace(0, 1, len(xs)))                  # braking then pushing
        th = np.array([leg_torques(x, fx, fz) for x, fx, fz in zip(xs, Fx, Fz)])
        torque_rows[gname] = {"duty": beta, "peak wheel force [N]": wheel_peak(W, beta), "peak shoulder [Nm]": float(np.nanmax(np.abs(th[:, 0]))), "peak knee [Nm]": float(np.nanmax(np.abs(th[:, 1])))}
    tau_stand = None
    if REAR_SHARE is not None:
        tau_stand = leg_torques(geo["axle_x"], 0.0, W / 4 * 2 * REAR_SHARE)
        torque_rows["standing, rear wheel"] = {"duty": 1.0, "peak wheel force [N]": W / 4 * 2 * REAR_SHARE, "peak shoulder [Nm]": abs(tau_stand[0]), "peak knee [Nm]": abs(tau_stand[1])}
    torques = pd.DataFrame(torque_rows).T
    torques["SF on stall (800 Nm)"] = LEG.stall_Nm / torques[["peak shoulder [Nm]", "peak knee [Nm]"]].max(axis=1)

    # cell 15: the walking speed the modules allow, holding the standing pose
    W0_LEG = LEG.no_load_rpm * 2 * math.pi / 60
    SWING_SPEED_FRACTION = inputs["SWING_SPEED_FRACTION"]
    leg_len = H_AXLE + R_WHEEL

    def v_walk_max(beta, stance=STANCE, sweep_fraction=1.0):
        w_sw = SWING_SPEED_FRACTION * W0_LEG
        T_min = sweep_fraction * stance / leg_len / ((1 - beta) * w_sw)
        return stance / (beta * T_min)
    walk_speed = pd.DataFrame({g: {"pure walk (wheels braked) [km/h]": v_walk_max(b) * 3.6, "wheel-walk (wheels roll half the stance) [km/h]": v_walk_max(b, sweep_fraction=0.5) * 3.6,
                                   "stride frequency at the limit [Hz]": v_walk_max(b) / STANCE * b} for g, b in BETA.items()}).T
    hold = None
    if tau_stand is not None:
        f_hold = min(abs(tau_stand[1]) / LEG.stall_Nm, 1.0)
        hold = {"brakes on (self-locking module) [W]": 4 * LEG.holding_power(abs(tau_stand[1])), "brakes off, four knees hold the hull [W]": 4 * f_hold * LEG.stall_A * LEG.voltage_V}

    # cell 26: the stand-up from a crouch (cell 17's corner)
    corner = sentinel_corner(M_TOTAL, budget, geo)
    LEVER_KNEE, M_SPRUNG_CORNER = corner["lever_knee_m"], corner["m_sprung_corner_kg"]
    leg_servo = orb.leg_servo()
    CROUCH = inputs["CROUCH"]
    KT, R_LEG = leg_servo.torque_constant, leg_servo.resistance

    def stand_up(kp=orb.LEG_KP, kd=orb.LEG_KD, dt=1e-3, t_end=4.0, gravity_ff=True):
        q_stand = orb.nominal_qpos(p)["FL_knee"]
        q = q_stand - CROUCH / LEVER_KNEE; qd = 0.0
        I_eq = M_SPRUNG_CORNER * LEVER_KNEE**2 + orb.LEG_ARMATURE          # the corner's mass reflected to the knee
        tau_g = M_SPRUNG_CORNER * G * LEVER_KNEE                            # gravity torque at the knee
        hist = []
        for t in np.arange(0, t_end, dt):
            ff = tau_g if gravity_ff else 0.0
            tau = float(cservo.servo_torque(q, qd, q_stand, 0.0, ff, kp=kp, kd=kd, stall=leg_servo.stall_torque, no_load_speed=leg_servo.no_load_speed))
            i = abs(tau) / KT; P_el = abs(tau * qd) + i**2 * R_LEG
            hist.append((t, (q - q_stand) * LEVER_KNEE, qd, tau, i, P_el))
            qdd = (tau - tau_g) / I_eq
            qd += qdd * dt; q += qd * dt
        return pd.DataFrame(hist, columns=["t", "dz_m", "qd", "tau", "i", "P_el"])
    su = stand_up()
    settled = su[np.abs(su.dz_m) < 0.005].t.min()
    E_standup = getattr(np, "trapezoid", getattr(np, "trapz", None))(su.P_el, su.t) * 4 / 3600

    # cell 27: thermal duty
    rows = {}
    if tau_stand is not None:
        rows["standing, brakes off (knee)"] = {"module": LEG.key, "torque [Nm]": abs(tau_stand[1]), "continuous [Nm]": LEG.rated_Nm, "stall [Nm]": LEG.stall_Nm}
    rows.update({
        "walk peak (knee)": {"module": LEG.key, "torque [Nm]": torques.loc["walk", "peak knee [Nm]"], "continuous [Nm]": LEG.rated_Nm, "stall [Nm]": LEG.stall_Nm},
        "stand-up peak (knee)": {"module": LEG.key, "torque [Nm]": su.tau.abs().max(), "continuous [Nm]": LEG.rated_Nm, "stall [Nm]": LEG.stall_Nm},
        "cruise 38 km/h on asphalt (hub)": {"module": HUB.key, "torque [Nm]": resistance(V_SPEC / 3.6, C_RR["asphalt"], 0) / 4 * R_W, "continuous [Nm]": HUB.rated_Nm, "stall [Nm]": HUB.stall_Nm},
        "20 % grade on gravel (hub)": {"module": HUB.key, "torque [Nm]": resistance(2.0, C_RR["gravel"], 20) / 4 * R_W, "continuous [Nm]": HUB.rated_Nm, "stall [Nm]": HUB.stall_Nm},
        "three-wheel limp, 20 % grade (hub)": {"module": HUB.key, "torque [Nm]": resistance(2.0, C_RR["gravel"], 20) / 3 * R_W, "continuous [Nm]": HUB.rated_Nm, "stall [Nm]": HUB.stall_Nm},
        "0–38 km/h run, peak (hub)": {"module": HUB.key, "torque [Nm]": HUB.stall_Nm * (1 - 0.0), "continuous [Nm]": HUB.rated_Nm, "stall [Nm]": HUB.stall_Nm},
    })
    thermal = pd.DataFrame(rows).T
    thermal["duty"] = np.where(thermal["torque [Nm]"].astype(float) <= thermal["continuous [Nm]"].astype(float), "continuous", np.where(thermal["torque [Nm]"].astype(float) <= thermal["stall [Nm]"].astype(float), "PEAK: minutes, thermal model needed", "EXCEEDS STALL"))

    # cell 33: the export's names
    return {"wheel_mode": {"top_speed_kmh": {f"{gr}% {surf}": v * 3.6 for (gr, surf), v in top_speed.items()}, "continuous_grade_deg": grade_cont,
                           "acceleration": accel.to_dict(orient="index"), "endurance": endurance.to_dict(orient="index")},
            "walking_mode": {"torques": torques.to_dict(orient="index"), "speed_limits": walk_speed.to_dict(orient="index"), "holding_power_W": hold},
            "stand_up": {"crouch_m": CROUCH, "settle_s": float(settled), "peak_torque_Nm": float(su.tau.abs().max()),
                         "peak_current_A": float(su.i.max()), "energy_Wh_4_corners": float(E_standup)},
            "thermal": {k: {kk: (str(vv) if kk == "duty" else vv) for kk, vv in r.items()} for k, r in thermal.to_dict(orient="index").items()}}


def sentinel_terrains(M_TOTAL: float, geo: dict, corner: dict, terrains: dict = TERRAINS) -> dict:
    """20 cell 17: the front corner (``corner``: ``sentinel_corner``) over each terrain, the rear corner on the same
    profile the wheelbase behind, the hull's pitch; the per-terrain ``summary`` (the cell's table, rounded as it is);
    and cell 22's ``rock_strike``: the rocky field's peak tyre force with the road's slope there as the longitudinal
    share."""
    K_SUSP, C_SUSP, K_TYRE = corner["k_susp_N_m"], corner["c_susp_N_s_m"], corner["k_tyre_N_m"]
    M_SPRUNG_CORNER, M_UNSPRUNG = corner["m_sprung_corner_kg"], corner["m_unsprung_kg"]

    def quarter_car(x, z_road, speed, dt=5e-4):
        r = rw.quarter_car(x, z_road, speed, k_susp=K_SUSP, c_susp=C_SUSP, k_tyre=K_TYRE, m_sprung=M_SPRUNG_CORNER,
                           m_unsprung=M_UNSPRUNG, static_corner_N=M_TOTAL * G / 4, dt=dt, g=G)
        return r["t"], r["zr"], r["Fs"], r["Ft"], r["As"], r["travel"], r["zs"]
    sims = {}
    for name, tr in terrains.items():
        x, z = rw.iso8608_profile(tr["length"], 0.005, tr["gd"], tr["seed"])
        if tr["rocks"]: z = rw.add_rocks(x, z, *tr["rocks"], seed=tr["seed"] + 10)
        if tr["drop"]: z = rw.add_drop(x, z, *tr["drop"])
        t, zr, Fs, Ft, As, travel, Zs = quarter_car(x, z, tr["speed"])
        z_rear = np.interp(x - geo["wheelbase"], x, z, left=0.0)                           # the rear corner, the wheelbase behind
        _, _, _, _, _, _, Zs_rear = quarter_car(x, z_rear, tr["speed"])
        pitch = np.degrees(np.arctan((Zs[:len(Zs_rear)] - Zs_rear) / geo["wheelbase"]))
        sims[name] = dict(t=t, zr=zr, Fs=Fs, Ft=Ft, As=As, travel=travel, pitch=pitch, pitch_rate=np.gradient(pitch, t[:len(pitch)]), duration_s=t[-1], speed=tr["speed"])
    summary = pd.DataFrame({k: {"speed_km_h": s["speed"] * 3.6, "tyre_force_max_N": s["Ft"].max(), "tyre_force_min_N": s["Ft"].min(), "leg_force_max_N": s["Fs"].max(),
                                "hull_accel_rms_g": np.sqrt(np.mean(s["As"] ** 2)) / G, "hull_accel_peak_g": np.abs(s["As"]).max() / G, "travel_max_mm": np.abs(s["travel"]).max() * 1000,
                                "pitch_rms_deg": np.sqrt(np.mean(s["pitch"] ** 2)), "pitch_rate_rms_deg_s": np.sqrt(np.mean(s["pitch_rate"] ** 2)),
                                "airborne_%": 100 * np.mean(s["Ft"] <= 1e-6)} for k, s in sims.items()}).T.round(2)
    # cell 22: the rock strike
    rock = sims["rocky field"]; i_rock = int(np.argmax(rock["Ft"]))
    slope_rock = np.gradient(rock["zr"], rock["t"] * rock["speed"])
    rock_strike = dict(fx=float(rock["Ft"][i_rock] * min(abs(slope_rock[i_rock]), 1.0)), fy=0.0, fz=float(rock["Ft"][i_rock]))
    return {"summary": summary.to_dict(orient="index"), "rock_strike": rock_strike}


def pin_geometry(p: dict, yield_MPa: float = PIN_YIELD_MPA) -> dict:
    """What 20 cell 23 reads of the design: the stub axle's and the knee pin's diameters and levers [mm], the pin steel's
    yield."""
    return {"lever_axle_mm": p["wheel_offset"] + p["wheel_width"] / 2,                   # stub axle: cantilever to the wheel centre
            "lever_pin_mm": (p["upper_leg_thickness"] / 2 + 4.0) / 2,                     # knee pin: bending over half the clevis gap
            "axle_diameter": p["axle_diameter"], "pin_diameter": p["pin_diameter"], "yield_MPa": yield_MPa}


def sentinel_pins(cases: dict, pins: dict) -> dict:
    """20 cell 23: the stub axle (cantilever, single shear) and the knee pin (double shear) under each case's resultant,
    von Mises sqrt(bending² + 3 shear²) and the safety factor on the pin steel; ``pins``: ``pin_geometry``."""
    lever_axle, lever_pin = pins["lever_axle_mm"], pins["lever_pin_mm"]
    out = {}
    for name, f in cases.items():
        F = math.sqrt(f["fx"]**2 + f["fy"]**2 + f["fz"]**2)
        s_axle = math.sqrt(leg.pin_bending(F, lever_axle, pins["axle_diameter"])**2 + 3 * leg.pin_shear(F, pins["axle_diameter"], 1)**2)
        s_pin = math.sqrt(leg.pin_bending(F, lever_pin, pins["pin_diameter"])**2 + 3 * leg.pin_shear(F, pins["pin_diameter"])**2)
        out[name] = {"resultant_N": F, "stub_axle_MPa": s_axle, "stub_axle_SF": pins["yield_MPa"] / s_axle,
                     "knee_pin_MPa": s_pin, "knee_pin_SF": pins["yield_MPa"] / s_pin}
    return out


# ------------------------------------------------------------------------------------------------ the Atlas (21)
def atlas_drives(p: dict, cad: dict, inputs: dict = ATLAS) -> dict:
    """21 cell 12: the lift screw at the rated load: force, loaded speed, the time to full height, the energy of a lift
    (DC-motor estimate), in cell 24's ``drives`` names."""
    fl = oar.forklift_geometry(p)
    m_load = inputs["m_load"]
    LIFT = act.get_linear(oar.LIFT)
    m_carriage = oar.FORKLIFT_KG["carriage frame (welded)"] + oar.forklift_masses(p, cad)["forks 2x (S690, CAD)"] + oar.FORKLIFT_KG["lift chain, rollers, guards"]
    FRICTION = inputs["FRICTION"]                       # mast rollers and the screw: 8 % of the load (input)
    F_rated = (m_carriage + m_load) * G * (1 + FRICTION)
    v_rated = LIFT.no_load_mm_s / 1000 * (1 - F_rated / LIFT.stall_N)
    t_full = fl["lift_max"] / v_rated
    kt = LIFT.stall_N / LIFT.stall_A                      # N per A (stall point)
    I = F_rated / kt
    E_lift = (F_rated * fl["lift_max"] + I ** 2 * (LIFT.voltage_V / LIFT.stall_A) * t_full) / 3600
    return {"lift_rated_N": F_rated, "lift_speed_m_s": v_rated, "full_lift_s": t_full, "energy_per_lift_Wh": E_lift}


def atlas_load_chart(M: float, X_CG: float, Z_CG: float, budget: dict, p: dict, inputs: dict = ATLAS) -> dict:
    """21 cells 8 (the load chart: tipping over the front wheels with SF 1.5, static and braking, and the front knees'
    limit), 10 (the stance table) and 18 (the ride of a front corner with and without the pallet) on the standing CG
    of the ChironLab episode (cell 6: ``X_CG``, ``Z_CG``); cell 24's ``load_chart``, ``stance``, ``ride``."""
    geo = orb.geometry(p)
    fl = oar.forklift_geometry(p)
    x_front = geo["shoulder_x"] + geo["axle_x"]                                         # cell 4
    # cell 8
    SF_TIP = inputs["SF_TIP"]
    KNEE = act.get(orb.LEG_ACTUATOR)
    x_rear = -geo["shoulder_x"] + geo["axle_x"]
    LEVER_KNEE = geo["L2"] * math.sin(geo["a2"])
    H_PALLET_CG = oas.PALLET["height"] + 0.5 * oas.PALLET["crate"][2] * oas.PALLET["crate_kg"] / oas.PALLET["total_kg"]   # above the forks' top

    def capacity(c_mm, lift_m=0.25, a=0.0):
        x_l = fl["heel_world_x"] + c_mm / 1000
        h_l = lift_m + fl["fork_t"] + H_PALLET_CG
        tip = M * (G * (x_front - X_CG) - a * Z_CG) / (SF_TIP * (G * (x_l - x_front) + a * h_l))
        # knees: front axle force with a load m: (M g (x_cg - x_rear) + m g (x_l - x_rear)) / wheelbase, per wheel / 2
        wheel_max = KNEE.stall_Nm / LEVER_KNEE
        knee = (2 * wheel_max * geo["wheelbase"] / G - M * (X_CG - x_rear)) / (x_l - x_rear)
        return tip, knee
    chart = pd.DataFrame({f"{c:.0f} mm": {"tipping/1.5, static [kg]": capacity(c)[0], "tipping/1.5, braking 1 m/s², 1.2 m up [kg]": capacity(c, 1.2, 1.0)[0],
                                          "front knees at stall [kg]": capacity(c)[1]} for c in (400, 500, 600, 800)}).T
    chart["capacity [kg]"] = chart.min(axis=1)
    # cell 10
    L1, L2 = geo["L1"], geo["L2"]
    m_load = inputs["m_load"]
    x_l = fl["heel_world_x"] + 0.5
    F_front = (M * G * (X_CG - x_rear) + m_load * G * (x_l - x_rear)) / geo["wheelbase"] / 2       # per front wheel [N]
    stance = pd.DataFrame({name: {"knee [N·m]": F_front * L2 * math.sin(math.radians(b)), "shoulder [N·m]": F_front * abs(-L1 * math.sin(math.radians(a)) + L2 * math.sin(math.radians(b))),
                                  "wheel stiffness kp/(L2 sin a2)² [kN/m]": orb.LEG_KP / (L2 * math.sin(math.radians(b))) ** 2 / 1000,
                                  "shoulder height [m]": L1 * math.cos(math.radians(a)) + L2 * math.cos(math.radians(b)) + geo["r_wheel"]}
                           for name, (a, b) in {"Sentinel 40°/55°": (40, 55), "Atlas 25°/15°": (math.degrees(geo["a1"]), math.degrees(geo["a2"]))}.items()}).T
    # cell 18
    M_UNSPRUNG = orb.PARTS_KG["wheels: tyres + rims 4x"] / 4 + act.get(orb.WHEEL_MOTOR).mass_g / 1000 + budget["lower legs 4x (Al 7075-T6, from CAD)"] / 8
    K_TYRE = inputs["K_TYRE"]
    K_SUSP = orb.LEG_KP / LEVER_KNEE ** 2
    C_SUSP = orb.LEG_KD / LEVER_KNEE ** 2

    def quarter_car(x, z_road, speed, m_s, dt=2e-4):
        r = rw.quarter_car(x, z_road, speed, k_susp=K_SUSP, c_susp=C_SUSP, k_tyre=K_TYRE, m_sprung=m_s, m_unsprung=M_UNSPRUNG, dt=dt, g=G)
        return r["t"], r["As"], r["Ft"]
    x_road, z_road = rw.iso8608_profile(*inputs["road"])
    F_front_empty = M * G * (X_CG - x_rear) / geo["wheelbase"] / 2
    rows = {}
    for label, F_corner, speed in (("empty, 3 m/s", F_front_empty, 3.0), ("rated pallet, 1 m/s", F_front, 1.0), ("rated pallet, 2 m/s", F_front, 2.0)):
        m_s = F_corner / G - M_UNSPRUNG
        t_, As, Ft = quarter_car(x_road, z_road, speed, m_s)
        rows[label] = {"sprung mass [kg]": m_s, "heave [Hz]": math.sqrt(K_SUSP / m_s) / (2 * math.pi), "hull accel RMS [g]": np.sqrt(np.mean(As ** 2)) / G,
                       "hull accel min [g]": As.min() / G, "time below −g [%]": 100 * np.mean(As < -G), "wheel load max [N]": Ft.max(),
                       "front knee max / stall": Ft.max() * LEVER_KNEE / KNEE.stall_Nm, "wheel airborne [%]": 100 * np.mean(Ft <= 1e-6)}
    ride = pd.DataFrame(rows).T
    return {"load_chart": chart.to_dict(orient="index"), "stance": stance.to_dict(orient="index"), "ride": ride.to_dict(orient="index")}


# ------------------------------------------------------------------------------------------------ the Manus (22)
def manus_sizing(p: dict, cad: dict, inputs: dict = MANUS) -> dict:
    """22 cell 8 (the static joint torques holding the log, pin 0.75 m below the shoulder, pincer down, against the
    reach; the heaviest log at 0.7 m with SF 1.5 on the shoulder), cell 10 (the cutting envelope per wire class) and
    cells 12, 14 (the shoulder servo's own mode with the outboard masses and the log); cell 20's names."""
    ag = omr.arm_geometry(p)
    m_log = inputs["LOG"]["mass_kg"]
    # cell 8
    x_sh = ag["x"]
    m_arm = omr.arm_masses(p, cad)

    def joint_torques(r, z_pin, elev, load_kg):
        """Static shoulder, elbow, wrist torques [N m] with the pin at (r forward, z_pin above the shoulder), the load at the pin."""
        q = omr.arm_ik((x_sh + r, ag["y"], ag["z"] + z_pin), elev, "L", p)
        if q is None:
            return (np.nan,) * 3
        _, sh, el, wr = q
        p1, p2, p3 = -sh, -(sh + el), -(sh + el + wr)
        elbow = ag["Lu"] * math.cos(p1); wrist = elbow + ag["Lf"] * math.cos(p2); pin = wrist + ag["Lp"] * math.cos(p3)
        jaw_c = pin + 0.1 * math.cos(p3)
        items = [(m_arm["upper arm (Al 6082, CAD)"], ag["Lu"] / 2 * math.cos(p1)), (m_arm["elbow drive (harmonic 150 Nm, brake)"], elbow),
                 (m_arm["forearm (Al 6082, CAD)"], elbow + ag["Lf"] / 2 * math.cos(p2)), (m_arm["wrist drive (harmonic 60 Nm, brake)"], wrist),
                 (m_arm["palm (wrist housing, screw mount)"] + m_arm["jaw drive (jaw screw 6 kN)"], wrist + ag["Lp"] / 2 * math.cos(p3)),
                 (m_arm["jaws 2x (tool steel, CAD)"], jaw_c), (load_kg, jaw_c)]
        tau_sh = G * sum(mi * xi for mi, xi in items)
        tau_el = G * sum(mi * (xi - elbow) for mi, xi in items[2:])
        tau_wr = G * sum(mi * (xi - wrist) for mi, xi in items[4:])
        return abs(tau_sh), abs(tau_el), abs(tau_wr)
    reach = np.linspace(0.3, 1.0, 15)
    rows = {}
    for r in reach:
        rows[f"{r:.2f} m"] = dict(zip(("shoulder", "elbow", "wrist"), joint_torques(r, -0.75, -math.pi / 2, m_log)))
    tq = pd.DataFrame(rows).T
    stall = {j: act.get(omr.ARM_ACTUATORS[j]).stall_Nm for j in ("shoulder", "elbow", "wrist")}
    heaviest = min(((stall["shoulder"] / 1.5 - joint_torques(0.7, -0.75, -math.pi / 2, 0)[0]) / (G * (0.7 + 0.1)), 1e9), key=float)
    # cell 10: cutting
    JAW = act.get(omr.ARM_ACTUATORS["jaw"])
    WIRES = inputs["WIRES"]
    cut_tab = {}
    for name, (dw, uts) in WIRES.items():
        F = 0.8 * uts * math.pi / 4 * dw ** 2
        x_max = JAW.stall_Nm / F
        cut_tab[name] = {"cutting force [kN]": F / 1000, "cuts up to x [mm] from the pin": x_max * 1000, "in the notch (40 mm)": bool(x_max >= ag["notch"])}
    # cells 12 and 14: the shoulder servo's mode
    Lu = p["upper_arm_length"]
    outboard = (m_arm["forearm (Al 6082, CAD)"] + m_arm["wrist drive (harmonic 60 Nm, brake)"] + m_arm["palm (wrist housing, screw mount)"]
                + m_arm["jaw drive (jaw screw 6 kN)"] + m_arm["jaws 2x (tool steel, CAD)"] + m_arm["elbow drive (harmonic 150 Nm, brake)"] + m_log)
    lever_out = inputs["lever_out"]                       # outboard masses' centroid beyond the elbow [m] (arm horizontal)
    kp_sh = omr.ARM_GAINS["shoulder"][0]
    I_out = outboard * (Lu / 1000 + lever_out) ** 2 + m_arm["upper arm (Al 6082, CAD)"] * (Lu / 1000) ** 2 / 3 + omr.ARM_GAINS["shoulder"][2]
    f_servo = math.sqrt(kp_sh / I_out) / (2 * math.pi)
    return {"arm_torques_holding_log": tq.to_dict(orient="index"), "heaviest_log_kg": heaviest, "cutting": cut_tab,
            "shoulder_servo_mode_hz": f_servo}


# ------------------------------------------------------------------------------------------------ the Sweeper (23)
def sweeper_sizing(p: dict, cad: dict, M: float, inputs: dict = SWEEPER) -> dict:
    """23 cell 8 (broom, fan, power and endurance), cell 10 (the arms' reach and torques at the picks and over the
    basket), cell 15 (the pick-up table on the recorded suction) and cell 21's forcing (the broom's and the fan's
    unbalance, the body's ride frequency on its tyres; the transmissibilities need the basket and hood modes, FEA);
    cell 27's names (``unbalance`` the forcing of its ``vibration``)."""
    sg = osr.sweeper_geometry(p)
    ag = omr.arm_geometry(p)
    suction = dict(inputs["suction"])
    # cell 8
    BROOM = act.get(osr.BROOM_DRIVE); FAN = act.get(osr.FAN_MOTOR)
    N_bristle, mu_bristle, r_eff = inputs["N_bristle"], inputs["mu_bristle"], 0.8 * sg["broom_r"]     # inputs: bristle load, friction, mean bristle radius
    w_broom = osr.BROOM_RPM * 2 * math.pi / 60
    tau_broom = mu_bristle * N_bristle * r_eff; P_broom = tau_broom * w_broom
    dp_downstream, eta_fan, eta_motor = inputs["dp_downstream"], inputs["eta_fan"], inputs["eta_motor"]
    dp_fan = cfd.SUCTION["fan_static_pressure_Pa"] + dp_downstream
    P_air = osr.FAN_FLOW * dp_fan; P_fan_shaft = P_air / eta_fan; P_fan_el = P_fan_shaft / eta_motor
    P_roll = orb.C_RR * M * G * 1.0; P_elec = inputs["P_elec"]
    P_total = P_roll + P_broom / 0.85 + P_fan_el + P_elec
    E_usable = inputs["E_usable_Wh"]
    hours = E_usable / P_total
    power = pd.DataFrame({"W": {"rolling resistance at 1 m/s (C_RR 0.03)": P_roll, "broom (bristles 100 N, μ 0.5, 150 rpm; drive η 0.85)": P_broom / 0.85,
                                f"fan ({osr.FAN_FLOW} m³/s at {dp_fan:.0f} Pa; η 0.55 × 0.9)": P_fan_el, "computer, sensors, display": P_elec, "TOTAL": P_total}})
    # cell 10
    down = -math.pi / 2
    bz1 = sg["basket"][5]
    targets = {"brick on the road (pick)": ((ag["x"] + 0.75, 0.0, 0.07 + swc.PICK["brick"]["grip_depth"] - sg["hull_z"]), "L"),
               "box on the road (pick)": ((ag["x"] + 0.75, 0.0, 0.15 + swc.PICK["box"]["grip_depth"] - sg["hull_z"]), "R"),
               "drop into the basket (L)": ((-0.45, ag["y"] + 0.02, bz1 + 0.36), "L"), "drop into the basket (R)": ((-0.45, -ag["y"] - 0.02, bz1 + 0.36), "R")}
    m_arm = omr.arm_masses(p, cad)
    stall = {j: act.get(omr.ARM_ACTUATORS[j]).stall_Nm for j in ("shoulder", "elbow", "wrist")}

    def torques(q, load_kg):
        _, sh, el, wr = q
        p1, p2, p3 = -sh, -(sh + el), -(sh + el + wr)
        elbow = ag["Lu"] * math.cos(p1); wrist = elbow + ag["Lf"] * math.cos(p2); pin = wrist + ag["Lp"] * math.cos(p3); jaw_c = pin + 0.1 * math.cos(p3)
        items = [(m_arm["upper arm (Al 6082, CAD)"], ag["Lu"] / 2 * math.cos(p1)), (m_arm["elbow drive (harmonic 150 Nm, brake)"], elbow),
                 (m_arm["forearm (Al 6082, CAD)"], elbow + ag["Lf"] / 2 * math.cos(p2)), (m_arm["wrist drive (harmonic 60 Nm, brake)"], wrist),
                 (m_arm["palm (wrist housing, screw mount)"] + m_arm["jaw drive (jaw screw 6 kN)"], wrist + ag["Lp"] / 2 * math.cos(p3)),
                 (m_arm["jaws 2x (tool steel, CAD)"], jaw_c), (load_kg, jaw_c)]
        return {"shoulder": abs(G * sum(mi * xi for mi, xi in items)), "elbow": abs(G * sum(mi * (xi - elbow) for mi, xi in items[2:])),
                "wrist": abs(G * sum(mi * (xi - wrist) for mi, xi in items[4:]))}
    rows = {}
    for name, (tgt, side) in targets.items():
        q = omr.arm_ik(tgt, down, side, p)
        load = oss.OBJECTS["brick"]["mass"] if "brick" in name or "(L)" in name else oss.OBJECTS["box"]["mass"]
        row = {"reachable": q is not None, "yaw [deg]": math.degrees(q[0]) if q else float("nan"), "extension": math.hypot(*(np.array(tgt[:2]) - np.array([ag["x"], omr.SIDES[side] * ag["y"]]))) and float("nan")}
        if q:
            tq = torques(q, load)
            d = math.sqrt((tgt[0] - ag["x"]) ** 2 + (tgt[1] - omr.SIDES[side] * ag["y"]) ** 2 + (tgt[2] - ag["z"]) ** 2)
            row["extension"] = d / (ag["Lu"] + ag["Lf"])
            row.update({f"{j} / stall": tq[j] / stall[j] for j in stall})
        rows[name] = row
    # cell 15
    pick = cfd.pickup_table(suction)
    # cell 21: the forcing
    m_disc, e_disc = osr.sweeper_masses(p, cad)["broom disc (PA6, CAD)"] + osr.SWEEPER_KG["bristle ring (PP tufts)"], inputs["e_disc"]   # inputs: disc mass, 2 mm eccentricity
    m_imp, grade = inputs["m_imp"], inputs["grade"]                                              # fan impeller 3 kg, ISO 21940 G 6.3 (e·ω = 6.3 mm/s)
    f_broom = osr.BROOM_RPM / 60; f_fan = inputs["fan_rpm"] / 60
    F_broom = m_disc * e_disc * (2 * math.pi * f_broom) ** 2
    e_fan = grade / (2 * math.pi * f_fan); F_fan = m_imp * e_fan * (2 * math.pi * f_fan) ** 2
    k_tyre = inputs["k_tyre"]; f_ride = math.sqrt(4 * k_tyre / M) / (2 * math.pi)
    return {"power_W": power["W"].to_dict(), "endurance_h": hours, "arms": rows, "pickup": pick.to_dict(orient="index"),
            "unbalance": {"f_broom": f_broom, "F_broom": F_broom, "f_fan": f_fan, "F_fan": F_fan, "f_ride": f_ride}}
