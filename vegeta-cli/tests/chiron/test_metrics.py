"""chiron.metrics on hand-checkable synthetic episode logs (protocol §6 and §9.2)."""
import ast
import math
from pathlib import Path

import numpy as np
import pytest

from vegeta.chiron import metrics as M

G = 9.81
SIDES = {"rl": (-0.05, 0.1), "fl": (0.05, 0.1), "rr": (-0.05, -0.1), "fr": (0.05, -0.1)}
BODY_X = {"front": 0.2, "rear": 0.0}


def quat_zyx(yaw=0.0, pitch=0.0, roll=0.0):
    """w, x, y, z of R = Rz(yaw) Ry(pitch) Rx(roll) (works elementwise on arrays)."""
    cy, sy = np.cos(np.asarray(yaw) / 2), np.sin(np.asarray(yaw) / 2)
    cp, sp = np.cos(np.asarray(pitch) / 2), np.sin(np.asarray(pitch) / 2)
    cr, sr = np.cos(np.asarray(roll) / 2), np.sin(np.asarray(roll) / 2)
    return np.stack([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy], axis=-1)


def make_log(T=200, dt=0.01, v=0.2, m=10.0):
    """Two bodies ('front', 'rear'), four feet each (rl, fl, rr, fr), three active joints per leg and two
    passive body joints; all feet loaded and still, the COM moving at v along +x."""
    t = np.arange(T) * dt
    bodies = ["front", "rear"]
    feet = [f"{b}_{s}" for b in bodies for s in SIDES]
    F = len(feet)
    joints, kinds, leg = [], [], []
    for i, f in enumerate(feet):
        for suf, kind in (("hy", "hip_yaw"), ("hp", "hip_pitch"), ("kn", "knee")):
            joints.append(f"{f}_{suf}")
            kinds.append(kind)
            leg.append(i)
    joints += ["body_yaw_12", "body_pitch_12"]
    kinds += ["body_yaw", "body_pitch"]
    leg += [-1, -1]
    J = len(joints)
    active = np.array([k not in ("body_yaw", "body_pitch") for k in kinds])
    nan_p = lambda x: np.where(active, x, np.nan)
    com = np.zeros((T, 3))
    com[:, 0] = 0.1 + v * t
    com[:, 2] = 0.15
    com_vel = np.zeros((T, 3))
    com_vel[:, 0] = v
    body_pos = np.zeros((T, 2, 3))
    for b, name in enumerate(bodies):
        body_pos[:, b, 0] = BODY_X[name] + v * t
        body_pos[:, b, 2] = 0.15
    foot_pos = np.zeros((T, F, 3))
    for i, f in enumerate(feet):
        b, s = f.split("_")
        foot_pos[:, i, 0] = BODY_X[b] + SIDES[s][0]
        foot_pos[:, i, 1] = SIDES[s][1]
    force = np.zeros((T, F, 3))
    force[..., 2] = m * G / F
    normal = np.zeros((T, F, 3))
    normal[..., 2] = 1.0
    q_range = np.tile(np.radians([-60.0, 60.0]), (J, 1))
    q_range[-2:] = np.radians([-45.0, 45.0])
    return {
        "t": t, "bodies": bodies, "body_group": bodies,
        "body_pos": body_pos, "body_quat": np.tile([1.0, 0, 0, 0], (T, 2, 1)),
        "body_angvel": np.zeros((T, 2, 3)), "body_linvel": np.tile([v, 0, 0], (T, 2, 1)),
        "com": com, "com_vel": com_vel, "ang_mom": np.zeros((T, 3)),
        "total_mass": m, "gravity": np.array([0.0, 0.0, -G]),
        "feet": feet, "foot_body": np.array([0] * 4 + [1] * 4), "foot_group": ["front"] * 4 + ["rear"] * 4,
        "foot_pos": foot_pos, "foot_force": force, "foot_normal": normal, "foot_contact_pos": foot_pos.copy(),
        "foot_mu": np.full(F, 0.8), "foot_jac": np.zeros((T, F, 3, 3)),
        "foot_joints": [[f"{f}_hy", f"{f}_hp", f"{f}_kn"] for f in feet],
        "joints": joints, "joint_kind": kinds, "joint_active": active, "joint_leg": np.array(leg),
        "q": np.zeros((T, J)), "qd": np.zeros((T, J)), "tau": np.zeros((T, J)),
        "tau_stall": nan_p(6.0), "tau_rated": nan_p(2.0), "qd_noload": nan_p(5.76), "i_stall": nan_p(3.0),
        "voltage": nan_p(12.0), "q_range": q_range,
        "belly_contact": np.zeros((T, 2), bool), "terrain_height_under_com": np.zeros(T),
        "v_target": v, "course_m": 1.5, "nominal_hip_height": 0.15, "disturbances": [],
        "robot": "test", "treatment": "flexible", "controller": "fixed",
        "terrain": {"kind": "flat", "level": 0.0, "seed": 3}, "seed": 3,
    }


# ---------------------------------------------------------------------------------------------- helpers
def test_orientation_conventions():
    q = quat_zyx(0.3, 0.2, -0.1)
    yaw, pitch, roll = M.euler_zyx(q)
    assert (yaw, pitch, roll) == pytest.approx((0.3, 0.2, -0.1))
    R = M.quat_to_matrix(quat_zyx(pitch=0.2))
    assert R[2, 0] == pytest.approx(-math.sin(0.2))              # positive pitch: nose (body x) down
    R = M.quat_to_matrix(quat_zyx(roll=0.2))
    assert R[2, 1] == pytest.approx(math.sin(0.2))               # positive roll: left side up, right down
    assert M.tilt_angle(quat_zyx(1.0, 0.0, 0.25)) == pytest.approx(0.25)    # tilt ignores yaw
    assert M.tilt_angle(quat_zyx(0.0, 0.3, 0.0)) == pytest.approx(0.3)


def test_pure_numpy_scipy():
    src = Path(M.__file__).parent
    for name in ("metrics.py", "stats.py"):
        tree = ast.parse((src / name).read_text())
        mods = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        mods |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module and n.level == 0}
        assert mods <= {"__future__", "math", "time", "typing", "numpy", "scipy", "itertools", "dataclasses"}, (name, mods)


# ------------------------------------------------------------------------------------ progress and speed
def test_distance_and_speed():
    log = make_log(T=201)
    d = M.distance_before_failure(log, {"success": False, "reason": "fall", "t_end": 1.0})
    assert d["progress_m"] == pytest.approx(0.2) and d["x_end_m"] == pytest.approx(0.3) and d["completed"] is False
    assert M.achieved_speed(log, {"t_end": 1.0}) == pytest.approx(0.2)
    assert M.distance_before_failure(log)["progress_m"] == pytest.approx(0.4) and M.distance_before_failure(log)["completed"] is None
    assert M.achieved_speed(log) == pytest.approx(0.2)


# -------------------------------------------------------------------------------------- body motion
def test_body_angular_motion_per_body_payload_and_worst():
    log = make_log(T=100)
    log["body_angvel"][:, 1, 0] = 0.3                         # rear: roll rate 0.3 rad/s
    log["body_angvel"][:, 0, 1] = np.where(np.arange(100) % 2, 0.1, -0.1)   # front: pitch rate ±0.1
    roll = np.zeros((100, 2))
    roll[:, 1] = np.radians(np.linspace(0, 10, 100))        # rear rolls 0..10 deg
    pitch = np.zeros((100, 2))
    pitch[:, 0] = np.radians(4.0)                           # front pitched 4 deg
    log["body_quat"] = quat_zyx(0.0, pitch, roll)
    out = M.body_angular_motion(log, payload="front")
    rear, front = out["per_body"]["rear"], out["per_body"]["front"]
    assert rear["roll_rate_rms_rad_s"] == pytest.approx(0.3) and front["pitch_rate_rms_rad_s"] == pytest.approx(0.1)
    assert rear["tilt_p95_deg"] == pytest.approx(np.percentile(np.linspace(0, 10, 100), 95))
    assert rear["roll_abs_p95_deg"] == pytest.approx(rear["tilt_p95_deg"]) and rear["tilt_max_deg"] == pytest.approx(10)
    assert front["pitch_abs_p95_deg"] == pytest.approx(4.0) and front["tilt_p95_deg"] == pytest.approx(4.0)
    # worst = max over bodies, per metric, never averaged
    assert out["worst_tilt_p95_deg"] == pytest.approx(rear["tilt_p95_deg"]) and out["worst_tilt_p95_deg_body"] == "rear"
    assert out["worst_pitch_abs_p95_deg"] == pytest.approx(4.0) and out["worst_pitch_abs_p95_deg_body"] == "front"
    assert out["payload_body"] == "front" and out["payload_tilt_p95_deg"] == pytest.approx(4.0)
    assert "payload_body" not in M.body_angular_motion(log)          # no payload given: none guessed
    with pytest.raises(ValueError):
        M.body_angular_motion(log, payload="tail")


def test_intersegment_angles_limit_band_and_hits():
    log = make_log(T=200)
    j = log["joints"].index("body_yaw_12")
    q = np.zeros(200)
    q[50:70] = np.radians(44.5)           # within 1 deg of +45: 20 samples
    q[100:110] = np.radians(-44.6)        # within 1 deg of -45: 10 samples
    q[150:160] = np.radians(40.0)         # large but not near the limit
    log["q"][:, j] = q
    out = M.intersegment_angles(log)
    yj = out["per_joint"]["body_yaw_12"]
    assert yj["kind"] == "body_yaw" and yj["max_abs_deg"] == pytest.approx(44.6)
    assert yj["near_limit_fraction"] == pytest.approx(30 / 200) and yj["limit_hits"] == 2
    assert yj["p95_abs_deg"] == pytest.approx(np.percentile(np.abs(np.degrees(q)), 95))
    assert out["per_joint"]["body_pitch_12"]["limit_hits"] == 0
    assert out["per_kind"]["body_yaw"]["max_abs_deg"] == pytest.approx(44.6)
    # a locked robot has no body joints
    lk = make_log(T=10)
    lk["joint_kind"] = [k if not k.startswith("body_") else "weld" for k in lk["joint_kind"]]
    assert M.intersegment_angles(lk) == {"per_joint": {}, "per_kind": {}}


def test_belly_contacts():
    log = make_log(T=200)
    log["belly_contact"][10:30, 0] = True
    log["belly_contact"][20:40, 1] = True
    out = M.belly_contacts(log)
    assert out["fraction"] == pytest.approx(30 / 200) and out["events"] == 1
    assert out["per_body"] == {"front": pytest.approx(0.1), "rear": pytest.approx(0.1)}


# ---------------------------------------------------------------------------------------------- slip
def test_foot_slip_zero_exact_and_unloaded_excluded():
    log = make_log(T=200)
    out = M.foot_slip(log)
    assert out["slip_total_m"] == 0.0 and out["n_episodes"] == 8 and out["fraction_over_tol"] == 0.0
    k = np.arange(200)
    log["foot_pos"][:, 0, 0] += 0.001 * np.clip(k - 50, 0, 20)     # foot 0 slides 20 mm while loaded
    log["foot_pos"][:, 1, 2] += 0.001 * np.clip(k - 50, 0, 20)     # foot 1 moves along the normal: no slip
    log["foot_pos"][:, 2, 0] += 0.001 * np.clip(k - 100, 0, 50)    # foot 2 moves 50 mm while UNloaded
    log["foot_force"][100:150, 2] = 0.0
    log["foot_normal"][100:150, 2] = np.nan
    out = M.foot_slip(log)
    assert out["per_foot_m"]["front_rl"] == pytest.approx(0.020)
    assert out["per_foot_m"]["front_fl"] == pytest.approx(0.0, abs=1e-12)
    assert out["per_foot_m"]["front_rr"] == pytest.approx(0.0, abs=1e-12)
    assert out["n_episodes"] == 9 and out["slip_total_m"] == pytest.approx(0.020)
    assert out["fraction_over_tol"] == pytest.approx(1 / 9)
    assert out["slip_per_m"] == pytest.approx(0.020 / (0.2 * 1.99))
    # a light touch (below 2 % of the weight) is not a loaded episode
    log["foot_force"][:, 7, 2] = 0.01 * 10 * G
    assert M.foot_slip(log)["n_episodes"] == 8


def test_foot_slip_in_the_tilted_contact_plane():
    log = make_log(T=50)
    a = math.radians(30)
    n = np.array([math.sin(a), 0.0, math.cos(a)])
    tan = np.array([math.cos(a), 0.0, -math.sin(a)])
    log["foot_normal"][:, 0] = n
    log["foot_force"][:, 0] = n * 20.0
    log["foot_normal"][:, 1] = n
    log["foot_force"][:, 1] = n * 20.0
    k = np.arange(50)[:, None]
    log["foot_pos"][:, 0] += 0.01 * np.clip(k / 49, 0, 1) * n        # along the normal: 0
    log["foot_pos"][:, 1] += 0.01 * np.clip(k / 49, 0, 1) * tan      # in the plane: 10 mm
    out = M.foot_slip(log)
    assert out["per_foot_m"]["front_rl"] == pytest.approx(0.0, abs=1e-12)
    assert out["per_foot_m"]["front_fl"] == pytest.approx(0.010)


# ------------------------------------------------------------------------------------ cost of transport
def test_cost_of_transport_positive_power_active_joints_only():
    log = make_log(T=201)                         # 2.0 s, 0.4 m
    J = {n: i for i, n in enumerate(log["joints"])}
    for name, tau, qd in (("front_rl_hy", 2.0, 1.5), ("front_rl_hp", 2.0, 1.5), ("front_rl_kn", 2.0, -1.0),
                          ("front_fl_kn", 2.0, -5.0), ("body_yaw_12", 5.0, 1.0)):
        log["tau"][:, J[name]], log["qd"][:, J[name]] = tau, qd
    out = M.cost_of_transport(log)
    assert out["energy_pos_J"] == pytest.approx(2 * 3.0 * 2.0)                    # negative and passive excluded
    assert out["cot_mech"] == pytest.approx(12.0 / (10 * G * 0.4))
    # k_t = 6/3 = 2 N m/A, R = 12/3 = 4 ohm, I = 1 A, I^2 R = 4 W:  7 + 7 + max(-2+4, 0) + max(-10+4, 0)
    assert out["energy_el_J"] == pytest.approx((7 + 7 + 2 + 0) * 2.0)
    assert out["cot_el_est"] == pytest.approx(32.0 / (10 * G * 0.4))
    assert out["power_mech_mean_W"] == pytest.approx(6.0) and out["duration_s"] == pytest.approx(2.0)
    log["com"][:, 0] = 0.1                                                          # no progress
    assert math.isnan(M.cost_of_transport(log)["cot_mech"])


def test_actuator_demand_groups_and_saturation():
    log = make_log(T=100)
    J = {n: i for i, n in enumerate(log["joints"])}
    knees = [J[f"{f}_kn"] for f in log["feet"]]
    log["tau"][:, knees] = 1.0
    log["tau"][50:, J["front_rl_kn"]] = 2.95                 # limit at w0/2 is 3.0: 2.95 >= 0.98*3
    log["qd"][50:, J["front_rl_kn"]] = 5.76 / 2
    log["qd"][:, J["rear_fr_hy"]] = 5.7                      # >= 0.98 * 5.76: speed-saturated
    out = M.actuator_demand(log)
    assert set(out) == {"hip_yaw", "hip_pitch", "knee"}      # passive body joints are not actuators
    kn = out["knee"]
    assert kn["peak_tau_Nm"] == pytest.approx(2.95)
    assert kn["rms_tau_worst_Nm"] == pytest.approx(math.sqrt(0.5 * 1 + 0.5 * 2.95 ** 2))
    assert kn["rms_tau_Nm"] == pytest.approx(math.sqrt((7 * 100 + 50 + 50 * 2.95 ** 2) / 800))
    assert kn["rms_over_rated_worst"] == pytest.approx(kn["rms_tau_worst_Nm"] / 2.0)
    assert kn["torque_sat_fraction_worst"] == pytest.approx(0.5) and kn["torque_sat_fraction"] == pytest.approx(0.5 / 8)
    hy = out["hip_yaw"]
    assert hy["speed_sat_fraction"] == pytest.approx(1 / 8) and hy["speed_sat_fraction_worst"] == 1.0
    assert hy["torque_sat_fraction"] == 0.0                  # tau = 0 below the (small) limit 0.0625 N m
    assert out["hip_pitch"]["peak_tau_Nm"] == 0.0


# ------------------------------------------------------------------------------------- support margin
def contact_log(cases, feet_xy, m=10.0):
    """One sample per case: (com_xy, {foot index: normal force N}); unlisted feet unloaded."""
    T, F = len(cases), len(feet_xy)
    com = np.zeros((T, 3))
    force = np.zeros((T, F, 3))
    normal = np.full((T, F, 3), np.nan)
    pos = np.zeros((T, F, 3))
    pos[:, :, :2] = np.asarray(feet_xy, float)
    for k, (c, loads) in enumerate(cases):
        com[k, :2] = c
        com[k, 2] = 0.2
        for i, fn in loads.items():
            force[k, i, 2] = fn
            normal[k, i] = (0, 0, 1)
    return {"t": np.arange(T) * 0.01, "com": com, "com_vel": np.zeros((T, 3)), "ang_mom": np.zeros((T, 3)),
            "foot_force": force, "foot_normal": normal, "foot_pos": pos, "foot_contact_pos": pos.copy(),
            "feet": [f"f{i}" for i in range(F)], "foot_mu": np.full(F, 0.8), "total_mass": m,
            "gravity": np.array([0, 0, -G]), "belly_contact": np.zeros((T, 1), bool), "nominal_hip_height": 0.15}


def test_support_margin_cases():
    sq = [(-0.1, -0.1), (0.1, -0.1), (0.1, 0.1), (-0.1, 0.1), (0.0, 0.1)]
    all4 = {0: 25.0, 1: 25.0, 2: 25.0, 3: 25.0}
    cases = [
        ((0.0, 0.0), all4),                                   # centre of the square: +0.1
        ((0.15, 0.0), all4),                                  # outside the right edge: -0.05
        ((0.1, 0.0), all4),                                   # on the edge: 0, not outside
        ((0.2, 0.2), all4),                                   # off the corner: -sqrt(2)*0.1
        ((0.0, 0.05), {3: 40.0, 2: 40.0}),                    # two contacts: minus distance to the segment
        ((0.0, 0.0), {2: 80.0}),                              # one contact: minus distance to the point
        ((0.0, 0.0), {}),                                     # none: nan
        ((0.0, 0.07), {3: 30.0, 4: 30.0, 2: 30.0}),           # three collinear contacts: -0.03
        ((0.05, 0.1), {3: 30.0, 4: 30.0, 2: 30.0}),           # on the collinear segment: 0
        ((0.05, 0.05), {0: 30.0, 1: 30.0, 3: 30.0, 2: 0.5}),  # foot 2 below 2 % of weight: triangle only
    ]
    out = M.support_margin(contact_log(cases, sq))
    s = out["s"]
    exp = [0.1, -0.05, 0.0, -math.sqrt(2) * 0.1, -0.05, -math.sqrt(2) * 0.1, np.nan, -0.03, 0.0, -0.1 / math.sqrt(2)]
    np.testing.assert_allclose(s, exp, atol=1e-12)
    assert out["min_m"] == pytest.approx(-math.sqrt(2) * 0.1)
    assert out["outside_fraction"] == pytest.approx(6 / 9) and out["no_contact_fraction"] == pytest.approx(0.1)
    assert out["n_contacts"].tolist() == [4, 4, 4, 4, 2, 1, 0, 3, 3, 3]
    assert out["p5_m"] == pytest.approx(np.percentile([e for e in exp if np.isfinite(e)], 5))


# ---------------------------------------------------------------------------- contact-force feasibility
def feas_log(com_xyz, acc=(0.0, 0.0, 0.0), mu=0.8, stall=6.0, half=0.3, m=10.0, T=5):
    corners = [(-half, -half), (half, -half), (half, half), (-half, half)]
    lg = contact_log([((0, 0), {i: m * G / 4 for i in range(4)})] * T, corners, m=m)
    lg["com"][:] = com_xyz
    lg["com_vel"] = np.outer(np.arange(T) * 0.01, acc)            # a = acc exactly
    lg["foot_mu"] = np.full(4, mu)
    lg["joints"] = [f"j{i}" for i in range(12)]
    lg["tau_stall"] = np.full(12, stall)
    lg["foot_joints"] = [[f"j{3 * i}", f"j{3 * i + 1}", f"j{3 * i + 2}"] for i in range(4)]
    lg["foot_jac"] = np.tile(0.2 * np.eye(3), (T, 4, 1, 1))       # every joint sees 0.2 m x the force
    return lg


def test_feasibility_quasi_static_inside_outside():
    out = M.contact_force_feasibility(feas_log((0.0, 0.0, 0.2)))
    assert out["infeasible_fraction"] == {"quasi_static": 0.0, "dynamic": 0.0, "quasi_static_act": 0.0, "dynamic_act": 0.0}
    assert out["n_evaluated"]["quasi_static"] == 5
    out = M.contact_force_feasibility(feas_log((0.5, 0.0, 0.2)))          # COM beyond the support square
    assert out["infeasible_fraction"]["quasi_static"] == 1.0 and out["infeasible_fraction"]["dynamic"] == 1.0
    out = M.contact_force_feasibility(feas_log((0.29, 0.0, 0.2)), every=2)
    assert out["infeasible_fraction"]["quasi_static"] == 0.0 and out["index"].tolist() == [0, 2, 4]


def test_feasibility_dynamic_needs_friction():
    a = (0.6 * G, 0.0, 0.0)               # needs |f_t| / f_n = 0.6 in total
    tight = M.contact_force_feasibility(feas_log((0.0, 0.0, 0.2), acc=a, mu=0.5))
    assert tight["infeasible_fraction"]["quasi_static"] == 0.0 and tight["infeasible_fraction"]["dynamic"] == 1.0
    ok = M.contact_force_feasibility(feas_log((0.0, 0.0, 0.2), acc=a, mu=0.8))
    assert ok["infeasible_fraction"]["dynamic"] == 0.0
    over = M.contact_force_feasibility(feas_log((0.0, 0.0, 0.2), acc=a), mu=0.5)    # mu override
    assert over["infeasible_fraction"]["dynamic"] == 1.0
    # the same acceleration with the COM so high that the pitching moment needs x_cop beyond the rear feet
    high = M.contact_force_feasibility(feas_log((0.0, 0.0, 0.6), acc=a, mu=0.8))
    assert high["infeasible_fraction"]["dynamic"] == 1.0


def test_feasibility_actuator_limited_and_belly_and_no_contact():
    weak = M.contact_force_feasibility(feas_log((0.0, 0.0, 0.2), stall=2.0))   # 4 x 2/0.2 = 40 N < 98 N
    assert weak["infeasible_fraction"]["quasi_static"] == 0.0 and weak["infeasible_fraction"]["quasi_static_act"] == 1.0
    assert M.contact_force_feasibility(feas_log((0.0, 0.0, 0.2), stall=6.0))["infeasible_fraction"]["quasi_static_act"] == 0.0
    lg = feas_log((0.0, 0.0, 0.2))
    lg["belly_contact"][2] = True
    out = M.contact_force_feasibility(lg, actuator_limits=False)
    assert out["n_belly_skipped"] == 1 and out["n_evaluated"] == {"quasi_static": 4, "dynamic": 4}
    lg = feas_log((0.0, 0.0, 0.2))
    lg["foot_force"][:] = 0.0
    out = M.contact_force_feasibility(lg, actuator_limits=False)
    assert out["infeasible_fraction"]["quasi_static"] == 1.0                      # nothing holds the weight
    lg["com_vel"] = np.outer(np.arange(5) * 0.01, (0, 0, -G))                      # free fall: a = g
    assert M.contact_force_feasibility(lg, actuator_limits=False)["infeasible_fraction"]["dynamic"] == 0.0


def test_feasibility_two_contacts_and_wrench_tolerance():
    # trot-like: only the diagonal pair (-0.3,-0.3) and (0.3,0.3) loaded
    def diag(com_xy, dL=(0.0, 0.0, 0.0)):
        lg = feas_log((com_xy[0], com_xy[1], 0.2))
        lg["foot_force"][:, [1, 3]] = 0.0
        lg["foot_force"][:, [0, 2], 2] = 10 * G / 2
        lg["ang_mom"] = np.outer(np.arange(5) * 0.01, dL)
        return lg
    on = M.contact_force_feasibility(diag((0.0, 0.0)), actuator_limits=False, wrench_tol=(0, 0))
    assert on["infeasible_fraction"]["quasi_static"] == 0.0                       # COM exactly over the line
    off = M.contact_force_feasibility(diag((0.02, -0.02)), actuator_limits=False)
    assert off["infeasible_fraction"]["quasi_static"] == 1.0                      # 28 mm off the line: falls
    # a 0.1 N m inconsistency of dL/dt ALONG the contact line (finite-difference noise): with the total force
    # fixed by Newton, two point contacts cannot produce it ...
    noisy = diag((0.0, 0.0), dL=(0.1 / math.sqrt(2), 0.1 / math.sqrt(2), 0.0))
    exact = M.contact_force_feasibility(noisy, actuator_limits=False, wrench_tol=(0, 0))
    assert exact["infeasible_fraction"]["dynamic"] == 1.0                         # ... breaks the exact test
    tol = M.contact_force_feasibility(noisy, actuator_limits=False)
    assert tol["wrench_tol"] == pytest.approx((0.02 * 10 * G, 0.02 * 10 * G * 0.15))
    assert tol["infeasible_fraction"]["dynamic"] == 0.0                           # within 2 % W x hip height


def test_feasibility_dynamic_accounts_for_the_logged_push():
    # a 1 N s sideways push over 50 ms on a 10 kg robot standing on ice (mu = 0.05: at most 4.9 N sideways)
    T = 50
    lg = feas_log((0.0, 0.0, 0.2), mu=0.05, T=T)
    t = lg["t"]
    lg["com_vel"] = np.zeros((T, 3))
    lg["com_vel"][:, 1] = 0.1 * np.clip((t - 0.2) / 0.05, 0, 1)              # dv = J / m
    lg["bodies"], lg["body_pos"] = ["trunk"], lg["com"][:, None, :].copy()
    lg["disturbances"] = [{"t_start": 0.2, "duration": 0.05, "impulse_Ns": 1.0, "body": "trunk", "direction": [0, 1, 0]}]
    out = M.contact_force_feasibility(lg, actuator_limits=False)
    assert out["infeasible_fraction"]["dynamic"] == 0.0                         # the push supplies the force
    blind = dict(lg, disturbances=[])
    assert M.contact_force_feasibility(blind, actuator_limits=False)["infeasible_fraction"]["dynamic"] == pytest.approx(6 / 50)
    lg["disturbances"][0]["body"] = "not logged"                                 # unknown point of application
    out = M.contact_force_feasibility(lg, actuator_limits=False)
    assert out["n_evaluated"]["dynamic"] == 50 - 6 and out["infeasible_fraction"]["dynamic"] == 0.0


def test_friction_pyramid_geometry():
    n = np.array([0.0, 0.0, 1.0])
    g = M.friction_pyramid(n, 0.5)
    assert g.shape == (3, 8) and np.allclose(g[2], 1.0)
    assert np.allclose(np.hypot(g[0], g[1]), 0.5)                                  # edges on the cone
    assert np.isclose(g[0].max(), 0.5)                                             # full mu along t1 = world x
    gc = M.friction_pyramid(n, 0.5, "circumscribed")
    assert np.allclose(np.hypot(gc[0], gc[1]), 0.5 / math.cos(math.pi / 8))
    tilted = np.array([math.sin(0.3), 0.0, math.cos(0.3)])
    gt = M.friction_pyramid(tilted, 0.8)
    proj = tilted @ gt
    assert np.allclose(proj, 1.0) and np.allclose(np.linalg.norm(gt - np.outer(tilted, proj), axis=0), 0.8)


# ---------------------------------------------------------------------------------------------- recovery
def push_log(v_low_until=4.0, tilt_until=4.5, T=1000):
    log = make_log(T=T)
    t = log["t"]
    k = np.arange(T)
    log["com_vel"][:, 0] = np.where((k >= 300) & (k < int(round(v_low_until * 100))), 0.05, 0.2)
    roll = np.where((k >= 300) & (k < int(round(tilt_until * 100))), np.radians(15.0), 0.0)
    log["body_quat"][:, 1] = quat_zyx(0.0, 0.0, roll)
    log["disturbances"] = [{"t_start": 3.0, "duration": 0.05, "impulse_Ns": 2.0, "body": "rear", "direction": (0, 1, 0)}]
    return log


def test_recovery_on_a_synthetic_trace():
    out = M.recovery(push_log())                          # tilt back at 4.50 s (speed already back)
    assert out["recovered"] and out["recovery_time_s"] == pytest.approx(4.50 - 3.05)
    assert out["v_pre_m_s"] == pytest.approx(0.2) and out["impulse_Ns"] == 2.0
    # speed-limited: centred 0.5 s average (51 samples) first reaches 0.14 = 0.2 - 30 % at sample 405
    out = M.recovery(push_log(v_low_until=4.0, tilt_until=3.5))
    assert out["recovered"] and out["recovery_time_s"] == pytest.approx(4.05 - 3.05)
    assert not M.recovery(push_log(), {"success": False, "reason": "fall", "t_end": 9.0})["recovered"]
    late = M.recovery(push_log(tilt_until=8.2))           # back after 5.15 s > 5 s
    assert not late["recovered"] and math.isnan(late["recovery_time_s"])
    short = M.recovery(push_log(tilt_until=9.5))          # never held for 1.0 s within the log
    assert not short["recovered"]
    with pytest.raises(ValueError):
        M.recovery(make_log(T=50))


# ----------------------------------------------------------------------------------- duty and phases
def gait_log(offsets, period=100, stance=75, T=1000, chatter=False):
    """Contact pattern: foot i loaded for ``stance`` samples from every touchdown ``offsets[i] + n·period``."""
    log = make_log(T=T)
    k = np.arange(T)
    for i, f in enumerate(log["feet"]):
        loaded = ((k - offsets[f]) % period < stance) & (k >= offsets[f])
        if chatter:
            loaded &= ~(((k - offsets[f]) % period == 30) & (k >= offsets[f]))   # 1-sample dropout mid-stance
        log["foot_force"][:, i, 2] = np.where(loaded, 30.0, 0.0)
        log["foot_normal"][~loaded, i] = np.nan
    return log


OFFS = {"front_rl": 10, "front_fl": 35, "front_rr": 60, "front_fr": 85,
        "rear_rl": 20, "rear_fl": 45, "rear_rr": 70, "rear_fr": 95}


def test_duty_factor_from_the_contact_record():
    out = M.duty_factor(gait_log(OFFS))
    assert out["mean"] == pytest.approx(0.75) and out["sd"] == pytest.approx(0.0, abs=1e-12)
    assert all(v["n_strides"] == 9 and v["mean"] == pytest.approx(0.75) for v in out["per_foot"].values())
    half = M.duty_factor(gait_log(OFFS, stance=40))
    assert half["mean"] == pytest.approx(0.40)
    noisy = gait_log(OFFS, chatter=True)
    assert M.duty_factor(noisy)["mean"] != pytest.approx(0.75)                     # chatter splits strides
    assert M.duty_factor(noisy, min_run_s=0.03)["mean"] == pytest.approx(0.75)      # debounced


def test_default_pairs_and_interlimb_phases():
    log = gait_log(OFFS)
    pairs = M.default_phase_pairs(log)
    assert ("front_rl", "front_fl", "ipsilateral") in pairs and ("front_rl", "front_rr", "contralateral") in pairs
    assert ("front_rl", "front_fr", "diagonal") in pairs and ("rear_rl", "rear_rr", "contralateral") in pairs
    assert ("front_rl", "rear_rl", "intersegmental") in pairs and len(pairs) == 7
    rel = M.interlimb_phases(log)["relations"]
    expect = {"ipsilateral:front_rl->front_fl": 0.25, "contralateral:front_rl->front_rr": 0.5,
              "diagonal:front_rl->front_fr": 0.75, "ipsilateral:rear_rl->rear_fl": 0.25,
              "intersegmental:front_rl->rear_rl": 0.10}
    for label, ph in expect.items():
        r = rel[label]
        assert r["mean_cycles"] == pytest.approx(ph) and r["sd_rad"] == pytest.approx(0.0, abs=1e-6)
        assert r["n"] == 9 and r["R"] == pytest.approx(1.0)
    custom = M.interlimb_phases(log, pairs=[("front_fl", "front_rl")])["relations"]
    assert custom["relation:front_fl->front_rl"]["mean_cycles"] == pytest.approx(0.75)
    # a body rotated by 90 deg keeps its left/right from its own frame
    rot = gait_log(OFFS)
    rot["body_quat"][:] = quat_zyx(math.pi / 2)
    p = rot["foot_pos"]
    rel_xy = p[:, :, :2] - rot["body_pos"][:, rot["foot_body"], :2]
    p[:, :, 0] = rot["body_pos"][:, rot["foot_body"], 0] - rel_xy[..., 1]
    p[:, :, 1] = rot["body_pos"][:, rot["foot_body"], 1] + rel_xy[..., 0]
    assert M.default_phase_pairs(rot) == pairs


def test_circular_stats():
    c = M.circular_stats([0.95, 0.05])
    assert c["mean_cycles"] == pytest.approx(0.0, abs=1e-12) or c["mean_cycles"] == pytest.approx(1.0)
    R = abs(np.mean(np.exp(2j * np.pi * np.array([0.95, 0.05]))))
    assert c["R"] == pytest.approx(R) and c["sd_rad"] == pytest.approx(math.sqrt(-2 * math.log(R)))
    assert c["sd_cycles"] == pytest.approx(c["sd_rad"] / (2 * math.pi))
    assert M.circular_stats([np.nan])["n"] == 0


def test_phase_recovery_after_a_push():
    T = 1200
    offs = {f: 10 for f in OFFS}
    log = gait_log(offs, stance=40, T=T)
    k = np.arange(T)
    rl_td = 10 + 100 * np.arange(12)
    for i, f in enumerate(log["feet"][:4]):
        base = {"front_rl": 0, "front_fl": 25, "front_rr": 50, "front_fr": 75}[f]
        loaded = np.zeros(T, bool)
        for n, td in enumerate(rl_td):
            d = 55 if (f == "front_fl" and n in (5, 6, 7)) else base       # FL shifted in strides 5-7
            loaded[td + d: td + d + 40] = True
        log["foot_force"][:, i, 2] = np.where(loaded[:T], 30.0, 0.0)
    push = {"t_start": 5.0, "duration": 0.05}
    pairs = [("front_rl", "front_fl"), ("front_rl", "front_rr"), ("front_rl", "front_fr")]
    out = M.phase_recovery(log, push, pairs)
    assert out["per_relation"]["relation:front_rl->front_fl"]["pre_mean_cycles"] == pytest.approx(0.25)
    assert out["per_relation"]["relation:front_rl->front_rr"]["time_s"] == pytest.approx(5.10 - 5.05)
    assert out["per_relation"]["relation:front_rl->front_fl"]["time_s"] == pytest.approx(8.10 - 5.05)
    assert out["recovered"] and out["time_s"] == pytest.approx(3.05)
    # FL never comes back: not recovered
    fl = log["feet"].index("front_fl")
    shifted = np.zeros(T, bool)
    for n, td in enumerate(rl_td):
        d = 55 if n >= 5 else 25
        shifted[td + d: td + d + 40] = True
    log["foot_force"][:, fl, 2] = np.where(shifted[:T], 30.0, 0.0)
    out = M.phase_recovery(log, push, pairs)
    assert not out["recovered"] and math.isnan(out["time_s"])


# ------------------------------------------------------------------------------------------- undulation
def test_body_undulation_on_a_sine():
    log = make_log(T=1001)
    t = log["t"]
    j = log["joints"].index("body_yaw_12")
    log["q"][:, j] = np.radians(10.0) * np.sin(2 * np.pi * 2.0 * t)
    log["com"][:, 1] = 0.01 * np.cos(2 * np.pi * 2.0 * t) + 0.001 * t
    yaw = np.radians(20.0) + np.radians(5.0) * np.sin(2 * np.pi * 2.0 * t)
    log["body_quat"][:, 0] = quat_zyx(yaw)
    log["body_quat"][:, 1] = quat_zyx(np.radians(40.0) - yaw)                      # mirror about 20 deg
    out = M.body_undulation(log, t_from=1.0)
    by = out["body_yaw_joints"]["body_yaw_12"]
    assert by["rms_deg"] == pytest.approx(10 / math.sqrt(2), rel=2e-3)
    assert by["p2p_deg"] == pytest.approx(20.0, abs=0.1) and by["freq_hz"] == pytest.approx(2.0, abs=0.02)
    assert out["body_yaw_rms_max_deg"] == pytest.approx(by["rms_deg"])
    assert out["com_lateral"]["amplitude_m"] == pytest.approx(0.01, rel=0.01)
    assert out["com_lateral"]["freq_hz"] == pytest.approx(2.0, abs=0.02)
    assert out["heading_mean_deg"] == pytest.approx(20.0, abs=1e-6)
    for b in ("front", "rear"):
        assert out["yaw_about_heading"][b]["rms_deg"] == pytest.approx(5 / math.sqrt(2), rel=2e-3)
        assert out["yaw_about_heading"][b]["p2p_deg"] == pytest.approx(10.0, abs=0.05)
    # the window ends at a push
    log["disturbances"] = [{"t_start": 6.0, "duration": 0.05, "impulse_Ns": 1.0}]
    assert M.body_undulation(log)["window_s"] == (1.0, 6.0)


def test_lateral_deviation_and_heading():
    log = make_log(T=100)
    log["com"][:, 1] = np.linspace(0, -0.12, 100)
    log["body_quat"][:, 0] = quat_zyx(np.radians(3.0))
    out = M.lateral_deviation_and_heading(log)
    assert out["lateral_max_m"] == pytest.approx(0.12) and out["heading_rms_deg"] == pytest.approx(3.0)
    assert out["heading_body"] == "front"
    assert M.lateral_deviation_and_heading(log, "rear")["heading_rms_deg"] == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------------------------------------------------ trial row
def test_trial_metrics_flat_row():
    log = gait_log(OFFS, T=1000)
    log["disturbances"] = [{"t_start": 3.0, "duration": 0.05, "impulse_Ns": 2.0, "body": "rear", "direction": (0, 1, 0)}]
    outcome = {"success": True, "reason": "success", "t_end": 9.99, "distance_m": 2.098}
    row = M.trial_metrics(log, outcome, payload="front", feasibility_every=50)
    for k in ("success", "reason", "t_end", "distance_m", "robot", "treatment", "controller", "terrain_kind",
              "terrain_level", "terrain_seed", "seed", "progress_m", "achieved_speed_m_s", "payload_tilt_p95_deg",
              "worst_tilt_p95_deg", "tilt_p95_deg@rear", "belly_contact_fraction", "slip_per_m", "cot_mech",
              "cot_el_est", "knee_peak_tau_Nm", "support_margin_min_m", "infeasible_fraction_quasi_static",
              "infeasible_fraction_dynamic_act", "duty_mean", "duty_mean@front_rl",
              "phase_mean_cycles@ipsilateral:front_rl->front_fl", "phase_sd_max_cycles", "body_yaw_rms_max_deg",
              "body_yaw_rms_deg@body_yaw_12", "com_lateral_amplitude_m", "worst_yaw_about_heading_rms_deg",
              "lateral_max_m", "heading_rms_deg", "recovered", "recovery_time_s", "phase_recovered",
              "body_yaw_max_abs_deg", "isa_limit_hits@body_pitch_12"):
        assert k in row, k
    assert all(v is None or isinstance(v, (str, bool, int, float)) for v in row.values())
    assert row["success"] is True and row["achieved_speed_m_s"] == pytest.approx(0.2)
    assert row["duty_mean"] == pytest.approx(0.75) and row["recovered"] is True and row["phase_recovered"] is True
    assert row["phase_mean_cycles@intersegmental:front_rl->rear_rl"] == pytest.approx(0.1)
    assert "infeasible_fraction_dynamic" not in M.trial_metrics(log, outcome, feasibility_every=None)
    res = M.trial_result(log, outcome, "front", None)
    assert res.kind == "chiron.metrics" and res.ok and res.metrics["duty_mean"] == pytest.approx(0.75)
    assert set(res.to_dict()) == {"kind", "status", "metrics", "artifacts", "messages", "duration_s", "execution", "metadata"}


def test_trial_metrics_skips_blocks_with_missing_keys():
    log = make_log(T=300)
    for k in ("tau", "foot_jac", "belly_contact"):
        del log[k]
    log["disturbances"] = [{"t_start": 1.0, "duration": 0.05, "impulse_Ns": None, "force_N": 40.0, "body": "rear"}]
    row = M.trial_metrics(log, {"success": False, "reason": "timeout", "t_end": 2.99}, feasibility_every=20)
    assert "cost_of_transport" in row["metrics_skipped"] and "belly_contacts" in row["metrics_skipped"]
    assert "cot_mech" not in row and "knee_peak_tau_Nm" not in row
    assert "infeasible_fraction_quasi_static" in row and "infeasible_fraction_quasi_static_act" not in row
    assert math.isnan(row["impulse_Ns"]) and row["recovered"] is True


@pytest.mark.parametrize("T", [0, 1, 2, 5])
def test_trial_metrics_on_tiny_logs(T):
    log = make_log(T=max(T, 1))
    if T == 0:
        log = {k: (v[:0] if isinstance(v, np.ndarray) and v.ndim and v.shape[0] == 1 else v) for k, v in log.items()}
    row = M.trial_metrics(log, {"success": False, "reason": "fall", "t_end": 0.0}, payload="front")
    assert row["n_samples"] == T and row["reason"] == "fall"
