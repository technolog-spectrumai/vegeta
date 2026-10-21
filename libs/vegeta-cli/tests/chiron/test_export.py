"""chiron.export: raw time series of an episode log (columns, units, values), gzip CSV round trip, run and
configuration tables — on a hand-checkable synthetic log and on a short ChironLab toy episode."""
import ast
import gzip
import math
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vegeta import chiron as ch
from vegeta.chiron import export as E
from vegeta.chiron import metrics as M

G = 9.81
W0 = 5.76          # servo no-load speed [rad/s]
STALL = 6.0        # servo stall torque [N·m]


def quat_zyx(yaw=0.0, pitch=0.0, roll=0.0):
    """w, x, y, z of R = Rz(yaw) Ry(pitch) Rx(roll) (elementwise on arrays)."""
    cy, sy = np.cos(np.asarray(yaw) / 2), np.sin(np.asarray(yaw) / 2)
    cp, sp = np.cos(np.asarray(pitch) / 2), np.sin(np.asarray(pitch) / 2)
    cr, sr = np.cos(np.asarray(roll) / 2), np.sin(np.asarray(roll) / 2)
    return np.stack([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy], axis=-1)


def synthetic_log(T=10, dt=0.01, m=10.0):
    """Two bodies, three feet, two actuated joints and one passive body joint, with known values:

    * body 'front' at (0.1 t, 0, 0.15) m with yaw 0.3, pitch 0.2, roll −0.1 rad and ω = (1, 2, 3) rad/s;
      'rear' level, ω = 0, touching the ground (belly) from sample 5 on;
    * foot 'a': force (3, 4, 10) N on a vertical normal → N = 10, tangential 5; 'b': in the air (force 0,
      normal nan); 'c': a tilted normal (0.6, 0, 0.8) with force 20 n + 2 t̂ → N = 20, tangential 2;
      weight 98.1 N, so 2 % = 1.962 N: a and c loaded, b not;
    * joint 'j_free' (actuated): q̇ = 2.88 rad/s (half ω₀, limit 3 N·m) and τ = 3 N·m → torque-saturated,
      not speed-saturated; 'j_fast': q̇ = 5.7 rad/s (≥ 0.98 ω₀) and τ = 0 → speed-saturated only (its torque
      limit is 0.0625 N·m and |τ| = 0 is below 98 % of it); 'body_yaw' passive, q = 0.3 rad.
    """
    t = np.arange(T) * dt
    body_pos = np.zeros((T, 2, 3))
    body_pos[:, 0, 0] = 0.1 * t
    body_pos[:, :, 2] = 0.15
    quat = np.zeros((T, 2, 4))
    quat[:, 0] = quat_zyx(0.3, 0.2, -0.1)
    quat[:, 1] = [1.0, 0.0, 0.0, 0.0]
    angvel = np.zeros((T, 2, 3))
    angvel[:, 0] = [1.0, 2.0, 3.0]
    belly = np.zeros((T, 2), bool)
    belly[5:, 1] = True
    n_tilt = np.array([0.6, 0.0, 0.8])
    t_hat = np.array([0.8, 0.0, -0.6])
    force = np.zeros((T, 3, 3))
    force[:, 0] = [3.0, 4.0, 10.0]
    force[:, 2] = 20 * n_tilt + 2 * t_hat
    normal = np.full((T, 3, 3), np.nan)
    normal[:, 0] = [0.0, 0.0, 1.0]
    normal[:, 2] = n_tilt
    foot_pos = np.zeros((T, 3, 3))
    foot_pos[:, :, 0] = [0.2, 0.0, -0.2]
    foot_pos[:, 1, 2] = 0.03
    q = np.zeros((T, 3))
    q[:, 2] = 0.3
    qd = np.zeros((T, 3))
    qd[:, 0], qd[:, 1] = 0.5 * W0, 5.7
    tau = np.zeros((T, 3))
    tau[:, 0] = 3.0
    nan_p = np.array([1.0, 1.0, np.nan])
    return {
        "t": t, "bodies": ["front", "rear"], "body_group": ["front", "rear"],
        "body_pos": body_pos, "body_quat": quat, "body_angvel": angvel, "belly_contact": belly,
        "com": np.column_stack([0.05 + 0.1 * t, np.zeros(T), np.full(T, 0.15)]),
        "com_vel": np.tile([0.1, 0.0, 0.0], (T, 1)), "total_mass": m, "gravity": np.array([0.0, 0.0, -G]),
        "feet": ["a", "b", "c"], "foot_body": np.array([0, 0, 1]), "foot_force": force, "foot_normal": normal,
        "foot_pos": foot_pos, "joints": ["j_free", "j_fast", "body_yaw"],
        "joint_kind": ["hip_pitch", "knee", "body_yaw"], "joint_active": np.array([True, True, False]),
        "q": q, "qd": qd, "tau": tau, "tau_stall": STALL * nan_p, "qd_noload": W0 * nan_p,
        "q_range": np.array([[-1.0, 1.0], [-1.0, 1.0], [-0.785, 0.785]]),
        "robot": "synthetic", "treatment": "spring", "controller": "fixed", "seed": 4, "v_target": 0.1,
        "terrain": {"kind": "flat"},
    }


# ----------------------------------------------------------------------------------------------- columns and values
def test_raw_frame_columns_values_and_units():
    log = synthetic_log()
    df = E.raw_frame(log, every=2, q0={"body_yaw": 0.1})
    assert df["t"].tolist() == pytest.approx([0.0, 0.02, 0.04, 0.06, 0.08])
    assert len(df) == 5 and df.attrs["every"] == 2 and df.attrs["log_dt_s"] == pytest.approx(0.02)
    expected = (["t"] + [f"{q}@{b}" for b in ("front", "rear") for q in
                         ("x", "y", "z", "roll", "pitch", "yaw", "wx", "wy", "wz", "belly_contact")]
                + [f"{q}@com" for q in ("x", "y", "z", "vx", "vy", "vz")]
                + [f"{q}@{f}" for f in "abc" for q in
                   ("normal_force", "tangential_force", "loaded", "foot_x", "foot_y", "foot_z")]
                + ["q@j_free", "qd@j_free", "tau@j_free", "torque_sat@j_free", "speed_sat@j_free",
                   "q@j_fast", "qd@j_fast", "tau@j_fast", "torque_sat@j_fast", "speed_sat@j_fast",
                   "q@body_yaw", "qd@body_yaw", "tau@body_yaw", "deflection@body_yaw"])
    assert list(df.columns) == expected
    # every column has a unit, and the units are the documented ones
    units = df.attrs["units"]
    assert set(units) == set(df.columns) and all(units.values())
    assert units["roll@front"] == "rad" and units["wz@rear"] == "rad/s" and units["vx@com"] == "m/s"
    assert units["normal_force@a"] == "N" and units["tau@j_free"] == "N·m" and units["deflection@body_yaw"] == "rad"
    assert E.column_units(["loaded@x", "t", "foot_z@b"]) == {"loaded@x": "bool", "t": "s", "foot_z@b": "m"}
    for q in E.QUANTITY_UNITS:                                   # the module docstring documents every quantity
        assert f"``{q}" in E.__doc__ or f"``{q}@" in E.__doc__, q
    r = df.iloc[3]
    # bodies: position, z-y-x Euler angles (metrics' convention), body-frame rates, belly contact
    assert r["x@front"] == pytest.approx(0.1 * 0.06) and r["z@front"] == pytest.approx(0.15)
    assert (r["yaw@front"], r["pitch@front"], r["roll@front"]) == pytest.approx((0.3, 0.2, -0.1))
    assert (r["roll@rear"], r["pitch@rear"], r["yaw@rear"]) == pytest.approx((0.0, 0.0, 0.0))
    assert (r["wx@front"], r["wy@front"], r["wz@front"]) == (1.0, 2.0, 3.0)
    assert df["belly_contact@rear"].tolist() == [False, False, False, True, True]
    assert df["belly_contact@front"].dtype == bool and not df["belly_contact@front"].any()
    assert r["x@com"] == pytest.approx(0.05 + 0.1 * 0.06) and r["vx@com"] == 0.1
    # feet: normal and tangential force, the 2 % loaded rule, the pad centre
    assert (r["normal_force@a"], r["tangential_force@a"]) == pytest.approx((10.0, 5.0))
    assert (r["normal_force@b"], r["tangential_force@b"]) == (0.0, 0.0)
    assert (r["normal_force@c"], r["tangential_force@c"]) == pytest.approx((20.0, 2.0))
    assert (r["loaded@a"], r["loaded@b"], r["loaded@c"]) == (True, False, True)
    np.testing.assert_array_equal(df[["loaded@a", "loaded@b", "loaded@c"]].to_numpy(), M.foot_loaded(log)[::2])
    assert (r["foot_x@a"], r["foot_z@b"]) == (0.2, 0.03)
    # joints: state, saturation flags (actuated joints only), deflection q − q0 (body joints only)
    assert (r["qd@j_free"], r["tau@j_free"]) == (0.5 * W0, 3.0)
    assert bool(r["torque_sat@j_free"]) and not bool(r["speed_sat@j_free"])
    assert bool(r["speed_sat@j_fast"]) and not bool(r["torque_sat@j_fast"])
    assert "torque_sat@body_yaw" not in df and "deflection@j_free" not in df
    assert r["deflection@body_yaw"] == pytest.approx(0.2) and df.attrs["q0_source"].startswith("argument")
    assert df.attrs["meta"]["treatment"] == "spring" and df.attrs["meta"]["seed"] == 4
    assert df.attrs["skipped"] == []


def test_saturation_flags_match_the_metrics_rule():
    rng = np.random.default_rng(0)
    log = synthetic_log(T=400)
    log["qd"][:, :2] = rng.uniform(-6.0, 6.0, (400, 2))
    log["tau"][:, :2] = rng.uniform(-6.0, 6.0, (400, 2))
    df = E.raw_frame(log, every=1, q0=0.0)
    demand = M.actuator_demand({**log, "tau_rated": 2.0 * np.array([1.0, 1.0, np.nan])})
    assert df["torque_sat@j_free"].mean() == pytest.approx(demand["hip_pitch"]["torque_sat_fraction"])
    assert df["speed_sat@j_fast"].mean() == pytest.approx(demand["knee"]["speed_sat_fraction"])


def test_q0_sources():
    log = synthetic_log()
    with pytest.warns(UserWarning, match="joint_springref"):
        df = E.raw_frame(log)
    assert df["deflection@body_yaw"].iloc[0] == pytest.approx(0.3) and df.attrs["q0_source"].startswith("assumed")
    log["joint_springref"] = np.array([0.0, 0.0, 0.05])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        df = E.raw_frame(log)
        assert df["deflection@body_yaw"].iloc[0] == pytest.approx(0.25)
        assert df.attrs["q0_source"] == "log joint_springref"
        assert E.raw_frame(log, q0=[0, 0, 0.3])["deflection@body_yaw"].iloc[0] == pytest.approx(0.0)
        assert E.raw_frame(log, q0=0.1)["deflection@body_yaw"].iloc[0] == pytest.approx(0.2)
    with pytest.raises(ValueError, match="missing"):
        E.raw_frame(log, q0={"j_free": 0.0})
    with pytest.raises(ValueError, match="not in the log"):
        E.raw_frame(log, q0={"body_yaw": 0.0, "nope": 0.0})
    with pytest.raises(ValueError, match="3 joints"):
        E.raw_frame(log, q0=[0.0, 0.1])
    with pytest.raises(ValueError):
        E.raw_frame(log, every=0)


def test_sparse_log_skips_blocks_and_tiny_logs_work():
    log = synthetic_log()
    for k in ("body_angvel", "foot_normal", "tau_stall", "belly_contact"):
        del log[k]
    df = E.raw_frame(log, every=3, q0=0.0)
    assert set(df.attrs["skipped"]) == {"body_angvel", "foot_forces", "saturation", "belly_contact"}
    assert "wx@front" not in df and "normal_force@a" not in df and "torque_sat@j_free" not in df
    assert "foot_x@a" in df and "q@j_free" in df and len(df) == 4
    empty = E.raw_frame({**synthetic_log(T=0), "t": np.zeros(0)}, q0=0.0)
    assert len(empty) == 0 and "roll@front" in empty
    with pytest.raises(ValueError, match="'com'"):
        E.raw_frame({**synthetic_log(), "bodies": ["com", "rear"]}, q0=0.0)


# ----------------------------------------------------------------------------------------------- files
def test_raw_csv_round_trip(tmp_path):
    log = synthetic_log(T=21)
    df = E.raw_frame(log, every=2, q0=0.1)
    path = E.write_raw_csv(log, tmp_path / "runs" / "episode.csv", every=2, q0=0.1)
    assert path == tmp_path / "runs" / "episode.csv.gz"
    assert path.read_bytes()[:2] == b"\x1f\x8b"                         # gzip magic
    header = gzip.open(path, "rt").readline().strip().split(",")
    assert header == list(df.columns)
    back = pd.read_csv(path)
    assert list(back.columns) == list(df.columns) and len(back) == len(df)
    for c in df.columns:
        if df[c].dtype == bool:
            assert back[c].dtype == bool and (back[c] == df[c]).all(), c
        else:
            np.testing.assert_allclose(back[c].to_numpy(), df[c].to_numpy(), rtol=1e-5, atol=1e-12, err_msg=c)
    assert E.write_raw_csv(log, tmp_path / "x.csv.gz", q0=0.1) == tmp_path / "x.csv.gz"


def test_runs_and_configs_csv(tmp_path):
    rows = pd.DataFrame([
        {"trial_id": "a1", "robot_kwargs.body_connection": "spring", "seed": 3, "terrain.rms": 0.027,
         "terrain.level": 0.15, "success": True, "reason": "success", "distance_m": 1.5000000000000002,
         "cot_mech": 1.234567890123, "disturbances": "", "extra": {"k": [1, 2]}},
        {"trial_id": "b2", "robot_kwargs.body_connection": "spring_damper", "seed": 3, "terrain.rms": 0.027,
         "terrain.level": 0.15, "success": False, "reason": "fall", "distance_m": 0.7321, "cot_mech": math.nan,
         "disturbances": "", "extra": None},
    ])
    p = E.write_runs_csv(rows, tmp_path / "runs.csv.gz")
    assert p.read_bytes()[:2] == b"\x1f\x8b"
    back = pd.read_csv(p)
    assert list(back.columns) == list(rows.columns)
    assert back["distance_m"].tolist() == rows["distance_m"].tolist()          # full precision
    assert back["cot_mech"].iloc[0] == rows["cot_mech"].iloc[0] and math.isnan(back["cot_mech"].iloc[1])
    assert back["success"].tolist() == [True, False] and back["seed"].tolist() == [3, 3]
    assert back["extra"].iloc[0] == '{"k": [1, 2]}'
    assert pd.read_csv(E.write_runs_csv(rows, tmp_path / "runs.csv")).shape == rows.shape

    @dataclass
    class Axis:
        k: float = 8.0          # N·m/rad
        c: float = 0.2          # N·m·s/rad

    configs = [
        {"treatment": "spring", "seed": 3, "robot_kwargs": {"body_k_pitch": 8.0, "body_limit_roll": (-0.35, 0.35)},
         "terrain": {"kind": "rough", "rms": 0.027, "seed": 3}, "yaw": Axis(c=0.0)},
        {"treatment": "spring_damper", "seed": 4, "robot_kwargs": {"body_k_pitch": 8.0, "body_c_pitch": 0.2},
         "terrain": {"kind": "rough", "rms": 0.027, "seed": 4}, "yaw": Axis()},
    ]
    assert E.flatten_config(configs[0]) == {
        "treatment": "spring", "seed": 3, "robot_kwargs.body_k_pitch": 8.0,
        "robot_kwargs.body_limit_roll": "[-0.35, 0.35]",
        "terrain.kind": "rough", "terrain.rms": 0.027, "terrain.seed": 3, "yaw.k": 8.0, "yaw.c": 0.0}
    p = E.write_configs_csv(configs, tmp_path / "configs.csv")
    back = pd.read_csv(p)
    assert list(back.columns) == ["treatment", "seed", "robot_kwargs.body_k_pitch", "robot_kwargs.body_limit_roll",
                                  "terrain.kind", "terrain.rms", "terrain.seed", "yaw.k", "yaw.c",
                                  "robot_kwargs.body_c_pitch"]
    assert back["seed"].tolist() == [3, 4] and back["terrain.seed"].tolist() == [3, 4]
    assert math.isnan(back["robot_kwargs.body_c_pitch"].iloc[0]) and back["robot_kwargs.body_c_pitch"].iloc[1] == 0.2
    with pytest.raises(TypeError):
        E.flatten_config(3)


def test_configs_from_trials(tmp_path):
    from vegeta.chiron import experiments as ex

    trials = [ex.Trial(robot="x.py:robot", robot_kwargs={"body_connection": b}, seed=s,
                       terrain={"kind": "rough", "rms": 0.01, "correlation_length": 0.04}, v_target=0.2)
              for b in ("spring", "spring_damper") for s in (0, 1)]
    back = pd.read_csv(E.write_configs_csv(trials, tmp_path / "trials.csv"))
    assert back["seed"].tolist() == [0, 1, 0, 1]
    assert back["robot_kwargs.body_connection"].tolist() == ["spring", "spring", "spring_damper", "spring_damper"]
    assert back["terrain.rms"].tolist() == [0.01] * 4 and back["v_target"].tolist() == [0.2] * 4


# ----------------------------------------------------------------------------------------------- independence
def test_export_is_pure_numpy_pandas_and_lazy():
    tree = ast.parse(Path(E.__file__).read_text())
    mods = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    mods |= {n.module.split(".")[0] for n in ast.walk(tree)
             if isinstance(n, ast.ImportFrom) and n.module and n.level == 0}
    rel = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level > 0}
    assert mods <= {"__future__", "dataclasses", "json", "math", "warnings", "pathlib", "typing", "numpy", "pandas"}
    assert rel <= {"metrics", "servo"}
    code = ("import sys, vegeta.chiron as ch; ch.export.raw_frame; "
            "print('pandas' in sys.modules, 'mujoco' in sys.modules, 'scipy' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
    assert out == ["False", "False", "False"]
    assert "export" in dir(ch)


# ----------------------------------------------------------------------------------------------- a real episode
SERVO = ch.Servo(stall_torque=STALL, rated_torque=2.0, no_load_speed=W0, stall_current=3.0, voltage=12.0, kp=40.0,
                 kd=0.8, source="test values (smart-servo class)")
TAIL_Q0 = 0.05        # rad: the tail hinge's spring rest angle


def toy_with_tail():
    """A four-legged toy trunk with a passive yaw-hinged tail (k 2 N·m/rad, c 0.05 N·m·s/rad, rest 0.05 rad)."""
    legs, feet, nominal = [], [], {}
    for leg, sx, sy in (("FL", 1, 1), ("FR", 1, -1), ("RL", -1, 1), ("RR", -1, -1)):
        shank = ch.Link(f"{leg}_shank", pos=(0, 0, -0.12),
                        joints=[ch.Joint(f"{leg}_knee", axis=(0, 1, 0), range=(-2.5, 0.5), tag="knee", servo=SERVO,
                                         leg=leg)],
                        geoms=[ch.Geom(f"{leg}_shank_g", "capsule", (0.01,), fromto=(0, 0, 0, 0, 0, -0.12), mass=0.05,
                                       role="visual"),
                               ch.Geom(f"{leg}_foot", "sphere", (0.015,), pos=(0, 0, -0.12), mass=0.02,
                                       friction=(0.8, 0.005, 0.0001), role="foot")])
        thigh = ch.Link(f"{leg}_thigh", pos=(0.12 * sx, 0.11 * sy, 0.0),
                        joints=[ch.Joint(f"{leg}_hip", axis=(0, 1, 0), range=(-1.5, 1.5), tag="hip_pitch",
                                         servo=SERVO, leg=leg)],
                        geoms=[ch.Geom(f"{leg}_thigh_g", "capsule", (0.012,), fromto=(0, 0, 0, 0, 0, -0.12),
                                       mass=0.1, role="visual")],
                        children=[shank])
        legs.append(thigh)
        feet.append(ch.FootSpec(leg, f"{leg}_foot", [f"{leg}_hip", f"{leg}_knee"], "trunk"))
        nominal[f"{leg}_hip"], nominal[f"{leg}_knee"] = 0.5, -1.0
    tail = ch.Link("tail", pos=(-0.25, 0.0, 0.0), log=True,
                   joints=[ch.Joint("tail_yaw", axis=(0, 0, 1), pos=(0.08, 0.0, 0.0), range=(-0.7, 0.7), stiffness=2.0,
                                    damping=0.05, springref=TAIL_Q0, tag="body_yaw")],
                   geoms=[ch.Geom("tail_box", "box", (0.08, 0.05, 0.02), mass=0.3, role="body")])
    legs.append(tail)
    nominal["tail_yaw"] = TAIL_Q0
    trunk = ch.Link("trunk", geoms=[ch.Geom("trunk_box", "box", (0.15, 0.1, 0.03), mass=2.0,
                                            friction=(0.5, 0.005, 0.0001), role="body")], children=legs, log=True)
    return ch.Robot("toy with tail", trunk, feet=feet, nominal_qpos=nominal, sources={"all": "test toy"})


class Hold:
    name = "hold"

    def reset(self, lab, seed):
        self.cmd = ch.Command(q_target=lab.nominal_command().q_target)

    def __call__(self, obs):
        return self.cmd


@pytest.fixture(scope="module")
def toy_episode():
    pytest.importorskip("mujoco")
    lab = ch.ChironLab(toy_with_tail(), ch.Flat(), timestep=0.001, control_dt=0.002, log_dt=0.01)
    return lab.run(Hold(), duration=0.4, settle=0.3, seed=0, info={"treatment": "toy"})


def test_raw_frame_of_a_lab_episode(toy_episode, tmp_path):
    log = toy_episode.log
    df = E.raw_frame(toy_episode, every=2, q0={"tail_yaw": TAIL_Q0})
    T = len(log["t"])
    assert len(df) == (T + 1) // 2 and df["t"].tolist() == pytest.approx(log["t"][::2].tolist())
    bodies, feet, joints = list(log["bodies"]), list(log["feet"]), list(log["joints"])
    assert bodies == ["trunk", "tail"] and "tail_yaw" in joints
    for b in bodies:
        quantities = ("x", "y", "z", "roll", "pitch", "yaw", "wx", "wy", "wz", "belly_contact")
        assert {f"{q}@{b}" for q in quantities} <= set(df)
    for f in feet:
        quantities = ("normal_force", "tangential_force", "loaded", "foot_x", "foot_y", "foot_z")
        assert {f"{q}@{f}" for q in quantities} <= set(df)
    actuated = [j for j, a in zip(joints, log["joint_active"]) if a]
    assert {c.split("@")[1] for c in df if c.startswith("torque_sat@")} == set(actuated)
    assert [c for c in df if c.startswith("deflection@")] == ["deflection@tail_yaw"]
    assert set(df.attrs["units"]) == set(df.columns) and df.attrs["skipped"] == []
    # the same numbers as the log and the metrics' definitions
    np.testing.assert_allclose(df[[f"normal_force@{f}" for f in feet]], M.foot_normal_force(log)[::2])
    np.testing.assert_array_equal(df[[f"loaded@{f}" for f in feet]], M.foot_loaded(log)[::2])
    np.testing.assert_allclose(df["deflection@tail_yaw"], log["q"][::2, joints.index("tail_yaw")] - TAIL_Q0)
    np.testing.assert_allclose(df["x@com"], log["com"][::2, 0])
    # physics: standing at rest, the feet carry the weight; the tail's yaw relative to the trunk is its hinge angle
    weight = log["total_mass"] * G
    total = df[[f"normal_force@{f}" for f in feet]].sum(axis=1)
    assert total.iloc[-5:].mean() == pytest.approx(weight, rel=0.03)
    assert df[[f"tangential_force@{f}" for f in feet]].iloc[-1].max() < 0.2 * weight
    rel_yaw = df["yaw@tail"] - df["yaw@trunk"]
    np.testing.assert_allclose(rel_yaw, df["deflection@tail_yaw"] + TAIL_Q0, atol=2e-3)
    assert np.abs(df["deflection@tail_yaw"]).max() < 0.01        # springs unloaded at the start, nothing pushes yaw
    # gzip CSV round trip of the real episode
    path = E.write_raw_csv(toy_episode, tmp_path / "toy.csv.gz", q0={"tail_yaw": TAIL_Q0})
    back = pd.read_csv(path)
    assert list(back.columns) == list(df.columns)
    flags = [c for c in df if df[c].dtype == bool]
    assert flags and (back[flags] == df[flags]).all().all() and (back[flags].dtypes == bool).all()
    values = [c for c in df if c not in flags]                  # an all-zero column (a passive τ) reads back as int
    np.testing.assert_allclose(back[values].to_numpy(float), df[values].to_numpy(float), rtol=1e-5, atol=1e-9)
