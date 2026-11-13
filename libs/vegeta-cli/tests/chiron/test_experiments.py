"""Paired trials: Trial identity and pickling, the cached process-pool runner, rows, paired-design helpers.

The toy robot and its controller are defined here and referenced by file path ('<this file>:toy_robot'), the
way a study references its design files. ``test_cli.py`` uses the same factories.
"""
import json
import math
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from vegeta.chiron import Command, FootSpec, Geom, Joint, Link, PhaseGenerator, Robot, Servo
from vegeta.chiron import experiments as ex

HERE = Path(__file__).resolve()
ROBOT = f"{HERE}:toy_robot"
WALK = f"{HERE}:toy_walk"
SERVO = Servo(stall_torque=6.0, rated_torque=2.0, no_load_speed=5.76, stall_current=3.0, voltage=12.0, kp=40.0,
              kd=0.8, source="test values (smart-servo class)")
THIGH = SHANK = 0.12          # [m]
LEGS = (("FL", 1, 1), ("FR", 1, -1), ("RL", -1, 1), ("RR", -1, -1))


# ----------------------------------------------------------------------------------------------- the toy
def toy_robot(trunk_mass: float = 2.0, treatment: str = "") -> Robot:
    """A four-legged toy (hip pitch + knee per leg, sphere feet, feet under the hips when standing)."""
    legs, feet, nominal = [], [], {}
    for leg, sx, sy in LEGS:
        shank = Link(f"{leg}_shank", pos=(0, 0, -THIGH),
                     joints=[Joint(f"{leg}_knee", axis=(0, 1, 0), range=(-2.5, 0.5), tag="knee", servo=SERVO, leg=leg)],
                     geoms=[Geom(f"{leg}_shank_g", "capsule", (0.01,), fromto=(0, 0, 0, 0, 0, -SHANK), mass=0.05,
                                 role="visual"),
                            Geom(f"{leg}_foot", "sphere", (0.015,), pos=(0, 0, -SHANK), mass=0.02,
                                 friction=(0.8, 0.005, 0.0001), role="foot")])
        thigh = Link(f"{leg}_thigh", pos=(0.12 * sx, 0.11 * sy, 0.0),
                     joints=[Joint(f"{leg}_hip", axis=(0, 1, 0), range=(-1.5, 1.5), tag="hip_pitch", servo=SERVO,
                                   leg=leg)],
                     geoms=[Geom(f"{leg}_thigh_g", "capsule", (0.012,), fromto=(0, 0, 0, 0, 0, -THIGH), mass=0.1,
                                 role="visual")],
                     children=[shank])
        legs.append(thigh)
        feet.append(FootSpec(leg, f"{leg}_foot", [f"{leg}_hip", f"{leg}_knee"], "trunk"))
        nominal[f"{leg}_hip"], nominal[f"{leg}_knee"] = 0.5, -1.0
    trunk = Link("trunk", geoms=[Geom("trunk_box", "box", (0.15, 0.1, 0.03), mass=trunk_mass,
                                      friction=(0.5, 0.005, 0.0001), role="body")], children=legs, log=True)
    robot = Robot("toy", trunk, feet=feet, nominal_qpos=nominal, sources={"all": "test toy"})
    if treatment:
        robot.treatment = treatment
    return robot


def _leg_ik(x, z):
    """Hip and knee angles for a foot at (x, z) from the hip (z < 0): absolute link angles a1, a2 from -z."""
    d = min(math.hypot(x, z), 0.999 * (THIGH + SHANK))
    beta = math.acos(d / (THIGH + SHANK))
    phi = math.atan2(-x, -z)
    return phi + beta, -2.0 * beta


class ToyWalk:
    """A lateral-sequence walk: stance feet move back at v_target, swing feet return on a lifted cosine."""

    def __init__(self, v_target=0.1, stride=0.08, lift=0.03, depth=0.2106, duty=0.75, name="toy walk"):
        self.v, self.stride, self.lift, self.depth, self.duty, self.name = v_target, stride, lift, depth, duty, name

    def reset(self, lab, seed):
        feet = list(lab.feet)
        base = {"RL": 0.0, "FL": 0.25, "RR": 0.5, "FR": 0.75}
        self.gen = PhaseGenerator(feet, {f: base[f] for f in feet}, self.duty, self.v / self.stride)
        self.gen.reset(offset=float(np.random.default_rng(seed).uniform()) if seed is not None else 0.0)
        act = {j: i for i, j in enumerate(lab.actuated_joints)}
        self.idx = [(act[f"{f}_hip"], act[f"{f}_knee"]) for f in feet]
        self.n = len(lab.actuated_joints)
        self.dt = lab.control_dt

    def __call__(self, obs):
        c = self.gen.phases
        q = np.empty(self.n)
        L = self.stride * self.duty
        for i, (ih, ik) in enumerate(self.idx):
            if c[i] < self.duty:
                x, lift = L / 2 - L * c[i] / self.duty, 0.0
            else:
                s = (c[i] - self.duty) / (1 - self.duty)
                x, lift = -L / 2 + L * (1 - math.cos(math.pi * s)) / 2, self.lift * (1 - math.cos(2 * math.pi * s)) / 2
            q[ih], q[ik] = _leg_ik(x, -self.depth + lift)
        phases = c.copy()
        self.gen.step(self.dt)
        return Command(q_target=q, leg_phase=phases, leg_stance=phases < self.duty)


def toy_walk(v_target=0.1, stride=0.08, lift=0.03):
    return ToyWalk(v_target, stride, lift)


def broken_controller(v_target=0.1):
    """A controller that fails in the middle of a run (a failed run, not invalid input)."""
    class Broken(ToyWalk):
        def __call__(self, obs):
            if obs.t > 0.2:
                raise FloatingPointError("controller diverged")
            return super().__call__(obs)

    return Broken(v_target)


def crashing_controller(v_target=0.1):
    """A controller whose process dies (as on a segfault): only ever used in a worker process."""
    class Crash(ToyWalk):
        def __call__(self, obs):
            if obs.t > 0.1:
                os._exit(3)
            return super().__call__(obs)

    return Crash(v_target)


def flat_or_ridges(kind="flat", level=None, seed=0, spacing=0.17):
    """A terrain factory (a study's normalisation): level = ridge height [m]."""
    if kind == "flat" or not level:
        return {"kind": "flat"}
    return {"kind": "long_bumps", "height": level, "spacing": spacing, "width": spacing / 2, "start": 0.3,
            "jitter": spacing / 4, "seed": seed, "level": level}


def stub_metrics(log, outcome, scale=1.0):
    """A cheap metrics function: progress and the number of samples."""
    com = np.asarray(log["com"])
    return {"stub_progress_m": float(com[-1, 0] - com[0, 0]) * scale, "n_samples": int(len(log["t"])),
            "reason": outcome["reason"]}


def toy_trial(seed=0, mass=2.0, stride=0.08, **kw):
    base = dict(robot=ROBOT, controller=WALK, robot_kwargs={"trunk_mass": mass}, controller_kwargs={"stride": stride},
                terrain={"kind": "flat"}, seed=seed, v_target=0.1, course_m=0.6, rules={"timeout": 1.0},
                settle=0.2, lab_kwargs={"control_dt": 0.002})
    base.update(kw)
    return ex.Trial(**base)


# ----------------------------------------------------------------------------------------------- Trial
def test_trial_identity_ignores_labels_and_is_stable_across_processes():
    a = toy_trial(seed=3)
    assert a.key() == toy_trial(seed=3).key()
    assert a.key() == a.replace(name="renamed", factors={"x": 1}).key()
    assert a.key() != toy_trial(seed=4).key()
    assert a.key() != toy_trial(seed=3, mass=2.5).key()
    assert a.key() != a.replace(version="v2").key()
    assert a.key() != a.key(metrics_fn=f"{HERE}:stub_metrics")
    assert a.key() == ex.Trial(**{**a.to_dict(), "robot_kwargs": {"trunk_mass": 2.0}}).key()
    # dict order does not matter
    assert toy_trial(rules={"timeout": 1.0, "max_tilt_deg": 50}).key() == \
        toy_trial(rules={"max_tilt_deg": 50, "timeout": 1.0}).key()
    code = (f"import sys; sys.path.insert(0, {str(HERE.parent)!r}); import test_experiments as t; "
            f"print(t.toy_trial(seed=3).key())")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == a.key()


def test_trial_pickles_and_resolves_the_terrain_seed():
    t = toy_trial(seed=7, terrain={"kind": "rough", "rms": 0.005, "correlation_length": 0.04, "start": 0.3,
                                   "level": 0.1},
                  disturbances=[{"body": "trunk", "t_start": 0.3, "duration": 0.05, "impulse": 0.5,
                                 "direction": (0, 1, 0)}])
    u = pickle.loads(pickle.dumps(t))
    assert u == t and u.key() == t.key()
    assert t.terrain_spec()["seed"] == 7
    assert toy_trial(seed=7, terrain={"kind": "rough", "rms": 0.005, "correlation_length": 0.04,
                                      "seed": 1}).terrain_spec()["seed"] == 1
    assert "seed" not in toy_trial(terrain={"kind": "cross_slope", "angle_deg": 5}).terrain_spec()
    flat = t.flat()
    assert flat["robot_factory"] == ROBOT and flat["terrain.kind"] == "rough" and flat["terrain.seed"] == 7
    assert flat["robot_kwargs.trunk_mass"] == 2.0 and flat["rules.timeout"] == 1.0
    assert json.loads(flat["disturbances"])[0]["impulse"] == 0.5
    with pytest.raises(TypeError):
        ex.Trial(robot=toy_robot)                       # factories are references, not objects
    with pytest.raises(TypeError):
        toy_trial(robot_kwargs={"x": object()}).key()   # no stable form -> no silent unstable hash


def test_load_ref_reuses_module_and_reports_bad_refs():
    assert ex.load_ref(ROBOT) is ex.load_ref(ROBOT)
    assert ex.load_ref("vegeta.chiron.metrics:trial_metrics").__name__ == "trial_metrics"
    with pytest.raises(ValueError):
        ex.load_ref("no_colon")
    with pytest.raises(FileNotFoundError):
        ex.load_ref("/nonexistent/robot.py:make")
    with pytest.raises(ValueError):
        ex.load_ref(f"{HERE}:no_such_factory")


def test_trial_run_passes_v_target_info_and_disturbances():
    t = toy_trial(seed=1, robot_kwargs={"treatment": "stiff"}, info={"payload": "trunk"},
                  disturbances=[{"body": "trunk", "t_start": 0.3, "duration": 0.05, "impulse": 0.4}])
    ep = t.run()
    log = ep.log
    assert log["controller"] == "toy walk" and log["treatment"] == "stiff"     # the robot's treatment attribute
    assert log["v_target"] == 0.1 and log["course_m"] == 0.6 and log["payload"] == "trunk"
    assert log["seed"] == 1 and log["terrain"]["kind"] == "flat"
    assert log["disturbances"][0]["impulse_Ns"] == pytest.approx(0.4)
    assert ep.outcome["reason"] in ("success", "fall", "stall", "off_course", "timeout")
    assert ep.outcome["t_end"] <= 1.0 + 1e-9
    assert "leg_phase" in log


# ----------------------------------------------------------------------------------------------- the runner
def _two_by_two():
    """2 configurations (trunk mass) x 2 strides on 2 paired seeds = 8 trials (a 2 x 2 design per seed)."""
    def make(mass, stride, seed):
        return toy_trial(seed=seed, mass=mass, stride=stride, name=f"m{mass}-s{stride}-{seed}")
    return ex.paired_trials(make, {"mass": [2.0, 2.5], "stride": [0.06, 0.08]}, ex.paired_seeds(2))


def test_run_trials_pool_rows_and_cache(tmp_path):
    trials = _two_by_two()
    assert len(trials) == 8 and len({t.key() for t in trials}) == 8
    stub = f"{HERE}:stub_metrics"
    t0 = time.perf_counter()
    df = ex.run_trials(trials, processes=2, cache_dir=tmp_path, metrics_fn=stub, progress=False)
    first = time.perf_counter() - t0
    assert len(df) == 8 and (df["status"] == "ok").all() and not df["cached"].any()
    assert df.attrs["factors"] == ["mass", "stride", "seed"] and df.attrs["n_run"] == 8
    assert list(df["name"]) == [t.name for t in trials]                      # input order, not completion order
    for col in ("trial_id", "mass", "stride", "seed", "robot_factory", "robot_kwargs.trunk_mass",
                "controller_factory", "controller_kwargs.stride", "terrain.kind", "v_target", "course_m",
                "rules.timeout", "lab_kwargs.control_dt", "metrics_fn", "success", "reason", "t_end", "distance_m",
                "x_end", "robot", "controller", "total_mass", "wall_time_s", "stub_progress_m", "n_samples"):
        assert col in df.columns, col
    assert "metric.reason" not in df.columns                                 # same value as the outcome's
    assert np.allclose(df["total_mass"], np.where(df["mass"] == 2.0, 2.68, 3.18))
    assert np.allclose(df["stub_progress_m"], df["distance_m"])
    assert len(list((tmp_path / "rows").glob("*.json"))) == 8
    # a re-run reads every row from the cache
    t0 = time.perf_counter()
    again = ex.run_trials(trials, processes=2, cache_dir=tmp_path, metrics_fn=stub, progress=False)
    assert again["cached"].all() and again.attrs["n_run"] == 0
    assert time.perf_counter() - t0 < 0.5 * first
    cols = [c for c in df.columns if c not in ("cached",)]
    assert again[cols].equals(df[cols])
    # one new trial runs, the rest stay cached; relabelling does not re-run
    more = [t.replace(name=t.name + "*") for t in trials] + [toy_trial(seed=9)]
    third = ex.run_trials(more, processes=2, cache_dir=tmp_path, metrics_fn=stub, progress=False)
    assert third.attrs["n_run"] == 1 and third["cached"].sum() == 8 and third["name"].iloc[0].endswith("*")
    # the cache can be read back on its own
    assert len(ex.load_cached(tmp_path)) == 9


def test_in_process_run_matches_the_pool_bitwise(tmp_path):
    trials = _two_by_two()[:2]
    stub = f"{HERE}:stub_metrics"
    a = ex.run_trials(trials, processes=1, metrics_fn=stub, progress=False)
    b = ex.run_trials(trials, processes=2, metrics_fn=stub, progress=False)
    assert list(a["distance_m"]) == list(b["distance_m"]) and list(a["t_end"]) == list(b["t_end"])


def test_default_metrics_in_the_workers(tmp_path):
    trials = [toy_trial(seed=s, metrics_kwargs={"feasibility_every": 20}) for s in (0, 1)]
    df = ex.run_trials(trials, processes=2, cache_dir=tmp_path, progress=False)
    assert (df["status"] == "ok").all(), df["error"].tolist()
    assert df["metrics_fn"].iloc[0] == "vegeta.chiron.metrics:trial_metrics"
    for col in ("achieved_speed_m_s", "cot_mech", "worst_tilt_p95_deg", "slip_per_m", "duty_mean",
                "support_margin_min_m", "metrics_kwargs.feasibility_every"):
        assert col in df.columns, col
    assert np.all(np.isfinite(df["cot_mech"]))


def test_errors_are_recorded_not_cached_or_raised(tmp_path):
    good = toy_trial(seed=0)
    bad = toy_trial(seed=0, controller_kwargs={"no_such_option": 1})
    broken = toy_trial(seed=0, controller=f"{HERE}:broken_controller", controller_kwargs={})
    stub = f"{HERE}:stub_metrics"
    df = ex.run_trials([good, bad, broken], processes=2, cache_dir=tmp_path, metrics_fn=stub, progress=False)
    assert list(df["status"]) == ["ok", "error", "error"]
    assert "no_such_option" in df["error"].iloc[1] and "diverged" in df["error"].iloc[2]
    assert df.attrs["n_failed"] == 2
    assert len(list((tmp_path / "rows").glob("*.json"))) == 1
    with pytest.raises(RuntimeError, match="no_such_option"):
        ex.run_trials([bad], processes=1, metrics_fn=stub, progress=False, on_error="raise")


def test_a_dying_worker_does_not_hang_the_sweep(tmp_path):
    crash = toy_trial(seed=0, controller=f"{HERE}:crashing_controller", controller_kwargs={}, name="crash")
    trials = [toy_trial(seed=0, name="a"), crash, toy_trial(seed=1, name="b")]
    df = ex.run_trials(trials, processes=2, cache_dir=tmp_path, metrics_fn=f"{HERE}:stub_metrics", progress=False)
    row = df.set_index("name").loc["crash"]
    assert row["status"] == "error" and "worker process died (exit code 3)" in row["error"]
    assert list(df["status"]) == ["ok", "error", "ok"]                       # the others are not taken down
    assert len(list((tmp_path / "rows").glob("*.json"))) == 2


def test_keep_logs_and_rerun_with_log(tmp_path):
    t = toy_trial(seed=2)
    stub = f"{HERE}:stub_metrics"
    df = ex.run_trials([t], processes=1, cache_dir=tmp_path, metrics_fn=stub, keep_logs=True, progress=False)
    path = Path(df["log_path"].iloc[0])
    assert path.is_file() and path.suffix == ".npz"
    loaded = ex.rerun_with_log(t, cache_dir=tmp_path, metrics_fn=stub)        # read from the kept log
    fresh = ex.rerun_with_log(t, metrics_fn=stub)                             # simulated again
    assert np.array_equal(loaded.log["com"], fresh.log["com"])
    assert np.array_equal(fresh.log["q"], t.run().log["q"])
    assert fresh.outcome["distance_m"] == df["distance_m"].iloc[0]
    geo = ex.rerun_with_log(t, cache_dir=tmp_path, metrics_fn=stub, log_geoms=True, save=tmp_path / "geo.npz")
    assert "geom_pose" in geo.log and (tmp_path / "geo.npz").is_file()
    assert np.array_equal(geo.log["com"], fresh.log["com"])                  # logging does not change the run


def test_terrain_factory_and_level_bookkeeping():
    t = toy_trial(seed=5, terrain=f"{HERE}:flat_or_ridges", terrain_kwargs={"kind": "ridges", "level": 0.01})
    terrain, extra = t.make_terrain()
    assert terrain.spec()["kind"] == "long_bumps" and terrain.spec()["seed"] == 5 and extra == {"level": 0.01}
    lab, ctrl, rules, info = t.build()
    assert info["terrain"] == {"level": 0.01} and rules.course_m == 0.6 and rules.timeout == 1.0
    assert t.flat()["terrain_factory"].endswith(":flat_or_ridges") and t.flat()["terrain_kwargs.level"] == 0.01
    spec = toy_trial(terrain={"kind": "long_bumps", "height": 0.01, "spacing": 0.17, "width": 0.085,
                              "level": 0.0556})
    assert spec.make_terrain()[1] == {"level": 0.0556}


def test_rules_need_a_course_or_a_duration():
    with pytest.raises(ValueError):
        toy_trial(course_m=None, rules={}).make_rules()
    with pytest.raises(ValueError):
        toy_trial(course_m=None).make_rules()                                 # rules without a course
    assert toy_trial(course_m=None, rules={}, duration=0.3).make_rules() is None
    ep = toy_trial(course_m=None, rules={}, duration=0.3, controller=None).run()
    assert ep.outcome["reason"] == "completed" and ep.log["controller"] == "stand"


# ----------------------------------------------------------------------------------------------- paired designs
def test_paired_helpers():
    assert ex.paired_seeds(3) == [0, 1, 2] and ex.paired_seeds(2, start=10) == [10, 11]
    a, b = ex.paired_seeds(30, stream="tuning"), ex.paired_seeds(30, stream="confirmation")
    assert a == ex.paired_seeds(30, stream="tuning") and not set(a) & set(b) and not set(a) & set(range(30))
    grid = ex.factor_grid({"body": ["rigid", "flex"]}, controller=["fixed", "adaptive"])
    assert grid == [{"body": "rigid", "controller": "fixed"}, {"body": "rigid", "controller": "adaptive"},
                    {"body": "flex", "controller": "fixed"}, {"body": "flex", "controller": "adaptive"}]
    with pytest.raises(TypeError):
        ex.factor_grid(body="rigid")
    trials = ex.paired_trials(lambda body, controller, seed: toy_trial(seed=seed, name=body),
                              {"body": ["rigid", "flex"], "controller": ["fixed", "adaptive"]}, [4, 5])
    assert len(trials) == 8 and trials[0].factors == {"body": "rigid", "controller": "fixed", "seed": 4}
    frame = ex.trials_frame(trials)
    assert list(frame.columns[:5]) == ["trial_id", "name", "body", "controller", "seed"] and len(frame) == 8


def test_pair_up_aligns_on_the_other_factors():
    import pandas as pd

    df = pd.DataFrame({"body": ["rigid", "flex"] * 4, "level": [1, 1, 2, 2] * 2, "seed": [0] * 4 + [1] * 4,
                       "success": [True, False, True, True, False, False, True, False]})
    df.attrs["factors"] = ["body", "level", "seed"]
    p = ex.pair_up(df.iloc[::-1], "body", "rigid", "flex", "success")
    assert list(p.columns) == ["level", "seed", "rigid", "flex"] and len(p) == 4
    row = p[(p["level"] == 1) & (p["seed"] == 0)].iloc[0]
    assert row["rigid"] and not row["flex"]
    with pytest.raises(ValueError):
        ex.pair_up(df, "body", "rigid", "flex", "success", on=["seed"])     # not unique without the level
