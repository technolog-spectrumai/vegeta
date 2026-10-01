"""benchmark/cleopatra: the trial lists of Cleopatra's body-joint study (docs/myropod_stability.md, Amendment D §12) and the runner that
writes its CSV files.

Treatments: ``spring`` (τ = −k (q − q₀)) and ``spring_damper`` (τ = −k (q − q₀) − c q̇), identical joints otherwise;
crossed with the controllers ``fixed`` (baseline, σ = 0) and ``adaptive`` (load feedback, σ = 0.6 with the
phase-rate floor of §13.1). Every cell of an experiment shares the seeds, hence the terrain and the initial gait
phase (``myropod_controller.phase_offset(seed)``): the runs pair across treatments and controllers.

Experiments (protocol counts are the defaults; ``scale`` reduces them for smoke tests and pilots, and the reduction
is recorded in every row as ``info.scale``):

=====================  ========================================================================================
``main``               flat, longitudinal bumps, alternating bumps, cross-slope, rough (steps optional) at every
                       level of §4, v = 0.2 m/s, treatment × controller, 30 seeds
``speed_maps``         rough and alternating bumps × every level × v ∈ {0.1, 0.2, 0.3} m/s × treatment ×
                       controller, 10 seeds
``disturbance``        flat, v = 0.2 m/s, a sideways impulse J ∈ {0, 1, 2, 3, 4, 6, 8} N·s on segment 2 at 3.0 s of
                       walking for 50 ms, treatment × controller, 30 seeds; course 2.0 m so the 5 s recovery window
                       fits before the finish
``onset``              flat, v ∈ {0.1, 0.2, 0.3, 0.4, 0.5} m/s, body-yaw stiffness k ∈ {2, 4, 8, 16} N·m/rad (the
                       same in both treatments), treatment × controller, 10 seeds
``damping``            rough and alternating bumps at h/L = 0.15, fixed controller, c ∈ {0, 0.05, 0.2, 0.8}
                       N·m·s/rad on every axis (c = 0 is the spring-only model), 30 seeds
``roll``               alternating bumps (h/L 0.15) and cross-slope (15°), body roll hinge on and off (identical in
                       both treatments), treatment × fixed controller, 30 seeds
=====================  ========================================================================================

§7.6 (equal controller tuning) is not part of this round: controller settings are held fixed in every comparison.

Terrain scaling (§4): heights as a fraction of the leg length L = femur + tibia = 0.18 m, spacings as a fraction of
the segment pitch P = 0.170 m; bumps P apart and P/2 wide with ±P/4 seeded jitter; alternating bumps under the left
and right foot tracks (y = ±0.131 m); steps alternate up and down every 2P; rough ground has an RMS height of
(h/L)·L and a correlation length of P/4; every course is flat for its first 0.3 m. Each row records the terrain in SI
and normalised form (``info.height_m``, ``info.h_over_L``, ``info.spacing_m``, ``info.spacing_over_P``,
``info.slope_deg``, ``info.rms_m``, ``info.correlation_m``).

Physics: ``LAB`` — 0.25 ms physics step (the §12.5 / §13.2 convergence result: 1 ms and 0.5 ms change outcomes and
speeds by up to 16 %, 0.125 ms agrees with 0.25 ms within 2 %), control 1 kHz, log 100 Hz.

    python3 benchmark/cleopatra/full_benchmark.py --scale full          # everything, JSON + CSV + plots
    import experiments as ce                                            # from benchmark/cleopatra
    df = ce.run_experiment("main", "benchmark/cleopatra/results/smoke", scale="smoke")
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from vegeta.chiron import experiments as cx
from vegeta.chiron import metrics as cm

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DESIGNS = REPO / "notebooks" / "designs"                 # Cleopatra's robot, controller and the shared plot helpers
if str(DESIGNS) not in sys.path:
    sys.path.insert(0, str(DESIGNS))
ROBOT = f"{DESIGNS / 'myropod_robot.py'}:cleopatra"
CONTROLLERS = {"fixed": f"{DESIGNS / 'myropod_controller.py'}:fixed",
               "adaptive": f"{DESIGNS / 'myropod_controller.py'}:adaptive"}
METRICS = f"{HERE / 'experiments.py'}:study_metrics"
VERSION = "amendment-D-1"            # bump when a factory, controller or metric changes (the cache cannot see code)

L_LEG = 0.18                          # m, femur + tibia
P_SEG = 0.170                         # m, segment pitch
TRACK_Y = 0.131                       # m, foot track: hip y 0.061 + foot out 0.070
START = 0.3                           # m, flat start
COURSE = 1.5                          # m
LAB = dict(timestep=0.00025, control_dt=0.001, log_dt=0.01)
TREATMENTS = ("spring", "spring_damper")
LEVELS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)       # h/L
SLOPES = (5.0, 10.0, 15.0, 20.0, 25.0)               # deg
SPEEDS = (0.1, 0.2, 0.3)                             # m/s
ONSET_SPEEDS = (0.1, 0.2, 0.3, 0.4, 0.5)            # m/s
ONSET_K = (2.0, 4.0, 8.0, 16.0)                      # N·m/rad, body yaw
IMPULSES = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)       # N·s
DAMPINGS = (0.0, 0.05, 0.2, 0.8)                     # N·m·s/rad
EXPERIMENTS = ("main", "speed_maps", "disturbance", "onset", "damping", "roll")

#: Counts per scale: seeds, and which levels/speeds are kept. 'full' is the protocol.
SCALES = {
    "full": dict(seeds_main=30, seeds_maps=10, seeds_dist=30, seeds_onset=10, seeds_sens=30, levels=LEVELS,
                 slopes=SLOPES, speeds=SPEEDS, onset_speeds=ONSET_SPEEDS, onset_k=ONSET_K, impulses=IMPULSES,
                 dampings=DAMPINGS),
    "pilot": dict(seeds_main=4, seeds_maps=2, seeds_dist=4, seeds_onset=2, seeds_sens=4, levels=(0.15, 0.30),
                  slopes=(15.0,), speeds=SPEEDS, onset_speeds=(0.1, 0.3, 0.5), onset_k=(2.0, 16.0),
                  impulses=(0.0, 3.0, 6.0), dampings=(0.0, 0.2, 0.8)),
    "smoke": dict(seeds_main=1, seeds_maps=1, seeds_dist=1, seeds_onset=1, seeds_sens=1, levels=(0.15,),
                  slopes=(15.0,), speeds=(0.2,), onset_speeds=(0.3,), onset_k=(8.0,), impulses=(3.0,),
                  dampings=(0.2,)),
}


def terrain_spec(kind: str, level: float) -> tuple[dict, dict]:
    """(chiron terrain spec, recorded terrain parameters) for a §4 terrain at a difficulty ``level`` (h/L for
    bumps, steps and rough; degrees for cross-slope; ignored for flat). The seed is added by the trial."""
    rec = dict(terrain=kind, level=float(level), height_m=0.0, h_over_L=0.0, spacing_m=0.0, spacing_over_P=0.0,
               slope_deg=0.0, rms_m=0.0, correlation_m=0.0)
    if kind == "flat":
        return {"kind": "flat", "level": 0.0}, rec
    if kind == "cross_slope":
        rec.update(slope_deg=float(level))
        return {"kind": "cross_slope", "angle_deg": float(level), "start": START, "level": float(level)}, rec
    h = float(level) * L_LEG
    if kind == "long_bumps":
        rec.update(height_m=h, h_over_L=float(level), spacing_m=P_SEG, spacing_over_P=1.0)
        return {"kind": "long_bumps", "height": h, "spacing": P_SEG, "width": P_SEG / 2, "start": START,
                "jitter": P_SEG / 4, "level": float(level)}, rec
    if kind == "alt_bumps":
        rec.update(height_m=h, h_over_L=float(level), spacing_m=P_SEG, spacing_over_P=1.0)
        return {"kind": "alt_bumps", "height": h, "spacing": P_SEG, "width": P_SEG / 2, "track_y": TRACK_Y,
                "start": START, "jitter": P_SEG / 4, "level": float(level)}, rec
    if kind == "steps":
        rec.update(height_m=h, h_over_L=float(level), spacing_m=2 * P_SEG, spacing_over_P=2.0)
        return {"kind": "steps", "height": h, "spacing": 2 * P_SEG, "start": START, "jitter": P_SEG / 4,
                "mode": "alternate", "level": float(level)}, rec
    if kind == "rough":
        rec.update(height_m=h, h_over_L=float(level), rms_m=h, correlation_m=P_SEG / 4)
        return {"kind": "rough", "rms": h, "correlation_length": P_SEG / 4, "start": START,
                "level": float(level)}, rec
    raise ValueError(f"unknown terrain {kind!r}")


def make_trial(experiment: str, treatment: str, controller: str, terrain: str, level: float, seed: int, *,
               v_target: float = 0.2, course_m: float = COURSE, robot_kwargs: dict | None = None,
               disturbances: list | None = None, scale: str = "full", raw_dir: str | None = None,
               raw_every: int = 2, feasibility_every: int = 10, extra_info: dict | None = None) -> cx.Trial:
    """One run of the study as a chiron Trial (everything that changes the physics is in the spec)."""
    tspec, rec = terrain_spec(terrain, level)
    rk = {"body_connection": treatment, **(robot_kwargs or {})}
    j = sum(float(d.get("impulse") or 0.0) for d in (disturbances or []))
    info = {"experiment": experiment, "scale": scale, "treatment": treatment, "controller_name": controller,
            "impulse_Ns": j, **rec, **(extra_info or {})}
    name = (f"{experiment}|{treatment}|{controller}|{terrain}:{level:g}|v{v_target:g}|J{j:g}|"
            + "|".join(f"{k}={rk[k]}" for k in sorted(rk) if k != "body_connection") + f"|s{seed}")
    raw_name = name.replace("|", "__").replace(":", "-").replace("=", "-")
    import myropod_robot as mr                                    # the springs' rest angles, for the raw deflections
    roll = mr.as_bool(rk.get("body_roll_axis", True), "body_roll_axis")
    axes = ("yaw", "pitch") + (("roll",) if roll else ())
    raw_q0 = {f"body {i}-{i + 1} {a}": float(rk.get(f"body_q0_{a}", 0.0)) for i in (1, 2) for a in axes}
    return cx.Trial(robot=ROBOT, controller=CONTROLLERS[controller], terrain=tspec, seed=int(seed),
                    robot_kwargs=rk, v_target=float(v_target), course_m=float(course_m),
                    disturbances=list(disturbances or []), lab_kwargs=dict(LAB), info=info,
                    metrics_kwargs={"raw_dir": raw_dir, "raw_name": raw_name, "raw_every": int(raw_every),
                                    "raw_q0": raw_q0, "feasibility_every": int(feasibility_every)},
                    version=VERSION, name=name, factors={})


def trials(experiment: str, *, scale: str = "full", raw_dir: str | None = None, raw_every: int = 2,
           feasibility_every: int = 10, include_steps: bool = False) -> list[cx.Trial]:
    """The trial list of one experiment (see the module docstring)."""
    s = SCALES[scale]
    kw = dict(scale=scale, raw_dir=raw_dir, raw_every=raw_every, feasibility_every=feasibility_every)
    out: list[cx.Trial] = []

    def cells(seeds, terrains, controllers=("fixed", "adaptive"), speeds=(0.2,), treatments=TREATMENTS, **mk):
        for seed in cx.paired_seeds(seeds):
            for terrain, level in terrains:
                for v in speeds:
                    for ctrl in controllers:
                        for tr in treatments:
                            out.append(make_trial(experiment, tr, ctrl, terrain, level, seed, v_target=v, **mk, **kw))

    if experiment == "main":
        terr = [("flat", 0.0)] + [(k, lv) for k in ("long_bumps", "alt_bumps", "rough") for lv in s["levels"]]
        terr += [("cross_slope", a) for a in s["slopes"]]
        if include_steps:
            terr += [("steps", lv) for lv in s["levels"]]
        cells(s["seeds_main"], terr)
    elif experiment == "speed_maps":
        cells(s["seeds_maps"], [(k, lv) for k in ("rough", "alt_bumps") for lv in s["levels"]], speeds=s["speeds"])
    elif experiment == "disturbance":
        for J in s["impulses"]:
            dist = [] if J == 0 else [{"body": "segment 2", "t_start": 3.0, "duration": 0.05, "impulse": float(J),
                                       "direction": (0.0, 1.0, 0.0)}]
            cells(s["seeds_dist"], [("flat", 0.0)], course_m=2.0, disturbances=dist,
                  extra_info={"impulse_level_Ns": float(J)})
    elif experiment == "onset":
        for k in s["onset_k"]:
            cells(s["seeds_onset"], [("flat", 0.0)], speeds=s["onset_speeds"], robot_kwargs={"body_k_yaw": float(k)},
                  extra_info={"k_yaw": float(k)})
    elif experiment == "damping":
        for c in s["dampings"]:
            rk = {"body_c_pitch": float(c), "body_c_yaw": float(c), "body_c_roll": float(c)}
            cells(s["seeds_sens"], [("rough", 0.15), ("alt_bumps", 0.15)], controllers=("fixed",),
                  treatments=("spring_damper",), robot_kwargs=rk, extra_info={"c_all": float(c)})
    elif experiment == "roll":
        for roll in (True, False):
            cells(s["seeds_sens"], [("alt_bumps", 0.15), ("cross_slope", 15.0)], controllers=("fixed",),
                  robot_kwargs={"body_roll_axis": roll}, extra_info={"roll_axis": bool(roll)})
    else:
        raise ValueError(f"unknown experiment {experiment!r}; known: {EXPERIMENTS}")
    return out


def study_metrics(log, outcome, *, raw_dir=None, raw_name=None, raw_every=2, raw_q0=None, feasibility_every=10,
                  **kw) -> dict:
    """Metrics of one run (in the worker): ``chiron.metrics.trial_metrics`` with the head as payload and segment 1 as
    the heading body, plus the run's raw time series written to ``<raw_dir>/<raw_name>.csv.gz``."""
    m = cm.trial_metrics(log, outcome, payload="head", heading_body="segment 1",
                         feasibility_every=feasibility_every or None, **kw)
    if raw_dir:
        from vegeta.chiron import export as cxp
        path = cxp.write_raw_csv(log, Path(raw_dir) / f"{raw_name}.csv.gz", every=int(raw_every), q0=raw_q0)
        m["raw_csv"] = str(path)
    return m


def configs_table() -> list[dict]:
    """Every treatment's configuration (per-axis k, c, q₀, limits, roll axis), the controllers, the physics settings
    and the failure rules — for configs.csv."""
    import myropod_robot as mr
    import myropod_controller as mc
    rows = []
    for tr in TREATMENTS:
        robot = mr.cleopatra(tr)
        conn = getattr(robot, "connection", None)
        row = {"kind": "treatment", "name": tr}
        if conn is not None and hasattr(mr, "connection_params"):
            row.update({f"body.{k}": v for k, v in mr.connection_params(conn).items()})
        row["total_mass_kg"] = robot.total_mass()
        rows.append(row)
    for name, fn in (("fixed", mc.fixed), ("adaptive", mc.adaptive)):
        ctrl = fn(0.2)
        row = {"kind": "controller", "name": name}
        row.update({f"ctrl.{k}": v for k, v in vars(ctrl.params).items() if not k.startswith("_")})
        rows.append(row)
    rows.append({"kind": "physics", "name": "lab", **{f"lab.{k}": v for k, v in LAB.items()},
                 "integrator": "implicitfast", "cone": "pyramidal", "contact_solref": "MuJoCo default (0.02, 1)",
                 "foot_mu": 0.8, "shell_mu": 0.5})
    rules = mr.failure_rules(0.2)
    rows.append({"kind": "failure_rules", "name": "protocol §5",
                 **{f"rules.{k}": v for k, v in vars(rules).items() if not k.startswith("_")}})
    return rows


def run_experiment(experiment: str, out_dir, *, scale: str = "full", processes: int = 4, raw_every: int = 2,
                   write_raw: bool = True, feasibility_every: int = 10, include_steps: bool = False,
                   progress: bool = True) -> pd.DataFrame:
    """Run one experiment (cached under ``out_dir/cache``) and write ``out_dir/<experiment>_runs.csv`` (one row per
    run: configuration, seeds, terrain parameters, outcome, every metric), ``out_dir/configs.csv`` and, for every
    run, ``out_dir/raw/<experiment>/<run>.csv.gz``. Returns the rows."""
    out = Path(out_dir)
    raw = out / "raw" / experiment if write_raw else None
    if raw is not None:
        raw.mkdir(parents=True, exist_ok=True)
    tl = trials(experiment, scale=scale, raw_dir=str(raw) if raw else None, raw_every=raw_every,
                feasibility_every=feasibility_every, include_steps=include_steps)
    df = cx.run_trials(tl, processes=processes, cache_dir=out / "cache", metrics_fn=METRICS, progress=progress)
    df.insert(0, "experiment", experiment)
    from vegeta.chiron import export as cxp
    cxp.write_runs_csv(df, out / f"{experiment}_runs.csv")
    cxp.write_configs_csv(configs_table(), out / "configs.csv")
    return df


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """Outcome counts per cell (failures shown, never dropped): runs, successes, each failure reason, errors."""
    d = df.copy()
    keys = [c for c in ("experiment", "info.terrain", "info.level", "v_target", "info.impulse_Ns", "info.k_yaw",
                        "info.c_all", "info.roll_axis", "info.controller_name", "info.treatment") if c in d.columns]
    d["reason"] = d["reason"].fillna(d.get("status", "error"))
    t = d.groupby(keys, dropna=False)["reason"].value_counts().unstack(fill_value=0)
    t.insert(0, "runs", t.sum(axis=1))
    return t.reset_index()
