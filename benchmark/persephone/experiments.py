"""benchmark/persephone: trial lists of Persephone's body-joint benchmark — spring-only vs spring–damper body joints
(Amendment D's treatments, identical joints) × baseline vs load-feedback control, on the hearth before the flue.

NOT YET VALIDATED: written at the close of the build without running a single trial. Run
``python3 benchmark/persephone/full_benchmark.py --scale smoke`` first; if Persephone does not walk at the smoke
scale, the gains, armature or gait parameters in persephone_robot.py / ``controller()`` below need tuning before
the full run means anything.

Scope: ground scenarios only — flat hearth, soot-rough hearth and small longitudinal bumps. The flue itself (a
150 mm pipe with a 90° knee and a vertical climb, notebook 17) needs static pipe geometry in Chiron and a bracing
controller; neither exists yet, so no pipe scenario is listed (stated in the report's limitations).

Speeds: Persephone's 'standard 60 g worm' servos turn at 30 rpm (3.1 rad/s); the hip-yaw return of a 40 mm stride
must fit a quarter of the gait period, which limits the gait to about 0.03–0.04 m/s. Speeds swept: 0.01, 0.02,
0.03 m/s; course 0.5 m (flat for the first 0.15 m).

=================  ======================================================================================
``hearth``         flat, rough (RMS 2, 4, 8 mm, correlation 17.5 mm) and longitudinal bumps (5, 10 mm high, 70 mm
                   apart) at 0.02 m/s; treatment × controller; 30 paired seeds
``speeds``         flat and rough RMS 4 mm × v ∈ {0.01, 0.02, 0.03} m/s; treatment × controller; 10 seeds
``onset``          flat, v ∈ {0.01, 0.02, 0.03}, body-yaw stiffness k ∈ {2, 8, 16} N·m/rad; treatment × fixed; 10 seeds
=================  ======================================================================================
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DESIGNS = HERE.parents[1] / "notebooks" / "designs"
for _p in (str(DESIGNS), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from vegeta.chiron import experiments as cx  # noqa: E402
from vegeta.chiron import metrics as cm  # noqa: E402

import persephone_robot as pr  # noqa: E402

ROBOT = f"{HERE / 'persephone_robot.py'}:persephone"
CONTROLLER = f"{HERE / 'experiments.py'}:controller"
METRICS = f"{HERE / 'experiments.py'}:study_metrics"
VERSION = "persephone-D-1"
TREATMENTS = ("spring", "spring_damper")
CONTROLLERS = {"fixed": 0.0, "adaptive": 0.6}
L_LEG = 0.078                         # m, femur + tibia
P_SEG = 0.070                         # m, segment pitch
START = 0.15                          # m, flat start
COURSE = 0.5                          # m
EXPERIMENTS = ("hearth", "speeds", "onset")
SCALES = {
    "full": dict(seeds_main=30, seeds_sweep=10, rough=(0.002, 0.004, 0.008), bumps=(0.005, 0.010),
                 speeds=(0.01, 0.02, 0.03), onset_k=(2.0, 8.0, 16.0)),
    "pilot": dict(seeds_main=3, seeds_sweep=2, rough=(0.004,), bumps=(0.010,), speeds=(0.01, 0.03),
                  onset_k=(2.0, 16.0)),
    "smoke": dict(seeds_main=1, seeds_sweep=1, rough=(0.004,), bumps=(), speeds=(0.02,), onset_k=(8.0,)),
}


def controller(v_target: float = 0.02, sigma: float = 0.0, robot=None, **kw):
    """Cleopatra's controller with Persephone's gait: stride 40 mm, swing 15 mm, the standing pad position
    (51.8 mm down, 48.2 mm out); ``sigma`` 0 = baseline fixed phase, 0.6 = load feedback (Amendment A/E)."""
    import myropod_controller as mc
    p = pr.PARAMS
    params = mc.ControllerParams(v_target=float(v_target), stride=0.04, swing_height=0.015, depth=p.foot_drop,
                                 foot_out=p.foot_out, sigma=float(sigma), **kw)
    return mc.CleopatraController(params, p, name="fixed" if sigma == 0 else "adaptive")


def terrain_spec(kind: str, level: float) -> tuple[dict, dict]:
    rec = dict(terrain=kind, level=float(level), height_m=0.0, h_over_L=0.0, spacing_m=0.0, rms_m=0.0,
               correlation_m=0.0)
    if kind == "flat":
        return {"kind": "flat", "level": 0.0}, rec
    if kind == "rough":
        rec.update(rms_m=float(level), h_over_L=float(level) / L_LEG, correlation_m=P_SEG / 4)
        return {"kind": "rough", "rms": float(level), "correlation_length": P_SEG / 4, "start": START,
                "level": float(level)}, rec
    if kind == "long_bumps":
        rec.update(height_m=float(level), h_over_L=float(level) / L_LEG, spacing_m=P_SEG)
        return {"kind": "long_bumps", "height": float(level), "spacing": P_SEG, "width": P_SEG / 2, "start": START,
                "jitter": P_SEG / 4, "level": float(level)}, rec
    raise ValueError(kind)


def make_trial(experiment, treatment, ctrl, terrain, level, seed, *, v_target=0.02, robot_kwargs=None, scale="full",
               raw_dir=None, raw_every=2, feasibility_every=0, extra_info=None) -> cx.Trial:
    tspec, rec = terrain_spec(terrain, level)
    rk = {"body_connection": treatment, **(robot_kwargs or {})}
    name = (f"{experiment}|{treatment}|{ctrl}|{terrain}:{level:g}|v{v_target:g}|"
            + "|".join(f"{k}={rk[k]}" for k in sorted(rk) if k != "body_connection") + f"|s{seed}")
    return cx.Trial(robot=ROBOT, controller=CONTROLLER, terrain=tspec, seed=int(seed), robot_kwargs=rk,
                    controller_kwargs={"sigma": CONTROLLERS[ctrl]}, v_target=float(v_target), course_m=COURSE,
                    lab_kwargs=dict(pr.LAB),
                    info={"experiment": experiment, "scale": scale, "treatment": treatment, "controller_name": ctrl,
                          **rec, **(extra_info or {})},
                    metrics_kwargs={"raw_dir": raw_dir, "raw_name": name.replace("|", "__").replace(":", "-")
                                    .replace("=", "-"), "raw_every": int(raw_every),
                                    "feasibility_every": int(feasibility_every)},
                    version=VERSION, name=name)


def trials(experiment: str, *, scale: str = "full", raw_dir=None, raw_every: int = 2, feasibility_every: int = 0):
    s = SCALES[scale]
    kw = dict(scale=scale, raw_dir=raw_dir, raw_every=raw_every, feasibility_every=feasibility_every)
    out = []

    def cells(seeds, terrains, speeds=(0.02,), controllers=tuple(CONTROLLERS), **mk):
        for seed in cx.paired_seeds(seeds):
            for terrain, level in terrains:
                for v in speeds:
                    for ctrl in controllers:
                        for tr in TREATMENTS:
                            out.append(make_trial(experiment, tr, ctrl, terrain, level, seed, v_target=v, **mk, **kw))

    if experiment == "hearth":
        cells(s["seeds_main"], [("flat", 0.0)] + [("rough", r) for r in s["rough"]]
              + [("long_bumps", b) for b in s["bumps"]])
    elif experiment == "speeds":
        cells(s["seeds_sweep"], [("flat", 0.0), ("rough", 0.004)], speeds=s["speeds"])
    elif experiment == "onset":
        for k in s["onset_k"]:
            cells(s["seeds_sweep"], [("flat", 0.0)], speeds=s["speeds"], controllers=("fixed",),
                  robot_kwargs={"body_k_yaw": float(k)}, extra_info={"k_yaw": float(k)})
    else:
        raise ValueError(f"unknown experiment {experiment!r}; known: {EXPERIMENTS}")
    return out


def study_metrics(log, outcome, *, raw_dir=None, raw_name=None, raw_every=2, feasibility_every=0, **kw) -> dict:
    """``chiron.metrics.trial_metrics`` (head as payload, segment 1 as heading) + the raw CSV of the run. The contact
    LP is off by default here (48 feet make it slow); pass --feasibility-every to switch it on."""
    m = cm.trial_metrics(log, outcome, payload="head", heading_body="segment 1",
                         feasibility_every=feasibility_every or None, **kw)
    if raw_dir:
        from vegeta.chiron import export as cxp
        m["raw_csv"] = str(cxp.write_raw_csv(log, Path(raw_dir) / f"{raw_name}.csv.gz", every=int(raw_every), q0=0.0))
    return m


def configs_table() -> list[dict]:
    rows = []
    for tr in TREATMENTS:
        r = pr.persephone(tr)
        rows.append({"kind": "treatment", "name": tr, "total_mass_kg": r.total_mass(), "notes": r.notes})
    rows.append({"kind": "physics", "name": "lab", **{f"lab.{k}": v for k, v in pr.LAB.items()},
                 "kp": pr.KP, "kd": pr.KD, "armature": pr.ARMATURE})
    for name, sigma in CONTROLLERS.items():
        rows.append({"kind": "controller", "name": name, "sigma": sigma, "stride": 0.04, "swing_height": 0.015})
    return rows


def run_experiment(experiment, out_dir, *, scale="full", processes=4, raw_every=2, write_raw=True,
                   feasibility_every=0, progress=True):
    out = Path(out_dir)
    raw = out / "raw" / experiment if write_raw else None
    if raw is not None:
        raw.mkdir(parents=True, exist_ok=True)
    tl = trials(experiment, scale=scale, raw_dir=str(raw) if raw else None, raw_every=raw_every,
                feasibility_every=feasibility_every)
    df = cx.run_trials(tl, processes=processes, cache_dir=out / "cache", metrics_fn=METRICS, progress=progress)
    df.insert(0, "experiment", experiment)
    from vegeta.chiron import export as cxp
    cxp.write_runs_csv(df, out / f"{experiment}_runs.csv")
    cxp.write_configs_csv(configs_table(), out / "configs.csv")
    return df
