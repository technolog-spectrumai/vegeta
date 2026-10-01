# Chiron — legged-robot dynamics on MuJoCo

Chiron (the centaur who trained the heroes) answers "does it actually walk?" for a legged design whose
geometry, masses and actuators are known. It wraps **MuJoCo**: gravity, inertia, friction cones, soft
contacts with the ground, belly contacts, passive spring–damper joints and servos limited by their motor's
torque–speed line. Its environment, **ChironLab**, is reusable: any robot described with Chiron's building
blocks (or imported from MJCF) walks in it on any Chiron terrain with any controller, under the same logging,
disturbances and failure rules, and every run returns the same **episode log**, which `chiron.metrics` reads.

It is **not** a controller-design tool (no optimisation, no learning, no trajectory planning), not a
kinematic animator (`notebooks/designs/gait.py` does that) and not a structural tool (joint torques and
contact forces go to Talos as load cases). Controllers and robots are yours; Chiron simulates them honestly.
Nothing is guessed: masses, inertias, friction, servo data and gains, terrain sizes and failure thresholds
are explicit inputs with their sources. Units are SI everywhere (m, kg, s, N, N·m, rad; degrees only where a
name says `_deg`). World frame: z up; a course runs along +x from x = 0.

## Building blocks (`robot.py`, MuJoCo-free dataclasses)
- **Servo**(stall_torque, rated_torque, no_load_speed, stall_current, voltage, kp, kd, armature=0, source) — a
  PD position loop whose torque is clipped to `|τ| ≤ τ_stall (1 − |ω|/ω₀)` (0 beyond ω₀, braking included).
  `Servo.from_actuator(entry, kp=, kd=)` reads a `designs/actuators.py` entry; `Servo.from_rpm(...)`.
- **Joint**(name, kind 'hinge'|'slide', axis, pos, range, stiffness, damping, springref, armature, frictionloss,
  tag, servo, leg) — `servo=None` is a passive joint (its spring and damper do no actuator work); `tag`
  classifies it for the metrics (`hip_yaw`, `hip_pitch`, `knee`, `hip_roll`, `body_pitch`, `body_yaw`,
  `body_roll`, ...); `leg` names its foot.
- **Geom**(name, type box|sphere|capsule|cylinder|ellipsoid, size, pos, quat, fromto, mass, friction, role, rgba)
  — `role`: `foot` (a pad named by a FootSpec), `body` (a shell: its ground contact is a *belly contact*),
  `link` (other colliding part) or `visual` (never collides). `mass=None` is massless.
- **PointMass**(name, mass, pos) — servos, batteries, boards placed where they are.
- **Link**(name, pos, quat, geoms, masses, joints, children, log, group) — no joints = welded to the parent;
  the root gets a free joint; `log=True` logs the body (frame x forward, y left, z up).
- **FootSpec**(name, geom, joints, body) and **Robot**(name, root, feet, nominal_qpos, nominal_base_height,
  nominal_hip_height, notes, sources): `total_mass()`, `summary()`, `validate()`, `to_mjcf(terrain)`.
- Third-party models: `Robot.from_mjcf(xml, RobotMeta(feet, logged_bodies, joint_tags, servos, ...))`; its own
  actuators are replaced by Chiron servos.

Robot-specific code lives with the designs, never in the package: `notebooks/designs/myropod_robot.py`
(Cleopatra), `myropod_controller.py`, `robot_dog_robot.py`, `robot_dog_controller.py`.

## Terrains (`terrain.py`)
Analytic, vectorised `height(x, y)` [m], turned into a MuJoCo height field (5 mm cells by default; flat ground
is a plane unless `flat_as_plane=False`). `Flat()`, `LongitudinalBumps(height, spacing, width, start, jitter,
seed)`, `AlternatingBumps(height, spacing, width, track_y, ...)`, `CrossSlope(angle_deg, start, transition)`,
`Steps(height, spacing, start, jitter, seed, mode)`, `Rough(rms, correlation_length, start, seed)` (Gaussian
spectrum, exact RMS), `Custom(fn)`. `terrain.spec()` ↔ `terrain_from_spec(dict)`, so the surface is recorded
with every run; equal specs give identical surfaces (paired trials). Nothing robot-specific: a study
normalises its levels (h/L, spacing/P) itself and passes metres.

## ChironLab (`lab.py`)
```python
from vegeta import chiron as ch

lab = ch.ChironLab(robot, ch.Rough(0.027, 0.0425, start=0.3, seed=3), timestep=0.001, control_dt=0.001, log_dt=0.01)
obs = lab.reset(seed=3)                      # standing, at rest, COM above (0, 0)
obs = lab.step(ch.Command(q_target={...}))   # one control step; obs.foot_force, obs.com, obs.foot_jac, ...
ep = lab.run(controller, rules=ch.FailureRules(course_m=1.5, v_target=0.2), settle=0.5, seed=3)
ep.outcome, ep.log["com"], ep.save("trial.npz"), ep.to_result()
```
- **Controller**: any object with `reset(lab, seed)` and `__call__(obs) -> Command` (or a plain callable);
  optional `settle_command(obs)` and `name`. `Command(q_target, qd_target, tau_ff, leg_phase, leg_stance)` —
  dict joint → value or an array over `lab.actuated_joints`; `leg_phase`/`leg_stance` are only logged.
  `PhaseGenerator(legs, base_phases, duty, frequency_hz, sigma)` gives fixed (σ = 0) or load-adaptive
  Tegotae phases `dφ/dt = ω(φ) − σ N cos φ`.
- **Observation**: time, joint q/q̇/τ, logged bodies' pose, body-frame angular velocity and world velocity,
  COM, COM velocity, angular momentum, per-foot contact force (ON the foot), normal (ground → foot), contact
  point and Jacobian, belly contacts. Kinematic and contact fields are one physics step old (a sensor);
  `observe(sync=True)` makes them simultaneous.
- **Servos** run at every physics step (a MuJoCo `general` actuator carries −kd·q̇ so the default
  `implicitfast` integrator treats it implicitly; the applied torque equals the clipped servo law exactly).
- **Disturbances**: `lab.add_disturbance(Disturbance(body, t_start, duration, impulse=J or force=F,
  direction))` delivers the exact impulse on the body's COM; `lab.apply_impulse(...)` from a controller.
- **FailureRules**(course_m, max_tilt_deg 60, min_height_fraction 0.4, low_height_time 0.5, stall_window 3,
  stall_fraction 0.1, stall_grace 2, lateral_limit 0.5, timeout = 2·course/v + 2 s) — exactly one outcome:
  `success` (COM passes x = course), `fall` (a logged body tilts > 60°, or the COM stays below 40 % of the hip
  height above the ground for 0.5 s), `stall` (< 10 % of v·3 s progress over a 3 s window after 2 s),
  `off_course` (|y| > 0.5 m), `timeout`. Without rules: `completed`; a shorter duration: `stopped`.
- **Episode**: `log`, `outcome` (success, reason, t_end, distance_m, x_end, detail), `meta` (timing,
  options, versions); `save`/`load` (.npz), `to_result()` (kind `chiron.episode`). Runs are deterministic:
  the same trial gives a bitwise-identical log; logging on or off does not change the trajectory.

## The episode log
A dict of numpy arrays, T samples every `log_dt` of walking time (0 at the end of the settle), B logged
bodies, F feet, J joints:

| keys | content |
|---|---|
| `t`, `bodies`, `body_group` | walking time [s]; body names and their groups |
| `body_pos`, `body_quat`, `body_angvel`, `body_linvel` | (T,B,·) COM position, w-x-y-z quaternion, body-frame ω, world COM velocity |
| `com`, `com_vel`, `ang_mom`, `total_mass`, `gravity` | whole robot; angular momentum about the COM |
| `feet`, `foot_body`, `foot_group`, `foot_joints`, `foot_mu` | foot names, the body each leg is mounted on, its joints, friction |
| `foot_pos`, `foot_force`, `foot_normal`, `foot_contact_pos`, `foot_jac` | pad centre; total contact force ON the foot (world, N; 0 off the ground); unit normal ground → foot and contact point (nan off the ground); (T,F,3,n) d(foot)/d(q_leg) |
| `joints`, `joint_kind`, `joint_active`, `joint_leg`, `q_range` | names, tags, actuated?, foot index or −1, limits |
| `q`, `qd`, `tau` | (T,J) angle, speed, applied servo torque (0 if passive) |
| `tau_stall`, `tau_rated`, `qd_noload`, `i_stall`, `voltage` | servo data per joint (nan if passive) |
| `belly_contact`, `terrain_height_under_com` | (T,B) any `body` geom touching; ground height under the COM |
| `leg_phase`, `leg_stance_cmd` | controller's cycle fraction and commanded stance per foot (when reported) |
| `v_target`, `course_m`, `nominal_hip_height`, `disturbances`, `robot`, `treatment`, `controller`, `terrain`, `seed` | the trial |

A log recorded on a real robot in this format works with the same metrics (blocks with missing keys are skipped).

## Metrics (`chiron.metrics`, pure numpy/scipy)
`trial_metrics(log, outcome, payload=None, feasibility_every=10)` returns one flat dict per trial; per-entity
values are keyed `<metric>@<body|joint|foot>`. Definitions are fixed by the pre-registered protocol,
[docs/myropod_stability.md](myropod_stability.md) §6 and §9.2; in short (loaded = normal force > 2 % of the weight):

| metric | definition |
|---|---|
| progress, achieved speed | COM x at the outcome − start; progress / walking time |
| body angular motion | per body RMS roll/pitch rate (body frame), p95 and max tilt, p95 \|roll\|, \|pitch\|; `worst_*` = max over bodies, `payload_*` |
| intersegment angles | per body joint: max and p95 \|q\|, fraction within 1° of a limit, limit hits |
| belly contacts | time fraction any shell touches, per body |
| foot slip | tangential path of the pad centre in the contact plane per loaded episode: per metre, p95, fraction > 5 mm |
| cost of transport | `E₊ = ∫Σ max(τq̇, 0) dt` over actuated joints, `CoT = E₊/(m g d)`; electrical estimate with `I = τ/k_t`, `P = max(τq̇ + I²R, 0)` |
| actuator demand | per joint group: peak and RMS τ, RMS τ/τ_rated, torque- and speed-saturated fractions (≥ 98 % of the line / ω₀) |
| support margin | signed distance of the COM projection to the hull of the loaded contacts: min, p5, fraction outside |
| contact-force feasibility | LP: do forces in 8-facet friction pyramids at the loaded contacts produce the quasi-static / dynamic wrench (optionally within the joint torque limits)? fraction infeasible |
| recovery | after a push: 0.5 s mean speed within ±30 % of the pre-push mean and every tilt < 10°, held 1 s, within 5 s |
| gait | measured duty factor per leg; interlimb touchdown phases (circular mean and SD); phase recovery after a push |
| undulation, heading | body-yaw joint RMS / peak-to-peak / FFT frequency, COM lateral oscillation, yaw about the mean heading; max \|y\|, heading RMS |

## Statistics (`chiron.stats`)
Trials, not frames, are the observations, and comparisons are paired on seeds: `wilson_ci`, `rate_ci`,
`mcnemar_exact`, `paired_difference_ci` (bootstrap, 10 000 resamples, fixed seed), `success_difference_ci`,
`difference_of_differences_ci` (interaction), `logistic_fit` (factors and interactions, ML), `recovery_curve`
(Wilson per impulse, logistic J50), `cell_rates` (heatmaps), `onset_speed`.

## Experiments (`chiron.experiments`)
```python
from vegeta.chiron import experiments as ex

def make(body, level, seed):          # protocol §12.3: body_connection = spring | spring_damper
    return ex.Trial(robot="designs/myropod_robot.py:cleopatra", robot_kwargs={"body_connection": body},
                    controller="designs/myropod_controller.py:fixed", seed=seed, v_target=0.2, course_m=1.5,
                    terrain={"kind": "rough", "rms": level * 0.18, "correlation_length": 0.0425, "start": 0.3,
                             "level": level}, lab_kwargs={"timestep": 0.00025}, info={"payload": "head"})

trials = ex.paired_trials(make, {"body": ["spring", "spring_damper"], "level": [0.15, 0.25]}, ex.paired_seeds(30))
df = ex.run_trials(trials, processes=4, cache_dir="runs/cache")         # one row per trial, metrics included
p = ex.pair_up(df, "body", "spring", "spring_damper", "success")        # aligned pairs for stats.mcnemar_exact
ep = ex.rerun_with_log(trials[0], log_geoms=True)                       # the full episode, for plots and videos
```
- **Trial** is plain, picklable data: factories as `'file.py:function'` / `'module:function'` strings with
  keyword arguments, the terrain spec (its seed defaults to the trial's), seed, `v_target`, `course_m`
  (failure rules; or `duration` without rules), `rules` overrides, `disturbances`, `lab_kwargs`, `info` (into
  the log), `metrics_kwargs`, `version`; `name` and `factors` are labels. A controller factory also receives
  `v_target` and `robot` when it takes them. `trial.run()` gives the Episode.
- **run_trials**(trials, processes=4, cache_dir, metrics_fn='vegeta.chiron.metrics:trial_metrics',
  keep_logs=False): each trial in its own forked process, `processes` at a time (a crash takes down only its
  own trial); metrics computed there; rows cached as `<cache_dir>/rows/<trial_id>.json` as soon as each trial
  ends. `trial_id` hashes everything that can change the result (not the labels); re-runs skip cached trials,
  a failed trial is recorded (`status='error'`) and retried next time. The cache does not see code changes:
  bump `version` or use a new directory. `keep_logs` saves every episode next to the rows.
- **Rows**: `trial_id`, `name`, the factors, the flattened spec (`robot_factory`, `robot_kwargs.*`,
  `terrain.*`, `seed`, ...), `status`, the outcome, timing and every metric.
- Helpers: `paired_seeds(n, stream=...)`, `factor_grid`, `paired_trials`, `pair_up`, `trials_frame` (the plan
  before running), `load_cached`.

## CLI
```bash
chiron info designs/myropod_robot.py:cleopatra -p body_connection=spring      # masses, bodies, joints, servos
chiron run designs/myropod_robot.py:cleopatra -p body_connection=spring_damper -p body_k_pitch=8 \
    --controller designs/myropod_controller.py:adaptive --terrain rough:0.027 -t correlation_length=0.0425 \
    -t start=0.3 --seed 3 --course 1.5 --v-target 0.2 --timestep 0.00025 --payload head --out runs/r3
chiron run ... --push "segment 2:3:3.0:0.05"                                  # BODY:IMPULSE[N s]:T_START:DURATION
chiron metrics runs/r3/episode.npz --json
xvfb-run -a chiron render runs/r3/episode.npz -o walk.mp4                     # record with --log-geoms
```
`--terrain KIND:LEVEL` sets the kind's difficulty parameter in SI (height or RMS in m, slope in degrees); the
other parameters come from `-t name=value` (a missing one is an error, never a default), or
`--terrain-factory file.py:fn` maps KIND:LEVEL with your own normalisation. `run` writes `episode.npz` and
`summary.json` (Result kind `chiron.run`: outcome and metrics; the trial spec in `metadata`). Exit codes: 0 the
run completed (whatever its outcome; `--require-success` makes a non-success 1), 1 the run, its metrics or
its video failed, 2 invalid input.

## Rendering (`chiron.viz`)
MuJoCo's renderer needs EGL/OSMesa; Chiron draws from the log with pyvista instead:
`ChironLab(..., log_geoms=True)` (or `--log-geoms`) records every geom's pose; `viz.frames(ep, camera)`,
`viz.to_video(frames, path)`, `viz.render(ep, path)`. Headless: `xvfb-run -a` with `PYVISTA_OFF_SCREEN=true`.

## Assumptions (stated with every result)
- MuJoCo soft contacts (default solref/solimp unless given) and pyramidal friction cones; contact stiffness and
  damping dissipate energy at every touchdown. Robot geoms touch only the terrain unless `self_collision=True`.
- A servo is a PD loop clipped to the stall line; no current limit, thermal model, backlash or compliance
  beyond `armature`. The electrical cost of transport is an estimate from stall data.
- Height-field resolution (5 mm) and the time step are numerical choices: check convergence for your robot
  (Cleopatra needs 0.25 ms; see `designs/myropod_robot.py`).

Notebooks: `20_chiron_lab` (the tool, on the robot dog and Cleopatra) and `21_cleopatra_stability` (the study).
