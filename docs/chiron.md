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
(Cleopatra), `myropod_controller.py`, `robot_dog_robot.py`, `robot_dog_controller.py`, and for a wheeled
machine `onager_robot.py` / `onager_controller.py` / `onager_scenario.py` (the Onager Sentinel: wheels are
unlimited hinges driven by a `Servo` with `kp = 0`, i.e. a velocity loop on the hub motor's torque–speed line,
with rolling resistance as a hinge `frictionloss`; the wheels are the robot's feet, so the foot metrics apply).

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
- **Servos** run at every physics step. An unclipped joint's −kd·q̇ is an implicit actuator bias (so
  `implicitfast` integrates the damping implicitly); a joint whose torque is clipped gets the clipped torque with
  the implicit derivative of the servo law it is actually on (the torque–speed line's slope, or none beyond the
  no-load speed) — before this fix a saturated joint was under-accelerated by I/(I + h·kd). The applied torque
  always equals the clipped servo law exactly.
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

## Four-quadrant drives (`Servo.four_quadrant`)

By default a servo's torque follows the DC-motor line in all four quadrants: zero at and beyond the no-load speed,
even when the joint is being backdriven. `Servo(..., four_quadrant=True)` gives a braking drive (torque opposing
the motion) its full stall torque at any speed — a regenerating motor, a braked or self-locking screw. The Onager
Atlas needs it: without it, a mast jolted past its screw's no-load speed loses all torque and falls. The hot loop
treats a four-quadrant joint clipped while braking as a constant torque (integrated explicitly). Off by default:
the Cleopatra and Persephone studies keep their law.

## Scenery: props, welds, hooks (`Prop`, `Weld`, `ChironLab.add_hook`)

Things the robot works on that are not the robot — a wire across the road, a log to lift, a post:

```python
post = ch.Prop(ch.Link("post", pos=(8.0, -1.3, 0.0), geoms=[ch.Geom("post_g", "box", (0.04, 0.04, 0.6), pos=(0, 0, 0.6))]))
wire = ch.Prop(ch.Link("wire_a", pos=(8.0, -1.3, 1.15), joints=[ch.Joint("wire_a_hinge", axis=(1, 0, 0))],
                       geoms=[ch.Geom("wire_a_g", "capsule", (0.0016,), fromto=(0, 0, 0, 0, 1.0, 0), mass=0.06)]),
               solref=(-2e7, -2e3))                         # direct stiffness: a thin wire the jaws squeeze
log = ch.Prop(ch.Link("log", pos=(16, 0, 0.075), geoms=[ch.Geom("log_g", "cylinder", (0.075, 0.4), mass=6.4)]), free=True)
lab = ch.ChironLab(robot, terrain, props=[post, wire, log], welds=[ch.Weld("wire_joint", "wire_a", "wire_b")])

def cutter(lab):                                         # scene logic: the wire parts when squeezed hard enough
    f, fn = lab.contact_force(["jaw_upper", "jaw_lower"], "wire_a_g")
    if fn > 7500 and lab.weld_active("wire_joint"):
        lab.set_weld("wire_joint", False); lab.log_event("wire", f"cut at {fn:.0f} N")
lab.add_hook(cutter)
```

* A `Prop` is a Link tree placed in the world: `free=True` gives its root a free joint (a loose object), otherwise
  it is fixed and its own passive Joints still move it. Its geoms collide with the robot, the terrain and the other
  props (the robot's own geoms still only touch the terrain and the props); `priority` 2 lets the prop's
  `solref`/`solimp`/friction govern its contacts with the robot. Prop joints are not robot joints (`lab.joint_names`,
  `total_mass` and the COM stay the robot's).
* A `Weld` holds two bodies (or a body and the world) in their reference relative pose; `lab.set_weld(name, active)`
  switches it at run time, `reset` restores it.
* `lab.add_hook(fn)`: `fn(lab)` every control step of walking time after the controller; `fn.reset(lab)` at reset.
  `lab.log_event(source, detail)` → `log["events"]`. `lab.contact_force(a, b)` → (force vector on `b` from `a`,
  summed normal force) from the last forward pass.
* `lab.body_force(body, force, torque=None)`: a force [N] at a body's COM (world) kept until the next call for that
  body — a scene hook's aerodynamic drag on a prop (the Sweeper's suction); `None` removes it; `reset` clears all.
* Logged: `props` (prop link names), `prop_pos`, `prop_quat` (T,P,·), `events`, `welds` (state at the end).
  Rendering (`viz.frames`) frames the robot and hides scenery farther than 25 robot sizes from it (litter a scene
  hook parks away once collected would otherwise stretch the camera's clipping range).
  Observation: `obs.prop_pos`, `obs.prop_quat`. Rendering draws the props with the robot.

## Water (`density`, `viscosity`, `Geom.fluidshape`)

`ChironLab(..., density=1000.0, viscosity=1.0e-3, wind=(0, 0.5, 0))` (or the same `SimOptions` fields; `wind` is the
medium's velocity — a current) turns on MuJoCo's fluid forces:
quadratic drag and viscous resistance on every body from its inertia box, or — on geoms with
`fluidshape="ellipsoid"` — from the geom's own ellipsoid, with added mass and lift (`fluidcoef` overrides MuJoCo's five
coefficients). **MuJoCo applies no buoyancy**: its fluid forces vanish at rest (checked: `qfrc_fluid` is zero for a
body at rest at density 1000). A scene supplies it, e.g. a hook that pushes each body up with ρ g V at its centre of
buoyancy through `lab.body_force` (the Sikarian Lobster's `lobster_scenario.Water`: CAD volumes per body, and the
tail thruster's force along its shroud axis). Weight-normalised metrics (`metrics.py`) still divide by the robot's
dry weight. Defaults are 0 (vacuum): existing models are unchanged.

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

## Cleopatra body joints: spring-only and spring-damper
The stability study's treatments ([docs/myropod_stability.md](myropod_stability.md) §12, Amendment D; Amendment
C's `rigid` / `flexible` / `flexible+yaw` are legacy). Both have the **identical** joints between segments 1–2 and
2–3 (the head is welded to segment 1): passive hinges `body <i>-<i+1> yaw|pitch|roll` (tags `body_yaw|pitch|roll`)
chained yaw → pitch → roll at the pin mid-gap; pitch and yaw always, roll per `body_roll_axis`; same k, q0, stops;
armature 0, friction loss 0, no actuator, no translation, no prescribed bending. q = 0 is the straight chain; yaw +
swings the rear segment's tail right, pitch + lifts it, roll is about +x. They differ only in c:

| treatment | joint torque [N·m] | damping |
|---|---|---|
| `spring` | `τ = −k (q − q0)` | c = 0 on every axis (any `body_c_*` given is ignored; `robot.notes` lists them) |
| `spring_damper` (default) | `τ = −k (q − q0) − c q̇` | c per axis |

| parameter (Python keyword = Myropod design parameter = CLI `-p`) | unit | default |
|---|---|---|
| `body_connection` | — | `spring_damper` (`spring`; legacy names need `legacy=True`) |
| `body_roll_axis` | bool | true (without it the roll values are unused and `body_q0_roll` must be 0) |
| `body_k_pitch`, `body_k_yaw`, `body_k_roll` | N·m/rad | 8 |
| `body_c_pitch`, `body_c_yaw`, `body_c_roll` | N·m·s/rad | 0.2 (spring_damper only) |
| `body_q0_pitch`, `body_q0_yaw`, `body_q0_roll` | rad | 0 (the spring's rest angle and the start pose) |
| `body_limit_pitch`, `body_limit_yaw`, `body_limit_roll` | rad | 0.785 (45°), 0.785 (45°), 0.349 (20°): stops at ±value |

```python
import myropod_robot as mr, myropod_controller as mc      # notebooks/designs on sys.path
robot = mr.cleopatra("spring")                            # or "spring_damper"; keywords as in the table
lab = mr.cleopatra_lab("spring_damper", terrain, robot_kw={"body_k_yaw": 4.0})   # LAB_OPTIONS: 0.25 ms step
ep = lab.run(mc.adaptive(0.2), rules=mr.failure_rules(0.2), seed=3, info={"treatment": "spring_damper"})
import myropod                                            # the Dedalus design (imports CadQuery)
p = myropod.Myropod().resolve(**mr.CLEO_MM, body_connection="spring")   # its parameters, Cleopatra's geometry
robot = mr.robot_from_design(p)                           # geometry [mm] + body joints → the same robot
```
```bash
dedalus params notebooks/designs/myropod.py:Myropod                           # lists the body_* parameters
chiron run notebooks/designs/myropod_robot.py:cleopatra -p body_connection=spring -p body_k_pitch=8 \
    -p body_roll_axis=false --controller notebooks/designs/myropod_controller.py:fixed --terrain flat \
    --course 1.5 --v-target 0.2 --timestep 0.00025 --seed 0 --out runs/s0
```
`-p` values are Python literals: `8`, `8.0` and `"8"` build the same joint; bools accept true/false, 1/0,
yes/no, on/off; `-p body_connection=rigid` exits 2 (legacy). In trials: `ex.Trial(robot=
"designs/myropod_robot.py:cleopatra", robot_kwargs={"body_connection": "spring", ...}, lab_kwargs=mr.LAB_OPTIONS)`.
The robot carries `treatment` (`spring`, `spring_damper`, `legacy:<name>`), `body_params` (the values as built:
`spring` reports c = 0) and `connection`; the log's `robot` is `cleopatra <treatment>`. The CAD ignores the body
parameters (its chain is drawn straight unless `bend_*_deg` bend it; q0 is not drawn). The controllers take the
leg lengths from the robot they are given.

**Checks** (`notebooks/designs/tests/test_body_joint_physics.py`; numbers at the lab's 0.25 ms step, printed with
`pytest -s`). *Model equivalence*: all 592 `MjModel` attributes compared — the two treatments differ only in
`dof_damping` on the body DOFs and in the model's own name. *c = 0*: `spring_damper` with every `body_c_* = 0`
reproduces `spring` bitwise (rough ground, both controllers; only the log's `robot` label differs). The other
checks run on a rig where only one hinge is free (the robot welded at its standing pose, gravity off; inertia
about the hinge 0.012–0.15 kg·m²). *Restoring torque*: a 0.8 N·m torque on the child segment deflects every axis
of both joints by τ/k (k = 6, 8, 10 N·m/rad; q0 = 0.05 rad) within 2e-7 relative, and `qfrc_spring` =
−k (q − q0), `qfrc_damper` = −c q̇ to 1e-12. *Damping*: in a free decay from q0 + 0.1 rad the energy lost equals
∫ c q̇² dt within 1.3e-5 relative; the logarithmic decrement (δ = 0.50–2.39 per period, ζ = 0.08–0.36) gives c
within 0.25 % (0.998–1.000 c).

**Where energy goes besides c** (so `spring` is not a lossless robot):
- *Contacts*: MuJoCo soft contacts with the default `solref` (0.02 s time constant, damping ratio 1 — critically
  damped) and `solimp` (0.9, 0.95, 0.001, 0.5, 2): every touchdown and belly contact loses energy.
- *Friction*: pads μ = 0.8, shells and head μ = 0.5, pyramidal cones, `condim` 3 (sliding only); a slipping
  contact dissipates μ N |v_slip|.
- *Hard stops*: the body hinges' ±45° / ±20° and the leg ranges are soft limit constraints with the same default
  solref/solimp; hitting one dissipates.
- *Leg servos*: 36 PD loops with kd = 0.8 N·m·s/rad, dissipating whenever a joint moves inside its unclipped
  regime, and the torque–speed line `|τ| ≤ τ_stall (1 − |ω|/ω₀)` (6 N·m, 5.76 rad/s), along which a braking joint
  absorbs energy. Their velocity derivative is integrated implicitly (see ChironLab, *Servo integration*). Where
  −kd stayed implicit for a clipped joint (the build before Amendment E §13.2), a saturated joint was
  under-accelerated by I/(I + h·kd), I its effective inertia at the standing pose (1/(M⁻¹)ᵢᵢ): knee 9.2e-5 kg·m²
  → 0.32 at 0.25 ms and 0.10 at 1 ms; hip yaw 3.2–3.9e-4 → 0.62–0.66 and 0.29–0.33; hip pitch 3.7–4.0e-4 →
  0.65–0.67 and 0.31–0.34.
- *Integrator* (`implicitfast`): velocity terms (c, servo kd) are implicit, springs semi-implicit (symplectic
  Euler). Measured on the rig with c = 0: no systematic loss — energy drift per cycle between −5e-9 and +1.5e-6 of
  E (all six hinges, 20 cycles, 0.25 ms), with a within-cycle fluctuation of 0.16–0.71 % of E; on the fastest hinge
  (body 2-3 roll, ω = 28.5 rad/s) the drift per cycle is 9e-8, 1.2e-6, 2.8e-6, 3.2e-6 and 2.8e-5 at 0.125, 0.25,
  0.5, 1 and 2 ms. With c > 0 the implicit damping shifts the identified c by −0.10 %, −0.20 %, −0.40 %, −0.79 % and
  −1.6 % at those steps, while the energy balance ΔE = ∫ c q̇² dt holds within 0.002 % (0.06 % at 2 ms).
- None from the joints themselves: armature 0 and friction loss 0 everywhere; no fluid drag (density and
  viscosity 0 unless a model sets them, see "Water"). Bearing friction, backlash and gear losses of a real body joint are not modelled.

## Assumptions (stated with every result)
- MuJoCo soft contacts (default solref/solimp unless given) and pyramidal friction cones; contact stiffness and
  damping dissipate energy at every touchdown. Robot geoms touch only the terrain unless `self_collision=True`.
- A servo is a PD loop clipped to the stall line; no current limit, thermal model, backlash or compliance
  beyond `armature`. The electrical cost of transport is an estimate from stall data.
- Height-field resolution (5 mm) and the time step are numerical choices: check convergence for your robot
  (Cleopatra needs 0.25 ms; see `designs/myropod_robot.py`).

Benchmarks: `benchmark/cleopatra/full_benchmark.py` and `benchmark/persephone/full_benchmark.py` (the study of docs/myropod_stability.md; see benchmark/README.md); `./user_tests.sh` runs the checks and the smoke benchmarks.
