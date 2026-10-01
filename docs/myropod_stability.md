# Myropod stability: protocol and metrics (pre-registration)

This document is written **before** any stability trial is run. It fixes the robot model, the treatments,
the controller, the terrains, the failure rules and every metric, so that results cannot redefine them.
It answers an engineer's review of notebook 18: prescribed kinematics (inverse kinematics on a planned
foot path) cannot demonstrate stability; that needs a dynamic simulation with gravity, inertia, friction,
contacts and actuator limits, and a controlled, paired comparison of flexible and locked bodies.

Code: `notebooks/designs/myropod_sim.py` (MuJoCo model, controller, trial runner),
`notebooks/designs/locomotion_metrics.py` (metrics and statistics), `notebooks/designs/stability_experiments.py`
(sweeps), notebook `20_cleopatra_stability.ipynb` (the experiment and its plots). Units are SI throughout
(m, kg, s, N, N·m); the design files and `gait.py` work in mm and are converted at the boundary.

## 1. Robot: Cleopatra, as in notebook 18

Geometry (`designs/myropod.py` with notebook 18's `CLEO` parameters): 3 segments 150 × 110 × 70 mm, joint gap 20 mm
(pitch 0.170 m), head 130 mm long, femur 80 mm, tibia 100 mm, leg Ø16 mm, foot pad Ø24 mm, hip pairs at ±37.5 mm
from the segment centre, hips at y = ±61 mm (seg_width/2 + 6), standing hip angle 40°, knee 85° (tibia below
horizontal): foot 70.0 mm outward and 151.0 mm below the hip.

Mass budget (notebook 18, 6.17 kg total), placed so the inertia is realistic:

| body | contents | mass |
|---|---|---|
| segment shell (each of 3) | printed shell, hatch, rib (CAD volume × 1.1 g/cm³) + 8 hip servos (hip yaw + hip pitch, 70 g each, at the hips) + 2 body-joint servos (120 g) + PCB 60 g + wiring 50 g | 259 + 560 + 240 + 60 + 50 = 1169 g |
| leg (each of 12) | femur and tibia (CAD leg volume split by length) + knee servo 70 g at the knee + 10 g pad | ≈ 49 + 70 (+pad inside the 49) |
| head | shell (CAD) + cameras, LWIR, microphones, radio 450 g | ≈ 636 g |
| segment 2 | + battery 444 g | |
| segment 1 | + compute 150 g | |

The head is welded to segment 1 in every configuration (the camera mount is not part of the treatment).
Leg joints per leg: hip yaw (axis: segment z), hip pitch (axis: horizontal, perpendicular to the leg plane;
positive = femur below horizontal), knee (parallel to hip pitch). Joint ranges: yaw ±60°, hip pitch −40°…+90°,
knee (relative) 0°…160°.

Actuators (`designs/actuators.py`): every leg joint is a `smart servo 6 Nm` — stall 6.0 N·m, rated (thermal)
2.0 N·m, stall current 3.0 A at 12 V, no-load 55 rpm (5.76 rad/s). A servo is modelled as a PD position loop
whose output torque is clipped to the DC-motor torque–speed line `|τ| ≤ τ_stall (1 − |ω|/ω_0)` (and to 0 beyond
ω_0); gains kp = 40 N·m/rad, kd = 0.8 N·m·s/rad (the same in every configuration and trial).
Contacts: foot pads are spheres, friction μ = 0.8 (rubber on rock); segment shells and the head can touch the
ground (belly contact), μ = 0.5. Time step 1 ms; control at 1 kHz; log at 100 Hz.

## 2. Treatments (the only difference between configurations)

| name | body joints between segments 1–2 and 2–3 | physical definition |
|---|---|---|
| `locked` | none: the segments are welded | rigid body; identical bodies and masses |
| `flexible` | passive pitch + yaw hinges at the joint pin | torsional spring k = 8 N·m/rad, damper c = 0.2 N·m·s/rad, limits ±45° |
| `flexible+roll` | passive pitch + yaw + roll | as `flexible`, plus roll k = 8 N·m/rad, c = 0.2, limit ±20° |

All three use the same robot, masses, legs, friction, actuator limits, controller, gains and initial conditions.
The passive joints have no actuator: their spring and damper do no actuator work. (An actively positioned hinge is
a fourth, different treatment; it is out of scope for this round.)

## 3. Controller (identical in every configuration; terrain-blind)

Each segment walks a lateral-sequence wave on its own four legs (rear-left, front-left, rear-right, front-right at
phases 0, 0.25, 0.5, 0.75; duty 0.75), the segments shifted by a twelfth of a stride from the tail to the head
(a metachronal wave), as in notebook 18. Foot targets are planned in each segment's own frame: stance — the foot
moves straight back at the target speed at the nominal depth below the hip (151 mm) and 70 mm outward; swing — it
returns forward on a cosine profile lifted by 50 mm. Stride 0.10 m (step frequency = v / stride). `gait.ik_myropod`
turns a foot target into hip yaw, hip pitch and knee; the servo loop tracks it. A heading correction makes the
left and right strides differ by `k_heading × yaw error of segment 1` (k_heading = 0.5 m/rad, clipped to ±30 %
of the stride). No body levelling, no terrain sensing, no reflexes: the comparison isolates the body. The
controller settings above are held fixed for the main comparison; a separate, equal-effort tuning study (§7)
gives each configuration the same small budget to find its best settings.

## 4. Terrains (paired: both configurations see the identical surface)

Heights are normalised by leg length L = femur + tibia = 0.18 m; spacings by the segment pitch P = 0.170 m. Every
course is 1.5 m long, starts flat for 0.3 m, and is a MuJoCo height field (5 mm cells). A trial's seed sets the
random parts (roughness field, obstacle position jitter of ±P/4) and the initial gait phase offset (uniform in one
stride); configurations in a pair share the seed.

| terrain | definition | difficulty parameter |
|---|---|---|
| `flat` | z = 0 | — |
| `long_bumps` | half-sine ridges across the path, spacing P, width P/2 | height h/L |
| `alt_bumps` | half-sine bumps under the left feet, then the right feet, alternating every P | height h/L |
| `cross_slope` | the plane tilted about the route axis | angle (deg) |
| `steps` | alternating up and down steps every 2P | height h/L |
| `rough` | isotropic random field, Gaussian spectrum, correlation length 0.25 P | RMS height h/L |

Levels: h/L ∈ {0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30}; cross-slope ∈ {0°, 5°, 10°, 15°, 20°, 25°}.
Target speeds: {0.1, 0.2, 0.3} m/s.

## 5. Trial and failure rules (defined before any run)

A trial starts with the robot standing on its feet at its nominal height, settles for 0.5 s, then walks the course.
Exactly one outcome is recorded:

- **success** — the whole-robot centre of mass (COM) passes x = course length;
- **fall** — any segment tilts more than 60° from vertical, or the COM is lower than 40 % of the nominal hip
  height above the ground beneath it for 0.5 s;
- **stall** — after the first 2 s, the COM advances less than 10 % of `v_target × 3 s` over any 3 s window;
- **off_course** — the COM leaves |y| ≤ 0.5 m;
- **timeout** — none of the above before `2 × course / v_target + 2 s`.

There is no intervention in simulation. The trial, not a frame, is the independent observation.

Disturbance trials: on flat ground at the target speed, after 3.0 s of walking, a sideways (+y) force on
segment 2's centre of mass for 50 ms delivers a known impulse `J = ∫F dt`; impulses
J ∈ {0, 1, 2, 3, 4, 6, 8} N·s (6.17 kg: 1 N·s ≈ 0.16 m/s of lateral velocity).

## 6. Metrics (per trial; every segment reported; the payload body is the head)

| metric | definition |
|---|---|
| traversal success | outcome == success; across trials: rate with a Wilson 95 % interval |
| distance before failure | COM progress along +x at the outcome; completed trials marked separately |
| achieved speed | COM progress / elapsed walking time (to the outcome) |
| recovery (disturbance) | recovered if, within 5 s after the push ends, the 0.5 s moving-average forward speed returns within ±30 % of its pre-push mean **and** every segment's tilt within 10°, both held for 1.0 s, with no failure; recovery time = start of that held interval minus the end of the push |
| body angular motion | per body: RMS roll rate and RMS pitch rate (body-frame ω_x, ω_y); 95th percentile of tilt (angle of the body z axis from vertical), of |roll| and of |pitch|; reported for the payload body and the worst body — never averaged across the body |
| intersegment angles | per body joint and axis: max and 95th-percentile |angle|, fraction of time within 1° of a limit, number of limit hits (entries into that band) |
| belly contacts | fraction of time any shell or the head touches the ground; per body |
| foot slip | for each loaded contact episode (normal force > 2 % of the robot's weight), the tangential path length of the foot centre in the contact plane; reported as slip per metre of progress, 95th-percentile slip per episode and the fraction of episodes slipping more than 5 mm |
| mechanical cost of transport | `E₊ = ∫ Σ_j max(τ_j q̇_j, 0) dt` over every actuated joint (all leg joints and any active body joint); `CoT_mech = E₊ / (m g d)`. Excludes electrical losses and holding power. |
| electrical cost of transport (estimate) | DC-motor model from the stall data: k_t = τ_stall / I_stall, R = V / I_stall, I = |τ| / k_t, `P = max(τ q̇ + I² R, 0)` per actuator; `CoT_el = E_el / (m g d)`. Labelled an estimate. |
| actuator demand | per joint group (hip yaw, hip pitch, knee, active body): peak |τ|, RMS τ, RMS τ / rated torque, fraction of time torque-saturated (|τ| ≥ 98 % of the speed-dependent limit) and speed-saturated (|q̇| ≥ 98 % of ω_0) |
| support margin | `s(t)` = signed distance from the COM's ground projection to the boundary of the convex hull of the loaded contact points' projections (positive inside); fewer than three loaded contacts: minus the distance to their hull (a segment or point). Reported: minimum, 5th percentile, fraction of time outside. A diagnostic on uneven ground, not a fall criterion (Bretl & Lall). |
| contact-force feasibility | second stage: at each logged sample, a linear program asks whether contact forces exist in linearised friction cones (8-facet pyramids, μ of the contact) at the loaded contacts that produce the net wrench the motion needs — quasi-static (`Σf = −m g`, moments about the COM zero) and dynamic (`Σf = m(a_COM − g)`, `Σ (p_i − c) × f_i = dL/dt`), optionally with each leg's joint torques `|J_iᵀ f_i| ≤ τ_stall`. Reported: fraction of samples infeasible. |
| lateral deviation, heading | max |y_COM|; RMS yaw of segment 1 |

## 7. Experiments and statistics

1. **Main comparison** (controller fixed): every terrain × level at v = 0.2 m/s, 30 paired seeds per cell, all
   three configurations. Success rates with Wilson 95 % intervals; paired difference by exact McNemar test on the
   discordant pairs; continuous metrics as paired differences with bootstrap 95 % intervals (10 000 resamples,
   fixed seed). Failures are shown next to every speed and energy result.
2. **Speed × difficulty maps**: `rough` and `alt_bumps`, all levels × all three speeds, 10 paired seeds per cell
   (a pilot density, stated on the plots); one heatmap of success per configuration; achieved speed reported.
3. **Disturbance**: flat, 0.2 m/s, every impulse level, 30 paired seeds; recovery probability vs impulse with
   Wilson intervals and a logistic fit (impulse at 50 % recovery).
4. **Energy**: cost of transport (mechanical and electrical estimate) vs achieved speed, successful and failed
   trials marked differently.
5. **Mechanism**: one representative pair on `alt_bumps`: synchronised traces of every segment's roll and pitch,
   the body-joint angles, contact normal forces, slip and the support margin.
6. **Equal tuning effort**: each configuration gets the same 12 controller settings (a fixed grid over stride
   length, swing height and stance depth) on `rough` at h/L = 0.15, 10 seeds each; its best setting is then run
   on 30 fresh paired seeds and compared.

Any reduction of these counts for runtime is logged where it happens and stated on the plots.

## 8. Precedent

Yasui et al. (2022), *Adaptive centipede walking via synergetic coupling between decentralized control and
flexible body dynamics* — in a 2-D centipede model, body flexibility and adaptive leg control both improved
rough-terrain walking. Bretl & Lall (2008), *Testing static equilibrium for legged robots* — the projected
support polygon is not sufficient on uneven ground; contact-force feasibility is.
