# vegeta-bionic — plan (v0)

A new distribution, **`vegeta-bionic`** (`libs/vegeta-bionic`, imported as `vegeta.bionic`), for the shared
primitives of human–robot limbs. It serves two lines that share most of their parts:

1. **teleoperation**: a glove that drives two robot pincers remotely, later the whole arm, and a 3D-printed scaled
   copy of the arm (a "puppet" or master arm) that you move by hand to teach the robot;
2. **assistive exoskeletons** for people with a limb disability, starting with light ones.

The same hand model, sensors, calibration, filters, actuator models and safety limits serve both. A glove that
*resists* your fingers (force feedback) and a glove that *helps* them (grip assist) are the same frame with the
motor's sign reversed.

## Decisions (design questions)

| question | decision |
|---|---|
| glove sensing | an IMU on the back of the hand (wrist orientation) + Hall-effect angle sensors on the finger joints |
| feedback | **force feedback is the goal**, reached iteratively: vibration motors for contact and grip force first, then brakes, then motors on the glove's linkages |
| first target | **simulated pincers in MuJoCo**: Onager Manus's two arms and jaws (`labs/land/designs/onager_manus_robot.py`) |
| module | a new distribution, `vegeta-bionic`, like `vegeta-core` |
| teach model | a **scaled printed copy of the arm** with a magnetic encoder in every joint |
| electronics | **decided later**: the glove is simulated first (`SimGlove`), its protocol defined in software |
| exoskeletons | grip, wrist and elbow/shoulder assistance are all wanted, none built now (TODO below) |
| safety | **medical-device rules from day one** for anything worn and powered (below) |

## What vegeta.bionic holds

Like the engineering tools, it is a plain Python library: dataclasses, explicit values, files out, no hidden
defaults that matter. It depends on `vegeta-cli` (as `vegeta-core` does) but its primitives do not import MuJoCo. The
MuJoCo scenes that use it live in a lab.

| module | what | first version |
|---|---|---|
| `hand` | a kinematic hand: wrist pose, per finger MCP/PIP/DIP flexion (+ thumb CMC, MCP abduction), per-user segment lengths and joint ranges | 15 DOF + wrist; forward kinematics of fingertips |
| `sensors` | sensor models: IMU (gyro bias, drift, noise, magnetic disturbance), Hall angle sensor (gain, offset, nonlinearity, cross-talk), encoder (resolution) | parameters from datasheet classes |
| `glove` | `GloveSpec` (which sensor on which joint), `SimGlove` (hand motion → sensor readings), `GloveState` (estimated hand) | a simulated glove driven by scripted or recorded hand motions |
| `estimate` | calibration (a few poses: flat, fist, pinch), orientation filter (complementary / Madgwick), angle estimation | offline and streaming |
| `retarget` | hand → robot commands: pinch aperture → jaw opening, squeeze → grip force; later the wrist pose → arm IK target; scaling, deadband, a clutch (engage/release), rate limits | the two-pincer mapping |
| `haptics` | feedback: pincer contact force → vibration patterns; later actuator models (brake, motor + cable or linkage) with torque limits and the haptic loop's stability (delay, passivity) | vibration first |
| `link` | the glove ↔ host message schema (versioned; sample, timestamp, sequence), transports: in-process (sim) now, serial/UDP when the electronics are chosen | in-process, with a latency/jitter/loss model |
| `teleop` | the session loop: read the glove, retarget, command the robot, return the feedback; record demonstrations (time series in, robot states out) and replay them | at the robot's control rate |
| `puppet` | the scaled master arm: joints, scale, encoders, the mapping to the robot arm | after the glove |
| `safety` | limits (force, torque, range of motion, speed), watchdog, dead-man/clutch, e-stop; for worn devices the risk file's controls | from the start |
| `exo` | assistive devices (TODO) | not now |

## Phases

| phase | deliverable | checks |
|---|---|---|
| **P0** | this plan | — |
| **P1** | `libs/vegeta-bionic` skeleton; `hand`, `sensors`, `glove.SimGlove`, `estimate`, `link` (in-process) | tests: calibration recovers joint angles within a stated error from noisy sensors; IMU drift bounded by the filter; message round-trip |
| **P2** | `retarget` + `haptics` (vibration) + `teleop`: a scripted hand drives Onager Manus's two pincers in MuJoCo to pick and pass objects; demonstrations recorded and replayed | pick success, aperture error, grip-force tracking, end-to-end latency budget, vibration level against contact force |
| **P3** | the glove's CAD (Dedalus): printed finger and back-of-hand mounts, sensor placement, three hand sizes; the electronics chosen (ESP32-S3 or Teensy class) and `link` over serial/UDP | mounts print (Mellonia), sensor ranges cover the hand's motion |
| **P4** | force feedback in simulation: a brake, then a motor, on the glove's linkages; stability of the haptic loop with delay | passivity margin; force fidelity; torque limits never exceeded |
| **P5** | the puppet arm: a scaled printed copy of the Manus arm (Dedalus) with magnetic encoders; teach by posing, record, replay in MuJoCo | joint tracking error; repeatability |
| **P6** | the whole arm from the glove: the wrist IMU (+ a tracker) → arm IK; the glove keeps the pincers | reach tasks in MuJoCo |

## P1 done (2026-11-03)

`libs/vegeta-bionic`: `hand` (20 joints, per-user sizes, fingertip kinematics, key poses), `sensors` (Hall angle,
IMU, encoder, each unit's errors from a seed), `motions` (calibration routine, grasp cycles, wrist tour), `glove`
(`SimGlove`: 14 sensed joints, the IMU), `estimate` (per-sensor calibration from held poses, the magnetometer's
hard-iron fit, a Mahony filter), `link` (versioned binary message, an in-process transport with latency, jitter, loss
and stale-message dropping), `safety` (`Watchdog`, `Limits`), `RISK.md`. Measured on the simulated glove: sensed joints
under 2° rms after calibration; unsensed DIPs ~3–5° rms (tendon coupling); the pinch aperture ~1.4 mm rms; the wrist
~2.7° mean once the magnetometer is calibrated (8° without: the hard-iron offset, not the filter, was the error).

## TODO: assistive exoskeletons (not now)

Each starts from the glove's primitives and gets its own risk file before any powered prototype is worn.

* **grip assist** — help close and open the fingers (after a stroke, an incomplete spinal injury, a nerve injury):
  the P4 glove frame with the motor helping instead of resisting; intent from residual finger motion or a trigger.
* **wrist and forearm** — support and assist wrist extension (wrist drop, tendon injury): one joint, a forearm cuff.
* **elbow and shoulder support** — assist lifting the arm against gravity: larger forces, a harness or a
  mounted frame, gravity compensation first.

## Safety: medical-device rules from day one

Anything worn and powered is treated as a medical device from the first drawing:

* a **risk management file** in the repo from P1 (ISO 14971 style): hazards, causes, controls, verification, residual
  risk, kept with the code it controls;
* limits enforced **in hardware** as well as in software: torque, force, range of motion and speed per joint, with
  the glove's or exoskeleton's mechanical end stops inside the user's own range;
* an always-available **release** (mechanical, without power) and a dead-man / clutch; a watchdog that removes power
  on lost communication;
* electrical safety and rehabilitation-robot practice as the reference for the hardware phases (IEC 60601-1, IEC
  80601-2-78), and the EU MDR classification settled before any user trial;
* **clinician input** and an ethics review before anyone with a disability wears a powered version.

Teleoperation (P1–P6) also has a risk file: the robot, not the glove, is the hazard there (the pincer's force and
speed limits, the clutch, the watchdog).

## Decisions (round 3: teleoperation)

| question | decision |
|---|---|
| gloves | **two**, one per pincer: the left hand drives the left pincer, the right hand the right (the primitives take N gloves) |
| mapping | **position and rate, switchable**: position (the jaw opening follows the thumb–index pinch, squeezing past contact commands grip force) for normal work; rate (finger flexion sets the closing speed) for fine adjustment; a clutch gesture freezes the pincer to relax or re-centre the hand; a switch gesture changes the mode |
| latency | **measured first**: `link` models latency, jitter and loss; P2 sweeps them and finds where grasping starts to fail, and the budget is set from that |
| demonstrations | **everything**: raw glove samples and the estimated hand pose; the robot's commands and states (joints, jaw opening, grip force); contact forces and object poses; camera video (rendered now, real later), all on one clock |

### The demonstration record

One recording is a folder: `meta.json` (who, which robot and gloves, calibrations, the mapping and its mode
switches, the software revision), a time-series file per stream on one clock (`glove_<side>`, `hand_<side>`,
`robot_command`, `robot_state`, `contacts`, `objects`), and the videos (`camera_<name>.mp4`) with their frame
timestamps. The streams are columnar (Parquet or NPZ) so they load straight into training code. Replay reads the
same folder.

## Open questions (next round)

* the glove's sizes: the hand-size range to cover, left and right
* the switch and clutch gestures: which ones are unambiguous and never happen by accident during grasping
* the latency sweep's tasks: which picks and passes in P2 define "grasping fails"
