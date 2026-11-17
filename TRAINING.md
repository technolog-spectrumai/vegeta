# Training Belisario's locomotion policy (MuJoCo, PyTorch, PPO)

The pipeline lives in `training/`: a Gymnasium-style environment round the gorilla-like frame of
`labs/legged/designs/space_unit_robots.bare` (Chiron's MJCF, its servos as PD torque limits), a PPO implementation
in PyTorch, an eight-stage curriculum, sessions of scenes, and scripts to train, resume, evaluate, run, visualise,
export to ONNX and compare against the programmed gaits. Nothing here has been run on this machine: the code was
written and inspected statically. Expect first-run issues and read the tracebacks; the design is laid out so each
part can be checked on its own (`training/tests`).

## Dependencies and setup (Ubuntu)

Python 3.11+. From the repository root:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e libs/vegeta-cli -e libs/ogma
pip install mujoco numpy pyyaml gymnasium tensorboard
pip install torch --index-url https://download.pytorch.org/whl/cu121    # CUDA 12.1 laptop; or plain "pip install torch" for CPU
pip install onnx onnxruntime                                            # for the export and its check
pip install pygame                                                      # only for the joystick in teleop
```

`cadquery` is not needed for training (the sims' CAD renderer is separate). Check the model builds:

```bash
python -c "from training.belisario_rl.robot_model import RobotModel; m = RobotModel(); print(m.summary()); print(m.missing_requirements())"
```

Run the unit tests (they build one environment, ~10 s):

```bash
pytest training/tests -q
```

## The robot model and what it lacks

`belisario_rl.robot_model.RobotModel` builds the frame through Chiron (`bare(casing)` → `to_mjcf`): 57 actuated hinges
(21 of Talia's limbs, the spine's 4, the hands' 32: a 2-module wrist, four fingers of three joints and a two-joint
thumb a side) on position servos whose torque is clipped to the module's torque–speed line, a free joint at the
pelvis, contact geoms `L`, `R` (the soles) and `L_knuckle`, `R_knuckle` (the backs of the middle fingers' middle
phalanges, the knuckle-walking contacts). By default the policy drives the 31 limb and spine joints and the hands
hold the knuckle-walking curl (`action.hands: held`); `action.hands: policy` gives it all 57.

`RobotModel.missing_requirements()` states what the model does not have, rather than inventing it:

* **no sensors in the MJCF**: the observations are computed from MuJoCo's state (base angular velocity, gravity in
  the base frame, joint positions and velocities, contact flags). The physical robot needs an IMU in the pelvis,
  joint encoders and contact switches on the soles and the knuckles to provide the same; the base linear velocity
  is *estimated* (finite differences of the pelvis position in simulation; an observer on the real machine).
* **no actuator dynamics beyond the PD and the torque–speed line**: no gear friction, backlash or thermal limit.
  Domain randomization covers some of it (joint friction and damping, motor strength, latency).
* **the physics step must be ≤ 1 ms** (Chiron's implicit servo integration); the configuration uses 1 kHz.

## Observations, actions, commands

Policy observation (`belisario_rl.env.ObsLayout`, every block deployable): joint positions as offsets from the default
pose (57), joint velocities (57), base angular velocity (3), projected gravity (3), estimated base linear velocity
(3), contact flags (4), the previous action (31, or 57 with `action.hands: policy`), the normalized command (5).
Critic extras (privileged, training only): true base velocity, height above the terrain, contact normal forces,
the current push, ground friction, mass scale, the goal offset.

Action: normalized joint-position offsets, `target = default + scale · a` (`scale` 0.5 rad), clipped to the joint
ranges; the PD runs at the physics rate with the torque limit of each servo (`belisario_rl.pd`).

Command (`belisario_rl.commands.Command`): `vx` [m/s], `vy` [m/s], `yaw_rate` [rad/s], `height` [m], `stop`. The
ranges are in the configuration (`commands:`); the policy sees them normalized to ±1 and a stop flag. A stop
zeroes the speeds in what the policy sees and the rewards give no forward credit. The same `Command` goes to the
programmed gaits (`belisario_rl.controllers.ProgrammedGaitController`) and to the learned policy
(`PolicyController`), so the two are swapped behind one interface.

## Configuration

`training/configs/base.yaml` holds every key (`belisario_rl.config.Config`): sim rates and episode length, command
ranges, observation scales and noise, the action and PD settings, reward weights, termination, terrain, domain
randomization, disturbances, PPO, the run. A stage (`training/configs/stages/<stage>.yaml`) overrides it; a session
(`session.yaml`) names the scenes, their weights and parameters and the stages to run. From the command line,
`--set section.key=value` overrides anything (`--set ppo.num_envs=4 --set run.device=cpu`).

### Sessions and scenes

A session file:

```yaml
name: belisario_knuckle_walk
base: base.yaml
config: {ppo: {num_envs: 8}}
stages: [1_stance, 2_shift, 3_walk]
scenes:
  - name: flat
    weight: 0.8
  - name: rough
    weight: 0.2
    params: {rms: 0.02, correlation_length: 0.4}
```

The environments are shared out among the scenes by weight (`belisario_rl.session.assign_scenes`). Scenes
(`belisario_rl.scenes`): `flat`, `rough`, `steps` (locomotion), `door` (the lever door of `gorilla_tasks`), `window`,
`obstacles` (reach past them), `wall`, `tree`, `hull`, `rope` (climb to a goal height; their props are exactly those
of `gorilla_scenarios`, the scenery the pre-programmed climbs were solved on). `training/configs/session_locomotion.yaml`
is the first experiment; `session.yaml` adds the tasks.

## The curriculum

| stage | what | notes |
|---|---|---|
| `1_stance` | a stable crouched stance on four | stop commands only, no randomization |
| `2_shift` | weight shifting | small height and lateral commands |
| `3_walk` | slow forward knuckle-walking | 0.1–0.3 m/s |
| `4_speed` | variable forward and backward speed | light randomization begins |
| `5_turn` | turning and lateral motion | |
| `6_transitions` | starting, stopping, direction changes | commands resampled every 2 s, sensor noise |
| `7_terrain` | uneven terrain | rough fields; steps through a session's scenes |
| `8_disturbance` | pushes at the pelvis | full randomization |

Each stage is a YAML of overrides; `--stages all` runs them in turn, every stage starting from the previous one's
last checkpoint; `--stage <name> --resume <checkpoint>` starts any stage from any checkpoint. The programmed gaits
can seed the learning as a reference: set `reward.imitation` > 0 and the environment's `reference`
(`belisario_rl.reference.ReferenceGait`, the knuckle walk of `gorilla_motions`) adds a term for staying near the
keyframes' joint angles; keep it small or zero once the policy walks.

## Commands

All from the repository root. Training:

```bash
python -m training.belisario_rl.scripts.train --session training/configs/session_locomotion.yaml --stage 1_stance
```

```bash
python -m training.belisario_rl.scripts.train --session training/configs/session_locomotion.yaml --stage 2_shift --resume training/runs/belisario_knuckle_walk/1_stance_<timestamp>/model_latest.pt
```

```bash
python -m training.belisario_rl.scripts.train --session training/configs/session_locomotion.yaml --stages all
```

The whole curriculum with the task scenes:

```bash
python -m training.belisario_rl.scripts.train --session training/configs/session.yaml --stages all --device cuda
```

Each stage makes a run directory `training/runs/<session name>/<stage>_<timestamp>/` with `config.yaml` (the resolved
configuration), `session.json`, TensorBoard events, `model_<iteration>.pt` every `ppo.save_every` iterations and
`model_latest.pt`. Seeds: `run.seed` (or `--seed`), applied to numpy, torch and each environment (`seed + index`).

TensorBoard:

```bash
tensorboard --logdir training/runs
```

Evaluation (deterministic: the action mean, no noise, no randomization, fixed seeds):

```bash
python -m training.belisario_rl.scripts.evaluate --checkpoint training/runs/<run>/model_latest.pt --scene flat --vx 0.25 --episodes 5
```

Deterministic inference printing the state (or running an exported ONNX actor):

```bash
python -m training.belisario_rl.scripts.infer --checkpoint training/runs/<run>/model_latest.pt --vx 0.2 --yaw 0.1
```

Visualisation (MuJoCo's viewer, or a movie):

```bash
python -m training.belisario_rl.scripts.visualize --checkpoint training/runs/<run>/model_latest.pt --scene rough --movie training/runs/walk.mp4
```

ONNX export (the observation normalizer folded in; a `.json` beside it with the observation blocks and the action's
meaning):

```bash
python -m training.belisario_rl.scripts.export_onnx --checkpoint training/runs/<run>/model_latest.pt --out training/exports/belisario_actor.onnx
```

Programmed gait against the learned policy (the same commands, the same scenes, tracking error, falls, energy):

```bash
python -m training.belisario_rl.scripts.compare_gaits --checkpoint training/runs/<run>/model_latest.pt --scene flat --vx 0.25
```

Keyboard or joystick control: see `training/CONTROL_EXAMPLE.md`.

## CPU and CUDA, memory

The environments are MuJoCo on the CPU in subprocesses (`ppo.num_envs`, one process each; `--no-subprocess` for
debugging). The networks are small (two MLPs 256-256-128, ~0.3 M parameters): the GPU holds the rollout buffer
(`num_envs × steps_per_env × (obs + critic + act)` floats, a few MB) and the minibatches, so a 4 GB card is far
more than enough; `run.device: auto` picks CUDA when available. On a laptop the bottleneck is the physics: 8
environments at 1 kHz with 50 Hz control is ~8 × 20 physics steps per policy step; expect roughly 1–3 k
environment steps per second across the processes. Memory on the CPU side: each environment is one MuJoCo model
(a few MB) plus the process; 8–16 environments fit comfortably in 8 GB. If the machine thermally throttles, lower
`ppo.num_envs` and raise `ppo.steps_per_env` to keep the batch size.

## Checkpoints

A checkpoint (`torch.save`) holds the actor-critic weights, the observation normalizer, the optimizer and learning
rate, the iteration, the stage, the dimensions and the full configuration; `--resume` restores all of it, and a
checkpoint from one stage starts the next. Dimensions must match the environment (a different `action.hands` or
command layout is a new policy).

## Integration with the programmed gaits

`gorilla_motions.knuckle_walk` (and the scenarios' sequences) are joint keyframes on a base track. In training they
serve as (1) the default pose the actions are offsets from (`quad_pose`), (2) an optional imitation reference, (3)
the baseline `compare_gaits` measures against, through the same `Command` interface. The scenario sequences
(`labs/legged/designs/gorilla_scenarios.py`: the wall, the tree, the hull, the rope, the window, the obstacles, the
door) are the starting references for the task scenes: `labs/legged/sims/space_gorilla_scenes.py` plays them.

## Issuing commands at inference

`PolicyController.act(obs, command, t)` takes a `Command`; the environment's `reset(options={"command": Command(vx=0.3)})`
fixes one for an episode, `evaluate`/`infer`/`visualize` take `--vx --vy --yaw --height --stop`, and `teleop` edits
one live. On a robot, build the observation vector in the layout the exported `.json` lists, normalize the command
with the same ranges (`CommandRanges.normalize`), run the ONNX actor, and send `default + scale · action` as the joint
targets to the position servos.

## Limitations and the first experiment

* Nothing has been executed here: the first run will find mistakes. Start with the tests, then a few iterations
  of stage `1_stance` with `--num-envs 2 --iterations 5 --no-subprocess`.
* The programmed knuckle walk only holds up in simulation with an upright assist; the policy has to learn balance
  the assist was providing, which is what stages 1–3 are for.
* The climbing scenes have no hold-aware observations (the policy does not see where a rung is beyond the goal
  offset in the critic); they are placeholders until a perception input exists. The door's lever needs the hand's
  curl, which `action.hands: policy` enables.
* The fingers' self-locking knuckle drives (`harmonic 12 Nm, brake`) are what lets a hang hold on the real machine;
  in MuJoCo a PD of the same torque cannot hold the frame's weight on four fingers for long - the physics climb is
  not expected to work before the policy learns it, and the kinematic references are what the sims play.

The recommended first experiment: `session_locomotion.yaml`, stage `1_stance` (~300 iterations, half an hour on a
laptop), then `3_walk` from its checkpoint (skip `2_shift` the first time) and watch `reward/track_lin_vel` and the
fall rate in TensorBoard; `evaluate` at 0.2 m/s; then `4_speed` onward.
