"""The configuration: nested dataclasses with defaults, loaded from YAML (``load_config``), overridden by a stage
(``curriculum``) and by command-line ``key=value`` pairs, snapshotted into every run directory."""
from __future__ import annotations

import copy
import dataclasses
import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = ["Config", "SimConfig", "CommandConfig", "ObsConfig", "ActionConfig", "RewardConfig", "TerminationConfig",
           "TerrainConfig", "RandomizationConfig", "DisturbanceConfig", "PPOConfig", "RunConfig", "load_config", "to_dict",
           "merge", "apply_overrides", "save_snapshot"]


@dataclass
class SimConfig:
    sim_hz: int = 1000                   #: the physics step (Chiron's default 1 ms)
    control_hz: int = 50                 #: the policy's step; decimation = sim_hz / control_hz
    episode_seconds: float = 12.0
    headless: bool = True
    casing: str = "none"                 #: the frame's casing (``space_unit_robots.bare``): none | plastic | lorica


@dataclass
class CommandConfig:
    vx: tuple = (-0.3, 0.4)              #: forward speed [m/s]
    vy: tuple = (-0.15, 0.15)            #: lateral speed [m/s]
    yaw_rate: tuple = (-0.5, 0.5)        #: [rad/s]
    height: tuple = (0.5, 0.7)           #: the pelvis above the ground [m] (0.62 on four)
    stop_probability: float = 0.15       #: a command that says stop and hold
    resample_seconds: float = 4.0        #: a new command every so often within an episode
    hold_still_speed: float = 0.03       #: below this the command is a stop (|vx|, |vy|, |yaw|)


@dataclass
class ObsConfig:
    joint_pos_scale: float = 1.0
    joint_vel_scale: float = 0.05
    ang_vel_scale: float = 0.25
    lin_vel_scale: float = 2.0
    noise_joint_pos: float = 0.01        #: sensor noise (rad), applied when randomization.sensor_noise is on
    noise_joint_vel: float = 0.5
    noise_ang_vel: float = 0.1
    noise_gravity: float = 0.03
    lin_vel_estimate: str = "finite_difference"   #: how the deployable base-velocity estimate is made: finite_difference | none
    clip: float = 10.0


@dataclass
class ActionConfig:
    scale: float = 0.5                   #: a policy output of ±1 is ±scale rad about the default pose
    clip: float = 1.0
    default_pose: str = "quad"           #: quad (on four) | stand
    kp_fraction: float = 1.0             #: the PD gains relative to the design's servo gains
    kd_fraction: float = 1.0
    torque_fraction: float = 1.0         #: of the servo's stall torque
    velocity_limit_fraction: float = 1.0 #: of the servo's no-load speed (the torque–speed line)
    latency_steps: int = 0               #: control steps of actuator latency (randomization may add)
    hands: str = "held"                  #: held (the hands' 26 joints stay at the default pose's curl, the policy drives the 31 limb and spine joints) | policy (all 57)


@dataclass
class RewardConfig:
    track_lin_vel: float = 1.5
    track_ang_vel: float = 0.75
    tracking_sigma: float = 0.25
    height: float = 0.5
    height_sigma: float = 0.08
    orientation: float = -1.0            #: projected gravity in the pelvis's xy, against the commanded pitch
    contact_stability: float = 0.3       #: three or more contacts down while moving slowly, no contact bouncing
    slip: float = -0.2
    energy: float = -2e-5                #: |τ·q̇|
    action_rate: float = -0.01
    joint_accel: float = -2.5e-7
    joint_vel: float = -1e-3
    coordination: float = 0.2            #: the lateral sequence: a hand and the foot of the same side not both swinging
    joint_limits: float = -1.0
    body_collision: float = -1.0
    fall: float = -50.0
    stop_still: float = 0.5              #: when stopped: stillness, and no credit for moving
    imitation: float = 0.0               #: optional: closeness to the programmed knuckle walk
    torque_saturation: float = -0.02
    alive: float = 0.05


@dataclass
class TerminationConfig:
    min_height: float = 0.3
    max_tilt_deg: float = 55.0           #: the pelvis's angle off the commanded pitch
    body_collision_terminates: bool = True
    body_collision_geoms: tuple = ("head", "ribcage", "lumbar", "pelvis_shell", "yoke")


@dataclass
class TerrainConfig:
    kind: str = "flat"                   #: flat | rough | steps
    rms: float = 0.02                    #: rough: RMS height [m]
    correlation_length: float = 0.3      #: rough [m]
    step_height: float = 0.05            #: steps
    step_spacing: float = 1.0
    extent: tuple = (-2.0, 8.0, -2.0, 2.0)
    cell: float = 0.02
    seed: int = 0
    friction: float = 0.9


@dataclass
class DisturbanceConfig:
    enabled: bool = False
    interval_seconds: tuple = (2.0, 5.0)
    force_xy: float = 150.0              #: push magnitude [N], applied for ``duration_seconds`` at the pelvis
    duration_seconds: float = 0.2


@dataclass
class RandomizationConfig:
    enabled: bool = False
    mass_scale: tuple = (0.9, 1.1)       #: per body
    joint_friction: tuple = (0.0, 0.5)   #: frictionloss [N·m]
    joint_damping_scale: tuple = (0.8, 1.2)
    motor_strength: tuple = (0.85, 1.15) #: kp and torque scale
    latency_steps: tuple = (0, 2)
    ground_friction: tuple = (0.5, 1.2)
    sensor_noise: bool = False
    pushes: bool = False
    init_pose_noise: float = 0.05        #: rad on the joints at reset
    init_yaw: float = 3.14159
    init_vel_noise: float = 0.1


@dataclass
class PPOConfig:
    num_envs: int = 8
    steps_per_env: int = 32              #: rollout length per environment
    epochs: int = 5
    minibatches: int = 4
    learning_rate: float = 3e-4
    adaptive_lr: bool = True             #: KL-adaptive (RSL-RL style)
    desired_kl: float = 0.01
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    value_clip: float = 0.2
    entropy_coef: float = 0.005
    value_coef: float = 1.0
    max_grad_norm: float = 1.0
    init_noise_std: float = 0.8
    actor_hidden: tuple = (256, 256, 128)
    critic_hidden: tuple = (256, 256, 128)
    activation: str = "elu"
    normalize_obs: bool = True
    normalize_rewards: bool = False
    iterations: int = 1500
    save_every: int = 50
    log_every: int = 1


@dataclass
class RunConfig:
    name: str = "belisario"
    log_dir: str = "training/runs"
    seed: int = 1
    device: str = "auto"                 #: auto | cpu | cuda
    deterministic_eval: bool = True
    stage: str = "1_stance"
    resume: str = ""                     #: a checkpoint path
    tensorboard: bool = True


@dataclass
class Config:
    sim: SimConfig = field(default_factory=SimConfig)
    commands: CommandConfig = field(default_factory=CommandConfig)
    obs: ObsConfig = field(default_factory=ObsConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    termination: TerminationConfig = field(default_factory=TerminationConfig)
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)
    disturbance: DisturbanceConfig = field(default_factory=DisturbanceConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    run: RunConfig = field(default_factory=RunConfig)


def to_dict(cfg) -> dict:
    """A plain dict (tuples as lists) of a config."""
    if is_dataclass(cfg):
        return {f.name: to_dict(getattr(cfg, f.name)) for f in fields(cfg)}
    if isinstance(cfg, (list, tuple)):
        return [to_dict(v) for v in cfg]
    return cfg


def _from_dict(cls, data: dict):
    kw = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        v = data[f.name]
        if is_dataclass(f.type) or (isinstance(f.default_factory, type) and is_dataclass(f.default_factory)):
            sub = f.default_factory()
            kw[f.name] = _from_dict(type(sub), v) if isinstance(v, dict) else sub
        elif isinstance(v, list):
            kw[f.name] = tuple(v)
        else:
            kw[f.name] = v
    return cls(**kw)


def merge(base: dict, over: dict) -> dict:
    """A deep merge: ``over``'s values replace ``base``'s, dict by dict."""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    """The defaults, then the YAML file at ``path`` (which may name a ``base`` file to start from), then ``overrides``."""
    data = to_dict(Config())
    if path is not None:
        data = merge(data, _load_yaml_chain(Path(path)))
    if overrides:
        data = merge(data, overrides)
    return _from_dict(Config, data)


def _load_yaml_chain(path: Path) -> dict:
    with open(path) as f:
        doc = yaml.safe_load(f) or {}
    base = doc.pop("base", None)
    if base:
        parent = _load_yaml_chain((path.parent / base).resolve())
        return merge(parent, doc)
    return doc


def apply_overrides(cfg: Config, pairs: list[str]) -> Config:
    """``section.key=value`` pairs from the command line (values parsed as YAML: numbers, lists, booleans)."""
    data = to_dict(cfg)
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"override {pair!r}: expected section.key=value")
        key, raw = pair.split("=", 1)
        value = yaml.safe_load(raw)
        node = data
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return _from_dict(Config, data)


def save_snapshot(cfg: Config, run_dir: str | Path) -> Path:
    """The run's configuration as YAML (and JSON) next to its checkpoints."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    data = to_dict(cfg)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    with open(run_dir / "config.json", "w") as f:
        json.dump(data, f, indent=1)
    return run_dir / "config.yaml"
