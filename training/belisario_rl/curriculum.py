"""The curriculum: eight stages, each a set of overrides on the configuration (commands, terrain, randomization,
disturbances, reward weights), applied in order or started from any one with a checkpoint. The YAML files in
``training/configs/stages/`` mirror these and take precedence when present, so a stage can be tuned without
touching the code."""
from __future__ import annotations

from pathlib import Path

from .config import Config, apply_overrides, load_config, merge, to_dict, _from_dict

__all__ = ["STAGES", "STAGE_ORDER", "stage_overrides", "apply_stage", "next_stage"]

STAGES = {
    "1_stance": {"description": "a stable crouched stance on four: stop commands only, the nominal model",
                 "commands": {"vx": [0.0, 0.0], "vy": [0.0, 0.0], "yaw_rate": [0.0, 0.0], "height": [0.6, 0.64], "stop_probability": 1.0},
                 "reward": {"stop_still": 1.0, "height": 1.0, "contact_stability": 0.5, "track_lin_vel": 0.0, "track_ang_vel": 0.0},
                 "terrain": {"kind": "flat"}, "randomization": {"enabled": False}, "disturbance": {"enabled": False},
                 "sim": {"episode_seconds": 6.0}, "ppo": {"iterations": 300}},
    "2_shift": {"description": "weight shifting: small height and lateral commands, the hands and feet learning to load and unload",
                "commands": {"vx": [0.0, 0.0], "vy": [-0.05, 0.05], "yaw_rate": [0.0, 0.0], "height": [0.55, 0.68], "stop_probability": 0.5},
                "reward": {"stop_still": 0.8, "height": 1.0, "contact_stability": 0.5, "track_lin_vel": 0.5, "track_ang_vel": 0.0},
                "terrain": {"kind": "flat"}, "randomization": {"enabled": False}, "disturbance": {"enabled": False},
                "sim": {"episode_seconds": 8.0}, "ppo": {"iterations": 400}},
    "3_walk": {"description": "slow forward knuckle-walking",
               "commands": {"vx": [0.1, 0.3], "vy": [0.0, 0.0], "yaw_rate": [0.0, 0.0], "height": [0.58, 0.66], "stop_probability": 0.1},
               "reward": {"track_lin_vel": 1.5, "coordination": 0.3, "contact_stability": 0.3},
               "terrain": {"kind": "flat"}, "randomization": {"enabled": False}, "disturbance": {"enabled": False},
               "sim": {"episode_seconds": 10.0}, "ppo": {"iterations": 1500}},
    "4_speed": {"description": "forward and backward at variable speed",
                "commands": {"vx": [-0.3, 0.4], "vy": [0.0, 0.0], "yaw_rate": [0.0, 0.0], "height": [0.55, 0.68], "stop_probability": 0.15},
                "terrain": {"kind": "flat"}, "randomization": {"enabled": True, "mass_scale": [0.95, 1.05], "motor_strength": [0.95, 1.05],
                                                               "joint_friction": [0.0, 0.2], "latency_steps": [0, 1], "sensor_noise": False},
                "disturbance": {"enabled": False}, "sim": {"episode_seconds": 12.0}, "ppo": {"iterations": 1500}},
    "5_turn": {"description": "turning and lateral motion",
               "commands": {"vx": [-0.3, 0.4], "vy": [-0.15, 0.15], "yaw_rate": [-0.5, 0.5], "height": [0.55, 0.68], "stop_probability": 0.15},
               "reward": {"track_ang_vel": 1.0},
               "terrain": {"kind": "flat"}, "randomization": {"enabled": True}, "disturbance": {"enabled": False},
               "sim": {"episode_seconds": 12.0}, "ppo": {"iterations": 1500}},
    "6_transitions": {"description": "starting, stopping and changing direction: commands resampled often",
                      "commands": {"vx": [-0.3, 0.4], "vy": [-0.15, 0.15], "yaw_rate": [-0.5, 0.5], "height": [0.5, 0.7], "stop_probability": 0.3,
                                   "resample_seconds": 2.0},
                      "reward": {"stop_still": 0.8, "action_rate": -0.02},
                      "terrain": {"kind": "flat"}, "randomization": {"enabled": True, "sensor_noise": True}, "disturbance": {"enabled": False},
                      "sim": {"episode_seconds": 14.0}, "ppo": {"iterations": 1500}},
    "7_terrain": {"description": "uneven terrain: rough fields and steps",
                  "commands": {"vx": [-0.2, 0.35], "vy": [-0.1, 0.1], "yaw_rate": [-0.4, 0.4], "height": [0.55, 0.68], "stop_probability": 0.2},
                  "terrain": {"kind": "rough", "rms": 0.03, "correlation_length": 0.35},
                  "randomization": {"enabled": True, "sensor_noise": True, "ground_friction": [0.4, 1.2]}, "disturbance": {"enabled": False},
                  "sim": {"episode_seconds": 14.0}, "ppo": {"iterations": 2000}},
    "8_disturbance": {"description": "pushes at the pelvis, full randomization",
                      "commands": {"vx": [-0.3, 0.4], "vy": [-0.15, 0.15], "yaw_rate": [-0.5, 0.5], "height": [0.5, 0.7], "stop_probability": 0.2},
                      "terrain": {"kind": "rough", "rms": 0.03, "correlation_length": 0.35},
                      "randomization": {"enabled": True, "sensor_noise": True, "pushes": True},
                      "disturbance": {"enabled": True, "force_xy": 200.0, "interval_seconds": [2.0, 5.0]},
                      "sim": {"episode_seconds": 14.0}, "ppo": {"iterations": 2000}},
}
STAGE_ORDER = list(STAGES)


def stage_overrides(name: str, stages_dir: str | Path | None = None) -> dict:
    """The stage's overrides: the YAML in ``stages_dir`` (default ``training/configs/stages``) when there is one,
    else the table above (``description`` dropped)."""
    if name not in STAGES:
        raise KeyError(f"stage {name!r}: one of {', '.join(STAGE_ORDER)}")
    d = Path(stages_dir) if stages_dir else Path(__file__).resolve().parents[1] / "configs" / "stages"
    f = d / f"{name}.yaml"
    if f.exists():
        import yaml
        with open(f) as fh:
            doc = yaml.safe_load(fh) or {}
        doc.pop("base", None)
        doc.pop("description", None)
        return doc
    return {k: v for k, v in STAGES[name].items() if k != "description"}


def apply_stage(cfg: Config, name: str, stages_dir=None) -> Config:
    data = merge(to_dict(cfg), stage_overrides(name, stages_dir))
    data["run"]["stage"] = name
    return _from_dict(Config, data)


def next_stage(name: str) -> str | None:
    i = STAGE_ORDER.index(name)
    return STAGE_ORDER[i + 1] if i + 1 < len(STAGE_ORDER) else None
