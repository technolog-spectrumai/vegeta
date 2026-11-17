"""Configuration loading, overrides, stages and sessions."""
from pathlib import Path

from kerchak_rl.config import apply_overrides, load_config, to_dict
from kerchak_rl.curriculum import STAGE_ORDER, apply_stage, next_stage, stage_overrides
from kerchak_rl.session import assign_scenes, load_session

ROOT = Path(__file__).resolve().parents[1]


def test_defaults_and_overrides():
    cfg = load_config(ROOT / "configs" / "base.yaml")
    assert cfg.ppo.actor_hidden == (256, 256, 128) and cfg.sim.control_hz == 50
    cfg = apply_overrides(cfg, ["ppo.num_envs=3", "commands.vx=[-0.1, 0.2]", "randomization.enabled=true"])
    assert cfg.ppo.num_envs == 3 and cfg.commands.vx == (-0.1, 0.2) and cfg.randomization.enabled
    d = to_dict(cfg)
    assert d["ppo"]["num_envs"] == 3


def test_stages_in_order():
    cfg = load_config(ROOT / "configs" / "base.yaml")
    for name in STAGE_ORDER:
        s = apply_stage(cfg, name, ROOT / "configs" / "stages")
        assert s.run.stage == name
        assert stage_overrides(name, ROOT / "configs" / "stages")
    assert apply_stage(cfg, "1_stance").commands.stop_probability == 1.0
    assert apply_stage(cfg, "8_disturbance").disturbance.enabled
    assert next_stage("1_stance") == "2_shift" and next_stage("8_disturbance") is None


def test_session_and_scene_assignment():
    s = load_session(ROOT / "configs" / "session.yaml")
    assert s.scenes and abs(sum(s.scene_weights().values()) - 1.0) < 1e-9
    entries = assign_scenes(s, 16)
    assert len(entries) == 16 and {e.name for e in entries} == {e.name for e in s.scenes}
    entries = assign_scenes(s, 3)
    assert len(entries) == 3
    loc = load_session(ROOT / "configs" / "session_locomotion.yaml")
    assert loc.stages[0] == "1_stance" and loc.config.ppo.num_envs == 8
