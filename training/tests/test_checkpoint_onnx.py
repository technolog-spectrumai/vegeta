"""Checkpoint serialisation and the ONNX export (skipped without torch / onnx)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _ac(obs=20, critic=5, act=6):
    from kerchak_rl.ppo.networks import ActorCritic
    return ActorCritic(obs, obs + critic, act, (256, 256, 128), (256, 256, 128), "elu", 0.8)


def test_actor_critic_shapes():
    ac = _ac()
    x = torch.zeros(3, 20)
    a = ac.act(x, deterministic=True)
    assert a.shape == (3, 6)
    assert ac.evaluate(torch.zeros(3, 25)).shape == (3,)
    assert [m.out_features for m in ac.actor if isinstance(m, torch.nn.Linear)] == [256, 256, 128, 6]


def test_ppo_update_runs():
    from kerchak_rl.config import PPOConfig
    from kerchak_rl.ppo.ppo import PPO
    cfg = PPOConfig(num_envs=2, steps_per_env=8, epochs=1, minibatches=2)
    ac = _ac()
    algo = PPO(ac, cfg, torch.device("cpu"))
    algo.init_buffer(20, 25, 6)
    for _ in range(8):
        o, c = torch.randn(2, 20), torch.randn(2, 25)
        algo.act(o, c)
        algo.record(o, c, torch.randn(2), torch.zeros(2))
    algo.finish_rollout(torch.randn(2, 25))
    stats = algo.update()
    assert set(stats) >= {"value_loss", "surrogate_loss", "entropy", "lr"}


def test_checkpoint_roundtrip(tmp_path):
    from kerchak_rl.ppo.normalizer import RunningMeanStd
    ac = _ac()
    norm = RunningMeanStd(20)
    norm.update(torch.randn(50, 20) * 3 + 1)
    ck = {"iteration": 7, "actor_critic": ac.state_dict(), "normalizer": norm.state_dict(), "critic_normalizer": None, "algo": None,
          "dims": {"obs": 20, "critic": 5, "act": 6}, "config": {"ppo": {"actor_hidden": [256, 256, 128], "critic_hidden": [256, 256, 128],
                                                                        "activation": "elu", "init_noise_std": 0.8}}, "stage": "test"}
    path = tmp_path / "model.pt"
    torch.save(ck, path)
    from kerchak_rl.export import build_actor_from_checkpoint
    ac2, norm2 = build_actor_from_checkpoint(torch.load(path, map_location="cpu", weights_only=False))
    x = torch.randn(4, 20)
    assert torch.allclose(ac.actor(norm(x)), ac2.actor(norm2(x)))
    assert norm2.frozen


def test_onnx_export(tmp_path):
    pytest.importorskip("onnx")
    from kerchak_rl.export import export_onnx
    from kerchak_rl.ppo.normalizer import RunningMeanStd
    ac = _ac()
    norm = RunningMeanStd(20)
    ck = {"iteration": 1, "actor_critic": ac.state_dict(), "normalizer": norm.state_dict(), "critic_normalizer": None, "algo": None,
          "dims": {"obs": 20, "critic": 5, "act": 6}, "config": {"ppo": {"actor_hidden": [256, 256, 128], "critic_hidden": [256, 256, 128],
                                                                        "activation": "elu", "init_noise_std": 0.8}}, "stage": "test"}
    path = export_onnx(ck, tmp_path / "actor.onnx", layout_blocks=[("x", 20)], action_meta={"scale_rad": 0.5})
    assert path.exists() and path.with_suffix(".json").exists()
    ort = pytest.importorskip("onnxruntime")
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = np.random.randn(1, 20).astype(np.float32)
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    ref = ac.actor(norm(torch.as_tensor(x))).detach().numpy()
    assert np.allclose(out, ref, atol=1e-4)
