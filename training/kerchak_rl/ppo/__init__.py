"""PPO for Kerchak, in PyTorch: a command-conditioned actor-critic with an asymmetric critic
(``networks.ActorCritic``), a running observation normaliser (``normalizer``), the clipped-surrogate update with
GAE and a KL-adaptive learning rate (``ppo.PPO``), and the training loop with checkpoints, TensorBoard and seeds
(``runner.Runner``).

Why not a library: no RL library is installed in this project (``rsl_rl``, ``stable_baselines3`` absent), and
this pipeline was written without network access, so no library's current API could be verified; the update
here follows the structure of RSL-RL's PPO (rollout, GAE, minibatch epochs, adaptive LR, separate critic inputs)
in a few hundred lines that a test can exercise. An RSL-RL runner can be dropped in later: ``vec_env`` already
hands out actor and critic observations separately, which is what its ``VecEnv`` expects.
"""
from .networks import ActorCritic
from .normalizer import RunningMeanStd
from .ppo import PPO, RolloutBuffer
from .runner import Runner

__all__ = ["ActorCritic", "RunningMeanStd", "PPO", "RolloutBuffer", "Runner"]
