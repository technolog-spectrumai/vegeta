# Driving the policy by hand: keyboard and joystick

`training/belisario_rl/scripts/teleop.py` runs a trained policy in MuJoCo's viewer and lets you edit its command
live. The command is the same `Command` the environment, the evaluation scripts and the programmed gaits take
(`vx`, `vy`, `yaw_rate`, `height`, `stop`), clipped to the configuration's ranges.

```bash
python -m training.belisario_rl.scripts.teleop --checkpoint training/runs/<run>/model_latest.pt --scene flat
```

Keys (with the viewer window focused):

| key | effect |
|---|---|
| W / S | forward speed +0.05 / −0.05 m/s |
| A / D | lateral speed +0.05 / −0.05 m/s |
| Q / E | yaw rate +0.1 / −0.1 rad/s |
| R / F | pelvis height +0.02 / −0.02 m |
| space | stop and hold (and release) |
| X | zero every command |

The current command prints in the terminal each time it changes.

Joystick (`pygame` installed):

```bash
python -m training.belisario_rl.scripts.teleop --checkpoint training/runs/<run>/model_latest.pt --joystick
```

The left stick's vertical axis is `vx`, its horizontal axis `vy`, the right stick's horizontal axis the yaw rate,
each scaled to the command ranges; a button (0) toggles stop. Axis and button numbers are the script's
`JOYSTICK_AXES` and `JOYSTICK_STOP_BUTTON`: change them for your controller.

## The same command from your own code

```python
from training.belisario_rl.commands import Command
from training.belisario_rl.controllers import PolicyController
from training.belisario_rl.env import BelisarioEnv
from training.belisario_rl.export import build_actor_from_checkpoint
import torch

ck = torch.load("training/runs/<run>/model_latest.pt", map_location="cpu", weights_only=False)
ac, norm = build_actor_from_checkpoint(ck)
from training.belisario_rl.config import _from_dict, Config
env = BelisarioEnv(_from_dict(Config, ck["config"]), eval_mode=True)
policy = PolicyController(ac, norm)
obs, info = env.reset(seed=0, options={"command": Command(vx=0.2)})
for step in range(500):
    command = Command(vx=0.2, yaw_rate=0.1 if step > 200 else 0.0)   # whatever your controller decides
    env.command = command
    obs, reward, terminated, truncated, info = env.step(policy.act(obs, command, env.t))
    if terminated or truncated:
        break
```

The environment re-reads `env.command` every step, so any source (a network message, a planner, a game pad) can
set it. On the robot, the exported ONNX actor takes the observation vector the export's `.json` describes, with the
command block normalized by `CommandRanges.normalize`.
