#!/usr/bin/env python3
"""Drive a trained policy from the keyboard in MuJoCo's viewer: the command (``commands.Command``) is edited by
key presses through the viewer's key callback and the policy follows it.

    python -m kerchak_rl.scripts.teleop --checkpoint training/runs/<run>/model_latest.pt

Keys (in the viewer window):  W/S forward speed +/-0.05 m/s   A/D lateral +/-0.05   Q/E yaw rate +/-0.1 rad/s
                              R/F height +/-0.02 m             space: stop and hold    X: zero everything

A joystick can feed the same ``Command`` (``pygame``'s joystick axes mapped to vx, vy, yaw rate): ``--joystick``
uses it when ``pygame`` is installed.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kerchak_rl.commands import Command, CommandRanges  # noqa: E402

JOYSTICK_AXES = {"vx": 1, "vy": 0, "yaw": 2}   #: pygame axis numbers: the left stick's vertical and horizontal, the right stick's horizontal
JOYSTICK_STOP_BUTTON = 0
from kerchak_rl.config import Config, _from_dict  # noqa: E402
from kerchak_rl.controllers import PolicyController  # noqa: E402
from kerchak_rl.env import KerchakEnv  # noqa: E402
from kerchak_rl.export import build_actor_from_checkpoint  # noqa: E402
from kerchak_rl.ppo.runner import load_checkpoint  # noqa: E402
from kerchak_rl.scenes import build_scene  # noqa: E402


class KeyCommand:
    """The command edited by keys (GLFW key codes: letters are their ASCII upper-case codes, space is 32)."""

    def __init__(self, ranges: CommandRanges, height: float):
        self.ranges = ranges
        self.cmd = Command(0.0, 0.0, 0.0, height, True)

    def key(self, keycode):
        c = self.cmd
        k = chr(keycode) if 0 <= keycode < 256 else ""
        if k == "W":
            c = Command(c.vx + 0.05, c.vy, c.yaw_rate, c.height, False)
        elif k == "S":
            c = Command(c.vx - 0.05, c.vy, c.yaw_rate, c.height, False)
        elif k == "A":
            c = Command(c.vx, c.vy + 0.05, c.yaw_rate, c.height, False)
        elif k == "D":
            c = Command(c.vx, c.vy - 0.05, c.yaw_rate, c.height, False)
        elif k == "Q":
            c = Command(c.vx, c.vy, c.yaw_rate + 0.1, c.height, False)
        elif k == "E":
            c = Command(c.vx, c.vy, c.yaw_rate - 0.1, c.height, False)
        elif k == "R":
            c = Command(c.vx, c.vy, c.yaw_rate, c.height + 0.02, c.stop)
        elif k == "F":
            c = Command(c.vx, c.vy, c.yaw_rate, c.height - 0.02, c.stop)
        elif keycode == 32:
            c = Command(0.0, 0.0, 0.0, c.height, True)
        elif k == "X":
            c = Command(0.0, 0.0, 0.0, c.height, False)
        self.cmd = self.ranges.clip(c)
        print(f"command: vx {self.cmd.vx:+.2f} vy {self.cmd.vy:+.2f} yaw {self.cmd.yaw_rate:+.2f} h {self.cmd.height:.2f} stop {self.cmd.stop}")


class JoystickCommand:
    def __init__(self, ranges: CommandRanges, height: float):
        import pygame
        pygame.init()
        pygame.joystick.init()
        self.js = pygame.joystick.Joystick(0)
        self.js.init()
        self.pygame, self.ranges, self.height = pygame, ranges, height

    def current(self) -> Command:
        self.pygame.event.pump()
        vx = -self.js.get_axis(JOYSTICK_AXES["vx"]) * self.ranges.vx[1]
        vy = -self.js.get_axis(JOYSTICK_AXES["vy"]) * self.ranges.vy[1]
        wz = -self.js.get_axis(JOYSTICK_AXES["yaw"]) * self.ranges.yaw_rate[1]
        stop = bool(self.js.get_button(JOYSTICK_STOP_BUTTON))
        return self.ranges.clip(Command(vx, vy, wz, self.height, stop))


def main(argv=None):
    ap = argparse.ArgumentParser(description="keyboard or joystick teleoperation of a Kerchak policy")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--scene", default="flat")
    ap.add_argument("--joystick", action="store_true")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    import mujoco.viewer
    ck = load_checkpoint(a.checkpoint, a.device)
    cfg = _from_dict(Config, ck["config"])
    env = KerchakEnv(cfg, scene=build_scene(a.scene, cfg), seed=3, eval_mode=True)
    ac, norm = build_actor_from_checkpoint(ck, a.device)
    ctrl = PolicyController(ac, norm, deterministic=True, device=a.device)
    ranges = CommandRanges.from_config(cfg.commands)
    keys = KeyCommand(ranges, env.default["height"])
    joy = JoystickCommand(ranges, env.default["height"]) if a.joystick else None
    obs, info = env.reset(options={"command": keys.cmd})
    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=keys.key) as viewer:
        t = 0.0
        while viewer.is_running():
            cmd = joy.current() if joy else keys.cmd
            env.command = cmd
            act = ctrl.act(obs, cmd, t)
            obs, r, term, trunc, info = env.step(act)
            t += env.dt
            viewer.sync()
            time.sleep(env.dt)
            if term:
                obs, info = env.reset(options={"command": cmd})
    env.close()


if __name__ == "__main__":
    main()
