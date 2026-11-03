"""Scripted hand motions: key poses on a timeline, joined smoothly, with the wrist turning — what drives a
``SimGlove`` (and later the simulated pincers) before a real hand does.

    m = calibration_routine()               # flat, fist, pinch, spread held in turn
    pose = m(3.2)                            # the HandPose at t = 3.2 s
    m.windows                                # {pose name: (t0, t1)}: when each key pose is held
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import _quat as Q
from .hand import HandPose, interpolate, key_pose


def _smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


@dataclass
class Motion:
    """Keyframes ``[(t, HandPose)]``: held between, joined by smoothstep; ``windows`` names the held stretches."""

    keys: list
    windows: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return float(self.keys[-1][0])

    def __call__(self, t: float) -> HandPose:
        ks = self.keys
        if t <= ks[0][0]:
            return ks[0][1]
        for (t0, a), (t1, b) in zip(ks[:-1], ks[1:]):
            if t <= t1:
                return interpolate(a, b, _smooth((t - t0) / (t1 - t0) if t1 > t0 else 1.0))
        return ks[-1][1]


def _with_wrist(p: HandPose, roll=0.0, pitch=0.0, yaw=0.0) -> HandPose:
    return HandPose(dict(p.joints), Q.from_euler(roll, pitch, yaw), p.wrist_pos.copy())


def calibration_routine(hold: float = 2.0, move: float = 1.0) -> Motion:
    """Hold flat, fist, pinch and spread for ``hold`` s each (the wrist still), ``move`` s between them."""
    keys, windows, t = [], {}, 0.0
    for name in ("flat", "fist", "pinch", "spread"):
        p = key_pose(name)
        keys += [(t, p), (t + hold, p)]
        windows[name] = (t + 0.3, t + hold - 0.1)
        t += hold + move
    return Motion(keys, windows)


def grasp_cycles(n: int = 5, period: float = 2.0, wrist_amp_deg: float = 40.0) -> Motion:
    """Open, pinch, fist and open again ``n`` times while the wrist rolls, pitches and turns."""
    keys, t = [], 0.0
    a = math.radians(wrist_amp_deg)
    seq = ["flat", "pinch", "fist", "flat"]
    for k in range(n):
        for j, name in enumerate(seq):
            ph = 2 * math.pi * (k * len(seq) + j) / (len(seq) * 1.7)
            keys.append((t, _with_wrist(key_pose(name), a * math.sin(ph), 0.5 * a * math.sin(1.3 * ph), a * math.sin(0.7 * ph))))
            t += period / len(seq)
    return Motion(keys)


def wrist_tour(duration: float = 120.0, amp_deg: float = 60.0, period: float = 8.0) -> Motion:
    """The hand still and flat, the wrist moving through roll, pitch and yaw for ``duration`` s (a filter's test)."""
    keys, t, a = [], 0.0, math.radians(amp_deg)
    while t <= duration:
        ph = 2 * math.pi * t / period
        keys.append((t, _with_wrist(key_pose("flat"), a * math.sin(ph), 0.6 * a * math.sin(0.73 * ph), a * math.sin(0.41 * ph))))
        t += period / 8
    return Motion(keys)
