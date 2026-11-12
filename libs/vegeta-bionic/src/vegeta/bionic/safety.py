"""Safety primitives shared by teleoperation and (later) worn devices. They are software layers on top of the hardware
limits ``docs/bionic.md`` requires, never a replacement for them.

* ``Watchdog`` — trips when no fresh input has arrived for ``timeout_s``; whoever reads it must then hold or release
  (the teleop: freeze the pincers; a worn device: remove power).
* ``Limits`` — clamps a command to a range and its rate of change to a maximum speed.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Watchdog:
    timeout_s: float = 0.1
    last: float | None = None

    def feed(self, t: float) -> None:
        self.last = t

    def ok(self, t: float) -> bool:
        return self.last is not None and t - self.last <= self.timeout_s


@dataclass
class Limits:
    lo: float
    hi: float
    rate: float                                       # units per second
    value: float | None = None

    def apply(self, target: float, dt: float) -> float:
        target = min(max(target, self.lo), self.hi)
        if self.value is None:
            self.value = target
        else:
            step = self.rate * max(dt, 0.0)
            self.value += min(max(target - self.value, -step), step)
        return self.value
