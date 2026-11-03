"""Haptic feedback, stage 1 of the iterative path (``docs/bionic.md``): a vibration motor per glove that tells the
operator about contact and grip force. Brakes and motors on the glove's linkages (force feedback) come later.

* ``VibrationMotor`` — an eccentric-rotating-mass motor: its vibration follows the drive with a first-order lag
  (``tau_s``, the spin-up), up to ``max_g``.
* ``ContactFeedback`` — what drives it: a short full-strength pulse at the onset of contact (``pulse_s``), then a
  steady level that grows with the grip force on a log scale (a force from ``force_ref`` to ``force_max`` maps to
  ``floor`` .. 1, so light touches are felt and heavy grips do not saturate).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class VibrationMotor:
    tau_s: float = 0.03
    max_g: float = 1.5
    level: float = 0.0                           # 0..1

    def step(self, drive: float, dt: float) -> float:
        a = 1.0 - math.exp(-max(dt, 0.0) / self.tau_s)
        self.level += (min(max(drive, 0.0), 1.0) - self.level) * a
        return self.level * self.max_g


@dataclass
class ContactFeedback:
    pulse_s: float = 0.08
    force_ref: float = 5.0
    force_max: float = 1500.0
    floor: float = 0.25
    _onset: float | None = None

    def drive(self, t: float, contact: bool, force: float) -> float:
        if not contact:
            self._onset = None
            return 0.0
        if self._onset is None:
            self._onset = t
        if t - self._onset < self.pulse_s:
            return 1.0
        f = max(force, self.force_ref)
        u = math.log(f / self.force_ref) / math.log(self.force_max / self.force_ref)
        return self.floor + (1.0 - self.floor) * min(max(u, 0.0), 1.0)
