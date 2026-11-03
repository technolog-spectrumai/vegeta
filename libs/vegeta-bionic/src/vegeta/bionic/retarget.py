"""From the hand to a pincer: the jaw opening and the grip force a robot pincer should follow.

Two modes, switched from the operator's interface (``teleop.OperatorUI``), and a clutch (also from the interface):

* **position** — the jaw opening follows the thumb–index pinch aperture (linearly between ``aperture_closed`` and
  ``aperture_open``); once the pincer reports contact, squeezing the fingers further (below the aperture at the moment
  of contact) commands grip force, ``grip_gain`` N per mm, up to ``grip_max``, and the jaws hold where they touched;
* **rate** — the index finger's flexion (MCP + PIP, 0 straight .. 1 fully bent) sets the closing speed: past
  ``rate_band`` it closes, under it opens, within it holds; for fine adjustment;
* **clutch** — the command freezes; on release, position mode re-references (the opening keeps where it was and
  follows the fingers' changes from there), so a clutch never makes the pincer jump.

Every command passes ``safety.Limits`` (opening range and jaw speed; force range and rate).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .hand import RANGES, HandPose, HandSize, pinch_aperture
from .safety import Limits


@dataclass
class PincerMap:
    """The mapping's constants (SI). Defaults: an adult hand (pinch 12-90 mm) to a pincer opening 0-0.20 m at the tips."""

    aperture_closed: float = 0.012
    aperture_open: float = 0.090
    opening_max: float = 0.20
    jaw_speed: float = 0.40                     # m/s at the tips (the drive's limit, or less)
    grip_gain: float = 60.0                     # N per mm of squeeze past contact
    grip_max: float = 1500.0                    # N at the tips
    grip_rate: float = 6000.0                   # N/s
    rate_band: tuple = (0.40, 0.60)             # rate mode: index flexion below opens, above closes
    rate_speed: float = 0.08                    # m/s at full flexion in rate mode

    def opening(self, aperture: float) -> float:
        u = (aperture - self.aperture_closed) / (self.aperture_open - self.aperture_closed)
        return min(max(u, 0.0), 1.0) * self.opening_max


@dataclass
class PincerCommand:
    t: float
    opening: float                              # m at the tips
    grip: float                                 # N squeeze (0 when not in contact)
    mode: str
    clutched: bool


def index_flexion(pose: HandPose) -> float:
    """The index finger's flexion, 0 (straight) .. 1 (MCP and PIP at their range's end)."""
    a = pose.joints["index_mcp"] + pose.joints["index_pip"]
    hi = RANGES["index_mcp"][1] + RANGES["index_pip"][1]
    return min(max(a / hi, 0.0), 1.0)


@dataclass
class PincerRetarget:
    """One pincer's mapping state. ``update(t, pose, size, contact)`` -> ``PincerCommand``."""

    side: str = "R"
    map: PincerMap = field(default_factory=PincerMap)
    mode: str = "position"
    clutched: bool = False

    def __post_init__(self):
        m = self.map
        self.open_lim = Limits(0.0, m.opening_max, m.jaw_speed)
        self.grip_lim = Limits(0.0, m.grip_max, m.grip_rate, value=0.0)
        self.offset = 0.0
        self.contact_aperture = None
        self.last = None
        self.t_prev = None

    # --- the operator's interface
    def set_mode(self, mode: str):
        if mode not in ("position", "rate"):
            raise ValueError("mode must be 'position' or 'rate'")
        self.mode = mode
        self.offset = None                       # position mode re-references on its next update

    def set_clutch(self, on: bool):
        if self.clutched and not on:
            self.offset = None                   # re-reference on release
        self.clutched = bool(on)

    # --- the loop
    def update(self, t: float, pose: HandPose | None, size: HandSize, contact: bool = False) -> PincerCommand:
        dt = 0.0 if self.t_prev is None else t - self.t_prev
        self.t_prev = t
        if pose is None or self.clutched:        # no fresh hand (watchdog) or clutched: hold the last command
            if self.last is None:
                self.last = PincerCommand(t, self.open_lim.apply(0.0, 0.0), 0.0, self.mode, self.clutched)
            self.last = PincerCommand(t, self.last.opening, self.last.grip, self.mode, self.clutched)
            return self.last
        a = pinch_aperture(pose, size)
        current = self.open_lim.value if self.open_lim.value is not None else self.map.opening(a)
        if self.mode == "position":
            if self.offset is None:
                self.offset = current - self.map.opening(a)
            target = self.map.opening(a) + self.offset
        else:
            u = index_flexion(pose)
            lo, hi = self.map.rate_band
            v = 0.0 if lo <= u <= hi else (-(u - hi) / (1 - hi) if u > hi else (lo - u) / lo) * self.map.rate_speed
            target = current + v * dt
        grip = 0.0
        if contact:
            if self.contact_aperture is None:
                self.contact_aperture, self.contact_opening = a, current
            squeeze_mm = max(self.contact_aperture - a, 0.0) * 1000.0
            if self.mode == "position":
                grip = self.map.grip_gain * squeeze_mm
                target = max(target, self.contact_opening)   # the jaws hold where they touched; opening the hand releases
            else:
                grip = self.map.grip_gain * max(index_flexion(pose) - self.map.rate_band[1], 0.0) * 50.0
                target = max(target, self.contact_opening)
        else:
            self.contact_aperture = None
        opening = self.open_lim.apply(target, dt)
        g = self.grip_lim.apply(grip, dt)
        self.last = PincerCommand(t, opening, g, self.mode, self.clutched)
        return self.last
