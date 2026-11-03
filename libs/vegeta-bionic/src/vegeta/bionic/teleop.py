"""The teleoperation loop's pieces, independent of any simulator: a glove channel (the simulated glove on the
operator's hand, its calibration, the downlink, the estimate, the watchdog), a delay line for what comes back (the
robot's contact force and the cameras: the uplink), the operator's interface (a scheduled mock for now: clutch and
mode switches), and ``TeleopSide``, one glove driving one pincer.

The loop, per control step at time t (the lab calls it)::

    side.operator_hand(t, pose)             # the operator's true hand -> glove sample -> downlink
    cmd = side.robot_step(t, contact)       # what arrived by t -> estimate -> retarget -> a PincerCommand
    side.feedback(t, contact, force)        # the robot's contact force -> uplink -> vibration on the glove

Latency is measured, not assumed (``docs/bionic.md``): ``down_latency_s`` / ``up_latency_s`` and their jitter and
loss are the sweep's variables.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .estimate import Calibration, Mahony, initial_orientation
from .glove import GloveSpec, SimGlove
from .haptics import ContactFeedback, VibrationMotor
from .hand import HandPose, HandSize
from .link import InProcessLink, encode
from . import motions
from .retarget import PincerMap, PincerRetarget
from .safety import Watchdog


@dataclass
class DelayLine:
    """Values put in at t come out at t + latency (``get(t)``: the newest one that has arrived, or ``default``)."""

    latency_s: float = 0.0
    default: object = None
    _q: deque = field(default_factory=deque)
    _out: object = None

    def put(self, t: float, value) -> None:
        self._q.append((t + self.latency_s, value))

    def get(self, t: float):
        while self._q and self._q[0][0] <= t:
            self._out = self._q.popleft()[1]
        return self.default if self._out is None else self._out


@dataclass
class OperatorUI:
    """The operator's interface, mocked: ``events`` = [(t, side, what, value)], ``what`` 'clutch' (bool) or 'mode'
    ('position' / 'rate'); ``apply(t, sides)`` fires the events that are due."""

    events: list = field(default_factory=list)
    fired: list = field(default_factory=list)

    def apply(self, t: float, sides: dict) -> None:
        due = [e for e in self.events if e[0] <= t]
        for e in due:
            _, side, what, value = e
            r = sides[side].retarget
            r.set_clutch(value) if what == "clutch" else r.set_mode(value)
            self.fired.append(e)
            self.events.remove(e)


class GloveChannel:
    """One glove: sample the operator's hand, send it down the link, and on the robot's side estimate the hand from
    what has arrived (the calibration is done once, offline, as the user would on putting the glove on)."""

    def __init__(self, side: str = "R", *, size: HandSize | None = None, seed: int = 0, latency_s: float = 0.005,
                 jitter_s: float = 0.0, loss: float = 0.0, watchdog_s: float = 0.1):
        self.spec = GloveSpec.default(side)
        self.size = size or HandSize(side=side)
        self.glove = SimGlove(self.spec, self.size, seed=seed)
        r = motions.calibration_routine()
        self.cal = Calibration.fit(self.glove.run(r)[0], r.windows, self.spec)
        self.cal.with_magnetometer(self.glove.run(motions.wrist_tour(20.0))[0])
        self.glove.seq, self.glove._prev = 0, None                      # the glove restarts for the session
        self.link = InProcessLink(latency_s, jitter_s, loss, seed=seed + 1000)
        self.watchdog = Watchdog(watchdog_s)
        self.filter = None
        self.t_last_sample = None
        self.pose = None
        self.next_sample_t = 0.0

    def operator_hand(self, t: float, pose: HandPose):
        """The glove samples at its own rate; call this every control step."""
        if t + 1e-9 >= self.next_sample_t:
            s = self.glove.sample(t, pose)
            self.link.send(t, encode(s))
            self.next_sample_t = t + 1.0 / self.spec.rate_hz
            return s
        return None

    def receive(self, t: float) -> HandPose | None:
        """The hand estimated from what arrived by ``t``; None when the watchdog has tripped (stale glove)."""
        for ta, s in self.link.receive(t):
            mag = s.mag - self.cal.mag_offset
            if self.filter is None:
                self.filter = Mahony(q0=initial_orientation(s.acc, mag))
                dt = 0.0
            else:
                dt = max(s.t - self.t_last_sample, 0.0)
            q = self.filter.update(s.gyro, s.acc, mag, dt)
            self.t_last_sample = s.t
            self.pose = HandPose(self.cal.angles(s), q.copy())
            self.watchdog.feed(ta)                                       # its arrival: a queue that stopped is stale, a slow link is not
        return self.pose if self.watchdog.ok(t) else None


class TeleopSide:
    """One glove driving one pincer, with the feedback coming back up."""

    def __init__(self, side: str, *, seed: int = 0, down_latency_s: float = 0.005, up_latency_s: float = 0.005,
                 jitter_s: float = 0.0, loss: float = 0.0, pincer_map: PincerMap | None = None):
        self.side = side
        self.channel = GloveChannel(side, seed=seed, latency_s=down_latency_s, jitter_s=jitter_s, loss=loss)
        self.retarget = PincerRetarget(side, pincer_map or PincerMap())
        self.up = DelayLine(up_latency_s, default=(False, 0.0))
        self.haptic = ContactFeedback()
        self.motor = VibrationMotor()
        self.vibration_g = 0.0
        self.t_prev = None

    def operator_hand(self, t: float, pose: HandPose):
        return self.channel.operator_hand(t, pose)

    def robot_step(self, t: float, contact: bool):
        pose = self.channel.receive(t)
        return self.retarget.update(t, pose, self.channel.size, contact), pose

    def feedback(self, t: float, contact: bool, force: float) -> float:
        """The robot's contact goes up the link; the glove's motor vibrates with what has arrived."""
        self.up.put(t, (contact, force))
        c, f = self.up.get(t)
        dt = 0.0 if self.t_prev is None else t - self.t_prev
        self.t_prev = t
        self.vibration_g = self.motor.step(self.haptic.drive(t, c, f), dt)
        return self.vibration_g

    def seen_contact(self, t: float) -> tuple:
        """What the operator knows of the contact now (delayed): (contact, force)."""
        return self.up.get(t)
