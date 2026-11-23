"""The race rules, one set for every course, and the referee that applies them to a trajectory (the MuJoCo run and
the mock flyer alike).

- **zones in order**: the racer must enter every zone in the published order; entering a later zone first does not
  count (a missed gate: the racer must go back; the referee keeps pointing at the missed one);
- **checkpoints**: every zone marked ``checkpoint`` (and the start) is a reset point: after a collision reset the
  racer restarts hovering at the last checkpoint passed, with ``reset_penalty_s`` added to its time;
- **collisions**: any contact between the aircraft and the scene (ground included once airborne) is a collision; the
  rule ``collision`` is ``reset`` (teleport to the last checkpoint, count it) or ``count`` (fly on, count it) or
  ``stop`` (the race ends, DNF); more than ``max_collisions`` ends the race as DNF;
- **lap completion**: the lap is complete when the finish zone is entered after every other zone, in order; the lap
  time runs from leaving the start zone to entering the finish zone, penalties included;
- **timeout**: ``timeout_s`` of racing time ends the race as DNF.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

STUCK = "stuck (time limit)"     # the stats' status of a lap the simulation stopped with the race still open


@dataclass
class Rules:
    collision: str = "reset"            # reset | count | stop
    reset_penalty_s: float = 5.0
    max_collisions: int = 5
    timeout_s: float = 300.0
    reset_hover_s: float = 1.0          # the racer hangs at the checkpoint this long after a reset
    text: str = field(default="see races.rules", repr=False)


@dataclass
class Referee:
    course: object
    rules: Rules = field(default_factory=Rules)

    def __post_init__(self):
        self.next = 1                     # the start zone is index 0: the racer begins inside it
        self.passed = [0]
        self.gate_times = {}
        self.collisions = []
        self.resets = []
        self.penalty = 0.0
        self.t_start = None
        self.t_finish = None
        self.status = "racing"            # racing | finished | dnf
        self.reason = ""
        self.last_checkpoint = 0
        self.reset_request = None         # (position, hover until) the runner must honour
        self.missed = []

    @property
    def zones(self):
        return self.course.zones

    def update(self, t: float, p, in_contact: bool, airborne: bool) -> list:
        """Advance the referee with the racer at ``p`` at time ``t``; returns the events of this step."""
        ev = []
        if self.status != "racing":
            return ev
        if self.t_start is None and not self.zones[0].contains(p):
            self.t_start = t
            ev.append((t, "start"))
        if in_contact and airborne:
            self.collisions.append((t, tuple(float(x) for x in p)))
            ev.append((t, f"collision {len(self.collisions)}"))
            if self.rules.collision == "stop" or len(self.collisions) > self.rules.max_collisions:
                self.status, self.reason = "dnf", "collision" if self.rules.collision == "stop" else "too many collisions"
                ev.append((t, "DNF: " + self.reason))
                return ev
            if self.rules.collision == "reset":
                z = self.zones[self.last_checkpoint]
                self.penalty += self.rules.reset_penalty_s
                self.resets.append((t, z.name))
                self.reset_request = (np.array(z.center, dtype=float), t + self.rules.reset_hover_s)
                self.next = self.last_checkpoint + 1
                ev.append((t, f"reset to {z.name} (+{self.rules.reset_penalty_s:.0f} s)"))
                return ev
        if self.next < len(self.zones):
            for j in range(self.next + 1, len(self.zones) - 1):         # a later zone entered first: a missed gate, noted once (the finish excepted: it can sit over the start)
                if self.zones[j].contains(p) and (j, self.next) not in self.missed:
                    self.missed.append((j, self.next))
                    ev.append((t, f"missed '{self.zones[self.next].name}' (entered '{self.zones[j].name}' first)"))
            if self.zones[self.next].contains(p):
                z = self.zones[self.next]
                self.passed.append(self.next)
                self.gate_times[z.name] = (t - (self.t_start or t)) + self.penalty
                ev.append((t, f"zone {self.next} '{z.name}'"))
                if z.kind == "checkpoint":
                    self.last_checkpoint = self.next
                self.next += 1
                if z.kind == "finish":
                    self.t_finish = t
                    self.status = "finished"
                    ev.append((t, f"finish: {self.lap_time:.1f} s"))
        if self.status == "racing" and self.t_start is not None and t - self.t_start > self.rules.timeout_s:
            self.status, self.reason = "dnf", "timeout"
            ev.append((t, "DNF: timeout"))
        return ev

    @property
    def lap_time(self):
        if self.t_finish is None or self.t_start is None:
            return float("nan")
        return self.t_finish - self.t_start + self.penalty

    def stats(self) -> dict:
        return {"status": self.status, "reason": self.reason, "lap time [s]": self.lap_time, "penalty [s]": self.penalty, "zones passed": f"{len(set(self.passed))} of {len(self.zones)}",
                "collisions": len(self.collisions), "resets": len(self.resets), "missed gates": len({m[1] for m in self.missed if m[1] not in self.passed}),
                "gates flown out of order (then recovered)": len({m[1] for m in self.missed if m[1] in self.passed})}
