"""Swarm algorithms for the air match, after Nomios (``technolog-spectrumai/nomios``, ``main`` at 60c9c24: ``swarm/formation.py``,
``swarm/assignment.py``, ``swarm/controllers.py``). Vegeta does not import Nomios; what the formation chase needs is here,
re-written for this simulation's frame (x forward, y left, z up) and its velocity commands, with Nomios's numbers:

- ``wedge_offsets`` — the V of migrating birds: the first member at the apex, arms ``WEDGE_ANGLE`` off the heading,
  members ``spacing`` apart along each arm, taking turns right and left (Nomios ``formation.wedge_offsets``).
- ``column_offsets`` — single file, ``spacing`` apart behind the head (Nomios's column with one member per section).
- ``PathReference`` — the point the formation hangs on, and its recorded path: a wedge slot is the offset turned to the
  reference's heading (a rigid body, with ω × r for its velocity); a column slot sits **on the path** at its arc
  length behind the head and moves along it at the reference's speed — "like a train", so the body bends round a
  corner and through a hoop exactly where the head went (Nomios ``GroupController._path_slots``).
- ``min_cost_assignment`` — the Hungarian algorithm (Kuhn–Munkres with potentials, as Nomios ``assignment.py``): when
  the formation changes, every member gets the new slot that makes the sum of squared moves smallest, so few paths cross.
- ``Morph`` — a change of formation: each slot moves from where its member's old slot was to its new place on a
  smoothstep that starts and ends at rest, over T = the longest move / (0.35 × top speed) kept between 2 and 8 s.
- ``slot_command`` — a member's wanted velocity: the slot's velocity fed forward plus ``SLOT_GAIN`` × the slot error
  (Nomios ``BoidController``, term 2); ``separation`` — the reflex added on top, quadratic in the intrusion into the
  margin (term 4), never traded against the tracking.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

WEDGE_ANGLE = 35.0                 # degrees between each arm of the V and the heading (Nomios)
SLOT_GAIN = 2.0                    # 1/s: closing speed per metre of slot error (Nomios ControllerParams.slot_gain)
PATH_STEP = 0.25                   # m between recorded path points (Nomios)
MORPH_SPEED = 0.35                 # the longest move of a change is flown at this share of the top speed (Nomios)
MORPH_T = (2.0, 8.0)               # ... and the change takes between these seconds
SEPARATION_MARGIN = 1.0            # m of gap below which members push apart
SEPARATION_GAIN = 6.0              # m/s at zero gap


def wedge_offsets(n: int, spacing: float, climb: float = 0.0) -> np.ndarray:
    """``n`` slots of a V in the formation's frame (x forward, y left, z up): the apex first, then right, left, right …
    ``climb`` raises each rank a little above the one ahead (out of its propeller wash)."""
    a = math.radians(WEDGE_ANGLE)
    out = [(0.0, 0.0, 0.0)]
    for i in range(1, n):
        k = (i + 1) // 2
        sign = -1.0 if i % 2 else 1.0                               # the first wingman on the right (-y)
        out.append((-k * math.cos(a) * spacing, sign * k * math.sin(a) * spacing, k * climb))
    return np.array(out)


def column_offsets(n: int, spacing: float) -> np.ndarray:
    """Single file: slot k is ``k × spacing`` behind the head."""
    return np.array([(-k * spacing, 0.0, 0.0) for k in range(n)])


def min_cost_assignment(cost) -> list:
    """For a square cost matrix, the assignment ``a`` (row i gets column a[i]) with the smallest total cost: the O(n³)
    shortest-augmenting-path Hungarian algorithm with row and column potentials (Nomios ``assignment.py``)."""
    n = len(cost)
    if n == 0:
        return []
    inf = math.inf
    u, v, match, way = [0.0] * (n + 1), [0.0] * (n + 1), [0] * (n + 1), [0] * (n + 1)
    for i in range(1, n + 1):
        match[0] = i
        j0 = 0
        minv, used = [inf] * (n + 1), [False] * (n + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = match[j0], inf, 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(n + 1):
                if used[j]:
                    u[match[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if match[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            match[j0] = match[j1]
            j0 = j1
    out = [0] * n
    for j in range(1, n + 1):
        out[match[j] - 1] = j - 1
    return out


def smoothstep(x: float) -> tuple:
    """(weight, rate) of a move that starts and ends at rest: 3x² − 2x³ and its derivative, for x in 0…1."""
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x), 6 * x * (1 - x)


class PathReference:
    """The reference point with its heading, its velocity and the path it flew (newest first)."""

    def __init__(self, pos, heading: float, keep_m: float):
        self.pos = np.asarray(pos, float).copy()
        self.heading, self.vel, self.yaw_rate = float(heading), np.zeros(3), 0.0
        back = np.array([-math.cos(heading), -math.sin(heading), 0.0]) * PATH_STEP
        n = int(math.ceil(keep_m / PATH_STEP)) + 4
        self.path = deque((self.pos + back * k for k in range(1, n + 1)), maxlen=n)

    def update(self, pos, vel, dt: float) -> None:
        pos, vel = np.asarray(pos, float), np.asarray(vel, float)
        if float(np.hypot(vel[0], vel[1])) > 0.5:
            h = math.atan2(vel[1], vel[0])
            dh = (h - self.heading + math.pi) % (2 * math.pi) - math.pi
            self.yaw_rate = dh / max(dt, 1e-6) if abs(dh) < 0.5 else 0.0
            self.heading = h
        self.pos, self.vel = pos.copy(), vel.copy()
        if float(np.linalg.norm(self.pos - self.path[0])) >= PATH_STEP:
            self.path.appendleft(self.pos.copy())                   # the oldest point drops off the end

    def rigid_slot(self, offset) -> tuple:
        """A wedge slot: the offset turned to the heading; its velocity is the reference's plus ω × r."""
        c, s = math.cos(self.heading), math.sin(self.heading)
        r = np.array([c * offset[0] - s * offset[1], s * offset[0] + c * offset[1], offset[2]])
        return self.pos + r, self.vel + self.yaw_rate * np.array([-r[1], r[0], 0.0])

    def path_slot(self, behind: float, lateral=(0.0, 0.0)) -> tuple:
        """A column slot: on the path ``behind`` metres of arc from the reference, moving along it at the reference's
        speed (past the path's end: straight on)."""
        pts = [self.pos] + list(self.path)
        acc, speed = 0.0, float(np.linalg.norm(self.vel))
        for k in range(len(pts) - 1):
            seg = pts[k + 1] - pts[k]
            length = float(np.linalg.norm(seg))
            if acc + length >= behind or k == len(pts) - 2:
                t = (behind - acc) / length if length > 1e-9 else 0.0
                along = -seg / length if length > 1e-9 else np.array([math.cos(self.heading), math.sin(self.heading), 0.0])
                left = np.array([-along[1], along[0], 0.0])
                return pts[k] + seg * t + left * lateral[0] + np.array([0.0, 0.0, lateral[1]]), along * speed
            acc += length
        return self.pos.copy(), self.vel.copy()


class Morph:
    """A change of formation in flight: who takes which new slot (the least sum of squared moves), and the weight of
    the new shape against the old one while the slots move over."""

    def __init__(self, positions, new_slots, v_max: float, t0: float):
        cost = [[float(np.sum((np.asarray(p) - np.asarray(q)) ** 2)) for q in new_slots] for p in positions]
        self.assign = min_cost_assignment(cost)                      # member i → new slot assign[i]
        longest = max(math.sqrt(cost[i][j]) for i, j in enumerate(self.assign))
        self.T = min(max(longest / (MORPH_SPEED * v_max), MORPH_T[0]), MORPH_T[1])
        self.t0 = t0
        self.crossings = self.count_crossings(positions, new_slots, self.assign)

    def weight(self, t: float) -> tuple:
        w, rate = smoothstep((t - self.t0) / self.T)
        return w, rate / self.T

    def done(self, t: float) -> bool:
        return t >= self.t0 + self.T

    @staticmethod
    def count_crossings(positions, new_slots, assign) -> int:
        """How many pairs of members' straight moves cross in plan view (the fewer, the safer the change)."""
        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
        n, k = len(positions), 0
        for i in range(n):
            for j in range(i + 1, n):
                a, b, c, d = positions[i], new_slots[assign[i]], positions[j], new_slots[assign[j]]
                if ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d):
                    k += 1
        return k


def slot_command(pos, slot_pos, slot_vel, v_max: float) -> np.ndarray:
    """A member's wanted velocity: the slot's velocity fed forward, plus the gain on the slot error (bounded)."""
    err = np.asarray(slot_pos, float) - np.asarray(pos, float)
    close = SLOT_GAIN * err
    n = float(np.linalg.norm(close))
    if n > v_max:
        close *= v_max / n
    return np.asarray(slot_vel, float) + close


def separation(pos, others, reach: float, margin: float = SEPARATION_MARGIN, gain: float = SEPARATION_GAIN) -> np.ndarray:
    """The reflex on top of the tracking: away from every member whose gap is under the margin, quadratic in how far
    it is inside it. ``others``: their positions; ``reach``: the sum of the two aircraft's radii."""
    push = np.zeros(3)
    for q in others:
        off = np.asarray(pos, float) - np.asarray(q, float)
        d = float(np.linalg.norm(off))
        gap = d - reach
        if gap < margin and d > 1e-6:
            s = 1.0 - max(gap, 0.0) / margin
            push += off / d * gain * s * s
    return push
