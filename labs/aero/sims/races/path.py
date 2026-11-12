"""The racing line through a course's zones and the speed schedule an aircraft's limits allow (the same for the
MuJoCo autopilot and the mock flyer): the corner speed each zone's clearance allows at the lateral acceleration the
thrust vector gives, braking and acceleration between zones, the top speed."""
from __future__ import annotations

import math

import numpy as np

G = 9.81


class Path:
    def __init__(self, points, radii, *, a_lateral: float, a_brake: float, a_accel: float, v_top: float, safety: float = 0.8, v_start: float = 2.0, v_end: float = 3.0):
        P = np.asarray(points, dtype=float)
        self.P = P
        seg = np.diff(P, axis=0)
        self.L = np.maximum(np.linalg.norm(seg, axis=1), 1e-6)
        self.s = np.concatenate([[0.0], np.cumsum(self.L)])
        self.tan = seg / self.L[:, None]
        v = np.full(len(P), v_top)
        for i in range(1, len(P) - 1):
            t0, t1 = self.tan[i - 1], self.tan[i]
            turn = math.acos(max(-1.0, min(1.0, float(np.dot(t0, t1)))))
            r_fit = radii[i] / max(math.tan(turn / 2), 0.05)           # a turn through 'turn' inside the zone's clearance
            r = min(r_fit, 80.0)
            v[i] = min(v_top, safety * math.sqrt(a_lateral * r), safety * math.sqrt(a_lateral * radii[i] * 3.0))
        v[0], v[-1] = v_start, v_end
        for i in range(len(P) - 2, -1, -1):
            v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * a_brake * self.L[i]))
        for i in range(1, len(P)):
            v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2 * a_accel * self.L[i - 1]))
        self.v = v
        self.limits = {"a_lateral": a_lateral, "a_brake": a_brake, "a_accel": a_accel, "v_top": v_top, "safety": safety}
        self.turn = np.zeros(len(P))
        for i in range(1, len(P) - 1):
            self.turn[i] = math.acos(max(-1.0, min(1.0, float(np.dot(self.tan[i - 1], self.tan[i])))))
        self.zone_half = np.full(len(P), 1.5)

    @property
    def length(self) -> float:
        return float(self.s[-1])

    def at(self, s: float):
        s = float(np.clip(s, 0.0, self.s[-1]))
        i = min(int(np.searchsorted(self.s, s, side="right")) - 1, len(self.L) - 1)
        f = (s - self.s[i]) / self.L[i]
        return self.P[i] + f * (self.P[i + 1] - self.P[i]), self.tan[i], self.v[i] + f * (self.v[i + 1] - self.v[i]), i

    def project(self, p, s_hint: float) -> float:
        best, s_best = 1e9, s_hint
        i0 = max(int(np.searchsorted(self.s, s_hint, side="right")) - 2, 0)
        for i in range(i0, min(i0 + 3, len(self.L))):
            d = p - self.P[i]
            f = float(np.clip(np.dot(d, self.tan[i]) / self.L[i], 0.0, 1.0))
            q = self.P[i] + f * (self.P[i + 1] - self.P[i])
            dist = float(np.linalg.norm(p - q))
            if dist < best:
                best, s_best = dist, self.s[i] + f * self.L[i]
        return max(s_best, s_hint - 0.5)

    def time_estimate(self) -> float:
        return float(sum(2 * self.L[i] / max(self.v[i] + self.v[i + 1], 0.5) for i in range(len(self.L))))

    def table(self):
        import pandas as pd
        return pd.DataFrame({"s [m]": self.s, "x": self.P[:, 0], "y": self.P[:, 1], "z": self.P[:, 2], "v schedule [m/s]": self.v})


def carrot(path: Path, s: float, nz: int, *, lookahead_s: float = 0.35, la_max: float = 6.0, parked: bool = False):
    """The moving reference for the autopilots (the mock and the MuJoCo run alike): a point a short way ahead on the
    line at the line's own speed — the look-ahead shortened at the next zone so the corner is not cut by more than
    the zone's half-size (the zones are flown through) — and, when the aircraft has already missed the zone
    (``parked``), the reference held at the zone with a walking-pace feedforward so the aircraft is drawn into its
    box. Returns (p_ref, v_ref, a_ref)."""
    s_zone = path.s[nz]
    _, _, v_here, _ = path.at(s)
    la_free = min(max(1.0, lookahead_s * v_here), la_max)
    half = float(path.zone_half[nz])
    la_cut = half / max(math.sin(path.turn[nz] / 2), 0.15)             # the cut a look-ahead la makes at a corner is ≈ la · sin(turn / 2)
    la = max(0.8, min(la_free, la_cut))
    if parked:
        la = min(la, max(s_zone - s, 0.0) + 0.2)
    p_ref, tan, v_sched, _ = path.at(s + la)
    _, tan_n, _, _ = path.at(s + la + 1.0)
    v_ref = tan * min(v_sched, v_here + 2.0)
    a_ref = (tan_n - tan) * v_sched ** 2 * 0.5
    if parked:
        v_ref = tan * min(float(np.linalg.norm(v_ref)), 1.5)
        a_ref = np.zeros(3)
    return p_ref, v_ref, a_ref


def resync(path: Path, s: float, next_zone_s: float, is_last: bool) -> tuple:
    """Past the next zone's station without entering it: back to it (a missed gate must be flown); returns
    (s, parked)."""
    if s > next_zone_s + 1.0 and not is_last:
        return max(next_zone_s - 0.5, 0.0), True
    return s, False


class Drops:
    """The MuJoCo autopilot's guard for a drop: a zone where the line turns from level flight into a descent steeper
    than 45° (the hangar's stair opening). Arriving with speed and looking past the corner, the reference jumps below
    the aircraft, the geometric controller asks for almost no thrust, the rotors sit at idle and leave no roll or
    pitch torque (the controller assumes unsaturated rotors), and the aircraft falls tumbling into the opening's edge.
    So: approach the top and stop on it, wait over it until aligned (``ALIGN_M`` horizontally, below ``ALIGN_V``),
    then descend along the steep segments at no more than ``DESCENT_V`` with the reference at most ``DESCENT_LEAD``
    below. Fixed rules of the autopilot, not tuning knobs."""

    ALIGN_M, ALIGN_V, DESCENT_V, DESCENT_LEAD, APPROACH_M, STEEP = 0.5, 0.8, 2.0, 1.0, 1.5, -0.7

    def __init__(self, path: Path):
        self.path = path
        n = len(path.tan)
        self.steep = [path.tan[i][2] < self.STEEP for i in range(n)]
        self.tops = {i for i in range(1, n) if self.steep[i] and not self.steep[i - 1]}
        self.released = set()
        self.mode = None                                                   # the last decision: approach, align, descend or None (races.video_panel shows it)

    def adjust(self, s: float, nz: int, p, v, ref: tuple) -> tuple:
        P = self.path
        self.mode = None
        if nz in self.tops and P.s[nz] - s < self.APPROACH_M:              # approaching a top: stop on it (re-armed after a reset)
            self.released.discard(nz)
            self.mode = "approach"
            return P.P[nz].copy(), np.zeros(3), np.zeros(3)
        top = nz - 1
        if top in self.tops and top not in self.released:                  # over the top: align before descending
            if np.linalg.norm(p[:2] - P.P[top][:2]) < self.ALIGN_M and np.linalg.norm(v[:2]) < self.ALIGN_V:
                self.released.add(top)
            else:
                self.mode = "align"
                return P.P[top].copy(), np.zeros(3), np.zeros(3)
        p_ref, v_ref, a_ref = ref
        if 0 <= nz - 1 < len(self.steep) and self.steep[nz - 1]:          # on the way down: slow, the reference close below
            self.mode = "descend"
            n = float(np.linalg.norm(v_ref))
            v_ref = v_ref * min(1.0, self.DESCENT_V / max(n, 1e-9))
            p_ref = p_ref.copy()
            p_ref[2] = max(p_ref[2], p[2] - self.DESCENT_LEAD)
            a_ref = np.zeros(3)
        return p_ref, v_ref, a_ref
