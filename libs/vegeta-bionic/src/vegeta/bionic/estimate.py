"""From glove samples back to a hand: per-sensor calibration from held key poses, joint angles (sensed and inferred),
the magnetometer's hard-iron offset from a wrist sweep, and the wrist's orientation from the IMU (a Mahony filter: gyro integration corrected by gravity and the magnetic
field, with gyro-bias estimation).

Calibration: the user holds the key poses of ``motions.calibration_routine``; their nominal angles (``hand.key_pose``)
stand for the user's own (a real user is off by a few degrees: ``pose_error_deg`` in the tests). Each sensor's model
counts = o + A sin(k (angle - centre)) is linear in (o, A), fitted by least squares over the held samples; the
angle is then centre + arcsin((counts - o) / A) / k. What the fit cannot remove — each unit's phase, cross-talk, drift
— is the residual error the tests measure.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import _quat as Q
from .glove import GloveSample, GloveSpec
from .hand import JOINTS, RANGES, HandPose, key_pose
from .sensors import EARTH_FIELD

DIP_PIP = 2.0 / 3.0                      # DIP flexion per PIP flexion (flexor tendon coupling)
RING_LITTLE_ABD = 0.5


@dataclass
class Calibration:
    spec: GloveSpec
    offset: dict = field(default_factory=dict)
    amplitude: dict = field(default_factory=dict)
    residual_counts: dict = field(default_factory=dict)
    mag_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def with_magnetometer(self, sweep_samples: list) -> "Calibration":
        """The magnetometer's hard-iron offset from a wrist sweep (``motions.wrist_tour``), kept with the calibration."""
        self.mag_offset, self.mag_radius, self.mag_residual = fit_hard_iron([s.mag for s in sweep_samples])
        return self

    @classmethod
    def fit(cls, samples: list, windows: dict, spec: GloveSpec, poses: dict | None = None) -> "Calibration":
        """Fit every sensor from the samples inside each held window (``{pose name: (t0, t1)}``), the poses' angles
        from ``poses`` (default: ``key_pose`` of each name)."""
        poses = poses or {n: key_pose(n) for n in windows}
        mounts = spec.mounts
        rows = {j: ([], []) for j in spec.sensed}
        for s in samples:
            for name, (t0, t1) in windows.items():
                if t0 <= s.t <= t1:
                    for i, j in enumerate(spec.sensed):
                        m = mounts[j]
                        rows[j][0].append([1.0, math.sin(m.k * (poses[name].joints[j] - m.centre))])
                        rows[j][1].append(float(s.hall[i]))
        cal = cls(spec)
        for j, (X, y) in rows.items():
            X, y = np.array(X), np.array(y)
            if len(np.unique(X[:, 1].round(6))) < 2:
                raise ValueError(f"the calibration poses do not move {j}: hold at least two poses that differ there")
            (o, A), *_ = np.linalg.lstsq(X, y, rcond=None)
            cal.offset[j], cal.amplitude[j] = float(o), float(A)
            cal.residual_counts[j] = float(np.sqrt(np.mean((X @ [o, A] - y) ** 2)))
        return cal

    def angles(self, sample: GloveSample) -> dict:
        """Every joint's angle [rad]: the sensed ones from their counts, the rest inferred; clamped to the ranges."""
        mounts = self.spec.mounts
        out = {}
        for i, j in enumerate(self.spec.sensed):
            m = mounts[j]
            u = (float(sample.hall[i]) - self.offset[j]) / self.amplitude[j]
            out[j] = m.centre + math.asin(max(-1.0, min(1.0, u))) / m.k
        for f in ("index", "middle", "ring", "little"):
            if f"{f}_dip" not in out and f"{f}_pip" in out:
                out[f"{f}_dip"] = DIP_PIP * out[f"{f}_pip"]
        out.setdefault("ring_abd", RING_LITTLE_ABD * out.get("little_abd", 0.0))
        for j in JOINTS:
            out.setdefault(j, 0.0)
        return {j: float(np.clip(a, *RANGES[j])) for j, a in out.items()}


def fit_hard_iron(mags) -> tuple:
    """The magnetometer's hard-iron offset from readings taken while the hand turns through many orientations (a
    sphere fit, |m - b|^2 = r^2, linear in b and r^2 - |b|^2): (offset [uT], field radius [uT], rms residual)."""
    M = np.asarray(mags, float)
    X = np.c_[2 * M, np.ones(len(M))]
    y = (M * M).sum(axis=1)
    sol, *_ = np.linalg.lstsq(X, y, rcond=None)
    b = sol[:3]
    r = math.sqrt(max(sol[3] + b @ b, 0.0))
    res = float(np.sqrt(np.mean((np.linalg.norm(M - b, axis=1) - r) ** 2)))
    return b, r, res


def initial_orientation(acc, mag) -> np.ndarray:
    """The orientation (sensor to world) from one accelerometer and magnetometer reading at rest (TRIAD)."""
    up_b = np.asarray(acc, float) / np.linalg.norm(acc)                       # at rest the accelerometer reads -g: up
    m_b = np.asarray(mag, float)
    east_b = np.cross(m_b, up_b)
    east_b /= np.linalg.norm(east_b)
    north_b = np.cross(up_b, east_b)
    up_w, mw = np.array([0.0, 0.0, 1.0]), EARTH_FIELD
    east_w = np.cross(mw, up_w)
    east_w /= np.linalg.norm(east_w)
    north_w = np.cross(up_w, east_w)
    R = np.column_stack([north_w, east_w, up_w]) @ np.column_stack([north_b, east_b, up_b]).T
    return _R_to_q(R)


def _R_to_q(R) -> np.ndarray:
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
        elif i == 1:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    return Q.normalize(q)


class Mahony:
    """Mahony's complementary filter on SO(3): q' = q (x) (omega - b + kp e)/2, b' = -ki e, where e is the error between
    the measured and the predicted gravity and magnetic field directions. ``kp`` [1/s], ``ki`` [1/s^2]."""

    def __init__(self, kp: float = 1.0, ki: float = 0.05, q0=None):
        self.kp, self.ki = kp, ki
        self.q = np.array([1.0, 0, 0, 0]) if q0 is None else Q.normalize(q0)
        self.bias = np.zeros(3)

    def update(self, gyro, acc, mag, dt: float) -> np.ndarray:
        Rt = Q.to_R(self.q).T
        e = np.zeros(3)
        a = np.asarray(acc, float)
        if np.linalg.norm(a) > 1e-6:
            e += np.cross(a / np.linalg.norm(a), Rt @ np.array([0.0, 0.0, 1.0]))
        m = np.asarray(mag, float)
        if np.linalg.norm(m) > 1e-6:
            mh = Rt @ EARTH_FIELD                                              # the field predicted in the body
            # the heading part only: both projected off the measured "up" so the field's dip does not fight gravity
            up = a / np.linalg.norm(a) if np.linalg.norm(a) > 1e-6 else Rt @ np.array([0.0, 0, 1])
            mp = m - np.dot(m, up) * up
            hp = mh - np.dot(mh, up) * up
            if np.linalg.norm(mp) > 1e-6 and np.linalg.norm(hp) > 1e-6:
                e += np.cross(mp / np.linalg.norm(mp), hp / np.linalg.norm(hp))
        self.bias -= self.ki * e * dt
        w = np.asarray(gyro, float) - self.bias + self.kp * e
        self.q = Q.normalize(self.q + 0.5 * dt * Q.mul(self.q, np.r_[0.0, w]))
        return self.q


def estimate_hand(samples: list, cal: Calibration, *, kp: float = 1.0, ki: float = 0.05) -> list:
    """HandPoses from a run of samples: the joints by ``cal``, the wrist by a Mahony filter started from the first
    sample (TRIAD)."""
    mo = cal.mag_offset
    f = Mahony(kp, ki, initial_orientation(samples[0].acc, samples[0].mag - mo))
    out, tp = [], samples[0].t
    for s in samples:
        q = f.update(s.gyro, s.acc, s.mag - mo, max(s.t - tp, 0.0))
        tp = s.t
        out.append(HandPose(cal.angles(s), q.copy()))
    return out
