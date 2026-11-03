"""Quaternions (w, x, y, z), unit, body-to-world. Kept here so vegeta.bionic needs nothing but numpy."""
from __future__ import annotations

import numpy as np


def normalize(q):
    q = np.asarray(q, float)
    return q / np.linalg.norm(q)


def mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def from_axis_angle(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    return np.r_[np.cos(angle / 2), np.sin(angle / 2) * axis]


def from_euler(roll, pitch, yaw):
    """ZYX (yaw, pitch, roll) Euler angles [rad] to a quaternion."""
    return mul(mul(from_axis_angle((0, 0, 1), yaw), from_axis_angle((0, 1, 0), pitch)), from_axis_angle((1, 0, 0), roll))


def to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def rotate(q, v):
    """v (world) seen in the body frame of q is to_R(q).T @ v; this returns to_R(q) @ v (body to world)."""
    return to_R(q) @ np.asarray(v, float)


def angle_between(a, b) -> float:
    """The rotation angle [rad] between two orientations."""
    d = abs(float(np.dot(normalize(a), normalize(b))))
    return 2.0 * np.arccos(min(1.0, d))


def angular_velocity(q0, q1, dt):
    """Body-frame angular velocity [rad/s] that turns q0 into q1 in dt."""
    dq = mul(conj(q0), q1)
    if dq[0] < 0:
        dq = -dq
    s = np.linalg.norm(dq[1:])
    if s < 1e-12:
        return np.zeros(3)
    ang = 2.0 * np.arctan2(s, dq[0])
    return dq[1:] / s * ang / dt
