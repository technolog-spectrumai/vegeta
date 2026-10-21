"""Rotations and frame conventions.

Body frame (Chiron, MuJoCo, Gazebo): x forward, y left, z up. Aero frame (flight mechanics): x forward, y right,
z down. Quaternions are (w, x, y, z)."""
import math

import numpy as np

AERO_FROM_BODY = np.diag([1.0, -1.0, -1.0])      # x fwd y left z up  <->  x fwd y right z down (its own inverse)


def quat_to_R(q) -> np.ndarray:
    """The rotation matrix of a unit quaternion (w, x, y, z): body → world."""
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def euler_from_R(R) -> tuple:
    """(roll right-wing-down, pitch nose-up, heading from +x towards +y) [rad] of a body frame (x fwd, y left, z up)."""
    pitch = math.asin(max(-1.0, min(1.0, R[2, 0])))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def yaw_quat(yaw: float) -> tuple:
    """The quaternion of a rotation by ``yaw`` [rad] about z."""
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
