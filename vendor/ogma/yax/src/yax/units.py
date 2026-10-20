"""Units and frames: the only conversions a consumer needs to place a Yax design in its own world."""
from __future__ import annotations

LENGTH = {"mm": 1e-3, "m": 1.0}                    # metres per unit
AXES = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}


def to_metres(value, unit: str):
    """A length, or a nested list of lengths, in metres."""
    if isinstance(value, (list, tuple)):
        return [to_metres(v, unit) for v in value]
    return value * LENGTH[unit]


def basis(frame: dict) -> list:
    """Rows ``forward, left, up`` of a frame ``{"up": "+z", "forward": "+x"}``, in that frame's coordinates."""
    up, fwd = AXES[frame["up"]], AXES[frame["forward"]]
    if sum(u * f for u, f in zip(up, fwd)) != 0:
        raise ValueError(f"frame {frame}: up and forward must be perpendicular axes")
    left = (up[1] * fwd[2] - up[2] * fwd[1], up[2] * fwd[0] - up[0] * fwd[2], up[0] * fwd[1] - up[1] * fwd[0])
    return [list(fwd), list(left), list(up)]


def rotation(source: dict, target: dict) -> list:
    """The 3x3 matrix ``R`` with ``p_target = R @ p_source``: the design's forward/up land on the target's forward/up.

    Example: a Z-up CAD frame into a Y-up, forward = +z (glTF) world: ``rotation({"up": "+z", "forward": "+x"},
    {"up": "+y", "forward": "+z"})``."""
    s, t = basis(source), basis(target)
    return [[sum(t[k][i] * s[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def apply(matrix, point) -> list:
    return [sum(matrix[i][j] * point[j] for j in range(3)) for i in range(3)]
