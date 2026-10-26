"""Rotating reference frame: the solver runs in a frame turning at ``omega`` about an axis parallel to x.

Geometry that rotates with the frame (a propeller's blades and hub) is then static, and an isolated propeller in
uniform inflow becomes a steady problem. The fluid feels two fictitious forces per unit volume,

    F = ρ [−2 Ω×u − Ω×(Ω×r)],   Ω = (ω, 0, 0),   r = (0, y − y_c, z − z_c),

Coriolis and centrifugal, added through Guo's forcing. The Coriolis force depends on the velocity it changes;
the half-force velocity ``u = (Σ f c + F/2)/ρ`` is therefore solved exactly (a 2×2 linear system per node).

Conventions:
* ``Simulation.velocity()`` is the velocity **relative to the rotating frame**; ``absolute_velocity()`` adds Ω×r.
* Velocities *given* to the simulation — initial fields, inlet and wall velocities on faces — are **absolute**
  (lab-frame) velocities; the boundaries convert them per node: ``u_rel = u_abs − Ω×r``, evaluated at the wall's
  halfway position.
* Pressure is the same in both frames, so a pressure outlet stays uniform.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .errors import InvalidParameterError


@dataclass(frozen=True)
class RotatingFrame:
    """A frame rotating at ``omega`` [rad per time step] about the axis ``(·, center[0], center[1])`` along +x
    (right-hand rule: positive ``omega`` turns +y towards +z). ``center`` is in node coordinates."""

    omega: float
    center: tuple[float, float] = (0.0, 0.0)
    axis: str = "x"

    def __post_init__(self) -> None:
        if self.axis != "x":
            raise InvalidParameterError("only rotation about the x axis is implemented (put the propeller axis on x)")
        if not math.isfinite(self.omega):
            raise InvalidParameterError(f"omega must be finite, got {self.omega!r}")
        object.__setattr__(self, "center", (float(self.center[0]), float(self.center[1])))

    def radial(self, shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
        """``y − y_c`` and ``z − z_c`` of the nodes, broadcastable to ``shape``."""
        y = (np.arange(shape[1], dtype=np.float64) - self.center[0]).reshape(1, -1, 1)
        z = (np.arange(shape[2], dtype=np.float64) - self.center[1]).reshape(1, 1, -1)
        return y, z

    def frame_velocity(self, shape: tuple[int, int, int]) -> np.ndarray:
        """Ω×r at the nodes, ``[3, nx, ny, nz]``."""
        y, z = self.radial(shape)
        out = np.zeros((3,) + shape)
        out[1] = -self.omega * z
        out[2] = self.omega * y
        return out

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"center": list(self.center)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RotatingFrame":
        return cls(omega=d["omega"], center=tuple(d["center"]), axis=d.get("axis", "x"))


def solve_velocity(b: np.ndarray, omega: float, sign: float) -> np.ndarray:
    """Solve ``u + sign·(Ω×u) = b`` for u, with Ω = (ω, 0, 0) (``b[3, ...]``).

    Pre-collision (``sign = +1``): ``b = (Σ f c + F_u/2)/ρ + ω² r/2``; post-collision state (``sign = −1``):
    ``b = (Σ f* c − F_u/2)/ρ − ω² r/2``, with ``F_u`` the uniform body force and ``r`` radial.
    """
    s = sign * omega
    d = 1.0 + omega * omega
    u = np.empty_like(b)
    u[0] = b[0]
    u[1] = (b[1] + s * b[2]) / d
    u[2] = (b[2] - s * b[1]) / d
    return u


def fictitious_force(rho: np.ndarray, u: np.ndarray, omega: float, ry: np.ndarray, rz: np.ndarray) -> np.ndarray:
    """``ρ [−2 Ω×u − Ω×(Ω×r)]`` per node, ``[3, ...]``."""
    out = np.zeros_like(u)
    out[1] = rho * (omega * omega * ry + 2.0 * omega * u[2])
    out[2] = rho * (omega * omega * rz - 2.0 * omega * u[1])
    return out
