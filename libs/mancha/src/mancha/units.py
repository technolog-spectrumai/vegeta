"""Physical ↔ lattice units. The solver works only in lattice units (dx = dt = 1, reference density 1);
:class:`UnitConverter` is the one place physical units enter or leave.

A converter is fixed by the cell size ``dx`` [m], the time step ``dt`` [s] and the physical reference density
``rho`` [kg/m³]. Two ways to choose ``dt``:

* :meth:`UnitConverter.from_velocity` — fix the lattice velocity of a reference speed (Mach-number control,
  ``u_lattice ≲ 0.1`` keeps compressibility errors ≈ Ma² below ~0.3 %);
* :meth:`UnitConverter.from_viscosity` — fix the lattice viscosity (relaxation-time control).

Both keep the Reynolds number: ``Re = U L / nu`` is the same in either system.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .errors import InvalidParameterError

CS2 = 1.0 / 3.0


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not (math.isfinite(value) and value > 0):
        raise InvalidParameterError(f"{name} must be a positive number, got {value!r}")
    return value


@dataclass(frozen=True)
class UnitConverter:
    dx: float          # cell size [m]
    dt: float          # time step [s]
    rho: float = 1.0   # physical density represented by lattice density 1 [kg/m³]

    def __post_init__(self) -> None:
        for name in ("dx", "dt", "rho"):
            _positive(name, getattr(self, name))

    @classmethod
    def from_velocity(cls, dx: float, velocity: float, lattice_velocity: float, rho: float = 1.0) -> "UnitConverter":
        """``dt`` such that the physical ``velocity`` [m/s] is ``lattice_velocity`` in lattice units."""
        dx, velocity, lattice_velocity = (_positive(n, v) for n, v in
                                          (("dx", dx), ("velocity", velocity), ("lattice_velocity", lattice_velocity)))
        return cls(dx, lattice_velocity * dx / velocity, rho)

    @classmethod
    def from_viscosity(cls, dx: float, viscosity: float, lattice_viscosity: float,
                       rho: float = 1.0) -> "UnitConverter":
        """``dt`` such that the physical kinematic ``viscosity`` [m²/s] is ``lattice_viscosity``."""
        dx, viscosity, lattice_viscosity = (_positive(n, v) for n, v in
                                            (("dx", dx), ("viscosity", viscosity),
                                             ("lattice_viscosity", lattice_viscosity)))
        return cls(dx, lattice_viscosity * dx * dx / viscosity, rho)

    # lattice -> physical
    def length(self, n: float) -> float:
        return n * self.dx

    def time(self, steps: float) -> float:
        return steps * self.dt

    def velocity(self, u_lattice):
        return u_lattice * (self.dx / self.dt)

    def viscosity(self, nu_lattice: float) -> float:
        return nu_lattice * self.dx * self.dx / self.dt

    def pressure(self, rho_lattice, rho_ref: float = 1.0):
        """Gauge pressure [Pa] of a lattice density, ``p − p_ref = cs² (ρ − ρ_ref) ρ_phys (dx/dt)²``."""
        return CS2 * (rho_lattice - rho_ref) * self.rho * (self.dx / self.dt) ** 2

    def force(self, f_lattice):
        """Force [N] from a lattice force (momentum per step): ``F = F_lu ρ dx⁴ / dt²``."""
        return f_lattice * self.rho * self.dx ** 4 / self.dt ** 2

    # physical -> lattice
    def to_lattice_length(self, length: float) -> float:
        return length / self.dx

    def to_lattice_time(self, t: float) -> float:
        return t / self.dt

    def to_lattice_velocity(self, u):
        return u * (self.dt / self.dx)

    def to_lattice_viscosity(self, nu: float) -> float:
        return nu * self.dt / (self.dx * self.dx)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def reynolds(velocity: float, length: float, viscosity: float) -> float:
    """``Re = U L / nu`` (any consistent unit system)."""
    return velocity * length / viscosity


def mach(lattice_velocity: float) -> float:
    """Lattice Mach number ``u / cs`` with ``cs = 1/√3``."""
    return lattice_velocity / math.sqrt(CS2)
