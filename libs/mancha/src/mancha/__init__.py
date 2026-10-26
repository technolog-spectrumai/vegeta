"""Mancha: a lattice Boltzmann (D3Q19) solver for incompressible flow, with a NumPy reference backend and a
Triton GPU backend. Lattice units throughout; :class:`UnitConverter` converts to and from physical units."""
__version__ = "0.1.0"

from .boundaries import FACES, BoundaryConditions  # noqa: E402
from .collision import BGK, CollisionModel  # noqa: E402
from .errors import (BackendUnavailableError, InsufficientMemoryError, InvalidParameterError,  # noqa: E402
                     ManchaError, NumericalInstabilityError)
from .lattice import D3Q19  # noqa: E402
from .simulation import Record, Simulation  # noqa: E402
from .units import UnitConverter  # noqa: E402

__all__ = ["BGK", "D3Q19", "FACES", "BackendUnavailableError", "BoundaryConditions", "CollisionModel",
           "InsufficientMemoryError", "InvalidParameterError", "ManchaError", "NumericalInstabilityError", "Record",
           "Simulation", "UnitConverter", "__version__"]
