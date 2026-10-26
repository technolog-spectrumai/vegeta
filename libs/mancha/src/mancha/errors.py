"""Mancha's exceptions: invalid parameters and numerical failures are raised, never silenced."""


class ManchaError(Exception):
    """Base class of Mancha's errors."""


class InvalidParameterError(ManchaError, ValueError):
    """A parameter that cannot give a valid simulation (tau <= 1/2, a non-tangential wall velocity, ...)."""


class NumericalInstabilityError(ManchaError, FloatingPointError):
    """The solution became non-finite or exceeded the lattice speed of sound."""


class InsufficientMemoryError(ManchaError, MemoryError):
    """The requested grid does not fit in the device memory."""


class BackendUnavailableError(ManchaError, RuntimeError):
    """The requested backend cannot run here (no Triton, no CUDA and no interpreter, ...)."""
