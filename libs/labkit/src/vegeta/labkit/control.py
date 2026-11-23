"""Mission building blocks shared by the labs' controllers (Onager, Pekari, Drongo, Nisus …)."""
from __future__ import annotations

from dataclasses import dataclass


def smooth(u: float) -> float:
    """Smoothstep on [0, 1] (0 below, 1 above)."""
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


@dataclass
class Phase:
    """One step of a mission: ``start(m, obs)`` once, then ``update(m, obs, tau) -> done`` every control step (tau:
    time in the phase); ``timeout`` [s] ends it anyway (logged)."""

    name: str
    update: object
    start: object = None
    timeout: float = 30.0
