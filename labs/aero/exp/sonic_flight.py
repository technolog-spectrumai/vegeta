"""Vertical flight of what leaves the blowdown pipe (notebook 51): kinetic energy at the muzzle and along the next 50 m
of climb, under gravity and air drag.

m dv/dt = -m g - ½ ρ v² Cd(M) A, with v up and dh = v dt. The air is still at sea level (ρ 1.2 kg/m³, a 343 m/s).

Drag coefficients on the frontal area, against Mach. They are rough engineering values, not measured here:
- **sphere**: 0.47 subcritical, rising through the transonic range to ~0.95 at Mach 1 (sphere drag curves, e.g.
  Hoerner, *Fluid-Dynamic Drag*, ch. 16; Miller & Bailey 1979).
- **cylinder + head, nose first**: forebody + blunt base + skin friction. Base drag 0.12 subsonic, rising to ~0.25 at
  Mach 1. Hemisphere forebody ~0.02 subsonic, paraboloid ~0.01, both rising past Mach 0.8 (the hemisphere more).
  Skin friction 0.004 per unit of wetted / frontal area (4 L/D). Hoerner chs. 3, 13, 16 for the orders of magnitude.
- **tumbling**: a long body without fins is statically unstable nose first (uniform rubber has its centre of mass
  mid-body). The worst case is broadside: Cd 1.2 on the side area D·L, averaged with the nose-first value.

The point is a comparison between shapes, and with the ±30 % these values carry the ranking is what to trust.
"""
from __future__ import annotations

import math

import numpy as np

G, RHO, A_SOUND = 9.81, 1.2, 343.0


def _ramp(M, sub, sonic, m0=0.6, m1=1.0):
    """sub below m0, sonic from m1, smooth (cosine) in between."""
    s = np.clip((M - m0) / (m1 - m0), 0.0, 1.0)
    return sub + (sonic - sub) * 0.5 * (1 - np.cos(np.pi * s))


def cd_sphere(M):
    return _ramp(M, 0.47, 0.95, 0.5, 1.0)


def cd_body(M, head: str, l_over_d: float):
    nose = _ramp(M, 0.02, 0.25, 0.8, 1.1) if head == "hemisphere" else _ramp(M, 0.01, 0.12, 0.85, 1.1)
    base = _ramp(M, 0.12, 0.25, 0.6, 1.0)
    return nose + base + 0.004 * 4 * l_over_d


def drag_area(ball, M, tumbling: bool = False) -> float:
    """Cd·A [m²] of a ``sonic_ball.Ball`` at Mach M."""
    D = ball.diameter * 1e-3
    A = math.pi / 4 * D * D
    if ball.shape == "sphere":
        return cd_sphere(M) * A
    cda = cd_body(M, ball.shape, ball.L / ball.diameter) * A
    if tumbling:
        cda = 0.5 * (cda + 1.2 * D * ball.L * 1e-3)
    return cda


def climb(ball, v0: float, height: float = 50.0, tumbling: bool = False, dt: float = 1e-5) -> dict:
    """Vertical climb from the muzzle at v0 [m/s]: h, v, kinetic energy, time; stops at ``height`` or the apex."""
    m = ball.mass
    h, v, t = 0.0, v0, 0.0
    rows = [(t, h, v)]
    while v > 0 and h < height:
        def acc(v):
            return -G - 0.5 * RHO * v * v * drag_area(ball, v / A_SOUND, tumbling) / m
        k1 = acc(v); k2 = acc(v + 0.5 * dt * k1); k3 = acc(v + 0.5 * dt * k2); k4 = acc(v + dt * k3)
        dv = dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        h += dt * (v + 0.5 * dv)
        v += dv
        t += dt
        rows.append((t, h, v))
    a = np.array(rows)
    out = {"t": a[:, 0], "h": a[:, 1], "v": np.maximum(a[:, 2], 0.0)}
    out["E"] = 0.5 * m * out["v"] ** 2
    out["reached"] = out["h"][-1] >= height - 1e-6
    out["apex"] = None if out["reached"] else out["h"][-1]
    return out


def energy_at(fl: dict, heights) -> np.ndarray:
    """Kinetic energy [J] at the given heights; 0 past the apex."""
    return np.array([np.interp(z, fl["h"], fl["E"]) if z <= fl["h"][-1] else 0.0 for z in heights])
