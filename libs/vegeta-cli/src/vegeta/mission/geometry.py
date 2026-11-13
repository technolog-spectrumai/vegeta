"""Small geometric helpers shared by the nodes: intercept time, closest approach."""
from __future__ import annotations

import math

import numpy as np

__all__ = ["intercept_time", "closest_approach", "predict", "intercept_on_track"]


def intercept_time(r, v_target, speed, t_max=60.0):
    """Earliest t with |r + v_target t| = speed t (a straight-line intercept at ``speed``); None if none within t_max.
    ``r``: target minus pursuer position."""
    r, v = np.asarray(r, float), np.asarray(v_target, float)
    a = float(v @ v) - speed ** 2
    b = 2.0 * float(r @ v)
    c = float(r @ r)
    if abs(a) < 1e-9:
        t = -c / b if b < 0 else None
        return t if t is not None and 0 < t <= t_max else None
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    s = math.sqrt(disc)
    roots = sorted(x for x in ((-b - s) / (2 * a), (-b + s) / (2 * a)) if x > 0)
    return roots[0] if roots and roots[0] <= t_max else None


def closest_approach(r, v_rel):
    """(time, distance) of the closest approach for relative position ``r`` (target − own) and relative velocity
    ``v_rel`` (target − own), straight lines; time clipped at 0."""
    r, v = np.asarray(r, float), np.asarray(v_rel, float)
    vv = float(v @ v)
    t = 0.0 if vv < 1e-9 else max(0.0, -float(r @ v) / vv)
    return t, float(np.linalg.norm(r + v * t))


def predict(pos, vel, turn_rate, dt):
    """The position ``dt`` ahead on a coordinated turn of ``turn_rate`` [rad/s] (straight when ~0)."""
    pos, vel = np.asarray(pos, float), np.asarray(vel, float)
    w = float(turn_rate)
    if abs(w) < 0.03:
        return pos + vel * dt
    s, c = math.sin(w * dt), math.cos(w * dt)
    vx, vy = vel[0], vel[1]
    return np.array([pos[0] + (vx * s - vy * (1 - c)) / w, pos[1] + (vx * (1 - c) + vy * s) / w, pos[2] + vel[2] * dt])


def intercept_on_track(own, k, speed, t_max=15.0, turning=True):
    """Intercept time of a track flying its estimated turn (``turning``) or straight, by fixed-point iteration."""
    w = k.turn_rate if (turning and k.p_turn > 0.5) else 0.0
    t = intercept_time(k.pos - own, k.vel, speed, t_max=t_max)
    t = t_max if t is None else t
    for _ in range(8):
        p = predict(k.pos, k.vel, w, t)
        t_new = min(float(np.linalg.norm(p - own)) / speed, t_max)
        if abs(t_new - t) < 0.05:
            break
        t = t_new
    return t, w
