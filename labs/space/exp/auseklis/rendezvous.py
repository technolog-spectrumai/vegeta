"""Approach gates (10 km, 1 km, 100 m, 10 m): navigation uncertainty, speed limits, passive safety and collision probability.

The navigation 1-sigma values are ASSUMED sensor-class numbers (differential GNSS far away, angles-only / LiDAR close);
they are inputs to be replaced by sensor datasheets and an EKF covariance study. The ``SafetyMonitor`` is deliberately
independent of guidance: it sees only a second range sensor and a table of limits, and can only command ``abort``."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import physics as ph
from .provenance import assumed


@dataclass(frozen=True)
class Gate:
    range_m: float
    sensor: str
    sigma_pos: float             # m (1-sigma, per axis)
    sigma_vel: float             # m/s
    v_max: float                 # m/s, conservative closing-speed limit
    keep_out: float              # m, hard abort radius around the target (centre-to-centre) for this gate's rules
    horizon_s: float = 5400.0    # s, how long the free drift must stay outside keep_out after a loss of control (ASSUMED)


GATES = [Gate(10_000, "relative GNSS / ground-aided + wide camera (angles-only)", float(assumed(60.0, "m", "", "sig_10km")), 0.05, 5.0, 5000.0, 5400.0),
         Gate(1_000, "narrow camera + angles-only range estimate / LiDAR acquisition", float(assumed(8.0, "m", "", "sig_1km")), 0.02, 0.5, 500.0, 5400.0),
         Gate(100, "LiDAR + camera pose estimation", float(assumed(0.6, "m", "", "sig_100m")), 0.005, 0.1, 30.0, 1800.0),
         Gate(10, "LiDAR + camera + proximity sensors", float(assumed(0.03, "m", "", "sig_10m")), 0.001, 0.02, 3.0, 300.0)]


def pc_2d(miss: float, sigma: float, hbr: float, n: int = 400) -> float:
    """Probability that the true separation is inside the combined hard-body radius ``hbr`` when the nominal miss distance
    is ``miss`` and the relative-position error is isotropic Gaussian ``sigma`` (encounter plane): numerical integral."""
    xs = np.linspace(-hbr, hbr, n)
    X, Y = np.meshgrid(xs, xs)
    inside = X ** 2 + Y ** 2 <= hbr ** 2
    pdf = np.exp(-((X - miss) ** 2 + Y ** 2) / (2 * sigma ** 2)) / (2 * math.pi * sigma ** 2)
    return float(pdf[inside].sum() * (xs[1] - xs[0]) ** 2)


def passive_safety(range_m: float, sigma_vel: float, alt_km: float = 550.0, horizon_s: float = 5400.0, samples: int = 300,
                   seed: int = 1, radial_offset: float = 0.0) -> dict:
    """Thrusters off at ``range_m`` behind the target: Monte-Carlo CW drift with a velocity error of ``sigma_vel`` per axis.
    Returns the distribution of the minimum separation over ``horizon_s``. Sampling is deterministic (``seed``)."""
    n = ph.mean_motion(ph.radius(alt_km))
    rng = np.random.default_rng(seed)
    ts = np.linspace(0, horizon_s, 400)
    mins = []
    for _ in range(samples):
        x0 = np.array([radial_offset, -range_m, 0.0, *rng.normal(0, sigma_vel, 3)])
        mins.append(min(np.linalg.norm(ph.cw_propagate(x0, n, t)[:3]) for t in ts[::8]))
    mins = np.array(mins)
    return {"p01_min_sep": float(np.percentile(mins, 1)), "median_min_sep": float(np.median(mins)), "worst": float(mins.min())}


def gate_table(hbr: float = 3.5) -> list[dict]:
    """Per gate: speed limit, time to go at the limit, 3-sigma position error vs. keep-out, passive-safety result, Pc."""
    out = []
    for g in GATES:
        ps = passive_safety(g.range_m, g.sigma_vel, horizon_s=g.horizon_s, samples=60)
        out.append({"range_m": g.range_m, "sensor": g.sensor, "sigma_pos_m": g.sigma_pos, "v_max_m_s": g.v_max,
                    "time_to_go_min_at_vmax": g.range_m / g.v_max / 60, "three_sigma_over_range": 3 * g.sigma_pos / g.range_m,
                    "keep_out_m": g.keep_out, "horizon_min": g.horizon_s / 60, "passive_p01_min_sep_m": ps["p01_min_sep"],
                    "passively_safe": ps["p01_min_sep"] > g.keep_out,
                    "pc_on_nominal_collision_course": pc_2d(0.0, max(g.sigma_pos, 1e-3), min(hbr, 10 * g.sigma_pos + hbr))})
    return out


class SafetyMonitor:
    """Independent last-line check. ``check(range_m, closing_speed, lateral_offset)`` -> 'ok' | 'abort'.
    Limits: speed <= ``v_limit(range)``, and while closing inside 100 m the lateral offset must stay inside the corridor."""

    def __init__(self, corridor_half_angle_deg: float = 10.0):
        self.tan = math.tan(math.radians(corridor_half_angle_deg))

    @staticmethod
    def v_limit(range_m: float) -> float:
        pts = sorted((g.range_m, g.v_max) for g in GATES)
        r = [p[0] for p in pts]
        v = [p[1] for p in pts]
        return float(np.exp(np.interp(math.log(max(range_m, 1.0)), np.log(r), np.log(v))))

    def check(self, range_m: float, closing_speed: float, lateral_offset: float = 0.0) -> str:
        if closing_speed > self.v_limit(range_m) * 1.2:
            return "abort"
        if range_m < 100 and closing_speed > 0 and abs(lateral_offset) > self.tan * range_m + 0.2:
            return "abort"
        return "ok"
