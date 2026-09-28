"""A timed parachute landing and what the payload feels: opening shock, descent in wind, touchdown.

``simulate_drop`` flies a point mass in the vertical plane from the moment the motor is cut: a ballistic fall
(the drone's own drag area) until the timer fires, the canopy inflating over its fill time (the opening shock
is the peak drag force during inflation), the steady descent in a mean wind with gusts (a first-order coloured
noise, seeded), and the touchdown onto a crush pad that stops the payload over a stroke. Everything is explicit
input; the output is the time history and the peaks the medicine sees. Pure numpy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

G = 9.81


@dataclass(frozen=True)
class Parachute:
    """A round canopy: nominal area, drag coefficient on that area, inflation (fill) time; ``area_m2`` = 0 is no canopy."""

    area_m2: float
    cd: float = 0.75
    fill_time_s: float = 1.2
    mass_kg: float = 0.0

    def terminal_speed(self, mass_kg: float, rho: float = 1.225) -> float:
        """Steady descent rate under the full canopy."""
        if self.area_m2 <= 0:
            return float("inf")
        return math.sqrt(2 * mass_kg * G / (rho * self.cd * self.area_m2))

    def area_for_descent(self, mass_kg: float, descent_m_s: float, rho: float = 1.225) -> float:
        """Canopy area that gives ``descent_m_s`` for ``mass_kg``."""
        return 2 * mass_kg * G / (rho * self.cd * descent_m_s ** 2)


@dataclass(frozen=True)
class Body:
    """The falling machine: mass and its drag area ``cd_a_m2`` without the canopy (a tumbling airframe)."""

    mass_kg: float
    cd_a_m2: float


@dataclass(frozen=True)
class Wind:
    """Mean horizontal wind (+x, m/s) and gusts: rms of a first-order coloured noise with a time scale."""

    mean_m_s: float = 0.0
    gust_rms_m_s: float = 0.0
    gust_time_s: float = 2.0
    vertical_rms_m_s: float = 0.0


@dataclass
class DropResult:
    t: np.ndarray
    x: np.ndarray
    z: np.ndarray
    vx: np.ndarray
    vz: np.ndarray
    g: np.ndarray                    # acceleration felt (drag / weight), in g
    phase: np.ndarray                # 0 fall, 1 inflating, 2 descent
    speed: np.ndarray                # |v| over the ground, m/s
    descent_speed: np.ndarray        # -vz: positive going down, m/s
    opening_g: float
    opening_time_s: float
    opening_altitude_m: float
    descent_rate_m_s: float
    touchdown_speed_m_s: float
    touchdown_vertical_m_s: float
    touchdown_g: float
    touchdown_pulse: tuple           # (peak acceleration m/s^2, duration s) of the half-sine crush
    drift_m: float
    duration_s: float
    landed: bool

    def summary(self) -> dict:
        return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in self.__dict__.items()
                if not isinstance(v, np.ndarray)}

    def peaks(self) -> dict:
        return {"opening_g": self.opening_g, "touchdown_g": self.touchdown_g, "descent_rate_m_s": self.descent_rate_m_s,
                "drift_m": self.drift_m, "duration_s": self.duration_s}


def gusts(n: int, dt: float, rms: float, time_s: float, rng) -> np.ndarray:
    """First-order (Ornstein-Uhlenbeck) gust series with the given rms and correlation time."""
    if rms <= 0:
        return np.zeros(n)
    a = math.exp(-dt / max(time_s, dt))
    out = np.empty(n)
    out[0] = rng.normal(0.0, rms)
    s = rms * math.sqrt(1 - a * a)
    for i in range(1, n):
        out[i] = a * out[i - 1] + rng.normal(0.0, s)
    return out


def crush_pulse(v_td: float, stroke_m: float) -> tuple[float, float, float]:
    """A half-sine stop of ``v_td`` over ``stroke_m``: ``(peak m/s^2, duration s, peak g)``."""
    if stroke_m <= 0 or v_td <= 0:
        return float("inf") if v_td > 0 else 0.0, 0.0, float("inf") if v_td > 0 else 0.0
    tau = 2 * stroke_m / v_td
    a_p = math.pi * v_td * v_td / (4 * stroke_m)
    return a_p, tau, a_p / G


def simulate_drop(body: Body, chute: Parachute, *, altitude_m: float, speed_m_s: float, timer_s: float,
                  wind: Wind = Wind(), crush_stroke_m: float = 0.05, rho: float = 1.225, dt: float = 0.01,
                  t_max_s: float = 600.0, seed: int = 0, glide_ratio: float = 0.0) -> DropResult:
    """Motor cut at ``altitude_m`` flying at ``speed_m_s`` (+x); the timer fires the canopy at ``timer_s``.

    ``glide_ratio``: lift / drag of the unpowered airframe while it still flies straight before the canopy
    opens (0 = it tumbles: drag only). ``crush_stroke_m``: the pad or legs that stop the payload at touchdown."""
    rng = np.random.default_rng(seed)
    n = int(t_max_s / dt) + 1
    gx = gusts(n, dt, wind.gust_rms_m_s, wind.gust_time_s, rng)
    gz = gusts(n, dt, wind.vertical_rms_m_s, wind.gust_time_s, np.random.default_rng(seed + 1))
    m = body.mass_kg + chute.mass_kg
    x, z, vx, vz = 0.0, altitude_m, speed_m_s, 0.0
    T = np.zeros(n); X = np.zeros(n); Z = np.zeros(n); VX = np.zeros(n); VZ = np.zeros(n); GG = np.zeros(n); PH = np.zeros(n, int)
    opening_g = 0.0; opening_t = float("nan"); opening_alt = float("nan")
    landed = False
    k = 0
    for k in range(n):
        t = k * dt
        wx, wz = wind.mean_m_s + gx[k], gz[k]
        rx, rz = vx - wx, vz - wz
        vr = math.hypot(rx, rz)
        if t < timer_s or chute.area_m2 <= 0:
            phase, area = 0, 0.0
        elif t < timer_s + chute.fill_time_s:
            phase, area = 1, chute.area_m2 * ((t - timer_s) / chute.fill_time_s) ** 2   # the canopy fills as the square of time
        else:
            phase, area = 2, chute.area_m2
        cda = body.cd_a_m2 + chute.cd * area
        fd = 0.5 * rho * vr * vr * cda
        fx = -fd * rx / vr if vr > 1e-9 else 0.0
        fz = -fd * rz / vr if vr > 1e-9 else 0.0
        if phase == 0 and glide_ratio > 0 and vr > 1e-9:          # a gliding airframe: lift normal to the relative wind
            fl = glide_ratio * 0.5 * rho * vr * vr * body.cd_a_m2
            fx += -fl * rz / vr
            fz += fl * rx / vr
        ax, az = fx / m, fz / m - G
        T[k], X[k], Z[k], VX[k], VZ[k], PH[k] = t, x, z, vx, vz, phase
        GG[k] = math.hypot(fx, fz) / (m * G)
        if phase in (1, 2) and GG[k] > opening_g and t < timer_s + 3 * chute.fill_time_s:
            opening_g, opening_t, opening_alt = GG[k], t, z
        vx += ax * dt; vz += az * dt
        x += vx * dt; z += vz * dt
        if z <= 0.0:
            landed = True
            break
    k = min(k + 1, n)
    sl = slice(0, k)
    v_td = math.hypot(VX[k - 1], VZ[k - 1]); v_vert = abs(VZ[k - 1])
    a_p, tau, g_td = crush_pulse(v_vert, crush_stroke_m) if landed else (0.0, 0.0, 0.0)
    steady = PH[sl] == 2
    descent = float(-np.median(VZ[sl][steady])) if steady.sum() > 10 else float("nan")
    return DropResult(T[sl], X[sl], Z[sl], VX[sl], VZ[sl], GG[sl], PH[sl], np.hypot(VX[sl], VZ[sl]), -VZ[sl], float(opening_g), float(opening_t), float(opening_alt),
                      descent, v_td, v_vert, g_td, (a_p, tau), float(X[k - 1]), float(T[k - 1]), landed)


def landing_scatter(body: Body, chute: Parachute, *, n: int = 50, seed: int = 0, wind_mean_range=(0.0, 8.0),
                    **kwargs) -> dict:
    """``n`` drops with random wind (uniform mean in ``wind_mean_range``, the gusts of ``kwargs['wind']``) and
    seeds: the landing radius and the peaks' spread. Returns arrays and a summary (95 % radius, max g)."""
    rng = np.random.default_rng(seed)
    base = kwargs.pop("wind", Wind())
    drifts, open_g, td_g, rates = [], [], [], []
    for i in range(n):
        w = Wind(float(rng.uniform(*wind_mean_range)), base.gust_rms_m_s, base.gust_time_s, base.vertical_rms_m_s)
        r = simulate_drop(body, chute, wind=w, seed=seed + 1000 + i, **kwargs)
        drifts.append(r.drift_m); open_g.append(r.opening_g); td_g.append(r.touchdown_g); rates.append(r.descent_rate_m_s)
    drifts, open_g, td_g = np.array(drifts), np.array(open_g), np.array(td_g)
    return {"drift_m": drifts, "opening_g": open_g, "touchdown_g": td_g, "descent_rate_m_s": np.array(rates),
            "drift_p95_m": float(np.percentile(drifts, 95)), "drift_mean_m": float(drifts.mean()),
            "opening_g_max": float(open_g.max()), "touchdown_g_max": float(td_g.max())}
