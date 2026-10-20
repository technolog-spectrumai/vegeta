"""Electric motors, the servo law and transmissions: ``og.motors``.

- **DC motor** (``DCMotor``, alias ``Motor``): the usual first-order brushless model — Kv, winding resistance,
  no-load current (Boreas's ``Motor``) — with ``fit_resistance`` (R from a published full-throttle point, ESC and
  wires included, as Nisus+/Falco fit theirs) and ``operating_point`` (the rpm where the motor's torque meets a
  load's, by bisection: Falco's ``_static_point`` and the frame study's inner loop).
- **The servo law** (Chiron's ``servo``): a PD position loop clipped to the DC-motor torque–speed line,
  ``torque_limit``, ``servo_torque``, ``implicit_slope``, ``saturation``.
- **Transmissions:** ``lever`` (a linear actuator on an arm → joint torque and speed) and ``gearbox`` (a ratio and
  an efficiency), the conversions the Onager, Pekari and Perseus robots did by hand.

Characterisation: ``tests/test_motors.py`` (``tests/golden/motors.json``, recorded from Vegeta and Chiron).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------------------------------------- DC motor
@dataclass(frozen=True)
class DCMotor:
    """Brushless DC motor as the usual first-order model: Kv, winding resistance, no-load current (Boreas's Motor)."""
    name: str
    kv_rpm_per_volt: float
    resistance_ohm: float
    no_load_current_a: float
    max_current_a: float
    mass_kg: float = 0.0
    source: str = "datasheet-style values"

    def __post_init__(self):
        if min(self.kv_rpm_per_volt, self.resistance_ohm, self.max_current_a) <= 0 or self.no_load_current_a < 0:
            raise ValueError("kv, resistance and max current must be > 0; no-load current >= 0")

    @property
    def kt(self) -> float:
        """Torque constant [N m / A] = 1 / Kv in rad/s per volt."""
        return 60 / (2 * math.pi * self.kv_rpm_per_volt)

    def voltage_for(self, rpm: float, torque: float) -> tuple[float, float]:
        """``(voltage, current)`` needed to hold ``rpm`` against a shaft ``torque`` [N m]."""
        current = self.no_load_current_a + torque / self.kt
        return rpm / self.kv_rpm_per_volt + current * self.resistance_ohm, current

    def efficiency(self, rpm: float, torque: float) -> float:
        v, i = self.voltage_for(rpm, torque)
        return (torque * rpm * 2 * math.pi / 60) / (v * i) if v * i > 0 else 0.0


Motor = DCMotor


def fit_resistance(voltage: float, rpm: float, kv: float, current: float) -> float:
    """The resistance that puts a motor of ``kv`` at a published point: R = (V − rpm/Kv) / I (whatever sits between
    the battery and the back-EMF — ESC, wires — is included)."""
    return (voltage - rpm / kv) / current


def operating_point(kv: float, R: float, I0: float, V: float, load_torque, *, lo: float = 300.0, iters: int = 60) -> tuple:
    """The rpm where a motor (``kv``, ``R``, ``I0``) at ``V`` volts full throttle meets ``load_torque(rpm)`` [N m]
    (e.g. a propeller's), by bisection between ``lo`` and the no-load rpm Kv·V; returns (rpm, current)."""
    kt = 60 / (2 * math.pi * kv)
    hi = kv * V
    for _ in range(iters):
        n = 0.5 * (lo + hi)
        Im = (V - n / kv) / R
        Qm = kt * (Im - I0)
        Qp = load_torque(n)
        lo, hi = (n, hi) if Qm > Qp else (lo, n)
    return n, (V - n / kv) / R


# ------------------------------------------------------------------------------------------------------- servo law
def torque_limit(qd, stall, no_load_speed, out=None):
    """Speed-dependent torque bound ``τ_stall · max(0, 1 − |q̇|/ω₀)`` [N·m] (arrays broadcast)."""
    qd = np.asarray(qd, dtype=float)
    stall = np.asarray(stall, dtype=float)
    w0 = np.asarray(no_load_speed, dtype=float)
    if out is None:
        return stall * np.maximum(0.0, 1.0 - np.abs(qd) / w0)
    np.abs(qd, out=out)
    np.divide(out, w0, out=out)
    np.subtract(1.0, out, out=out)
    np.maximum(out, 0.0, out=out)
    np.multiply(out, stall, out=out)
    return out


def servo_torque(q, qd, q_target, qd_target=0.0, tau_ff=0.0, *, kp, kd, stall, no_load_speed, out=None,
                 work=None, four_quadrant=False):
    """Servo output torque [N·m] for joint angles ``q`` [rad] and speeds ``qd`` [rad/s].

    ``q_target``/``qd_target`` are the commanded angle and speed, ``tau_ff`` a feed-forward torque; ``kp``
    [N·m/rad], ``kd`` [N·m·s/rad], ``stall`` [N·m] and ``no_load_speed`` [rad/s] are the servo's data (scalars
    or one value per joint). Returns ``clip(τ_pd, ±τ_stall·max(0, 1 − |q̇|/ω₀))``: the torque never exceeds the
    torque–speed line and is zero at or beyond the no-load speed.

    ``out`` and ``work`` (float arrays of the result's shape) avoid allocations in a simulation loop.
    """
    if out is None or np.any(four_quadrant):
        q = np.asarray(q, dtype=float)
        qd = np.asarray(qd, dtype=float)
        tau = kp * (np.asarray(q_target, dtype=float) - q) + kd * (np.asarray(qd_target, dtype=float) - qd) + tau_ff
        lim = torque_limit(qd, stall, no_load_speed)
        if np.any(four_quadrant):                     # braking: the full stall torque at any speed
            lim = np.where(np.asarray(four_quadrant, dtype=bool) & (tau * qd < 0),
                           np.broadcast_to(np.asarray(stall, dtype=float), np.shape(lim)), lim)
        res = np.clip(tau, -lim, lim)
        if out is not None:
            out[...] = res
            return out
        return res
    if work is None:
        work = np.empty_like(out)
    np.subtract(q_target, q, out=out)
    np.multiply(out, kp, out=out)
    np.subtract(qd_target, qd, out=work)
    np.multiply(work, kd, out=work)
    np.add(out, work, out=out)
    np.add(out, tau_ff, out=out)
    torque_limit(qd, stall, no_load_speed, out=work)
    np.minimum(out, work, out=out)
    np.negative(work, out=work)
    np.maximum(out, work, out=out)
    return out


def implicit_slope(tau, qd, clipped, kd, stall, no_load_speed, step_dv=None, out=None):
    """The servo torque's velocity derivative ``b = ∂τ/∂q̇`` [N·m·s/rad] for an implicit integrator (``b ≤ 0``).

    ``tau`` [N·m] is the servo torque (already clipped), ``qd`` [rad/s] the joint speed, ``clipped`` (bool) marks
    the joints whose PD torque exceeded the torque–speed line, ``kd`` [N·m·s/rad], ``stall`` [N·m] and
    ``no_load_speed`` ω₀ [rad/s] the servo data (scalars or one value per joint). ``step_dv`` [rad/s] (optional):
    the speed change the torque alone gives over one time step, ``h·|τ|/I`` (I the joint's inertia [kg·m²]).

    An integrator that is implicit in the velocity (MuJoCo's ``implicitfast``) solves
    ``(M − h·b) Δq̇ = h·f`` with the torque linearised as ``τ + b·Δq̇``; ``b`` must be the slope of the regime the
    joint is in, or a saturated joint is integrated with a damping it does not have:

    * unclipped: ``−kd`` (the PD law);
    * clipped and pulling (``τ·q̇ ≥ 0``, ``|q̇| < ω₀``): ``−τ_stall/ω₀``, the slope of the torque–speed line;
    * clipped at or beyond the no-load speed: 0 (``τ ≡ 0``);
    * clipped and braking (``τ·q̇ < 0``): the line's slope there is ``+τ_stall/ω₀`` (less speed, more torque), which
      an implicit step cannot take (``M − h·b`` would lose definiteness): 0, i.e. explicit — unless the torque stops
      the joint within the step (``step_dv ≥ |q̇|``); the joint then ends the step past zero speed, pulling on the
      line, and gets the line's slope ``−τ_stall/ω₀``.

    ``out`` (a float array of the result's shape) is filled and returned.
    """
    tau = np.asarray(tau, dtype=float)
    qd = np.asarray(qd, dtype=float)
    clipped = np.asarray(clipped, dtype=bool)
    w0 = np.asarray(no_load_speed, dtype=float)
    if out is None:
        out = np.empty(np.broadcast_shapes(tau.shape, qd.shape, clipped.shape))
    np.negative(np.broadcast_to(np.asarray(kd, dtype=float), out.shape), out=out)
    np.copyto(out, 0.0, where=clipped)
    speed = np.abs(qd)
    line = clipped & (speed < w0)
    pulling = tau * qd >= 0.0
    if step_dv is not None:
        pulling |= np.asarray(step_dv, dtype=float) >= speed
    line &= pulling
    np.copyto(out, -np.asarray(stall, dtype=float) / w0, where=line)
    return out


def saturation(tau, qd, stall, no_load_speed, fraction=0.98):
    """Boolean arrays (torque_saturated, speed_saturated): ``|τ| ≥ fraction · τ_max(q̇)`` and ``|q̇| ≥ fraction · ω₀``
    (the protocol's saturation definitions, §6 actuator demand)."""
    lim = torque_limit(qd, stall, no_load_speed)
    tau = np.abs(np.asarray(tau, dtype=float))
    torque_sat = (tau >= fraction * lim) & (lim > 0) | (lim <= 0)
    speed_sat = np.abs(np.asarray(qd, dtype=float)) >= fraction * np.asarray(no_load_speed, dtype=float)
    return torque_sat, speed_sat


# --------------------------------------------------------------------------------------------------- transmissions
def lever(force, speed, arm):
    """A linear actuator (stall or rated ``force`` [N], no-load ``speed`` [m/s]) on an ``arm`` [m] → the joint's
    (torque [N m], angular speed [rad/s]) = (F·arm, v/arm)."""
    return force * arm, speed / arm


def gearbox(torque, speed, ratio, efficiency: float = 1.0):
    """A motor's (``torque`` [N m], ``speed`` [rad/s]) through a reduction ``ratio`` (> 1 slows) with ``efficiency``
    → the output's (torque·ratio·η, speed/ratio)."""
    return torque * ratio * efficiency, speed / ratio


__all__ = ["DCMotor", "Motor", "fit_resistance", "operating_point", "torque_limit", "servo_torque", "implicit_slope",
           "saturation", "lever", "gearbox"]
