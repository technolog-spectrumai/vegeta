"""The servo law: a PD position loop whose torque is clipped to the DC-motor torque–speed line.

    τ_pd  = kp (q* − q) + kd (q̇* − q̇) + τ_ff
    τ_max = τ_stall · max(0, 1 − |q̇| / ω₀)          (the torque–speed line; 0 beyond the no-load speed ω₀)
    τ     = clip(τ_pd, −τ_max, +τ_max)

The limit is the motoring line of a DC motor at its supply voltage, applied symmetrically (the same bound
whether the torque drives or brakes the joint), as fixed by the pre-registered locomotion-stability
protocol in docs/ (§1). Pure numpy, vectorised over joints; ``out`` lets a simulation loop reuse its buffers.
``implicit_slope`` gives the law's velocity derivative for an implicit integrator (the rule ``ChironLab`` applies,
inlined, at every physics step).
"""
from __future__ import annotations

import numpy as np


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
