"""The Onager Sentinel's patrol scenario in ChironLab (the Sentinel notebook §7, ``onager/scenarios/onager_patrol.py``): a gravel road
with a speed bump at 3 m/s (11 km/h), then two partial failures of the drive and two responses to compare.

* **road** — ISO-8608-like rough gravel (RMS 15 mm, correlation 0.3 m, from x = 2 m) with a 120 mm speed bump
  1.2 m long at x = 12 m (``gravel_road``);
* **failures** (``FAILURES``) — at 6 s the front-left hub motor loses power (it freewheels), at 9 s the
  rear-right wheel seizes (braked on its motor's torque line);
* **responses** (``RESPONSES``) — ``'drag'``: keep driving, the seized tyre skids; ``'limp'``: the three-wheel limp
  (the hull shifts 0.4 m forward over 1.5 s, the seized wheel lifts 0.1 m, speed down to 1.5 m/s).

``run(lab, response)`` returns the Episode with the controller's events in ``ep.log['events']``; ``timeseries(ep)``
and ``phase_table(ep)`` give the numbers (speed, heading, tilt, corner loads, wheel torques and power per phase).
Mechanical wheel power is Σ |τ ω|; the electrical estimate adds the copper loss (τ / k_t)² R from the hub motor's
stall point (``Servo.torque_constant`` / ``resistance``) — a DC-motor estimate, stated as such.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from vegeta import chiron as ch

import onager_controller as oc
import onager_robot as orb

__all__ = ["ROAD", "BUMP", "V_PATROL", "FAILURES", "RESPONSES", "DURATION", "PHASES", "gravel_road", "controller",
           "run", "timeseries", "phase_table", "tilt_deg", "yaw_deg"]

ROAD = dict(rms=0.015, correlation_length=0.30, start=2.0, seed=3, extent=(-3.0, 45.0, -4.0, 4.0), cell=0.05)
BUMP = dict(x=12.0, height=0.12, length=1.2)
V_PATROL = 3.0                                                  # m/s (11 km/h: a patrol speed on gravel)
FAILURES = [oc.Failure(6.0, "FL", "motor_off"), oc.Failure(9.0, "RR", "seized")]
RESPONSES = {
    "drag": dict(lift_seized=0.0, v_limp=None),
    "limp": dict(lift_seized=0.10, shift_x=0.40, shift_s=1.5, v_limp=1.5),
}
DURATION = 16.0
PHASES = [("nominal", 2.0, 6.0), ("FL motor off", 6.0, 9.0), ("RR seized + response", 10.5, DURATION)]


def gravel_road() -> ch.Custom:
    rough = ch.Rough(ROAD["rms"], ROAD["correlation_length"], start=ROAD["start"], seed=ROAD["seed"],
                     extent=ROAD["extent"], cell=ROAD["cell"])
    x0, h, L = BUMP["x"], BUMP["height"], BUMP["length"]

    def ground(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        bump = h * np.cos(np.pi * (x - x0) / L) ** 2 * (np.abs(x - x0) < L / 2)
        return rough.height(x, y) + bump

    return ch.Custom(ground, name=f"gravel road (RMS {ROAD['rms'] * 1000:.0f} mm) with a {h * 1000:.0f} mm speed bump at x = {x0:g} m")


def controller(response: str, v_target: float = V_PATROL, failures=FAILURES, **kw) -> oc.Drive:
    opts = dict(RESPONSES[response])
    opts.update(kw)
    return oc.Drive(v_target, failures=list(failures), name=f"patrol/{response}", **opts)


def run(lab: ch.ChironLab, response: str, duration: float = DURATION, seed: int = 0, settle: float = 0.5, **kw):
    ctrl = controller(response, **kw)
    ep = lab.run(ctrl, duration=duration, rules=None, settle=settle, seed=seed,
                 info={"controller": ctrl.name, "treatment": response, "v_target": ctrl.v_target})
    ep.log["events"] = list(ctrl.events)
    return ep


def tilt_deg(quat) -> np.ndarray:
    q = np.asarray(quat, dtype=float)
    return np.degrees(np.arccos(np.clip(1 - 2 * (q[..., 1] ** 2 + q[..., 2] ** 2), -1.0, 1.0)))


def yaw_deg(quat) -> np.ndarray:
    q = np.asarray(quat, dtype=float)
    return np.degrees(np.arctan2(2 * (q[..., 0] * q[..., 3] + q[..., 1] * q[..., 2]),
                                 1 - 2 * (q[..., 2] ** 2 + q[..., 3] ** 2)))


def timeseries(ep) -> pd.DataFrame:
    """Per log sample: t, x, y, v (COM, forward), yaw, tilt, hull z, per-wheel normal load ``fz_<W>``, longitudinal
    contact force ``fx_<W>``, wheel torque ``tau_<W>``, mechanical power ``p_<W>`` and the electrical estimate
    ``pe_<W>``; ``p_mech`` / ``p_el`` the sums."""
    log = ep.log
    t = np.asarray(log["t"])
    com, vel = np.asarray(log["com"]), np.asarray(log["com_vel"])
    quat = np.asarray(log["body_quat"])[:, 0]
    names = list(log["joints"])
    q = np.asarray(log["qd"])
    tau = np.asarray(log["tau"])
    ff = np.asarray(log["foot_force"])
    feet = list(log["feet"])
    motor = orb.wheel_servo()
    kt, R = motor.torque_constant, motor.resistance
    df = pd.DataFrame({"t": t, "x": com[:, 0], "y": com[:, 1], "v": vel[:, 0], "yaw_deg": yaw_deg(quat),
                       "tilt_deg": tilt_deg(quat), "hull_z": np.asarray(log["body_pos"])[:, 0, 2]})
    p_mech = np.zeros_like(t)
    p_el = np.zeros_like(t)
    for i, w in enumerate(feet):
        j = names.index(orb.wheel_joint(w))
        df[f"fz_{w}"] = ff[:, i, 2]
        df[f"fx_{w}"] = ff[:, i, 0]
        df[f"tau_{w}"] = tau[:, j]
        pm = np.abs(tau[:, j] * q[:, j])
        pe = pm + (tau[:, j] / kt) ** 2 * R
        df[f"p_{w}"], df[f"pe_{w}"] = pm, pe
        p_mech += pm
        p_el += pe
    df["p_mech"], df["p_el"] = p_mech, p_el
    return df


def phase_table(ep, phases=PHASES, seized: str = "RR") -> pd.DataFrame:
    """Per phase: mean speed, distance, RMS heading, max tilt, mean mechanical and electrical wheel power, the
    seized wheel's mean drag (−fx) and airborne fraction, the max corner load."""
    ts = timeseries(ep)
    feet = [c[3:] for c in ts.columns if c.startswith("fz_")]
    rows = {}
    for name, t0, t1 in phases:
        s = ts[(ts.t >= t0) & (ts.t < t1)]
        if not len(s):
            continue
        rows[name] = {
            "window_s": f"{t0:g}–{t1:g}", "mean_speed_m_s": s.v.mean(), "distance_m": s.x.iloc[-1] - s.x.iloc[0],
            "yaw_rms_deg": math.sqrt((s.yaw_deg ** 2).mean()), "lateral_max_m": s.y.abs().max(),
            "tilt_max_deg": s.tilt_deg.max(), "wheel_power_mech_W": s.p_mech.mean(), "wheel_power_el_W": s.p_el.mean(),
            f"{seized}_drag_N": (-s[f"fx_{seized}"]).mean(), f"{seized}_airborne": (s[f"fz_{seized}"] < 1.0).mean(),
            "corner_load_max_N": s[[f"fz_{w}" for w in feet]].max().max(),
            "cost_of_transport": s.p_mech.mean() / max(1e-9, s.v.mean() * ep.log["total_mass"] * orb.G),
        }
    return pd.DataFrame(rows).T
