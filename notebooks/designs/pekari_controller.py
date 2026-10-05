"""Skid-steer driving for the Pekari Rover in ChironLab (notebook 30 §8): a list of legs — drive straight for a
distance, turn by an angle on a radius — and the tables of a run.

``TrackDrive`` is a ChironLab controller (``reset(lab, seed)``, ``__call__(obs) -> Command``):

* **legs** — ``Leg("straight", 3.0, v=0.8)`` holds the heading for 3 m; ``Leg("turn", 90.0, v=0.5, radius=1.0)``
  turns 90° left (negative: right) at a yaw rate v / R (``radius`` 0: a pivot turn at ``v`` track speed);
* **steering** — the reference heading advances at the commanded yaw rate (held on straight legs); the side belt
  speeds are v ∓ (ω_ref + k_h (ψ_ref − ψ)) B/2, so the heading error is fed back; v ramps at ``accel``;
* **rollers** — every roller on a side gets ω = v_side / r_i (``pekari_rover_robot.pekari``'s velocity servos).

The controller never reads the terrain. ``timeseries(ep)`` and ``leg_table(ep)`` give the numbers (path, speed,
heading, tilt, side belt forces and power, slip).

    import pekari_rover_robot as prr, pekari_controller as pc
    lab = prr.pekari_lab(chiron.Flat())
    ep = lab.run(pc.TrackDrive(pc.MISSION), duration=14.0, rules=None)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from vegeta.chiron import Command

import pekari_rover_robot as prr

__all__ = ["Leg", "MISSION", "COURSE", "TrackDrive", "yaw_of", "uneven_ground", "run", "timeseries", "leg_table"]


def yaw_of(quat) -> float:
    w, x, y, z = quat
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


@dataclass
class Leg:
    """``kind`` 'straight' (``value`` = distance [m]) or 'turn' (``value`` = heading change [deg], + left);
    ``v`` [m/s] the speed of the rover's centre (a pivot turn: of the tracks); ``radius`` [m] of a turn."""

    kind: str
    value: float
    v: float = 0.8
    radius: float = 1.0

    def __post_init__(self):
        if self.kind not in ("straight", "turn"):
            raise ValueError("Leg.kind must be 'straight' or 'turn'")


#: Notebook 30 §8: a short run, a 90° left turn on a 1 m radius, and on.
MISSION = [Leg("straight", 3.0, v=0.8), Leg("turn", 90.0, v=0.5, radius=1.0), Leg("straight", 2.0, v=0.8)]


#: The uneven ground of notebook 30 §8: rough soil (ISO-8608-like, RMS 15 mm, correlation 0.3 m) from x = 0.5 m, a
#: 60 mm log lying across the path at x = 1.5 m and three 40 mm stones on the way and in the turn.
COURSE = {"rms": 0.015, "correlation_length": 0.3, "start": 0.5, "seed": 4, "extent": (-1.5, 6.5, -2.0, 5.0),
          "cell": 0.02, "log_x": 1.5, "log_h": 0.06, "stones": [(2.4, 0.12, 0.04), (3.9, 0.35, 0.04), (4.45, 2.2, 0.04)]}


def uneven_ground(course=COURSE):
    """``ch.Custom``: the rough soil plus the log (a half cylinder of radius ``log_h``) and the stones (smooth
    40 mm bumps, 0.12 m across)."""
    from vegeta import chiron as ch

    rough = ch.Rough(course["rms"], course["correlation_length"], start=course["start"], seed=course["seed"],
                     extent=course["extent"], cell=course["cell"])
    x0, h = course["log_x"], course["log_h"]

    def ground(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = rough.height(x, y) + h * np.sqrt(np.clip(1 - ((x - x0) / h) ** 2, 0, None))
        for sx, sy, sh in course["stones"]:
            z = z + sh * np.exp(-((x - sx) ** 2 + (y - sy) ** 2) / (2 * 0.04 ** 2))
        return z

    return ch.Custom(ground, name=f"rough soil (RMS {course['rms'] * 1000:.0f} mm), a {h * 1000:.0f} mm log at "
                                  f"x = {x0:g} m, {len(course['stones'])} stones")


def run(lab, legs=MISSION, duration: float = 16.0, settle: float = 0.5, seed: int = 0, **kw):
    """Run ``TrackDrive(legs)`` and return the Episode with the controller's events in ``ep.log['events']``."""
    ctrl = TrackDrive(legs, **kw)
    ep = lab.run(ctrl, duration=duration, rules=None, settle=settle, seed=seed, info={"controller": ctrl.name})
    ep.log["events"] = list(ctrl.events)
    return ep


class TrackDrive:
    def __init__(self, legs=MISSION, *, accel: float = 1.0, k_heading: float = 3.0, name: str = "track drive"):
        self.legs = list(legs)
        self.accel, self.k_heading, self.name = float(accel), float(k_heading), name
        self.events = []

    # ---- ChironLab protocol
    def reset(self, lab, seed=None):
        self.lab = lab
        g = lab.robot.geometry
        self.B = g["B"]
        self.radius = {f"{side}_{name}": r for side in prr.SIDES for name, _, _, r, *_ in lab.robot.rollers}
        self.joints = {side: [f"{side}_{name}" for name, *_ in lab.robot.rollers] for side in prr.SIDES}
        self.i = 0
        self.t_leg = None
        self.v = 0.0
        self.psi_ref = None
        self.psi_unwrapped = None
        self.psi_prev = None
        self.dist = 0.0
        self.xy_prev = None
        self.turned = 0.0
        self.done = not self.legs                            # no legs: stand still (a standing check)
        self.t_prev = None
        self.events = []

    def settle_command(self, obs):
        return self._command(0.0, 0.0)

    def _command(self, v_left, v_right):
        qd = {}
        for side, v in (("L", v_left), ("R", v_right)):
            for j in self.joints[side]:
                qd[j] = v / self.radius[j]
        return Command(q_target={}, qd_target=qd)

    def _heading(self, obs):
        psi = yaw_of(np.asarray(obs.base_quat))
        if self.psi_prev is None:
            self.psi_unwrapped = psi
        else:
            d = (psi - self.psi_prev + math.pi) % (2 * math.pi) - math.pi
            self.psi_unwrapped += d
        self.psi_prev = psi
        return self.psi_unwrapped

    def __call__(self, obs):
        t = float(obs.t)
        dt = 0.0 if self.t_prev is None else t - self.t_prev
        self.t_prev = t
        psi = self._heading(obs)
        xy = np.asarray(obs.com)[:2]
        if self.psi_ref is None:
            self.psi_ref = psi
            self.t_leg = t
            if self.legs:
                self.events.append([t, "leg", f"1: {self.legs[0].kind} {self.legs[0].value:g}"])
        if self.xy_prev is not None:
            self.dist += float(np.linalg.norm(xy - self.xy_prev))
        self.xy_prev = xy
        if self.done:
            self.v = max(0.0, self.v - self.accel * dt)
            return self._command(self.v, self.v)
        leg = self.legs[self.i]
        self.v = min(leg.v, self.v + self.accel * dt) if self.v < leg.v else max(leg.v, self.v - self.accel * dt)
        omega = 0.0
        if leg.kind == "turn":
            target = math.radians(leg.value)
            rate = (leg.v / leg.radius if leg.radius > 0 else 2 * leg.v / self.B) * math.copysign(1.0, target)
            step = rate * dt
            if abs(self.turned + step) >= abs(target):
                step = target - self.turned
            self.turned += step
            self.psi_ref += step
            omega = step / dt if dt > 0 else 0.0
            finished = abs(self.turned) >= abs(target) - 1e-9 and abs(self.psi_ref - psi) < math.radians(3.0)
        else:
            finished = self.dist >= leg.value
        w = omega + self.k_heading * (self.psi_ref - psi)
        v_c = 0.0 if (leg.kind == "turn" and leg.radius == 0) else self.v
        cmd = self._command(v_c - w * self.B / 2, v_c + w * self.B / 2)
        if finished:
            self.i += 1
            self.dist, self.turned = 0.0, 0.0
            if self.i >= len(self.legs):
                self.done = True
                self.events.append([t, "done", f"course done, heading {math.degrees(psi):.1f} deg"])
            else:
                nxt = self.legs[self.i]
                self.events.append([t, "leg", f"{self.i + 1}: {nxt.kind} {nxt.value:g}"])
        return cmd


def _tilt_deg(quat):
    q = np.asarray(quat, dtype=float)
    return np.degrees(np.arccos(np.clip(1 - 2 * (q[..., 1] ** 2 + q[..., 2] ** 2), -1.0, 1.0)))


def timeseries(ep, robot=None):
    """Per log sample: t, x, y, v (ground speed of the COM), yaw [deg], tilt [deg], the side belt forces ``F_L`` /
    ``F_R`` (Σ τ_i / r_i), belt speeds ``vb_L`` / ``vb_R`` (mean ω_i r_i), mechanical power ``P_mech`` (Σ τ ω) and the
    leg index from the events."""
    import pandas as pd

    log = ep.log
    t = np.asarray(log["t"])
    com, vel = np.asarray(log["com"]), np.asarray(log["com_vel"])
    quat = np.asarray(log["body_quat"])[:, 0]
    names = list(log["joints"])
    qd, tau = np.asarray(log["qd"]), np.asarray(log["tau"])
    rl = prr.rollers() if robot is None else robot.rollers
    yaw = np.unwrap(np.array([yaw_of(q) for q in quat]))
    df = pd.DataFrame({"t": t, "x": com[:, 0], "y": com[:, 1], "v": np.hypot(vel[:, 0], vel[:, 1]),
                       "yaw_deg": np.degrees(yaw), "tilt_deg": _tilt_deg(quat), "z": com[:, 2]})
    p = np.zeros_like(t)
    for side in prr.SIDES:
        F = np.zeros_like(t)
        vb = np.zeros_like(t)
        for name, _, _, r, *_ in rl:
            j = names.index(f"{side}_{name}")
            F += tau[:, j] / r
            vb += qd[:, j] * r / len(rl)
            p += tau[:, j] * qd[:, j]
        df[f"F_{side}"], df[f"vb_{side}"] = F, vb
    df["P_mech"] = p
    leg = np.zeros(len(t), dtype=int)
    for ev in log.get("events", []):
        if ev[1] in ("leg", "done"):
            leg[t >= ev[0]] += 1
    df["leg"] = leg
    return df


def leg_table(ep, legs=MISSION, robot=None):
    """Per leg: duration, distance, mean speed, heading change, the turn radius reached, max tilt, mean side belt
    forces, slip (1 − ground speed / mean belt speed) and mechanical energy."""
    import pandas as pd

    ts = timeseries(ep, robot)
    rows = {}
    for k, lg in enumerate(legs, start=1):
        s = ts[ts.leg == k]
        if len(s) < 2:
            continue
        dist = float(np.sum(np.hypot(np.diff(s.x), np.diff(s.y))))
        dpsi = float(s.yaw_deg.iloc[-1] - s.yaw_deg.iloc[0])
        T = float(s.t.iloc[-1] - s.t.iloc[0])
        vb = float(((s.vb_L + s.vb_R) / 2).mean())
        rows[f"{k}: {lg.kind} {lg.value:g}"] = {
            "duration [s]": T, "distance [m]": dist, "mean speed [m/s]": dist / T if T > 0 else 0.0,
            "heading change [deg]": dpsi,
            "turn radius [m]": dist / abs(math.radians(dpsi)) if lg.kind == "turn" and dpsi else np.nan,
            "max tilt [deg]": float(s.tilt_deg.max()), "F_L mean [N]": float(s.F_L.mean()), "F_R mean [N]": float(s.F_R.mean()),
            "slip": 1 - (dist / T) / vb if T > 0 and vb > 0 else np.nan,
            "energy [J]": float(np.trapezoid(s.P_mech, s.t)),
        }
    return pd.DataFrame(rows).T
