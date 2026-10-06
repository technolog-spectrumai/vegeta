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

**Trials** (notebook 30 §7.2, ``TRIALS``): the unchanged rover on three grounds, each run under ``FailureRules`` so
the episode — and its movie — ends at the frame where the mission fails (fall, stall, off course, timeout) or at
the finish line:

* ``micro_hills`` — an irregular egg-crate of hills the rover's own radius: hill diameter twice the rover's length
  (1.44 m), the steepest slope 15°, sized numerically (``A`` from the field's max gradient on a fine grid);
* ``steep_hill`` — flat, up a 30° ramp, a plateau, down the other side;
* ``mud`` — flat rough ground with a zone of water-ish soil that moves under the tracks (``MudHook``: the rollers'
  friction drops to the mud's μ, the hull meets the Bekker compaction resistance of ``terramechanics.SOILS['mud']``
  plus a viscous drag, and the soft layer drifts sideways and pushes the rover with it). MuJoCo's ground stays
  rigid: sinkage shows up as resistance and lost grip, not as a rut.

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

__all__ = ["Leg", "MISSION", "COURSE", "TrackDrive", "yaw_of", "uneven_ground", "run", "timeseries", "leg_table",
           "micro_hills", "steep_hill", "mud_flat", "max_slope_deg", "MudHook", "TRIALS", "trial_rules", "run_trial",
           "trial_table", "end_card"]


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


# ----------------------------------------------------------------------------------------------- the trials (§7.2)
ROVER_LENGTH = 0.72                                     # m, PekariRover.overall(p)['length'] of the default design


def max_slope_deg(terrain, extent, cell: float = 0.01) -> float:
    """The steepest gradient [deg] of ``terrain.height`` on a ``cell`` grid over ``extent`` (x0, x1, y0, y1)."""
    x = np.arange(extent[0], extent[1], cell)
    y = np.arange(extent[2], extent[3], cell)
    X, Y = np.meshgrid(x, y)
    Z = terrain.height(X, Y)
    gy, gx = np.gradient(Z, cell)
    return float(np.degrees(np.arctan(np.hypot(gx, gy).max())))


def micro_hills(diameter: float = 2 * ROVER_LENGTH, slope_deg: float = 15.0, start: float = 0.5, length: float = 7.0,
                extent=(-1.5, 8.5, -3.0, 3.0)):
    """Irregular micro-hills: an egg-crate z = A sin(kx + φ(y)) sin(ky + φ(x)) whose hills (the positive lobes) are
    ``diameter`` across (k = π / diameter) with slowly varying phases so no two rows match; ``A`` is set so the
    steepest gradient of the field is ``slope_deg`` (measured on a 10 mm grid). Flat before ``start`` (a 0.5 m
    blend) and after ``start + length``."""
    from vegeta import chiron as ch

    k = math.pi / diameter

    def field(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return np.sin(k * x + 0.6 * np.sin(0.37 * k * y)) * np.sin(k * y + 0.5 * np.sin(0.41 * k * x) + 0.9)

    blend = 1.5

    def unit(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        w_in = np.clip((x - start) / blend, 0.0, 1.0)
        w_out = np.clip((start + length - x) / blend, 0.0, 1.0)
        w = (w_in * w_out) ** 2 * (3 - 2 * w_in * w_out)                 # a smooth step in and out
        return field(x, y) * w

    # the ground scales with A, blends included: size it on the whole course
    A = math.tan(math.radians(slope_deg)) / math.tan(math.radians(
        max_slope_deg(ch.Custom(unit), (start - 0.5, start + length + 0.5, -1.5, 1.5))))

    def ground(x, y):
        return A * unit(x, y)

    t = ch.Custom(ground, name=f"micro-hills: diameter {diameter:.2f} m, slopes up to {slope_deg:g} deg, "
                               f"height ±{A * 1000:.0f} mm")
    t.amplitude, t.diameter, t.extent = A, diameter, extent
    return t


def steep_hill(angle_deg: float = 30.0, height: float = 0.9, start: float = 1.0, plateau: float = 1.0, knee: float = 0.3,
               extent=(-1.5, 9.5, -2.0, 2.0)):
    """Flat, up a ``angle_deg`` ramp to ``height``, a ``plateau``, down the other side, flat — the knees rounded
    over ``knee`` m (a box average of the piecewise profile). The ramp's horizontal run is height / tan(angle)."""
    from vegeta import chiron as ch

    run = height / math.tan(math.radians(angle_deg))
    xs = np.array([start, start + run, start + run + plateau, start + 2 * run + plateau])
    zs = np.array([0.0, height, height, 0.0])
    offsets = np.linspace(-knee / 2, knee / 2, 7)

    def ground(x, y):
        x = np.asarray(x, dtype=float)
        z = np.zeros_like(x)
        for d in offsets:
            z = z + np.interp(x + d, xs, zs, left=0.0, right=0.0)
        return z / len(offsets) + 0.0 * np.asarray(y, dtype=float)

    t = ch.Custom(ground, name=f"a {angle_deg:g} deg hill, {height:g} m high (ramp run {run:.2f} m), plateau {plateau:g} m")
    t.knots, t.extent, t.top = (xs, zs), extent, (start + run, start + run + plateau)
    return t


def mud_flat(zone=(1.5, 4.5), rms: float = 0.008, extent=(-1.5, 8.5, -3.0, 3.0), seed: int = 7):
    """Flat rough ground (RMS ``rms``) — the mud ``zone`` (x0, x1) lives in ``MudHook``, not in the height."""
    from vegeta import chiron as ch

    t = ch.Rough(rms, 0.25, start=0.3, seed=seed, extent=extent, cell=0.02)
    t.zone, t.extent = tuple(zone), extent
    return t


class MudHook:
    """Water-ish, moving soil in the zone (x0, x1), as a ChironLab scene hook (``lab.add_hook``).

    Every control step: ``f`` = the fraction of the rover's contact length inside the zone (from the hull's x). The
    rollers' sliding friction is ``mu_mud`` when the hull is in the zone (``mu_dry`` outside; MuJoCo's rigid ground
    keeps the robot's value). On the hull, a force ``−f (R_c + c_v |v|) v̂`` — R_c the Bekker compaction resistance
    of both tracks on ``soil`` at the rover's weight (``terramechanics``), ``c_v`` the viscous drag of soil moving
    with the tracks — plus the soft layer's drift ``f · flow · W`` (world x, y; a fraction of the weight) with a
    slow random part (``gust``, seeded). The hull's velocity is taken from its position between calls."""

    def __init__(self, zone=(1.5, 4.5), *, soil: str = "mud", flow=(0.0, -0.15), gust: float = 0.05, c_v: float = 20.0,
                 mu_mud: float = 0.25, mu_dry: float = 0.6, seed: int = 0):
        self.zone, self.soil, self.flow, self.gust, self.c_v = tuple(zone), soil, np.asarray(flow, dtype=float), gust, c_v
        self.mu_mud, self.mu_dry, self.seed = mu_mud, mu_dry, seed

    def reset(self, lab):
        import terramechanics as tm

        self.lab = lab
        self.hull = lab._body_id("hull")
        self.geoms = [lab._geom_id(f.geom) for f in lab.robot.feet]
        self.L = lab.robot.geometry["L"]
        W = lab.total_mass * prr.G
        self.W = W
        self.R_c = 2 * tm.compaction_resistance_track(W / 2, lab.robot.geometry["b"], self.L, self.soil)
        self.rng = np.random.default_rng(self.seed)
        self.noise = np.zeros(2)
        self.prev = None
        self.inside = False
        self._set_mu(self.mu_dry)

    def _set_mu(self, mu):
        for g in self.geoms:
            self.lab.model.geom_friction[g, 0] = mu

    def fraction(self, x):
        lo, hi = x - self.L / 2, x + self.L / 2
        return float(np.clip(min(hi, self.zone[1]) - max(lo, self.zone[0]), 0.0, self.L) / self.L)

    def __call__(self, lab):
        pos = np.array(lab.data.xpos[self.hull][:2])
        t = lab.time
        v = np.zeros(2)
        if self.prev is not None and t > self.prev[0]:
            v = (pos - self.prev[1]) / (t - self.prev[0])
        self.prev = (t, pos.copy())
        f = self.fraction(pos[0])
        inside = f > 0.0
        if inside != self.inside:
            self.inside = inside
            self._set_mu(self.mu_mud if inside else self.mu_dry)
            lab.log_event("mud", "into the mud" if inside else "out of the mud")
        if f <= 0.0:
            lab.body_force("hull", None)
            return
        speed = float(np.linalg.norm(v))
        drag = -(self.R_c + self.c_v * speed) * (v / speed if speed > 1e-6 else np.zeros(2))
        self.noise = 0.999 * self.noise + 0.0447 * self.rng.normal(0.0, self.gust, 2)     # a slow (~1 s) random walk
        push = (self.flow + self.noise) * self.W
        F = f * (drag + push)
        lab.body_force("hull", (F[0], F[1], 0.0))


#: The three trials: terrain, legs (one straight run over the course), failure rules and hooks.
TRIALS = {
    "micro_hills": dict(terrain=micro_hills, course_m=6.5, v=0.6, max_tilt_deg=40.0, lateral_limit=1.0, hooks=()),
    "steep_hill": dict(terrain=steep_hill, course_m=7.5, v=0.5, max_tilt_deg=50.0, lateral_limit=1.0, hooks=()),
    "mud": dict(terrain=mud_flat, course_m=6.0, v=0.6, max_tilt_deg=40.0, lateral_limit=1.0, hooks=(MudHook,)),
}


def trial_rules(name: str):
    """The ``FailureRules`` of a trial: success past ``course_m``; fall at ``max_tilt_deg``; off course beyond
    ``lateral_limit``; stall when the COM advances less than 10 % of v × 3 s; timeout 2 × course / v + 2 s."""
    from vegeta.chiron import FailureRules

    tr = TRIALS[name]
    return FailureRules(course_m=tr["course_m"], max_tilt_deg=tr["max_tilt_deg"], lateral_limit=tr["lateral_limit"],
                        stall_window=3.0, stall_fraction=0.1, stall_grace=2.0, v_target=tr["v"])


def run_trial(name: str, robot=None, *, log_geoms: bool = True, seed: int = 0, **lab_kw):
    """Run one trial of ``TRIALS``: the rover (``robot`` or ``prr.pekari()``) drives straight at the trial's speed
    until the rules end it. Returns the Episode (``ep.outcome``, ``ep.log['events']`` with the hooks' events)."""
    tr = TRIALS[name]
    terrain = tr["terrain"]()
    opts = dict(course_extent=getattr(terrain, "extent", prr.LAB_OPTIONS["course_extent"]), log_geoms=log_geoms)
    opts.update(lab_kw)
    lab = prr.pekari_lab(terrain, robot=robot, **opts)
    for hook in tr["hooks"]:
        lab.add_hook(hook())
    rules = trial_rules(name)
    ctrl = TrackDrive([Leg("straight", tr["course_m"] + 1.0, v=tr["v"])], name=f"trial/{name}")
    ep = lab.run(ctrl, rules=rules, settle=0.5, seed=seed, info={"controller": ctrl.name, "trial": name, "v_target": tr["v"]})
    ep.log["events"] = sorted(list(ctrl.events) + list(lab.events), key=lambda e: e[0])
    ep.terrain = terrain
    return ep


def trial_table(episodes: dict, robot=None):
    """Per trial: outcome, reason and detail, time, distance along x, max tilt, min speed after 2 s, mean side belt
    forces, max |side force|, energy."""
    import pandas as pd

    rows = {}
    for name, ep in episodes.items():
        ts = timeseries(ep, robot)
        o = ep.outcome
        late = ts[ts.t >= 2.0]
        rows[name] = {"outcome": "success" if o.get("success") else "FAIL", "reason": o.get("reason"), "detail": o.get("detail"),
                      "t_end [s]": o.get("t_end"), "distance x [m]": o.get("distance_m"), "max tilt [deg]": float(ts.tilt_deg.max()),
                      "min speed after 2 s [m/s]": float(late.v.min()) if len(late) else np.nan,
                      "max |y| [m]": float(ts.y.abs().max()), "end heading [deg]": float(ts.yaw_deg.iloc[-1]),
                      "F_L mean [N]": float(ts.F_L.mean()), "F_R mean [N]": float(ts.F_R.mean()),
                      "|F| max [N]": float(ts[["F_L", "F_R"]].abs().max().max()),
                      "energy [J]": float(np.trapezoid(ts.P_mech, ts.t))}
    return pd.DataFrame(rows).T


def end_card(images: list, ep, fps: int = 25, hold_s: float = 1.5) -> list:
    """The movie's last frame held for ``hold_s`` with the outcome written on it (red: a failure, green: success),
    so a failed mission's movie ends at the frame where it failed."""
    import cv2

    o = ep.outcome
    ok = bool(o.get("success"))
    text = ("FINISHED" if ok else "FAILED: " + str(o.get("reason", "")).upper()) + f"  t = {o.get('t_end', 0):.1f} s"
    detail = str(o.get("detail", ""))
    last = images[-1].copy()
    colour = (46, 125, 50) if ok else (198, 40, 40)
    cv2.rectangle(last, (0, last.shape[0] - 70), (last.shape[1], last.shape[0]), (20, 20, 20), -1)
    cv2.putText(last, text, (16, last.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2, cv2.LINE_AA)
    cv2.putText(last, detail[:90], (16, last.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    return list(images) + [last] * int(round(hold_s * fps))
