"""Motions for the design-stage robots: keyframed joint trajectories worked out kinematically (no dynamics, no
controller) and carried in the Yax export as the machine's part data ``data:motions`` (tag ``motion``).

A motion is a list of keyframes: joint angles [rad] over one cycle (``loop``) or one transition, with the base's
velocity in its own frame (x forward) when the motion travels. Foot and hand paths are turned into joint angles by
inverse kinematics on the robot's own MuJoCo model (``Kin``: the Chiron robot's ``to_mjcf``, the base held still),
so the angles agree with the exported geometry; undulations, flaps and rotor spins are written as joint
functions directly. Every motion carries its ``check``: the worst IK error, the joint-limit margin and, for gaits,
the stance feet's height spread.

Block shape (``data:motions``)::

    {"format": "vegeta-motions/1", "base_height_m": …, "motions": {name: {
        "kind": "gait" | "transition" | "cycle" | "rates", "loop": bool, "period_s": …,
        "joints": [names], "t": [s], "q": [[rad per joint] per t], "rates": {joint: rad/s} (kind "rates"),
        "base": {"velocity_m_s": [vx, vy, wz]} | None, "feet": {foot: {"phase", "duty"}} (gaits),
        "params": {…}, "check": {…}, "source": "…"}}}
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["Kin", "Motion", "block", "legged_gait", "joint_cycle", "ik_transition", "froude_speed", "G", "smooth"]

G = 9.81


def smooth(u):
    """0 → 1 with zero slope at both ends."""
    u = min(max(u, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * u)


def froude_speed(froude, leg_length_m):
    """Speed at a Froude number v² / (g L): ~0.15 a walk, ~0.5 a trot, ≥ 1 a bound or gallop (Alexander 1984)."""
    return math.sqrt(froude * G * leg_length_m)


class Kin:
    """The robot's kinematics: its MJCF compiled by MuJoCo, the free base fixed at the origin pose of q = 0."""

    def __init__(self, robot):
        import mujoco

        self.mj = mujoco
        self.robot = robot
        self.model = mujoco.MjModel.from_xml_string(robot.to_mjcf())
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.hinges = [i for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
        self.names = [m.joint(i).name for i in self.hinges]
        self.index = {n: k for k, n in enumerate(self.names)}
        self.qadr = np.array([m.jnt_qposadr[i] for i in self.hinges])
        self.dadr = np.array([m.jnt_dofadr[i] for i in self.hinges])
        self.lo = np.array([m.jnt_range[i][0] if m.jnt_limited[i] else -np.inf for i in self.hinges])
        self.hi = np.array([m.jnt_range[i][1] if m.jnt_limited[i] else np.inf for i in self.hinges])
        self.q0 = np.zeros(len(self.hinges))

    def set(self, q):
        d = self.data
        d.qpos[:] = self.model.qpos0
        d.qpos[self.qadr] = q
        self.mj.mj_kinematics(self.model, d)
        self.mj.mj_comPos(self.model, d)

    def body(self, name):
        return self.model.body(name).id

    def geom_point(self, geom):
        """(body id, local point) of a geom's centre."""
        g = self.model.geom(geom)
        return int(g.bodyid[0]), np.array(self.model.geom_pos[g.id])

    def world(self, body, local, q=None):
        if q is not None:
            self.set(q)
        d = self.data
        return d.xpos[body] + d.xmat[body].reshape(3, 3) @ local

    def ik(self, targets, q_init=None, q_ref=None, joints=None, iters=60, reg=1e-3, tol=1e-5):
        """Joint angles putting each target point at its goal: ``targets`` = [(body id, local point, goal xyz)],
        moving ``joints`` (names; default all hinges) within their ranges, regularised towards ``q_ref``
        (default q = 0, the design pose). Returns (q, worst error [m])."""
        mj, m, d = self.mj, self.model, self.data
        q = np.array(self.q0 if q_init is None else q_init, dtype=float)
        q_ref = self.q0 if q_ref is None else np.asarray(q_ref)
        cols = np.arange(len(self.names)) if joints is None else np.array([self.index[j] for j in joints])
        jp = np.zeros((3, m.nv))
        err = np.inf
        for _ in range(iters):
            self.set(q)
            rows, J = [], []
            for body, local, goal in targets:
                p = d.xpos[body] + d.xmat[body].reshape(3, 3) @ local
                rows.append(np.asarray(goal) - p)
                mj.mj_jac(m, d, jp, None, p, body)
                J.append(jp[:, self.dadr[cols]].copy())
            e, J = np.concatenate(rows), np.vstack(J)
            err = float(np.max(np.linalg.norm(e.reshape(-1, 3), axis=1)))
            if err < tol:
                break
            r = np.sqrt(reg) * (q_ref[cols] - q[cols])
            A = np.vstack([J, np.sqrt(reg) * np.eye(len(cols))])
            b = np.concatenate([e, r])
            dq = np.linalg.lstsq(A, b, rcond=None)[0]
            q[cols] = np.clip(q[cols] + dq, self.lo[cols], self.hi[cols])
        return q, err

    def margin(self, Q, joints=None):
        """Smallest distance [rad] of the trajectory ``Q`` (rows) from a joint limit (negative: outside)."""
        cols = np.arange(len(self.names)) if joints is None else np.array([self.index[j] for j in joints])
        Q = np.asarray(Q)[:, cols]
        lo, hi = self.lo[cols], self.hi[cols]
        return float(np.min(np.minimum(Q - lo, hi - Q)))


class Motion:
    def __init__(self, name, kind, joints, t, Q, *, loop=True, period=None, base=None, params=None, check=None,
                 source="", feet=None, rates=None):
        self.name, self.kind, self.joints, self.t, self.Q = name, kind, list(joints), list(t), np.asarray(Q)
        self.loop, self.period, self.base, self.params = loop, period, base, params or {}
        self.check, self.source, self.feet, self.rates = check or {}, source, feet, rates

    def to_data(self):
        out = {"kind": self.kind, "loop": self.loop, "period_s": round(self.period, 4) if self.period else None,
               "joints": self.joints, "t": [round(x, 4) for x in self.t],
               "q": [[round(float(v), 5) for v in row] for row in self.Q],
               "base": self.base, "params": self.params, "check": self.check, "source": self.source}
        if self.feet:
            out["feet"] = self.feet
        if self.rates:
            out["rates"] = {k: round(v, 3) for k, v in self.rates.items()}
        return out


def block(motions, base_height=None):
    """The ``data:motions`` block and its tag (``motion``) for a machine's parts function."""
    return {"motions": {"format": "vegeta-motions/1",
                        "base_height_m": None if base_height is None else round(base_height, 4),
                        "motions": {m.name: m.to_data() for m in motions}},
            "tags": {"data:motions*": "motion"}}


def _moving(kin, Q, tol=1e-6):
    """The joints ``Q`` (rows of all hinges) takes away from the design pose at some time."""
    Q = np.asarray(Q)
    return [n for k, n in enumerate(kin.names) if np.max(np.abs(Q[:, k] - kin.q0[k])) > tol]


def legged_gait(kin, name, feet, *, period, duty, stride, step_height, frames=32, foot_points=None, extra=None,
                lateral=None, crouch=0.0, q_start=None, source="", params=None):
    """A periodic gait in the base frame: each foot in ``feet`` ({geom: phase 0..1}) slides back ``stride`` [m]
    on the ground during its stance (``duty`` of the cycle) and swings forward on a raised half-sine of
    ``step_height``; the base travels forward at stride / (duty · period). ``foot_points`` ({geom: [local
    offsets]}) tracks several points of a foot (a flat sole); ``lateral(t)`` [m] shifts all feet sideways (the
    base's sway, seen from the base); ``crouch`` [m] lowers the base (the feet rise in its frame: straight-legged
    designs need it to bend their knees; ``q_start`` {joint: rad} seeds the IK off a straight-leg singularity);
    ``extra(t, q)`` sets joints no foot drives (arm swing). Returns a Motion."""
    pts = {}
    for g in feet:
        body, c = kin.geom_point(g)
        offs = (foot_points or {}).get(g, [np.zeros(3)])
        pts[g] = [(body, c + np.asarray(o), kin.world(body, c + np.asarray(o), kin.q0)) for o in offs]
    ts = [period * i / frames for i in range(frames)]
    Q, q, worst, stance_z = [], kin.q0.copy(), 0.0, []
    for j, v in (q_start or {}).items():
        q[kin.index[j]] = v
    for t in ts:
        targets = []
        dy = lateral(t) if lateral else 0.0
        for g, ph in feet.items():
            s = (t / period + ph) % 1.0
            if s < duty:
                dx, dz = stride * (0.5 - s / duty), 0.0
            else:
                u = (s - duty) / (1.0 - duty)
                dx, dz = stride * (-0.5 + smooth(u)), step_height * math.sin(math.pi * u)
            for body, local, p0 in pts[g]:
                targets.append((body, local, p0 + np.array([dx, dy, dz + crouch])))
        q, err = kin.ik(targets, q_init=q)
        if extra:
            q = extra(t, q.copy())
        kin.set(q)
        for g, ph in feet.items():
            if (t / period + ph) % 1.0 < duty:
                for body, local, p0 in pts[g]:
                    stance_z.append(kin.world(body, local)[2] - p0[2] - crouch)
        worst = max(worst, err)
        Q.append(q.copy())
    joints = _moving(kin, Q)
    cols = [kin.index[j] for j in joints]
    v = stride / (duty * period)
    check = {"ik_error_max_m": round(worst, 6), "limit_margin_min_rad": round(kin.margin(Q, joints), 4),
             "stance_height_spread_m": round(float(np.ptp(stance_z)) if stance_z else 0.0, 6)}
    return Motion(name, "gait", joints, ts, np.asarray(Q)[:, cols], period=period, base={"velocity_m_s": [round(v, 4), 0.0, 0.0]},
                  feet={g: {"phase": ph, "duty": duty} for g, ph in feet.items()}, check=check, source=source,
                  params=dict(params or {}, stride_m=round(stride, 4), step_height_m=round(step_height, 4),
                              duty=duty, speed_m_s=round(v, 4), crouch_m=round(crouch, 4)))


def joint_cycle(kin, name, fn, *, period, frames=32, loop=True, kind="cycle", source="", params=None, base=None):
    """A motion written as joint functions: ``fn(t)`` → {joint: angle offset from the design pose [rad]}."""
    ts = [period * i / frames for i in range(frames + (0 if loop else 1))]
    Q = []
    for t in ts:
        q = kin.q0.copy()
        for j, v in fn(t).items():
            q[kin.index[j]] = v
        Q.append(np.clip(q, kin.lo, kin.hi))
    raw = [[fn(t).get(j, 0.0) for j in kin.names] for t in ts]
    clipped = float(np.max(np.abs(np.asarray(raw) - np.asarray(Q))))
    joints = _moving(kin, Q)
    cols = [kin.index[j] for j in joints]
    return Motion(name, kind, joints, ts, np.asarray(Q)[:, cols], loop=loop, period=period, base=base, source=source,
                  params=params or {}, check={"limit_margin_min_rad": round(kin.margin(Q, joints), 4),
                                              "clipped_max_rad": round(clipped, 4)})


def ik_transition(kin, name, keyposes, *, duration, frames=24, loop=False, source="", params=None, joints=None,
                  extra=None):
    """A motion through ``keyposes`` = [(fraction of the duration, [(body, local, goal), …]), …] solved by IK and
    eased between (smooth steps); ``extra(t, q)`` adds direct joint values (a claw closing)."""
    ts = [duration * i / frames for i in range(frames + (0 if loop else 1))]
    Q, q, worst = [], kin.q0.copy(), 0.0
    for t in ts:
        u = t / duration
        k = max(i for i, (f, _) in enumerate(keyposes) if f <= u + 1e-9)
        if k + 1 < len(keyposes):
            f0, a = keyposes[k]
            f1, b = keyposes[k + 1]
            w = smooth((u - f0) / (f1 - f0))
            targets = [(ba, la, (1 - w) * np.asarray(ga) + w * np.asarray(gb)) for (ba, la, ga), (_, _, gb) in zip(a, b)]
        else:
            targets = keyposes[k][1]
        q, err = kin.ik(targets, q_init=q, joints=joints)
        if extra:
            q = np.clip(extra(t, q.copy()), kin.lo, kin.hi)
        worst = max(worst, err)
        Q.append(q.copy())
    moving = _moving(kin, Q)
    cols = [kin.index[j] for j in moving]
    return Motion(name, "transition" if not loop else "cycle", moving, ts, np.asarray(Q)[:, cols], loop=loop,
                  period=duration, source=source, params=params or {},
                  check={"ik_error_max_m": round(worst, 6), "limit_margin_min_rad": round(kin.margin(Q, moving), 4)})
