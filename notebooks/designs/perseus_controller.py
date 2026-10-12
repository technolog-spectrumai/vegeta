"""The Perseus mission controller for ChironLab: drive to the tree, cut the branches one by one, drive on.

A ``CutMission`` is a list of ``onager_controller.Phase`` s run in order (the Onager Manus pattern, without the
Sentinel's wheel-legs). Underneath, the tracks are driven as ``pekari_controller.TrackDrive`` does — every roller a
velocity target at the side's belt speed, skid steering with a heading hold — here from a speed target ``m.v``
[m/s] and a heading target the phases set (``drive_to``: brake to a stop at a world x). On top, the boom:

* **joint moves** — smoothstep in joint space from the commanded pose to an inverse-kinematics target;
* **line moves** — the bar root along a straight world line, the bar's elevation fixed, IK every control step;
* **feed** — the bar root pushed along the bar's normal (into the branch) at ``feed_speed`` while the scene's
  ``SawCutter`` reports the bar engaged, backing off while it reports a jam, until the branch is cut;
* **brake** — while the speed target is zero every roller is held at the angle it stopped at (a speed command
  proportional to its angle error, capped: the parking brake a real track has; the roller model's velocity servos
  would otherwise let the boom's reactions and a slope roll the machine);
* MuJoCo's gravity torques on the boom's own links (``qfrc_bias``) are fed forward.

The controller only knows what a perception system would tell it: where the trunk stands and where each branch
leaves it (``Scene``). Whether the saw gets through is the scene hook's physics: the bar's contact force and the
cutting-rate model of ``cutting``.

    import perseus_controller as pcc
    mission = pcc.CutMission(pcc.branch_job(scene, robot))
"""
from __future__ import annotations

import math

import numpy as np

from vegeta.chiron import Command

import onager_controller as oc
import pekari_controller as pc
import pekari_rover_robot as prr
import perseus_robot as pbr

__all__ = ["Phase", "CutMission", "smooth", "world_to_hull", "hull_to_world", "branch_job", "stand_distance",
           "branch_targets"]

smooth, Phase = oc.smooth, oc.Phase


def _rot(quat) -> np.ndarray:
    w, x, y, z = quat
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def world_to_hull(obs, p) -> np.ndarray:
    """A world point in the hull frame (origin at the hull centre, the root body)."""
    return _rot(obs.base_quat).T @ (np.asarray(p, dtype=float) - np.asarray(obs.base_pos, dtype=float))


def hull_to_world(obs, p) -> np.ndarray:
    return _rot(obs.base_quat) @ np.asarray(p, dtype=float) + np.asarray(obs.base_pos, dtype=float)


class CutMission:
    """Phases on top of a track drive and the boom (see the module). Phases set ``m.v`` (speed target, m/s) and
    ``m.arm`` (the commanded joint angles yaw, sh, el, wr); ``m.log`` records [t, phase, note]; ``m.memory`` keeps
    what the mission has seen; ``m.state`` is the running phase's scratch."""

    def __init__(self, phases: list, *, accel: float = 0.5, k_heading: float = 3.0, k_brake: float = 4.0,
                 v_brake: float = 0.15, name: str = "cut mission"):
        self.phases = list(phases)
        self.accel, self.k_heading, self.name = float(accel), float(k_heading), name
        self.k_brake, self.v_brake = float(k_brake), float(v_brake)

    # ---- ChironLab protocol
    def reset(self, lab, seed=None):
        self.lab = lab
        robot = lab.robot
        self.v_ = robot.variant
        self.p = robot.params
        g = robot.geometry
        self.B = g["B"]
        self.radius = {f"{side}_{name}": r for side in prr.SIDES for name, _, _, r, *_ in robot.rollers}
        self.track_joints = {side: [f"{side}_{name}" for name, *_ in robot.rollers] for side in prr.SIDES}
        self.idx = {n: i for i, n in enumerate(lab.joint_names)}
        self.dof = {n: int(lab._jd[lab._joint_index[n]]) for n in lab.actuated_joints}
        self.arm = np.array(pbr.stow_q(self.v_), dtype=float)
        self.v, self.psi_ref, self.latched = 0.0, None, None
        self.psi_prev, self.psi_unwrapped = None, None
        self.i, self.t0, self.started, self.t_prev = 0, 0.0, False, None
        self.log, self.state, self.memory = [], {}, {}

    def settle_command(self, obs):
        return self._command(obs, 0.0)

    def _heading(self, obs):
        psi = pc.yaw_of(np.asarray(obs.base_quat))
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
        while self.i < len(self.phases):
            ph = self.phases[self.i]
            if not self.started:
                self.t0, self.started, self.state = t, True, {}
                if ph.start is not None:
                    ph.start(self, obs)
                self.log.append([t, ph.name, "start"])
            tau = t - self.t0
            done = bool(ph.update(self, obs, tau))
            if not done and tau >= ph.timeout:
                self.log.append([t, ph.name, f"timeout after {ph.timeout:g} s"])
                done = True
            if not done:
                break
            self.log.append([t, ph.name, "done"])
            self.i += 1
            self.started = False
        if self.i >= len(self.phases):
            self.v = 0.0
        return self._command(obs, dt)

    @property
    def phase(self) -> str:
        return self.phases[self.i].name if self.i < len(self.phases) else "finished"

    @property
    def finished(self) -> bool:
        return self.i >= len(self.phases)

    # ---- driving
    def drive_to(self, obs, x_stop: float, v_max: float, tol: float = 0.03) -> bool:
        """Brake to a stop at world x: v = min(v_max, √(2 a |Δx|), |v| + a·dt); done within ``tol`` and slow."""
        x = float(obs.base_pos[0])
        dx = x_stop - x
        v_brake = math.sqrt(2 * self.accel * max(0.0, abs(dx) - tol / 2))
        speed = min(v_max, v_brake, abs(self.v) + self.accel * self.lab.control_dt)
        self.v = math.copysign(speed, dx)
        done = abs(dx) < tol and abs(float(obs.com_vel[0])) < 0.05
        if done:
            self.v = 0.0
        return done

    def _command(self, obs, dt: float) -> Command:
        psi = self._heading(obs)
        if self.psi_ref is None:
            self.psi_ref = psi
        qd = {}
        if abs(self.v) > 1e-9:
            self.latched = None
            w = self.k_heading * (self.psi_ref - psi)
            for side, v_side in (("L", self.v - w * self.B / 2), ("R", self.v + w * self.B / 2)):
                for j in self.track_joints[side]:
                    qd[j] = v_side / self.radius[j]
        else:                                                   # the brake: every roller held where it stopped
            if self.latched is None:
                self.latched = {j: float(obs.q[self.idx[j]]) for side in self.track_joints for j in self.track_joints[side]}
                self.psi_ref = psi
            w = self.k_heading * (self.psi_ref - psi)             # and the heading held (a rocking hull walks)
            for side, sgn in (("L", -1.0), ("R", 1.0)):
                for j in self.track_joints[side]:
                    lim = self.v_brake / self.radius[j]
                    hold = self.k_brake * (self.latched[j] - float(obs.q[self.idx[j]]))
                    qd[j] = float(np.clip(hold + sgn * w * self.B / 2 / self.radius[j], -lim, lim))
        q_t, ff = {}, {}
        bias = self.lab.data.qfrc_bias
        for n, val in zip(pbr.ARM_JOINTS, self.arm):
            q_t[n] = float(val)
            qd[n] = 0.0
            ff[n] = float(bias[self.dof[n]])                        # the boom's own gravity
        return Command(q_target=q_t, qd_target=qd, tau_ff=ff)

    # ---- the boom
    def arm_target(self, obs, p_world, elevation: float):
        """IK in the hull frame for a world bar-root position; None when out of reach."""
        return pbr.arm_ik(world_to_hull(obs, p_world), elevation, self.p, self.v_)

    def bar_root_world(self, obs) -> np.ndarray:
        """The commanded bar root in the world (forward kinematics of the commanded angles)."""
        (x, y, z), _ = pbr.arm_fk(self.arm, self.p, self.v_)
        return hull_to_world(obs, (x, y, z))


# ----------------------------------------------------------------------------------------------- phase pieces
def _joint_move(target_fn, T):
    """Move the boom in joint space to ``target_fn(m, obs)`` (yaw, sh, el, wr) over T s."""
    def start(m, obs):
        q1 = target_fn(m, obs)
        if q1 is None:
            m.log.append([float(obs.t), m.phase, "boom: target out of reach, holding the pose"])
            q1 = m.arm
        m.state.update(q0=m.arm.copy(), q1=np.asarray(q1, dtype=float))

    def update(m, obs, tau):
        u = smooth(tau / T)
        m.arm = m.state["q0"] + (m.state["q1"] - m.state["q0"]) * u
        return tau >= T
    return start, update


def _line_move(p0_fn, p1_fn, elevation, T):
    """Move the bar root along a straight world line over T s, the bar at ``elevation``."""
    def start(m, obs):
        m.state.update(p0=np.asarray(p0_fn(m, obs), dtype=float), p1=np.asarray(p1_fn(m, obs), dtype=float))

    def update(m, obs, tau):
        u = smooth(tau / T)
        p = m.state["p0"] + (m.state["p1"] - m.state["p0"]) * u
        q = m.arm_target(obs, p, elevation)
        if q is not None:
            m.arm = np.asarray(q)
        elif not m.state.get("unreachable"):
            m.state["unreachable"] = True
            m.log.append([float(obs.t), m.phase, "boom: target out of reach, holding the last pose"])
        return tau >= T
    return start, update


def _feed(branch, direction, feed_speed, elevation, retreat=2.0, band=(0.5, 0.8)):
    """Push the bar root along ``direction`` (world unit vector, into the branch) under a force loop on the
    cutter's reading: push at ``feed_speed`` [m/s] below ``band[0]`` × the window's top, back off at that speed
    above ``band[1]`` × it, hold in between; back off at ``retreat`` × the speed while the cutter reports a jam
    (the boom's servos deflect under the push: backing off slowly leaves the bar pressed). Done when the
    branch's weld is released (the operator's feel for the saw, as a regulator)."""
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)

    def start(m, obs):
        m.state.update(p=m.bar_root_world(obs).copy(), t_prev=float(obs.t))

    def update(m, obs, tau):
        t = float(obs.t)
        dt = t - m.state["t_prev"]
        m.state["t_prev"] = t
        cutter = m.lab.cutter
        if branch.name in cutter.cut_at:
            return True
        f = cutter.force.get(branch.name, 0.0)
        if branch.name in cutter.jammed:
            step = -retreat * feed_speed * dt
        elif f < band[0] * cutter.F_max:
            step = feed_speed * dt
        elif f > band[1] * cutter.F_max:
            step = -feed_speed * dt
        else:
            step = 0.0
        m.state["p"] = m.state["p"] + d * step
        q = m.arm_target(obs, m.state["p"], elevation)
        if q is not None:
            m.arm = np.asarray(q)
        return False
    return start, update


def branch_targets(scene, robot, br, standoff: float, approach: float, cut_from: str = "top") -> dict:
    """The bar-root targets for one branch in the world as functions of the shoulder's world position: ``above``
    (the bar across the branch ``standoff`` above/below it), ``near`` (``approach`` away) and ``through`` (the cut
    done), plus the cut point. The bar lies level, pointing from the shoulder to the cut point in plan, the branch
    under the middle of its cutting edge (the nose clear: the kickback rule)."""
    g = robot.arm
    cut = np.asarray(br.cut_point(scene.cut_s), dtype=float)
    s_bar = (g["Lbar"] - g["bar_w"] / 2) / 2 + g["bar_w"] / 4
    sign = 1.0 if cut_from == "top" else -1.0

    def root(sh, off_z):
        d = cut - np.asarray(sh, dtype=float)
        d[2] = 0.0
        d = d / np.linalg.norm(d)
        return cut - d * s_bar + np.array([0.0, 0.0, off_z])
    return {"cut": cut, "above": lambda sh: root(sh, sign * (standoff + br.d / 2)),
            "near": lambda sh: root(sh, sign * (approach + br.d / 2)), "through": lambda sh: root(sh, -sign * br.d / 2)}


def _offsets(robot, standoff, approach) -> tuple:
    """Default stand-off and approach [m]: 10 % and 5 % of the reach."""
    R = robot.arm["reach"]
    return (0.10 * R if standoff is None else standoff), (0.05 * R if approach is None else approach)


def stand_distance(scene, robot, *, standoff: float | None = None, approach: float | None = None,
                   cut_from: str = "top", n: int = 150) -> float:
    """The hull-centre-to-trunk distance [m] from which every branch's targets (above, near and through the cut)
    are within the boom's reach: the third quarter of the feasible interval scanned over 0.2–1.0 × the reach
    (further out leaves the carry pose room before the trunk); raises when none is (which branches are out of
    reach)."""
    g = robot.arm
    v_ = robot.variant
    z_hull = g["z_hull"]
    standoff, approach = _offsets(robot, standoff, approach)
    cands = np.linspace(0.2 * g["reach"], 1.0 * g["reach"], n)
    ok = []
    worst = {}
    slack = 0.05                                               # the stop's tolerance and the ground's height
    for s in cands:
        x_hull = scene.x_tree - s
        good = True
        for br in scene.branches:
            tg = branch_targets(scene, robot, br, standoff, approach, cut_from)
            for key in ("above", "near", "through"):
                for dx_, dz_ in ((0.0, 0.0), (slack, slack), (-slack, -slack), (slack, -slack), (-slack, slack)):
                    sh = (x_hull + dx_ + g["x"], g["y"], g["z_world"] + dz_)
                    pw = tg[key](sh)
                    ph = (pw[0] - x_hull - dx_, pw[1], pw[2] - z_hull - dz_)
                    if pbr.arm_ik(ph, 0.0, robot.params, v_) is None:
                        good = False
                        worst[br.name] = worst.get(br.name, 0) + 1
        ok.append(good)
    ok = np.asarray(ok)
    if not ok.any():
        bad = sorted(worst, key=worst.get, reverse=True)
        raise RuntimeError(f"no stand distance reaches every branch: out of reach most often {bad[:2]} "
                           f"(reach {g['reach']:.2f} m, shoulder {g['z_world']:.2f} m up)")
    idx = np.flatnonzero(ok)
    return float(cands[idx[(3 * len(idx)) // 4]])             # towards the far end: room for the carry pose


def branch_job(scene, robot, *, v_max: float = 0.8, feed_speed: float = 0.02, standoff: float | None = None,
               approach: float | None = None, cut_from: str = "top") -> list:
    """The phases of the mission on ``scene`` (``perseus_scenario.Scene``): drive to the stand-off from the trunk,
    then for each branch (lowest first): retract to the carry pose (the bar raised, clear of the trunk), swing to
    the branch's bearing still retracted, extend so the bar lies across the branch at ``standoff`` [m] above it
    (``cut_from`` 'top') or below ('bottom'),
    the branch under the middle of the bar's cutting edge (the nose clear: the kickback rule), move in, feed
    through, back off; then retract, stow and back out to ``scene.x_end``. The cut
    point on each branch is ``scene.cut_s`` [m] from the trunk. The chain runs throughout: a swing that sweeps the
    bar through a neighbouring branch cuts it too (the scene hook counts every engaged contact)."""
    v_ = robot.variant
    tree = np.asarray(scene.trunk_base, dtype=float)
    standoff, approach = _offsets(robot, standoff, approach)
    # stop where every cut is inside the boom's reach: the hull centre at x_trunk - stand_dist
    stand_dist = (scene.stand_dist if scene.stand_dist is not None
                  else stand_distance(scene, robot, standoff=standoff, approach=approach, cut_from=cut_from))
    x_stop = tree[0] - stand_dist
    sign = 1.0 if cut_from == "top" else -1.0

    def set_v(x):
        def update(m, obs, tau):
            return m.drive_to(obs, x, v_max)
        return update

    def wait(T):
        return lambda m, obs, tau: tau >= T

    def stow_q(m, obs):
        return pbr.stow_q(v_)

    def carry(m, obs):
        return pbr.carry_q(v_, float(m.arm[0]))

    phases = []
    add = lambda name, upd, start=None, timeout=30.0: phases.append(Phase(name, upd, start, timeout))  # noqa: E731
    add("drive to the tree", set_v(x_stop), timeout=60.0)
    add("settle", wait(0.5))
    g = robot.arm

    def shoulder(m, obs):
        return hull_to_world(obs, (g["x"], g["y"], g["z"]))
    for br in sorted(scene.branches, key=lambda b: b.z):
        tg = branch_targets(scene, robot, br, standoff, approach, cut_from)
        above = lambda m, obs, f=tg["above"]: f(shoulder(m, obs))        # noqa: E731
        near = lambda m, obs, f=tg["near"]: f(shoulder(m, obs))          # noqa: E731
        def swing(m, obs, f=above):
            q1 = m.arm_target(obs, f(m, obs), 0.0)
            return None if q1 is None else pbr.carry_q(v_, float(q1[0]))
        st, up = _joint_move(carry, 3.0)
        add(f"retract before {br.name}", up, st, timeout=10.0)
        st, up = _joint_move(swing, 3.0)
        add(f"swing to {br.name}", up, st, timeout=10.0)                       # retracted: the saw clears the trunk
        st, up = _joint_move(lambda m, obs, f=above: m.arm_target(obs, f(m, obs), 0.0), 4.0)
        add(f"boom to {br.name} (Ø {br.d * 1e3:.0f} mm)", up, st, timeout=10.0)
        st, up = _line_move(above, near, 0.0, 2.0)
        add(f"{br.name}: move in", up, st)
        st, up = _feed(br, (0.0, 0.0, -sign), feed_speed, 0.0)
        add(f"{br.name}: cut", up, st, timeout=scene.cut_timeout)
        st, up = _line_move(lambda m, obs: m.bar_root_world(obs), above, 0.0, 2.0)
        add(f"{br.name}: back off", up, st)
    st, up = _joint_move(carry, 3.0)
    add("retract", up, st, timeout=10.0)
    st, up = _joint_move(stow_q, 4.0)
    add("stow", up, st)
    add("back out", set_v(scene.x_end), timeout=60.0)
    return phases
