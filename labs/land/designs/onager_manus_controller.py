"""The Onager Manus mission controller for ChironLab: drive, stop, work with the arms, drive on.

A ``Mission`` (``onager_controller.PhasedMission`` with arms) is a list of ``Phase`` s run in order; each phase
updates the speed target, the crouch and the arm and jaw targets until it reports done. Underneath, the Sentinel's
``onager_controller.Drive`` keeps driving the chassis (wheel speed loop, heading hold, the legs' gravity and
contact-force feed-forward). On top of it:

* **speed** — a stop at a world x: ``v = min(v_max, √(2 a (x_stop − x)), v_prev + a·dt)``, then 0 (the wheels'
  speed loop holds the robot);
* **crouch** — the hull lowered by ``c`` [m] on its legs, every axle kept under its shoulder at the same x
  (two-link inverse kinematics, as the three-wheel limp does);
* **arms** — joint-space moves (smoothstep from the current command to an inverse-kinematics target),
  straight-line pincer moves (the jaw pin along a line in the world, pincer elevation fixed, IK every control step),
  yaw swings; MuJoCo's gravity torques of the robot's own links (``qfrc_bias``) are fed forward;
* **jaws** — an opening angle, or a **grip**: the jaws push with a set torque whatever their angle (target = the
  current angle − τ/kp), or **close** fully (the drive's stall torque: cutting).

The controller only knows what a perception system would tell it: where the wire runs (``Scene``) and the log's
pose when it reaches for it (read from the simulation: ideal perception, stated as such). Whether the wire parts and
whether the log stays in the jaws is physics.

    import onager_manus_controller as omc
    mission = omc.Mission(omc.cut_and_clear(scene))
"""
from __future__ import annotations

import math

import numpy as np

import onager_controller as oc
import onager_manus_robot as omr

__all__ = ["Phase", "Mission", "smooth", "world_to_hull", "cut_and_clear"]


smooth, Phase = oc.smooth, oc.Phase


def world_to_hull(obs, p) -> np.ndarray:
    """A world point in the hull frame (origin at the hull centre, the root body)."""
    w, x, y, z = obs.base_quat
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return R.T @ (np.asarray(p, dtype=float) - np.asarray(obs.base_pos, dtype=float))


class Mission(oc.PhasedMission):
    """``onager_controller.PhasedMission`` with the two arms and their jaws (see the module)."""

    def __init__(self, phases: list, *, accel: float = 0.6, name: str = "manus mission"):
        super().__init__(phases, accel=accel, name=name)

    def tool_reset(self, lab):
        self.kp_jaw = omr.ARM_GAINS["jaw"][0]
        self.stall_jaw = omr.act.get(omr.ARM_ACTUATORS["jaw"]).stall_Nm
        self.arm = {s: np.array([omr.STOW["yaw"], omr.STOW["shoulder"], omr.STOW["elbow"], omr.STOW["wrist"]])
                    for s in omr.SIDES}
        self.jaw = {s: ("angle", omr.STOW["jaw"]) for s in omr.SIDES}

    def arm_target(self, obs, side: str, p_world, elevation: float):
        """IK in the hull frame for a world jaw-pin position; None when out of reach."""
        return omr.arm_ik(world_to_hull(obs, p_world), elevation, side, self.lab.robot.params)

    def tool_command(self, obs, q_t, qd_t, ff):
        bias = self.lab.data.qfrc_bias
        for side in omr.SIDES:
            names = omr.arm_joints(side)
            for n, v in zip(names[:4], self.arm[side]):
                q_t[n] = float(v)
                qd_t[n] = 0.0
                ff[n] = float(bias[self.dof[n]])                       # the arm's own gravity
            mode, val = self.jaw[side]
            for n in names[4:]:
                ff[n] = float(bias[self.dof[n]])
                qd_t[n] = 0.0
                if mode == "angle":
                    q_t[n] = float(val)
                elif mode == "grip":                                    # push with torque val whatever the angle
                    q_t[n] = float(obs.q[self.idx[n]]) - float(val) / self.kp_jaw
                else:                                                   # "close": the drive's full force
                    q_t[n] = float(obs.q[self.idx[n]]) - 2.0 * self.stall_jaw / self.kp_jaw


# ----------------------------------------------------------------------------------------------- the mission
def _joint_move(side, target_fn, T):
    """Phase pieces: move arm ``side`` in joint space to ``target_fn(m, obs)`` (yaw, sh, el, wr) over T s."""
    def start(m, obs):
        q1 = target_fn(m, obs)
        if q1 is None:
            raise RuntimeError(f"{side} arm: target out of reach")
        m.state.update(q0=m.arm[side].copy(), q1=np.asarray(q1, dtype=float))

    def update(m, obs, tau):
        u = smooth(tau / T)
        m.arm[side] = m.state["q0"] + (m.state["q1"] - m.state["q0"]) * u
        return tau >= T
    return start, update


def _line_move(side, p0_fn, p1_fn, elevation, T):
    """Move the jaw pin of arm ``side`` along a straight world line over T s, pincer at ``elevation``."""
    def start(m, obs):
        m.state.update(p0=np.asarray(p0_fn(m, obs), dtype=float), p1=np.asarray(p1_fn(m, obs), dtype=float))

    def update(m, obs, tau):
        u = smooth(tau / T)
        p = m.state["p0"] + (m.state["p1"] - m.state["p0"]) * u
        q = m.arm_target(obs, side, p, elevation)
        if q is not None:
            m.arm[side] = np.asarray(q)
        elif not m.state.get("unreachable"):
            m.state["unreachable"] = True
            m.log.append([float(obs.t), m.phase, f"{side} arm: target out of reach, holding the last pose"])
        return tau >= T
    return start, update


def _pin_world(m, obs, side):
    """The current commanded jaw-pin position in the world (forward kinematics of the commanded arm angles)."""
    (x, y, z), _ = omr.arm_fk(m.arm[side], side, m.lab.robot.params)
    w, qx, qy, qz = obs.base_quat
    R = np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - w * qz), 2 * (qx * qz + w * qy)],
                  [2 * (qx * qy + w * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - w * qx)],
                  [2 * (qx * qz - w * qy), 2 * (qy * qz + w * qx), 1 - 2 * (qx * qx + qy * qy)]])
    return R @ np.array([x, y, z]) + np.asarray(obs.base_pos)


def cut_and_clear(scene, *, v_max: float = 1.5, cut_arm: str = "R", lift_arm: str = "L", jaw_open_cut: float = 25.0,
                  jaw_open_log: float = 50.0, grip_torque: float = 60.0, crouch: float = 0.30, lift: float = 0.55,
                  swing_deg: float = 80.0) -> list:
    """The phases of notebook 21's mission on ``scene`` (``onager_manus_scenario.Scene``): drive to the wire, cut
    it with the ``cut_arm`` pincer (its notch on the wire), stow, drive through, stop at the log, crouch, take the log
    in the middle with the ``lift_arm`` pincer pointing down, lift it, swing it over the side, put it down, release,
    stow, stand up, drive on."""
    g = omr.arm_geometry()
    rad = math.radians
    s_cut = omr.SIDES[cut_arm]
    wire = scene.wire_point(s_cut * g["y"])                      # where the cut arm meets the wire (world)
    notch, jl = g["notch"], g["Lj"]
    # stop so that the jaw pin in the cut pose is r_cut ahead of the shoulder: the hull centre at x_wire − ...
    r_cut = 0.85
    x_stop_wire = wire[0] - notch - r_cut - g["x"]
    pre = lambda m, obs: wire + np.array([-notch - jl - 0.06, 0.0, 0.0])          # tips 6 cm short of the wire  # noqa: E731
    above = lambda m, obs: pre(m, obs) + np.array([0.0, 0.0, 0.25])             # noqa: E731
    cut_pose = lambda m, obs: wire + np.array([-notch, 0.0, 0.0])               # the notch on the wire  # noqa: E731

    log_c = scene.log_center()
    s_lift = omr.SIDES[lift_arm]
    r_log = 0.65                                                 # the log 0.65 m ahead of the shoulders (8 cm from the wheels)
    x_stop_log = log_c[0] - r_log - g["x"]
    # log centre below the jaw pin when gripping: the hooked tips (0.22 m out) then close under its middle. (Straight
    # jaws on a round log from above form a wedge that pushes it down and the arm up: the first runs lost the log.)
    grip_depth = 0.16

    def log_now(m, obs):
        """The log's centre as perception sees it when the arm reaches for it (the simulated pose), kept for the
        rest of the mission."""
        if m.memory.get("log") is None:
            m.memory["log"] = np.asarray(obs.prop_pos[m.lab.prop_bodies.index("log")], dtype=float).copy()
        return m.memory["log"]

    log_above = lambda m, obs: log_now(m, obs) + np.array([0.0, 0.0, grip_depth + 0.30])   # noqa: E731
    log_grip = lambda m, obs: log_now(m, obs) + np.array([0.0, 0.0, grip_depth])           # noqa: E731
    down = -math.pi / 2

    def set_v(x_stop):
        def update(m, obs, tau):
            return m.drive_to(obs, x_stop, v_max)
        return update

    def wait(T):
        return lambda m, obs, tau: tau >= T

    def jaw(side, mode, val=None):
        def start(m, obs):
            m.jaw[side] = (mode, val)
        return start

    def crouch_to(c, T):
        def start(m, obs):
            m.state["c0"] = m.crouch

        def update(m, obs, tau):
            m.crouch = m.state["c0"] + (c - m.state["c0"]) * smooth(tau / T)
            return tau >= T
        return start, update

    def until_cut(m, obs, tau):
        return not m.lab.weld_active(scene.WIRE_WELD) and tau > 0.3

    def ik_to(side, p_fn, elev):
        return lambda m, obs: m.arm_target(obs, side, p_fn(m, obs), elev)

    def stow_q(m, obs):
        return (omr.STOW["yaw"], omr.STOW["shoulder"], omr.STOW["elbow"], omr.STOW["wrist"])

    def swing_target(m, obs):
        q = m.arm[lift_arm].copy()
        q[0] += s_lift * rad(swing_deg)
        return q

    def both(*starts):
        """Several start functions as one (bound now: the loop variables change later)."""
        fs = tuple(starts)

        def start(m, obs):
            for f in fs:
                f(m, obs)
        return start

    phases = []
    add = lambda name, upd, start=None, timeout=30.0: phases.append(Phase(name, upd, start, timeout))  # noqa: E731
    add("drive to the wire", set_v(x_stop_wire), timeout=40.0)
    add("settle", wait(0.5))
    st, up = _joint_move(cut_arm, ik_to(cut_arm, above, 0.0), 3.0)
    add(f"{cut_arm} arm: over the wire", up, both(jaw(cut_arm, "angle", rad(jaw_open_cut)), st))
    st, up = _line_move(cut_arm, above, pre, 0.0, 1.5)
    add(f"{cut_arm} arm: down in front of the wire", up, st)
    st, up = _line_move(cut_arm, pre, cut_pose, 0.0, 2.0)
    add(f"{cut_arm} arm: notch onto the wire", up, st)
    add("cut", until_cut, jaw(cut_arm, "close"), timeout=6.0)
    add("open the jaws", wait(1.0), jaw(cut_arm, "angle", rad(jaw_open_cut)))
    st, up = _line_move(cut_arm, cut_pose, pre, 0.0, 1.5)
    add(f"{cut_arm} arm: back off", up, st)
    st, up = _joint_move(cut_arm, stow_q, 3.0)
    add(f"{cut_arm} arm: stow", up, both(jaw(cut_arm, "angle", 0.0), st))
    add("drive through to the log", set_v(x_stop_log), timeout=60.0)
    add("settle", wait(0.5))
    st, up = crouch_to(crouch, 2.0)
    add("crouch", up, st)

    def creep(m, obs, tau):                                      # crouching moves the hull: close in on the log seen now
        x_stop = log_now(m, obs)[0] - r_log - g["x"]
        x = float(obs.base_pos[0])
        m.v = float(np.clip(0.8 * (x_stop - x), -0.2, 0.2))
        return abs(x_stop - x) < 0.01 and abs(float(obs.com_vel[0])) < 0.02
    add("creep to the log", creep, timeout=8.0)
    add("settle", wait(0.5), lambda m, obs: setattr(m, "v", 0.0))
    st, up = _joint_move(lift_arm, ik_to(lift_arm, log_above, down), 3.0)
    add(f"{lift_arm} arm: over the log", up, both(jaw(lift_arm, "angle", rad(jaw_open_log)), st))
    st, up = _line_move(lift_arm, log_above, log_grip, down, 2.0)
    add(f"{lift_arm} arm: jaws around the log", up, st)
    add("grip", wait(1.5), jaw(lift_arm, "grip", grip_torque))
    st, up = _line_move(lift_arm, log_grip, lambda m, obs: log_grip(m, obs) + np.array([0, 0, lift]), down, 2.5)
    add("lift", up, st)
    add("hold", wait(1.0))
    st, up = _joint_move(lift_arm, swing_target, 3.0)
    add("swing over the side", up, st)
    st, up = _line_move(lift_arm, lambda m, obs: _pin_world(m, obs, lift_arm),
                        lambda m, obs: _pin_world(m, obs, lift_arm) + np.array([0, 0, -lift + 0.02]), down, 2.0)
    add("put it down", up, st)
    add("release", wait(1.0), jaw(lift_arm, "angle", rad(jaw_open_log)))
    st, up = _line_move(lift_arm, lambda m, obs: _pin_world(m, obs, lift_arm),
                        lambda m, obs: _pin_world(m, obs, lift_arm) + np.array([0, 0, 0.3]), down, 1.5)
    add(f"{lift_arm} arm: up", up, st)
    st, up = _joint_move(lift_arm, stow_q, 3.0)
    add(f"{lift_arm} arm: stow", up, both(jaw(lift_arm, "angle", 0.0), st))
    st, up = crouch_to(0.0, 2.0)
    add("stand up", up, st)
    add("drive on", set_v(scene.x_end), timeout=30.0)
    return phases
