"""Wheel-mode driving for the Onager Sentinel in ChironLab, with partial failures of the drive.

``Drive`` is a ChironLab controller (``reset(lab, seed)``, ``__call__(obs) -> Command``):

* **wheels** — every hub motor gets a speed target ω = v / r; a heading hold adds a differential (skid-steer):
  the left and right sides differ by ``k_heading × yaw + k_lateral × y`` (rad/s), clipped to ``±diff_clip``;
* **legs** — shoulders and knees hold the standing pose through their position servos (an active suspension:
  kp / kd of ``onager_robot.leg_servo``) with a contact-force feed-forward: each leg's servos add
  ``−Jᵀ (f_x, f_y, W/4)`` from the lab's foot Jacobians — the measured horizontal contact force of its wheel
  (``force_ff``: the drive, braking and drag forces that would otherwise swing the leg on its servo compliance;
  a seized tyre's drag of ~700 N is ~700 N·m at the shoulder, 25° on a 1600 N·m/rad servo) and its nominal share
  of the weight (``gravity_ff``; ``load_ff='measured'``: the wheel's measured normal force instead, for a machine
  whose payload moves its weight between the wheels; ``'axle'``: the mean of its axle's two wheels, so a wheel that
  lifts on uneven ground still pushes down — the Atlas's stiff stance) — so the servos' stiffness is spent on
  deviations, not on the standing loads;
* **wheel speed loop** — the hub motor's velocity servo is proportional (``onager_robot.wheel_servo``: τ = kd
  (ω_target − ω), on the torque–speed line) and droops under load; the controller adds the integral term of a PI
  loop as a feed-forward torque, ``k_i ∫(ω_target − ω) dt`` (anti-windup), so a loaded wheel still reaches its
  speed. Both terms are clamped to ``tau_wheel_max`` (default 150 N·m: a wheel's tractive force acts ~1 m below
  the shoulder, and the 800 N·m shoulder module cannot react the hub motor's 240 N·m stall torque through the leg —
  it would swing the leg to its limit): the speed error the P term sees is clipped to ``tau_wheel_max / kd``;
* **speed ramp** — v rises from 0 over ``ramp_s`` (the hub motors stay on their torque–speed line otherwise);
* **failures** — ``Failure(t, wheel, mode)`` at walking time ``t`` on wheel ``'FL'``…: ``'motor_off'`` (the hub
  motor loses power: the controller commands zero torque — ω_target = the wheel's own speed — and the wheel
  freewheels), ``'seized'`` (the wheel is braked: ω_target = 0 on the motor's full torque line, up to its stall
  torque; the tyre skids), ``'knee_locked'`` (the knee module loses power and its spring-applied brake engages:
  modelled as the servo holding the angle of the failure instant, with the servo's compliance and its
  feed-forward — the brake carries the load without current). The controller's *response* to a seized wheel (``lift_seized`` [m] > 0) is the three-wheel limp:
  the CG must leave the seized corner's side of the diagonal through the two neighbouring wheels, so over
  ``shift_s`` the three good legs move the hull ``shift_x`` [m] along x away from the seized wheel (every axle
  re-placed by the two-link inverse kinematics at the same height) while the seized leg folds its wheel up by
  ``lift_seized``; each good leg's gravity feed-forward then carries W/3. After any seizure the speed target drops
  to ``v_limp`` (None = keep it), whether the wheel is lifted or dragged. Failures and the shift are logged in ``Drive.events``.

The controller never reads the terrain; the hull level follows from the servo compliance. ``Stand`` holds the pose.

    import onager_robot as orb, onager_controller as oc
    lab = orb.onager_lab(chiron.Flat())
    ep = lab.run(oc.Drive(3.0, failures=[oc.Failure(4.0, "FL", "motor_off")]), duration=10.0, rules=None)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from vegeta.chiron import Command

import onager_robot as orb

__all__ = ["Failure", "Drive", "Stand", "yaw_of", "smooth", "Phase", "PhasedMission"]


def _ik(x_axle: float, z_down: float, L1: float, L2: float):
    """Two-link planar IK (``gait.ik_two_link_planar``): (a1, a2) [deg] for an axle ``x_axle`` ahead of the shoulder
    and ``z_down`` below it; None when out of reach."""
    import gait

    return gait.ik_two_link_planar(x_axle, -z_down, L1, L2)


def yaw_of(quat) -> float:
    w, x, y, z = quat
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


@dataclass
class Failure:
    """A partial failure at walking time ``t`` [s] on ``wheel`` ('FL', 'FR', 'RL', 'RR'): ``mode`` 'motor_off',
    'seized' or 'knee_locked'."""

    t: float
    wheel: str
    mode: str

    def __post_init__(self):
        if self.mode not in ("motor_off", "seized", "knee_locked"):
            raise ValueError(f"unknown failure mode {self.mode!r}")
        if self.wheel not in orb.LEGS:
            raise ValueError(f"unknown wheel {self.wheel!r}")


class Drive:
    """Wheel-mode drive at ``v_target`` [m/s] with a heading hold and scripted partial failures (see the module)."""

    def __init__(self, v_target: float, *, failures=(), ramp_s: float = 1.5, k_heading: float = 30.0,
                 k_lateral: float = 3.0, diff_clip: float = 4.0, gravity_ff: bool = True, lift_seized: float = 0.10,
                 shift_x: float = 0.30, shift_s: float = 1.5, v_limp: float | None = 1.5, k_i: float = 60.0,
                 force_ff: bool = True, tau_wheel_max: float = 150.0, crouch: float = 0.0, load_ff: str = "nominal",
                 name: str | None = None):
        self.v_limp = v_limp
        self.force_ff = force_ff
        if load_ff not in ("nominal", "measured", "axle"):
            raise ValueError("load_ff must be 'nominal', 'measured' or 'axle'")
        self.load_ff = load_ff
        self.tau_wheel_max = float(tau_wheel_max)
        self.v_target = float(v_target)
        self.k_i = float(k_i)
        self.shift_x, self.shift_s = float(shift_x), float(shift_s)
        self.failures = sorted((Failure(*f) if isinstance(f, tuple) else f for f in failures), key=lambda f: f.t)
        self.ramp_s, self.k_heading, self.k_lateral, self.diff_clip = ramp_s, k_heading, k_lateral, diff_clip
        self.gravity_ff, self.lift_seized, self.crouch = gravity_ff, lift_seized, float(crouch)
        self.name = name or ("drive" if not self.failures else "drive+failures")
        self.events = []

    # ---- ChironLab protocol
    def reset(self, lab, seed=None):
        self.lab = lab
        self.r = lab.robot.geometry["r_wheel"]
        self.L2 = lab.robot.geometry["L2"]
        self.L1 = lab.robot.geometry["L1"]
        self.a2 = lab.robot.geometry["a2"]
        self.axle_x = lab.robot.geometry["axle_x"]
        self.h_axle = lab.robot.geometry["h_axle"]
        self.locked = {}                                     # knee joint -> angle held after a knee_locked failure
        self.limp = None                                     # (t_start, seized wheel) of the three-wheel limp
        self.weight = lab.total_mass * orb.G
        self.q0 = {n: float(lab.nominal_q[lab._joint_index[n]]) for n in lab.actuated_joints}
        self.idx = {n: i for i, n in enumerate(lab.joint_names)}
        self.feet = list(lab.feet)
        self.state = {leg: None for leg in orb.LEGS}        # wheel -> active failure mode
        self.events = []
        self.yaw0 = None
        self.integral = {leg: 0.0 for leg in orb.LEGS}
        self.t_prev = None
        self.stall = {leg: min(orb.act.get(orb.WHEEL_MOTOR).stall_Nm, self.tau_wheel_max) for leg in orb.LEGS}
        self.dw_max = self.tau_wheel_max / orb.WHEEL_KD              # speed error the P loop may see
        if self.crouch:
            # lower the hull by `crouch` [m] with the knees (the wheel rises by L2 sin a2 per rad of knee flexion)
            dq = self.crouch / (self.L2 * math.sin(self.a2))
            for leg in orb.LEGS:
                self.q0[f"{leg}_knee"] -= dq

    def settle_command(self, obs):
        return self._command(obs, 0.0, 0.0)

    def __call__(self, obs):
        t = float(obs.t)
        if self.yaw0 is None:
            self.yaw0 = yaw_of(obs.base_quat)
        for f in self.failures:
            if t >= f.t and self.state[f.wheel] != f.mode and (f.wheel, f.mode) not in {(e[1], e[2]) for e in self.events}:
                self.state[f.wheel] = f.mode
                self.integral[f.wheel] = 0.0
                self.events.append((t, f.wheel, f.mode))
                if f.mode == "knee_locked":
                    jk = orb.leg_joints(f.wheel)[1]
                    self.locked[jk] = float(obs.q[self.idx[jk]])
                if f.mode == "seized" and self.lift_seized > 0 and self.limp is None:
                    self.limp = (t, f.wheel)
                    self.events.append((t, f.wheel, f"limp: hull shifts {self.shift_x:+.2f} m, wheel lifts {self.lift_seized:.2f} m"))
        v = self.v_target * min(1.0, t / self.ramp_s) if self.ramp_s > 0 else self.v_target
        if self.v_limp is not None and any(m == "seized" for m in self.state.values()):
            v = min(v, self.v_limp)                              # a seized wheel: home at the reduced speed, lifted or dragged
        dt = 0.0 if self.t_prev is None else t - self.t_prev
        self.t_prev = t
        return self._command(obs, v, dt)

    # ---- the command
    def command(self, obs, v: float, dt: float) -> Command:
        """The chassis command at speed ``v`` [m/s] for a controller that schedules the speed itself (failures
        and the limp are not evaluated: call the Drive itself for those)."""
        if self.yaw0 is None:
            self.yaw0 = yaw_of(obs.base_quat)
        return self._command(obs, v, dt)

    def _command(self, obs, v: float, dt: float) -> Command:
        yaw = yaw_of(obs.base_quat) - (self.yaw0 or 0.0)
        y = float(obs.com[1])
        diff = float(np.clip(self.k_heading * yaw + self.k_lateral * y, -self.diff_clip, self.diff_clip))
        w0 = v / self.r
        q_target, qd_target, tau_ff = dict(self.q0), {}, {}
        n_carry = len(self.feet)
        if self.limp is not None:                            # the three-wheel limp: re-place every axle
            t0, seized = self.limp
            u = min(1.0, max(0.0, (float(obs.t) - t0) / self.shift_s))
            u = u * u * (3 - 2 * u)
            dx = -orb.LEGS[seized][0] * self.shift_x * u     # the hull moves away from the seized wheel ...
            n_carry = len(self.feet) - 1
            for leg in self.feet:
                js, jk, _ = orb.leg_joints(leg)
                lift = self.lift_seized * u if leg == seized else 0.0
                ik = _ik(self.axle_x - dx, self.h_axle - lift, self.L1, self.L2)   # ... so the axles move the other way
                if ik is not None:
                    q_target[js], q_target[jk] = orb.angles_to_q(*ik)
        for i, leg in enumerate(self.feet):
            sx, sy = orb.LEGS[leg]
            js, jk, jw = orb.leg_joints(leg)
            # a left wheel slows down when the hull points left (yaw > 0): turn right
            w = w0 + (diff if sy > 0 else -diff)
            mode = self.state.get(leg)
            w_now = float(obs.qd[self.idx[jw]])
            qd_target[jw] = w_now + float(np.clip(w - w_now, -self.dw_max, self.dw_max))
            if mode is None and self.k_i > 0:
                err = w - w_now
                self.integral[leg] = float(np.clip(self.integral[leg] + self.k_i * err * dt, -self.stall[leg], self.stall[leg]))
                tau_ff[jw] = self.integral[leg]
            if self.gravity_ff or self.force_ff:
                jac = obs.foot_jac[i]                                    # (3, n_leg): shoulder, knee, wheel
                f = np.zeros(3)
                if self.gravity_ff:
                    f[2] = self.weight / n_carry
                    if self.load_ff == "measured" and not (mode == "seized" and self.limp is not None):
                        f[2] = max(0.0, float(obs.foot_force[i][2]))       # what this wheel carries now (a payload)
                    elif self.load_ff == "axle" and not (mode == "seized" and self.limp is not None):
                        pair = [k for k, w in enumerate(self.feet) if orb.LEGS[w][0] == sx]
                        f[2] = max(0.0, float(np.mean([obs.foot_force[k][2] for k in pair])))   # the axle's mean
                if self.force_ff:
                    f[:2] = obs.foot_force[i][:2]                        # the measured horizontal contact force
                tau = -jac.T @ f
                tau_ff[js], tau_ff[jk] = float(tau[0]), float(tau[1])
            if mode == "motor_off":
                qd_target[jw] = float(obs.qd[self.idx[jw]])              # zero torque: freewheel
                tau_ff[jw] = 0.0
            elif mode == "seized":
                qd_target[jw] = 0.0                                      # braked on the motor's torque line
                tau_ff[jw] = 0.0
                if self.limp is not None and self.limp[1] == leg:
                    tau_ff[jk] = tau_ff[js] = 0.0                        # lifted: carries nothing
            elif mode == "knee_locked":
                q_target[jk] = self.locked[jk]                           # the brake holds the failure angle (the
                # feed-forward stays: a spring-applied brake carries the standing load without current)
        return Command(q_target=q_target, qd_target=qd_target, tau_ff=tau_ff)


def smooth(u: float) -> float:
    """Smoothstep on [0, 1] (0 below, 1 above)."""
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


@dataclass
class Phase:
    """One step of a mission: ``start(m, obs)`` once, then ``update(m, obs, tau) -> done`` every control step (tau:
    time in the phase); ``timeout`` [s] ends it anyway (logged)."""

    name: str
    update: object
    start: object = None
    timeout: float = 30.0


class PhasedMission:
    """A list of ``Phase`` s run in order on top of a ``Drive`` that keeps the chassis going (wheel speed loop,
    heading hold, the legs' feed-forward). Phases set ``self.v`` (the speed target, m/s) and ``self.crouch`` (the
    hull lowered on its legs, axles kept under the shoulders); subclasses add their tools in ``tool_command``.
    ``self.log`` records [t, phase, note]; ``self.memory`` keeps what the mission has seen."""

    def __init__(self, phases: list, *, accel: float = 0.6, drive_kw: dict | None = None, name: str = "mission"):
        self.phases = list(phases)
        self.accel = float(accel)
        self.drive_kw = dict(drive_kw or {})
        self.name = name

    def reset(self, lab, seed=None):
        self.lab = lab
        self.drive = Drive(0.0, ramp_s=0.0, name=f"{self.name}: chassis", **self.drive_kw)
        self.drive.reset(lab, seed)
        g = lab.robot.geometry
        self.axle_x, self.h_axle, self.L1, self.L2 = g["axle_x"], g["h_axle"], g["L1"], g["L2"]
        self.idx = {n: i for i, n in enumerate(lab.joint_names)}
        self.dof = {n: int(lab._jd[lab._joint_index[n]]) for n in lab.actuated_joints}
        self.v, self.crouch = 0.0, 0.0
        self.i, self.t0, self.started, self.t_prev = 0, 0.0, False, None
        self.log, self.state, self.memory = [], {}, {}
        self.tool_reset(lab)

    def tool_reset(self, lab):
        pass

    def tool_command(self, obs, q_t: dict, qd_t: dict, ff: dict) -> None:
        pass

    def settle_command(self, obs):
        return self._command(obs, 0.0)

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

    def drive_to(self, obs, x_stop: float, v_max: float, tol: float = 0.02) -> bool:
        """Brake to a stop at world x (forward or backward): v = ±min(v_max, √(2 a |Δx|), |v| + a·dt); done within
        ``tol`` [m] and below 5 cm/s."""
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
        cmd = self.drive.command(obs, self.v, dt)
        q_t, qd_t, ff = cmd.q_target, cmd.qd_target, cmd.tau_ff
        if self.crouch:
            for leg in orb.LEGS:
                js, jk, _ = orb.leg_joints(leg)
                ik = _ik(self.axle_x, self.h_axle - self.crouch, self.L1, self.L2)
                if ik is not None:
                    q_t[js], q_t[jk] = orb.angles_to_q(*ik)
        self.tool_command(obs, q_t, qd_t, ff)
        return Command(q_target=q_t, qd_target=qd_t, tau_ff=ff)


class Stand:
    """Hold the standing pose (with the gravity feed-forward), wheels at zero speed."""

    name = "stand"

    def __init__(self, gravity_ff: bool = True):
        self.drive = Drive(0.0, gravity_ff=gravity_ff, ramp_s=0.0, name="stand")

    def reset(self, lab, seed=None):
        self.drive.reset(lab, seed)

    def settle_command(self, obs):
        return self.drive.settle_command(obs)

    def __call__(self, obs):
        return self.drive(obs)
