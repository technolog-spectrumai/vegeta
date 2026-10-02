"""The Sikarian Lobster's mission controller for ChironLab: walk on the bottom, work with the pincers, point the tail
and drive the thruster.

``Mission`` runs a list of ``Phase`` s (``onager_controller.Phase``: ``start(m, obs)``, ``update(m, obs, tau) ->
done``, a timeout) and holds a target for every joint:

* **legs** — a **tripod gait** (``TRIPODS``: FL·MR·RL against FR·ML·RR, duty 0.5): the swinging tripod lifts
  (hip pitch, a half sine of ``lift``) and swings its hip yaw from −``stride`` to +``stride``, the standing tripod
  sweeps back; the stride shrinks below ``V_NOMINAL`` (0.1 m/s); ``turn`` shortens one side's stride (a heading hold); ``m.walk = (v, turn)`` or None (stand);
  the gait finishes its half cycle before standing;
* **claws** — joint targets from a planar inverse kinematics of the cutter notch with the jaws at a set elevation
  (``claw_ik``: shoulder, elbow, wrist), straight-line moves of the notch in the hull frame, an integral term on the arm joints (their wet weight sags the light servos), jaw modes
  as the Onager Manus's (an angle, a grip with a set torque, or close with the drive's full torque);
* **tail and thruster** — ``thrust_line(theta)``: the two tail pitch joints that point the thrust ``theta`` above
  the hull's +x **and** put its line through the centre of gravity (no pitching moment: the jet vectors without
  tipping the robot); the yaw joints add a yaw moment; ``m.rpm`` goes to the Water hook (``lab.thruster_rpm``).

    import lobster_controller as lc
    mission = lc.Mission(lc.cut_and_enter(scene))
"""
from __future__ import annotations

import math

import numpy as np

from vegeta.chiron import Command

import lobster_robot as lr
from onager_controller import Phase, smooth

__all__ = ["Phase", "Mission", "smooth", "claw_ik", "claw_fk", "thrust_line", "yaw_of", "pitch_of", "world_to_hull", "hull_to_world"]


def _R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def yaw_of(q) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def pitch_of(q) -> float:
    """The hull's nose-up pitch [rad]."""
    w, x, y, z = q
    return -math.asin(max(-1.0, min(1.0, -2 * (x * z - w * y))))


def world_to_hull(obs, p):
    return _R(obs.base_quat).T @ (np.asarray(p, dtype=float) - np.asarray(obs.base_pos, dtype=float))


def hull_to_world(obs, p):
    return _R(obs.base_quat) @ np.asarray(p, dtype=float) + np.asarray(obs.base_pos, dtype=float)


# ----------------------------------------------------------------------------------------------- kinematics
def claw_ik(target, side: str, elevation: float = 0.0, g: dict | None = None):
    """(shoulder yaw, shoulder, elbow, wrist) [rad] that put the cutter notch at ``target`` (hull frame, m) with the
    jaws pointing ``elevation`` [rad] above horizontal; the notch is ``cutter`` along the jaw from its pin, the pin
    ``Lh`` from the wrist; elbow up. None when out of reach."""
    g = lr.geometry() if g is None else g
    sx, sy, sz = g["claw"][0], lr.CLAWS[side] * g["claw"][1], g["claw"][2]
    dx, dy, dz = target[0] - sx, target[1] - sy, target[2] - sz
    yaw = math.atan2(dy, dx)
    r = math.hypot(dx, dy)
    tip = g["Lh"] + g["cutter"]
    wr, wz = r - tip * math.cos(elevation), dz - tip * math.sin(elevation)       # the wrist
    L1, L2 = g["Lu"], g["Lp"]
    d = math.hypot(wr, wz)
    if d > L1 + L2 - 1e-9 or d < abs(L1 - L2) + 1e-9:
        return None
    beta = math.acos(max(-1.0, min(1.0, (L1 * L1 + d * d - L2 * L2) / (2 * L1 * d))))
    e1 = math.atan2(wz, wr) + beta                                   # upper arm elevation (elbow up)
    ex, ez = L1 * math.cos(e1), L1 * math.sin(e1)
    e2 = math.atan2(wz - ez, wr - ex)                                # palm elevation
    return yaw, e1, e1 - e2, e2 - elevation


def claw_fk(q, side: str, g: dict | None = None):
    """Cutter notch (hull frame) and the jaws' elevation for (shoulder yaw, shoulder, elbow, wrist)."""
    g = lr.geometry() if g is None else g
    yaw, sh, el, wr = q
    e1, e2, e3 = sh, sh - el, sh - el - wr
    tip = g["Lh"] + g["cutter"]
    r = g["Lu"] * math.cos(e1) + g["Lp"] * math.cos(e2) + tip * math.cos(e3)
    z = g["Lu"] * math.sin(e1) + g["Lp"] * math.sin(e2) + tip * math.sin(e3)
    sx, sy, sz = g["claw"][0], lr.CLAWS[side] * g["claw"][1], g["claw"][2]
    return np.array([sx + r * math.cos(yaw), sy + r * math.sin(yaw), sz + z]), e3


def thrust_line(theta: float, cg=(0.0, 0.0, 0.0), g: dict | None = None, best_effort: bool = False):
    """(tail1_pitch, tail2_pitch) [rad] with the thrust ``theta`` above the hull's +x (the force on the robot; the jet
    leaves the opposite way) and its line through ``cg`` (hull frame). The pitch hinges turn about +y (+ lifts the
    tip), so the thrust's elevation is −(q1 + q2); q1 is found by bisection within the tail's ranges
    (``lobster_robot.TAIL_PITCH_RANGE``; ``cg`` may be any point the line should pass). None if no line through the
    CG exists (the tail is too short to get below it for steep upward thrust); ``best_effort``: then the closest
    line the joint limits allow (a pitching moment remains)."""
    g = lr.geometry() if g is None else g
    r1 = [math.radians(v) for v in lr.TAIL_PITCH_RANGE[1]]
    r2 = [math.radians(v) for v in lr.TAIL_PITCH_RANGE[2]]
    total = -theta
    x0, _, z0 = g["tail0"]
    L1, L2h = g["L1"], g["L2"] + g["shroud_L"] / 2
    d = np.array([math.cos(theta), math.sin(theta)])                 # thrust direction in the xz plane (x, z)

    def miss(q1):
        q2 = total - q1
        hx = x0 - L1 * math.cos(q1) - L2h * math.cos(q1 + q2)
        hz = z0 + L1 * math.sin(q1) + L2h * math.sin(q1 + q2)
        rx, rz = cg[0] - hx, cg[2] - hz
        return rx * d[1] - rz * d[0]                                  # cross((cg − hub), d): 0 on the line

    lo, hi = max(r1[0], total - r2[1]), min(r1[1], total - r2[0])
    if lo > hi:
        return None
    qs = np.linspace(lo, hi, 121)
    f = np.array([miss(q) for q in qs])
    k = np.where(np.sign(f[:-1]) != np.sign(f[1:]))[0]
    if not len(k):
        if not best_effort:
            return None
        q1 = float(qs[int(np.argmin(np.abs(f)))])                    # the closest line the joints allow
        return q1, total - q1
    a, b = qs[k[0]], qs[k[0] + 1]
    for _ in range(40):
        mid = 0.5 * (a + b)
        if np.sign(miss(mid)) == np.sign(miss(a)):
            a = mid
        else:
            b = mid
    q1 = 0.5 * (a + b)
    return q1, total - q1


# ----------------------------------------------------------------------------------------------- the mission
V_NOMINAL = 0.10        # m/s: Nefri's full stride (±stride_deg at period 1.2 s); slower walking shortens the stride (the
                        # variant's ``mission["v_nominal"]`` once the Mission is reset on a lab)


class Mission:
    """Phases over the Lobster's joints (see the module). ``m.log`` [t, phase, note]; ``m.memory`` what it has seen."""

    def __init__(self, phases: list, *, name: str = "lobster mission", stride_deg: float | None = None,
                 lift_deg: float | None = None, period: float | None = None):
        self.phases, self.name = list(phases), name
        self._gait_kw = (stride_deg, lift_deg, period)
        self._set_gait(lr.NEFRI_V)

    def _set_gait(self, variant):
        """The gait's stride, lift and period: the constructor's values, else the variant's mission tuning."""
        ms = variant.mission
        stride_deg, lift_deg, period = self._gait_kw
        self.stride = math.radians(ms["stride_deg"] if stride_deg is None else stride_deg)
        self.lift = math.radians(ms["lift_deg"] if lift_deg is None else lift_deg)
        self.period = float(ms["period"] if period is None else period)
        self.v_nominal = float(ms["v_nominal"])

    # ---- ChironLab controller protocol
    def reset(self, lab, seed=None):
        self.lab = lab
        self.g = lab.robot.lobster
        self.variant = getattr(lab.robot, "variant", lr.NEFRI_V)
        self._set_gait(self.variant)
        self.idx = {n: i for i, n in enumerate(lab.joint_names)}
        self.kp_jaw = self.variant.gains["jaw"][0]
        self.stall_jaw = lr.act.get(self.variant.servo_keys["jaw"]).stall_Nm
        self.q = dict(lab.robot.nominal_qpos)
        self.claw = {s: np.array([self.q[f"{s}_shoulder_yaw"], self.q[f"{s}_shoulder"], self.q[f"{s}_elbow"], self.q[f"{s}_wrist"]])
                     for s in lr.CLAWS}
        self.jaw = {s: ("angle", 0.0) for s in lr.CLAWS}
        self.claw_i = {s: np.zeros(4) for s in lr.CLAWS}               # integral sag correction of the arm joints
        self.tail = np.zeros(4)                                       # tail1_yaw, tail1_pitch, tail2_yaw, tail2_pitch
        self.legs = {leg: [0.0, 0.0] for leg in lr.LEGS}             # yaw, lift
        self.rpm = 0.0
        self.walk = None
        self.gait_phase, self.cycles = 0.0, 0
        self.i, self.started, self.t0, self.t_prev = 0, False, 0.0, None
        self.log, self.state, self.memory = [], {}, {}

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
            self.walk, self.rpm = None, 0.0
        return self._command(obs, dt)

    @property
    def phase(self) -> str:
        return self.phases[self.i].name if self.i < len(self.phases) else "finished"

    @property
    def finished(self) -> bool:
        return self.i >= len(self.phases)

    # ---- the gait
    def _gait(self, dt):
        """Advance the tripod gait; when ``walk`` is None finish the half cycle and stand."""
        if self.walk is None and self.gait_phase == 0.0:
            for leg in lr.LEGS:
                self.legs[leg] = [0.0, 0.0]
            return
        v, turn = self.walk if self.walk is not None else (0.0, 0.0)
        before = self.gait_phase
        self.gait_phase += dt / self.period
        if self.walk is None and (before < 0.5 <= self.gait_phase or self.gait_phase >= 1.0):
            self.gait_phase = 0.0                                      # stop at the end of a half cycle
            for leg in lr.LEGS:
                self.legs[leg] = [0.0, 0.0]
            return
        self.gait_phase %= 1.0
        direction = 1.0 if v >= 0 else -1.0
        for k, tripod in enumerate(lr.TRIPODS):
            ph = (self.gait_phase + 0.5 * k) % 1.0
            swing = ph < 0.5
            u = ph / 0.5 if swing else (ph - 0.5) / 0.5
            for leg in tripod:
                side = lr.LEGS[leg][1]
                A = self.stride * min(1.0, abs(v) / self.v_nominal) * float(np.clip(1.0 - side * turn, 0.2, 1.8))   # turn > 0: shorter left strides
                if swing:
                    yaw = direction * A * (2 * smooth(u) - 1)
                    lift = self.lift * math.sin(math.pi * u)
                else:
                    yaw = direction * A * (1 - 2 * u)
                    lift = 0.0
                self.legs[leg] = [yaw, lift]

    def _command(self, obs, dt) -> Command:
        self._gait(dt)
        q_t, qd_t = {}, {}
        for leg in lr.LEGS:
            y, l = lr.leg_joints(leg)
            q_t[y], q_t[l] = self.legs[leg]
        for s in lr.CLAWS:
            names = lr.claw_joints(s)
            # the arm's wet weight (a worm drive and steel jaws in the palm) sags a light servo: an integral term
            # moves the target until the joint sits where it was asked to (k_i 4 /s, at most ±15°)
            err = np.array([float(v) - float(obs.q[self.idx[n]]) for n, v in zip(names[:4], self.claw[s])])
            self.claw_i[s] = np.clip(self.claw_i[s] + 4.0 * err * dt, -0.26, 0.26)
            for n, v, c in zip(names[:4], self.claw[s], self.claw_i[s]):
                q_t[n] = float(v + c)
            mode, val = self.jaw[s]
            for n in names[4:]:
                if mode == "angle":
                    q_t[n] = float(val)
                elif mode == "grip":
                    q_t[n] = float(obs.q[self.idx[n]]) - float(val) / self.kp_jaw
                else:                                                     # "close": the drive's full torque
                    q_t[n] = float(obs.q[self.idx[n]]) - 2.0 * self.stall_jaw / self.kp_jaw
        for n, v in zip(lr.TAIL_JOINTS, self.tail):
            q_t[n] = float(v)
        qd_t = {n: 0.0 for n in q_t}
        self.lab.thruster_rpm = self.rpm
        return Command(q_target=q_t, qd_target=qd_t)

    # ---- helpers for phases
    def cg_hull(self, obs):
        """The centre of gravity in the hull frame (MuJoCo's subtree COM of the root)."""
        return world_to_hull(obs, np.asarray(obs.com, dtype=float))

    def steer(self, obs, pitch_ref: float, yaw: float = 0.0, kp: float = 2.0, kd: float = 0.6) -> None:
        """Swim attitude: the tail pitches the thrust line off the CG to make a pitching moment that holds the
        hull's pitch at ``pitch_ref`` [rad, nose up]: thrust pointed down at the stern lifts the nose. Both pitch
        joints take half; ``yaw`` [rad] over the yaw joints (+ swings the tip right: the nose turns right)."""
        pitch = pitch_of(obs.base_quat)
        rate = float(np.asarray(obs.body_angvel)[0][1]) if np.ndim(obs.body_angvel) > 1 else 0.0   # about the hull's y: nose down +
        delta = float(np.clip(kp * (pitch_ref - pitch) - kd * (-rate), -0.9, 0.9))
        self.tail = np.array([yaw / 2, delta / 2, yaw / 2, delta / 2])

    def aim(self, obs, theta: float, yaw: float = 0.0, world: bool = False, through=None) -> bool:
        """Point the thrust ``theta`` above the hull's +x (``world``: above the horizon — the hull's pitch is taken
        out) through the CG (or ``through``, a hull-frame point), or as close as the tail allows; ``yaw`` [rad] split
        over the two yaw joints. Returns whether the line passes through the point."""
        if world:
            theta = theta - pitch_of(obs.base_quat)
        point = self.cg_hull(obs) if through is None else np.asarray(through, dtype=float)
        exact = thrust_line(theta, point, self.g)
        sol = exact or thrust_line(theta, point, self.g, best_effort=True)
        if sol is None:
            return False
        self.tail = np.array([yaw / 2, sol[0], yaw / 2, sol[1]])
        return exact is not None


# ----------------------------------------------------------------------------------------------- phase pieces
def wait(T):
    return lambda m, obs, tau: tau >= T


def walk_to(x_stop, v=0.10, k_heading=1.5, y_line=0.0, k_lat=2.0):
    """Walk along +x until the hull is at ``x_stop`` (world), holding the heading and the line y = ``y_line``. The
    gait finishes its half cycle after the stop is called (~v·period/4 further): the call comes that much early."""
    def update(m, obs, tau):
        x = float(obs.base_pos[0])
        err = -yaw_of(obs.base_quat) * k_heading + (y_line - float(obs.base_pos[1])) * k_lat
        m.walk = (v, float(np.clip(err, -0.6, 0.6)))           # turn > 0 turns left
        if x >= x_stop - abs(v) * m.period * 0.25 or m.state.get("stopping"):
            m.state["stopping"] = True
            m.walk = None
            return m.gait_phase == 0.0
        return False
    return update


def creep_to(x_stop, v=0.04, tol=0.012, k_heading=1.5, blocked_after=1.5, blocked_mm=3.0):
    """Short steps forward or back (|v|, the tripod gait) until the hull is within ``tol`` of ``x_stop`` — or until
    it is **blocked**: within 5 cm of the stop but not advancing ``blocked_mm`` in ``blocked_after`` s (the body
    pressed against the rope it is creeping onto: the legs cannot push it further, and need not)."""
    def update(m, obs, tau):
        x = float(obs.base_pos[0])
        dx = x_stop - x
        hist = m.state.setdefault("creep_x", [])
        hist.append((tau, x))
        past = [xx for tt, xx in hist if tt <= tau - blocked_after]
        blocked = bool(past) and abs(dx) < 0.05 and abs(x - past[-1]) < blocked_mm / 1000
        if blocked and not m.state.get("stopping"):
            m.log.append([float(obs.t), m.phases[m.i].name, f"blocked {dx * 1000:.0f} mm short: stop"])
        if m.state.get("stopping") or abs(dx) < tol or blocked:
            m.state["stopping"] = True
            m.walk = None
            return m.gait_phase == 0.0
        m.walk = (math.copysign(v, dx), float(np.clip(-yaw_of(obs.base_quat) * k_heading, -0.6, 0.6)))
        return False
    return update


def stand(T=0.6):
    def start(m, obs):
        m.walk = None
    return start, wait(T)


def claw_line(side, p0_fn, p1_fn, T, elevation: float = 0.0):
    """Move the notch of claw ``side`` along a straight line (hull frame) over T s, the jaws at ``elevation``."""
    def start(m, obs):
        m.state.update(p0=np.asarray(p0_fn(m, obs), dtype=float), p1=np.asarray(p1_fn(m, obs), dtype=float))

    def update(m, obs, tau):
        u = smooth(tau / T)
        q = claw_ik(m.state["p0"] + (m.state["p1"] - m.state["p0"]) * u, side, elevation, m.g)
        if q is not None:
            m.claw[side] = np.asarray(q)
        elif not m.state.get("unreachable"):
            m.state["unreachable"] = True
            m.log.append([float(obs.t), m.phase, f"{side} claw: target out of reach, holding"])
        return tau >= T
    return start, update


def claw_joint_move(side, q_fn, T):
    def start(m, obs):
        q1 = q_fn(m, obs)
        if q1 is None:                                    # out of reach: stay, and say so
            m.log.append([float(obs.t), m.phase, f"{side} claw: target out of reach, holding"])
            q1 = m.claw[side].copy()
        m.state.update(q0=m.claw[side].copy(), q1=np.asarray(q1, dtype=float))

    def update(m, obs, tau):
        m.claw[side] = m.state["q0"] + (m.state["q1"] - m.state["q0"]) * smooth(tau / T)
        return tau >= T
    return start, update


def jaw(side, mode, val=None):
    def start(m, obs):
        m.jaw[side] = (mode, val)
    return start


def stow_q(m, obs):
    return (lr.STOW["shoulder_yaw"], lr.STOW["shoulder"], lr.STOW["elbow"], lr.STOW["wrist"])


def notch_now(side):
    return lambda m, obs: claw_fk(m.claw[side], side, m.g)[0]


def both(*starts):
    fs = tuple(starts)

    def start(m, obs):
        for f in fs:
            if f is not None:
                f(m, obs)
    return start
