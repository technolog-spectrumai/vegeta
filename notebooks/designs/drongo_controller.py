"""Drongo's flight controller and its delivery missions in ChironLab (notebook 08b).

**Flight controller** (``Flight``). It cascades two loops and a mixer:

* *position* — the force the rotors must make, ``F = m (a_ref + g ẑ) − k_x e_p − k_v e_v − k_i ∫e_p``: a PD loop on the
  position and velocity errors to a reference (``Ref``: position, velocity, acceleration, yaw) with the reference
  acceleration fed forward and a small integral term (it absorbs what the mass estimate misses). ``m`` is Drongo's
  mass plus the item it carries; the tilt is limited to ``tilt_max_deg``;
* *attitude* — the frame's z axis turned onto ``F`` (a geometric controller on SO(3), Lee, Leok & McClamroch 2010):
  torques ``M = −k_R e_R − k_Ω Ω`` with gains from Drongo's inertia (MuJoCo's mass matrix);
* *mixer* — collective thrust and the three torques to the four rotor thrusts of the X layout (arms at 45°); out of
  range, the yaw torque is dropped first, then the roll/pitch torques are scaled until every rotor is between its
  idle and maximum thrust.

**Missions** (``Mission``, a list of ``Phase`` s like ``onager_controller.PhasedMission``): each phase moves the
reference (``move``: a straight line with a jerk-smoothed trapezoidal speed profile; ``touchdown``: down at a slow
speed until the skids carry Drongo), switches between flying and standing on the ground (``spool_down``: the rotors
ramp to zero once landed; ``takeoff``: they spool up and the reference climbs), or works the pincer (``grip``: each jaw
pushes with a set force, as ``onager_manus_controller`` grips; ``release``: open). ``delivery(scene, plan)`` builds a
whole mission: take off from the kitchen pad, land over the potato, grip it, climb, fly to the drop zone and **drop**
it into the hungry people's net from at most ``plan.release_above_net`` (variant ``drop``) or **place** it: land on the
zone so the potato touches it, let go (variant ``place``); back for the cream; the same; home.

The controller knows only what its sensors and the order tell it: Drongo's state (ideal), where the items, the net and
the zone are, and what each item weighs. Whether an item stays in the jaws, falls into the net or lands gently is
physics (the scene's hooks judge it).

    import drongo_controller as dc
    mission = dc.Mission(dc.delivery(scene, dc.Plan()), dc.Flight(prop))
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from vegeta.chiron import Command

import drongo_robot as dr
from onager_controller import Phase, smooth

__all__ = ["Ref", "Profile", "Flight", "Mission", "Plan", "delivery", "Phase", "smooth", "quat_to_R"]


def quat_to_R(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


@dataclass
class Ref:
    """The reference the flight controller follows (world, SI)."""

    p: np.ndarray = field(default_factory=lambda: np.zeros(3))
    v: np.ndarray = field(default_factory=lambda: np.zeros(3))
    a: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw: float = 0.0


class Profile:
    """Distance along a straight move against time: a trapezoidal speed profile (``v_max``, ``a_max``), its position
    averaged over a sliding window of ``smooth_s`` so the acceleration ramps instead of stepping (a jerk limit of
    a_max / smooth_s). ``at(t)`` → (s, v, a); ``T`` its duration."""

    def __init__(self, dist: float, v_max: float, a_max: float, smooth_s: float = 0.3, dt: float = 0.005):
        self.dist = float(max(dist, 0.0))
        if self.dist < 1e-9:
            self.T, self.t, self.s, self.v, self.a = 0.0, np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2)
            return
        t_acc = v_max / a_max
        if a_max * t_acc ** 2 >= self.dist:                                  # triangular: never reaches v_max
            t_acc = math.sqrt(self.dist / a_max)
            v_top, t_flat = a_max * t_acc, 0.0
        else:
            v_top, t_flat = v_max, (self.dist - a_max * t_acc ** 2) / v_max
        T0 = 2 * t_acc + t_flat
        n_win = max(1, int(round(smooth_s / dt)))
        t = np.arange(0.0, T0 + n_win * dt + dt, dt)
        tt = np.minimum(t, T0)
        s = np.where(tt < t_acc, 0.5 * a_max * tt ** 2,
                     np.where(tt < t_acc + t_flat, 0.5 * a_max * t_acc ** 2 + v_top * (tt - t_acc),
                              self.dist - 0.5 * a_max * (T0 - tt) ** 2))
        pad = np.concatenate([np.zeros(n_win - 1), s])
        s = np.convolve(pad, np.ones(n_win) / n_win, mode="valid")          # causal moving average
        self.t, self.s = t, s
        self.v = np.gradient(s, dt)
        self.a = np.gradient(self.v, dt)
        self.T = float(t[-1])

    def at(self, t: float) -> tuple:
        if self.T <= 0 or t >= self.T:
            return self.dist, 0.0, 0.0
        return (float(np.interp(t, self.t, self.s)), float(np.interp(t, self.t, self.v)),
                float(np.interp(t, self.t, self.a)))


class Flight:
    """Position → attitude → rotor thrusts (see the module). ``omega_pos`` [rad/s] and ``zeta_pos`` set the position
    loop (k_x = m ω², k_v = 2 ζ m ω), ``omega_att`` / ``zeta_att`` the roll-pitch loop and ``omega_yaw`` the yaw
    loop on Drongo's inertia; ``ki`` [1/s] the integral's rate relative to k_x, ``i_max`` [N] its bound;
    ``tilt_max_deg`` the largest tilt the position loop may ask for; ``idle`` the thrust share every rotor keeps in
    the air."""

    def __init__(self, prop: dr.Propulsion, *, omega_pos: float = 2.4, zeta_pos: float = 1.0, ki: float = 0.4,
                 i_max: float = 2.0, omega_att: float = 16.0, zeta_att: float = 0.8, omega_yaw: float = 5.0,
                 tilt_max_deg: float = 35.0, idle: float = 0.03):
        self.prop = prop
        self.omega_pos, self.zeta_pos, self.ki, self.i_max = omega_pos, zeta_pos, ki, i_max
        self.omega_att, self.zeta_att, self.omega_yaw = omega_att, zeta_att, omega_yaw
        self.tilt_max = math.radians(tilt_max_deg)
        self.idle = idle

    def reset(self, lab):
        import mujoco

        robot = lab.robot
        self.m0 = float(lab.total_mass)
        d = mujoco.MjData(lab.model)
        d.qpos[:] = lab.data.qpos
        mujoco.mj_forward(lab.model, d)
        M = np.zeros((lab.model.nv, lab.model.nv))
        try:
            mujoco.mj_fullM(lab.model, d, M)                                    # MuJoCo ≥ 3.5: (m, d, dst)
        except TypeError:
            mujoco.mj_fullM(lab.model, M, d.qM)                                 # older: (m, dst, qM)
        a = int(lab.model.jnt_dofadr[lab._root_jnt])
        self.J = np.diag(M[a + 3:a + 6, a + 3:a + 6]).copy()                 # roll, pitch, yaw inertia [kg·m²]
        w, z = self.omega_att, self.zeta_att
        self.kR = np.array([self.J[0] * w * w, self.J[1] * w * w, self.J[2] * self.omega_yaw ** 2])
        self.kW = np.array([2 * z * self.J[0] * w, 2 * z * self.J[1] * w, 2 * 0.9 * self.J[2] * self.omega_yaw])
        pos, spin = dr.rotor_layout(robot.params)
        self.A = np.vstack([np.ones(4), pos[:, 1], -pos[:, 0], spin * self.prop.torque_per_thrust_m])
        self.Ainv = np.linalg.inv(self.A)
        self.integral = np.zeros(3)
        self.last = {"F": np.zeros(3), "tilt_deg": 0.0, "T": np.zeros(4), "saturated": False}

    def reset_integral(self):
        self.integral[:] = 0.0

    def mix(self, T: float, M: np.ndarray) -> tuple:
        """Rotor thrusts [N] for a collective thrust ``T`` and body torques ``M``; True when they had to be limited."""
        lo, hi = self.idle * self.prop.max_thrust_N, self.prop.max_thrust_N
        T = float(np.clip(T, 4 * lo, 4 * hi))
        u = self.Ainv @ np.array([T, M[0], M[1], M[2]])
        if lo - 1e-9 <= u.min() and u.max() <= hi + 1e-9:
            return u, False
        u = self.Ainv @ np.array([T, M[0], M[1], 0.0])                        # drop the yaw torque first
        if not (lo - 1e-9 <= u.min() and u.max() <= hi + 1e-9):
            k_lo, k_hi = 0.0, 1.0                                             # then scale roll and pitch
            for _ in range(20):
                k = 0.5 * (k_lo + k_hi)
                v = self.Ainv @ np.array([T, k * M[0], k * M[1], 0.0])
                if lo - 1e-9 <= v.min() and v.max() <= hi + 1e-9:
                    k_lo = k
                else:
                    k_hi = k
            u = self.Ainv @ np.array([T, k_lo * M[0], k_lo * M[1], 0.0])
        return np.clip(u, lo, hi), True

    def attitude(self, R: np.ndarray, omega: np.ndarray, b3: np.ndarray, yaw: float) -> np.ndarray:
        """Body torques [N·m] that turn the frame's z axis onto ``b3`` at heading ``yaw``."""
        b1c = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        b2 = np.cross(b3, b1c)
        b2 /= np.linalg.norm(b2)
        Rd = np.column_stack([np.cross(b2, b3), b2, b3])
        E = 0.5 * (Rd.T @ R - R.T @ Rd)
        eR = np.array([E[2, 1], E[0, 2], E[1, 0]])
        return -self.kR * eR - self.kW * omega

    def thrusts(self, obs, ref: Ref, m_payload: float, dt: float) -> np.ndarray:
        """The four rotor thrusts [N] to follow ``ref`` carrying ``m_payload`` [kg]."""
        m = self.m0 + m_payload
        p = np.asarray(obs.base_pos, dtype=float)
        v = np.asarray(obs.body_linvel[0], dtype=float)
        R = quat_to_R(obs.base_quat)
        omega = np.asarray(obs.body_angvel[0], dtype=float)
        wp = self.omega_pos
        kx, kv = m * wp * wp, 2 * self.zeta_pos * m * wp
        ep, ev = p - ref.p, v - ref.v
        self.integral = np.clip(self.integral + kx * self.ki * ep * dt, -self.i_max, self.i_max)
        F = m * (ref.a + np.array([0.0, 0.0, dr.G])) - kx * ep - kv * ev - self.integral
        F[2] = max(F[2], 0.25 * m * dr.G)
        h = math.hypot(F[0], F[1])
        if h > F[2] * math.tan(self.tilt_max):                                # tilt limit: shorten the horizontal part
            F[:2] *= F[2] * math.tan(self.tilt_max) / h
        b3 = F / np.linalg.norm(F)
        M = self.attitude(R, omega, b3, ref.yaw)
        T, sat = self.mix(float(F @ R[:, 2]), M)
        self.last = {"F": F, "tilt_deg": math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2])))), "T": T, "saturated": sat}
        return T

    def ground(self, obs, share: float) -> np.ndarray:
        """On the skids: a fraction ``share`` of the hover thrust, the attitude held level."""
        R = quat_to_R(obs.base_quat)
        omega = np.asarray(obs.body_angvel[0], dtype=float)
        M = self.attitude(R, omega, np.array([0.0, 0.0, 1.0]), 0.0) * share
        u = self.Ainv @ np.array([share * self.m0 * dr.G, M[0], M[1], M[2]])
        return np.clip(u, 0.0, self.prop.max_thrust_N)


class Mission:
    """A list of ``Phase`` s run in order (``update(m, obs, tau) -> done``, ``start(m, obs)``, ``timeout``), on top of
    ``Flight``. Phases set ``self.ref`` (the reference), ``self.mode`` ('air' or 'ground'; on the ground
    ``self.ground_share`` of the hover thrust), ``self.jaws`` (('open', q) or ('grip', F per pad)) and
    ``self.payload`` (the item carried, from ``dr.ITEMS``). ``self.log`` holds [t, phase, note]; the rotor command
    goes to ``lab.rotors`` (the ``drongo_robot.Rotors`` hook)."""

    def __init__(self, phases: list, flight: Flight, *, name: str = "Drongo delivery"):
        self.phases, self.flight, self.name = list(phases), flight, name

    def reset(self, lab, seed=None):
        self.lab = lab
        self.flight.reset(lab)
        self.rotors = lab.rotors
        self.idx = {n: i for i, n in enumerate(lab.joint_names)}
        self.kp_jaw = dr.JAW_GAINS[0]
        g = lab.robot.geometry
        self.skid_h, self.travel, self.ready = g["skid_h"], g["travel"], g["ready"]
        a = lab._root_qadr
        self.ref = Ref(p=np.array(lab.data.qpos[a:a + 3], dtype=float))
        self.mode, self.ground_share, self.payload = "ground", 0.0, None
        self.jaws = ("open", self.ready)
        self.i, self.t0, self.started, self.t_prev = 0, 0.0, False, None
        self.log, self.state, self.memory = [], {}, {}

    @property
    def phase(self) -> str:
        return self.phases[self.i].name if self.i < len(self.phases) else "finished"

    @property
    def finished(self) -> bool:
        return self.i >= len(self.phases)

    @property
    def mass(self) -> float:
        return self.flight.m0 + (self.payload["mass_kg"] if self.payload else 0.0)

    def note(self, obs, text: str):
        self.log.append([float(obs.t), self.phase, text])

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
        if self.mode == "air":
            cmd = self.flight.thrusts(obs, self.ref, self.payload["mass_kg"] if self.payload else 0.0, dt)
        else:
            cmd = self.flight.ground(obs, self.ground_share)
        self.rotors.command = cmd
        return self._jaw_command(obs)

    def _jaw_command(self, obs) -> Command:
        """Open: both jaws to ``val``. Grip: both push inwards with ``val`` [N] from their mean position — the pinion
        between the racks keeps the jaws symmetric, so a jaw ahead of the other is pulled back with the servo's
        stiffness."""
        mode, val = self.jaws
        if mode == "open":
            return Command(q_target={j: float(val) for j in dr.JAW_JOINTS})
        mean = 0.5 * sum(float(obs.q[self.idx[j]]) for j in dr.JAW_JOINTS)
        return Command(q_target={j: mean - float(val) / self.kp_jaw for j in dr.JAW_JOINTS})


# ----------------------------------------------------------------------------------------------- the delivery
@dataclass
class Plan:
    """How Drongo flies the delivery (SI). ``v_cruise``/``a_cruise`` along the way at ``scene.cruise_alt``;
    ``v_climb``, ``v_descent``, ``a_vertical``; ``v_touchdown`` the last ``touchdown_from`` metres onto the ground
    (``v_place`` onto the drop zone with an item: calmly); ``hop_alt`` the hop from the kitchen pad to the first item;
    ``grip`` the squeeze per pad [N] per item; ``grip_s`` / ``release_s`` the time the jaws get; ``hover_s`` the steady
    hover before a drop; ``release_above_net`` the item's bottom above the net when it is let go (the net takes at most
    5 m)."""

    v_cruise: float = 6.0
    a_cruise: float = 3.0
    v_climb: float = 2.5
    v_descent: float = 2.0
    a_vertical: float = 2.0
    v_touchdown: float = 0.35
    v_place: float = 0.25
    touchdown_from: float = 0.6
    hop_alt: float = 1.5
    grip: dict = field(default_factory=lambda: dict(dr.GRIP_N))
    grip_s: float = 0.8
    release_s: float = 0.8
    hover_s: float = 1.0
    release_above_net: float = 4.0
    items: tuple = ("potato", "cream")


def _move(target_fn, v_max, a_max, settle_tol=0.12, max_extra=4.0):
    """Phase pieces: a straight move of the reference from where it is to ``target_fn(m, obs)``; done when the
    profile has ended and Drongo is within ``settle_tol`` of the target and slow (or ``max_extra`` s later)."""
    def start(m, obs):
        p1 = np.asarray(target_fn(m, obs), dtype=float)
        p0 = m.ref.p.copy()
        d = p1 - p0
        L = float(np.linalg.norm(d))
        m.state.update(p0=p0, p1=p1, u=d / L if L > 1e-9 else np.zeros(3), prof=Profile(L, v_max, a_max))

    def update(m, obs, tau):
        s, v, a = m.state["prof"].at(tau)
        u = m.state["u"]
        m.ref.p, m.ref.v, m.ref.a = m.state["p0"] + u * s, u * v, u * a
        if tau < m.state["prof"].T:
            return False
        err = float(np.linalg.norm(np.asarray(obs.base_pos) - m.state["p1"]))
        slow = float(np.linalg.norm(obs.body_linvel[0])) < 0.25
        return (err < settle_tol and slow) or tau > m.state["prof"].T + max_extra
    return start, update


def delivery(scene, plan: Plan | None = None) -> list:
    """The phases of the whole delivery on ``scene`` (``drongo_scenario.Scene``; variant 'drop' or 'place')."""
    plan = plan or Plan()
    phases = []
    add = lambda name, upd, start=None, timeout=60.0: phases.append(Phase(name, upd, start, timeout))  # noqa: E731
    vc, ac, av = plan.v_cruise, plan.a_cruise, plan.a_vertical

    def landed_z(m, x, y):
        return scene.ground(x, y) + m.skid_h

    def above(x, y, alt):
        return lambda m, obs: np.array([x, y, scene.ground(x, y) + alt])

    def jaws(mode, val):
        def start(m, obs):
            m.jaws = (mode, val)
        return start

    def wait(T):
        return lambda m, obs, tau: tau >= T

    def takeoff(alt):
        """Spool up and climb straight up to ``alt`` above the ground here."""
        st, up = _move(lambda m, obs: np.array([m.ref.p[0], m.ref.p[1], scene.ground(m.ref.p[0], m.ref.p[1]) + alt]),
                       plan.v_climb, av)

        def start(m, obs):
            m.ref = Ref(p=np.asarray(obs.base_pos, dtype=float).copy())
            m.mode = "air"
            m.flight.reset_integral()
            st(m, obs)
        return start, up

    def touchdown(v, what="landed"):
        """Down at ``v`` until something stops Drongo: the skids carry it (their load > 25 % of its weight), or its
        descent has stalled — it sinks slower than a third of ``v`` while the reference has gone 15 mm below it, as a
        flight controller's land detector judges it — for 0.1 s. Then the reference stops 20 mm below Drongo."""
        def start(m, obs):
            m.state["held"] = 0.0

        def update(m, obs, tau):
            m.ref.v = np.array([0.0, 0.0, -v])
            m.ref.a = np.zeros(3)
            m.ref.p = m.ref.p + m.ref.v * m.lab.control_dt
            load = float(np.sum(obs.foot_normal_force))
            z, vz = float(obs.base_pos[2]), float(obs.body_linvel[0][2])
            stalled = vz > -v / 3 and m.ref.p[2] < z - 0.015 and tau > 0.3
            m.state["held"] = m.state["held"] + m.lab.control_dt if (load > 0.25 * m.mass * dr.G or stalled) else 0.0
            if m.state["held"] >= 0.1:
                m.ref.p = np.array([float(obs.base_pos[0]), float(obs.base_pos[1]), z - 0.02])
                m.ref.v = np.zeros(3)
                m.note(obs, f"{what} at {float(obs.base_pos[0]):.2f}, {float(obs.base_pos[1]):.2f} m "
                            f"(skids {load:.1f} N{', descent stalled' if stalled and load <= 0.25 * m.mass * dr.G else ''})")
                return True
            return False
        return start, update

    def spool_down(T=0.5):
        def start(m, obs):
            m.mode = "ground"
            m.state["share0"] = 0.6

        def update(m, obs, tau):
            m.ground_share = m.state["share0"] * (1.0 - smooth(tau / T))
            return tau >= T
        return start, update

    def grip(item):
        F = plan.grip[item]

        def start(m, obs):
            m.jaws = ("grip", F)

        def update(m, obs, tau):
            if tau < plan.grip_s:
                return False
            q = [float(obs.q[m.idx[j]]) for j in dr.JAW_JOINTS]
            gap = dr.pad_gap(q, m.lab.robot.geometry)
            want = dr.ITEMS[item]["diameter_m"]
            if abs(gap - want) < 0.012:
                m.payload = dr.ITEMS[item]
                m.note(obs, f"{item} in the jaws (gap {gap * 1000:.0f} mm, squeeze {F:.0f} N per pad)")
            else:
                m.note(obs, f"no {item} in the jaws (gap {gap * 1000:.0f} mm, {want * 1000:.0f} mm expected)")
            return True
        return start, update

    def release(item):
        def start(m, obs):
            m.jaws = ("open", m.travel)
            m.note(obs, f"release the {item} at {float(obs.base_pos[2]) - m.skid_h:.2f} m (its bottom)")

        def update(m, obs, tau):
            if tau >= plan.release_s:
                m.payload = None
                return True
            return False
        return start, update

    def land_at(x, y, v_last, label):
        """From the reference's height: down at the descent speed to ``touchdown_from`` above landing, then
        ``v_last`` until the skids carry, then spool down."""
        st, up = _move(lambda m, obs: np.array([x, y, landed_z(m, x, y) + plan.touchdown_from]), plan.v_descent, av)
        add(f"descend {label}", up, st)
        st, up = touchdown(v_last)
        add(f"touch down {label}", up, st, timeout=20.0)
        st, up = spool_down()
        add(f"spool down {label}", up, st)

    hx, hy = scene.home
    zx, zy = scene.zone
    m_open = scene.robot_geometry["travel"]
    first = True
    for item in plan.items:
        ix, iy = scene.item_at(item)
        if first:
            st, up = takeoff(plan.hop_alt)
            add("take off", up, st)
            st, up = _move(above(ix, iy, plan.hop_alt), vc / 2, ac)
            add(f"hop over the {item}", up, st)
            first = False
        else:
            st, up = _move(above(ix, iy, scene.cruise_alt), vc, ac)
            add(f"fly back to the {item}", up, st)
        add(f"open the jaws over the {item}", wait(0.3), jaws("open", m_open))
        land_at(ix, iy, plan.v_touchdown, f"over the {item}")
        st, up = grip(item)
        add(f"grip the {item}", up, st)
        st, up = takeoff(scene.cruise_alt)
        add(f"climb with the {item}", up, st)
        if scene.variant == "drop":
            st, up = _move(above(zx, zy, scene.cruise_alt), vc, ac)
            add(f"fly the {item} to the net", up, st)
            z_rel = scene.net_height + plan.release_above_net
            st, up = _move(lambda m, obs: np.array([zx, zy, z_rel + m.skid_h]), plan.v_descent, av, settle_tol=0.08)
            add(f"down to the release height", up, st)
            add("steady hover", wait(plan.hover_s))
            st, up = release(item)
            add(f"drop the {item}", up, st)
            st, up = _move(above(zx, zy, scene.cruise_alt), plan.v_climb, av)
            add("climb back to cruise", up, st)
        else:
            sx, sy = scene.spot(item)
            st, up = _move(above(sx, sy, scene.cruise_alt), vc, ac)
            add(f"fly the {item} to the zone", up, st)
            st, up = _move(lambda m, obs, x=sx, y=sy: np.array([x, y, landed_z(m, x, y) + plan.touchdown_from]),
                           plan.v_descent, av)
            add(f"descend with the {item} to the zone", up, st)
            st, up = touchdown(plan.v_place, f"the {item} touches the zone")
            add(f"lower the {item} onto the zone", up, st, timeout=20.0)
            add("hold", wait(0.3))
            st, up = release(item)
            add(f"let go of the {item}", up, st)
            st, up = _move(lambda m, obs: np.array([m.ref.p[0], m.ref.p[1], scene.ground(*m.ref.p[:2]) + scene.cruise_alt]),
                           plan.v_climb, av)
            add("climb back to cruise", up, st)
    st, up = _move(above(hx, hy, scene.cruise_alt), vc, ac)
    add("fly home", up, st)
    add("close the jaws", wait(0.3), jaws("open", scene.robot_geometry["ready"]))
    land_at(hx, hy, plan.v_touchdown, "on the kitchen pad")
    return phases
