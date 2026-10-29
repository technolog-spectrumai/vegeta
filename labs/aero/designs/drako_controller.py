"""Drako's flight controller and its cargo missions in ChironLab (``drako_scenario``).

**Flight** is Drongo's cascade (``drongo_controller.Flight``: position PID with the reference acceleration fed forward,
a geometric attitude controller on SO(3)) on six rotors: the mixer is the pseudo-inverse of the flat-six allocation
(``ogma.rotors.mix``: yaw dropped first, then roll/pitch scaled, when a rotor would leave [idle, T_max]). The
controller's mass is Drako's plus the package it carries (it knows the manifest); the position loop's integral takes
up a steady wind.

**Missions** (``Mission``): a list of ``Phase`` s, as Drongo's. ``leg(scene, a, b, ...)`` flies from pad ``a`` to pad
``b``: spool up, climb straight up to the leg's cruise altitude (the highest terrain within ``corridor_m`` of the
straight line plus ``clearance_m``), cruise at the speed for the load (40 km/h with a package, 60 km/h without), descend
over the pad, touch down slowly, spool down. ``pick(package)`` closes the latches on a package standing under the
cradle; ``drop(package)`` opens them after landing on the pad (the package stays on the ground). ``mission(scene, plan)``
chains the legs of ``plan.route``.

The controller knows Drako's state (ideal sensors), the map (terrain heights) and the pads; not the wind.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from vegeta.chiron import Command
from vegeta.labkit.control import Phase, smooth
from ogma import rotors
from ogma.frames import quat_to_R

from drongo_controller import Profile, Ref, _move
import drako_robot as dk

G = 9.81


class Flight:
    """Position → attitude → six thrusts. Gains as Drongo's, scaled by Drako's mass and inertia."""

    def __init__(self, *, omega_pos: float = 1.2, zeta_pos: float = 1.0, ki: float = 0.3, i_max: float = 60.0,
                 omega_att: float = 8.0, zeta_att: float = 0.8, omega_yaw: float = 2.5, tilt_max_deg: float = 32.0,
                 idle: float = 0.04):
        self.omega_pos, self.zeta_pos, self.ki, self.i_max = omega_pos, zeta_pos, ki, i_max
        self.omega_att, self.zeta_att, self.omega_yaw = omega_att, zeta_att, omega_yaw
        self.tilt_max, self.idle = math.radians(tilt_max_deg), idle

    def reset(self, lab):
        import mujoco
        self.air = lab.air
        self.m0 = float(lab.total_mass)
        d = mujoco.MjData(lab.model)
        d.qpos[:] = lab.data.qpos
        mujoco.mj_forward(lab.model, d)
        M = np.zeros((lab.model.nv, lab.model.nv))
        try:
            mujoco.mj_fullM(lab.model, d, M)
        except TypeError:
            mujoco.mj_fullM(lab.model, M, d.qM)
        a = int(lab.model.jnt_dofadr[lab._root_jnt])
        self.J = np.diag(M[a + 3:a + 6, a + 3:a + 6]).copy()
        w, z = self.omega_att, self.zeta_att
        self.kR = np.array([self.J[0] * w * w, self.J[1] * w * w, self.J[2] * self.omega_yaw ** 2])
        self.kW = np.array([2 * z * self.J[0] * w, 2 * z * self.J[1] * w, 2 * 0.9 * self.J[2] * self.omega_yaw])
        pos, spin = self.air.pos, self.air.spin
        self.A = np.vstack([np.ones(6), pos[:, 1], -pos[:, 0], spin * self.air.kM])
        self.Ainv = np.linalg.pinv(self.A)
        self.integral = np.zeros(3)
        self.last = {"F": np.zeros(3), "saturated": False}

    def reset_integral(self):
        self.integral[:] = 0.0

    @property
    def T_max(self) -> float:
        return self.air.T_max

    def attitude(self, R, omega, b3, yaw):
        b1c = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        b2 = np.cross(b3, b1c)
        b2 /= np.linalg.norm(b2)
        Rd = np.column_stack([np.cross(b2, b3), b2, b3])
        E = 0.5 * (Rd.T @ R - R.T @ Rd)
        return -self.kR * np.array([E[2, 1], E[0, 2], E[1, 0]]) - self.kW * omega

    def thrusts(self, obs, ref: Ref, m_payload: float, dt: float) -> np.ndarray:
        m = self.m0 + m_payload
        p, v = np.asarray(obs.base_pos, float), np.asarray(obs.body_linvel[0], float)
        R, omega = quat_to_R(obs.base_quat), np.asarray(obs.body_angvel[0], float)
        wp = self.omega_pos
        kx, kv = m * wp * wp, 2 * self.zeta_pos * m * wp
        ep, ev = p - ref.p, v - ref.v
        self.integral = np.clip(self.integral + kx * self.ki * ep * dt, -self.i_max, self.i_max)
        F = m * (ref.a + np.array([0.0, 0.0, G])) - kx * ep - kv * ev - self.integral
        F[2] = max(F[2], 0.25 * m * G)
        h = math.hypot(F[0], F[1])
        if h > F[2] * math.tan(self.tilt_max):
            F[:2] *= F[2] * math.tan(self.tilt_max) / h
        M = self.attitude(R, omega, F / np.linalg.norm(F), ref.yaw)
        T, sat = rotors.mix(self.Ainv, float(F @ R[:, 2]), M, self.idle * self.T_max, self.T_max)
        self.last = {"F": F, "saturated": sat}
        return T

    def ground(self, obs, share: float, m_payload: float = 0.0) -> np.ndarray:
        R, omega = quat_to_R(obs.base_quat), np.asarray(obs.body_angvel[0], float)
        M = self.attitude(R, omega, np.array([0.0, 0.0, 1.0]), 0.0) * share
        u = self.Ainv @ np.array([share * (self.m0 + m_payload) * G, M[0], M[1], M[2]])
        return np.clip(u, 0.0, self.T_max)


class Mission:
    """Phases in order on top of ``Flight``. Phases set ``ref``, ``mode`` ('air' / 'ground' with ``ground_share``),
    ``latches`` ('open' / 'closed') and ``payload`` (the package carried: its manifest entry). ``log``: [t, phase, note]."""

    def __init__(self, phases, flight: Flight, *, name="Drako cargo"):
        self.phases, self.flight, self.name = list(phases), flight, name

    def reset(self, lab, seed=None):
        self.lab = lab
        self.flight.reset(lab)
        self.cargo = lab.cargo
        self.leg_h = dk.design_params()["leg_height"] / 1000
        a = lab._root_qadr
        self.ref = Ref(p=np.array(lab.data.qpos[a:a + 3], float))
        self.mode, self.ground_share, self.payload, self.latches = "ground", 0.0, None, "open"
        self.i, self.t0, self.started, self.t_prev = 0, 0.0, False, None
        self.log, self.state = [], {}

    @property
    def phase(self):
        return self.phases[self.i].name if self.i < len(self.phases) else "finished"

    @property
    def finished(self):
        return self.i >= len(self.phases)

    @property
    def m_payload(self):
        return self.payload["mass_kg"] if self.payload else 0.0

    @property
    def mass(self):
        return self.flight.m0 + self.m_payload

    def note(self, obs, text):
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
            cmd = self.flight.thrusts(obs, self.ref, self.m_payload, dt)
        else:
            cmd = self.flight.ground(obs, self.ground_share, self.m_payload)
        self.lab.air.command = cmd
        q = 0.0 if self.latches == "closed" else dk.LATCH_OPEN_RAD
        return Command(q_target={f"{n}_hinge": q for n in dk.LATCHES})


@dataclass
class Plan:
    """How Drako flies (SI). ``route``: [(from pad, to pad, the package carried or None)];
    ``v_loaded`` / ``v_empty`` cruise speeds; ``clearance_m`` above the highest terrain in a ``corridor_m`` band;
    ``swap_battery_at`` the pads where the battery is swapped while landed."""

    route: list = field(default_factory=lambda: [("helipad", "hut", "medical"), ("hut", "helipad", "samples"),
                                                 ("helipad", "village", "groceries"), ("village", "helipad", None)])
    v_loaded: float = 40 / 3.6
    v_empty: float = 60 / 3.6
    a_cruise: float = 1.5
    v_climb: float = 4.0
    v_descent: float = 3.0
    a_vertical: float = 1.0
    approach_m: float = 8.0
    v_touchdown: float = 0.5
    clearance_m: float = 50.0
    corridor_m: float = 60.0
    latch_s: float = 1.5
    swap_battery_at: tuple = ("helipad",)


def cruise_altitude(scene, a, b, plan: Plan) -> float:
    """The highest terrain within ``plan.corridor_m`` of the straight line a → b, plus the clearance [m, world z]."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = float(np.linalg.norm(b - a))
    u = (b - a) / max(L, 1e-9)
    n = np.array([-u[1], u[0]])
    s = np.linspace(0.0, L, max(2, int(L / 10) + 1))
    pts = [a + u * si + n * o for si in s for o in (-plan.corridor_m, 0.0, plan.corridor_m)]
    P = np.array(pts)
    return float(np.max(scene.height(P[:, 0], P[:, 1]))) + plan.clearance_m


def mission(scene, plan: Plan | None = None) -> list:
    """The phases of ``plan.route`` on ``scene``."""
    plan = plan or Plan()
    phases = []
    add = lambda name, upd, start=None, timeout=900.0: phases.append(Phase(name, upd, start, timeout))  # noqa: E731

    def wait(T):
        return lambda m, obs, tau: tau >= T

    def takeoff():
        def start(m, obs):
            m.ref = Ref(p=np.asarray(obs.base_pos, float).copy())
            m.mode = "air"
            m.flight.reset_integral()

        def update(m, obs, tau):                   # spool up to the hover thrust, then lift 0.5 m
            m.ref.p = m.ref.p + np.array([0.0, 0.0, 0.3 * m.lab.control_dt]) * (tau > 0.5)
            return tau > 2.5
        return start, update

    def touchdown(label):
        def start(m, obs):
            m.state["held"] = 0.0

        def update(m, obs, tau):
            v = plan.v_touchdown
            m.ref.v, m.ref.a = np.array([0.0, 0.0, -v]), np.zeros(3)
            m.ref.p = m.ref.p + m.ref.v * m.lab.control_dt
            load = float(np.sum(obs.foot_normal_force))
            z, vz = float(obs.base_pos[2]), float(obs.body_linvel[0][2])
            stalled = vz > -v / 3 and m.ref.p[2] < z - 0.02 and tau > 0.5
            m.state["held"] = m.state["held"] + m.lab.control_dt if (load > 0.3 * m.mass * G or stalled) else 0.0
            if m.state["held"] >= 0.2:
                m.ref.p, m.ref.v = np.array([float(obs.base_pos[0]), float(obs.base_pos[1]), z - 0.03]), np.zeros(3)
                m.note(obs, f"landed {label} at {float(obs.base_pos[0]):.1f}, {float(obs.base_pos[1]):.1f} m")
                return True
            return False
        return start, update

    def spool_down(T=1.0):
        def start(m, obs):
            m.mode = "ground"
            m.state["s0"] = 0.6

        def update(m, obs, tau):
            m.ground_share = m.state["s0"] * (1.0 - smooth(tau / T))
            return tau >= T
        return start, update

    def latch(pkg):
        def start(m, obs):
            m.latches = "closed"

        def update(m, obs, tau):
            if tau < plan.latch_s:
                return False
            if m.cargo.latch(pkg, obs.t):
                m.payload = scene.packages[pkg]
                m.note(obs, f"{pkg} latched ({m.payload['mass_kg']:g} kg)")
            else:
                m.note(obs, f"{pkg} not under the cradle: nothing latched")
            return True
        return start, update

    def unlatch(pkg):
        def start(m, obs):
            m.latches = "open"
            m.cargo.unlatch(pkg, obs.t)
            m.note(obs, f"{pkg} released")

        def update(m, obs, tau):
            if tau >= plan.latch_s:
                m.payload = None
                return True
            return False
        return start, update

    def swap(pad):
        def start(m, obs):
            m.lab.air.swap_battery(obs.t)
            m.note(obs, f"battery swapped at the {pad}")
        return start, wait(30.0)

    carried = None
    for k, (a, b, pkg) in enumerate(plan.route):
        pa, pb = scene.pad(a), scene.pad(b)
        if pkg:
            st, up = latch(pkg)
            add(f"latch the {pkg} at the {a}", up, st)
        carried = pkg
        alt = cruise_altitude(scene, pa[:2], pb[:2], plan)
        v = plan.v_loaded if pkg else plan.v_empty
        st, up = takeoff()
        add(f"take off from the {a}", up, st, timeout=10.0)
        st, up = _move(lambda m, obs, alt=alt: np.array([m.ref.p[0], m.ref.p[1], alt]), plan.v_climb, plan.a_vertical,
                       settle_tol=1.0, max_extra=10.0)
        add(f"climb to {alt:.0f} m", up, st)
        st, up = _move(lambda m, obs, pb=pb, alt=alt: np.array([pb[0], pb[1], alt]), v, plan.a_cruise, settle_tol=2.0,
                       max_extra=30.0)
        add(f"cruise to the {b}" + (f" with the {pkg}" if pkg else ""), up, st)
        st, up = _move(lambda m, obs, pb=pb: np.array([pb[0], pb[1], pb[2] + m.leg_h + plan.approach_m]),
                       plan.v_descent, plan.a_vertical, settle_tol=0.5, max_extra=20.0)
        add(f"descend over the {b}", up, st)
        st, up = touchdown(f"on the {b}")
        add(f"touch down on the {b}", up, st, timeout=60.0)
        st, up = spool_down()
        add(f"spool down on the {b}", up, st)
        if carried:
            st, up = unlatch(carried)
            add(f"release the {carried} at the {b}", up, st)
        if b in plan.swap_battery_at and k < len(plan.route) - 1:
            st, up = swap(b)
            add(f"swap the battery at the {b}", up, st)
    return phases
