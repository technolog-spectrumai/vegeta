"""Cleopatra's controllers for ChironLab: the fixed gait of protocol §3 and the adaptive (Tegotae) gait of §9.1.

One class, ``CleopatraController``, whose only difference between the two is the load-feedback gain ``sigma``:
``fixed(v)`` is σ = 0, ``adaptive(v)`` is σ = 0.6 rad/(N·s). The same code drives every body treatment (it is
terrain-blind and does not know whether the body joints are locked).

**Phases** (``vegeta.chiron.gaits.PhaseGenerator``). Each segment walks a lateral-sequence wave on its four legs —
rear-left, front-left, rear-right, front-right at cycle fractions 0, 0.25, 0.5, 0.75, duty 0.75 — and segment i
(1 = behind the head) is shifted by ``(3 − i)/12`` of a stride (notebook 18's metachronal wave). A trial's seed
adds one random offset, uniform in a stride, to every leg (protocol §4). Stride frequency = v / stride. With
σ > 0 every leg integrates ``dφ/dt = ω(φ) − σ N cos φ`` with its foot's normal ground force N (the lab's contact
force: a contact sensor), swing φ ∈ [0, π), stance φ ∈ [π, 2π); the initial phases are the fixed controller's, so
σ = 0 *is* the fixed controller. ``min_rate`` (default 0.25 for σ > 0) bounds the phase rate below at
``min_rate · ω(φ)``: a loaded leg is slowed but never frozen (see ``ControllerParams``).

**Foot targets** (in each segment's own frame, relative to the hip; the segment's tilt is not compensated — no
body levelling): stance — the foot moves straight back at the target speed, ``depth`` (0.151 m) below the hip and
``foot_out`` (0.070 m) outward, from +L/2 to −L/2 where L = duty × stride is the stance sweep; swing — it returns
forward on a cosine profile, ``x = −L/2 + L (1 − cos πs)/2``, lifted by ``swing_height`` (0.05 m) on
``(1 − cos 2πs)/2`` (s = progress through the swing). ``gait.ik_myropod`` turns a target into hip yaw, hip pitch
and the knee's absolute angle; the knee joint is ``knee_abs − hip_pitch``. (``ik='gait'``, the default, calls
``gait.ik_myropod`` per leg; ``ik='numpy'`` is the same formula vectorised over the legs — equal to it to 1e-12 rad,
tested — for many-legged batches; for 12 legs both take ≈ 50 µs.)

**Heading** (§3): the left and right strides differ by ``k_heading × yaw of segment 1`` (0.5 m/rad), clipped to
±30 % of the stride (a left yaw lengthens the left stride, which turns the robot back to the right).

**Servo targets**: joint angles only, as a position servo receives them (its PD acts on the measured speed). With
``velocity_feedforward`` the targets' rate is sent too (it drives saturated servos harder; off by default).

**Start** (§5, §11.2: standing on the feet at the nominal height, zero velocities, then the 0.5 s settle): with
``start='stance'`` (default) the controller's ``reset`` poses the robot with every foot on the ground at its gait
position for the first phase (stance feet where the stance schedule puts them, the swinging leg's foot below its
swing position) at the nominal height, holds that pose through the settle and starts walking from it — no jump
in the targets. ``start='gait'`` poses the swinging legs in the air instead; ``start='nominal'`` keeps the lab's
standing pose (all feet under the hips) and starts the gait from it.

Units: SI (m, s, rad, N); ``gait.ik_myropod`` works in mm and degrees and is converted at the boundary.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace

import numpy as np

import gait
import myropod_robot as mr
from vegeta import chiron as ch

__all__ = ["ControllerParams", "CleopatraController", "FlooredPhaseGenerator", "fixed", "adaptive", "with_params",
           "BASE_PHASES", "PHASE_STREAM", "phase_offset", "foot_target", "leg_ik", "legs_ik"]

#: Lateral-sequence walk within a segment (protocol §3, notebook 18): cycle fraction of each leg.
BASE_PHASES = {"rear-left": 0.0, "front-left": 0.25, "rear-right": 0.5, "front-right": 0.75}
#: Random stream tag for the initial phase offset: ``default_rng([PHASE_STREAM, seed])`` is independent of the
#: terrain's ``default_rng(seed)`` (both come from the trial's seed, §4).
PHASE_STREAM = 7121


@dataclass(frozen=True)
class ControllerParams:
    """Controller settings (protocol §3 and §9.1; SI).

    Protocol values: ``v_target`` [m/s]; ``stride`` [m] 0.10 (step frequency = v / stride); ``swing_height`` [m]
    0.05; ``depth`` [m] 0.151 (pad centre below the hip in stance); ``foot_out`` [m] 0.070; ``duty`` 0.75;
    ``k_heading`` [m/rad] 0.5 and ``heading_clip`` 0.30 (of the stride); ``sigma`` [rad/(N·s)] the Tegotae gain
    (0 = the fixed controller; 0.6 for §9.1's comparison); ``wave_shift`` 1/12 stride between neighbouring
    segments; ``base_phases`` within a segment.

    Implementation choices (not protocol numbers): ``min_rate`` — for σ > 0 the phase rate never drops below
    ``min_rate × ω(φ)``. At §9.1's design point (6.7 N per foot at 0.2 m/s, σN/ω ≈ 0.48) it is inactive; it acts
    only where σN cos φ > (1 − min_rate) ω, which freezes a leg (σN > ω) under the plain rule — Cleopatra's
    head-loaded front legs (≈ 17 N) on the flexible bodies, and every leg at 0.1 m/s (σN/ω ≈ 0.96). 0 = the
    plain rule. ``sigma_ref_speed`` — if set, σ is scaled by v / sigma_ref_speed (keeps σN/ω at every speed;
    off). ``velocity_feedforward`` (off), ``start`` 'stance' | 'gait' | 'nominal', ``random_offset`` (draw the
    seed's phase offset, §4), ``ik`` 'numpy' | 'gait' (see the module docstring).
    """

    v_target: float = 0.2
    stride: float = 0.10
    swing_height: float = 0.05
    depth: float = 0.151
    foot_out: float = 0.070
    duty: float = 0.75
    k_heading: float = 0.5
    heading_clip: float = 0.30
    sigma: float = 0.0
    wave_shift: float = 1.0 / 12.0
    base_phases: dict = field(default_factory=lambda: dict(BASE_PHASES))
    min_rate: float = 0.25
    sigma_ref_speed: float | None = None
    velocity_feedforward: bool = False
    start: str = "stance"
    random_offset: bool = True
    ik: str = "gait"

    def __post_init__(self):
        if self.start not in ("stance", "gait", "nominal"):
            raise ValueError("start must be 'stance', 'gait' or 'nominal'")
        if self.ik not in ("numpy", "gait"):
            raise ValueError("ik must be 'numpy' or 'gait'")
        if not 0.0 <= self.min_rate < 1.0:
            raise ValueError("min_rate must be in [0, 1)")
        if self.v_target <= 0 or self.stride <= 0:
            raise ValueError("v_target and stride must be positive")

    @property
    def frequency_hz(self) -> float:
        """Stride frequency [Hz] = v / stride."""
        return self.v_target / self.stride

    @property
    def sigma_eff(self) -> float:
        """The Tegotae gain used [rad/(N·s)]: ``sigma``, scaled by v / sigma_ref_speed when that is set."""
        if self.sigma_ref_speed:
            return self.sigma * self.v_target / self.sigma_ref_speed
        return self.sigma


def phase_offset(seed) -> float:
    """The trial's initial gait phase offset (cycle fraction in [0, 1)) from its seed; 0 for None."""
    if seed is None:
        return 0.0
    return float(np.random.default_rng([PHASE_STREAM, int(seed)]).uniform(0.0, 1.0))


def foot_target(c, duty, stride, depth, foot_out, swing_height, sy):
    """Foot target relative to the hip, segment frame [m], for cycle fraction(s) ``c`` (arrays broadcast).

    Stance (c < duty): x from +L/2 to −L/2 at constant speed, z = −depth; swing: x = −L/2 + L (1 − cos πs)/2,
    z = −depth + swing_height (1 − cos 2πs)/2; L = duty × stride; y = sy × foot_out. Returns (x, y, z)."""
    c = np.asarray(c, dtype=float)
    L = duty * np.asarray(stride, dtype=float)
    st = c < duty
    s_st = c / duty
    s_sw = (c - duty) / (1.0 - duty)
    x = np.where(st, L / 2 - L * s_st, -L / 2 + L * (1.0 - np.cos(np.pi * s_sw)) / 2)
    z = np.where(st, -depth, -depth + swing_height * (1.0 - np.cos(2 * np.pi * s_sw)) / 2)
    y = np.broadcast_to(np.asarray(sy, dtype=float) * foot_out, x.shape)
    return x, y, z


def leg_ik(rel_m, femur_m, tibia_m, sy):
    """``gait.ik_myropod`` in SI for one leg: (hip_yaw, hip_pitch, knee) [rad] for a foot at ``rel_m`` [m]
    (relative to the hip, segment frame), the knee as the physical (relative) joint angle; plus reachability."""
    (yaw, hip, knee_abs), ok = gait.ik_myropod((rel_m[0] * 1000.0, rel_m[1] * 1000.0, rel_m[2] * 1000.0),
                                               femur_m * 1000.0, tibia_m * 1000.0, sy)
    d2r = math.pi / 180.0
    return (yaw * d2r, hip * d2r, (knee_abs - hip) * d2r), ok


def legs_ik(x, y, z, femur_m, tibia_m, sy):
    """``gait.ik_myropod`` vectorised over legs (the same formula, step for step, in its mm units):
    arrays of foot targets [m] and sides → (yaw, hip_pitch, knee) [rad] arrays (knee relative) and ``ok``."""
    f, t = femur_m * 1000.0, tibia_m * 1000.0
    x = np.asarray(x, dtype=float) * 1000.0
    out_y = np.asarray(sy, dtype=float) * np.asarray(y, dtype=float) * 1000.0
    yaw = np.arctan2(x, np.maximum(out_y, 1e-6))
    r, d = np.hypot(x, out_y), -np.asarray(z, dtype=float) * 1000.0
    rho = np.hypot(r, d)
    ok = (abs(f - t) + 1e-6 < rho) & (rho < f + t - 1e-6)
    if not ok.all():
        k = np.where(rho >= f + t - 1e-6, (f + t - 2) / rho, (abs(f - t) + 2) / rho)
        k = np.where(ok, 1.0, k)
        r, d, rho = r * k, d * k, rho * k
    gamma = np.arctan2(d, r)
    alpha = np.arccos(np.clip((f ** 2 + rho ** 2 - t ** 2) / (2 * f * rho), -1.0, 1.0))
    hip = gamma - alpha
    knee_abs = np.arctan2(d - f * np.sin(hip), r - f * np.cos(hip))
    return yaw, hip, knee_abs - hip, ok


class FlooredPhaseGenerator(ch.PhaseGenerator):
    """``chiron.gaits.PhaseGenerator`` whose phase rate is bounded below by ``min_rate`` × the free rate
    (dφ/dt ≥ min_rate · ω(φ)): a loaded leg is slowed by the Tegotae term but never stopped. ``min_rate = 0`` or
    ``sigma = 0`` is the base class exactly."""

    def __init__(self, legs, base_phases, duty, frequency_hz, sigma=0.0, min_rate=0.0):
        super().__init__(legs, base_phases, duty, frequency_hz, sigma=sigma)
        self.min_rate = float(min_rate)

    def step(self, dt, normal_forces=None):
        if self.sigma == 0.0 or self.min_rate <= 0.0:
            return super().step(dt, normal_forces)
        rate = np.maximum(self.rate(normal_forces), self.min_rate * self.frequency_hz)
        self.c = np.mod(self.c + dt * rate, 1.0)
        self.t += dt
        return self.c.copy(), self.c < self.duty


def _yaw_of(q) -> float:
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class CleopatraController:
    """The §3 / §9.1 controller for ChironLab: ``reset(lab, seed)``, ``settle_command(obs)``, ``__call__(obs)``.

    ``params``: ControllerParams; ``robot_params``: the robot's ``CleopatraParams`` (femur and tibia lengths for
    the IK); ``name``: 'fixed' / 'adaptive' (logged). Works with every treatment of ``myropod_robot.cleopatra``
    (feet named ``s<i> <leg>``; the root link — the head — welded square to segment 1). After ``reset``:
    ``offset`` (the seed's phase offset), ``gen`` (the phase generator); ``unreachable`` counts targets the IK had
    to pull into reach.
    """

    def __init__(self, params: ControllerParams | None = None, robot_params: mr.CleopatraParams | None = None,
                 name: str | None = None):
        self.params = params or ControllerParams()
        self.robot_params = robot_params or mr.CleopatraParams()
        self.name = name or ("fixed" if self.params.sigma == 0 else "adaptive")
        self.gen = None
        self.offset = 0.0
        self.unreachable = 0

    def __repr__(self):
        return f"CleopatraController({self.name}, {asdict(self.params)})"

    # ---- setup
    def reset(self, lab, seed=None):
        """Bind to ``lab`` (leg layout, joint order), draw the seed's phase offset and pose the robot at its
        start (see ``ControllerParams.start``)."""
        p = self.params
        feet = list(lab.feet)
        legs, base = [], {}
        for f in feet:
            seg_s, key = f.split(" ", 1)
            seg = int(seg_s[1:])
            legs.append((seg, key, mr.LEG_SIDES[key][1]))
            base[f] = (p.base_phases[key] + (3 - seg) * p.wave_shift) % 1.0
        self._legs = legs
        self._sy = np.array([leg[2] for leg in legs], dtype=float)
        self._left = self._sy > 0
        act = {j: i for i, j in enumerate(lab.actuated_joints)}
        self._jidx = np.array([[act[j] for j in lab.foot_joints[i]] for i in range(len(feet))], dtype=int)
        self._dt = lab.control_dt
        self._n_act = len(lab.actuated_joints)
        self.gen = FlooredPhaseGenerator(feet, base, p.duty, p.frequency_hz, sigma=p.sigma_eff, min_rate=p.min_rate)
        self.offset = phase_offset(seed) if p.random_offset else 0.0
        self.gen.reset(offset=self.offset)
        if p.start == "nominal":
            self._q0 = lab.nominal_command().q_target.copy()
        else:
            self._q0 = self._targets(0.0, grounded=(p.start == "stance"))
        self.unreachable = 0
        self._q_prev = self._q0.copy()
        if p.start != "nominal":                     # re-pose the robot in place (same root pose) at its start
            obs = lab.observe()
            qpos = {j: float(self._q0[i]) for i, j in enumerate(lab.actuated_joints)}
            lab.reset(seed=seed, base_pos=tuple(float(v) for v in obs.base_pos),
                      base_quat=tuple(float(v) for v in obs.base_quat), qpos=qpos)

    def initial_pose(self) -> np.ndarray:
        """Joint targets at the start (array over ``lab.actuated_joints``; after ``reset``)."""
        return self._q0.copy()

    # ---- the control law
    def strides(self, heading_error: float) -> np.ndarray:
        """Per-foot stride [m]: left and right differ by k_heading × heading error, clipped to ±heading_clip."""
        p = self.params
        diff = float(np.clip(p.k_heading * heading_error, -p.heading_clip * p.stride, p.heading_clip * p.stride))
        return np.where(self._left, p.stride + diff / 2, p.stride - diff / 2)

    def _targets(self, heading_error: float, grounded: bool = False) -> np.ndarray:
        p, rp = self.params, self.robot_params
        x, y, z = foot_target(self.gen.c, p.duty, self.strides(heading_error), p.depth, p.foot_out,
                              0.0 if grounded else p.swing_height, self._sy)
        q = np.empty(self._n_act)
        if p.ik == "numpy":
            yaw, hip, knee, ok = legs_ik(x, y, z, rp.femur, rp.tibia, self._sy)
            q[self._jidx[:, 0]], q[self._jidx[:, 1]], q[self._jidx[:, 2]] = yaw, hip, knee
            self.unreachable += int((~ok).sum())
        else:
            for i in range(len(self._legs)):
                angles, ok = leg_ik((x[i], y[i], z[i]), rp.femur, rp.tibia, self._sy[i])
                self.unreachable += 0 if ok else 1
                q[self._jidx[i]] = angles
        return q

    def settle_command(self, obs):
        """Hold the starting pose during the settle phase."""
        return ch.Command(q_target=self._q0)

    def __call__(self, obs):
        p = self.params
        heading = _yaw_of(obs.base_quat)             # the root (head) is welded square to segment 1: same yaw
        phases = self.gen.phases
        stance = phases < p.duty
        q = self._targets(heading)
        qd = (q - self._q_prev) / self._dt if p.velocity_feedforward else None
        self._q_prev = q
        self.gen.step(self._dt, obs.foot_normal_force if self.gen.sigma != 0.0 else None)
        return ch.Command(q_target=q, qd_target=qd, leg_phase=phases, leg_stance=stance)


def fixed(v_target: float = 0.2, robot_params: mr.CleopatraParams | None = None, **kw) -> CleopatraController:
    """The fixed controller of §3 (σ = 0) at ``v_target`` [m/s]; ``kw`` override ControllerParams fields."""
    return CleopatraController(ControllerParams(v_target=v_target, sigma=0.0, **kw), robot_params, name="fixed")


def adaptive(v_target: float = 0.2, sigma: float = 0.6, robot_params: mr.CleopatraParams | None = None,
             **kw) -> CleopatraController:
    """The adaptive (Tegotae) controller of §9.1: σ = 0.6 rad/(N·s) by default; ``kw`` override
    ControllerParams fields (e.g. ``min_rate=0`` for the plain rule)."""
    return CleopatraController(ControllerParams(v_target=v_target, sigma=sigma, **kw), robot_params,
                               name="adaptive")


def with_params(ctrl: CleopatraController, **kw) -> CleopatraController:
    """A fresh controller like ``ctrl`` with some ControllerParams changed (same name)."""
    return CleopatraController(replace(ctrl.params, **kw), ctrl.robot_params, ctrl.name)
