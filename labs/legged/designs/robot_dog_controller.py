"""Gaits for Cerberus (notebook 16's robot dog) in ChironLab: the lateral-sequence walk of notebook 16 §9 and a trot.

A ChironLab controller (``reset(lab, seed)``, ``__call__(obs) -> Command``) that plans every foot in its hip's
frame (body axes, origin at the hip point) and turns it into joint targets with ``gait.ik_dog``:

* **stance** — the foot moves straight back at the commanded speed, ``depth`` below the hip (the standing
  hip-to-foot-centre height of notebook 16), directly under the hip in the leg plane, from ``+L/2`` to ``−L/2``
  about ``x_offset``; ``L = v · duty / f`` is the stance travel (notebook 16 §8/§9: a foot lands half a stance
  ahead of its hip and leaves half a stance behind);
* **swing** — it returns forward, lifted by ``swing_lift`` (notebook 16: 60 mm). ``swing_profile='linear'`` is
  notebook 16's curve (x linear in the swing, lift ``sin(π u)``); ``'smooth'`` (default) is a cubic in x that
  leaves and lands at the stance speed, lifting off on ``sin(π u)`` and landing on a raised cosine, so the foot
  clears the ground at once and touches down at ground speed with no vertical velocity; ``'cosine'`` is x on a
  cosine with the ``sin(π u)`` lift.

**Pattern** (``GAITS``): the walk is rear-left, front-left, rear-right, front-right at cycle fractions 0, 0.25,
0.5, 0.75, duty 0.75 (notebook 16 §9: three feet down at any time); the trot moves the diagonals together
(front-left with rear-right at 0, the other pair at 0.5), duty 0.5 (notebook 16 §2/§8). Phases come from
``vegeta.chiron.PhaseGenerator`` (stance while ``c < duty``, as in ``gait.simulate_steps``); ``sigma > 0``
switches on its Tegotae load feedback with each foot's normal force.

**Stride frequency** (``stride_hz``): ``'notebook'`` (default) is the frequency notebook 16 gives its own gait —
the walk of §9 (Froude stride at 0.6 m/s: 1.11 Hz) and the trot of §8 (at 1.5 m/s: 2.04 Hz) — with the stride
length scaled to the commanded speed; ``'froude'`` evaluates notebook 16's Froude stride (``froude_stride``) at the
commanded speed (slow, long strides: at 0.3 m/s a 0.68 Hz walk, which this open-loop gait cannot balance); or a
number [Hz].

Additions a dynamic simulation needs and notebook 16's kinematic plan did not (each can be switched off):
a ramp from the standing pose over ``ramp_s`` (speed, swing lift and the stance centre blend from notebook 16's
standing pose, foot 20 mm ahead of the hip, into the gait); a heading correction (left and right stance travel
differ by ``k_heading × yaw``, clipped to ±``heading_clip`` of the travel — the stability protocol's §3 rule;
``k_lateral`` also steers back towards y = 0); a load feed-forward (``load_ff``: each
commanded-stance leg's servos add ``−Jᵀ (0, 0, m g / n_stance)``, the torque that carries its share of the weight,
from ChironLab's foot Jacobians; ``'contact'`` shares it only among stance legs that touch the ground — tried and
found worse: when a leg unloads, the others push harder and the dog hops); joint-velocity feed-forward
(``velocity_ff``: the backward difference of the joint targets; off by default); an optional lateral body sway
away from the swinging legs (``sway``; off by default). No terrain sensing, no body levelling, no reflexes
(``sigma > 0`` adds the Tegotae phase reflex).

    import robot_dog_robot as rdr, robot_dog_controller as rdc
    lab = rdr.dog_lab(chiron.Flat())            # ChironLab(rdr.dog_robot(), terrain, **rdr.LAB_OPTIONS)
    ep = lab.run(rdc.DogGait("walk", 0.3), rules=chiron.FailureRules(course_m=1.5, v_target=0.3), seed=0)
"""
from __future__ import annotations

import math

import numpy as np

from vegeta.chiron import Command, PhaseGenerator

import gait
import robot_dog_robot as rdr

__all__ = ["GAITS", "G", "froude_stride", "notebook_stride_hz", "DogGait", "Stand"]

G = 9.81
#: Gait patterns — duty factor, each leg's phase (cycle fraction) and the speed at which notebook 16 runs the gait
#: (§9: walk at 0.6 m/s; §8: trot at 1.5 m/s), which fixes its stride frequency.
GAITS = {
    "walk": {"duty": 0.75, "phases": {"RL": 0.0, "FL": 0.25, "RR": 0.5, "FR": 0.75}, "notebook_speed": 0.6},
    "trot": {"duty": 0.50, "phases": {"FL": 0.0, "RR": 0.0, "FR": 0.5, "RL": 0.5}, "notebook_speed": 1.5},
}


def froude_stride(v: float, h_stand: float, g: float = G) -> tuple:
    """Notebook 16's stride for speed ``v`` [m/s] at hip height ``h_stand`` [m] (§6, coarse Froude scaling):
    ``stride = 0.9 h √(v / √(g h)) · 2 + 0.2`` [m]; returns ``(stride_m, frequency_hz)``."""
    stride = 0.9 * h_stand * math.sqrt(v / math.sqrt(g * h_stand)) * 2 + 0.2
    return stride, v / stride


def notebook_stride_hz(gait_name: str, h_stand: float) -> float:
    """The stride frequency notebook 16 gives the gait (§9 walk at 0.6 m/s, §8 trot at 1.5 m/s) [Hz]."""
    return froude_stride(GAITS[gait_name]["notebook_speed"], h_stand)[1]


def _yaw(quat) -> float:
    w, x, y, z = quat
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class Stand:
    """Hold notebook 16's standing pose (a ChironLab controller)."""

    name = "stand"

    def reset(self, lab, seed):
        self._cmd = lab.nominal_command()

    def __call__(self, obs):
        return self._cmd


class DogGait:
    """A walk or trot for the dog (see the module docstring).

    ``gait_name`` 'walk' | 'trot' (``GAITS``); ``v`` target speed [m/s]; ``stride_hz`` 'notebook' | 'froude' | Hz;
    ``duty``/``phases`` override the pattern; ``swing_lift`` [m] (notebook 16: 0.06); ``depth`` foot centre below
    the hip in stance [m] (None = notebook 16's standing height); ``x_offset`` stance centre ahead of the hip [m]
    (0: notebook 16 §9); ``swing_profile`` 'smooth' | 'linear' (notebook 16) | 'cosine'; ``ramp_s`` speed ramp from
    standstill [s] (None = one stride period); ``k_heading`` [m/rad], ``heading_clip`` (fraction of the stance
    travel), ``k_lateral`` [rad/m]; ``load_ff`` 'stance' (= True) | 'contact' | False and ``contact_threshold``
    (fraction of the weight), ``velocity_ff`` feed-forwards; ``sway`` peak lateral body sway [m]
    with ``sway_lead`` (cycle fraction of the ramps around each swing); ``sigma`` [rad/(N·s)] Tegotae gain (0 =
    fixed schedule); ``random_phase``: start at a seeded random point of the stride (uniform in one stride, as the
    stability protocol does); ``geometry``: ``robot_dog_robot.geometry()`` of the simulated dog.
    """

    def __init__(self, gait_name: str = "walk", v: float = 0.3, *, stride_hz="notebook", duty: float | None = None,
                 phases: dict | None = None, swing_lift: float = 0.06, depth: float | None = None,
                 x_offset: float = 0.0, swing_profile: str = "smooth", ramp_s: float | None = None,
                 k_heading: float = 0.5, heading_clip: float = 0.3, k_lateral: float = 0.0, load_ff="stance",
                 contact_threshold: float = 0.02, velocity_ff: bool = False, sway: float = 0.0,
                 sway_lead: float = 0.125, sigma: float = 0.0, random_phase: bool = False,
                 geometry: dict | None = None, name: str | None = None):
        if gait_name not in GAITS:
            raise ValueError(f"gait must be one of {sorted(GAITS)}")
        if swing_profile not in ("linear", "cosine", "smooth"):
            raise ValueError("swing_profile must be 'linear', 'cosine' or 'smooth'")
        if v < 0:
            raise ValueError("v must be non-negative")
        self.gait = gait_name
        self.v = float(v)
        self.geom = dict(geometry or rdr.geometry())
        pattern = GAITS[gait_name]
        self.duty = float(duty if duty is not None else pattern["duty"])
        self.base_phases = dict(phases if phases is not None else pattern["phases"])
        if stride_hz == "notebook":
            self.stride_hz = notebook_stride_hz(gait_name, self.geom["h_stand"])
        elif stride_hz == "froude":
            self.stride_hz = froude_stride(max(self.v, 1e-9), self.geom["h_stand"])[1]
        else:
            self.stride_hz = float(stride_hz)
        if not self.stride_hz > 0:
            raise ValueError("stride_hz must be positive")
        self.stride_rule = stride_hz if isinstance(stride_hz, str) else "given"
        self.swing_lift = float(swing_lift)
        self.depth = float(depth if depth is not None else self.geom["h_stand"])
        self.x_offset = float(x_offset)
        self.swing_profile = swing_profile
        self.ramp_s = float(ramp_s) if ramp_s is not None else 1.0 / self.stride_hz
        self.k_heading, self.heading_clip, self.k_lateral = float(k_heading), float(heading_clip), float(k_lateral)
        if load_ff not in (True, False, "contact", "stance"):
            raise ValueError("load_ff must be 'contact', 'stance' (= True) or False")
        self.load_ff = "stance" if load_ff is True else load_ff
        self.contact_threshold = float(contact_threshold)
        self.velocity_ff = bool(velocity_ff)
        self.sway, self.sway_lead = float(sway), float(sway_lead)
        self.sigma = float(sigma)
        self.random_phase = bool(random_phase)
        self.name = name or f"{gait_name} {self.v:g} m/s" + (f" sigma {self.sigma:g}" if self.sigma else "")
        self._L1, self._L2 = self.geom["L1"] * 1000.0, self.geom["L2"] * 1000.0      # gait.ik_dog works in mm

    def settings(self) -> dict:
        """The controller's settings (plain values, for the episode log)."""
        keys = ("gait", "v", "duty", "base_phases", "stride_hz", "stride_rule", "swing_lift", "depth", "x_offset",
                "swing_profile", "ramp_s", "k_heading", "heading_clip", "k_lateral", "load_ff", "contact_threshold",
                "velocity_ff", "sway", "sway_lead", "sigma", "random_phase")
        return {k: getattr(self, k) for k in keys}

    # ---- the stride
    @property
    def stance_length(self) -> float:
        """Foot travel during one stance at full speed [m]."""
        return self.v * self.duty / self.stride_hz

    def foot_target(self, c: float, length: float, x_centre: float | None = None, lift: float | None = None
                    ) -> tuple:
        """Foot relative to its hip (x ahead, z up) [m] and the stance flag at cycle fraction ``c``, for a stance
        travel ``length`` [m] about ``x_centre`` (default ``x_offset``) with a swing ``lift`` (default
        ``swing_lift``)."""
        duty = self.duty
        xc = self.x_offset if x_centre is None else x_centre
        h = self.swing_lift if lift is None else lift
        if c < duty:
            return xc + length / 2 * (1 - 2 * c / duty), -self.depth, True
        u = (c - duty) / (1 - duty)
        x0 = xc - length / 2
        if self.swing_profile == "linear":                    # notebook 16 §8/§9
            return x0 + length * u, -self.depth + h * math.sin(math.pi * u), False
        if self.swing_profile == "cosine":
            return x0 + length * (1 - math.cos(math.pi * u)) / 2, -self.depth + h * math.sin(math.pi * u), False
        # 'smooth': cubic Hermite in x leaving and landing at the stance speed (dx/du = −L (1 − duty)/duty); the
        # foot lifts off on sin(π u) (clears the ground at once) and lands on (1 + cos 2π(u − ½))/2 (no vertical
        # speed at touchdown)
        m = -length * (1 - duty) / duty
        u2, u3 = u * u, u * u * u
        x = (2 * u3 - 3 * u2 + 1) * x0 + (u3 - 2 * u2 + u) * m + (-2 * u3 + 3 * u2) * (x0 + length) + (u3 - u2) * m
        zl = math.sin(math.pi * u) if u <= 0.5 else 0.5 * (1.0 + math.cos(2.0 * math.pi * (u - 0.5)))
        return x, -self.depth + h * zl, False

    def swing_weight(self, c: float) -> float:
        """For the sway: 1 while the leg swings, 0 deep in stance, raised-cosine ramps ``sway_lead`` (cycle
        fraction) before lift-off and after touchdown."""
        duty, lead = self.duty, self.sway_lead
        if c >= duty:
            return 1.0
        if lead > 0 and c >= duty - lead:
            return 0.5 * (1.0 - math.cos(math.pi * (c - duty + lead) / lead))
        if lead > 0 and c < lead:
            return 0.5 * (1.0 + math.cos(math.pi * c / lead))
        return 0.0

    def leg_angles(self, x: float, y: float, z: float) -> tuple:
        """Joint targets (hip roll, hip pitch, knee) [rad] for a foot at (x, y, z) [m] from the hip (body axes)
        and whether it is reachable (``gait.ik_dog``)."""
        (roll, a1, a2), ok = gait.ik_dog((x * 1000.0, y * 1000.0, z * 1000.0), self._L1, self._L2)
        return rdr.angles_to_q(roll, a1, a2), ok

    # ---- ChironLab controller interface
    def reset(self, lab, seed):
        self.legs = list(lab.feet)
        missing = [leg for leg in self.legs if leg not in self.base_phases or leg not in rdr.LEGS]
        if missing:
            raise ValueError(f"no phase or side for feet {missing}")
        self.side = np.array([rdr.LEGS[leg][1] for leg in self.legs], dtype=float)     # +1 left, -1 right
        self.pg = PhaseGenerator(self.legs, self.base_phases, self.duty, self.stride_hz, sigma=self.sigma)
        offset = float(np.random.default_rng(seed).uniform()) if (self.random_phase and seed is not None) else 0.0
        self.phase_offset = offset
        self.pg.reset(offset=offset)
        act_index = {n: i for i, n in enumerate(lab.actuated_joints)}
        self.leg_cols = [[act_index[j] for j in rdr.leg_joints(leg)] for leg in self.legs]
        self.yaw0 = None
        self.weight = lab.total_mass * G
        self.dt = lab.control_dt
        self.prev_q = None
        self.unreachable = 0
        # sway normalisation: the peak of the summed swing weights over one stride of the fixed schedule
        cs = np.linspace(0.0, 1.0, 400, endpoint=False)
        sums = [abs(sum(-self.side[i] * self.swing_weight((c + self.pg.base[i]) % 1.0) for i in range(len(self.legs))))
                for c in cs]
        self._sway_norm = max(sums) if max(sums) > 0 else 1.0
        n = len(lab.actuated_joints)
        self._q = lab.nominal_command().q_target.copy()
        self._qd = np.zeros(n)
        self._ff = np.zeros(n)

    def __call__(self, obs):
        t = obs.t
        if self.yaw0 is None:
            self.yaw0 = _yaw(obs.base_quat)
        if self.sigma:
            phases, stance = self.pg.step(self.dt, obs.foot_normal_force)
        else:
            phases, stance = self.pg.step(self.dt)
        # start from the standing pose: speed, lift and the stance centre (notebook 16's standing foot is foot_x
        # ahead of the hip) blend into the gait over ramp_s, so the first targets equal the standing pose
        ramp = min(1.0, max(t, 0.0) / self.ramp_s) if self.ramp_s > 0 else 1.0
        length = self.v * ramp * self.duty / self.stride_hz
        x_centre = self.x_offset + (1.0 - ramp) * (self.geom["foot_x"] - self.x_offset)
        lift = self.swing_lift * ramp
        # heading: turned left of the reference (yaw > 0) -> longer left stance, shorter right stance; the
        # reference turns back towards the route when the COM has drifted sideways (k_lateral)
        yaw = _yaw(obs.base_quat) - self.yaw0
        if self.k_lateral:
            yaw += self.k_lateral * float(obs.com[1])
        dl = float(np.clip(self.k_heading * yaw, -self.heading_clip * length, self.heading_clip * length))
        # lateral sway: the body moves away from the side whose legs swing (all feet shift the other way)
        y_body = 0.0
        if self.sway:
            y_body = ramp * self.sway / self._sway_norm * sum(-self.side[i] * self.swing_weight(float(phases[i]))
                                                               for i in range(len(self.legs)))
        q = self._q
        for i in range(len(self.legs)):
            x, z, _ = self.foot_target(float(phases[i]), length + self.side[i] * dl / 2, x_centre, lift)
            angles, ok = self.leg_angles(x, -y_body, z)
            self.unreachable += 0 if ok else 1
            q[self.leg_cols[i]] = angles
        qd = None
        if self.velocity_ff:
            qd = self._qd
            if self.prev_q is None:
                qd[:] = 0.0
            else:
                np.subtract(q, self.prev_q, out=qd)
                qd /= self.dt
            self.prev_q = q.copy()
        ff = None
        if self.load_ff:
            ff = self._ff
            ff[:] = 0.0
            carrying = stance
            if self.load_ff == "contact":                 # only legs in stance that touch the ground carry load
                carrying = stance & (obs.foot_normal_force > self.contact_threshold * self.weight)
            n_st = int(carrying.sum())
            if n_st:
                fz = self.weight / n_st
                jac = obs.foot_jac
                for i in np.nonzero(carrying)[0]:
                    ff[self.leg_cols[i]] = -fz * jac[i][2]             # −Jᵀ (0, 0, fz)
        return Command(q_target=q.copy(), qd_target=None if qd is None else qd.copy(),
                       tau_ff=None if ff is None else ff.copy(), leg_phase=phases, leg_stance=stance)
