"""Cleopatra as a Chiron robot: the three-segment Myropod of notebook 18 and its body treatments (Amendment D).

The pre-registered stability protocol (``docs/myropod_stability.md``) fixes everything here: §1 the robot (geometry,
masses, joints, actuators, friction) and §12 (Amendment D, 2026-10-01, which supersedes §2 and §11.2) the
treatments. ``cleopatra()`` returns a ``chiron.Robot``. Both treatments have the **identical intersegment joints**
between segments 1–2 and 2–3 (the head is welded to segment 1): passive hinges on pitch and yaw always, roll per
``body_roll_axis`` (default on), chained yaw → pitch → roll at the joint pin, with the same stiffness k [N·m/rad],
neutral angle q0 [rad] and hard stops [rad] per axis, armature 0, friction loss 0, translations constrained, no
body actuator and no prescribed bending. They differ in nothing but the joint damping c:

* ``spring`` — joint torque ``τ = −k (q − q0)``; c = 0 on every axis (any ``body_c_*`` given is ignored, and
  ``robot.notes`` says so);
* ``spring_damper`` — ``τ = −k (q − q0) − c q̇`` with c [N·m·s/rad] per axis (default 0.2).

Protocol defaults (``BODY_DEFAULTS``): k = 8 N·m/rad, c = 0.2 N·m·s/rad, q0 = 0 rad, limits ±45° (0.785 rad) pitch
and yaw, ±20° (0.349 rad) roll. The same names are Myropod design parameters (``designs/myropod.py``, §12.3):
``design_connection`` and ``robot_from_design`` build the robot from a Dedalus parameter dict, and Chiron's CLI
passes them as ``-p body_connection=spring -p body_k_pitch=8``.

Amendment C's treatments ('rigid', 'flexible', 'flexible+yaw', §11.2) are **legacy (pre-Amendment D)**: they are
built only with ``legacy=True``, labelled ``legacy:<name>`` and never pooled with the Amendment D results; §2's
'locked' and 'flexible+roll' are refused.

Frames (Chiron's convention): x forward (the head at +x), y left, z up. Every segment's link frame sits at the
segment's centre at **hip height** (the CAD's ``seg_height / 2`` above the shell's belly), so a hip is at
``(±hip_x, ±hip_y, 0)`` in its segment's frame; the head's frame is at its centre at the same height. The chain
grows backwards from the root link, the head (it floats freely; Chiron logs the bodies in tree order: head,
segment 1, segment 2, segment 3): head → segment 1 (welded, 0.160 m behind it) → segment 2 → segment 3, one pitch
(0.170 m) apart; a body joint sits at the pin halfway across the joint gap. Body-joint signs: yaw about +z
(+ swings the rear segment's tail to the right), pitch about +y (+ lifts the rear segment's tail), roll about +x;
q = 0 is the straight chain.

Legs: hip yaw (axis: segment z; positive sweeps the foot forward), hip pitch (horizontal, perpendicular to the
leg plane; positive = femur below horizontal) and knee (parallel to hip pitch; the **relative** angle, positive =
tibia folds further down). These are ``gait.ik_myropod``'s conventions: its knee is the tibia's absolute angle
below horizontal, so the physical knee joint is ``knee_abs − hip_pitch``. At zero every leg sticks straight out
sideways.

Units: SI here (m, kg, s, N·m, rad); the design files (``myropod.py``, ``gait.py``) work in mm and degrees for
geometry and are converted at the boundary (the body-joint design parameters are already SI). Every number
carries its source in ``CleopatraParams`` and ``Robot.sources``.

Promoted from ``notebooks/designs/myropod_robot.py`` as it was proven there; the notebook copy may move on.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field

import numpy as np

from . import actuators as act
from vegeta import chiron as ch

__all__ = ["CLEO_MM", "CleopatraParams", "BodyAxis", "BodyConnection", "TREATMENTS", "LEGACY_TREATMENTS",
           "RETIRED_TREATMENTS", "AXIS_ORDER", "BODY_DEFAULTS", "DESIGN_KEYS", "LIMIT_PITCH", "LIMIT_YAW",
           "LIMIT_ROLL", "LEG_KEYS", "LEG_SIDES", "SEGMENTS", "LEG_RANGES", "KP", "KD", "cleopatra",
           "cleopatra_servo", "canonical_treatment", "make_connection", "connection_params", "design_connection",
           "design_params", "robot_from_design", "as_bool", "leg_name", "leg_joints", "body_joints", "nominal_pose",
           "mass_budget", "hip_position", "stance_foot", "LAB_OPTIONS", "LAB_OPTIONS_PROTOCOL", "COURSE_M",
           "cleopatra_lab", "failure_rules"]

#: Notebook 18's ``CLEO`` parameters for ``designs/myropod.py`` [mm, deg] (Cleopatra) — the protocol's §1 geometry.
CLEO_MM = dict(n_segments=3, seg_length=150.0, seg_width=110.0, seg_height=70.0, shell_thickness=3.0, joint_gap=20.0,
               joint_diameter=40.0, joint_pin_diameter=10.0, femur_length=80.0, tibia_length=100.0, leg_diameter=16.0,
               foot_diameter=24.0, hip_angle_deg=40.0, knee_angle_deg=85.0, head_length=130.0, tether=False)

#: The four legs of a segment and their (sx, sy) signs (front +1 / rear −1, left +1 / right −1), as ``gait.LEGS``.
LEG_SIDES = {"front-left": (1, 1), "front-right": (1, -1), "rear-left": (-1, 1), "rear-right": (-1, -1)}
LEG_KEYS = tuple(LEG_SIDES)
SEGMENTS = ("segment 1", "segment 2", "segment 3")
#: Protocol §1 leg joint ranges [rad]: yaw ±60°, hip pitch −40°…+90°, knee (relative) 0°…160°.
LEG_RANGES = {"hip_yaw": (math.radians(-60.0), math.radians(60.0)),
              "hip_pitch": (math.radians(-40.0), math.radians(90.0)),
              "knee": (0.0, math.radians(160.0))}
#: Protocol §1 servo gains (the same in every configuration and trial) [N·m/rad, N·m·s/rad].
KP, KD = 40.0, 0.8


# ----------------------------------------------------------------------------------------------- the robot's numbers
@dataclass(frozen=True)
class CleopatraParams:
    """Every physical input of the model (SI). Defaults are the protocol's §1 values; sources below.

    Geometry — ``designs/myropod.py`` with notebook 18's ``CLEO`` (``CLEO_MM``):
    ``seg_length/width/height`` 0.150 × 0.110 × 0.070 m shells, ``shell_thickness`` 3 mm, ``joint_gap`` 0.020 m
    (pitch 0.170 m), ``head_length`` 0.130 m (a loft from the segment's 110 × 70 mm section to ``head_taper`` = 85 %
    of it at the front), ``femur`` 0.080 m, ``tibia`` 0.100 m, ``leg_diameter`` 0.016 m (the tibia bar is 0.8 × as in
    the CAD), ``foot_diameter`` 0.024 m (a sphere), hip pairs at ``±hip_x`` = ±seg_length × leg_pair_spacing / 2 =
    ±0.0375 m, hips at ``±hip_y`` = seg_width / 2 + 6 mm = ±0.061 m, standing hip angle 40° and tibia 85° below
    horizontal (pad centre 70.0 mm outward and 151.0 mm below the hip).

    Masses — notebook 18's recorded budget (6.17 kg): ``shell_mass`` 259.2 g per segment (printed shell + hatch +
    rib + bosses + necks, CAD volume × 1.1 g/cm³), ``leg_cad_mass`` 196.7 g per four legs (CAD leg volume × 1.1
    g/cm³; the 10 g ``pad_mass`` is inside it, the rest is split femur/tibia by length), servos from
    ``actuators.py``: every leg joint a ``smart servo 6 Nm`` (70 g; hip yaw + hip pitch at the hip on the segment,
    the knee servo at the knee on the femur), two body-joint servos ``smart servo 12 Nm`` (120 g) per segment
    (carried in every treatment: same masses), ``pcb_mass`` 60 g and ``wiring_mass`` 50 g per segment,
    ``head_shell_mass`` 188 g (CAD) + ``head_payload`` 450 g (4K day camera 180 g, NIR camera 120 g, LWIR 90 g,
    microphones + IMU + radio 60 g), ``battery_mass`` 444 g (4S2P 21700: 8 × 48 g cells + 60 g BMS) in segment 2,
    ``compute_mass`` 150 g in segment 1.

    Placement (protocol §1: "placed so the inertia is realistic"; the positions are this model's choices, stated
    here): each shell's mass is spread over its six walls in proportion to their area (a thin-walled box); the
    head's likewise over a box of its mean section; the cameras and LWIR (390 g) sit ``camera_x`` = 40 mm ahead of
    the head's centre (25 mm behind its front face), the microphones/IMU/radio (60 g) at its centre; the body-joint
    servos on the segment's centre line at ``±body_servo_x`` = ±50 mm (one towards each end); the battery on the
    floor of segment 2 (``battery_z`` = −15 mm), compute at ``compute_z`` = +10 mm, the PCB under the hatch
    (``pcb_z`` = +25 mm), the wiring at the centre.

    Contacts (§1): foot pads μ = 0.8 (rubber on rock), shells and head μ = 0.5; the legs' bars do not touch the
    ground (§1 lists only pads, shells and head) unless ``cleopatra(leg_links_collide=True)``.
    """

    seg_length: float = 0.150
    seg_width: float = 0.110
    seg_height: float = 0.070
    shell_thickness: float = 0.003
    joint_gap: float = 0.020
    head_length: float = 0.130
    head_taper: float = 0.85
    femur: float = 0.080
    tibia: float = 0.100
    leg_diameter: float = 0.016
    tibia_diameter_ratio: float = 0.8
    foot_diameter: float = 0.024
    leg_pair_spacing: float = 0.5
    hip_y_offset: float = 0.006
    hip_angle_deg: float = 40.0
    knee_angle_deg: float = 85.0
    # masses [kg]
    shell_mass: float = 0.2592
    leg_cad_mass: float = 0.1967
    pad_mass: float = 0.010
    leg_servo: str = "smart servo 6 Nm"
    body_servo: str = "smart servo 12 Nm"
    body_servos_per_segment: int = 2
    pcb_mass: float = 0.060
    wiring_mass: float = 0.050
    head_shell_mass: float = 0.188
    head_payload: tuple = (("4K day camera", 0.180), ("NIR night camera", 0.120), ("LWIR", 0.090),
                           ("microphones, IMU, radio", 0.060))
    battery_mass: float = 0.444
    compute_mass: float = 0.150
    # placement [m]
    camera_x: float = 0.040
    body_servo_x: float = 0.050
    battery_z: float = -0.015
    compute_z: float = 0.010
    pcb_z: float = 0.025
    # contacts
    foot_mu: float = 0.8
    shell_mu: float = 0.5

    # ---- derived geometry
    @property
    def pitch(self) -> float:
        """Segment pitch [m] (``Myropod.pitch_length``)."""
        return self.seg_length + self.joint_gap

    @property
    def hip_x(self) -> float:
        """Leg pairs at ±hip_x from the segment centre [m] (``Myropod.hip_x``)."""
        return self.seg_length * self.leg_pair_spacing / 2

    @property
    def hip_y(self) -> float:
        """Hips at ±hip_y [m] (seg_width / 2 + 6 mm, the CAD's hip bosses)."""
        return self.seg_width / 2 + self.hip_y_offset

    @property
    def head_offset(self) -> float:
        """Head centre ahead of segment 1's centre [m] (``Myropod.frames``: (head + seg)/2 + gap)."""
        return (self.head_length + self.seg_length) / 2 + self.joint_gap

    @property
    def foot_out(self) -> float:
        """Standing pad centre outward from the hip [m] (0.0700)."""
        a, b = math.radians(self.hip_angle_deg), math.radians(self.knee_angle_deg)
        return self.femur * math.cos(a) + self.tibia * math.cos(b)

    @property
    def foot_drop(self) -> float:
        """Standing pad centre below the hip [m] (0.1510)."""
        a, b = math.radians(self.hip_angle_deg), math.radians(self.knee_angle_deg)
        return self.femur * math.sin(a) + self.tibia * math.sin(b)

    @property
    def hip_height(self) -> float:
        """Standing hip height above flat ground [m]: the pad centre's drop plus the pad radius (0.1630)."""
        return self.foot_drop + self.foot_diameter / 2

    @property
    def leg_servo_mass(self) -> float:
        return act.get(self.leg_servo).mass_g / 1000.0

    @property
    def body_servo_mass(self) -> float:
        return act.get(self.body_servo).mass_g / 1000.0

    @property
    def femur_mass(self) -> float:
        """Femur bar [kg]: the CAD leg mass without the pad, split by length."""
        rest = self.leg_cad_mass / 4 - self.pad_mass
        return rest * self.femur / (self.femur + self.tibia)

    @property
    def tibia_mass(self) -> float:
        rest = self.leg_cad_mass / 4 - self.pad_mass
        return rest * self.tibia / (self.femur + self.tibia)


# ----------------------------------------------------------------------------------------------- body joints
#: The treatments (protocol §12.1, Amendment D). They differ in nothing but the body joints' damping c.
TREATMENTS = ("spring", "spring_damper")
#: Amendment C's treatments (§11.2): legacy (pre-Amendment D) — built only with ``legacy=True``, never pooled.
LEGACY_TREATMENTS = ("rigid", "flexible", "flexible+yaw")
#: §2's names (retired by Amendment C): refused.
RETIRED_TREATMENTS = ("locked", "flexible+roll")
#: Hinge order at a body joint, from the front segment to the rear one: the yaw pin, then pitch, then roll.
AXIS_ORDER = ("yaw", "pitch", "roll")
AXIS_VECTOR = {"yaw": (0.0, 0.0, 1.0), "pitch": (0.0, 1.0, 0.0), "roll": (1.0, 0.0, 0.0)}
#: Protocol §12.1 hard stops [rad]: ±45° pitch and yaw, ±20° roll.
LIMIT_PITCH, LIMIT_YAW, LIMIT_ROLL = math.radians(45.0), math.radians(45.0), math.radians(20.0)
#: The body-joint parameters with the protocol's §12.1 defaults — the same names as the Myropod design parameters
#: (§12.3) and ``cleopatra()``'s keywords: treatment; roll hinge on/off; per axis stiffness k [N·m/rad], damping
#: c [N·m·s/rad] (spring_damper only), neutral angle q0 [rad] and the symmetric hard stop ±limit [rad].
BODY_DEFAULTS = {
    "body_connection": "spring_damper", "body_roll_axis": True,
    "body_k_pitch": 8.0, "body_k_yaw": 8.0, "body_k_roll": 8.0,
    "body_c_pitch": 0.2, "body_c_yaw": 0.2, "body_c_roll": 0.2,
    "body_q0_pitch": 0.0, "body_q0_yaw": 0.0, "body_q0_roll": 0.0,
    "body_limit_pitch": LIMIT_PITCH, "body_limit_yaw": LIMIT_YAW, "body_limit_roll": LIMIT_ROLL,
}
#: Myropod design-parameter names read by ``design_connection`` (= ``BODY_DEFAULTS``' keys).
DESIGN_KEYS = tuple(BODY_DEFAULTS)
_LEGACY_AXES = {"rigid": (), "flexible": ("pitch", "roll"), "flexible+yaw": AXIS_ORDER}
_TRUE, _FALSE = ("true", "yes", "on", "1"), ("false", "no", "off", "0")


def as_bool(value, name: str = "value") -> bool:
    """A bool from a Python/numpy bool, 0/1, or 'true'/'false', 'yes'/'no', 'on'/'off', '1'/'0' (any case) — the
    forms a design dict or ``chiron run -p name=value`` gives; anything else raises ``ValueError``."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in _TRUE + _FALSE:
        return value.strip().lower() in _TRUE
    raise ValueError(f"{name} expects a bool (true/false, 1/0, yes/no, on/off), got {value!r}")


def _num(value, name: str) -> float:
    """A finite float from a number or a numeric string (not a bool)."""
    if isinstance(value, (bool, np.bool_)) or value is None:
        raise ValueError(f"{name} expects a number, got {value!r}")
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} expects a number, got {value!r}") from None
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return x


def _limits(value, name: str) -> tuple:
    """Hard stops [rad]: a positive number L → (−L, L); a pair → (lo, hi)."""
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) != 2:
            raise ValueError(f"{name} expects a number L (stops at ±L) or a pair (lo, hi) [rad], got {value!r}")
        return (_num(value[0], name), _num(value[1], name))
    lim = _num(value, name)
    if lim <= 0:
        raise ValueError(f"{name} must be positive (stops at ±{name}) [rad], got {value!r}")
    return (-lim, lim)


@dataclass(frozen=True)
class BodyAxis:
    """One passive body-joint axis: joint torque ``τ = −k (q − q0) − c q̇`` [N·m] plus the hard stops.

    ``stiffness`` k [N·m/rad] (≥ 0); ``damping`` c [N·m·s/rad] (≥ 0; 0 in the 'spring' treatment); ``q0`` [rad] the
    neutral angle — the spring's rest angle; the segments start there with the springs unloaded; ``limits``
    (lo, hi) [rad] the hard stops, absolute joint angles with lo < q0 < hi, within ±π."""

    stiffness: float = 8.0
    damping: float = 0.2
    q0: float = 0.0
    limits: tuple = (-LIMIT_PITCH, LIMIT_PITCH)

    def __post_init__(self):
        for k in ("stiffness", "damping", "q0"):
            object.__setattr__(self, k, _num(getattr(self, k), k))
        if len(self.limits) != 2:
            raise ValueError("limits must be (lo, hi) [rad]")
        lo, hi = (_num(v, "limits") for v in self.limits)
        object.__setattr__(self, "limits", (lo, hi))
        if not -math.pi <= lo < hi <= math.pi:
            raise ValueError(f"limits must satisfy -pi <= lo < hi <= pi [rad], got {(lo, hi)}")
        if not lo < self.q0 < hi:
            raise ValueError(f"q0 = {self.q0} rad must lie strictly within the limits {(lo, hi)} rad")
        if self.stiffness < 0 or self.damping < 0:
            raise ValueError("stiffness [N·m/rad] and damping [N·m·s/rad] must be non-negative")


_DEFAULT_AXES = {a: BodyAxis(BODY_DEFAULTS[f"body_k_{a}"], BODY_DEFAULTS[f"body_c_{a}"], BODY_DEFAULTS[f"body_q0_{a}"],
                             (-BODY_DEFAULTS[f"body_limit_{a}"], BODY_DEFAULTS[f"body_limit_{a}"]))
                 for a in AXIS_ORDER}


@dataclass(frozen=True)
class BodyConnection:
    """The intersegment joints between segments 1–2 and 2–3 (identical at both; the head is welded to segment 1).

    ``treatment`` 'spring' | 'spring_damper' (protocol §12.1), or with ``legacy=True`` Amendment C's 'rigid' |
    'flexible' | 'flexible+yaw' (§11.2, pre-Amendment D). ``axes``: the hinged axes in chain order (yaw → pitch →
    roll) — Amendment D: ('yaw', 'pitch', 'roll') or ('yaw', 'pitch') without the roll hinge; legacy: () welds the
    segments ('rigid'), ('pitch', 'roll') 'flexible', all three 'flexible+yaw'. ``yaw``/``pitch``/``roll``: each
    axis's ``BodyAxis`` **as built** (a 'spring' connection has c = 0 on every axis). An axis that is not hinged is
    held at its q0, built into the link's orientation before the hinges (Amendment D: only q0 = 0 there; legacy
    'rigid' welds at the neutral angles). ``notes``: what the builder ignored (not part of equality).

    Use ``make_connection`` (the design-parameter names) or ``design_connection`` (a design dict) to build one.
    """

    treatment: str = "spring_damper"
    axes: tuple = AXIS_ORDER
    yaw: BodyAxis = _DEFAULT_AXES["yaw"]
    pitch: BodyAxis = _DEFAULT_AXES["pitch"]
    roll: BodyAxis = _DEFAULT_AXES["roll"]
    legacy: bool = False
    notes: str = field(default="", compare=False)

    def __post_init__(self):
        axes = tuple(self.axes)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "legacy", as_bool(self.legacy, "legacy"))
        if any(a not in AXIS_ORDER for a in axes) or tuple(a for a in AXIS_ORDER if a in axes) != axes:
            raise ValueError(f"axes {axes} must be distinct names from {AXIS_ORDER}, in that (chain) order")
        canonical_treatment(self.treatment, self.legacy)
        if self.treatment in LEGACY_TREATMENTS:
            if not self.legacy:
                raise ValueError("a legacy treatment needs legacy=True")
            if axes != _LEGACY_AXES[self.treatment]:
                raise ValueError(f"legacy {self.treatment!r} hinges {_LEGACY_AXES[self.treatment]}, not {axes}")
            return
        if self.legacy:
            raise ValueError(f"{self.treatment!r} is an Amendment D treatment, not legacy")
        if axes not in (AXIS_ORDER, ("yaw", "pitch")):
            raise ValueError(f"Amendment D hinges pitch and yaw always and roll optionally (§12.1), not {axes}")
        if self.treatment == "spring" and any(self.axis(a).damping != 0.0 for a in axes):
            raise ValueError("'spring' has joint damping c = 0 on every body axis (§12.1)")
        for a in AXIS_ORDER:
            if a not in axes and self.axis(a).q0 != 0.0:
                raise ValueError(f"the {a} axis is not hinged: its q0 must be 0 (a fixed offset is not modelled)")

    @property
    def hinged(self) -> tuple:
        """The hinged axes in chain order (empty for the legacy 'rigid')."""
        return self.axes

    @property
    def roll_axis(self) -> bool:
        return "roll" in self.axes

    def axis(self, name: str) -> BodyAxis:
        return getattr(self, name)


def canonical_treatment(treatment: str, legacy: bool = False) -> str:
    """Check a treatment name: 'spring' | 'spring_damper' (protocol §12.1); Amendment C's 'rigid' | 'flexible' |
    'flexible+yaw' only with ``legacy=True`` (pre-Amendment D: reported as legacy, never pooled); §2's 'locked' and
    'flexible+roll' and anything else raise ``ValueError``."""
    name = str(treatment)
    if name in TREATMENTS:
        return name
    if name in LEGACY_TREATMENTS:
        if as_bool(legacy, "legacy"):
            return name
        raise ValueError(f"treatment {name!r} is legacy (pre-Amendment D, protocol §11.2): pass legacy=True to build "
                         f"it (its results are never pooled with {TREATMENTS}); Amendment D uses {TREATMENTS}")
    if name in RETIRED_TREATMENTS:
        raise ValueError(f"treatment {name!r} (protocol §2) was retired by Amendment C and is not built; "
                         f"use one of {TREATMENTS} (§12.1)")
    raise ValueError(f"unknown treatment {name!r}: use one of {TREATMENTS} (protocol §12.1)")


def make_connection(body_connection: str = BODY_DEFAULTS["body_connection"], *,
                    body_roll_axis=BODY_DEFAULTS["body_roll_axis"],
                    body_k_pitch=8.0, body_k_yaw=8.0, body_k_roll=8.0,
                    body_c_pitch=0.2, body_c_yaw=0.2, body_c_roll=0.2,
                    body_q0_pitch=0.0, body_q0_yaw=0.0, body_q0_roll=0.0,
                    body_limit_pitch=LIMIT_PITCH, body_limit_yaw=LIMIT_YAW, body_limit_roll=LIMIT_ROLL,
                    legacy=False) -> BodyConnection:
    """The ``BodyConnection`` for the body-joint parameters (names and defaults: ``BODY_DEFAULTS``, protocol §12.1).

    ``body_connection`` 'spring' | 'spring_damper'; ``body_roll_axis`` (bool; ``as_bool`` forms accepted) adds the
    roll hinge; per axis ``body_k_*`` k [N·m/rad], ``body_c_*`` c [N·m·s/rad] (spring_damper only: 'spring' builds
    c = 0 whatever these say and records the ignored values in ``notes``), ``body_q0_*`` q0 [rad], ``body_limit_*``
    [rad] (a positive L: stops at ±L; or a pair (lo, hi)). Numbers may be ints, floats or numeric strings (the CLI's
    ``-p`` values). Without the roll hinge, ``body_k_roll``/``body_c_roll``/``body_limit_roll`` are unused and
    ``body_q0_roll`` must be 0. 'spring_damper' accepts c = 0 (the §12.5 check that it then reproduces 'spring'
    bitwise); the treatment as run has c > 0. ``legacy=True`` unlocks Amendment C's names (see
    ``canonical_treatment``), built with Amendment C's hinged axes and these per-axis values (``body_roll_axis``
    unused)."""
    t = canonical_treatment(body_connection, legacy)
    roll_on = as_bool(body_roll_axis, "body_roll_axis")
    given = {"k": {"pitch": body_k_pitch, "yaw": body_k_yaw, "roll": body_k_roll},
             "c": {"pitch": body_c_pitch, "yaw": body_c_yaw, "roll": body_c_roll},
             "q0": {"pitch": body_q0_pitch, "yaw": body_q0_yaw, "roll": body_q0_roll},
             "limit": {"pitch": body_limit_pitch, "yaw": body_limit_yaw, "roll": body_limit_roll}}
    k = {a: _num(v, f"body_k_{a}") for a, v in given["k"].items()}
    c = {a: _num(v, f"body_c_{a}") for a, v in given["c"].items()}
    q0 = {a: _num(v, f"body_q0_{a}") for a, v in given["q0"].items()}
    lim = {a: _limits(v, f"body_limit_{a}") for a, v in given["limit"].items()}
    notes = []
    if t in LEGACY_TREATMENTS:
        axes = _LEGACY_AXES[t]
        notes.append(f"LEGACY (pre-Amendment D) treatment {t!r} (protocol §11.2), never pooled with {TREATMENTS}; "
                     f"body_roll_axis unused")
    else:
        axes = AXIS_ORDER if roll_on else ("yaw", "pitch")
        if not roll_on:
            notes.append("no roll hinge (body_roll_axis false): roll held at 0; body_k_roll, body_c_roll and "
                         "body_limit_roll unused")
    if t == "spring":
        ignored = {f"body_c_{a}": c[a] for a in axes if c[a] != 0.0}
        notes.append("'spring': joint damping c = 0 on every body axis"
                     + (f" (ignored {ignored} [N·m·s/rad])" if ignored else ""))
        c = dict.fromkeys(AXIS_ORDER, 0.0)
    axis = {a: BodyAxis(k[a], c[a], q0[a], lim[a]) for a in AXIS_ORDER}
    return BodyConnection(t, axes, yaw=axis["yaw"], pitch=axis["pitch"], roll=axis["roll"],
                          legacy=t in LEGACY_TREATMENTS, notes="; ".join(notes))


def connection_params(conn: BodyConnection) -> dict:
    """The body-joint parameters of a connection **as built**, by ``DESIGN_KEYS`` name (SI; for records and CSV):
    'spring' reports c = 0; a symmetric stop is its L, an asymmetric one the pair (lo, hi). Legacy connections add
    ``legacy: True``."""
    out = {"body_connection": conn.treatment, "body_roll_axis": conn.roll_axis}
    for q in ("k", "c", "q0", "limit"):
        for a in ("pitch", "yaw", "roll"):
            ax = conn.axis(a)
            lo, hi = ax.limits
            out[f"body_{q}_{a}"] = {"k": ax.stiffness, "c": ax.damping, "q0": ax.q0,
                                    "limit": hi if lo == -hi else (lo, hi)}[q]
    if conn.legacy:
        out["legacy"] = True
    return out


def design_connection(p: Mapping) -> BodyConnection:
    """The ``BodyConnection`` from a Myropod design-parameter dict (protocol §12.3), e.g.
    ``Myropod().resolve(body_connection='spring')``: the ``DESIGN_KEYS`` present (a resolved Myropod has them all;
    missing ones take the §12.1 defaults); other keys are ignored."""
    return make_connection(**{k: p[k] for k in DESIGN_KEYS if k in p})


def design_params(p: Mapping, **overrides) -> CleopatraParams:
    """CleopatraParams with the geometry of a Myropod design-parameter dict [mm, deg] (e.g. ``CLEO_MM`` or a
    resolved ``Myropod``); masses and placement stay the recorded values (``overrides`` change any field)."""
    mm = 1e-3
    geo = dict(seg_length=p["seg_length"] * mm, seg_width=p["seg_width"] * mm, seg_height=p["seg_height"] * mm,
               shell_thickness=p["shell_thickness"] * mm, joint_gap=p["joint_gap"] * mm,
               head_length=p["head_length"] * mm, femur=p["femur_length"] * mm, tibia=p["tibia_length"] * mm,
               leg_diameter=p["leg_diameter"] * mm, foot_diameter=p["foot_diameter"] * mm,
               hip_angle_deg=float(p["hip_angle_deg"]), knee_angle_deg=float(p["knee_angle_deg"]))
    if "leg_pair_spacing" in p:
        geo["leg_pair_spacing"] = float(p["leg_pair_spacing"])
    geo.update(overrides)
    return CleopatraParams(**geo)


#: Myropod geometry keys read by ``robot_from_design`` [mm, deg] (notebook 18's ``CLEO_MM`` plus the pair spacing).
GEOMETRY_KEYS = tuple(CLEO_MM) + ("leg_pair_spacing",)


def robot_from_design(p: Mapping | None = None, **kw) -> ch.Robot:
    """Cleopatra from a Myropod (Dedalus) design-parameter dict: one configuration for the CAD and the dynamics
    (protocol §12.3).

    Geometry [mm, deg] (``GEOMETRY_KEYS``; missing keys take notebook 18's ``CLEO_MM``) → ``design_params``; the
    body-joint keys (``DESIGN_KEYS``) → ``design_connection``. ``n_segments`` must be 3 and ``leg_sweep_deg`` 0
    (the Chiron model has neither more segments nor swept legs); CAD-only keys (part, version, bend_*, tether,
    pincer_*, case_*, guide_*, joint and pin diameters) are ignored. ``kw``: ``cleopatra``'s other keywords (kp,
    kd, armature, leg_links_collide). Example::

        import myropod, myropod_robot as mr
        p = myropod.Myropod().resolve(**mr.CLEO_MM, body_connection="spring", body_k_yaw=4.0)
        robot = mr.robot_from_design(p)
    """
    p = dict(p or {})
    clash = sorted(set(kw) & ({"connection", "params", "legacy"} | set(DESIGN_KEYS)))
    if clash:
        raise TypeError(f"robot_from_design: {clash} come from the design dict, not keywords")
    geo = {**CLEO_MM, **{k: p[k] for k in GEOMETRY_KEYS if k in p}}
    if int(geo["n_segments"]) != 3:
        raise ValueError(f"Cleopatra has 3 segments; the design dict says n_segments = {geo['n_segments']}")
    if float(p.get("leg_sweep_deg", 0.0)) != 0.0:
        raise ValueError("leg_sweep_deg must be 0: the Chiron model has no swept legs")
    return cleopatra(connection=design_connection(p), params=design_params(geo), **kw)


# ----------------------------------------------------------------------------------------------- names and poses
def cleopatra_servo(kp: float = KP, kd: float = KD, armature: float = 0.0, key: str = "smart servo 6 Nm") -> ch.Servo:
    """The leg servo (protocol §1): ``actuators.py``'s ``smart servo 6 Nm`` — stall 6.0 N·m, rated 2.0 N·m, 3.0 A
    at 12 V, no-load 55 rpm (5.76 rad/s) — as a PD loop (kp 40 N·m/rad, kd 0.8 N·m·s/rad) clipped to the
    torque–speed line. ``armature`` [kg·m²]: reflected rotor inertia; the protocol and datasheet give none, so 0."""
    return ch.Servo.from_actuator(act.get(key), kp=kp, kd=kd, armature=armature)


def leg_name(segment: int, key: str) -> str:
    """Foot / leg name, e.g. ``leg_name(1, 'front-left') == 's1 front-left'`` (notebook 18's names)."""
    return f"s{segment} {key}"


def leg_joints(segment: int, key: str) -> list:
    """The leg's actuated joints, hip to foot: hip yaw, hip pitch, knee."""
    n = leg_name(segment, key)
    return [f"{n} hip_yaw", f"{n} hip_pitch", f"{n} knee"]


def body_joints(connection="spring_damper", n_segments: int = 3, *, roll_axis=True, legacy=False) -> list:
    """Names of the passive body joints, ``body <i>-<i+1> <axis>`` in chain order, of a ``BodyConnection`` or a
    treatment name (with ``roll_axis``; both Amendment D treatments have the same joints; legacy 'rigid': none)."""
    if isinstance(connection, BodyConnection):
        conn = connection
    else:
        conn = make_connection(connection, body_roll_axis=roll_axis, legacy=legacy)
    return [f"body {i}-{i + 1} {a}" for i in range(1, n_segments) for a in conn.hinged]


def hip_position(p: CleopatraParams, key: str) -> tuple:
    """Hip position in its segment's frame [m]."""
    sx, sy = LEG_SIDES[key]
    return (sx * p.hip_x, sy * p.hip_y, 0.0)


def stance_foot(p: CleopatraParams, key: str) -> tuple:
    """Standing pad centre relative to its hip, segment frame [m]: foot_out outward, foot_drop down."""
    sy = LEG_SIDES[key][1]
    return (0.0, sy * p.foot_out, -p.foot_drop)


def nominal_pose(p: CleopatraParams | None = None, connection: BodyConnection | None = None) -> dict:
    """The standing pose (protocol §1): every leg at hip yaw 0, hip pitch 40°, knee 85° − 40° = 45° (relative);
    every body hinge of ``connection`` at its q0 (springs unloaded, §12.3). Joint → angle [rad]."""
    p = p or CleopatraParams()
    q = {}
    for s in (1, 2, 3):
        for key in LEG_KEYS:
            yaw, pitch, knee = leg_joints(s, key)
            q[yaw] = 0.0
            q[pitch] = math.radians(p.hip_angle_deg)
            q[knee] = math.radians(p.knee_angle_deg - p.hip_angle_deg)
    if connection is not None:
        for name in body_joints(connection):
            q[name] = connection.axis(name.rsplit(" ", 1)[1]).q0
    return q


# ----------------------------------------------------------------------------------------------- building blocks
SHELL_RGBA = (0.35, 0.37, 0.40, 1.0)
LEG_RGBA = (0.17, 0.18, 0.19, 1.0)
HEAD_RGBA = (0.30, 0.32, 0.35, 1.0)
PAD_RGBA = (0.10, 0.10, 0.10, 1.0)


def _plates(prefix, length, width, height, thickness, mass, rgba):
    """Six wall plates of a thin-walled box (outer size length × width × height) centred at the link origin, the
    mass shared by area: visual (non-colliding) geoms that carry the shell's mass and inertia."""
    t = thickness
    walls = [  # name, area, half sizes, position
        ("top", length * width, (length / 2, width / 2, t / 2), (0.0, 0.0, height / 2 - t / 2)),
        ("bottom", length * width, (length / 2, width / 2, t / 2), (0.0, 0.0, -height / 2 + t / 2)),
        ("left", length * height, (length / 2, t / 2, height / 2 - t), (0.0, width / 2 - t / 2, 0.0)),
        ("right", length * height, (length / 2, t / 2, height / 2 - t), (0.0, -width / 2 + t / 2, 0.0)),
        ("front", width * height, (t / 2, width / 2 - t, height / 2 - t), (length / 2 - t / 2, 0.0, 0.0)),
        ("rear", width * height, (t / 2, width / 2 - t, height / 2 - t), (-length / 2 + t / 2, 0.0, 0.0)),
    ]
    total = sum(w[1] for w in walls)
    return [ch.Geom(f"{prefix} wall {n}", "box", hs, pos=pos, mass=mass * a / total, role="visual", rgba=rgba)
            for n, a, hs, pos in walls]


def _leg(p: CleopatraParams, segment: int, key: str, servo: ch.Servo, leg_role: str) -> ch.Link:
    """One leg: a femur link (hip yaw then hip pitch at the hip — the pitch axis turns with the yaw) and a tibia
    link (knee at the femur's end) carrying the foot pad."""
    sx, sy = LEG_SIDES[key]
    name = leg_name(segment, key)
    yaw_j, pitch_j, knee_j = leg_joints(segment, key)
    # Signs (gait.fk_myropod): yaw + sweeps the foot forward (+x) → left legs turn about −z, right legs about +z;
    # pitch + puts the femur below horizontal → left legs (reaching +y) turn about −x, right legs about +x.
    yaw_axis = (0.0, 0.0, -float(sy))
    pitch_axis = (-float(sy), 0.0, 0.0)
    r_leg, r_tib, r_pad = p.leg_diameter / 2, p.leg_diameter * p.tibia_diameter_ratio / 2, p.foot_diameter / 2
    bar_friction = (p.foot_mu, 0.005, 0.0001)
    tibia = ch.Link(
        f"{name} tibia", pos=(0.0, sy * p.femur, 0.0),
        joints=[ch.Joint(knee_j, axis=pitch_axis, range=LEG_RANGES["knee"], tag="knee", servo=servo, leg=name)],
        geoms=[ch.Geom(f"{name} tibia bar", "capsule", (r_tib,), fromto=(0, 0, 0, 0, sy * p.tibia, 0),
                       mass=p.tibia_mass, role=leg_role, friction=bar_friction, rgba=LEG_RGBA),
               ch.Geom(f"{name} foot", "sphere", (r_pad,), pos=(0.0, sy * p.tibia, 0.0), mass=p.pad_mass,
                       role="foot", friction=(p.foot_mu, 0.005, 0.0001), rgba=PAD_RGBA)])
    return ch.Link(
        f"{name} femur", pos=hip_position(p, key),
        joints=[ch.Joint(yaw_j, axis=yaw_axis, range=LEG_RANGES["hip_yaw"], tag="hip_yaw", servo=servo, leg=name),
                ch.Joint(pitch_j, axis=pitch_axis, range=LEG_RANGES["hip_pitch"], tag="hip_pitch", servo=servo,
                         leg=name)],
        geoms=[ch.Geom(f"{name} femur bar", "capsule", (r_leg,), fromto=(0, 0, 0, 0, sy * p.femur, 0),
                       mass=p.femur_mass, role=leg_role, friction=bar_friction, rgba=LEG_RGBA)],
        masses=[ch.PointMass(f"{name} knee servo", p.leg_servo_mass, (0.0, sy * p.femur, 0.0))],
        children=[tibia])


def _segment(p: CleopatraParams, index: int, servo: ch.Servo, leg_role: str) -> ch.Link:
    """Segment ``index`` (1-based) with its legs, without its connection to the segment ahead."""
    name = f"segment {index}"
    geoms = [ch.Geom(f"{name} shell", "box", (p.seg_length / 2, p.seg_width / 2, p.seg_height / 2), role="body",
                     friction=(p.shell_mu, 0.005, 0.0001), rgba=SHELL_RGBA)]
    geoms += _plates(name, p.seg_length, p.seg_width, p.seg_height, p.shell_thickness, p.shell_mass, SHELL_RGBA)
    masses = []
    for key in LEG_KEYS:                      # hip yaw + hip pitch servos of each leg, at its hip
        masses.append(ch.PointMass(f"{name} {key} hip servos", 2 * p.leg_servo_mass, hip_position(p, key)))
    if p.body_servos_per_segment == 2:
        masses += [ch.PointMass(f"{name} body servo front", p.body_servo_mass, (p.body_servo_x, 0.0, 0.0)),
                   ch.PointMass(f"{name} body servo rear", p.body_servo_mass, (-p.body_servo_x, 0.0, 0.0))]
    elif p.body_servos_per_segment:
        masses.append(ch.PointMass(f"{name} body servos", p.body_servos_per_segment * p.body_servo_mass,
                                   (0.0, 0.0, 0.0)))
    masses += [ch.PointMass(f"{name} PCB", p.pcb_mass, (0.0, 0.0, p.pcb_z)),
               ch.PointMass(f"{name} wiring", p.wiring_mass, (0.0, 0.0, 0.0))]
    if index == 1:
        masses.append(ch.PointMass("compute", p.compute_mass, (0.0, 0.0, p.compute_z)))
    if index == 2:
        masses.append(ch.PointMass("battery", p.battery_mass, (0.0, 0.0, p.battery_z)))
    legs = [_leg(p, index, key, servo, leg_role) for key in LEG_KEYS]
    return ch.Link(name, geoms=geoms, masses=masses, children=legs, log=True, group=name)


def _head(p: CleopatraParams) -> ch.Link:
    """The head — the robot's root link; segment 1 is welded behind it in every treatment (§1, §12.1). A tapered
    shell (110 × 70 mm at the back, 85 % at the front) modelled as a thin-walled box of its mean section."""
    w = p.seg_width * (1 + p.head_taper) / 2
    h = p.seg_height * (1 + p.head_taper) / 2
    geoms = [ch.Geom("head shell", "box", (p.head_length / 2, w / 2, h / 2), role="body",
                     friction=(p.shell_mu, 0.005, 0.0001), rgba=HEAD_RGBA)]
    geoms += _plates("head", p.head_length, w, h, p.shell_thickness, p.head_shell_mass, HEAD_RGBA)
    payload = dict(p.head_payload)
    cams = sum(m for k, m in payload.items() if "camera" in k or "LWIR" in k)
    rest = sum(payload.values()) - cams
    masses = [ch.PointMass("head cameras and LWIR", cams, (p.camera_x, 0.0, 0.0)),
              ch.PointMass("head microphones, IMU, radio", rest, (0.0, 0.0, 0.0))]
    return ch.Link("head", geoms=geoms, masses=masses, log=True, group="head")


def _quat_axis(axis, angle):
    s = math.sin(angle / 2)
    return (math.cos(angle / 2), axis[0] * s, axis[1] * s, axis[2] * s)


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def _rotate(q, v):
    w, x, y, z = q
    R = ((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
         (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
         (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)))
    return tuple(sum(R[i][j] * v[j] for j in range(3)) for i in range(3))


def _connect(child: ch.Link, i: int, p: CleopatraParams, conn: BodyConnection) -> None:
    """Hang segment i+1 (``child``) behind segment i at the pin (half a pitch behind segment i's centre): the fixed
    rotation of the axes that are not hinged (identity in Amendment D: their q0 is 0), then the passive hinges about
    the pin in chain order, each with its k, c, q0 (spring reference) and stops; armature and friction loss 0."""
    q = (1.0, 0.0, 0.0, 0.0)
    for a in AXIS_ORDER:
        if a not in conn.hinged:
            q = _quat_mul(q, _quat_axis(AXIS_VECTOR[a], conn.axis(a).q0))
    half = p.pitch / 2
    off = _rotate(q, (-half, 0.0, 0.0))
    child.pos = (-half + off[0], off[1], off[2])
    child.quat = q
    joints = []
    for a in conn.hinged:
        ax = conn.axis(a)
        joints.append(ch.Joint(f"body {i}-{i + 1} {a}", axis=AXIS_VECTOR[a], pos=(half, 0.0, 0.0), range=ax.limits,
                               stiffness=ax.stiffness, damping=ax.damping, springref=ax.q0, armature=0.0,
                               frictionloss=0.0, tag=f"body_{a}"))
    child.joints = joints


# ----------------------------------------------------------------------------------------------- the robot
def cleopatra(body_connection: str = BODY_DEFAULTS["body_connection"], *,
              body_roll_axis=BODY_DEFAULTS["body_roll_axis"],
              body_k_pitch=8.0, body_k_yaw=8.0, body_k_roll=8.0,
              body_c_pitch=0.2, body_c_yaw=0.2, body_c_roll=0.2,
              body_q0_pitch=0.0, body_q0_yaw=0.0, body_q0_roll=0.0,
              body_limit_pitch=LIMIT_PITCH, body_limit_yaw=LIMIT_YAW, body_limit_roll=LIMIT_ROLL,
              legacy=False, connection: BodyConnection | None = None, params: CleopatraParams | None = None,
              kp: float = KP, kd: float = KD, armature: float = 0.0, leg_links_collide: bool = False) -> ch.Robot:
    """Cleopatra (protocol §1) with an Amendment D body treatment (§12.1).

    ``body_connection``: 'spring' (joint torque −k (q − q0), c = 0) or 'spring_damper' (−k (q − q0) − c q̇).
    Both build the **identical** body joints — hinges on yaw and pitch, plus roll when ``body_roll_axis`` (bool;
    'true'/'false', 1/0 accepted), chained yaw → pitch → roll at the pins between segments 1–2 and 2–3, with
    ``body_k_<axis>`` k [N·m/rad] (default 8), ``body_q0_<axis>`` q0 [rad] (0; the spring's rest angle and the
    start pose), ``body_limit_<axis>`` [rad] stops at ± the value (π/4 pitch and yaw, π/9 roll; or a pair (lo,
    hi)), armature 0, friction loss 0, no actuator — and differ only in ``body_c_<axis>`` c [N·m·s/rad] (default
    0.2), which 'spring' replaces by 0 (``robot.notes`` records the ignored values). Without the roll hinge the
    roll values are unused and ``body_q0_roll`` must be 0. Numbers may be ints, floats or numeric strings
    (``chiron run -p body_k_pitch=8``).

    ``legacy=True`` unlocks Amendment C's 'rigid' | 'flexible' | 'flexible+yaw' (pre-Amendment D; labelled
    ``legacy:<name>``, never pooled). ``connection``: a ready ``BodyConnection`` (from ``make_connection`` or
    ``design_connection``) instead of the ``body_*`` keywords, which must then stay at their defaults.
    ``params``: the robot's numbers (``CleopatraParams``; ``design_params`` / ``robot_from_design`` read a Myropod
    design dict). ``kp``/``kd`` [N·m/rad, N·m·s/rad] and ``armature`` [kg·m²] of every leg servo (§1: 40, 0.8; no
    armature given). ``leg_links_collide`` lets the femur and tibia bars touch the ground (§1: only pads, shells
    and head do).

    Returns a ``chiron.Robot`` named ``cleopatra <treatment>``: logged bodies head, segment 1–3 (groups = their
    names); 12 feet ``s<i> <leg>`` on their segments; 36 actuated leg joints ``s<i> <leg> hip_yaw|hip_pitch|knee``;
    passive body hinges ``body <i>-<i+1> yaw|pitch|roll`` (tags ``body_yaw|body_pitch|body_roll``); the standing
    pose of §1 with the body hinges at q0 (springs unloaded) and its base height (the root frame — the head's centre,
    level with the hips — above flat ground), 0.163 m for a straight chain. Bookkeeping attributes: ``treatment``
    ('spring', 'spring_damper' or 'legacy:<name>'), ``legacy``, ``connection`` (the BodyConnection as built),
    ``body_params`` (``connection_params``), ``params``.
    """
    body = {"body_connection": body_connection, "body_roll_axis": body_roll_axis,
            "body_k_pitch": body_k_pitch, "body_k_yaw": body_k_yaw, "body_k_roll": body_k_roll,
            "body_c_pitch": body_c_pitch, "body_c_yaw": body_c_yaw, "body_c_roll": body_c_roll,
            "body_q0_pitch": body_q0_pitch, "body_q0_yaw": body_q0_yaw, "body_q0_roll": body_q0_roll,
            "body_limit_pitch": body_limit_pitch, "body_limit_yaw": body_limit_yaw, "body_limit_roll": body_limit_roll}
    legacy = as_bool(legacy, "legacy")
    if connection is None:
        connection = make_connection(**body, legacy=legacy)
    else:
        changed = sorted(k for k, v in body.items() if not _same(v, BODY_DEFAULTS[k]))
        if changed:
            raise ValueError(f"pass either connection= or the body keywords {changed}, not both")
        if connection.legacy and not legacy:
            raise ValueError(f"connection {connection.treatment!r} is legacy (pre-Amendment D): pass legacy=True")
    p = params or CleopatraParams()
    servo = cleopatra_servo(kp, kd, armature, p.leg_servo)
    leg_role = "link" if leg_links_collide else "visual"
    segs = [_segment(p, i, servo, leg_role) for i in (1, 2, 3)]
    head = _head(p)                                   # the root link (free joint); segment 1 is welded behind it
    segs[0].pos = (-p.head_offset, 0.0, 0.0)
    head.children.append(segs[0])
    for i in (1, 2):                                  # segment i+1 hangs behind segment i
        _connect(segs[i], i, p, connection)
        segs[i - 1].children.append(segs[i])
    feet = [ch.FootSpec(leg_name(s, k), f"{leg_name(s, k)} foot", leg_joints(s, k), f"segment {s}")
            for s in (1, 2, 3) for k in LEG_KEYS]
    label = f"legacy:{connection.treatment}" if connection.legacy else connection.treatment
    sources = {
        "geometry": "designs/myropod.py with notebook 18's CLEO parameters (CLEO_MM); protocol §1",
        "masses": "notebook 18 mass budget (CAD volume x 1.1 g/cm^3, actuators.py servo masses, recorded payload); "
                  "protocol §1 table",
        "leg servo": f"actuators.py '{p.leg_servo}': {act.get(p.leg_servo).source}; kp {kp}, kd {kd} (protocol §1)",
        "body connection": ("protocol §11.2 (Amendment C) — LEGACY, pre-Amendment D" if connection.legacy else
                            "protocol §12.1 (Amendment D, 2026-10-01): identical passive hinges (pitch, yaw, roll "
                            "configurable), defaults k 8 N·m/rad, c 0.2 N·m·s/rad (spring_damper; 0 in spring), "
                            "q0 0 rad, limits ±45° pitch and yaw, ±20° roll"),
        "friction": "protocol §1: pads 0.8 (rubber on rock), shells and head 0.5",
        "placement": "CleopatraParams docstring (thin-walled shells, payload positions)",
    }
    per_axis = "; ".join(
        f"{a}: k {connection.axis(a).stiffness:g} N·m/rad, c {connection.axis(a).damping:g} N·m·s/rad, "
        f"q0 {connection.axis(a).q0:g} rad, limits [{connection.axis(a).limits[0]:.6g}, "
        f"{connection.axis(a).limits[1]:.6g}] rad" for a in connection.hinged)
    hinges = (f"passive hinges {' -> '.join(connection.hinged)} at body joints 1-2 and 2-3 ({per_axis}); armature 0, "
              f"friction loss 0, no body actuator, translations constrained" if connection.hinged else
              "segments welded at the neutral angles (no body degree of freedom)")
    notes = ("LEGACY (pre-Amendment D; never pooled with Amendment D results). " if connection.legacy else "") \
        + f"Cleopatra, treatment '{label}': {hinges}" \
        + (f"; {connection.notes}" if connection.notes else "") \
        + f"; leg servo armature {armature} kg·m²; params {asdict(p)}"
    bent = any(connection.axis(a).q0 != 0.0 for a in connection.hinged)
    robot = ch.Robot(f"cleopatra {label}", head, feet=feet, nominal_qpos=nominal_pose(p, connection),
                     # a bent chain (q0 ≠ 0) stands with its lowest foot on the ground (Chiron computes it)
                     nominal_base_height=None if bent else p.hip_height, nominal_hip_height=p.hip_height,
                     notes=notes, sources=sources)
    robot.validate()
    robot.treatment = label              # plain attributes for bookkeeping (not part of chiron.Robot's fields)
    robot.legacy = connection.legacy
    robot.connection = connection
    robot.body_params = connection_params(connection)
    robot.params = p
    return robot


def _same(a, b) -> bool:
    """Whether a body keyword equals its default (numbers by value, so 8 == 8.0)."""
    if isinstance(b, bool):
        try:
            return as_bool(a) == b
        except ValueError:
            return False
    if isinstance(b, str):
        return a == b
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return False


#: ChironLab settings for Cleopatra [s]. Protocol §1: control at 1 kHz, log at 100 Hz, time step 1 ms — but the
#: previous build failed the timestep-convergence check (§11.5, §12.5) at 1 ms and 0.5 ms: with the servo's kd
#: integrated implicitly even when its torque is clipped, a saturated joint is under-accelerated by M / (M + dt·kd)
#: (M its effective inertia: ≈ 9e-5 kg·m² for a knee at the standing pose → 0.10 at 1 ms, 0.32 at 0.25 ms;
#: Amendment E §13.2). At 0.25 ms the walking speed and tilt agreed with 0.125 ms within 4 % and 1°, so the physics
#: step is 0.25 ms and the controller still runs at 1 kHz. §13.2: once the core gives a clipped joint its clipped
#: torque with no implicit damping, the study's step is re-chosen by the §12.5 check and recorded in the study
#: notebook. ``LAB_OPTIONS_PROTOCOL`` keeps the literal 1 ms.
LAB_OPTIONS = dict(timestep=0.00025, control_dt=0.001, log_dt=0.01)
LAB_OPTIONS_PROTOCOL = dict(timestep=0.001, control_dt=0.001, log_dt=0.01)
#: Protocol §4: every course is 1.5 m long and starts flat for 0.3 m.
COURSE_M = 1.5


def cleopatra_lab(body_connection: str = BODY_DEFAULTS["body_connection"], terrain=None, *,
                  robot: ch.Robot | None = None, robot_kw: dict | None = None, **lab_kw) -> ch.ChironLab:
    """A ChironLab with Cleopatra — ``cleopatra(body_connection, **robot_kw)`` (e.g. ``robot_kw={'body_k_yaw':
    4.0}``), or ``robot`` — on ``terrain`` (default flat) with ``LAB_OPTIONS`` (overridden by ``lab_kw``, e.g.
    ``course_extent``, ``flat_as_plane``, ``gravity``)."""
    opts = dict(LAB_OPTIONS)
    opts.update(lab_kw)
    return ch.ChironLab(robot if robot is not None else cleopatra(body_connection, **(robot_kw or {})), terrain, **opts)


def failure_rules(v_target: float, course_m: float = COURSE_M, **kw) -> ch.FailureRules:
    """The protocol's §5 rules for a trial at ``v_target`` [m/s] (tilt 60°, COM below 40 % of the hip height for
    0.5 s, stall after 2 s over 3 s windows at 10 %, |y| ≤ 0.5 m, timeout 2·course/v + 2 s)."""
    return ch.FailureRules(course_m=course_m, v_target=v_target, **kw)


def mass_budget(p: CleopatraParams | None = None) -> dict:
    """Notebook 18's budget recomputed from the parameters [kg] (the robot's ``total_mass()`` must equal it)."""
    p = p or CleopatraParams()
    seg = (p.shell_mass + p.leg_cad_mass + 12 * p.leg_servo_mass + p.body_servos_per_segment * p.body_servo_mass
           + p.pcb_mass + p.wiring_mass)
    head = p.head_shell_mass + sum(m for _, m in p.head_payload)
    return {"per segment": seg, "segments": 3 * seg, "head": head, "battery": p.battery_mass,
            "compute": p.compute_mass, "total": 3 * seg + head + p.battery_mass + p.compute_mass}


if __name__ == "__main__":  # pragma: no cover
    for t in TREATMENTS:
        r = cleopatra(t)
        print(f"{t:13s} {r.total_mass():.4f} kg, {len(r.actuated_joints())} servos, body joints "
              f"{[(j.name, j.stiffness, j.damping) for j in r.joints() if j.servo is None]}")
