"""Cleopatra as a Chiron robot: the three-segment Myropod of notebook 18 and its body-connection treatments.

The pre-registered stability protocol (``docs/myropod_stability.md``) fixes everything here: §1 the robot (geometry,
masses, joints, actuators, friction) and §11 (Amendment C, which supersedes §2) the treatments. ``cleopatra()``
returns a ``chiron.Robot``; the treatments differ **only** in how segments 1–2 and 2–3 are joined, never in a body,
a mass, an inertia, a leg, a friction or an actuator:

* ``rigid`` — the rear segment is welded to the front one at the neutral angles (no degree of freedom);
* ``flexible`` — passive spring–damper hinges at the joint pin on pitch and roll (k = 8 N·m/rad, c = 0.2 N·m·s/rad,
  θ₀ = 0°, limits ±45° pitch, ±20° roll);
* ``flexible+yaw`` — as ``flexible`` plus a yaw hinge (k = 8, c = 0.2, ±45°; its stiffness is ``yaw_stiffness``).

Every per-axis value (k, c, θ₀, limits) is a field of ``BodyConnection`` / ``BodyAxis``; ``design_connection`` and
``design_params`` read the same settings from a Myropod design-parameter dict (§11.3).

Frames (Chiron's convention): x forward (the head at +x), y left, z up. Every segment's link frame sits at the
segment's centre at **hip height** (the CAD's ``seg_height / 2`` above the shell's belly), so a hip is at
``(±hip_x, ±hip_y, 0)`` in its segment's frame; the head's frame is at its centre at the same height. The chain
grows backwards from the root link, the head (it floats freely; Chiron logs the bodies in tree order: head,
segment 1, segment 2, segment 3): head → segment 1 (welded, 0.160 m behind it) → segment 2 → segment 3, one pitch
(0.170 m) apart; a body joint sits at the pin halfway across the joint gap. Body-joint signs: yaw about +z
(+ swings the rear segment's tail to the right), pitch about +y (+ lifts the rear segment's tail), roll about +x;
hinges are chained yaw → pitch → roll.

Legs: hip yaw (axis: segment z; positive sweeps the foot forward), hip pitch (horizontal, perpendicular to the
leg plane; positive = femur below horizontal) and knee (parallel to hip pitch; the **relative** angle, positive =
tibia folds further down). These are ``gait.ik_myropod``'s conventions: its knee is the tibia's absolute angle
below horizontal, so the physical knee joint is ``knee_abs − hip_pitch``. At zero every leg sticks straight out
sideways.

Units: SI here (m, kg, s, N·m, rad); the design files (``myropod.py``, ``gait.py``) work in mm and degrees and are
converted at the boundary. Every number carries its source in ``CleopatraParams`` and ``Robot.sources``.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import actuators as act
from vegeta import chiron as ch

__all__ = ["CLEO_MM", "CleopatraParams", "BodyAxis", "BodyConnection", "TREATMENTS", "TREATMENT_CONNECTIONS",
           "LEGACY_TREATMENTS", "LEG_KEYS", "LEG_SIDES", "SEGMENTS", "LEG_RANGES", "KP", "KD", "cleopatra",
           "cleopatra_servo", "canonical_treatment", "treatment_connection", "design_connection", "design_params",
           "leg_name", "leg_joints", "body_joints", "nominal_pose", "mass_budget", "hip_position", "stance_foot",
           "LAB_OPTIONS", "LAB_OPTIONS_PROTOCOL", "COURSE_M", "cleopatra_lab", "failure_rules"]

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


# ----------------------------------------------------------------------------------------------- body connections
@dataclass(frozen=True)
class BodyAxis:
    """One passive body-joint axis (protocol §11.2): torque ``τ = −k (θ − θ₀) − c θ̇`` plus hard stops.

    ``stiffness`` k [N·m/rad], ``damping`` c [N·m·s/rad], ``neutral_deg`` θ₀ [deg] (the spring's rest angle; the
    segments start there and a rigid connection is welded there), ``limits_deg`` (lo, hi) [deg] hard stops
    (absolute joint angles)."""

    stiffness: float = 8.0
    damping: float = 0.2
    neutral_deg: float = 0.0
    limits_deg: tuple = (-45.0, 45.0)

    def __post_init__(self):
        lo, hi = self.limits_deg
        if not lo < hi:
            raise ValueError("limits_deg must be (lo, hi) with lo < hi")
        if not lo <= self.neutral_deg <= hi:
            raise ValueError("neutral_deg must lie within limits_deg")
        if self.stiffness < 0 or self.damping < 0:
            raise ValueError("stiffness and damping must be non-negative")


#: Protocol §11.2 defaults: k = 8 N·m/rad, c = 0.2 N·m·s/rad, θ₀ = 0°, limits ±45° pitch, ±20° roll, ±45° yaw.
PITCH_AXIS = BodyAxis(8.0, 0.2, 0.0, (-45.0, 45.0))
ROLL_AXIS = BodyAxis(8.0, 0.2, 0.0, (-20.0, 20.0))
YAW_AXIS = BodyAxis(8.0, 0.2, 0.0, (-45.0, 45.0))
#: Hinge order at a body joint, from the front segment to the rear one (yaw pin, then pitch pin, then roll).
AXIS_ORDER = ("yaw", "pitch", "roll")
AXIS_VECTOR = {"yaw": (0.0, 0.0, 1.0), "pitch": (0.0, 1.0, 0.0), "roll": (1.0, 0.0, 0.0)}


@dataclass(frozen=True)
class BodyConnection:
    """How segments 1–2 and 2–3 are joined (protocol §11.2; the head is welded to segment 1 in every mode).

    ``mode`` 'rigid' — the rear segment is welded to the front one (no degree of freedom), oriented at the neutral
    angles of ``yaw``/``pitch``/``roll``; 'flexible' — passive spring–damper hinges at the joint pin on ``axes``
    (default pitch and roll; yaw optional), each with its ``BodyAxis``. An axis that is not hinged is fixed at its
    neutral angle (built into the link's orientation, before the hinges). Translations stay constrained.
    """

    mode: str = "flexible"
    axes: tuple = ("pitch", "roll")
    pitch: BodyAxis = PITCH_AXIS
    roll: BodyAxis = ROLL_AXIS
    yaw: BodyAxis = YAW_AXIS

    def __post_init__(self):
        if self.mode not in ("rigid", "flexible"):
            raise ValueError("mode must be 'rigid' or 'flexible'")
        bad = [a for a in self.axes if a not in AXIS_ORDER]
        if bad:
            raise ValueError(f"unknown body axes {bad}; use {AXIS_ORDER}")

    @property
    def hinged(self) -> tuple:
        """The hinged axes in chain order (empty when rigid)."""
        return () if self.mode == "rigid" else tuple(a for a in AXIS_ORDER if a in self.axes)

    def axis(self, name: str) -> BodyAxis:
        return getattr(self, name)


#: The treatments run (protocol §11.2; they supersede §2's locked / flexible / flexible+roll).
TREATMENT_CONNECTIONS = {
    "rigid": BodyConnection(mode="rigid"),
    "flexible": BodyConnection(mode="flexible", axes=("pitch", "roll")),
    "flexible+yaw": BodyConnection(mode="flexible", axes=("yaw", "pitch", "roll")),
}
TREATMENTS = tuple(TREATMENT_CONNECTIONS)
#: §2's retired names that have an identical §11 meaning (the same hinged axes, the same values). §2's 'flexible'
#: (pitch + yaw) has none: 'flexible' now means pitch + roll.
LEGACY_TREATMENTS = {"locked": "rigid", "flexible+roll": "flexible+yaw"}


def canonical_treatment(treatment: str) -> str:
    """'rigid' | 'flexible' | 'flexible+yaw' (§2's 'locked' and 'flexible+roll' are mapped; anything else fails)."""
    name = LEGACY_TREATMENTS.get(treatment, treatment)
    if name not in TREATMENT_CONNECTIONS:
        raise ValueError(f"treatment {treatment!r}: use one of {TREATMENTS} (protocol §11.2)")
    return name


def treatment_connection(treatment: str, yaw_stiffness: float | None = None) -> BodyConnection:
    """The BodyConnection of a named treatment, with an optional body-yaw stiffness [N·m/rad] (§9.3 exp. 8)."""
    conn = TREATMENT_CONNECTIONS[canonical_treatment(treatment)]
    if yaw_stiffness is not None:
        conn = replace(conn, yaw=replace(conn.yaw, stiffness=float(yaw_stiffness)))
    return conn


#: Myropod design-parameter names read by ``design_connection`` (protocol §11.3; [deg] for angles).
DESIGN_KEYS = ("body_connection", "body_yaw", *(f"{a}_{q}" for a in AXIS_ORDER
                                                for q in ("stiffness", "damping", "neutral_deg", "limit_deg")))


def design_connection(p: dict) -> BodyConnection:
    """The BodyConnection from a Myropod design-parameter dict (protocol §11.3), e.g. ``Myropod.resolve(...)``.

    Reads ``body_connection`` ('rigid' | 'flexible'), ``body_yaw`` (bool: the optional yaw hinge) and per axis
    ``<axis>_stiffness`` [N·m/rad], ``<axis>_damping`` [N·m·s/rad], ``<axis>_neutral_deg`` [deg] and
    ``<axis>_limit_deg`` [deg] (symmetric stops ±limit) for axis in yaw, pitch, roll. Missing keys take the §11.2
    defaults."""
    defaults = {"yaw": YAW_AXIS, "pitch": PITCH_AXIS, "roll": ROLL_AXIS}
    axes = {}
    for a, d in defaults.items():
        lim = p.get(f"{a}_limit_deg")
        axes[a] = BodyAxis(float(p.get(f"{a}_stiffness", d.stiffness)), float(p.get(f"{a}_damping", d.damping)),
                           float(p.get(f"{a}_neutral_deg", d.neutral_deg)),
                           (-float(lim), float(lim)) if lim is not None else d.limits_deg)
    hinged = ("yaw", "pitch", "roll") if bool(p.get("body_yaw", False)) else ("pitch", "roll")
    return BodyConnection(mode=str(p.get("body_connection", "flexible")), axes=hinged, **axes)


def design_params(p: dict, **overrides) -> CleopatraParams:
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


def body_joints(treatment="rigid", n_segments: int = 3) -> list:
    """Names of the passive body joints of a treatment name or BodyConnection (empty when rigid)."""
    conn = treatment if isinstance(treatment, BodyConnection) else treatment_connection(treatment)
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
    every body hinge at its neutral angle (springs unloaded, §11.2). Joint → angle [rad]."""
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
            q[name] = math.radians(connection.axis(name.rsplit(" ", 1)[1]).neutral_deg)
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
    """The head — the robot's root link; segment 1 is welded behind it in every treatment (§1, §11.2). A tapered
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
    """Hang segment i+1 (``child``) behind segment i at the pin (half a pitch behind segment i's centre): fixed
    rotation for the axes that are not hinged (all of them when rigid), then the hinges about the pin."""
    q = (1.0, 0.0, 0.0, 0.0)
    for a in AXIS_ORDER:
        if a not in conn.hinged:
            q = _quat_mul(q, _quat_axis(AXIS_VECTOR[a], math.radians(conn.axis(a).neutral_deg)))
    half = p.pitch / 2
    off = _rotate(q, (-half, 0.0, 0.0))
    child.pos = (-half + off[0], off[1], off[2])
    child.quat = q
    joints = []
    for a in conn.hinged:
        ax = conn.axis(a)
        joints.append(ch.Joint(f"body {i}-{i + 1} {a}", axis=AXIS_VECTOR[a], pos=(half, 0.0, 0.0),
                               range=(math.radians(ax.limits_deg[0]), math.radians(ax.limits_deg[1])),
                               stiffness=ax.stiffness, damping=ax.damping, springref=math.radians(ax.neutral_deg),
                               tag=f"body_{a}"))
    child.joints = joints


# ----------------------------------------------------------------------------------------------- the robot
def cleopatra(treatment: str = "rigid", yaw_stiffness: float | None = None, *,
              connection: BodyConnection | None = None, params: CleopatraParams | None = None, kp: float = KP,
              kd: float = KD, armature: float = 0.0, leg_links_collide: bool = False) -> ch.Robot:
    """Cleopatra (protocol §1) with a body treatment (§11.2).

    ``treatment``: 'rigid', 'flexible' (pitch + roll hinges) or 'flexible+yaw' (pitch + roll + yaw); §2's
    'locked' and 'flexible+roll' are accepted as the same physics ('rigid', 'flexible+yaw'). ``yaw_stiffness``
    [N·m/rad] sets the body-yaw spring (None = 8; §9.3 experiment 8 varies it). ``connection``: a full
    ``BodyConnection`` instead of a named treatment (every per-axis k, c, θ₀, limit). ``params``: the robot's
    numbers (``CleopatraParams``; ``design_params`` builds them from a Myropod parameter dict). ``kp``/``kd``
    [N·m/rad, N·m·s/rad] and ``armature`` [kg·m²] of every leg servo (§1: 40, 0.8; no armature given).
    ``leg_links_collide`` lets the femur and tibia bars touch the ground (§1: only pads, shells and head do).

    Returns a ``chiron.Robot``: logged bodies head, segment 1–3 (groups = their names); 12 feet ``s<i> <leg>``
    on their segments; 36 actuated leg joints ``s<i> <leg> hip_yaw|hip_pitch|knee``; passive body hinges
    ``body <i>-<i+1> yaw|pitch|roll``; the standing pose of §1 with body hinges at their neutral angles, and its
    base height (the root frame — the head's centre, level with the hips — above flat ground), 0.163 m.
    """
    if connection is None:
        connection = treatment_connection(treatment, yaw_stiffness)
        name = canonical_treatment(treatment)
    else:
        if yaw_stiffness is not None:
            connection = replace(connection, yaw=replace(connection.yaw, stiffness=float(yaw_stiffness)))
        name = "custom"
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
    sources = {
        "geometry": "designs/myropod.py with notebook 18's CLEO parameters (CLEO_MM); protocol §1",
        "masses": "notebook 18 mass budget (CAD volume x 1.1 g/cm^3, actuators.py servo masses, recorded payload); "
                  "protocol §1 table",
        "leg servo": f"actuators.py '{p.leg_servo}': {act.get(p.leg_servo).source}; kp {kp}, kd {kd} (protocol §1)",
        "body connection": "protocol §11.2 (Amendment C): rigid weld, or hinges with k 8 N·m/rad, c 0.2 N·m·s/rad, "
                           "θ0 0°, limits ±45° pitch, ±20° roll, ±45° yaw",
        "friction": "protocol §1: pads 0.8 (rubber on rock), shells and head 0.5",
        "placement": "CleopatraParams docstring (thin-walled shells, payload positions)",
    }
    notes = (f"Cleopatra, treatment '{name}': {connection.mode}"
             + (f" ({', '.join(connection.hinged)})" if connection.hinged else "")
             + f"; connection {asdict(connection)}; armature {armature}; params {asdict(p)}")
    robot = ch.Robot(f"cleopatra {name}", head, feet=feet, nominal_qpos=nominal_pose(p, connection),
                     nominal_base_height=p.hip_height, nominal_hip_height=p.hip_height, notes=notes, sources=sources)
    robot.validate()
    robot.treatment = name               # plain attributes for bookkeeping (not part of chiron.Robot's fields)
    robot.connection = connection
    robot.params = p
    return robot


#: ChironLab settings for Cleopatra. Protocol §1: control at 1 kHz, log at 100 Hz, time step 1 ms — but the §11.5
#: timestep-convergence check fails at 1 ms (and 0.5 ms): the servos' kp = 40 N·m/rad on the light leg links
#: (tibia + pad ≈ 2e-4 kg·m²) is integrated explicitly, and when a torque is clipped the implicit kd term
#: under-accelerates the joint by I / (I + dt·kd) (≈ 1/5 at 1 ms). At 0.25 ms the walking speed and tilt agree
#: with 0.125 ms within 4 % and 1° (both integrators); so the physics step is 0.25 ms and the controller still
#: runs at 1 kHz. ``LAB_OPTIONS_PROTOCOL`` keeps the literal 1 ms.
LAB_OPTIONS = dict(timestep=0.00025, control_dt=0.001, log_dt=0.01)
LAB_OPTIONS_PROTOCOL = dict(timestep=0.001, control_dt=0.001, log_dt=0.01)
#: Protocol §4: every course is 1.5 m long and starts flat for 0.3 m.
COURSE_M = 1.5


def cleopatra_lab(treatment: str = "rigid", terrain=None, *, robot: ch.Robot | None = None,
                  robot_kw: dict | None = None, **lab_kw) -> ch.ChironLab:
    """A ChironLab with Cleopatra (``cleopatra(treatment, **robot_kw)`` or ``robot``) on ``terrain`` (default flat)
    with ``LAB_OPTIONS`` (overridden by ``lab_kw``, e.g. ``course_extent``, ``flat_as_plane``)."""
    opts = dict(LAB_OPTIONS)
    opts.update(lab_kw)
    return ch.ChironLab(robot if robot is not None else cleopatra(treatment, **(robot_kw or {})), terrain, **opts)


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
              f"{[j.name for j in r.joints() if j.servo is None]}")
