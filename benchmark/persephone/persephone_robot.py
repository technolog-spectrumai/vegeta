"""Persephone as a Chiron robot: the 12-segment chimney Myropod of notebook 17, with Amendment D's body joints.

Built from Cleopatra's building blocks (``notebooks/designs/myropod_robot.py``: segments, legs, head, the passive
body joints and their 'spring' / 'spring_damper' treatments) with Persephone's geometry and mass budget and a loop
over ``n_segments`` (12) instead of Cleopatra's three. NOT YET VALIDATED: written at the close of the build without
running it; check it with ``user_tests.sh persephone-smoke`` before trusting any result.

Geometry (designs/myropod.py defaults, notebook 17): segments 60 × 44 × 34 mm, shell 2.5 mm, joint gap 10 mm (pitch
70 mm), head 80 mm, femur 36 mm, tibia 42 mm, leg Ø9 mm, pad Ø12 mm, standing hip 20° and tibia 70° below
horizontal (pad 48.2 mm out, 51.8 mm down), hip pairs at ±15 mm, hips at y = ±28 mm.

Masses (notebook 17's servo-sizing result, 10.96 kg): per segment shell + hatch + rib 38.7 g (CAD 35 204 mm³ ×
1.1 g/cm³), legs 4 × 7.45 g (CAD 6 772 mm³ × 1.1 g/cm³; 2 g pad inside), 12 leg servos 'standard 60 g worm'
(actuators.py; 720 g), 2 body-joint servos 'mini 32 g' (64 g, carried as mass: the joints are passive), PCB 14 g,
wiring 12 g — 878.5 g; head shell 34.7 g (CAD 31 525 mm³) + 75 g (2 lights 12 g, camera 25 g, IMU 8 g, radio and
antenna 30 g); tail tether gland 25 g; the segment modules of notebook 17's configuration (285 g in total:
lights, sensors, battery backups, relay dropper, sampler).

Servo: 'standard 60 g worm' — stall 2.0 N·m, 30 rpm no-load, 3.0 A at 7.4 V, self-locking. The model is the same
PD + torque–speed-line servo as Cleopatra's (the worm's self-locking — holding with no current — is not modelled;
holding power is accounted as zero in notebook 17, not here). Gains kp = 15 N·m/rad, kd = 0.3 N·m·s/rad (scaled
from Cleopatra's 40 / 0.8 by the stall torque, 2 / 6 ≈ 0.33 → 13, rounded up). Armature 5e-4 kg·m² per leg joint:
the worm gear's reflected rotor inertia (a coreless micro-motor of about 1e-8 kg·m² through a ratio of order 200 —
an estimate, stated as such); without it Persephone's 4 g tibia would need a physics step of a few microseconds.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
DESIGNS = HERE.parents[1] / "notebooks" / "designs"
if str(DESIGNS) not in sys.path:
    sys.path.insert(0, str(DESIGNS))

import myropod_robot as mr  # noqa: E402
from vegeta import chiron as ch  # noqa: E402

N_SEGMENTS = 12
KP, KD, ARMATURE = 15.0, 0.3, 5e-4
LAB = dict(timestep=0.00025, control_dt=0.001, log_dt=0.01)
#: notebook 17's module configuration, one type per segment from the head back, and the module masses [kg]
CONFIG = ("lights", "sensor", "plain", "battery backup", "plain", "relay dropper", "plain", "sensor", "plain",
          "sampler", "plain", "battery backup")
MODULE_MASS = {"plain": 0.0, "battery backup": 0.045, "sensor": 0.030, "relay dropper": 0.060, "sampler": 0.055,
               "lights": 0.020}
TETHER_GLAND = 0.025

PARAMS = replace(
    mr.CleopatraParams(),
    seg_length=0.060, seg_width=0.044, seg_height=0.034, shell_thickness=0.0025, joint_gap=0.010,
    head_length=0.080, head_taper=0.85, femur=0.036, tibia=0.042, leg_diameter=0.009, foot_diameter=0.012,
    hip_angle_deg=20.0, knee_angle_deg=70.0,
    shell_mass=0.0387, leg_cad_mass=0.0298, pad_mass=0.002,
    leg_servo="standard 60 g worm", body_servo="mini 32 g", body_servos_per_segment=2,
    pcb_mass=0.014, wiring_mass=0.012, head_shell_mass=0.0347,
    head_payload=(("lights 2 x 6 g", 0.012), ("camera", 0.025), ("IMU", 0.008), ("radio and antenna", 0.030)),
    battery_mass=0.0, compute_mass=0.0, camera_x=0.020, body_servo_x=0.020, battery_z=-0.008, compute_z=0.005,
    pcb_z=0.010,
)


def persephone(body_connection: str = "spring_damper", *, n_segments: int = N_SEGMENTS, body_roll_axis=True,
               body_k_pitch=8.0, body_k_yaw=8.0, body_k_roll=8.0, body_c_pitch=0.2, body_c_yaw=0.2,
               body_c_roll=0.2, body_q0_pitch=0.0, body_q0_yaw=0.0, body_q0_roll=0.0,
               body_limit_pitch=mr.LIMIT_PITCH, body_limit_yaw=mr.LIMIT_YAW, body_limit_roll=mr.LIMIT_ROLL,
               kp: float = KP, kd: float = KD, armature: float = ARMATURE, modules: bool = True) -> ch.Robot:
    """Persephone with an Amendment D body treatment ('spring' or 'spring_damper'; identical joints, the same
    keywords and units as ``myropod_robot.cleopatra``)."""
    conn = mr.make_connection(body_connection, body_roll_axis=body_roll_axis, body_k_pitch=body_k_pitch,
                              body_k_yaw=body_k_yaw, body_k_roll=body_k_roll, body_c_pitch=body_c_pitch,
                              body_c_yaw=body_c_yaw, body_c_roll=body_c_roll, body_q0_pitch=body_q0_pitch,
                              body_q0_yaw=body_q0_yaw, body_q0_roll=body_q0_roll, body_limit_pitch=body_limit_pitch,
                              body_limit_yaw=body_limit_yaw, body_limit_roll=body_limit_roll)
    p = PARAMS
    servo = mr.cleopatra_servo(kp, kd, armature, p.leg_servo)
    n = int(n_segments)
    segs = [mr._segment(p, i, servo, "visual") for i in range(1, n + 1)]
    if modules:
        for i, seg in enumerate(segs):
            kind = CONFIG[i % len(CONFIG)]
            if MODULE_MASS[kind]:
                seg.masses.append(ch.PointMass(f"segment {i + 1} module ({kind})", MODULE_MASS[kind], (0.0, 0.0, 0.0)))
        segs[-1].masses.append(ch.PointMass("tether gland", TETHER_GLAND, (-p.seg_length / 2, 0.0, 0.0)))
    head = mr._head(p)
    segs[0].pos = (-p.head_offset, 0.0, 0.0)
    head.children.append(segs[0])
    for i in range(1, n):
        mr._connect(segs[i], i, p, conn)
        segs[i - 1].children.append(segs[i])
    feet = [ch.FootSpec(mr.leg_name(s, k), f"{mr.leg_name(s, k)} foot", mr.leg_joints(s, k), f"segment {s}")
            for s in range(1, n + 1) for k in mr.LEG_KEYS]
    q = {}
    for s in range(1, n + 1):
        for key in mr.LEG_KEYS:
            yaw, pitch, knee = mr.leg_joints(s, key)
            q[yaw], q[pitch], q[knee] = 0.0, mr.math.radians(p.hip_angle_deg), mr.math.radians(p.knee_angle_deg -
                                                                                                p.hip_angle_deg)
    for name in mr.body_joints(conn, n_segments=n, roll_axis=conn.roll_axis if hasattr(conn, "roll_axis") else True):
        q[name] = conn.axis(name.rsplit(" ", 1)[1]).q0
    robot = ch.Robot(f"persephone {conn.treatment}", head, feet=feet, nominal_qpos=q,
                     nominal_base_height=p.hip_height, nominal_hip_height=p.hip_height,
                     notes=f"Persephone ({n} segments), treatment '{conn.treatment}'; built from myropod_robot blocks; "
                           f"leg servo {p.leg_servo} kp {kp} kd {kd} armature {armature}",
                     sources={"geometry": "designs/myropod.py defaults (notebook 17)",
                              "masses": "notebook 17 servo-sizing result (10.96 kg)",
                              "servo": f"actuators.py '{p.leg_servo}'; gains and armature: this file's docstring"})
    robot.validate()
    robot.treatment = conn.treatment
    robot.connection = conn
    robot.params = p
    return robot


def persephone_lab(body_connection: str = "spring_damper", terrain=None, **lab_kw) -> ch.ChironLab:
    """A ChironLab with Persephone on ``terrain`` (default flat) with ``LAB`` (0.25 ms; not yet convergence-checked
    for Persephone — run ``user_tests.sh persephone-smoke``)."""
    return ch.ChironLab(persephone(body_connection), terrain, **{**LAB, **lab_kw})
