"""Apheloria as a Chiron robot — the modular Myropod that curls into a ball (notebook 19, designs/apheloria.py).

Built from the Myropod blocks of ``myropod_robot.py`` (segment shells, legs with hip yaw / hip pitch / knee, the
head) with Apheloria's geometry, plus an armour plate on every segment and ACTIVE body joints: one pitch hinge per
joint, driven by the 'qdd 60 Nm' actuator notebook 19 sizes for curling (60 N·m stall, 200 rpm). Curling the 9 links
(head + 8 segments) by −40° at each of the 8 joints (neck included; 360°/9, belly inward) closes the ring.

Geometry (designs/apheloria.py defaults): body 150 mm long, Ø110 mm (a 150 × 110 × 110 mm shell here), joint gap
18 mm (pitch 168 mm), head 150 mm, femur 80 mm, tibia 95 mm, leg Ø18 mm, pad Ø26 mm, standing hip 30°, tibia 75°.

Masses (notebook 19's base segment, 3.838 kg; the split of the 0.927 kg of printed and machined parts is this model's
estimate, stated here): shell 0.30 kg, aluminium armour plate 0.36 kg (an ellipsoidal cap 220 mm wide on the back),
legs 4 × 66.8 g (CAD), 12 leg servos 'smart servo 12 Nm' (1.44 kg), 2 body actuators 'qdd 60 Nm' (1.30 kg; one
drives the joint ahead, the other is the yaw axis of the design, carried as mass), MCU/IMU/bus 90 g, wiring 80 g.
Head: shell 0.71 kg + stereo cameras 0.36, lidar 0.35, illuminators 0.15, computer 0.20, neck actuator 0.65 kg
(notebook 19). No payload modules (the 'standard' configuration's modules add mass the joints may not lift:
notebook 19's curl-up safety factor is 1.13 with them).
"""
from __future__ import annotations

import math
from dataclasses import replace

import myropod_robot as mr
from vegeta import chiron as ch

N_SEGMENTS = 8
CURL_DEG = -360.0 / (N_SEGMENTS + 1)          # per joint: −40° (negative pitch = the rear segment's tail goes down)
PLATE_MASS = 0.36
LEG_KP, LEG_KD = 60.0, 1.5                    # smart servo 12 Nm (scaled from Cleopatra's 40/0.8 at 6 N·m)
BODY_KP, BODY_KD = 400.0, 20.0                # qdd 60 Nm position loop
LAB = dict(timestep=0.0005, control_dt=0.001, log_dt=0.04)

PARAMS = replace(
    mr.CleopatraParams(),
    seg_length=0.150, seg_width=0.110, seg_height=0.110, shell_thickness=0.003, joint_gap=0.018, head_length=0.150,
    femur=0.080, tibia=0.095, leg_diameter=0.018, foot_diameter=0.026, hip_angle_deg=30.0, knee_angle_deg=75.0,
    shell_mass=0.300, leg_cad_mass=0.267, pad_mass=0.015, leg_servo="smart servo 12 Nm", body_servo="qdd 60 Nm",
    body_servos_per_segment=2, pcb_mass=0.090, wiring_mass=0.080, head_shell_mass=0.71,
    head_payload=(("stereo cameras", 0.36), ("lidar", 0.35), ("illuminators", 0.15), ("computer", 0.20),
                  ("neck actuator", 0.65)),
    battery_mass=0.0, compute_mass=0.0, camera_x=0.05, body_servo_x=0.05, pcb_z=0.03,
)

#: Leg poses [rad]: standing (protocol-style 30° / 75°) and tucked against the belly for the ball.
STAND = {"hip_yaw": 0.0, "hip_pitch": math.radians(30.0), "knee": math.radians(75.0 - 30.0)}
TUCK = {"hip_yaw": 0.0, "hip_pitch": math.radians(-25.0), "knee": math.radians(150.0)}


def body_joint(i: int) -> str:
    """Joint i: 0 = the neck (head–segment 1), i ≥ 1 = between segments i and i+1."""
    return "neck pitch" if i == 0 else f"body {i}-{i + 1} pitch"


def apheloria(n_segments: int = N_SEGMENTS, *, plates: bool = True) -> ch.Robot:
    """Apheloria (head + ``n_segments``) with active body pitch joints; legs as in the Myropods."""
    p = PARAMS
    leg_servo = mr.cleopatra_servo(LEG_KP, LEG_KD, 0.002, p.leg_servo)
    body_servo = ch.Servo.from_actuator(mr.act.get(p.body_servo), kp=BODY_KP, kd=BODY_KD, armature=0.05)
    segs = []
    for i in range(1, n_segments + 1):
        seg = mr._segment(p, i, leg_servo, "visual")
        if plates:
            seg.geoms.append(ch.Geom(f"segment {i} plate", "ellipsoid", (p.seg_length / 2 + 0.006, 0.110, 0.025),
                                     pos=(0.0, 0.0, p.seg_height / 2 + 0.008), mass=PLATE_MASS, role="body",
                                     friction=(0.5, 0.005, 0.0001), rgba=(0.24, 0.25, 0.27, 1.0)))
        segs.append(seg)
    head = mr._head(p)
    head.children.append(segs[0])
    half = p.pitch / 2
    for k in range(0, n_segments):            # k = 0: the neck (segment 1 behind the head), k ≥ 1: segment k+1
        child = segs[k]
        child.pos = (-p.head_offset if k == 0 else -p.pitch, 0.0, 0.0)
        pin = p.head_offset - p.head_length / 2 - p.joint_gap / 2 if k == 0 else half
        child.joints = [ch.Joint(body_joint(k), axis=(0.0, 1.0, 0.0), pos=(pin, 0.0, 0.0),
                                 range=(math.radians(-60.0), math.radians(60.0)), tag="body_pitch", servo=body_servo)]
        if k >= 1:
            segs[k - 1].children.append(child)
    feet = [ch.FootSpec(mr.leg_name(s, k), f"{mr.leg_name(s, k)} foot", mr.leg_joints(s, k), f"segment {s}")
            for s in range(1, n_segments + 1) for k in mr.LEG_KEYS]
    robot = ch.Robot("apheloria", head, feet=feet, nominal_qpos=pose(n_segments, legs=STAND, curl=0.0),
                     nominal_base_height=p.hip_height, nominal_hip_height=p.hip_height,
                     notes="Apheloria: Myropod blocks, active body pitch joints (qdd 60 Nm), armour plates",
                     sources={"geometry": "designs/apheloria.py", "masses": "notebook 19 base segment (3.838 kg)"})
    robot.validate()
    robot.params = p
    return robot


def pose(n_segments: int = N_SEGMENTS, *, legs=STAND, curl: float = 0.0) -> dict:
    """Joint angles [rad]: every leg in ``legs`` (dict hip_yaw/hip_pitch/knee), every body joint at ``curl``."""
    q = {}
    for s in range(1, n_segments + 1):
        for key in mr.LEG_KEYS:
            yaw, pitch, knee = mr.leg_joints(s, key)
            q[yaw], q[pitch], q[knee] = legs["hip_yaw"], legs["hip_pitch"], legs["knee"]
    for k in range(0, n_segments):
        q[body_joint(k)] = float(curl)
    return q
