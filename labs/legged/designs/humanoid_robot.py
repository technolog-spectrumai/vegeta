"""Talia, the humanoid (``humanoid.py``), as a Chiron robot for export: pelvis root, a waist-yaw torso with a fixed
head, two 4-joint arms (shoulder pitch/roll, shoulder yaw, elbow) and two 6-joint legs (hip yaw/roll/pitch, knee,
ankle pitch/roll), 21 actuated joints (one waist yaw). Dimensions are ``humanoid.Humanoid``'s parameter defaults; masses are the
shapes' volumes × ``body_kit.DENSITY`` plus the listed parts. Not tuned for simulation (no balance controller).

    import humanoid_robot as hr
    robot = hr.talia()
"""
from __future__ import annotations

from vegeta.chiron import Geom, Joint, Link, PointMass

import body_kit as bk

__all__ = ["P", "PARTS_KG", "talia"]

mm = 1e-3
#: humanoid.py's parameter defaults [mm, deg] (the test checks they match)
P = {"pelvis_width": 260.0, "pelvis_depth": 150.0, "pelvis_height": 110.0, "torso_width": 320.0, "torso_depth": 180.0,
     "torso_height": 420.0, "hip_spacing": 180.0, "thigh_length": 380.0, "shin_length": 380.0, "leg_width": 80.0,
     "ankle_height": 70.0, "foot_length": 240.0, "foot_width": 110.0, "foot_thickness": 30.0,
     "upper_arm_length": 300.0, "forearm_length": 280.0, "arm_diameter": 70.0, "neck_length": 70.0,
     "head_width": 170.0, "head_depth": 190.0, "head_height": 210.0}

#: listed parts [kg] (estimates for a 1.56 m research humanoid)
PARTS_KG = {"battery 14S 10 Ah": 4.2, "computer, IMU, cameras": 1.2, "frame and wiring": 3.0}


def talia(name: str = "Talia"):
    p = P
    leg_servo, arm_servo = bk.servo("qdd 60 Nm", kp=300.0, kd=3.0), bk.servo("qdd 24 Nm", kp=60.0, kd=0.8)
    act_leg, act_arm = 0.65, 0.48
    Wp, Dp, Hp = p["pelvis_width"] * mm, p["pelvis_depth"] * mm, p["pelvis_height"] * mm
    Wt, Dt, Ht = p["torso_width"] * mm, p["torso_depth"] * mm, p["torso_height"] * mm
    # legs: hip yaw, roll, pitch at the hip (short stubs), knee, then the foot on the ankle (pitch, roll)
    legs = []
    for side, sy in (("L", 1), ("R", -1)):
        L1, L2 = p["thigh_length"] * mm, p["shin_length"] * mm
        ha = p["ankle_height"] * mm
        sole = Geom(f"{side}_foot", "box", (p["foot_length"] * mm / 2, p["foot_width"] * mm / 2,
                                           p["foot_thickness"] * mm / 2),
                    pos=(p["foot_length"] * mm * 0.2, 0.0, -ha + p["foot_thickness"] * mm / 2),
                    mass=bk.box_mass(0.12, 0.055, 0.015, bk.DENSITY["shell"]), role="foot",
                    friction=(0.9, 0.005, 0.0001), rgba=bk.BLACK)
        foot = Link(f"{side}_foot_link", pos=(0, 0, -L2),
                    joints=[Joint(f"{side}_ankle_pitch", axis=(0, 1, 0), range=(-0.9, 0.7), tag="ankle_pitch",
                                  servo=leg_servo, leg=side),
                            Joint(f"{side}_ankle_roll", axis=(1, 0, 0), range=(-0.4, 0.4), tag="ankle_roll",
                                  servo=leg_servo, leg=side)],
                    geoms=[sole], masses=[PointMass(f"{side}_ankle_actuators", 2 * act_leg, (0, 0, 0.12))])
        shin = Link(f"{side}_shin", pos=(0, 0, -L1),
                    joints=[Joint(f"{side}_knee", axis=(0, 1, 0), range=(0.0, 2.3), tag="knee", servo=leg_servo, leg=side)],
                    geoms=[bk.capsule(f"{side}_shin", 0.03, (0, 0, 0), (0, 0, -L2))], masses=[], children=[foot])
        sole.name = side
        thigh = Link(f"{side}_thigh", pos=(0, sy * p["hip_spacing"] * mm / 2, -Hp / 2),
                     joints=[Joint(f"{side}_hip_yaw", axis=(0, 0, 1), range=(-0.6, 0.6), tag="hip_yaw", servo=leg_servo, leg=side),
                             Joint(f"{side}_hip_roll", axis=(1, 0, 0), range=(-0.5, 0.5), tag="hip_roll", servo=leg_servo, leg=side),
                             Joint(f"{side}_hip_pitch", axis=(0, 1, 0), range=(-2.0, 0.8), tag="hip_pitch", servo=leg_servo, leg=side)],
                     geoms=[bk.capsule(f"{side}_thigh", 0.04, (0, 0, 0), (0, 0, -L1))],
                     masses=[PointMass(f"{side}_hip_actuators", 3 * act_leg), PointMass(f"{side}_knee_actuator", act_leg)],
                     children=[shin])
        legs.append(thigh)
    arms = []
    for side, sy in (("L", 1), ("R", -1)):
        La, Lf, r = p["upper_arm_length"] * mm, p["forearm_length"] * mm, p["arm_diameter"] * mm / 2
        fore = Link(f"{side}_forearm", pos=(0, 0, -La),
                    joints=[Joint(f"{side}_elbow", axis=(0, 1, 0), range=(-2.4, 0.0), tag="elbow", servo=arm_servo)],
                    geoms=[bk.capsule(f"{side}_forearm", r * 0.84, (0, 0, 0), (0, 0, -Lf)),
                           bk.box(f"{side}_hand", (0.03, 0.012, 0.045), (0, 0, -Lf - 0.04), role="link", rgba=bk.DARK)])
        upper = Link(f"{side}_upper_arm", pos=(0, sy * (Wt / 2 + r + 0.01), Ht - r * 1.2),
                     joints=[Joint(f"{side}_shoulder_pitch", axis=(0, 1, 0), range=(-3.0, 1.0), tag="shoulder_pitch", servo=arm_servo),
                             Joint(f"{side}_shoulder_roll", axis=(1, 0, 0), range=(-0.3, 2.5) if sy > 0 else (-2.5, 0.3),
                                   tag="shoulder_roll", servo=arm_servo),
                             Joint(f"{side}_shoulder_yaw", axis=(0, 0, 1), range=(-1.5, 1.5), tag="shoulder_yaw", servo=arm_servo)],
                     geoms=[bk.capsule(f"{side}_upper_arm", r, (0, 0, 0), (0, 0, -La))],
                     masses=[PointMass(f"{side}_shoulder_actuators", 3 * act_arm), PointMass(f"{side}_elbow_actuator", act_arm)],
                     children=[fore])
        arms.append(upper)
    Wh, Dh, Hh = p["head_width"] * mm, p["head_depth"] * mm, p["head_height"] * mm
    torso = Link("torso", pos=(0, 0, Hp / 2),
                 joints=[Joint("waist_yaw", axis=(0, 0, 1), range=(-1.2, 1.2), tag="waist_yaw", servo=leg_servo)],
                 geoms=[bk.box("torso_shell", (Dt / 2, Wt / 2, Ht / 2), (0, 0, Ht / 2)),
                        bk.capsule("neck", Wh * 0.22, (0, 0, Ht), (0, 0, Ht + p["neck_length"] * mm), role="body", rgba=bk.DARK),
                        bk.box("head", (Dh / 2, Wh / 2, Hh / 2), (0, 0, Ht + p["neck_length"] * mm + Hh / 2))],
                 masses=[PointMass("battery", PARTS_KG["battery 14S 10 Ah"], (-Dt / 4, 0, Ht * 0.45)),
                         PointMass("electronics", PARTS_KG["computer, IMU, cameras"], (0, 0, Ht + 0.12)),
                         PointMass("waist_actuator", act_leg)],
                 children=arms)
    pelvis = Link("pelvis", log=True, group="body",
                  geoms=[bk.box("pelvis_shell", (Dp / 2, Wp / 2, Hp / 2))],
                  masses=[PointMass("frame_and_wiring", PARTS_KG["frame and wiring"])], children=[torso] + legs)
    sources = {"geometry": "designs/humanoid.py (Humanoid parameter defaults)",
               "masses": "body_kit.DENSITY × shape volumes + humanoid_robot.PARTS_KG",
               "actuators": "ogma.actuators: qdd 60 Nm (legs, waist), qdd 24 Nm (arms)"}
    r = bk.robot(name, pelvis, "Talia: 1.56 m biped humanoid, 21 actuated joints (design stage).", sources)
    return r
