"""ARACHNE as a Chiron robot (``vegeta.chiron``): the body, eight two-servo legs and a foot module on each, from
``arachne_specs`` (geometry, the BOM's masses, the micro servo). Units at the Chiron boundary: SI.

Frames: x along the heading (up the wall when climbing), y left, z out of the wall; the wall is the plane z = 0
(Chiron's ground). Per leg ``L1``…``R4`` (front to rear):

* ``<leg>_yaw`` — hip yaw, axis z through the hip; positive turns the leg anticlockwise seen from outside the wall;
* ``<leg>_lift`` — hip lift, axis in the wall plane across the leg (radial × z); positive lifts the foot off the
  wall;
* ``<leg>_knee`` — knee, axis parallel to the lift's; positive folds the tibia up. (The real knee servo sits on the
  yaw horn and drives the tibia by a pushrod parallelogram, so its angle is the tibia's absolute angle: knee servo =
  lift + knee here.) The ankle is passive and compliant (±18° free, a hard stop at 25° for peeling) and is not a
  joint of this model: the foot is the pad, a box on the tibia's end, plus a sphere ``<leg>`` (role 'foot') at the
  pad's centre for contact and gait targets.

    import arachne_robot as ar
    robot = ar.arachne()                 # 172 g with the 20 g payload, 24 servos
"""
from __future__ import annotations

import math

from vegeta.chiron import Geom, Joint, Link, PointMass, Servo

import arachne_specs as sp
import body_kit as bk

__all__ = ["servo", "arachne", "foot_geom", "pad_geom", "MASS_KG"]

mm = 1e-3
DARK, PAD, SHELL = (0.2, 0.2, 0.22, 1.0), (0.95, 0.55, 0.15, 1.0), (0.85, 0.86, 0.88, 1.0)


def servo(kp=0.6, kd=0.004) -> Servo:
    """The micro servo at 1S (``arachne_specs.SERVO``): stall and speed scaled from the 4.8 V rating, 40 % of stall
    continuous. kp: 0.6 N·m/rad holds a foot's 0.3 N preload at 50 mm within ~1.5°."""
    s = sp.SERVO
    stall = sp.servo_stall_nm()
    return Servo(stall, stall * s["continuous_fraction"], sp.servo_speed_rad_s(), s["stall_current_a"],
                 s["volt_run"], kp, kd, source=f"{s['model']}: 0.8 kg·cm and 0.09 s/60° at 4.8 V, scaled to "
                                                f"{s['volt_run']} V")


def foot_geom(leg):
    return leg


def pad_geom(leg):
    return f"{leg}_pad"


def _q_axis_angle(axis, angle):
    s = math.sin(angle / 2)
    return (math.cos(angle / 2), axis[0] * s, axis[1] * s, axis[2] * s)


#: masses [kg] from the BOM (each leg: the lift and knee servos and their bracket on the yaw horn, the femur and
#: tibia, the pushrod, the foot module + its magnet, the Hall sensor on the tibia; the body: the rest, the yaw servos
#: included)
def _masses(payload_g):
    b = sp.mass_budget()
    leg_link, pushrod, bracket = 0.75e-3, 0.12e-3, 0.35e-3
    foot = (sp.FOOT_PAD["mass_g"] + 0.07 + 0.03) * 1e-3
    hall = 0.05e-3
    servo_kg = sp.SERVO["mass_g"] * 1e-3
    per_leg = leg_link + pushrod + bracket + foot + hall + 2 * servo_kg
    total = sum(b.values()) * 1e-3 + payload_g * 1e-3
    return {"per_leg": per_leg, "link": leg_link, "pushrod": pushrod, "bracket": bracket, "foot": foot, "hall": hall,
            "lift_servo": servo_kg, "knee_servo": servo_kg, "body": total - 8 * per_leg, "total": total,
            "battery": sp.BATTERY["mass_g"] * 1e-3, "payload": payload_g * 1e-3}


MASS_KG = _masses(sp.PAYLOAD_G)


def arachne(payload_g=sp.PAYLOAD_G, *, kp=0.6, kd=0.004, foot_mu=0.6, name="Arachne"):
    """ARACHNE as a Chiron ``Robot`` (``payload_g`` on the back plate, above the body centre)."""
    m = _masses(payload_g)
    sv = servo(kp, kd)
    B = sp.BODY
    zc = (B["belly_clearance"] + B["height"] / 2) * mm                    # the body centre above the wall
    legs = []
    rad = math.radians
    for leg in sp.LEGS:
        P = {k: tuple(c * mm for c in v) for k, v in sp.leg_points(leg).items()}
        u = sp.radial(leg)
        axis = (u[1], -u[0], 0.0)                                         # radial × z: + lifts the foot
        rel = lambda a, b: tuple(a[k] - b[k] for k in range(3))           # noqa: E731
        pad = sp.FOOT_PAD["pad"]
        yaw_ang = math.atan2(u[1], u[0])
        tibia = Link(f"{leg}_tibia", pos=rel(P["knee"], P["lift"]),
                     joints=[Joint(f"{leg}_knee", axis=axis, range=tuple(rad(a) for a in sp.LEG["knee_range_deg"]),
                                   tag="knee", servo=sv, leg=leg)],
                     geoms=[Geom(f"{leg}_tibia", "capsule", (0.0015,), fromto=(0, 0, 0) + rel(P["ankle"], P["knee"]),
                                 mass=m["link"] * 0.45 + m["hall"], role="link", rgba=DARK),
                            Geom(pad_geom(leg), "box", (pad[0] / 2 * mm, pad[1] / 2 * mm, 0.0008),
                                 pos=rel((P["pad"][0], P["pad"][1], 0.0012), P["knee"]),
                                 quat=_q_axis_angle((0, 0, 1), yaw_ang), mass=m["foot"] * 0.6, role="visual", rgba=PAD),
                            Geom(foot_geom(leg), "sphere", (0.002,), pos=rel((P["pad"][0], P["pad"][1], 0.002), P["knee"]),
                                 mass=m["foot"] * 0.4, role="foot", friction=(foot_mu, 0.005, 0.0001), rgba=PAD)])
        femur = Link(f"{leg}_femur", pos=rel(P["lift"], P["yaw"]),
                     joints=[Joint(f"{leg}_lift", axis=axis, range=tuple(rad(a) for a in sp.LEG["lift_range_deg"]),
                                   tag="lift", servo=sv, leg=leg)],
                     geoms=[Geom(f"{leg}_femur", "capsule", (0.0018,), fromto=(0, 0, 0) + rel(P["knee"], P["lift"]),
                                 mass=m["link"] * 0.55 + m["pushrod"], role="link", rgba=DARK)],
                     children=[tibia])
        knee_dz = sp.LEG["knee_servo_above"] * mm                          # the knee servo stacked on the lift servo
        hipl = Link(f"{leg}_hip", pos=(P["yaw"][0], P["yaw"][1], P["yaw"][2] - zc),
                    joints=[Joint(f"{leg}_yaw", axis=(0, 0, 1), range=tuple(rad(a) for a in sp.LEG["yaw_range_deg"]),
                                  tag="yaw", servo=sv, leg=leg)],
                    masses=[PointMass(f"{leg}_lift_servo", m["lift_servo"], tuple(c * 0.5 for c in rel(P["lift"], P["yaw"]))),
                            PointMass(f"{leg}_knee_servo", m["knee_servo"],
                                      tuple(c * 0.5 for c in rel(P["lift"], P["yaw"])[:2]) + (knee_dz,)),
                            PointMass(f"{leg}_bracket", m["bracket"], tuple(c * 0.5 for c in rel(P["lift"], P["yaw"])))],
                    children=[femur])
        legs.append(hipl)
    L, W, H = B["length"] * mm, B["width"] * mm, B["height"] * mm
    bx, by, bz = (c * mm for c in B["battery"])
    body = Link("body", pos=(0.0, 0.0, zc), log=True, group="body",
                geoms=[Geom("chassis", "box", (L / 2, W / 2, H / 2), mass=m["body"] - m["battery"] - m["payload"],
                            role="body", friction=(0.4, 0.005, 0.0001), rgba=SHELL)],
                masses=[PointMass("battery", m["battery"], (0.0, 0.0, -H / 2 + B["plate"] * mm + bz / 2)),
                        PointMass("payload", m["payload"], (0.0, 0.0, H / 2 + 0.006))] if m["payload"] else
                       [PointMass("battery", m["battery"], (0.0, 0.0, -H / 2 + B["plate"] * mm + bz / 2))],
                children=legs)
    notes = (f"ARACHNE: palm-sized eight-legged wall climber, {m['total'] * 1000:.0f} g with a {payload_g:.0f} g "
             f"payload, 24 micro servos (hip yaw, hip lift, knee per leg), swappable foot modules.")
    sources = {"geometry": "designs/arachne_specs.py (BODY, LEG, leg_points)",
               "masses": "arachne_specs.BOM (catalogue and estimated part masses)",
               "actuators": sp.SOURCES["ES9051"], "adhesion": "arachne_specs.ADHESIVE / SPINES (design allowables)"}
    return bk.robot(name, body, notes, sources)
