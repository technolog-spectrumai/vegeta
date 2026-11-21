"""TARANTULA as a Chiron robot (``vegeta.chiron``) for the gait and stability simulations (``tarantula_gaits``,
``sims/tarantula_sim.py``, ``./tarantula.sh``): the geometry of the CAD (``spider_specs.TARANTULA``), a mass budget,
and the tendon drive of §5 as each joint's servo. Units at the Chiron boundary: SI. Frames: x forward, y left, z up,
the ground z = 0; the body frame's origin is the prosoma's centre.

Per leg ``L1``…``R4`` (front to rear), three driven joints — the motors are in the body, the joints are driven by
tendons and pushrods, so the leg links carry no motor mass:

* ``<leg>_yaw`` — the coxa about the vertical hip axis on the carapace's rim;
* ``<leg>_lift`` — the femur about the hip-lift axis (radial × z; + lifts the foot);
* ``<leg>_knee`` — the tibia about the knee (the same axis; + folds the tibia up). The metatarsus and the tarsus
  ride on the tibia (their tendon-coupled joint is held at its standing angle here) and the foot is a contact sphere
  under the pad's centre (role 'foot').

The drive (``spider_actuation.pick_motor`` for TARANTULA's joint demand): a 1000:1 micro metal gearmotor on a 1.5:1
pulley — 1.62 N·m stall, 2.2 rad/s no-load at the joint; self-locking (four-quadrant: it holds and brakes at full
torque). Masses: ``MASS_G`` (class estimates, 1.2 kg + the 0.3 kg payload on the deck).

    import tarantula_robot as tr
    robot = tr.tarantula()              # 1.5 kg, 24 joints
"""
from __future__ import annotations

import math

import body_kit as bk
import spider_actuation as sa
import spider_specs as ss
from vegeta.chiron import Geom, Joint, Link, PointMass, Servo

__all__ = ["LEGS", "MASS_G", "PAYLOAD_G", "STAND_Z", "drive", "leg_frame", "mass_kg", "tarantula"]

mm = 1e-3
S = ss.TARANTULA
LEGS = [f"{s}{i}" for s in "LR" for i in (1, 2, 3, 4)]
PAYLOAD_G = 238.5                                                        # 0.3 kg less the dragline pod
#: the body centre's height standing (the CAD pose) [m]
STAND_Z = ss.body_z(S) * mm
#: the mass budget [g] (assumed class estimates; the motors from §5's pick)
MASS_G = {
    "24 gearmotors (1000:1 micro metal) + encoders": 24 * (10.5 + 0.4),
    "battery (2S2P Li-ion 18650, 7.2 V 5 Ah)": 200.0,
    "board, IMU, motor drivers, wiring": 75.0,
    "carapace, abdomen shells, armour plates": 210.0,
    "frame, tendon pulleys and sleeves": 70.0,
    "sensor turret (camera + lidar)": 80.0,
    "cutting jaws": 54.0,
    "dragline pod (spool, winch, brake, launcher, harpoon)": 61.5,
    "8 legs (tubes, knuckles, springs, tendons, feet)": 8 * 26.0,
    "spare (fasteners, cables)": 41.4,
}
_COLOURS = {"shell": (0.42, 0.30, 0.22, 1.0), "leg": (0.18, 0.18, 0.2, 1.0), "foot": (0.95, 0.55, 0.15, 1.0)}


def mass_kg(payload_g=PAYLOAD_G):
    return (sum(MASS_G.values()) + payload_g) * 1e-3


def drive(kp=50.0, kd=0.5, derate=1.0, armature=2e-3) -> Servo:
    """The joint drive: the lightest body motor and pulley meeting TARANTULA's joint demand (§5), as a self-locking
    position servo. ``derate`` scales its torque (a weak battery, a hot motor); ``armature`` [kg·m²]: the reflected
    rotor inertia of a 1500:1 drive (assumed order of magnitude)."""
    d = sa.joint_demand("TARANTULA")
    m = sa.pick_motor(d["peak_nm"], d["speed_rad_s"])
    name, _, stall, w0, src = next(x for x in sa.MOTORS if x[0] == m["name"])
    r = m["ratio"]
    return Servo(stall * r * derate, stall * r * 0.3 * derate, w0 / r, 1.5, 6.0, kp, kd, armature=armature,
                 source=f"{name} × {r:g} pulley ({src})", four_quadrant=True)


def leg_frame(leg):
    """(hip point (x, y) [mm, body frame], radial unit vector u (x, y), side ±1)."""
    (hx, hy), splay, sy = ss.hip_points(S, leg)
    a = math.radians(splay)
    return (hx, hy), (math.sin(a), sy * math.cos(a)), sy


def tarantula(payload_g=PAYLOAD_G, *, kp=50.0, kd=0.5, derate=1.0, foot_mu=0.8, foot_r_mm=10.0, name="Tarantula",
              payload_x_mm=None):
    """TARANTULA as a Chiron ``Robot`` (``payload_g`` on the deck over the abdomen, at ``payload_x_mm`` — default the
    deck's centre)."""
    sv = drive(kp, kd, derate)
    c = ss.leg_chain(S)
    zc = ss.body_z(S)                                                     # mm; the body frame's origin height
    legs_g = MASS_G["8 legs (tubes, knuckles, springs, tendons, feet)"] / 8
    split = {"coxa": 0.12, "femur": 0.30, "tibia": 0.33, "foot": 0.25}   # of a leg's mass (assumed)
    rad = math.radians
    legs = []
    for leg in LEGS:
        (hx, hy), u, _ = leg_frame(leg)
        P = {k: (c[k][0] * u[0], c[k][0] * u[1], c[k][1] - zc) for k in ("lift", "knee", "meta", "tarsus", "toe")}
        mid = (c["tarsus"][0] + c["toe"][0]) / 2
        P["foot"] = (mid * u[0], mid * u[1], foot_r_mm - zc)
        rel = lambda a, b: tuple((a[k] - b[k]) * mm for k in range(3))
        axis = (u[1], -u[0], 0.0)
        tube = S["tube"]
        tibia = Link(f"{leg}_tibia", pos=rel(P["knee"], P["lift"]),
                     joints=[Joint(f"{leg}_knee", axis=axis, range=(rad(-75), rad(75)), tag="knee", servo=sv, leg=leg)],
                     geoms=[Geom(f"{leg}_tibia", "capsule", (tube[1] * mm,), fromto=(0, 0, 0) + rel(P["meta"], P["knee"]),
                                 mass=legs_g * split["tibia"] * 0.6e-3, role="link", rgba=_COLOURS["leg"]),
                            Geom(f"{leg}_metatarsus", "capsule", (tube[2] * mm,),
                                 fromto=rel(P["meta"], P["knee"]) + rel(P["tarsus"], P["knee"]),
                                 mass=legs_g * split["tibia"] * 0.4e-3, role="link", rgba=_COLOURS["leg"]),
                            Geom(leg, "sphere", (foot_r_mm * mm,), pos=rel(P["foot"], P["knee"]),
                                 mass=legs_g * split["foot"] * 1e-3, role="foot", friction=(foot_mu, 0.005, 0.0001),
                                 rgba=_COLOURS["foot"])])
        femur = Link(f"{leg}_femur", pos=rel(P["lift"], (0, 0, 0)),
                     joints=[Joint(f"{leg}_lift", axis=axis, range=(rad(-55), rad(65)), tag="lift", servo=sv, leg=leg)],
                     geoms=[Geom(f"{leg}_femur", "capsule", (tube[0] * mm,), fromto=(0, 0, 0) + rel(P["knee"], P["lift"]),
                                 mass=legs_g * split["femur"] * 1e-3, role="link", rgba=_COLOURS["leg"])],
                     children=[tibia])
        hip = Link(f"{leg}_hip", pos=(hx * mm, hy * mm, (c["z_lift"] - zc) * mm),
                   joints=[Joint(f"{leg}_yaw", axis=(0, 0, 1), range=(rad(-40), rad(40)), tag="yaw", servo=sv, leg=leg)],
                   geoms=[Geom(f"{leg}_coxa", "capsule", (tube[0] * 1.2 * mm,), fromto=(0, 0, 0) + rel(P["lift"], (0, 0, 0)),
                               mass=legs_g * split["coxa"] * 1e-3, role="link", rgba=_COLOURS["leg"])],
                   children=[femur])
        legs.append(hip)
    L, W, H = (v * mm for v in S["prosoma"])
    L2, W2, H2 = (v * mm for v in S["opisthosoma"])
    xo = -(S["prosoma"][0] / 2 + S["waist"] + S["opisthosoma"][0] / 2) * 0.94 * mm
    zo = (S["belly"] + S["opisthosoma"][2] / 2 - zc) * mm
    shells = MASS_G["carapace, abdomen shells, armour plates"] * 1e-3
    px = xo if payload_x_mm is None else payload_x_mm * mm
    masses = [PointMass("motors", MASS_G["24 gearmotors (1000:1 micro metal) + encoders"] * 1e-3, (0.0, 0.0, 0.0)),
              PointMass("battery", MASS_G["battery (2S2P Li-ion 18650, 7.2 V 5 Ah)"] * 1e-3, (xo, 0.0, zo)),
              PointMass("board", MASS_G["board, IMU, motor drivers, wiring"] * 1e-3, (0.0, 0.0, 0.014)),
              PointMass("frame", MASS_G["frame, tendon pulleys and sleeves"] * 1e-3, (0.0, 0.0, 0.0)),
              PointMass("turret", MASS_G["sensor turret (camera + lidar)"] * 1e-3, (0.046, 0.0, H / 2 + 0.02)),
              PointMass("jaws", MASS_G["cutting jaws"] * 1e-3, (0.095, 0.0, -0.015)),
              PointMass("dragline pod", MASS_G["dragline pod (spool, winch, brake, launcher, harpoon)"] * 1e-3,
                        (xo - 0.058, 0.0, zo + 0.028)),
              PointMass("spare", MASS_G["spare (fasteners, cables)"] * 1e-3, (-0.05, 0.0, 0.0))]
    if payload_g:
        masses.append(PointMass("payload", payload_g * 1e-3, (px, 0.0, zo + H2 / 2 + 0.01)))
    body = Link("body", pos=(0.0, 0.0, zc * mm), log=True, group="body",
                geoms=[Geom("prosoma", "box", (L / 2 * 0.85, W / 2 * 0.85, H / 2), mass=shells * 0.55, role="body",
                            friction=(0.5, 0.005, 0.0001), rgba=_COLOURS["shell"]),
                       Geom("opisthosoma", "box", (L2 / 2 * 0.8, W2 / 2 * 0.8, H2 / 2), pos=(xo, 0.0, zo),
                            mass=shells * 0.45, role="body", friction=(0.5, 0.005, 0.0001), rgba=_COLOURS["shell"])],
                masses=masses, children=legs)
    notes = (f"TARANTULA: 1.5 kg-class eight-legged walker and climber, {mass_kg(payload_g):.2f} kg with a "
             f"{payload_g:.0f} g payload; 24 tendon-driven joints from body gearmotors (docs/spiders.md §5, §9).")
    sources = {"geometry": "designs/spider_specs.py (TARANTULA, leg_chain, hip_points)",
               "masses": "tarantula_robot.MASS_G (class estimates)", "drive": sv.source}
    return bk.robot(name, body, notes, sources)
