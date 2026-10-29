"""The zoomorphic designs' numbers, without CadQuery: each animal's parameter defaults [mm, deg] and the geometry both
its CAD design (``zoo_*.py``) and its robot model (``zoo_robots.py``) derive from them, so the two always agree.
x forward, y left, z up, ground at z = 0."""
from __future__ import annotations

import math

__all__ = ["QUADRUPED", "DOG", "BEAR", "DEER", "ELEPHANT", "SNAKE", "BAT", "OCTOPUS", "DESCRIPTIONS", "leg_points",
           "hip_height", "body_centre", "head_centre", "arm_points", "bat_wing_points", "octopus_arm_points", "slither_points"]

#: what every quadruped parameter means (the designs' Parameter descriptions)
DESCRIPTIONS = {
    "body_length": "torso, shoulder to rump", "body_width": "torso width", "body_height": "torso depth",
    "hip_x": "front and rear hip axes from the body centre", "hip_y": "leg planes from the centre line",
    "upper_leg_length": "hip/shoulder to knee/elbow", "lower_leg_length": "knee/elbow to hock/wrist",
    "pastern_length": "hock/wrist to the foot (0 = two-segment leg)", "leg_bend_deg": "segment angle from vertical",
    "leg_radius": "upper leg; the lower segments taper to 70 %", "foot_radius": "foot pad",
    "neck_length": "body front to the head centre", "neck_angle_deg": "neck above horizontal",
    "neck_radius": "", "head_length": "", "head_width": "", "head_height": "", "snout_length": "0 = none",
    "ear_size": "0 = none", "tail_length": "0 = none", "tail_angle_deg": "tail above horizontal (negative: hanging)",
}

QUADRUPED = {
    "body_length": 600.0, "body_width": 240.0, "body_height": 220.0, "hip_x": 220.0, "hip_y": 90.0,
    "upper_leg_length": 220.0, "lower_leg_length": 220.0, "pastern_length": 80.0, "leg_bend_deg": 22.0,
    "leg_radius": 30.0, "foot_radius": 28.0, "neck_length": 260.0, "neck_angle_deg": 40.0, "neck_radius": 50.0,
    "head_length": 220.0, "head_width": 130.0, "head_height": 130.0, "snout_length": 110.0, "ear_size": 70.0,
    "tail_length": 300.0, "tail_angle_deg": 25.0,
}

#: "Laika": a medium dog, digitigrade three-segment legs, pricked ears, raised tail
DOG = dict(QUADRUPED)

#: "Ursa": heavy, wide, short plantigrade legs (two segments, big paws), round small ears, stub tail
BEAR = dict(QUADRUPED, body_length=1000.0, body_width=480.0, body_height=440.0, hip_x=340.0, hip_y=170.0,
            upper_leg_length=300.0, lower_leg_length=280.0, pastern_length=0.0, leg_bend_deg=10.0, leg_radius=65.0,
            foot_radius=70.0, neck_length=300.0, neck_angle_deg=15.0, neck_radius=120.0, head_length=320.0,
            head_width=260.0, head_height=240.0, snout_length=120.0, ear_size=60.0, tail_length=80.0,
            tail_angle_deg=10.0)

#: "Cerva": a slim deer, long three-segment legs, long raised neck, antlers (``antler_*`` in the design)
DEER = dict(QUADRUPED, body_length=800.0, body_width=260.0, body_height=300.0, hip_x=300.0, hip_y=85.0,
            upper_leg_length=330.0, lower_leg_length=330.0, pastern_length=180.0, leg_bend_deg=14.0,
            leg_radius=26.0, foot_radius=22.0, neck_length=420.0, neck_angle_deg=55.0, neck_radius=55.0,
            head_length=260.0, head_width=120.0, head_height=120.0, snout_length=90.0, ear_size=90.0,
            tail_length=90.0, tail_angle_deg=30.0, antler_length=420.0, antler_tines=3.0)

#: "Gaja": an elephant, pillar legs (two near-straight segments, pad feet), trunk, tusks, big ears
ELEPHANT = dict(QUADRUPED, body_length=1600.0, body_width=900.0, body_height=900.0, hip_x=560.0, hip_y=300.0,
                upper_leg_length=520.0, lower_leg_length=480.0, pastern_length=0.0, leg_bend_deg=3.0,
                leg_radius=120.0, foot_radius=140.0, neck_length=520.0, neck_angle_deg=10.0, neck_radius=300.0,
                head_length=600.0, head_width=560.0, head_height=620.0, snout_length=0.0, ear_size=520.0,
                tail_length=500.0, tail_angle_deg=-70.0, trunk_segments=8.0, trunk_segment_length=150.0,
                tusk_length=450.0)

#: "Naga": a serpent of identical segments (two-axis joints), a wedge head
SNAKE = {"segments": 16.0, "segment_length": 140.0, "body_radius": 45.0, "tail_radius": 15.0, "wave_deg": 28.0,
         "head_length": 170.0, "head_width": 120.0, "head_height": 70.0}

#: "Noctis": a bat with folding wings: humerus, forearm, three finger bones and a membrane on each side
BAT = {"body_length": 260.0, "body_width": 110.0, "body_height": 100.0, "humerus_length": 180.0,
       "forearm_length": 260.0, "finger_length": 300.0, "finger_spread_deg": 70.0, "fold": 0.0, "bone_radius": 7.0,
       "membrane_thickness": 2.0, "head_size": 80.0, "ear_size": 60.0, "leg_length": 110.0, "stand_height": 140.0}

#: "Kraken": an octopus, an egg mantle over a head ring and eight tapered arms
OCTOPUS = {"mantle_length": 420.0, "mantle_width": 320.0, "head_radius": 150.0, "arms": 8.0, "arm_segments": 8.0,
           "arm_length": 900.0, "arm_radius": 45.0, "arm_curl_deg": 18.0, "eye_diameter": 50.0}


# ------------------------------------------------------------------------------------------- quadruped geometry
def _segments(p):
    return [L for L in (p["upper_leg_length"], p["lower_leg_length"], p["pastern_length"]) if L > 0]


def hip_height(p) -> float:
    """Hip axes above the ground, standing [mm]: the segments at ±leg_bend from vertical, plus the foot pad."""
    b = math.radians(p["leg_bend_deg"])
    return sum(L * math.cos(b) for L in _segments(p)) + p["foot_radius"]


def leg_points(p, front: bool, sy: int):
    """The leg's joint points hip → foot centre [mm, world]: front legs bend the elbow back, rear legs the knee
    forward (then the hock back), so the feet stand roughly under the hips."""
    b = math.radians(p["leg_bend_deg"])
    signs = (-1, 1, 1) if front else (1, -1, 1)
    x = (1 if front else -1) * p["hip_x"]
    z = hip_height(p)
    pts = [(x, sy * p["hip_y"], z)]
    for L, s in zip(_segments(p), signs):
        x += s * L * math.sin(b)
        z -= L * math.cos(b)
        pts.append((x, sy * p["hip_y"], z))
    return pts


def body_centre(p):
    return (0.0, 0.0, hip_height(p) + p["body_height"] * 0.2)


def head_centre(p):
    """Neck from the top front of the body at ``neck_angle_deg`` to the head centre."""
    bx, _, bz = body_centre(p)
    a = math.radians(p["neck_angle_deg"])
    n0 = (bx + p["body_length"] / 2 - p["neck_radius"] * 0.5, 0.0, bz + p["body_height"] * 0.2)
    return n0, (n0[0] + p["neck_length"] * math.cos(a), 0.0, n0[2] + p["neck_length"] * math.sin(a))


def arm_points(start, n, seg_len, heading_deg, bend_deg, plane="xz"):
    """Points of a curling chain (trunks, tails, tentacles): ``n`` segments, each turning ``bend_deg``."""
    pts, h = [tuple(start)], math.radians(heading_deg)
    for _ in range(n):
        x, y, z = pts[-1]
        if plane == "xz":
            pts.append((x + seg_len * math.cos(h), y, z + seg_len * math.sin(h)))
        else:
            pts.append((x + seg_len * math.cos(h), y + seg_len * math.sin(h), z))
        h += math.radians(bend_deg)
    return pts


# ------------------------------------------------------------------------------------------- bat and octopus
def bat_wing_points(p, sy: int):
    """Shoulder, elbow, wrist and three finger tips of one wing [mm]. ``fold`` 0 = spread, 1 = folded along the
    body (the bones swing back and the fingers close)."""
    f = p["fold"]
    zc = p["stand_height"]

    def step(a, L, deg):                                 # deg from lateral (+y outwards), positive = backwards
        t = math.radians(deg)
        return (a[0] - L * math.sin(t), a[1] + sy * L * math.cos(t), a[2])

    sh = (p["body_length"] * 0.15, sy * p["body_width"] / 2, zc + p["body_height"] * 0.25)
    elbow = step(sh, p["humerus_length"], 10 + 80 * f)
    wrist = step(elbow, p["forearm_length"], -10 - 80 * f)
    spread = p["finger_spread_deg"]
    tips = [step(wrist, p["finger_length"] * (1 - 0.15 * i), (1 - f) * (i * spread / 2 - 5) + f * (90 + 4 * i))
            for i in range(3)]
    return sh, elbow, wrist, tips


def octopus_arm_points(p, i: int):
    """Arm ``i``'s points from the head ring outwards, curling down to the ground then up at the tip."""
    n = int(p["arms"])
    phi = 2 * math.pi * (i + 0.5) / n
    r0 = p["head_radius"] * 0.8
    z0 = p["arm_radius"] * 1.0 + 40.0
    seg = p["arm_length"] / p["arm_segments"]
    pts = [(r0 * math.cos(phi), r0 * math.sin(phi), z0 + 20.0)]
    for k in range(int(p["arm_segments"])):
        x, y, z = pts[-1]
        rise = math.radians(-25 + k * p["arm_curl_deg"] * 0.6)
        pts.append((x + seg * math.cos(rise) * math.cos(phi), y + seg * math.cos(rise) * math.sin(phi),
                    max(z + seg * math.sin(rise), p["arm_radius"] * 0.5)))
    return pts


def slither_points(p):
    """The snake's joint points from the head backwards, on the ground in a sine wave [mm] (``planar_chain``'s
    rule: each segment turns ``wave_deg`` × sin(1.1 i))."""
    pts, h = [(0.0, 0.0, p["body_radius"])], math.pi
    for i in range(int(p["segments"])):
        h += math.radians(p["wave_deg"]) * math.sin(i * 1.1)
        x, y, z = pts[-1]
        pts.append((x + p["segment_length"] * math.cos(h), y + p["segment_length"] * math.sin(h), z))
    return pts
