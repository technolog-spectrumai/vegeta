"""The monster designs' numbers, without CadQuery (as ``zoo_specs`` for the animals): each monster is a variant of
an animal or of the humanoid, or a new body plan, with its parameter defaults [mm, deg] and derived geometry.

* **Fenrir** (wolf) — Laika the dog, scaled up, with a dorsal ridge and fangs;
* **Wyvern** — a hexacopter: an egg fuselage with a neck, head and tail, six rotor arms and two manipulator arms;
* **Chiron-C** (centaur) — a horse-like quadruped (Cerva's legs) with a humanoid torso, head and two manipulator
  arms rising from the withers (the nearest machine in Vegeta is the land lab's Onager Manus: two arms on a
  wheel-leg chassis);
* **Hirudo** (giant leech) — Naga the snake: fewer, fatter, flattened segments, a sucker at each end;
* **Leshy** (forest spirit) — Cerva the deer: taller, a long beard of twigs and huge branching antlers;
* **Wraith** — a helium balloon with a gondola and octopus arms hanging from it;
* **Ghoul** — Talia the humanoid: thin limbs, long arms with a pincer on each hand, deeply bent knees, a dorsal
  ridge.
"""
from __future__ import annotations

import math

import zoo_specs as zs

__all__ = ["WOLF", "WYVERN", "CENTAUR", "LEECH", "LESHY", "WRAITH", "GHOUL", "rotor_centres", "wyvern_arm_points",
           "centaur_torso", "wraith_arm_points", "LESHY_ANTLER_SCALE"]

WOLF = dict(zs.DOG, body_length=900.0, body_width=330.0, body_height=320.0, hip_x=330.0, hip_y=120.0,
            upper_leg_length=300.0, lower_leg_length=300.0, pastern_length=110.0, leg_radius=40.0, foot_radius=36.0,
            neck_length=320.0, neck_angle_deg=25.0, neck_radius=80.0, head_length=320.0, head_width=190.0,
            head_height=170.0, snout_length=180.0, ear_size=110.0, tail_length=500.0, tail_angle_deg=-15.0,
            ridge_spikes=9.0, spike_height=90.0, fang_length=60.0)

WYVERN = {"body_length": 900.0, "body_width": 300.0, "body_height": 260.0, "cruise_height": 1500.0,
          "rotor_arm_length": 650.0, "rotor_radius": 230.0, "rotor_arm_radius": 18.0, "motor_radius": 40.0,
          "neck_length": 450.0, "neck_angle_deg": 30.0, "head_length": 300.0, "head_width": 150.0,
          "head_height": 140.0, "tail_length": 900.0, "manip_upper_length": 380.0, "manip_lower_length": 360.0,
          "manip_radius": 30.0, "claw_length": 140.0, "horn_length": 160.0}

#: the centaur: the horse half is Cerva's legs on a longer, heavier barrel; the torso rises from the withers
CENTAUR = dict(zs.DEER, body_length=1300.0, body_width=520.0, body_height=520.0, hip_x=480.0, hip_y=150.0,
               upper_leg_length=420.0, lower_leg_length=420.0, pastern_length=220.0, leg_radius=45.0,
               foot_radius=40.0, neck_length=0.0, neck_angle_deg=80.0, neck_radius=150.0, head_length=210.0,
               head_width=180.0, head_height=230.0, snout_length=0.0, ear_size=0.0, tail_length=550.0,
               tail_angle_deg=-40.0, antler_length=0.0, antler_tines=0.0,
               torso_height=520.0, torso_width=420.0, torso_depth=260.0, upper_arm_length=360.0,
               forearm_length=340.0, arm_radius=40.0)

LEECH = dict(zs.SNAKE, segments=10.0, segment_length=220.0, body_radius=110.0, tail_radius=80.0, wave_deg=18.0,
             head_length=200.0, head_width=240.0, head_height=140.0, flatten=0.55, sucker_radius=150.0,
             ring_ridges=1.0)

LESHY = dict(zs.DEER, body_length=1100.0, body_width=380.0, body_height=420.0, hip_x=400.0, hip_y=120.0,
             upper_leg_length=500.0, lower_leg_length=480.0, pastern_length=260.0, leg_radius=40.0,
             neck_length=600.0, neck_angle_deg=60.0, neck_radius=80.0, head_length=340.0, head_width=170.0,
             head_height=170.0, ear_size=120.0, tail_length=200.0, antler_length=900.0, antler_tines=5.0,
             beard_twigs=7.0, beard_length=350.0)

WRAITH = {"envelope_length": 2600.0, "envelope_diameter": 1700.0, "float_height": 2600.0, "gondola_length": 600.0,
          "gondola_width": 360.0, "gondola_height": 260.0, "suspension_length": 350.0, "arms": 6.0,
          "arm_segments": 8.0, "arm_length": 1600.0, "arm_radius": 40.0, "arm_curl_deg": 14.0, "fin_size": 500.0,
          "eye_diameter": 120.0}

GHOUL = {"pelvis_width": 230.0, "torso_width": 280.0, "torso_depth": 150.0, "torso_height": 400.0, "waist_taper": 0.6,
         "hip_spacing": 160.0, "thigh_length": 400.0, "shin_length": 420.0, "leg_width": 55.0, "leg_thickness": 45.0,
         "joint_boss_diameter": 70.0, "knee_bend_deg": 32.0, "upper_arm_length": 420.0, "forearm_length": 420.0,
         "arm_diameter": 50.0, "elbow_bend_deg": 35.0, "neck_length": 50.0, "head_width": 150.0, "head_depth": 200.0,
         "head_height": 180.0, "eye_diameter": 40.0, "battery_bay_depth": 0.0, "ridge_spikes": 6.0}


# --------------------------------------------------------------------------------------------------- wyvern
def rotor_centres(p):
    """Six rotor hubs [mm, world] on arms at 30° + k·60° around the body centre, a little above the body."""
    zc = p["cruise_height"]
    return [(p["rotor_arm_length"] * math.cos(math.radians(30 + 60 * k)),
             p["rotor_arm_length"] * math.sin(math.radians(30 + 60 * k)), zc + p["body_height"] * 0.35)
            for k in range(6)]


def wyvern_arm_points(p, sy):
    """One manipulator (a talon arm) under the body: shoulder, elbow, wrist [mm], reaching forward and down."""
    zc = p["cruise_height"]
    sh = (p["body_length"] * 0.15, sy * p["body_width"] * 0.3, zc - p["body_height"] * 0.4)
    el = (sh[0] - p["manip_upper_length"] * 0.3, sh[1] + sy * 20, sh[2] - p["manip_upper_length"] * 0.95)
    wr = (el[0] + p["manip_lower_length"] * 0.8, el[1], el[2] - p["manip_lower_length"] * 0.6)
    return sh, el, wr


# --------------------------------------------------------------------------------------------------- centaur
def centaur_torso(p):
    """Waist (on the withers), shoulder height and the two shoulder points [mm]."""
    bx, _, bz = zs.body_centre(p)
    waist = (bx + p["body_length"] * 0.42, 0.0, bz + p["body_height"] * 0.3)
    top = waist[2] + p["torso_height"]
    shoulders = [(waist[0], sy * (p["torso_width"] * 0.4 + p["arm_radius"] * 0.5), top - p["torso_height"] * 0.2)
                 for sy in (1, -1)]
    return waist, top, shoulders


# --------------------------------------------------------------------------------------------------- wraith
def wraith_arm_points(p, i):
    """Arm ``i`` hanging from the gondola's rim, curling outwards [mm]."""
    n = int(p["arms"])
    phi = 2 * math.pi * (i + 0.5) / n
    zg = p["float_height"] - p["envelope_diameter"] / 2 - p["suspension_length"] - p["gondola_height"] / 2
    r0 = p["gondola_width"] * 0.4
    pts = [(r0 * math.cos(phi) * 1.4, r0 * math.sin(phi), zg - p["gondola_height"] * 0.25)]
    seg = p["arm_length"] / p["arm_segments"]
    for k in range(int(p["arm_segments"])):
        x, y, z = pts[-1]
        out = math.radians(10 + k * p["arm_curl_deg"])          # from straight down, outwards
        pts.append((x + seg * math.sin(out) * math.cos(phi), y + seg * math.sin(out) * math.sin(phi),
                    z - seg * math.cos(out)))
    return pts
