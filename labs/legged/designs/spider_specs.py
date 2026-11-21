"""The second-generation spiders' numbers, without CadQuery: ARACHNE-2 (palm-sized wall climber), TARANTULA (a bigger
walker and climber) and SALTY (a hydraulic jumping spider) — docs/spiders.md §3, §5, §6. mm and degrees.

**The leg** (all three; the spider's segments, simplified): the **coxa** turns about the hip-yaw axis; the **femur**
rises from the hip-lift joint to a high **knee** (the femur–patella joint); the **tibia** (patella and tibia as one)
falls to the **metatarsus** joint; the **metatarsus** falls to the **tarsus**, the foot. Driven: hip yaw, hip lift,
knee — every motor in the body (``motor_bank``), the lift and knee through pushrods and tendons in sleeves along the
femur, a torsion spring at the knee extending it (the spider's hemolymph pressure, simplified). The metatarsus–tarsus
joint is coupled to the knee by a tendon with a spring-loaded stop, so the foot turns onto a new surface. Angles:
``femur_rise_deg`` above the ground plane, the others below it.
"""
from __future__ import annotations

import math

__all__ = ["ARACHNE2", "SALTY", "SPIDERS", "TARANTULA", "body_z", "cutter_gap", "cutter_linkage", "hip_points",
           "leg_chain", "rim_y"]

#: ARACHNE-2 — the palm-sized wall climber, spider-like legs (docs/spiders.md §3)
ARACHNE2 = {
    "name": "ARACHNE-2", "mass_g": 180.0,
    "prosoma": (70.0, 56.0, 20.0), "opisthosoma": (56.0, 44.0, 16.0), "waist": 6.0, "belly": 7.0,
    "hip_x": (24.0, 9.0, -6.0, -21.0), "splay_deg": (55.0, 20.0, -15.0, -45.0),
    "coxa": 10.0, "femur": 34.0, "tibia": 36.0, "metatarsus": 24.0, "tarsus": 10.0,
    "femur_rise_deg": 50.0, "tibia_drop_deg": 35.0, "metatarsus_drop_deg": 50.0, "tarsus_drop_deg": 8.0,
    "tube": (2.4, 1.6, 1.3, 1.0),                                         # segment tube radii: femur, tibia, meta, tarsus
    "pad": (14.0, 11.0, 2.6), "claw": 5.0, "motor": (6.0, 20.0), "motors": 24,
    "battery": (40.0, 20.0, 6.0), "board": (40.0, 30.0), "bank": (6, 2, 2),
}

#: TARANTULA — the bigger walker and climber (1.5 kg, reach ~250 mm; tendons and body gearmotors: §5)
TARANTULA = {
    "name": "TARANTULA", "mass_g": 1500.0,
    "prosoma": (165.0, 140.0, 48.0), "opisthosoma": (150.0, 125.0, 75.0), "waist": 16.0, "belly": 26.0,
    "hip_x": (64.0, 22.0, -20.0, -62.0), "splay_deg": (55.0, 20.0, -15.0, -45.0),
    "coxa": 30.0, "femur": 95.0, "tibia": 105.0, "metatarsus": 70.0, "tarsus": 30.0,
    "femur_rise_deg": 45.0, "tibia_drop_deg": 35.0, "metatarsus_drop_deg": 55.0, "tarsus_drop_deg": 10.0,
    "tube": (9.0, 8.0, 6.5, 5.0),
    "pad": (40.0, 30.0, 6.0), "claw": 14.0, "motor": (12.0, 10.0, 26.0), "motors": 24,
    #: the foot of §10: a backing plate, a rubber sole under the heel (swappable: the fibrillar pad, the magnet), and
    #: a microspine comb at the toe — ``rows`` × ``per_row`` spines at ``pitch`` [mm], each on its own finger flexure,
    #: the rows ``dx`` apart from ``x0`` (from the foot's centre), tilted ``tilt_deg`` so the tips trail to the heel
    #: (they hook when the foot is drawn towards the body: the climbing stroke, and the drag that sets them)
    "spine_comb": {"rows": 5, "per_row": 32, "pitch": 0.9, "tilt_deg": 30.0, "x0": 6.0, "dx": 2.5, "reach": 0.3},
    #: the dragline pod at the spinnerets (§12): a spool of UHMWPE line (``drum`` core radius, ``flange`` radius,
    #: ``width`` [mm]) driven by a gearmotor through a clutch, a band brake for controlled descent, and a spring
    #: launcher firing a barbed harpoon backwards and up (``elevation_deg``) along the line's exit; the pod's centre
    #: ``at`` (dx, dz) from the abdomen's centre
    "cords": "routed",                                                    # tendons over the knuckles (spiders.tendon_routes)
    "winch": {"drum": 10.0, "flange": 17.0, "width": 18.0, "at": (-58.0, 28.0), "line_mm": 0.8, "line_m": 10.0,
              "tube": (12.0, 8.0, 80.0), "elevation_deg": 25.0, "harpoon": (5.0, 70.0, 14.0), "spring_mm": 50.0},
    "battery": (110.0, 45.0, 32.0), "board": (110.0, 80.0), "bank": (6, 2, 2),
    "turret": (46.0, 34.0), "shell_plates": 5,
    #: the chelicerae: two bypass-shear jaws under the turret (the left jaw above the pivot plane, the right below; their
    #: straight cutting edges, ``edge_y`` inside the pivot, cross the midline and slide past each other), each on a
    #: vertical pin in front of a hollow, open-fronted cheliceral housing; a leadscrew pushes a crossbar yoke forward
    #: and two links (from the yoke's ends at ±``yoke_y``) push the jaws' outward tails forward — the jaws close
    #: (``spider_actuation.cutter`` sizes it). Jaw coordinates: the left jaw's, from its pivot, x forward, y to the left;
    #: the right jaw mirrors it.
    "cutter": {"pivot": (101.0, 9.0, 44.0), "jaw_t": 7.0, "edge_y": 10.5, "tail": (-4.0, 13.0), "link": 16.0,
               "yoke_y": 19.0, "open_deg": 30.0, "cut_zone": (6.0, 30.0), "housing_at": (70.0, 96.0, 34.0, 66.0),
               "motor_at": (55.0, 44.0), "lead_mm": 0.7},
}

#: SALTY — the jumping spider: legs III jump (hydraulic extensor cylinders, a gas accumulator, a latch), legs I, II, IV
#: walk; four front eyes are cameras; a dragline tether reel at the spinnerets (§6)
SALTY = {
    "name": "SALTY", "mass_g": 250.0,
    "prosoma": (62.0, 56.0, 30.0), "opisthosoma": (56.0, 46.0, 32.0), "waist": 5.0, "belly": 9.0,
    "plan": "box",                                                        # the boxy salticid carapace, a flat face
    "hip_x": (16.0, 4.0, -8.0, -21.5), "splay_deg": (60.0, 25.0, -20.0, -50.0),
    "coxa": 7.0, "femur": 24.0, "tibia": 22.0, "metatarsus": 17.0, "tarsus": 8.0,
    "femur_rise_deg": 40.0, "tibia_drop_deg": 55.0, "metatarsus_drop_deg": 65.0, "tarsus_drop_deg": 10.0,
    "tube": (2.4, 1.8, 1.4, 1.1),
    "jump_leg": 1.45,                                                     # legs III are this much longer, crouched:
    "jump_angles": (65.0, 55.78, 70.0, 10.0),                             # femur rise, tibia/meta/tarsus drop (same hip)
    "pad": (11.0, 9.0, 2.4), "claw": 4.0, "motor": (6.0, 20.0), "motors": 20,
    "eyes": ((15.0, 8.5), (7.0, 40.0)),                                   # principal (Ø, y), anterior lateral (Ø, look-out °)
    "cylinder": (10.0, 26.0), "accumulator": (14.0, 34.0), "pump": (12.0, 10.0, 14.0), "reel": (16.0, 8.0),
    "battery": (24.0, 16.0, 8.0), "board": (36.0, 30.0), "bank": (5, 2, 2), "bank_at": (-4.0, -3.0),
}

SPIDERS = {"arachne2": ARACHNE2, "tarantula": TARANTULA, "salty": SALTY}


def body_z(spec):
    """The prosoma's mid-height: the hip ring (the hip-lift axes) unless that would sink the belly below ``belly``."""
    H = spec["prosoma"][2]
    return max(leg_chain(spec)["z_lift"], spec["belly"] + H / 2)


def rim_y(spec, x):
    """Half-width of the prosoma's plan at ``x``: an ellipse, or (``plan`` 'box') a rectangle with rounded corners."""
    L, W, _ = spec["prosoma"]
    if spec.get("plan") == "box":
        rc = 0.2 * W
        dx = abs(x) - (L / 2 - rc)
        return W / 2 if dx <= 0 else W / 2 - rc + math.sqrt(max(rc * rc - dx * dx, 0.0))
    return W / 2 * math.sqrt(max(1.0 - (x / (L / 2)) ** 2, 0.0))


def hip_points(spec, leg):
    """(hip-yaw axis (x, y), splay [deg], side ±1) of ``leg`` ('L1'…'R4'): just outside the prosoma's rim at its hip x
    (the coxa's yaw knuckle stands on a boss on the rim, leaving the inside to the motor bank)."""
    sy = 1 if leg[0] == "L" else -1
    i = int(leg[1]) - 1
    hx = spec["hip_x"][i]
    return (hx, sy * (rim_y(spec, hx) + 1.3 * spec["tube"][0])), spec["splay_deg"][i], sy


def cutter_linkage(spec, open_deg):
    """TARANTULA's jaw linkage at an opening ``open_deg`` (each jaw turned that far outward from closed): the left
    jaw's tail point (x, y) and the yoke's x (the links' back ends; the right jaw mirrors)."""
    c = spec["cutter"]
    px, py, _ = c["pivot"]
    tx, ty = c["tail"]
    a = math.radians(open_deg)
    T = (px + tx * math.cos(a) - ty * math.sin(a), py + tx * math.sin(a) + ty * math.cos(a))
    return {"tail": T, "yoke_x": T[0] - math.sqrt(c["link"] ** 2 - (T[1] - c["yoke_y"]) ** 2)}


def cutter_gap(spec, open_deg, a):
    """The gap between the two cutting edges ``a`` mm in front of the pivots at an opening ``open_deg`` (negative: the
    edges have crossed — the bypass blades overlap by that much)."""
    c = spec["cutter"]
    t = math.radians(open_deg)
    return 2 * (c["pivot"][1] + a * math.sin(t) - c["edge_y"] * math.cos(t))


def leg_chain(spec, scale=1.0, angles=None):
    """The leg's joints in its own frame (r outward along the leg, z up, the ground at z = 0) — hip-lift axis, knee,
    metatarsus joint, tarsus joint, toe — and the lift-axis height that puts the foot pad on the ground.
    ``scale`` lengthens the femur, tibia and metatarsus and ``angles`` (femur rise, tibia, metatarsus, tarsus drop
    [deg]) replaces the spec's — SALTY's jumping legs, latched in their crouch (``jump_angles``: chosen so the hip sits
    at the walking legs' height)."""
    angles = angles or (spec["femur_rise_deg"], spec["tibia_drop_deg"], spec["metatarsus_drop_deg"],
                        spec["tarsus_drop_deg"])
    a1, a2, a3, a4 = (math.radians(a) for a in angles)
    f, t, m, ta = spec["femur"] * scale, spec["tibia"] * scale, spec["metatarsus"] * scale, spec["tarsus"]
    drop = f * math.sin(a1) - t * math.sin(a2) - m * math.sin(a3) - ta * math.sin(a4)
    z_lift = spec["pad"][2] + 1.0 - drop
    r = spec["coxa"]
    lift = (r, z_lift)
    knee = (r + f * math.cos(a1), z_lift + f * math.sin(a1))
    meta = (knee[0] + t * math.cos(a2), knee[1] - t * math.sin(a2))
    tars = (meta[0] + m * math.cos(a3), meta[1] - m * math.sin(a3))
    toe = (tars[0] + ta * math.cos(a4), tars[1] - ta * math.sin(a4))
    return {"lift": lift, "knee": knee, "meta": meta, "tarsus": tars, "toe": toe, "z_lift": z_lift,
            "reach": toe[0], "knee_height": knee[1]}
