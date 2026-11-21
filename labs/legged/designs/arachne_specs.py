"""ARACHNE — a palm-sized eight-legged wall climber: the numbers, without CadQuery or MuJoCo.

The budget comes first and fixes the dimensions (``docs/arachne.md`` §2): a target gross mass, what adhesion can
carry at that mass, what the actuators can hold, what the battery can run. Every number is either a
**datasheet / literature value** (``SOURCES``) or an **assumption** (``ASSUMPTIONS``), stated so the tethered tests
of ``docs/arachne.md`` §9 can replace it with a measurement. Units: mm, g, N, V, A, W unless named.

Frames: the body frame has x along the heading (up the wall when climbing), y to the left, z out of the wall; the
wall is the plane z = 0. Legs are numbered from the front, L1…L4 on the left (+y), R1…R4 on the right.
"""
from __future__ import annotations

import math

__all__ = ["BODY", "LEG", "LEGS", "FOOT_PAD", "FOOT_SPINE", "FOOT_HYBRID", "SURFACES", "ADHESIVE", "SPINES",
           "SERVO", "BATTERY", "ELECTRONICS", "BOM", "SOURCES", "ASSUMPTIONS", "mass_budget", "gross_mass_g",
           "PAYLOAD_G", "leg_names", "hip", "G", "servo_stall_nm", "servo_speed_rad_s", "radial", "TAIL",
           "leg_points", "tibia_drop_deg"]

G = 9.81

# ----------------------------------------------------------------------------------------------- geometry [mm]
#: the body: a two-plate sandwich (belly plate, top plate, 19 mm apart) with the 8 hip-yaw servos standing between
#: them in two rows of four, shafts up through the top plate; the battery lies on the belly plate in the 22 mm
#: corridor between the rows, the main board on the top plate. Packaging sets the size: a lift + knee servo pair is
#: ~17 mm wide across the leg and must clear its neighbour while yawing, so the hips are 26 mm apart.
BODY = {"length": 100.0, "width": 64.0, "plate": 1.2, "gap": 19.0, "belly_clearance": 6.0,
        "hip_x": (39.0, 13.0, -13.0, -39.0), "hip_y": 26.0, "battery": (50.0, 20.0, 7.0)}
BODY["height"] = BODY["gap"] + 2 * BODY["plate"]

#: one leg, three servos: hip yaw (axis z, in the body) → hip lift (axis in the wall plane, across the leg, on the
#: yaw horn) → femur → knee (axis parallel to the lift's) → tibia → passive compliant ankle → foot module. The knee
#: servo rides beside the lift servo on the yaw horn and drives the tibia through a carbon pushrod along the femur
#: (a parallelogram: the servo sets the tibia's angle), so the leg carries no servo beyond the hip. A two-servo leg
#: (yaw + lift, fixed knee) cannot keep a planted foot still while the body moves: its foot can only follow an arc
#: round the hip (docs/arachne.md §3). Splay: the leg's resting direction from the body's +y axis, positive towards
#: the front (spider-like: front legs forward, rear back).
LEG = {"coxa": 17.0, "femur": 30.0, "tibia": 30.0, "femur_rise_deg": 25.0, "lift_axis_height": 12.0,
       "knee_servo_above": 8.3,
       "splay_deg": (50.0, 17.0, -17.0, -50.0), "yaw_range_deg": (-40.0, 40.0), "lift_range_deg": (-45.0, 95.0),
       "knee_range_deg": (-60.0, 70.0), "pushrod": (0.8, 30.0),
       "ankle_tilt_free_deg": 18.0, "ankle_stop_deg": 25.0, "ankle_height": 6.0}

LEGS = [f"{s}{i}" for s in "LR" for i in (1, 2, 3, 4)]


def leg_names():
    return list(LEGS)


def hip(leg):
    """The hip-yaw axis of ``leg`` in the body frame [mm] and its splay [deg] (left legs mirrored)."""
    side, i = leg[0], int(leg[1]) - 1
    sy = 1 if side == "L" else -1
    return (BODY["hip_x"][i], sy * BODY["hip_y"], 0.0), LEG["splay_deg"][i]


def radial(leg):
    """The unit vector along ``leg`` in the wall plane (outward, splayed)."""
    (_, sy, _), splay = hip(leg)
    s = 1 if sy > 0 else -1
    a = math.radians(splay)
    return (math.sin(a), s * math.cos(a), 0.0)


def leg_points(leg):
    """The standing leg [mm, body frame, z out of the wall from the wall]: hip-yaw axis foot, hip-lift axis, knee,
    ankle top and pad centre. The femur rises ``femur_rise_deg``; the knee is a fixed bend that brings the tibia
    down to the ankle (the knee angle follows); the pad lies on the wall."""
    (hx, hy, _), _ = hip(leg)
    u = radial(leg)
    z0 = LEG["lift_axis_height"]
    lift = (hx + u[0] * LEG["coxa"], hy + u[1] * LEG["coxa"], z0)
    a = math.radians(LEG["femur_rise_deg"])
    knee = (lift[0] + u[0] * LEG["femur"] * math.cos(a), lift[1] + u[1] * LEG["femur"] * math.cos(a),
            z0 + LEG["femur"] * math.sin(a))
    dz = knee[2] - LEG["ankle_height"]
    run = math.sqrt(LEG["tibia"] ** 2 - dz ** 2)
    ankle = (knee[0] + u[0] * run, knee[1] + u[1] * run, LEG["ankle_height"])
    return {"yaw": (hx, hy, z0), "lift": lift, "knee": knee, "ankle": ankle, "pad": (ankle[0], ankle[1], 0.0)}


def tibia_drop_deg():
    """The tibia's angle below the wall-parallel plane at the standing pose."""
    p = leg_points("L1")
    dz = p["knee"][2] - p["ankle"][2]
    return math.degrees(math.asin(dz / LEG["tibia"]))


# ----------------------------------------------------------------------------------------------- feet [mm, g]
#: the adhesive foot module (glass, acrylic, varnished wood): a pad on a soft backing on a passive ankle
FOOT_PAD = {"pad": (18.0, 14.0), "backing_foam": 1.0, "backing_plate": 0.5, "adhesive_film": 0.35,
            "pivot_height": 0.6, "pivot_height_simple": 1.9, "magnet": (2.0, 1.0), "mass_g": 1.0}
#: the ankle: two crossed silicone strips whose virtual pivot lies ``pivot_height`` above the adhesive interface
#: (0.6 mm) rather than on top of the backing (a simple ball/pin ankle: 1.9 mm) — shear then puts almost no peel
#: moment on the pad edge (arachne_statics)

#: the bark kit's passive tail: a spring-steel strip behind the rear legs whose rubber tip rests on the surface below
#: the body: a compression-only support farther from the top feet than the rear feet are, so the tipping moment of the
#: off-wall centre of mass needs less pull-off at the top feet. It lifts off when the robot heads down.
TAIL = {"x": -105.0, "stiffness_n_m": 250.0, "mass_g": 2.5, "strip": (60.0, 6.0, 0.2)}
#: the microspine foot module (bark, brick): sixteen spines in two staggered rows of eight, each on its own
#: flexure, angled 50° to the surface, tips facing down the leg's load so a downward drag engages them
FOOT_SPINE = {"spines": 16, "rows": 2, "spine_angle_deg": 50.0, "tip_radius_um": 12.0, "pin_diameter": 0.2,
              "flexure_travel": 2.5, "row_pitch": 1.6, "mass_g": 1.4}
#: the hybrid module: a shorter pad with a row of six spines at its toe
FOOT_HYBRID = {"pad": (14.0, 14.0), "spines": 6, "mass_g": 1.2}

# ----------------------------------------------------------------------------------------------- adhesion
#: design allowables for the fibrillar pad at low preload (~0.3 N on 2.5 cm²) and quick release — deliberately far
#: below tape ratings: Setex DA tapes are rated 10–40 N/cm² (100–400 kPa) in shear after a firm press and dwell;
#: Stickybot's directional stalks gave ~2.6 kPa normal (1 N on 3.9 cm²). Values per surface, kPa; replace with the
#: §9 test results (T1, T2).
ADHESIVE = {"shear_kpa": {"glass": 10.0, "acrylic": 8.0, "varnished wood": 5.0, "sanded wood": 1.5, "bark": 0.0,
                          "painted wall": 2.0},
            "normal_kpa": {"glass": 3.0, "acrylic": 2.5, "varnished wood": 1.5, "sanded wood": 0.4, "bark": 0.0,
                           "painted wall": 0.6},
            "preload_n": 0.3, "drag_mm": 1.5, "peel_force_n": 0.05}

#: design allowables for the microspines: SpinybotII reports 1–2 N per spine/asperity contact (25 µm tips, hard
#: rock); bark is softer and wood fibres tear, so 0.15 N per engaged spine, 30 % of spines engaged; spines resist a
#: pull-off only through their angle (~0.25 × the shear they carry)
SPINES = {"per_spine_n": {"bark": 0.15, "sanded wood": 0.08, "brick": 0.2, "painted wall": 0.03, "glass": 0.0,
                          "acrylic": 0.0, "varnished wood": 0.01},
          "engaged_fraction": 0.3, "pulloff_over_shear": 0.25, "engage_drag_mm": 3.0}

#: which module for which surface, and why (docs/arachne.md §4)
SURFACES = {
    "glass": ("pad", "smooth and stiff: the fibres make full contact; spines find no asperity"),
    "acrylic": ("pad", "smooth, slightly softer; static charge collects dust — clean the pads"),
    "varnished wood": ("pad", "smooth film over the grain; the pad holds, with less margin than glass"),
    "sanded wood": ("hybrid", "roughness of several µm and open pores: the pad loses most of its contact, the toe "
                              "spines catch the grain"),
    "painted wall": ("hybrid", "texture defeats the pad on part of each foot; spines catch the bumps"),
    "bark": ("spine", "deep, soft, dusty roughness: no pad contact at all; the spines hook fibres and plate edges"),
    "brick": ("spine", "hard, coarse asperities: Spinybot's home ground"),
}

# ----------------------------------------------------------------------------------------------- actuators
#: a 4 g-class coreless digital micro servo (e.g. EMAX ES9051: 4.1 g, 0.8 kg·cm and 0.09 s/60° at 4.8 V, rated
#: 3.6–5.3 V by some sellers, 4.0–5.5 V by others; 19.7 × 8.3 × 23.3 mm over tabs and shaft). Run directly from 1S:
#: torque scaled with voltage. ``case``: the body without tabs (length, width, height), ``shaft_from_end``.
SERVO = {"model": "4 g-class coreless micro servo (EMAX ES9051 class)", "mass_g": 4.1, "count": 24,
         "case": (16.0, 8.3, 16.5), "tabs": (19.7, 1.2), "shaft_from_end": 5.0,
         "stall_kgcm_4v8": 0.8, "speed_s60_4v8": 0.09, "volt_ref": 4.8, "volt_run": 3.8,
         "stall_current_a": 0.35, "idle_current_a": 0.008, "continuous_fraction": 0.4}

#: a 1S LiPo, flat, in the belly against the wall side (the CoM low)
BATTERY = {"chemistry": "1S LiPo pouch, 702050 size", "capacity_mah": 600, "volt_nominal": 3.7, "mass_g": 13.5,
           "size_mm": (50.0, 20.0, 7.0), "usable_fraction": 0.8}

#: the electronics (one custom board; the parts commercially available)
ELECTRONICS = {"mcu": "ESP32-S3-MINI-1 module", "servo_driver": "2 × PCA9685 (32 PWM channels, 24 used)",
               "imu": "BMI270",
               "foot_sensors": "8 × TI TMAG5273 3-axis Hall (two I²C buses, four addresses each)",
               "tof": "2 × VL53L4CD time-of-flight (wall ahead, edge below)", "fuel_gauge": "MAX17048",
               "current_sense": "INA219 on the servo rail", "power_w": 0.35}

PAYLOAD_G = 20.0

# ----------------------------------------------------------------------------------------------- bill of materials
#: (item, qty, unit mass [g], source class, note). Source classes: 'commercial' (catalogue part), 'custom'
#: (printed / cut / moulded in-house or by a service from our files), 'specialist' (needs a supplier or a
#: clean-room process: the fibrillar adhesive).
BOM = [
    ("micro servo, 4 g class (hip yaw, hip lift, knee ×8)", 24, 4.1, "commercial", "EMAX ES9051 class; one type "
                                                                                     "everywhere, buy 6 spares"),
    ("chassis: belly + top plates, servo pockets, side walls", 1, 12.0, "custom", "SLS PA12 (service) or FDM "
                                                                                  "PETG-CF"),
    ("coxa C-bracket on the yaw horn, carrying the lift + knee servos", 8, 0.6, "custom", "SLS PA12"),
    ("femur (with pushrod guide) and tibia (with ankle socket)", 8, 0.75, "custom", "SLS PA12; FDM nylon first"),
    ("knee pushrod Ø0.8 mm carbon, 30 mm, with two ball links", 8, 0.12, "commercial", "carbon rod + micro links"),
    ("foot module — adhesive pad (18 × 14 mm)", 8, 1.0, "custom", "clip + silicone ankle + FR4 plate + foam"),
    ("fibrillar dry-adhesive film for pads", 8, 0.07, "specialist", "Setex-class tape, or cast mushroom fibres"),
    ("foot module — microspine (16 spines), set of 8 (bark kit)", 0, 1.4, "custom", "fitted instead of the pads"),
    ("passive tail (bark kit): spring-steel strip + rubber tip", 0, 2.5, "custom", "fitted for bark"),
    ("NdFeB magnet Ø2 × 1 mm (one per foot)", 8, 0.03, "commercial", "N52 disc"),
    ("TMAG5273 3-axis Hall sensor on the tibia tip", 8, 0.05, "commercial", "with a 4-wire flex lead"),
    ("main board: ESP32-S3-MINI-1, 2 × PCA9685, BMI270, MAX17048, INA219", 1, 4.5, "custom", "2-layer 0.8 mm "
                                                                                           "PCB, PCBA service"),
    ("VL53L4CD time-of-flight sensor (bare module)", 2, 0.3, "commercial", "front-up and front-down"),
    ("1S LiPo 600 mAh, 50 × 20 × 7 mm (702050)", 1, 13.5, "commercial", "with a JST-PH 2.0 lead"),
    ("wiring: servo leads shortened, flex to the feet", 1, 4.0, "custom", "silicone 32 AWG"),
    ("fasteners: M1.6 / M2 screws, threaded inserts", 1, 2.0, "commercial", ""),
    ("tether eye and belay clip (tests)", 1, 0.5, "commercial", "Dyneema line in the tests"),
]

SOURCES = {
    "Waalbot II": "Murphy, Kute, Mengüç, Sitti, IJRR 30(1):118–133, 2011: untethered 85 g, carries up to 100 g on "
                  "smooth vertical surfaces, up to 5 cm/s, fibrillar dry adhesive feet on rotating legs",
    "SpinybotII": "Kim et al., IEEE ICAR 2005; Asbeck et al., IJRR 2006: 1–2 N per spine/asperity contact, "
                  "~25 µm spine tips, many spines on compliant suspensions sharing the load",
    "Setex DA": "Setex Technologies tape data: shear 10–40 N/cm² (DA 310 … DA 910/110)",
    "Stickybot stalks": "Santos et al. 2007: ~1 N for a 3.9 cm² patch pulled along the stalks (~2.6 kPa)",
    "ES9051": "retailer data: 4.1 g, 0.8 kg·cm, 0.09 s/60° at 4.8 V, coreless, plastic gears",
}

ASSUMPTIONS = {
    "adhesive allowables": "ADHESIVE: one tenth to one hundredth of tape ratings, for a quick, low-preload contact",
    "spine allowables": "SPINES: 0.15 N per engaged spine on bark, 30 % engaged",
    "servo at 1S": "torque and speed scale with voltage (3.8 V: 79 % of the 4.8 V rating); stall current 0.35 A",
    "servo current under load": "I = idle + stall × (torque / stall torque); 40 % of stall usable continuously",
    "foot stiffness": "shear 400 N/m, normal 800 N/m per attached foot (pad, ankle, link and servo backlash)",
    "electronics": "0.35 W average (MCU at moderate duty, sensors, PWM driver)",
}


def mass_budget():
    """{item: g} from ``BOM`` (fitted items only)."""
    return {item: qty * m for item, qty, m, _, _ in BOM if qty}


def gross_mass_g(payload_g=PAYLOAD_G):
    return sum(mass_budget().values()) + payload_g


def servo_stall_nm(volt=None):
    """Stall torque at ``volt`` [N·m] (scaled from the 4.8 V rating)."""
    v = SERVO["volt_run"] if volt is None else volt
    return SERVO["stall_kgcm_4v8"] * G / 100.0 * v / SERVO["volt_ref"]


def servo_speed_rad_s(volt=None):
    v = SERVO["volt_run"] if volt is None else volt
    return (math.pi / 3) / SERVO["speed_s60_4v8"] * v / SERVO["volt_ref"]
