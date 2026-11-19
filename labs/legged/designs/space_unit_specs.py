"""The space game units' numbers, without CadQuery: bulky armoured humanoid robots on Talia's frame
(``humanoid.Humanoid`` variants [mm, deg]) — the **Marine** (the line unit: a carbine and a combat shield), the
**Marine with a chainsaw** (the carbine and a chainsaw), the **Breacher** (a tower shield and a power maul) and the
**Centurion** (the carbine, the imperial banner, a feathered crest) and the **Heavy** (a rotary cannon fed from its
backpack); every unit has both hands busy — and the materials of their
armour and weapons (Yax's materials shape)."""
from __future__ import annotations

__all__ = ["MARINE", "MARINE_EXTRA", "MARINE_SAW", "MARINE_SAW_EXTRA", "CENTURION", "CENTURION_EXTRA", "BREACHER", "BREACHER_EXTRA", "HEAVY", "HEAVY_EXTRA", "MATERIALS", "ROSTER"]

#: the Marine: Humanoid parameters on a gorilla-human mix (a gorilla's long, wide barrel ribcage down to the pelvis,
#: long thick arms with the hands at the knees, short legs and broad flaring iliac wings - on a human's S-curved spine,
#: for standing upright; a square helmet, the forearms held forward 60 deg at the carbine), the skeleton's and the
#: armour's extras in ``MARINE_EXTRA``
MARINE = {"pelvis_width": 560.0, "pelvis_depth": 270.0, "pelvis_height": 170.0, "torso_width": 660.0,
          "torso_depth": 400.0, "torso_height": 600.0, "waist_taper": 0.82, "shell_fillet": 26.0, "hip_spacing": 360.0,
          "thigh_length": 360.0, "shin_length": 340.0, "leg_width": 135.0, "leg_thickness": 115.0,
          "joint_boss_diameter": 150.0, "ankle_height": 95.0, "foot_length": 340.0, "foot_width": 180.0,
          "foot_thickness": 48.0, "knee_bend_deg": 15.0, "upper_arm_length": 540.0, "forearm_length": 500.0,
          "arm_diameter": 140.0, "elbow_bend_deg": 60.0, "neck_length": 40.0, "head_width": 230.0, "head_depth": 240.0,
          "head_height": 230.0, "eye_diameter": 18.0, "battery_bay_depth": 0.0,
          # the skeleton: an S-curved spine of vertebrae set back in the chest, ribs to a sternum, clavicles and
          # scapulae, iliac wings on a sacrum, femurs, tibia and fibula, every joint a mechanical rotary module
          "spine_segments": 12, "spine_curve": 28.0, "spine_bow": 0.5, "spine_set_back": 0.3, "vertebra_diameter": 120.0,
          "rib_count": 10, "rib_section": 24.0, "sternum_width": 70.0, "clavicle_diameter": 30.0, "scapula_width": 200.0,
          "iliac_wing_height": 170.0, "femur_diameter": 66.0, "tibia_diameter": 58.0, "fibula_diameter": 26.0,
          "joint_module_length": 150.0, "joint_module_diameter": 150.0, "finger_length": 190.0, "finger_diameter": 28.0,
          # the armour
          "pauldron_diameter": 300.0, "pauldron_thickness": 45.0, "chest_plate_thickness": 35.0, "lorica_bands": 6, "helmet_thickness": 12.0,
          "backpack_width": 400.0, "backpack_depth": 170.0, "backpack_height": 430.0, "exhaust_diameter": 70.0,
          "exhaust_height": 160.0, "visor_height": 45.0, "greave_thickness": 30.0, "belt_height": 70.0,
          # the carbine (the right fist) and the combat shield (the left forearm)
          "carbine_length": 640.0, "receiver_height": 150.0, "receiver_width": 70.0, "barrel_diameter": 26.0,
          "magazine_height": 170.0,
          "shield_width": 600.0, "shield_height": 1600.0, "shield_thickness": 24.0, "shield_boss_diameter": 160.0}
MARINE_EXTRA = {"spine_segments": "vertebrae, sacrum to the neck", "spine_curve": "the spine's curve: the S's amplitude (a human's) or the bow's depth (a gorilla's)",
                "spine_bow": "the spine between a human's S (0) and a gorilla's single backward bow (1); 0.5: the hybrid, a shallow lordosis under a long kyphosis",
                "spine_set_back": "the spine's place in the chest, a fraction of the torso depth behind its middle",
                "vertebra_diameter": "the lumbar vertebral bodies (the thoracic thinner)",
                "rib_count": "rib pairs on the upper vertebrae", "rib_section": "a rib's diameter", "sternum_width": "",
                "clavicle_diameter": "", "scapula_width": "the shoulder blades behind the ribs",
                "iliac_wing_height": "the pelvis's wings above the hip modules", "femur_diameter": "the thigh bone's shaft",
                "tibia_diameter": "the shin bone", "fibula_diameter": "the thin bone beside it",
                "joint_module_length": "a rotary joint module along its axis (hips, knees, ankles)", "joint_module_diameter": "",
                "finger_length": "the gripping hand's fingers (four, three phalanges each)", "finger_diameter": "",
                "lorica_bands": "the lorica segmentata's girth hoops round the ribs", "helmet_thickness": "the galea's plates: the brow band, cheek and nape guards, the comb",
                "pauldron_diameter": "the shoulder guards' span", "pauldron_thickness": "their plates",
                "chest_plate_thickness": "the lorica's bands", "backpack_width": "the power pack",
                "backpack_depth": "", "backpack_height": "", "exhaust_diameter": "the two exhaust stacks",
                "exhaust_height": "above the pack", "visor_height": "the helmet's visor slit",
                "greave_thickness": "the shin plates", "belt_height": "the waist belt",
                "carbine_length": "muzzle to butt", "receiver_height": "", "receiver_width": "",
                "barrel_diameter": "", "magazine_height": "below the receiver",
                "shield_width": "the body shield, covering the Marine from the shins to the helmet", "shield_height": "",
                "shield_thickness": "", "shield_boss_diameter": "the shield's boss"}

#: the Marine with a chainsaw in the left fist instead of the shield (the carbine stays)
MARINE_SAW = {k: v for k, v in MARINE.items() if not k.startswith("shield_")} | \
    {"saw_bar_length": 620.0, "saw_bar_width": 110.0, "saw_bar_thickness": 16.0, "saw_body_length": 240.0,
     "saw_body_height": 160.0, "saw_teeth": 16}
MARINE_SAW_EXTRA = {k: v for k, v in MARINE_EXTRA.items() if not k.startswith("shield_")} | \
    {"saw_bar_length": "the chainsaw's bar, body to tip", "saw_bar_width": "", "saw_bar_thickness": "",
     "saw_body_length": "the motor body at the fist", "saw_body_height": "", "saw_teeth": "teeth along the bar's edge"}

#: the Centurion: the Marine's frame and carbine, the imperial banner in the left fist (a pole with a skull in a gear as
#: the finial and a large flat phased-array antenna under it: the squad's uplink) and a feathered crest along the helmet
CENTURION = {k: v for k, v in MARINE.items() if not k.startswith("shield_")} | \
    {"pole_length": 2300.0, "pole_diameter": 36.0, "antenna_width": 520.0, "antenna_height": 340.0, "antenna_thickness": 34.0,
     "antenna_tilt_deg": 25.0, "gear_diameter": 240.0, "gear_teeth": 12, "gear_thickness": 30.0, "skull_diameter": 120.0,
     "crest_length": 340.0, "crest_height": 170.0, "crest_width": 60.0, "crest_feathers": 9}
CENTURION_EXTRA = {k: v for k, v in MARINE_EXTRA.items() if not k.startswith("shield_")} | \
    {"pole_length": "the banner pole, butt to finial", "pole_diameter": "",
     "antenna_width": "the phased-array panel on the pole (a Starlink-class flat antenna)", "antenna_height": "", "antenna_thickness": "",
     "antenna_tilt_deg": "the panel's face tipped back from vertical",
     "gear_diameter": "the finial: a gear with a skull in it", "gear_teeth": "", "gear_thickness": "", "skull_diameter": "",
     "crest_length": "the feathered crest along the helmet, front to back", "crest_height": "above the helmet", "crest_width": "",
     "crest_feathers": "feathers in the row"}

#: the Breacher: the Marine's frame with a tower shield on the left forearm and a power maul in the right fist
BREACHER = dict(MARINE, elbow_bend_deg=20.0,
                shield_width=680.0, shield_height=1750.0, shield_thickness=30.0, shield_boss_diameter=200.0,    # the tower shield: the whole body
                maul_length=720.0, maul_head_length=200.0, maul_head_diameter=130.0, maul_shaft_diameter=40.0)
BREACHER_EXTRA = dict(MARINE_EXTRA, shield_width="the tower shield", shield_height="", shield_thickness="",
                      shield_boss_diameter="the shield's boss", maul_length="the power maul, butt to head",
                      maul_head_length="", maul_head_diameter="", maul_shaft_diameter="")
for _k in ("carbine_length", "receiver_height", "receiver_width", "barrel_diameter", "magazine_height"):
    BREACHER.pop(_k)
    BREACHER_EXTRA.pop(_k)

#: the Heavy: the Marine's frame, wider, a rotary cannon on the right forearm fed from a bigger backpack
#: The upright humanoid: the same skeleton, spine and hands at a human's proportions (1.95 m), long legs and short
#: arms; lighter than Belisario and meant to walk, not to knuckle-walk - the same tasks from standing.
_HUMANOID_207 = dict(MARINE, pelvis_width=420.0, pelvis_depth=230.0, pelvis_height=150.0, torso_width=520.0, torso_depth=300.0, torso_height=560.0,
                     hip_spacing=260.0, thigh_length=470.0, shin_length=440.0, leg_width=110.0, leg_thickness=95.0, upper_arm_length=330.0,
                     forearm_length=300.0, arm_diameter=105.0, elbow_bend_deg=20.0, finger_length=150.0, finger_diameter=24.0,
                     femur_diameter=56.0, tibia_diameter=50.0, fibula_diameter=22.0, foot_length=300.0, foot_width=120.0, rib_count=10,
                     iliac_wing_height=150.0, scapula_width=170.0, vertebra_diameter=100.0)   # a human's proportions at 2.07 m
HUMANOID_SCALE = 0.87                                                      #: the humanoid: a tall human, 1.80 m (every length of the 2.07 m proportions by this)
HUMANOID = {k: (v * HUMANOID_SCALE if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in ("waist_taper", "rib_count") and not k.endswith("_deg") else v)
            for k, v in _HUMANOID_207.items()}

HEAVY = {k: v for k, v in MARINE.items() if not k.startswith("shield_")}
HEAVY = dict(HEAVY, torso_width=720.0, torso_depth=430.0, hip_spacing=400.0, pelvis_width=600.0,
             backpack_width=480.0, backpack_depth=240.0, backpack_height=480.0, elbow_bend_deg=75.0,
             cannon_length=820.0, cannon_diameter=130.0, cannon_barrels=6, cannon_barrel_diameter=22.0,
             feed_diameter=60.0)
HEAVY_EXTRA = dict({k: v for k, v in MARINE_EXTRA.items() if not k.startswith("shield_")}, cannon_length="the rotary cannon, breech to muzzle", cannon_diameter="the barrel cluster",
                   cannon_barrels="", cannon_barrel_diameter="", feed_diameter="the ammunition feed from the pack")
for _k in ("carbine_length", "receiver_height", "receiver_width", "barrel_diameter", "magazine_height"):
    HEAVY.pop(_k)
    HEAVY_EXTRA.pop(_k)

#: name -> (specs, extras, what it carries)
ROSTER = {"marine": (MARINE, MARINE_EXTRA, "a carbine in the right fist, a combat shield on the left forearm"),
          "marine_saw": (MARINE_SAW, MARINE_SAW_EXTRA, "a carbine in the right fist, a chainsaw in the left"),
          "centurion": (CENTURION, CENTURION_EXTRA, "a carbine in the right fist, the imperial banner in the left, a feathered crest"),
          "breacher": (BREACHER, BREACHER_EXTRA, "a tower shield on the left forearm, a power maul in the right fist"),
          "heavy": (HEAVY, HEAVY_EXTRA, "a rotary cannon on the right forearm, fed from the backpack")}

MATERIALS = {
    "Ti-6Al-4V": {"density_kg_m3": 4430.0, "E_pa": 114e9, "nu": 0.34, "yield_pa": 880e6, "ultimate_pa": 950e6,
                  "source": "titanium alloy, annealed (ASM handbook): the armour plates, pauldrons and shield"},
    "4140 steel": {"density_kg_m3": 7850.0, "E_pa": 205e9, "nu": 0.29, "yield_pa": 655e6, "ultimate_pa": 1020e6,
                   "source": "AISI 4140, quenched and tempered: barrels, the maul's head, the cannon"},
    "7075 aluminium": {"density_kg_m3": 2810.0, "E_pa": 71.7e9, "nu": 0.33, "yield_pa": 503e6, "ultimate_pa": 572e6,
                       "source": "7075-T6: receivers, the backpack's casing, the maul's shaft"},
    "feathers": {"density_kg_m3": 90.0, "E_pa": None, "nu": None, "yield_pa": None, "ultimate_pa": None,
                 "source": "assumed: a packed crest of dyed feathers on a comb (bulk density)"},
    "polycarbonate": {"density_kg_m3": 1200.0, "E_pa": 2.4e9, "nu": 0.37, "yield_pa": 62e6, "ultimate_pa": 70e6,
                      "source": "PC sheet: the visor"},
}

#: The gibbon: Kerchak's fork for the trees, fully arboreal (a siamang's size). A gibbon's and a small monkey's tricks
#: for climbing: arms far longer than the legs (the forearm longer than the upper arm, as in the gibbons), long
#: hook hands with a short, set-back thumb, a short trunk with a broad, shallow chest (the shoulders wide on long
#: clavicles, the scapulae flat on the back: the arm hangs and swings overhead freely), a light pelvis and short legs.
#: The joint modules are small (``space_unit_robots.gibbon``) and the casing thin: about 15 kg in all.
GIBBON = dict(MARINE, pelvis_width=150.0, pelvis_depth=85.0, pelvis_height=60.0, torso_width=250.0, torso_depth=115.0, torso_height=270.0,
              waist_taper=0.78, shell_fillet=8.0, hip_spacing=110.0, thigh_length=210.0, shin_length=195.0, leg_width=50.0, leg_thickness=44.0,
              joint_boss_diameter=55.0, ankle_height=35.0, foot_length=150.0, foot_width=60.0, foot_thickness=20.0, knee_bend_deg=15.0,
              upper_arm_length=280.0, forearm_length=305.0, arm_diameter=46.0, elbow_bend_deg=15.0, neck_length=25.0,
              head_width=115.0, head_depth=125.0, head_height=110.0, eye_diameter=12.0,
              spine_segments=10, spine_curve=10.0, vertebra_diameter=34.0, rib_count=8, rib_section=9.0, sternum_width=26.0,
              clavicle_diameter=12.0, scapula_width=80.0, iliac_wing_height=60.0, femur_diameter=22.0, tibia_diameter=20.0,
              fibula_diameter=8.0, joint_module_length=55.0, joint_module_diameter=55.0, finger_length=125.0, finger_diameter=13.0,
              hand_scale=0.62, thumb_scale=0.45, scapula_turn_deg=6.0, scapula_y=0.3, scapula_height=0.42)

import math as _math

#: The hand (mm): a palm, four fingers of three phalanges each (index, middle, ring, little at their own lengths,
#: ``finger_length`` the middle one's), an opposable thumb (a metacarpal on an opposition joint, two phalanges on a
#: flexion joint). The hand's frame: the wrist at the origin, the palm hanging -z, its face +x, the thumb's side ±y.
HAND = {"palm_width": 95.0, "palm_thickness": 36.0, "palm_length": 105.0, "finger_spacing": 23.0,
        "finger_fractions": (0.88, 1.0, 0.92, 0.74), "phalanx_fractions": (0.44, 0.31, 0.25),
        "thumb_metacarpal": 75.0, "thumb_proximal": 45.0, "thumb_distal": 35.0, "thumb_diameter": 32.0,
        "thumb_base": (12.0, 44.0, -70.0), "thumb_rest_deg": 70.0, "thumb_tip_bend_deg": 25.0}
#: A hand's single curl command [0 open … 2 a fist] drives every joint: the angle [rad] per unit of curl.
FINGER_CURL = {"mcp": 0.8, "pip": 0.95, "dip": 0.6}
THUMB_CURL = {"oppose": 0.6, "curl": 0.75}
KNUCKLE_CURL = 1.4                                                         #: the fingers flexed for knuckle-walking (the hand's rest)


def hand_layout(P, sy):
    """The hand's skeleton for the ``sy`` side (+1 left) in the hand's frame [mm]: the palm's box (thickness, width,
    length), each finger's root, phalanx lengths and radius, the thumb's base, metacarpal direction, lengths, radius
    and joint axes (opposition about the palm's long axis, flexion about an axis across the metacarpal: positive
    bends the thumb down over the fingers)."""
    hs, ts = P.get("hand_scale", 1.0), P.get("thumb_scale", 1.0)       # a smaller palm; a shorter, set-back thumb (a gibbon's)
    H = dict(HAND, palm_width=HAND["palm_width"] * hs, palm_thickness=HAND["palm_thickness"] * hs, palm_length=HAND["palm_length"] * hs,
             finger_spacing=HAND["finger_spacing"] * hs, thumb_metacarpal=HAND["thumb_metacarpal"] * ts, thumb_proximal=HAND["thumb_proximal"] * ts,
             thumb_distal=HAND["thumb_distal"] * ts, thumb_diameter=HAND["thumb_diameter"] * max(ts, 0.6),
             thumb_base=tuple(v * hs for v in HAND["thumb_base"]))
    fl, fd = P["finger_length"], P["finger_diameter"]
    fingers = []
    for k, frac in enumerate(H["finger_fractions"]):
        L = fl * frac
        fingers.append({"root": (4.0 * hs, sy * (1.5 - k) * H["finger_spacing"], -H["palm_length"]),
                        "lengths": tuple(L * f for f in H["phalanx_fractions"]), "radius": fd / 2 * (0.85 if k == 3 else 1.0)})
    a = _math.radians(H["thumb_rest_deg"])
    bx, by, bz = H["thumb_base"]
    return {"palm": (H["palm_thickness"], H["palm_width"], H["palm_length"]), "fingers": fingers,
            "thumb": {"base": (bx, sy * by, bz), "dir": (_math.cos(a), sy * _math.sin(a), 0.0),
                      "lengths": (H["thumb_metacarpal"], H["thumb_proximal"], H["thumb_distal"]), "radius": H["thumb_diameter"] / 2,
                      "oppose_axis": (0.0, 0.0, -sy), "curl_axis": (-sy * _math.sin(a), _math.cos(a), 0.0),
                      "tip_bend": _math.radians(H["thumb_tip_bend_deg"])}}


def _rot(axis, a):
    x, y, z = axis
    c, s_, t = _math.cos(a), _math.sin(a), 1 - _math.cos(a)
    return ((t * x * x + c, t * x * y - s_ * z, t * x * z + s_ * y), (t * x * y + s_ * z, t * y * y + c, t * y * z - s_ * x),
            (t * x * z - s_ * y, t * y * z + s_ * x, t * z * z + c))


def _mv(R, v):
    return tuple(sum(R[i][j] * v[j] for j in range(3)) for i in range(3))


def finger_chain(finger, curl):
    """The finger's joints and tip in the hand's frame [mm] at the hand's ``curl``: [root, pip, dip, tip]."""
    x0, y, z0 = finger["root"]
    th, pts, pos = 0.0, [(x0, y, z0)], (x0, y, z0)
    for L, key in zip(finger["lengths"], ("mcp", "pip", "dip")):
        th += FINGER_CURL[key] * curl
        pos = (pos[0] + L * _math.sin(th), y, pos[2] - L * _math.cos(th))
        pts.append(pos)
    return pts


def thumb_chain(thumb, curl):
    """The thumb's joints and tip in the hand's frame [mm] at the hand's ``curl``: [base, mcp, ip, tip]."""
    oppose, flex = THUMB_CURL["oppose"] * curl, THUMB_CURL["curl"] * curl
    Ro = _rot(thumb["oppose_axis"], oppose)
    m = _mv(Ro, thumb["dir"])
    ax = _mv(Ro, thumb["curl_axis"])
    Lm, Lp, Ld = thumb["lengths"]
    base = thumb["base"]
    mcp = tuple(base[i] + Lm * m[i] for i in range(3))
    d1 = _mv(_rot(ax, flex), m)
    ip = tuple(mcp[i] + Lp * d1[i] for i in range(3))
    d2 = _mv(_rot(ax, flex + thumb["tip_bend"]), m)
    tip = tuple(ip[i] + Ld * d2[i] for i in range(3))
    return [base, mcp, ip, tip]
