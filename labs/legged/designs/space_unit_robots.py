"""The space game units as Chiron robots, for export: Talia's model (``humanoid_robot.talia``) at each unit's
numbers, re-actuated for the weight (industrial modules: ``cycloidal 1600 Nm, brake`` at the hips and knees,
``cycloidal 800 Nm, brake`` at the ankles and the waist, ``harmonic 150 Nm, brake`` in the arms - the leg modules
sized by the march in MuJoCo) with the armour and the weapons as rigid geoms on the links, their masses
the CAD volumes (``space_units.armour_parts`` / ``weapon_parts``) × the densities of ``space_unit_specs.MATERIALS``.

* **Marine** — the carbine along the right forearm (the receiver and the stock, the barrel, the magazine): at the
  low ready with the elbow bent 60° (``nominal_qpos``), level when the forearm comes up; the combat shield on the
  left forearm.
* **Marine with a chainsaw** — the carbine, and the chainsaw along the left forearm (the body on the fist, the bar
  ahead).
* **Centurion** — the carbine, the imperial banner in the left fist (the pole vertical in the world like the maul, the
  skull-in-a-gear emblem on top and the flat antenna panel under it, tipped back), the feathered crest on the helmet.
* **Breacher** — the tower shield along the left forearm, its face on the forearm's outer side (upright with the
  elbow at 20°), the power maul in the right fist.
* **Heavy** — the rotary cannon along the right forearm (the elbow at 75° so it points ahead), its mount and the feed
  tube; the bigger backpack.

21 actuated joints each (Talia's).
"""
from __future__ import annotations

import math

from vegeta.chiron import Geom

import body_kit as bk
import humanoid_robot as hr
import space_unit_specs as su

__all__ = ["CAD_MM3", "MATERIAL", "carried_masses", "marine", "marine_saw", "centurion", "breacher", "heavy", "parts_marine",
           "parts_marine_saw", "parts_centurion", "parts_breacher", "parts_heavy"]

mm = 1e-3

#: CAD volumes [mm³] of every armour and weapon piece at the defaults (the test rebuilds them)
CAD_MM3 = {
    "marine": {"pauldrons": 12801070.1, "chest_plate": 4292690.0, "backpack": 30210811.9, "belt": 8645803.5,
               "greaves": 7406507.2, "boots": 9714685.2, "receiver": 4149663.7, "barrel": 235682.6, "magazine": 957495.8,
               "shield": 24197865.1, "strap": 4237199.4},
    "marine_saw": {"pauldrons": 12801070.1, "chest_plate": 4292690.0, "backpack": 30210811.9, "belt": 8645803.5,
                   "greaves": 7406507.2, "boots": 9714685.2, "receiver": 4149663.7, "barrel": 235682.6, "magazine": 957495.8,
                   "saw_body": 5425953.0, "saw_bar": 1217654.7},
    "centurion": {"pauldrons": 12801070.1, "chest_plate": 4292690.0, "backpack": 30210811.9, "belt": 8645803.5,
                  "greaves": 7406507.2, "boots": 9714685.2, "receiver": 4149663.7, "barrel": 235682.6, "magazine": 957495.8,
                  "banner_pole": 2992328.7, "antenna": 6616219.8, "emblem": 1503971.9, "crest": 3086414.2},
    "breacher": {"pauldrons": 12801070.1, "chest_plate": 4292690.0, "backpack": 30210811.9, "belt": 8645803.5,
                 "greaves": 7406507.2, "boots": 9714685.2, "shield": 37308855.0, "strap": 4683735.6, "shaft": 1150818.8,
                 "head": 3029269.4},
    "heavy": {"pauldrons": 12801070.1, "chest_plate": 4777344.3, "backpack": 56211187.1, "belt": 9487263.8,
              "greaves": 7406507.2, "boots": 9714685.2, "cannon": 4701116.9, "mount": 9047537.8, "feed": 2091164.1},
}
#: piece -> material; the plates are hollow shells and the backpack a casing, so a fill factor says how much of the
#: CAD volume is metal (the rest: padding, the pack's cells and electronics counted in the casing's density)
MATERIAL = {"pauldrons": "Ti-6Al-4V", "chest_plate": "Ti-6Al-4V", "backpack": "7075 aluminium", "belt": "Ti-6Al-4V",
            "greaves": "Ti-6Al-4V", "boots": "Ti-6Al-4V", "receiver": "7075 aluminium", "barrel": "4140 steel",
            "magazine": "7075 aluminium", "shield": "Ti-6Al-4V", "strap": "7075 aluminium", "shaft": "7075 aluminium",
            "head": "4140 steel", "cannon": "4140 steel", "mount": "7075 aluminium", "feed": "7075 aluminium",
            "saw_body": "7075 aluminium", "saw_bar": "4140 steel", "banner_pole": "7075 aluminium", "antenna": "7075 aluminium",
            "emblem": "4140 steel", "crest": "feathers"}
FILL = {"pauldrons": 0.22, "chest_plate": 0.5, "backpack": 0.35, "belt": 0.3, "greaves": 0.5, "boots": 0.25,
        "receiver": 0.5, "barrel": 1.0, "magazine": 0.6, "shield": 0.3, "strap": 0.4, "shaft": 0.6, "head": 1.0,
        "cannon": 0.7, "mount": 0.4, "feed": 0.3, "saw_body": 0.45, "saw_bar": 0.8, "banner_pole": 0.35, "antenna": 0.22,
        "emblem": 0.6, "crest": 1.0}
ARMOUR = ("pauldrons", "chest_plate", "backpack", "belt", "greaves", "boots", "crest")


def carried_masses(unit: str) -> dict:
    """{piece: kg} of a unit's armour and weapons: CAD volume × fill × density."""
    return {k: v * 1e-9 * FILL[k] * su.MATERIALS[MATERIAL[k]]["density_kg_m3"] for k, v in CAD_MM3[unit].items()}


def _link(root, name):
    for lk, _ in root.walk():
        if lk.name == name:
            return lk
    raise KeyError(name)


def _reactuate(r):
    """Industrial joint modules for the weight, sized by the march in MuJoCo (``sims/space_squad_march.py``: a
    400 N m module sags and saturates under a 300 kg unit in single support): hips and knees ``cycloidal 1600 Nm,
    brake``, ankles and the waist ``cycloidal 800 Nm, brake``, arms ``harmonic 150 Nm, brake``; the actuator point
    masses scaled to the modules."""
    from ogma import actuators as act
    big, mid, arm = (bk.servo("cycloidal 1600 Nm, brake", kp=6000.0, kd=60.0), bk.servo("cycloidal 800 Nm, brake", kp=3000.0, kd=30.0),
                     bk.servo("harmonic 150 Nm, brake", kp=400.0, kd=6.0))
    m_big, m_mid, m_arm = (act.get(k).mass_g / 1000.0 for k in ("cycloidal 1600 Nm, brake", "cycloidal 800 Nm, brake", "harmonic 150 Nm, brake"))
    for lk, _ in r.root.walk():
        for j in lk.joints:
            j.servo = arm if j.tag in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow") else \
                big if j.tag in ("hip_yaw", "hip_roll", "hip_pitch", "knee") else mid
        for pm in lk.masses:
            if "actuator" in pm.name:
                if "shoulder" in pm.name:
                    pm.mass = 3 * m_arm
                elif "elbow" in pm.name:
                    pm.mass = m_arm
                elif "hip" in pm.name:
                    pm.mass = 3 * m_big
                elif "knee" in pm.name:
                    pm.mass = m_big
                elif "ankle" in pm.name:
                    pm.mass = 2 * m_mid
                else:
                    pm.mass = m_mid                                        # the waist
    r.sources["actuators"] = ("ogma.actuators: cycloidal 1600 Nm, brake (hips, knees), cycloidal 800 Nm, brake (ankles, waist), "
                              "harmonic 150 Nm, brake (arms); sized by sims/space_squad_march.py")
    return r


def _armour(r, unit, P):
    """The armour as boxes on the links, each at its CAD mass."""
    m = carried_masses(unit)
    Wt, Dt, Ht = P["torso_width"] * mm, P["torso_depth"] * mm, P["torso_height"] * mm
    Dp, Wp, Hp = P["pelvis_depth"] * mm, P["pelvis_width"] * mm, P["pelvis_height"] * mm
    d, R = P["arm_diameter"] * mm, P["pauldron_diameter"] * mm / 2
    ti, al = (0.42, 0.45, 0.50, 1.0), (0.30, 0.32, 0.36, 1.0)
    torso = _link(r.root, "torso")
    t = P["chest_plate_thickness"] * mm
    torso.geoms += [
        Geom("chest_plate", "box", (t / 2, Wt * 0.39, Ht * 0.26), pos=(Dt / 2 + t / 2 - 0.008, 0.0, Ht * 0.6),
             mass=m["chest_plate"], role="body", rgba=ti),
        Geom("backpack", "box", (P["backpack_depth"] * mm / 2, P["backpack_width"] * mm / 2, P["backpack_height"] * mm / 2),
             pos=(-Dt / 2 - P["backpack_depth"] * mm / 2 + 0.008, 0.0, Ht * 0.5), mass=m["backpack"], role="body", rgba=al)]
    pelvis = r.root
    pelvis.geoms.append(Geom("belt", "box", (Dp / 2 + 0.018, Wp / 2 + 0.018, P["belt_height"] * mm / 2),
                             pos=(0.0, 0.0, Hp / 2 - P["belt_height"] * mm / 2 + 0.01), mass=m["belt"], role="body", rgba=ti))
    for side, sy in (("L", 1), ("R", -1)):
        up = _link(r.root, f"{side}_upper_arm")
        up.geoms.append(Geom(f"{side}_pauldron", "sphere", (R,), pos=(0.0, 0.0, d * 0.35), mass=m["pauldrons"] / 2,
                             role="body", rgba=ti))
        shin = _link(r.root, f"{side}_shin")
        L2 = P["shin_length"] * mm
        shin.geoms.append(Geom(f"{side}_greave", "box", (P["greave_thickness"] * mm / 2, P["leg_thickness"] * mm * 0.62, L2 * 0.36),
                               pos=(0.03 + P["greave_thickness"] * mm / 2, 0.0, -L2 * 0.52), mass=m["greaves"] / 2, role="body", rgba=ti))
        foot = _link(r.root, f"{side}_foot_link")
        Lf, Wf, ha = P["foot_length"] * mm, P["foot_width"] * mm, P["ankle_height"] * mm
        foot.geoms.append(Geom(f"{side}_boot", "box", (Lf * 0.31, Wf * 0.53, ha * 0.65), pos=(Lf * 0.25, 0.0, -ha + ha * 0.62),
                               mass=m["boots"] / 2, role="body", rgba=ti))
    return r


def _unit(unit, name, P, notes):
    p = {k: v for k, v in P.items() if k in hr.P}
    r = hr.talia(name, p, notes=notes)
    _reactuate(r)
    _armour(r, unit, P)
    e = math.radians(P["elbow_bend_deg"])
    r.nominal_qpos = {"L_elbow": -e, "R_elbow": -e}
    r.sources["geometry"] = f"designs/space_units.py ({name}: space_unit_specs.{unit.upper()})"
    r.sources["carried"] = "space_units.armour_parts / weapon_parts: CAD volumes × fill × space_unit_specs.MATERIALS"
    return r


#: a box built along x (its length) and z (its height) turned into the forearm's frame: x -> down the forearm (-z),
#: z -> the forearm's outer side (+x)
ALONG = (math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)


def marine():
    P = su.MARINE
    r = _unit("marine", "Marine", P, "Marine: a 1.86 m armoured humanoid (the line unit) with a carbine and a combat shield, "
                                     "21 actuated joints (design stage).")
    m = carried_masses("marine")
    _carbine(r, P, m)
    _shield(r, P, m)
    return r


def _shield(r, P, m):
    """The shield (``shield_*`` of the specs) standing vertical in the world ahead of the left forearm's middle, its
    foot ``space_units.SHIELD_FOOT`` above the ground, and its strap on the forearm."""
    import humanoid as hu

    Lf, d = P["forearm_length"] * mm, P["arm_diameter"] * mm
    t, W, H = P["shield_thickness"] * mm, P["shield_width"] * mm, P["shield_height"] * mm
    e = math.radians(P["elbow_bend_deg"])
    up, fwd = (math.sin(e), 0.0, math.cos(e)), (math.cos(e), 0.0, -math.sin(e))     # the world's up and +x in the bent forearm
    level = (math.cos(-e / 2), 0.0, math.sin(-e / 2), 0.0)
    hz = -Lf - 0.04
    zm = hz + Lf * 0.5                                                    # the forearm's middle (forearm frame)
    # the forearm's middle above the ground, from the CAD's build pose
    pcad = {k: v for k, v in P.items()}
    hh = hu.Humanoid.hip_height(pcad) * mm
    zs = hh + P["pelvis_height"] * mm / 2 + P["torso_height"] * mm - d * 0.6
    mid_world = zs - P["upper_arm_length"] * mm - Lf * 0.5 * math.cos(e)
    dz = 0.25 + H / 2 - mid_world
    xo = max(d / 2 + t / 2 + 0.025, P["torso_depth"] * mm / 2 + P["chest_plate_thickness"] * mm + 0.05 + t / 2 - Lf * 0.5 * math.sin(e))
    fore_l = _link(r.root, "L_forearm")
    fore_l.geoms += [
        Geom("shield", "box", (t / 2 + 0.006, W / 2, H / 2), pos=(fwd[0] * xo + up[0] * dz, 0.0, zm + fwd[2] * xo + up[2] * dz),
             quat=level, mass=m["shield"], role="link", rgba=(0.42, 0.45, 0.50, 1.0)),
        Geom("shield_strap", "box", (d * 0.65 + 0.02, d * 0.65, d * 0.55), pos=(d * 0.35, 0.0, zm), mass=m["strap"], role="link",
             rgba=bk.DARK)]


def _carbine(r, P, m):
    """The carbine along the right forearm."""
    Lf, d = P["forearm_length"] * mm, P["arm_diameter"] * mm
    L, rh, w = P["carbine_length"] * mm, P["receiver_height"] * mm, P["receiver_width"] * mm
    hz = -Lf - 0.04                                                       # the fist on the forearm's axis
    x = d * 0.55 + rh * 0.35
    rec_len = L * 0.42
    zc = hz - L * 0.08                                                    # the receiver's centre, ahead of the fist
    zb = zc - rec_len / 2
    fore = _link(r.root, "R_forearm")
    fore.geoms += [
        Geom("carbine_receiver", "box", (rec_len / 2 + L * 0.14, w / 2, rh / 2), pos=(x, 0.0, zc + L * 0.14), quat=ALONG,
             mass=m["receiver"], role="link", rgba=bk.DARK),
        Geom("carbine_barrel", "capsule", (P["barrel_diameter"] * mm / 2,), fromto=(x + rh * 0.2, 0.0, zb + 0.005, x + rh * 0.2, 0.0, zb - L * 0.30),
             mass=m["barrel"], role="link", rgba=bk.BLACK),
        Geom("carbine_magazine", "box", (w * 0.65, w * 0.45, P["magazine_height"] * mm / 2),
             pos=(x - rh / 2 - P["magazine_height"] * mm / 2 + 0.008, 0.0, zc - rec_len * 0.2), quat=ALONG, mass=m["magazine"],
             role="link", rgba=bk.DARK)]


def marine_saw():
    P = su.MARINE_SAW
    r = _unit("marine_saw", "Marine (chainsaw)", P, "Marine with a chainsaw: a 1.86 m armoured humanoid with a carbine in the "
                                                     "right fist and a chainsaw in the left, 21 actuated joints (design stage).")
    m = carried_masses("marine_saw")
    _carbine(r, P, m)
    Lf, d = P["forearm_length"] * mm, P["arm_diameter"] * mm
    L, W, t = P["saw_bar_length"] * mm, P["saw_bar_width"] * mm, P["saw_bar_thickness"] * mm
    bl, bh = P["saw_body_length"] * mm, P["saw_body_height"] * mm
    hz = -Lf - 0.04
    x = d * 0.5 + bh * 0.3
    zb0 = hz - bl * 0.35 - bl / 2 + 0.01
    fore_l = _link(r.root, "L_forearm")
    fore_l.geoms += [
        Geom("saw_body", "box", (bh / 2, bh * 0.4, bl / 2), pos=(x, 0.0, hz - bl * 0.35), mass=m["saw_body"], role="link", rgba=bk.DARK),
        Geom("saw_bar", "box", (W / 2 + 0.01, t * 0.8, L / 2), pos=(x, 0.0, zb0 - L / 2), mass=m["saw_bar"], role="link", rgba=bk.BLACK)]
    return r


def centurion():
    P = su.CENTURION
    r = _unit("centurion", "Centurion", P, "Centurion: a 1.86 m armoured humanoid with a carbine, the imperial banner (a skull "
                                           "in a gear on the pole) and a feathered crest, 21 actuated joints (design stage).")
    m = carried_masses("centurion")
    _carbine(r, P, m)
    Lf, d = P["forearm_length"] * mm, P["arm_diameter"] * mm
    L, pd = P["pole_length"] * mm, P["pole_diameter"] * mm
    e = math.radians(P["elbow_bend_deg"])
    down, up = (-math.sin(e), 0.0, -math.cos(e)), (math.sin(e), 0.0, math.cos(e))     # the world's vertical in the bent forearm
    hand = (0.0, 0.0, -Lf - 0.04)
    bottom = tuple(hand[i] + down[i] * L * 0.3 for i in range(3))
    top = tuple(hand[i] + up[i] * L * 0.7 for i in range(3))
    level = (math.cos(-e / 2), 0.0, math.sin(-e / 2), 0.0)                # a world-level box in the bent forearm
    fwd = (math.cos(e), 0.0, -math.sin(e))                                # the world's +x in the bent forearm
    # the gear's axis along the world's x: the cylinder's z turned onto x (pitch +90 deg), then the forearm's tilt
    qa, qb = (math.cos(-e / 2), 0.0, math.sin(-e / 2), 0.0), (math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)
    level_x = (qa[0] * qb[0] - qa[2] * qb[2], 0.0, qa[0] * qb[2] + qa[2] * qb[0], 0.0)
    G = P["gear_diameter"] * mm
    W, H, t = P["antenna_width"] * mm, P["antenna_height"] * mm, P["antenna_thickness"] * mm
    za = -G * 0.95 - H * 0.55                                              # the panel's centre under the gear (world, from the top)
    xa = pd * 0.5 + t / 2 + 0.03
    tilt = math.radians(P["antenna_tilt_deg"])
    qt = (math.cos(-tilt / 2), 0.0, math.sin(-tilt / 2), 0.0)             # the panel tipped back, then the forearm's tilt
    tipped = (level[0] * qt[0] - level[2] * qt[2], 0.0, level[0] * qt[2] + level[2] * qt[0], 0.0)
    fore_l = _link(r.root, "L_forearm")
    fore_l.geoms += [
        Geom("banner_pole", "capsule", (pd / 2,), fromto=bottom + top, mass=m["banner_pole"] * 0.8, role="link", rgba=bk.DARK),
        Geom("banner_bracket", "box", (xa / 2, pd * 0.8, pd * 0.7), pos=tuple(top[i] + up[i] * za + fwd[i] * xa / 2 for i in range(3)),
             quat=level, mass=m["banner_pole"] * 0.2, role="link", rgba=bk.DARK),
        Geom("antenna", "box", (t / 2 + 0.004, W / 2, H / 2), pos=tuple(top[i] + up[i] * za + fwd[i] * xa for i in range(3)),
             quat=tipped, mass=m["antenna"], role="link", rgba=(0.9, 0.9, 0.92, 1.0)),
        Geom("emblem", "cylinder", (G / 2, P["gear_thickness"] * mm / 2), pos=tuple(top[i] - up[i] * G * 0.45 for i in range(3)),
             quat=level_x, mass=m["emblem"], role="link", rgba=(0.75, 0.65, 0.3, 1.0))]
    torso = _link(r.root, "torso")
    Ht, Hh = P["torso_height"] * mm, P["head_height"] * mm
    zh = Ht + P["neck_length"] * mm + Hh
    torso.geoms.append(Geom("crest", "box", (P["crest_length"] * mm / 2, P["crest_width"] * mm / 2, P["crest_height"] * mm / 2),
                            pos=(-P["crest_length"] * mm * 0.1, 0.0, zh + 0.012 + P["crest_height"] * mm / 2), mass=m["crest"], role="body",
                            rgba=(0.75, 0.1, 0.12, 1.0)))
    return r


def breacher():
    P = su.BREACHER
    r = _unit("breacher", "Breacher", P, "Breacher: a 1.86 m armoured humanoid with a tower shield and a power maul, "
                                         "21 actuated joints (design stage).")
    m = carried_masses("breacher")
    _shield(r, P, m)
    Lf, d = P["forearm_length"] * mm, P["arm_diameter"] * mm
    hz = -Lf - 0.04
    L, hd, hl, sd = P["maul_length"] * mm, P["maul_head_diameter"] * mm, P["maul_head_length"] * mm, P["maul_shaft_diameter"] * mm
    e = math.radians(P["elbow_bend_deg"])
    down, up = (-math.sin(e), 0.0, -math.cos(e)), (math.sin(e), 0.0, math.cos(e))     # the world's vertical in the bent forearm
    hand = (0.0, 0.0, hz)
    bottom = tuple(hand[i] + down[i] * L * 0.3 for i in range(3))
    top = tuple(hand[i] + up[i] * L * 0.7 for i in range(3))
    fore_r = _link(r.root, "R_forearm")
    fore_r.geoms += [
        Geom("maul_shaft", "capsule", (sd / 2,), fromto=bottom + top, mass=m["shaft"], role="link", rgba=bk.DARK),
        Geom("maul_head", "cylinder", (hd / 2, hl / 2), pos=tuple(top[i] - up[i] * hd * 0.1 for i in range(3)),
             quat=(math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0), mass=m["head"], role="link", rgba=bk.BLACK)]
    return r


def heavy():
    P = su.HEAVY
    r = _unit("heavy", "Heavy", P, "Heavy: a 1.86 m armoured humanoid on a wider frame with a rotary cannon fed from its "
                                   "backpack, 21 actuated joints (design stage).")
    m = carried_masses("heavy")
    Lf, d = P["forearm_length"] * mm, P["arm_diameter"] * mm
    L, D = P["cannon_length"] * mm, P["cannon_diameter"] * mm
    mid = (0.0, 0.0, -Lf * 0.55)
    axis0 = (mid[0], mid[1] - d * 0.9, mid[2] - d * 0.25)
    breech, muzzle = (axis0[0], axis0[1], axis0[2] + L * 0.25), (axis0[0], axis0[1], axis0[2] - L * 0.75)
    fore = _link(r.root, "R_forearm")
    fore.geoms += [
        Geom("cannon", "capsule", (D / 2,), fromto=breech + muzzle, mass=m["cannon"], role="link", rgba=bk.BLACK),
        Geom("cannon_mount", "box", (d * 0.8, d * 1.1, d * 0.75), pos=(mid[0], mid[1] - d * 0.45, mid[2] - d * 0.1),
             mass=m["mount"], role="link", rgba=bk.DARK)]
    torso = _link(r.root, "torso")
    Dt, Ht = P["torso_depth"] * mm, P["torso_height"] * mm
    torso.geoms.append(Geom("cannon_feed", "capsule", (P["feed_diameter"] * mm / 2,),
                            fromto=(-Dt / 2 - P["backpack_depth"] * mm / 2, -P["backpack_width"] * mm * 0.3, Ht * 0.5 - P["backpack_height"] * mm / 2 + 0.02,
                                    -Dt / 2 - 0.06, -P["torso_width"] * mm / 2 - 0.16, Ht * 0.25),
                            mass=m["feed"], role="body", rgba=bk.DARK))
    return r


def _parts(unit, motions_fn):
    import motion_kit as mk
    kin, ms = motions_fn()
    out = mk.block(ms, _base_height(kin))
    pieces = CAD_MM3[unit]
    out["materials"] = {k: dict(su.MATERIALS[k]) for k in sorted({MATERIAL[p] for p in pieces})}
    masses = carried_masses(unit)
    out["carried"] = {k: {"material": MATERIAL[k], "volume_m3": round(v * 1e-9, 9), "fill": FILL[k], "mass_kg": round(masses[k], 4)}
                      for k, v in pieces.items()}
    out["tags"]["data:carried*"] = "cad"
    return out


def _base_height(kin):
    import humanoid_motions as hum
    return hum.base_height(kin)


def parts_marine(machine) -> dict:
    import space_unit_motions as sm
    return _parts("marine", sm.motions_marine)


def parts_marine_saw(machine) -> dict:
    import space_unit_motions as sm
    return _parts("marine_saw", sm.motions_marine_saw)


def parts_centurion(machine) -> dict:
    import space_unit_motions as sm
    return _parts("centurion", sm.motions_centurion)


def parts_breacher(machine) -> dict:
    import space_unit_motions as sm
    return _parts("breacher", sm.motions_breacher)


def parts_heavy(machine) -> dict:
    import space_unit_motions as sm
    return _parts("heavy", sm.motions_heavy)
