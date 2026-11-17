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

25 actuated joints each (Talia's 21 and the spine's four: a lumbar and a thoracic pitch/roll module).
"""
from __future__ import annotations

import math

from vegeta.chiron import Geom, Joint, Link, PointMass

import body_kit as bk
import humanoid_robot as hr
import space_unit_specs as su

__all__ = ["CAD_MM3", "MATERIAL", "carried_masses", "marine", "marine_saw", "centurion", "breacher", "heavy", "parts_marine",
           "parts_marine_saw", "parts_centurion", "parts_breacher", "parts_heavy"]

mm = 1e-3

#: CAD volumes [mm³] of every armour and weapon piece at the defaults (the test rebuilds them)
CAD_MM3 = {
    "marine": {"helmet": 1914219.9, "shoulder_guards": 7235516.9, "lorica": 16051563.8, "arm_lames": 7271690.0, "chest_bay": 11194344.7, "belly_bay": 3806772.1, "backpack": 30210811.9, "belt": 13319528.4, "greaves": 8973334.3, "boots": 9714685.2, "receiver": 4239405.9, "barrel": 235682.6, "magazine": 957495.8, "shield": 24197865.1, "handle": 461906.1},
    "marine_saw": {"helmet": 1914219.9, "shoulder_guards": 7235516.9, "lorica": 16051563.8, "arm_lames": 7271690.0, "chest_bay": 11194344.7, "belly_bay": 3806772.1, "backpack": 30210811.9, "belt": 13319528.4, "greaves": 8973334.3, "boots": 9714685.2, "receiver": 4239405.9, "barrel": 235682.6, "magazine": 957495.8, "saw_body": 5557142.5, "saw_bar": 1217654.7},
    "centurion": {"helmet": 1914219.9, "shoulder_guards": 7235516.9, "lorica": 16051563.8, "arm_lames": 7271690.0, "chest_bay": 11194344.7, "belly_bay": 3806772.1, "backpack": 30210811.9, "belt": 13319528.4, "greaves": 8973334.3, "boots": 9714685.2, "crest": 3086414.2, "receiver": 4239405.9, "barrel": 235682.6, "magazine": 957495.8, "banner_pole": 2992328.7, "antenna": 6616219.8, "emblem": 1503971.9},
    "breacher": {"helmet": 1914219.9, "shoulder_guards": 7235516.9, "lorica": 16051563.8, "arm_lames": 7271689.9, "chest_bay": 11194344.7, "belly_bay": 3806772.1, "backpack": 30210811.9, "belt": 13319528.4, "greaves": 8973334.3, "boots": 9714685.2, "shield": 37308855.0, "handle": 624022.0, "shaft": 1150818.8, "head": 3029269.4},
    "heavy": {"helmet": 1914219.9, "shoulder_guards": 7235516.9, "lorica": 17271722.0, "arm_lames": 7271690.0, "chest_bay": 13267510.8, "belly_bay": 4076459.9, "backpack": 56211187.1, "belt": 14221020.7, "greaves": 8973334.3, "boots": 9714685.2, "cannon": 4701116.9, "mount": 14398915.1, "feed": 2288040.3},
}
#: piece -> material; the plates are hollow shells and the backpack a casing, so a fill factor says how much of the
#: CAD volume is metal (the rest: padding, the pack's cells and electronics counted in the casing's density)
MATERIAL = {"belly_bay": "7075 aluminium", "helmet": "Ti-6Al-4V", "shoulder_guards": "Ti-6Al-4V", "lorica": "Ti-6Al-4V", "arm_lames": "Ti-6Al-4V", "chest_bay": "7075 aluminium", "backpack": "7075 aluminium", "belt": "Ti-6Al-4V",
            "greaves": "Ti-6Al-4V", "boots": "Ti-6Al-4V", "receiver": "7075 aluminium", "barrel": "4140 steel",
            "magazine": "7075 aluminium", "shield": "Ti-6Al-4V", "handle": "7075 aluminium", "shaft": "7075 aluminium",
            "head": "4140 steel", "cannon": "4140 steel", "mount": "7075 aluminium", "feed": "7075 aluminium",
            "saw_body": "7075 aluminium", "saw_bar": "4140 steel", "banner_pole": "7075 aluminium", "antenna": "7075 aluminium",
            "emblem": "4140 steel", "crest": "feathers"}
FILL = {"belly_bay": 0.3, "helmet": 0.6, "shoulder_guards": 0.5, "lorica": 0.5, "arm_lames": 0.5, "chest_bay": 0.3, "backpack": 0.35, "belt": 0.3, "greaves": 0.5, "boots": 0.25,
        "receiver": 0.5, "barrel": 1.0, "magazine": 0.6, "shield": 0.3, "handle": 0.5, "shaft": 0.6, "head": 1.0,
        "cannon": 0.7, "mount": 0.4, "feed": 0.3, "saw_body": 0.45, "saw_bar": 0.8, "banner_pole": 0.35, "antenna": 0.22,
        "emblem": 0.6, "crest": 1.0}
ARMOUR = ("belly_bay", "helmet", "shoulder_guards", "lorica", "arm_lames", "chest_bay", "backpack", "belt", "greaves", "boots", "crest")


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
            if j.tag in ("spine_pitch", "spine_roll"):
                continue                                                   # the spine's modules: set in _spine
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
                elif "ankle" in pm.name or "spine" in pm.name:
                    pm.mass = 2 * m_mid
                else:
                    pm.mass = m_mid                                        # the waist
    r.sources["actuators"] = ("ogma.actuators: cycloidal 1600 Nm, brake (hips, knees), cycloidal 800 Nm, brake (ankles, waist), "
                              "harmonic 150 Nm, brake (arms); sized by sims/space_squad_march.py")
    return r


def _spine(r, P):
    """Talia's one torso link becomes a spine: the lumbar link (waist yaw plus a lumbar pitch/roll module at the
    sacrum) and the thorax (a thoracic pitch/roll module at the lumbar's top) carrying the ribcage, the shoulder
    girdle, the head and the arms. The segments give the trunk its compliance: the modules are held by their PD
    (``cycloidal 800 Nm, brake``, stiff, well damped), so the upper body can lean into a step without the pelvis
    tilting with it. The geoms: the lumbar column, the ribcage, the yoke; the masses as Talia's."""
    import dataclasses
    torso = _link(r.root, "torso")
    Wt, Dt, Ht = P["torso_width"] * mm, P["torso_depth"] * mm, P["torso_height"] * mm
    xs = -P["spine_set_back"] * Dt
    split = 0.42 * Ht                                                     # the thoracic module's height above the sacrum
    spine_servo = bk.servo("cycloidal 800 Nm, brake", kp=3000.0, kd=300.0)
    pitch_range, roll_range = (-0.35, 0.35), (-0.25, 0.25)
    thorax = Link("thorax", pos=(0.0, 0.0, split),
                  joints=[Joint("spine_thoracic_pitch", axis=(0, 1, 0), range=pitch_range, tag="spine_pitch", servo=spine_servo),
                          Joint("spine_thoracic_roll", axis=(1, 0, 0), range=roll_range, tag="spine_roll", servo=spine_servo)],
                  geoms=[], masses=[PointMass("spine_thoracic_actuators", 2 * 11.0)], children=list(torso.children))
    for arm in thorax.children:                                           # the arms hang from the thorax now
        arm.pos = (arm.pos[0], arm.pos[1], arm.pos[2] - split)
    for g in list(torso.geoms):                                           # the head and the neck move to the thorax
        if g.name in ("neck", "head"):
            if g.fromto is not None:
                f = g.fromto
                g.fromto = (f[0], f[1], f[2] - split, f[3], f[4], f[5] - split)
            else:
                g.pos = (g.pos[0], g.pos[1], g.pos[2] - split)
            thorax.geoms.append(g)
    torso.geoms = [g for g in torso.geoms if g.name not in ("torso_shell", "neck", "head")]
    for pm in list(torso.masses):
        if pm.name in ("battery", "electronics"):
            pm.pos = (pm.pos[0], pm.pos[1], pm.pos[2] - split)
            thorax.masses.append(pm)
            torso.masses.remove(pm)
    torso.masses.append(PointMass("spine_lumbar_actuators", 2 * 11.0))
    torso.joints += [Joint("spine_lumbar_pitch", axis=(0, 1, 0), range=pitch_range, tag="spine_pitch", servo=spine_servo),
                     Joint("spine_lumbar_roll", axis=(1, 0, 0), range=roll_range, tag="spine_roll", servo=spine_servo)]
    torso.geoms.append(bk.capsule("lumbar", P["vertebra_diameter"] * mm * 0.5, (xs, 0, 0.0), (xs + 0.02, 0, split)))
    torso.children = [thorax]
    thorax.geoms += [bk.box("ribcage", (Dt / 2 * 0.92, Wt / 2 * 0.9, (Ht - split) / 2 * 0.86), (0.0, 0.0, (Ht - split) / 2 * 0.95), rho=bk.DENSITY["shell"] * 0.45),
                     bk.box("yoke", (Dt * 0.2, Wt / 2 + 0.02, 0.035), (0.0, 0.0, Ht - split - P["arm_diameter"] * mm * 0.6))]
    return r


def _armour(r, unit, P):
    """The armour as boxes on the links, each at its CAD mass."""
    m = carried_masses(unit)
    Wt, Dt, Ht = P["torso_width"] * mm, P["torso_depth"] * mm, P["torso_height"] * mm
    Dp, Wp, Hp = P["pelvis_depth"] * mm, P["pelvis_width"] * mm, P["pelvis_height"] * mm
    d, R = P["arm_diameter"] * mm, P["pauldron_diameter"] * mm / 2
    ti, al = (0.42, 0.45, 0.50, 1.0), (0.30, 0.32, 0.36, 1.0)
    thorax, zt = _link(r.root, "thorax"), 0.42 * Ht                        # on the ribs: the thorax link
    t = P["chest_plate_thickness"] * mm
    n = int(P["lorica_bands"])
    hb = Ht * 0.52 / n
    for k in range(n):                                                    # the lorica's girth hoops, a band each
        thorax.geoms.append(Geom(f"lorica_{k}", "box", ((Dt + 2 * t + k * 0.003) / 2, (Wt * 0.9 + 2 * t + k * 0.004) / 2, hb * 0.62),
                                 pos=(0.0, 0.0, Ht * 0.34 + (k + 0.5) * hb - zt), mass=m["lorica"] / (n + 1), role="body", rgba=ti))
    thorax.geoms += [
        Geom("breast_band", "box", (t / 2, Wt * 0.25, Ht * 0.08), pos=(Dt / 2 + t * 1.5 - 0.008, 0.0, Ht * 0.86 - zt),
             mass=m["lorica"] / (n + 1), role="body", rgba=ti),
        Geom("chest_bay", "box", (Dt * 0.17, Wt * 0.2, Ht * 0.13), pos=(-Dt * 0.05, 0.0, Ht * 0.6 - zt), mass=m["chest_bay"],
             role="body", rgba=al)]                                       # the electronics' casing inside the ribcage
    _link(r.root, "torso").geoms.append(Geom("belly_bay", "box", (Dp * 0.21, Wp * 0.25, Ht * 0.1), pos=(0.02, 0.0, Ht * 0.17), mass=m["belly_bay"],
                                             role="body", rgba=al))       # the power electronics in front of the lumbar spine
    thorax.geoms += [
        Geom("backpack", "box", (P["backpack_depth"] * mm / 2, P["backpack_width"] * mm / 2, P["backpack_height"] * mm / 2),
             pos=(-Dt / 2 - P["backpack_depth"] * mm / 2 + 0.008, 0.0, Ht * 0.5 - zt), mass=m["backpack"], role="body", rgba=al)]
    Wh, Dh, Hh, th = P["head_width"] * mm, P["head_depth"] * mm, P["head_height"] * mm, P["helmet_thickness"] * mm
    zh = Ht - zt + P["neck_length"] * mm + Hh / 2
    thorax.geoms += [                                                     # the galea: the brow band, the cheek guards, the nape
        Geom("helmet_brow", "box", (Dh / 2 + th, Wh / 2 + th, Hh * 0.11), pos=(0.0, 0.0, zh + Hh * 0.28), mass=m["helmet"] * 0.4, role="body", rgba=ti),
        Geom("helmet_nape", "box", (Dh * 0.2, Wh * 0.45, th / 2), pos=(-Dh * 0.62, 0.0, zh - Hh * 0.05),
             quat=(math.cos(math.radians(17.5)), 0.0, math.sin(math.radians(17.5)), 0.0), mass=m["helmet"] * 0.2, role="body", rgba=ti)]
    for side, sy in (("L", 1), ("R", -1)):
        thorax.geoms.append(Geom(f"{side}_cheek_guard", "box", (Dh * 0.25, th / 2, Hh * 0.3), pos=(Dh * 0.12, sy * (Wh / 2 + th / 2 + 0.002), zh - Hh * 0.12),
                                 mass=m["helmet"] * 0.2, role="body", rgba=ti))
    pelvis = r.root
    pelvis.geoms.append(Geom("belt", "box", (Dp / 2 + 0.018, Wp / 2 + 0.018, P["belt_height"] * mm / 2),
                             pos=(0.0, 0.0, Hp / 2 - P["belt_height"] * mm / 2 + 0.01), mass=m["belt"], role="body", rgba=ti))
    Lf = P["forearm_length"] * mm
    for side, sy in (("L", 1), ("R", -1)):
        up = _link(r.root, f"{side}_upper_arm")
        for k in range(3):                                                # the shoulder guards: three layered plates
            a = -sy * math.radians(12 + 24 * k)
            up.geoms.append(Geom(f"{side}_shoulder_guard_{k}", "box", (R * 0.75, R * 0.25, P["pauldron_thickness"] * mm * 0.25),
                                 pos=(0.0, sy * (R * 0.2 + k * R * 0.26), d * 0.35 + R * 0.62 - k * R * 0.28),
                                 quat=(math.cos(a / 2), math.sin(a / 2), 0.0, 0.0), mass=m["shoulder_guards"] / 6, role="body", rgba=ti))
        fore = _link(r.root, f"{side}_forearm")
        fl, fd = P["finger_length"] * mm, P["finger_diameter"] * mm
        fore.geoms = [g for g in fore.geoms if g.name != f"{side}_hand"]
        wrist = -Lf
        fore.geoms.append(bk.box(f"{side}_palm", (d * 0.47, d * 0.25, d * 0.45), (0.0, 0.0, wrist - d * 0.45), role="link", rgba=bk.DARK))
        for k in range(4):                                                # the gripping hand: four curled fingers
            y = (k - 1.5) * (d * 0.5 / 3.4)
            pos, ang, seg = (d * 0.47, y, wrist - d * 0.86), -20.0, fl / 3
            for j in range(3):
                ang += 55.0
                nxt = (pos[0] - seg * math.cos(math.radians(ang)), y, pos[2] - seg * math.sin(math.radians(ang)))
                fore.geoms.append(bk.capsule(f"{side}_finger_{k}_{j}", fd * (0.5 - 0.05 * j), pos, nxt, role="link", rgba=bk.DARK))
                pos = nxt
        shin = _link(r.root, f"{side}_shin")
        L2, rt, gt = P["shin_length"] * mm, P["tibia_diameter"] * mm / 2, P["greave_thickness"] * mm
        for k in range(4):                                                # the greave's bands
            shin.geoms.append(Geom(f"{side}_greave_{k}", "cylinder", (rt + gt + 0.002 + k * 0.002, L2 * 0.085), pos=(0.0, 0.0, -L2 * (0.3 + 0.14 * k)),
                                   mass=m["greaves"] / 8, role="body", rgba=ti))
        La = P["upper_arm_length"] * mm
        for k in range(3):                                                # the arm lames: the upper arm's and the forearm's
            up.geoms.append(Geom(f"{side}_upper_lame_{k}", "cylinder", (d / 2 + 0.014 + k * 0.003, La * 0.1), pos=(0.0, 0.0, -La * (0.3 + 0.17 * k)),
                                 mass=m["arm_lames"] / 12, role="body", rgba=ti))
            fore_l = _link(r.root, f"{side}_forearm")
            fore_l.geoms.append(Geom(f"{side}_fore_lame_{k}", "cylinder", (d * 0.42 + 0.012 + k * 0.003, Lf * 0.1), pos=(0.0, 0.0, -Lf * (0.32 + 0.17 * k)),
                                     mass=m["arm_lames"] / 12, role="body", rgba=ti))
        foot = _link(r.root, f"{side}_foot_link")
        Lf, Wf, ha = P["foot_length"] * mm, P["foot_width"] * mm, P["ankle_height"] * mm
        foot.geoms.append(Geom(f"{side}_boot", "box", (Lf * 0.31, Wf * 0.53, ha * 0.65), pos=(Lf * 0.25, 0.0, -ha + ha * 0.62),
                               mass=m["boots"] / 2, role="body", rgba=ti))
    return r


#: the linear actuators' volumes [mm³] per group (both sides), rebuilt by the test from ``space_units.actuators``;
#: each weighs volume × 0.5 (the barrel and the rod round the piston) × 7075 aluminium
ACTUATOR_MM3 = {"hip_struts": 793039.6, "knee_struts": 806947.4, "ankle_struts": 469454.7, "shoulder_struts": 445491.6,
                "elbow_struts": 469919.6, "spine_struts": 582587.8}
ACTUATOR_LINK = {"hip_struts": "{s}_thigh", "knee_struts": "{s}_thigh", "ankle_struts": "{s}_shin", "shoulder_struts": "{s}_upper_arm",
                 "elbow_struts": "{s}_upper_arm", "spine_struts": "torso"}


def actuator_masses() -> dict:
    """{group: kg} of the linear actuators."""
    return {k: v * 1e-9 * 0.5 * su.MATERIALS["7075 aluminium"]["density_kg_m3"] for k, v in ACTUATOR_MM3.items()}


def _struts(r, P):
    """The linear actuators as point masses on the links they ride (per side)."""
    for name, kg in actuator_masses().items():
        if kg <= 0:
            continue
        for side in ("L", "R"):
            lk = ACTUATOR_LINK[name].format(s=side)
            z = {"hip_struts": -P["thigh_length"] * mm * 0.3, "knee_struts": -P["thigh_length"] * mm * 0.6, "ankle_struts": -P["shin_length"] * mm * 0.5,
                 "shoulder_struts": -P["upper_arm_length"] * mm * 0.2, "elbow_struts": -P["upper_arm_length"] * mm * 0.7, "spine_struts": P["torso_height"] * mm * 0.2}[name]
            _link(r.root, lk).masses.append(PointMass(f"{name}_{side}", kg / 2, (0.03, 0.0, z)))
    return r


def _unit(unit, name, P, notes):
    p = {k: v for k, v in P.items() if k in hr.P}
    r = hr.talia(name, p, notes=notes)
    _spine(r, P)
    _reactuate(r)
    _armour(r, unit, P)
    _struts(r, P)
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
                                     "25 actuated joints (design stage).")
    m = carried_masses("marine")
    _carbine(r, P, m)
    _shield(r, P, m)
    return r


def su_shield_foot():
    import space_units
    return space_units.SHIELD_FOOT * mm


def _shield(r, P, m):
    """The shield (``shield_*`` of the specs) held in the left hand: the handle bar across the fist where the fingers
    close, the plate standing vertical in the world ahead of it, its foot ``space_units.SHIELD_FOOT`` above the ground."""
    import humanoid as hu

    Lf, d = P["forearm_length"] * mm, P["arm_diameter"] * mm
    t, W, H = P["shield_thickness"] * mm, P["shield_width"] * mm, P["shield_height"] * mm
    e = math.radians(P["elbow_bend_deg"])
    up, fwd = (math.sin(e), 0.0, math.cos(e)), (math.cos(e), 0.0, -math.sin(e))     # the world's up and +x in the bent forearm
    level = (math.cos(e / 2), 0.0, math.sin(e / 2), 0.0)
    grip = (d * 0.45, 0.0, -Lf - d * 0.55)                                # where the curled fingers close (forearm frame)
    # the grip's height above the ground, from the CAD's build pose
    pcad = {k: v for k, v in P.items()}
    hh = hu.Humanoid.hip_height(pcad) * mm
    zs = hh + P["pelvis_height"] * mm / 2 + P["torso_height"] * mm - d * 0.6
    grip_world = zs - P["upper_arm_length"] * mm - (Lf + d * 0.55) * math.cos(e) + d * 0.45 * math.sin(e)
    grip_x_world = (Lf + d * 0.55) * math.sin(e) + d * 0.45 * math.cos(e)
    dz = su_shield_foot() + H / 2 - grip_world
    xo = max(d * 0.5 + t / 2, P["torso_depth"] * mm / 2 + P["chest_plate_thickness"] * mm + 0.05 + t / 2 - grip_x_world)
    fore_l = _link(r.root, "L_forearm")
    fore_l.geoms += [
        Geom("shield", "box", (t / 2 + 0.006, W / 2, H / 2), pos=tuple(grip[i] + fwd[i] * xo + up[i] * dz for i in range(3)),
             quat=level, mass=m["shield"], role="link", rgba=(0.42, 0.45, 0.50, 1.0)),
        Geom("shield_handle", "capsule", (0.016,), fromto=(grip[0], -d * 0.6, grip[2], grip[0], d * 0.6, grip[2]), mass=m["handle"] * 0.3, role="link",
             rgba=bk.DARK),
        Geom("shield_bracket", "box", (xo / 2, 0.02, 0.03), pos=tuple(grip[i] + fwd[i] * xo / 2 for i in range(3)), quat=level,
             mass=m["handle"] * 0.7, role="link", rgba=bk.DARK)]


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
    torso = _link(r.root, "thorax")
    Ht, Hh = P["torso_height"] * mm, P["head_height"] * mm
    zh = Ht - 0.42 * Ht + P["neck_length"] * mm + Hh                      # on the thorax link
    torso.geoms.append(Geom("crest", "box", (P["crest_length"] * mm / 2, P["crest_width"] * mm / 2, P["crest_height"] * mm / 2),
                            pos=(-P["crest_length"] * mm * 0.1, 0.0, zh + 0.012 + P["crest_height"] * mm / 2), mass=m["crest"], role="body",
                            rgba=(0.75, 0.1, 0.12, 1.0)))
    return r


def breacher():
    P = su.BREACHER
    r = _unit("breacher", "Breacher", P, "Breacher: a 1.86 m armoured humanoid with a tower shield and a power maul, "
                                         "25 actuated joints (design stage).")
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
    torso = _link(r.root, "thorax")
    Dt, Ht = P["torso_depth"] * mm, P["torso_height"] * mm
    zt = 0.42 * Ht                                                        # on the thorax link
    torso.geoms.append(Geom("cannon_feed", "capsule", (P["feed_diameter"] * mm / 2,),
                            fromto=(-Dt / 2 - P["backpack_depth"] * mm / 2, -P["backpack_width"] * mm * 0.3, Ht * 0.5 - P["backpack_height"] * mm / 2 + 0.02 - zt,
                                    -Dt / 2 - 0.06, -P["torso_width"] * mm / 2 - 0.16, Ht * 0.25 - zt),
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
