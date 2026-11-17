"""The space game units (CAD): bulky armoured humanoid robots on ``humanoid.Humanoid``'s frame, both hands busy —
the **Marine** (a carbine and a combat shield), the **Marine with a chainsaw** (the carbine and a chainsaw), the
**Centurion** (the carbine, the imperial banner, a feathered crest), the **Breacher** (a tower shield and a power maul) and the **Heavy** (a rotary cannon fed from its backpack). The armour: shoulder domes (pauldrons), a breastplate, a power pack with two exhaust stacks, a waist belt,
greaves over the shins, knee domes, boots and a visor slit cut into the square helmet. Numbers in
``space_unit_specs``; mm, x forward, y left, z up, the ground at z = 0.

``part``: as ``Humanoid`` (the whole machine, the trunk, a thigh, a shin, a foot), plus ``skeleton`` (the bare
frame), ``armour`` (the plates alone) and ``weapon`` (what the unit carries). The frame is a skeleton: an S-curved
spine of vertebrae set back in the chest, ribs round to a sternum, clavicles and scapulae, a pelvis of iliac wings
on a sacrum, femurs, tibia and fibula - and every joint a mechanical rotary module (``joint_module``); the armour
goes on top of the bones. ``armour_parts`` and ``weapon_parts`` give the pieces by name, for the
masses (``space_unit_robots.CAD_MM3``: the test rebuilds them).
"""
from __future__ import annotations

import math

import cadquery as cq
from vegeta.dedalus import Parameter

import shape_kit as sk
import space_unit_specs as su
from fantasy_humanoids import _add, hand_frame
from humanoid import Humanoid

__all__ = ["SpaceMarine", "SpaceMarineSaw", "Centurion", "Breacher", "Heavy", "carbine", "tower_shield", "combat_shield", "chainsaw",
           "banner", "crest", "maul", "rotary_cannon", "forearm_frame", "skeleton_trunk", "femur", "tibia", "foot_bones",
           "joint_module", "spine_x", "gripper", "galea", "actuators", "piston"]


def _torso_frame(p):
    """(z of the pelvis top, torso width, depth, height, the shoulder height) in the build pose."""
    hh = Humanoid.hip_height(p)
    z0 = hh + p["pelvis_height"] / 2
    return z0, p["torso_width"], p["torso_depth"], p["torso_height"], z0 + p["torso_height"] - p["arm_diameter"] * 0.6


# ---- the weapons: each returns {piece: solid}, in the build pose ------------------------------------------------
def forearm_frame(p, sy):
    """The bent forearm's frame as CadQuery sees it: a part built with the hand at the origin, the forearm running up
    +z to the elbow and the forearm's outer side along +x, rotated into the build pose and set on the ``sy`` hand."""
    c, _, _ = hand_frame(p, sy)
    e = p["elbow_bend_deg"]

    def place(shape):
        return shape.rotate((0, 0, 0), (0, 1, 0), -e).translate(c)
    return place


def carbine(p):
    """The carbine in the right fist, along the forearm (at rest the forearm hangs bent 60°, so it points at the ground
    ahead - the low ready; the forearm raised level aims it): a receiver on the forearm's outer side above the fist,
    the barrel forward, a stock back along the forearm, a magazine hanging from the receiver. Pieces: ``receiver`` (the
    receiver, the stock and the grip, aluminium), ``barrel`` (steel), ``magazine`` (aluminium)."""
    place = forearm_frame(p, -1)
    L, rh, w, d = p["carbine_length"], p["receiver_height"], p["receiver_width"], p["arm_diameter"]
    x = d * 0.55 + rh * 0.35                                              # the receiver's centre line off the forearm
    rec_len = L * 0.42
    z = L * 0.08                                                          # the receiver's centre ahead of the fist (-z)
    receiver = sk.rounded_box(rh, w, rec_len, (x, 0, -z), fillet=12)
    stock = sk.rounded_box(rh * 0.75, w * 0.8, L * 0.28, (x - rh * 0.05, 0, -z + rec_len / 2 + L * 0.13), fillet=10)
    grip = sk.rounded_box(d * 0.9, w * 0.7, d * 0.5, (d * 0.2, 0, 0), fillet=8)
    zb = -z - rec_len / 2
    barrel = sk.rod((x + rh * 0.2, 0, zb + 5), (x + rh * 0.2, 0, zb - L * 0.30), p["barrel_diameter"] / 2)
    shroud = sk.rod((x + rh * 0.2, 0, zb + 5), (x + rh * 0.2, 0, zb - L * 0.14), p["barrel_diameter"] * 0.95)
    sight = sk.rounded_box(rh * 0.25, w * 0.35, rec_len * 0.5, (x + rh / 2 + rh * 0.1, 0, -z), fillet=5)
    mag = sk.rounded_box(p["magazine_height"], w * 0.9, w * 1.3, (x - rh / 2 - p["magazine_height"] / 2 + 8, 0, -z - rec_len * 0.2), fillet=8)
    return {"receiver": place(sk.union_all([receiver, stock, grip, sight])), "barrel": place(barrel.union(shroud)),
            "magazine": place(mag)}


def tower_shield(p):
    """A shield held in the left hand: a handle bar across the fist (the gripper's fingers close round it), a bracket
    from the handle to the plate's back, the plate standing vertical in the world ahead of the fist, covering the
    body from the shins (its foot ``SHIELD_FOOT`` above the ground) to above the helmet - a titanium plate with a
    rim, a boss and a vision slit. Pieces: ``shield`` (titanium), ``handle`` (the bar and the bracket, aluminium).
    The size is ``shield_*`` of the unit's specs."""
    c, u, _ = hand_frame(p, 1)
    d, t, W, H = p["arm_diameter"], p["shield_thickness"], p["shield_width"], p["shield_height"]
    grip = _add(c, (-d * 0.55, u), (d * 0.45, (1.0, 0.0, 0.0)))          # where the curled fingers close, ahead of the palm
    x = max(grip[0] + d * 0.5 + t / 2, p["torso_depth"] / 2 + p["chest_plate_thickness"] + 50 + t / 2)    # clear of the chest
    zc = SHIELD_FOOT + H / 2
    plate = sk.rounded_box(t, W, H, (x, grip[1], zc), fillet=min(40.0, t / 2 - 1))
    rim = cq.Workplane("YZ").rect(W + 16, H + 16).rect(W - 30, H - 30).extrude(t * 0.4).translate((x + t / 2, grip[1], zc))
    boss = sk.disc((x + t / 2 + 10, grip[1], zc), p["shield_boss_diameter"] / 2, 40, axis="x")
    slit = cq.Workplane("XY").box(t * 3, W * 0.45, 28).translate((x, grip[1], zc + H * 0.36))
    plate = plate.cut(slit)
    handle = sk.rod((grip[0], grip[1] - d * 0.6, grip[2]), (grip[0], grip[1] + d * 0.6, grip[2]), 16)
    bracket = sk.rounded_box(x - grip[0], 40, 60, ((x + grip[0]) / 2, grip[1], grip[2]), fillet=6)
    for sy in (-1, 1):                                                    # the handle's cheeks, on to the plate
        bracket = bracket.union(sk.rounded_box(x - grip[0], 14, 70, ((x + grip[0]) / 2, grip[1] + sy * d * 0.6, grip[2]), fillet=4))
    return {"shield": sk.union_all([plate, rim, boss]), "handle": handle.union(bracket)}


SHIELD_FOOT = 220.0                                                       # mm: the shields' foot above the ground


combat_shield = tower_shield


def chainsaw(p):
    """The chainsaw in the left fist, along the forearm like the carbine (the bar points at the ground ahead at rest):
    the motor body on the fist with its rear handle, the bar forward with a chain of teeth round its edge, a front hoop
    handle. Pieces: ``saw_body`` (aluminium), ``saw_bar`` (steel)."""
    place = forearm_frame(p, 1)
    d, L, W, t = p["arm_diameter"], p["saw_bar_length"], p["saw_bar_width"], p["saw_bar_thickness"]
    bl, bh = p["saw_body_length"], p["saw_body_height"]
    x = d * 0.5 + bh * 0.3                                                # the body's centre line off the forearm
    body = sk.rounded_box(bh, bh * 0.8, bl, (x, 0, -bl * 0.35), fillet=14)
    grip = sk.rounded_box(d * 0.9, bh * 0.5, d * 0.5, (d * 0.2, 0, 0), fillet=8)
    rear = sk.rounded_box(bh * 0.35, bh * 0.3, bl * 0.5, (x - bh * 0.1, 0, bl * 0.25), fillet=8)
    hoop = sk.rod((x - bh * 0.45, -bh * 0.45, -bl * 0.75), (x + bh * 0.6, -bh * 0.45, -bl * 0.75), 12).union(
        sk.rod((x - bh * 0.45, -bh * 0.45, -bl * 0.75), (x - bh * 0.45, 0, -bl * 0.75), 12)).union(
        sk.rod((x + bh * 0.6, -bh * 0.45, -bl * 0.75), (x + bh * 0.6, 0, -bl * 0.75), 12))
    zb0 = -bl * 0.35 - bl / 2 + 10                                        # the bar leaves the body's front
    bar = sk.rounded_box(W, t, L, (x, 0, zb0 - L / 2), fillet=min(W / 2 - 1, 40.0))
    teeth = []
    n = int(p["saw_teeth"])
    for k in range(n):                                                    # along the outer (+x) edge and round the tip
        zt = zb0 - L * (k + 0.5) / n
        teeth.append(cq.Workplane("XY").box(14, t * 1.6, L / n * 0.45).translate((x + W / 2 + 4, 0, zt)))
    for k in range(3):
        teeth.append(cq.Workplane("XY").box(12, t * 1.6, 12).translate((x + W / 2 - 20 - 30 * k, 0, zb0 - L - 5)))
    chain = cq.Workplane("XY").box(10, t * 1.3, L).translate((x + W / 2 + 1, 0, zb0 - L / 2))
    return {"saw_body": place(sk.union_all([body, grip, rear, hoop])), "saw_bar": place(sk.union_all([bar, chain] + teeth))}


def banner(p):
    """The imperial banner in the left fist: a pole standing vertical through the fist (its butt by the knee), the
    finial - a gear with a skull in it - and under it, on a bracket, the squad's uplink: a large flat phased-array
    antenna panel (a Starlink-class dish) facing forward, tipped back ``antenna_tilt_deg``. Pieces: ``banner_pole``
    (the pole and the bracket, aluminium), ``antenna`` (the panel, aluminium casing), ``emblem`` (the gear and the
    skull, steel)."""
    c, _, _ = hand_frame(p, 1)
    L, pd = p["pole_length"], p["pole_diameter"]
    bottom, top = (c[0], c[1], c[2] - L * 0.3), (c[0], c[1], c[2] + L * 0.7)
    pole = sk.rod(bottom, top, pd / 2).union(sk.ball(bottom, pd * 0.8)).union(
        sk.rod((c[0], c[1], c[2] - pd * 1.5), (c[0], c[1], c[2] + pd * 1.5), pd * 0.7))
    G, gt, n = p["gear_diameter"], p["gear_thickness"], int(p["gear_teeth"])
    gz = top[2] - G * 0.45
    W, H, t, tilt = p["antenna_width"], p["antenna_height"], p["antenna_thickness"], p["antenna_tilt_deg"]
    za = gz - G * 0.5 - H * 0.55                                          # the panel's centre under the gear
    xa = top[0] + pd * 0.5 + t / 2 + 30
    panel = sk.rounded_box(t, W, H, (0, 0, 0), fillet=min(t / 2 - 1, 12.0)).rotate((0, 0, 0), (0, 1, 0), -tilt).translate((xa, top[1], za))
    face = sk.rounded_box(6, W * 0.92, H * 0.88, (0, 0, 0), fillet=5).rotate((0, 0, 0), (0, 1, 0), -tilt).translate(
        (xa + (t / 2 + 2) * math.cos(math.radians(tilt)), top[1], za + (t / 2 + 2) * math.sin(math.radians(tilt))))
    bracket = sk.rounded_box(xa - top[0] + pd, pd * 1.6, pd * 1.4, ((top[0] + xa) / 2, top[1], za + H * 0.25), fillet=6).union(
        sk.rounded_box(xa - top[0] + pd, pd * 1.6, pd * 1.4, ((top[0] + xa) / 2, top[1], za - H * 0.25), fillet=6))
    gear = sk.disc((top[0], top[1], gz), G * 0.42, gt, axis="x").cut(sk.disc((top[0], top[1], gz), G * 0.33, gt + 2, axis="x"))
    for k in range(n):                                                    # the gear faces forward, like the panel
        a = 2 * math.pi * k / n
        tooth = cq.Workplane("XY").box(gt, G * 0.1, G * 0.09).translate((0, G * 0.45, 0)).rotate((0, 0, 0), (1, 0, 0), math.degrees(a)) \
            .translate((top[0], top[1], gz))
        gear = gear.union(tooth)
    sd = p["skull_diameter"]
    skull = sk.ball((top[0], top[1], gz + sd * 0.08), sd / 2).union(sk.rounded_box(sd * 0.55, sd * 0.7, sd * 0.42, (top[0], top[1], gz - sd * 0.33), fillet=10))
    for sy in (-1, 1):
        skull = skull.cut(sk.ball((top[0] + sd * 0.4, top[1] + sy * sd * 0.2, gz + sd * 0.12), sd * 0.16))
    skull = skull.cut(cq.Workplane("XY").box(sd * 0.3, sd * 0.12, sd * 0.2).translate((top[0] + sd * 0.45, top[1], gz - sd * 0.1)))
    spokes = sk.rod((top[0], top[1], gz - G * 0.42), (top[0], top[1], gz + G * 0.42), pd * 0.3)
    emblem = sk.union_all([gear, skull, spokes])
    return {"banner_pole": pole.union(bracket), "antenna": panel.union(face), "emblem": emblem}


def crest(p):
    """The feathered crest along the helmet's centre line: a comb on the helmet, a row of feathers standing from it,
    each a thin rounded fin leaning back. Returns one solid (the comb and the feathers)."""
    z0, Wt, Dt, Ht, _ = _torso_frame(p)
    zh = z0 + Ht + p["neck_length"] + p["head_height"]
    Lc, Hc, Wc, n = p["crest_length"], p["crest_height"], p["crest_width"], int(p["crest_feathers"])
    comb = sk.rounded_box(Lc, Wc * 0.5, 24, (-Lc * 0.1, 0, zh + 6), fillet=6)
    feathers = [comb]
    for k in range(n):
        x = -Lc * 0.1 - Lc / 2 + Lc * (k + 0.5) / n
        h = Hc * (0.75 + 0.25 * math.sin(math.pi * (k + 0.5) / n))
        fin = sk.rounded_box(Lc / n * 1.15, Wc, h, (x, 0, zh + 12 + h / 2), fillet=min(Wc / 2 - 1, 14.0))
        feathers.append(fin.rotate((x, 0, zh + 12), (x, 1, zh + 12), -18.0))
    return sk.union_all(feathers)


def maul(p):
    """The power maul in the right fist: a shaft through the fist, the steel head crosswise at the top, a pommel.
    Pieces: ``shaft`` (aluminium), ``head`` (steel)."""
    c, _, _ = hand_frame(p, -1)
    L, hd, hl, sd = p["maul_length"], p["maul_head_diameter"], p["maul_head_length"], p["maul_shaft_diameter"]
    bottom, top = (c[0], c[1], c[2] - L * 0.3), (c[0], c[1], c[2] + L * 0.7)
    shaft = sk.rod(bottom, top, sd / 2).union(sk.ball(bottom, sd * 0.8)).union(sk.rod((c[0], c[1], c[2] - sd * 1.5), (c[0], c[1], c[2] + sd * 1.5), sd * 0.7))
    head = sk.disc((top[0], top[1], top[2] - hd * 0.1), hd / 2, hl, axis="x")
    head = head.union(sk.disc((top[0] + hl / 2, top[1], top[2] - hd * 0.1), hd * 0.42, 40, axis="x"))
    head = head.union(sk.disc((top[0] - hl / 2, top[1], top[2] - hd * 0.1), hd * 0.42, 40, axis="x"))
    return {"shaft": shaft, "head": head}


def rotary_cannon(p):
    """The rotary cannon on the right forearm: a mount block on the forearm, the barrel cluster forward along it,
    breech and muzzle rings, the feed tube from the backpack. Pieces: ``cannon`` (steel), ``mount`` (aluminium),
    ``feed`` (aluminium)."""
    c, u, _ = hand_frame(p, -1)
    L, D, n, bd = p["cannon_length"], p["cannon_diameter"], int(p["cannon_barrels"]), p["cannon_barrel_diameter"]
    d = p["arm_diameter"]
    mid = _add(c, (-p["forearm_length"] * 0.55, u))
    # the cannon's axis: along the forearm, offset outward (-y, the right side) and down from it
    axis0 = (mid[0], mid[1] - d * 0.9, mid[2] - d * 0.25)
    breech = _add(axis0, (-L * 0.25, u))
    muzzle = _add(axis0, (L * 0.75, u))
    pieces = [sk.rod(breech, muzzle, D * 0.18)]
    v = (0.0, 1.0, 0.0)                                                  # a frame perpendicular to u for the barrels' circle
    wv = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
    r = D / 2 - bd / 2
    for k in range(n):
        a = 2 * math.pi * k / n
        off = tuple(r * (math.cos(a) * v[i] + math.sin(a) * wv[i]) for i in range(3))
        pieces.append(sk.rod(_add(breech, (0.3, u), (1.0, off)), _add(muzzle, (1.0, off)), bd / 2))
    rings = [sk.rod(_add(axis0, (s, u)), _add(axis0, (s + 30, u)), D / 2 + 6) for s in (-L * 0.25, -L * 0.07, L * 0.42, L * 0.70)]
    cannon = sk.union_all(pieces + rings)
    mount = sk.rounded_box(d * 1.6, d * 2.2, d * 1.5, (mid[0], mid[1] - d * 0.45, mid[2] - d * 0.1), fillet=12)
    z0, Wt, Dt, Ht, _ = _torso_frame(p)
    pack_bottom = (-Dt / 2 - p["backpack_depth"] / 2, -p["backpack_width"] * 0.3, z0 + Ht * 0.5 - p["backpack_height"] / 2 + 20)
    feed = sk.polyline_limb([pack_bottom, (pack_bottom[0] - 60, pack_bottom[1] - 120, pack_bottom[2] - 140),
                             (breech[0] - 60, breech[1] - 40, breech[2] + 60), breech], [p["feed_diameter"] / 2, p["feed_diameter"] / 2])
    return {"cannon": cannon, "mount": mount, "feed": feed}


# ---- the skeleton: bones and mechanical joint modules, the armour goes on top -----------------------------------
def joint_module(c, p, axis="y", scale=1.0):
    """A rotary joint module: a cylinder along ``axis`` at ``c`` (the actuator's housing) with a flange ring and a
    pin boss on each face - what stands where a human joint is."""
    r, L = p["joint_module_diameter"] / 2 * scale, p["joint_module_length"] * scale
    body = sk.disc(c, r, L, axis=axis)
    flange = sk.disc(c, r * 1.08, L * 0.22, axis=axis)
    u = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    out = body.union(flange)
    for sgn in (-1, 1):
        face = _add(c, (sgn * L / 2, u))
        out = out.union(sk.disc(face, r * 0.45, L * 0.16, axis=axis)).union(sk.disc(face, r * 0.2, L * 0.4, axis=axis))
    return out


def spine_x(p, u):
    """The spine's x (build pose, mm) at fraction ``u`` of the torso height: set back in the chest, the top coming
    forward under the neck, and its curve - a human's S (lumbar forward, thoracic back) or, ``spine_bow``, a
    gorilla's single backward bow from the sacrum to the shoulders (no lumbar lordosis: the spine of an animal that
    carries its trunk pitched on four and only sometimes stands)."""
    back = -p["spine_set_back"] * p["torso_depth"]
    forward = 0.8 * sk_smooth((u - 0.55) / 0.45) if u > 0.55 else 0.0
    bow = p.get("spine_bow", 0.0)
    curve = (1 - bow) * p["spine_curve"] * math.sin(2 * math.pi * u) - bow * p["spine_curve"] * 1.2 * math.sin(math.pi * u)
    return back * (1 - forward) + curve


def sk_smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def _rib(p, i_pts, r):
    return sk.polyline_limb(i_pts, [r])


def skeleton_trunk(p):
    """{piece: solid} of the trunk's skeleton in the build pose: the pelvis (a sacrum block between two iliac wings,
    the hip joint modules under them, a pubic bar in front), the spine (``spine_segments`` vertebral bodies with
    discs, spinous and transverse processes, on the S curve ``spine_x``), the ribcage (``rib_count`` rib pairs from
    the upper vertebrae round to the sternum, sloping down to the front), the shoulder girdle (clavicles from the
    sternum's top to the shoulder joint modules, scapulae behind the upper ribs), the neck and the helmet-head."""
    hh = Humanoid.hip_height(p)
    Wp, Dp, Hp = p["pelvis_width"], p["pelvis_depth"], p["pelvis_height"]
    z0, Wt, Dt, Ht, zs = _torso_frame(p)
    N, dv = int(p["spine_segments"]), p["vertebra_diameter"]
    xs0 = spine_x(p, 0.0)
    # the pelvis
    sacrum = sk.rounded_box(Dp * 0.55, Wp * 0.36, Hp * 1.1, (xs0 + 10, 0, hh + Hp * 0.1), fillet=14)
    pelvis = sacrum
    hy = p["hip_spacing"] / 2
    for sy in (-1, 1):
        wing = (sk.rounded_box(Dp * 0.5, 16.0, p["iliac_wing_height"], (0, 0, 0), fillet=7)
                .rotate((0, 0, 0), (1, 0, 0), -sy * 22).rotate((0, 0, 0), (0, 0, 1), sy * 18)
                .translate((xs0 + Dp * 0.1, sy * (Wp / 2 - 30), hh + Hp / 2 + p["iliac_wing_height"] * 0.42)))
        pelvis = pelvis.union(wing)
        pelvis = pelvis.union(sk.rod((xs0 + 10, 0, hh + Hp * 0.5), (xs0 + Dp * 0.1, sy * (Wp / 2 - 30), hh + Hp / 2 + p["iliac_wing_height"] * 0.42), 20))
        pelvis = pelvis.union(joint_module((0, sy * hy, hh), p, "y"))                       # the hip module
        pelvis = pelvis.union(sk.rod((xs0 + 10, sy * Wp * 0.15, hh + Hp * 0.3), (0, sy * hy, hh), 22))   # the ilium down to the hip
        pelvis = pelvis.union(sk.rod((0, sy * hy, hh), (Dp * 0.32, sy * 30, hh - Hp * 0.25), 18))          # to the pubis
    pelvis = pelvis.union(sk.rounded_box(28, 90, 36, (Dp * 0.32, 0, hh - Hp * 0.25), fillet=8))              # the pubic bar
    # the spine
    seg = Ht / N
    spine = None
    for i in range(N):
        u = (i + 0.5) / N
        x, z = spine_x(p, u), z0 - 12 + (i + 0.5) * seg
        d_i = dv * (1.12 - 0.3 * u)
        body = sk.disc((x, 0, z), d_i / 2, seg * 0.66, axis="z")
        disc = sk.disc((x, 0, z + seg * 0.5), d_i * 0.42, seg * 0.4, axis="z")
        spinous = sk.rounded_box(dv * 0.5, 18, seg * 0.5, (x - d_i / 2 - dv * 0.2, 0, z - seg * 0.1), fillet=5)
        transverse = sk.rod((x - d_i * 0.25, -d_i * 0.75, z), (x - d_i * 0.25, d_i * 0.75, z), 9)
        v = body.union(disc).union(spinous).union(transverse)
        spine = v if spine is None else spine.union(v)
    # the ribs and the sternum
    nr, rr = int(p["rib_count"]), p["rib_section"] / 2
    x_front = Dt / 2 - rr - 6
    ribs, fronts = [], []
    for k in range(nr):
        i = N - nr + k                                                     # rib k on vertebra i (k = 0 the lowest)
        u = (i + 0.5) / N
        x, z = spine_x(p, u), z0 - 12 + (i + 0.5) * seg
        d_i = dv * (1.12 - 0.3 * u)
        wk = Wt * (0.78 + 0.2 * math.sin(math.pi * (k + 0.5) / nr)) / 2 - rr      # the rib's half width
        drop = Wt * 0.11 + seg * 0.25
        D = x_front - x
        for sy in (-1, 1):
            pts = [(x - d_i * 0.1, sy * d_i * 0.45, z), (x - d_i * 0.3, sy * wk * 0.6, z - drop * 0.12),
                   (x + D * 0.18, sy * wk * 0.96, z - drop * 0.35), (x + D * 0.5, sy * wk, z - drop * 0.62),
                   (x + D * 0.82, sy * wk * 0.72, z - drop * 0.88), (x_front, sy * p["sternum_width"] * 0.3, z - drop)]
            ribs.append(_rib(p, pts, rr))
        fronts.append(z - drop)
    sternum = sk.rounded_box(rr * 2.2, p["sternum_width"], fronts[-1] - fronts[0] + rr * 4 + 30,
                             (x_front, 0, (fronts[-1] + fronts[0]) / 2 + 15), fillet=9)
    ribcage = sk.union_all(ribs).union(sternum)
    # the shoulder girdle: clavicles to the shoulder modules, scapulae behind the upper ribs, the neck and the head
    d = p["arm_diameter"]
    girdle = None
    for sy in (-1, 1):
        cy = sy * (Wt / 2 + 10)
        module = joint_module((0, cy, zs), p, "y", scale=d * 1.3 / p["joint_module_diameter"])
        clav = sk.rod((x_front - 4, sy * p["sternum_width"] * 0.3, fronts[-1] + rr * 2 + 24), (0, cy - sy * d * 0.3, zs + d * 0.15),
                      p["clavicle_diameter"] / 2)
        top_x = spine_x(p, (N - 1.5) / N)
        scap = (sk.rounded_box(12, p["scapula_width"], Ht * 0.3, (0, 0, 0), fillet=8).rotate((0, 0, 0), (0, 0, 1), sy * 25)
                .translate((top_x - dv * 0.55 - 20, sy * (Wt * 0.26), z0 + Ht * 0.7)))
        acromion = sk.rod((top_x - dv * 0.55 - 20, sy * (Wt * 0.26), z0 + Ht * 0.7), (0, cy - sy * d * 0.2, zs), 13)
        part = module.union(clav).union(scap).union(acromion)
        girdle = part if girdle is None else girdle.union(part)
    zn = z0 + Ht
    x_top = spine_x(p, 1.0)
    neck = sk.rod((x_top, 0, zn - seg * 0.4), (0, 0, zn + p["neck_length"] + 5), p["head_width"] * 0.2)
    for j in range(3):                                                     # the cervical vertebrae up the neck
        f = (j + 0.5) / 3
        neck = neck.union(sk.disc((x_top * (1 - f), 0, zn - seg * 0.2 + f * (p["neck_length"] + seg * 0.2)), p["head_width"] * 0.24, 10, axis="z"))
    Wh, Dh, Hh = p["head_width"], p["head_depth"], p["head_height"]
    zh = zn + p["neck_length"] + Hh / 2
    head = cq.Workplane("XY").box(Dh, Wh, Hh).edges().fillet(min(35.0, Wh / 2 - 1, Hh / 2 - 1)).translate((0, 0, zh))
    eyes = (cq.Workplane("YZ").pushPoints([(-Wh * 0.2, zh + Hh * 0.08), (Wh * 0.2, zh + Hh * 0.08)])
            .circle(p["eye_diameter"] / 2).extrude(30).translate((Dh / 2 - 15, 0, 0)))
    return {"pelvis": pelvis, "spine": spine, "ribcage": ribcage, "girdle": girdle, "neck": neck, "head": head.cut(eyes)}


def galea(p):
    """The Roman helmet over the square head: a brow band, two cheek guards hinged beside the face, a nape guard
    sloping off the back, and a comb along the top for the plume (the Centurion's feathered crest rides on it)."""
    z0, Wt, Dt, Ht, _ = _torso_frame(p)
    Wh, Dh, Hh = p["head_width"], p["head_depth"], p["head_height"]
    zh = z0 + Ht + p["neck_length"] + Hh / 2
    t = p["helmet_thickness"]
    brow = sk.rounded_box(Dh + 2 * t, Wh + 2 * t, Hh * 0.22, (0, 0, zh + Hh * 0.28), fillet=12).cut(
        sk.rounded_box(Dh - 4, Wh - 4, Hh * 0.3, (0, 0, zh + Hh * 0.28), fillet=10))
    cap = sk.rounded_box(Dh * 0.8, Wh * 0.8, t, (0, 0, zh + Hh / 2 + t / 2 - 1), fillet=10)
    comb = sk.rounded_box(Dh * 0.7, 18, Hh * 0.22, (-Dh * 0.05, 0, zh + Hh / 2 + t + Hh * 0.1), fillet=6)
    parts = brow.union(cap).union(comb)
    for sy in (-1, 1):                                                    # the cheek guards, hinged at the brow band
        cheek = (sk.rounded_box(Dh * 0.5, t, Hh * 0.6, (0, 0, 0), fillet=8).rotate((0, 0, 0), (0, 0, 1), sy * 8)
                 .translate((Dh * 0.12, sy * (Wh / 2 + t / 2 + 2), zh - Hh * 0.12)))
        hinge = sk.rod((Dh * 0.12, sy * (Wh / 2 + 1), zh + Hh * 0.2), (Dh * 0.12, sy * (Wh / 2 + t + 6), zh + Hh * 0.2), 8)
        parts = parts.union(cheek).union(hinge)
    nape = (sk.rounded_box(Dh * 0.4, Wh * 0.9, t, (0, 0, 0), fillet=8).rotate((0, 0, 0), (0, 1, 0), 35)
            .translate((-Dh / 2 - Dh * 0.12, 0, zh - Hh * 0.05)))
    nape = nape.union(sk.rounded_box(t, Wh * 0.7, Hh * 0.3, (-Dh / 2 - t / 2 - 1, 0, zh + Hh * 0.08), fillet=6))
    return parts.union(nape)


def piston(a, b, barrel=44.0, rod=22.0):
    """A linear actuator from ``a`` (the barrel's eye) to ``b`` (the rod's eye): the barrel over 55 % of the length,
    the rod through, an eye ball at each end."""
    av, bv = cq.Vector(*a), cq.Vector(*b)
    m = av + (bv - av) * 0.55
    return (sk.rod(a, tuple(m.toTuple()), barrel / 2).union(sk.rod(a, b, rod / 2))
            .union(sk.ball(a, barrel * 0.6)).union(sk.ball(b, rod * 1.1)))


def actuators(p):
    """{piece: solid} of the linear actuators: the pistons that work the joints beside their rotary modules (the
    modules give the joint its bearing and brake, the pistons the heavy lifting), laid across the skeleton the way
    muscles would go - the hip struts from the iliac wings to the femurs, the knee struts down the thighs to the
    tibiae's plateaus, the ankle struts from the tibiae to the heels, the shoulder struts from the scapulae to the
    upper arms, the elbow struts to the forearms, the spine struts from the pelvis to the ribcage's front."""
    hh = Humanoid.hip_height(p)
    z0, Wt, Dt, Ht, zs = _torso_frame(p)
    d, L1, L2 = p["arm_diameter"], p["upper_arm_length"], p["forearm_length"]
    a = p["knee_bend_deg"]
    sa, ca = math.sin(math.radians(a)), math.cos(math.radians(a))
    Lt, Ls = p["thigh_length"], p["shin_length"]
    Dp, Hp = p["pelvis_depth"], p["pelvis_height"]
    groups = {k: [] for k in ("hip_struts", "knee_struts", "ankle_struts", "shoulder_struts", "elbow_struts", "spine_struts")}
    xs0 = spine_x(p, 0.0)
    x_front = Dt / 2 - p["rib_section"] / 2 - 6
    for sy in (-1, 1):
        y = sy * p["hip_spacing"] / 2
        kx, kz = Lt * sa, hh - Lt * ca
        groups["hip_struts"].append(piston((xs0 + Dp * 0.35, y + sy * 20, hh + Hp * 0.55), (Lt * 0.5 * sa + p["femur_diameter"] * 0.9, y, hh - Lt * 0.5 * ca)))
        groups["knee_struts"].append(piston((Lt * 0.35 * sa + p["femur_diameter"] * 0.9, y, hh - Lt * 0.35 * ca), (kx + p["tibia_diameter"] * 0.9, y, kz - Ls * 0.22)))
        groups["ankle_struts"].append(piston((kx - p["tibia_diameter"] * 0.9, y, kz - Ls * 0.35), (-p["foot_length"] * 0.2, y, p["ankle_height"] * 0.6), barrel=36, rod=18))
        cy = sy * (Wt / 2 + 10 + d * 0.5)
        top_x = spine_x(p, (int(p["spine_segments"]) - 1.5) / int(p["spine_segments"]))
        groups["shoulder_struts"].append(piston((top_x - p["vertebra_diameter"] * 0.55 - 20, sy * Wt * 0.26, z0 + Ht * 0.62), (-d * 0.55, cy, zs - L1 * 0.4), barrel=36, rod=18))
        c, u, _ = hand_frame(p, sy)
        fore_mid = _add(c, (-L2 * 0.5, u))
        groups["elbow_struts"].append(piston((d * 0.6, cy, zs - L1 * 0.55), (fore_mid[0] + d * 0.1, fore_mid[1], fore_mid[2] + d * 0.5), barrel=36, rod=18))
        groups["spine_struts"].append(piston((Dp * 0.3, sy * 60, hh + Hp * 0.5), (x_front - 30, sy * 90, z0 + Ht * 0.42), barrel=40, rod=20))
    return {k: sk.union_all(v) for k, v in groups.items()}


def gripper(p):
    """A gripping hand at the wrist (the forearm's end at the origin, the forearm up +z, its outer side +x): a palm
    block, four fingers of three phalanges on knuckle modules along the palm's lower edge, curled in (round a grip:
    the carbine's, the pole, the maul's shaft), a thumb pad opposite. ``finger_length`` is the whole finger."""
    d, Lf, df = p["arm_diameter"], p["finger_length"], p["finger_diameter"]
    palm = sk.rounded_box(d * 0.95, d * 0.5, d * 0.9, (0, 0, -d * 0.45), fillet=8)
    wrist = sk.disc((0, 0, 0), d * 0.4, d * 0.3, axis="z")
    out = palm.union(wrist)
    seg = Lf / 3
    for k in range(4):
        y = (k - 1.5) * (d * 0.5 / 3.4)
        pts, ang, pos = [(d * 0.47, y, -d * 0.86)], -20.0, (d * 0.47, y, -d * 0.86)
        for j in range(3):                                                # each phalanx curls 55° further
            ang += 55.0
            pos = (pos[0] + seg * math.cos(math.radians(ang)) * -1, y, pos[2] - seg * math.sin(math.radians(ang)))
            pts.append(pos)
        finger = sk.polyline_limb(pts, [df / 2, df * 0.45, df * 0.4])
        out = out.union(finger).union(sk.disc((d * 0.47, y, -d * 0.86), df * 0.55, df * 0.9, axis="y"))   # the knuckle module
    thumb = sk.rounded_box(d * 0.3, d * 0.5, d * 0.35, (-d * 0.55, 0, -d * 0.55), fillet=6)
    return out.union(thumb)


def femur(p):
    """The thigh bone, the hip module at the origin, the knee at (0, 0, -thigh_length): a neck from the hip module's
    side boss down into the shaft, the greater trochanter, the shaft bowing forward a little, the condyles (two
    knuckles) holding the knee module."""
    L, df, dm = p["thigh_length"], p["femur_diameter"], p["joint_module_diameter"]
    neck = sk.rod((0, 0, 0), (df * 0.2, 0, -dm * 0.75), df * 0.42)
    troch = sk.ball((df * 0.15, 0, -dm * 0.6), df * 0.55)
    shaft = sk.polyline_limb([(df * 0.15, 0, -dm * 0.6), (df * 0.5, 0, -L * 0.45), (0, 0, -L + dm * 0.55)], [df / 2, df / 2])
    condyles = sk.ball((0, df * 0.4, -L + dm * 0.5), df * 0.5).union(sk.ball((0, -df * 0.4, -L + dm * 0.5), df * 0.5))
    knee = joint_module((0, 0, -L), p, "y", scale=0.9)
    return neck.union(troch).union(shaft).union(condyles).union(knee)


def tibia(p):
    """The shin: the knee at the origin (the knee module belongs to the femur; the tibia's plateau clamps it), the
    tibia straight down to the ankle module, the fibula beside it, the ankle module at (0, 0, -shin_length)."""
    L, dt, dfb, dm = p["shin_length"], p["tibia_diameter"], p["fibula_diameter"], p["joint_module_diameter"]
    plateau = sk.rounded_box(dt * 1.3, dt * 1.6, dt * 0.6, (0, 0, -dm * 0.55), fillet=8)
    clamp = sk.rod((0, -dt * 0.8, 0), (0, dt * 0.8, 0), dm * 0.32)                     # round the knee module's pin
    clamp = clamp.union(sk.rod((0, 0, 0), (0, 0, -dm * 0.6), dt * 0.5))                 # the yoke down to the plateau
    shaft = sk.polyline_limb([(0, 0, -dm * 0.5), (0, 0, -L + dm * 0.5)], [dt / 2])
    fib = sk.polyline_limb([(-dt * 0.3, dt * 0.95, -dm * 0.7), (-dt * 0.35, dt * 0.9, -L + dm * 0.55)], [dfb / 2])
    fib = fib.union(sk.rod((0, 0, -dm * 0.7), (-dt * 0.3, dt * 0.95, -dm * 0.7), dfb * 0.45))
    ankle = joint_module((0, 0, -L), p, "y", scale=0.8)
    return plateau.union(clamp).union(shaft).union(fib).union(ankle)


def foot_bones(p):
    """The foot: the ankle module's roll stage (a smaller module across), a heel block behind, the sole plate (what
    the boot goes over) and a toe bar."""
    Lf, Wf, ha, t = p["foot_length"], p["foot_width"], p["ankle_height"], p["foot_thickness"]
    roll = joint_module((0, 0, -ha * 0.45), p, "x", scale=0.5)
    heel = sk.rounded_box(Lf * 0.3, Wf * 0.6, ha * 0.7, (-Lf * 0.22, 0, -ha + t / 2 + ha * 0.3), fillet=10)
    sole = sk.rounded_box(Lf, Wf, t, (Lf * 0.2, 0, -ha + t / 2), fillet=min(10.0, t / 2 - 1))
    arch = sk.rod((0, 0, -ha * 0.45), (Lf * 0.3, 0, -ha + t), Wf * 0.18)
    toes = sk.rod((Lf * 0.62, -Wf * 0.45, -ha + t), (Lf * 0.62, Wf * 0.45, -ha + t), t * 0.9)
    return roll.union(heel).union(sole).union(arch).union(toes)


class SpaceMarine(Humanoid):
    """The Marine: Talia's frame at the Marine's numbers (1.86 m, the ork's bulk on longer legs) in power armour,
    a carbine in the right fist and a combat shield on the left forearm. ``part``: as ``Humanoid``, plus 'armour' and
    'weapon'."""

    SPECS, EXTRA, NAME = su.MARINE, su.MARINE_EXTRA, "Marine"

    @staticmethod
    def weapon_fn(p):
        return {**carbine(p), **combat_shield(p)}

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        cls.parameters = cls._parameters()

    @classmethod
    def _parameters(cls):
        extra = [Parameter(k, float(cls.SPECS[k]), "deg" if k.endswith("_deg") else ("" if k == "cannon_barrels" else "mm"),
                           min=0.0, description=d) for k, d in cls.EXTRA.items()]
        base = sk.with_defaults(Humanoid.parameters, **{k: v for k, v in cls.SPECS.items() if k not in cls.EXTRA})
        return [Parameter("part", "humanoid", choices=("humanoid", "skeleton", "actuators", "trunk", "thigh", "shin", "foot", "armour", "weapon"),
                          description="what to build")] + [q for q in base if q.name != "part"] + extra

    def armour_parts(self, p):
        """{piece: solid} of the armour, in the build pose: the shoulder guards (three layered plates a side), the
        lorica (girth hoops round the ribs with their clasps, the breast band on top: lorica segmentata, so the
        trunk can bend), the chest bay (the electronics' casing
        inside the ribcage, bolted to the thoracic spine, aluminium), the belly bay (the power electronics' casing in
        front of the lumbar spine, under the lowest hoop, aluminium), the arm lames (three bands round each upper arm and
        forearm, hinged like the lorica's hoops), the backpack (with the exhausts), the belt, the greaves (four bands down each shin on a hinge spine, with the knee domes),
        the boots. All
        titanium but the casings (aluminium)."""
        z0, Wt, Dt, Ht, zs = _torso_frame(p)
        hh = Humanoid.hip_height(p)
        d, Hp, Wp, Dp = p["arm_diameter"], p["pelvis_height"], p["pelvis_width"], p["pelvis_depth"]
        R = p["pauldron_diameter"] / 2
        # the shoulder guards: three layered plates stepping down over each shoulder (lorica segmentata), on a post
        guards = []
        for sy in (-1, 1):
            cy = sy * (Wt / 2 + 10 + d * 0.5)
            post = sk.rod((0, cy, zs - d * 0.2), (0, cy, zs + d * 0.35 + R * 0.55), d * 0.3)
            part = post.union(sk.disc((0, cy, zs + d * 0.1), d * 0.55, d * 0.9, axis="y"))
            for k in range(3):
                plate = (sk.rounded_box(R * 1.5, R * 0.5, p["pauldron_thickness"] * 0.5, (0, 0, 0), fillet=6)
                         .rotate((0, 0, 0), (1, 0, 0), -sy * (12 + 24 * k))
                         .translate((0, cy + sy * (R * 0.2 + k * R * 0.26), zs + d * 0.35 + R * 0.62 - k * R * 0.28)))
                part = part.union(plate)
            guards.append(part)
        t = p["chest_plate_thickness"]
        # the lorica: girth hoops round the ribs, each a thin band overlapping the one below, split front and back
        # by the clasps' seam (a box frame cut hollow), the uppermost the breast band
        n = int(p["lorica_bands"])
        hb = Ht * 0.52 / n
        lorica = None
        for k in range(n):
            zb = z0 + Ht * 0.34 + (k + 0.5) * hb
            wb, db = Wt * 0.9 + 2 * t + k * 4, Dt + 2 * t + k * 3
            band = sk.rounded_box(db, wb, hb * 1.25, (0, 0, zb), fillet=min(10.0, t / 2 - 1)).cut(
                sk.rounded_box(db - 2 * t * 0.6, wb - 2 * t * 0.6, hb * 1.5, (0, 0, zb), fillet=8))
            band = band.union(sk.rounded_box(t * 0.8, 46, hb * 0.5, (db / 2 + t * 0.2, 0, zb), fillet=4))     # the clasp
            lorica = band if lorica is None else lorica.union(band)
        # the breastplate: a short plate on the top band, standing on it (the chest's crest)
        chest = sk.rounded_box(t, Wt * 0.5, Ht * 0.16, (Dt / 2 + t * 1.5 - 8, 0, z0 + Ht * 0.86), fillet=min(16.0, t / 2 - 1))
        lorica = lorica.union(chest)
        # the belly bay: the power electronics' casing in front of the lumbar spine, under the lorica's lowest hoop
        xl = spine_x(p, 0.15)
        belly = sk.rounded_box(Dp * 0.42, Wp * 0.5, Ht * 0.2, (xl + Dp * 0.38, 0, z0 + Ht * 0.17), fillet=12)
        belly = belly.union(sk.rod((xl, 0, z0 + Ht * 0.17), (xl + Dp * 0.38, 0, z0 + Ht * 0.17), 20))
        # the chest bay: the electronics' casing inside the ribcage, between the thoracic spine and the sternum
        xs = spine_x(p, 0.62)
        xb = (xs + Dt / 2) / 2
        bay = sk.rounded_box(Dt / 2 - xs - 60, Wt * 0.4, Ht * 0.26, (xb, 0, z0 + Ht * 0.6), fillet=12)
        bay = bay.union(sk.rod((xs, 0, z0 + Ht * 0.6), (xb, 0, z0 + Ht * 0.6), 20))                   # bolted to the spine
        Wb, Db, Hb = p["backpack_width"], p["backpack_depth"], p["backpack_height"]
        pack = sk.rounded_box(Db, Wb, Hb, (-Dt / 2 - Db / 2 + 8, 0, z0 + Ht * 0.5), fillet=18)
        for sy in (-1, 1):
            ex = p["exhaust_diameter"] / 2
            base = (-Dt / 2 - Db / 2 + 8, sy * Wb * 0.3, z0 + Ht * 0.5 + Hb / 2 - 10)
            pack = pack.union(sk.rod(base, (base[0], base[1], base[2] + p["exhaust_height"]), ex)) \
                .union(sk.rod((base[0], base[1], base[2] + p["exhaust_height"] - 20), (base[0], base[1], base[2] + p["exhaust_height"]), ex * 1.25))
        belt = sk.rounded_box(Dp + 36, Wp + 36, p["belt_height"], (0, 0, hh + Hp / 2 - p["belt_height"] / 2 + 10), fillet=14)
        belt = belt.union(sk.rounded_box(Dp * 0.3, Wp * 0.3, p["belt_height"] * 1.3, (Dp / 2 + 14, 0, hh + Hp / 2 - p["belt_height"] / 2 + 10), fillet=8))
        a = p["knee_bend_deg"]
        L1, L2, gt = p["thigh_length"], p["shin_length"], p["greave_thickness"]
        w = p["leg_width"] * 0.8
        greaves, boots = [], []
        for sy in (-1, 1):
            y = sy * p["hip_spacing"] / 2
            kx, kz = L1 * math.sin(math.radians(a)), hh - L1 * math.cos(math.radians(a))
            rt = p["tibia_diameter"] / 2
            plate = None
            for k in range(4):                                            # the greave: four overlapping bands down the shin
                zb = -L2 * (0.3 + 0.14 * k)
                band = (sk.disc((0, 0, zb), rt + gt + 2 + k * 2, L2 * 0.17, axis="z")
                        .cut(sk.disc((0, 0, zb), rt + 2 + k * 2 - 4, L2 * 0.2, axis="z"))
                        .union(sk.rounded_box(gt * 0.8, 36, L2 * 0.08, (rt + gt + 2 + k * 2, 0, zb), fillet=3)))   # the buckle
                plate = band if plate is None else plate.union(band)
            plate = plate.union(sk.rod((rt - 4, 0, -L2 * 0.3), (rt - 4, 0, -L2 * 0.72), 10))               # the spine the bands hinge on
            plate = plate.rotate((0, 0, 0), (0, 1, 0), a).translate((kx, y, kz))
            knee = sk.ball((kx, y, kz), p["leg_thickness"] * 0.72)
            greaves.append(plate.union(knee))
            Lf, Wf, ha = p["foot_length"], p["foot_width"], p["ankle_height"]
            boots.append(sk.rounded_box(Lf * 0.62, Wf * 1.06, ha * 1.3, (Lf * 0.25, y, ha * 0.62), fillet=16))
        # the arm lames: three bands round the upper arm, three round the forearm, each overlapping the one above
        lames = []
        for sy in (-1, 1):
            cy = sy * (Wt / 2 + 10 + d * 0.5)
            L1a, L2a, ra = p["upper_arm_length"], p["forearm_length"], d / 2
            part = None
            for k in range(3):
                zb = zs - L1a * (0.3 + 0.17 * k)
                band = (sk.disc((0, cy, zb), ra + 14 + k * 3, L1a * 0.2, axis="z").cut(sk.disc((0, cy, zb), ra - 3 + k * 3, L1a * 0.24, axis="z"))
                        .union(sk.rounded_box(14, 10, L1a * 0.1, (ra + 14 + k * 3, cy, zb), fillet=3)))
                part = band if part is None else part.union(band)
            place = forearm_frame(p, sy)
            rf = d * 0.42
            for k in range(3):
                zb = L2a * (0.32 + 0.17 * k)
                band = (sk.disc((0, 0, zb), rf + 12 + k * 3, L2a * 0.2, axis="z").cut(sk.disc((0, 0, zb), rf - 3 + k * 3, L2a * 0.24, axis="z"))
                        .union(sk.rounded_box(14, 10, L2a * 0.1, (rf + 12 + k * 3, 0, zb), fillet=3)))
                part = part.union(place(band))
            lames.append(part)
        return {"helmet": galea(p), "shoulder_guards": sk.union_all(guards), "lorica": lorica, "arm_lames": sk.union_all(lames),
                "chest_bay": bay, "belly_bay": belly, "backpack": pack, "belt": belt, "greaves": sk.union_all(greaves), "boots": sk.union_all(boots)}

    def actuator_parts(self, p):
        """{piece: solid} of the linear actuators (``actuators``), in the build pose."""
        return actuators(p)

    def weapon_parts(self, p):
        return type(self).weapon_fn(p)

    def _arm(self, p, side):
        """Talia's arm (the shoulder boss, the upper arm, the elbow, the bent forearm) ending in a gripping hand: a
        palm block and four fingers of three phalanges each, curled round what the fist holds (``gripper``)."""
        d, L1, L2 = p["arm_diameter"], p["upper_arm_length"], p["forearm_length"]
        e = math.radians(p["elbow_bend_deg"])
        shoulder = cq.Workplane("XZ").circle(d * 0.65).extrude(d * 0.45, both=True)
        upper = cq.Workplane("XY").circle(d / 2).extrude(-L1)
        elbow = cq.Workplane("XY").sphere(d * 0.48).translate((0, 0, -L1))
        fore = (cq.Workplane("XY").circle(d * 0.42).extrude(-L2)
                .rotate((0, 0, 0), (0, 1, 0), -p["elbow_bend_deg"]).translate((0, 0, -L1)))
        hx, hz = L2 * math.sin(e), -L1 - L2 * math.cos(e)
        hand = gripper(p).rotate((0, 0, 0), (0, 1, 0), -p["elbow_bend_deg"]).translate((hx, 0, hz))
        return shoulder.union(upper).union(elbow).union(fore).union(hand)

    # the skeleton in place of Talia's shells: Humanoid.build places these parts (the hip, knee and ankle frames)
    def _trunk(self, p):
        return sk.union_all(list(skeleton_trunk(p).values()))

    def _thigh(self, p):
        return femur(p)

    def _shin(self, p):
        return tibia(p)

    def _foot(self, p):
        return foot_bones(p)

    def skeleton_parts(self, p):
        """{piece: solid} of the trunk's skeleton, in the build pose (``skeleton_trunk``)."""
        return skeleton_trunk(p)

    def _visor(self, p):
        z0, Wt, Dt, Ht, _ = _torso_frame(p)
        zh = z0 + Ht + p["neck_length"] + p["head_height"] / 2
        return cq.Workplane("XY").box(40, p["head_width"] * 0.72, p["visor_height"]).translate(
            (p["head_depth"] / 2, 0, zh + p["head_height"] * 0.08))

    def build(self, p):
        if p["part"] == "armour":
            return sk.union_all(list(self.armour_parts(p).values()))
        if p["part"] == "weapon":
            return sk.union_all(list(self.weapon_parts(p).values()))
        if p["part"] == "actuators":
            return sk.union_all(list(self.actuator_parts(p).values()))
        body = super().build(dict(p, part="humanoid" if p["part"] == "skeleton" else p["part"]))
        if p["part"] != "humanoid":
            return body                                                   # a part, or the bare skeleton
        whole = sk.union_all([body] + list(self.armour_parts(p).values()) + list(self.weapon_parts(p).values()))
        return whole.cut(self._visor(p))


class SpaceMarineSaw(SpaceMarine):
    """The Marine with a chainsaw in the left fist instead of the shield (the carbine stays in the right)."""

    SPECS, EXTRA, NAME = su.MARINE_SAW, su.MARINE_SAW_EXTRA, "Marine (chainsaw)"

    @staticmethod
    def weapon_fn(p):
        return {**carbine(p), **chainsaw(p)}


class Centurion(SpaceMarine):
    """The Centurion: the Marine with the imperial banner in the left fist (a skull in a gear on the pole, the squad's
    flat antenna under it) and a feathered crest along the helmet; the carbine in the right."""

    SPECS, EXTRA, NAME = su.CENTURION, su.CENTURION_EXTRA, "Centurion"

    @staticmethod
    def weapon_fn(p):
        return {**carbine(p), **banner(p)}

    def armour_parts(self, p):
        return {**super().armour_parts(p), "crest": crest(p)}


class Breacher(SpaceMarine):
    """The Breacher: the Marine's armour with a tower shield on the left forearm and a power maul in the right fist."""

    SPECS, EXTRA, NAME = su.BREACHER, su.BREACHER_EXTRA, "Breacher"

    @staticmethod
    def weapon_fn(p):
        return {**tower_shield(p), **maul(p)}


class Heavy(SpaceMarine):
    """The Heavy: the Marine's armour on a wider frame, a rotary cannon on the right forearm fed from a bigger pack."""

    SPECS, EXTRA, NAME = su.HEAVY, su.HEAVY_EXTRA, "Heavy"
    weapon_fn = staticmethod(rotary_cannon)


SpaceMarine.parameters = SpaceMarine._parameters()
