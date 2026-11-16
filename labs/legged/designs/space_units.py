"""The space game units (CAD): bulky armoured humanoid robots on ``humanoid.Humanoid``'s frame, both hands busy —
the **Marine** (a carbine and a combat shield), the **Marine with a chainsaw** (the carbine and a chainsaw), the
**Breacher** (a tower shield and a power maul) and the **Heavy** (a rotary cannon fed from its backpack). The armour: shoulder domes (pauldrons), a breastplate, a power pack with two exhaust stacks, a waist belt,
greaves over the shins, knee domes, boots and a visor slit cut into the square helmet. Numbers in
``space_unit_specs``; mm, x forward, y left, z up, the ground at z = 0.

``part``: as ``Humanoid`` (the whole machine, the trunk, a thigh, a shin, a foot), plus ``armour`` (the plates alone)
and ``weapon`` (what the unit carries). ``armour_parts`` and ``weapon_parts`` give the pieces by name, for the
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

__all__ = ["SpaceMarine", "SpaceMarineSaw", "Breacher", "Heavy", "carbine", "tower_shield", "combat_shield", "chainsaw", "maul",
           "rotary_cannon", "forearm_frame"]


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
    """A shield strapped to the left forearm, its plate along the forearm and its face on the forearm's outer side (the
    Breacher's tower shield with the forearm hanging nearly straight stands upright facing ahead; the Marine's combat
    shield on its bent forearm faces ahead and down): a titanium plate with a rim and a boss, the strap block round the
    forearm. Pieces: ``shield`` (titanium), ``strap`` (aluminium). The size is ``shield_*`` of the unit's specs."""
    place = forearm_frame(p, 1)
    d, t, W, H = p["arm_diameter"], p["shield_thickness"], p["shield_width"], p["shield_height"]
    zm = p["forearm_length"] * 0.5                                        # the forearm's middle, up from the fist
    x = d / 2 + t / 2 + 25
    plate = sk.rounded_box(t, W, H, (x, 0, zm + H * 0.08), fillet=min(40.0, t / 2 - 1))
    rim = cq.Workplane("YZ").rect(W + 16, H + 16).rect(W - 30, H - 30).extrude(t * 0.4).translate((x + t / 2, 0, zm + H * 0.08))
    boss = sk.disc((x + t / 2 + 10, 0, zm + H * 0.08), p["shield_boss_diameter"] / 2, 40, axis="x")
    strap = sk.rounded_box(d * 1.3 + 40, d * 1.3, d * 1.1, (d * 0.35, 0, zm), fillet=10)
    return {"shield": place(sk.union_all([plate, rim, boss])), "strap": place(strap)}


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
        return [Parameter("part", "humanoid", choices=("humanoid", "trunk", "thigh", "shin", "foot", "armour", "weapon"),
                          description="what to build")] + [q for q in base if q.name != "part"] + extra

    def armour_parts(self, p):
        """{piece: solid} of the armour, in the build pose: pauldrons, breastplate, backpack (with the exhausts), belt,
        greaves (with the knee domes), boots. All titanium but the backpack's casing (aluminium)."""
        z0, Wt, Dt, Ht, zs = _torso_frame(p)
        hh = Humanoid.hip_height(p)
        d, Hp, Wp, Dp = p["arm_diameter"], p["pelvis_height"], p["pelvis_width"], p["pelvis_depth"]
        R = p["pauldron_diameter"] / 2
        pauldrons = []
        for sy in (-1, 1):
            cy = sy * (Wt / 2 + 10 + d * 0.5)
            dome = sk.ball((0, cy, zs + d * 0.35), R).cut(sk.ball((0, cy, zs + d * 0.35), R - p["pauldron_thickness"]))
            dome = dome.intersect(cq.Workplane("XY").box(2 * R + 2, 2 * R + 2, R + 2).translate((0, cy, zs + d * 0.35 + R / 2 + 1)))
            post = sk.rod((0, cy, zs - d * 0.2), (0, cy, zs + d * 0.35 + R - p["pauldron_thickness"] / 2), d * 0.3)
            pauldrons.append(dome.union(post).union(sk.disc((0, cy, zs + d * 0.1), d * 0.55, d * 0.9, axis="y")))
        t = p["chest_plate_thickness"]
        chest = sk.rounded_box(t, Wt * 0.78, Ht * 0.52, (Dt / 2 + t / 2 - 8, 0, z0 + Ht * 0.6), fillet=min(16.0, t / 2 - 1))
        chest = chest.union(sk.rounded_box(t * 0.6, Wt * 0.3, Ht * 0.18, (Dt / 2 + t + 2, 0, z0 + Ht * 0.66), fillet=8))
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
            plate = (sk.rounded_box(gt, p["leg_thickness"] * 1.25, L2 * 0.72, (w / 2 + gt / 2 - 6, 0, -L2 * 0.52), fillet=10)
                     .rotate((0, 0, 0), (0, 1, 0), a).translate((kx, y, kz)))
            knee = sk.ball((kx, y, kz), p["leg_thickness"] * 0.72)
            greaves.append(plate.union(knee))
            Lf, Wf, ha = p["foot_length"], p["foot_width"], p["ankle_height"]
            boots.append(sk.rounded_box(Lf * 0.62, Wf * 1.06, ha * 1.3, (Lf * 0.25, y, ha * 0.62), fillet=16))
        return {"pauldrons": sk.union_all(pauldrons), "chest_plate": chest, "backpack": pack, "belt": belt,
                "greaves": sk.union_all(greaves), "boots": sk.union_all(boots)}

    def weapon_parts(self, p):
        return type(self).weapon_fn(p)

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
        body = super().build(dict(p, part=p["part"]))
        if p["part"] != "humanoid":
            return body
        whole = sk.union_all([body] + list(self.armour_parts(p).values()) + list(self.weapon_parts(p).values()))
        return whole.cut(self._visor(p))


class SpaceMarineSaw(SpaceMarine):
    """The Marine with a chainsaw in the left fist instead of the shield (the carbine stays in the right)."""

    SPECS, EXTRA, NAME = su.MARINE_SAW, su.MARINE_SAW_EXTRA, "Marine (chainsaw)"

    @staticmethod
    def weapon_fn(p):
        return {**carbine(p), **chainsaw(p)}


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
