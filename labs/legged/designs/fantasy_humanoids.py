"""Fantasy humanoids (CAD): the hand frame of ``humanoid.Humanoid``'s arms, a pincer the ghoul carries on each hand
(``monsters.Ghoul``), and the ork — a bulky humanoid with tusks holding a broom in its right hand and a trash-bin
lid strapped to its left forearm. Numbers in ``fantasy_humanoid_specs``; mm, x forward, y left, z up."""
import math

import cadquery as cq
from vegeta.dedalus import Parameter

import fantasy_humanoid_specs as fh
import shape_kit as sk
from humanoid import Humanoid

__all__ = ["hand_frame", "pincer", "Ork", "broom", "bin_lid"]


def hand_frame(p, sy):
    """(hand centre, forearm direction u, lateral v) of the ``sy`` arm (+1 left) in ``Humanoid.build``'s pose [mm]:
    the shoulder at (0, ±(torso_width/2 + 10 + arm_diameter/2), shoulder height), the forearm bent forward by
    ``elbow_bend_deg``."""
    hh = Humanoid.hip_height(p)
    zs = hh + p["pelvis_height"] / 2 + p["torso_height"] - p["arm_diameter"] * 0.6
    y = sy * (p["torso_width"] / 2 + 10 + p["arm_diameter"] * 0.5)
    e = math.radians(p["elbow_bend_deg"])
    u = (math.sin(e), 0.0, -math.cos(e))
    c = (p["forearm_length"] * math.sin(e), y, zs - p["upper_arm_length"] - p["forearm_length"] * math.cos(e))
    return c, u, (0.0, 1.0, 0.0)


def _add(a, *terms):
    return tuple(a[k] + sum(s * v[k] for s, v in terms) for k in range(3))


def pincer(p, sy, length, open_deg, radius=7.0):
    """Two hooked jaws past the hand, opening sideways by ``open_deg`` each, their tips curling back in, on a
    knuckle block: the ghoul's pincer (a lobster's claw on a human hand)."""
    c, u, v = hand_frame(p, sy)
    d = p["arm_diameter"]
    base = _add(c, (d * 0.6, u))
    knuckle = sk.capsule(_add(base, (-d * 0.2, v)), _add(base, (d * 0.2, v)), radius * 1.6)
    a, hook = math.radians(open_deg + 10), math.radians(35)
    jaws = []
    for j in (-1, 1):
        b = _add(base, (j * d * 0.18, v))
        m = _add(b, (length * 0.6 * math.cos(a), u), (j * length * 0.6 * math.sin(a), v))
        t = _add(m, (length * 0.45 * math.cos(hook), u), (-j * length * 0.45 * math.sin(hook), v))
        jaws.append(sk.polyline_limb([b, m, t], [radius, radius * 0.7]))
        for k in (1, 2, 3):                                               # teeth on the inner edge
            q = _add(b, (length * 0.2 * k * math.cos(a), u), (j * length * 0.2 * k * math.sin(a), v))
            jaws.append(sk.capsule(q, _add(q, (-j * radius * 1.6, v)), radius * 0.4))
    return sk.union_all([knuckle] + jaws)


def broom(p):
    """The broom in the ork's right hand: an ash pole through the fist, slanting forward ``broom_tilt_deg``, its
    head (a stock with a block of straw bristles) on the ground. Returns (pole + stock, bristles)."""
    c, _, _ = hand_frame(p, -1)
    tilt = math.radians(p["broom_tilt_deg"])
    head_z = p["bristle_length"] + p["stock_height"] / 2
    run = (c[2] - head_z) * math.tan(tilt)
    bottom = (c[0] + run, c[1], head_z)
    top = (c[0] - p["pole_above_hand"] * math.sin(tilt), c[1], c[2] + p["pole_above_hand"] * math.cos(tilt))
    pole = sk.rod(bottom, top, p["pole_diameter"] / 2).union(sk.ball(top, p["pole_diameter"] / 2))
    stock = sk.rounded_box(p["stock_depth"], p["broom_width"], p["stock_height"], bottom, fillet=8)
    bl = p["bristle_length"]
    bristles = cq.Workplane("XY").box(p["stock_depth"] * 1.4, p["broom_width"] * 1.1, bl).translate(
        (bottom[0], bottom[1], bl / 2 + 1.0))
    for k in range(-6, 7):                                               # tufts: slots up the bristle block
        bristles = bristles.cut(cq.Workplane("XY").box(p["stock_depth"] * 2, 3, bl * 0.7)
                                .translate((bottom[0], bottom[1] + k * p["broom_width"] / 13, bl * 0.35)))
    return pole.union(stock), bristles


def bin_lid(p):
    """The trash-bin lid on the ork's left forearm: a dished galvanised disc with three pressed rings and a rim,
    facing outward (+y), its handle and strap round the forearm. Returns (lid,)."""
    c, u, _ = hand_frame(p, 1)
    mid = _add(c, (-p["forearm_length"] * 0.45, u))
    R, t = p["lid_diameter"] / 2, p["lid_thickness"]
    y0 = mid[1] + p["arm_diameter"] / 2 + 4                              # the handle (8 mm proud) sunk into the strap
    disc = cq.Workplane("XZ").circle(R).extrude(-t)                       # XZ extrudes along -y: flip to +y
    Rs = R * 2.2                                                          # the dome: a sheet-thick spherical cap
    cap = cq.Workplane("XY").sphere(Rs).cut(cq.Workplane("XY").sphere(Rs - t)).translate((0, -Rs + p["lid_dish"], 0))
    dome = cap.intersect(cq.Workplane("XZ").circle(R).extrude(-p["lid_dish"]))
    rings = [cq.Workplane("XZ").circle(R * f + 3).circle(R * f).extrude(-t - 3) for f in (0.35, 0.6, 0.85)]
    rim = cq.Workplane("XZ").circle(R + 4).circle(R - 2).extrude(-t - 6)
    handle = sk.capsule((-R * 0.25, -8, 0), (R * 0.25, -8, 0), 7).union(sk.capsule((-R * 0.25, -8, 0), (-R * 0.25, 0, 0), 7)) \
        .union(sk.capsule((R * 0.25, -8, 0), (R * 0.25, 0, 0), 7))
    lid = sk.union_all([disc, dome] + rings + [rim, handle])
    e = math.degrees(math.atan2(u[0], -u[2]))                             # the disc's plane along the forearm
    return (lid.rotate((0, 0, 0), (0, 1, 0), e).translate((mid[0], y0, mid[2])),)


_ORK_EXTRA = [Parameter(k, float(fh.ORK[k]), "deg" if k.endswith("_deg") else "mm", min=0.0, description=d)
              for k, d in fh.ORK_EXTRA.items()]


class Ork(Humanoid):
    """The ork: a bulky humanoid (Talia's frame scaled up and thickened: a broad chest, thick limbs, short heavy
    legs, a big low head with a jutting jaw, two tusks and a heavy brow) holding a broom in its right hand and a
    trash-bin lid strapped to its left forearm (a shield). ``part``: as ``Humanoid``, plus 'broom' and 'lid'."""

    parameters = [Parameter("part", "humanoid", choices=("humanoid", "trunk", "thigh", "shin", "foot", "broom", "lid"),
                            description="what to build")] + \
        [q for q in sk.with_defaults(Humanoid.parameters, **{k: v for k, v in fh.ORK.items() if k not in fh.ORK_EXTRA})
         if q.name != "part"] + _ORK_EXTRA

    def face(self, p):
        """Tusks from the lower jaw and a heavy brow, on the head of ``Humanoid._trunk``."""
        hh = self.hip_height(p)
        zn = hh + p["pelvis_height"] / 2 + p["torso_height"]
        Wh, Dh, Hh = p["head_width"], p["head_depth"], p["head_height"]
        zh = zn + p["neck_length"] + Hh / 2
        jaw = sk.rounded_box(Dh * 0.35, Wh * 0.9, Hh * 0.3, (Dh * 0.38, 0, zh - Hh * 0.38), fillet=15)
        brow = sk.rounded_box(Dh * 0.2, Wh * 1.02, Hh * 0.12, (Dh * 0.45, 0, zh + Hh * 0.2), fillet=8)
        tusks = [sk.polyline_limb([(Dh * 0.5, sy * Wh * 0.3, zh - Hh * 0.35), (Dh * 0.6, sy * Wh * 0.33, zh - Hh * 0.15),
                                   (Dh * 0.55, sy * Wh * 0.36, zh + p["tusk_length"] - Hh * 0.15)], [9.0, 5.0])
                 for sy in (-1, 1)]
        return [jaw, brow] + tusks

    def build(self, p):
        if p["part"] == "broom":
            a, b = broom(p)
            return a.union(b)
        if p["part"] == "lid":
            return bin_lid(p)[0]
        body = super().build(dict(p, part=p["part"]))
        if p["part"] != "humanoid":
            return body
        a, b = broom(p)
        return sk.union_all([body] + self.face(p) + [a, b, bin_lid(p)[0]])
