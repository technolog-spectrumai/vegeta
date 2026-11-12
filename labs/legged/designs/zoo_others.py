import math

from vegeta.dedalus import Design

import shape_kit as sk
import zoo_specs as zs
from zoo_quadruped import parameters_of


class Snake(Design):
    """Naga, the snake-like robot: ``segments`` identical capsule segments (each joint two-axis, pitch + yaw)
    tapering from ``body_radius`` to ``tail_radius``, laid on the ground in a slither wave of ``wave_deg``, and a
    wedge head with two eyes. x forward (the head), y left, z up; ``part``: the snake or one segment."""

    spec = zs.SNAKE
    parameters = parameters_of(zs.SNAKE, ("snake", "segment"))

    def spine(self, p):
        pts = zs.slither_points(p)
        n = len(pts) - 1
        r0, r1 = p["body_radius"], p["tail_radius"]
        return sk.polyline_limb(pts, [r0 + (r1 - r0) * i / max(n - 1, 1) for i in range(n)]), pts

    def head(self, p):
        r0 = p["body_radius"]
        Lh, Wh, Hh = p["head_length"], p["head_width"], p["head_height"]
        head = sk.egg(Lh, Wh, Hh, (Lh * 0.4, 0, max(Hh / 2, r0)), nose=0.45, tail=0.9)
        for sy in (-1, 1):
            head = head.cut(sk.ball((Lh * 0.55, sy * Wh * 0.32, max(Hh / 2, r0) + Hh * 0.2), Hh * 0.1))
        return head

    def build(self, p):
        if p["part"] == "segment":
            r = p["body_radius"]
            return sk.capsule((0, 0, 0), (p["segment_length"], 0, 0), r).cut(
                sk.rod((-r, 0, 0), (p["segment_length"] + r, 0, 0), r * 0.6))
        body, _ = self.spine(p)
        return body.union(self.head(p))


class Bat(Design):
    """Noctis, the bat robot with folding wings: a furry egg body, a head with tall ears, and per side a humerus,
    a forearm and three finger bones carrying a membrane (body → elbow → wrist → finger tips → leg). ``fold``: 0 =
    wings spread, 1 = folded along the body (``zoo_specs.bat_wing_points``). Two short legs with hook feet; the
    body is ``stand_height`` above the ground. ``part``: the bat or one (left) wing."""

    spec = zs.BAT
    parameters = parameters_of(zs.BAT, ("bat", "wing"))

    def wing(self, p, sy):
        sh, el, wr, tips = zs.bat_wing_points(p, sy)
        r = p["bone_radius"]
        bones = [sk.capsule(sh, el, r * 1.4), sk.capsule(el, wr, r * 1.2)] + [sk.capsule(wr, t, r * 0.8) for t in tips]
        if p["fold"] < 0.8:
            t = p["membrane_thickness"]
            hip = (-p["body_length"] * 0.35, sy * p["body_width"] * 0.35, sh[2] - 10)
            panels = [sk.membrane([sh, el, wr, tips[0]], t), sk.membrane([wr, tips[0], tips[1]], t),
                      sk.membrane([wr, tips[1], tips[2]], t), sk.membrane([sh, wr, tips[2], hip], t)]
            bones += panels
        return sk.union_all(bones)

    def build(self, p):
        if p["part"] == "wing":
            return self.wing(p, 1)
        z = p["stand_height"]
        L, W, H = p["body_length"], p["body_width"], p["body_height"]
        body = sk.egg(L, W, H, (0, 0, z), nose=0.7, tail=0.5)
        hs = p["head_size"]
        hc = (L / 2 + hs * 0.3, 0, z + H * 0.2)
        head = sk.egg(hs * 1.1, hs, hs * 0.9, hc, nose=0.5, tail=0.9)
        for sy in (-1, 1):
            head = head.union(sk.membrane([(hc[0] - hs * 0.2, sy * hs * 0.25, hc[2] + hs * 0.3),
                                           (hc[0] + hs * 0.15, sy * hs * 0.25, hc[2] + hs * 0.3),
                                           (hc[0] - hs * 0.1, sy * hs * 0.4, hc[2] + hs * 0.3 + p["ear_size"])], 4.0))
            head = head.cut(sk.ball((hc[0] + hs * 0.45, sy * hs * 0.2, hc[2] + hs * 0.1), hs * 0.08))
        legs = []
        for sy in (-1, 1):
            a = (-L * 0.3, sy * W * 0.3, z - H * 0.3)
            f = (-L * 0.35, sy * W * 0.35, 8.0)
            legs.append(sk.capsule(a, f, p["bone_radius"]))
            legs.append(sk.capsule(f, (f[0] + 30, f[1], 8.0), p["bone_radius"] * 0.8))
        return sk.union_all([body, head] + legs + [self.wing(p, sy) for sy in (-1, 1)])


class Octopus(Design):
    """Kraken, the octopus robot: an egg mantle rising behind a round head with two eyes, and ``arms`` tapered arms
    of ``arm_segments`` segments that spread on the ground and curl up at the tips
    (``zoo_specs.octopus_arm_points``). ``part``: the octopus or one arm."""

    spec = zs.OCTOPUS
    parameters = parameters_of(zs.OCTOPUS, ("octopus", "arm"))

    def arm(self, p, i):
        pts = zs.octopus_arm_points(p, i)
        n = len(pts) - 1
        r = p["arm_radius"]
        return sk.polyline_limb(pts, [r * (1 - 0.8 * k / n) for k in range(n)])

    def build(self, p):
        if p["part"] == "arm":
            return self.arm(p, 0)
        rh = p["head_radius"]
        zc = p["arm_radius"] + 40.0 + rh * 0.8
        head = sk.ball((0, 0, zc), rh)
        for sy in (-1, 1):
            head = head.cut(sk.ball((rh * 0.62, sy * rh * 0.5, zc + rh * 0.38), p["eye_diameter"] / 2))
        a = math.radians(35)
        Lm = p["mantle_length"]
        mantle = sk.egg(Lm, p["mantle_width"], p["mantle_width"] * 0.9,
                        (-Lm * 0.35 * math.cos(a), 0, zc + rh * 0.5 + Lm * 0.35 * math.sin(a)), nose=0.8, tail=0.45)
        mantle = mantle.rotate((0, 0, zc + rh * 0.5), (0, 1, 0), 35)
        return sk.union_all([head, mantle] + [self.arm(p, i) for i in range(int(p["arms"]))])
