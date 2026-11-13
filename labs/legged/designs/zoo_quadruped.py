import math

import cadquery as cq
from vegeta.dedalus import Design, Parameter

import shape_kit as sk
import zoo_specs as zs

UNITLESS = {"antler_tines", "trunk_segments"}


def parameters_of(spec, part_choices=("animal", "body", "leg")):
    """A Design parameter list from a ``zoo_specs`` dict (mm and deg; counts unitless)."""
    out = [Parameter("part", part_choices[0], choices=part_choices, description="what to build")]
    for k, v in spec.items():
        unit = "" if k in UNITLESS else ("deg" if k.endswith("_deg") else "mm")
        out.append(Parameter(k, float(v), unit, min=-90.0 if k.endswith("_deg") else 0.0,
                             description=zs.DESCRIPTIONS.get(k, "")))
    return out


class Quadruped(Design):
    """A four-legged animal robot: an egg-shaped torso, four legs (two or three capsule segments through
    ``zoo_specs.leg_points``, front elbows back and rear knees forward, a ball foot), a neck to a head with a snout,
    two eyes and two ears, and a tail. x forward, y left, z up, ground at z = 0. ``part``: the animal, the body
    (torso + neck + head + tail) or one front leg (world frame). Subclasses set ``spec`` and add ``extras``."""

    spec = zs.QUADRUPED
    parameters = parameters_of(zs.QUADRUPED)

    def leg(self, p, front, sy):
        pts = zs.leg_points(p, front, sy)
        r = p["leg_radius"]
        radii = [r, r * 0.8, r * 0.7][: len(pts) - 1]
        leg = sk.polyline_limb(pts, radii)
        return leg.union(sk.ball(pts[-1], p["foot_radius"]))

    def head(self, p, hc):
        Lh, Wh, Hh = p["head_length"], p["head_width"], p["head_height"]
        head = sk.egg(Lh, Wh, Hh, hc, nose=0.7, tail=0.8)
        if p["snout_length"] > 0:
            head = head.union(sk.capsule((hc[0] + Lh * 0.3, 0, hc[2] - Hh * 0.15),
                                         (hc[0] + Lh * 0.3 + p["snout_length"], 0, hc[2] - Hh * 0.2), Hh * 0.22))
        eyes = [sk.ball((hc[0] + Lh * 0.3, sy * Wh * 0.3, hc[2] + Hh * 0.12), Hh * 0.08) for sy in (-1, 1)]
        for e in eyes:
            head = head.cut(e)
        if p["ear_size"] > 0:
            for sy in (-1, 1):
                head = head.union(self.ear(p, hc, sy))
        return head

    def ear(self, p, hc, sy):
        """A pricked triangular ear on top of the head."""
        e, Wh, Hh = p["ear_size"], p["head_width"], p["head_height"]
        b = (hc[0] - p["head_length"] * 0.1, sy * Wh * 0.3, hc[2] + Hh * 0.35)
        return sk.membrane([(b[0] - e * 0.3, b[1], b[2]), (b[0] + e * 0.3, b[1], b[2]),
                            (b[0] - e * 0.1, b[1] + sy * e * 0.2, b[2] + e)], 8.0)

    def tail(self, p):
        if p["tail_length"] <= 0:
            return None
        bc = zs.body_centre(p)
        n = 5
        start = (bc[0] - p["body_length"] / 2 + 20, 0, bc[2] + p["body_height"] * 0.15)
        pts = zs.arm_points(start, n, p["tail_length"] / n, 180 - p["tail_angle_deg"], -6 if p["tail_angle_deg"] > 0 else 4)
        r = max(p["leg_radius"] * 0.5, 8.0)
        return sk.polyline_limb(pts, [r * (1 - 0.12 * i) for i in range(n)])

    def extras(self, p, hc):
        """Subclass hook: more shapes (antlers, trunk, tusks) given the head centre."""
        return []

    def body(self, p):
        bc = zs.body_centre(p)
        torso = sk.egg(p["body_length"], p["body_width"], p["body_height"], bc)
        n0, hc = zs.head_centre(p)
        neck = sk.capsule(n0, hc, p["neck_radius"])
        return sk.union_all([torso, neck, self.head(p, hc), self.tail(p)] + self.extras(p, hc))

    def build(self, p):
        if p["part"] == "leg":
            return self.leg(p, True, 1)
        body = self.body(p)
        if p["part"] == "body":
            return body
        legs = [self.leg(p, front, sy) for front in (True, False) for sy in (-1, 1)]
        return sk.union_all([body] + legs)


class Dog(Quadruped):
    """Laika, the dog-like robot: a medium quadruped, digitigrade three-segment legs, pricked ears, a raised tail.
    (The lab's Cerberus, ``robot_dog.py``, is the engineered QDD dog; Laika is the animal-shaped one.)"""

    spec = zs.DOG
    parameters = parameters_of(zs.DOG)


class Bear(Quadruped):
    """Ursa, the bear-like robot: heavy and wide, short plantigrade legs (two segments) with big flat paws, a round
    head with small round ears, a stub tail."""

    spec = zs.BEAR
    parameters = parameters_of(zs.BEAR)

    def leg(self, p, front, sy):
        pts = zs.leg_points(p, front, sy)
        r = p["leg_radius"]
        leg = sk.polyline_limb(pts, [r, r * 0.85])
        x, y, _ = pts[-1]
        paw = sk.rounded_box(p["foot_radius"] * 2.6, p["foot_radius"] * 1.8, p["foot_radius"] * 0.9,
                             (x + p["foot_radius"] * 0.5, y, p["foot_radius"] * 0.45), fillet=20.0)
        return leg.union(paw)

    def ear(self, p, hc, sy):
        e = p["ear_size"]
        return sk.ball((hc[0] - p["head_length"] * 0.15, sy * p["head_width"] * 0.35, hc[2] + p["head_height"] * 0.42), e / 2)


class Deer(Quadruped):
    """Cerva, the deer-like robot: slim, long three-segment legs with small hooves, a long raised neck, big ears and
    branching antlers (``antler_length``, ``antler_tines`` per side)."""

    spec = zs.DEER
    parameters = parameters_of(zs.DEER)

    def antler(self, p, hc, sy, scale=1.0):
        L = p["antler_length"] * scale
        b = (hc[0] - p["head_length"] * 0.05, sy * p["head_width"] * 0.25, hc[2] + p["head_height"] * 0.4)
        top = (b[0] - L * 0.35, b[1] + sy * L * 0.45, b[2] + L * 0.8)
        beam = sk.capsule(b, top, 10.0 * scale ** 0.5)
        tines = []
        n = int(p["antler_tines"])
        for i in range(n):
            t = (i + 1) / (n + 1)
            s = tuple(b[k] + t * (top[k] - b[k]) for k in range(3))
            tines.append(sk.capsule(s, (s[0] + L * 0.28, s[1] + sy * L * 0.05, s[2] + L * 0.3), 6.0 * scale ** 0.5))
        return sk.union_all([beam] + tines)

    def extras(self, p, hc):
        return [self.antler(p, hc, sy) for sy in (-1, 1)] if p["antler_length"] > 0 else []


class Elephant(Quadruped):
    """Gaja, the elephant robot: pillar legs ending in wide pads, a big domed head, a curling trunk of
    ``trunk_segments`` segments, two tusks and large flat ears; a thin hanging tail."""

    spec = zs.ELEPHANT
    parameters = parameters_of(zs.ELEPHANT)

    def leg(self, p, front, sy):
        pts = zs.leg_points(p, front, sy)
        r = p["leg_radius"]
        leg = sk.polyline_limb(pts, [r, r * 0.95])
        x, y, _ = pts[-1]
        return leg.union(sk.disc((x, y, p["foot_radius"] * 0.35), p["foot_radius"], p["foot_radius"] * 0.7))

    def ear(self, p, hc, sy):
        e = p["ear_size"]
        b = (hc[0] - p["head_length"] * 0.2, sy * p["head_width"] * 0.45, hc[2])
        return sk.membrane([(b[0], b[1], b[2] + e * 0.5), (b[0] - e * 0.7, b[1] + sy * e * 0.3, b[2] + e * 0.3),
                            (b[0] - e * 0.8, b[1] + sy * e * 0.35, b[2] - e * 0.4), (b[0] - e * 0.1, b[1], b[2] - e * 0.6)],
                           25.0)

    def trunk_points(self, p, hc):
        start = (hc[0] + p["head_length"] * 0.4, 0, hc[2] - p["head_height"] * 0.1)
        return zs.arm_points(start, int(p["trunk_segments"]), p["trunk_segment_length"], -70, -8)

    def extras(self, p, hc):
        pts = self.trunk_points(p, hc)
        n = len(pts) - 1
        trunk = sk.polyline_limb(pts, [p["head_width"] * 0.14 * (1 - 0.06 * i) for i in range(n)])
        tusks = []
        if p["tusk_length"] > 0:
            for sy in (-1, 1):
                a = (hc[0] + p["head_length"] * 0.3, sy * p["head_width"] * 0.18, hc[2] - p["head_height"] * 0.3)
                tusks.append(sk.capsule(a, (a[0] + p["tusk_length"] * 0.8, a[1] * 1.3, a[2] - p["tusk_length"] * 0.2), 28.0))
                tusks.append(sk.capsule((a[0] + p["tusk_length"] * 0.8, a[1] * 1.3, a[2] - p["tusk_length"] * 0.2),
                                        (a[0] + p["tusk_length"], a[1] * 1.2, a[2] + p["tusk_length"] * 0.15), 22.0))
        return [trunk] + tusks
