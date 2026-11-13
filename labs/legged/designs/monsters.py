"""The monster designs (CAD): variants of the zoomorphic animals and the humanoid, and three new body plans.
Numbers in ``monster_specs``. x forward, y left, z up, ground at z = 0."""
import math

from vegeta.dedalus import Design, Parameter

import monster_specs as ms
import shape_kit as sk
import zoo_specs as zs
import fantasy_humanoid_specs as fh
import fantasy_humanoids as fhu
from humanoid import Humanoid
from zoo_others import Octopus, Snake
from zoo_quadruped import Deer, Dog, Quadruped, parameters_of


def _ridge(p, n, height, r=14.0):
    """Spikes along the back of a quadruped, shoulder to rump."""
    bx, _, bz = zs.body_centre(p)
    top = bz + p["body_height"] / 2 - 10
    xs = [bx + p["body_length"] * (0.35 - 0.7 * i / max(n - 1, 1)) for i in range(n)]
    return [sk.capsule((x, 0, top - 15), (x - height * 0.3, 0, top + height * (1 - 0.4 * abs(i - n / 2) / n)), r)
            for i, x in enumerate(xs)]


class Wolf(Dog):
    """Fenrir, the four-legged wolf: Laika the dog scaled to a big wolf, with a low tail, a dorsal ridge of
    ``ridge_spikes`` spikes and two fangs on the snout."""

    spec = ms.WOLF
    parameters = parameters_of(ms.WOLF)

    def extras(self, p, hc):
        out = _ridge(p, int(p["ridge_spikes"]), p["spike_height"]) if p["ridge_spikes"] > 0 else []
        if p["fang_length"] > 0:
            x = hc[0] + p["head_length"] * 0.3 + p["snout_length"] * 0.8
            z = hc[2] - p["head_height"] * 0.2 - p["head_height"] * 0.18
            out += [sk.capsule((x, sy * p["head_width"] * 0.12, z), (x + 10, sy * p["head_width"] * 0.12,
                                                                     z - p["fang_length"]), 8.0) for sy in (-1, 1)]
        return out


class Wyvern(Design):
    """The wyvern: a hexacopter dragon. An egg fuselage, a neck rising forward to a horned head, a long tapered
    tail, six rotor arms (motor pods and rotor discs at ``monster_specs.rotor_centres``) and two manipulator arms
    hanging under the body (shoulder, elbow, wrist and a three-finger claw). Shown hovering at ``cruise_height``;
    ``part``: the wyvern, the airframe (body + rotor arms) or one manipulator."""

    spec = ms.WYVERN
    parameters = parameters_of(ms.WYVERN, ("wyvern", "airframe", "manipulator"))

    def manipulator(self, p, sy):
        sh, el, wr = ms.wyvern_arm_points(p, sy)
        r = p["manip_radius"]
        arm = sk.polyline_limb([sh, el, wr], [r, r * 0.8])
        c = p["claw_length"]
        claws = [sk.polyline_limb([wr, (wr[0] + c * 0.6, wr[1] + sy * k * c * 0.35, wr[2] - c * 0.4),
                                   (wr[0] + c * 0.75, wr[1] + sy * k * c * 0.4, wr[2] - c * 0.85)], [r * 0.4, r * 0.3])
                 for k in (-1, 0, 1)]
        return sk.union_all([arm] + claws)

    def airframe(self, p):
        zc = p["cruise_height"]
        body = sk.egg(p["body_length"], p["body_width"], p["body_height"], (0, 0, zc), nose=0.6, tail=0.5)
        parts = [body]
        for c in ms.rotor_centres(p):
            parts.append(sk.capsule((c[0] * 0.15, c[1] * 0.15, c[2] - 30), c, p["rotor_arm_radius"]))
            parts.append(sk.disc((c[0], c[1], c[2] + 25), p["motor_radius"], 60.0))
            parts.append(sk.disc((c[0], c[1], c[2] + 53), p["rotor_radius"], 6.0))
        return sk.union_all(parts)

    def build(self, p):
        if p["part"] == "manipulator":
            return self.manipulator(p, 1)
        frame = self.airframe(p)
        if p["part"] == "airframe":
            return frame
        zc = p["cruise_height"]
        a = math.radians(p["neck_angle_deg"])
        n0 = (p["body_length"] * 0.4, 0, zc + p["body_height"] * 0.15)
        hc = (n0[0] + p["neck_length"] * math.cos(a), 0, n0[2] + p["neck_length"] * math.sin(a))
        neck = sk.capsule(n0, hc, p["head_height"] * 0.3)
        head = sk.egg(p["head_length"], p["head_width"], p["head_height"], (hc[0] + p["head_length"] * 0.25, 0, hc[2]),
                      nose=0.45, tail=0.9)
        horns = [sk.capsule((hc[0], sy * p["head_width"] * 0.3, hc[2] + p["head_height"] * 0.3),
                            (hc[0] - p["horn_length"], sy * p["head_width"] * 0.45, hc[2] + p["horn_length"] * 0.6), 12.0)
                 for sy in (-1, 1)]
        tail, tpts = sk.planar_chain((-p["body_length"] * 0.45, 0, zc), 6, p["tail_length"] / 6, p["body_height"] * 0.18,
                                  p["body_height"] * 0.05, heading_deg=185, bend_deg=4, plane="xz")
        tx, _, tz = tpts[-1]
        tip = sk.membrane([(tx + 40, 0, tz - 90), (tx - 160, 0, tz), (tx + 40, 0, tz + 90)], 8.0)
        return sk.union_all([frame, neck, head, tail, tip] + horns + [self.manipulator(p, sy) for sy in (-1, 1)])


class Centaur(Quadruped):
    """Chiron-C, the centaur: a horse barrel on four long three-segment legs (Cerva's), a tail, and a humanoid
    torso rising from the withers with a head and two manipulator arms (shoulder, elbow, a gripper paddle)."""

    spec = ms.CENTAUR
    parameters = parameters_of(ms.CENTAUR)

    def head(self, p, hc):                       # the horse half has no head: the torso carries it
        return None

    def body(self, p):
        bc = zs.body_centre(p)
        torso_h = sk.egg(p["body_length"], p["body_width"], p["body_height"], bc, nose=0.8, tail=0.7)
        waist, top, shoulders = ms.centaur_torso(p)
        chest = sk.egg(p["torso_height"], p["torso_width"], p["torso_depth"], (0, 0, 0), nose=0.85, tail=0.7) \
            .rotate((0, 0, 0), (0, 1, 0), -90).translate((waist[0], 0, waist[2] + p["torso_height"] / 2))
        hc = (waist[0] + 20, 0, top + 40 + p["head_height"] / 2)
        head = sk.egg(p["head_length"], p["head_width"], p["head_height"], hc, nose=0.75, tail=0.85)
        for sy in (-1, 1):
            head = head.cut(sk.ball((hc[0] + p["head_length"] * 0.45, sy * p["head_width"] * 0.2, hc[2] + 20), 18))
        neck = sk.capsule((waist[0], 0, top - 40), hc, p["head_width"] * 0.25)
        arms = []
        for s in shoulders:
            el = (s[0] + 60, s[1], s[2] - p["upper_arm_length"])
            wr = (el[0] + p["forearm_length"] * 0.85, el[1], el[2] - p["forearm_length"] * 0.5)
            arms.append(sk.polyline_limb([s, el, wr], [p["arm_radius"], p["arm_radius"] * 0.85]))
            arms.append(sk.rounded_box(120, 30, 90, (wr[0] + 50, wr[1], wr[2] - 20), fillet=8))
        return sk.union_all([torso_h, chest, neck, head, self.tail(p)] + arms)


class Leech(Snake):
    """Hirudo, the giant leech: Naga the snake with fewer, fatter segments flattened to ``flatten`` of their
    width (each a lofted ellipse ring), a ring ridge per segment and a round sucker at each end (the front one
    with a toothed mouth)."""

    spec = ms.LEECH
    parameters = parameters_of(ms.LEECH, ("leech", "segment"))

    def segment(self, a, b, r, f):
        L = math.dist(a, b)
        seg = sk.egg(L * 1.25, 2 * r, 2 * r * f, (0, 0, 0), nose=0.85, tail=0.85)
        yaw = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
        mid = tuple((a[k] + b[k]) / 2 for k in range(3))
        return seg.rotate((0, 0, 0), (0, 0, 1), yaw).translate(mid)

    def build(self, p):
        pts = zs.slither_points(dict(p, body_radius=p["body_radius"] * p["flatten"]))
        n = len(pts) - 1
        r0, r1, f = p["body_radius"], p["tail_radius"], p["flatten"]
        segs = [self.segment(pts[i], pts[i + 1], r0 + (r1 - r0) * i / max(n - 1, 1), f) for i in range(n)]
        if p["part"] == "segment":
            return segs[0].translate(tuple(-c for c in pts[0]))
        zr = r0 * f
        mouth = sk.disc((p["sucker_radius"] * 0.3, 0, zr), p["sucker_radius"], zr * 1.4, axis="x")
        mouth = mouth.cut(sk.disc((p["sucker_radius"] * 0.3 + zr * 0.5, 0, zr), p["sucker_radius"] * 0.6, zr, axis="x"))
        teeth = [sk.capsule((p["sucker_radius"] * 0.3, p["sucker_radius"] * 0.5 * math.cos(t), zr + p["sucker_radius"] * 0.5 * math.sin(t)),
                            (p["sucker_radius"] * 0.3 + 40, p["sucker_radius"] * 0.35 * math.cos(t),
                             zr + p["sucker_radius"] * 0.35 * math.sin(t)), 8.0) for t in [k * math.pi / 4 for k in range(8)]]
        tail = sk.disc((pts[-1][0], pts[-1][1], 20), p["sucker_radius"] * 1.1, 40.0)
        return sk.union_all(segs + [mouth, tail] + teeth)


class Leshy(Deer):
    """The Leshy, the forest spirit: Cerva the deer grown tall and gaunt, with huge antlers of ``antler_tines``
    tines and a beard of ``beard_twigs`` twigs hanging from the jaw."""

    spec = ms.LESHY
    parameters = parameters_of(ms.LESHY)

    def extras(self, p, hc):
        out = super().extras(p, hc)
        n, L = int(p["beard_twigs"]), p["beard_length"]
        for i in range(n):
            y = (i - (n - 1) / 2) * p["head_width"] * 0.6 / max(n - 1, 1)
            a = (hc[0] + p["head_length"] * 0.1, y, hc[2] - p["head_height"] * 0.35)
            out.append(sk.polyline_limb([a, (a[0] - 20, y * 1.2, a[2] - L * 0.5), (a[0] + 10, y * 1.4, a[2] - L * (0.8 + 0.2 * (i % 2)))],
                                        [10.0, 7.0]))
        return out


class Wraith(Design):
    """The wraith: a helium envelope (an egg with four tail fins) floating at ``float_height``, a gondola slung
    under it on four struts with two glowing eyes, and ``arms`` octopus arms hanging from the gondola and curling
    outwards (``monster_specs.wraith_arm_points``). ``part``: the wraith or one arm."""

    spec = ms.WRAITH
    parameters = parameters_of(ms.WRAITH, ("wraith", "arm"))

    def arm(self, p, i):
        pts = ms.wraith_arm_points(p, i)
        n = len(pts) - 1
        r = p["arm_radius"]
        return sk.polyline_limb(pts, [r * (1 - 0.8 * k / n) for k in range(n)])

    def build(self, p):
        if p["part"] == "arm":
            return self.arm(p, 0)
        zc, D, L = p["float_height"], p["envelope_diameter"], p["envelope_length"]
        env = sk.egg(L, D, D, (0, 0, zc), nose=0.6, tail=0.35)
        f = p["fin_size"]
        fins = [sk.membrane([(-L * 0.35, 0, zc), (-L * 0.5, 0, zc), (-L * 0.55, f * math.cos(t), zc + f * math.sin(t)),
                             (-L * 0.4, f * 0.6 * math.cos(t), zc + f * 0.6 * math.sin(t))], 10.0)
                for t in (0, math.pi / 2, math.pi, 3 * math.pi / 2)]
        zg = zc - D / 2 - p["suspension_length"] - p["gondola_height"] / 2
        gon = sk.egg(p["gondola_length"], p["gondola_width"], p["gondola_height"], (0, 0, zg), nose=0.7, tail=0.6)
        for sy in (-1, 1):
            gon = gon.cut(sk.ball((p["gondola_length"] * 0.42, sy * p["gondola_width"] * 0.18, zg + 20),
                                  p["eye_diameter"] / 2))
        struts = [sk.capsule((sx * p["gondola_length"] * 0.3, sy * p["gondola_width"] * 0.3, zg + p["gondola_height"] * 0.3),
                             (sx * L * 0.15, sy * D * 0.15, zc - D * 0.42), 10.0)
                  for sx in (-1, 1) for sy in (-1, 1)]
        return sk.union_all([env, gon] + fins + struts + [self.arm(p, i) for i in range(int(p["arms"]))])


_GHOUL_EXTRA = [Parameter("pincer_length", fh.PINCER["pincer_length"], "mm", min=20, description="each jaw"),
                Parameter("pincer_open_deg", fh.PINCER["pincer_open_deg"], "deg", min=0, max=60,
                          description="each jaw out from the forearm's line"),
                Parameter("ridge_spikes", ms.GHOUL["ridge_spikes"], "", min=0, description="spikes down the spine")]


class Ghoul(Humanoid):
    """The ghoul: Talia the humanoid gone feral — a narrow waist, thin legs bent deep at the knee, long arms ending
    in a pincer each (two hooked, toothed jaws past the hand, ``fantasy_humanoids.pincer``), a small head with big
    eyes and a ridge of spikes down the back."""

    parameters = sk.with_defaults(Humanoid.parameters, **{k: v for k, v in ms.GHOUL.items()
                                                           if k not in ("ridge_spikes",)}) + _GHOUL_EXTRA

    def build(self, p):
        robot = super().build(p)
        if p["part"] != "humanoid":
            return robot
        hh = self.hip_height(p)
        extra = [fhu.pincer(p, sy, p["pincer_length"], p["pincer_open_deg"]) for sy in (-1, 1)]
        n = int(p["ridge_spikes"])
        z0 = hh + p["pelvis_height"] / 2
        for i in range(n):
            z = z0 + p["torso_height"] * (0.15 + 0.8 * i / max(n - 1, 1))
            x = -p["torso_depth"] / 2
            extra.append(sk.capsule((x + 10, 0, z), (x - 70, 0, z + 40), 12.0))
        return sk.union_all([robot] + extra)
