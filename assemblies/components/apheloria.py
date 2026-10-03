"""Promoted from ``notebooks/designs/apheloria.py`` as it was proven there; the notebook copy may move on."""
import math
import cadquery as cq
import numpy as np
from vegeta.dedalus import Design, Parameter


class Apheloria(Design):
    """Apheloria, the Myropod that rolls into a ball (the pill millipede). Large modular segments —
    a cylindrical body with a domed armour plate (tergite) on its back, four legs (two pairs) per
    segment, a two-axis joint between segments — and a head segment with the sensor suite. Curled
    nose-under with 360°/n of pitch at every joint, the body cylinders form a ring and the armour
    plates, cut as wedges of one sphere, close into a ball; the legs fold into the ring.

    ``mode``: "walk" (straight, legs down) or "ball" (curled, legs folded). ``part``: the whole
    unit, one segment with its plate, the plate alone, one leg (hip bore at the origin, reaching +y),
    or the head. x along the body (head at +x), y left, z up; in walk mode the belly is at z = 0."""

    parameters = [
        Parameter("part", "apheloria", choices=("apheloria", "segment", "plate", "leg", "head"), description="what to build"),
        Parameter("mode", "walk", choices=("walk", "ball"), description="straight and walking, or curled into the ball"),
        Parameter("n_segments", 8, "", min=4, max=20, description="body segments behind the head (the head counts as one more link of the ring)"),
        Parameter("seg_length", 150.0, "mm", min=50, description="body cylinder length"),
        Parameter("body_diameter", 110.0, "mm", min=30),
        Parameter("shell_thickness", 3.0, "mm", min=1),
        Parameter("joint_gap", 18.0, "mm", min=5, description="neck between two segments (the two-axis joint)"),
        Parameter("joint_diameter", 44.0, "mm", min=10),
        Parameter("joint_pin_diameter", 10.0, "mm", min=2),
        Parameter("plate_thickness", 4.0, "mm", min=1, description="armour plate (a wedge of the ball's sphere)"),
        Parameter("plate_standoff", 8.0, "mm", min=0, description="plate above the body cylinder, at the back"),
        Parameter("plate_width", 220.0, "mm", min=20, description="plate width across the body (y)"),
        Parameter("femur_length", 80.0, "mm", min=10),
        Parameter("tibia_length", 95.0, "mm", min=10),
        Parameter("leg_diameter", 18.0, "mm", min=3),
        Parameter("foot_diameter", 26.0, "mm", min=5),
        Parameter("hip_angle_deg", 30.0, "deg", min=-80, max=90, description="femur below horizontal, reaching outwards (walk mode)"),
        Parameter("knee_angle_deg", 75.0, "deg", min=0, max=170, description="tibia below horizontal (walk mode)"),
        Parameter("leg_pair_spacing", 0.5, "", min=0.2, max=0.9, description="distance between the two leg pairs, as a fraction of the segment length"),
        Parameter("head_length", 150.0, "mm", min=50),
    ]

    # ---- geometry shared with the notebook ----
    @staticmethod
    def pitch_length(p):
        return p["seg_length"] + p["joint_gap"]

    @staticmethod
    def n_links(p):
        return p["n_segments"] + 1                            # the head is the first link of the ring

    def coil_angle_deg(self, p):
        return 360.0 / self.n_links(p)

    def coil_radius(self, p):
        """Centre-line radius of the curled body (the ring through the segment centres)."""
        return self.pitch_length(p) / (2 * math.sin(math.radians(self.coil_angle_deg(p)) / 2))

    def ball_radius(self, p):
        return self.coil_radius(p) + p["body_diameter"] / 2 + p["plate_standoff"] + p["plate_thickness"]

    @staticmethod
    def hip_x(p):
        return p["seg_length"] * p["leg_pair_spacing"] / 2

    @staticmethod
    def leg_reach(p, hip_deg=None, knee_deg=None):
        a = math.radians(p["hip_angle_deg"] if hip_deg is None else hip_deg)
        b = math.radians(p["knee_angle_deg"] if knee_deg is None else knee_deg)
        y_knee = p["femur_length"] * math.cos(a)
        y = y_knee + p["tibia_length"] * math.cos(b)
        z = p["femur_length"] * math.sin(a) + p["tibia_length"] * math.sin(b)
        return {"reach_y": max(y, y_knee), "foot_y": y, "drop_z": z, "stance_width": p["body_diameter"] + 2 * 8 + 2 * y + p["foot_diameter"]}

    # ---- parts (segment frame: body axis x, centre of the cylinder at the origin, back at +z) ----
    def _body(self, p, length):
        D, t = p["body_diameter"], p["shell_thickness"]
        body = cq.Workplane("YZ").circle(D / 2).extrude(length / 2, both=True)
        body = body.cut(cq.Workplane("YZ").circle(D / 2 - t).extrude(length / 2 - t, both=True))
        return body

    def _plate(self, p, length):
        """Armour plate: the wedge of the ball's sphere above this segment, cut to the plate width."""
        Rb, t = self.ball_radius(p), p["plate_thickness"]
        Rc = self.coil_radius(p)
        half = self.coil_angle_deg(p) / 2
        centre = (0, 0, -Rc)                                   # the ball's centre, below the belly
        shell = cq.Workplane("XY").sphere(Rb).translate(centre).cut(cq.Workplane("XY").sphere(Rb - t).translate(centre))
        # the wedge: a big triangle in XZ with its apex at the ball's centre, +/- half the coil angle about +z, extruded across y
        L = 2 * Rb
        wedge = (cq.Workplane("XZ").polyline([(0, -Rc), (L * math.sin(math.radians(half)), -Rc + L * math.cos(math.radians(half))),
                                              (-L * math.sin(math.radians(half)), -Rc + L * math.cos(math.radians(half)))]).close()
                 .extrude(p["plate_width"] / 2, both=True))
        plate = shell.intersect(wedge)
        # keep only the part above the body's top (a cap), with a small gap at the wedge edges so neighbours do not touch
        plate = plate.intersect(cq.Workplane("XY").box(length + p["joint_gap"] * 2, p["plate_width"], Rb).translate((0, 0, Rb / 2 + p["body_diameter"] / 2 - 10)))
        apex = Rb - Rc                                          # the crown of this plate
        pad = cq.Workplane("XY").circle(12).extrude(5).translate((0, 0, apex - 3))   # impact pad: a flat on the crown (also the FEA load region)
        return plate.union(pad)

    def pad_z(self, p):
        return self.ball_radius(p) - self.coil_radius(p) + 2.0

    def _leg(self, p, hip_deg=None, knee_deg=None):
        d, f, tb = p["leg_diameter"], p["femur_length"], p["tibia_length"]
        a = math.radians(p["hip_angle_deg"] if hip_deg is None else hip_deg)
        b = math.radians(p["knee_angle_deg"] if knee_deg is None else knee_deg)
        knee = (f * math.cos(a), -f * math.sin(a))
        foot = (knee[0] + tb * math.cos(b), knee[1] - tb * math.sin(b))
        def bar(a_, b_, w, th):
            L = math.hypot(b_[0] - a_[0], b_[1] - a_[1]); ang = math.degrees(math.atan2(b_[1] - a_[1], b_[0] - a_[0]))
            return cq.Workplane("YZ").center((a_[0] + b_[0]) / 2, (a_[1] + b_[1]) / 2).slot2D(L + w, w, ang).extrude(th / 2, both=True)
        hip = cq.Workplane("YZ").circle(d / 2 + 4).extrude(10, both=True)
        leg = hip.union(bar((0, 0), knee, d, d)).union(bar(knee, foot, d * 0.8, d * 0.8))
        leg = leg.union(cq.Workplane("YZ").center(*knee).circle(d / 2 + 2).extrude(d / 2 + 2, both=True))
        leg = leg.union(cq.Workplane("XY").sphere(p["foot_diameter"] / 2).translate((0, foot[0], foot[1])))
        return leg.cut(cq.Workplane("YZ").circle(p["joint_pin_diameter"] / 2).extrude(20, both=True))

    def _necks(self, p, shape, length):
        g, dj = p["joint_gap"], p["joint_diameter"]
        for sx in (-1, 1):
            r = dj / 2 - 4.0 if sx > 0 else dj / 2
            shape = shape.union(cq.Workplane("YZ").circle(r).extrude(g / 2 + 3).translate((sx * length / 2 - (0 if sx > 0 else g / 2 + 3), 0, 0)))
        pin = cq.Workplane("XY").center(length / 2 + g / 4 + 1.5, 0).circle(p["joint_pin_diameter"] / 2).extrude(dj * 2, both=True)
        return shape.cut(pin).cut(pin.mirror("YZ"))

    def _segment(self, p, with_plate=True):
        Ls, D = p["seg_length"], p["body_diameter"]
        seg = self._body(p, Ls)
        boss_r = p["leg_diameter"] / 2 + 6
        for sy in (-1, 1):
            for sx in (-1, 1):
                seg = seg.union(cq.Workplane("XZ").center(sx * self.hip_x(p), -D * 0.15).circle(boss_r).extrude(8)
                                .translate((0, sy * (D / 2 - 6 + (8 if sy > 0 else 0)), 0)))
        seg = self._necks(p, seg, Ls)
        # a round sensor port on each side (the picture's blue eyes), a service hatch under the plate
        seg = seg.cut(cq.Workplane("XZ").circle(9).extrude(D, both=True).translate((0, 0, D * 0.1)))
        if with_plate:
            plate = self._plate(p, Ls)
            studs = cq.Workplane("XY").pushPoints([(sx * Ls * 0.3, sy * p["plate_width"] * 0.3) for sx in (-1, 1) for sy in (-1, 1)]).circle(4).extrude(p["plate_standoff"] + 6).translate((0, 0, D / 2 - 6))
            seg = seg.union(plate).union(studs)
        return seg

    def _head(self, p):
        Lh, D = p["head_length"], p["body_diameter"]
        head = self._body(p, Lh)
        # a domed nose with the sensor suite: stereo cameras, lidar, illuminators
        nose = (cq.Workplane("XY").sphere(D / 2 - 1.0).translate((Lh / 2 - 2.0, 0, 0))
                .intersect(cq.Workplane("XY").box(D, D + 2, D + 2).translate((Lh / 2 - 2.0 + D / 2, 0, 0))))
        head = head.union(nose)
        eyes = cq.Workplane("YZ").pushPoints([(-D * 0.22, D * 0.1), (D * 0.22, D * 0.1)]).circle(10).extrude(30).translate((Lh / 2 + D * 0.3, 0, 0))
        lidar = cq.Workplane("XY").circle(14).extrude(22).translate((Lh * 0.1, 0, D / 2 - 2))
        antenna = cq.Workplane("XY").circle(2.5).extrude(120).translate((-Lh * 0.3, 0, D / 2))
        head = head.cut(eyes).union(lidar).union(antenna)
        g, dj = p["joint_gap"], p["joint_diameter"]
        neck = cq.Workplane("YZ").circle(dj / 2).extrude(g / 2 + 3).translate((-Lh / 2 - g / 2 - 3, 0, 0))
        pin = cq.Workplane("XY").center(-Lh / 2 - g / 4 - 1.5, 0).circle(p["joint_pin_diameter"] / 2).extrude(dj * 2, both=True)
        head = head.union(neck).cut(pin)
        plate = self._plate(p, Lh)
        return head.union(plate)

    def frames(self, p):
        """(position, pitch about y) of every link, head first; walk: a straight line; ball: a ring curled nose-under."""
        n, L = self.n_links(p), self.pitch_length(p)
        if p["mode"] == "walk":
            out, x = [], 0.0
            for i in range(n):
                step = 0.0 if i == 0 else ((p["head_length"] + p["seg_length"]) / 2 + p["joint_gap"] if i == 1 else L)
                x -= step
                out.append((np.array([x, 0.0, 0.0]), 0.0))
            return out
        Rc, da = self.coil_radius(p), self.coil_angle_deg(p)
        C = np.array([0.0, 0.0, -Rc])
        out = []
        for i in range(n):
            a = -i * da                                        # the chain curls under: each link behind is pitched further nose-down... seen from +y
            ar = math.radians(a)
            pos = C + Rc * np.array([-math.sin(ar), 0.0, math.cos(ar)])
            out.append((pos, a))
        return out

    def _place(self, shape, pos, pitch):
        return shape.rotate((0, 0, 0), (0, 1, 0), -pitch).translate(tuple(float(v) for v in pos))

    def _assembly(self, p):
        ball = p["mode"] == "ball"
        hip, knee = (70.0, 150.0) if ball else (None, None)   # ball: legs folded under the belly into the ring
        leg = self._leg(p, hip, knee)
        D, hx = p["body_diameter"], self.hip_x(p)
        legs = None
        for sx in (-1, 1):
            l = leg.translate((sx * hx, D / 2 + 2, -D * 0.15)); r = leg.mirror("XZ").translate((sx * hx, -D / 2 - 2, -D * 0.15))
            pair = l.union(r); legs = pair if legs is None else legs.union(pair)
        seg = self._segment(p).union(legs)
        head = self._head(p)
        body = None
        for i, (pos, pitch) in enumerate(self.frames(p)):
            link = head if i == 0 else seg
            placed = self._place(link, pos, pitch)
            body = placed if body is None else body.union(placed)
        if not ball:                                            # walk mode: belly (feet) on the ground
            reach = self.leg_reach(p)
            body = body.translate((0, 0, D * 0.15 + reach["drop_z"]))
        return body

    def build(self, p):
        if p["part"] == "segment":
            return self._segment(p)
        if p["part"] == "plate":
            return self._plate(p, p["seg_length"])
        if p["part"] == "leg":
            return self._leg(p)
        if p["part"] == "head":
            return self._head(p)
        return self._assembly(p)
