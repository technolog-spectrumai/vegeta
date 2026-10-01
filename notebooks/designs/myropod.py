import math
import cadquery as cq
import numpy as np
from vegeta.dedalus import Design, Parameter


class Myropod(Design):
    """The Myropod family: segmented crawlers for chimneys, flues, ducts and tight dangerous spaces. A
    chain of identical hollow shell segments joined by two-axis (pitch + yaw) joints; every segment
    carries **two pairs of two-link legs** (four legs, like a myriapod), so each segment can stand and
    brace on its own and the controller can treat it as one simple unit. A head with two lights and a
    camera, a tether gland at the tail. The default parameters are **Persephone**, the chimney member
    of the family (44 x 34 mm segments, 12 of them); other members change the segment count and size.

    ``version``: 1 = the crawler itself; 2 = transport kit (the crawler coiled around its tether drum in
    a carry case, plus the flue-mouth insertion guide as ``part="guide"``); 3 = version 1 plus two
    pincers on the head. ``bend_yaw_deg`` / ``bend_pitch_deg`` bend every joint from ``bend_from`` on
    (a flue elbow, or 360/n to coil). x along the body (head at +x), y left, z up; the straight
    crawler lies with its belly at z = 0 and the feet below.

    ``part`` selects the whole crawler (as its version), a segment, a leg (hip bore at the origin,
    reaching +y), the head, one pincer (pivot bore at the origin, jaw along +x), the carry case or
    the insertion guide."""

    parameters = [
        Parameter("part", "crawler", choices=("crawler", "segment", "leg", "head", "pincer", "case", "guide"), description="what to build"),
        Parameter("version", 1, "", choices=(1, 2, 3), description="1 crawler, 2 transport kit (coiled in the carry case), 3 with pincers"),
        Parameter("n_segments", 12, "", min=3, max=40, description="body segments behind the head"),
        Parameter("seg_length", 60.0, "mm", min=15, description="shell length of one segment"),
        Parameter("seg_width", 44.0, "mm", min=10),
        Parameter("seg_height", 34.0, "mm", min=8),
        Parameter("shell_thickness", 2.5, "mm", min=0.5),
        Parameter("joint_gap", 10.0, "mm", min=3, description="neck between two segments (the joint)"),
        Parameter("joint_diameter", 16.0, "mm", min=3, description="neck diameter"),
        Parameter("joint_pin_diameter", 5.0, "mm", min=1),
        Parameter("femur_length", 36.0, "mm", min=5, description="hip to knee"),
        Parameter("tibia_length", 42.0, "mm", min=5, description="knee to the foot pad"),
        Parameter("leg_diameter", 9.0, "mm", min=1.5),
        Parameter("foot_diameter", 9.0, "mm", min=2),
        Parameter("hip_angle_deg", 20.0, "deg", min=-60, max=80, description="femur below horizontal, reaching outwards"),
        Parameter("knee_angle_deg", 70.0, "deg", min=0, max=150, description="tibia below horizontal"),
        Parameter("leg_sweep_deg", 0.0, "deg", min=-60, max=60, description="legs swept forward (+) in the top view"),
        Parameter("leg_pair_spacing", 0.5, "", min=0.2, max=0.9, description="distance between the two leg pairs of a segment, as a fraction of its length"),
        Parameter("head_length", 80.0, "mm", min=20),
        Parameter("tether_diameter", 8.0, "mm", min=1),
        Parameter("tether", True, "", description="tether gland and a stub of tether at the tail (Persephone); False for untethered members"),
        Parameter("bend_yaw_deg", 0.0, "deg", min=-60, max=60, description="yaw at every joint from bend_from on"),
        Parameter("bend_pitch_deg", 0.0, "deg", min=-60, max=60, description="pitch (nose down +) at every joint from bend_from on"),
        Parameter("bend_from", 0, "", min=0, description="first joint that bends (0 = head joint)"),
        Parameter("pincer_length", 55.0, "mm", min=10),
        Parameter("pincer_thickness", 6.0, "mm", min=1),
        Parameter("pincer_open_deg", 25.0, "deg", min=0, max=90, description="each jaw rotated out from closed"),
        Parameter("case_wall", 3.0, "mm", min=1),
        Parameter("case_clearance", 6.0, "mm", min=0, description="radial gap around the coiled crawler"),
        Parameter("drum_diameter", 120.0, "mm", min=20, description="tether drum in the middle of the case"),
        Parameter("guide_diameter", 150.0, "mm", min=40, description="inner diameter of the insertion guide (half-pipe)"),
        Parameter("guide_radius", 220.0, "mm", min=50, description="bend radius of the insertion guide"),
    ]

    # ---- kinematics shared with the notebook ----
    @staticmethod
    def pitch_length(p):
        return p["seg_length"] + p["joint_gap"]

    @staticmethod
    def hip_x(p):
        """The two leg pairs of a segment sit at +/- hip_x from its centre."""
        return p["seg_length"] * p["leg_pair_spacing"] / 2

    @staticmethod
    def coil_yaw_deg(p):
        return 360.0 / p["n_segments"]

    @staticmethod
    def leg_reach(p):
        """Horizontal (y) and vertical (z, positive down) reach of one leg from its hip, plus the stance width."""
        a, b = math.radians(p["hip_angle_deg"]), math.radians(p["knee_angle_deg"])
        y_knee = p["femur_length"] * math.cos(a)
        y = y_knee + p["tibia_length"] * math.cos(b)
        z = p["femur_length"] * math.sin(a) + p["tibia_length"] * math.sin(b)
        return {"reach_y": max(y, y_knee), "foot_y": y, "drop_z": z, "stance_width": p["seg_width"] + 2 * y + p["foot_diameter"]}

    def frames(self, p):
        """Position and (yaw, pitch) of every link: index 0 is the head, 1..n the segments."""
        n, L = p["n_segments"], self.pitch_length(p)
        pos, yaw, pitch = np.zeros(3), 0.0, 0.0
        out = [(pos.copy(), yaw, pitch)]
        for j in range(n):                                   # joint j sits between link j and link j+1
            if j >= p["bend_from"]:
                yaw += p["bend_yaw_deg"]; pitch += p["bend_pitch_deg"]
            y, t = math.radians(yaw), math.radians(pitch)
            d = np.array([math.cos(y) * math.cos(t), math.sin(y) * math.cos(t), -math.sin(t)])
            step = (p["head_length"] + p["seg_length"]) / 2 + p["joint_gap"] if j == 0 else L
            pos = pos - d * step                              # the chain grows towards -x behind the head
            out.append((pos.copy(), yaw, pitch))
        return out

    # ---- parts ----
    def _segment(self, p):
        Ls, W, H, t = p["seg_length"], p["seg_width"], p["seg_height"], p["shell_thickness"]
        zc = H / 2
        shell = cq.Workplane("XY").box(Ls, W, H).translate((0, 0, zc)).edges("|X").fillet(min(6.0, H / 3))
        shell = shell.cut(cq.Workplane("XY").box(Ls - 2 * t, W - 2 * t, H - 2 * t).translate((0, 0, zc)))
        # armour plate on top (the picture's riveted hatch), a stiffening rib on the belly
        hatch = cq.Workplane("XY").box(Ls * 0.6, W * 0.45, 2.0).translate((0, 0, H + 1.0))
        rib = cq.Workplane("XY").box(Ls - 6, 6, 3.0).translate((0, 0, -1.5))
        shell = shell.union(hatch).union(rib)
        # hip bosses on both sides (leg motors), joint necks on both ends, with pin bores across them
        boss_r = p["leg_diameter"] / 2 + 4
        for sy in (-1, 1):
            for sx in (-1, 1):                                # two leg pairs per segment
                x = sx * self.hip_x(p)
                shell = shell.union(cq.Workplane("XZ").center(x, zc).circle(boss_r).extrude(6).translate((0, sy * (W / 2 + (6 if sy > 0 else 0)), 0)))
        g, dj = p["joint_gap"], p["joint_diameter"]
        for sx in (-1, 1):                                    # front: the tongue (thinner); back: the fork (full diameter)
            r = dj / 2 - 2.0 if sx > 0 else dj / 2
            neck = cq.Workplane("YZ").center(0, zc).circle(r).extrude(g / 2 + 2).translate((sx * (Ls / 2) - (0 if sx > 0 else g / 2 + 2), 0, 0))
            shell = shell.union(neck)
        pin = cq.Workplane("XY").center(Ls / 2 + g / 4 + 1.0, 0).circle(p["joint_pin_diameter"] / 2).extrude(H * 2).translate((0, 0, -H / 2))   # clear of the shell end face
        return shell.cut(pin).cut(pin.mirror("YZ"))

    def _leg(self, p):
        """One leg in its own frame: hip bore at the origin (axis x), femur reaching +y and down."""
        d, f, tb = p["leg_diameter"], p["femur_length"], p["tibia_length"]
        a, b = math.radians(p["hip_angle_deg"]), math.radians(p["knee_angle_deg"])
        knee = (f * math.cos(a), -f * math.sin(a))
        foot = (knee[0] + tb * math.cos(b), knee[1] - tb * math.sin(b))
        hip = cq.Workplane("YZ").circle(d / 2 + 2).extrude(5, both=True)
        def bar(a, b, w, th):                                  # a rounded bar from a to b in the YZ plane, thickness along x
            L = math.hypot(b[0] - a[0], b[1] - a[1])
            ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
            return cq.Workplane("YZ").center((a[0] + b[0]) / 2, (a[1] + b[1]) / 2).slot2D(L + w, w, ang).extrude(th / 2, both=True)
        femur = bar((0, 0), knee, d, d)
        tibia = bar(knee, foot, d * 0.8, d * 0.8)
        kneeb = cq.Workplane("YZ").center(*knee).circle(d / 2 + 1).extrude(d / 2 + 1, both=True)
        pad = cq.Workplane("XY").sphere(p["foot_diameter"] / 2).translate((0, foot[0], foot[1]))
        leg = hip.union(femur).union(tibia).union(kneeb).union(pad)
        return leg.cut(cq.Workplane("YZ").circle(p["joint_pin_diameter"] / 2).extrude(10, both=True))

    def _head(self, p):
        Lh, W, H, t = p["head_length"], p["seg_width"], p["seg_height"], p["shell_thickness"]
        zc = H / 2
        head = (cq.Workplane("YZ").rect(W, H).workplane(offset=Lh).rect(W * 0.85, H * 0.85).loft().translate((-Lh / 2, 0, zc))
                .edges("not |Y").fillet(3.0))
        head = head.cut(cq.Workplane("YZ").rect(W - 2 * t, H - 2 * t).workplane(offset=Lh - 2 * t).rect(W * 0.85 - 2 * t, H * 0.85 - 2 * t).loft()
                        .translate((-Lh / 2 + t, 0, zc)))
        # two lights and a camera on the front face, a short antenna on top, the neck at the back
        front = Lh / 2
        eyes = cq.Workplane("YZ").pushPoints([(-W * 0.28, zc), (W * 0.28, zc)]).circle(4.5).extrude(10, both=True).translate((front - 2, 0, 0))
        cam = cq.Workplane("YZ").center(0, zc).rect(12, 9).extrude(10, both=True).translate((front - 2, 0, 0))
        antenna = cq.Workplane("XY").circle(1.5).extrude(40).translate((-Lh * 0.2, 0, H))
        neck = cq.Workplane("YZ").center(0, zc).circle(p["joint_diameter"] / 2).extrude(p["joint_gap"] / 2 + 2).translate((-Lh / 2 - p["joint_gap"] / 2 - 2, 0, 0))
        head = head.cut(eyes).cut(cam).union(antenna).union(neck)
        pin = cq.Workplane("XY").center(-Lh / 2 - p["joint_gap"] / 4 - 1.0, 0).circle(p["joint_pin_diameter"] / 2).extrude(2 * H).translate((0, 0, -H / 2))
        return head.cut(pin)

    def _pincer(self, p):
        """One jaw: pivot bore at the origin (axis z), a curved arm along +x closing towards -y, with teeth."""
        L, th = p["pincer_length"], p["pincer_thickness"]
        w0, w1 = 12.0, 5.0
        pts = [(0, -w0 / 2), (L * 0.55, -w0 / 2 + 2), (L, -w1 / 2 - 6), (L, w1 / 2 - 6), (L * 0.55, w0 / 2 - 2), (0, w0 / 2)]
        jaw = cq.Workplane("XY").polyline(pts).close().extrude(th / 2, both=True)
        jaw = jaw.union(cq.Workplane("XY").circle(w0 / 2 + 2).extrude(th / 2, both=True))
        for i in range(4):                                     # teeth on the inside edge
            x = L * 0.45 + i * L * 0.14
            jaw = jaw.union(cq.Workplane("XY").polyline([(x - 3, -w0 / 2 + 2 - 0.1), (x + 3, -w0 / 2 + 2 - 0.1), (x, -w0 / 2 - 3)]).close().extrude(th / 2, both=True))
        return jaw.cut(cq.Workplane("XY").circle(p["joint_pin_diameter"] / 2).extrude(th, both=True))

    def coil_radius(self, p):
        """Centre-line radius of the crawler coiled with coil_yaw_deg per joint."""
        return self.pitch_length(p) / (2 * math.sin(math.radians(self.coil_yaw_deg(p)) / 2))

    def case_radius(self, p):
        """Inner radius of the carry case around the coiled crawler (legs tucked as the notebook sets them)."""
        reach = self.leg_reach(p)
        return self.coil_radius(p) + p["seg_width"] / 2 + 6 + reach["reach_y"] + p["foot_diameter"] / 2 + p["case_clearance"]

    def _case(self, p):
        """A round carry case: floor, outer wall around the coiled crawler, the tether drum in the middle, a handle, four feet."""
        reach = self.leg_reach(p)
        r_out = self.case_radius(p)
        wall, H = p["case_wall"], p["seg_height"] + reach["drop_z"] + 8
        case = cq.Workplane("XY").circle(r_out + wall).extrude(wall)
        case = case.union(cq.Workplane("XY").circle(r_out + wall).circle(r_out).extrude(H))
        drum = (cq.Workplane("XY").circle(p["drum_diameter"] / 2).circle(p["drum_diameter"] / 2 - 2 * wall).extrude(H)
                .union(cq.Workplane("XY").circle(p["drum_diameter"] / 2 + 10).extrude(wall).translate((0, 0, H - wall))))
        handle = (cq.Workplane("XZ").center(0, H + 30).rect(2 * r_out + 2 * wall, 60).extrude(10, both=True)
                  .cut(cq.Workplane("XZ").center(0, H + 30).rect(2 * r_out - 2 * wall - 20, 40).extrude(12, both=True))
                  .cut(cq.Workplane("XY").circle(r_out - wall).extrude(H).translate((0, 0, 0))))
        pitch = r_out * 0.6
        feet = cq.Workplane("XY").pushPoints([(sx * pitch, sy * pitch) for sx in (-1, 1) for sy in (-1, 1)]).circle(12).extrude(8).translate((0, 0, -8))
        return case.union(drum).union(handle).union(feet)

    def _guide(self, p):
        """Insertion guide: a quarter-bend half-pipe that turns the crawler from the hearth (horizontal) up into the flue.
        The bend axis is y; the trough is the outer half of the bend, the side the body slides on."""
        D, R, wall = p["guide_diameter"], p["guide_radius"], p["case_wall"] + 1
        outer = cq.Workplane("XY").center(R, 0).circle(D / 2 + wall).revolve(90, (-R, 0, 0), (-R, 1, 0))
        inner = cq.Workplane("XY").center(R, 0).circle(D / 2).revolve(90, (-R, 0, 0), (-R, 1, 0))
        tube = outer.cut(inner)
        core = cq.Workplane("XZ").circle(R).extrude(D, both=True)             # everything closer to the bend axis than R goes
        return tube.cut(core)

    def _place(self, shape, pos, yaw, pitch):
        return shape.rotate((0, 0, 0), (0, 1, 0), pitch).rotate((0, 0, 0), (0, 0, 1), yaw).translate(tuple(float(v) for v in pos))

    def _assembly(self, p):
        frames = self.frames(p)
        seg, leg, head = self._segment(p), self._leg(p), self._head(p)
        zc = p["seg_height"] / 2
        W = p["seg_width"]
        hx = self.hip_x(p)
        legs = None
        for sx in (-1, 1):
            leg_l = leg.rotate((0, 0, 0), (0, 0, 1), -p["leg_sweep_deg"]).translate((sx * hx, W / 2 + 6, zc))
            leg_r = leg.mirror("XZ").rotate((0, 0, 0), (0, 0, 1), p["leg_sweep_deg"]).translate((sx * hx, -W / 2 - 6, zc))
            pair = leg_l.union(leg_r)
            legs = pair if legs is None else legs.union(pair)
        body = None
        for i, (pos, yaw, pitch) in enumerate(frames):
            if i == 0:
                link = head
                if p["version"] == 3:
                    hx = p["head_length"] / 2 - 6
                    for sy, sgn in ((1, 1), (-1, -1)):
                        jaw = self._pincer(p)
                        if sy < 0:
                            jaw = jaw.mirror("XZ")
                        jaw = jaw.rotate((0, 0, 0), (0, 0, 1), sy * p["pincer_open_deg"]).translate((hx, sy * (W / 2 - 4), zc))
                        pin = cq.Workplane("XY").circle(p["joint_pin_diameter"] / 2 - 0.1).extrude(p["pincer_thickness"] + 8, both=True).translate((hx, sy * (W / 2 - 4), zc))
                        link = link.union(jaw).union(pin)
            else:
                link = seg.union(legs)
                if i == len(frames) - 1 and p["tether"]:        # tail: tether gland and a stub of tether
                    gland = cq.Workplane("YZ").center(0, zc).circle(p["tether_diameter"] / 2 + 3).extrude(12).translate((-p["seg_length"] / 2 - 12, 0, 0))
                    tether = cq.Workplane("YZ").center(0, zc).circle(p["tether_diameter"] / 2).extrude(60).translate((-p["seg_length"] / 2 - 70, 0, 0))
                    link = link.union(gland).union(tether)
            placed = self._place(link, pos, yaw, pitch)
            body = placed if body is None else body.union(placed)
        return body

    def build(self, p):
        if p["part"] == "segment":
            return self._segment(p)
        if p["part"] == "leg":
            return self._leg(p)
        if p["part"] == "head":
            return self._head(p)
        if p["part"] == "pincer":
            return self._pincer(p)
        if p["part"] == "case":
            return self._case(p)
        if p["part"] == "guide":
            return self._guide(p)
        if p["version"] == 2:
            q = dict(p, bend_yaw_deg=self.coil_yaw_deg(p), bend_pitch_deg=0.0, bend_from=0, version=1)
            crawler = self._assembly(q)
            centre = np.mean([f[0] for f in self.frames(q)[1:]], axis=0)       # the ring's centre from the segment positions
            crawler = crawler.translate((-float(centre[0]), -float(centre[1]), self.leg_reach(p)["drop_z"] + p["case_wall"] + 2))
            return self._case(p).union(crawler)
        return self._assembly(p)
