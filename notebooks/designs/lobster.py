import math

import cadquery as cq
import numpy as np
from vegeta.dedalus import Design, Parameter
from vegeta.dedalus.examples import Propeller

from onager_manus import OnagerManus


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


class SikarianLobster(Design):
    """Sikarian Lobster: the Sikarian line's underwater walker — a reconnaissance and blocking node for river
    crossings, bridge piers, harbours and outfall pipes. A faceted, free-flooding shell over a sealed aluminium
    **pressure housing** (electronics, battery) trimmed with syntactic foam to sink slightly; **six simple legs** (two
    joints each: hip yaw swings the leg fore and aft, hip pitch lifts it) to stand and walk on the bottom; **two
    pincer arms** at the front (shoulder yaw, shoulder pitch, elbow pitch, wrist pitch, a jaw pair on a pin: the right jaw has a
    hardened cutter near its pin, the left one serrated grip); a **tail of two segments**, each on a yaw + pitch joint,
    ending in a **shrouded propeller**: the tail points the jet, so the one thruster swims, steers, lifts and presses
    the robot onto the bottom. Masts for the radio antenna and the hydrophone array on the roof. x forward, y left, z
    up, the bottom at z = 0, standing.

    The defaults are **Nefri** (v1, the small member: ~6 kg, cuts a Ø10 mm PP rope, walks into a Ø600 mm pipe).

    ``part``: ``robot`` (standing, tail posed by ``tail_yaw_deg`` / ``tail_pitch_deg`` at both joints, claws by
    ``claw_*``), ``shell``, ``housing`` (tube and end caps), ``leg`` (one leg: hip pivot at the origin, reaching +y,
    the foot below), ``upper_arm`` (shoulder bore at the origin, along +x), ``jaw`` (one jaw: pivot bore at the origin,
    inner edge on z = 0 along +x, the cutter notch at ``cutter_x``), ``tail_segment`` (joint at the origin, along −x),
    ``shroud`` (axis +x through the origin), ``propeller`` (axis +x, hub at the origin) and ``cfd`` (the body the CFD
    sees: shell, tail segments and shroud, posed; legs, arms and masts left out — they barely see the jet)."""

    parameters = [
        Parameter("part", "robot", choices=("robot", "shell", "housing", "leg", "upper_arm", "jaw", "tail_segment",
                                            "shroud", "propeller", "cfd"), description="what to build"),
        Parameter("shell_length", 320.0, "mm", min=100, description="the faceted shell (datasheet 0.30-0.34 m)"),
        Parameter("shell_width", 200.0, "mm", min=60),
        Parameter("shell_height", 130.0, "mm", min=40),
        Parameter("shell_thickness", 3.0, "mm", min=1, description="ABS/ASA shell, free-flooding (water inside)"),
        Parameter("shell_chamfer", 28.0, "mm", min=1, description="the armour facets"),
        Parameter("shell_bottom", 75.0, "mm", min=20, description="belly over the bottom, standing"),
        Parameter("housing_diameter", 110.0, "mm", min=30, description="pressure housing, outer (Al 6082 tube)"),
        Parameter("housing_wall", 3.0, "mm", min=1, description="0.1 MPa at 8 m: thin wall, buckling checked in notebook 24 §7"),
        Parameter("housing_length", 230.0, "mm", min=50, description="over the end caps"),
        Parameter("cap_thickness", 8.0, "mm", min=3, description="the two flat end caps (O-ring sealed)"),
        Parameter("hip_spacing", 95.0, "mm", min=20, description="front and rear hips from the middle pair, along x"),
        Parameter("hip_height", 35.0, "mm", min=0, description="hip axes above the shell bottom"),
        Parameter("coxa_length", 22.0, "mm", min=5, description="hip boss outward of the shell side"),
        Parameter("femur_length", 70.0, "mm", min=10, description="hip pitch axis to the knee bend, horizontal"),
        Parameter("tibia_length", 105.0, "mm", min=10, description="knee bend to the foot centre"),
        Parameter("tibia_splay_deg", 10.0, "deg", min=0, max=45, description="tibia outward of vertical"),
        Parameter("leg_width", 16.0, "mm", min=3, description="leg plate depth (in the leg's plane)"),
        Parameter("leg_thickness", 8.0, "mm", min=2, description="leg plate thickness (Al 6082 / PA12-CF)"),
        Parameter("foot_diameter", 20.0, "mm", min=4, description="rubber foot ball"),
        Parameter("claw_y", 55.0, "mm", min=10, description="shoulders either side of the centre line, at the nose"),
        Parameter("claw_height", 70.0, "mm", min=0, description="shoulders above the shell bottom"),
        Parameter("upper_arm_length", 90.0, "mm", min=20),
        Parameter("palm_length", 45.0, "mm", min=10, description="elbow to the wrist"),
        Parameter("hand_length", 22.0, "mm", min=5, description="wrist to the jaw pin (the jaw drive's housing)"),
        Parameter("arm_width", 22.0, "mm", min=5),
        Parameter("jaw_length", 70.0, "mm", min=20, description="jaw pin to the tip"),
        Parameter("jaw_thickness", 6.0, "mm", min=2, description="hardened steel finger plate"),
        Parameter("jaw_depth", 18.0, "mm", min=5),
        Parameter("jaw_pin_diameter", 6.0, "mm", min=2),
        Parameter("cutter_x", 14.0, "mm", min=4, description="the cutter notch from the jaw pin"),
        Parameter("cutter_depth", 4.0, "mm", min=1),
        Parameter("hook_length", 0.0, "mm", min=0, description="hooked tip turned inwards (0: straight jaws — the cut ends of a rope slip off them)"),
        Parameter("hook_width", 8.0, "mm", min=2),
        Parameter("claw_pitch_deg", 20.0, "deg", min=-60, max=80, description="upper arm above horizontal (pose)"),
        Parameter("claw_elbow_deg", 40.0, "deg", min=-30, max=150, description="palm below the upper arm (pose)"),
        Parameter("claw_wrist_deg", 0.0, "deg", min=-90, max=90, description="hand below the palm (pose)"),
        Parameter("jaw_open_deg", 20.0, "deg", min=0, max=60, description="jaw open (pose)"),
        Parameter("tail_height", 70.0, "mm", min=0, description="first tail joint above the shell bottom, at the stern"),
        Parameter("tail_segment_1", 85.0, "mm", min=20, description="first tail segment, joint to joint"),
        Parameter("tail_segment_2", 70.0, "mm", min=20, description="second segment, joint to the shroud's front"),
        Parameter("tail_diameter", 50.0, "mm", min=10),
        Parameter("tail_yaw_deg", 0.0, "deg", min=-60, max=60, description="each tail joint's yaw (pose; + about +z: the tip to the right)"),
        Parameter("tail_pitch_deg", 0.0, "deg", min=-60, max=60, description="each tail joint's pitch (pose; + about +y: the tip up)"),
        Parameter("shroud_length", 60.0, "mm", min=10),
        Parameter("shroud_inner", 82.0, "mm", min=20, description="duct bore"),
        Parameter("shroud_wall", 5.0, "mm", min=1),
        Parameter("prop_diameter", 76.0, "mm", min=10, description="propeller tip to tip (tip gap 3 mm)"),
        Parameter("prop_pitch", 70.0, "mm", min=5),
        Parameter("prop_blades", 3, min=2, max=7),
        Parameter("prop_hub", 20.0, "mm", min=4),
        Parameter("mast_height", 90.0, "mm", min=0, description="antenna and hydrophone masts on the roof"),
        Parameter("mast_diameter", 7.0, "mm", min=2),
    ]

    # ---- derived geometry shared with the Chiron model and the CFD ------------------------------------------------
    @staticmethod
    def hips(p):
        """The six hip pivots [mm], world, standing: {name: (x, y, z, side)}; side +1 left, −1 right."""
        z = p["shell_bottom"] + p["hip_height"]
        y = p["shell_width"] / 2 + p["coxa_length"]
        out = {}
        for k, x in (("F", p["hip_spacing"]), ("M", 0.0), ("R", -p["hip_spacing"])):
            out[f"{k}L"] = (x, y, z, 1)
            out[f"{k}R"] = (x, -y, z, -1)
        return out

    @staticmethod
    def foot_offset(p):
        """The foot centre from its hip pivot [mm] (left leg, joints at 0): (0, out, −down)."""
        s = math.radians(p["tibia_splay_deg"])
        return (0.0, p["femur_length"] + p["tibia_length"] * math.sin(s), -p["tibia_length"] * math.cos(s))

    @staticmethod
    def standing_clearance(p):
        """The foot balls' bottom above z = 0 with the legs at 0 (negative: the belly height must come down)."""
        _, _, dz = SikarianLobster.foot_offset(p)
        return p["shell_bottom"] + p["hip_height"] + dz - p["foot_diameter"] / 2

    @staticmethod
    def tail_frames(p, yaw_deg=None, pitch_deg=None):
        """Tail joints and orientations [mm]: a list of (joint point, rotation matrix) for segment 1 and 2, and the
        propeller (hub point, rotation): each segment along its frame's −x; joint i is a yaw (+z) then a pitch (+y)
        hinge. The propeller's thrust acts along the frame's +x (towards the body), the jet leaves along −x."""
        yaw = math.radians(p["tail_yaw_deg"] if yaw_deg is None else yaw_deg)
        pitch = math.radians(p["tail_pitch_deg"] if pitch_deg is None else pitch_deg)
        p0 = np.array([-p["shell_length"] / 2, 0.0, p["shell_bottom"] + p["tail_height"]])
        R1 = _rz(yaw) @ _ry(pitch)
        p1 = p0 + R1 @ np.array([-p["tail_segment_1"], 0.0, 0.0])
        R2 = R1 @ _rz(yaw) @ _ry(pitch)
        hub = p1 + R2 @ np.array([-p["tail_segment_2"] - p["shroud_length"] / 2, 0.0, 0.0])
        return [(p0, R1), (p1, R2)], (hub, R2)

    @staticmethod
    def thrust_axis(p, yaw_deg=None, pitch_deg=None):
        """(propeller hub [m], unit thrust direction) in the robot frame: the force on the robot; the jet is opposite."""
        _, (hub, R) = SikarianLobster.tail_frames(p, yaw_deg, pitch_deg)
        return hub / 1000.0, R @ np.array([1.0, 0.0, 0.0])

    @staticmethod
    def jaw_params(p):
        """The Manus jaw's parameters at this size (``OnagerManus._jaw`` builds the finger)."""
        return {"jaw_length": p["jaw_length"], "jaw_thickness": p["jaw_thickness"], "jaw_depth": p["jaw_depth"],
                "jaw_pin_diameter": p["jaw_pin_diameter"], "notch_x": p["cutter_x"], "notch_depth": p["cutter_depth"],
                "hook_length": p["hook_length"], "hook_width": p["hook_width"]}

    # ---- parts -----------------------------------------------------------------------------------------------------
    def _shell(self, p):
        L, W, H, t, c = p["shell_length"], p["shell_width"], p["shell_height"], p["shell_thickness"], p["shell_chamfer"]
        outer = cq.Workplane("XY").box(L, W, H).edges("|Y").chamfer(c).edges("|X").chamfer(c * 0.6)
        inner = cq.Workplane("XY").box(L - 2 * t, W - 2 * t, H - 2 * t).edges("|Y").chamfer(max(c - t, 1.0)).edges("|X").chamfer(max(c * 0.6 - t, 1.0))
        shell = outer.cut(inner)
        # free-flooding: drain slots in the belly, an opening for the tail and the claws
        for x in (-L * 0.3, 0.0, L * 0.3):
            shell = shell.cut(cq.Workplane("XY").box(30.0, W * 0.5, 3 * t).translate((x, 0, -H / 2)))
        shell = shell.cut(cq.Workplane("YZ").circle(p["tail_diameter"] / 2 + 4).extrude(3 * t, both=True)
                          .translate((-L / 2, 0, p["tail_height"] - H / 2)))
        # the camera brow on the nose: three lens bosses
        for y in (-W * 0.28, 0.0, W * 0.28):
            shell = shell.union(cq.Workplane("YZ").circle(11.0).extrude(8.0).translate((L / 2 - 6.0, y, H * 0.05)))
        return shell.translate((0, 0, p["shell_bottom"] + H / 2))

    def _housing(self, p):
        r, t, L, tc = p["housing_diameter"] / 2, p["housing_wall"], p["housing_length"], p["cap_thickness"]
        tube = cq.Workplane("YZ").circle(r).circle(r - t).extrude(L / 2 - tc, both=True)
        caps = None
        for s in (1, -1):
            cap = cq.Workplane("YZ").circle(r).extrude(tc / 2, both=True).translate((s * (L / 2 - tc / 2), 0, 0))
            caps = cap if caps is None else caps.union(cap)
        return tube.union(caps)

    def _leg(self, p):
        """Hip pivot at the origin (yaw axis z), the coxa boss, the femur along +y, the tibia down and outward."""
        w, t, F = p["leg_width"], p["leg_thickness"], p["femur_length"]
        boss = cq.Workplane("XY").circle(w * 0.8).extrude(w * 0.8, both=True)
        femur = cq.Workplane("XY").box(t, F, w).translate((0, F / 2, 0))
        s = math.radians(p["tibia_splay_deg"])
        T = p["tibia_length"]
        tibia = (cq.Workplane("XY").box(t, w, T).translate((0, 0, -T / 2))
                 .rotate((0, 0, 0), (1, 0, 0), math.degrees(s)).translate((0, F, 0)))
        knee = cq.Workplane("YZ").circle(w * 0.6).extrude(t / 2, both=True).translate((0, F, 0))
        fx, fy, fz = self.foot_offset(p)
        foot = cq.Workplane("XY").sphere(p["foot_diameter"] / 2).translate((fx, fy, fz))
        return boss.union(femur).union(knee).union(tibia).union(foot)

    def _upper_arm(self, p):
        L, w = p["upper_arm_length"], p["arm_width"]
        arm = cq.Workplane("XY").box(L, w * 0.7, w).translate((L / 2, 0, 0))
        bosses = (cq.Workplane("XZ").circle(w * 0.7).extrude(w * 0.45, both=True)
                  .union(cq.Workplane("XZ").center(L, 0).circle(w * 0.6).extrude(w * 0.45, both=True)))
        return arm.union(bosses)

    def _jaw(self, p):
        return OnagerManus._jaw(self, self.jaw_params(p))

    def _tail_segment(self, L, p):
        d = p["tail_diameter"]
        seg = cq.Workplane("YZ").circle(d / 2).extrude(L).translate((-L, 0, 0))
        knuckle = cq.Workplane("XY").sphere(d * 0.42)
        return seg.union(knuckle)

    def _shroud(self, p):
        r, t, L = p["shroud_inner"] / 2, p["shroud_wall"], p["shroud_length"]
        duct = cq.Workplane("YZ").circle(r + t).circle(r).extrude(L / 2, both=True)
        # a lip on the intake (an accelerating duct: the inlet flares)
        lip = cq.Workplane("YZ").circle(r + t + 3).circle(r + 1).extrude(4.0).translate((L / 2 - 4.0, 0, 0))
        strut = cq.Workplane("XY").box(6.0, 2 * r, 3.0)                       # the hub strut across the bore
        motor = cq.Workplane("YZ").circle(p["prop_hub"] * 0.75).extrude(L * 0.35).translate((0.0, 0, 0))
        return duct.union(lip).union(strut).union(motor)

    def _propeller(self, p):
        prop = Propeller().generate(diameter=p["prop_diameter"], pitch=p["prop_pitch"], blades=int(p["prop_blades"]),
                                    hub_diameter=p["prop_hub"], hub_height=p["prop_hub"] * 0.7, bore=3.0,
                                    chord_root=p["prop_diameter"] * 0.16, chord_max=p["prop_diameter"] * 0.26,
                                    chord_tip=p["prop_diameter"] * 0.12, thickness=0.12, camber=0.05, stations=8)
        return cq.Workplane(obj=prop.shape.rotate((0, 0, 0), (0, 1, 0), 90))          # axis z -> +x

    def _tail(self, p, with_prop=True):
        yaw, pitch = p["tail_yaw_deg"], p["tail_pitch_deg"]
        (j1, _), (j2, _) = self.tail_frames(p)[0]
        hub, _ = self.tail_frames(p)[1]

        def posed(part, n):          # rotations about world axes, applied innermost first: R = Rz Ry (Rz Ry)
            for _ in range(n):
                part = part.rotate((0, 0, 0), (0, 1, 0), pitch).rotate((0, 0, 0), (0, 0, 1), yaw)
            return part
        seg1 = posed(self._tail_segment(p["tail_segment_1"], p), 1).translate(tuple(j1))
        seg2 = posed(self._tail_segment(p["tail_segment_2"], p), 2).translate(tuple(j2))
        shroud = posed(self._shroud(p), 2).translate(tuple(hub))
        out = seg1.union(seg2).union(shroud)
        if with_prop:
            out = out.union(posed(self._propeller(p).translate((-p["shroud_length"] * 0.15, 0, 0)), 2).translate(tuple(hub)))
        return out

    def _claw(self, p, side):
        """One pincer arm in its pose, in the world (side +1 left, −1 right)."""
        x0 = p["shell_length"] / 2 - 10.0
        y0 = side * p["claw_y"]
        z0 = p["shell_bottom"] + p["claw_height"]
        a, e, w = p["claw_pitch_deg"], p["claw_elbow_deg"], p["claw_wrist_deg"]
        Lu, Lp, Lh = p["upper_arm_length"], p["palm_length"], p["hand_length"]
        upper = self._upper_arm(p).rotate((0, 0, 0), (0, 1, 0), -a).translate((x0, y0, z0))
        elbow = (x0 + Lu * math.cos(math.radians(a)), y0, z0 + Lu * math.sin(math.radians(a)))
        down = e - a                                                     # palm below horizontal
        palm = (cq.Workplane("XY").box(Lp, p["arm_width"] * 0.8, p["arm_width"] * 1.1).translate((Lp / 2, 0, 0))
                .rotate((0, 0, 0), (0, 1, 0), down).translate(elbow))
        wrist = (elbow[0] + Lp * math.cos(math.radians(down)), y0, elbow[2] - Lp * math.sin(math.radians(down)))
        hdown = down + w                                                 # hand (and jaws) below horizontal
        hand = (cq.Workplane("XZ").circle(p["arm_width"] * 0.55).extrude(p["arm_width"] * 0.4, both=True)
                .union(cq.Workplane("XY").box(Lh, p["arm_width"] * 0.7, p["arm_width"]).translate((Lh / 2, 0, 0)))
                .rotate((0, 0, 0), (0, 1, 0), hdown).translate(wrist))
        pin = (wrist[0] + Lh * math.cos(math.radians(hdown)), y0, wrist[2] - Lh * math.sin(math.radians(hdown)))
        jaws = None
        for flip in (1, -1):
            j = self._jaw(p).translate((0, 0, -p["jaw_depth"] / 2))
            if flip < 0:
                j = j.mirror("XY")
            j = j.rotate((0, 0, 0), (0, 1, 0), -flip * p["jaw_open_deg"]).rotate((0, 0, 0), (0, 1, 0), hdown).translate(pin)
            jaws = j if jaws is None else jaws.union(j)
        return upper.union(palm).union(hand).union(jaws)

    def build(self, p):
        part = p["part"]
        if part == "shell":
            return self._shell(p)
        if part == "housing":
            return self._housing(p)
        if part == "leg":
            return self._leg(p)
        if part == "upper_arm":
            return self._upper_arm(p)
        if part == "jaw":
            return self._jaw(p)
        if part == "tail_segment":
            return self._tail_segment(p["tail_segment_1"], p)
        if part == "shroud":
            return self._shroud(p)
        if part == "propeller":
            return self._propeller(p)
        if p["housing_diameter"] >= p["shell_height"] - 2 * p["shell_thickness"]:
            raise ValueError("the pressure housing must fit inside the shell (housing_diameter < shell_height − 2 t)")
        if p["housing_length"] >= p["shell_length"] - 2 * p["shell_chamfer"]:
            raise ValueError("the pressure housing must fit inside the shell (housing_length < shell_length − 2 chamfer)")
        shell = self._shell(p)
        if part == "cfd":
            solid = cq.Workplane("XY").box(p["shell_length"], p["shell_width"], p["shell_height"]).edges("|Y").chamfer(p["shell_chamfer"]) \
                .edges("|X").chamfer(p["shell_chamfer"] * 0.6).translate((0, 0, p["shell_bottom"] + p["shell_height"] / 2))
            return solid.union(self._tail(p, with_prop=False))
        zc = p["shell_bottom"] + p["shell_height"] / 2
        robot = shell.union(self._housing(p).translate((0, 0, zc)))
        for name, (x, y, z, side) in self.hips(p).items():
            leg = self._leg(p)
            if side < 0:
                leg = leg.mirror("XZ")
            robot = robot.union(leg.translate((x, y, z)))
        for side in (1, -1):
            robot = robot.union(self._claw(p, side))
        robot = robot.union(self._tail(p))
        roof = p["shell_bottom"] + p["shell_height"]
        if p["mast_height"] > 0:
            for x, y, h in ((-p["shell_length"] * 0.15, p["shell_width"] * 0.2, 1.0), (-p["shell_length"] * 0.05, -p["shell_width"] * 0.2, 0.8)):
                robot = robot.union(cq.Workplane("XY").circle(p["mast_diameter"] / 2).extrude(p["mast_height"] * h).translate((x, y, roof - 2)))
            robot = robot.union(cq.Workplane("XY").box(30, 24, 14).translate((p["shell_length"] * 0.12, 0, roof + 5)))   # hydrophone puck / sonar
        return robot
