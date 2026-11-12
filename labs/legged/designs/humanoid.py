import math
import cadquery as cq
from vegeta.dedalus import Design, Parameter


class Humanoid(Design):
    """Biped humanoid robot ("Talia"): a pelvis block, a torso with a chest plate and a battery bay, a neck and a
    head with two camera eyes, two arms (shoulder boss, upper arm, elbow, forearm, gripper paddle) and two legs
    (hip boss, thigh, knee boss, shin, ankle and a flat foot), standing with slightly bent knees. x forward,
    y left, z up, ground at z = 0. ``part`` selects the whole robot, the trunk (pelvis + torso + head), one thigh,
    one shin or one foot (limbs in their own frame: the upper joint at the origin, hanging down -z, the joint
    axis along y) for FEA and printing."""

    parameters = [
        Parameter("part", "humanoid", choices=("humanoid", "trunk", "thigh", "shin", "foot"), description="what to build"),
        Parameter("pelvis_width", 260.0, "mm", min=80, description="hip to hip, outside"),
        Parameter("pelvis_depth", 150.0, "mm", min=40),
        Parameter("pelvis_height", 110.0, "mm", min=30),
        Parameter("torso_width", 320.0, "mm", min=100, description="at the shoulders"),
        Parameter("torso_depth", 180.0, "mm", min=50),
        Parameter("torso_height", 420.0, "mm", min=100, description="waist to shoulders"),
        Parameter("waist_taper", 0.75, "", min=0.4, max=1.0, description="waist width / shoulder width"),
        Parameter("shell_fillet", 20.0, "mm", min=1),
        Parameter("hip_spacing", 180.0, "mm", min=40, description="between the two leg planes"),
        Parameter("thigh_length", 380.0, "mm", min=80, description="hip to knee"),
        Parameter("shin_length", 380.0, "mm", min=80, description="knee to ankle"),
        Parameter("leg_width", 80.0, "mm", min=15, description="thigh width at the hip; the shin is 80 %"),
        Parameter("leg_thickness", 60.0, "mm", min=10),
        Parameter("joint_boss_diameter", 90.0, "mm", min=15, description="hip and knee actuator housings"),
        Parameter("ankle_height", 70.0, "mm", min=20, description="ankle axis above the sole"),
        Parameter("foot_length", 240.0, "mm", min=60),
        Parameter("foot_width", 110.0, "mm", min=30),
        Parameter("foot_thickness", 30.0, "mm", min=8),
        Parameter("knee_bend_deg", 15.0, "deg", min=0, max=60, description="thigh ahead of vertical; the shin returns the ankle under the hip"),
        Parameter("upper_arm_length", 300.0, "mm", min=50),
        Parameter("forearm_length", 280.0, "mm", min=50),
        Parameter("arm_diameter", 70.0, "mm", min=15),
        Parameter("elbow_bend_deg", 20.0, "deg", min=0, max=140, description="forearm ahead of the upper arm"),
        Parameter("neck_length", 70.0, "mm", min=10),
        Parameter("head_width", 170.0, "mm", min=40),
        Parameter("head_depth", 190.0, "mm", min=40),
        Parameter("head_height", 210.0, "mm", min=40),
        Parameter("eye_diameter", 28.0, "mm", min=4),
        Parameter("pin_diameter", 14.0, "mm", min=2, description="joint bores"),
        Parameter("battery_bay_depth", 60.0, "mm", min=0, description="pocket in the back of the torso (0 = none)"),
    ]

    # ---- derived geometry (kept here so CAD and mechanics agree) ----
    @staticmethod
    def hip_height(p):
        """Hip axis above the ground, standing: the thigh ahead of vertical by knee_bend, the shin back under the hip."""
        a = math.radians(p["knee_bend_deg"])
        return (p["thigh_length"] + p["shin_length"]) * math.cos(a) + p["ankle_height"]

    @staticmethod
    def total_height(p):
        return (Humanoid.hip_height(p) + p["pelvis_height"] / 2 + p["torso_height"] + p["neck_length"]
                + p["head_height"])

    def _thigh(self, p):
        L, w, t = p["thigh_length"], p["leg_width"], p["leg_thickness"]
        plate = (cq.Workplane("XZ").polyline([(-w / 2, 0), (w / 2, 0), (w * 0.4, -L), (-w * 0.4, -L)]).close()
                 .extrude(t / 2, both=True))
        bosses = (cq.Workplane("XZ").circle(p["joint_boss_diameter"] / 2).extrude(t / 2, both=True)
                  .union(cq.Workplane("XZ").center(0, -L).circle(w * 0.4 + 4).extrude(t / 2, both=True)))
        pocket = cq.Workplane("XZ").center(0, -L / 2).rect(w * 0.35, L * 0.5).extrude(t * 0.3, both=True)
        leg = plate.union(bosses).cut(pocket.translate((0, t / 2, 0))).cut(pocket.translate((0, -t / 2, 0)))
        holes = (cq.Workplane("XZ").circle(p["pin_diameter"] / 2).extrude(t, both=True)
                 .union(cq.Workplane("XZ").center(0, -L).circle(p["pin_diameter"] / 2).extrude(t, both=True)))
        return leg.cut(holes)

    def _shin(self, p):
        L, w, t = p["shin_length"], p["leg_width"] * 0.8, p["leg_thickness"]
        knee = cq.Workplane("XZ").circle(w / 2 + 4).extrude(t / 2, both=True)
        rod = (cq.Workplane("XY").rect(w, t * 0.9).workplane(offset=-L).rect(w * 0.6, t * 0.7).loft(ruled=True))
        ankle = cq.Workplane("XZ").center(0, -L).circle(w * 0.35).extrude(t / 2, both=True)
        leg = knee.union(rod).union(ankle)
        return leg.cut(cq.Workplane("XZ").circle(p["pin_diameter"] / 2).extrude(t, both=True)) \
                  .cut(cq.Workplane("XZ").center(0, -L).circle(p["pin_diameter"] / 2).extrude(t, both=True))

    def _foot(self, p):
        """Foot in its own frame: the ankle axis at the origin, the sole at z = -ankle_height, toe at +x."""
        Lf, Wf, Tf, ha = p["foot_length"], p["foot_width"], p["foot_thickness"], p["ankle_height"]
        sole = (cq.Workplane("XY").box(Lf, Wf, Tf).edges("|Z").fillet(min(Wf / 2 - 1, 25.0))
                .translate((Lf * 0.2, 0, -ha + Tf / 2)))
        bracket = cq.Workplane("XZ").rect(p["leg_width"] * 0.5, ha * 2).extrude(Wf * 0.3, both=True) \
            .intersect(cq.Workplane("XY").box(Lf, Wf, ha).translate((0, 0, -ha / 2)))
        foot = sole.union(bracket).union(cq.Workplane("XZ").circle(p["leg_width"] * 0.28).extrude(Wf * 0.3, both=True))
        return foot.cut(cq.Workplane("XZ").circle(p["pin_diameter"] / 2).extrude(Wf, both=True))

    def _arm(self, p, side):
        """One arm hanging from its shoulder at the origin, the elbow bent forward."""
        d, L1, L2 = p["arm_diameter"], p["upper_arm_length"], p["forearm_length"]
        e = math.radians(p["elbow_bend_deg"])
        shoulder = cq.Workplane("XZ").circle(d * 0.65).extrude(d * 0.45, both=True)
        upper = cq.Workplane("XY").circle(d / 2).extrude(-L1)
        elbow = cq.Workplane("XY").sphere(d * 0.48).translate((0, 0, -L1))
        fore = (cq.Workplane("XY").circle(d * 0.42).extrude(-L2)
                .rotate((0, 0, 0), (0, 1, 0), -p["elbow_bend_deg"]).translate((0, 0, -L1)))
        hx, hz = L2 * math.sin(e), -L1 - L2 * math.cos(e)
        hand = (cq.Workplane("XY").box(d * 0.9, d * 0.35, d * 1.3).edges("|Y").fillet(d * 0.15)
                .rotate((0, 0, 0), (0, 1, 0), -p["elbow_bend_deg"]).translate((hx, 0, hz)))
        return shoulder.union(upper).union(elbow).union(fore).union(hand)

    def _trunk(self, p):
        hh = self.hip_height(p)
        Wp, Dp, Hp = p["pelvis_width"], p["pelvis_depth"], p["pelvis_height"]
        pelvis = cq.Workplane("XY").box(Dp, Wp, Hp).edges().fillet(min(p["shell_fillet"], Hp / 2 - 1)).translate((0, 0, hh))
        # hip actuator housings below the pelvis
        for sy in (-1, 1):
            pelvis = pelvis.union(cq.Workplane("XZ").circle(p["joint_boss_diameter"] / 2).extrude(p["leg_thickness"] * 0.4, both=True)
                                  .translate((0, sy * (p["hip_spacing"] / 2 - p["leg_thickness"] * 0.9), hh)))
        z0 = hh + Hp / 2
        Wt, Dt, Ht = p["torso_width"], p["torso_depth"], p["torso_height"]
        torso = (cq.Workplane("XY").rect(Dt * 0.85, Wt * p["waist_taper"]).workplane(offset=Ht).rect(Dt, Wt)
                 .loft(ruled=True).translate((0, 0, z0)))
        # chest plate: a raised panel on the front
        chest = cq.Workplane("YZ").rect(Wt * 0.6, Ht * 0.35).extrude(8).translate((Dt / 2 - 4, 0, z0 + Ht * 0.68))
        trunk = pelvis.union(torso).union(chest)
        if p["battery_bay_depth"] > 0:
            bay = cq.Workplane("YZ").rect(Wt * 0.5, Ht * 0.5).extrude(p["battery_bay_depth"] + 5) \
                .translate((-Dt / 2 - 5, 0, z0 + Ht * 0.45))
            trunk = trunk.cut(bay)
        # shoulder bosses
        zs = z0 + Ht - p["arm_diameter"] * 0.6
        for sy in (-1, 1):
            trunk = trunk.union(cq.Workplane("XZ").circle(p["arm_diameter"] * 0.6).extrude(25, both=True)
                                .translate((0, sy * (Wt / 2 + 10), zs)))
        # neck and head with two camera eyes and a visor slot
        zn = z0 + Ht
        neck = cq.Workplane("XY").circle(p["head_width"] * 0.22).extrude(p["neck_length"] + 10).translate((0, 0, zn - 5))
        Wh, Dh, Hh = p["head_width"], p["head_depth"], p["head_height"]
        zh = zn + p["neck_length"] + Hh / 2
        head = cq.Workplane("XY").box(Dh, Wh, Hh).edges().fillet(min(35.0, Wh / 2 - 1, Hh / 2 - 1)).translate((0, 0, zh))
        eyes = (cq.Workplane("YZ").pushPoints([(-Wh * 0.2, zh + Hh * 0.08), (Wh * 0.2, zh + Hh * 0.08)])
                .circle(p["eye_diameter"] / 2).extrude(30).translate((Dh / 2 - 15, 0, 0)))
        return trunk.union(neck).union(head.cut(eyes))

    def build(self, p):
        if p["part"] == "thigh":
            return self._thigh(p)
        if p["part"] == "shin":
            return self._shin(p)
        if p["part"] == "foot":
            return self._foot(p)
        if p["hip_spacing"] >= p["pelvis_width"]:
            raise ValueError("hip_spacing must be smaller than the pelvis width")
        trunk = self._trunk(p)
        if p["part"] == "trunk":
            return trunk
        hh = self.hip_height(p)
        a = p["knee_bend_deg"]
        L1 = p["thigh_length"]
        robot = trunk
        for sy in (-1, 1):
            y = sy * p["hip_spacing"] / 2
            # thigh forward by a (rotate about +y by -a), shin back by a: the ankle lands under the hip
            thigh = self._thigh(p).rotate((0, 0, 0), (0, 1, 0), -a).translate((0, y, hh))
            kx, kz = L1 * math.sin(math.radians(a)), hh - L1 * math.cos(math.radians(a))
            shin = self._shin(p).rotate((0, 0, 0), (0, 1, 0), a).translate((kx, y, kz))
            foot = self._foot(p).translate((0, y, p["ankle_height"]))
            zs = hh + p["pelvis_height"] / 2 + p["torso_height"] - p["arm_diameter"] * 0.6
            arm = self._arm(p, sy).translate((0, sy * (p["torso_width"] / 2 + 10 + p["arm_diameter"] * 0.5), zs))
            robot = robot.union(thigh).union(shin).union(foot).union(arm)
        return robot
