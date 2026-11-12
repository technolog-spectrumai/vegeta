import math
import cadquery as cq
from vegeta.dedalus import Design, Parameter


class OnagerSentinel(Design):
    """Onager Sentinel SX-1: the reconnaissance unit of the Onager series of land robots — a wheel-leg
    hybrid. An armoured hull with a sensor turret and a telescopic mast on its back, four two-link legs
    (shoulder actuator boss on the hull, upper leg plate, knee boss, lower leg plate) each ending in a
    driven wheel (hub motor). Wheel mode: the legs act as an active suspension and the wheels drive;
    walking mode: the legs step with the wheels locked. x forward, y left, z up, ground at z = 0.

    ``part`` selects the whole robot, the hull (with the turret and mast), one upper leg, one lower leg
    (both in their own frame: the joint bore at the origin, the leg hanging down -z, the bore axis along
    y) or one wheel, for FEA and the mass budget."""

    parameters = [
        Parameter("part", "robot", choices=("robot", "hull", "upper_leg", "lower_leg", "wheel"), description="what to build"),
        Parameter("hull_length", 2000.0, "mm", min=400, description="hull length (the datasheet's 2.6 m is over the wheels)"),
        Parameter("hull_width", 780.0, "mm", min=200),
        Parameter("hull_height", 520.0, "mm", min=100),
        Parameter("hull_chamfer", 60.0, "mm", min=1, description="chamfer of the hull edges (the armour facets)"),
        Parameter("hull_bottom", 760.0, "mm", min=100, description="ground clearance of the hull floor when standing"),
        Parameter("shoulder_x", 900.0, "mm", min=50, description="shoulders from the hull centre, fore and aft"),
        Parameter("shoulder_boss_diameter", 220.0, "mm", min=20, description="shoulder actuator housing on the hull side"),
        Parameter("shoulder_boss_length", 90.0, "mm", min=10),
        Parameter("upper_leg_length", 520.0, "mm", min=50, description="shoulder to knee"),
        Parameter("upper_leg_width", 140.0, "mm", min=20, description="plate width at the shoulder"),
        Parameter("upper_leg_thickness", 50.0, "mm", min=5, description="depth of the machined H-section (pockets from both faces)"),
        Parameter("upper_leg_taper", 0.7, "", min=0.3, max=1.0),
        Parameter("lower_leg_length", 540.0, "mm", min=50, description="knee to the wheel axle"),
        Parameter("lower_leg_width", 120.0, "mm", min=20, description="plate width at the knee"),
        Parameter("lower_leg_thickness", 44.0, "mm", min=5),
        Parameter("lower_leg_taper", 0.75, "", min=0.3, max=1.0),
        Parameter("hip_angle_deg", 40.0, "deg", min=0, max=80, description="upper leg behind vertical (knee back) when standing"),
        Parameter("knee_angle_deg", 55.0, "deg", min=0, max=120, description="lower leg ahead of vertical (wheel forward) when standing"),
        Parameter("pin_diameter", 40.0, "mm", min=5, description="shoulder and knee pins"),
        Parameter("wheel_diameter", 560.0, "mm", min=100),
        Parameter("wheel_width", 180.0, "mm", min=20),
        Parameter("hub_diameter", 260.0, "mm", min=20, description="hub motor housing inside the wheel"),
        Parameter("wheel_offset", 30.0, "mm", min=0, description="gap between the lower leg and the wheel"),
        Parameter("axle_diameter", 45.0, "mm", min=5),
        Parameter("turret_length", 500.0, "mm", min=0, description="sensor turret box on the hull roof (0 = none)"),
        Parameter("turret_width", 420.0, "mm", min=0),
        Parameter("turret_height", 220.0, "mm", min=0),
        Parameter("mast_height", 450.0, "mm", min=0, description="telescopic mast above the turret, stowed (0 = none)"),
        Parameter("mast_diameter", 60.0, "mm", min=5),
        Parameter("sensor_head", 150.0, "mm", min=0, description="E/O-IR sensor head cube on the mast"),
    ]

    # ---- derived geometry shared by the notebook (kept here so CAD and mechanics agree) ----
    @staticmethod
    def axle_height(p):
        """Wheel axle below the shoulder when standing [mm]."""
        a1, a2 = math.radians(p["hip_angle_deg"]), math.radians(p["knee_angle_deg"])
        return p["upper_leg_length"] * math.cos(a1) + p["lower_leg_length"] * math.cos(a2)

    @staticmethod
    def axle_x(p):
        """Axle ahead (+) of the shoulder, in the leg's plane [mm]."""
        a1, a2 = math.radians(p["hip_angle_deg"]), math.radians(p["knee_angle_deg"])
        return -p["upper_leg_length"] * math.sin(a1) + p["lower_leg_length"] * math.sin(a2)

    @staticmethod
    def shoulder_height(p):
        """Shoulder axis above the ground when standing [mm]: wheel radius + the leg's vertical reach."""
        return p["wheel_diameter"] / 2 + OnagerSentinel.axle_height(p)

    @staticmethod
    def overall(p):
        """Overall length, width and height [mm] of the standing robot (the datasheet numbers)."""
        ax = OnagerSentinel.axle_x(p)
        length = 2 * p["shoulder_x"] + 2 * abs(ax) + p["wheel_diameter"]
        width = p["hull_width"] + 2 * (p["shoulder_boss_length"] + p["upper_leg_thickness"] + p["lower_leg_thickness"]
                                       + p["wheel_offset"] + p["wheel_width"])
        hull_top = p["hull_bottom"] + p["hull_height"]
        height = hull_top + p["turret_height"] + p["mast_height"] + p["sensor_head"]
        return {"length": length, "width": width, "height": height, "hull_top": hull_top}

    def _plate_leg(self, L, w, t, k, pin, boss_extra, bore_depth):
        """A tapered plate link (XZ plane, thickness along y) with round bosses and pin bores at both ends."""
        plate = (cq.Workplane("XZ").polyline([(-w / 2, 0), (w / 2, 0), (w * k / 2, -L), (-w * k / 2, -L)]).close()
                 .extrude(t / 2, both=True))
        bosses = (cq.Workplane("XZ").circle(w / 2).extrude(t / 2, both=True)
                  .union(cq.Workplane("XZ").center(0, -L).circle(w * k / 2 + boss_extra).extrude(t / 2, both=True)))
        leg = plate.union(bosses)
        # pockets from both faces: a machined H-section (a web of 24 % of the depth between two flanges)
        pocket = cq.Workplane("XZ").center(0, -L / 2).rect(w * 0.5, L * 0.66).extrude(bore_depth, both=True)
        leg = leg.cut(pocket.translate((0, t / 2, 0))).cut(pocket.translate((0, -t / 2, 0)))
        holes = (cq.Workplane("XZ").circle(pin / 2).extrude(t, both=True)
                 .union(cq.Workplane("XZ").center(0, -L).circle(pin / 2).extrude(t, both=True)))
        return leg.cut(holes)

    def _upper_leg(self, p):
        return self._plate_leg(p["upper_leg_length"], p["upper_leg_width"], p["upper_leg_thickness"], p["upper_leg_taper"],
                               p["pin_diameter"], 8.0, p["upper_leg_thickness"] * 0.38)

    def _lower_leg(self, p):
        # knee bore at the top, the axle bore at the bottom (the axle is a stub: the lower leg carries the hub motor)
        return self._plate_leg(p["lower_leg_length"], p["lower_leg_width"], p["lower_leg_thickness"], p["lower_leg_taper"],
                               p["axle_diameter"], 10.0, p["lower_leg_thickness"] * 0.38)

    def _wheel(self, p):
        """A wheel in its own frame: axle along y, centred at the origin."""
        r, wdt = p["wheel_diameter"] / 2, p["wheel_width"]
        tyre = cq.Workplane("XZ").circle(r).extrude(wdt / 2, both=True).edges().fillet(wdt * 0.18)
        hub = cq.Workplane("XZ").circle(p["hub_diameter"] / 2).extrude(wdt * 0.6, both=True)
        bore = cq.Workplane("XZ").circle(p["axle_diameter"] / 2).extrude(wdt, both=True)
        return tyre.union(hub).cut(bore)

    def _hull(self, p):
        Lh, Wh, Hh, ch = p["hull_length"], p["hull_width"], p["hull_height"], p["hull_chamfer"]
        z0 = p["hull_bottom"]
        hull = cq.Workplane("XY").box(Lh, Wh, Hh, centered=(True, True, False)).translate((0, 0, z0)).edges("|Y").chamfer(ch)
        hull = hull.edges("|X").chamfer(ch * 0.6)
        # sensor cluster at the front face (the E/O block of the picture): a shallow box proud of the glacis
        nose = cq.Workplane("XY").box(120.0, Wh * 0.5, Hh * 0.45, centered=(True, True, False)).translate((Lh / 2 + 20, 0, z0 + Hh * 0.3))
        hull = hull.union(nose)
        # shoulder bosses on both sides, fore and aft
        zs = self.shoulder_height(p)
        for sx in (-1, 1):
            for sy in (-1, 1):
                boss = (cq.Workplane("XZ").circle(p["shoulder_boss_diameter"] / 2).extrude(p["shoulder_boss_length"])
                        .translate((sx * p["shoulder_x"], sy * (Wh / 2 + p["shoulder_boss_length"]) if sy > 0 else sy * Wh / 2, zs)))
                hull = hull.union(boss)
        if p["turret_length"] > 0 and p["turret_height"] > 0:
            tur = (cq.Workplane("XY").box(p["turret_length"], p["turret_width"], p["turret_height"], centered=(True, True, False))
                   .translate((-Lh * 0.1, 0, z0 + Hh)).edges("|Z").chamfer(min(30.0, p["turret_width"] * 0.1)))
            hull = hull.union(tur)
        if p["mast_height"] > 0:
            zt = z0 + Hh + p["turret_height"]
            mast = cq.Workplane("XY").circle(p["mast_diameter"] / 2).extrude(p["mast_height"]).translate((-Lh * 0.1 + p["turret_length"] * 0.3, 0, zt))
            hull = hull.union(mast)
            if p["sensor_head"] > 0:
                s = p["sensor_head"]
                head = cq.Workplane("XY").box(s, s, s, centered=(True, True, False)).translate((-Lh * 0.1 + p["turret_length"] * 0.3, 0, zt + p["mast_height"]))
                hull = hull.union(head)
        return hull

    def build(self, p):
        if p["part"] == "upper_leg":
            return self._upper_leg(p)
        if p["part"] == "lower_leg":
            return self._lower_leg(p)
        if p["part"] == "wheel":
            return self._wheel(p)
        if p["shoulder_x"] >= p["hull_length"] / 2:
            raise ValueError("shoulder_x must be smaller than half the hull length")
        hull = self._hull(p)
        if p["part"] == "hull":
            return hull
        robot = hull
        zs = self.shoulder_height(p)
        a1, a2 = p["hip_angle_deg"], p["knee_angle_deg"]
        L1, L2 = p["upper_leg_length"], p["lower_leg_length"]
        y_upper = p["hull_width"] / 2 + p["shoulder_boss_length"] + p["upper_leg_thickness"] / 2 + 1.0
        y_lower = y_upper + p["upper_leg_thickness"] / 2 + p["lower_leg_thickness"] / 2 + 1.0
        y_wheel = y_lower + p["lower_leg_thickness"] / 2 + p["wheel_offset"] + p["wheel_width"] / 2
        for sx in (-1, 1):
            for sy in (-1, 1):
                shoulder = (sx * p["shoulder_x"], sy * y_upper, zs)
                upper = self._upper_leg(p).rotate((0, 0, 0), (0, 1, 0), a1).translate(shoulder)
                knee = (shoulder[0] - L1 * math.sin(math.radians(a1)), sy * y_lower, zs - L1 * math.cos(math.radians(a1)))
                lower = self._lower_leg(p).rotate((0, 0, 0), (0, 1, 0), -a2).translate(knee)
                axle = (knee[0] + L2 * math.sin(math.radians(a2)), sy * y_wheel, knee[2] - L2 * math.cos(math.radians(a2)))
                wheel = self._wheel(p).translate(axle)
                stub = (cq.Workplane("XZ").circle(p["axle_diameter"] / 2).extrude((y_wheel - y_lower) / 2 + 2.0, both=True)
                        .translate((axle[0], sy * (y_lower + y_wheel) / 2, axle[2])))
                robot = robot.union(upper).union(lower).union(wheel).union(stub)
        return robot
