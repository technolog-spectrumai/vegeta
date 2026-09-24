import math
import cadquery as cq
from vegeta.dedalus import Design, Parameter


class Rover(Design):
    """Small four-wheel rover for rugged terrain: a tub chassis (plate + side rails), four trailing
    suspension arms on pivots in the rails, wheels on the arm ends. Chassis top at z = 0, wheels
    below. ``part`` selects the whole vehicle, the chassis, or one suspension arm (for FEA/printing)."""

    parameters = [
        Parameter("part", "rover", choices=("rover", "chassis", "arm"), description="what to build"),
        Parameter("length", 320.0, "mm", min=100, description="chassis length"),
        Parameter("width", 200.0, "mm", min=60, description="chassis width (outside the rails)"),
        Parameter("plate_thickness", 5.0, "mm", min=1),
        Parameter("rail_height", 30.0, "mm", min=5, description="side rails below the plate"),
        Parameter("rail_thickness", 6.0, "mm", min=1),
        Parameter("arm_length", 90.0, "mm", min=20, description="pivot to wheel axle"),
        Parameter("arm_width", 18.0, "mm", min=4),
        Parameter("arm_height", 12.0, "mm", min=3),
        Parameter("arm_taper", 0.6, "", min=0.3, max=1.0),
        Parameter("pivot_diameter", 8.0, "mm", min=2),
        Parameter("axle_diameter", 6.0, "mm", min=2),
        Parameter("arm_angle_deg", 25.0, "deg", min=0, max=60, description="arm droop below horizontal at ride height"),
        Parameter("wheel_diameter", 110.0, "mm", min=20),
        Parameter("wheel_width", 40.0, "mm", min=5),
        Parameter("wheel_offset", 12.0, "mm", min=0, description="gap between arm and wheel"),
        Parameter("pivot_x", 60.0, "mm", min=0, description="pivot distance from each chassis end"),
        Parameter("payload_cutout", 0.0, "mm", min=0, description="square cutout in the plate (0 = none)"),
    ]

    def _arm(self, p):
        L, w, h, t = p["arm_length"], p["arm_width"], p["arm_height"], p["arm_taper"]
        arm = (cq.Workplane("XY").polyline([(0, -w / 2), (L, -w * t / 2), (L, w * t / 2), (0, w / 2)]).close()
               .extrude(h).translate((0, 0, -h / 2)))
        bosses = (cq.Workplane("XY").circle(w / 2).extrude(h).translate((0, 0, -h / 2))
                  .union(cq.Workplane("XY").center(L, 0).circle(w * t / 2 + 2).extrude(h).translate((0, 0, -h / 2))))
        arm = arm.union(bosses)
        holes = (cq.Workplane("XY").circle(p["pivot_diameter"] / 2).extrude(2 * h).translate((0, 0, -h))
                 .union(cq.Workplane("XY").center(L, 0).circle(p["axle_diameter"] / 2).extrude(2 * h).translate((0, 0, -h))))
        # the arm rotates about Y (pivot axis along Y): built in XZ so its thickness is along Y
        return arm.cut(holes).rotate((0, 0, 0), (1, 0, 0), 90)

    def build(self, p):
        if p["part"] == "arm":
            return self._arm(p)
        Lc, Wc, tp = p["length"], p["width"], p["plate_thickness"]
        rh, rt = p["rail_height"], p["rail_thickness"]
        if p["pivot_x"] >= Lc / 2:
            raise ValueError("pivot_x must be smaller than half the chassis length")
        plate = cq.Workplane("XY").box(Lc, Wc, tp, centered=(True, True, False)).translate((0, 0, -tp)).edges("|Z").fillet(8)
        chassis = plate
        for sy in (-1, 1):
            rail = cq.Workplane("XY").box(Lc - 20, rt, rh, centered=(True, True, False)).translate((0, sy * (Wc - rt) / 2 - sy * 2, -tp - rh))
            chassis = chassis.union(rail)
        if p["payload_cutout"] > 0:
            chassis = chassis.cut(cq.Workplane("XY").rect(p["payload_cutout"], p["payload_cutout"]).extrude(50).translate((0, 0, -25)))
        # pivot holes through the rails
        px, pz = Lc / 2 - p["pivot_x"], -tp - rh / 2
        for sx in (-1, 1):
            chassis = chassis.cut(cq.Workplane("XZ").center(sx * px, pz).circle(p["pivot_diameter"] / 2).extrude(Wc, both=True))
        if p["part"] == "chassis":
            return chassis
        rover = chassis
        ang = math.radians(p["arm_angle_deg"])
        L, h, w = p["arm_length"], p["arm_height"], p["wheel_width"]
        for sx in (-1, 1):                      # front (+x) arms lead forward, rear arms trail backward
            for sy in (-1, 1):
                arm = self._arm(p).rotate((0, 0, 0), (0, 1, 0), math.degrees(ang))      # droop: tip goes down
                if sx < 0:
                    arm = arm.rotate((0, 0, 0), (0, 0, 1), 180)
                y_arm = sy * (Wc / 2 + h / 2 + 1.0)                                       # just outside the rail
                arm = arm.translate((sx * px, y_arm, pz))
                ax_x, ax_z = sx * (px + L * math.cos(ang)), pz - L * math.sin(ang)
                stub = h / 2 + p["wheel_offset"] + w + 3.0
                y_wheel = sy * (abs(y_arm) + h / 2 + p["wheel_offset"] + w / 2)
                wheel = (cq.Workplane("XZ").center(ax_x, ax_z).circle(p["wheel_diameter"] / 2).extrude(w / 2, both=True)
                         .edges().chamfer(3).translate((0, y_wheel, 0)))
                axle = (cq.Workplane("XZ").center(ax_x, ax_z).circle(p["axle_diameter"] / 2).extrude(stub / 2, both=True)
                        .translate((0, sy * (abs(y_arm) + stub / 2), 0)))
                rover = rover.union(arm).union(wheel).union(axle)
        return rover
