import math
import cadquery as cq
from vegeta.dedalus import Design, Parameter


class QuadFrame(Design):
    """X-frame quadcopter: round centre plate, four tapered arms, round motor pads. One printable solid.
    Motor and stack bolt holes are through holes; loads and supports attach to their cylindrical faces."""

    parameters = [
        Parameter("wheelbase", 250.0, "mm", min=100, description="motor-to-motor diagonal"),
        Parameter("arm_width", 12.0, "mm", min=4, description="arm width at the root"),
        Parameter("arm_height", 6.0, "mm", min=2, description="arm thickness (Z)"),
        Parameter("taper", 0.7, "", min=0.3, max=1.0, description="arm width at the pad / at the root"),
        Parameter("plate_size", 70.0, "mm", min=30, description="centre plate diameter"),
        Parameter("plate_thickness", 6.0, "mm", min=1, description="centre plate thickness (>= arm_height: the arms cross inside it)"),
        Parameter("pad_diameter", 30.0, "mm", min=10, description="motor pad diameter"),
        Parameter("motor_pattern", 16.0, "mm", min=5, description="motor bolt square (M3, 16x16)"),
        Parameter("motor_hole", 3.2, "mm", min=1),
        Parameter("stack_pattern", 30.5, "mm", min=10, description="flight-controller stack bolt square"),
        Parameter("stack_hole", 3.2, "mm", min=1),
        Parameter("canopy_height", 0.0, "mm", min=0, description="elliptic dome over the plate (0 = none)"),
    ]

    def build(self, p):
        R = p["wheelbase"] / 2
        w, h, t = p["arm_width"], p["arm_height"], p["plate_thickness"]
        if p["pad_diameter"] < p["motor_pattern"] * math.sqrt(2) + p["motor_hole"] + 3.0:
            raise ValueError("pad_diameter leaves less than 1.5 mm of wall around the motor bolt holes")
        if t < h:
            raise ValueError("plate_thickness must be >= arm_height (the four arms cross inside the plate)")
        s = p["plate_size"]
        frame = cq.Workplane("XY").circle(s / 2).extrude(t)
        pads = cq.Workplane("XY")
        holes = cq.Workplane("XY")
        stack = p["stack_pattern"] / 2
        holes = holes.pushPoints([(stack, stack), (-stack, stack), (stack, -stack), (-stack, -stack)]) \
            .circle(p["stack_hole"] / 2).extrude(50)
        for k in range(4):
            ang = math.radians(45 + 90 * k)
            cx, cy = R * math.cos(ang), R * math.sin(ang)
            arm = (cq.Workplane("XY")
                   .polyline([(0, -w / 2), (R, -w * p["taper"] / 2), (R, w * p["taper"] / 2), (0, w / 2)])
                   .close().extrude(h).rotate((0, 0, 0), (0, 0, 1), math.degrees(ang)))
            pad = cq.Workplane("XY").center(cx, cy).circle(p["pad_diameter"] / 2).extrude(h)
            frame = frame.union(arm).union(pad)
            m = p["motor_pattern"] / 2
            holes = holes.pushPoints([(cx + m, cy + m), (cx - m, cy + m), (cx + m, cy - m), (cx - m, cy - m)]) \
                .circle(p["motor_hole"] / 2).extrude(50)
        if p["canopy_height"] > 0:
            dome = (cq.Workplane("XY").workplane(offset=t)
                    .ellipse(s / 2 - 6, s / 2 - 6).extrude(p["canopy_height"])
                    .faces(">Z").edges().fillet(p["canopy_height"] * 0.9))
            frame = frame.union(dome)
        return frame.cut(holes)
