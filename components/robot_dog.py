import math
import cadquery as cq
from vegeta.dedalus import Design, Parameter


class RobotDog(Design):
    """Quadruped robot dog ("Cerberus"): a rounded body shell with a payload deck on its back, a head
    with two camera eyes, and four two-link legs (hip motor boss, upper leg plate, knee boss, slim
    lower leg with a rubber foot), all knees pointing backwards. x forward, y left, z up, ground at
    z = 0. ``part`` selects the whole dog, the body (shell + deck + head), one upper leg or one
    lower leg (both in their own frame: the joint bore at the origin, the leg hanging down -z, the
    bore axis along y) for FEA and printing."""

    parameters = [
        Parameter("part", "dog", choices=("dog", "body", "upper_leg", "lower_leg"), description="what to build"),
        Parameter("body_length", 520.0, "mm", min=150, description="shell length without the head"),
        Parameter("body_width", 240.0, "mm", min=80),
        Parameter("body_height", 130.0, "mm", min=40),
        Parameter("shell_fillet", 28.0, "mm", min=1, description="rounding of the shell edges"),
        Parameter("hip_x", 190.0, "mm", min=20, description="hips from the body centre, fore and aft"),
        Parameter("hip_boss_diameter", 64.0, "mm", min=10, description="hip motor housing on the body side"),
        Parameter("hip_boss_length", 30.0, "mm", min=4),
        Parameter("upper_leg_length", 200.0, "mm", min=30, description="hip to knee"),
        Parameter("upper_leg_width", 44.0, "mm", min=8, description="plate width at the hip"),
        Parameter("upper_leg_thickness", 16.0, "mm", min=3),
        Parameter("upper_leg_taper", 0.65, "", min=0.3, max=1.0),
        Parameter("lower_leg_length", 210.0, "mm", min=30, description="knee to the foot centre"),
        Parameter("lower_leg_diameter", 24.0, "mm", min=4, description="at the knee; tapers to 60 %"),
        Parameter("foot_diameter", 34.0, "mm", min=6),
        Parameter("hip_angle_deg", 35.0, "deg", min=0, max=80, description="upper leg behind vertical (knee back)"),
        Parameter("knee_angle_deg", 40.0, "deg", min=0, max=120, description="lower leg ahead of vertical (foot forward)"),
        Parameter("pin_diameter", 10.0, "mm", min=2, description="hip and knee pins"),
        Parameter("head_length", 150.0, "mm", min=20),
        Parameter("head_height", 95.0, "mm", min=20),
        Parameter("head_drop", 25.0, "mm", min=0, description="head centre below the body centre"),
        Parameter("deck_length", 300.0, "mm", min=0, description="payload deck pocket on the back (0 = none)"),
        Parameter("deck_width", 200.0, "mm", min=0),
        Parameter("deck_depth", 6.0, "mm", min=0),
        Parameter("deck_hole_diameter", 5.5, "mm", min=1, description="four mounting holes, M5 clearance"),
        Parameter("deck_hole_pitch", 200.0, "mm", min=10, description="square bolt pattern of the deck"),
    ]

    # ---- derived geometry shared by the notebook (kept here so CAD and mechanics agree) ----
    @staticmethod
    def standing_height(p):
        a1, a2 = math.radians(p["hip_angle_deg"]), math.radians(p["knee_angle_deg"])
        return p["upper_leg_length"] * math.cos(a1) + p["lower_leg_length"] * math.cos(a2)

    def _upper_leg(self, p):
        L, w, t, k = p["upper_leg_length"], p["upper_leg_width"], p["upper_leg_thickness"], p["upper_leg_taper"]
        plate = (cq.Workplane("XZ").polyline([(-w / 2, 0), (w / 2, 0), (w * k / 2, -L), (-w * k / 2, -L)]).close()
                 .extrude(t / 2, both=True))
        bosses = (cq.Workplane("XZ").circle(w / 2).extrude(t / 2, both=True)
                  .union(cq.Workplane("XZ").center(0, -L).circle(w * k / 2 + 3).extrude(t / 2, both=True)))
        leg = plate.union(bosses)
        # lightening pocket in the middle of the plate, both faces
        pocket = cq.Workplane("XZ").center(0, -L / 2).rect(w * 0.35, L * 0.55).extrude(t, both=True)
        leg = leg.cut(cq.Workplane("XZ").center(0, -L / 2).rect(w * 0.35, L * 0.55).extrude(t * 0.3, both=True)
                      .translate((0, t / 2, 0))).cut(cq.Workplane("XZ").center(0, -L / 2).rect(w * 0.35, L * 0.55)
                                                     .extrude(t * 0.3, both=True).translate((0, -t / 2, 0)))
        holes = (cq.Workplane("XZ").circle(p["pin_diameter"] / 2).extrude(t, both=True)
                 .union(cq.Workplane("XZ").center(0, -L).circle(p["pin_diameter"] / 2).extrude(t, both=True)))
        return leg.cut(holes)

    def _lower_leg(self, p):
        L, d = p["lower_leg_length"], p["lower_leg_diameter"]
        t = p["upper_leg_thickness"]
        # knee clevis: a round boss with the pin bore, same thickness as the upper leg plate
        knee = cq.Workplane("XZ").circle(d / 2 + 5).extrude(t / 2, both=True)
        # tapered rod, slightly swept back (a curve like the picture), as a loft of circles
        rod = (cq.Workplane("XY").circle(d / 2).workplane(offset=-L * 0.5).center(-0.04 * L, 0).circle(d * 0.8 / 2)
               .workplane(offset=-L * 0.5).center(0.04 * L, 0).circle(d * 0.6 / 2).loft(ruled=False))
        foot = cq.Workplane("XY").sphere(p["foot_diameter"] / 2).translate((0, 0, -L))
        leg = knee.union(rod).union(foot)
        return leg.cut(cq.Workplane("XZ").circle(p["pin_diameter"] / 2).extrude(t, both=True))

    def _body(self, p):
        Lb, Wb, Hb = p["body_length"], p["body_width"], p["body_height"]
        h = self.standing_height(p)
        zc = h + 8.0                                                                   # body centre a little above the hips
        shell = cq.Workplane("XY").box(Lb, Wb, Hb).translate((0, 0, zc)).edges().fillet(min(p["shell_fillet"], Hb / 2 - 1))
        # hip motor bosses on both sides, fore and aft
        for sx in (-1, 1):
            for sy in (-1, 1):
                boss = (cq.Workplane("XZ").center(sx * p["hip_x"], h).circle(p["hip_boss_diameter"] / 2)
                        .extrude(p["hip_boss_length"]).translate((0, sy * (Wb / 2 + (p["hip_boss_length"] if sy > 0 else 0)), 0)))
                shell = shell.union(boss)
        # head: a rounded box ahead, lower, with two eye bores and a nose light
        Lh, Hh = p["head_length"], p["head_height"]
        head = (cq.Workplane("XY").box(Lh, Wb * 0.55, Hh).translate((Lb / 2 + Lh / 2 - 15, 0, zc - p["head_drop"]))
                .edges().fillet(min(18.0, Hh / 2 - 1)))
        eyes = (cq.Workplane("YZ").pushPoints([(-Wb * 0.14, zc - p["head_drop"] + Hh * 0.12), (Wb * 0.14, zc - p["head_drop"] + Hh * 0.12)])
                .circle(9).extrude(20).translate((Lb / 2 + Lh - 15 - 10, 0, 0)))
        shell = shell.union(head).cut(eyes)
        # tail antenna
        tail = cq.Workplane("XY").circle(4).extrude(70).translate((-Lb / 2 + 25, 0, zc + Hb / 2 - 10)).rotate((-Lb / 2 + 25, 0, zc), (-Lb / 2 + 25, 1, zc), -35)
        shell = shell.union(tail)
        # payload deck: a shallow pocket on the back with four bolt holes
        if p["deck_length"] > 0 and p["deck_width"] > 0:
            top = zc + Hb / 2
            pocket = cq.Workplane("XY").rect(p["deck_length"], p["deck_width"]).extrude(p["deck_depth"] + 5).translate((0, 0, top - p["deck_depth"]))
            shell = shell.cut(pocket)
            pitch = p["deck_hole_pitch"] / 2
            holes = cq.Workplane("XY").pushPoints([(sx * pitch, sy * pitch) for sx in (-1, 1) for sy in (-1, 1)]).circle(p["deck_hole_diameter"] / 2).extrude(40).translate((0, 0, top - 35))
            shell = shell.cut(holes)
        return shell

    def build(self, p):
        if p["part"] == "upper_leg":
            return self._upper_leg(p)
        if p["part"] == "lower_leg":
            return self._lower_leg(p)
        body = self._body(p)
        if p["part"] == "body":
            return body
        if p["hip_x"] >= p["body_length"] / 2:
            raise ValueError("hip_x must be smaller than half the body length")
        h = self.standing_height(p)
        a1, a2 = p["hip_angle_deg"], p["knee_angle_deg"]
        L1 = p["upper_leg_length"]
        dog = body
        y_leg = p["body_width"] / 2 + p["hip_boss_length"] + p["upper_leg_thickness"] / 2 + 1.0
        for sx in (-1, 1):
            for sy in (-1, 1):
                # knee back: rotate about +y by +a1 takes (0,0,-L1) to (-L1 sin a1, 0, -L1 cos a1)
                upper = self._upper_leg(p).rotate((0, 0, 0), (0, 1, 0), a1).translate((sx * p["hip_x"], sy * y_leg, h))
                kx, kz = sx * p["hip_x"] - L1 * math.sin(math.radians(a1)), h - L1 * math.cos(math.radians(a1))
                lower = self._lower_leg(p).rotate((0, 0, 0), (0, 1, 0), -a2).translate((kx, sy * y_leg, kz))
                pins = (cq.Workplane("XZ").center(sx * p["hip_x"], h).circle(p["pin_diameter"] / 2 - 0.1).extrude(p["upper_leg_thickness"] + 6, both=True)
                        .union(cq.Workplane("XZ").center(kx, kz).circle(p["pin_diameter"] / 2 - 0.1).extrude(p["upper_leg_thickness"] + 6, both=True))
                        .translate((0, sy * y_leg, 0)))
                dog = dog.union(upper).union(lower).union(pins)
        return dog
