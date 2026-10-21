"""Perseus — a tracked branch-cutting rover (notebook 37) and its small garden version, Perseus Mini (notebook 38).

Both are the Pekari Rover's track architecture (``pekari_rover.PekariRover``: hull between two track modules, rear
drive sprocket, front idler on a tensioner, road wheels on bogies, a belt of hinged links) with one boom on the roof
ending in an electric chainsaw: a pedestal (yaw), the **boom** (shoulder pitch), the **stick** (elbow pitch) and the
**saw head** (wrist pitch) carrying the saw body and its bar. On Perseus the boom and the stick are moved by two
electric cylinders (ball screws with brakes, as the Onager Atlas's lift and tilt) in the layout of a small excavator
— the lift cylinder from a lug on the pedestal to the underside of the boom, the stick cylinder from the top of the
boom to the stick's heel behind the elbow; the yaw and the wrist are joint modules. Perseus Mini has joint modules at
every axis (``cylinder_diameter`` 0: no cylinders). x forward, y left, z up, ground at z = 0 (the grouser tips).

``part``: the Pekari parts (``rover`` = the whole machine with the boom in its ready pose, ``hull``, ``track_module``,
``sprocket``, ``idler``, ``road_wheel``, ``track_link``, ``pinion``) plus ``pedestal`` (own frame: its base on z = 0),
``boom`` and ``stick`` (own frames: the inboard pin bore at the origin, the link along +x, the bore axis along y;
lugs for the cylinders), ``saw_head`` (the wrist bore at the origin, the mount, the saw body and the bar along +x,
the chain plane y = 0) and ``cylinder`` (the lift cylinder at its ready-pose length, along +x from its base lug).
Static methods keep CAD and mechanics in step: ``shoulder_point``, ``reach``, ``arm_points`` (the pins and the bar
tip for given elevations), ``lift_lever`` / ``stick_lever`` (the cylinders' lever arms about their pivots) and
``cylinder_length``.
"""
import math
from dataclasses import replace

try:
    import cadquery as cq
except ImportError:                     # the static geometry needs no CAD kernel
    cq = None
from vegeta.dedalus import Parameter

from pekari_rover import PekariRover

#: The Perseus chassis: Pekari's parameters scaled to a ~180 kg machine (mm); no basket (the boom takes the roof).
PERSEUS_CHASSIS = {
    "hull_length": 1100.0, "hull_width": 520.0, "hull_height": 300.0, "hull_chamfer": 30.0, "ground_clearance": 150.0,
    "sensor_box": 140.0, "basket_length": 0.0, "basket_width": 0.0, "basket_height": 0.0, "basket_wall": 1.5,
    "basket_x": 0.0, "frame_offset": 100.0, "frame_thickness": 8.0, "track_width": 150.0, "track_pitch": 50.0,
    "link_thickness": 14.0, "grouser_height": 12.0, "guide_height": 20.0, "guide_width": 25.0, "pin_diameter": 10.0,
    "sprocket_teeth": 15, "sprocket_x": -450.0, "sprocket_z": 260.0, "idler_diameter": 220.0, "idler_x": 450.0,
    "idler_z": 200.0, "road_wheel_diameter": 140.0, "road_wheels": 6, "road_wheel_spacing": 150.0,
    "return_roller_diameter": 80.0, "return_roller_z": 300.0, "axle_diameter": 20.0, "sprocket_bore": 25.0,
    "pinion_module": 1.5, "pinion_teeth": 18, "pinion_width": 20.0, "pinion_bore": 10.0,
}
#: The Mini chassis: a Pekari-class machine (~30 kg) on 100 mm tracks.
MINI_CHASSIS = {
    "hull_length": 600.0, "hull_width": 320.0, "hull_height": 150.0, "hull_chamfer": 15.0, "ground_clearance": 100.0,
    "sensor_box": 90.0, "basket_length": 0.0, "basket_width": 0.0, "basket_height": 0.0, "basket_wall": 1.5,
    "basket_x": 0.0, "frame_offset": 65.0, "frame_thickness": 6.0, "track_width": 90.0, "track_pitch": 36.0,
    "link_thickness": 8.0, "grouser_height": 8.0, "guide_height": 14.0, "guide_width": 18.0, "pin_diameter": 6.0,
    "sprocket_teeth": 12, "sprocket_x": -320.0, "sprocket_z": 160.0, "idler_diameter": 130.0, "idler_x": 320.0,
    "idler_z": 120.0, "road_wheel_diameter": 80.0, "road_wheels": 4, "road_wheel_spacing": 140.0,
    "return_roller_diameter": 45.0, "return_roller_z": 185.0, "axle_diameter": 12.0, "sprocket_bore": 14.0,
    "pinion_module": 1.0, "pinion_teeth": 18, "pinion_width": 12.0, "pinion_bore": 6.0,
}
#: The Perseus boom (mm, deg).
PERSEUS_ARM = {
    "boom_x": 200.0, "pedestal_height": 150.0, "pedestal_diameter": 220.0,
    "boom_length": 900.0, "boom_width": 80.0, "boom_depth": 120.0,
    "stick_length": 800.0, "stick_width": 70.0, "stick_depth": 100.0, "arm_wall": 5.0,
    "arm_boss_diameter": 100.0, "arm_pin_diameter": 35.0,
    "head_length": 250.0, "saw_body_length": 320.0, "saw_body_width": 120.0, "saw_body_height": 180.0,
    "bar_length": 400.0, "bar_width": 75.0, "bar_thickness": 5.0,
    "cylinder_diameter": 50.0, "lift_lug_x": 300.0, "lift_rod_frac": 0.45, "stick_lug_frac": 0.55, "stick_heel": 200.0,
    "boom_deg": 45.0, "stick_deg": -15.0, "head_deg": 0.0,
}
#: The Mini boom: joint modules, no cylinders.
MINI_ARM = {
    "boom_x": 120.0, "pedestal_height": 80.0, "pedestal_diameter": 120.0,
    "boom_length": 400.0, "boom_width": 50.0, "boom_depth": 70.0,
    "stick_length": 350.0, "stick_width": 45.0, "stick_depth": 60.0, "arm_wall": 3.0,
    "arm_boss_diameter": 70.0, "arm_pin_diameter": 16.0,
    "head_length": 120.0, "saw_body_length": 200.0, "saw_body_width": 90.0, "saw_body_height": 110.0,
    "bar_length": 150.0, "bar_width": 45.0, "bar_thickness": 4.0,
    "cylinder_diameter": 0.0, "lift_lug_x": 0.0, "lift_rod_frac": 0.45, "stick_lug_frac": 0.55, "stick_heel": 0.0,
    "boom_deg": 45.0, "stick_deg": -15.0, "head_deg": 0.0,
}

PARTS = ("rover", "hull", "track_module", "sprocket", "idler", "road_wheel", "track_link", "pinion", "pedestal", "boom",
         "stick", "saw_head", "cylinder")
PARTS_MINI = PARTS[:-1]                              # no cylinders on the Mini


def _pekari_params(defaults):
    """Pekari's parameters (without ``part``) with the given defaults; the basket parameters stay (0 = none)."""
    out = []
    for q in PekariRover.parameters:
        if q.name == "part":
            continue
        if q.name in defaults and q.default != defaults[q.name]:
            desc = q.description
            if q.name.startswith("basket"):
                desc = "no basket on Perseus: the boom takes the roof"
            q = replace(q, default=defaults[q.name], description=desc)
        out.append(q)
    return out


def _arm_params(arm):
    d = arm
    return [
        Parameter("boom_x", d["boom_x"], "mm", description="the pedestal (shoulder yaw axis) ahead of the hull centre"),
        Parameter("pedestal_height", d["pedestal_height"], "mm", min=20, description="shoulder pitch axis above the roof"),
        Parameter("pedestal_diameter", d["pedestal_diameter"], "mm", min=20),
        Parameter("boom_length", d["boom_length"], "mm", min=100, description="shoulder pitch to elbow"),
        Parameter("boom_width", d["boom_width"], "mm", min=10, description="rectangular tube, across (y)"),
        Parameter("boom_depth", d["boom_depth"], "mm", min=10, description="rectangular tube, bending depth (z)"),
        Parameter("stick_length", d["stick_length"], "mm", min=100, description="elbow to wrist pitch"),
        Parameter("stick_width", d["stick_width"], "mm", min=10),
        Parameter("stick_depth", d["stick_depth"], "mm", min=10),
        Parameter("arm_wall", d["arm_wall"], "mm", min=1, description="wall of the boom and stick tubes (Al 6082-T6)"),
        Parameter("arm_boss_diameter", d["arm_boss_diameter"], "mm", min=20, description="joint housings at the pins"),
        Parameter("arm_pin_diameter", d["arm_pin_diameter"], "mm", min=5, description="arm joint bores"),
        Parameter("head_length", d["head_length"], "mm", min=20, description="wrist pitch to the bar root"),
        Parameter("saw_body_length", d["saw_body_length"], "mm", min=50, description="the saw's powerhead (x)"),
        Parameter("saw_body_width", d["saw_body_width"], "mm", min=20),
        Parameter("saw_body_height", d["saw_body_height"], "mm", min=20),
        Parameter("bar_length", d["bar_length"], "mm", min=50, description="the bar beyond the body (cutting length)"),
        Parameter("bar_width", d["bar_width"], "mm", min=20, description="the bar's height (z), its nose radius is half"),
        Parameter("bar_thickness", d["bar_thickness"], "mm", min=1, description="the bar plate (the chain's kerf is wider)"),
        Parameter("cylinder_diameter", d["cylinder_diameter"], "mm", min=0,
                  description="the lift and stick cylinders' barrels (0 = joint modules at the shoulder and elbow)"),
        Parameter("lift_lug_x", d["lift_lug_x"], "mm", min=0, description="lift cylinder base lug ahead of the shoulder, at the roof"),
        Parameter("lift_rod_frac", d["lift_rod_frac"], "", min=0.1, max=0.9, description="lift cylinder rod on the boom, fraction of its length"),
        Parameter("stick_lug_frac", d["stick_lug_frac"], "", min=0.1, max=0.9, description="stick cylinder base on the boom, fraction of its length"),
        Parameter("stick_heel", d["stick_heel"], "mm", min=0, description="stick cylinder rod lug behind the elbow (the stick's heel)"),
        Parameter("boom_deg", d["boom_deg"], "deg", min=-30, max=100, description="boom above horizontal (ready pose)"),
        Parameter("stick_deg", d["stick_deg"], "deg", min=-120, max=90, description="stick above horizontal (ready pose)"),
        Parameter("head_deg", d["head_deg"], "deg", min=-120, max=120, description="bar above horizontal (ready pose)"),
    ]


class PerseusRover(PekariRover):
    """Perseus: the Pekari track architecture at ~180 kg with one chainsaw boom (see the module)."""

    parameters = ([Parameter("part", "rover", choices=PARTS, description="what to build")]
                  + _pekari_params(PERSEUS_CHASSIS) + _arm_params(PERSEUS_ARM))

    # ---- derived geometry shared with the robot module ----
    @staticmethod
    def shoulder_point(p):
        """The shoulder pitch axis (x, y, z) [mm] in the vehicle frame."""
        return (p["boom_x"], 0.0, p["ground_clearance"] + p["hull_height"] + p["pedestal_height"])

    @staticmethod
    def reach(p):
        """Shoulder pitch axis to the bar tip, arm straight [mm]."""
        return p["boom_length"] + p["stick_length"] + p["head_length"] + p["bar_length"]

    @staticmethod
    def arm_points(p, boom_deg=None, stick_deg=None, head_deg=None):
        """The pins and the bar in the xz plane [mm] for the elevations (deg above horizontal; None = the ready
        pose): ``shoulder``, ``elbow``, ``wrist``, ``bar_root``, ``bar_tip`` as (x, z) in the vehicle frame."""
        b = p["boom_deg"] if boom_deg is None else boom_deg
        s = p["stick_deg"] if stick_deg is None else stick_deg
        h = p["head_deg"] if head_deg is None else head_deg
        sx, _, sz = PerseusRover.shoulder_point(p)
        ex, ez = sx + p["boom_length"] * math.cos(math.radians(b)), sz + p["boom_length"] * math.sin(math.radians(b))
        wx, wz = ex + p["stick_length"] * math.cos(math.radians(s)), ez + p["stick_length"] * math.sin(math.radians(s))
        ch, sh = math.cos(math.radians(h)), math.sin(math.radians(h))
        rx, rz = wx + p["head_length"] * ch, wz + p["head_length"] * sh
        tx, tz = rx + p["bar_length"] * ch, rz + p["bar_length"] * sh
        return {"shoulder": (sx, sz), "elbow": (ex, ez), "wrist": (wx, wz), "bar_root": (rx, rz), "bar_tip": (tx, tz)}

    @staticmethod
    def _lift_ends(p, boom_deg):
        """Lift cylinder base lug and rod lug (x, z) [mm] relative to the shoulder pin."""
        a = (p["lift_lug_x"], -0.9 * p["pedestal_height"])                   # the lug's eye just above the roof
        t = math.radians(boom_deg)
        u, v = p["lift_rod_frac"] * p["boom_length"], -(p["boom_depth"] / 2 + 0.3 * p["boom_depth"])
        b = (u * math.cos(t) - v * math.sin(t), u * math.sin(t) + v * math.cos(t))
        return a, b

    @staticmethod
    def _stick_ends(p, boom_deg, stick_deg):
        """Stick cylinder base lug (on the boom) and rod lug (the stick's heel) (x, z) [mm] relative to the elbow pin."""
        tb, ts = math.radians(boom_deg), math.radians(stick_deg)
        u, v = (p["stick_lug_frac"] - 1.0) * p["boom_length"], p["boom_depth"] / 2 + 0.3 * p["boom_depth"]
        a = (u * math.cos(tb) - v * math.sin(tb), u * math.sin(tb) + v * math.cos(tb))
        u, v = -p["stick_heel"], 0.8 * p["stick_depth"]
        b = (u * math.cos(ts) - v * math.sin(ts), u * math.sin(ts) + v * math.cos(ts))
        return a, b

    @staticmethod
    def _lever(a, b):
        """Perpendicular distance of the line through ``a`` and ``b`` from the origin (the pivot)."""
        dx, dz = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dz)
        return abs(a[0] * dz - a[1] * dx) / L if L > 0 else 0.0

    @staticmethod
    def lift_lever(p, boom_deg=None):
        """The lift cylinder's lever arm about the shoulder [mm] at the boom elevation (0 without cylinders)."""
        if p["cylinder_diameter"] <= 0:
            return 0.0
        a, b = PerseusRover._lift_ends(p, p["boom_deg"] if boom_deg is None else boom_deg)
        return PerseusRover._lever(a, b)

    @staticmethod
    def stick_lever(p, boom_deg=None, stick_deg=None):
        """The stick cylinder's lever arm about the elbow [mm] at the elevations (0 without cylinders)."""
        if p["cylinder_diameter"] <= 0:
            return 0.0
        a, b = PerseusRover._stick_ends(p, p["boom_deg"] if boom_deg is None else boom_deg,
                                        p["stick_deg"] if stick_deg is None else stick_deg)
        return PerseusRover._lever(a, b)

    @staticmethod
    def cylinder_length(p, which="lift", boom_deg=None, stick_deg=None):
        """Lug-to-lug length [mm] of the lift or stick cylinder at the elevations."""
        b = p["boom_deg"] if boom_deg is None else boom_deg
        s = p["stick_deg"] if stick_deg is None else stick_deg
        a, c = PerseusRover._lift_ends(p, b) if which == "lift" else PerseusRover._stick_ends(p, b, s)
        return math.hypot(c[0] - a[0], c[1] - a[1])

    @staticmethod
    def has_cylinders(p):
        return p["cylinder_diameter"] > 0

    # ---- parts ----
    def _tube(self, L, w, d, wall, boss_d, pin_d):
        """A rectangular tube along +x from the inboard bore with round bosses and bores at both ends (as the
        Onager Manus arm tubes)."""
        tube = cq.Workplane("XY").box(L, w, d).translate((L / 2, 0, 0))
        tube = tube.cut(cq.Workplane("XY").box(L - boss_d, w - 2 * wall, d - 2 * wall).translate((L / 2, 0, 0)))
        for x in (0.0, L):
            boss = cq.Workplane("XZ").center(x, 0).circle(boss_d / 2).extrude(w / 2, both=True)
            tube = tube.union(boss)
        for x in (0.0, L):
            tube = tube.cut(cq.Workplane("XZ").center(x, 0).circle(pin_d / 2).extrude(w, both=True))
        return tube

    def _lug(self, x, z, w, size, z_face=0.0):
        """A lug (a cylinder's eye) at (x, z) in the part's frame, ``w`` wide, on a block reaching back to the
        part's face at ``z_face`` so the two fuse; with its pin bore."""
        z0, z1 = sorted((z_face, z + (size / 2 if z > z_face else -size / 2)))
        block = cq.Workplane("XY").box(size, w, z1 - z0).translate((x, 0, (z0 + z1) / 2))
        eye = cq.Workplane("XZ").center(x, z).circle(size / 2).extrude(w / 2, both=True)
        lug = block.union(eye)
        return lug.cut(cq.Workplane("XZ").center(x, z).circle(size / 5).extrude(w, both=True))

    def _pedestal(self, p):
        h_col = p["pedestal_height"] - p["arm_boss_diameter"] / 2
        out = cq.Workplane("XY").circle(p["pedestal_diameter"] / 2).extrude(h_col)
        # the clevis for the boom's inboard boss on top: two cheek plates either side of the boom
        w, t = p["boom_width"] + 4.0, p["arm_wall"] + 2.0
        d_b = p["arm_boss_diameter"]
        for s in (-1, 1):
            cheek = cq.Workplane("XY").box(d_b, t, d_b + 10.0).translate((0, s * (w + t) / 2, h_col - 10.0 + (d_b + 10.0) / 2))
            out = out.union(cheek)
        if self.has_cylinders(p):
            a, _ = self._lift_ends(p, p["boom_deg"])
            out = out.union(self._lug(a[0], p["pedestal_height"] + a[1], w + 2 * t, 0.9 * p["cylinder_diameter"],
                                      z_face=0.0).translate((0, 0, 0)))
        return out

    def _boom(self, p):
        boom = self._tube(p["boom_length"], p["boom_width"], p["boom_depth"], p["arm_wall"], p["arm_boss_diameter"], p["arm_pin_diameter"])
        if self.has_cylinders(p):
            size = 0.9 * p["cylinder_diameter"]
            _, b = self._lift_ends(p, 0.0)                                  # in the boom's own frame (elevation 0)
            boom = boom.union(self._lug(b[0], b[1], p["boom_width"], size, z_face=-p["boom_depth"] / 2 + 1.0))
            a, _ = self._stick_ends(p, 0.0, 0.0)
            boom = boom.union(self._lug(p["boom_length"] + a[0], a[1], p["boom_width"], size, z_face=p["boom_depth"] / 2 - 1.0))
        return boom

    def _stick(self, p):
        stick = self._tube(p["stick_length"], p["stick_width"], p["stick_depth"], p["arm_wall"], p["arm_boss_diameter"], p["arm_pin_diameter"])
        if self.has_cylinders(p) and p["stick_heel"] > 0:
            _, b = self._stick_ends(p, 0.0, 0.0)
            Lh_ = p["stick_heel"] + p["arm_boss_diameter"] / 2
            heel = cq.Workplane("XY").box(Lh_, p["stick_width"], p["stick_depth"] * 0.8).translate(
                (-Lh_ / 2 + p["arm_boss_diameter"] / 2, 0, 0.4 * p["stick_depth"]))
            stick = stick.union(heel).union(self._lug(b[0], b[1], p["stick_width"], 0.9 * p["cylinder_diameter"],
                                                      z_face=0.4 * p["stick_depth"]))
        return stick

    def _saw_head(self, p):
        """Wrist bore at the origin; the mount to the saw body, the body, the bar with a rounded nose along +x."""
        Lh, Lb, wb, hb = p["head_length"], p["saw_body_length"], p["saw_body_width"], p["saw_body_height"]
        mount_L = max(Lh - Lb, 0.0) + 40.0
        mount = cq.Workplane("XY").box(mount_L, p["stick_width"], p["stick_depth"]).translate((mount_L / 2, 0, 0))
        boss = cq.Workplane("XZ").circle(p["arm_boss_diameter"] / 2).extrude(p["stick_width"] / 2, both=True)
        head = mount.union(boss)
        body = cq.Workplane("XY").box(Lb, wb, hb).translate((Lh - Lb / 2, 0, 0)).edges("|X").fillet(min(12.0, wb / 5))
        head = head.union(body)
        Lbar, wbar, tbar = p["bar_length"], p["bar_width"], p["bar_thickness"]
        bar = cq.Workplane("XY").box(Lbar - wbar / 2, tbar, wbar).translate((Lh + (Lbar - wbar / 2) / 2, 0, 0))
        nose = cq.Workplane("XZ").center(Lh + Lbar - wbar / 2, 0).circle(wbar / 2).extrude(tbar / 2, both=True)
        head = head.union(bar).union(nose)
        return head.cut(cq.Workplane("XZ").circle(p["arm_pin_diameter"] / 2).extrude(p["stick_width"], both=True))

    def _cylinder(self, p, length):
        """A cylinder (barrel + rod) of lug-to-lug ``length`` along +x from the origin."""
        d = p["cylinder_diameter"]
        barrel = cq.Workplane("YZ").circle(d / 2).extrude(0.6 * length).translate((0.05 * length, 0, 0))
        rod = cq.Workplane("YZ").circle(0.55 * d / 2).extrude(0.9 * length).translate((0.05 * length, 0, 0))
        eyes = cq.Workplane("YZ").circle(0.45 * d).extrude(0.6 * d, both=True)
        return barrel.union(rod).union(eyes).union(eyes.translate((length, 0, 0)))

    def _place(self, part, elevation_deg, point):
        """Rotate a +x part to ``elevation_deg`` above horizontal (about y) and move its origin to ``point``."""
        return part.rotate((0, 0, 0), (0, 1, 0), -elevation_deg).translate(point)

    def _arm_assembly(self, p):
        sx, sy, sz = self.shoulder_point(p)
        pts = self.arm_points(p)
        roof = p["ground_clearance"] + p["hull_height"]
        out = self._pedestal(p).translate((sx, sy, roof))
        out = out.union(self._place(self._boom(p), p["boom_deg"], (sx, sy, sz)))
        ex, ez = pts["elbow"]
        out = out.union(self._place(self._stick(p), p["stick_deg"], (ex, sy, ez)))
        wx, wz = pts["wrist"]
        out = out.union(self._place(self._saw_head(p), p["head_deg"], (wx, sy, wz)))
        if self.has_cylinders(p):
            for which, (ox, oz) in (("lift", (sx, sz)), ("stick", (ex, ez))):
                a, b = (self._lift_ends(p, p["boom_deg"]) if which == "lift"
                        else self._stick_ends(p, p["boom_deg"], p["stick_deg"]))
                L = math.hypot(b[0] - a[0], b[1] - a[1])
                ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
                out = out.union(self._place(self._cylinder(p, L), ang, (ox + a[0], sy, oz + a[1])))
        return out

    def build(self, p):
        part = p["part"]
        if part in ("pedestal", "boom", "stick", "saw_head", "cylinder", "rover"):
            r_ped = p["pedestal_diameter"] / 2
            x_front = p["hull_length"] / 2 - p["hull_chamfer"] - (p["sensor_box"] if p["sensor_box"] > 0 else 0.0)
            if p["boom_x"] + r_ped > x_front or p["boom_x"] - r_ped < -p["hull_length"] / 2 + p["hull_chamfer"]:
                raise ValueError("the pedestal must stand on the roof behind the sensor block")
            if p["head_length"] < p["saw_body_length"] * 0.5:
                raise ValueError("head_length must leave room for the saw body (at least half its length)")
            if self.has_cylinders(p):
                if p["lift_lug_x"] <= 0 or p["stick_heel"] <= 0:
                    raise ValueError("cylinders need lift_lug_x and stick_heel > 0")
                if self.lift_lever(p) < p["cylinder_diameter"] or self.stick_lever(p) < 0.5 * p["cylinder_diameter"]:
                    raise ValueError("a cylinder's lever arm is too short in the ready pose (its lug geometry)")
        if part == "pedestal":
            return self._pedestal(p)
        if part == "boom":
            return self._boom(p)
        if part == "stick":
            return self._stick(p)
        if part == "saw_head":
            return self._saw_head(p)
        if part == "cylinder":
            if not self.has_cylinders(p):
                raise ValueError("no cylinders: cylinder_diameter is 0")
            return self._cylinder(p, self.cylinder_length(p, "lift"))
        if part != "rover":
            return super().build(p)
        rover = super().build({**p, "part": "rover"})
        return rover.union(self._arm_assembly(p))


class PerseusMini(PerseusRover):
    """Perseus Mini: the same machine at ~30 kg (Pekari-class tracks and drives, a one-hand pruning chainsaw, joint
    modules instead of cylinders) for small branches and gardening (notebook 38)."""

    parameters = ([Parameter("part", "rover", choices=PARTS_MINI, description="what to build")]
                  + _pekari_params(MINI_CHASSIS) + _arm_params(MINI_ARM))
