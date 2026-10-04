from dataclasses import replace

import cadquery as cq
from vegeta.dedalus import Parameter

from .onager_manus import OnagerManus

#: The Sentinel and Manus parameters the Sweeper changes: a bulky, low, rounded street-cleaning body (the hull
#: floor 300 mm over the road so the disc broom and the suction hood fit under it), shorter legs set in a fixed
#: stance (no active suspension: the leg modules' brakes hold the ride height), the Manus arms set further back
#: with longer links so that a pincer reaches both the road ahead of the front wheels and the basket behind it.
SWEEPER_CHASSIS = {
    "hull_length": (2200.0, "a bulky body: the hopper, fan, battery and ducting inside"),
    "hull_width": (1000.0, "the body spans the broom and the hood"),
    "hull_height": (650.0, "floor at 300 mm, roof at 950 mm"),
    "hull_chamfer": (150.0, "rounded: the hull edges are filleted with this radius (the Sentinel chamfers)"),
    "hull_bottom": (300.0, "the hull floor over the road: room for the hood (170 mm) and its duct"),
    "shoulder_x": (950.0, "shoulders fore and aft of the long hull"),
    "upper_leg_length": (420.0, "short legs: the hull sits low (Sentinel 520)"),
    "lower_leg_length": (420.0, "(Sentinel 540)"),
    "hip_angle_deg": (40.0, "upper leg behind vertical: the fixed stance"),
    "knee_angle_deg": (50.0, "lower leg ahead of vertical: the fixed stance (Sentinel 55°)"),
    "turret_length": (0.0, "no turret"),
    "turret_height": (0.0, "no turret"),
    "mast_height": (250.0, "the display post on the nose (OnagerSweeper places it at display_x)"),
    "sensor_head": (150.0, "the display / sensor head cube on the post"),
    "arm_x": (600.0, "shoulder pedestals: the pincers reach the road ahead of the front wheels and the basket"),
    "arm_y": (300.0, "pedestals either side of the centre line"),
    "upper_arm_length": (700.0, "longer than the Manus's 600: the basket is 1.1 m behind the shoulders"),
    "forearm_length": (650.0, "(Manus 550)"),
}


def _manus_params():
    out = []
    for p in OnagerManus.parameters:
        if p.name == "part":
            continue
        if p.name in SWEEPER_CHASSIS:
            v, d = SWEEPER_CHASSIS[p.name]
            p = replace(p, default=v, description=d)
        out.append(p)
    return out


class OnagerSweeper(OnagerManus):
    """Onager Sweeper: the Onager series' outdoor street-cleaning unit — the wheel-leg chassis of the Sentinel set
    low in a fixed stance (the leg brakes hold the ride height: no active suspension), a bulky rounded body, a
    **rotary disc broom** under the hull centre with a **suction hood** right behind it (a square duct up to the fan
    and the hopper inside the hull), the Manus's two **pincer arms** on the roof — they pick up what the broom
    cannot (a bottle, a brick, a branch) and drop it into the **basket** on the back of the roof — and a display
    post on the nose. x forward, y left, z up, ground at z = 0.

    ``part`` adds to the Manus's parts: ``broom_disc`` (the disc with its spindle boss, own frame: the spindle
    axis z through the origin, the disc's top face on z = 0, bristles not modelled), ``hood`` (the CFD body in the
    world frame: hood, duct, broom disc and housing; no bristles — they are porous), ``hood_shell`` (the hood and
    its duct alone, world frame: its mass and FEA) and ``basket`` (own frame: its floor's underside on z = 0, for
    FEA)."""

    parameters = [Parameter("part", "robot", choices=("robot", "hull", "upper_leg", "lower_leg", "wheel", "upper_arm",
                                                      "forearm", "jaw", "broom_disc", "hood", "hood_shell", "basket"),
                            description="what to build")] + _manus_params() + [
        Parameter("display_x", 900.0, "mm", min=0, description="the display post on the roof, ahead of the hull centre"),
        Parameter("broom_x", 350.0, "mm", description="disc broom centre ahead of the hull centre"),
        Parameter("broom_diameter", 600.0, "mm", min=100, description="disc broom over the bristle tips"),
        Parameter("broom_disc_thickness", 20.0, "mm", min=5, description="the Al disc the bristle tufts are set in"),
        Parameter("bristle_length", 60.0, "mm", min=10, description="bristles below the disc; their tips on the road"),
        Parameter("broom_housing_diameter", 200.0, "mm", min=50, description="the drive housing between the disc and the floor"),
        Parameter("hood_x", -100.0, "mm", description="suction hood centre (behind the broom)"),
        Parameter("hood_width", 500.0, "mm", min=100, description="hood across the track (y)"),
        Parameter("hood_length", 250.0, "mm", min=50, description="hood along the track (x): the time a piece of litter spends under the duct"),
        Parameter("hood_height", 170.0, "mm", min=50, description="top of the hood roof above the road"),
        Parameter("hood_gap", 30.0, "mm", min=5, description="the lips above the road: the air comes in here (the front lip is a rubber flap bulky litter pushes through)"),
        Parameter("hood_wall", 6.0, "mm", min=2, description="hood and duct wall (Al sheet)"),
        Parameter("duct_inner", 140.0, "mm", min=50, description="square suction duct, inner width"),
        Parameter("duct_wall", 10.0, "mm", min=2, description="duct wall (Al): the CFD mesh blocks meet inside it"),
        Parameter("basket_x", -650.0, "mm", description="basket centre (on the back of the roof): its front wall clear of a dropped box"),
        Parameter("basket_length", 700.0, "mm", min=100),
        Parameter("basket_width", 900.0, "mm", min=100),
        Parameter("basket_height", 400.0, "mm", min=50, description="basket walls above the roof"),
        Parameter("basket_wall", 2.0, "mm", min=1, description="basket sheet (Al 5754)"),
    ]

    # ---- derived geometry shared with the Chiron robot and the CFD
    @staticmethod
    def hood_box(p):
        """The hood's outer box in the world [mm]: (xmin, ymin, zmin, xmax, ymax, zmax); zmin is the lip height."""
        return (p["hood_x"] - p["hood_length"] / 2, -p["hood_width"] / 2, p["hood_gap"],
                p["hood_x"] + p["hood_length"] / 2, p["hood_width"] / 2, p["hood_height"])

    # ---- parts
    def _hull(self, p):
        """The rounded body with the shoulder bosses, the nose block and the display post (no turret, no mast at the
        Sentinel's place)."""
        Lh, Wh, Hh, r = p["hull_length"], p["hull_width"], p["hull_height"], p["hull_chamfer"]
        z0 = p["hull_bottom"]
        hull = cq.Workplane("XY").box(Lh, Wh, Hh, centered=(True, True, False)).translate((0, 0, z0))
        hull = hull.edges("|Y").fillet(r).edges("|X").fillet(min(r * 0.6, Hh / 2 - r - 1))
        nose = (cq.Workplane("XY").box(100.0, Wh * 0.5, Hh * 0.35, centered=(True, True, False))
                .translate((Lh / 2 - 20, 0, z0 + Hh * 0.3)).edges("|X").fillet(30.0))
        hull = hull.union(nose)
        zs = self.shoulder_height(p)
        for sx in (-1, 1):
            for sy in (-1, 1):
                boss = (cq.Workplane("XZ").circle(p["shoulder_boss_diameter"] / 2).extrude(p["shoulder_boss_length"])
                        .translate((sx * p["shoulder_x"], sy * (Wh / 2 + p["shoulder_boss_length"]) if sy > 0 else sy * Wh / 2, zs)))
                hull = hull.union(boss)
        roof = z0 + Hh
        if p["mast_height"] > 0:
            post = cq.Workplane("XY").circle(p["mast_diameter"] / 2).extrude(p["mast_height"]).translate((p["display_x"], 0, roof))
            hull = hull.union(post)
            if p["sensor_head"] > 0:
                s = p["sensor_head"]
                head = (cq.Workplane("XY").box(s * 0.6, s * 1.6, s, centered=(True, True, False))
                        .translate((p["display_x"], 0, roof + p["mast_height"])).edges("|Y").fillet(s * 0.15))
                hull = hull.union(head)
        return hull

    def _broom_disc(self, p):
        """The disc with its spindle boss (own frame: top face on z = 0, the disc below it), the spindle bore."""
        r, t = p["broom_diameter"] / 2 - p["bristle_length"] * 0.3, p["broom_disc_thickness"]
        disc = cq.Workplane("XY").circle(r).extrude(-t)
        boss = cq.Workplane("XY").circle(p["broom_housing_diameter"] / 2 * 0.6).extrude(t * 1.5)
        bore = cq.Workplane("XY").circle(20.0).extrude(t * 4, both=True)
        return disc.union(boss).cut(bore)

    def _bristles(self, p):
        """The bristle ring (drawn solid; porous in the flow) under the disc."""
        r = p["broom_diameter"] / 2
        return (cq.Workplane("XY").circle(r).circle(r * 0.7).extrude(p["bristle_length"])
                .translate((p["broom_x"], 0, 0)))

    def _broom_assembly(self, p, bristles=True):
        """Disc, housing and (optionally) bristles in the world."""
        z_disc = p["bristle_length"]
        disc = self._broom_disc(p).translate((p["broom_x"], 0, z_disc + p["broom_disc_thickness"]))
        housing = (cq.Workplane("XY").circle(p["broom_housing_diameter"] / 2)
                   .extrude(p["hull_bottom"] - (z_disc + p["broom_disc_thickness"]) + 10.0)
                   .translate((p["broom_x"], 0, z_disc + p["broom_disc_thickness"])))
        out = disc.union(housing)
        return out.union(self._bristles(p)) if bristles else out

    def _hood(self, p, duct_top):
        """The hood (open underneath, lips at hood_gap) and its square duct up to ``duct_top`` [mm, world]."""
        x0, y0, z0, x1, y1, z1 = self.hood_box(p)
        w = p["hood_wall"]
        L, W, H = x1 - x0, y1 - y0, z1 - z0
        outer = cq.Workplane("XY").box(L, W, H, centered=(True, True, False)).translate((p["hood_x"], 0, z0))
        inner = cq.Workplane("XY").box(L - 2 * w, W - 2 * w, H - w, centered=(True, True, False)).translate((p["hood_x"], 0, z0 - 1.0))
        hood = outer.cut(inner)
        di, dw = p["duct_inner"], p["duct_wall"]
        duct = (cq.Workplane("XY").box(di + 2 * dw, di + 2 * dw, duct_top - (z1 - w), centered=(True, True, False))
                .translate((p["hood_x"], 0, z1 - w)))
        bore = (cq.Workplane("XY").box(di, di, duct_top - (z1 - w) + 2.0, centered=(True, True, False))
                .translate((p["hood_x"], 0, z1 - w - 1.0)))
        return hood.union(duct).cut(bore)

    def _basket(self, p):
        """Sheet-metal basket: floor and four walls (own frame: the floor's underside on z = 0)."""
        L, W, H, t = p["basket_length"], p["basket_width"], p["basket_height"], p["basket_wall"]
        outer = cq.Workplane("XY").box(L, W, H, centered=(True, True, False))
        inner = cq.Workplane("XY").box(L - 2 * t, W - 2 * t, H, centered=(True, True, False)).translate((0, 0, t))
        basket = outer.cut(inner)
        # a rolled rim (a 20 mm lip outwards) stiffens the walls
        rim = cq.Workplane("XY").box(L + 40.0, W + 40.0, 3.0, centered=(True, True, False)).translate((0, 0, H - 3.0))
        rim = rim.cut(cq.Workplane("XY").box(L - 2 * t, W - 2 * t, 10.0, centered=(True, True, False)).translate((0, 0, H - 6.0)))
        return basket.union(rim)

    def build(self, p):
        part = p["part"]
        if part == "broom_disc":
            return self._broom_disc(p)
        if part == "basket":
            return self._basket(p)
        if part == "hood":
            return self._hood(p, p["hull_bottom"] + 20.0).union(self._broom_assembly(p, bristles=False))
        if part == "hood_shell":
            return self._hood(p, p["hull_bottom"] + 10.0)
        x0, y0, z0, x1, y1, z1 = self.hood_box(p)
        if z1 + p["hood_wall"] >= p["hull_bottom"]:
            raise ValueError("the hood must fit under the hull floor (hood_height + wall < hull_bottom)")
        if abs(p["broom_x"] - p["hood_x"]) < p["broom_diameter"] / 2 + p["hood_length"] / 2:
            raise ValueError("the disc broom and the hood overlap (broom_x, hood_x)")
        body = super().build(p)
        if part not in ("robot", "hull"):
            return body
        roof = p["hull_bottom"] + p["hull_height"]
        body = body.union(self._hood(p, p["hull_bottom"] + 10.0)).union(self._broom_assembly(p))
        body = body.union(self._basket(p).translate((p["basket_x"], 0, roof)))
        return body
