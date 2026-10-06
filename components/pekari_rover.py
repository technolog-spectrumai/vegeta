import math
import cadquery as cq
from vegeta.dedalus import Design, Parameter


def _hull_of_circles(circles, n=360):
    """Convex hull [(x, z)] of circles ``[(x, z, r)]``, each sampled at ``n`` points (Andrew's monotone chain)."""
    pts = sorted({(round(x + r * math.cos(2 * math.pi * k / n), 9), round(z + r * math.sin(2 * math.pi * k / n), 9))
                  for x, z, r in circles for k in range(n)})

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _perimeter(poly):
    return sum(math.dist(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly)))


class PekariRover(Design):
    """Pekari Rover: a small tracked (caterpillar) rover, ~20 kg with ~5 kg payload — a hull box between two track
    modules, a sheet-metal basket on the roof for the payload (the default configuration; ``basket_height`` 0 = none). Each track module: a side frame plate on the hull, a rear drive sprocket (raised), a front idler on a
    tensioner slide, road wheels in pairs on bogies, a return roller, and a belt of hinged links (pins across the
    width, a grouser under each link, a centre guide horn on top running in the grooves of the wheels; the sprocket
    is two toothed discs either side of the horns, seating on the link knuckles). x forward, y left, z up, ground at
    z = 0 (the grouser tips).

    ``part`` selects the whole rover, the hull, the basket (own frame: the floor's underside on z = 0), one track module (in its own frame: the track centre plane at
    y = 0), or one sprocket, idler, road wheel, track link or the gearbox's final-stage sun pinion (each in its own
    frame, the axis along y through the origin; the link with its rear pin at x = −pitch/2), for FEA and the mass
    budget."""

    parameters = [
        Parameter("part", "rover", choices=("rover", "hull", "track_module", "sprocket", "idler", "road_wheel",
                                            "track_link", "pinion", "basket"), description="what to build"),
        Parameter("hull_length", 560.0, "mm", min=100),
        Parameter("hull_width", 300.0, "mm", min=50),
        Parameter("hull_height", 140.0, "mm", min=30),
        Parameter("hull_chamfer", 15.0, "mm", min=1),
        Parameter("ground_clearance", 90.0, "mm", min=20, description="hull floor above the ground"),
        Parameter("sensor_box", 80.0, "mm", min=0, description="sensor block on the hull roof, front (0 = none)"),
        Parameter("basket_length", 380.0, "mm", min=0, description="payload basket on the roof (the default configuration)"),
        Parameter("basket_width", 260.0, "mm", min=0),
        Parameter("basket_height", 120.0, "mm", min=0, description="0 = no basket"),
        Parameter("basket_wall", 1.5, "mm", min=0.5, description="Al sheet"),
        Parameter("basket_x", -70.0, "mm", description="basket centre along x (behind the sensor block)"),
        Parameter("frame_offset", 56.0, "mm", min=10, description="hull side to the track centre plane"),
        Parameter("frame_thickness", 6.0, "mm", min=1, description="track side frame plate"),
        Parameter("track_width", 80.0, "mm", min=20),
        Parameter("track_pitch", 31.0, "mm", min=8, description="link pitch (pin to pin)"),
        Parameter("link_thickness", 8.0, "mm", min=2, description="link body, the pin at mid-thickness"),
        Parameter("grouser_height", 6.0, "mm", min=0),
        Parameter("guide_height", 12.0, "mm", min=2, description="centre guide horn above the link"),
        Parameter("guide_width", 16.0, "mm", min=2),
        Parameter("pin_diameter", 5.0, "mm", min=1),
        Parameter("sprocket_teeth", 12, "", min=6, max=40),
        Parameter("sprocket_x", -290.0, "mm", description="drive sprocket axle (rear)"),
        Parameter("sprocket_z", 140.0, "mm", min=20),
        Parameter("idler_diameter", 110.0, "mm", min=20),
        Parameter("idler_x", 290.0, "mm", description="idler axle (front), on the tensioner slide"),
        Parameter("idler_z", 110.0, "mm", min=20),
        Parameter("road_wheel_diameter", 70.0, "mm", min=20),
        Parameter("road_wheels", 4, "", min=2, max=10, description="per side, in pairs on bogies"),
        Parameter("road_wheel_spacing", 120.0, "mm", min=20),
        Parameter("return_roller_diameter", 40.0, "mm", min=10),
        Parameter("return_roller_z", 162.0, "mm", min=20),
        Parameter("axle_diameter", 10.0, "mm", min=2, description="road wheel, idler and roller axles"),
        Parameter("sprocket_bore", 12.0, "mm", min=2, description="gearbox output shaft"),
        Parameter("pinion_module", 0.8, "mm", min=0.2, description="gearbox final stage"),
        Parameter("pinion_teeth", 18, "", min=8, max=60),
        Parameter("pinion_width", 10.0, "mm", min=1),
        Parameter("pinion_bore", 5.0, "mm", min=1),
    ]

    # ---- derived geometry shared with the notebook (kept here so CAD and mechanics agree) ----
    @staticmethod
    def track_center_y(p):
        """Track centre plane from the vehicle centre [mm]."""
        return p["hull_width"] / 2 + p["frame_offset"]

    @staticmethod
    def track_gauge(p):
        """Track centre to track centre [mm] (B of the skid-steer formulas)."""
        return 2 * PekariRover.track_center_y(p)

    @staticmethod
    def sprocket_pitch_radius(p):
        """Pitch radius [mm] of the sprocket: the pins' circle, p / (2 sin(π/z))."""
        return p["track_pitch"] / (2 * math.sin(math.pi / p["sprocket_teeth"]))

    @staticmethod
    def road_wheel_z(p):
        """Road wheel axle height [mm]: the wheel runs on the link's top face."""
        return p["grouser_height"] + p["link_thickness"] + p["road_wheel_diameter"] / 2

    @staticmethod
    def road_wheel_x(p):
        """Road wheel axles along x [mm], centred on the hull."""
        n, s = int(p["road_wheels"]), p["road_wheel_spacing"]
        return [(i - (n - 1) / 2) * s for i in range(n)]

    @staticmethod
    def contact_length(p):
        """Ground contact length of one track [mm]: the outer road wheels' span plus one wheel diameter (the belt
        between and just beyond the wheels carries the load)."""
        return (int(p["road_wheels"]) - 1) * p["road_wheel_spacing"] + p["road_wheel_diameter"]

    @staticmethod
    def belt_circles(p):
        """The circles the belt's pin line wraps [(x, z, r) mm]: the sprocket's pitch circle, the idler, the outer
        road wheels and the return roller (each wheel's radius plus half a link: the pins sit mid-link)."""
        h = p["link_thickness"] / 2
        zr = PekariRover.road_wheel_z(p)
        xs = PekariRover.road_wheel_x(p)
        return [(p["sprocket_x"], p["sprocket_z"], PekariRover.sprocket_pitch_radius(p)),
                (p["idler_x"], p["idler_z"], p["idler_diameter"] / 2 + h),
                (xs[0], zr, p["road_wheel_diameter"] / 2 + h), (xs[-1], zr, p["road_wheel_diameter"] / 2 + h),
                (0.0, p["return_roller_z"], p["return_roller_diameter"] / 2 + h)]

    @staticmethod
    def belt_length(p):
        """Length of the belt's pin line [mm] (the convex hull of ``belt_circles``)."""
        return _perimeter(_hull_of_circles(PekariRover.belt_circles(p), 720))

    @staticmethod
    def link_count(p):
        """Links per track [-] (the pin line rounded up to whole links) and the take-up [mm] the idler slide
        absorbs (n p − belt length; the idler moves about half of it)."""
        L = PekariRover.belt_length(p)
        n = math.ceil(L / p["track_pitch"])
        return n, n * p["track_pitch"] - L

    @staticmethod
    def overall(p):
        """Overall length, width and height [mm] of the rover."""
        out = [(x, z, r + p["link_thickness"] / 2 + p["grouser_height"]) for x, z, r in PekariRover.belt_circles(p)]
        xmin = min(x - r for x, _, r in out)
        xmax = max(x + r for x, _, r in out)
        top_track = max(z + r for _, z, r in out)
        hull_top = p["ground_clearance"] + p["hull_height"]
        height = max(top_track, hull_top + (p["sensor_box"] * 0.5 if p["sensor_box"] > 0 else 0.0),
                     hull_top + (p["basket_height"] if p["basket_height"] > 0 else 0.0))
        return {"length": max(xmax, p["hull_length"] / 2) - min(xmin, -p["hull_length"] / 2),
                "width": 2 * (PekariRover.track_center_y(p) + p["track_width"] / 2), "height": height,
                "hull_top": hull_top}

    @staticmethod
    def _tooth_points(m, z, alpha_deg=20.0, n_points=8):
        """One involute tooth [(x, y) mm] on +x from root to root (the same construction as ``gears.involute_profile``)."""
        a = math.radians(alpha_deg)
        r = m * z / 2.0
        rb, ra, rf = r * math.cos(a), r + m, r - 1.25 * m
        inv = lambda t: math.tan(t) - t  # noqa: E731
        half = math.pi / (2 * z) + inv(a)
        r0 = max(rb, rf)
        flank = []
        for k in range(n_points + 1):
            rr = r0 + (ra - r0) * k / n_points
            th = half - inv(math.acos(min(1.0, rb / rr)))
            flank.append((rr * math.cos(th), rr * math.sin(th)))
        upper = [(rf * math.cos(half), rf * math.sin(half))] + flank
        return upper + [(x, -y) for x, y in reversed(upper)]

    # ---- parts ----
    def _grooved_wheel(self, p, d, bore):
        """A wheel the width of the track with a groove for the guide horns, axis along y."""
        b, r = p["track_width"], d / 2
        groove_w = p["guide_width"] + 6.0
        groove_r = max(r - p["guide_height"] - 3.0, bore / 2 + 3.0)
        wheel = cq.Workplane("XZ").circle(r).extrude(b / 2, both=True)
        ring = cq.Workplane("XZ").circle(r + 1.0).circle(groove_r).extrude(groove_w / 2, both=True)
        hole = cq.Workplane("XZ").circle(bore / 2).extrude(b, both=True)
        return wheel.cut(ring).cut(hole)

    def _sprocket(self, p):
        """Two toothed discs either side of the guide horns on a hub; the tooth gaps seat the link knuckles."""
        R = self.sprocket_pitch_radius(p)
        z = int(p["sprocket_teeth"])
        knuckle = p["link_thickness"] / 2 + 1.5
        disc_t, b = 10.0, p["track_width"]
        y_disc = (p["guide_width"] / 2 + 4.0 + b / 2) / 2                      # the middle of the free band
        tip = R + knuckle + 1.0
        disc = cq.Workplane("XZ").circle(tip).extrude(disc_t / 2, both=True)
        for k in range(z):
            t = 2 * math.pi * k / z
            seat = (cq.Workplane("XZ").center(R * math.cos(t), R * math.sin(t)).circle(knuckle + 0.5)
                    .extrude(disc_t, both=True))
            disc = disc.cut(seat)
        disc = disc.cut(cq.Workplane("XZ").circle(R - knuckle - 8.0).circle(28.0).extrude(disc_t, both=True))
        spokes = None
        for k in range(4):
            spoke = (cq.Workplane("XZ").rect(2 * (R - knuckle - 6.0), 12.0).extrude(disc_t / 2, both=True)
                     .rotate((0, 0, 0), (0, 1, 0), 45.0 * k))
            spokes = spoke if spokes is None else spokes.union(spoke)
        disc = disc.union(spokes)
        hub = cq.Workplane("XZ").circle(28.0).extrude(y_disc + disc_t / 2, both=True)
        out = hub.union(disc.translate((0, y_disc, 0))).union(disc.translate((0, -y_disc, 0)))
        return out.cut(cq.Workplane("XZ").circle(p["sprocket_bore"] / 2).extrude(b, both=True))

    def _link(self, p):
        """One track link, pins along y at x = ±pitch/2, mid-thickness at z = 0: body, grouser below, guide horn on
        top, two knuckles at the front pin (outer quarters), one at the rear pin (middle half), pin bores."""
        P, b, t = p["track_pitch"], p["track_width"], p["link_thickness"]
        kr = t / 2 + 1.5
        gap = 0.6
        body = cq.Workplane("XY").box(P - 2 * kr - 2 * gap, b, t)
        q = b / 4
        for y0, y1, x0, x1 in ((-b / 2, -q - gap / 2, 0.0, P / 2), (q + gap / 2, b / 2, 0.0, P / 2),
                               (-q + gap / 2, q - gap / 2, -P / 2, 0.0)):
            web = cq.Workplane("XY").box(x1 - x0, y1 - y0, t).translate(((x0 + x1) / 2, (y0 + y1) / 2, 0))
            knuckle = (cq.Workplane("XZ").center(x1 if x1 > 0 else x0, 0).circle(kr).extrude(-(y1 - y0))
                       .translate((0, y0, 0)))
            body = body.union(web).union(knuckle)
        if p["grouser_height"] > 0:
            body = body.union(cq.Workplane("XY").box(P * 0.3, b, p["grouser_height"])
                              .translate((0, 0, -t / 2 - p["grouser_height"] / 2)))
        horn = (cq.Workplane("XY").box(P * 0.45, p["guide_width"], p["guide_height"])
                .translate((0, 0, t / 2 + p["guide_height"] / 2)).edges("|Y and >Z").fillet(min(3.0, P * 0.1)))
        body = body.union(horn)
        for x in (-P / 2, P / 2):
            body = body.cut(cq.Workplane("XZ").center(x, 0).circle(p["pin_diameter"] / 2).extrude(b, both=True))
        return body

    def _pinion(self, p):
        m, z = p["pinion_module"], int(p["pinion_teeth"])
        tooth = list(reversed(self._tooth_points(m, z)))                   # counter-clockwise, root to root
        pts = []
        for k in range(z):
            t = 2 * math.pi * k / z
            c, s = math.cos(t), math.sin(t)
            pts += [(x * c - y * s, x * s + y * c) for x, y in tooth]
        gear = cq.Workplane("XZ").polyline(pts).close().extrude(p["pinion_width"] / 2, both=True)
        return gear.cut(cq.Workplane("XZ").circle(p["pinion_bore"] / 2).extrude(p["pinion_width"], both=True))

    def _belt(self, p):
        """The belt as a band (pin line ± half a link, the grousers outside): visual and mass of the closed loop."""
        h = p["link_thickness"] / 2
        circles = self.belt_circles(p)
        inner = _hull_of_circles([(x, z, r - h + 0.5) for x, z, r in circles], 72)
        outer = _hull_of_circles([(x, z, r + h + p["grouser_height"]) for x, z, r in circles], 72)
        b = p["track_width"]
        ring = cq.Workplane("XZ").polyline(outer).close().extrude(b / 2, both=True)
        return ring.cut(cq.Workplane("XZ").polyline(inner).close().extrude(b, both=True))

    def _track_module(self, p):
        """One track in its own frame (track centre plane y = 0, the hull side at −y)."""
        b, ft = p["track_width"], p["frame_thickness"]
        y_frame = -(b / 2 + 10.0 + ft / 2)
        y_bogie = -(b / 2 + 3.0 + 3.0)
        zr = self.road_wheel_z(p)
        xs = self.road_wheel_x(p)
        x0, x1 = min(p["sprocket_x"], p["idler_x"]), max(p["sprocket_x"], p["idler_x"])
        z_lo, z_hi = zr + 12.0, min(p["return_roller_z"], p["sprocket_z"], p["idler_z"]) + 20.0
        frame = (cq.Workplane("XZ").center((x0 + x1) / 2, (z_lo + z_hi) / 2).rect(x1 - x0, z_hi - z_lo)
                 .extrude(ft / 2, both=True).edges("|Y").fillet(min(15.0, (z_hi - z_lo) / 3)).translate((0, y_frame, 0)))
        out = frame
        for x, z, d, bore in [(p["idler_x"], p["idler_z"], p["idler_diameter"], p["axle_diameter"]),
                              (0.0, p["return_roller_z"], p["return_roller_diameter"], p["axle_diameter"])] + \
                             [(x, zr, p["road_wheel_diameter"], p["axle_diameter"]) for x in xs]:
            out = out.union(self._grooved_wheel(p, d, bore).translate((x, 0, z)))
        out = out.union(self._sprocket(p).translate((p["sprocket_x"], 0, p["sprocket_z"])))
        z_piv = zr + 22.0
        for i in range(0, len(xs) - 1, 2):
            xa, xb = xs[i], xs[i + 1]
            bogie = (cq.Workplane("XZ").polyline([(xa - 10, zr - 8), (xb + 10, zr - 8), (xb + 10, zr + 8),
                                                  ((xa + xb) / 2 + 14, z_piv + 8), ((xa + xb) / 2 - 14, z_piv + 8),
                                                  (xa - 10, zr + 8)]).close().extrude(3.0, both=True)
                     .translate((0, y_bogie, 0)))
            out = out.union(bogie)
        # axles: from the frame through the wheels' hubs
        for x, z in [(p["idler_x"], p["idler_z"]), (0.0, p["return_roller_z"]), (p["sprocket_x"], p["sprocket_z"])] + \
                    [((xs[i] + xs[i + 1]) / 2, z_piv) for i in range(0, len(xs) - 1, 2)]:
            axle = (cq.Workplane("XZ").center(x, z).circle(p["axle_diameter"] / 2)
                    .extrude((abs(y_frame) + ft / 2 + 2.0) / 2, both=True).translate((0, y_frame / 2, 0)))
            out = out.union(axle)
        return out.union(self._belt(p))

    def _basket(self, p):
        """Sheet-metal basket: floor, four walls and a 10 mm rim lip outwards (own frame: the floor's underside on
        z = 0), as the Onager Sweeper's."""
        L, W, H, t = p["basket_length"], p["basket_width"], p["basket_height"], p["basket_wall"]
        outer = cq.Workplane("XY").box(L, W, H, centered=(True, True, False))
        inner = cq.Workplane("XY").box(L - 2 * t, W - 2 * t, H, centered=(True, True, False)).translate((0, 0, t))
        rim = cq.Workplane("XY").box(L + 20.0, W + 20.0, 2.0, centered=(True, True, False)).translate((0, 0, H - 2.0))
        rim = rim.cut(cq.Workplane("XY").box(L - 2 * t, W - 2 * t, 10.0, centered=(True, True, False)).translate((0, 0, H - 6.0)))
        return outer.cut(inner).union(rim)

    @staticmethod
    def has_basket(p):
        return p["basket_height"] > 0 and p["basket_length"] > 0 and p["basket_width"] > 0

    def _hull(self, p):
        Lh, Wh, Hh, ch = p["hull_length"], p["hull_width"], p["hull_height"], p["hull_chamfer"]
        z0 = p["ground_clearance"]
        hull = cq.Workplane("XY").box(Lh, Wh, Hh, centered=(True, True, False)).translate((0, 0, z0)).edges("|Y").chamfer(ch)
        if p["sensor_box"] > 0:
            s = p["sensor_box"]
            box = (cq.Workplane("XY").box(s, 1.5 * s, 0.5 * s, centered=(True, True, False))
                   .translate((Lh / 2 - s / 2 - ch, 0, z0 + Hh)))
            hull = hull.union(box)
        return hull

    def build(self, p):
        part = p["part"]
        if part == "sprocket":
            return self._sprocket(p)
        if part == "idler":
            return self._grooved_wheel(p, p["idler_diameter"], p["axle_diameter"])
        if part == "road_wheel":
            return self._grooved_wheel(p, p["road_wheel_diameter"], p["axle_diameter"])
        if part == "track_link":
            return self._link(p)
        if part == "pinion":
            return self._pinion(p)
        if part == "basket":
            if not self.has_basket(p):
                raise ValueError("no basket: basket_length, basket_width and basket_height must be > 0")
            return self._basket(p)
        if p["idler_x"] <= p["sprocket_x"]:
            raise ValueError("the idler (front) must be ahead of the sprocket (rear)")
        xs = self.road_wheel_x(p)
        if xs[0] - p["road_wheel_diameter"] / 2 < p["sprocket_x"] or xs[-1] + p["road_wheel_diameter"] / 2 > p["idler_x"]:
            raise ValueError("the road wheels must fit between the sprocket and the idler")
        if p["frame_offset"] < p["track_width"] / 2 + 10.0 + p["frame_thickness"]:
            raise ValueError("frame_offset must leave room for half the track, the frame plate and a 10 mm gap")
        if self.has_basket(p):
            x_front = p["hull_length"] / 2 - (p["sensor_box"] + p["hull_chamfer"] if p["sensor_box"] > 0 else 0.0)
            if p["basket_x"] - p["basket_length"] / 2 < -p["hull_length"] / 2 or p["basket_x"] + p["basket_length"] / 2 > x_front:
                raise ValueError("the basket must sit on the roof, behind the sensor block")
        if int(p["road_wheels"]) % 2:
            raise ValueError("road_wheels must be even (pairs on bogies)")
        if part == "track_module":
            return self._track_module(p)
        hull = self._hull(p)
        if part == "hull":
            return hull
        yc = self.track_center_y(p)
        module = self._track_module(p)
        left = module.translate((0, yc, 0))
        right = module.mirror("XZ").translate((0, -yc, 0))
        rover = hull.union(left).union(right)
        if self.has_basket(p):
            roof = p["ground_clearance"] + p["hull_height"]
            rover = rover.union(self._basket(p).translate((p["basket_x"], 0, roof)))
        return rover
