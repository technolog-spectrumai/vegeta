import math

try:
    import cadquery as cq
except ImportError:                 # the sizing and the studies run without the CAD
    cq = None
from vegeta.dedalus import Design, Parameter


class Drako(Design):
    """Drako: a heavy-lift hexacopter for lifting and transporting packages (10 kg design, 15 kg maximum; tote 600 x 400 x 350 mm).

    Six carbon-tube arms (``arm_count`` = 6, flat-six layout: one arm straight forward) bolted between two carbon
    centre plates; 30x10 propellers on 12S 120 kV motors at a 1650 mm motor-to-motor diagonal. Under the lower plate hangs a
    cargo cradle: a rectangular frame with four hook latches, one servo each, that close on a standard tote's lip. The
    package is held between four tall landing legs on two skids, so Drako lands over a package, latches it and takes
    off; at the drop point it lands (or hovers low) and opens the latches. x forward (along arm 0), y left, z up; z = 0
    the lower plate's underside, the ground at z = −``leg_height`` when landed.

    ``part``: ``drako`` (everything, with the bought parts — motors, propeller discs, battery, a package — drawn as
    blocks), ``frame`` (plates and arms), ``plate`` (one centre plate), ``arm`` (one arm with its motor mount),
    ``gear`` (legs and skids), ``cradle`` (cargo frame with latches). Single parts are in the assembly's frame.
    The sizing (masses, hover thrust, endurance, one-motor-out margin) is ``Drako.sizing(p)``."""

    parameters = [
        Parameter("part", "drako", choices=("drako", "frame", "plate", "arm", "gear", "cradle"),
                  description="what to build"),
        Parameter("arm_count", 6, "", min=6, max=6, description="hexacopter"),
        Parameter("wheelbase", 1650.0, "mm", min=400, description="motor-to-motor diagonal"),
        Parameter("prop_diameter", 30.0, "in", min=10, description="propeller diameter"),
        Parameter("plate_diameter", 420.0, "mm", min=120, description="centre plates (hexagon across corners)"),
        Parameter("plate_thickness", 3.0, "mm", min=1, description="carbon plate"),
        Parameter("plate_gap", 50.0, "mm", min=15, description="between the plates (arm clamps, wiring)"),
        Parameter("arm_diameter", 40.0, "mm", min=10, description="carbon arm tube outer diameter"),
        Parameter("arm_wall", 2.0, "mm", min=0.5, description="arm tube wall"),
        Parameter("mount_diameter", 90.0, "mm", min=20, description="motor mount disc"),
        Parameter("mount_thickness", 6.0, "mm", min=2),
        Parameter("leg_height", 450.0, "mm", min=150, description="lower plate above the ground, landed"),
        Parameter("leg_spread", 720.0, "mm", min=150, description="skid-to-skid distance (y)"),
        Parameter("skid_length", 800.0, "mm", min=200, description="skid tube along x"),
        Parameter("leg_diameter", 25.0, "mm", min=6),
        Parameter("cradle_x", 620.0, "mm", min=100, description="cargo frame inner length (x)"),
        Parameter("cradle_y", 420.0, "mm", min=100, description="cargo frame inner width (y)"),
        Parameter("cradle_bar", 20.0, "mm", min=5, description="cargo frame square tube"),
        Parameter("package_x", 600.0, "mm", min=50, description="package (tote) length"),
        Parameter("package_y", 400.0, "mm", min=50),
        Parameter("package_z", 350.0, "mm", min=20),
        Parameter("payload", 10.0, "kg", min=0, description="design payload (25 min point)"),
        Parameter("max_payload", 15.0, "kg", min=0, description="maximum payload"),
        Parameter("battery_cells", 12, "", min=6, description="12S semi-solid Li-ion (44.4 V nominal), hot-swappable 2 x 6S"),
        Parameter("battery_capacity", 64.0, "Ah", min=1),
        Parameter("show_bought", True, "", description="draw motors, propellers, battery and the package"),
    ]

    # ---- sizing: plain momentum theory with a figure of merit, round masses from catalogue parts
    MASSES_KG = {"frame (plates, folding arms, gear, cradle)": 4.0, "6 x motor 12S 120 kV + 80 A FOC ESC": 6 * 0.95,
                 "6 x 30-inch carbon propeller": 6 * 0.15,
                 "FC (triple IMU), dual GNSS, 45 km datalink, camera, wiring": 0.9,
                 "latches (4 servos) and package sensor": 0.6}

    @staticmethod
    def battery_mass(p):
        """semi-solid-state Li-ion pack at ~260 Wh/kg (cells, BMS, case)."""
        return p["battery_cells"] * 3.7 * p["battery_capacity"] / 260.0

    @classmethod
    def sizing(cls, p, rho=1.225, figure_of_merit=0.666, max_thrust_per_motor=13.85, reserve=0.15, eta_drive=0.85,
               mission_factor=0.70, cda_body=0.25, cda_package=0.15):
        """Take-off mass, hover power, endurance and one-motor-out margin [SI, kg]. ``max_thrust_per_motor`` in kgf is
        the 30x10 at 80 % of 120 kV x 44.4 V (4262 rpm) from Boreas BEMT (``drako_robot.boreas_system``), and
        ``figure_of_merit`` that BEMT's static figure of merit; ``reserve`` the battery fraction left at landing."""
        g = 9.81
        empty = sum(cls.MASSES_KG.values()) + cls.battery_mass(p)
        mtow = empty + p["payload"]
        n = p["arm_count"]
        disc = n * math.pi * (p["prop_diameter"] * 0.0254 / 2) ** 2
        w = mtow * g
        energy = p["battery_cells"] * 3.7 * p["battery_capacity"] * 3600 * (1 - reserve)
        t_max = n * max_thrust_per_motor
        # one motor out on a hexacopter: the opposite motor is throttled with it to keep yaw/roll trim, 4 left
        t_motor_out = (n - 2) * max_thrust_per_motor
        def hover(m):
            return (m * g) ** 1.5 / math.sqrt(2 * rho * disc) / figure_of_merit / eta_drive

        def tilt_at(v, m, cda):
            """Body tilt [deg] to hold ``v`` [m/s] level: tan = drag / weight."""
            return math.degrees(math.atan(0.5 * rho * v * v * cda / (m * g)))

        mmax = empty + p["max_payload"]
        # mission endurance: hover endurance x mission_factor (climb, cruise, gusts, descent; field-reported ratio)
        return {
            "empty_kg": round(empty, 2), "battery_kg": round(cls.battery_mass(p), 2),
            "battery_Wh": round(p["battery_cells"] * 3.7 * p["battery_capacity"]),
            "tow_design_kg": round(mtow, 2), "mtow_max_kg": round(mmax, 2),
            "disc_loading_max_N_m2": round(mmax * g / disc, 1),
            "thrust_to_weight_empty": round(t_max / empty, 2),
            "thrust_to_weight_design": round(t_max / mtow, 2),
            "thrust_to_weight_max": round(t_max / mmax, 2),
            "motor_out_T_W_max": round(t_motor_out / mmax, 2),
            "hover_power_empty_W": round(hover(empty)), "hover_power_design_W": round(hover(mtow)),
            "hover_power_max_W": round(hover(mmax)),
            "endurance_empty_min": round(energy / hover(empty) / 60 * mission_factor, 1),
            "endurance_design_min": round(energy / hover(mtow) / 60 * mission_factor, 1),
            "endurance_max_min": round(energy / hover(mmax) / 60 * mission_factor, 1),
            "tilt_80kmh_empty_deg": round(tilt_at(80 / 3.6, empty, cda_body), 1),
            "tilt_40kmh_max_deg": round(tilt_at(40 / 3.6, mmax, cda_body + cda_package), 1),
            "range_empty_km_at_60kmh": round(energy / hover(empty) / 3600 * mission_factor * 60 * 0.9, 1),
            "range_design_km_at_40kmh": round(energy / hover(mtow) / 3600 * mission_factor * 40, 1),
        }

    # ---- parts
    def _arm_angles(self, p):
        return [360.0 * k / p["arm_count"] for k in range(p["arm_count"])]

    def _hexagon(self, p, z):
        r = p["plate_diameter"] / 2
        pts = [(r * math.cos(math.radians(30 + 60 * k)), r * math.sin(math.radians(30 + 60 * k))) for k in range(6)]
        return cq.Workplane("XY").polyline(pts).close().extrude(p["plate_thickness"]).translate((0, 0, z))

    def _plates(self, p):
        lower = self._hexagon(p, 0.0)
        upper = self._hexagon(p, p["plate_thickness"] + p["plate_gap"])
        return lower.union(upper)

    def _arm_z(self, p):
        return p["plate_thickness"] + p["plate_gap"] / 2

    def _arm(self, p, angle=0.0):
        R, ra = p["wheelbase"] / 2, p["arm_diameter"] / 2
        r_in = p["plate_diameter"] / 2 * 0.55          # tube runs into the plate sandwich for the clamps
        z = self._arm_z(p)
        tube = (cq.Workplane("YZ").center(0, z).circle(ra).circle(ra - p["arm_wall"])
                .extrude(R - r_in).translate((r_in, 0, 0)))
        mount = (cq.Workplane("XY").center(R, 0).circle(p["mount_diameter"] / 2).extrude(p["mount_thickness"])
                 .translate((0, 0, z + ra)))
        return tube.union(mount).rotate((0, 0, 0), (0, 0, 1), angle)

    def _gear(self, p):
        h, sy, L = p["leg_height"], p["leg_spread"] / 2, p["skid_length"]
        r = p["leg_diameter"] / 2
        gear = None
        for s in (1, -1):
            skid = cq.Workplane("YZ").center(s * sy, -h + r).circle(r).extrude(L / 2, both=True)
            gear = skid if gear is None else gear.union(skid)
            for sx in (0.35 * L, -0.35 * L):
                a = cq.Vector(sx * 0.45, s * p["plate_diameter"] * 0.3, 0.0)
                b = cq.Vector(sx, s * sy, -h + r)
                gear = gear.union(cq.Workplane("XY").add(cq.Solid.makeCylinder(r, (b - a).Length, a, b - a)))
        return gear

    def _cradle_z(self, p):
        """Cradle frame top: just above the package top when the package stands on the ground under a landed Drako."""
        return -p["leg_height"] + p["package_z"] + p["cradle_bar"]

    def _cradle(self, p):
        X, Y, b = p["cradle_x"], p["cradle_y"], p["cradle_bar"]
        z = self._cradle_z(p)
        outer = cq.Workplane("XY").rect(X + 2 * b, Y + 2 * b).extrude(-b)
        frame = outer.cut(cq.Workplane("XY").rect(X, Y).extrude(-b)).translate((0, 0, z))
        # hangers from the lower plate to the frame
        for sx in (1, -1):
            for sy in (1, -1):
                post = cq.Workplane("XY").box(b, b, -z).translate((sx * 0.3 * X, sy * (Y / 2 + b / 2), z / 2))
                frame = frame.union(post)
        # four hook latches on the long sides, hanging inside the frame to catch the tote lip
        for sx in (0.25, -0.25):
            for sy in (1, -1):
                hook = (cq.Workplane("XY").box(30, 6, 40)
                        .translate((sx * X, sy * (Y / 2 - 3), z - b - 20)))
                tip = cq.Workplane("XY").box(30, 12, 6).translate((sx * X, sy * (Y / 2 - 9), z - b - 37))
                servo = cq.Workplane("XY").box(40, 20, 38).translate((sx * X, sy * (Y / 2 + b + 10), z - b / 2))
                frame = frame.union(hook).union(tip).union(servo)
        return frame

    def _bought(self, p):
        R, z = p["wheelbase"] / 2, self._arm_z(p) + p["arm_diameter"] / 2 + p["mount_thickness"]
        rp = p["prop_diameter"] * 25.4 / 2
        top = 2 * p["plate_thickness"] + p["plate_gap"]
        out = cq.Workplane("XY").box(300, 160, 110).translate((0, 0, top + 55))          # 12S 64 Ah
        for a in self._arm_angles(p):
            cx, cy = R * math.cos(math.radians(a)), R * math.sin(math.radians(a))
            out = out.union(cq.Workplane("XY").center(cx, cy).circle(50).extrude(45).translate((0, 0, z)))
            out = out.union(cq.Workplane("XY").center(cx, cy).circle(rp).extrude(2.0).translate((0, 0, z + 55)))
        pkg = cq.Workplane("XY").box(p["package_x"], p["package_y"], p["package_z"]) \
            .translate((0, 0, -p["leg_height"] + p["package_z"] / 2))
        return out.union(pkg)

    def check(self, p):
        R, rp = p["wheelbase"] / 2, p["prop_diameter"] * 25.4 / 2
        tip_gap = 2 * R * math.sin(math.pi / p["arm_count"]) - 2 * rp
        if tip_gap < 20.0:
            raise ValueError(f"neighbouring propeller tips are {tip_gap:.0f} mm apart (< 20 mm): widen wheelbase")
        if p["cradle_x"] < p["package_x"] + 10 or p["cradle_y"] < p["package_y"] + 10:
            raise ValueError("the package does not fit inside the cradle with 5 mm a side")
        if p["leg_spread"] / 2 - p["leg_diameter"] < p["cradle_y"] / 2 + p["cradle_bar"] + 40:
            raise ValueError("the skids are inside the cradle and its servos: widen leg_spread")
        if self._cradle_z(p) > -20.0:
            raise ValueError("the package reaches the plates: raise leg_height or use a lower package")

    def build(self, p):
        self.check(p)
        part = p["part"]
        if part == "plate":
            return self._hexagon(p, 0.0)
        if part == "arm":
            return self._arm(p)
        if part == "gear":
            return self._gear(p)
        if part == "cradle":
            return self._cradle(p)
        frame = self._plates(p)
        for a in self._arm_angles(p):
            frame = frame.union(self._arm(p, a))
        if part == "frame":
            return frame
        body = frame.union(self._gear(p)).union(self._cradle(p))
        if p["show_bought"]:
            body = body.union(self._bought(p))
        return body


if __name__ == "__main__":
    defaults = {q.name: q.default for q in Drako.parameters}
    for k, v in Drako.sizing(defaults).items():
        print(f"{k:34s} {v}")
