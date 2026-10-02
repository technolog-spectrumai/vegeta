from dataclasses import replace

import cadquery as cq
from vegeta.dedalus import Parameter

from onager import OnagerSentinel


#: The Sentinel parameters the Atlas changes: no turret (a logistics unit; the sensor mast stays), and the
#: logistics stance — upper legs 25° and lower legs 15° from vertical (Sentinel 40° / 55°), so the legs carry the
#: load through the structure: with a 200 kg pallet the front knee torque drops from ~1.6 kN·m (over the 800 N·m
#: modules) to ~0.5 kN·m (notebook 21 §3); the hull rides higher, keeping the shoulders where they were on its sides.
ATLAS_CHASSIS = {"turret_length": (0.0, "no turret on the Atlas"),
                 "turret_height": (0.0, "no turret: the mast stands on the roof"),
                 "hip_angle_deg": (25.0, "upper leg behind vertical: the logistics stance (Sentinel 40°)"),
                 "knee_angle_deg": (15.0, "lower leg ahead of vertical: the logistics stance (Sentinel 55°)"),
                 "hull_bottom": (1045.0, "ground clearance of the hull floor in the logistics stance")}


def _sentinel_params():
    out = []
    for p in OnagerSentinel.parameters:
        if p.name == "part":
            continue
        if p.name in ATLAS_CHASSIS:
            v, desc = ATLAS_CHASSIS[p.name]
            p = replace(p, default=v, description=desc)
        out.append(p)
    return out


class OnagerAtlas(OnagerSentinel):
    """Onager Atlas: the Sentinel's wheel-leg chassis carrying a forklift — a mast of two steel uprights ahead of the
    front wheels on a tilt pivot under the hull's nose, a lift carriage running on the uprights and two forks. It
    picks up a pallet, carries it and sets it down. x forward, y left, z up, ground at z = 0; ``lift`` raises the
    carriage (0: the forks' undersides ``fork_ground`` above the ground), ``tilt_deg`` tilts the mast back.

    ``part`` adds to the Sentinel's (each in its own frame): ``fork`` (one fork: the heel corner at the origin, the
    tine along +x, the shank up +z, its width along y), ``mast`` (the two uprights with their crossbars: the tilt
    pivot at the origin, uprights up +z from ``mast_bottom`` to ``mast_top``)."""

    parameters = [Parameter("part", "robot", choices=("robot", "hull", "upper_leg", "lower_leg", "wheel", "fork", "mast"),
                            description="what to build")] + _sentinel_params() + [
        Parameter("pivot_x", 1200.0, "mm", min=0, description="mast tilt pivot ahead of the hull centre"),
        Parameter("pivot_z", 1065.0, "mm", min=0, description="mast tilt pivot above the ground (standing)"),
        Parameter("upright_y", 320.0, "mm", min=50, description="uprights either side of the centre line"),
        Parameter("upright_width", 60.0, "mm", min=10, description="rectangular tube, across (y)"),
        Parameter("upright_depth", 100.0, "mm", min=10, description="rectangular tube, fore-aft (x)"),
        Parameter("upright_wall", 6.0, "mm", min=1, description="S355 tube wall"),
        Parameter("mast_bottom", -1005.0, "mm", description="uprights' lower end relative to the pivot"),
        Parameter("mast_top", 1020.0, "mm", description="uprights' upper end relative to the pivot"),
        Parameter("carriage_offset", 80.0, "mm", min=0, description="carriage plate ahead of the uprights' axis"),
        Parameter("carriage_width", 760.0, "mm", min=100),
        Parameter("carriage_height", 450.0, "mm", min=50),
        Parameter("carriage_thickness", 30.0, "mm", min=5),
        Parameter("fork_y", 175.0, "mm", min=20, description="tines either side of the centre line (Euro-pallet openings)"),
        Parameter("fork_length", 1000.0, "mm", min=200, description="tine length from the heel"),
        Parameter("fork_width", 80.0, "mm", min=20),
        Parameter("fork_thickness", 30.0, "mm", min=5, description="tine and shank thickness (S690 high-strength steel)"),
        Parameter("fork_shank", 450.0, "mm", min=50, description="shank height above the tine"),
        Parameter("fork_ground", 40.0, "mm", min=0, description="forks' underside above the ground at lift 0"),
        Parameter("lift_max", 1200.0, "mm", min=100, description="carriage travel"),
        Parameter("lift", 150.0, "mm", min=0, description="carriage height in the built pose"),
        Parameter("tilt_deg", 0.0, "deg", min=-10, max=15, description="mast tilt back in the built pose"),
    ]

    # ---- derived geometry shared by the notebook and the Chiron robot
    @staticmethod
    def heel_x(p):
        """Fork heel (the tines' root) ahead of the hull centre [mm], mast vertical."""
        return p["pivot_x"] + p["upright_depth"] / 2 + p["carriage_offset"] + p["carriage_thickness"] / 2 + p["fork_thickness"]

    @staticmethod
    def load_centre_x(p, load_centre=500.0):
        """The rated load centre (``load_centre`` mm from the heel) ahead of the hull centre [mm]."""
        return OnagerAtlas.heel_x(p) + load_centre

    # ---- parts
    def _fork(self, p):
        L, w, t, h = p["fork_length"], p["fork_width"], p["fork_thickness"], p["fork_shank"]
        tine = cq.Workplane("XY").box(L, w, t).translate((L / 2, 0, t / 2))
        shank = cq.Workplane("XY").box(t, w, h + t).translate((-t / 2, 0, (h + t) / 2))
        heel = cq.Workplane("XZ").center(-t / 2, t / 2).rect(t, t).extrude(w / 2, both=True)   # fills the corner
        fork = tine.union(shank).union(heel)
        # taper the tine tip over its last 150 mm to 40 % thickness (entry into the pallet)
        cut = (cq.Workplane("XZ").polyline([(L - 150, t), (L + 1, t), (L + 1, 0.4 * t)]).close().extrude(w, both=True))
        return fork.cut(cut)

    def _mast(self, p):
        """Two rectangular-tube uprights and two crossbars, the pivot at the origin."""
        w, d, wall = p["upright_width"], p["upright_depth"], p["upright_wall"]
        z0, z1 = p["mast_bottom"], p["mast_top"]
        H = z1 - z0
        mast = None
        for sy in (-1, 1):
            tube = cq.Workplane("XY").box(d, w, H).translate((0, sy * p["upright_y"], z0 + H / 2))
            tube = tube.cut(cq.Workplane("XY").box(d - 2 * wall, w - 2 * wall, H + 2).translate((0, sy * p["upright_y"], z0 + H / 2)))
            mast = tube if mast is None else mast.union(tube)
        span = 2 * p["upright_y"] + w
        for z in (z0 + 40.0, z1 - 40.0):
            mast = mast.union(cq.Workplane("XY").box(d * 0.8, span, 20.0).translate((0, 0, z)))
        # the pivot: a cross tube (Ø70 × 8) at the pivot height
        mast = mast.union(cq.Workplane("XZ").circle(35.0).circle(27.0).extrude(span / 2, both=True))
        return mast

    def _forklift(self, p):
        """Mast, carriage and forks in the world, in the built pose (lift, tilt), plus the frame to the hull."""
        tilt = p["tilt_deg"]
        piv = (p["pivot_x"], 0.0, p["pivot_z"])
        mast = self._mast(p)
        # carriage and forks in the mast frame: the forks' underside fork_ground above the ground at lift 0
        z_fork = p["fork_ground"] - p["pivot_z"] + p["lift"]
        xc = p["upright_depth"] / 2 + p["carriage_offset"]
        carriage = (cq.Workplane("XY").box(p["carriage_thickness"], p["carriage_width"], p["carriage_height"])
                    .translate((xc, 0, z_fork + p["carriage_height"] / 2)))
        x_heel = xc + p["carriage_thickness"] / 2 + p["fork_thickness"]
        for sy in (-1, 1):
            carriage = carriage.union(self._fork(p).translate((x_heel, sy * p["fork_y"], z_fork)))
        # the two arms that carry the pivot from the hull's nose
        hull_front = p["hull_length"] / 2
        arms = None
        for sy in (-1, 1):
            a = (cq.Workplane("XY").box(p["pivot_x"] - hull_front + 40.0, 50.0, 80.0)
                 .translate(((hull_front + p["pivot_x"]) / 2 - 20.0, sy * (p["upright_y"] - 70.0), p["pivot_z"])))
            arms = a if arms is None else arms.union(a)
        moving = mast.union(carriage).rotate((0, 0, 0), (0, 1, 0), -tilt).translate(piv)
        return moving.union(arms)

    def build(self, p):
        part = p["part"]
        if part == "fork":
            return self._fork(p)
        if part == "mast":
            return self._mast(p)
        if p["lift"] > p["lift_max"]:
            raise ValueError("lift exceeds lift_max")
        body = super().build(dict(p, part=part))
        if part == "robot":
            body = body.union(self._forklift(p))
        return body
