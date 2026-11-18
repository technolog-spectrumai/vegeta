"""The Pekari's golf-ball launcher: CAD (Dedalus / CadQuery).

A tennis-ball machine's layout, shrunk to golf balls and set on a pan-tilt mount on the Pekari's payload basket:

    basket (hopper, ~100 balls) -> agitator -> feed tube up the hollow pan axis -> breech
      -> two flywheels one above the other (top/bottom: their speed difference is the backspin)
      -> a vented smooth tube -> muzzle

Frame: x along the tube (forward at pan 0, tilt 0), y left, z up, the origin on the pan axis at the mount's base
(the basket's lid). The tilt axis is along y at ``pivot_height``. Lengths in mm.

``part``: ``launcher`` (the mount and everything on it, at ``tilt_deg``), ``tube``, ``flywheel``, ``ball`` (a smooth
sphere), ``ball_dimpled`` (a golf ball with ``dimples`` spherical dimples on a Fibonacci lattice), for the mass budget,
the FEA and the CFD. The numbers the hand calculation uses (``pekari_golf_aero``) are the defaults here.
"""
from __future__ import annotations

import math

try:
    import cadquery as cq
except ImportError:          # the layout and masses need no CAD kernel
    cq = None
from vegeta.dedalus import Design, Parameter

import pekari_golf_aero as aero

AL = 2.70e-3      # g/mm3
STEEL = 7.85e-3
PU = 1.20e-3
PETG = 1.27e-3


class PekariGolf(Design):
    """The golf-ball launcher on the Pekari (see the module)."""

    parameters = [
        Parameter("part", "launcher", choices=("launcher", "tube", "flywheel", "ball", "ball_dimpled"), description="what to build"),
        Parameter("tilt_deg", 10.0, "deg", min=-10, max=55, description="the tube's elevation"),
        Parameter("ball_diameter", aero.D_BALL * 1e3, "mm", min=30),
        Parameter("dimples", 336, "", min=0, max=500, description="dimples on ball_dimpled"),
        Parameter("dimple_diameter", 3.8, "mm", min=0.5),
        Parameter("dimple_depth", 0.25, "mm", min=0.02),
        Parameter("bore", aero.Tube().bore * 1e3, "mm", min=31, description="tube bore: ball + clearance"),
        Parameter("tube_wall", 2.0, "mm", min=0.5),
        Parameter("tube_length", aero.Tube().length * 1e3, "mm", min=50, description="nip to muzzle"),
        Parameter("vent_slots", 6, "", min=0, max=20, description="slots per side along the first 2/3 (the air ahead escapes)"),
        Parameter("wheel_diameter", aero.Flywheels().diameter * 1e3, "mm", min=40),
        Parameter("wheel_width", aero.Flywheels().width * 1e3, "mm", min=10),
        Parameter("tyre", 6.0, "mm", min=1, description="polyurethane tyre thickness"),
        Parameter("squeeze", aero.Flywheels().squeeze * 1e3, "mm", min=0, description="ball diameter minus the nip gap"),
        Parameter("motor_diameter", 42.0, "mm", min=10, description="outrunner BLDC can (a 4240-class, ~400 W)"),
        Parameter("motor_length", 40.0, "mm", min=10),
        Parameter("pivot_height", 140.0, "mm", min=60, description="tilt axis above the mount base (Armageddon's PIVOT_UP)"),
        Parameter("nip_behind_pivot", 70.0, "mm", min=0, description="the wheels' nip behind the tilt axis: the housing balances the tube"),
        Parameter("turntable_diameter", 130.0, "mm", min=50),
        Parameter("feed_bore", 46.0, "mm", min=44, description="feed tube up the hollow pan axis"),
    ]

    # --- the layout (no CAD kernel) ------------------------------------------------------------

    @staticmethod
    def layout(p):
        p = dict(p)
        rw = p["wheel_diameter"] / 2
        gap = p["ball_diameter"] - p["squeeze"]
        zc = gap / 2 + rw                                    # wheel axes above/below the tube axis
        x_nip = -p["nip_behind_pivot"]
        x_muzzle = x_nip + p["tube_length"]
        return {"wheel_r": rw, "gap": gap, "wheel_z": zc, "x_nip": x_nip, "x_muzzle": x_muzzle,
                "tube_od": p["bore"] + 2 * p["tube_wall"], "pivot_z": p["pivot_height"],
                "muzzle_from_pivot": x_muzzle}

    def masses(self, **kw):
        """A mass budget of the parts on the tilt axis (g) and their moment about it (N m, at tilt 0), by hand."""
        p = self.resolve(**kw)
        L = self.layout(p)
        rw, w = L["wheel_r"], p["wheel_width"]
        tube_len = p["tube_length"]
        tube = math.pi / 4 * (L["tube_od"] ** 2 - p["bore"] ** 2) * tube_len * AL * 0.85       # minus the slots
        hub = math.pi * (rw - p["tyre"]) ** 2 * w * AL * 0.55                                  # lightened disc
        tyre = math.pi * (rw ** 2 - (rw - p["tyre"]) ** 2) * w * PU
        motor = 2 * 190.0                                                                       # g, 4240-class
        housing = 2 * 3.0 * 160 * 150 * AL + 120.0                                              # side plates + bolts
        parts = {"tube": (tube, L["x_nip"] + tube_len / 2), "flywheels": (2 * (hub + tyre), L["x_nip"]),
                 "motors": (motor, L["x_nip"]), "housing": (housing, L["x_nip"] + 10), "breech": (60.0, L["x_nip"] - 40)}
        total = sum(m for m, _ in parts.values())
        moment = sum(m * x for m, x in parts.values()) * 1e-3 * 9.81 * 1e-3
        return {"parts_g": {k: round(m, 1) for k, (m, _) in parts.items()}, "total_g": total, "tilt_moment_Nm": moment,
                "wheel_g": hub + tyre, "wheel_inertia_kgm2": (hub * (rw - p["tyre"]) ** 2 / 2 + tyre * (rw - p["tyre"] / 2) ** 2) * 1e-9}

    # --- the geometry ------------------------------------------------------------------------

    def build(self, p):
        part = p["part"]
        if part == "ball":
            return cq.Workplane("XY").sphere(p["ball_diameter"] / 2)
        if part == "ball_dimpled":
            return self._dimpled(p)
        if part == "tube":
            return self._tube(p)
        if part == "flywheel":
            return self._flywheel(p)
        return self._launcher(p)

    def _dimpled(self, p):
        R = p["ball_diameter"] / 2
        ball = cq.Workplane("XY").sphere(R)
        n = p["dimples"]
        if n == 0:
            return ball
        a, h = p["dimple_diameter"] / 2, p["dimple_depth"]
        rc = (a * a + h * h) / (2 * h)                     # the cutter sphere through the dimple's rim at depth h
        d = R - h + rc                                      # its centre from the ball's
        golden = math.pi * (3 - math.sqrt(5))
        cutters = None
        for i in range(n):
            z = 1 - 2 * (i + 0.5) / n
            r = math.sqrt(1 - z * z)
            t = golden * i
            c = (d * r * math.cos(t), d * r * math.sin(t), d * z)
            s = cq.Solid.makeSphere(rc, cq.Vector(*c))
            cutters = s if cutters is None else cutters.fuse(s)
        return ball.cut(cq.Workplane("XY").add(cutters))

    def _tube(self, p):
        L = self.layout(p)
        od, bore, length = L["tube_od"], p["bore"], p["tube_length"]
        t = cq.Workplane("YZ").circle(od / 2).circle(bore / 2).extrude(length)
        flange = cq.Workplane("YZ").circle(od / 2 + 6).circle(bore / 2).extrude(5)
        t = t.union(flange)
        n = p["vent_slots"]
        if n:
            span = length * 2 / 3 - 20
            pitch = span / n
            for side in (1, -1):
                for k in range(n):
                    slot = (cq.Workplane("XY").box(pitch * 0.6, od, 4).translate((25 + pitch * (k + 0.5), 0, 0))
                            .rotate((0, 0, 0), (1, 0, 0), 45 * side))
                    t = t.cut(slot)
        return t

    def _flywheel(self, p):
        rw, w, ty = p["wheel_diameter"] / 2, p["wheel_width"], p["tyre"]
        hub = cq.Workplane("XZ").circle(rw - ty).extrude(w / 2, both=True)
        for k in range(5):                                  # lightening holes
            a = 2 * math.pi * k / 5
            rr = (rw - ty) * 0.55
            hub = hub.cut(cq.Workplane("XZ").center(rr * math.cos(a), rr * math.sin(a)).circle((rw - ty) * 0.18).extrude(w, both=True))
        hub = hub.cut(cq.Workplane("XZ").circle(4).extrude(w, both=True))      # 8 mm shaft
        tyre = cq.Workplane("XZ").circle(rw).circle(rw - ty).extrude(w / 2, both=True)
        groove = cq.Workplane("XZ").circle(rw + 1).extrude(w / 2, both=True).cut(
            cq.Solid.makeTorus(rw + p["ball_diameter"] / 2 - 1.0, p["ball_diameter"] / 2, pnt=cq.Vector(0, 0, 0), dir=cq.Vector(0, 1, 0)))
        return hub.union(tyre.intersect(groove))

    def _launcher(self, p):
        L = self.layout(p)
        pz = L["pivot_z"]
        # pan stage: a turntable ring with the hollow axis (the feed comes up through it) and a yoke
        tt = cq.Workplane("XY").circle(p["turntable_diameter"] / 2).circle(p["feed_bore"] / 2 + 3).extrude(18)
        yoke_w = p["wheel_width"] + 2 * p["motor_length"] + 30
        plate = cq.Workplane("XY").box(70, 6, pz + 25).translate((0, 0, (pz + 25) / 2 + 9))
        yoke = plate.translate((0, yoke_w / 2 + 3, 0)).union(plate.translate((0, -yoke_w / 2 - 3, 0)))
        bearing = cq.Workplane("XZ").circle(14).extrude(yoke_w / 2 + 6, both=True).translate((0, 0, pz))
        mount = tt.union(yoke).union(bearing.cut(cq.Workplane("XZ").circle(14).extrude(yoke_w / 2 - 6, both=True).translate((0, 0, pz))))
        # the tilting cradle, built at tilt 0 about the tilt axis (0, 0, pz), then rotated
        tube = self._tube(p).translate((L["x_nip"], 0, 0))
        fw = self._flywheel(p)
        wheels = fw.translate((L["x_nip"], 0, L["wheel_z"])).union(fw.translate((L["x_nip"], 0, -L["wheel_z"])))
        motor = cq.Workplane("XZ").circle(p["motor_diameter"] / 2).extrude(p["motor_length"])
        motors = None
        for zc, side in ((L["wheel_z"], 1), (-L["wheel_z"], -1)):          # the motors on opposite sides
            y = p["wheel_width"] / 2 + 4 + p["motor_length"] if side > 0 else -(p["wheel_width"] / 2 + 4)   # XZ extrudes to -y
            m = motor.translate((L["x_nip"], y, zc))
            motors = m if motors is None else motors.union(m)
        hh = 2 * L["wheel_z"] + p["wheel_diameter"] + 16
        side_plate = cq.Workplane("XY").box(p["wheel_diameter"] + 50, 3, hh).translate((L["x_nip"] + 10, 0, 0))
        housing = side_plate.translate((0, p["wheel_width"] / 2 + 2, 0)).union(side_plate.translate((0, -p["wheel_width"] / 2 - 2, 0)))
        trunnion = cq.Workplane("XZ").circle(10).extrude(yoke_w / 2 + 4, both=True)
        breech = cq.Workplane("YZ").circle(L["tube_od"] / 2).circle(p["bore"] / 2).extrude(45).translate((L["x_nip"] - p["wheel_diameter"] / 2 - 45, 0, 0))
        cradle = tube.union(wheels).union(motors).union(housing).union(trunnion).union(breech)
        cradle = cradle.translate((0, 0, pz)).rotate((0, 0, pz), (0, 1, 0), -p["tilt_deg"])
        # the feed hose from the hollow pan axis to the breech (a straight stand-in for the flexible hose)
        bx = L["x_nip"] - p["wheel_diameter"] / 2 - 45
        t = math.radians(p["tilt_deg"])
        end = cq.Vector(bx * math.cos(t), 0, pz + bx * math.sin(t))      # the breech, tilted with the cradle
        start = cq.Vector(0, 0, 18)
        axis = end - start
        feed = cq.Solid.makeCylinder(p["feed_bore"] / 2 + 2, axis.Length, start, axis.normalized())
        feed = cq.Workplane("XY").add(feed).cut(cq.Workplane("XY").add(cq.Solid.makeCylinder(p["feed_bore"] / 2, axis.Length, start, axis.normalized())))
        return mount.union(cradle).union(feed)


def export(outdir="out/pekari_golf", **kw):
    """STEP + STL of the launcher, the tube, a flywheel and both balls."""
    from pathlib import Path
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for part in ("launcher", "tube", "flywheel", "ball", "ball_dimpled"):
        g = PekariGolf().generate(part=part, **kw)
        g.export_step(str(out / f"{part}.step"))
        g.export_stl(str(out / f"{part}.stl"), tolerance=0.05 if part.startswith("ball") else 0.2)
        paths[part] = g.measure()
    return paths
