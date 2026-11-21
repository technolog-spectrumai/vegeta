"""Kerchak's quad, autonomous: the same quad (``gorilla_quad``) drives itself, and a human or Kerchak can drive it.

**Hardware added** (``AutonomyKit``, CadQuery; on the sprung frame unless stated):

* two 3D lidars - one on the nose (forward 270°) and one on a mast at the trunk's back (rear 270°), above the rider's
  knees and the trunk's tools; three layers each in the sim (0°, -4°, -9°), 20 m;
* a stereo camera bar under the handlebar pod (for the logs; the sim plans on the lidars);
* two GNSS antennas on the trunk rails (RTK heading), an IMU in the VCU;
* drive-by-wire: the four wheel motors were electric already; the steering column gets a motor and a torque sensor
  (``STEER_BY_WIRE``: it steers in AUTO, assists the rider in MANUAL, and feels a rider's hands turning the bar - the
  takeover); twist-grip throttle and brake levers read as signals; regenerative and by-wire friction brakes;
* the seat's load cell (who rides: ``gorilla_quad_riders.classify_seat_load``), the telescopic column (the bar's
  place for the rider), air springs at the corners and the seat (pressurised for the load);
* three mushroom e-stops (both rear trunk corners, the bar's centre) and a mode beacon on the rear mast (green AUTO,
  blue MANUAL, red STOP).

**Software** (pure Python here; the sim feeds it sensors and applies its commands):

* ``Perception`` - lidar hits to obstacle points: the ground and anything lower than ``OBSTACLE_MIN_H`` (a bump the
  suspension takes) dropped;
* ``Planner`` - follows the route; an obstacle in its corridor ahead gets a detour (the smallest side offset that
  clears it by ``SAFETY['clearance']``, blended in and out), and the speed is capped by the
  free distance ahead (stopping in it at ``a_plan``), by the curvature (the rider profile's lateral g) and by the
  route; nothing clears -> it stops short; the swerve's length scales with the speed, so a stopped quad can still
  crawl round;
* ``Guardian`` - always on, whoever drives: the arc the quad is on now is swept ahead; an obstacle inside the
  stopping distance (``a_aeb``) plus a margin brakes the quad (automatic emergency braking); the rider keeps the bar;
* ``Arbiter`` - the mode: nobody aboard -> AUTO only; a rider can engage AUTO (the button) and takes back control by
  turning the bar against it (torque over ``STEER_BY_WIRE['takeover_Nm']``) or twisting the throttle; an e-stop
  latches STOP.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cadquery as cq
import numpy as np

import gorilla_quad as gq
from vegeta.dedalus import Design, Parameter

__all__ = ["LIDARS", "STEER_BY_WIRE", "SAFETY", "Perception", "Planner", "Guardian", "Arbiter", "AutonomyKit", "kit_mesh",
           "lidar_rays", "OBSTACLE_MIN_H"]

#: the lidars: mount on the frame [m, vehicle frame], heading [rad], field of view [rad], layers' pitch [rad], range
LIDARS = {
    "front": {"pos": (1.02, 0.0, 0.80), "yaw": 0.0, "fov": math.radians(270.0), "layers": (0.0, -math.radians(4.0), -math.radians(9.0)),
              "beams": 91, "range": 20.0, "rate_hz": 10.0},
    "rear": {"pos": (-1.12, 0.0, 1.30), "yaw": math.pi, "fov": math.radians(270.0), "layers": (0.0, -math.radians(4.0), -math.radians(9.0)),
             "beams": 91, "range": 20.0, "rate_hz": 10.0},
}
OBSTACLE_MIN_H = 0.16          #: lower than this is a bump the suspension takes (the course's are 0.10-0.12 m) [m]
#: the steer-by-wire column: its motor's torque, the rider's torque that takes over, how fast it may turn
STEER_BY_WIRE = {"motor_Nm": 120.0, "takeover_Nm": 15.0, "takeover_s": 0.2, "rate_rad_s": 1.2}
#: the margins: the quad's half width with its fenders, the clearance kept, the planning and emergency decelerations
SAFETY = {"half_width": 1.06, "clearance": 0.45, "a_plan": 1.5, "a_aeb": 4.5, "stop_gap": 1.2, "horizon": 22.0,
          "max_offset": 3.0, "react_s": 0.15}


# ---------------------------------------------------------------- sensing
def lidar_rays(name):
    """(unit directions in the vehicle frame, n) of a lidar's beams, all layers."""
    L = LIDARS[name]
    az = L["yaw"] + np.linspace(-L["fov"] / 2, L["fov"] / 2, L["beams"])
    dirs = []
    for el in L["layers"]:
        dirs.append(np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.full_like(az, np.sin(el))], axis=1))
    return np.concatenate(dirs)


class Perception:
    """Lidar hits (world points) -> obstacle points: the ground and the low bumps dropped."""

    def __init__(self, ground_z=0.0):
        self.ground_z = ground_z

    def obstacles(self, hits):
        hits = np.asarray(hits, dtype=float).reshape(-1, 3)
        return hits[hits[:, 2] > self.ground_z + OBSTACLE_MIN_H]


# ---------------------------------------------------------------- planning
@dataclass
class Plan:
    steer_path: callable          #: y(x) the vehicle should follow (the route with the detour)
    v_max: float                  #: the speed cap [m/s]
    status: str                   #: "route", "detour left/right", "stopping: blocked"
    offset: float = 0.0
    free_m: float = math.inf


@dataclass
class Planner:
    """The route ``route_y(x)`` (a lane along +x) with detours round obstacles and a speed cap. The obstacle points
    are kept in a short-lived map (a 0.1 m grid, ``memory_s``): one scan sees a far box as a point or two, the map
    gathers its face as the quad closes in, and the detour is checked again each scan and widened if it no longer
    clears."""

    route_y: callable
    profile: dict
    detour: dict | None = None                      #: {"x0": where it begins, "x1": where it ends, "offset": m, "ramp": m}
    memory_s: float = 2.0
    cells: dict = field(default_factory=dict)

    def path_y(self, x):
        y = self.route_y(x)
        d = self.detour
        if d is None:
            return y
        r = d["ramp"]
        if x <= d["x0"] - r or x >= d["x1"] + r:
            return y
        u = 1.0 if d["x0"] <= x <= d["x1"] else (x - (d["x0"] - r)) / r if x < d["x0"] else 1.0 - (x - d["x1"]) / r
        u = 0.5 - 0.5 * math.cos(math.pi * min(max(u, 0.0), 1.0))
        return y + d["offset"] * u

    def remember(self, t, pts):
        for p in np.asarray(pts, dtype=float).reshape(-1, 3):
            self.cells[(int(math.floor(p[0] / 0.1)), int(math.floor(p[1] / 0.1)))] = (t, p)
        self.cells = {k: v for k, v in self.cells.items() if t - v[0] <= self.memory_s}
        return np.array([v[1] for v in self.cells.values()]) if self.cells else np.zeros((0, 3))

    def _blocking(self, pts, x_now, offset=0.0, x0=None, x1=None, half=None):
        """The obstacle points ahead inside the corridor of the route, shifted ``offset`` between ``x0`` and ``x1``."""
        if len(pts) == 0:
            return pts
        half = SAFETY["half_width"] + SAFETY["clearance"] if half is None else half
        ahead = pts[(pts[:, 0] > x_now) & (pts[:, 0] < x_now + SAFETY["horizon"])]
        if len(ahead) == 0:
            return ahead
        yc = np.array([self.route_y(x) for x in ahead[:, 0]])
        if offset:
            yc = yc + np.where((ahead[:, 0] >= x0) & (ahead[:, 0] <= x1), offset, 0.0)
        return ahead[np.abs(ahead[:, 1] - yc) < half]

    def update(self, x, v, pts, t=0.0):
        pts = self.remember(t, pts)
        if self.detour is not None and x > self.detour["x1"] + self.detour["ramp"]:
            self.detour = None
        status = "route"
        block = self._blocking(pts, x)
        if len(block):
            xo0, xo1 = float(block[:, 0].min()), float(block[:, 0].max()) + 0.6       # its far side is hidden: assume 0.6 m deep
            d = self.detour
            stale = d is None or xo0 < d["x0"] or xo1 > d["x1"] or len(self._blocking(pts, x, d["offset"], d["x0"], d["x1"]))
            if stale:
                ramp = max(3.0, 1.6 * abs(v) + 2.0)
                best = None
                for off in sorted(np.arange(-SAFETY["max_offset"], SAFETY["max_offset"] + 1e-9, 0.25), key=lambda o: (abs(o), o)):
                    if off == 0.0:
                        continue
                    if not len(self._blocking(pts, x, off, xo0 - 1.0, xo1 + 1.0)):
                        best = off
                        break
                if best is not None and xo0 - x > 0.6 * ramp:
                    if d is not None and np.sign(d["offset"]) == np.sign(best):
                        best = best if abs(best) >= abs(d["offset"]) else d["offset"]   # keep the side, never narrow it
                    self.detour = {"x0": xo0 - 1.0, "x1": xo1 + 1.0, "offset": float(best), "ramp": ramp}
        if self.detour is not None:
            off = self.detour["offset"]
            status = f"detour {'left' if off > 0 else 'right'} {abs(off):.2f} m"
        # the free distance along the planned path
        free = math.inf
        ahead = pts[(pts[:, 0] > x) & (pts[:, 0] < x + SAFETY["horizon"])] if len(pts) else pts
        if len(ahead):
            yp = np.array([self.path_y(px) for px in ahead[:, 0]])
            hit = ahead[np.abs(ahead[:, 1] - yp) < SAFETY["half_width"] + 0.15]
            if len(hit):
                free = float(hit[:, 0].min() - x) - 0.95       # the nose is 0.95 m ahead of the middle
        v_cap = self.profile["v_max"]
        if free < math.inf:
            v_cap = min(v_cap, math.sqrt(max(0.0, 2 * SAFETY["a_plan"] * (free - SAFETY["stop_gap"]))))
            v_cap = max(v_cap, 0.6) if free - SAFETY["stop_gap"] > 0.3 else v_cap      # creep up to the stop line
            if free - SAFETY["stop_gap"] < 0.3:
                status = "stopping: blocked"
        xs = x + np.array([2.0, 4.0, 6.0])
        ys = np.array([self.path_y(v_) for v_ in xs])
        k = abs((ys[2] - 2 * ys[1] + ys[0]) / 4.0)
        if k > 1e-4:
            v_cap = min(v_cap, math.sqrt(self.profile["lat_g"] * 9.81 / k))
        return Plan(self.path_y, v_cap, status, self.detour["offset"] if self.detour else 0.0, free)


def pure_pursuit(x, y, yaw, v, path_y, wheelbase):
    """The steering angle that puts the quad on ``path_y``."""
    ld = max(2.0, 0.4 * abs(v) + 1.2)
    xa = x + ld * math.cos(yaw)
    alpha = math.atan2(path_y(xa) - y, xa - x) - yaw
    return math.atan2(2 * wheelbase * math.sin(alpha), ld)


# ---------------------------------------------------------------- safety and modes
class Guardian:
    """Automatic emergency braking, whoever drives: the arc the quad is on is swept ahead; an obstacle point inside
    the stopping distance (plus reaction and a gap) brakes the quad."""

    def __init__(self):
        self.braking = False

    def check(self, x, y, yaw, v, steer, pts, wheelbase):
        pts = np.asarray(pts, dtype=float).reshape(-1, 3)
        if len(pts) == 0 or v < 0.2:
            self.braking = self.braking and v > 0.05
            return self.braking
        d_stop = v * SAFETY["react_s"] + v * v / (2 * SAFETY["a_aeb"]) + 0.6
        kappa = math.tan(steer) / wheelbase
        c, s = math.cos(yaw), math.sin(yaw)
        local = np.stack([(pts[:, 0] - x) * c + (pts[:, 1] - y) * s, -(pts[:, 0] - x) * s + (pts[:, 1] - y) * c], axis=1)
        front = local[(local[:, 0] > 0.0) & (local[:, 0] < 0.95 + d_stop)]
        if len(front):
            arc_y = 0.5 * kappa * front[:, 0] ** 2                         # the arc's lateral offset at that distance
            hit = np.abs(front[:, 1] - arc_y) < SAFETY["half_width"] + 0.1
            self.braking = bool(hit.any()) or (self.braking and v > 0.05)
        else:
            self.braking = self.braking and v > 0.05
        return self.braking


class Arbiter:
    """The driving mode: ``AUTO`` (the planner steers and drives), ``MANUAL`` (the rider), ``STOP`` (latched by an
    e-stop). Nobody aboard: AUTO only. The rider takes over by turning the bar against the motor or twisting the
    throttle; engages AUTO with the button."""

    def __init__(self, rider):
        self.rider = rider
        self.mode = "AUTO" if rider == "none" else "MANUAL"
        self._push = 0.0
        self.events = []

    def update(self, t, dt, bar_torque=0.0, throttle=False, auto_button=False, estop=False):
        if estop:
            if self.mode != "STOP":
                self.events.append((t, "STOP (e-stop)"))
            self.mode = "STOP"
            return self.mode
        if self.mode == "STOP":
            return self.mode
        if self.rider == "none":
            self.mode = "AUTO"
            return self.mode
        if self.mode == "MANUAL" and auto_button:
            self.mode = "AUTO"
            self.events.append((t, "AUTO engaged by the rider"))
        elif self.mode == "AUTO":
            self._push = self._push + dt if abs(bar_torque) > STEER_BY_WIRE["takeover_Nm"] else 0.0
            if throttle or self._push >= STEER_BY_WIRE["takeover_s"]:
                self.mode = "MANUAL"
                self._push = 0.0
                self.events.append((t, "MANUAL: the rider took over"))
        return self.mode


# ---------------------------------------------------------------- CAD: the autonomy kit (mm, vehicle frame)
class AutonomyKit(Design):
    """The parts the autonomy adds (``gorilla_quad_autonomy`` docstring), each in the vehicle frame at its place:
    ``lidar_front``, ``rear_mast`` (with its lidar and beacon), ``camera_bar``, ``gnss``, ``estops``,
    ``steer_actuator`` (motor, gearbox and torque sensor round the column's foot), ``column_telescope`` (the column's
    sliding stage), ``air_seat`` (the bellows under the seat pan), or ``kit`` (all)."""

    parameters = [Parameter("part", "kit", choices=("kit", "lidar_front", "rear_mast", "beacon", "camera_bar", "gnss", "estops", "steer_actuator",
                                                    "column_telescope", "air_seat"), description="what to build")]

    @staticmethod
    def _puck(pos, r=52.0, h=75.0):
        x, y, z = (v * 1000 for v in pos)
        body = cq.Workplane("XY").circle(r).extrude(h).translate((x, y, z - h / 2)).edges().fillet(8)
        window = cq.Workplane("XY").circle(r + 1).circle(r - 4).extrude(28).translate((x, y, z - 10))
        return body.union(window)

    def lidar_front(self, p):
        x, y, z = LIDARS["front"]["pos"]
        bracket = cq.Workplane("XY").box(90, 120, (z - 0.62) * 1000 - 37).translate((x * 1000 - 30, 0, 620 + ((z - 0.62) * 1000 - 37) / 2))
        return self._puck(LIDARS["front"]["pos"]).union(bracket)

    def rear_mast(self, p):
        x, y, z = LIDARS["rear"]["pos"]
        t = gq.GEOM["trunk"]
        z0 = (t["floor_z"] + t["h"]) * 1000
        mast = cq.Workplane("XY").circle(20).extrude(z * 1000 - 37 - z0).translate((x * 1000, 0, z0))
        return self._puck(LIDARS["rear"]["pos"]).union(mast)

    def beacon(self, p):
        x, y, z = LIDARS["rear"]["pos"]
        return cq.Workplane("XY").circle(34).extrude(70).translate((x * 1000, 0, z * 1000 + 38)).edges(">Z").fillet(20)

    def camera_bar(self, p):
        bx = gq.GEOM["column_base"][0]
        bar = cq.Workplane("XY").box(50, 300, 45).translate((bx * 1000 + 160, 0, 760)).edges("|X").fillet(10)
        for s in (-1, 1):
            bar = bar.union(cq.Workplane("YZ").circle(14).extrude(12).translate((bx * 1000 + 185, s * 110, 760)))
        return bar

    def gnss(self, p):
        t = gq.GEOM["trunk"]
        z = (t["floor_z"] + t["h"]) * 1000 + 12
        out = None
        for s in (-1, 1):
            a = (cq.Workplane("XY").circle(70).extrude(22).translate((t["x"] * 1000 + 120, s * (t["w"] * 500), z)).edges(">Z").fillet(10))
            out = a if out is None else out.union(a)
        return out

    def estops(self, p):
        t = gq.GEOM["trunk"]
        out = None
        pts = [((t["x"] - t["len"] / 2) * 1000 + 30, s * (t["w"] * 500 - 40), (t["floor_z"] + t["h"]) * 1000 + 15) for s in (-1, 1)]
        pts.append((gq.GEOM["column_base"][0] * 1000 + 40, 0.0, 1000.0))
        for x, y, z in pts:
            base = cq.Workplane("XY").box(60, 60, 40).translate((x, y, z))
            cap = cq.Workplane("XY").circle(30).extrude(22).translate((x, y, z + 20)).edges(">Z").fillet(10)
            b = base.union(cap)
            out = b if out is None else out.union(b)
        return out

    def steer_actuator(self, p):
        base = np.asarray(gq.GEOM["column_base"]) * 1000
        ax = gq._column_axis()
        foot = base - 180 * ax
        motor = cq.Workplane("XY").circle(55).extrude(160).translate((0, 0, -80))
        return motor.rotate((0, 0, 0), (0, 1, 0), -gq.GEOM["column_rake_deg"]).translate(tuple(foot))

    def column_telescope(self, p):
        base = np.asarray(gq.GEOM["column_base"]) * 1000
        rail = cq.Workplane("XY").box(360, 70, 40).translate((base[0] - 120, 0, base[2] - 10))
        return rail

    def air_seat(self, p):
        x = gq.GEOM["pelvis"][0] * 1000
        z = (gq.GEOM["seat_top"] - 0.10) * 1000
        out = None
        for k in range(3):
            ring = cq.Workplane("XY").circle(120 - 12 * (k % 2)).extrude(36).translate((x + 40, 0, z - 40 * (k + 1))).edges().fillet(12)
            out = ring if out is None else out.union(ring)
        return out

    def build(self, p):
        if p["part"] != "kit":
            return getattr(self, p["part"])(p)
        parts = [getattr(self, n)(p) for n in ("lidar_front", "rear_mast", "beacon", "camera_bar", "gnss", "estops", "steer_actuator", "column_telescope", "air_seat")]
        return cq.Compound.makeCompound([w.val() for w in parts])


KIT_COLORS = {"lidar_front": (0.10, 0.10, 0.11, 1.0), "rear_mast": (0.10, 0.10, 0.11, 1.0), "beacon": (0.2, 0.8, 0.3, 1.0), "camera_bar": (0.15, 0.15, 0.17, 1.0),
              "gnss": (0.92, 0.92, 0.90, 1.0), "estops": (0.85, 0.10, 0.08, 1.0), "steer_actuator": (0.25, 0.27, 0.30, 1.0),
              "column_telescope": (0.45, 0.47, 0.50, 1.0), "air_seat": (0.12, 0.12, 0.13, 1.0)}


def kit_mesh():
    """The kit's pieces for ``kerchak_render``, all on the quad's sprung body: [(body, triangles, rgba)] in
    ``KIT_COLORS``' order (the beacon's colour is the mode's, set per frame)."""
    d = AutonomyKit()
    p = d.resolve()
    pieces = []
    for name, rgba in KIT_COLORS.items():
        pieces.append(("quad_chassis", gq._tess(getattr(d, name)(p)), rgba))
    return pieces
