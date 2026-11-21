"""Kerchak's golf-ball bazooka: Armageddon's golf-ball launcher (the Pekari's turret: ``armageddon/launcher.py`` -
two counter-rotating flywheels throwing golf balls down a smooth tube, rims 20-70 m/s, a ball every 0.4 s at most)
made a shoulder weapon for a gorilla, and the saddle scabbard it rides in on his quad (``gorilla_quad``), beside his
right hip, where a dragoon's carbine rides on a horse.

The weapon, in its own frame (origin at the rear handle's bar, where his right fist closes; x along the tube to
the muzzle, z up):

* the tube 0.70 m (44 mm bore for a 42.7 mm ball), its axis ``TUBE_Z`` above the fist;
* the flywheel housing behind the grip (two 100 mm wheels, their motors), a 24-ball hopper on it;
* two D-handles under it, rear and fore, each a 50 mm bar across the weapon - the handlebar's bar, held the same
  way (his fists from behind and below);
* a long stock to the shoulder pad 0.55 m behind the grip - his arms are 1.04 m, a human-length stock would fold
  them double;
* a battery in the stock, a sight on the tube.

``SPEC`` has the numbers; the ball's flight (``fly``) is the game's: gravity and a smooth sphere's drag against its
Reynolds number (``balls.drag_coefficient``), still air. ``elevation_for(range, height)`` finds the tube's
elevation that puts a ball there.
"""
from __future__ import annotations

import math

import cadquery as cq
import numpy as np

import gorilla_quad as gq
from vegeta.dedalus import Design, Parameter

__all__ = ["SPEC", "BALL", "GRIP", "FORE", "PAD", "MUZZLE", "TUBE_Z", "SCABBARD", "Bazooka", "fly", "elevation_for",
           "hand_to_weapon", "weapon_to_hand", "hand_in_weapon", "bazooka_mesh", "scabbard_mesh", "drag_force"]

#: the golf ball (Armageddon's balls.py)
BALL = {"radius": 0.021335, "mass": 0.04593, "rho_air": 1.225, "nu_air": 1.48e-5}
#: the weapon (the launcher's own numbers from Armageddon's launcher.py; the rest for the shoulder version)
SPEC = {"rim_speed_range": (20.0, 70.0), "rim_speed": 45.0, "grip": 0.97, "feed_s": 0.4, "scatter_rad": math.radians(0.25),
        "scatter_speed": 0.01, "hopper": 24, "tube_m": 0.70, "bore_mm": 44.0,
        "mass_kg": {"tube (CF, 44 mm bore)": 0.9, "flywheels + 2 BLDC motors": 1.6, "housing": 1.1, "hopper": 0.4,
                    "stock, pad, grips": 1.4, "battery 6S 2.2 Ah in the stock": 0.4, "sight, electronics": 0.3}}

TUBE_Z = 0.10                   #: the tube's axis above the handles' bars [m]
GRIP = np.zeros(3)              #: the rear handle's bar centre: the weapon frame's origin
FORE = np.array([0.40, 0.0, 0.0])        #: the fore handle's bar centre (his left fist)
PAD = np.array([-0.55, 0.0, TUBE_Z - 0.02])   #: the shoulder pad's centre
MUZZLE = np.array([0.55, 0.0, TUBE_Z])   #: where a ball leaves

#: the scabbard on the quad: the weapon's frame in the quad's (it lies level, the muzzle forward, beside his right hip)
SCABBARD = {"pos": (-0.08, -0.60, 0.80), "yaw": 0.0}


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


# ---------------------------------------------------------------- the fists on the handles
#: both handles are D-handles: a 50 mm bar across the weapon (along its y) under the body - the same bar his fists
#: close on at the handlebar, held the same way: from behind and below, the wrist back and down from the bar
WRIST_DIR = (-0.75, 0.0, -0.66)


def hold_in_weapon(side):
    """The hold (``gorilla_scenarios.bar``) of ``side``'s fist, in the weapon's own frame: the right on the rear
    handle, the left on the fore handle."""
    import gorilla_scenarios as gs
    return gs.bar(GRIP if side == "R" else FORE, axis=(0.0, 1.0, 0.0), n=(-1.0, 0.0, 0.0), wrist_dir=WRIST_DIR)


def hand_in_weapon(side):
    """``side``'s hand pose (rotation, origin) in the weapon's frame with the fist closed on its handle."""
    import gorilla_scenarios as gs
    h = hold_in_weapon(side)
    R = gs.hand_frame_on(h, side)
    return R, h["c"] - R @ gs.GRIP


def weapon_to_hand(side, W_rot, W_pos):
    """The hand's pose (rotation, origin) that closes ``side``'s fist on its handle of the weapon at (``W_rot``, ``W_pos``)."""
    R, p = hand_in_weapon(side)
    return W_rot @ R, W_pos + W_rot @ p


def hand_to_weapon(side, H_rot, H_pos):
    """The weapon's pose (rotation, origin) held in ``side``'s fist at (``H_rot``, ``H_pos``)."""
    R, p = hand_in_weapon(side)
    W_rot = H_rot @ R.T
    return W_rot, H_pos - W_rot @ p


# ---------------------------------------------------------------- the ball's flight
def drag_coefficient(speed):
    re = speed * 2 * BALL["radius"] / BALL["nu_air"]
    if re <= 2.5e5:
        return 0.47
    if re >= 4.0e5:
        return 0.12
    return 0.47 - 0.35 * (re - 2.5e5) / 1.5e5


def drag_force(v):
    """The air's force on a ball moving at ``v`` [m/s, 3-vector] (N)."""
    v = np.asarray(v, dtype=float)
    s = float(np.linalg.norm(v))
    return -0.5 * BALL["rho_air"] * drag_coefficient(s) * math.pi * BALL["radius"] ** 2 * s * v


def fly(p, v, until_x, dt=1e-3, t_max=4.0):
    """A ball from ``p`` at ``v`` (x forward, z up) to where it crosses ``until_x``: (height there, time)."""
    p, v = np.array(p, dtype=float), np.array(v, dtype=float)
    t = 0.0
    while p[0] < until_x and t < t_max:
        a = drag_force(v) / BALL["mass"] + np.array([0.0, 0.0, -9.81])
        v += a * dt
        p += v * dt
        t += dt
    return float(p[2]), t


def elevation_for(dx, dz, speed=None):
    """The tube's elevation [rad] that puts a ball ``dx`` ahead and ``dz`` above the muzzle (the lower of the two arcs)."""
    speed = SPEC["rim_speed"] * SPEC["grip"] if speed is None else speed
    lo, hi = math.radians(-10.0), math.radians(40.0)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        z, _ = fly((0.0, 0.0, 0.0), (speed * math.cos(mid), 0.0, speed * math.sin(mid)), dx)
        lo, hi = (mid, hi) if z < dz else (lo, mid)
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------- CAD (mm, the weapon's frame)
def _cyl_x(r, x0, x1, y=0.0, z=0.0):
    return cq.Workplane("YZ").circle(r).extrude(x1 - x0).translate((x0, y, z))


class Bazooka(Design):
    """The golf-ball bazooka (``part="bazooka"``) in its frame, or the saddle scabbard (``part="scabbard"``) in the
    quad's frame."""

    parameters = [Parameter("part", "bazooka", choices=("bazooka", "scabbard"), description="what to build")]

    def bazooka(self, p):
        mm = 1000.0
        tz = TUBE_Z * mm
        bore = SPEC["bore_mm"] / 2
        tube = _cyl_x(bore + 4, -60, MUZZLE[0] * mm, z=tz).cut(_cyl_x(bore, -70, MUZZLE[0] * mm + 10, z=tz))
        muzzle = _cyl_x(bore + 9, MUZZLE[0] * mm - 40, MUZZLE[0] * mm, z=tz).cut(_cyl_x(bore, MUZZLE[0] * mm - 50, MUZZLE[0] * mm + 10, z=tz))
        housing = cq.Workplane("XY").box(170, 150, 130).translate((-140, 0, tz)).edges("|Y").fillet(25)
        wheels = [cq.Workplane("XZ").circle(50).extrude(14, both=True).translate((-120, s * 45, tz)) for s in (-1, 1)]
        hopper = cq.Workplane("XY").box(150, 110, 120).translate((-150, 0, tz + 125)).edges("|Z").fillet(20)
        feed = cq.Workplane("XY").circle(24).extrude(70).translate((-120, 0, tz + 40))
        handles = []
        for hx in (GRIP[0] * mm, FORE[0] * mm):                            # D-handles: a 50 mm bar across, two lugs up into the body
            handles.append(cq.Workplane("XZ").circle(25).extrude(70, both=True).translate((hx, 0, 0)))
            for sy in (-1, 1):
                handles.append(cq.Workplane("XY").box(30, 14, tz - 20).translate((hx, sy * 77, (tz - 20) / 2 + 10)))
        trigger = cq.Workplane("XY").box(16, 30, 26).translate((GRIP[0] * mm + 34, 0, 18))
        stock = cq.Workplane("XY").box(PAD[0] * -mm - 210, 60, 70).translate(((PAD[0] * mm - 210) / 2, 0, tz - 25))
        pad = cq.Workplane("XY").box(40, 110, 170).translate((PAD[0] * mm, 0, PAD[2] * mm)).edges("|X").fillet(30)
        sight = cq.Workplane("XY").box(90, 26, 40).translate((200, 0, tz + bore + 24))
        parts = [tube, muzzle, housing, *wheels, hopper, feed, *handles, trigger, stock, pad, sight]
        return cq.Compound.makeCompound([w.val() for w in parts])

    def scabbard(self, p):
        """Two U-cradles (under the stock and near the muzzle) on stays down to the frame's lower rail."""
        mm = 1000.0
        x0, y0, z0 = (v * mm for v in SCABBARD["pos"])
        tz = z0 + TUBE_Z * mm
        out = []
        for dx, r in ((-420.0, 48.0), (500.0, 36.0)):
            cradle = (cq.Workplane("YZ").circle(r + 8).circle(r).extrude(40).translate((x0 + dx - 20, y0, tz - (25 if dx < 0 else 0)))
                      .cut(cq.Workplane("XY").box(60, 160, 80).translate((x0 + dx, y0, tz + 40))))
            stay = gq._tube((x0 + dx, y0, tz - r - 30), (x0 + dx, -gq.GEOM["frame_y"] * mm, gq.GEOM["rail_z"] * mm), 10)
            out += [cradle.val(), stay]
        return cq.Compound.makeCompound(out)

    def build(self, p):
        return getattr(self, p["part"])(p)


BAZOOKA_RGBA = (0.18, 0.30, 0.20, 1.0)
SCABBARD_RGBA = (0.35, 0.25, 0.15, 1.0)


def bazooka_mesh(body="bazooka"):
    d = Bazooka()
    return [(body, gq._tess(d.bazooka(d.resolve())), BAZOOKA_RGBA)]


def scabbard_mesh():
    d = Bazooka()
    return [("quad_chassis", gq._tess(d.scabbard(d.resolve(part="scabbard"))), SCABBARD_RGBA)]


def scabbard_pose():
    """The weapon's pose in the quad's frame when it rides in the scabbard."""
    return _rot_z(SCABBARD["yaw"]), np.asarray(SCABBARD["pos"], dtype=float)
