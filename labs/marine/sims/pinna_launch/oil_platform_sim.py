"""Pinna at an oil platform: the carrier, carrying only one submarine and one quadcopter, arrives at a fixed steel
jacket (``labs/marine/designs/oil_platform.py``: ~110 m of water, 4 battered legs, X-braces, risers, the topsides),
holds station on DP off its south-west corner, flies the quad on a visual inspection pass and puts the submarine
over the side on its motion-compensated davit; the submarine dives towards the nearest leg.

* **the submarine: Squilla** (``squilla.py``), the resident inspection/intervention AUV - Nephro's hull with eight
  tunnel thrusters for a stiff hover, a work arm and a hold arm, rated 300 m: the jacket's legs, nodes and anodes
  (CP stab, UT, close visual, cut and clear) are its job, and at 139 kg it is the only one of the two candidates a
  davit can launch. Homarus, the work-class ROV, is 4.2 t on a lift umbilical and a TMS: it needs a support vessel's
  A-frame LARS, not a davit. Squilla is still over the Walrus davit's 60 kg rating: the aft port davit is fitted with
  the uprated AHC winch (``pinna.DAVIT_UPRATED``: 3 kN line pull) for this job.
* **the quad: Gannet, flying empty** (11.4 kg): it is the quad Pinna's pads, locks and swap robot serve; without
  the crate it hovers ~35 min and holds station in 14 m/s of wind - an offshore topsides/flare inspection needs the
  wind margin and the endurance more than a small frame (Drongo, a 250-class quad, has minutes of hover and no
  margin in offshore wind). Its downward camera and lidar are the inspection sensors here.
* **the scene**: the jacket's centre 42 m north of Pinna's station, so its nearest leg (A1) is ~25 m from the
  launching davit (a compressed stand-off for the clip; a real 500 m safety zone needs the platform's permit). The
  carrier comes in from the west at ~4 m/s and stops on station (``track``).

``squilla_agent``, ``inspection_mission``, ``JacketScenery`` are used by the ``oil_platform_deploy`` clip.
"""
from __future__ import annotations

import math

import numpy as np

from . import world
from .gannet_sim import GannetAgent
from .walrus_sim import WalrusAgent, _ramp
from .world import G, RHO_SW, min_jerk

STATION = np.array([30.0, 0.0])            # Pinna's DP station (it starts at the origin, where its locks were made)
JACKET_AT = STATION + np.array([0.0, 42.0])   # the jacket's centre (m, world), 42 m north of the station
ARRIVE = (0.0, 30.0, 8.0)                  # 30 m in from the west in 8 s


def track(t):
    """Pinna's way over the ground: in from the west, stopped on station."""
    x, v, _ = min_jerk(t, 0.0, ARRIVE[2], [ARRIVE[0]], [ARRIVE[1]])
    return float(x[0]), 0.0, float(v[0]), 0.0


def leg_point(leg: str, z: float) -> np.ndarray:
    """A jacket leg's centre at height z (world)."""
    from oil_platform import LEGS, Jacket
    p = Jacket.p()
    sx, sy = LEGS[leg]
    h = p["half_top"] + (p["top"] - z) / p["batter"]
    return np.array([JACKET_AT[0] + sx * h, JACKET_AT[1] + sy * h, z])


class JacketScenery:
    """The jacket as drawn meshes (no collisions: neither machine is flown into it)."""

    name = "jacket"
    designs = ("jacket",)

    def xml(self, use_meshes):
        from . import meshes
        col = meshes.JACKET_PARTS
        if use_meshes:
            vis = "".join(world._vis_mesh(f"jacket_{k}", c) for k, c in col.items())
        else:
            vis = "".join(f'<geom class="visual" type="cylinder" size="1 61" pos="{12 * sx} {12 * sy} -49" rgba="0.8 0.62 0.2 1"/>'
                          for sx in (-1, 1) for sy in (-1, 1))
        return {"world": f'<body name="jacket" pos="{JACKET_AT[0]} {JACKET_AT[1]} 0">{vis}</body>'}

    def bind(self, sim):
        pass

    def control(self, sim, t, dt):
        pass

    def hud(self, sim):
        return []


class InspectionMission:
    """Gannet's inspection pass: lift-off following the pad, out to the jacket's south-west corner, down to the
    splash zone 7 m off leg A1, up the leg under the cellar deck, out round the deck's corner and up the west face to
    the flare tower; the nose (its camera) on the leg or the tower throughout."""

    def __init__(self, pad_xy, z_pad, t_release=9.0):
        self.t_rel, self.t_lift = t_release, t_release + 1.4
        self.t_track = 1e9
        self.z_pad = z_pad
        a1 = leg_point("A1", 4.0)
        off = np.array([-0.70, -0.70, 0.0]) * 7.0
        tower = np.array([JACKET_AT[0] + 7.0, JACKET_AT[1] + 7.0, 45.0])
        J = np.array([JACKET_AT[0], JACKET_AT[1], 0.0])
        self.wp = [(self.t_lift + 5.0, J + [-21.0, -25.0, 14.0], "transit to the jacket's south-west corner", a1),
                   (self.t_lift + 9.0, a1 + off, "down to the splash zone, 7 m off leg A1", a1),
                   (self.t_lift + 16.0, leg_point("A1", 15.0) + off, "inspection pass up leg A1 (nodes, anodes, risers)", leg_point("A1", 15.0)),
                   (self.t_lift + 21.0, J + [-21.0, -20.0, 33.0], "out round the cellar deck's corner", tower),
                   (self.t_lift + 27.0, J + [-24.0, 2.0, 42.0], "up the west face: the topsides and the flare tower", tower)]
        self._p_lift = None

    def phase(self, t, landed=False):
        if t < self.t_rel:
            return "on the pad, harpoon locked"
        if t < self.t_lift:
            return "lift-off from the moving deck"
        for t1, _, what, _ in self.wp:
            if t < t1:
                return what
        return "hover off the flare tower: close visual"

    def ref(self, t, pad_p, pad_v):
        if t < self.t_lift:
            dz, vz, az = min_jerk(t, self.t_rel, self.t_lift, [0.0], [1.5])
            return pad_p + np.array([0, 0, 0.04 + dz[0]]), pad_v + np.array([0, 0, vz[0]]), np.array([0, 0, az[0]]), 0.0
        if self._p_lift is None:
            self._p_lift = pad_p + np.array([0.0, 0.0, 1.54])
        t0, p0 = self.t_lift, self._p_lift
        look = self.wp[0][3]
        for t1, p1, _, look in self.wp:
            if t < t1:
                p, v, a = min_jerk(t, t0, t1, p0, p1)
                break
            t0, p0 = t1, p1
        else:
            p, v, a = p0.copy(), np.zeros(3), np.zeros(3)
        d = look - p
        return p, v, a, math.atan2(d[1], d[0])


def squilla_card() -> dict:
    from squilla import Squilla
    p, s = Squilla.p(), Squilla.sizing()
    m = s["mass_in_air_kg"]
    L, D = p["length"] / 1000, p["diameter"] / 1000
    th = s["thrust_N"]
    dummy = [{"at": (0, 0, 0), "axis": (1, 0, 0), "max_N": 0.0}] * 6
    return {"p": p, "s": s, "mass": m, "volume": (m + 1.0) / RHO_SW,                 # trimmed 1 kg buoyant
            "L": 2.6, "W": 0.48, "H": 0.74, "thrusters": dummy,
            "t_max": np.array([th["surge"], th["lateral"], th["vertical"], 40.0, 0.5 * th["vertical"], 0.5 * th["lateral"]]),
            "J": np.array([m * (D / 2) ** 2 / 2, m * L * L / 12, m * L * L / 12]),
            "cda": np.array([0.25 * math.pi * (D / 2) ** 2, 1.0 * L * D, 1.1 * L * D])}


class SquillaAgent(WalrusAgent):
    """Squilla on the compensated davit: the Walrus's launch (hoist, latch cradle, luff out, AHC lowering through the
    splash zone, the hook's release) with Squilla's body; then its own thrusters (a body-axis wrench limited by the
    card's surge, lateral and vertical thrust) take it to 3.5 m off leg A1 at 12 m depth, nose to the leg."""

    name = "squilla"
    bname = "squilla"
    designs = ("squilla",)
    X_COM = 0.15                                   # the CG along the CAD's axis (m)
    COM = np.array([0.0, 0.0, 0.538])              # body origin at the skids' bottom, the axis 0.538 m above
    COB = np.array([0.0, 0.0, 0.57])
    EYE = np.array([0.0, 0.0, 0.86])
    card = staticmethod(squilla_card)

    def __init__(self, **kw):
        kw = {"t_hoist": (9.0, 11.7), "t_swing": (11.7, 15.0), "t_lower": 16.0, "pay_out": 0.8} | kw
        super().__init__(**kw)
        self.goal = leg_point("A1", -12.0) + np.array([-0.3, -3.5, 0.0])
        self.T = np.zeros(6)
        self.davit.P = self.davit.P + np.array([STATION[0], STATION[1], 0.0])     # the earth-fixed sheave point, on station

    def xml(self, use_meshes):
        c = self.c
        from . import meshes
        col = meshes.SQUILLA_PARTS
        pos = (-self.X_COM, 0.0, self.COM[2])
        if use_meshes:
            vis = "".join(world._vis_mesh(f"squilla_{k}", rgba, pos=pos) for k, rgba in col.items())
        else:
            vis = f'<geom class="visual" type="capsule" size="0.2" fromto="-1 0 {self.COM[2]} 1.1 0 {self.COM[2]}" rgba="0.95 0.8 0.15 1"/>'
        J = c["J"]
        out = super().xml(use_meshes)
        body = f"""<body name="squilla" pos="{world._f(self.start)}" quat="{world._f(self.QUAT)}">
<freejoint name="squilla_free"/>
<inertial pos="{world._f(self.COM)}" mass="{c['mass']:.3f}" diaginertia="{world._f(J)}"/>
{vis}
<geom name="s_hull" class="drone" type="capsule" size="0.2" fromto="-1.0 0 {self.COM[2]} 1.05 0 {self.COM[2]}"/>
<geom name="s_skids" class="drone" type="box" size="0.6 0.22 0.05" pos="0 0 0.05"/>
<site name="squilla_eye" pos="{world._f(self.EYE)}" size="0.03"/>
<geom class="visual" type="cylinder" size="0.015 0.06" pos="{world._f(self.EYE - [0, 0, 0.06])}" rgba="0.8 0.8 0.1 1"/>
</body>"""
        return {"world": body, "carrier": out["carrier"],
                "tendon": out["tendon"].replace("walrus_cable", "squilla_cable").replace("walrus_eye", "squilla_eye"),
                "equality": '<weld name="squilla_latch" body1="davit_boom" body2="squilla" active="false" solref="0.02 1"/>'}

    def _thrust(self, com, R, eta, F, tau):
        if self.sub < 0.5:
            return F, tau
        return F + R @ self.T[:3], tau + R @ self.T[3:]

    def _allocate(self, t, com, v, R, w, eta):
        c = self.c
        m = c["mass"]
        T = 16.0
        p_ref, v_ref, a_ref = min_jerk(t - self.t0, 1.0, 1.0 + T, self.p0, self.goal)
        Fw = m * (0.5 * (p_ref - com) + 1.4 * (v_ref - v) + a_ref)
        Fw[2] += m * G - RHO_SW * G * c["volume"] * self.sub
        Fb = R.T @ Fw
        Rd = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])           # nose north, level
        E = Rd.T @ R - R.T @ Rd
        eR = 0.5 * np.array([E[2, 1], E[0, 2], E[1, 0]])
        J = c["J"]
        Mb = -J * 4.0 * eR - J * 2 * 0.9 * 2.0 * w
        u = np.concatenate([Fb, Mb])
        return np.clip(u, -c["t_max"], c["t_max"])

    def phase(self, t):
        if t < self.t_hoist[0]:
            return "on its cradle, hooked on"
        if t < self.t_hoist[1]:
            return "davit: hoist into the latch cradle"
        if t < self.t_swing[1]:
            return "davit: boom luffs outboard"
        if self.released is None:
            return "AHC lowering through the splash zone" if self.sub > 0 else "AHC lowering, compensated"
        if t < self.released + 1.0:
            return "hook released, thrusters on"
        return "diving to leg A1 (12 m), nose to the leg"

    def hud(self, sim):
        com, v, R, w = self.state()
        dist = float(np.linalg.norm(self.goal - com))
        tail = f"   {dist:4.1f} m to its work spot at leg A1" if self.released is not None else f"   cable {self.L:4.2f} m"
        return [f"Squilla {self.c['mass']:.0f} kg   depth {self.depth():+5.1f} m   speed {np.linalg.norm(v):4.2f} m/s{tail}   phase: {self.phase(sim.t)}"]

    def trace(self, sim, span=10.0):
        return None


def inspection_gannet() -> GannetAgent:
    g = world.geometry()
    pad = 3
    return GannetAgent(pad=pad, crate=False, mission=InspectionMission(g["pads"][pad], g["z_pad"]))
