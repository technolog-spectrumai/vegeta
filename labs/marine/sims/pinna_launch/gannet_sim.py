"""Gannet in MuJoCo: the 16.4 kg cargo quad (``labs/aero/designs/gannet.py``) with its 5 kg crate, locked on a pad of
Pinna's flight deck by the harpoon probe, taking off, climbing to 30 m, transitioning to a 15 m/s cruise, flying a
circuit and coming back to land on the moving pad (the probe locks again on touch-down).

* **the airframe**: one free body with the card's mass (empty + crate) and an estimated inertia; drawn with the CAD
  meshes (body, arms, rotor discs, skids, crate), colliding with simple shapes (the skids as capsules, the crate, the
  body) - the skids are what touches the pad.
* **the rotors**: four thrusts along the body's z at the CAD's rotor positions (28 in props, ``Gannet._rotor_at``),
  each a first-order lag of its command (``MOTOR_TAU``) clipped to the card's maximum (thrust/weight 3.55), with the
  reaction torque ``K_Q`` x thrust about z; applied as a wrench on the body (as ``races.sim`` does). The electrical
  power shown is momentum theory with the card's figure of merit and drive efficiency.
* **the air**: the frame's drag 1/2 rho CdA |v_rel| v_rel with the card's CdA against the wind (``world.Wind``).
* **the controller** (``Controller``): the geometric (Lee) position and attitude controller the lab's quads fly
  (``drongo_controller`` / ``races.flight``): a PID on the position error gives the force vector (tilt-limited), the
  attitude error on SO(3) the moments, a mixer the four thrusts. The guidance (``Mission``) gives the reference: a
  minimum-jerk lift-off that follows the pad (the deck's motion comes from Pinna's INS), the climb, the transition
  as a speed ramp, the circuit at 15 m/s, the approach from astern into the wind, a hover 5 m over the pad tracking
  its heave and roll, and the descent at 0.6 m/s relative to the deck down to the touch-down.
"""
from __future__ import annotations

import math

import numpy as np

from . import world
from .world import G, RHO_AIR, min_jerk

MOTOR_TAU = 0.07          # s, 28 in props on 6 kW-peak motors (assumed)
K_Q = 0.018               # m, reaction torque per newton of thrust (assumed for a 28 in prop)
PAD = 3                   # Pinna's pad (-2.6, -2.4): the starboard forward one


def card() -> dict:
    from gannet import Gannet
    p, s = Gannet.p(), Gannet.sizing()
    a = p["arm"] / math.sqrt(2) / 1000
    m = s["mtow_kg"]
    t_max = s["thrust_weight"] * m * G / 4
    return {"p": p, "s": s, "mass": m, "rotors": np.array([(a, a), (a, -a), (-a, a), (-a, -a)]), "spin": np.array([1, -1, -1, 1]),
            "t_max": t_max, "disc": math.pi * (p["prop_in"] * 0.0254 / 2) ** 2, "cda": p["drag_area"],
            "J": np.array([1.05, 1.15, 1.95]), "com_z": 0.33, "rotor_z": 0.52}


class Controller:
    """Lee's geometric controller: position PID -> force vector (tilt <= ``tilt_max``) -> attitude PD on SO(3)."""

    def __init__(self, m, J, rotors, spin, t_max, *, wn=1.6, zeta=1.0, ki=0.35, w_att=7.0, z_att=0.85, tilt_max_deg=32.0):
        self.m, self.J, self.t_max = m, np.asarray(J), t_max
        self.kp, self.kd, self.ki = wn * wn, 2 * zeta * wn, ki
        self.kR = self.J * w_att ** 2
        self.kR[2] = self.J[2] * 3.0 ** 2
        self.kW = 2 * z_att * self.J * w_att
        self.kW[2] = 2 * 0.9 * self.J[2] * 3.0
        self.tilt = math.radians(tilt_max_deg)
        self.A = np.vstack([np.ones(4), rotors[:, 1], -rotors[:, 0], spin * K_Q])
        self.Ainv = np.linalg.inv(self.A)
        self.I = np.zeros(3)

    def reset(self):
        self.I[:] = 0.0

    def thrusts(self, p, v, R, w, p_ref, v_ref, a_ref, yaw, dt, drag_ff=np.zeros(3)):
        e, ev = p_ref - p, v_ref - v
        self.I = np.clip(self.I + e * dt, -3.0, 3.0)
        a = a_ref + self.kp * e + self.kd * ev + self.ki * self.I
        F = self.m * (a + np.array([0.0, 0.0, G])) - drag_ff
        F[2] = max(F[2], 0.2 * self.m * G)
        h = math.hypot(F[0], F[1])
        if h > F[2] * math.tan(self.tilt):
            F[:2] *= F[2] * math.tan(self.tilt) / h
        zb = F / np.linalg.norm(F)
        xc = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        yb = np.cross(zb, xc); yb /= np.linalg.norm(yb)
        Rd = np.column_stack([np.cross(yb, zb), yb, zb])
        eR = 0.5 * _vee(Rd.T @ R - R.T @ Rd)
        tau = -self.kR * eR - self.kW * w + np.cross(w, self.J * w)
        f = float(F @ R[:, 2])
        T = self.Ainv @ np.array([f, *tau])
        return np.clip(T, 0.0, self.t_max), F


def _vee(M):
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


class Mission:
    """Gannet's guidance from the pad and back: phase times and the reference (p, v, a, yaw) at t."""

    def __init__(self, pad_xy, z_pad, *, t_release=1.5, alt=30.0, cruise=15.0, out_x=110.0, radius=45.0, back_x=-110.0):
        self.pad_xy, self.z_pad = np.array(pad_xy, float), z_pad
        self.t_rel, self.alt, self.V, self.out_x, self.r, self.back_x = t_release, alt, cruise, out_x, radius, back_x
        self.t_lift = t_release + 1.4                         # 1.5 m over the pad, following it
        self.t_climb = self.t_lift + 7.0                      # to the cruise altitude
        self.acc = 2.5                                        # m/s^2 transition
        y0 = pad_xy[1]
        # the circuit: out along +x at y0, a left turn of radius r (to port), back along -x, a left turn, the approach
        self.legs = []
        x_start = pad_xy[0]
        t = self.t_climb
        t_acc = self.V / self.acc
        x_acc = x_start + 0.5 * self.acc * t_acc ** 2
        self.legs.append(("accel", t, t + t_acc, x_start))
        t += t_acc
        self.legs.append(("out", t, t + (out_x - x_acc) / self.V, x_acc))
        t += (out_x - x_acc) / self.V
        self.legs.append(("turn1", t, t + math.pi * radius / self.V, None))
        t += math.pi * radius / self.V
        self.legs.append(("back", t, t + (out_x - back_x) / self.V, None))
        t += (out_x - back_x) / self.V
        self.legs.append(("turn2", t, t + math.pi * radius / self.V, None))
        t += math.pi * radius / self.V
        self.t_approach = t                                   # at (back_x, y0) heading +x at V, alt
        self.hover_h = 5.0
        dist = (pad_xy[0] - back_x)
        self.t_over = t + 2.0 * dist / self.V                 # a decelerating min-jerk-like profile to the hover spot
        self.t_track = self.t_over + 2.5                      # hover over the pad, tracking it
        self.v_desc = 0.6
        self.t_touch_plan = self.t_track + self.hover_h / self.v_desc
        self.y0 = y0
        self._p_lift = None

    def phase(self, t, landed=False) -> str:
        if landed:
            return "landed: harpoon locked"
        if t < self.t_rel:
            return "on the pad: harpoon locked, rotors spinning up"
        if t < self.t_lift:
            return "lift-off: lock released, following the pad"
        if t < self.t_climb:
            return "vertical climb to 30 m"
        for name, t0, t1, _ in self.legs:
            if t < t1:
                return {"accel": "transition to cruise", "out": "cruise 15 m/s", "turn1": "circuit: turn", "back": "circuit: downwind leg",
                        "turn2": "circuit: turn to final"}[name]
        if t < self.t_over:
            return "approach from astern, into the wind"
        if t < self.t_track:
            return "hover 5 m over the pad, tracking the deck"
        return "descent onto the moving deck"

    def ref(self, t, pad_p, pad_v):
        """(p, v, a, yaw) at t; ``pad_p``, ``pad_v`` the pad's centre now (world)."""
        z0 = np.zeros(3)
        x0, y0 = self.pad_xy
        if t < self.t_lift:
            dz, vz, az = min_jerk(t, self.t_rel, self.t_lift, [0.0], [1.5])
            off = np.array([0.0, 0.0, 0.04 + dz[0]])
            return pad_p + off, pad_v + np.array([0, 0, vz[0]]), np.array([0, 0, az[0]]), 0.0
        if t < self.t_climb:                                   # from where the lift-off left it (the pad has moved)
            if self._p_lift is None:
                self._p_lift = pad_p + np.array([0.0, 0.0, 1.54])
            p, v, a = min_jerk(t, self.t_lift, self.t_climb, self._p_lift, [x0, y0, self.alt])
            return p, v, a, 0.0
        for name, t0, t1, xs in self.legs:
            if t < t1:
                tau = t - t0
                if name == "accel":
                    return np.array([xs + 0.5 * self.acc * tau ** 2, y0, self.alt]), np.array([self.acc * tau, 0, 0]), np.array([self.acc, 0, 0]), 0.0
                if name == "out":
                    return np.array([xs + self.V * tau, y0, self.alt]), np.array([self.V, 0, 0]), z0, 0.0
                if name in ("turn1", "turn2"):
                    th = self.V * tau / self.r
                    cx, cy, th0 = (self.out_x, y0 + self.r, -math.pi / 2) if name == "turn1" else (self.back_x, y0 + self.r, math.pi / 2)
                    ang = th0 + th
                    p = np.array([cx + self.r * math.cos(ang), cy + self.r * math.sin(ang), self.alt])
                    v = self.V * np.array([-math.sin(ang), math.cos(ang), 0.0])
                    a = -(self.V ** 2 / self.r) * np.array([math.cos(ang), math.sin(ang), 0.0])
                    return p, v, a, math.atan2(v[1], v[0])
                if name == "back":
                    return np.array([self.out_x - self.V * tau, y0 + 2 * self.r, self.alt]), np.array([-self.V, 0, 0]), z0, math.pi
        if t < self.t_over:                                   # decelerate from V to the hover spot (quintic with v0 = V)
            T = self.t_over - self.t_approach
            s = (t - self.t_approach) / T
            p0 = np.array([self.back_x, y0, self.alt])
            p1 = np.array([pad_p[0], pad_p[1], self.z_pad + self.hover_h])
            v0 = np.array([self.V, 0, 0])
            # cubic Hermite in each axis: p(0)=p0, v(0)=v0, p(1)=p1, v(1)=0
            h00, h10, h01 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s, -2 * s ** 3 + 3 * s ** 2
            d00, d10, d01 = (6 * s ** 2 - 6 * s) / T, (3 * s ** 2 - 4 * s + 1), (-6 * s ** 2 + 6 * s) / T
            dd00, dd10, dd01 = (12 * s - 6) / T ** 2, (6 * s - 4) / T, (-12 * s + 6) / T ** 2
            p = h00 * p0 + h10 * T * v0 + h01 * p1
            v = d00 * p0 + d10 * v0 + d01 * p1
            a = dd00 * p0 + dd10 * v0 + dd01 * p1
            return p, v, a, 0.0
        if t < self.t_track:
            k = min(1.0, (t - self.t_over) / 1.5)            # blend into the pad's motion
            return pad_p + np.array([0, 0, self.hover_h]), k * pad_v, z0, 0.0
        h = max(self.hover_h - self.v_desc * (t - self.t_track), -0.15)
        vz = -self.v_desc if h > -0.15 else 0.0
        return pad_p + np.array([0, 0, h]), pad_v + np.array([0, 0, vz]), z0, 0.0


class GannetAgent:
    name = "gannet"

    def __init__(self, pad: int = PAD, mission_kw: dict | None = None, crate: bool = True, mission=None):
        self.c = card()
        self.crate = crate
        if not crate:                                            # flies empty (an inspection): the card's empty mass
            self.c["mass"] = self.c["s"]["empty_kg"]
        self.pad = pad
        g = world.geometry()
        self.pad_xy = g["pads"][pad]
        self.mission = mission or Mission(self.pad_xy, g["z_pad"], **(mission_kw or {}))
        self.landed = False
        self.locked = True
        self.T = np.zeros(4)
        self.log = []
        self.touch = None

    def xml(self, use_meshes: bool) -> dict:
        c, (x, y), z = self.c, self.pad_xy, world.geometry()["z_pad"]
        col = __import__("pinna_launch.meshes", fromlist=["colours"]).colours()
        vis = ""
        if use_meshes:
            for part in ("body", "arms", "rotors", "skids") + (("crate",) if self.crate else ()):
                vis += world._vis_mesh(f"gannet_{part}", col[f"gannet_{part}"])
        else:
            vis = ('<geom class="visual" type="box" size="0.21 0.17 0.08" pos="0 0 0.42" rgba="0.12 0.38 0.55 1"/>'
                   '<geom class="visual" type="box" size="0.2 0.15 0.1" pos="0 0 0.13" rgba="0.85 0.55 0.25 1"/>')
        J = c["J"]
        body = f"""<body name="gannet" pos="{x} {y} {z + 0.002:.4f}">
<freejoint name="gannet_free"/>
<inertial pos="0 0 {c['com_z']}" mass="{c['mass']:.3f}" diaginertia="{J[0]} {J[1]} {J[2]}"/>
{vis}
<geom name="g_skid_l" class="drone" type="capsule" size="0.014" fromto="-0.35 0.45 0.014 0.35 0.45 0.014" friction="1.0 0.02 0.002"/>
<geom name="g_skid_r" class="drone" type="capsule" size="0.014" fromto="-0.35 -0.45 0.014 0.35 -0.45 0.014" friction="1.0 0.02 0.002"/>
{'<geom name="g_crate" class="drone" type="box" size="0.2 0.15 0.1" pos="0 0 0.13"/>' if self.crate else ''}
<geom name="g_body" class="drone" type="box" size="0.21 0.17 0.1" pos="0 0 0.42"/>
<site name="gannet_cam" pos="0 0 0.3"/>
</body>"""
        eq = '<weld name="gannet_lock" body1="carrier" body2="gannet" solref="0.003 1"/>'
        return {"world": body, "equality": eq}

    def bind(self, sim):
        self.sim = sim
        self.b = sim.body("gannet")
        j = sim.joint("gannet_free")
        self.qa, self.va = int(sim.model.jnt_qposadr[j]), int(sim.model.jnt_dofadr[j])
        self.geoms = sim.body_geoms(self.b)
        self.pad_geom = {sim.geom(f"c_pad{self.pad}")}
        c = self.c
        self.ctrl = Controller(c["mass"], c["J"], c["rotors"], c["spin"], c["t_max"])
        self.pad_local = np.array([self.pad_xy[0], self.pad_xy[1], world.geometry()["z_pad"]])
        self.alpha = 1.0 - math.exp(-sim.dt_control / MOTOR_TAU)

    # state
    def state(self):
        d = self.sim.data
        p = d.qpos[self.qa:self.qa + 3].copy()
        v = d.qvel[self.va:self.va + 3].copy()
        w = d.qvel[self.va + 3:self.va + 6].copy()            # body frame angular velocity (free joint)
        R = d.xmat[self.b].reshape(3, 3).copy()
        return p, v, R, w

    def control(self, sim, t, dt):
        c, d = self.c, sim.data
        p, v, R, w = self.state()
        com = d.xipos[self.b]
        pad_p, pad_v = sim.carrier_point(self.pad_local)
        hover = c["mass"] * G / 4
        if self.locked and not self.landed:
            cmd = np.full(4, hover * 0.75 * min(1.0, t / max(self.mission.t_rel, 1e-3)))
            if t >= self.mission.t_rel:
                d.eq_active[sim.eq("gannet_lock")] = 0
                self.locked = False
                self.ctrl.reset()
        elif self.landed:
            cmd = np.zeros(4)
        else:
            p_ref, v_ref, a_ref, yaw = self.mission.ref(t, pad_p, pad_v)
            air = sim.wind.at(p[2]) - v
            drag = 0.5 * RHO_AIR * c["cda"] * np.linalg.norm(air) * air
            # the reference is for the skids' centre: the controller flies the body origin (the skids)
            cmd, _ = self.ctrl.thrusts(p, v, R, w, p_ref, v_ref, a_ref, yaw, dt, drag_ff=drag)
            if t > self.mission.t_track and sim.contact_between(self.geoms, self.pad_geom):
                self.touch = {"t": t, "v_rel": (v - pad_v).tolist(), "offset": (p - pad_p).tolist()}
                sim.relock("gannet_lock")
                self.landed = True
                cmd = np.zeros(4)
        self.T += (cmd - self.T) * self.alpha
        # the wrench at the COM: thrusts along body z at the rotors, reaction torques, drag
        zb = R[:, 2]
        F = zb * self.T.sum()
        tau = np.zeros(3)
        for (rx, ry), s, Ti in zip(c["rotors"], c["spin"], self.T):
            r = R @ np.array([rx, ry, c["rotor_z"]]) + d.xpos[self.b] - com
            tau += np.cross(r, zb * Ti) + zb * s * K_Q * Ti
        air = sim.wind.at(p[2]) - v
        F = F + 0.5 * RHO_AIR * c["cda"] * np.linalg.norm(air) * air
        d.xfrc_applied[self.b, :3] += F
        d.xfrc_applied[self.b, 3:] += tau
        self.log.append((t, *p, *v, float(self.T.sum())))

    def power_W(self):
        c = self.c
        fm, eta = c["p"]["fm"], c["p"]["eta"]
        return float(sum(max(T, 0.0) ** 1.5 / math.sqrt(2 * RHO_AIR * c["disc"]) for T in self.T) / (fm * eta) + c["p"]["avionics_w"])

    def hud(self, sim) -> list:
        p, v, R, w = self.state()
        pad_p, _ = sim.carrier_point(self.pad_local)
        air = np.linalg.norm(v - sim.wind.at(p[2]))
        return [f"Gannet  {self.c['mass']:.1f} kg{' (5 kg crate)' if self.crate else ''}   alt {p[2]:5.1f} m ASL  ({p[2] - pad_p[2]:5.1f} m over the pad)   "
                f"ground {np.linalg.norm(v[:2]):4.1f} m/s  climb {v[2]:+4.1f} m/s  air {air:4.1f} m/s",
                f"thrust {self.T.sum():4.0f} N of {4 * self.c['t_max']:.0f}   power {self.power_W() / 1000:4.2f} kW   "
                f"phase: {self.mission.phase(sim.t, self.landed)}"]

    def focus(self, sim):
        return self.sim.data.xipos[self.b].copy()
