"""Fulmar in MuJoCo: the 20.2 kg, 3.6 m cargo fixed-wing (``labs/aero/designs/fulmar.py`` on ``fixed_wing.FixedWing``)
shot off Pinna's pneumatic catapult and, in the recovery clip, flown into the net at the flight deck's aft edge.

* **the airframe**: one free body with the card's mass and an estimated inertia (the wing's mass spread over the
  span, the pack and the crate in the pod), drawn with the CAD mesh (the FixedWing airframe and the cargo pod; its
  frame is x aft from the wing root's leading edge, turned here to x forward with the centre of gravity at 28 % of
  the root chord), colliding as a fuselage capsule, the wing and the tailplane as thin boxes.
* **the aerodynamics** (``Aero``): the card's numbers - wing area, aspect ratio, CD0 0.032, Oswald 0.8, CL max 1.35 -
  with a linear lift slope (Helmbold, finite wing) blended past the stall into a flat plate (the Beard-McLain
  sigmoid), induced drag, and stability derivatives of class values for a conventional high-aspect-ratio UAV
  (assumed: Cm_alpha -0.9, Cm_q -15, Cl_p -0.5, Cn_beta 0.08, ...) with elevator, aileron and rudder; the thrust of
  the two motors falls linearly with airspeed from its static value (``T_STATIC``, assumed). All forces on the CG
  from the relative wind (``world.Wind``).
* **the catapult**: the shuttle on the rail (a slide joint of the carrier) runs a prescribed constant-acceleration
  stroke of 2.8 m to the card's launch speed (1.25 V_stall, 18 m/s: ~5.9 g), then brakes; the Fulmar sits on it
  held by a weld (the cradle) that opens at the end of the stroke.
* **the autopilot** (``Autopilot``): pitch attitude with rate damping and gain scheduling on dynamic pressure, the
  flight-path angle from an altitude error (climb-out at 10 deg), roll from a heading / cross-track law with a
  coordinated-turn feed-forward, a yaw damper, throttle on airspeed.
* **the net** (``FulmarNetAgent``): the approach from 130 m astern at 1.15 V_stall airspeed into the wind, the glide
  path aimed at the net's centre as it moves with the deck; the net-capture hook engages on contact (a weld to the
  net panel) and the panel's sprung slide (``world.carrier_xml``) absorbs the closing speed.
"""
from __future__ import annotations

import math

import numpy as np

from . import world
from .world import G, RHO_AIR, euler_zyx

T_STATIC = 120.0          # N, two motors at full throttle, static (assumed: thrust/weight 0.6)
V_ZERO_THRUST = 45.0      # m/s where the thrust falls to zero (assumed)
X_CG = 0.112              # m aft of the wing root's leading edge (28 % of the root chord)
Z_CG = -0.05              # m (the pod and the pack low)
DERIV = {"cm0": 0.03, "cm_a": -0.9, "cm_q": -15.0, "cm_de": -1.3, "cl_de": 0.35, "cy_b": -0.35, "cy_dr": 0.12,
         "cl_b": 0.07, "cl_p": -0.5, "cl_r": 0.1, "cl_da": 0.25, "cn_b": 0.08, "cn_r": -0.12, "cn_dr": 0.06, "cn_da": -0.01}


def card() -> dict:
    from fulmar import Fulmar
    p, s = Fulmar.p(), Fulmar.sizing()
    S, b = s["wing_area_m2"], p["span"] / 1000
    AR = b * b / S
    c = S / b
    cla = 2 * math.pi * AR / (2 + math.sqrt(AR * AR + 4))       # Helmbold
    cl0 = cla * math.radians(4.0)                                 # 4 % camber: zero lift at about -4 deg
    return {"p": p, "s": s, "mass": s["mtow_kg"], "S": S, "b": b, "c": c, "AR": AR, "cla": cla, "cl0": cl0,
            "cd0": p["cd0"], "e": p["oswald"], "cl_max": p["cl_max"], "alpha_stall": (p["cl_max"] - cl0) / cla,
            "v_stall": s["stall_m_s"], "v_launch": s["launch_m_s"], "stroke": s["catapult_stroke_m"],
            "J": np.array([4.6, 2.3, 6.6])}


class Aero:
    """Forces and moments on the CG (body frame: x forward, y left, z up) from the relative air and the controls."""

    def __init__(self, c):
        self.c = c

    def coeffs(self, alpha):
        c = self.c
        M, a0 = 120.0, c["alpha_stall"] + 0.03          # a sharp blend: CL peaks near the card's CL max at the stall angle
        sig = (1 + math.exp(-M * (alpha - a0)) + math.exp(M * (alpha + a0))) / ((1 + math.exp(-M * (alpha - a0))) * (1 + math.exp(M * (alpha + a0))))
        cl_lin = c["cl0"] + c["cla"] * alpha
        cl = (1 - sig) * cl_lin + sig * 2 * math.copysign(1, alpha) * math.sin(alpha) ** 2 * math.cos(alpha)
        cd = (1 - sig) * (c["cd0"] + cl_lin ** 2 / (math.pi * c["e"] * c["AR"])) + sig * (c["cd0"] + 2 * math.sin(alpha) ** 2)
        return cl, cd, sig

    def wrench(self, vb, wb, de, da, dr):
        """``vb`` the air-relative velocity of the CG in body axes, ``wb`` the body rates; deflections in rad
        (elevator trailing edge down +, aileron roll right +, rudder nose left +). Returns (F, M, info)."""
        c, D = self.c, DERIV
        V = float(np.linalg.norm(vb))
        if V < 0.5:
            return np.zeros(3), np.zeros(3), {"V": V, "alpha": 0.0, "beta": 0.0, "cl": 0.0}
        u, v, w = vb
        alpha = math.atan2(-w, u)
        beta = math.asin(max(-1.0, min(1.0, v / V)))
        q = 0.5 * RHO_AIR * V * V
        S, b, ch = c["S"], c["b"], c["c"]
        cl, cd, sig = self.coeffs(alpha)
        p, qb, r = wb
        q_conv = -qb                                                # body y is to port: nose-up rate is -wb[1]
        ph, qh, rh = p * b / (2 * V), q_conv * ch / (2 * V), r * b / (2 * V)
        L, Dg = q * S * cl, q * S * cd
        lift_dir = np.array([math.sin(alpha), 0.0, math.cos(alpha)])
        F = L * lift_dir - Dg * vb / V
        F[1] += q * S * (D["cy_b"] * beta + D["cy_dr"] * dr)
        cm = D["cm0"] + D["cm_a"] * alpha + D["cm_q"] * qh + D["cm_de"] * de
        cm *= (1 - sig) + sig * 0.3
        Mx = q * S * b * (D["cl_b"] * beta + D["cl_p"] * ph + D["cl_r"] * rh + D["cl_da"] * da)
        My = -q * S * ch * cm
        Mz = q * S * b * (D["cn_b"] * beta + D["cn_r"] * rh + D["cn_dr"] * dr + D["cn_da"] * da)
        return F, np.array([Mx, My, Mz]), {"V": V, "alpha": alpha, "beta": beta, "cl": cl, "q": q}


def thrust(throttle, V):
    return T_STATIC * max(0.0, min(1.0, throttle)) * max(0.0, 1.0 - V / V_ZERO_THRUST)


class Autopilot:
    """Classic loops: pitch (and its rate) by the elevator, flight path from altitude, roll from heading, yaw damper,
    throttle on airspeed. Gains scheduled on the dynamic pressure about ``V_REF``."""

    V_REF = 22.0

    def __init__(self, c):
        self.c = c
        self.i_theta = 0.0
        self.i_v = 0.0

    def __call__(self, st, cmd, dt):
        """``st``: theta, phi, psi, p, q_conv, r (body), V air, alpha, beta, z, vz; ``cmd``: alt, psi, v (or None
        with gamma/throttle given)."""
        k = min(2.5, (self.V_REF / max(st["V"], 8.0)) ** 2)
        if "gamma" in cmd:
            gamma = cmd["gamma"]
        else:
            gamma = max(math.radians(-8), min(math.radians(cmd.get("gamma_max", 10.0)), 0.06 * (cmd["alt"] - st["z"]) - 0.0 * st["vz"]))
        # the angle of attack the lift needs at this airspeed (the lift equation, not the measured one: feeding the
        # measured AoA back into the pitch command runs away when the aircraft slows), capped below the stall, and a
        # stall guard on the measured AoA
        c = self.c
        cl_req = c["mass"] * G * math.cos(st["phi"]) ** -1 / (0.5 * RHO_AIR * max(st["V"], 8.0) ** 2 * c["S"])
        a_req = max(0.0, min(math.radians(9.0), (cl_req - c["cl0"]) / c["cla"]))
        theta_c = gamma + a_req - 2.0 * max(0.0, st["alpha"] - math.radians(10.0))
        e = theta_c - st["theta"]
        self.i_theta = max(-0.15, min(0.15, self.i_theta + e * dt))
        de = k * (-0.9 * e - 0.6 * self.i_theta + 0.12 * st["q_conv"]) - 0.02
        de = max(-0.35, min(0.35, de))
        dpsi = (cmd["psi"] - st["psi"] + math.pi) % (2 * math.pi) - math.pi
        phi_c = max(-math.radians(30), min(math.radians(30), -1.2 * dpsi - cmd.get("phi_ff", 0.0)))
        da = k * (0.8 * (phi_c - st["phi"]) - 0.12 * st["p"])
        da = max(-0.3, min(0.3, da))
        dr = max(-0.3, min(0.3, k * (0.6 * st["beta"] - 0.15 * st["r"])))
        if "throttle" in cmd:
            th = cmd["throttle"]
        else:
            ev = cmd["v"] - st["V"]
            self.i_v = max(-0.5, min(0.5, self.i_v + ev * dt))
            th = 0.3 + 0.25 * ev + 0.15 * self.i_v
        return de, da, dr, max(0.0, min(1.0, th))


class _FulmarBase:
    """The airframe, its aerodynamics and autopilot as a MuJoCo free body (shared by the launch and the net)."""

    name = "fulmar"

    def __init__(self):
        self.c = card()
        self.aero = Aero(self.c)
        self.ap = Autopilot(self.c)
        self.u = (0.0, 0.0, 0.0, 0.0)
        self.info = {"V": 0.0, "alpha": 0.0, "beta": 0.0, "cl": 0.0}
        self.log = []

    def body_xml(self, use_meshes, pos, quat, extra=""):
        c = self.c
        col = __import__("pinna_launch.meshes", fromlist=["colours"]).colours()
        if use_meshes:      # the CAD frame (x aft, z up, LE at 0) turned 180 deg about z, the CG at the body origin
            vis = world._vis_mesh("fulmar_aircraft", col["fulmar_aircraft"], pos=(X_CG, 0.0, -Z_CG), quat=(0, 0, 0, 1))
        else:
            vis = '<geom class="visual" type="box" size="1 0.1 0.1" rgba="0.95 0.6 0.07 1"/><geom class="visual" type="box" size="0.18 1.8 0.02" rgba="0.95 0.6 0.07 1"/>'
        J = c["J"]
        return f"""<body name="fulmar" pos="{_f(pos)}" quat="{_f(quat)}">
<freejoint name="fulmar_free"/>
<inertial pos="0 0 0" mass="{c['mass']:.3f}" diaginertia="{J[0]} {J[1]} {J[2]}"/>
{vis}
<geom name="f_fuse" class="drone" type="capsule" size="0.1" fromto="{X_CG + 0.55:.3f} 0 {-Z_CG:.3f} {X_CG - 1.4:.3f} 0 {-Z_CG:.3f}"/>
<geom name="f_pod" class="drone" type="box" size="0.26 0.15 0.1" pos="{X_CG - 0.12:.3f} 0 {-Z_CG - 0.29:.3f}"/>
<geom name="f_wing" class="drone" type="box" size="0.16 1.8 0.03" pos="{X_CG - 0.16:.3f} 0 {-Z_CG + 0.03:.3f}"/>
<geom name="f_tail" class="drone" type="box" size="0.14 0.55 0.02" pos="{X_CG - 1.36:.3f} 0 {-Z_CG + 0.05:.3f}"/>
<geom name="f_hook" class="drone" type="sphere" size="0.06" pos="{X_CG + 0.62:.3f} 0 {-Z_CG + 0.1:.3f}"/>
{extra}
</body>"""

    def bind(self, sim):
        self.sim = sim
        self.b = sim.body("fulmar")
        j = sim.joint("fulmar_free")
        self.qa, self.va = int(sim.model.jnt_qposadr[j]), int(sim.model.jnt_dofadr[j])
        self.geoms = sim.body_geoms(self.b)

    def state(self):
        d = self.sim.data
        p = d.qpos[self.qa:self.qa + 3].copy()
        v = d.qvel[self.va:self.va + 3].copy()
        w = d.qvel[self.va + 3:self.va + 6].copy()
        R = d.xmat[self.b].reshape(3, 3).copy()
        return p, v, R, w

    def flight_state(self, sim):
        p, v, R, w = self.state()
        air = sim.wind.at(p[2])
        vb = R.T @ (v - air)
        roll, pitch, yaw = euler_zyx(R)
        V = float(np.linalg.norm(vb))
        alpha = math.atan2(-vb[2], vb[0]) if V > 0.5 else 0.0
        beta = math.asin(max(-1, min(1, vb[1] / V))) if V > 0.5 else 0.0
        return {"theta": pitch, "phi": roll, "psi": yaw, "p": w[0], "q_conv": -w[1], "r": w[2], "V": V, "alpha": alpha,
                "beta": beta, "z": p[2], "vz": v[2], "x": p[0], "y": p[1]}, (p, v, R, w, vb)

    def apply(self, sim, de, da, dr, th):
        p, v, R, w = self.state()
        vb = R.T @ (v - sim.wind.at(p[2]))
        F, M, info = self.aero.wrench(vb, w, de, da, dr)
        T = thrust(th, info["V"])
        F = F + np.array([T, 0.0, 0.0])
        sim.data.xfrc_applied[self.b, :3] += R @ F
        sim.data.xfrc_applied[self.b, 3:] += R @ M
        self.u, self.info = (de, da, dr, th), info | {"T": T}
        self.log.append((sim.t, *p, *v, info["V"], info["alpha"]))

    def hud_common(self, sim):
        p, v, R, w = self.state()
        roll, pitch, yaw = euler_zyx(R)
        i = self.info
        return (f"Fulmar  {self.c['mass']:.1f} kg, 3.6 m   alt {p[2]:5.1f} m ASL   ground {np.linalg.norm(v):4.1f} m/s   "
                f"air {i['V']:4.1f} m/s (stall {self.c['v_stall']:.1f})   climb {v[2]:+4.1f} m/s",
                f"pitch {math.degrees(pitch):+5.1f} deg  AoA {math.degrees(i['alpha']):+5.1f} deg  CL {i['cl']:4.2f}   thrust {i.get('T', 0):4.0f} N   "
                f"elev {math.degrees(self.u[0]):+4.1f}  ail {math.degrees(self.u[1]):+4.1f} deg")

    def focus(self, sim):
        return self.sim.data.xipos[self.b].copy()


def _f(v):
    return world._f(v)


class FulmarLaunchAgent(_FulmarBase):
    """On the catapult's shuttle (held by the cradle's weld), the stroke, the release, the climb-out to 60 m."""

    def __init__(self, t_fire: float = 2.0, alt: float = 60.0):
        super().__init__()
        self.t_fire = t_fire
        self.alt = alt
        c = self.c
        self.a_cat = c["v_launch"] ** 2 / (2 * c["stroke"])
        self.t_stroke = c["v_launch"] / self.a_cat
        self.released = False
        self.t_release = None
        self.peak_g = 0.0
        self._v_prev = None
        self.pitch0 = math.radians(3.0)                         # the cradle holds it 3 deg nose up

    def _cradle_pose(self):
        g = world.geometry()
        yaw = math.radians(world.CATAPULT_YAW_DEG)
        x0 = g["rail_x"][0]
        s = 0.4 + 0.15                                         # along the rail from its aft end: the shuttle + a bit
        pos = np.array([x0 + s * math.cos(yaw), s * math.sin(yaw), g["rail_top"] + 0.08 + 0.29 + 0.1 + 0.05])
        # the CG ahead of the shuttle's centre by 0.1 m
        pos[:2] += 0.1 * np.array([math.cos(yaw), math.sin(yaw)])
        qz = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
        qp = np.array([math.cos(-self.pitch0 / 2), 0, math.sin(-self.pitch0 / 2), 0])     # nose up: negative about +y
        q = np.zeros(4)
        import mujoco
        mujoco.mju_mulQuat(q, qz, qp)
        return pos, q

    def xml(self, use_meshes):
        pos, q = self._cradle_pose()
        return {"world": self.body_xml(use_meshes, pos, q),
                "equality": '<weld name="fulmar_cradle" body1="shuttle" body2="fulmar" solref="0.002 1"/>',
                "contact": '<exclude body1="shuttle" body2="fulmar"/>'}

    def shuttle(self, t):
        """The shuttle's prescribed stroke: rest, constant acceleration over the stroke, a hard brake."""
        tf, a, ts = self.t_fire, self.a_cat, self.t_stroke
        if t < tf:
            return 0.0, 0.0
        if t < tf + ts:
            tau = t - tf
            return 0.5 * a * tau * tau, a * tau
        v0, s0 = a * ts, self.c["stroke"]
        ab = v0 ** 2 / (2 * 0.6)                               # stops in 0.6 m
        tau = min(t - tf - ts, v0 / ab)
        return s0 + v0 * tau - 0.5 * ab * tau * tau, v0 - ab * tau

    def bind(self, sim):
        super().bind(sim)
        sim.prescribe("shuttle_x", self.shuttle)

    def control(self, sim, t, dt):
        p, v, R, w = self.state()
        if self._v_prev is not None and t > self.t_fire - 0.1 and (self.t_release is None or t < self.t_release + 0.05):
            acc = np.linalg.norm(v - self._v_prev) / dt
            self.peak_g = max(self.peak_g, acc / G)
        self._v_prev = v
        if not self.released and t >= self.t_fire + self.t_stroke:
            sim.data.eq_active[sim.eq("fulmar_cradle")] = 0
            self.released, self.t_release = True, t
            self.v_release = float(np.linalg.norm(v))
        st, _ = self.flight_state(sim)
        yaw0 = math.radians(world.CATAPULT_YAW_DEG)
        if not self.released:
            th = 1.0 if t > self.t_fire - 0.5 else 0.3 * max(0.0, t - 0.5) / 1.0
            self.apply(sim, 0.0, 0.0, 0.0, min(th, 1.0))
            return
        # climb-out on the launch heading to 25 m, then turn to the shore heading (port, 30 deg) and climb on
        ts = t - self.t_release
        psi = yaw0 if ts < 4.0 else yaw0 + math.radians(30) * min(1.0, (ts - 4.0) / 3.0)
        cmd = {"alt": self.alt, "psi": psi, "throttle": 1.0, "gamma_max": 10.0 if ts > 0.6 else 6.0}
        de, da, dr, th = self.ap(st, cmd, dt)
        self.apply(sim, de, da, dr, th)

    def phase(self, t):
        if t < self.t_fire - 0.5:
            return "on the catapult: cradle locked, motors spinning up"
        if t < self.t_fire:
            return "full throttle, catapult armed"
        if not self.released:
            return f"catapult stroke: {self.a_cat / G:.1f} g over {self.c['stroke']:.2f} m"
        ts = t - self.t_release
        return "released: climb-out on the launch heading" if ts < 4.0 else "climb-out, turning for the shore"

    def hud(self, sim):
        a, b = self.hud_common(sim)
        s, _ = self.shuttle(sim.t)
        extra = f"   stroke {s:4.2f}/{self.c['stroke']:.2f} m" if not self.released or sim.t < self.t_release + 1.5 else ""
        rel = f"   release {self.v_release:4.1f} m/s, peak {self.peak_g:.1f} g" if self.released else ""
        return [a, b + extra, f"phase: {self.phase(sim.t)}{rel}"]


class FulmarNetAgent(_FulmarBase):
    """The net recovery: from 130 m astern at 1.15 V_stall airspeed, aimed at the net's centre; the hook catches."""

    def __init__(self, dist: float = 130.0, t_cut: float = None):
        super().__init__()
        g = world.geometry()
        self.dist = dist
        self.z_net = g["net_bottom"] + g["net_size"][1] / 2
        self.v_app = 1.15 * self.c["v_stall"]
        self.caught = None
        self.t_cut = t_cut
        self.net_max, self.peak_g_catch, self._hist = 0.0, 0.0, []

    def xml(self, use_meshes):
        g = world.geometry()
        z = self.z_net + 3.0
        pos = (g["net_x"] - self.dist, 0.4, z)
        cl = self.c["mass"] * G / (0.5 * RHO_AIR * self.v_app ** 2 * self.c["S"])
        alpha = (cl - self.c["cl0"]) / self.c["cla"]
        q = (math.cos(-alpha / 2), 0, math.sin(-alpha / 2), 0)
        return {"world": self.body_xml(use_meshes, pos, q),
                "equality": '<weld name="fulmar_hook" body1="net" body2="fulmar" active="false" solref="0.02 1"/>'}

    def bind(self, sim):
        super().bind(sim)
        self.net_geom = {sim.geom("net_panel")}
        d = sim.data
        wind = sim.wind.at(self.z_net + 3.0)
        # start in trimmed flight: the air speed v_app along +x through the wind
        d.qvel[self.va:self.va + 3] = np.array([self.v_app + wind[0], wind[1], 0.0])
        self.alpha_trim = None

    def control(self, sim, t, dt):
        st, (p, v, R, w, vb) = self.flight_state(sim)
        if self.caught is None and sim.contact_between(self.geoms, self.net_geom):
            net_v = sim.data.qvel[int(sim.model.jnt_dofadr[sim.joint("net_x")])]
            self.caught = {"t": t, "ground_speed": float(np.linalg.norm(v)), "closing": float(v[0]), "air": st["V"],
                           "z": float(p[2]), "y": float(p[1])}
            sim.relock("fulmar_hook")
            self._hist = [(t, v.copy())]
        if self.caught is not None:
            nx = float(sim.data.qpos[int(sim.model.jnt_qposadr[sim.joint("net_x")])])
            self.net_max = max(self.net_max, nx)
            self._hist.append((t, v.copy()))                    # the arrest's deceleration over 50 ms windows
            while self._hist and t - self._hist[0][0] > 0.05:
                t0, v0 = self._hist.pop(0)
                self.peak_g_catch = max(self.peak_g_catch, float(np.linalg.norm(v - v0)) / (t - t0) / G)
            self.u, self.info = (0.0, 0.0, 0.0, 0.0), self.info | {"T": 0.0}
            self.apply(sim, 0.0, 0.0, 0.0, 0.0)
            return
        # the glide path to the net's centre (as it moves): the flight-path angle to it, the heading to it
        net_p, _ = sim.carrier_point((world.geometry()["net_x"], 0.0, self.z_net))
        dx = net_p[0] - p[0]
        gamma = math.atan2(net_p[2] - p[2], max(dx, 5.0))
        psi = math.atan2(net_p[1] - p[1], max(dx, 5.0)) * 1.6
        cmd = {"gamma": max(math.radians(-6), min(math.radians(4), gamma + 0.5 * (gamma - math.atan2(v[2], max(v[0], 1.0))))),
               "psi": psi, "v": self.v_app}
        de, da, dr, th = self.ap(st, cmd, dt)
        self.apply(sim, de, da, dr, th)

    def hud(self, sim):
        a, b = self.hud_common(sim)
        g = world.geometry()
        net_p, _ = sim.carrier_point((g["net_x"], 0.0, self.z_net))
        p, v, R, w = self.state()
        if self.caught is None:
            ph = f"approach at 1.15 V_stall ({self.v_app:.1f} m/s air) into the wind, {net_p[0] - p[0] - 0.7:5.1f} m to the net"
        else:
            nx = sim.data.qpos[int(sim.model.jnt_qposadr[sim.joint("net_x")])]
            c = self.caught
            ph = f"caught: hook in the net at {c['closing']:.1f} m/s closing ({c['air']:.1f} m/s air); net travel {nx:4.2f} m (max {self.net_max:.2f})"
        return [a, b, "phase: " + ph]
