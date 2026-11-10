"""Walrus in MuJoCo: the 35 kg clam crawler (``labs/marine/designs/walrus.py``) launched by Pinna's aft port davit
robot: hoisted off its deck chocks, swung over the side by the luffing boom, lowered through the splash zone on the
winch, released under water, and its eight ducted thrusters (layout C: four vectored 140 mm, four vertical 120 mm)
taking over to dive away from the hull towards the bed.

* **the body**: one free body with the card's vehicle mass (the tether stays on the reel), its volume from the card's
  sink weight (``sizing()['sink_kg']``: it is 2.9 kg heavy in sea water, so it sits on the bed), the centre of
  buoyancy above the centre of gravity (the foam on top): it rights itself. Drawn with the CAD meshes, colliding as
  a box (the tracks' envelope).
* **the cable**: a spatial tendon from the boom's sheave to the lifting eye with a length limit (a rope: it pulls,
  never pushes) that the winch sets every step (``Winch``); the boom's luffing angle is prescribed (the robot's
  trajectory). The cable's pull is MuJoCo's tendon-limit force.
* **the water**: buoyancy rho g V times the submerged fraction of the hull's height under the (heaving) surface, at
  the centre of buoyancy; quadratic drag per body axis (C_D A from the envelope's faces) and rotational damping,
  both scaled by the submerged fraction; a 0.2 m/s tidal current. No added mass and no wave slamming (assumed away).
* **the thrusters**: ``walrus.thruster_layout`` and ``walrus.allocation`` - the card's positions, axes and maximum
  thrust (the disc-loading sizing) - reversible, each a first-order lag, working only when wet.
* **the controller**: a PD on position (the depth profile of the dive, a stand-off from the hull) and on attitude
  (level, heading outboard) gives the body wrench, the card's allocation matrix (pseudo-inverse) the eight thrusts,
  clipped to the card's maximum.
"""
from __future__ import annotations

import math

import numpy as np

from . import world
from .world import G, RHO_SW, min_jerk

THRUSTER_TAU = 0.15       # s
CURRENT = np.array([-0.2, 0.0, 0.0])
DAVIT = (-11.5, 1)        # the aft port davit
BOOM = 1.85               # m, pivot to sheave
COM = np.array([0.02, 0.0, 0.17])
COB = np.array([0.0, 0.0, 0.22])
EYE = np.array([0.02, 0.0, 0.45])


def card() -> dict:
    from walrus import Walrus, allocation, thruster_layout
    p, s = Walrus.p(), Walrus.sizing()
    m = s["mass_wet_vehicle_kg"]
    vol = (m - s["sink_kg"]) / RHO_SW
    L, W, H = p["length"] / 1000, p["width"] / 1000, p["height"] / 1000
    th = thruster_layout(p)
    B = np.array(allocation(p))
    return {"p": p, "s": s, "mass": m, "volume": vol, "L": L, "W": W, "H": H, "thrusters": th, "B": B,
            "Bpinv": np.linalg.pinv(B), "t_max": np.array([t["max_N"] for t in th]),
            "J": np.array([m / 12 * (W * W + H * H), m / 12 * (L * L + H * H), m / 12 * (L * L + W * W)]),
            "cda": np.array([1.0 * W * H, 1.1 * L * H, 1.2 * L * W])}


class WalrusAgent:
    name = "walrus"

    def __init__(self, t_hoist=(0.5, 2.5), t_swing=(2.5, 7.0), t_lower=7.5, pay_out=0.7, release_depth=0.35, dive_to=-6.0):
        self.c = card()
        g = world.geometry()
        self.zd = g["z_deck"]
        x, side = DAVIT
        self.pivot = np.array([x, side * (g["beam"] / 2 - 0.45), self.zd + 2.6])
        self.phi_in, self.phi_out = math.radians(-50), math.radians(50)
        tip_in = self.pivot + BOOM * np.array([0, math.sin(self.phi_in), math.cos(self.phi_in)])
        self.start = np.array([x, tip_in[1], self.zd + 0.103])            # on its chocks (0.1 m)
        self.L0 = float(tip_in[2] - (self.start[2] + EYE[2])) + 0.02
        self.L_hoist = self.L0 - 0.46
        self.t_hoist, self.t_swing, self.t_lower, self.pay_out = t_hoist, t_swing, t_lower, pay_out
        self.release_depth, self.dive_to = release_depth, dive_to
        self.released = None
        self.T = np.zeros(len(self.c["thrusters"]))
        self.L = self.L0
        self.tension = 0.0
        self.sub = 0.0
        self.log = []

    # the davit robot's trajectories
    def luff(self, t):
        if self.released is not None:                       # after the release the boom comes back inboard
            q, v, _ = min_jerk(t, self.released + 1.0, self.released + 6.0, [self.phi_out], [self.phi_in])
            return float(q[0]), float(v[0])
        q, v, _ = min_jerk(t, self.t_swing[0], self.t_swing[1], [self.phi_in], [self.phi_out])
        return float(q[0]), float(v[0])

    def winch(self, t):
        if t < self.t_hoist[1]:
            return float(min_jerk(t, *self.t_hoist, [self.L0], [self.L_hoist])[0][0])
        if t < self.t_lower:
            return self.L_hoist
        ramp = min(1.0, (t - self.t_lower) / 0.8)
        return self.L_hoist + self.pay_out * max(0.0, (t - self.t_lower) - 0.4 * ramp)

    def xml(self, use_meshes):
        c = self.c
        col = __import__("pinna_launch.meshes", fromlist=["colours"]).colours()
        vis = ""
        if use_meshes:
            for part in ("tracks", "frame", "housing", "gantry", "basket", "thrusters", "lights", "arms"):
                vis += world._vis_mesh(f"walrus_{part}", col[f"walrus_{part}"])
        else:
            vis = '<geom class="visual" type="box" size="0.45 0.35 0.19" pos="0 0 0.19" rgba="0.85 0.85 0.8 1"/>'
        J = c["J"]
        q = (math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4))       # facing outboard (+y)
        body = f"""<body name="walrus" pos="{world._f(self.start)}" quat="{world._f(q)}">
<freejoint name="walrus_free"/>
<inertial pos="{world._f(COM)}" mass="{c['mass']:.3f}" diaginertia="{world._f(J)}"/>
{vis}
<geom name="w_hull" class="drone" type="box" size="{c['L'] / 2:.3f} {c['W'] / 2:.3f} {c['H'] / 2:.3f}" pos="0 0 {c['H'] / 2:.3f}"/>
<site name="walrus_eye" pos="{world._f(EYE)}" size="0.03"/>
<geom class="visual" type="cylinder" size="0.012 0.04" pos="{world._f(EYE - [0, 0, 0.03])}" rgba="0.8 0.8 0.1 1"/>
</body>"""
        chocks = "".join(f'<geom class="hullvis" type="box" size="0.4 0.06 0.05" pos="{self.start[0]} {self.start[1] + dy:.3f} {self.zd + 0.05:.3f}" rgba="0.3 0.3 0.3 1"/>'
                         for dy in (-0.3, 0.3))
        tendon = (f'<spatial name="walrus_cable" limited="true" range="0 {self.L0:.4f}" width="0.012" rgba="0.95 0.95 0.9 1" solreflimit="0.01 1" margin="0.02">'
                  f'<site site="boom_tip"/><site site="walrus_eye"/></spatial>')
        return {"world": body, "carrier": chocks, "tendon": tendon}

    def bind(self, sim):
        self.sim = sim
        m = sim.model
        self.b = sim.body("walrus")
        j = sim.joint("walrus_free")
        self.qa, self.va = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
        self.ten = sim.tendon("walrus_cable")
        sim.prescribe("davit_luff", self.luff)
        self.alpha = 1.0 - math.exp(-sim.dt_control / THRUSTER_TAU)
        self.at = np.array([t["at"] for t in self.c["thrusters"]])
        self.ax = np.array([t["axis"] for t in self.c["thrusters"]])

    def state(self):
        d = self.sim.data
        R = d.xmat[self.b].reshape(3, 3).copy()
        com = d.xipos[self.b].copy()
        v = d.qvel[self.va:self.va + 3].copy()            # the free joint's linear velocity is of the body origin
        w = d.qvel[self.va + 3:self.va + 6].copy()        # body frame
        v_com = v + R @ np.cross(w, COM)
        return com, v_com, R, w

    def _cable_tension(self, sim):
        d, m = sim.data, sim.model
        import mujoco
        T = 0.0
        for i in range(d.nefc):
            if d.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON and d.efc_id[i] == self.ten:
                T += abs(float(d.efc_force[i]))
        return T

    def control(self, sim, t, dt):
        c, d, m = self.c, sim.data, sim.model
        com, v, R, w = self.state()
        eta = sim.sea.eta(t)
        # the winch and the hook
        if self.released is None:
            self.L = self.winch(t)
            m.tendon_range[self.ten, 1] = self.L
            self.tension += (self._cable_tension(sim) - self.tension) * (1 - math.exp(-dt / 0.3))     # the pull, low-passed
            top = com[2] - COM[2] + c["H"]
            if t > self.t_lower and eta - top > self.release_depth:
                self.released = t
                m.tendon_limited[self.ten] = 0
                m.tendon_rgba[self.ten, 3] = 0.0
                self.t0 = t
                self.p0 = com.copy()
                self.tension = 0.0
        # the water: buoyancy, drag, current
        z_bot = com[2] - COM[2]
        self.sub = sub = float(np.clip((eta - z_bot) / c["H"], 0.0, 1.0))
        F, tau = np.zeros(3), np.zeros(3)
        if sub > 0:
            B = RHO_SW * G * c["volume"] * sub
            r_b = R @ (COB - COM)
            F += np.array([0, 0, B])
            tau += np.cross(r_b, np.array([0, 0, B]))
            vb = R.T @ (v - CURRENT)
            Fd = -0.5 * RHO_SW * c["cda"] * np.abs(vb) * vb - 8.0 * vb
            F += R @ (Fd * sub)
            tau += R @ (-(3.0 + 25.0 * np.abs(w)) * w * sub)
        else:
            tau += R @ (-4.0 * w)                                 # the cradle's guides on the boom head and the air
        # the thrusters (after the release)
        cmd = np.zeros_like(self.T)
        if self.released is not None:
            cmd = self._allocate(t, com, v, R, w, eta)
        self.T += (cmd - self.T) * self.alpha
        for k in range(len(self.T)):
            pk = R @ (self.at[k] - COM)
            wet = (com[2] + pk[2]) < eta - 0.03
            if not wet:
                continue
            fk = R @ self.ax[k] * self.T[k]
            F += fk
            tau += np.cross(pk, fk)
        d.xfrc_applied[self.b, :3] += F
        d.xfrc_applied[self.b, 3:] += tau
        self.log.append((t, *com, *v, self.tension, sub, self._cable_tension(sim) if self.released is None else 0.0))

    def _allocate(self, t, com, v, R, w, eta):
        c = self.c
        m = c["mass"]
        tr = t - self.t0
        # the reference: hold 1 s, then down at ~0.35 m/s to dive_to, stepping 2.5 m out from the hull
        z1 = self.dive_to
        T_dive = max(4.0, abs(z1 - self.p0[2]) / 0.42)
        p_ref, v_ref, a_ref = min_jerk(tr, 1.0, 1.0 + T_dive, self.p0, self.p0 + np.array([0.0, 3.0, z1 - self.p0[2]]))
        Fw = m * (0.6 * (p_ref - com) + 1.6 * (v_ref - v) + a_ref)
        Fw[2] += m * G - RHO_SW * G * c["volume"] * self.sub
        Fb = R.T @ Fw
        yaw = math.atan2(R[1, 0], R[0, 0])
        Rd = np.array([[math.cos(math.pi / 2), -math.sin(math.pi / 2), 0], [math.sin(math.pi / 2), math.cos(math.pi / 2), 0], [0, 0, 1]])
        eR = 0.5 * np.array([(Rd.T @ R - R.T @ Rd)[2, 1], (Rd.T @ R - R.T @ Rd)[0, 2], (Rd.T @ R - R.T @ Rd)[1, 0]])
        J = c["J"]
        Mb = -J * 9.0 * eR * 3 - J * 2 * 0.9 * 3.0 * w * 2
        wrench = np.concatenate([Fb, Mb + np.cross(COM - np.array([0, 0, c["H"] / 2]), Fb)])
        u = c["Bpinv"] @ wrench
        k = np.max(np.abs(u) / c["t_max"])
        if k > 1:
            u /= k                                                # scale the whole vector: the direction is kept
        self.ref = p_ref
        return u

    def depth(self):
        com, v, R, w = self.state()
        return self.sim.sea.eta(self.sim.t) - com[2]

    def phase(self, t):
        if t < self.t_hoist[0]:
            return "on its chocks, hooked to the davit"
        if t < self.t_hoist[1]:
            return "davit robot: hoist"
        if t < self.t_swing[1]:
            return "davit robot: boom luffs outboard over the side"
        if self.released is None:
            return "winch pays out: through the splash zone" if self.sub > 0 else "winch pays out: lowering"
        if t < self.released + 1.0:
            return "hook released: thrusters take over"
        return "diving under its 8 thrusters, away from the hull"

    def hud(self, sim):
        com, v, R, w = self.state()
        t = sim.t
        dep = self.depth()
        thr = f"thrusters {np.abs(self.T).sum():4.0f} N total (max {self.c['t_max'].sum():.0f})" if self.released is not None else "thrusters off"
        return [f"Walrus  {self.c['mass']:.1f} kg, {self.c['s']['sink_kg']:+.1f} kg in sea water   depth {dep:+5.2f} m   speed {np.linalg.norm(v):4.2f} m/s   "
                f"submerged {100 * self.sub:3.0f} %",
                f"cable {self.L:4.2f} m, pull {self.tension:4.0f} N   {thr}   phase: {self.phase(t)}"]

    def focus(self, sim):
        return self.sim.data.xipos[self.b].copy()
