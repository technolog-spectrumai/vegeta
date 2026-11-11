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
COM = np.array([0.02, 0.0, 0.17])
COB = np.array([0.0, 0.0, 0.22])
EYE = np.array([0.02, 0.0, 0.45])


def _ramp(t, t0, t1):
    return 0.0 if t <= t0 else 1.0 if t >= t1 else (t - t0) / (t1 - t0)


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


class CompensatedDavit:
    """The aft port davit robot (``pinna.DAVIT_AHC``): slew, luff and the AHC winch, each a servo with the card's rate
    limit and lag behind its command. The nominal job: hoist, luff outboard, pay out. With ``compensated`` the MRU's
    measurement of the deck (the true motion ``latency_s`` late, plus a slowly wandering error of the card's accuracy)
    gives the deck's pose, the controller solves the slew and luff that put the sheave back on its earth-fixed point
    and pays the rope out or in by the sheave's remaining vertical error, so the hook holds its earth-frame path."""

    def __init__(self, agent, compensated: bool, seed: int = 3):
        from pinna import DAVIT_AHC
        self.a, self.ag, self.on = DAVIT_AHC, agent, compensated
        self.Lb = DAVIT_AHC["boom_m"]
        g = world.geometry()
        x, side = DAVIT
        self.pivot = np.array([x, side * (g["beam"] / 2 - 0.45), g["z_deck"] + DAVIT_AHC["post_m"]])
        self.phi_in, self.phi_out = math.radians(DAVIT_AHC["inboard_luff_deg"]), math.radians(DAVIT_AHC["working_luff_deg"])
        self.P = self.pivot + self.Lb * np.array([0.0, math.sin(self.phi_out), math.cos(self.phi_out)])   # the earth-fixed sheave point
        self.P_local = self.P.copy()                                                 # the same point in the ship's frame
        self.q = {"slew": 0.0, "luff": self.phi_in, "rope": 0.0}
        self.qd = {k: 0.0 for k in self.q}
        self.rng = np.random.default_rng(seed)
        self.err = np.zeros(3)
        self._hist = []
        m = DAVIT_AHC["mru"]
        self.sig = np.array([m["heave_accuracy_m"] / 2, math.radians(m["roll_pitch_accuracy_deg"]), math.radians(m["roll_pitch_accuracy_deg"])])
        self.k = 0.0
        self._cmd = {}

    def tip0(self, phi):
        return self.pivot + self.Lb * np.array([0.0, math.sin(phi), math.cos(phi)])

    def mru(self, sim, t, dt):
        """The deck's (heave, roll, pitch) as the MRU reports them: ``latency_s`` late, with a slow error."""
        a = math.exp(-dt / 2.0)
        self.err = a * self.err + math.sqrt(1 - a * a) * self.sig * self.rng.standard_normal(3)
        return sim.sea.state(max(0.0, t - self.a["mru"]["latency_s"]))[0] + self.err

    def update(self, sim, t, dt, L_nominal, phi_nominal, k):
        """New commands for the three axes, then each servo moves toward its command (rate limit, first-order lag)."""
        self.k = k
        cmd = {"slew": 0.0, "luff": phi_nominal, "rope": L_nominal}
        if k > 0:
            pos, R = sim.sea.pose(t, self.mru(sim, t, dt))
            if sim.track:                                                   # the DP position (from the GNSS, as the MRU's)
                pos[:2] += sim.track(t)[:2]
            qs = R.T @ (self.P - pos)                                       # where the sheave must be, in the ship's frame
            d = qs - self.pivot
            psi = math.atan2(-d[0], d[1])
            rho = math.hypot(d[0], d[1])
            lo, hi = (math.radians(v) for v in self.a["luff"]["range_deg"])
            phi = min(max(math.asin(min(rho / self.Lb, 1.0)), lo), hi)
            smax = math.radians(self.a["slew"]["range_deg"][1])
            psi = min(max(psi, -smax), smax)
            u = np.array([-math.sin(psi) * math.sin(phi), math.cos(psi) * math.sin(phi), math.cos(phi)])
            ez = (pos + R @ (self.pivot + self.Lb * u))[2] - self.P[2]       # what the winch takes up
            cmd = {"slew": k * psi, "luff": k * phi + (1 - k) * phi_nominal, "rope": L_nominal + k * ez}
        lag = self.a["servo_lag_s"]
        rate = {"slew": math.radians(self.a["slew"]["rate_deg_s"]), "luff": math.radians(self.a["luff"]["rate_deg_s"]),
                "rope": self.a["winch"]["rope_speed_m_s"]}
        if self.q["rope"] == 0.0:
            self.q["rope"] = cmd["rope"]
        acc = {"slew": 4 * rate["slew"], "luff": 4 * rate["luff"], "rope": self.a["winch"]["rope_accel_m_s2"]}
        for key in self.q:                                                  # a servo: the command's rate fed forward,
            ff = (cmd[key] - self._cmd.get(key, cmd[key])) / dt             # the error closed with the lag; rate and
            v = ff + (cmd[key] - self.q[key]) / lag                         # acceleration limits
            v = max(-rate[key], min(rate[key], v))
            v = max(self.qd[key] - acc[key] * dt, min(self.qd[key] + acc[key] * dt, v))
            self.qd[key] = v
            self.q[key] += v * dt
        self._cmd = cmd

    def joint(self, key):
        return lambda t: (self.q[key], self.qd[key])


class WalrusAgent:
    name = "walrus"
    bname = "walrus"
    COM, COB, EYE = COM, COB, EYE
    QUAT = (math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4))         # facing outboard (+y)
    card = staticmethod(card)

    def __init__(self, t_hoist=(0.5, 3.2), t_swing=(3.2, 7.2), t_lower=8.5, pay_out=0.8, release_depth=0.35, dive_to=-6.0,
                 compensated: bool = True):
        self.c = self.card()
        g = world.geometry()
        self.zd = g["z_deck"]
        x, side = DAVIT
        self.davit = CompensatedDavit(self, compensated)
        self.compensated = compensated
        tip_in = self.davit.tip0(self.davit.phi_in)
        self.start = np.array([x, tip_in[1], self.zd + 0.103])            # on its chocks (0.1 m)
        self.L0 = float(tip_in[2] - (self.start[2] + self.EYE[2])) + 0.02
        self.L_hoist = 0.62                                                 # up under the boom head's latch cradle
        self.latched = False
        self.t_hoist, self.t_swing, self.t_lower, self.pay_out = t_hoist, t_swing, t_lower, pay_out
        self.release_depth, self.dive_to = release_depth, dive_to
        self.released = None
        self.wet_at = None
        self.T = np.zeros(len(self.c["thrusters"]))
        self.L = self.L0
        self.tension = 0.0
        self.sub = 0.0
        self.log = []
        self.comp_log = []           # t, k, sheave (earth), hook (earth), deck point (earth), hook reference z, in water

    # the davit robot's nominal trajectories (the compensation adds to them)
    def luff(self, t):
        if self.released is not None:                       # after the release the boom comes back inboard
            return float(min_jerk(t, self.released + 1.5, self.released + 6.5, [self.davit.phi_out], [self.davit.phi_in])[0][0])
        return float(min_jerk(t, self.t_swing[0], self.t_swing[1], [self.davit.phi_in], [self.davit.phi_out])[0][0])

    def k_comp(self, t):
        """How much compensation: none on deck and in the swing, on over the side, off after the release."""
        if not self.compensated:
            return 0.0
        k = _ramp(t, self.t_swing[1], self.t_swing[1] + 1.0)
        if self.released is not None:
            k *= 1.0 - _ramp(t, self.released + 0.5, self.released + 1.5)
        return k

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
<inertial pos="{world._f(self.COM)}" mass="{c['mass']:.3f}" diaginertia="{world._f(J)}"/>
{vis}
<geom name="w_hull" class="drone" type="box" size="{c['L'] / 2:.3f} {c['W'] / 2:.3f} {c['H'] / 2:.3f}" pos="0 0 {c['H'] / 2:.3f}"/>
<site name="walrus_eye" pos="{world._f(self.EYE)}" size="0.03"/>
<geom class="visual" type="cylinder" size="0.012 0.04" pos="{world._f(self.EYE - [0, 0, 0.03])}" rgba="0.8 0.8 0.1 1"/>
</body>"""
        chocks = "".join(f'<geom class="hullvis" type="box" size="0.4 0.06 0.05" pos="{self.start[0]} {self.start[1] + dy:.3f} {self.zd + 0.05:.3f}" rgba="0.3 0.3 0.3 1"/>'
                         for dy in (-0.3, 0.3))
        tendon = (f'<spatial name="walrus_cable" limited="true" range="0 {self.L0:.4f}" width="0.012" rgba="0.95 0.95 0.9 1" solreflimit="0.01 1" margin="0.02">'
                  f'<site site="boom_tip"/><site site="walrus_eye"/></spatial>')
        return {"world": body, "carrier": chocks, "tendon": tendon,
                "equality": '<weld name="walrus_latch" body1="davit_boom" body2="walrus" active="false" solref="0.02 1"/>'}

    def bind(self, sim):
        self.sim = sim
        m = sim.model
        self.b = sim.body(self.bname)
        j = sim.joint(self.bname + "_free")
        self.qa, self.va = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
        self.ten = sim.tendon(self.bname + "_cable")
        sim.prescribe("davit_luff", self.davit.joint("luff"))
        sim.prescribe("davit_slew", self.davit.joint("slew"))
        self.eye = sim.site(self.bname + "_eye")
        self.tip = sim.site("boom_tip")
        self.alpha = 1.0 - math.exp(-sim.dt_control / THRUSTER_TAU)
        self.at = np.array([t["at"] for t in self.c["thrusters"]])
        self.ax = np.array([t["axis"] for t in self.c["thrusters"]])

    def state(self):
        d = self.sim.data
        R = d.xmat[self.b].reshape(3, 3).copy()
        com = d.xipos[self.b].copy()
        v = d.qvel[self.va:self.va + 3].copy()            # the free joint's linear velocity is of the body origin
        w = d.qvel[self.va + 3:self.va + 6].copy()        # body frame
        v_com = v + R @ np.cross(w, self.COM)
        return com, v_com, R, w

    def _cradle(self, sim):
        """The latch cradle is gimballed and runs on a guide along the rope: it holds the Walrus plumb and level under
        the sheave, the rope's length down (the AHC winch's, compensation included) - the weld's target pose, in the
        boom's frame, follows the boom and the winch; the swing-out cannot set it swinging."""
        import mujoco
        d, m = sim.data, sim.model
        bb = sim.body("davit_boom")
        qw = np.array([math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)])          # level, facing outboard
        Rw = world.quat_to_R(qw)
        pw = d.site_xpos[self.tip] - np.array([0.0, 0.0, self.L]) - Rw @ self.EYE
        Rb = d.xmat[bb].reshape(3, 3)
        qi, rq = np.zeros(4), np.zeros(4)
        mujoco.mju_negQuat(qi, d.xquat[bb])
        mujoco.mju_mulQuat(rq, qi, qw)
        e = sim.eq(self.bname + "_latch")
        m.eq_data[e, 0:3] = 0.0
        m.eq_data[e, 3:6] = Rb.T @ (pw - d.xpos[bb])
        m.eq_data[e, 6:10] = rq

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
        if not self.latched and self.t_hoist[1] + 0.2 <= t < self.t_lower:          # the latch cradle closes on the hoisted Walrus
            self.L_hoist = float(np.linalg.norm(d.site_xpos[self.tip] - d.site_xpos[self.eye]))   # the rope as it is when latched
            self.latched = True
            d.eq_active[sim.eq(self.bname + "_latch")] = 1
        if self.latched:
            self._cradle(sim)
        if self.latched and t >= self.t_lower:                                     # and opens when the winch starts paying out
            d.eq_active[sim.eq(self.bname + "_latch")] = 0
            self.latched = False
        self.davit.update(sim, t, dt, self.winch(t), self.luff(t), self.k_comp(t))
        if self.released is None:
            self.L = self.davit.q["rope"]
            m.tendon_range[self.ten, 1] = self.L + (0.05 if self.latched else 0.0)      # latched: the cradle carries it, the rope just slack
            self.tension += (self._cable_tension(sim) - self.tension) * (1 - math.exp(-dt / 0.3))     # the pull, low-passed
            top = com[2] - self.COM[2] + c["H"]
            if t > self.t_lower and eta - top > self.release_depth:
                self.released = t
                m.tendon_limited[self.ten] = 0
                m.tendon_rgba[self.ten, 3] = 0.0
                self.t0 = t
                self.p0 = com.copy()
                self.tension = 0.0
        # the water: buoyancy, drag, current
        z_bot = com[2] - self.COM[2]
        self.sub = sub = float(np.clip((eta - z_bot) / c["H"], 0.0, 1.0))
        F, tau = np.zeros(3), np.zeros(3)
        if sub > 0:
            B = RHO_SW * G * c["volume"] * sub
            r_b = R @ (self.COB - self.COM)
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
        F, tau = self._thrust(com, R, eta, F, tau)
        d.xfrc_applied[self.b, :3] += F
        d.xfrc_applied[self.b, 3:] += tau
        self.log.append((t, *com, *v, self.tension, sub, self._cable_tension(sim) if self.released is None else 0.0))
        if self.released is None:
            if self.wet_at is None and sub > 0:
                self.wet_at = t
            deck, _ = sim.carrier_point(self.davit.P_local)            # the point of the deck's frame where the sheave stands at rest
            self.comp_log.append((t, self.k_comp(t), *d.site_xpos[self.tip], *d.site_xpos[self.eye], *deck,
                                  self.davit.P[2] - self.winch(t), sub))

    def _thrust(self, com, R, eta, F, tau):
        """The thrusters' wrench (each one only when it is wet) added to F, tau."""
        for k in range(len(self.T)):
            pk = R @ (self.at[k] - self.COM)
            if (com[2] + pk[2]) >= eta - 0.03:
                continue
            fk = R @ self.ax[k] * self.T[k]
            F = F + fk
            tau = tau + np.cross(pk, fk)
        return F, tau

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
        wrench = np.concatenate([Fb, Mb + np.cross(self.COM - np.array([0, 0, c["H"] / 2]), Fb)])
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
            return "davit robot: latched in the cradle, the boom luffs outboard"
        if t < self.t_lower:
            return "over the side, latched: compensation takes the deck's motion out" if self.compensated else "over the side, latched"
        if self.released is None:
            return "winch pays out: through the splash zone" if self.sub > 0 else "winch pays out: lowering"
        if t < self.released + 1.0:
            return "hook released: thrusters take over"
        return "diving under its 8 thrusters, away from the hull"

    def residual(self, t0=None, t1=None) -> dict:
        """The hook's (the lifting eye's) and the sheave's motion in the earth frame over the lowering in air (from 1.5 s
        after the boom is out to 0.3 s before the Walrus touches the water), against the deck's motion at the davit:
        the deviation from the earth-fixed path (the sheave's point, the rope's nominal length), rms and peak (m)."""
        L = np.array(self.comp_log)
        if not len(L):
            return {}
        t0 = self.t_swing[1] + 1.5 if t0 is None else t0
        t1 = (self.wet_at - 0.3 if self.wet_at else L[-1, 0]) if t1 is None else t1
        L = L[(L[:, 0] >= t0) & (L[:, 0] <= t1)]
        if len(L) < 10:
            return {}
        P = self.davit.P
        sheave, hook, deck, zref = L[:, 2:5], L[:, 5:8], L[:, 8:11], L[:, 11]

        def stats(dv, dh):
            return {"vertical_rms_m": float(np.sqrt(np.mean(dv ** 2))), "vertical_peak_m": float(np.max(np.abs(dv))),
                    "horizontal_rms_m": float(np.sqrt(np.mean(dh ** 2))), "horizontal_peak_m": float(np.max(np.abs(dh)))}
        hz = hook[:, 2] - (zref - self.EYE[2] * 0 - 0.0)
        hz = hz - np.mean(hz)                                     # the eye hangs a fixed offset under the reference
        return {"window_s": [round(float(t0), 2), round(float(t1), 2)], "compensated": self.compensated,
                "deck_at_davit": stats(deck[:, 2] - P[2], np.hypot(deck[:, 0] - P[0], deck[:, 1] - P[1])),
                "sheave": stats(sheave[:, 2] - P[2], np.hypot(sheave[:, 0] - P[0], sheave[:, 1] - P[1])),
                "hook": stats(hz, np.hypot(hook[:, 0] - P[0], hook[:, 1] - P[1]))}

    def hud(self, sim):
        com, v, R, w = self.state()
        t = sim.t
        dep = self.depth()
        thr = f"thrusters {np.abs(self.T).sum():4.0f} N (max {self.c['t_max'].sum():.0f})" if self.released is not None else "thrusters off"
        out = [f"Walrus {self.c['mass']:.1f} kg   depth {dep:+5.2f} m   speed {np.linalg.norm(v):4.2f} m/s   cable {self.L:4.2f} m, pull {self.tension:4.0f} N   {thr}",
               f"phase: {self.phase(t)}"]
        mode = "ON: MRU + AHC winch + luff + slew" if self.compensated else "OFF: plain davit"
        if self.comp_log and self.released is None and t >= self.t_swing[1]:
            row = self.comp_log[-1]
            P = self.davit.P
            deck = np.array(row[8:11]) - P
            hook = np.array(row[5:8])
            hr = self.residual(t0=self.t_swing[1] + 1.5, t1=t) or {}
            rms = f"   rms so far {100 * hr['hook']['vertical_rms_m']:.0f} cm vert, {100 * hr['hook']['horizontal_rms_m']:.0f} cm horiz" if hr else ""
            hz = hook[2] - row[11]
            out.append(f"compensation {mode}   deck at davit dz {deck[2]:+5.2f} m, dxy {np.hypot(deck[0], deck[1]):4.2f} m   "
                       f"hook (earth) dz {100 * (hz - self._hz0):+4.0f} cm, dxy {100 * np.hypot(hook[0] - P[0], hook[1] - P[1]):3.0f} cm{rms}")
        else:
            out.append(f"compensation {mode}")
        return out

    @property
    def _hz0(self):
        """The eye's mean height under the reference (it hangs below the sheave by the rope and the bridle)."""
        L = np.array(self.comp_log)
        m = L[:, 0] >= self.t_swing[1] + 1.5
        return float(np.mean(L[m, 7] - L[m, 11])) if m.any() else float(L[-1, 7] - L[-1, 11])

    def trace(self, sim, span=10.0):
        """For the overlay's strip chart: the last ``span`` s of the deck's and the hook's vertical motion (m)."""
        L = np.array(self.comp_log)
        if not len(L) or self.released is not None or sim.t < self.t_swing[1]:
            return None
        L = L[L[:, 0] >= max(sim.t - span, self.t_swing[1])]
        return {"t": L[:, 0], "deck at davit": L[:, 10] - self.davit.P[2], "hook (earth)": L[:, 7] - L[:, 11] - self._hz0}

    def focus(self, sim):
        return self.sim.data.xipos[self.b].copy()
