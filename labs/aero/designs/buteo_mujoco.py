"""BUTEO in MuJoCo: ARIGO-1's simulator (``arigo1_mujoco``) with BUTEO's wing-borne coefficients (``buteo_model``) and
four lift rotors; a hover pilot; and the two halves of the unattended cycle:

- ``takeoff_transition``: from the pad, straight up to 15 m, the pusher to full while the rotors hold the height, the
  rotors off once the wing carries it; the time from the cue to safe wing-borne flight.
- ``approach_land``: wing-borne toward the pad into the wind, the pusher off and the rotors on (the back transition),
  to a hover over the pad, down onto it; the miss from the pad's centre, the sink and the tilt at touchdown.
- ``hover_hold``: 20 s over a point in wind and turbulence: the position error, the tilt, the rotors' saturation.

**The rotors.** Each gives a thrust along the body's up axis at its place on the booms and a reaction torque in yaw
(KQ x thrust); a commanded thrust is reached through a first-order lag (ROTOR_TAU). HAND: no thrust loss in a
crossflow, no rotor-wing interference, the electrical power from the hover's by T^1.5.

**The hover pilot.** Position -> acceleration (PD + a slow integral for the wind) -> bank, pitch and the pusher (the
pusher takes the forward force, so the wing stays level into the wind); height -> total thrust (PD + integral, less
the wing's own lift); attitude -> moments (PD) -> the four thrusts (a 4 x 4 mix, clipped; yaw gives way first).
Yaw is only damped: the fins turn the nose into the wind (weathervaning), as quad-planes are flown.
Wind: ``raptors_robust.Turbulence`` (steady + Gauss-Markov gusts, 1.5 x).
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

import arigo1_model as am
import arigo1_mujoco as amj
import buteo as B
import buteo_model as M

ROTOR_TAU = 0.06                  # HAND: rotor thrust lag [s]
Z_GROUND = 0.09                   # the CG's height standing on the pad (the model's contact box)
H_TRANSITION = 15.0
H_APPROACH = 20.0
H_HOVER = 8.0
BLUFF_CDA = np.array([0.012, 0.115, 0.30])      # HAND: Cd x A [m2] along the body, across it (pod, fins, booms), through the wing's plane
BLUFF_X_M = -0.30                 # HAND: where the side force acts, behind the CG [m] (it turns the nose into the wind)
BLUFF_DAMP = 0.004                # HAND: rate damping of the bluff body [N m s / (m/s)]
LAND = {"vz_m_s": 1.5, "tilt_deg": 12.0, "miss_m": B.PAD_TOLERANCE_M}


class SimButeo(amj.Sim):
    def __init__(self, variant=None, propulsion=None, mesh=None, bandit=False, C=None):
        self.C = C or M.get()
        self.m = mujoco.MjModel.from_xml_string(amj.mjcf(self.C, mesh, bandit))
        self.d = mujoco.MjData(self.m)
        self.bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "arigo")
        self.controls = np.array([0.0, 0.0, 0.0, 0.5])
        self.wind = np.zeros(3)
        self.info = {}
        self.lift_cmd = np.zeros(4)                # commanded thrusts [N]
        self.lift = np.zeros(4)                    # actual
        self.energy_J = 0.0

    def step(self, n=1):
        L = self.C["lift"]
        Th = self.C["mass"] * am.G / 4
        for _ in range(n):
            s = self.state()
            F, Mo, info = self._aero(s)
            self.lift += (np.clip(self.lift_cmd, 0.0, L["T_max_each_N"]) - self.lift) * (amj.DT / ROTOR_TAU)
            for T, (x, y, sp) in zip(self.lift, L["rotors"]):
                F[2] -= T
                Mo[0] += -y * T
                Mo[1] += x * T
                Mo[2] += sp * L["kq_m"] * T
            P_lift = float(np.sum((L["hover_power_W"] - 15.0) / 4 * (np.maximum(self.lift, 0.0) / Th) ** 1.5))
            P_push = self.controls[3] * self.C["thrust"]["propulsor"]["power_w_per_unit"] if self.controls[3] > 0.02 else 0.0
            self.energy_J += (P_lift + P_push + 15.0) * amj.DT
            R = self.R()
            self.d.xfrc_applied[self.bid, :3] = R @ (F * amj.FLIP)
            self.d.xfrc_applied[self.bid, 3:] = R @ (Mo * amj.FLIP)
            self.info = {**info, "load_factor": float(-F[2] / (self.C["mass"] * am.G))}
            mujoco.mj_step(self.m, self.d)

    def _aero(self, s):
        """The wing-borne model where it holds (the air from ahead, within ~35 deg of the nose); a bluff body where
        it does not (a tailwind or a crosswind in the hover: the linear derivatives are meaningless there), blended
        between. The bluff body (HAND): drag areas Cd x A along, across and through the wing's plane, the side force
        acting BLUFF_X_M behind the CG (the fins: it weathervanes). The pusher's thrust is added unblended."""
        C = self.C
        v = np.asarray(s["v_air_frd"], dtype=float)
        V = float(np.linalg.norm(v))
        thr = self.controls[3]
        ctl = (*self.controls[:3], 0.0)
        F, Mo, info = am.aero(C, v, s["omega_frd"], *ctl)
        c = v[0] / V if V > 1e-6 else 1.0
        w = min(1.0, max(0.0, (c - 0.55) / 0.27))                  # 1 inside ~35 deg of the nose, 0 beyond ~57 deg
        w = w * w * (3 - 2 * w)
        if w < 1.0:
            Fb = -0.5 * am.RHO * V * v * BLUFF_CDA
            Mb = np.array([0.0, 0.0, BLUFF_X_M * Fb[1]]) - np.asarray(s["omega_frd"]) * BLUFF_DAMP * V
            F, Mo = w * F + (1 - w) * Fb, w * Mo + (1 - w) * Mb
        F = F + np.array([thr * am.max_thrust(C, max(v[0], 0.0)), 0.0, 0.0])
        return F, Mo, info

    def stand(self, heading_deg=0.0, xy=(0.0, 0.0)):
        """On the pad, at rest."""
        self.start(max(1.3 * M.stall(self.C), 12.0), 30.0, heading_deg)
        ps = math.radians(heading_deg)
        self.d.qpos[:3] = (xy[0], xy[1], Z_GROUND)
        self.d.qpos[3:7] = (math.cos(ps / 2), 0, 0, math.sin(ps / 2))
        self.d.qvel[:] = 0.0
        self.controls[:] = 0.0
        self.lift[:] = 0.0
        self.lift_cmd[:] = 0.0
        self.energy_J = 0.0
        mujoco.mj_forward(self.m, self.d)


class HoverPilot:
    """See the module's notes. ``__call__(pos_c, z_c, vz_c=0)`` sets the rotors, the pusher and neutral surfaces."""

    def __init__(self, sim: SimButeo):
        self.s = sim
        C = sim.C
        L = C["lift"]
        self.A = np.array([[1.0] * 4, [-y for _, y, _ in L["rotors"]], [x for x, _, _ in L["rotors"]],
                           [sp * L["kq_m"] for _, _, sp in L["rotors"]]])
        self.Tmax = L["T_max_each_N"]
        self.iz = 0.0
        self.ixy = np.zeros(2)
        self.sat = 0
        self.n = 0
        self.max_tilt = 0.0
        self.V13 = 1.3 * M.stall(C)

    def __call__(self, pos_c=None, z_c=10.0, vz_c=0.0, a_fwd=None, pusher=True, dt=0.01):
        s, C = self.s.state(), self.s.C
        m = C["mass"]
        p, q, r = s["omega_frd"]
        ps = s["psi"]
        v = s["v_world"]
        # horizontal: world acceleration wanted -> forward / left in the heading frame
        if pos_c is not None:
            e = np.asarray(pos_c[:2]) - s["pos"][:2]
            self.ixy = np.clip(self.ixy + e * dt, -6.0, 6.0)
            a = 1.6 * e - 2.6 * v[:2] + 0.5 * self.ixy
            n = float(np.linalg.norm(a))
            a = a * (4.0 / n) if n > 4.0 else a
            a_f = a[0] * math.cos(ps) + a[1] * math.sin(ps)
            a_l = -a[0] * math.sin(ps) + a[1] * math.cos(ps)
        else:
            a_f, a_l = (a_fwd or 0.0), -1.5 * (-v[0] * math.sin(ps) + v[1] * math.cos(ps))    # no drift sideways
        thr = 0.0
        if pusher and a_f > 0:
            Tp = max(am.max_thrust(C, max(s["v_air_frd"][0], 0.0)), 1e-3)
            thr = min(1.0, m * a_f / Tp)
            a_f -= thr * Tp / m
        th_c = max(-0.30, min(0.30, -a_f / am.G))
        ph_c = max(-0.35, min(0.35, -a_l / am.G))
        # height: total thrust, less what the wing carries
        ez = z_c - s["pos"][2]
        self.iz = max(-4.0, min(4.0, self.iz + ez * dt))
        az = 2.5 * ez + 3.0 * (vz_c - v[2]) + 0.8 * self.iz
        Vair = float(np.linalg.norm(s["v_air_frd"]))
        wing = m * am.G * min(1.0, (Vair / self.V13) ** 2) * (1.0 if s["v_air_frd"][0] > 0 else 0.0)
        Ttot = max(0.0, (m * (am.G + az) - wing) / max(math.cos(s["phi"]) * math.cos(s["theta"]), 0.5))
        Ixx, Iyy, Izz = C["inertia"]
        Mx = Ixx * (60.0 * (ph_c - s["phi"]) - 12.0 * p)
        My = Iyy * (60.0 * (th_c - s["theta"]) - 12.0 * q)
        Mz = -Izz * 2.5 * r
        T = np.linalg.solve(self.A, np.array([Ttot, Mx, My, Mz]))
        if (T < 0).any() or (T > self.Tmax).any():
            T = np.linalg.solve(self.A, np.array([Ttot, Mx, My, 0.0]))            # yaw gives way first
            if (T < 0).any() or (T > self.Tmax).any():
                self.sat += 1
        self.n += 1
        self.s.lift_cmd[:] = np.clip(T, 0.0, self.Tmax)
        self.s.controls[:] = (0.0, 0.0, 0.0, thr)
        self.max_tilt = max(self.max_tilt, math.degrees(max(abs(s["phi"]), abs(s["theta"]))))


def _wind(rng, W, wind_to_deg):
    import raptors_robust as RB
    return RB.Turbulence(rng, W, wind_to_deg) if rng is not None else (lambda dt, V: np.array([W * math.cos(math.radians(wind_to_deg)), W * math.sin(math.radians(wind_to_deg)), 0.0]))


def takeoff_transition(sim: SimButeo, W=0.0, rng=None, heading_deg=0.0, wind_to_deg=180.0) -> dict:
    """Pad -> H_TRANSITION -> wing-borne. Safe flight: the rotors off, >= 1.3 x stall, at or above 10 m."""
    C = sim.C
    Vs = M.stall(C)
    sim.stand(heading_deg)
    turb = _wind(rng, W, wind_to_deg)
    hp = HoverPilot(sim)
    ap = amj.Autopilot(sim)
    phase, t_ph, t_safe, why, z_min_after = "climb", 0.0, None, None, 1e9
    blend, t0 = 1.0, sim.d.time
    while sim.d.time - t0 < 25.0:
        s = sim.state()
        t = sim.d.time - t0
        sim.wind[:] = turb(0.01, max(s["V"], 3.0))
        if phase == "climb":
            hp(pos_c=(0.0, 0.0), z_c=H_TRANSITION, vz_c=0.0)
            if s["pos"][2] > H_TRANSITION - 1.0:
                phase, t_ph = "accelerate", t
        elif phase == "accelerate":
            hp(pos_c=None, z_c=H_TRANSITION, a_fwd=10.0)                       # the pusher at full, the rotors hold the height
            if s["v_air_frd"][0] >= 1.2 * Vs:
                phase, t_ph = "wing", t
        else:
            blend = max(0.0, 1.0 - (t - t_ph) / 1.0)
            ap(phi_c=0.0, theta_c=math.radians(8.0), throttle=1.0)
            keep = sim.controls.copy()
            hp(pos_c=None, z_c=max(H_TRANSITION, s["pos"][2]), a_fwd=10.0)
            sim.lift_cmd *= blend
            sim.controls[:] = keep
            if blend <= 0.0 and s["V"] >= 1.3 * Vs and s["pos"][2] >= 10.0:
                t_safe = t
                break
        sim.step(5)
        s = sim.state()
        if t > 1.0:
            z_min_after = min(z_min_after, s["pos"][2]) if phase != "climb" else z_min_after
        if phase != "climb" and s["pos"][2] < 1.0:
            why = "ground"
            break
        if max(abs(s["phi"]), abs(s["theta"])) > math.radians(60.0):
            why = "upset"
            break
    sim.wind[:] = 0.0
    ok = t_safe is not None
    return {"ok": ok, "why": None if ok else (why or "slow"), "t_safe_s": None if t_safe is None else round(t_safe, 2),
            "max_tilt_deg": round(hp.max_tilt, 1), "saturated_share": round(hp.sat / max(hp.n, 1), 3),
            "lowest_after_climb_m": round(z_min_after, 1) if z_min_after < 1e8 else None, "energy_Wh": round(sim.energy_J / 3600.0, 2)}


def approach_land(sim: SimButeo, W=0.0, rng=None, wind_off_deg=0.0, start_m=150.0) -> dict:
    """Wing-borne at H_APPROACH, ``start_m`` downwind of the pad, flying into the wind toward it: back transition,
    hover over the pad, down onto it."""
    C = sim.C
    Vs = M.stall(C)
    sim.start(1.3 * Vs, H_APPROACH)
    sim.d.qpos[0] = -start_m
    sim.d.qvel[0] = max(1.3 * Vs - W, 2.0)
    sim.lift[:] = 0.0
    sim.lift_cmd[:] = 0.0
    sim.energy_J = 0.0
    mujoco.mj_forward(sim.m, sim.d)
    turb = _wind(rng, W, 180.0 + wind_off_deg)
    hp = HoverPilot(sim)
    ap = amj.Autopilot(sim)
    phase, td, z_c, t0 = "cruise", None, H_HOVER, sim.d.time
    while sim.d.time - t0 < 120.0:
        s = sim.state()
        sim.wind[:] = turb(0.01, max(s["V"], 3.0))
        x, z = s["pos"][0], s["pos"][2]
        gs = float(np.linalg.norm(s["v_world"][:2]))
        if phase == "cruise":
            ap(phi_c=max(-0.3, min(0.3, 0.02 * s["pos"][1] + 0.8 * s["v_world"][1] / max(s["V"], 3.0))),
               theta_c=math.radians(sim.trim["alpha_deg"]) + 0.05 * (H_APPROACH - z), V_c=1.3 * Vs)
            if -x < 25.0 + 2.5 * gs:                                            # the distance the back transition needs
                phase = "brake"
        elif phase == "brake":
            hp(pos_c=None, z_c=H_HOVER, a_fwd=-3.0, pusher=False)
            if gs < 5.0 or x > -3.0:
                phase = "hover"
        elif phase == "hover":
            hp(pos_c=(0.0, 0.0), z_c=H_HOVER)
            if math.hypot(x, s["pos"][1]) < 0.15 and gs < 0.5:
                phase, z_c = "descend", z
        else:
            off = math.hypot(x, s["pos"][1])
            hold = off > 0.15 and z > 0.5                                       # off the centre: wait, do not come down
            rate = 0.0 if hold else (1.0 if z > 2.0 else 0.4)
            z_c -= rate * 0.01
            hp(pos_c=(0.0, 0.0), z_c=z_c, vz_c=-rate)
            if z <= Z_GROUND + 0.02:
                td = {"x": float(x), "y": float(s["pos"][1]), "vz": float(-s["v_world"][2]),
                      "tilt": math.degrees(max(abs(s["phi"]), abs(s["theta"]))), "t": sim.d.time - t0}
                break
        sim.step(5)
        if max(abs(sim.state()["phi"]), abs(sim.state()["theta"])) > math.radians(60.0):
            break
    sim.wind[:] = 0.0
    sim.lift_cmd[:] = 0.0
    if td is None:
        return {"ok": False, "why": f"no touchdown ({phase})", "max_tilt_deg": round(hp.max_tilt, 1)}
    miss = math.hypot(td["x"], td["y"])
    checks = {"miss": miss <= LAND["miss_m"], "sink": td["vz"] <= LAND["vz_m_s"], "tilt": td["tilt"] <= LAND["tilt_deg"]}
    ok = all(checks.values())
    return {"ok": ok, "why": None if ok else next(k for k, v in checks.items() if not v), "miss_m": round(miss, 3),
            "sink_m_s": round(td["vz"], 2), "tilt_deg": round(td["tilt"], 1), "time_s": round(td["t"], 1),
            "max_tilt_deg": round(hp.max_tilt, 1), "saturated_share": round(hp.sat / max(hp.n, 1), 3),
            "energy_Wh": round(sim.energy_J / 3600.0, 2)}


def hover_hold(sim: SimButeo, W=0.0, rng=None, seconds=20.0, wind_to_deg=180.0) -> dict:
    """Hold a point at H_HOVER in the wind (started nose into it): position error, tilt, saturation, power."""
    sim.stand(0.0)
    sim.d.qpos[2] = H_HOVER
    mujoco.mj_forward(sim.m, sim.d)
    turb = _wind(rng, W, wind_to_deg)
    hp = HoverPilot(sim)
    err, t0 = [], sim.d.time
    while sim.d.time - t0 < seconds + 5.0:
        s = sim.state()
        sim.wind[:] = turb(0.01, max(s["V"], 3.0))
        hp(pos_c=(0.0, 0.0), z_c=H_HOVER)
        sim.step(5)
        s = sim.state()
        if sim.d.time - t0 > 5.0:
            err.append(math.hypot(s["pos"][0], s["pos"][1]))
        if max(abs(s["phi"]), abs(s["theta"])) > math.radians(60.0) or s["pos"][2] < 1.0:
            return {"ok": False, "why": "lost", "max_tilt_deg": round(hp.max_tilt, 1)}
    sim.wind[:] = 0.0
    e = np.array(err)
    return {"ok": bool(e.max() < 2.0), "rms_error_m": round(float(np.sqrt((e ** 2).mean())), 2), "max_error_m": round(float(e.max()), 2),
            "max_tilt_deg": round(hp.max_tilt, 1), "saturated_share": round(hp.sat / max(hp.n, 1), 3),
            "mean_power_W": round(sim.energy_J / (sim.d.time - t0), 0)}


def robustness(sim_factory, n=12, winds=(0.0, 4.0, 8.0, 12.0, 16.0), seed=0) -> dict:
    """Per wind (gusts 1.5 x): take-off-and-transition success (the pad's heading to the wind random) and pad landing
    success (approach into the wind +-20 deg); the operating wind: both >= 95 %."""
    import raptors_robust as RB
    rng = np.random.default_rng(seed)
    per = []
    for W in winds:
        T = [takeoff_transition(sim_factory(), W, rng, heading_deg=rng.uniform(0, 360.0)) for _ in range(n)]
        Ld = [approach_land(sim_factory(), W, rng, wind_off_deg=rng.uniform(-20.0, 20.0)) for _ in range(n)]
        misses = [x["miss_m"] for x in Ld if "miss_m" in x]
        per.append({"wind_m_s": W, "gust_m_s": round(RB.GUST_RATIO * W, 1), "launch": RB._rate(T), "recovery": RB._rate(Ld),
                    "t_safe_median_s": float(np.median([x["t_safe_s"] for x in T if x["t_safe_s"]])) if any(x["t_safe_s"] for x in T) else None,
                    "miss_median_m": round(float(np.median(misses)), 3) if misses else None,
                    "miss_p90_m": round(float(np.percentile(misses, 90)), 3) if misses else None})
        if per[-1]["launch"]["rate"] < 0.5 and per[-1]["recovery"]["rate"] < 0.5:
            break
    op = 0.0
    for p in per:
        if p["launch"]["rate"] >= 0.95 and p["recovery"]["rate"] >= 0.95:
            op = p["wind_m_s"]
        else:
            break
    return {"per_wind": per, "operating_wind_m_s": op, "operating_gust_m_s": round(RB.GUST_RATIO * op, 1), "samples": n,
            "rule": f"take-off: wing-borne safe flight in 25 s, no upset; landing: miss <= {LAND['miss_m']} m, sink <= {LAND['vz_m_s']} m/s, "
                    f"tilt <= {LAND['tilt_deg']} deg; gusts 1.5 x steady"}
