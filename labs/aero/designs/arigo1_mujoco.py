"""ARIGO-1 in MuJoCo: a free rigid body whose aerodynamic force and moment come from ``arigo1_model.aero`` every
step (applied through ``xfrc_applied``), an autopilot, the stability tests and the manoeuvres (notebook 39).

MuJoCo does the rigid-body integration, the ground contact (the belly landing) and the pictures; the aerodynamics
are the model's (lattice + hand estimates, see ``arigo1_model``) — the runs are as good as those coefficients, and
the open-loop tests are compared with the model's own linear modes so a disagreement shows up as a bug, not a result.

World: x east, y north, z up (MuJoCo's). Body: x forward, y left, z up; converted to x forward, y right, z down
(``FRD``) for the aerodynamics.
"""
from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

import arigo1 as a1
import arigo1_model as am

DT = 0.002
FLIP = np.array([1.0, -1.0, -1.0])               # FLU <-> FRD


def mjcf(C: dict, mesh: str | None = None, bandit: bool = False) -> str:
    """The model as MJCF: ground, the aircraft (a free body with the model's mass and inertia, a belly box for contact,
    the CAD as a visual mesh when given), optionally a bandit (a mocap box)."""
    Ixx, Iyy, Izz = C["inertia"]
    b, L = C["b"], 0.72
    vis = ""
    if mesh:
        # the CAD frame: x aft from the nose, z up, mm -> the body: x forward, origin at the CG
        vis = (f'<geom type="mesh" mesh="arigo" contype="0" conaffinity="0" rgba="0.75 0.72 0.62 1" '
               f'pos="{C["x_cg_m"]:.4f} 0 {-C["z_cg_m"]:.4f}" euler="0 0 180"/>')
    else:
        vis = (f'<geom type="box" size="{L / 2:.3f} 0.035 0.035" contype="0" conaffinity="0" rgba="0.7 0.7 0.6 1"/>'
               f'<geom type="box" size="0.12 {b / 2:.3f} 0.008" pos="0 0 0" contype="0" conaffinity="0" rgba="0.75 0.72 0.62 1"/>')
    asset = f'<mesh name="arigo" file="{mesh}" scale="0.001 0.001 0.001"/>' if mesh else ""
    band = ('<body name="bandit" mocap="true" pos="40 0 30"><geom type="box" size="0.18 0.18 0.05" rgba="0.8 0.15 0.1 1" '
            'contype="0" conaffinity="0"/></body>') if bandit else ""
    return f"""<mujoco model="arigo1">
  <option timestep="{DT}" gravity="0 0 -9.80665" density="0" viscosity="0" integrator="RK4"/>
  <visual><global offwidth="1280" offheight="720"/><quality shadowsize="2048"/><map znear="0.05" zfar="2000"/></visual>
  <asset>{asset}
    <texture name="grid" type="2d" builtin="checker" rgb1="0.32 0.42 0.30" rgb2="0.28 0.37 0.26" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="200 200"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.55 0.7 0.9" rgb2="0.9 0.93 0.97" width="256" height="256"/>
  </asset>
  <worldbody>
    <light pos="0 0 300" dir="0.2 0.3 -1" directional="true"/>
    <geom type="plane" size="2000 2000 1" material="grid"/>
    <body name="arigo" pos="0 0 30">
      <freejoint/>
      <inertial pos="0 0 0" mass="{C['mass']:.4f}" diaginertia="{Ixx:.5f} {Iyy:.5f} {Izz:.5f}"/>
      <geom name="belly" type="box" size="0.2 0.04 0.02" pos="0 0 -0.045" rgba="0 0 0 0" friction="0.5 0.01 0.001"/>
      {vis}
    </body>
    {band}
  </worldbody>
</mujoco>"""


class Sim:
    """One aircraft in MuJoCo. ``controls`` (de, da, dr [deg], throttle 0-1) are held between steps; ``wind`` [m/s,
    world]."""

    def __init__(self, variant="hybrid", propulsion="pusher", mesh=None, bandit=False):
        self.C = am.coefficients(variant, propulsion)
        self.m = mujoco.MjModel.from_xml_string(mjcf(self.C, mesh, bandit))
        self.d = mujoco.MjData(self.m)
        self.bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "arigo")
        self.controls = np.array([0.0, 0.0, 0.0, 0.5])
        self.wind = np.zeros(3)
        self.info = {}

    def start(self, V=18.0, h=40.0, heading_deg=0.0):
        """Level trimmed flight at V, altitude h, heading (0 = +x). Above the top level speed (no trim: the thrust
        is short) the controls of the 26 m/s trim, at V — as if diving into the manoeuvre."""
        try:
            t = am.trim(self.C, V)
            assert t["ok"]
        except (np.linalg.LinAlgError, AssertionError):
            t = {**am.trim(self.C, 26.0), "V": V, "ok": False, "note": "beyond the top level speed: entered from a dive"}
        al = math.radians(t["alpha_deg"])
        mujoco.mj_resetData(self.m, self.d)
        ps = math.radians(heading_deg)
        q_yaw = np.array([math.cos(ps / 2), 0, 0, math.sin(ps / 2)])
        q_pitch = np.array([math.cos(-al / 2), 0, math.sin(-al / 2), 0])     # nose up = -rotation about y_left? (y left: + is nose down)
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, q_yaw, q_pitch)
        self.d.qpos[:3] = (0, 0, h)
        self.d.qpos[3:7] = q
        self.d.qvel[:3] = (V * math.cos(ps), V * math.sin(ps), 0.0)
        self.controls[:] = (t["elevator_deg"], 0, 0, t["throttle"])
        self.trim = t
        mujoco.mj_forward(self.m, self.d)
        return t

    # ----------------------------------------------------------------------------------------------------- state
    def R(self):
        return self.d.xmat[self.bid].reshape(3, 3)

    def state(self) -> dict:
        R = self.R()
        v_w = self.d.qvel[:3].copy()
        v_air = R.T @ (v_w - self.wind) * FLIP
        w = self.d.qvel[3:6] * FLIP
        x_b, y_b, z_b = R[:, 0], R[:, 1], R[:, 2]
        theta = math.asin(max(-1, min(1, x_b[2])))
        phi = math.atan2(y_b[2], z_b[2])
        psi = math.atan2(x_b[1], x_b[0])
        return {"pos": self.d.qpos[:3].copy(), "v_world": v_w, "v_air_frd": v_air, "omega_frd": w, "phi": phi, "theta": theta, "psi": psi,
                "V": float(np.linalg.norm(v_air)), "t": self.d.time}

    def step(self, n=1):
        for _ in range(n):
            s = self.state()
            F, M, info = am.aero(self.C, s["v_air_frd"], s["omega_frd"], *self.controls)
            R = self.R()
            self.d.xfrc_applied[self.bid, :3] = R @ (F * FLIP)
            self.d.xfrc_applied[self.bid, 3:] = R @ (M * FLIP)
            self.info = {**info, "load_factor": float(-F[2] / (self.C["mass"] * am.G))}
            mujoco.mj_step(self.m, self.d)


class Autopilot:
    """Bank and load factor (or pitch angle) commands -> elevator, aileron; a turn coordinator on the rudder; speed
    on the throttle. Gains by hand for the model's modes (short period ~7 rad/s, roll time constant ~0.2 s)."""

    def __init__(self, sim: Sim):
        self.s = sim
        self.i_q = 0.0

    def __call__(self, phi_c=0.0, n_c=None, theta_c=None, V_c=18.0, throttle=None, protect=True):
        s, C = self.s.state(), self.s.C
        p, q, r = s["omega_frd"]
        V = max(s["V"], 3.0)
        if n_c is None:
            th_c = 0.0 if theta_c is None else theta_c
            n_c = math.cos(s["theta"]) / max(math.cos(s["phi"]), 0.3) + 2.0 * (th_c - s["theta"]) * V / am.G
        n_av = 0.5 * am.RHO * V * V * C["S"] * C["CL_MAX"] / (C["mass"] * am.G)     # what the wing can give at this speed
        n_c = max(-0.5 * n_av, min(n_c, 0.9 * n_av))                                  # the alpha limiter
        q_c = am.G * (n_c - math.cos(s["theta"]) * math.cos(s["phi"])) / V
        e = q_c - q
        self.i_q = max(-1.0, min(1.0, self.i_q + e * DT))
        de = self.s.trim["elevator_deg"] - (180 / math.pi) * (0.4 * e + 1.2 * self.i_q) * (18.0 / V)
        vb = s["v_air_frd"]
        al = math.atan2(vb[2], vb[0])
        a_lim = C["alpha_stall"] - math.radians(2.0)                  # alpha protection: nose down past the limit
        if protect and al > a_lim:
            de += 4.0 * math.degrees(al - a_lim)
            self.i_q = min(self.i_q, 0.0) if e > 0 else self.i_q
        err_phi = (phi_c - s["phi"] + math.pi) % (2 * math.pi) - math.pi
        p_c = max(-6.0, min(6.0, 5.0 * err_phi))
        da = (180 / math.pi) * 0.12 * (p_c - p) * (18.0 / V) ** 2
        r_c = am.G * math.sin(s["phi"]) * math.cos(s["theta"]) / V
        dr = -(180 / math.pi) * 0.3 * (r_c - r)
        thr = throttle if throttle is not None else max(0.0, min(1.0, self.s.trim["throttle"] + 0.15 * (V_c - s["V"])))
        self.s.controls[:] = (max(-am.DE_MAX, min(am.DE_MAX, de)), max(-am.DA_MAX, min(am.DA_MAX, da)),
                              max(-am.DR_MAX, min(am.DR_MAX, dr)), thr)


# ------------------------------------------------------------------------------------------------- stability tests
def _peaks(t, y):
    i = [k for k in range(1, len(y) - 1) if (y[k] - y[k - 1]) * (y[k + 1] - y[k]) < 0]
    return np.array(t)[i], np.array(y)[i]


def _identify(t, y):
    """Period and damping from the extremes of a decaying oscillation (log decrement), None if not oscillating."""
    tp, yp = _peaks(t, y - np.mean(y[-len(y) // 4:]))
    if len(tp) < 3:
        return {"period_s": None, "zeta": None}
    T = 2 * float(np.median(np.diff(tp)))
    a = np.abs(yp)
    d = [math.log(a[k] / a[k + 2]) for k in range(len(a) - 2) if a[k + 2] > 1e-9 and a[k] > 1e-9]
    if not d:
        return {"period_s": T, "zeta": None}
    dl = float(np.median(d))
    return {"period_s": round(T, 3), "zeta": round(dl / math.sqrt(4 * math.pi ** 2 + dl * dl), 3)}


def run(sim, seconds, controls_fn=None, record=None):
    """Advance ``seconds``; ``controls_fn(t, sim)`` may set ``sim.controls``; returns the time history."""
    hist = {k: [] for k in ("t", "alpha", "beta", "V", "p", "q", "r", "phi", "theta", "psi", "z", "x", "y", "n", "stall")}
    t0 = sim.d.time
    while sim.d.time - t0 < seconds:
        if controls_fn:
            controls_fn(sim.d.time - t0, sim)
        sim.step(5)
        s, i = sim.state(), sim.info
        for k, v in (("t", sim.d.time - t0), ("alpha", math.degrees(i["alpha"])), ("beta", math.degrees(i["beta"])), ("V", s["V"]),
                     ("p", math.degrees(s["omega_frd"][0])), ("q", math.degrees(s["omega_frd"][1])), ("r", math.degrees(s["omega_frd"][2])),
                     ("phi", math.degrees(s["phi"])), ("theta", math.degrees(s["theta"])), ("psi", math.degrees(s["psi"])),
                     ("z", s["pos"][2]), ("x", s["pos"][0]), ("y", s["pos"][1]), ("n", i["load_factor"]), ("stall", i["stall"])):
            hist[k].append(float(v))
        if record is not None:
            record(sim)
        if s["pos"][2] < 0.3:
            break
    return {k: np.array(v) for k, v in hist.items()}


def stability_tests(variant="hybrid", propulsion="pusher", V=18.0, post_stall_cm_factor=1.0) -> dict:
    """The open-loop (controls held at trim) and closed-loop tests; every number from the run, the model's linear
    modes beside them."""
    out = {"variant": variant, "propulsion": propulsion, "V_m_s": V}
    sim = Sim(variant, propulsion)
    sim.C = C = {**sim.C, "post_stall_cm_factor": post_stall_cm_factor}
    out["post_stall_cm_factor"] = post_stall_cm_factor
    md = am.modes(C, V)
    out["trim"] = {k: (round(float(v), 4) if not isinstance(v, bool) else v) for k, v in md["trim"].items()}
    out["linear_modes"] = {k: {a: (round(b, 4) if isinstance(b, float) else b) for a, b in r.items()} for k, r in am.name_modes(md).items()}
    tr = sim.start(V, 60.0)
    de0, thr0 = tr["elevator_deg"], tr["throttle"]

    # 1. hands-off: does the trim hold for 20 s?
    h = run(sim, 20.0)
    out["hands_off_20s"] = {"altitude_change_m": round(float(h["z"][-1] - 60.0), 2), "speed_change_m_s": round(float(h["V"][-1] - V), 2),
                            "bank_deg_end": round(float(h["phi"][-1]), 2), "max_alpha_deg": round(float(h["alpha"].max()), 2)}

    # 2. elevator doublet (+-5 deg, 0.25 s each): the short period in alpha (first 3 s), the phugoid in speed (60 s)
    sim.start(V, 80.0)
    fn = lambda t, s: s.controls.__setitem__(0, de0 + (5.0 if t < 0.25 else -5.0 if t < 0.5 else 0.0))
    h = run(sim, 60.0, fn)
    k3 = h["t"] <= 3.0
    out["elevator_doublet"] = {"short_period_from_alpha": _identify(h["t"][k3 & (h["t"] > 0.5)], h["alpha"][k3 & (h["t"] > 0.5)]),
                               "phugoid_from_speed": _identify(h["t"][h["t"] > 3.0], h["V"][h["t"] > 3.0]),
                               "peak_alpha_deg": round(float(h["alpha"].max()), 2), "peak_load_factor": round(float(h["n"].max()), 2),
                               "altitude_change_60s_m": round(float(h["z"][-1] - 80.0), 1)}

    # 3. rudder doublet (+-10 deg): the Dutch roll in sideslip
    sim.start(V, 60.0)
    fn = lambda t, s: s.controls.__setitem__(2, 10.0 if t < 0.2 else -10.0 if t < 0.4 else 0.0)
    h = run(sim, 8.0, fn)
    k = h["t"] > 0.5
    out["rudder_doublet"] = {"dutch_roll_from_beta": _identify(h["t"][k], h["beta"][k]), "peak_beta_deg": round(float(np.abs(h["beta"]).max()), 2)}

    # 4. aileron pulse (10 deg, 0.3 s), then hands off for 15 s: roll mode and the spiral
    sim.start(V, 80.0)
    fn = lambda t, s: s.controls.__setitem__(1, 10.0 if t < 0.3 else 0.0)
    h = run(sim, 15.0, fn)
    out["aileron_pulse"] = {"peak_roll_rate_deg_s": round(float(np.abs(h["p"]).max()), 1), "bank_after_1s_deg": round(float(np.interp(1.0, h["t"], h["phi"])), 1),
                            "bank_after_15s_deg": round(float(h["phi"][-1]), 1), "altitude_change_m": round(float(h["z"][-1] - 80.0), 1)}

    # 5. full-aileron roll from level: time to 90 and 360 degrees (instantaneous agility)
    sim.start(V, 80.0)
    acc = {"phi": 0.0}
    def roll(t, s):
        s.controls[1] = am.DA_MAX
        acc["phi"] += s.state()["omega_frd"][0] * DT * 5
    h = run(sim, 3.0, roll)
    phi_un = np.cumsum(np.radians(h["p"])) * DT * 5
    t90 = float(np.interp(math.pi / 2, phi_un, h["t"])) if phi_un.max() > math.pi / 2 else None
    t360 = float(np.interp(2 * math.pi, phi_un, h["t"])) if phi_un.max() > 2 * math.pi else None
    out["full_aileron_roll"] = {"steady_roll_rate_deg_s": round(float(np.median(h["p"][h["t"] > 0.8])), 1), "time_to_90deg_s": t90 and round(t90, 3),
                                "time_to_360deg_s": t360 and round(t360, 3), "pb_2V": round(float(np.radians(np.median(h["p"][h["t"] > 0.8])) * C["b"] / (2 * V)), 3)}

    # 6. vertical gust: a 3 m/s up-gust for 1 s
    sim.start(V, 60.0)
    def gust(t, s):
        s.wind[:] = (0, 0, 3.0 if 0.2 < t < 1.2 else 0.0)
    h = run(sim, 10.0, gust)
    sim.wind[:] = 0
    out["vertical_gust_3m_s"] = {"peak_alpha_deg": round(float(h["alpha"].max()), 2), "peak_load_factor": round(float(h["n"].max()), 2),
                                 "altitude_change_m": round(float(h["z"][-1] - 60.0), 1), "stalled": bool(h["stall"].max() > 0.5)}

    # 7. power-off stall: idle, the autopilot holds the nose 10 degrees up while the speed bleeds until the wing
    #    stalls; the elevator then held full up 1.5 s (the abuse); then let go (trim elevator, trim throttle)
    for ab, key in ((0.0, "stall_and_release"), (1.5, "stall_full_up_1_5s_then_release")):
        sim.start(V, 100.0)
        ap = Autopilot(sim)
        st = {"t_stall": None}
        def stall(t, s, ab=ab, ap=ap, st=st):
            if st["t_stall"] is None:
                ap(theta_c=math.radians(10.0), throttle=0.0, protect=False)
                if s.info.get("stall", 0) > 0.5:
                    st["t_stall"] = t
            elif t - st["t_stall"] < ab:
                s.controls[:] = (-am.DE_MAX, 0, 0, 0.0)
            else:
                s.controls[:] = (de0, 0, 0, thr0)
        h = run(sim, 25.0, stall)
        ts = st["t_stall"]
        if ts is None:
            out[key] = {"note": "did not stall in 25 s"}
        else:
            rel = ts + ab
            after = h["t"] > rel
            upright = np.abs(h["theta"]) < 60
            rec = [t for t, a, sv in zip(h["t"][after], h["alpha"][after], h["stall"][after]) if sv < 0.1]
            level = [t for t, th, sv in zip(h["t"][after], h["theta"][after], h["stall"][after]) if sv < 0.1 and abs(th) < 5 and t > (rec[0] if rec else 1e9)]
            out[key] = {"stall_speed_m_s": round(float(np.interp(ts, h["t"], h["V"])), 2), "max_alpha_deg": round(float(h["alpha"].max()), 1),
                                        "max_bank_deg": round(float(np.abs(h["phi"][upright]).max()), 1),
                                        "departed": bool((np.abs(h["phi"]) > 60)[upright].any() or h["alpha"].max() > 45),
                                        "unstalled_after_release_s": rec and round(rec[0] - rel, 2),
                                        "nose_level_after_release_s": level and round(level[0] - rel, 2),
                                        "altitude_lost_m": round(float(np.interp(ts, h["t"], h["z"]) - h["z"][h["t"] >= ts].min()), 1),
                                        "alpha_stall_model_deg": round(math.degrees(C["alpha_stall"]), 1)}

    # 8. closed loop: a 60 degree banked level turn held 10 s (autopilot), full throttle
    sim.start(V, 80.0)
    ap = Autopilot(sim)
    h = run(sim, 10.0, lambda t, s: ap(phi_c=math.radians(60.0), theta_c=0.0, throttle=1.0))
    k = h["t"] > 4.0
    rate = float(np.median(np.abs(np.gradient(np.unwrap(np.radians(h["psi"][k])), h["t"][k]))))
    Vm = float(np.median(h["V"][k]))
    out["turn_60deg_closed_loop"] = {"turn_rate_deg_s": round(math.degrees(rate), 1), "radius_m": round(Vm / rate, 1) if rate else None,
                                     "speed_m_s": round(Vm, 2), "load_factor": round(float(np.median(h["n"][k])), 2),
                                     "altitude_change_m": round(float(h["z"][-1] - 80.0), 1)}
    out["verdict"] = {
        "statically_stable_pitch": bool(C["Cma"] < 0), "short_period_damped": out["linear_modes"].get("short_period", {}).get("stable"),
        "dutch_roll_zeta": out["linear_modes"].get("dutch_roll", {}).get("zeta"),
        "spiral_time_to_double_s": out["linear_modes"].get("spiral", {}).get("t_half_or_double_s"),
        "recovers_from_stall_hands_off": bool(out["stall_and_release"].get("unstalled_after_release_s") and not out["stall_and_release"].get("departed"))}
    return out


# --------------------------------------------------------------------------------------------------- manoeuvres
MANOEUVRES = ("break_turn", "loop", "split_s", "aileron_roll", "stall_recovery", "chase")


ENTRY = {"break_turn": (28.0, 80.0), "loop": (34.0, 60.0), "split_s": (22.0, 160.0), "aileron_roll": (22.0, 60.0),
         "stall_recovery": (18.0, 100.0), "chase": (20.0, 60.0)}           # entry speed [m/s], altitude [m]


def manoeuvre(name, sim, record=None, V=None):
    """Fly one manoeuvre under the autopilot from level flight at its entry speed; returns its history and a summary.
    The load factor commanded is capped at 90 % of what CL_max gives at the moment's speed."""
    V0, h0 = ENTRY[name]
    V = V or V0
    sim.start(V, h0)
    ap = Autopilot(sim)
    st = {"pitch": 0.0, "phase": 0}
    if name == "break_turn":           # full throttle, 4.5 g asked (the limiter caps it), banked for a level turn at that g
        def fn(t, s):
            V_ = s.state()["V"]
            n = min(4.5, 0.9 * 0.5 * am.RHO * V_ * V_ * s.C["S"] * s.C["CL_MAX"] / (s.C["mass"] * am.G))
            ap(phi_c=math.acos(1 / max(n, 1.05)) if t < 6.5 else 0.0, n_c=n if 0.3 < t < 6.5 else None, theta_c=0.0, throttle=1.0)
        T = 9.0
    elif name == "loop":               # pull 4 g until a full turn of pitch
        def fn(t, s):
            st["pitch"] += s.state()["omega_frd"][1] * DT * 5
            if t < 1.5:
                ap(theta_c=0.0, throttle=1.0, V_c=24.0)
            elif st["pitch"] < 2 * math.pi - 0.3:
                ap(phi_c=0.0, n_c=4.0, throttle=1.0)
            else:
                ap(theta_c=0.0, V_c=V)
        T = 10.0
    elif name == "split_s":            # half roll inverted, pull through half a loop
        def fn(t, s):
            st["pitch"] += s.state()["omega_frd"][1] * DT * 5 if t > 1.0 else 0.0
            if t < 1.0:
                ap(phi_c=math.pi, n_c=0.0, throttle=0.3)
                st["pitch"] = 0.0
            elif st["pitch"] < math.pi - 0.2:
                ap(phi_c=math.pi, n_c=4.5, throttle=0.0)
            else:
                ap(theta_c=0.0, V_c=V)
        T = 8.0
    elif name == "aileron_roll":       # full aileron for one turn, a little nose up first
        def fn(t, s):
            st["pitch"] += s.state()["omega_frd"][0] * DT * 5 if t > 1.0 else 0.0
            if t < 1.0:
                ap(theta_c=math.radians(15.0), throttle=1.0)
            elif st["pitch"] < 2 * math.pi - 0.4:
                s.controls[1] = am.DA_MAX
            else:
                ap(theta_c=0.0)
        T = 5.0
    elif name == "stall_recovery":     # idle, nose held 12 deg up until the wing stalls, 1 s more, then the autopilot recovers
        def fn(t, s):
            if st.get("t_stall") is None or t - st["t_stall"] < 1.0:
                ap(theta_c=math.radians(12.0), throttle=0.0, protect=False)
                if st.get("t_stall") is None and s.info.get("stall", 0) > 0.5:
                    st["t_stall"] = t
            else:
                ap(theta_c=math.radians(-10.0) if s.state()["V"] < 15.0 else 0.0, throttle=1.0)
        T = 10.0
    elif name == "chase":              # a bandit zig-zags at 14 m/s; pure pursuit (bank toward it, height to it)
        bid = mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_BODY, "bandit")
        mid = sim.m.body_mocapid[bid] if bid >= 0 else -1
        def fn(t, s):
            b = np.array([40.0 + 14.0 * t, 25.0 * math.sin(0.5 * t), 60.0 + 8.0 * math.sin(0.3 * t)])
            if mid >= 0:
                s.d.mocap_pos[mid] = b
            sd = s.state()
            bv = np.array([14.0, 12.5 * math.cos(0.5 * t), 2.4 * math.cos(0.3 * t)])
            rel0 = b - sd["pos"]
            t_go = np.linalg.norm(rel0) / max(sd["V"], 5.0)                   # lead pursuit: aim where it will be
            rel = rel0 + bv * t_go
            psi_c = math.atan2(rel[1], rel[0])
            e = (psi_c - sd["psi"] + math.pi) % (2 * math.pi) - math.pi
            st.setdefault("dist", []).append(float(np.linalg.norm(rel0)))
            ap(phi_c=max(-1.2, min(1.2, -2.5 * e)), theta_c=math.atan2(rel[2], math.hypot(rel[0], rel[1])), throttle=1.0)
        T = 20.0
    else:
        raise ValueError(name)
    h = run(sim, T, fn, record)
    summary = {"entry_speed_m_s": V, "duration_s": round(float(h["t"][-1]), 2), "peak_load_factor": round(float(h["n"].max()), 2), "min_load_factor": round(float(h["n"].min()), 2),
               "max_alpha_deg": round(float(h["alpha"].max()), 1), "min_speed_m_s": round(float(h["V"].min()), 2), "max_speed_m_s": round(float(h["V"].max()), 2),
               "altitude_change_m": round(float(h["z"][-1] - h["z"][0]), 1), "min_altitude_m": round(float(h["z"].min()), 1),
               "stalled": bool(h["stall"].max() > 0.5), "max_roll_rate_deg_s": round(float(np.abs(h["p"]).max()), 0)}
    if "dist" in st:
        d = np.array(st["dist"])
        summary.update({"bandit_start_m": round(float(d[0]), 1), "bandit_closest_m": round(float(d.min()), 1),
                        "time_to_within_5m_s": (round(float(h["t"][np.argmax(d < 5.0)]), 2) if (d < 5.0).any() else None)})
    if name == "break_turn":
        k = (h["t"] > 2.0) & (h["t"] < 6.0)
        rate = np.abs(np.gradient(np.unwrap(np.radians(h["psi"][k])), h["t"][k]))
        summary["turn_rate_deg_s"] = round(float(np.degrees(np.median(rate))), 1)
        summary["radius_m"] = round(float(np.median(h["V"][k]) / np.median(rate)), 1)
    return h, summary


def export_mesh(folder, variant="hybrid", propulsion="pusher") -> str:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"arigo1_{variant}_{propulsion}.stl"
    if not path.exists():
        a1.Arigo1().generate(**a1.variant(variant, propulsion), show_props=True).export_stl(str(path), tolerance=0.5)
    return str(path)
