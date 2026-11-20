"""RAPTORS robustness models (the ARIGO product lines BUTEO, ARPELLA, ASTOR, SPARFELL): launch success under
dispersion (KPIs B4, S8, and SPARFELL's throw) and the operating wind (B6), on the MuJoCo models with wind and
turbulence. See ``reports/arigo_catalog/KPIS.md``.

**Wind.** A steady wind W from a direction, plus turbulence: on each axis a first-order Gauss-Markov process (a
Dryden-like spectrum; time constant L/V, L = TURB_L_M; sigma per axis), the vertical at TURB_W_SHARE of the
horizontal. The gust ratio follows the KPI's 12/18 m/s envelope: a 3-sigma gust peak = GUST_RATIO x W.

**Launch trials** (one per draw), flown as the cards' launches (``arigo1_mujoco.takeoff``, ``arigo_mission.perch_drop``):

- *perch*: the spring push-off (speed +-15 %), the perch's nose-down attitude +-5 deg, a fixed perch: the wind
  from any direction (unless the station turns into the wind: ``turning_station``);
- *rail*: the rail's exit ground speed 1.3 x calm stall +-5 %, pitch +-2 deg, a fixed rail: the wind from any
  direction (unless it turns);
- *hand*: thrown into the wind (+-30 deg), 12 +-2 m/s over the ground, +-5 deg pitch, release 1.8 m.

A trial **succeeds** when it reaches safe flight (10 m, or the perch's height, at 1.3 x stall airspeed) within 12 s
without touching the ground (0.3 m) or rolling past 60 deg.

**Recovery trials**: a belly landing flown into the wind (+-20 deg) on the landing sim's 6-deg path at 1.3 x stall
plus half the wind, or a parachute
descent (HAND sink and aiming error). A landing **succeeds** with a touchdown energy (ground speed) under 300 J, a
miss under 30 m, a sink under 3 m/s and a bank under 20 deg at touchdown.

**Operating wind**: the top of the operating band — from the lightest wind on the grid at which launch and recovery
both succeed >= 95 % (calm air can fail: a throw slower than the stall, a heavy landing with no headwind) up through
every stronger wind that also does; interpolated at the top. Calm-air success is reported on its own. **Launch success** (B4/S8): over the KPI's envelope, the steady wind
drawn uniformly in 0-12 m/s with its gusts.

Every dispersion value is HAND (marked); samples per point are a run option (statistical error ~ sqrt(p(1-p)/n)).
"""
from __future__ import annotations

import math

import numpy as np

import arigo1_model as am
import arigo1_mujoco as amj

TURB_L_M = 50.0                    # HAND: turbulence length scale near the ground (Dryden low altitude ~ h)
TURB_W_SHARE = 0.6                 # HAND: vertical sigma / horizontal sigma
GUST_RATIO = 1.5                   # gust peak / steady wind: the KPI's 12 / 18 m/s
HAND_SPEED = (12.0, 2.0)           # HAND: throw ground speed mean, sd
HAND_DIR_DEG = 30.0                # HAND: throw direction off the wind (uniform +-)
PUSH_SD = 0.15                     # HAND: spring push-off speed, relative sd
RAIL_SD = 0.05                     # HAND: rail exit speed, relative sd
LAND_DIR_DEG = 20.0                # HAND: approach direction off the wind (uniform +-)
CHUTE_DEPLOY_M = 30.0              # HAND: parachute deployment height
CHUTE_AIM_ERR = 0.3                # HAND: the share of the wind drift left after aiming (sd)
SUCCESS = {"energy_J": 300.0, "miss_m": 30.0, "sink_m_s": 3.0, "bank_deg": 20.0}
WINDS = (0.0, 4.0, 8.0, 12.0, 16.0, 20.0)


class Turbulence:
    """Steady wind + Gauss-Markov turbulence; ``__call__(dt, V)`` returns the wind vector [m/s, world]."""

    def __init__(self, rng, W, heading_from_deg):
        self.rng, self.W = rng, W
        a = math.radians(heading_from_deg)                 # the direction the wind blows TO, in the world frame
        self.steady = np.array([W * math.cos(a), W * math.sin(a), 0.0])
        self.sig = (GUST_RATIO - 1.0) * W / 3.0 * np.array([1.0, 1.0, TURB_W_SHARE])
        self.x = self.rng.normal(0.0, 1.0, 3) * self.sig

    def __call__(self, dt, V):
        tau = TURB_L_M / max(V, 3.0)
        self.x += -self.x / tau * dt + self.sig * math.sqrt(2.0 * dt / tau) * self.rng.normal(0.0, 1.0, 3)
        return self.steady + self.x


def _Vs(C):
    return math.sqrt(C["mass"] * am.G / (0.5 * am.RHO * C["S"] * C["CL_MAX"]))


def _set_attitude(sim, pitch_deg):
    import mujoco
    th = math.radians(-pitch_deg)                          # + pitch = nose up; the quaternion's y rotation is nose-down positive
    sim.d.qpos[3:7] = (math.cos(th / 2), 0.0, math.sin(th / 2), 0.0)
    sim.d.qvel[3:] = 0.0
    mujoco.mj_forward(sim.m, sim.d)


def launch_trial(sim, mode, rng, W, turning_station=False, perch_push=8.0, perch_h=15.0) -> dict:
    """One launch in wind W (with gusts); returns success and why."""
    import mujoco
    import arigo_mission as M
    C = sim.C
    Vs = _Vs(C)
    if mode == "hand":
        wind_to = 180.0 + rng.uniform(-HAND_DIR_DEG, HAND_DIR_DEG)          # thrown into the wind (wind blows toward -x)
        Vg, pitch, h0 = rng.normal(*HAND_SPEED), rng.normal(0.0, 5.0), amj.RELEASE_HEIGHT_M
    elif mode == "rail":
        wind_to = 180.0 if turning_station else rng.uniform(0.0, 360.0)
        Vg, pitch, h0 = 1.3 * Vs * rng.normal(1.0, RAIL_SD), 8.0 + rng.normal(0.0, 2.0), amj.RELEASE_HEIGHT_M
    else:                                                                   # perch
        wind_to = 180.0 if turning_station else rng.uniform(0.0, 360.0)
        Vg = perch_push * rng.normal(1.0, PUSH_SD)
        pitch, h0 = -M.PERCH_NOSE_DOWN_DEG + rng.normal(0.0, 5.0), perch_h
    sim.start(max(1.15 * Vs, 8.0), h0)
    _set_attitude(sim, pitch)
    th = math.radians(pitch)
    sim.d.qvel[:3] = (Vg * math.cos(th), 0.0, Vg * math.sin(th))
    mujoco.mj_forward(sim.m, sim.d)
    turb = Turbulence(rng, W, wind_to)
    sim.wind[:] = turb(0.0, Vg)
    ap = amj.Autopilot(sim)
    target_h = perch_h if mode == "perch" else amj.SAFE_HEIGHT_M
    st = {"pulled": mode != "perch", "t_safe": None, "why": None}
    t0 = sim.d.time
    while sim.d.time - t0 < 12.0:
        s = sim.state()
        sim.wind[:] = turb(0.01, s["V"])
        if not st["pulled"] and s["V"] >= 1.2 * Vs:
            st["pulled"] = True
        if mode == "perch" and not st["pulled"]:
            ap(phi_c=0.0, theta_c=math.radians(pitch), throttle=1.0)
        else:
            ap(phi_c=0.0, theta_c=math.radians(12.0) if s["V"] > 1.25 * Vs else 0.0, throttle=1.0)
        sim.step(5)
        s = sim.state()
        if s["pos"][2] < 0.3:
            st["why"] = "ground"
            break
        if abs(math.degrees(s["phi"])) > 60.0:
            st["why"] = "upset"
            break
        if s["pos"][2] >= target_h and s["V"] >= 1.3 * Vs and st["pulled"] and (mode != "perch" or sim.d.time - t0 > 0.5):
            st["t_safe"] = sim.d.time - t0
            break
    sim.wind[:] = 0.0
    ok = st["t_safe"] is not None
    return {"ok": ok, "why": None if ok else (st["why"] or "slow"), "t_safe_s": st["t_safe"]}


def landing_trial(sim, recovery, rng, W) -> dict:
    """One recovery in wind W (with gusts); returns success, why and the touchdown values."""
    import arigo_mission as M
    import arigo_products as AP
    C = sim.C
    m = C["mass"]
    if recovery == "parachute":
        t_desc = CHUTE_DEPLOY_M / AP.PARACHUTE_SINK_M_S
        gust = abs(rng.normal(0.0, (GUST_RATIO - 1.0) * W / 3.0))
        miss = abs(rng.normal(0.0, CHUTE_AIM_ERR)) * W * t_desc + abs(rng.normal(0.0, 3.0))
        vg = math.hypot(AP.PARACHUTE_SINK_M_S, W + gust)                    # the canopy drifts with the wind
        E = 0.5 * m * vg * vg
        ok = E <= SUCCESS["energy_J"] and miss <= SUCCESS["miss_m"]
        return {"ok": ok, "why": None if ok else ("energy" if E > SUCCESS["energy_J"] else "miss"),
                "energy_J": round(E, 0), "miss_m": round(miss, 1)}
    Vs = _Vs(C)
    Va = 1.3 * Vs + 0.5 * W                                                 # the pilot's rule: half the wind on the approach speed
    gs = math.radians(M.GLIDE_SLOPE_DEG)
    aim = M.APPROACH_HEIGHT_M / math.tan(gs)
    sim.start(Va, M.APPROACH_HEIGHT_M)
    sim.d.qvel[:3] = (max(Va - W, 2.0), 0.0, 0.0)                           # the approach into the wind: ground speed
    turb = Turbulence(rng, W, 180.0 + rng.uniform(-LAND_DIR_DEG, LAND_DIR_DEG))
    ap = amj.Autopilot(sim)
    td, flare = None, False
    t0 = sim.d.time
    while sim.d.time - t0 < 150.0:
        s = sim.state()
        sim.wind[:] = turb(0.01, s["V"])
        x, z = s["pos"][0], s["pos"][2]
        if td is None and z < 0.12:
            td = {"V": float(np.linalg.norm(s["v_world"])), "vz": float(s["v_world"][2]), "x": x, "y": float(s["pos"][1]),
                  "bank": abs(math.degrees(s["phi"]))}
            break
        if z > M.FLARE_HEIGHT_M and not flare:
            h_path = max(0.0, (aim - x) * math.tan(gs))
            # the path over the ground, the heading held on the runway line (y -> 0)
            phi_c = max(-0.35, min(0.35, 0.02 * s["pos"][1] + 0.8 * s["v_world"][1] / max(s["V"], 3.0)))   # y left, + bank turns right
            ap(phi_c=phi_c, theta_c=-gs + 0.08 * (h_path - z) + math.radians(sim.trim["alpha_deg"]), V_c=Va)
        else:
            flare = True
            vz_c = -min(2.2, 0.5 + 0.08 * max(0.0, x - (aim - 10.0)))    # touchdown-point control: past the aim, sink faster
            ap(phi_c=0.0, theta_c=math.radians(2.0) + 0.15 * (vz_c - s["v_world"][2]), throttle=0.0)
        sim.step(5)
        if abs(math.degrees(sim.state()["phi"])) > 60.0:
            td = {"V": float(np.linalg.norm(sim.state()["v_world"])), "vz": -9.0, "x": x, "y": 0.0, "bank": 90.0}
            break
    sim.wind[:] = 0.0
    if td is None:
        return {"ok": False, "why": "no touchdown"}
    E = 0.5 * m * td["V"] ** 2
    miss = math.hypot(td["x"] - aim, td["y"])
    checks = {"energy": E <= SUCCESS["energy_J"], "miss": miss <= SUCCESS["miss_m"], "sink": -td["vz"] <= SUCCESS["sink_m_s"],
              "bank": td["bank"] <= SUCCESS["bank_deg"]}
    ok = all(checks.values())
    return {"ok": ok, "why": None if ok else next(k for k, v in checks.items() if not v), "energy_J": round(E, 0),
            "miss_m": round(miss, 1), "sink_m_s": round(-td["vz"], 2), "bank_deg": round(td["bank"], 1)}


def _rate(trials):
    n = len(trials)
    k = sum(1 for t in trials if t["ok"])
    why = {}
    for t in trials:
        if not t["ok"]:
            why[t["why"]] = why.get(t["why"], 0) + 1
    p = k / n if n else 0.0
    return {"rate": round(p, 3), "n": n, "sd": round(math.sqrt(max(p * (1 - p), 1e-9) / max(n, 1)), 3), "fails": why}


def robustness(sim_factory, C, launch_mode, recovery, n_launch=24, n_land=8, winds=WINDS, seed=0,
               turning_station=False, perch_push=8.0, perch_h=15.0, envelope_max=12.0) -> dict:
    """Launch success over the envelope (0-envelope_max m/s with gusts), and per wind: launch and recovery success;
    the operating wind (both >= 95 %). ``launch_mode``: perch / rail / hand; ``recovery``: belly / parachute."""
    rng = np.random.default_rng(seed)
    env = [launch_trial(sim_factory(), launch_mode, rng, rng.uniform(0.0, envelope_max), turning_station, perch_push, perch_h)
           for _ in range(n_launch)]
    per = []
    for W in winds:
        L = [launch_trial(sim_factory(), launch_mode, rng, W, turning_station, perch_push, perch_h) for _ in range(n_launch // 2)]
        R = [landing_trial(sim_factory(), recovery, rng, W) for _ in range(n_land)]
        per.append({"wind_m_s": W, "gust_m_s": round(GUST_RATIO * W, 1), "launch": _rate(L), "recovery": _rate(R),
                    "both": round(_rate(L)["rate"] * _rate(R)["rate"], 3)})
        if per[-1]["launch"]["rate"] < 0.5 and per[-1]["recovery"]["rate"] < 0.5:
            break                                              # far past the limit: stronger winds only cost time
    # the operating band: from the lightest wind that works (calm air can fail: a slow throw, a heavy landing) up
    # through every wind that works; its top is the operating wind
    good = [p["launch"]["rate"] >= 0.95 and p["recovery"]["rate"] >= 0.95 for p in per]
    lo = next((i for i, g in enumerate(good) if g), None)
    op, w_lo = 0.0, None
    for a, b in zip(per[lo:] if lo is not None else [], (per[lo + 1:] if lo is not None else []) + [None]):
        ok_a = a["launch"]["rate"] >= 0.95 and a["recovery"]["rate"] >= 0.95
        if not ok_a:
            break
        w_lo = a["wind_m_s"] if w_lo is None else w_lo
        op = a["wind_m_s"]
        if b is not None:
            ok_b = b["launch"]["rate"] >= 0.95 and b["recovery"]["rate"] >= 0.95
            if not ok_b:                                       # interpolate the crossing of 95 % on the weaker of the two
                fa = min(a["launch"]["rate"], a["recovery"]["rate"])
                fb = min(b["launch"]["rate"], b["recovery"]["rate"])
                if fa > fb:
                    op = a["wind_m_s"] + (fa - 0.95) / (fa - fb) * (b["wind_m_s"] - a["wind_m_s"])
    return {"launch_mode": launch_mode, "recovery": recovery, "turning_station": turning_station,
            "launch_success_envelope": _rate(env), "per_wind": per,
            "operating_wind_m_s": round(op, 1), "operating_gust_m_s": round(GUST_RATIO * op, 1),
            "lightest_working_wind_m_s": w_lo,
            "calm": {"launch": per[0]["launch"]["rate"], "recovery": per[0]["recovery"]["rate"]},
            "samples": {"launch_envelope": n_launch, "launch_per_wind": n_launch // 2, "recovery_per_wind": n_land, "seed": seed},
            "rule": "success: safe flight in 12 s (launch); energy < 300 J, miss < 30 m, sink < 3 m/s, bank < 20 deg (recovery); "
                    "operating wind: both >= 95 %; gusts 1.5 x steady (3-sigma)"}


def summary(rb) -> dict:
    return {"launch_success_pct": round(100 * rb["launch_success_envelope"]["rate"], 1),
            "operating_wind_m_s": rb["operating_wind_m_s"], "operating_gust_m_s": rb["operating_gust_m_s"],
            "recovery_success_calm_pct": round(100 * rb["per_wind"][0]["recovery"]["rate"], 1),
            "calm_success_pct": round(100 * min(rb["per_wind"][0]["launch"]["rate"], rb["per_wind"][0]["recovery"]["rate"]), 1),
            "lightest_working_wind_m_s": rb.get("lightest_working_wind_m_s")}
