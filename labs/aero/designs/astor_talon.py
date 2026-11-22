"""ASTOR-T's talons: two legs under the belly, built as a bird's, that strike the bandit and hold it (2026-11-22).

**How it is meant to work.** ASTOR comes at the bandit from behind and above; just before the pass the two legs are
thrown down and forward from the belly (``talon_deploy``). Each is three bones on sprung, damped joints (thigh,
shin, shank) and a foot of three toes forward and one back with curved claws. The feet take the bandit and snap shut
(a spring let go by the touch; a ratchet holds them, as a raptor's tendon lock does); the knees and ankles fold
under the blow, which is the leg's give; each hip can swing its foot sideways in the last instant. The strike is the
end of the flight: ASTOR cuts its motor and opens its parachute, and the two come down together, the bandit held
under it. (The first version, 2026-11-22, was one arm with a rake of hooks: less travel, no grip, no last correction.)

**What is computed here.**

- ``strike_loads``: two bodies that stick: the energy the arm must take (the reduced mass at the closing speed), the
  mean force over the arm's travel, the hinge moment, the acceleration of each, ASTOR's pitch rate after a strike
  below its centre of gravity, the sink rate of the pair under ASTOR's parachute.
- ``strike``: one sortie in ``astor_v_flight.SimV``: proportional navigation with the tracker of ``astor_netdrop``,
  aimed so the rake (not the fuselage) passes through the bandit; what is measured is how far the rake's centre
  passes from the bandit and at what closing speed. A catch: inside the rake's reach at no more than the arm's
  closing-speed limit.

Bandit masses, the arm's and the parachute's numbers are HAND estimates, marked.
"""
from __future__ import annotations

import math

import numpy as np

import arigo1_model as am
import arigo1_mujoco as amj
import astor_netdrop as ND
import astor_t as T
import astor_v_flight as VF

ARM_G = 200.0                      # HAND: the two legs (astor_t's budget)
STROKE_FRACTION = 0.6              # the thrown leg shortens by this share of its length as knee and ankle fold under the strike
STEER_DEG = 20.0                   # each hip can swing the foot this far sideways in the last instant (the bird's own correction)
REACH_Z_M = 0.18                   # a catch in height: the open foot's depth and the legs' give (HAND)
BANDIT_HALF_M = 0.15               # the bandit's own half width (HAND)
TERMINAL_M, TERMINAL_KZ, TERMINAL_VZ = 150.0, 1.5, 5.0     # the last stretch: within this range; climb at 1.5 /s of the height error, 5 m/s at most
PASS_SPEED = 1.3                   # the pass over a slow bandit, x the spread stall. Slower does not fly: the flight model's lift
                                   # falls away 2 deg before the nominal stall angle, where the angle-of-attack limiter also sits, so
                                   # at 1.15 x stall ASTOR sat on the limiter and sank 5 m under the bandit (2026-11-22)
V_CLOSE_MAX = 18.0                 # the legs' design closing speed [m/s]: 1.3 x stall (24 m/s) on a 7 m/s bandit is 17
ANGLE_NOISE = 0.005                # the camera places the bandit to this angle [rad] (HAND): 0.1 m at 20 m; never worse than noise_m
BANDITS_KG = {"armageddon": 1.5, "quad": 1.5, "fast": 4.0}       # HAND: a small multirotor; a 50 m/s fixed-wing drone
CHUTE_CDA = 0.8 * 1.3              # ASTOR's 0.8 m2 canopy (astor.py), Cd 1.3 (HAND +-0.3)


def leg_drop(p) -> float:
    """The thrown feet below the hips [m] (the three bones at astor_t's thrown angles)."""
    return sum(L * math.cos(math.radians(a)) for L, a in zip((p["leg_thigh"], p["leg_shin"], p["leg_shank"]), (35.0, -12.0, 28.0))) / 1000.0


def reach_across(p) -> float:
    """How far to the side of the centre plane the bandit's centre may pass and still be taken: a hip's offset, the open
    foot's half span, the bandit's half width, and what the hip can swing the foot."""
    return (p["leg_y"] + p["toe_length"] * math.sin(math.radians(32.0))) / 1000.0 + BANDIT_HALF_M + leg_drop(p) * math.sin(math.radians(STEER_DEG))


def strike_loads(v_close, bandit_kg, p=None, mass_kg=None) -> dict:
    p = p or T.resolve()
    m = mass_kg if mass_kg is not None else T.budget(p)["total_g"] / 1000.0          # the arm is in ASTOR-T's budget
    mu = m * bandit_kg / (m + bandit_kg)
    E = 0.5 * mu * v_close ** 2
    L = leg_drop(p)
    travel = L * STROKE_FRACTION
    F = E / travel
    r = p["fuselage_diameter"] / 2000.0 + 0.008 + L                   # the feet below the axis
    Iyy = float(VF.get("t", 12.0)["inertia"][1])
    q = mu * v_close * r / Iyy                                           # the strike's impulse at the arm's depth
    both = m + bandit_kg
    sink = lambda mm: math.sqrt(2 * mm * am.G / (am.RHO * CHUTE_CDA))
    return {"closing_m_s": v_close, "bandit_kg": bandit_kg, "astor_kg": round(m, 2), "energy_J": round(E, 0), "arm_travel_mm": round(travel * 1000),
            "mean_force_N": round(F, 0), "force_per_leg_N": round(F / 2, 0), "astor_g": round(F / m / am.G, 1), "bandit_g": round(F / bandit_kg / am.G, 1),
            "rake_below_axis_mm": round(r * 1000), "pitch_rate_after_deg_s": round(math.degrees(q)), "speed_lost_m_s": round(mu * v_close / m, 1),
            "sink_under_parachute_m_s": {"astor_alone": round(sink(m), 1), "with_the_bandit": round(sink(both), 1)},
            "flies_on": False, "note": "the pitch rate and the load under the belly end the flight: motor cut, parachute"}


class CruisingQuad:
    """A quadcopter flying straight and level, not evading: the easiest target (speed, height, start ahead)."""

    def __init__(self, speed=9.0, height=40.0, ahead=220.0, offset=0.0, heading_deg=0.0):
        h = math.radians(heading_deg)
        self.p = [ahead, offset, height]
        self.v = (speed * math.cos(h), speed * math.sin(h), 0.0)
        self.caught = None

    def step(self, dt, t, hunter=None):
        if self.caught is None:
            for i in range(3):
                self.p[i] += self.v[i] * dt


def strike(name="t", bandit_kind="armageddon", seed=0, latency_s=0.10, noise_m=0.3, duration_s=None, record=None) -> dict:
    """One sortie: every pass's miss (the rake's centre from the bandit, across and in height) and closing speed."""
    rng = np.random.default_rng(seed)
    fast = bandit_kind == "fast"
    p = T.resolve()
    drop = p["fuselage_diameter"] / 2000.0 + 0.008 + leg_drop(p)
    REACH_Y_M = reach_across(p)
    sim = VF.SimV(name, bandit=True)
    Vs = VF.stall(VF.get(name, VF.SWEEPS[0]))
    if fast:
        bandit = ND.FastBandit(seed=seed)
        sim.start(60.0, bandit.p[2])
    elif bandit_kind == "quad":
        bandit = CruisingQuad(offset=float(rng.uniform(-25, 25)), heading_deg=float(rng.uniform(-20, 20)))
        sim.start(1.4 * Vs, bandit.p[2] + 6.0)
    else:
        bandit = ND.ArmageddonBandit(centre=(250.0 + rng.uniform(-30, 30), rng.uniform(-40, 40)), seed=seed)
        sim.start(1.4 * Vs, 30.0, heading_deg=float(rng.uniform(-15, 15)))
    T_end = duration_s or (45.0 if fast else 90.0)
    ap = amj.Autopilot(sim)
    cam = ND.Camera(latency_s, noise_m, seed, window_s=1.5 if fast else 0.5, turning=fast)
    import mujoco
    bid = mujoco.mj_name2id(sim.m, mujoco.mjtObj.mjOBJ_BODY, "bandit")
    mid = sim.m.body_mocapid[bid] if bid >= 0 else -1
    passes, best, was_closing, caught, min_alt, phase, t_phase = [], None, False, None, 1e9, "attack", 0.0
    alpha_f = 0.0
    t0 = sim.d.time
    while sim.d.time - t0 < T_end and caught is None:
        t = sim.d.time - t0
        s = sim.state()
        if mid >= 0:
            sim.d.mocap_pos[mid] = bandit.p
        cam.sig = min(noise_m, ANGLE_NOISE * math.dist(bandit.p, s["pos"]))      # close in, the camera sees it better
        b, vb = cam.see(t, bandit.p, bandit.v)
        sb = float(np.linalg.norm(vb[:2]))
        aim = np.asarray(b) + np.array([0.0, 0.0, drop])                 # the fuselage passes the arm's depth above the bandit
        r = aim - s["pos"]
        rng_h = math.hypot(r[0], r[1])
        if phase == "attack":
            phi, gam = amj.pn_command(sim, aim, vb)
            n_av = 0.5 * am.RHO * s["V"] ** 2 * sim.C["S"] * sim.C["CL_MAX"] / (sim.C["mass"] * am.G)
            phi_max = math.acos(min(1.0, 1.0 / max(0.85 * n_av, 1.0001)))
            low = s["pos"][2] + 2.0 * min(s["v_world"][2], 0.0) < ND.FLOOR_M
            # the pass is flown as slowly as the wing allows over a slow bandit; a fast one is overtaken at +8 m/s
            V_c = min(72.0, sb + (8.0 if rng_h < 150.0 else 20.0)) if fast else max(PASS_SPEED * Vs, sb + min(14.0, 4.0 + 0.2 * rng_h))
            if rng_h < TERMINAL_M:
                # the last stretch: the height is flown as a climb rate in proportion to the height error (a flight path
                # angle, plus the angle of attack being flown). Pointing the nose along the line of sight, as the chase
                # does, passed 3 m low every time: at this speed the wing flies 6 deg nose-up.
                alpha_f += (float(sim.info.get("alpha", 0.0)) - alpha_f) * 0.01 / 0.5
                vz = max(-TERMINAL_VZ, min(TERMINAL_VZ, TERMINAL_KZ * float(r[2])))
                gam = max(-0.3, min(0.3, vz / max(s["V"], 10.0) + alpha_f))
            else:
                alpha_f = float(sim.info.get("alpha", 0.0))
            # the throttle holds the speed AND the height: the autopilot's own loop holds the speed alone, and over a
            # slow bandit (the speed wanted below the speed flown) it shut the motor and glided into the ground
            thr = 1.0 if low else max(0.0, min(1.0, sim.trim["throttle"] + 0.15 * (V_c - s["V"]) + 0.06 * float(r[2])))
            ap(phi_c=0.0 if low else max(-phi_max, min(phi_max, phi)), theta_c=math.radians(12.0) if low else gam, V_c=V_c, throttle=thr)
        else:
            ap(phi_c=math.radians(35.0), theta_c=math.radians(8.0), V_c=max(1.4 * Vs, sb + 8.0), throttle=0.9)
            if t - t_phase > (4.0 if not fast else 3.0):
                phase = "attack"
        sim.step(5)
        bandit.step(0.01, t, hunter=tuple(s["pos"]))
        s = sim.state()
        min_alt = min(min_alt, float(s["pos"][2]))
        # the rake's centre: the arm's depth under the fuselage; the miss in the plane across the relative velocity
        rel = np.asarray(bandit.p) - (np.asarray(s["pos"]) - np.array([0.0, 0.0, drop]))
        vrel = np.asarray(s["v_world"]) - np.asarray(bandit.v)
        vc = float(np.linalg.norm(vrel))
        closing = float(rel @ vrel) > 0.0
        d = float(np.linalg.norm(rel))
        if closing and (best is None or d < best["d"]) and d < 30.0:
            u = vrel / max(vc, 1e-6)
            perp = rel - (rel @ u) * u
            side = np.cross(np.array([0.0, 0.0, 1.0]), u)
            side /= max(float(np.linalg.norm(side)), 1e-6)
            best = {"d": d, "t": round(t, 2), "across_m": round(float(perp @ side), 2), "height_m": round(float(perp[2]), 2),
                    "closing_m_s": round(vc, 1), "astor_m_s": round(s["V"], 1), "bank_deg": round(math.degrees(s["phi"]), 0)}
        if was_closing and not closing and best is not None:               # the pass is over: its closest point stands
            hit = abs(best["across_m"]) <= REACH_Y_M and abs(best["height_m"]) <= REACH_Z_M and best["closing_m_s"] <= V_CLOSE_MAX
            body = abs(best["across_m"]) <= 0.25 and REACH_Z_M < best["height_m"] < drop + 0.1          # the bandit above the feet: it met the legs or the fuselage
            best.pop("d")
            passes.append({**best, "caught": bool(hit), "hit_the_fuselage": bool(body)})
            if hit:
                caught = passes[-1]
            best, phase, t_phase = None, "away", t
        was_closing = closing
        if record is not None:
            record(sim, bandit, t)
        if s["pos"][2] < 1.0:
            break
    miss = [math.hypot(q["across_m"], q["height_m"]) for q in passes]
    return {"design": name, "bandit": bandit_kind, "seed": seed, "caught": caught is not None, "passes": len(passes), "t_catch_s": None if caught is None else caught["t"],
            "best_miss_m": round(min(miss), 2) if miss else None, "median_miss_m": round(float(np.median(miss)), 2) if miss else None,
            "closing_m_s": [q["closing_m_s"] for q in passes], "fuselage_hits": sum(1 for q in passes if q["hit_the_fuselage"]),
            "min_altitude_m": round(min_alt, 1), "crashed": bool(min_alt < 1.0), "pass_list": passes}


def checks() -> dict:
    p = T.resolve()
    return {"legs_g": ARM_G, "feet_below_axis_mm": round((p["fuselage_diameter"] / 2000.0 + 0.008 + leg_drop(p)) * 1000),
            "reach": {"across_m": round(reach_across(p), 3), "of_which_hip_steering_m": round(leg_drop(p) * math.sin(math.radians(STEER_DEG)), 3),
                      "height_m": REACH_Z_M, "closing_limit_m_s": V_CLOSE_MAX},
            "strike": {f"{k} bandit, {v:.0f} m/s": strike_loads(v, BANDITS_KG[k]) for k, v in (("armageddon", 14.0), ("armageddon", 17.0), ("quad", 15.0), ("fast", 8.0))}}
