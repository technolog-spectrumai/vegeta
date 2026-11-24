"""GOSHAWK in a wood: one flight through standing trunks (2026-11-24).

The question ``goshawk.forest()`` answers on paper (is the turn's radius under the distance between stems?) is flown
here: the MuJoCo aircraft of ``astor_v_flight`` ("goshawk": its flaps, its propellers' wash, its limiter) through a
stand of trunks, steered by a short-sighted planner.

- **The stand** (``stand``): a strip ``LENGTH_M`` long and ``WIDTH_M`` wide, trunks on a square grid of the density's
  spacing (100 / sqrt(stems per hectare)), each moved at random by up to ``JITTER`` of the spacing, radius 0.12-0.28 m
  (HAND). The first ``CLEAR_M`` are open. Trunks only: no branches, no ground, no canopy; the flight is level at 4 m.
- **What the aircraft sees**: every trunk within ``SEE_M`` and inside +-``FOV_DEG`` of its heading, exactly (no camera
  model: the planner is the subject here, not the perception).
- **The planner** (``plan``): every ``PLAN_S`` it tries ``N_ARCS`` steady turns (banks from full left to full right at
  the load factor the wing gives now), follows each for ``HORIZON_S``, throws out those that pass a trunk closer than
  its radius + the wing's reach at that bank + ``MARGIN_M``, and takes the one that best keeps the way ahead (+x), the
  strip's middle, and the bank it already has. With no arc clear it takes the one that hits last.
- **The turn**: flaps down; in a steep turn the throttle goes up (the propellers' wash is lift) and the crow brake
  holds the speed.
- **A hit**: the aircraft's centre closer to a trunk's axis than the trunk's radius + the half span x |cos(bank)|
  (rolled on its side the wing passes a gap its span would not) + 0.05 m.

Outcomes: ``through`` (the far end reached), ``hit``, ``strayed`` (out of the strip's side), ``lost`` (below 1 m or stalled).
Flaps stay down (the wood is flown slowly); the wings are not drawn in (that state has no aerodynamics yet).
"""
from __future__ import annotations

import math

import numpy as np

import arigo1_model as am
import arigo1_mujoco as amj
import astor_v_flight as VF
import goshawk as GH

LENGTH_M, WIDTH_M, CLEAR_M, HEIGHT_M = 150.0, 80.0, 15.0, 4.0
JITTER, TRUNK_R_M = 0.35, (0.12, 0.28)
SEE_M, FOV_DEG = 30.0, 70.0
PLAN_S, HORIZON_S, N_ARCS, MARGIN_M = 0.05, 2.0, 11, 0.35
HOLD_S = (0.4, 0.9, 2.0)                        # a candidate holds its bank this long, then rolls level: a dodge, an S, or a turn
LIFT_SHARE, BANK_MAX_DEG = 0.85, 72.0           # the planner counts on this share of the wing's lift; the steepest bank it asks
ROLL_DEG_S = 110.0                              # how fast the bank follows its command in this simulation (seen in the first flights): the arcs are drawn with it
TURN_EFF = 0.72                                 # the track turns this share of g tan(bank) / V in a one-second bank (a scripted 60 deg bank at 14 m/s: 45 deg against 62 planned)
KEEP_BONUS = 0.8                                # the arc already being flown is kept unless another is better by this (the first planner changed sides every 0.2 s and flew straight into a trunk)
TURN_THROTTLE = 0.8                             # throttle added in a steep turn: the propellers' wash is lift; the crow brake holds the speed
KZ, VZ_MAX = 1.2, 3.0                           # the height loop [1/s, m/s]
KV, KA, = 0.25, 2.5                             # load factor per m/s of climb-rate error; elevator [deg] per deg of angle-of-attack error


def stand(stems_per_ha: float, seed: int) -> np.ndarray:
    """The trunks: rows of (x, y, radius) [m]."""
    rng = np.random.default_rng(1000 + seed)
    d = 100.0 / math.sqrt(stems_per_ha)
    xs = np.arange(CLEAR_M + d / 2, LENGTH_M + 30.0, d)
    ys = np.arange(-WIDTH_M / 2 - 2 * d, WIDTH_M / 2 + 2 * d + 1e-9, d)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    X = X + rng.uniform(-JITTER, JITTER, X.shape) * d
    Y = Y + rng.uniform(-JITTER, JITTER, Y.shape) * d
    R = rng.uniform(*TRUNK_R_M, X.shape)
    return np.stack([X.ravel(), Y.ravel(), R.ravel()], axis=1)


def reach(phi: float, half_span: float) -> float:
    """How far the aircraft reaches sideways from its centre at this bank [m]: the half span turned, never less than
    the body and a propeller."""
    return max(half_span * abs(math.cos(phi)), 0.18)


def wash_factor(C, V) -> float:
    """What the propellers' wash at full throttle adds to the wing's lift at the limiter's angle, as a factor on it
    (astor_v_flight.powered's own formula and limit)."""
    pw = C.get("powered")
    if not pw:
        return 1.0
    T = am.max_thrust(C, V)
    a_lim = C["alpha_stall"] - math.radians(C.get("alpha_margin_deg", VF.ALPHA_MARGIN_DEG))
    cl = float(np.interp(math.degrees(a_lim), C["cfd"]["alpha_deg"], C["cfd"]["CL"]))
    vj = math.sqrt(V * V + 2 * (T / 2) / (am.RHO * pw["disc_m2"]))
    dL = VF.SLIP_EFF * 0.5 * am.RHO * (vj * vj - V * V) * 2 * pw["prop_m"] * math.sqrt((V + vj) / (2 * vj)) * pw["chord_m"] * cl
    turn = min(max(a_lim + math.radians(12.0) * float(C.get("flap_share", 0.0)), 0.0), math.radians(40.0))
    dL = min(dL, T / max(vj - V, 0.5) * vj * math.sin(turn))
    return 1.0 + dL / (0.5 * am.RHO * V * V * C["S"] * max(cl, 0.1))


def pitch(sim, s, n_c, lift_factor) -> float:
    """The elevator [deg] for a load factor, by the angle of attack: the angle that gives it on the model's own lift
    table (with the wash's share), the elevator that balances the model's moment there and in the turn's pitch rate,
    and a correction on the angle's error. (The shared autopilot's pitch loop took ~2 s to build a turn's load factor on
    this aircraft: its gains are another's. In a wood the turn is over in half a second.)"""
    C = sim.C
    V = max(s["V"], 5.0)
    a_c = VF.alpha_for_g(C, V, max(n_c, 0.2) / max(lift_factor, 1.0))
    a_c = min(a_c, C["alpha_stall"] - math.radians(C.get("alpha_margin_deg", VF.ALPHA_MARGIN_DEG)))
    al = math.atan2(s["v_air_frd"][2], max(s["v_air_frd"][0], 1.0))
    q_ss = am.G / V * max(n_c - 1.0 / max(n_c, 0.5), 0.0)             # the pitch rate of the steady turn [rad/s]
    cm = float(np.interp(math.degrees(a_c), C["cfd"]["alpha_deg"], C["cfd"]["Cm"])) + C["Cmq"] * q_ss * C["c"] / (2 * V)
    k = VF.SAS.get(C.get("design"), {})
    damper = k.get("pitch_k", 0.0) * k.get("pitch_v_ref", V) / V * math.degrees(q_ss)      # what the pitch damper will add: taken out
    return max(-am.DE_MAX, min(am.DE_MAX, -cm / C["Cmde"] - damper + KA * math.degrees(al - a_c)))


def plan(x, y, psi, V, phi_now, trees, n_av, half_span, phi_prev=None):
    """The bank to fly [rad] and how clear its arc is [m] (negative: it hits)."""
    phi_max = min(math.radians(BANK_MAX_DEG), math.acos(min(1.0, 1.0 / max(LIFT_SHARE * n_av, 1.0001))))
    dx, dy = trees[:, 0] - x, trees[:, 1] - y
    dist = np.hypot(dx, dy)
    bearing = np.abs((np.arctan2(dy, dx) - psi + math.pi) % (2 * math.pi) - math.pi)
    seen = trees[(dist < SEE_M) & (bearing < math.radians(FOV_DEG))]
    ts = np.arange(0.1, HORIZON_S + 1e-9, 0.1)
    step = math.radians(ROLL_DEG_S) * 0.1
    best = None
    for hold in HOLD_S:
        for phi in np.linspace(-phi_max, phi_max, N_ARCS):
            if hold < HOLD_S[-1] and abs(phi) < 1e-6:
                continue
            # the arc as it will be flown: the bank goes to ``phi`` at ROLL_DEG_S, is held ``hold`` seconds, then rolls level
            ph, cur = np.empty_like(ts), phi_now
            for k, tk in enumerate(ts):
                cur += max(-step, min(step, (phi if tk <= hold else 0.0) - cur))
                ph[k] = cur
            w = -TURN_EFF * am.G * np.tan(ph) / max(V, 3.0)           # a right bank (+) turns clockwise (the track's angle falls)
            hd = psi + np.cumsum(w) * 0.1
            px, py = x + np.cumsum(V * np.cos(hd)) * 0.1, y + np.cumsum(V * np.sin(hd)) * 0.1
            need = np.maximum(half_span * np.abs(np.cos(ph)), 0.18) + MARGIN_M
            if len(seen):
                gap = np.hypot(px[:, None] - seen[None, :, 0], py[:, None] - seen[None, :, 1]) - seen[None, :, 2] - need[:, None]
                clear = float(gap.min())
                t_hit = float(ts[int(np.argmax(gap.min(axis=1) < 0))]) if clear < 0 else HORIZON_S + 1.0
            else:
                clear, t_hit = 9.0, HORIZON_S + 1.0
            edge = WIDTH_M / 2 - 3.0 - float(np.abs(py).max())        # stay inside the strip
            same = phi_prev is not None and abs(phi - phi_prev[0]) < 1e-6 and hold == phi_prev[1]
            # a clear arc: the way made good along the strip, room to spare, the strip's middle, an end pointing down it
            score = ((float(px[-1]) - x) / (V * HORIZON_S) * 3.0 + 1.2 * min(clear, 1.5) + 0.8 * math.cos(float(hd[-1])) - 0.05 * abs(float(py[-1])) + min(edge, 0.0)
                     + (KEEP_BONUS if same else 0.0) if clear >= 0 else -100.0 + 10.0 * t_hit + clear)
            if best is None or score > best[0]:
                best = (score, float(phi), clear, hold)
    return best[1], best[2], best[3]


def fly(stems_per_ha=700.0, speed=11.0, seed=0, record=None, sim=None) -> dict:
    """One flight through one stand at a commanded speed."""
    trees = stand(stems_per_ha, seed)
    sim = sim or VF.SimV("goshawk", bandit=False)
    rng = np.random.default_rng(seed)
    C0 = VF.with_flap(VF.get("goshawk"), 1.0)
    Vs = VF.stall(C0)
    half = sim.C["b"] / 2
    sim.start(max(speed, 1.15 * VF.stall(VF.get("goshawk"))), HEIGHT_M)
    sim.d.qpos[1] = float(rng.uniform(-4.0, 4.0))
    sim.flap_cmd = 1.0
    ap = amj.Autopilot(sim)
    t0, phi_c, t_plan, alpha_f, hold = sim.d.time, 0.0, -1.0, 0.0, HOLD_S[-1]
    outcome, closest, n_peak, bank_peak, x_end, v_sum, n_steps, clear_now = None, 9.0, 0.0, 0.0, 0.0, 0.0, 0, 9.0
    while outcome is None:
        t = sim.d.time - t0
        s = sim.state()
        x, y, z = (float(v) for v in s["pos"])
        V = s["V"]
        n_wing = 0.5 * am.RHO * V * V * sim.C["S"] * sim.C["CL_MAX"] / (sim.C["mass"] * am.G)
        if t - t_plan >= PLAN_S:
            n_av = n_wing * wash_factor(sim.C, max(V, 5.0))            # with the propellers' wash at full throttle
            psi_track = math.atan2(float(s["v_world"][1]), float(s["v_world"][0]))
            phi_c, clear_now, hold = plan(x, y, psi_track, V, s["phi"], trees, n_av, half, (phi_c, hold))
            t_plan = t
        alpha_f += (float(sim.info.get("alpha", 0.0)) - alpha_f) * amj.DT * 5 / 0.3
        vz = max(-VZ_MAX, min(VZ_MAX, KZ * (HEIGHT_M - z)))
        steep = min(1.0, abs(phi_c) / math.radians(50.0))
        thr = max(0.0, min(1.0, sim.trim["throttle"] + 0.4 * (speed - V) + 0.10 * (HEIGHT_M - z) + TURN_THROTTLE * steep))
        sim.brake = max(0.0, min(1.0, (V - speed) / 0.8))             # the crow brake holds the speed against that throttle
        ap(phi_c=phi_c, theta_c=max(-0.3, min(0.4, vz / max(V, 5.0) + alpha_f)), V_c=speed, throttle=thr)
        lf = 1.0 + (wash_factor(sim.C, max(V, 5.0)) - 1.0) * thr
        n_c = max(0.3, min((1.0 + KV * (vz - float(s["v_world"][2]))) / max(math.cos(s["phi"]), 0.25), LIFT_SHARE * n_wing * lf / LIFT_SHARE))
        sim.controls[0] = pitch(sim, s, n_c, lf)                      # the elevator is this file's own (see ``pitch``)
        sim.step(5)
        s = sim.state()
        x, y, z = (float(v) for v in s["pos"])
        gap = float((np.hypot(trees[:, 0] - x, trees[:, 1] - y) - trees[:, 2] - reach(s["phi"], half) - 0.05).min())
        closest = min(closest, gap)
        n_peak, bank_peak = max(n_peak, float(sim.info.get("load_factor", 0.0))), max(bank_peak, abs(math.degrees(s["phi"])))
        v_sum, n_steps, x_end = v_sum + s["V"], n_steps + 1, x
        if record is not None:
            record(sim, trees, t, phi_c, clear_now)
        if gap < 0.0:
            outcome = "hit"
        elif abs(y) > WIDTH_M / 2 + 1.0:
            outcome = "strayed"                                              # out of the strip's side: it turned away and did not come back
        elif z < 1.0 or s["V"] < 0.85 * Vs or abs(math.degrees(s["theta"])) > 70.0:
            outcome = "lost"
        elif x >= LENGTH_M:
            outcome = "through"
        elif t > 60.0:
            outcome = "lost"
    return {"seed": seed, "stems_per_ha": stems_per_ha, "spacing_m": round(100.0 / math.sqrt(stems_per_ha), 2), "speed_cmd_m_s": speed, "outcome": outcome,
            "through": outcome == "through", "distance_m": round(x_end, 1), "trunks_passed": int(((trees[:, 0] < x_end) & (np.abs(trees[:, 1]) < WIDTH_M / 2)).sum()),
            "closest_m": round(closest, 2), "mean_speed_m_s": round(v_sum / max(n_steps, 1), 1), "peak_load_factor": round(n_peak, 2), "peak_bank_deg": round(bank_peak),
            "duration_s": round(sim.d.time - t0, 1), "stall_flaps_m_s": round(Vs, 1), "height_end_m": round(z, 1)}


if __name__ == "__main__":
    import sys
    a = sys.argv[1:]
    print(fly(float(a[0]) if a else 700.0, float(a[1]) if len(a) > 1 else 11.0, int(a[2]) if len(a) > 2 else 0))
