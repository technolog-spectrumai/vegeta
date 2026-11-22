"""ASTOR-T's win rate against quadcopters, for a design: the sortie the design study flies (2026-11-22).

A hawk does not take every bird and sometimes drops one. The requirement here is the same kind: ASTOR wins about
80 % of its sorties and is itself almost never lost.

**A sortie's outcome**

- ``carried``   WIN  : the quadcopter caught and still on the line 10 s later, ASTOR flying;
- ``dropped``   WIN  : caught, its rotors fouled, and let go (it falls); ASTOR flying;
- ``escaped``   draw : caught, rotors still turning, let go because ASTOR was losing the pull; ASTOR flying;
- ``no_catch``  draw : no catch in the passes allowed (three, 75 s); ASTOR flying;
- ``lost``      LOSS : ASTOR hits the ground or is out of control at the end.

**The quadcopters** (one drawn per sortie, HAND proportions — change ``POPULATION``): 1.0-2.0 kg; flying straight at
8-14 m/s (35 %), weaving +-25 deg every 8 s at 10 m/s (30 %), or jinking +-60 deg every 2 s at 8 m/s once ASTOR is
within 60 m (35 %). When caught its rotors are fouled with probability ``p_foul`` (HAND: 0.5 with the claws alone,
0.8 with a net between the feet); otherwise it pulls with twice (70 %) or four times (30 %) its weight in a direction
drawn at random for the sortie.

**The design choices** (``DEFAULT_DESIGN``): motor power; wing chord; the catch's reach (the two feet, or a light net
frame between and beside them); the line's length; the pass speed; whether the line lets go when ASTOR is losing.

The flight is ``astor_catch``'s model (legs and quadcopter as bodies). Guidance: proportional navigation, the last
150 m on a height loop, a miss followed by a climbing turn and another pass.
"""
from __future__ import annotations

import math

import numpy as np

import arigo1_model as am
import arigo1_mujoco as amj
import astor_catch as AC
import astor_netdrop as ND
import astor_t as T
import astor_talon as TA
import astor_v_flight as VF

DEFAULT_DESIGN = {"power_kw": 4.0, "chord": 1.0, "reach": "feet", "line_m": 1.2, "pass_speed": 1.3, "release": True}
LEVELS = {"power_kw": (2.0, 4.0), "chord": (1.0, 1.2), "reach": ("feet", "net"), "line_m": (1.2, 2.0), "pass_speed": (1.3, 1.45), "release": (False, True)}
REACH = {"feet": dict(across=None, height=TA.REACH_Z_M, p_foul=0.5, mass_kg=0.0),            # across: the legs' own (astor_talon.reach_across)
         "net": dict(across=0.50, height=0.30, p_foul=0.8, mass_kg=0.035)}                  # a 0.7 x 0.45 m net on a wire frame between the feet (HAND)
POPULATION = {"behaviour": (("straight", 0.35), ("weave", 0.30), ("jink", 0.35)), "thrust_over_weight": ((2.0, 0.7), (4.0, 0.3)), "mass_kg": (1.0, 2.0)}
MAX_PASSES, HUNT_S, CARRY_S, HEIGHT_M = 3, 130.0, 10.0, 120.0     # a second pass costs ~40 s (a full turn, then the chase again)
MIN_RADIUS_M = 0.15                # the caught quadcopter no nearer the hips than this (the belly is there; and the line's model is singular at zero)
RELEASE_FLOOR_M = 70.0             # the line lets go this high at the latest if ASTOR is still going down: a pull-out takes 35-60 m
_configured = [None]


class Quad:
    """The quadcopter before the catch: straight, weaving, or jinking when the hunter is near."""

    def __init__(self, kind, rng, height=HEIGHT_M):
        self.kind, self.rng = kind, rng
        self.speed = {"straight": float(rng.uniform(8.0, 14.0)), "weave": 10.0, "jink": 8.0}[kind]
        self.h0 = math.radians(float(rng.uniform(-20, 20)))
        self.heading = self.h0
        self.p = [220.0, float(rng.uniform(-25, 25)), height]
        self.v = (self.speed * math.cos(self.heading), self.speed * math.sin(self.heading), 0.0)
        self.phase, self.jink, self.jink_at = float(rng.uniform(0, 8.0)), 0.0, 0.0

    def step(self, dt, t, hunter=None):
        if self.kind == "weave":
            self.heading = self.h0 + math.radians(25.0) * math.sin(2 * math.pi * (t + self.phase) / 8.0)
        elif self.kind == "jink" and hunter is not None and math.dist(hunter[:2], self.p[:2]) < 60.0:
            if t >= self.jink_at:
                self.jink, self.jink_at = math.radians(60.0) * (1.0 if self.rng.random() < 0.5 else -1.0), t + 2.0
            want = self.h0 + self.jink
            self.heading += max(-1.6 * dt, min(1.6 * dt, want - self.heading))       # 1.6 rad/s, Armageddon's turn rate
        self.v = (self.speed * math.cos(self.heading), self.speed * math.sin(self.heading), 0.0)
        for i in range(3):
            self.p[i] += self.v[i] * dt


def configure(design) -> dict:
    """Set this process to a design (the airframe's part of it); returns the design completed with the defaults."""
    dz = {**DEFAULT_DESIGN, **(design or {})}
    key = (dz["power_kw"], dz["chord"])
    if _configured[0] != key:
        T.configure(dz["power_kw"], dz["chord"])
        _configured[0] = key
    return dz


def hunt(design=None, seed=0, latency_s=0.10, noise_m=0.3, record=None, mesh=None) -> dict:
    """One sortie of a design against one quadcopter drawn from the population."""
    import mujoco
    dz = configure(design)
    rng = np.random.default_rng(seed)
    pick = lambda pairs: pairs[int(np.searchsorted(np.cumsum([w for _, w in pairs]), rng.random()))][0]
    kind = pick(POPULATION["behaviour"])
    q_kg = float(rng.uniform(*POPULATION["mass_kg"]))
    R = REACH[dz["reach"]]
    fouled = bool(rng.random() < R["p_foul"])
    tw = 0.0 if fouled else float(pick(POPULATION["thrust_over_weight"]))
    u = rng.normal(size=3)
    pull_dir = u / np.linalg.norm(u)
    sim = AC.build(mesh)
    m, d = sim.m, sim.d
    m.body_inertia[sim.ids["quad"]] *= q_kg / float(m.body_mass[sim.ids["quad"]])
    m.body_mass[sim.ids["quad"]] = q_kg
    m.body_mass[sim.bid] += R["mass_kg"]
    mujoco.mj_setConst(m, d)                                              # the solver's constants follow the new masses (left stale, the grip went soft and blew up)
    jh, js = (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ("hip", "stretch"))
    dv = int(m.jnt_dofadr[js])
    Vs = VF.stall(VF.get("t", VF.SWEEPS[0]))
    bandit = Quad(kind, rng)
    sim.start(1.4 * Vs, bandit.p[2] + 6.0)
    g = AC.leg_geometry()
    reach_y = R["across"] or TA.reach_across(T.resolve())
    qa, qv = sim.quad_qpos, sim.quad_qvel
    ap = amj.Autopilot(sim)
    cam = ND.Camera(latency_s, noise_m, seed, window_s=0.5)
    ma = float(m.body_mass[sim.bid]) + AC.LEG_KG + AC.FOOT_KG
    Vs_loaded = Vs * math.sqrt((ma + q_kg) / ma)
    alpha_f, was_ahead, passes, phase, t_phase = 0.0, None, [], "attack", 0.0
    state, t_catch, t_release, outcome = "hunt", None, None, None
    st = {"z_min": 1e9, "q_peak": 0.0, "line_force": 0.0, "tail_gap": 9.0, "v_min": 1e9}
    t0 = d.time
    t_last = -1.0
    while outcome is None:
        t = d.time - t0
        if t < t_last:                                                    # MuJoCo reset an unstable step: not a flight result
            outcome = "sim_error"
            break
        t_last = t
        s = sim.state()
        if state == "hunt":
            d.qpos[qa:qa + 3], d.qpos[qa + 3:qa + 7] = bandit.p, (1, 0, 0, 0)
            d.qvel[qv:qv + 3], d.qvel[qv + 3:qv + 6] = bandit.v, 0.0
            cam.sig = min(noise_m, TA.ANGLE_NOISE * math.dist(bandit.p, s["pos"]))
            b, vb = cam.see(t, bandit.p, bandit.v)
            sb = float(np.linalg.norm(vb[:2]))
            if phase == "attack":
                aim = np.asarray(b) - (d.xpos[sim.ids["foot"]] - np.asarray(s["pos"]))
                r = aim - s["pos"]
                rng_h = math.hypot(r[0], r[1])
                phi, gam = amj.pn_command(sim, aim, vb)
                n_av = 0.5 * am.RHO * s["V"] ** 2 * sim.C["S"] * sim.C["CL_MAX"] / (sim.C["mass"] * am.G)
                phi_max = math.acos(min(1.0, 1.0 / max(0.85 * n_av, 1.0001)))
                V_c = max(dz["pass_speed"] * Vs, sb + min(14.0, 4.0 + 0.2 * rng_h))
                if rng_h < TA.TERMINAL_M:
                    alpha_f += (float(sim.info.get("alpha", 0.0)) - alpha_f) * 0.01 / 0.5
                    vz = max(-TA.TERMINAL_VZ, min(TA.TERMINAL_VZ, AC.TERMINAL_KZ * float(r[2])))
                    gam = max(-0.3, min(0.3, vz / max(s["V"], 10.0) + alpha_f))
                else:
                    alpha_f = float(sim.info.get("alpha", 0.0))
                thr = max(0.0, min(1.0, sim.trim["throttle"] + 0.15 * (V_c - s["V"]) + 0.06 * float(r[2])))
                ap(phi_c=max(-phi_max, min(phi_max, phi)), theta_c=gam, V_c=V_c, throttle=thr)
            else:
                # after a miss ASTOR is ahead of the quadcopter: a full level turn brings it back BEHIND it, for another
                # pass from the tail at the same closing speed (turning straight back met it head-on at 25-35 m/s,
                # faster than the feet can take)
                los = np.asarray(b) - np.asarray(s["pos"])
                vh = np.asarray(s["v_world"])
                ang = math.degrees(math.acos(max(-1.0, min(1.0, float(los[:2] @ vh[:2]) / max(float(np.linalg.norm(los[:2]) * np.linalg.norm(vh[:2])), 1e-6)))))
                same_way = float(vh[:2] @ np.asarray(vb)[:2]) > 0.0
                dz_h = float(los[2]) + 6.0
                ap(phi_c=math.radians(40.0), theta_c=math.radians(max(-6.0, min(8.0, 2.0 * dz_h))) + float(sim.info.get("alpha", 0.0)), throttle=0.7)
                if t - t_phase > 6.0 and ang < 20.0 and same_way and float(np.linalg.norm(los[:2])) > 60.0:
                    phase, was_ahead = "attack", None
            if t > HUNT_S or (len(passes) >= MAX_PASSES and phase == "away" and t - t_phase > 2.0):
                outcome = "no_catch"
        else:
            ta = t - (t_catch if state == "carry" else t_release)
            if state == "carry":
                v_air = d.qvel[qv:qv + 3] - sim.wind
                F = -0.5 * am.RHO * AC.QUAD_CDA * float(np.linalg.norm(v_air)) * v_air
                if tw > 0.0 and ta > 0.3:
                    F = F + tw * q_kg * am.G * pull_dir
                d.xfrc_applied[sim.ids["quad"], :3] = F
                ext = g["length"] + float(d.qpos[sim.stretch_q]) - dz["line_m"]
                d.qfrc_applied[dv] = -(AC.LINE_K * ext + AC.LINE_C * max(float(d.qvel[dv]), 0.0)) if ext > 0.0 else 0.0
                st["line_force"] = max(st["line_force"], -float(d.qfrc_applied[dv]))
                Rb = d.xmat[sim.bid].reshape(3, 3)
                qb = Rb.T @ (d.xpos[sim.ids["quad"]] - d.xpos[sim.bid])
                px = min(max(qb[0], -0.43), -0.17)
                st["tail_gap"] = min(st["tail_gap"], math.hypot(qb[0] - px, qb[2] + 0.13, qb[1]) - 0.22)
                # the line lets go when ASTOR is losing: slow and sinking after the first seconds, or the ground near
                vz_now = float(s["v_world"][2])
                losing = (ta > 2.5 and vz_now < -2.0 and s["V"] < 1.25 * Vs_loaded) or (ta > 5.0 and vz_now < -3.0) or (s["pos"][2] < RELEASE_FLOOR_M and vz_now < -5.0)
                if dz["release"] and losing:
                    AC._grip(sim, False)
                    # the empty feet on their line: reeled back gently (left free, the 80 g on a slack 1.3 m line shook the
                    # integrator apart and MuJoCo reset the flight)
                    m.jnt_stiffness[js], m.dof_damping[dv], m.dof_damping[m.jnt_dofadr[jh]] = 30.0, 6.0, 0.3
                    d.qfrc_applied[dv] = 0.0
                    d.xfrc_applied[sim.ids["quad"], :3] = 0.0
                    state, t_release = "released", t
                elif ta >= CARRY_S:
                    ok = abs(math.degrees(s["phi"])) < 45 and s["V"] > 1.05 * Vs_loaded and vz_now > -0.5
                    outcome = "carried" if ok else ("lost" if not dz["release"] else "carried_sinking")
            fast_enough = s["V"] > 1.2 * (Vs_loaded if state == "carry" else Vs)
            ap(phi_c=0.0, theta_c=math.radians(8.0 if fast_enough else -2.0), throttle=1.0)
            if state == "released" and ta >= 5.0:
                outcome = "dropped" if fouled else "escaped"
        if s["pos"][2] < 1.5:
            outcome = "lost"
            break
        sim.step(5)
        if state == "hunt":
            bandit.step(0.01, t, hunter=tuple(s["pos"]))
            rel = np.asarray(bandit.p) - d.xpos[sim.ids["foot"]]
            vrel = np.asarray(sim.state()["v_world"]) - np.asarray(bandit.v)
            vc = float(np.linalg.norm(vrel))
            un = vrel / max(vc, 1e-6)
            ahead = float(rel @ un)
            if phase == "attack" and was_ahead is not None and was_ahead > 0.0 >= ahead and float(np.linalg.norm(rel)) < 25.0:
                perp = rel - ahead * un
                side = np.cross(np.array([0.0, 0.0, 1.0]), un)
                side /= max(float(np.linalg.norm(side)), 1e-6)
                across, height = float(perp @ side), float(perp[2])
                taken = bool(abs(across) <= reach_y and abs(height) <= R["height"] and vc <= TA.V_CLOSE_MAX)
                passes.append({"t": round(t, 1), "across_m": round(across, 2), "height_m": round(height, 2), "closing_m_s": round(vc, 1), "taken": taken})
                if taken:
                    d.qpos[qa:qa + 3], d.qvel[qv:qv + 3] = bandit.p, bandit.v
                    mujoco.mj_forward(m, d)
                    Rq = d.xmat[sim.ids["quad"]].reshape(3, 3)
                    off = Rq.T @ (d.xpos[sim.ids["foot"]] - d.xpos[sim.ids["quad"]])
                    m.eq_data[sim.ids["grip"], 3:6] = off * min(1.0, 0.16 / max(float(np.linalg.norm(off)), 1e-6))
                    AC._grip(sim, True)
                    m.jnt_stiffness[jh], m.dof_damping[m.jnt_dofadr[jh]], m.jnt_range[jh] = 0.0, 0.02, (-3.1, 3.1)
                    m.jnt_stiffness[js], m.dof_damping[dv], m.jnt_range[js] = 0.0, 0.0, (-(g["length"] - MIN_RADIUS_M), 2.0 * dz["line_m"] - g["length"])
                    state, t_catch = "carry", t
                else:
                    phase, t_phase = "away", t
            was_ahead = ahead
        else:
            s2 = sim.state()
            st["z_min"], st["v_min"] = min(st["z_min"], float(s2["pos"][2])), min(st["v_min"], s2["V"])
            qd = abs(math.degrees(float(s2["omega_frd"][1])))
            st["q_peak"] = max(st["q_peak"], qd)
        if record is not None:
            record(sim, t, state)
    s = sim.state()
    if outcome not in ("lost", "sim_error") and (abs(math.degrees(s["phi"])) > 70 or s["V"] < 0.9 * Vs):
        outcome = "lost"                                                  # flying, but not in control
    if outcome == "carried_sinking":
        outcome = "lost" if s["pos"][2] < 20.0 else "escaped"
    return {"seed": seed, "design": dz, "outcome": outcome, "win": outcome in ("carried", "dropped"), "astor_lost": outcome == "lost",
            "quadcopter": {"behaviour": kind, "mass_kg": round(q_kg, 2), "speed_m_s": round(bandit.speed, 1), "rotors_fouled": fouled, "thrust_over_weight": tw},
            "passes": passes, "caught": t_catch is not None, "t_catch_s": None if t_catch is None else round(t_catch, 1),
            "after_the_catch": None if t_catch is None else {"height_lost_m": round(max(0.0, HEIGHT_M - st["z_min"]), 1), "speed_min_m_s": round(st["v_min"], 1),
                                                             "pitch_rate_peak_deg_s": round(st["q_peak"]), "line_force_max_N": round(st["line_force"]),
                                                             "nearest_the_tail_m": round(st["tail_gap"], 2), "released": t_release is not None},
            "astor": {"mass_kg": round(ma, 2), "stall_m_s": round(Vs, 1), "duration_s": round(d.time - t0, 1)}}
