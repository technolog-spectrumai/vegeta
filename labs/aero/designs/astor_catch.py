"""ASTOR-T takes a quadcopter with its talons and tries to fly on — the catch as physics (2026-11-22).

``astor_talon`` counted a catch when the feet passed within reach, and its video drew a simple fall afterwards. Here
the legs and the quadcopter are bodies in the MuJoCo model:

- **the legs** (both as one): a link on a hinge at the hips — at the centre of gravity, 43 mm under it — that swings
  freely against a damper, and a foot that slides out along the leg against a spring and damper (the half-flexed
  leg stretching: knee and ankle unfolding);
- **the quadcopter**: a free body with mass and drag. Until the catch it is flown along its straight path; at the
  catch (the feet within reach as it passes) a ball joint locks it to the foot where the foot is, and from then on
  nothing is scripted: its 15 m/s of closing speed is taken by the leg's spring, the hinge's damper and ASTOR;
- **ASTOR**: its flight model (``astor_v_flight``), the angle-of-attack limiter, full power from the catch; the
  controller holds the wings level and the nose a little down until the speed is back, then climbs.

What it answers: the pitch rate the catch gives now, whether ASTOR stays in control, the height it loses, and
whether it climbs away carrying the quadcopter. Not modelled: the legs' and the hanging quadcopter's effect on
ASTOR's own aerodynamics beyond the quadcopter's drag; the claws' grip (it holds); the quadcopter's rotors (stopped).
HAND numbers are marked.
"""
from __future__ import annotations

import math

import numpy as np

import arigo1_model as am
import arigo1_mujoco as amj
import astor_netdrop as ND
import astor_t as T
import astor_talon as TA
import astor_v_flight as VF

LEG_KG, FOOT_KG = 0.12, 0.08       # the two legs' links and feet (astor_t's 200 g)
LINE_REST_M, LINE_MAX_M = 1.2, 2.4 # the feet's line: slack to this length from the hips, then elastic; its end stop
LINE_K, LINE_C = 300.0, 20.0       # N/m, N s/m: a bungee that takes the 15 m/s over about a metre (~120 N, HAND)
HIP_STOP_DEG = 100.0               # the legs' stop: 40 deg aft of straight down, the quadcopter then still under and ahead of the fins
CATCH_DAMP = 1.5                   # N m s/rad once the catch has let the hips go (HAND): takes the swing's energy; it is also what pitches ASTOR
HINGE_K = 3.0                      # N m/rad: holds the legs thrown in flight (5 deg of sag under their weight); gives way in the catch
HINGE_DAMP = 0.6                   # N m s/rad at the hips (HAND): what it resists it also passes into the fuselage as pitch
STRETCH_M = 0.17                   # how far the thrown leg shortens as knee and ankle fold (astor_talon: 60 % of its length)
STRETCH_K = 8000.0                 # N/m: the knees' and ankles' springs as one along the leg (stores ~115 J in the stretch)
STRETCH_C = 120.0                  # N s/m (HAND): well damped, so the quadcopter does not bounce back
TERMINAL_KZ = 0.7                  # 1/s: the height loop in the last stretch. astor_talon's 1.5 swung +-0.5 m every 5 s with these legs
TERMINAL_KI = 0.0                  # 1/s2: its integral (tried: it fed the swing; left out)
QUAD_KG, QUAD_CDA = TA.BANDITS_KG["quad"], 0.030     # 1.5 kg; drag area hanging broadside with stopped rotors (HAND +-0.015)
QUAD_GEOMS = ('<geom type="box" size="0.07 0.045 0.03" rgba="0.15 0.15 0.17 1" contype="0" conaffinity="0"/>'
              + "".join(f'<geom type="capsule" fromto="0 0 0 {0.16 * cx:.3f} {0.16 * cy:.3f} 0.01" size="0.008" rgba="0.8 0.15 0.1 1" contype="0" conaffinity="0"/>'
                        f'<geom type="cylinder" pos="{0.16 * cx:.3f} {0.16 * cy:.3f} 0.035" size="0.09 0.003" rgba="0.6 0.6 0.65 0.55" contype="0" conaffinity="0"/>'
                        for cx, cy in ((1, 1), (1, -1), (-1, 1), (-1, -1))))


def leg_geometry(p=None) -> dict:
    """The thrown legs in ASTOR's body frame (x forward, z up, origin at the centre of gravity) [m]."""
    p = p or T.resolve()
    seg, ang = (p["leg_thigh"], p["leg_shin"], p["leg_shank"]), T.LEG_THROWN
    fwd = sum(L * math.sin(math.radians(a)) for L, a in zip(seg, ang)) / 1000.0
    down = TA.leg_drop(p)
    hip = np.array([VF.get("t", 12.0)["x_cg_m"] - p["talon_hinge_x"] / 1000.0, 0.0, -(p["fuselage_diameter"] / 2000.0 + 0.008)])
    return {"hip": hip, "foot_from_hip": np.array([fwd, 0.0, -down]), "length": math.hypot(fwd, down), "leg_y": p["leg_y"] / 1000.0}


def build(mesh=None):
    """``SimV`` for ASTOR-T with the legs and the quadcopter added to the model."""
    import mujoco
    g = leg_geometry()
    hip, ff, y = g["hip"], g["foot_from_hip"], g["leg_y"]
    ax = ff / np.linalg.norm(ff)
    legs = "".join(f'<geom type="capsule" fromto="0 {s * y:.3f} 0 {ff[0]:.3f} {s * y:.3f} {ff[2]:.3f}" size="0.006" rgba="0.93 0.63 0 1" contype="0" conaffinity="0"/>' for s in (1, -1))
    toes = "".join(f'<geom type="capsule" fromto="0 {s * y:.3f} 0 {0.05 * math.cos(math.radians(f)) * d:.3f} {s * y + 0.05 * math.sin(math.radians(f)):.3f} -0.025" size="0.004" '
                   f'rgba="0.93 0.63 0 1" contype="0" conaffinity="0"/>' for s in (1, -1) for f, d in ((0, 1), (32, 1), (-32, 1), (0, -0.7)))
    body = (f'<body name="leg" pos="{hip[0]:.4f} 0 {hip[2]:.4f}">'
            f'<joint name="hip" type="hinge" axis="0 1 0" damping="{HINGE_DAMP}" stiffness="{HINGE_K}" springref="0" limited="true" range="-20 {HIP_STOP_DEG}"/>'
            f'<site name="hip_s" pos="0 0 0" size="0.003"/><inertial pos="{ff[0] / 2:.4f} 0 {ff[2] / 2:.4f}" mass="{LEG_KG}" diaginertia="0.0009 0.0009 0.0002"/>{legs}'
            f'<body name="foot" pos="{ff[0]:.4f} 0 {ff[2]:.4f}">'
            f'<joint name="stretch" type="slide" axis="{ax[0]:.4f} 0 {ax[2]:.4f}" stiffness="{STRETCH_K}" damping="{STRETCH_C}" springref="0" limited="true" range="{-STRETCH_M} 0.01"/>'
            f'<site name="foot_s" pos="0 0 0" size="0.003"/><inertial pos="0 0 0" mass="{FOOT_KG}" diaginertia="0.0001 0.0001 0.0001"/>{toes}</body></body>')
    quad = (f'<body name="quad" pos="0 0 0"><freejoint name="quad_free"/><inertial pos="0 0 0" mass="{QUAD_KG}" diaginertia="0.012 0.012 0.02"/>{QUAD_GEOMS}</body>')
    orig = amj.mjcf

    def mjcf(C, mesh_=None, bandit=False):
        x = orig(C, mesh_, False)
        i = x.index("</body>", x.index('<body name="arigo"'))
        x = x[:i] + body + x[i:]
        x = x.replace("</worldbody>", quad + "</worldbody>"
                      + '<equality><connect name="grip" body1="foot" body2="quad" anchor="0 0 0" active="false"/></equality>'
                      + '<tendon><spatial name="line" width="0.004" rgba="0.95 0.95 0.9 1"><site site="hip_s"/><site site="foot_s"/></spatial></tendon>', 1)
        return x.replace("</worldbody><equality>", "</worldbody>\n  <equality>")
    amj.mjcf = mjcf
    try:
        sim = VF.SimV("t", mesh=mesh, bandit=False)
    finally:
        amj.mjcf = orig
    m = sim.m
    m.body_mass[sim.bid] -= LEG_KG + FOOT_KG                             # the legs are their own bodies now
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST            # the foot's stiff, heavily damped slide shook under RK4 (47 mm in free flight)
    nm = lambda t_, n: mujoco.mj_name2id(m, t_, n)
    sim.ids = {"quad": nm(mujoco.mjtObj.mjOBJ_BODY, "quad"), "foot": nm(mujoco.mjtObj.mjOBJ_BODY, "foot"), "leg": nm(mujoco.mjtObj.mjOBJ_BODY, "leg"),
               "grip": nm(mujoco.mjtObj.mjOBJ_EQUALITY, "grip")}
    jq = nm(mujoco.mjtObj.mjOBJ_JOINT, "quad_free")
    sim.quad_qpos, sim.quad_qvel = int(m.jnt_qposadr[jq]), int(m.jnt_dofadr[jq])
    sim.hip_q = int(m.jnt_qposadr[nm(mujoco.mjtObj.mjOBJ_JOINT, "hip")])
    sim.stretch_q = int(m.jnt_qposadr[nm(mujoco.mjtObj.mjOBJ_JOINT, "stretch")])
    return sim


def _grip(sim, on: bool):
    if hasattr(sim.d, "eq_active"):
        sim.d.eq_active[sim.ids["grip"]] = 1 if on else 0
    else:
        sim.m.eq_active[sim.ids["grip"]] = 1 if on else 0


def catch(seed=0, latency_s=0.10, noise_m=0.3, seconds_after=12.0, pull_up=True, record=None, mesh=None, sim=None, height=120.0,
          quad_thrust=0.0, quad_pull="up") -> dict:
    """One sortie against the quadcopter cruising straight: the approach of ``astor_talon.strike``, then the catch and
    what follows, as physics. ``pull_up`` False: the controls held as they were at the catch (what the airframe does
    alone)."""
    import mujoco
    rng = np.random.default_rng(seed)
    sim = sim or build(mesh)
    m, d = sim.m, sim.d
    jh0, js0 = (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ("hip", "stretch"))
    m.jnt_stiffness[jh0], m.dof_damping[m.jnt_dofadr[jh0]], m.jnt_range[jh0] = HINGE_K, HINGE_DAMP, (math.radians(-20.0), math.radians(HIP_STOP_DEG))
    m.jnt_stiffness[js0], m.dof_damping[m.jnt_dofadr[js0]], m.jnt_range[js0] = STRETCH_K, STRETCH_C, (-STRETCH_M, 0.01)      # (a sim handed in may have flown a catch)
    d.qfrc_applied[:] = 0.0
    _grip(sim, False)
    C0 = VF.get("t", VF.SWEEPS[0])
    Vs = VF.stall(C0)
    bandit = TA.CruisingQuad(height=height, offset=float(rng.uniform(-25, 25)), heading_deg=float(rng.uniform(-20, 20)))
    sim.start(1.4 * Vs, bandit.p[2] + 6.0)
    g = leg_geometry()
    drop = -(g["hip"][2] + g["foot_from_hip"][2])                        # the feet under the centre of gravity
    fwd = g["hip"][0] + g["foot_from_hip"][0]
    reach_y = TA.reach_across(T.resolve())
    qa, qv = sim.quad_qpos, sim.quad_qvel
    ap = amj.Autopilot(sim)
    cam = ND.Camera(latency_s, noise_m, seed, window_s=0.5)
    alpha_f, caught, t_catch, log, was_ahead = 0.0, None, None, [], None
    i_z = 0.0
    ma = float(m.body_mass[sim.bid]) + LEG_KG + FOOT_KG
    Vs_loaded = Vs * math.sqrt((ma + QUAD_KG) / ma)
    stats = {"q_peak": 0.0, "theta_min": 1e9, "theta_max": -1e9, "z_min": 1e9, "v_min": 1e9, "stretch_max": 0.0, "hip_max": 0.0, "n_max": 0.0, "alpha_max": -1e9}
    t0 = d.time
    held = None
    while True:
        t = d.time - t0
        s = sim.state()
        if caught is None:
            d.qpos[qa:qa + 3], d.qpos[qa + 3:qa + 7] = bandit.p, (1, 0, 0, 0)       # the quadcopter flown along its path
            d.qvel[qv:qv + 3], d.qvel[qv + 3:qv + 6] = bandit.v, 0.0
            cam.sig = min(noise_m, TA.ANGLE_NOISE * math.dist(bandit.p, s["pos"]))
            b, vb = cam.see(t, bandit.p, bandit.v)
            sb = float(np.linalg.norm(vb[:2]))
            # the fuselage is aimed so the FEET (ahead of and under the centre of gravity) pass through the quadcopter
            off = d.xpos[sim.ids["foot"]] - np.asarray(s["pos"])           # where the feet really hang (the hips are free)
            aim = np.asarray(b) - off
            r = aim - s["pos"]
            rng_h = math.hypot(r[0], r[1])
            phi, gam = amj.pn_command(sim, aim, vb)
            n_av = 0.5 * am.RHO * s["V"] ** 2 * sim.C["S"] * sim.C["CL_MAX"] / (sim.C["mass"] * am.G)
            phi_max = math.acos(min(1.0, 1.0 / max(0.85 * n_av, 1.0001)))
            V_c = max(TA.PASS_SPEED * Vs, sb + min(14.0, 4.0 + 0.2 * rng_h))
            if rng_h < TA.TERMINAL_M:
                alpha_f += (float(sim.info.get("alpha", 0.0)) - alpha_f) * 0.01 / 0.5
                # + the error's integral: slowing to the pass speed ASTOR sinks steadily, and the feet passed 0.2-0.45 m low
                i_z = max(-0.6, min(0.6, i_z + float(r[2]) * 0.01)) if abs(float(r[2])) < 1.0 else 0.0      # only close to the height wanted
                vz = max(-TA.TERMINAL_VZ, min(TA.TERMINAL_VZ, TERMINAL_KZ * float(r[2]) + TERMINAL_KI * i_z))
                gam = max(-0.3, min(0.3, vz / max(s["V"], 10.0) + alpha_f))
            else:
                alpha_f = float(sim.info.get("alpha", 0.0))
            thr = max(0.0, min(1.0, sim.trim["throttle"] + 0.15 * (V_c - s["V"]) + 0.06 * float(r[2])))
            ap(phi_c=max(-phi_max, min(phi_max, phi)), theta_c=gam, V_c=V_c, throttle=thr)
        else:
            ta = t - t_catch
            v_air = d.qvel[qv:qv + 3] - sim.wind
            d.xfrc_applied[sim.ids["quad"], :3] = -0.5 * am.RHO * QUAD_CDA * float(np.linalg.norm(v_air)) * v_air
            if quad_thrust > 0.0 and ta > 0.3:                             # the quadcopter fights (after 0.3 s to right itself): full thrust one way
                vh = np.array([s["v_world"][0], s["v_world"][1], 0.0])
                fw = vh / max(float(np.linalg.norm(vh)), 1e-6)             # ASTOR's direction of flight
                pull = {"up": np.array([0.0, 0.0, 1.0]), "down": np.array([0.0, 0.0, -1.0]), "back": -fw, "ahead": fw,
                        "back_up": (-fw + np.array([0.0, 0.0, 1.0])) / math.sqrt(2.0)}[quad_pull]
                d.xfrc_applied[sim.ids["quad"], :3] += quad_thrust * pull
            dv = int(m.jnt_dofadr[js0])
            ext = g["length"] + float(d.qpos[sim.stretch_q]) - LINE_REST_M              # the line beyond its rest length
            d.qfrc_applied[dv] = -(LINE_K * ext + LINE_C * max(float(d.qvel[dv]), 0.0)) if ext > 0.0 else 0.0
            stats["line_max"] = max(stats.get("line_max", 0.0), g["length"] + float(d.qpos[sim.stretch_q]))
            stats["line_force"] = max(stats.get("line_force", 0.0), -float(d.qfrc_applied[dv]))
            # how near the quadcopter comes to the tail (body frame: the fins' lower edge, 0.17-0.43 m aft and 0.13 m under the CG)
            Rb = d.xmat[sim.bid].reshape(3, 3)
            qb = Rb.T @ (d.xpos[sim.ids["quad"]] - d.xpos[sim.bid])
            px = min(max(qb[0], -0.43), -0.17)
            stats["tail_gap"] = min(stats.get("tail_gap", 9.0), math.hypot(qb[0] - px, qb[2] + 0.13, qb[1]) - 0.22)
            if pull_up:
                # full power at once; the nose held a little down while the speed comes back, then the climb
                fast_enough = s["V"] > 1.2 * Vs_loaded
                ap(phi_c=0.0, theta_c=math.radians(8.0 if fast_enough else -2.0), throttle=1.0)
            else:
                sim.controls[:] = held
            if ta >= seconds_after or s["pos"][2] < 1.0:
                break
        sim.step(5)
        if caught is None:
            bandit.step(0.01, t)
            foot = d.xpos[sim.ids["foot"]].copy()
            rel = np.asarray(bandit.p) - foot
            vrel = np.asarray(sim.state()["v_world"]) - np.asarray(bandit.v)
            u = vrel / max(float(np.linalg.norm(vrel)), 1e-6)
            ahead = float(rel @ u)                                        # + : the quadcopter still ahead of the feet
            if was_ahead is not None and was_ahead > 0.0 >= ahead:        # it passes the feet now
                perp = rel - ahead * u
                side = np.cross(np.array([0.0, 0.0, 1.0]), u)
                side /= max(float(np.linalg.norm(side)), 1e-6)
                across, height = float(perp @ side), float(perp[2])
                vc = float(np.linalg.norm(vrel))
                caught = {"across_m": round(across, 2), "height_m": round(height, 2), "closing_m_s": round(vc, 1),
                          "taken": bool(abs(across) <= reach_y and abs(height) <= TA.REACH_Z_M and vc <= TA.V_CLOSE_MAX)}
                if not caught["taken"]:
                    break
                d.qpos[qa:qa + 3], d.qvel[qv:qv + 3] = bandit.p, bandit.v
                mujoco.mj_forward(m, d)
                Rq = d.xmat[sim.ids["quad"]].reshape(3, 3)
                off = Rq.T @ (d.xpos[sim.ids["foot"]] - d.xpos[sim.ids["quad"]])
                n = float(np.linalg.norm(off))
                m.eq_data[sim.ids["grip"], 3:6] = off * min(1.0, 0.16 / max(n, 1e-6))     # where on the quadcopter the claws take hold
                _grip(sim, True)
                # the feet have gripped: they tear away from the legs and stay tied to the hips by the line. In the model the
                # leg becomes that line: free to point anywhere from the hips, free to lengthen, elastic beyond its rest length
                jh, js = (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ("hip", "stretch"))
                m.jnt_stiffness[jh], m.dof_damping[m.jnt_dofadr[jh]], m.jnt_range[jh] = 0.0, 0.02, (-3.1, 3.1)
                m.jnt_stiffness[js], m.dof_damping[m.jnt_dofadr[js]], m.jnt_range[js] = 0.0, 0.0, (-(g["length"] - 0.02), LINE_MAX_M - g["length"])
                t_catch, held = t, sim.controls.copy()
                z_catch, v_catch = float(s["pos"][2]), float(s["V"])
            was_ahead = ahead
            if t > 60.0:
                break
        else:
            s2, i2 = sim.state(), sim.info
            q_deg = math.degrees(float(s2["omega_frd"][1]))
            stats["q_peak"] = q_deg if abs(q_deg) > abs(stats["q_peak"]) else stats["q_peak"]
            th = math.degrees(s2["theta"])
            stats["theta_min"], stats["theta_max"] = min(stats["theta_min"], th), max(stats["theta_max"], th)
            stats["z_min"], stats["v_min"] = min(stats["z_min"], float(s2["pos"][2])), min(stats["v_min"], s2["V"])

            stats["hip_max"] = max(stats["hip_max"], math.degrees(float(d.qpos[sim.hip_q])))
            stats["n_max"] = max(stats["n_max"], abs(float(i2["load_factor"])))
            stats["alpha_max"] = max(stats["alpha_max"], math.degrees(float(i2["alpha"])))
            log.append((round(t - t_catch, 2), round(float(s2["pos"][2]) - z_catch, 2), round(s2["V"], 1), round(th, 1), round(q_deg), round(math.degrees(s2["phi"])),
                        round(math.degrees(float(d.qpos[sim.hip_q]))), round(float(d.qpos[sim.stretch_q]) * 1000), round(float(d.qvel[2]), 1)))
        if record is not None:
            record(sim, bandit if caught is None else None, t, caught is not None)
    out = {"seed": seed, "pass": caught, "caught": bool(caught and caught["taken"]), "pull_up": pull_up,
           "quadcopter_fights": None if quad_thrust <= 0 else {"thrust_N": quad_thrust, "thrust_over_weight": round(quad_thrust / (QUAD_KG * am.G), 1), "pulls": quad_pull}}
    if out["caught"]:
        s = sim.state()
        last = [x for x in log if x[0] >= seconds_after - 2.0] or log[-1:]
        climb = float(np.mean([x[8] for x in last]))
        out.update({"t_catch_s": round(t_catch, 2), "speed_at_catch_m_s": round(v_catch, 1), "stall_loaded_m_s": round(Vs_loaded, 1),
                    "pitch_rate_peak_deg_s": round(stats["q_peak"]), "pitch_min_deg": round(stats["theta_min"]), "pitch_max_deg": round(stats["theta_max"]),
                    "height_lost_m": round(max(0.0, z_catch - stats["z_min"]), 1), "speed_min_m_s": round(stats["v_min"], 1), "alpha_max_deg": round(stats["alpha_max"], 1),
                    "load_factor_max": round(stats["n_max"], 1), "line_longest_m": round(stats["line_max"], 2), "line_force_max_N": round(stats["line_force"]),
                    "quadcopter_nearest_the_tail_m": round(stats["tail_gap"], 2),
                    "end": {"t_after_s": log[-1][0], "height_change_m": log[-1][1], "speed_m_s": log[-1][2], "pitch_deg": log[-1][3], "bank_deg": log[-1][5], "climb_m_s": round(climb, 1)},
                    "flies_on": bool(log[-1][0] >= seconds_after - 0.1 and abs(log[-1][5]) < 45 and log[-1][2] > 1.05 * Vs_loaded and climb > -0.5),
                    "history_t_dz_V_pitch_q_bank_hip_stretch_vz": log[::10]})
    return out
