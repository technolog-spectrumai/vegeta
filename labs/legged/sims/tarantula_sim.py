"""TARANTULA's gait and stability in MuJoCo (``./tarantula.sh``; cases and summaries in ``CASES``).

The robot (``tarantula_robot``: 1.5 kg, 24 tendon-driven joints, self-locking gearmotor drives) tracks a gait's
keyframes (``tarantula_gaits``) with each joint's PD loop clipped to its drive's torque–speed line when driving and
full stall torque when braking or holding (a worm drive holds). The world can tilt (gravity rotated: an incline of
``incline_deg``, the robot heading ``heading_deg`` from straight up the slope), the ground can be a Chiron terrain
(rough, steps), the feet can hold on (``attach``: an adhesion actuator per foot, on in stance and off in swing, with
``tarantula_attachment``'s pull and shear capacities), the body can be pushed, a leg can fail.

**Measured** (``simulate`` returns a flat dict): the speed achieved and planned, heading drift and sideways drift,
the dynamic stability margin (the CoM projected along gravity inside the polygon of the feet actually in contact;
its minimum and the time it is negative), the worst body roll and pitch, the lowest body height and the belly's
contacts, the feet's slip while in stance, peak joint torque over stall and the time saturated, the mean electrical
power, the cost of transport and the runtime on the battery, and whether it fell: tipped (roll or pitch past 50°),
slid (more than 0.25 m down the slope) or stopped (achieved under 20 % of the planned speed).

Assumptions: the drive's electrical power is its mechanical output / 0.25 (gearmotor + worm + tendon, §5) plus 0.03 W
idle per motor; the electronics and the turret draw 3 W; a holding worm costs no current. Spines are isotropic here
(real ones grip only up the slope: ``tarantula_attachment`` covers that).
"""
from __future__ import annotations

import itertools
import math
import xml.etree.ElementTree as ET

import numpy as np

__all__ = ["BATTERY_WH", "CASES", "cases", "run_case", "simulate", "summary"]

BATTERY_WH = 7.2 * 5.0 * 0.85                                             # 2S2P 18650, 85 % usable
ELECTRONICS_W = 3.0
DRIVE_EFF = 0.25
IDLE_W = 0.03


def _gravity(incline_deg, heading_deg):
    b, h = math.radians(incline_deg), math.radians(heading_deg)
    return 9.81 * np.array([-math.sin(b) * math.cos(h), math.sin(b) * math.sin(h), -math.cos(b)])


def _terrain(spec):
    from vegeta.chiron.terrain import Flat, Rough, Steps
    if not spec or spec.get("kind") == "flat":
        return Flat()
    ext = (-0.6, 1.8, -0.7, 0.7)
    if spec["kind"] == "rough":
        return Rough(spec["rms_mm"] * 1e-3, spec.get("corr_mm", 80.0) * 1e-3, start=0.35, seed=spec.get("seed", 0),
                     extent=ext, cell=0.01)
    if spec["kind"] == "steps":
        return Steps(spec["height_mm"] * 1e-3, spec.get("spacing_mm", 300.0) * 1e-3, start=0.35,
                     mode=spec.get("mode", "alternate"), end=1.8)
    raise ValueError(f"terrain {spec!r}")


def _attach_params(attach):
    """(pull-off per foot [N], friction μ) of ``attach`` = 'pad:glass', 'spine:brick', 'magnet:bare steel', …"""
    import tarantula_attachment as ta
    module, surface = attach.split(":", 1)
    tau, sig, per_spine, mu, steel = ta.SURFACES[surface]
    if module == "pad":
        a, b = ta.MODULES["pad"]["pad_mm"]
        A = a * b * 1e-6
        pull, shear = sig * 1e3 * A, tau * 1e3 * A
        return pull, (shear / pull if pull > 0 else mu)
    if module == "spine":
        import arachne_specs as asp
        cap = per_spine * ta.MODULES["spine"]["spines"] * asp.SPINES["engaged_fraction"]
        pull = cap * asp.SPINES["pulloff_over_shear"]
        return pull, (cap / pull if pull > 0 else mu)
    if module == "magnet":
        m = ta.MODULES["magnet"]
        if not steel:
            return 0.0, mu
        return (m["pull_n"] if surface == "bare steel" else m["pull_painted_n"]), m["mu"]
    raise ValueError(attach)


def _hull_margin(points, c):
    import tarantula_gaits as tg
    if len(points) < 3:
        return -math.inf
    return tg.edge_distance(tg.hull(points), c)


def simulate(gait="wave4", posture="standing", *, stride=None, swing_s=None, step_height=None, payload_g=300.0,
             payload_x_mm=None, mu=0.8, incline_deg=0.0, heading_deg=0.0, terrain=None, cycles=4, attach=None,
             push=None, fault=None, derate=1.0, kp=50.0, kd=0.5, timestep=0.001, settle_s=0.6, seed=0, trace_fps=None):
    """One run; returns a flat dict of results (the module docstring lists them).

    ``terrain``: None/{'kind': 'flat'}, {'kind': 'rough', 'rms_mm', 'corr_mm', 'seed'} or {'kind': 'steps',
    'height_mm', 'spacing_mm'}. ``attach``: None or 'module:surface' (``tarantula_attachment``). ``push``:
    {'t', 'force_n', 'dir_deg' (0 forward, 90 left), 'dur_s'} on the body. ``fault``: {'leg', 'mode': 'limp' (its
    drives go slack) or 'locked' (they stop where they are and hold), 't'}. Gait time t counts from the end of the
    settle. ``trace_fps``: also return ``trace`` — every link body's position [m] and quaternion at that frame
    rate, with the feet attached (adhesion on) per frame — for a video (``tools/tarantula_video.py``)."""
    import motion_kit as mk
    import mujoco
    import tarantula_gaits as tg
    import tarantula_robot as tr
    from vegeta.chiron.robot import SimOptions

    drop = tg.POSTURES[posture] if isinstance(posture, str) else float(posture)
    robot = tr.tarantula(payload_g, kp=kp, kd=kd, derate=derate, foot_mu=mu, payload_x_mm=payload_x_mm)
    kin = mk.Kin(robot)
    motion = tg.gait(gait, kin=kin, stride=stride, swing_s=swing_s, step_height=step_height, body_drop_mm=drop)
    g = _gravity(incline_deg, heading_deg)
    tilted = incline_deg > 0 or attach
    opt = SimOptions(timestep=timestep, gravity=tuple(g), cone="elliptic", impratio=3.0,
                     noslip_iterations=2 if tilted else 0, course_extent=(-0.6, 1.8, -0.7, 0.7),
                     heightfield_cell=0.01)
    root = ET.fromstring(robot.to_mjcf(terrain=_terrain(terrain), options=opt))
    pull, mu_att = (0.0, mu)
    if attach:
        pull, mu_att = _attach_params(attach)
        for gm in root.iter("geom"):
            if gm.get("name", "") in tr.LEGS:
                gm.set("friction", f"{mu_att:.4f} 0.005 0.0001")
                gm.set("margin", "0.003")                                    # adhesion acts within 3 mm of the wall
                gm.set("gap", "0.0")
        act = root.find("actuator")
        for f in tr.LEGS:
            ET.SubElement(act, "adhesion", {"name": f"adh:{f}", "body": f"{f}_tibia", "ctrlrange": "0 1",
                                            "gain": f"{pull:.5f}"})
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    names = list(kin.names)
    qadr = np.array([model.joint(n).qposadr[0] for n in names])
    dadr = np.array([model.joint(n).dofadr[0] for n in names])
    act_servo = np.array([model.actuator(f"servo:{n}").id for n in names])
    act_adh = {f: model.actuator(f"adh:{f}").id for f in tr.LEGS} if attach else {}
    sv = tr.drive(kp, kd, derate)
    stall = sv.stall_torque
    T = motion.period
    ts = np.array(list(motion.t) + [T])
    Qk = np.vstack([motion.Q, motion.Q[:1]])

    def q_ref(t):
        tt = t % T
        i = min(int(np.searchsorted(ts, tt, side="right")) - 1, len(ts) - 2)
        w = (tt - ts[i]) / (ts[i + 1] - ts[i])
        return (1 - w) * Qk[i] + w * Qk[i + 1]

    groups, twist = tg.GAITS[gait]
    n = len(groups)
    win = T / n
    group_of = {f: k for k, grp in enumerate(groups) for f in grp}

    def swing_u(f, t):
        s = ((t - group_of[f] * win) % T) / T
        return s * n if s < 1.0 / n else None

    free = model.joint("body:free")
    data.qpos[free.qposadr[0] + 2] = tr.STAND_Z - drop * 1e-3 + (0.0 if attach else 0.002)   # on a wall: feet touching
    data.qpos[qadr] = q_ref(0.0)
    mujoco.mj_forward(model, data)
    dt = model.opt.timestep
    n_settle, n_run = int(settle_s / dt), int(cycles * T / dt)
    body = model.body("body").id
    foot_ids = {f: model.geom(f).id for f in tr.LEGS}
    foot_of = {v: k for k, v in foot_ids.items()}
    shell_ids = {model.geom("prosoma").id, model.geom("opisthosoma").id}
    ground_ids = {i for i in range(model.ngeom) if model.geom_bodyid[i] == 0}
    ghat = g / np.linalg.norm(g)
    gp = ghat - ghat[2] * np.array([0.0, 0.0, 1.0])                       # downhill, in the ground plane
    gpn = gp / np.linalg.norm(gp) if np.linalg.norm(gp) > 1e-6 else None
    v_plan = np.array(motion.base["velocity_m_s"][:2]) if not tg.GAITS[gait][1][2] else np.zeros(2)
    fault_cols = [k for k, nm in enumerate(names) if fault and nm.startswith(fault["leg"] + "_")]
    frozen = None
    tq_hist_sat, steps_run = 0, 0
    x0 = yaw0 = None
    ssm_min, ssm_neg, few_contacts = math.inf, 0, 0
    roll_max = pitch_max = 0.0
    z_min = math.inf
    belly_steps, slip_max, slip_sum, slip_n = 0, 0.0, 0.0, 0
    tau_peak = {"yaw": 0.0, "lift": 0.0, "knee": 0.0}
    energy = 0.0
    anchor = {}
    fell, why = False, None
    roll_after_push = 0.0
    c6 = np.zeros(6)
    trace = None
    if trace_fps:
        tbodies = [model.body(i).name for i in range(1, model.nbody)]
        tids = np.arange(1, model.nbody)
        trace = {"bodies": tbodies, "fps": trace_fps, "t": [], "pos": [], "quat": [], "attached": [], "slope_deg": incline_deg,
                 "gravity": g.tolist()}
        next_frame = -settle_s
    for step in range(n_settle + n_run):
        t_g = step * dt - settle_s
        tg_ = max(t_g, 0.0)
        qr = q_ref(tg_)
        qdr = (q_ref(tg_ + dt) - qr) / dt if t_g >= 0 else np.zeros_like(qr)
        q, qd = data.qpos[qadr].copy(), data.qvel[dadr].copy()
        if fault and t_g >= fault["t"] and fault["mode"] == "locked":
            if frozen is None:
                frozen = q[fault_cols].copy()
            qr[fault_cols], qdr[fault_cols] = frozen, 0.0
        tq = kp * (qr - q) + kd * (qdr - qd)
        lim = np.where(tq * qd < 0, stall, sv.torque_limit(qd))
        tq = np.clip(tq, -lim, lim)
        if fault and t_g >= fault["t"] and fault["mode"] == "limp":
            tq[fault_cols] = 0.0
        data.ctrl[act_servo] = tq + kd * qd
        if attach:
            for f in tr.LEGS:
                u = None if t_g < 0 else swing_u(f, tg_)
                lv = 1.0 if u is None else (max(0.0, 1 - u / 0.1) if u < 0.1 else (max(0.0, (u - 0.9) / 0.1)))
                if fault and fault["leg"] == f and t_g >= fault["t"]:
                    lv = 0.0
                data.ctrl[act_adh[f]] = lv
        data.xfrc_applied[body] = 0.0
        if push and t_g >= push["t"] and t_g < push["t"] + push.get("dur_s", 0.2):
            a = math.radians(push["dir_deg"])
            R = data.xmat[body].reshape(3, 3)
            data.xfrc_applied[body, :3] = push["force_n"] * (R @ np.array([math.cos(a), math.sin(a), 0.0]))
        mujoco.mj_step(model, data)
        if trace is not None and t_g >= next_frame - 1e-9:
            trace["t"].append(round(t_g, 4))
            trace["pos"].append(np.round(data.xpos[tids], 5).tolist())
            trace["quat"].append(np.round(data.xquat[tids], 5).tolist())
            trace["attached"].append([f for f in tr.LEGS if not attach or data.ctrl[act_adh[f]] > 0.5])
            next_frame += 1.0 / trace_fps
        if t_g < 0:
            continue
        steps_run += 1
        if x0 is None:
            x0 = data.xpos[body].copy()
            R0 = data.xmat[body].reshape(3, 3)
            yaw0 = math.atan2(R0[1, 0], R0[0, 0])
        # contacts: the feet down (and where), the belly
        down = {}
        belly = False
        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = c.geom1, c.geom2
            other = g2 if g1 in ground_ids else (g1 if g2 in ground_ids else None)
            if other is None:
                continue
            if other in foot_of:
                mujoco.mj_contactForce(model, data, i, c6)
                if c6[0] > 1e-3 or attach:
                    down[foot_of[other]] = c.pos.copy()
            elif other in shell_ids:
                belly = True
        belly_steps += belly
        com = data.subtree_com[body].copy()
        if len(down) >= 3:
            pts = np.array(list(down.values()))
            gz = float(np.mean(pts[:, 2]))
            tt = -(com[2] - gz) / ghat[2] if ghat[2] < -1e-6 else 0.0      # along gravity onto the feet's plane
            cp = com + tt * ghat if ghat[2] < -1e-6 else com
            m_ = _hull_margin(pts[:, :2], cp[:2])
            ssm_min = min(ssm_min, m_)
            ssm_neg += m_ < 0
        else:
            few_contacts += 1
        # stance slip: a foot in contact during its stance stays put
        for f in tr.LEGS:
            if f in down and swing_u(f, tg_) is None and not (fault and fault["leg"] == f):
                p = data.geom_xpos[foot_ids[f]].copy()
                if f not in anchor:
                    anchor[f] = p
                s_ = float(np.linalg.norm((p - anchor[f])[:2]))
                slip_max = max(slip_max, s_)
            else:
                if f in anchor:
                    slip_sum += float(np.linalg.norm((data.geom_xpos[foot_ids[f]] - anchor[f])[:2]))
                    slip_n += 1
                anchor.pop(f, None)
        R = data.xmat[body].reshape(3, 3)
        roll = math.degrees(math.atan2(R[2, 1], R[2, 2]))
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, -R[2, 0]))))
        roll_max, pitch_max = max(roll_max, abs(roll)), max(pitch_max, abs(pitch))
        if push and t_g >= push["t"]:
            roll_after_push = max(roll_after_push, abs(roll), abs(pitch))
        z_min = min(z_min, float(data.xpos[body][2]))
        act_tq = data.actuator_force[act_servo]
        for k, nm in enumerate(names):
            kind = nm.split("_")[1]
            tau_peak[kind] = max(tau_peak[kind], abs(float(act_tq[k])))
        tq_hist_sat += int(np.sum(np.abs(act_tq) >= 0.98 * stall))
        energy += (float(np.sum(np.maximum(act_tq * qd, 0.0))) / DRIVE_EFF + IDLE_W * len(names)) * dt
        # sliding: how far it is downhill of where the gait should have taken it
        disp = data.xpos[body] - x0
        c0, s0 = math.cos(yaw0), math.sin(yaw0)
        plan = np.array([c0 * v_plan[0] - s0 * v_plan[1], s0 * v_plan[0] + c0 * v_plan[1], 0.0]) * t_g
        slid = float(np.dot(disp - plan, gpn)) if gpn is not None else 0.0
        if abs(roll) > 50 or abs(pitch) > 50:
            fell, why = True, "tipped"
        elif slid > 0.15:
            fell, why = True, "slid"
        if fell:
            break
    elapsed = steps_run * dt
    disp = (data.xpos[body] - x0) if x0 is not None else np.zeros(3)
    R = data.xmat[body].reshape(3, 3)
    yaw = math.atan2(R[1, 0], R[0, 0])
    dyaw = math.degrees((yaw - (yaw0 or 0.0) + math.pi) % (2 * math.pi) - math.pi)
    Rb = np.array([[math.cos(yaw0 or 0), -math.sin(yaw0 or 0)], [math.sin(yaw0 or 0), math.cos(yaw0 or 0)]])
    fwd, side = (Rb.T @ disp[:2]) if x0 is not None else (0.0, 0.0)
    planned = motion.params.get("speed_mm_s")
    vx, vy, wz = twist
    achieved = (fwd if vx else side if vy else 0.0) * 1000 / max(elapsed, 1e-9)
    rate = dyaw / max(elapsed, 1e-9)
    if not fell and planned and achieved < 0.2 * planned:
        fell, why = True, "stopped"
    yaw_plan = motion.params.get("yaw_rate_deg_s")
    if not fell and yaw_plan and rate < 0.2 * yaw_plan:
        fell, why = True, "stopped"
    mass = robot.total_mass()
    p_mean = energy / max(elapsed, 1e-9) + ELECTRONICS_W
    v = abs(achieved) * 1e-3
    return {"gait": gait, "posture": posture if isinstance(posture, str) else f"drop {drop:g} mm",
            "simulated_s": round(elapsed, 3), "fell": fell, "failure": why,
            "speed_mm_s": round(achieved, 2), "planned_speed_mm_s": round(planned, 2) if planned else None,
            "tracking": round(achieved / planned, 3) if planned else None,
            "yaw_rate_deg_s": round(rate, 2), "planned_yaw_rate_deg_s": motion.params.get("yaw_rate_deg_s"),
            "heading_drift_deg": round(dyaw, 2) if not wz else None,
            "sideways_drift_mm": round(float(side) * 1000, 1) if vx else None,
            "dynamic_margin_min_mm": round(ssm_min * 1000, 1) if math.isfinite(ssm_min) else None,
            "dynamic_margin_negative_s": round(ssm_neg * dt, 3), "under_3_feet_s": round(few_contacts * dt, 3),
            "static_margin_min_mm": round(motion.check["static_margin_min_m"] * 1000, 1),
            "roll_max_deg": round(roll_max, 2), "pitch_max_deg": round(pitch_max, 2),
            "tilt_after_push_deg": round(roll_after_push, 2) if push else None,
            "body_z_min_mm": round(z_min * 1000, 1), "belly_contact_s": round(belly_steps * dt, 3),
            "stance_slip_max_mm": round(slip_max * 1000, 2),
            "stance_slip_mean_mm": round(slip_sum / slip_n * 1000, 2) if slip_n else 0.0,
            "torque_peak_over_stall": {k: round(v_ / stall, 3) for k, v_ in tau_peak.items()},
            "saturated_fraction": round(tq_hist_sat / max(steps_run * len(names), 1), 4),
            "power_w": round(p_mean, 2), "cost_of_transport": round(p_mean / (mass * 9.81 * v), 2) if v > 1e-4 else None,
            "runtime_h": round(BATTERY_WH / p_mean, 2), "mass_kg": round(mass, 3),
            "attach_pull_n": round(pull, 2) if attach else None, "attach_mu": round(mu_att, 3) if attach else None,
            **({"trace": trace} if trace is not None else {})}


# --------------------------------------------------------------------------------------------- the studies
def _walk_cases(quick):
    if quick:
        return [{"gait": gg, "posture": "standing", "payload_g": 300.0, "mu": 0.8} for gg in ("wave4", "tetrapod")]
    return [{"gait": gg, "posture": p, "payload_g": w, "mu": m}
            for gg, p, w, m in itertools.product(("wave8", "wave4", "tetrapod", "turn", "crab"), ("standing", "crouched"),
                                                 (0.0, 300.0), (0.5, 0.8))]


def _slope_cases(quick):
    if quick:
        return [{"gait": "wave4", "posture": "crouched", "incline_deg": a, "heading_deg": h, "mu": 0.8}
                for a in (15.0, 30.0) for h in (0.0, 90.0)]
    return [{"gait": gg, "posture": p, "incline_deg": a, "heading_deg": h, "mu": m}
            for gg, p, a, h, m in itertools.product(("wave8", "wave4", "tetrapod"), ("standing", "crouched"),
                                                    (0.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0), (0.0, 90.0, 180.0),
                                                    (0.6, 0.9))]


def _rough_cases(quick):
    ters = [{"kind": "rough", "rms_mm": r, "corr_mm": 80.0} for r in (10.0, 20.0, 35.0)] + \
           [{"kind": "steps", "height_mm": h, "spacing_mm": 300.0} for h in (30.0, 60.0, 90.0)]
    if quick:
        return [{"gait": "wave4", "posture": "standing", "terrain": t, "step_height": 80.0, "seed": 0}
                for t in (ters[1], ters[4])]
    out = []
    for gg, t, h, s in itertools.product(("wave8", "wave4", "tetrapod"), ters, (40.0, 80.0), (0, 1)):
        t = dict(t, seed=s) if t["kind"] == "rough" else t
        if t["kind"] == "steps" and s:
            continue
        out.append({"gait": gg, "posture": "standing", "terrain": t, "step_height": h, "seed": s})
    return out


def _push_cases(quick):
    if quick:
        return [{"gait": "wave4", "posture": "standing", "push": {"t": 2.0, "force_n": 20.0, "dir_deg": 90.0,
                                                                   "dur_s": 0.2}}]
    return [{"gait": gg, "posture": p, "push": {"t": 2.0, "force_n": f, "dir_deg": d, "dur_s": 0.2}}
            for gg, p, f, d in itertools.product(("wave4", "tetrapod"), ("standing", "crouched"), (10.0, 20.0, 40.0, 80.0),
                                                 (0.0, 90.0, 180.0))]


def _fault_cases(quick):
    if quick:
        return [{"gait": "wave4", "posture": "standing", "fault": {"leg": "L2", "mode": "limp", "t": 1.0}}]
    return [{"gait": gg, "posture": "standing", "fault": {"leg": f, "mode": m, "t": 1.0}}
            for gg, f, m in itertools.product(("wave8", "wave4", "tetrapod"), ("L1", "L2", "L3", "L4"),
                                              ("limp", "locked"))]


def _climb_cases(quick):
    feet = [("pad", "glass"), ("spine", "brick"), ("spine", "bark"), ("magnet", "bare steel"), ("pad", "painted wall")]
    if quick:
        return [{"gait": "wave4", "posture": "crouched", "attach": "spine:brick", "incline_deg": 90.0, "mu": 0.8,
                 "cycles": 2}]
    return [{"gait": gg, "posture": "crouched", "attach": f"{m}:{s}", "incline_deg": a, "mu": 0.8, "cycles": 3}
            for (m, s), a, gg in itertools.product(feet, (45.0, 60.0, 75.0, 90.0), ("wave8", "wave4"))]


def _gaits_cases(quick):
    import tarantula_gaits as tg
    if quick:
        return [{"gait": gg, "posture": "standing"} for gg in ("wave4", "tetrapod")]
    return [{"gait": gg, "posture": p, "stride": s, "step_height": h}
            for gg, p, s, h in itertools.product(tuple(tg.GAITS), tuple(tg.POSTURES), (0.75, 1.0, 1.25), (30.0, 60.0))]


def _gaits_run(case):
    import tarantula_gaits as tg
    stride = tg.DEFAULTS[case["gait"]][0] * case.get("stride", 1.0)
    m = tg.gait(case["gait"], stride=stride, step_height=case.get("step_height"),
                body_drop_mm=tg.POSTURES[case["posture"]])
    c = m.check
    return dict(case, stride=stride, **{k: m.params[k] for k in ("period_s", "duty", "stance_min")},
                speed_mm_s=m.params.get("speed_mm_s"), yaw_rate_deg_s=m.params.get("yaw_rate_deg_s"),
                ik_error_max_mm=round(c["ik_error_max_m"] * 1000, 3), limit_margin_min_rad=round(c["limit_margin_min_rad"], 3),
                static_margin_min_mm=round(c["static_margin_min_m"] * 1000, 1),
                tip_deg_min={k: round(v, 1) for k, v in c["tip_deg_min"].items()},
                ok=bool(c["ik_error_max_m"] < 1e-3 and c["limit_margin_min_rad"] > 0 and c["static_margin_min_m"] > 0))


def _sim_run(case):
    return dict(case, **simulate(**case))


def _ok(r):
    return not r.get("error") and not r["fell"]


def _walk_summary(rs):
    return {f"{r['gait']} {r['posture']} payload {r['payload_g']:.0f} g μ {r['mu']}": {
        "ok": _ok(r), "speed_mm_s": r.get("speed_mm_s"), "tracking": r.get("tracking"), "power_w": r.get("power_w"),
        "cost_of_transport": r.get("cost_of_transport"), "runtime_h": r.get("runtime_h"),
        "dynamic_margin_min_mm": r.get("dynamic_margin_min_mm"), "drift_deg": r.get("heading_drift_deg")}
        for r in rs if not r.get("error")}


def _slope_summary(rs):
    out = {}
    for r in rs:
        if r.get("error"):
            continue
        k = f"{r['gait']} {r['posture']} heading {r['heading_deg']:.0f}° μ {r['mu']}"
        cur = out.setdefault(k, {"steepest_ok_deg": None, "first_failure": None})
        if _ok(r) and (cur["steepest_ok_deg"] is None or r["incline_deg"] > cur["steepest_ok_deg"]):
            cur["steepest_ok_deg"] = r["incline_deg"]
        if not _ok(r) and (cur["first_failure"] is None or r["incline_deg"] < cur["first_failure"]["incline_deg"]):
            cur["first_failure"] = {"incline_deg": r["incline_deg"], "how": r.get("failure")}
    return out


def _rough_summary(rs):
    out = {}
    for r in rs:
        if r.get("error"):
            continue
        t = r["terrain"]
        tk = f"rough {t['rms_mm']:.0f} mm" if t["kind"] == "rough" else f"steps {t['height_mm']:.0f} mm"
        k = f"{r['gait']} on {tk}, step {r['step_height']:.0f} mm"
        cur = out.setdefault(k, {"runs": 0, "ok": 0, "belly_contact_s": 0.0, "speed_mm_s": []})
        cur["runs"] += 1
        cur["ok"] += _ok(r)
        cur["belly_contact_s"] = max(cur["belly_contact_s"], r.get("belly_contact_s", 0.0))
        cur["speed_mm_s"].append(r.get("speed_mm_s"))
    return out


def _push_summary(rs):
    out = {}
    for r in rs:
        if r.get("error"):
            continue
        k = f"{r['gait']} {r['posture']} push {r['push']['dir_deg']:.0f}°"
        cur = out.setdefault(k, {"survives_up_to_n": 0.0, "falls_at_n": None})
        f = r["push"]["force_n"]
        if _ok(r):
            cur["survives_up_to_n"] = max(cur["survives_up_to_n"], f)
        elif cur["falls_at_n"] is None or f < cur["falls_at_n"]:
            cur["falls_at_n"] = f
    return out


def _fault_summary(rs):
    return {f"{r['gait']} {r['fault']['leg']} {r['fault']['mode']}": {
        "ok": _ok(r), "speed_mm_s": r.get("speed_mm_s"), "tracking": r.get("tracking"),
        "drift_deg": r.get("heading_drift_deg"), "dynamic_margin_min_mm": r.get("dynamic_margin_min_mm"),
        "failure": r.get("failure")} for r in rs if not r.get("error")}


def _climb_summary(rs):
    out = {}
    for r in rs:
        if r.get("error"):
            continue
        k = f"{r['attach']} {r['gait']}"
        cur = out.setdefault(k, {"steepest_ok_deg": None, "slip_max_mm": 0.0})
        if _ok(r) and (cur["steepest_ok_deg"] is None or r["incline_deg"] > cur["steepest_ok_deg"]):
            cur["steepest_ok_deg"] = r["incline_deg"]
        cur["slip_max_mm"] = max(cur["slip_max_mm"], r.get("stance_slip_max_mm") or 0.0)
    return out


def _gaits_summary(rs):
    return {f"{r['gait']} {r['posture']} stride {r['stride']:.0f} step {r['step_height']:.0f} mm"
            if "step_height" in r and r["step_height"] is not None else f"{r['gait']} {r['posture']}": {
                "ok": r.get("ok"), "speed_mm_s": r.get("speed_mm_s"),
                "static_margin_min_mm": r.get("static_margin_min_mm"), "tip_deg_min": r.get("tip_deg_min")}
            for r in rs if not r.get("error")}


#: name -> (cases(quick), run(case), summary(results), about)
CASES = {
    "tarantula-gaits": (_gaits_cases, _gaits_run, _gaits_summary,
                        "kinematic gait checks: IK, joint limits, static stability and tip-over angles"),
    "tarantula-walk": (_walk_cases, _sim_run, _walk_summary,
                       "MuJoCo walking on flat ground: gait × posture × payload × friction"),
    "tarantula-slope": (_slope_cases, _sim_run, _slope_summary,
                        "MuJoCo on inclines 0–40°: gait × posture × heading (up, across, down) × friction"),
    "tarantula-rough": (_rough_cases, _sim_run, _rough_summary,
                        "MuJoCo on rough ground (10–35 mm RMS) and steps (30–90 mm): gait × step height"),
    "tarantula-push": (_push_cases, _sim_run, _push_summary,
                       "MuJoCo push recovery: 10–80 N for 0.2 s from the front, side and back"),
    "tarantula-faults": (_fault_cases, _sim_run, _fault_summary,
                         "MuJoCo with a failed leg (limp or locked): gait × which leg"),
    "tarantula-climb": (_climb_cases, _sim_run, _climb_summary,
                        "MuJoCo climbing 45–90° with pads, spines or magnets (adhesion on in stance, off in swing)"),
}


def cases(name, quick):
    return CASES[name][0](quick)


def run_case(name, case):
    return CASES[name][1](case)


def summary(name, results):
    return CASES[name][2](results)
