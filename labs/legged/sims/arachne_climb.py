"""ARACHNE climbing a wall in MuJoCo: the robot model (``arachne_robot``) on a plane that is the wall (gravity along
the wall, ``heading_deg`` from straight up), its 24 servos tracking a gait's keyframes (``arachne_gaits``) through
PD loops clipped to the micro servo's torque–speed line when driving (full stall torque when braking: a real servo
brakes a back-driven joint), and a MuJoCo **adhesion actuator** on each foot switched by
the gait's attach / peel phases.

**Feet.** Each pad is its own body on the tibia: a normal spring (the foam, ``arachne_statics.K_N``) and two ankle
hinges free to ±18° with hard limits (the crossed-flexure ankle and its peel stops). The pad is a collision box
(18 × 14 mm), so a pad loaded off-centre lifts at an edge — peel emerges from the contacts. Adhesion: a normal pull of σ·A when attached (``arachne_specs.ADHESIVE['normal_kpa']`` × the pad area,
scattered per foot by ±``adhesion_scatter``); friction μ = τ/σ, so with no other normal load a pad holds τ·A in shear
(frictional adhesion: pushing a pad in raises its shear capacity, pulling it off lowers it).

**Run.** Settle 0.5 s with every foot attached, then ``cycles`` gait cycles. ``fault_foot`` loses its adhesion at
``fault_t`` and never gets it back (the open-loop gait carries on: the recovery behaviour of ``arachne_control`` is not
in this run — it shows what the other seven feet do alone).

**Out** (``run``): distance climbed and speed, the worst slip of an attached foot (stance and test phases; the
deliberate drag excluded), detachments, a fall, body
pitch/roll, peak joint torque / stall, mean electrical power and runtime on the battery, adhesion utilisation.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np

__all__ = ["build_model", "run", "surface_contact", "SURFACE_DEFAULT"]

SURFACE_DEFAULT = "glass"


def surface_contact(surface, module):
    """(pull-off force per foot [N], friction μ): a pad holds σ·A off the wall and τ·A in shear (μ = τ/σ); a spine
    foot holds n·f·engaged in shear and a quarter of that off the wall (μ = 4) — isotropic here, so its sideways grip
    is optimistic (real spines only grip along their direction: arachne_statics models that)."""
    import arachne_specs as sp
    if module == "spine":
        cap = sp.SPINES["per_spine_n"][surface] * sp.FOOT_SPINE["spines"] * sp.SPINES["engaged_fraction"]
        pull = cap * sp.SPINES["pulloff_over_shear"]
        return pull, (cap / pull if pull > 0 else 0.6)
    a, b = sp.FOOT_PAD["pad"]
    area = a * b * 1e-6
    sig = sp.ADHESIVE["normal_kpa"][surface] * 1e3
    tau = sp.ADHESIVE["shear_kpa"][surface] * 1e3
    return sig * area, (tau / sig if sig > 0 else 0.6)


def build_model(robot, *, surface, module="pad", heading_deg=0.0, adhesion_scatter=0.0, seed=0, timestep=0.0005,
                noslip=4):
    """(MjModel, info) of ``robot`` on the wall: gravity along the wall, pad boxes colliding with adhesion
    actuators, foot spheres off."""
    import mujoco
    from vegeta.chiron.robot import SimOptions
    import arachne_specs as sp

    h = math.radians(heading_deg)
    g = (-sp.G * math.cos(h), sp.G * math.sin(h), 0.0)
    # noslip iterations: MuJoCo's soft contacts otherwise let a loaded pad creep (≈ 3× the slip at 0 here)
    opt = SimOptions(timestep=timestep, gravity=g, cone="elliptic", impratio=10.0, contact_solref=(0.002, 1.0),
                     noslip_iterations=noslip)
    root = ET.fromstring(robot.to_mjcf(options=opt))
    pull, mu = surface_contact(surface, module)
    rng = np.random.default_rng(seed)
    gains = {}
    for gm in root.iter("geom"):
        if gm.get("name", "") in sp.LEGS:                                 # the contact spheres: off
            gm.set("contype", "0")
            gm.set("conaffinity", "0")
    # each pad becomes its own body on the tibia: a normal spring (the foam, K_N) and two ankle hinges free to
    # ±ankle_tilt_free_deg (the crossed flexure; the limits are its peel stops)
    import arachne_statics as st
    free = math.radians(sp.LEG["ankle_tilt_free_deg"])
    for leg in sp.LEGS:
        tib = next(bd for bd in root.iter("body") if bd.get("name") == f"{leg}_tibia")
        pad = next(gm for gm in tib.findall("geom") if gm.get("name") == f"{leg}_pad")
        tib.remove(pad)
        foot = ET.SubElement(tib, "body", {"name": f"{leg}_foot", "pos": pad.get("pos"), "quat": pad.get("quat", "1 0 0 0")})
        ET.SubElement(foot, "joint", {"name": f"{leg}_foam", "type": "slide", "axis": "0 0 1", "stiffness": f"{st.K_N}",
                                      "damping": "0.02", "range": "-0.002 0.002", "limited": "true", "armature": "1e-5"})
        for ax, nm in (("1 0 0", "roll"), ("0 1 0", "pitch")):
            ET.SubElement(foot, "joint", {"name": f"{leg}_ankle_{nm}", "type": "hinge", "axis": ax, "stiffness": "0.003",
                                          "damping": "2e-5", "range": f"{-free:.4f} {free:.4f}", "limited": "true",
                                          "armature": "1e-7"})
        pad.set("pos", "0 0 0")
        pad.set("quat", "1 0 0 0")
        pad.set("contype", "2")
        pad.set("conaffinity", "1")
        pad.set("condim", "3")
        pad.set("friction", f"{mu:.4f} 0.005 0.0001")
        pad.set("margin", "0.001")
        pad.set("gap", "0.0")
        pad.set("group", "0")
        foot.append(pad)
    act = root.find("actuator")
    if act is None:
        act = ET.SubElement(root, "actuator")
    for f in sp.LEGS:
        k = float(rng.uniform(1 - adhesion_scatter, 1 + adhesion_scatter)) if adhesion_scatter else 1.0
        gains[f] = pull * k
        ET.SubElement(act, "adhesion", {"name": f"adh:{f}", "body": f"{f}_foot", "ctrlrange": "0 1",
                                        "gain": f"{gains[f]:.5f}"})
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return model, {"mu": mu, "adhesion_n": gains, "shear_capacity_n": {f: mu * v for f, v in gains.items()},
                   "gravity": g}


def _adhesion_level(phase, u_in_phase):
    if phase in ("stance", "test", "drag"):
        return 1.0
    if phase == "preload":
        return u_in_phase
    if phase == "peel":
        return max(0.0, 1.0 - 2.0 * u_in_phase)
    return 0.0


def run(gait="climb", surface=SURFACE_DEFAULT, payload_g=20.0, heading_deg=0.0, cycles=3, adhesion_scatter=0.0,
        seed=0, fault_foot=None, fault_t=1.5, timestep=0.0005, settle_s=0.5, noslip=4):
    """One simulated climb; returns a flat dict of results (see the module docstring)."""
    import mujoco
    import arachne_gaits as ag
    import arachne_robot as ar
    import arachne_specs as sp
    import motion_kit as mk

    robot = ar.arachne(payload_g)
    kin = mk.Kin(robot)
    motion = ag.wall_gait(kin, gait, surface=surface)
    module = "spine" if surface in ("bark", "brick") else "pad"
    model, info = build_model(robot, surface=surface, module=module, heading_deg=heading_deg,
                              adhesion_scatter=adhesion_scatter, seed=seed, timestep=timestep, noslip=noslip)
    data = mujoco.MjData(model)
    names = [model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
             and model.joint(i).name.split("_")[1] in ("yaw", "lift", "knee")]
    qadr = np.array([model.joint(n).qposadr[0] for n in names])
    dadr = np.array([model.joint(n).dofadr[0] for n in names])
    act_servo = np.array([model.actuator(f"servo:{n}").id for n in names])
    act_adh = {f: model.actuator(f"adh:{f}").id for f in sp.LEGS}
    sv = ar.servo()
    kp, kd = sv.kp, sv.kd
    stall = sv.stall_torque
    # keyframes over a cycle for every hinge (the motion lists only the joints that move)
    T = motion.period
    ts = np.array(motion.t + [T])
    Qk = np.tile(kin.q0, (len(ts), 1))
    for k, j in enumerate(motion.joints):
        col = kin.index[j]
        Qk[:-1, col] = motion.Q[:, k]
        Qk[-1, col] = motion.Q[0, k]
    order = [kin.index[n] for n in names]                                 # kin order → model order
    groups, swing_t, stride, twist, drag, _ = ag.GAITS[gait]
    group_of = {f: g for g, grp in enumerate(groups) for f in grp}
    kin.set(kin.q0)
    o = kin.data.xpos[kin.body("body")].copy()
    p0 = {f: kin.data.geom_xpos[kin.model.geom(f).id] - o for f in sp.LEGS}
    win = T / len(groups)
    phase_start = dict(ag.STEP)
    phase_names = [n for n, _ in ag.STEP]

    def phase_of(f, t):
        s = ((t - group_of[f] * win) % T) / T
        sw = win / T
        if s >= sw:
            return "stance", 0.0
        u = s / sw
        k = max(i for i, n in enumerate(phase_names) if u >= phase_start[n])
        a = phase_start[phase_names[k]]
        b = phase_start[phase_names[k + 1]] if k + 1 < len(phase_names) else 1.0
        return phase_names[k], (u - a) / (b - a)

    def q_ref(t):
        tt = t % T
        i = min(np.searchsorted(ts, tt, side="right") - 1, len(ts) - 2)
        w = (tt - ts[i]) / (ts[i + 1] - ts[i])
        return (1 - w) * Qk[i] + w * Qk[i + 1]

    data.qpos[qadr] = q_ref(0.0)[order]                                  # start in the gait's first pose: no
    mujoco.mj_forward(model, data)                                        # leg fights a planted foot
    dt = model.opt.timestep
    n_settle = int(settle_s / dt)
    n_run = int(cycles * T / dt)
    body = model.body("body").id
    pad_ids = {f: model.geom(f"{f}_pad").id for f in sp.LEGS}
    tib_ids = {f: model.body(f"{f}_tibia").id for f in sp.LEGS}
    x0 = None
    slip_max, detach, util_max, pitch_max, roll_max = 0.0, 0, 0.0, 0.0, 0.0
    tau_peak = {"yaw": 0.0, "lift": 0.0, "knee": 0.0}
    energy = 0.0
    anchor = {}
    fell = False
    I_stall, I_idle, V = sp.SERVO["stall_current_a"], sp.SERVO["idle_current_a"], sp.SERVO["volt_run"]
    for step in range(n_settle + n_run):
        t_g = step * dt - settle_s                                       # gait time (negative while settling)
        tg = max(t_g, 0.0)
        qr = q_ref(tg)[order]
        qr_n = q_ref(tg + dt)[order]
        qdr = (qr_n - qr) / dt if t_g >= 0 else np.zeros_like(qr)
        q, qd = data.qpos[qadr], data.qvel[dadr]
        tq = kp * (qr - q) + kd * (qdr - qd)
        lim = np.where(tq * qd < 0, stall, sv.torque_limit(qd))           # braking: full torque (four-quadrant)
        tq = np.clip(tq, -lim, lim)
        data.ctrl[act_servo] = tq + kd * qd                              # the actuator adds −kd·qd itself
        levels, phases = {}, {}
        for f in sp.LEGS:
            ph, u = ("stance", 0.0) if t_g < 0 else phase_of(f, tg)
            phases[f] = ph
            lv = _adhesion_level(ph, u)
            if fault_foot == f and t_g >= fault_t:
                lv = 0.0
            levels[f] = lv
            data.ctrl[act_adh[f]] = lv
        mujoco.mj_step(model, data)
        if t_g < 0:
            continue
        if x0 is None:
            x0 = data.xpos[body].copy()
        # per-foot contact forces on the pads
        fn = {f: 0.0 for f in sp.LEGS}
        ft = {f: 0.0 for f in sp.LEGS}
        c6 = np.zeros(6)
        for i in range(data.ncon):
            c = data.contact[i]
            for f, gid in pad_ids.items():
                if c.geom1 == gid or c.geom2 == gid:
                    mujoco.mj_contactForce(model, data, i, c6)
                    fn[f] += c6[0]
                    ft[f] += math.hypot(c6[1], c6[2])
        for f in sp.LEGS:
            if phases[f] in ("stance", "test") and f != fault_foot:
                pc = data.geom_xpos[pad_ids[f]].copy()
                if f not in anchor:
                    anchor[f] = pc
                slip_max = max(slip_max, float(np.linalg.norm((pc - anchor[f])[:2])))
                if fn[f] <= 1e-6:
                    detach += 1
                util_max = max(util_max, ft[f] / max(info["mu"] * max(fn[f], 1e-9), 1e-9) if fn[f] > 0 else 0.0)
            else:
                anchor.pop(f, None)
        R = data.xmat[body].reshape(3, 3)
        pitch_max = max(pitch_max, abs(math.degrees(math.asin(max(-1, min(1, -R[2, 0]))))))
        roll_max = max(roll_max, abs(math.degrees(math.atan2(R[2, 1], R[2, 2]))))
        act_tq = data.actuator_force[act_servo]
        for k, n in enumerate(names):
            kind = n.split("_")[1]
            tau_peak[kind] = max(tau_peak[kind], abs(float(act_tq[k])))
        cur = I_idle * len(names) + I_stall * float(np.sum(np.minimum(1.0, np.abs(act_tq) / stall)))
        energy += cur * V * dt
        z_body = data.xpos[body][2]
        if z_body > 0.06 or (data.xpos[body] - x0)[0] < -0.05 * math.cos(math.radians(heading_deg)) - 0.02:
            fell = True
            break
    elapsed = (step + 1 - n_settle) * dt
    disp = (data.xpos[body] - x0) if x0 is not None else np.zeros(3)
    p_mean = energy / max(elapsed, 1e-9) + sp.ELECTRONICS["power_w"]
    wh = sp.BATTERY["capacity_mah"] / 1000 * sp.BATTERY["volt_nominal"] * sp.BATTERY["usable_fraction"]
    return {"gait": gait, "surface": surface, "module": module, "noslip": noslip, "payload_g": payload_g, "heading_deg": heading_deg, "seed": seed,
            "adhesion_scatter": adhesion_scatter, "fault_foot": fault_foot, "simulated_s": round(elapsed, 3),
            "fell": fell, "climbed_mm": round(float(disp[0]) * 1000, 2), "sideways_mm": round(float(disp[1]) * 1000, 2),
            "speed_mm_s": round(float(disp[0]) * 1000 / max(elapsed, 1e-9), 2),
            "planned_speed_mm_s": motion.params.get("speed_mm_s"),
            "slip_max_mm": round(slip_max * 1000, 3), "detached_steps": detach,
            "friction_use_max": round(util_max, 3), "pitch_max_deg": round(pitch_max, 2),
            "roll_max_deg": round(roll_max, 2),
            "torque_peak_over_stall": {k: round(v / stall, 3) for k, v in tau_peak.items()},
            "power_w": round(p_mean, 2), "runtime_min": round(wh / p_mean * 60, 1), "mu": round(info["mu"], 3),
            "adhesion_n_mean": round(float(np.mean(list(info["adhesion_n"].values()))), 4)}
