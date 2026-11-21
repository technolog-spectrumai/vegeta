"""ARACHNE's gaits (``motion_kit``), exported as ``data:motions`` (tag ``motion``) and checked frame by frame against
the statics (``arachne_statics``): every keyframe carries how many feet are attached and the worst foot's adhesion
margin, and every gait its joint torques, servo currents and power.

**A step** (one foot; the fractions are of its swing window):

* 0.00–0.12 **peel** — the foot lifts 2 mm while the tibia rolls the ankle onto its 25° peel stop: the pad comes off
  heel first, a line of fibres at a time, at a small force (``ADHESIVE['peel_force_n']``), never pulled off flat;
* 0.12–0.55 **swing** — up 6 mm and across to the new spot;
* 0.55–0.70 **approach + preload** — down until the pad presses 0.375 mm into the wall (0.3 N at the pad's 800 N/m);
* 0.70–0.85 **drag** — 1.5 mm along gravity (3 mm on spines): loads the fibres in shear (adhesion rises with
  shear for these adhesives) or engages the spines;
* 0.85–1.00 **test** — the foot holds still while the controller pulls 50 % of its expected load through it and
  watches the Hall sensor (``arachne_control``); only then does it count as attached.

A foot counts as attached from the end of its drag to its next peel. **Gaits** (feet moving together in a group):

* ``climb`` — the wave: four groups of two diagonal feet, six attached always (glass, acrylic): 13 mm/s;
* ``climb_fast`` — the alternating tetrapod, four attached (glass and acrylic only: margin ≥ 2 checked): 33 mm/s;
* ``climb_careful`` — one foot at a time, seven attached (varnished wood; bark with spines + tail): 5 mm/s;
* ``turn`` — the wave stepping round the body centre, 10° per cycle;
* ``crab`` — the careful gait sideways, the heading held up (bark: spines only grip along their direction);
* ``park`` — all feet re-placed one at a time into the parking stance (legs along gravity: low holding torque);
* ``reattach`` — the recovery step when a foot fails its test: retract, move 3 mm, preload, drag, test again.
"""
from __future__ import annotations

import math

import numpy as np

import arachne_specs as sp
import arachne_statics as st
import motion_kit as mk

__all__ = ["GAITS", "wall_gait", "motions", "parts", "STEP", "PRELOAD_M", "IK_REG"]

PRELOAD_M = sp.ADHESIVE["preload_n"] / st.K_N                               # 0.375 mm into the wall
#: the IK's pull towards the standing pose, scaled to a robot whose feet move millimetres (motion_kit's default
#: suits a metre-sized robot and would leave ~1 mm errors here)
IK_REG = 1e-9
#: the step's phases (fractions of the swing window): (name, start)
STEP = (("peel", 0.0), ("swing", 0.12), ("preload", 0.55), ("drag", 0.70), ("test", 0.85))

#: name -> (groups of feet moving together, swing time per group [s], stride [mm] (turn: rad), twist (vx, vy, wz)
#: direction, drag [mm], surfaces it is for). Bark and brick gaits run on the spine feet with the tail.
GAITS = {
    "climb": ([("L1", "R3"), ("R2", "L4"), ("R1", "L3"), ("L2", "R4")], 0.5, 20.0, (1, 0, 0), 1.5,
              ("glass", "acrylic")),
    "climb_fast": ([("L1", "R2", "L3", "R4"), ("R1", "L2", "R3", "L4")], 0.6, 20.0, (1, 0, 0), 1.5,
                   ("glass", "acrylic")),
    "climb_careful": ([("L4",), ("R2",), ("L3",), ("R1",), ("R4",), ("L2",), ("R3",), ("L1",)], 0.5, 20.0,
                      (1, 0, 0), 1.5, ("varnished wood",)),
    "turn": ([("L1", "R3"), ("R2", "L4"), ("R1", "L3"), ("L2", "R4")], 0.5, math.radians(10.0), (0, 0, 1), 1.5,
             ("glass", "acrylic")),
    "crab": ([("L4",), ("R2",), ("L3",), ("R1",), ("R4",), ("L2",), ("R3",), ("L1",)], 0.5, 15.0, (0, 1, 0), 3.0,
             ("bark", "brick")),
}


def _foot_offsets(kin, group_of, groups, period, t, stride, twist, drag, p0):
    """Each foot's target offset from its standing point at time ``t`` [m] and whether it is attached."""
    n = len(groups)
    window = period / n
    vx, vy, wz = twist
    out = {}
    for f, p in p0.items():
        g = group_of[f]
        s = ((t - g * window) % period) / period                         # 0 at this foot's lift-off
        sw = window / period                                              # the swing fraction
        # the stance displacement D the body makes over the stance (foot moves -D relative to the body)
        if wz:
            ang = stride
            D = np.array([-ang * p[1], ang * p[0], 0.0])                  # rotation by +stride about the centre
        else:
            D = np.array([vx, vy, 0.0]) * stride * 1e-3
        dg = np.array([-1.0, 0.0, 0.0]) * drag * 1e-3                    # the drag: along gravity (climbing: −x)
        if s < sw:                                                        # the swing window
            u = s / sw
            ph = [name for name, a in STEP if u >= a][-1]
            start = -D / 2                                                # where stance ended
            end = D / 2 - dg                                              # where the next stance starts, before drag
            if ph == "peel":
                w = u / STEP[1][1]
                off, z = start, 0.002 * w
            elif ph == "swing":
                w = mk.smooth((u - STEP[1][1]) / (STEP[2][1] - STEP[1][1]))
                off = start + (end - start) * w
                z = 0.002 * (1 - w) + 0.006 * math.sin(math.pi * w)         # from the peel height, 6 mm arc
            elif ph == "preload":
                w = mk.smooth((u - STEP[2][1]) / (STEP[3][1] - STEP[2][1]))
                off, z = end, -PRELOAD_M * w
            elif ph == "drag":
                w = mk.smooth((u - STEP[3][1]) / (STEP[4][1] - STEP[3][1]))
                off, z = end + dg * w, -PRELOAD_M
            else:                                                         # test
                off, z = end + dg, -PRELOAD_M
            attached = ph == "test"
        else:                                                             # stance: the body moves on
            u = (s - sw) / (1 - sw)
            off, z, attached = D / 2 + (-D) * u, -PRELOAD_M, True
            ph = "stance"
        out[f] = (np.array([off[0], off[1], z]), attached, ph)
    return out


def wall_gait(kin, name, *, frames=48, surface=None, module="pad", tail=False):
    """One of ``GAITS`` as a Motion: IK of the 24 servos on the feet's targets, and per keyframe the attached feet,
    their adhesion margin (``arachne_statics.check`` on ``surface``, climbing: wall, heading up) and the servo
    currents; ``check`` holds the worst of each, ``params`` the speed, the cycle and the mean electrical power."""
    groups, swing_t, stride, twist, drag, surfaces = GAITS[name]
    surface = surface or surfaces[0]
    if surface in ("bark", "brick"):
        module, tail = "spine", True
    n = len(groups)
    period = swing_t * n
    group_of = {f: g for g, grp in enumerate(groups) for f in grp}
    kin.set(kin.q0)
    o = kin.data.xpos[kin.body("body")].copy()
    p0 = {f: kin.data.geom_xpos[kin.model.geom(f).id] - o for f in sp.LEGS}
    pts = {f: kin.geom_point(f) for f in sp.LEGS}
    z0 = {f: kin.data.xmat[kin.body(f"{f}_tibia")].reshape(3, 3).T @ np.array([0.0, 0.0, 1.0]) for f in sp.LEGS}
    ts = [period * i / frames for i in range(frames)]
    Q, q, worst, margins, n_att, cur_mean, tilt_max = [], kin.q0.copy(), 0.0, [], [], [], 0.0
    for t in ts:
        offs = _foot_offsets(kin, group_of, groups, period, t, stride, twist, drag, p0)
        targets = [(pts[f][0], pts[f][1], o + p0[f] + offs[f][0]) for f in sp.LEGS]
        q, err = kin.ik(targets, q_init=q, reg=IK_REG)
        worst = max(worst, err)
        Q.append(q.copy())
        att = [f for f in sp.LEGS if offs[f][1]]
        n_att.append(len(att))
        c = st.check(kin, att, q=q, surface=surface, module=module, tail=tail)
        margins.append(c["margin"])
        tau = st.joint_torques(kin, c["forces"], q)
        moving = [j for j in kin.names if j.split("_")[0] in [f for f in sp.LEGS if not offs[f][1]]]
        cur_mean.append(sum(st.servo_currents(tau, moving).values()))
        kin.set(q)                                                        # the pad's tilt: the ankle must absorb it
        for f in att:
            zt = kin.data.xmat[kin.body(f"{f}_tibia")].reshape(3, 3) @ z0[f]
            tilt_max = max(tilt_max, math.degrees(math.acos(max(-1.0, min(1.0, float(zt[2]))))))
    moving = mk._moving(kin, Q)
    cols = [kin.index[j] for j in moving]
    Ts = period * (1 - 1 / n)                                             # each foot's stance time
    if twist[2]:
        speed = {"yaw_rate_deg_s": round(math.degrees(stride) / Ts, 3)}
        base = {"velocity_m_s": [0.0, 0.0, round(stride / Ts, 5)], "note": "velocity_m_s = [vx, vy, yaw rate]"}
    else:
        v = stride * 1e-3 / Ts
        speed = {"speed_mm_s": round(v * 1000, 2)}
        base = {"velocity_m_s": [round(v * twist[0], 5), round(v * twist[1], 5), 0.0]}
    I = float(np.mean(cur_mean))
    P = I * sp.SERVO["volt_run"] + sp.ELECTRONICS["power_w"]
    wh = sp.BATTERY["capacity_mah"] / 1000 * sp.BATTERY["volt_nominal"] * sp.BATTERY["usable_fraction"]
    m = mk.Motion(name, "gait", moving, ts, np.asarray(Q)[:, cols], period=period, base=base,
                  source=f"arachne_gaits.wall_gait('{name}')",
                  params=dict(speed, groups=[list(g) for g in groups], swing_s=swing_t,
                              stride=stride if twist[2] else stride * 1e-3, drag_mm=drag, surface=surface,
                              module=module, tail=tail, servo_current_mean_a=round(I, 3),
                              power_w=round(P, 2), runtime_min=round(wh / P * 60, 1),
                              attached_min=min(n_att), step_phases=dict(STEP)),
                  check={"ik_error_max_m": round(worst, 6), "limit_margin_min_rad": round(kin.margin(Q, moving), 4),
                         "attached_min": min(n_att), "adhesion_margin_min": round(min(margins), 3),
                         "ankle_tilt_max_deg": round(tilt_max, 2)})
    m.attached = n_att
    m.margins = margins
    return m


def motions(robot=None):
    """(kin, [the gaits on their own surfaces]): climb, climb_fast, climb_careful, turn on glass / varnished wood,
    and crab on bark with the spine feet."""
    import arachne_robot as ar
    kin = mk.Kin(robot or ar.arachne())
    return kin, [wall_gait(kin, g) for g in GAITS]


def parts(machine) -> dict:
    """``data:motions`` for the export (the body's centre above the wall as the base height)."""
    kin, ms = motions()
    kin.set(kin.q0)
    return mk.block(ms, float(kin.data.xpos[kin.body("body")][2]))
