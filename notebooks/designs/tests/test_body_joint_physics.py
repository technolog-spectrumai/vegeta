"""Body-joint physics of Cleopatra's two Amendment D treatments (protocol docs/myropod_stability.md §12.1, §12.5).

* **Model equivalence** — 'spring' and 'spring_damper' compile to MuJoCo models that are identical, field by field
  over every ``mujoco.MjModel`` attribute (body masses, inertias, ipos, geoms, joints' ranges / stiffness / springref
  / axes, actuators, options, statistics, visual settings, ...), except ``dof_damping`` on the body-joint DOFs (c vs
  0) and the model's own name (it carries the treatment).
* **c = 0 equivalence** — 'spring_damper' with every ``body_c_*`` = 0 gives a bitwise-identical episode log to
  'spring' (same seed, terrain and controller).
* **Restoring torque** — on a rig where every other DOF is held (the robot welded at its standing pose, only the
  tested hinge free, gravity off), a known external torque on the child segment deflects the hinge by τ/k (within
  2 %), and MuJoCo's passive force on the hinge is ``−k (q − q0)`` (plus ``−c q̇``).
* **Damping dissipation** — a free decay of one hinge on the same rig: the energy lost equals ``∫ c q̇² dt`` (within
  3 %) and the logarithmic decrement gives c (within 5 %).
* **Numerical dissipation** — with c = 0 the energy drift per oscillation cycle at the lab's time step is measured
  and reported (not asserted to be zero; a loose sanity bound only), and its time-step dependence is printed.

Units: SI (m, kg, s, N·m, rad, J). The measured numbers are printed (``pytest -s``) and stored as test properties
(``--junitxml``). Run:
cd /home/user/vegeta/notebooks/designs && python3 -m pytest -q -s tests/test_body_joint_physics.py
"""
import math

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

import myropod_controller as mc  # noqa: E402
import myropod_robot as mr  # noqa: E402
from vegeta import chiron as ch  # noqa: E402

H = mr.LAB_OPTIONS["timestep"]                       # the lab's physics step [s] (0.25 ms)
AXES = ("yaw", "pitch", "roll")
JOINTS = [f"body {i}-{i + 1} {a}" for i in (1, 2) for a in AXES]
ZERO_C = {"body_c_pitch": 0.0, "body_c_yaw": 0.0, "body_c_roll": 0.0}
#: per-axis values that differ, so a mixed-up axis shows (k [N·m/rad], c [N·m·s/rad])
K = {"yaw": 6.0, "pitch": 8.0, "roll": 10.0}
C = {"yaw": 0.15, "pitch": 0.2, "roll": 0.25}


def _report(record_property, label, **values):
    """Print the measured numbers (visible with -s) and keep them as test properties."""
    print(f"\n[{label}] " + ", ".join(f"{k} = {v:.6g}" if isinstance(v, float) else f"{k} = {v}"
                                    for k, v in values.items()))
    for k, v in values.items():
        record_property(k, v)


def compiled(robot, **lab_kw):
    """The robot's complete MJCF (``LAB_OPTIONS``, flat ground) compiled afresh — independent of anything a lab
    writes into its own model while it runs."""
    lab = mr.cleopatra_lab(robot=robot, **lab_kw)
    return mujoco.MjModel.from_xml_string(lab.xml)


def held_rig(robot, joint, timestep=H, gravity=(0.0, 0.0, 0.0)):
    """A MuJoCo model of ``robot`` with only ``joint`` free: every other joint (the root's free joint, the legs,
    the other body hinges) is removed and its link welded where the standing pose (``nominal_qpos``; body hinges at
    q0) puts it; the root is welded to the world 1 m up (no contact). The tested hinge keeps every property of the
    full model and sits at its reference (q = 0 is the straight chain). Returns (rig, full model, joint id in full).
    """
    opts = dict(mr.LAB_OPTIONS, timestep=timestep, control_dt=timestep, log_dt=timestep, gravity=gravity)
    lab = mr.cleopatra_lab(robot=robot, **opts)
    full = mujoco.MjModel.from_xml_string(lab.xml)
    jt = mujoco.mjtObj.mjOBJ_JOINT
    jid = mujoco.mj_name2id(full, jt, joint)
    assert jid >= 0, joint
    d = mujoco.MjData(full)
    for i, name in enumerate(lab.joint_names):
        j = mujoco.mj_name2id(full, jt, name)
        q = 0.0 if name == joint else float(lab.nominal_q[i])
        if full.jnt_bodyid[j] == full.jnt_bodyid[jid]:
            assert name == joint or q == 0.0, "the other hinges of the tested link must stand at 0"
        d.qpos[full.jnt_qposadr[j]] = q
    free = [j for j in range(full.njnt) if full.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    assert len(free) == 1
    d.qpos[full.jnt_qposadr[free[0]]:full.jnt_qposadr[free[0]] + 7] = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    mujoco.mj_kinematics(full, d)
    spec = mujoco.MjSpec.from_string(lab.xml)
    for a in list(spec.actuators):
        spec.delete(a)
    for b in spec.bodies:                                    # bake the posed frames into the bodies
        if b.name == "world":
            continue
        i = mujoco.mj_name2id(full, mujoco.mjtObj.mjOBJ_BODY, b.name)
        pa = full.body_parentid[i]
        b.pos = d.xmat[pa].reshape(3, 3).T @ (d.xpos[i] - d.xpos[pa])
        inv, rel = np.zeros(4), np.zeros(4)
        mujoco.mju_negQuat(inv, d.xquat[pa])
        mujoco.mju_mulQuat(rel, inv, d.xquat[i])
        b.quat = rel
    for j in list(spec.joints):
        if j.name != joint:
            spec.delete(j)
    rig = spec.compile()
    assert rig.nv == 1 and rig.njnt == 1
    # the tested hinge is the robot's hinge, unchanged
    fd, fq = full.jnt_dofadr[jid], full.jnt_qposadr[jid]
    assert rig.jnt_stiffness[0] == full.jnt_stiffness[jid] and rig.qpos_spring[0] == full.qpos_spring[fq]
    assert rig.dof_damping[0] == full.dof_damping[fd] and rig.dof_armature[0] == full.dof_armature[fd] == 0.0
    assert rig.dof_frictionloss[0] == full.dof_frictionloss[fd] == 0.0
    np.testing.assert_array_equal(rig.jnt_range[0], full.jnt_range[jid])
    np.testing.assert_array_equal(rig.jnt_axis[0], full.jnt_axis[jid])
    np.testing.assert_array_equal(rig.jnt_pos[0], full.jnt_pos[jid])
    child = rig.jnt_bodyid[0]
    assert rig.body_subtreemass[child] == pytest.approx(full.body_subtreemass[full.jnt_bodyid[jid]], rel=1e-12)
    return rig, full, jid


def inertia(m, d) -> float:
    """The rig's one-DOF inertia about the hinge [kg·m²]."""
    M = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, M)
    return float(M[0, 0])


def free_decay(rig, amplitude=0.1, duration=None, cycles=20):
    """Release the rig's hinge from q0 + ``amplitude`` [rad] at rest and integrate. Returns a dict: inertia M
    [kg·m²], k, c, the samples t [s], x = q − q0 [rad], v = q̇ [rad/s], energy E = ½ M v² + ½ k x² [J], the
    dissipation integral D(t) = ∫ c q̇² dt [J] (trapezoid), and the extremes of x (time [s], |x| [rad]; parabolic
    refinement). ``duration`` [s] default: ``cycles`` periods without damping, else until e^-12 of the amplitude."""
    d = mujoco.MjData(rig)
    q0, k, c = float(rig.qpos_spring[0]), float(rig.jnt_stiffness[0]), float(rig.dof_damping[0])
    d.qpos[0] = q0 + amplitude
    mujoco.mj_forward(rig, d)
    M = inertia(rig, d)
    h = rig.opt.timestep
    if duration is None:
        duration = cycles * 2 * math.pi / math.sqrt(k / M) if c == 0 else 12.0 / (c / (2 * M))
    n = int(round(duration / h))
    q, v = np.empty(n + 1), np.empty(n + 1)
    q[0], v[0] = d.qpos[0], d.qvel[0]
    lo, hi = rig.jnt_range[0]
    for i in range(n):
        mujoco.mj_step(rig, d)
        q[i + 1], v[i + 1] = d.qpos[0], d.qvel[0]
    assert lo < q.min() and q.max() < hi, "the decay must stay off the hard stops"
    assert d.ncon == 0 and d.nefc == 0, "no contact and no active constraint on the rig"
    t = np.arange(n + 1) * h
    x = q - q0
    E = 0.5 * M * v ** 2 + 0.5 * k * x ** 2
    D = c * np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] ** 2 + v[:-1] ** 2) * h)])
    i = np.nonzero(((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:])) | ((x[1:-1] < x[:-2]) & (x[1:-1] <= x[2:])))[0] + 1
    ext_t, ext_x = [0.0], [abs(x[0])]
    for j in i:
        y0, y1, y2 = x[j - 1], x[j], x[j + 1]
        off = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
        ext_t.append(t[j] + off * h)
        ext_x.append(abs(y1 - 0.25 * (y0 - y2) * off))
    return dict(M=M, k=k, c=c, h=h, t=t, x=x, v=v, E=E, D=D, ext_t=np.array(ext_t), ext_x=np.array(ext_x),
                amplitude=amplitude)


def decrement(r):
    """Logarithmic decrement from the extremes of a free decay: the damped period T_d [s], δ = ln(x_n / x_{n+1})
    per period (a straight-line fit of ln|x| at the extremes, which are T_d/2 apart), the damping ratio ζ, and the
    damping and stiffness it implies with the rig's inertia M: c = 2 M δ / T_d, k = M ((2π/T_d)² + (δ/T_d)²)."""
    keep = r["ext_x"] > 1e-6 * r["amplitude"]
    t, x = r["ext_t"][keep], r["ext_x"][keep]
    Td = 2.0 * float(np.mean(np.diff(t)))
    sigma = -float(np.polyfit(t, np.log(x), 1)[0])          # decay rate ζω_n [1/s]
    delta = sigma * Td
    return dict(Td=Td, delta=delta, zeta=delta / math.sqrt(4 * math.pi ** 2 + delta ** 2), n_extremes=int(len(t)),
                c=2 * r["M"] * sigma, k=r["M"] * ((2 * math.pi / Td) ** 2 + sigma ** 2))


def energy_drift(r):
    """Undamped decay: relative energy drift per cycle (slope of a line through E at every positive peak of x,
    divided by the initial energy) and the relative within-cycle fluctuation (max E − min E) / E0."""
    x, E = r["x"], r["E"]
    i = np.nonzero((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]))[0] + 1
    Ep = np.concatenate([[E[0]], E[i]])
    slope = float(np.polyfit(np.arange(len(Ep)), Ep, 1)[0])
    return dict(per_cycle=slope / E[0], fluctuation=float((E.max() - E.min()) / E[0]), cycles=int(len(Ep) - 1))


# ----------------------------------------------------------------------------------------------- model equivalence
def _struct_diff(x, y, path, out):
    """Compare two MuJoCo structs (MjOption, MjStatistic, MjVisual and its parts) field by field."""
    for f in sorted(dir(x)):
        if f.startswith("_"):
            continue
        a, b = getattr(x, f), getattr(y, f)
        if callable(a):
            continue
        if isinstance(a, np.ndarray):
            if not np.array_equal(a, b):
                out.append(f"{path}.{f}")
        elif isinstance(a, (int, float, bool, str)):
            if a != b:
                out.append(f"{path}.{f}")
        else:
            _struct_diff(a, b, f"{path}.{f}", out)


def _model_diff(a, b):
    """Every MjModel attribute compared: (names checked, names that differ). Methods and per-object accessors
    (``m.body(i)``, ``m.jnt(...)``, ...) are views of the same arrays and are skipped."""
    checked, differ = [], []
    for name in sorted(dir(a)):
        if name.startswith("_"):
            continue
        va = getattr(a, name)
        if callable(va):
            continue
        vb = getattr(b, name)
        if isinstance(va, np.ndarray):
            same = va.shape == vb.shape and va.dtype == vb.dtype and \
                np.array_equal(va, vb, equal_nan=va.dtype.kind in "fc")
            if not same:
                differ.append(name)
        elif isinstance(va, (int, float, bool, str, bytes)):
            if va != vb:
                differ.append(name)
        elif type(va).__module__.startswith("mujoco"):
            _struct_diff(va, vb, name, differ)
        else:
            raise AssertionError(f"unchecked MjModel attribute {name!r} of type {type(va).__name__}")
        checked.append(name)
    return checked, differ


@pytest.mark.parametrize("body_kw", [{}, {"body_roll_axis": False, "body_k_yaw": 4.0, "body_q0_pitch": 0.05,
                                          "body_limit_yaw": 0.6, "body_c_pitch": 0.3}], ids=["defaults", "custom"])
def test_treatment_models_differ_only_in_body_joint_damping(body_kw, record_property):
    """§12.3: the two treatments' compiled models are identical field by field except the body DOFs' damping."""
    a = compiled(mr.cleopatra("spring", **body_kw))
    b = compiled(mr.cleopatra("spring_damper", **body_kw))
    checked, differ = _model_diff(a, b)
    for must in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat", "geom_size",
                 "geom_pos", "geom_friction", "geom_contype", "jnt_range", "jnt_stiffness", "qpos_spring", "jnt_axis",
                 "jnt_pos", "dof_armature", "dof_frictionloss", "actuator_biasprm", "actuator_gainprm", "opt", "stat",
                 "vis"):
        assert must in checked
    name_fields = {"names", "nnames"} | {n for n in checked if n.startswith("name_") and n.endswith("adr")}
    assert set(differ) - name_fields == {"dof_damping"}, differ
    # the names: only the model's own name differs ("cleopatra spring" / "cleopatra spring_damper")
    na, nb = a.names.split(b"\0"), b.names.split(b"\0")
    assert (na[0], nb[0]) == (b"cleopatra spring", b"cleopatra spring_damper") and na[1:] == nb[1:]
    shift = len(nb[0]) - len(na[0])
    assert b.nnames - a.nnames == shift
    for f in name_fields - {"names", "nnames"}:
        np.testing.assert_array_equal(getattr(b, f) - getattr(a, f), shift, err_msg=f)
    # the damping: c on the body hinges (0 for 'spring'), the same 0 everywhere else
    jt = mujoco.mjtObj.mjOBJ_JOINT
    body = {mujoco.mj_id2name(a, jt, j): int(a.jnt_dofadr[j]) for j in range(a.njnt)
            if (mujoco.mj_id2name(a, jt, j) or "").startswith("body ")}
    robot = mr.cleopatra("spring_damper", **body_kw)
    assert list(body) == mr.body_joints(robot.connection)
    others = np.setdiff1d(np.arange(a.nv), list(body.values()))
    np.testing.assert_array_equal(a.dof_damping[others], b.dof_damping[others])
    assert not a.dof_damping.any()                                         # spring: no viscous damping anywhere
    for name, dof in body.items():
        assert a.dof_damping[dof] == 0.0
        assert b.dof_damping[dof] == robot.connection.axis(name.rsplit(" ", 1)[1]).damping > 0.0
    _report(record_property, f"model equivalence {body_kw or 'defaults'}", fields_checked=len(checked),
            differing=",".join(sorted(differ)))


@pytest.mark.parametrize("controller", ["fixed", "adaptive"])
def test_spring_damper_with_zero_c_reproduces_spring_bitwise(controller):
    """§12.5: the spring–damper model with c = 0 reproduces the spring-only trajectory bitwise (same seed, terrain
    — random rough ground, h/L = 0.15 — and controller)."""
    spec = {"kind": "rough", "rms": 0.15 * 0.18, "correlation_length": 0.25 * 0.17, "start": 0.3, "seed": 21}
    make = mc.fixed if controller == "fixed" else mc.adaptive
    eps = []
    for t, kw in (("spring", {}), ("spring_damper", ZERO_C)):
        lab = mr.cleopatra_lab(t, ch.terrain_from_spec(spec), robot_kw=kw, course_extent=(-0.6, 1.2, -0.4, 0.4))
        eps.append(lab.run(make(0.2), duration=2.0, seed=21, info={"treatment": "c=0 check"}))
    a, b = eps[0].log, eps[1].log
    assert set(a) == set(b)
    differ = []
    for key in a:
        x, y = a[key], b[key]
        if isinstance(x, np.ndarray):
            same = x.shape == y.shape and x.dtype == y.dtype and np.array_equal(x, y, equal_nan=x.dtype.kind in "fc")
        elif isinstance(x, float) and isinstance(y, float) and math.isnan(x):
            same = math.isnan(y)                              # e.g. v_target / course_m of a run without rules
        else:
            same = x == y
        if not same:
            differ.append(key)
    assert differ == ["robot"], differ                          # only the label: 'cleopatra spring' / '..._damper'
    assert eps[0].outcome == eps[1].outcome
    assert np.abs(a["q"][:, [i for i, j in enumerate(a["joints"]) if j.startswith("body ")]]).max() > 1e-3


# ----------------------------------------------------------------------------------------------- restoring torque
@pytest.mark.parametrize("joint", JOINTS)
@pytest.mark.parametrize("treatment", mr.TREATMENTS)
def test_static_deflection_under_a_known_torque_is_tau_over_k(treatment, joint, record_property):
    """§12.5: a known external torque τ on the child segment (everything else held, gravity off) deflects the hinge
    to q0 + τ/k (within 2 %); MuJoCo's passive force on it is −k (q − q0) − c q̇ exactly."""
    axis = joint.rsplit(" ", 1)[1]
    q0, tau = 0.05, 0.8                                      # [rad] on the tested axis; [N·m]
    robot = mr.cleopatra(treatment, **{f"body_k_{a}": v for a, v in K.items()},
                         **{f"body_c_{a}": v for a, v in C.items()}, **{f"body_q0_{axis}": q0})
    rig, full, jid = held_rig(robot, joint)
    k, c = K[axis], (C[axis] if treatment == "spring_damper" else 0.0)
    assert rig.jnt_stiffness[0] == k and rig.qpos_spring[0] == q0 and rig.dof_damping[0] == c
    d = mujoco.MjData(rig)
    d.qpos[0] = q0                                           # spring unloaded: at rest
    mujoco.mj_forward(rig, d)
    M = inertia(rig, d)
    child = rig.jnt_bodyid[0]
    world_axis = d.xaxis[0].copy()                           # a hinge's axis is fixed while it turns about it
    ramp, hold = 1.0, (4.0 if c == 0 else 2.0 + 12.0 / (c / (2 * M)))   # [s]: cosine ramp, then constant τ
    n_ramp, n = int(round(ramp / H)), int(round((ramp + hold) / H))
    q = np.empty(n)
    for i in range(n):
        s = min(1.0, i / n_ramp)
        d.xfrc_applied[child, 3:] = tau * 0.5 * (1.0 - math.cos(math.pi * s)) * world_axis   # torque [N·m], world
        mujoco.mj_step(rig, d)
        q[i] = d.qpos[0]
    tail = q[-int(round(2.0 / H)):]                          # the last 2 s (> 2 periods): equilibrium = mid-range
    deflection = 0.5 * (tail.max() + tail.min()) - q0
    assert deflection == pytest.approx(tau / k, rel=0.02)
    # passive force [N·m] at the final state and at arbitrary states: −k (q − q0) and −c q̇
    states = [(float(d.qpos[0]), float(d.qvel[0])), (q0 + 0.2, 0.0), (q0 - 0.15, 0.7), (q0, -1.3)]
    for qq, vv in states:
        d.qpos[0], d.qvel[0] = qq, vv
        d.xfrc_applied[:] = 0.0
        mujoco.mj_forward(rig, d)
        assert d.qfrc_spring[0] == pytest.approx(-k * (qq - q0), rel=1e-12, abs=1e-15)
        assert d.qfrc_damper[0] == pytest.approx(-c * vv, rel=1e-12, abs=1e-15)
        assert d.qfrc_passive[0] == pytest.approx(-k * (qq - q0) - c * vv, rel=1e-12, abs=1e-15)
    # and on the full, free robot (every DOF free) the body hinge's passive force is the same law
    fd = mujoco.MjData(full)
    mujoco.mj_resetData(full, fd)
    fd.qpos[full.jnt_qposadr[jid]] = q0 + 0.1
    fd.qvel[full.jnt_dofadr[jid]] = 0.4
    mujoco.mj_forward(full, fd)
    assert fd.qfrc_spring[full.jnt_dofadr[jid]] == pytest.approx(-k * 0.1, rel=1e-12)
    assert fd.qfrc_damper[full.jnt_dofadr[jid]] == pytest.approx(-c * 0.4, rel=1e-12, abs=1e-15)
    _report(record_property, f"restoring torque {treatment} {joint}", tau_Nm=tau, k=k, q0=q0,
            deflection_rad=deflection, tau_over_k_rad=tau / k, ratio=deflection / (tau / k),
            residual_oscillation_rad=0.5 * float(tail.max() - tail.min()), inertia_kgm2=M)


# ----------------------------------------------------------------------------------------------- damping dissipation
@pytest.mark.parametrize("joint", JOINTS)
def test_free_decay_energy_loss_and_log_decrement_give_c(joint, record_property):
    """§12.5: a free decay of one hinge (other DOFs held, gravity off): the energy lost equals ∫ c q̇² dt (within
    3 %) and the logarithmic decrement gives c (within 5 %), at the lab's time step."""
    axis = joint.rsplit(" ", 1)[1]
    robot = mr.cleopatra("spring_damper", **{f"body_k_{a}": v for a, v in K.items()},
                         **{f"body_c_{a}": v for a, v in C.items()})
    rig, _, _ = held_rig(robot, joint)
    r = free_decay(rig, amplitude=0.1)
    assert r["c"] == C[axis] and r["k"] == K[axis] and r["h"] == H
    E0, E1 = float(r["E"][0]), float(r["E"][-1])
    assert E0 == pytest.approx(0.5 * K[axis] * 0.1 ** 2, rel=1e-12)
    assert E1 < 1e-6 * E0                                    # decayed (e^-12 of the amplitude)
    lost, dissipated = E0 - E1, float(r["D"][-1])
    assert dissipated == pytest.approx(lost, rel=0.03)
    # the balance holds along the way too (every 0.1 s), not only at the end
    step = int(round(0.1 / H))
    np.testing.assert_allclose(r["D"][::step], E0 - r["E"][::step], rtol=0, atol=0.03 * E0)
    dec = decrement(r)
    assert dec["n_extremes"] >= 6
    assert dec["c"] == pytest.approx(C[axis], rel=0.05)
    assert dec["k"] == pytest.approx(K[axis], rel=0.05)
    zeta = C[axis] / (2 * math.sqrt(K[axis] * r["M"]))
    _report(record_property, f"decay spring_damper {joint}", inertia_kgm2=r["M"], k=K[axis], c=C[axis], zeta=zeta,
            energy_lost_J=lost, integral_c_qd2_J=dissipated, balance_ratio=dissipated / lost, Td_s=dec["Td"],
            log_decrement=dec["delta"], c_from_decrement=dec["c"], c_ratio=dec["c"] / C[axis],
            k_from_period=dec["k"], extremes=dec["n_extremes"])


# ----------------------------------------------------------------------------------------------- numerical dissipation
@pytest.mark.parametrize("joint", JOINTS)
def test_spring_only_numerical_energy_drift_is_reported(joint, record_property):
    """§12.2: with c = 0 the hinge is conservative; whatever energy the integrator loses or gains per cycle at the
    lab's time step is measured and reported. Only a loose sanity bound is asserted (|drift| < 1e-4 per cycle), not
    zero."""
    axis = joint.rsplit(" ", 1)[1]
    robot = mr.cleopatra("spring", **{f"body_k_{a}": v for a, v in K.items()})
    rig, _, _ = held_rig(robot, joint)
    assert rig.dof_damping[0] == 0.0 and rig.opt.timestep == H
    r = free_decay(rig, amplitude=0.1, cycles=20)
    dr = energy_drift(r)
    assert dr["cycles"] >= 19
    assert abs(dr["per_cycle"]) < 1e-4
    assert dr["fluctuation"] < 0.02
    _report(record_property, f"numerical drift spring {joint} at {H * 1e3:g} ms", inertia_kgm2=r["M"], k=K[axis],
            omega_rad_s=math.sqrt(K[axis] / r["M"]), drift_per_cycle=dr["per_cycle"],
            within_cycle_fluctuation=dr["fluctuation"], cycles=dr["cycles"],
            net_change=float((r["E"][-1] - r["E"][0]) / r["E"][0]))


def test_numerical_dissipation_versus_time_step(record_property):
    """Report (no protocol tolerance): the c = 0 energy drift per cycle and the c identified from the decrement of
    the fastest hinge (body 2-3 roll, the lightest child) at 0.125, 0.25, 0.5, 1 and 2 ms."""
    joint = "body 2-3 roll"
    rows = []
    for h in (0.000125, 0.00025, 0.0005, 0.001, 0.002):
        rig, _, _ = held_rig(mr.cleopatra("spring"), joint, timestep=h)
        dr = energy_drift(free_decay(rig, amplitude=0.1, cycles=20))
        rig, _, _ = held_rig(mr.cleopatra("spring_damper"), joint, timestep=h)
        r = free_decay(rig, amplitude=0.1)
        dec = decrement(r)
        balance = float(r["D"][-1] / (r["E"][0] - r["E"][-1]))
        rows.append((h, dr["per_cycle"], dr["fluctuation"], dec["c"] / r["c"], balance))
        _report(record_property, f"{joint} at {h * 1e3:g} ms", spring_drift_per_cycle=dr["per_cycle"],
                spring_fluctuation=dr["fluctuation"], spring_damper_c_ratio=dec["c"] / r["c"],
                spring_damper_balance=rows[-1][4])
    assert all(abs(row[1]) < 1e-3 and abs(row[3] - 1.0) < 0.05 for row in rows)
