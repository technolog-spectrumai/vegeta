"""Gait simulation shared by the product notebooks (robot dog, Myropods): kinematics only, no dynamics.

A walking machine is a set of *bodies* (one for the dog, one per segment for a Myropod), each with four
hips. Every body rides at its standing height over the ground under its own hips and pitches and rolls
with the ground between them; a stance foot stays where it landed while the body passes over it, a swing
foot flies to a landing point ahead of its hip on the ground. The inverse kinematics turns each foot
position into joint angles — hip roll + hip pitch + knee for a dog leg (the leg plane hangs down from a
hip axis along y), hip yaw + hip pitch + knee for a Myropod leg (the leg reaches sideways from a hip axis
along x) — and the joint torques follow from the ground force on the foot. Every input is explicit.
"""
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ---- rotations (right-hand rule; the same convention as pyvista's rotate_x / rotate_y / rotate_z) ----
def Rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# ---- terrain ----
def make_terrain(seed=3, n_rocks=14, x_range=(400.0, 3800.0), y_range=(-320.0, 320.0), rock_height=(30.0, 70.0), rock_width=(80.0, 140.0),
                 waves=((45.0, 2600.0, 0.0), (20.0, 900.0, 0.6))):
    """Ground height z(x, y) [mm]: long sine waves (amplitude, wavelength, y-skew) and seeded gaussian rocks."""
    rng = np.random.default_rng(seed)
    rocks = [(rng.uniform(*x_range), rng.uniform(*y_range), rng.uniform(*rock_height), rng.uniform(*rock_width)) for _ in range(n_rocks)]

    def terrain(x, y):
        z = 0.0
        for amp, lam, skew in waves:
            z = z + amp * np.sin(2 * np.pi * (x + skew * y) / lam)
        for rx, ry, rh, rw in rocks:
            z = z + rh * np.exp(-((x - rx) ** 2 + (y - ry) ** 2) / rw**2)
        return z

    terrain.rocks = rocks
    return terrain


def flat(x, y):
    return 0.0 * np.asarray(x)


# ---- bodies ----
LEGS = {"front-left": (1, 1), "front-right": (1, -1), "rear-left": (-1, 1), "rear-right": (-1, -1)}


def body_pose(x_body, hip_x, hip_y, h_stand, terrain, y_body=0.0):
    """One body over the ground: centre, rotation matrix (pitch nose-up with the front/rear slope, roll with the
    left/right slope, both from the ground under the hips), pitch, roll, and the four hips in world coordinates."""
    zf, zr = float(terrain(x_body + hip_x, y_body)), float(terrain(x_body - hip_x, y_body))
    zl, zrt = float(terrain(x_body, y_body + hip_y)), float(terrain(x_body, y_body - hip_y))
    th = math.atan2(zf - zr, 2 * hip_x)
    ph = math.atan2(zl - zrt, 2 * hip_y)
    R = Ry(-th) @ Rx(ph)                                   # body axes in the world (pyvista: rotate_x(roll) then rotate_y(-pitch))
    centre = np.array([x_body, y_body, (zf + zr) / 2 + h_stand])
    hips = {leg: centre + R @ np.array([sx * hip_x, sy * hip_y, 0.0]) for leg, (sx, sy) in LEGS.items()}
    return centre, R, th, ph, hips


# ---- inverse kinematics ----
def ik_two_link_planar(x_foot, z_foot, L1, L2):
    """Dog leg in its plane: upper-leg angle behind vertical and lower-leg angle ahead of vertical (knee back) [deg]
    for a foot at (x ahead, z below the hip); None when out of reach."""
    r = math.hypot(x_foot, z_foot)
    if r > L1 + L2 - 1e-6 or r < abs(L1 - L2) + 1e-6:
        return None
    knee_int = math.acos(max(-1, min(1, (L1**2 + L2**2 - r**2) / (2 * L1 * L2))))
    phi = math.atan2(x_foot, -z_foot)
    alpha = math.acos(max(-1, min(1, (L1**2 + r**2 - L2**2) / (2 * L1 * r))))
    a1 = -(phi - alpha)
    a2 = math.pi - knee_int - a1
    return math.degrees(a1), math.degrees(a2)


def ik_dog(rel, L1, L2):
    """Hip roll, hip pitch, knee [deg] for a foot at ``rel`` (body frame) under a dog hip; (angles, reachable)."""
    roll = math.atan2(rel[1], -rel[2]) if rel[2] < 0 else math.atan2(rel[1], 1e-6)
    xp, zp = rel[0], -math.hypot(rel[1], rel[2])
    ik = ik_two_link_planar(xp, zp, L1, L2)
    ok = ik is not None
    if not ok:
        k = (L1 + L2 - 2) / math.hypot(xp, zp)
        ik = ik_two_link_planar(xp * k, zp * k, L1, L2)
    return (math.degrees(roll), ik[0], ik[1]), ok


def ik_myropod(rel, femur, tibia, sy):
    """Hip yaw (sweep forward +), hip pitch (femur below horizontal), knee (tibia below horizontal) [deg] for a
    Myropod leg on side ``sy`` (+1 left) with a foot at ``rel`` (body frame); (angles, reachable)."""
    out_y = sy * rel[1]                                    # outward reach
    yaw = math.degrees(math.atan2(rel[0], max(out_y, 1e-6)))
    r, d = math.hypot(rel[0], out_y), -rel[2]              # horizontal reach in the leg plane, drop below the hip
    rho = math.hypot(r, d)
    ok = abs(femur - tibia) + 1e-6 < rho < femur + tibia - 1e-6
    if not ok:
        k = (femur + tibia - 2) / rho if rho >= femur + tibia - 1e-6 else (abs(femur - tibia) + 2) / rho
        r, d, rho = r * k, d * k, rho * k
    gamma = math.atan2(d, r)
    alpha = math.acos(max(-1, min(1, (femur**2 + rho**2 - tibia**2) / (2 * femur * rho))))
    hip = gamma - alpha
    knee = math.atan2(d - femur * math.sin(hip), r - femur * math.cos(hip))
    return (yaw, math.degrees(hip), math.degrees(knee)), ok


# ---- joint torques for a ground force on the foot ----
def torques_dog(angles, rel, F_body, L1, L2):
    """[N m] at hip roll, hip pitch, knee (lengths in mm, force in N)."""
    roll, a1, a2 = angles
    tau_roll = float(np.cross(rel, F_body)[0]) / 1000
    Fp = Rx(-math.radians(roll)) @ F_body
    xp, zp = rel[0], -math.hypot(rel[1], rel[2])
    kx, kz = -L1 * math.sin(math.radians(a1)), -L1 * math.cos(math.radians(a1))
    return tau_roll, (xp * Fp[2] - zp * Fp[0]) / 1000, ((xp - kx) * Fp[2] - (zp - kz) * Fp[0]) / 1000


def torques_myropod(angles, rel, F_body, femur, tibia, sy):
    """[N m] at hip yaw, hip pitch, knee for a Myropod leg."""
    yaw, hip, knee = angles
    tau_yaw = float(np.cross(rel, F_body)[2]) / 1000
    # into the leg plane: outward o, down d; the force's components along them
    o_hat = np.array([math.sin(math.radians(yaw)), sy * math.cos(math.radians(yaw)), 0.0])
    Fo, Fd = float(F_body @ o_hat), -float(F_body[2])
    r, d = float(rel @ o_hat), -rel[2]
    kr, kd = femur * math.cos(math.radians(hip)), femur * math.sin(math.radians(hip))
    tau_hip = (r * Fd - d * Fo) / 1000
    tau_knee = ((r - kr) * Fd - (d - kd) * Fo) / 1000
    return tau_yaw, tau_hip, tau_knee


# ---- the stepping machine ----
@dataclass
class StepResult:
    times: np.ndarray
    feet: dict                      # leg -> (T, 3) world positions
    stance: dict                    # leg -> (T,) bool
    joints: dict                    # leg -> (T, 3) joint angles [deg]
    steps: pd.DataFrame             # one row per landing
    unreachable: int = 0
    extra: dict = field(default_factory=dict)


def simulate_steps(times, pose_fn, legs, phases, duty, stride_mm, period_s, terrain, ik_fn, *, swing_lift=60.0, torque_fn=None,
                   force_body_fn=None, joint_names=("j1", "j2", "j3"), peak_Nm=None, body_names=None, foot_out=0.0):
    """Step the feet of one or more bodies through ``times``.

    ``pose_fn(t)`` returns a list of (centre, R, pitch, roll, hips) — one per body. ``legs`` maps a leg name to
    (body index, sx, sy) and ``phases`` to its phase in the stride; ``ik_fn(rel, sx, sy)`` -> (angles, ok);
    ``torque_fn(angles, rel, F_body, sx, sy)`` -> torques; ``force_body_fn(R)`` -> the ground force on a landing
    foot in the body frame. ``foot_out`` puts each foot that far outward (sideways) from its hip — 0 for a dog
    (feet under the hips), the standing foot reach for a Myropod. A landing becomes one row of the step table."""
    dt = float(times[1] - times[0])
    state, feet, stance, joints, steps, unreachable = {}, {}, {}, {}, [], 0
    poses0 = pose_fn(float(times[0]))
    for leg, (b, sx, sy) in legs.items():
        hip0 = poses0[b][4][_key(sx, sy)]
        x0 = hip0[0] + (stride_mm * duty / 2 * (1 - 2 * phases[leg] / duty) if phases[leg] < duty else 0.0)
        y0 = hip0[1] + sy * foot_out
        state[leg] = {"foot": np.array([x0, y0, float(terrain(x0, y0))]), "lift": None, "land": None, "was_stance": True}
        feet[leg], stance[leg], joints[leg] = [], [], []
    for t_s in times:
        poses = pose_fn(float(t_s))
        for leg, (b, sx, sy) in legs.items():
            centre, R, th, ph, hips = poses[b]
            hip = hips[_key(sx, sy)]
            phase = (t_s / period_s + phases[leg]) % 1.0
            st = state[leg]
            if phase < duty:
                if st["lift"] is not None:
                    st["foot"] = st["land"].copy(); st["lift"] = None
                foot, in_stance = st["foot"], True
            else:
                u = (phase - duty) / (1 - duty)
                if st["lift"] is None:
                    st["lift"] = st["foot"].copy()
                    hips_land = pose_fn(float(t_s) + (1 - u) * (1 - duty) * period_s)[b][4]
                    xl, yl = hips_land[_key(sx, sy)][0] + stride_mm * duty / 2, hips_land[_key(sx, sy)][1] + sy * foot_out
                    st["land"] = np.array([xl, yl, float(terrain(xl, yl))])
                lift, land = st["lift"], st["land"]
                foot = lift + (land - lift) * u
                clear = max(0.0, float(terrain(foot[0], foot[1])) - foot[2] + 15.0)
                foot = foot + np.array([0.0, 0.0, (swing_lift + clear) * math.sin(math.pi * u)])
                in_stance = False
            rel = R.T @ (foot - hip)
            angles, ok = ik_fn(rel, sx, sy)
            unreachable += 0 if ok else 1
            if in_stance and not st["was_stance"] and torque_fn is not None and force_body_fn is not None:
                F_body = force_body_fn(R)
                taus = torque_fn(angles, rel, F_body, sx, sy)
                row = {"t [s]": float(t_s), "leg": leg, "x [mm]": foot[0], "y [mm]": foot[1], "ground z [mm]": foot[2],
                       "body pitch [deg]": math.degrees(th), "body roll [deg]": math.degrees(ph)}
                if body_names is not None:
                    row["body"] = body_names[b]
                row.update({f"{n} [deg]": a for n, a in zip(joint_names, angles)})
                row.update({f"τ {n} [Nm]": v for n, v in zip(joint_names, taus)})
                if peak_Nm:
                    row["worst joint / peak motor"] = max(abs(v) for v in taus) / peak_Nm
                row["reachable"] = ok
                steps.append(row)
            st["was_stance"] = in_stance
            feet[leg].append(foot.copy()); stance[leg].append(in_stance); joints[leg].append(angles)
    tab = pd.DataFrame(steps)
    if len(tab):
        tab.index = [f"step {i + 1}" for i in range(len(tab))]
    return StepResult(np.asarray(times), {k: np.array(v) for k, v in feet.items()}, {k: np.array(v) for k, v in stance.items()},
                      {k: np.array(v) for k, v in joints.items()}, tab, unreachable)


def _key(sx, sy):
    return {(1, 1): "front-left", (1, -1): "front-right", (-1, 1): "rear-left", (-1, -1): "rear-right"}[(sx, sy)]


def gait_diagram(ax, result, legs_order=None):
    names = legs_order or list(result.stance)
    for j, leg in enumerate(names):
        ax.fill_between(result.times, j - 0.4, j + 0.4, where=result.stance[leg], color="#3f4347", step="mid")
    ax.set(yticks=range(len(names)), yticklabels=names, xlabel="time [s]", title="gait diagram (dark = foot on the ground)")
    ax.grid(alpha=0.3, axis="x")


def fk_myropod(angles, femur, tibia, sy):
    """Knee and foot positions relative to the hip (body frame) for a Myropod leg — the inverse of ``ik_myropod``."""
    yaw, hip, knee = (math.radians(a) for a in angles)
    o_hat = np.array([math.sin(yaw), sy * math.cos(yaw), 0.0]); down = np.array([0.0, 0.0, -1.0])
    k = femur * (math.cos(hip) * o_hat + math.sin(hip) * down)
    return k, k + tibia * (math.cos(knee) * o_hat + math.sin(knee) * down)


def fk_dog(angles, L1, L2):
    """Knee and foot positions relative to the hip (body frame) for a dog leg — the inverse of ``ik_dog``."""
    roll, a1, a2 = (math.radians(a) for a in angles)
    k_p = np.array([-L1 * math.sin(a1), 0.0, -L1 * math.cos(a1)])
    f_p = k_p + np.array([L2 * math.sin(a2), 0.0, -L2 * math.cos(a2)])
    return Rx(roll) @ k_p, Rx(roll) @ f_p
