"""Dolphin launches an Emerillion, in MuJoCo (``dolphin.md``): the submarine surfaced and rolling gently in the swell
(a mocap hull driven by a sine), the hangar hatch opening, the lift raising the drone nose-up on the deck pad, the
latch letting go, the EDF's thrust taking the drone up through the hover, the pitch-over and the acceleration into
level flight. No OpenGL here: the frames are drawn from the model's geoms with matplotlib and ffmpeg makes the mp4.

    python dolphin_launch.py [out.mp4]      # ~25 s of simulation; a minute to render

The drone is a point-mass-with-attitude model of the X-wing Emerillion (3.1 kg, 44 N from the two wing-tip fans, a
0.7 x 0.16 m wing at Cl up to 1.2, so it needs ~19 m/s before the wing carries it): thrust along its axis, a PD on
its attitude (torques about the centre of mass), lift once it has airspeed. The geoms are the X-wing's: the slender
fuselage with its canopy, the short wing with a fan pod at each tip, the four tail fins it stands on.
It is the launch *sequence* — timing, clearances, the hull's motion under the drone — not the aerodynamics.
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile

import mujoco
import numpy as np

RHO_AIR, G = 1.2, 9.81
DRONE = dict(mass=3.11, thrust_max=44.0, wing_area=0.7 * 0.16, cl_max=1.2, span=0.7, length=1.05)
HULL = dict(length=3.6, d=0.6)

XML = f"""
<mujoco model="dolphin_launch">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <default>
    <default class="hullg"><geom contype="1" conaffinity="1"/></default>
    <default class="droneg"><geom contype="2" conaffinity="2"/></default>
  </default>
  <worldbody>
    <geom name="sea" type="plane" size="30 30 0.1" rgba="0.45 0.65 0.85 1" pos="0 0 0" contype="3" conaffinity="3"/>
    <body name="hull" mocap="true" pos="0 0 0.05" childclass="hullg">
      <geom name="hull_body" type="cylinder" size="0.3 1.08" euler="0 90 0" rgba="0.17 0.37 0.54 1" pos="0 0 0.3"/>
      <geom name="hull_nose" type="sphere" size="0.3" pos="1.08 0 0.3" rgba="0.17 0.37 0.54 1"/>
      <geom name="hull_tail" type="cylinder" size="0.14 0.5" euler="0 90 0" pos="-1.58 0 0.3" rgba="0.17 0.37 0.54 1"/>
      <geom name="sail" type="box" size="0.25 0.09 0.15" pos="0.45 0 0.75" rgba="0.2 0.27 0.33 1"/>
      <geom name="mast" type="cylinder" size="0.02 0.2" pos="0.6 0 1.1" rgba="0.2 0.27 0.33 1"/>
      <geom name="bay" type="box" size="0.62 0.21 0.02" pos="-0.72 0 0.58" rgba="0.5 0.55 0.6 1"/>
      <body name="hatch" pos="-0.72 0.22 0.61">
        <joint name="hatch" type="hinge" axis="1 0 0" range="0 2.2" damping="2"/>
        <geom name="hatch_g" type="box" size="0.62 0.22 0.01" pos="0 -0.22 0" rgba="0.5 0.55 0.6 1"/>
      </body>
      <body name="lift" pos="-0.72 0 0.3">
        <joint name="lift" type="slide" axis="0 0 1" range="0 0.32" damping="20"/>
        <geom name="pad" type="cylinder" size="0.22 0.008" pos="0 0 0" rgba="0.9 0.49 0.13 1"/>
        <geom name="cone" type="cylinder" size="0.05 0.04" pos="0 0 0.04" rgba="0.9 0.49 0.13 1"/>
      </body>
    </body>
    <body name="drone" pos="-0.72 0 0.42" childclass="droneg">
      <freejoint name="drone"/>
      <geom name="fus" type="capsule" size="0.045 0.47" pos="0 0 0.6" rgba="0.95 0.77 0.06 1" mass="1.9"/>
      <geom name="canopy" type="ellipsoid" size="0.07 0.045 0.11" pos="0.05 0 0.74" rgba="0.55 0.75 0.9 1" mass="0.05"/>
      <geom name="wing" type="box" size="0.08 0.3 0.009" pos="0 0 0.5" rgba="0.95 0.77 0.06 1" mass="0.3"/>
      <geom name="pod1" type="cylinder" size="0.047 0.13" pos="0 0.303 0.5" rgba="0.3 0.3 0.3 1" mass="0.33"/>
      <geom name="pod2" type="cylinder" size="0.047 0.13" pos="0 -0.303 0.5" rgba="0.3 0.3 0.3 1" mass="0.33"/>
      <geom name="fin1" type="box" size="0.065 0.003 0.09" pos="0.1 0.1 0.12" euler="0 0 45" rgba="0.3 0.3 0.3 1" mass="0.05"/>
      <geom name="fin2" type="box" size="0.065 0.003 0.09" pos="-0.1 0.1 0.12" euler="0 0 -45" rgba="0.3 0.3 0.3 1" mass="0.05"/>
      <geom name="fin3" type="box" size="0.065 0.003 0.09" pos="-0.1 -0.1 0.12" euler="0 0 45" rgba="0.3 0.3 0.3 1" mass="0.05"/>
      <geom name="fin4" type="box" size="0.065 0.003 0.09" pos="0.1 -0.1 0.12" euler="0 0 -45" rgba="0.3 0.3 0.3 1" mass="0.05"/>
      <site name="cg" pos="0 0 0.5"/>
    </body>
  </worldbody>
  <equality>
    <weld name="latch" body1="lift" body2="drone" relpose="0 0 0.12 1 0 0 0"/>
  </equality>
  <actuator>
    <position name="hatch_act" joint="hatch" kp="40"/>
    <position name="lift_act" joint="lift" kp="400"/>
  </actuator>
</mujoco>
"""

# the sequence [s]: hatch opens, lift rises, the latch releases, hover climb, pitch-over, level flight
SEQ = dict(hatch=(1.0, 4.0), lift=(4.0, 8.0), release=9.0, hover_to=15.0, climb_m=12.0, pitch_over=16.0, end=21.0, away_m=120.0)


def _quat_to_mat(q):
    r = np.zeros(9)
    mujoco.mju_quat2Mat(r, q)
    return r.reshape(3, 3)


def controller(m, d, t):
    """Hatch and lift by schedule; the drone: thrust along its axis and attitude torques, lift from airspeed."""
    d.ctrl[0] = 2.1 * np.clip((t - SEQ["hatch"][0]) / (SEQ["hatch"][1] - SEQ["hatch"][0]), 0, 1)
    d.ctrl[1] = 0.32 * np.clip((t - SEQ["lift"][0]) / (SEQ["lift"][1] - SEQ["lift"][0]), 0, 1)
    d.eq_active[0] = 1 if t < SEQ["release"] else 0
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "drone")
    R = _quat_to_mat(d.xquat[bid])
    axis = R[:, 2]                                                      # the drone's nose (its +z in the body frame)
    v = d.cvel[bid][3:6].copy()
    pos = d.xpos[bid]
    d.xfrc_applied[bid][:] = 0.0
    if t < SEQ["release"]:
        return
    # the wanted axis: vertical while climbing, then pitching over towards +x
    if t < SEQ["pitch_over"] or pos[2] < 3.0:
        # hover: tilt the axis against the lateral velocity and the offset from the launch line, hold the climb height
        want = np.array([-0.15 * v[0] - 0.05 * pos[0], -0.15 * v[1] - 0.05 * pos[1], 1.0])
        want /= np.linalg.norm(want)
        vz_want = 2.0 if pos[2] < SEQ["climb_m"] else 0.0
        thrust = DRONE["mass"] * G * np.clip(1.0 + 0.3 * (vz_want - v[2]), 0.6, 1.33)
    else:
        # pitch over at full thrust, no faster than the airspeed lets the wing carry the weight (20° + 4°/(m/s))
        va_now = max(float(np.dot(v, axis)), 0.0)
        ang = math.radians(min(78.0, 20.0 + 4.0 * va_now))
        want = np.array([math.sin(ang), -0.15 * v[1] - 0.05 * pos[1], math.cos(ang)])
        want /= np.linalg.norm(want)
        thrust = DRONE["thrust_max"]
    # attitude PD: torque to turn the axis onto the wanted one, damped on the angular rate
    err = np.cross(axis, want)
    w = d.cvel[bid][0:3]
    torque = 15.0 * err - 2.5 * w
    # lift: the wing's normal is the body +x (the wing lies in the fuselage's y-z plane... its chord along body -z);
    # airspeed component along the axis gives lift normal to it, up to Cl_max
    va = float(np.dot(v, axis))
    q_dyn = 0.5 * RHO_AIR * max(va, 0.0) ** 2
    lift_dir = np.array([0.0, 0.0, 1.0]) - axis * axis[2]
    nrm = np.linalg.norm(lift_dir)
    lift = np.zeros(3) if nrm < 1e-6 else lift_dir / nrm * min(q_dyn * DRONE["wing_area"] * DRONE["cl_max"], DRONE["mass"] * G * 1.2)
    drag = -0.5 * RHO_AIR * 0.02 * np.linalg.norm(v) * v
    d.xfrc_applied[bid][0:3] = axis * thrust + lift + drag
    d.xfrc_applied[bid][3:6] = torque


def hull_motion(d, t):
    """The swell: a slow roll and heave of the surfaced hull (sea state 2-ish)."""
    roll = math.radians(4.0) * math.sin(2 * math.pi * t / 6.0)
    heave = 0.06 * math.sin(2 * math.pi * t / 5.0)
    d.mocap_pos[0] = [0.0, 0.0, 0.05 + heave]
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.array([1.0, 0.0, 0.0]), roll)
    d.mocap_quat[0] = q


def run(record_every=0.1):
    m = mujoco.MjModel.from_xml_string(XML)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    frames, t_next = [], 0.0
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "drone")
    while d.time < SEQ["end"] and np.linalg.norm(d.xpos[bid][:2]) < SEQ["away_m"]:
        hull_motion(d, d.time)
        controller(m, d, d.time)
        mujoco.mj_step(m, d)
        if d.time >= t_next:
            frames.append((d.time, snapshot(m, d)))
            t_next += record_every
    return m, frames


def snapshot(m, d):
    out = []
    for i in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if name == "sea":
            continue
        out.append((m.geom_type[i], d.geom_xpos[i].copy(), d.geom_xmat[i].reshape(3, 3).copy(), m.geom_size[i].copy(), m.geom_rgba[i].copy()))
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "drone")
    return {"geoms": out, "drone": d.xpos[bid].copy(), "speed": float(np.linalg.norm(d.cvel[bid][3:6]))}


# --- drawing (the same geom-polygon approach as the land lab's teleop drawings)
def _box(c, R, h):
    s = np.array([[a, b, e] for a in (-1, 1) for b in (-1, 1) for e in (-1, 1)]) * h
    v = s @ R.T + c
    f = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
    return [v[k] for k in f]


def _cyl(c, R, r, hl, n=12, caps=True):
    t = np.linspace(0, 2 * np.pi, n + 1)
    ring = np.c_[r * np.cos(t), r * np.sin(t)]
    out = []
    for k in range(n):
        a, b = ring[k], ring[k + 1]
        out.append(np.array([[a[0], a[1], -hl], [b[0], b[1], -hl], [b[0], b[1], hl], [a[0], a[1], hl]]) @ R.T + c)
    if caps:
        for z in (-hl, hl):
            out.append(np.c_[ring, np.full(len(ring), z)] @ R.T + c)
    return out


def _sphere(c, r, n=6, R=None):
    """A sphere of radius ``r``, or with ``r`` a 3-vector and ``R`` the ellipsoid's axes."""
    out = []
    rr = np.broadcast_to(np.asarray(r, float), (3,))
    R = np.eye(3) if R is None else R
    th, ph = np.linspace(0, np.pi, n + 1), np.linspace(0, 2 * np.pi, 2 * n + 1)
    for i in range(n):
        for j in range(2 * n):
            pts = [(th[i], ph[j]), (th[i + 1], ph[j]), (th[i + 1], ph[j + 1]), (th[i], ph[j + 1])]
            out.append(np.array([[rr[0] * math.sin(a) * math.cos(b), rr[1] * math.sin(a) * math.sin(b), rr[2] * math.cos(a)] for a, b in pts]) @ R.T + c)
    return out


def draw_frame(ax, snap, t):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    for gtype, c, R, sz, rgba in snap["geoms"]:
        if gtype == mujoco.mjtGeom.mjGEOM_BOX:
            polys = _box(c, R, sz[:3])
        elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
            polys = _cyl(c, R, sz[0], sz[1])
        elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
            polys = _cyl(c, R, sz[0], sz[1], caps=False) + _sphere(c + R[:, 2] * sz[1], sz[0], 4) + _sphere(c - R[:, 2] * sz[1], sz[0], 4)
        elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            polys = _sphere(c, sz[0])
        elif gtype == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            polys = _sphere(c, sz[:3], 5, R)
        else:
            continue
        ax.add_collection3d(Poly3DCollection(polys, facecolors=[rgba], edgecolors=[rgba * [0.6, 0.6, 0.6, 1]], linewidths=0.2, shade=True))
    # the sea and the camera following the drone up
    dz = snap["drone"][2]
    pad = np.array([-0.72, 0.0, 0.7])
    mid = 0.5 * (pad + snap["drone"])
    span = max(2.6, 0.62 * np.linalg.norm(snap["drone"] - pad) + 1.5)
    X, Y = np.meshgrid(np.linspace(mid[0] - span, mid[0] + span, 2), np.linspace(mid[1] - span, mid[1] + span, 2))
    ax.plot_surface(X, Y, np.zeros_like(X), color="#7fb3d5", alpha=0.35)
    ax.set_xlim(mid[0] - span, mid[0] + span); ax.set_ylim(mid[1] - span, mid[1] + span); ax.set_zlim(max(mid[2] - span, -0.5), max(mid[2] - span, -0.5) + 2 * span)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=14, azim=-50); ax.set_axis_off()
    phase = ("hatch opening" if t < SEQ["lift"][0] else "the lift raises the X-wing" if t < SEQ["release"] else
             "hover climb" if t < SEQ["pitch_over"] else "pitch-over into level flight")
    ax.set_title(f"t = {t:4.1f} s — {phase};  drone at {dz:.1f} m, {snap['speed']:.1f} m/s", fontsize=11, loc="left")


def main(out="dolphin_launch.mp4"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m, frames = run()
    tmp = tempfile.mkdtemp()
    for k, (t, snap) in enumerate(frames):
        fig = plt.figure(figsize=(10.24, 7.2))
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        draw_frame(ax, snap, t)
        fig.suptitle("Dolphin launches an Emerillion (MuJoCo; the swell as a 4° roll and 6 cm heave)", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(tmp, f"f{k:04d}.png"), dpi=72)
        plt.close(fig)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "10", "-i", os.path.join(tmp, "f%04d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", out], check=True)
    last = frames[-1][1]
    return out, len(frames), last["drone"], last["speed"]


if __name__ == "__main__":
    print(main(*sys.argv[1:]))
