"""Cleopatra, "Operation Pipe Entry" (``./cleopatra.sh mission``): a night-time MuJoCo short. Cleopatra — the
three-segment, twelve-legged Myropod (``designs/myropod_robot.py``, notebook 18), 6.2 kg, here with two small pincer
arms under its head (a gripper, which carries the pendrive, and a cutter, as TARANTULA has) and a camera-and-lidar
turret on its head — crosses the street outside the house of Operation Silent Drop to the intake of the bandit's
air-conditioning duct (a 260 mm pipe), crawls in, along the ground run, up the riser on the house wall and through
the wall. At the vent over the desk the cutter snips the grille's slats one by one where a 120 mm hole is to be (each
drops onto the desk once cut at both ends), and Cleopatra climbs out, walks to the laptop, crouches, and the gripper plugs the pendrive into the laptop's back port; after the
copy it unplugs it and Cleopatra backs out the way it came (time-lapsed). The bandit sleeps in his armchair.

How it moves (scripted kinematics, rendered by MuJoCo; no dynamics): the head follows a path — the yard, the duct's
centre line, the desk — and the segments trail it, each turned to the path's tangent at its own place, the passive
yaw → pitch → roll body joints between them set from the segments' relative rotation (within their ±45° / ±20°
stops). The twelve legs walk with a free gait: every foot stays where it stands — on the ground, the pipe's wall or
the desk — and the body advances only while each planted foot is within reach of where it wants to be; the legs step
four groups in turn, each group one leg per segment, so every segment always keeps three feet down; a stepping foot
lifts off along the surface normal (up from the ground, in towards the pipe's axis) and sets down ahead. Each leg's
three joints (hip yaw, hip pitch, knee) come from its own analytic inverse kinematics.

    ./cleopatra.sh mission                     # runs/cleopatra/cleopatra_mission.mp4 (1280 × 720, 30 fps)
    ./cleopatra.sh mission --text              # with captions

Needs mujoco, cadquery (the CAD → meshes, cached under runs/cleopatra/mission_cache), Pillow, ffmpeg.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
ROOT = LAB.parent.parent
for _p in (LAB / "designs", LAB / "common", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import tarantula_mission as tm
import tarantula_video as tv

sm, rot_z, rot_y, mat2quat, slerp, _s = tm.sm, tm.rot_z, tm.rot_y, tm.mat2quat, tm.slerp, tm._s

# ---- the set [m]
R_DUCT = 0.13                                                             # the duct's inner radius (260 mm)
R_ELBOW = 0.30                                                            # the elbows' centre-line radius
X_MOUTH, X_RISER, Z_IN, X_END = 1.70, 0.45, 0.90, -0.42                   # intake, riser, indoor height, vent
WALL = (-0.25, 0.0)                                                       # the house wall (x from, to)
DESK = (-1.10, -0.25, -0.75, 0.75, 0.75)                                  # x0, x1, y0, y1, top
STAND_H = 0.0                                                             # the laptop sits on the desk
LAPTOP = (-0.85, 0.0)                                                     # its centre (x, y); screen faces -x
HIP_H = 0.163                                                             # standing body height over the ground
SPEEDUP = 5.0                                                             # the retreat, time-lapsed
COLORS = {"shell": (0.36, 0.38, 0.41), "head": (0.27, 0.29, 0.32), "leg": (0.14, 0.15, 0.16),
          "pad": (0.06, 0.06, 0.06)}


# --------------------------------------------------------------------------------------------- the CAD as meshes
def cleo_meshes(cache: Path):
    """{part: stl path} — Cleopatra's CAD (``myropod.Myropod`` at notebook 18's ``CLEO_MM``) cut onto the Chiron
    links, each in its link frame: the head and a segment (centred at hip height), and the femur and tibia of a
    left and a right leg (straight out sideways, as the links are at q = 0)."""
    import cadquery as cq
    import myropod
    import myropod_robot as mr
    key = hashlib.sha1((LAB / "designs" / "myropod.py").read_bytes() + b"cleo-links-1").hexdigest()[:12]
    out = cache / f"cleo_{key}"
    index = out / "index.json"
    if index.exists():
        return json.loads(index.read_text())
    out.mkdir(parents=True, exist_ok=True)
    print("cleopatra_mission: tessellating the CAD (once; cached in", out, ")", flush=True)
    d = myropod.Myropod()
    p = d.resolve(**mr.CLEO_MM)
    zc = p["seg_height"] / 2
    dl, f, tb, pad = p["leg_diameter"], p["femur_length"], p["tibia_length"], p["foot_diameter"]

    def bar(length, w):
        return cq.Workplane("YZ").center(length / 2, 0).slot2D(length + w, w, 0).extrude(w / 2, both=True)

    femur = cq.Workplane("YZ").circle(dl / 2 + 2).extrude(5, both=True).union(bar(f, dl))
    tibia = (bar(tb, dl * 0.8).union(cq.Workplane("YZ").circle(dl / 2 + 1).extrude(dl / 2 + 1, both=True))
             .union(cq.Workplane("XY").sphere(pad / 2).translate((0, tb, 0))))
    parts = {"head": d._head(p).translate((0, 0, -zc)), "segment": d._segment(p).translate((0, 0, -zc)),
             "femur_l": femur, "tibia_l": tibia, "femur_r": femur.mirror("XZ"), "tibia_r": tibia.mirror("XZ")}
    idx = {}
    for name, w in parts.items():
        Vs, Fs, n0 = [], [], 0
        for sol in w.vals():
            vs, fs = sol.tessellate(0.25, 0.25)
            Vs.append(np.array([v.toTuple() for v in vs]) * 1e-3)
            Fs.append(np.array(fs) + n0)
            n0 += len(vs)
        path = out / f"{name}.stl"
        tv._stl(path, np.vstack(Vs), np.vstack(Fs))
        idx[name] = str(path)
    index.write_text(json.dumps(idx))
    return idx


# --------------------------------------------------------------------------------------------- the path
def _rmf(P, up0=(0.0, 0.0, 1.0)):
    """Tangents and rotation-minimising 'up' vectors along the polyline P (double reflection)."""
    T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    U = np.zeros_like(P)
    u = np.asarray(up0, float) - np.dot(up0, T[0]) * T[0]
    U[0] = u / np.linalg.norm(u)
    for i in range(len(P) - 1):
        v1 = P[i + 1] - P[i]
        c1 = v1 @ v1
        if c1 < 1e-16:
            U[i + 1] = U[i]
            continue
        rL = U[i] - (2 / c1) * (v1 @ U[i]) * v1
        tL = T[i] - (2 / c1) * (v1 @ T[i]) * v1
        v2 = T[i + 1] - tL
        c2 = v2 @ v2
        U[i + 1] = rL - (2 / c2) * (v2 @ rL) * v2 if c2 > 1e-16 else rL
        U[i + 1] -= (U[i + 1] @ T[i + 1]) * T[i + 1]
        U[i + 1] /= np.linalg.norm(U[i + 1])
    return T, U


def _resample(P, ds=0.004):
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    keep = np.concatenate([[True], seg > 1e-9])
    P, s = P[keep], s[keep]
    su = np.arange(0.0, s[-1], ds)
    return np.column_stack([np.interp(su, s, P[:, k]) for k in range(3)]), su


def duct_centreline():
    """The duct's centre line: the ground run from the intake, an elbow up, the riser on the wall, an elbow in, and
    the run through the wall to the vent over the desk."""
    pts = [np.array([X_MOUTH, 0.0, R_DUCT]), np.array([X_RISER + R_ELBOW, 0.0, R_DUCT])]
    a = np.linspace(0, math.pi / 2, 24)
    c1 = np.array([X_RISER + R_ELBOW, 0.0, R_DUCT + R_ELBOW])
    pts += [c1 + R_ELBOW * np.array([-math.sin(t), 0.0, -math.cos(t)]) for t in a[1:]]
    c2 = np.array([X_RISER - R_ELBOW, 0.0, Z_IN - R_ELBOW])
    pts += [c2 + R_ELBOW * np.array([math.cos(t), 0.0, math.sin(t)]) for t in a]
    pts.append(np.array([X_END, 0.0, Z_IN]))
    P, s = _resample(np.array(pts), 0.004)
    return P, s


def robot_path():
    """The head's path (centre at hip height): a lead-in behind the start (for the trailing segments), the yard, in at
    the intake, the duct's centre line, out over the desk to the laptop's back port, crouching at the end."""
    C, _ = duct_centreline()
    start = np.array([3.55, -1.25, HIP_H])
    pre = start + np.array([0.6, -0.3, 0.0])
    a0 = np.array([X_MOUTH + 0.35, 0.0, HIP_H])
    yard = [pre, start]
    for u in np.linspace(0, 1, 60)[1:]:                                   # a cubic from the start to the intake
        b = (1 - u) ** 3 * start + 3 * (1 - u) ** 2 * u * (start + np.array([-0.7, 0.35, 0])) \
            + 3 * (1 - u) * u ** 2 * (a0 + np.array([0.6, 0, 0])) + u ** 3 * a0
        yard.append(b)
    lead = [a0 + (C[0] - a0) * u for u in np.linspace(0, 1, 20)[1:]]      # down into the duct's axis
    desk_z = DESK[4] + HIP_H
    port = port_point()
    head_x_end = port[0] + PLUG_REACH_X                                   # the gripper reaches the rest of the way
    crouch_z = DESK[4] + 0.13
    d0 = np.array([X_END - 0.06, 0.0, desk_z])
    d1 = np.array([head_x_end + 0.08, 0.0, desk_z])
    d2 = np.array([head_x_end, 0.0, crouch_z])
    desk = [d0, d1] + [d1 + (d2 - d1) * sm(u) for u in np.linspace(0, 1, 20)[1:]]
    P, s = _resample(np.array(yard + lead + list(C[1:]) + desk), 0.004)
    T, U = _rmf(P)
    s_start = float(np.linalg.norm(pre[:2] - start[:2]))
    return P, s, T, U, s_start


# ---- Cleopatra's pincer arms (like TARANTULA's pedipalps), head frame [m]: a shoulder (yaw + pitch) under the
# head's front, an upper arm, an elbow, a forearm and a pincer — the gripper on the left (it carries the pendrive),
# the cutter on the right
ARM = {"base_x": 0.055, "base_y": 0.035, "base_z": -0.028, "upper": 0.115, "fore": 0.110, "finger": 0.022}
HOLE_R = 0.12                                                             # the hole the cutter cuts in the grille
ARM_TUCK = {"g": np.array([0.11, 0.035, -0.06]), "c": np.array([0.105, -0.035, -0.058])}
USB_HALF = 0.0225                                                         # the pendrive body's half length
USB_PLUG = USB_HALF + 0.012                                               # its centre to the plug's tip
GRIP_IN = 0.010                                                           # the gripper's tip to its grip point
PLUG_REACH_X = 0.19                                                       # the head stops this far behind the port


def port_point():
    """The laptop's back USB port (the back face's centre line, 10 mm above the desk)."""
    return np.array([LAPTOP[0] + 0.115, LAPTOP[1], DESK[4] + STAND_H + 0.010])


# --------------------------------------------------------------------------------------------- the scene
def build(width, height, cache: Path):
    import mujoco
    import myropod_robot as mr
    from vegeta.chiron.robot import SimOptions
    robot = mr.cleopatra("spring_damper")
    opt = SimOptions(timestep=0.001, gravity=(0.0, 0.0, -9.81))
    root = ET.fromstring(robot.to_mjcf(options=opt))
    meshes = cleo_meshes(cache)
    vis = tv._child(root, "visual")
    tv._child(vis, "global").attrib.update({"offwidth": str(width), "offheight": str(height), "fovy": "45"})
    tv._child(vis, "quality").attrib.update({"shadowsize": "4096", "offsamples": "8"})
    tv._child(vis, "headlight").attrib.update({"ambient": "0.17 0.18 0.22", "diffuse": "0.14 0.14 0.16",
                                               "specular": "0.02 0.02 0.02"})
    tv._child(vis, "map").attrib.update({"znear": "0.004", "zfar": "60", "haze": "0.2"})
    asset = tv._child(root, "asset")
    A = lambda tag, **kw: ET.SubElement(asset, tag, {k: str(v) for k, v in kw.items()})
    A("texture", name="sky", type="skybox", builtin="gradient", rgb1="0.05 0.07 0.15", rgb2="0 0 0.02", mark="random",
      random="0.0025", markrgb="0.9 0.9 1", width="800", height="800")
    A("texture", name="grass", type="2d", builtin="flat", rgb1="0.13 0.2 0.1", rgb2="0.08 0.12 0.06", mark="random",
      random="0.3", markrgb="0.2 0.3 0.12", width="512", height="512")
    A("material", name="grass", texture="grass", texrepeat="10 10", texuniform="true", specular="0.02")
    tile = tv._brick_png(cache / "brick.png")
    A("texture", name="brick", type="2d", file=str(cache / "brick.png"))
    A("material", name="brick", texture="brick", texrepeat=f"{1 / tile:.3f} {1 / tile:.3f}", texuniform="true",
      specular="0.05")
    A("texture", name="boards", type="2d", builtin="gradient", rgb1="0.42 0.28 0.16", rgb2="0.3 0.19 0.1",
      width="256", height="256")
    A("material", name="floor", texture="boards", texrepeat="8 8", texuniform="true", specular="0.15")
    A("material", name="plaster", rgba="0.62 0.6 0.55 1", specular="0.02")
    A("material", name="desk", rgba="0.36 0.23 0.13 1", specular="0.2")
    A("material", name="duct", rgba="0.78 0.8 0.84 0.22", specular="0.6", shininess="0.6")
    A("material", name="metal", rgba="0.6 0.62 0.65 1", specular="0.5", shininess="0.5")
    A("material", name="dark", rgba="0.1 0.1 0.11 1", specular="0.3")
    A("material", name="screen", rgba="0.25 0.45 0.85 1", emission="0.6")
    A("material", name="bar", rgba="0.2 0.95 0.35 1", emission="0.9")
    A("material", name="glow", rgba="1 0.8 0.5 1", emission="1")
    A("material", name="headlamp", rgba="1 0.97 0.85 1", emission="1")
    A("material", name="usb_led", rgba="0.2 0.6 1 1", emission="0")
    A("material", name="window", rgba="0.95 0.72 0.4 1", emission="0.45")
    A("material", name="lens", rgba="0.1 0.2 0.35 1", specular="1", shininess="1")
    A("material", name="lidar", rgba="0.9 0.2 0.15 1", emission="0.6")
    A("texture", name="paving", type="2d", builtin="checker", rgb1="0.27 0.26 0.25", rgb2="0.23 0.22 0.21",
      width="512", height="512")
    A("material", name="paving", texture="paving", texrepeat="4 4", texuniform="true", specular="0.08")
    A("material", name="stone", rgba="0.62 0.6 0.56 1", specular="0.05")
    A("material", name="wood", rgba="0.3 0.19 0.11 1", specular="0.2")
    A("material", name="blade", rgba="0.85 0.87 0.9 1", specular="0.8", shininess="0.8")
    for k, (r, g, b) in COLORS.items():
        A("material", name=f"cleo:{k}", rgba=f"{r} {g} {b} 1", specular="0.45", shininess="0.45")
    for name, path in meshes.items():
        A("mesh", name=f"m:{name}", file=path)
    world = root.find("worldbody")
    for gm in world.findall("geom"):                                      # the Chiron ground: the yard and the floor
        if gm.get("type") == "plane":
            gm.set("size", "15 15 0.1")
            gm.set("material", "paving")
            gm.attrib.pop("rgba", None)
    G = lambda parent, **kw: ET.SubElement(parent, "geom", {"contype": "0", "conaffinity": "0", "group": "1",
                                                            **{k: str(v) for k, v in kw.items()}})
    # ---- the robot: collision geoms out of sight, its CAD on the links; the pendrive's bracket and the headlamps
    bodies = {b.get("name"): b for b in root.iter("body")}
    for b in bodies.values():
        for gm in b.findall("geom"):
            if not gm.get("name", "").startswith("pm:"):
                gm.set("group", "3")
    G(bodies["head"], type="mesh", mesh="m:head", material="cleo:head", mass="0")
    for i in (1, 2, 3):
        G(bodies[f"segment {i}"], type="mesh", mesh="m:segment", material="cleo:shell", mass="0")
        for key in mr.LEG_KEYS:
            side = "l" if mr.LEG_SIDES[key][1] > 0 else "r"
            n = mr.leg_name(i, key)
            G(bodies[f"{n} femur"], type="mesh", mesh=f"m:femur_{side}", material="cleo:leg", mass="0")
            G(bodies[f"{n} tibia"], type="mesh", mesh=f"m:tibia_{side}", material="cleo:pad", mass="0")
    for sy in (-1, 1):                                                    # the arms' mounts
        G(bodies["head"], type="box", size="0.008 0.007 0.006",
          pos=f"{ARM['base_x'] - 0.004} {sy * ARM['base_y']} {ARM['base_z'] + 0.004}", material="dark", mass="0")
    # the sensor turret, as on TARANTULA: a camera with its lens, a lidar puck on top
    G(bodies["head"], type="box", size="0.026 0.021 0.014", pos="-0.005 0 0.05", material="dark", mass="0")
    G(bodies["head"], type="cylinder", size="0.0095 0.006", pos="0.024 0 0.051", zaxis="1 0 0", material="metal",
      mass="0")
    G(bodies["head"], type="cylinder", size="0.0065 0.0015", pos="0.031 0 0.051", zaxis="1 0 0", material="lens",
      mass="0")
    G(bodies["head"], type="cylinder", size="0.013 0.006", pos="-0.008 0 0.071", material="metal", mass="0")
    G(bodies["head"], type="cylinder", size="0.0135 0.002", pos="-0.008 0 0.069", material="lidar", mass="0")
    for sy in (-1, 1):
        G(bodies["head"], type="cylinder", size="0.0045 0.002", pos=f"0.066 {sy * 0.031} 0", zaxis="1 0 0",
          material="headlamp", mass="0")
    ET.SubElement(bodies["head"], "light", {"name": "headlamp", "pos": "0.07 0 0.005", "dir": "1 0 -0.08",
                                            "diffuse": "0.9 0.88 0.8", "specular": "0.2 0.2 0.2", "cutoff": "38",
                                            "exponent": "6", "attenuation": "1 0 2", "castshadow": "false"})
    # ---- the house: the wall (brick outside, plaster in), the room, the roof
    x0, x1 = WALL
    G(world, name="wall_out", type="box", size="0.004 3 1.2", pos=f"{x1 - 0.004} 0 1.2", material="brick")
    G(world, name="wall", type="box", size=f"{(x1 - x0) / 2 - 0.004} 3 1.2", pos=f"{(x0 + x1) / 2 - 0.002} 0 1.2",
      material="plaster")
    G(world, type="box", size="0.05 3 1.2", pos="-4.05 0 1.2", material="plaster")
    for sy in (-1, 1):
        G(world, type="box", size="1.92 0.05 1.2", pos=f"-2.08 {sy * 3.0} 1.2", material="plaster")
    G(world, type="box", size="2.1 3.05 0.045", pos="-2.0 0 2.355", material="dark")
    G(world, type="box", size="0.27 3.05 0.07", pos=f"{x1 - 0.03} 0 2.43", material="stone")   # the coping
    G(world, type="box", size="0.012 3.05 0.008", pos=f"{x1 + 0.228} 0 2.352", material="stone")
    G(world, type="box", size="1.9 2.95 0.006", pos="-2.13 0 0.006", material="floor")
    for k, y in enumerate((-1.4, 1.25)):                                  # two windows, one lit (as before)
        G(world, type="box", size="0.004 0.26 0.36", pos=f"{x1 + 0.003} {y} 1.35",
          material="window" if k else "dark")
        G(world, type="box", size="0.04 0.3 0.02", pos=f"{x1 + 0.04} {y} 0.91", material="stone")
        G(world, type="box", size="0.004 0.26 0.36", pos=f"{x0 - 0.003} {y} 1.35",
          material="window" if k else "dark")
    lant = np.array([x1 + 0.07, -0.55, 1.68])                             # the lantern
    G(world, type="box", size="0.035 0.01 0.01", pos=_s(lant + [-0.035, 0, 0.09]), material="dark")
    G(world, type="box", size="0.05 0.05 0.075", pos=_s(lant), rgba="0.1 0.1 0.1 0.35")
    G(world, type="sphere", size="0.028", pos=_s(lant), material="glow")
    G(world, type="box", size="0.055 0.055 0.01", pos=_s(lant + [0, 0, 0.083]), material="dark")
    ET.SubElement(world, "light", {"name": "lantern", "pos": _s(lant + [0.06, 0.05, -0.05]), "dir": "0.45 0.35 -1",
                                   "diffuse": "1.0 0.66 0.32", "specular": "0.2 0.15 0.1", "cutoff": "75",
                                   "exponent": "2", "attenuation": "1 0 0.25", "castshadow": "true"})
    tt = np.array([x1 + 0.42, 1.55, 0.75])                                # the terrace table, bottle and mug
    G(world, type="box", size="0.36 0.42 0.018", pos=_s(tt - [0, 0, 0.018]), material="wood")
    for sx in (-0.32, 0.32):
        for sy in (-0.38, 0.38):
            G(world, type="box", size="0.02 0.02 0.357", pos=_s(tt + [sx, sy, -0.393]), material="wood")
    G(world, type="cylinder", size="0.035 0.11", pos=_s(tt + [0.24, -0.27, 0.11]), rgba="0.1 0.35 0.15 0.85")
    G(world, type="cylinder", size="0.013 0.04", pos=_s(tt + [0.24, -0.27, 0.25]), rgba="0.1 0.35 0.15 0.85")
    G(world, type="cylinder", size="0.04 0.045", pos=_s(tt + [0.16, 0.26, 0.045]), rgba="0.85 0.85 0.8 1")
    # ---- the duct (translucent, ribbed), its intake and vent
    C, _ = duct_centreline()
    step = 12
    for i in range(0, len(C) - step, step):
        a, b = C[i], C[min(i + step, len(C) - 1)]
        G(world, type="cylinder", size=f"{R_DUCT + 0.003:.4f}", fromto=_s(np.concatenate([a, b])), material="duct")
    G(world, type="cylinder", size=f"{R_DUCT + 0.02:.4f} 0.012", pos=_s(C[0]), zaxis="1 0 0", material="metal")
    G(world, type="box", size="0.004 0.17 0.17", pos=f"{X_MOUTH + 0.25} 0.45 0.02", euler="0 1.45 0.4",
      material="metal")                                                   # the intake's grille, unscrewed
    for k in range(-5, 6):
        G(world, type="box", size="0.0045 0.15 0.003", pos=f"{X_MOUTH + 0.25} {0.45 + 0.025 * k:.3f} 0.026",
          euler="0 1.45 0.4", material="dark")
    gx = X_END - 0.006                                                    # the vent's grille: a frame and slats
    for sy in (-1, 1):
        G(world, type="box", size="0.005 0.012 0.16", pos=f"{gx} {sy * 0.148} {Z_IN}", rgba="0.75 0.76 0.78 1")
        G(world, type="box", size="0.005 0.16 0.012", pos=f"{gx} 0 {Z_IN + sy * 0.148}", rgba="0.75 0.76 0.78 1")
    for k in range(-6, 7):
        z = 0.022 * k
        w = math.sqrt(max(HOLE_R ** 2 - z * z, 0.0))                      # the slat's part inside the hole
        for sy in (-1, 1):
            a, b = (w, 0.136) if w > 0 else (0.0, 0.136)
            G(world, type="box", size=f"0.004 {(b - a) / 2:.4f} 0.0045", pos=f"{gx} {sy * (a + b) / 2:.4f} {Z_IN + z:.4f}",
              rgba="0.7 0.71 0.74 1")
        if w > 0.004:                                                     # the slat's middle: cut free, it drops
            sl = ET.SubElement(world, "body", {"name": f"slat{k + 6}", "mocap": "true", "pos": f"{gx} 0 {Z_IN + z:.4f}"})
            G(sl, type="box", size=f"0.004 {w:.4f} 0.0045", rgba="0.7 0.71 0.74 1")
    # ---- the yard: the AC unit by the riser, bushes
    G(world, type="box", size="0.3 0.25 0.38", pos=f"{X_RISER + 0.1} -0.7 0.38", material="metal")
    G(world, type="cylinder", size="0.18 0.005", pos=f"{X_RISER + 0.41} -0.7 0.42", zaxis="1 0 0", material="dark")
    # ---- the room: the desk, the laptop on its stand, the lamp, the armchair and the sleeping bandit
    dx0, dx1, dy0, dy1, dz = DESK
    G(world, name="desk", type="box", size=f"{(dx1 - dx0) / 2} {(dy1 - dy0) / 2} 0.02",
      pos=f"{(dx0 + dx1) / 2} {(dy0 + dy1) / 2} {dz - 0.02}", material="desk")
    for sx in (dx0 + 0.05, dx1 - 0.05):
        for sy in (dy0 + 0.05, dy1 - 0.05):
            G(world, type="box", size=f"0.025 0.025 {(dz - 0.04) / 2}", pos=f"{sx} {sy} {(dz - 0.04) / 2}",
              material="desk")
    lx, ly = LAPTOP
    G(world, type="box", size="0.115 0.165 0.01", pos=f"{lx} {ly} {dz + STAND_H + 0.01}", material="metal",
      rgba="0.55 0.56 0.6 1")
    port = port_point()
    G(world, type="box", size="0.002 0.0075 0.0035", pos=_s(port + [0.0005, 0, 0]), material="dark")
    scr = ET.SubElement(world, "body", {"name": "screen", "pos": f"{lx + 0.115} {ly} {dz + STAND_H + 0.02}",
                                        "euler": "0 0.3 0"})
    G(scr, type="box", size="0.005 0.165 0.11", pos="-0.006 0 0.11", material="metal", rgba="0.5 0.51 0.55 1")
    G(scr, name="display", type="box", size="0.0008 0.152 0.097", pos="-0.0115 0 0.11", material="screen")
    G(scr, name="bar_bg", type="box", size="0.0008 0.1 0.008", pos="-0.0125 0 0.09", material="dark")
    G(scr, name="bar", type="box", size="0.0008 0.0005 0.0065", pos="-0.0132 -0.1 0.09", material="bar")
    lamp = np.array([dx0 + 0.15, dy1 - 0.2, dz])
    G(world, type="cylinder", size="0.07 0.01", pos=_s(lamp + [0, 0, 0.01]), material="dark")
    G(world, type="capsule", size="0.008", fromto=_s(np.concatenate([lamp + [0, 0, 0.02], lamp + [0.1, -0.05, 0.38]])),
      material="dark")
    G(world, type="cylinder", size="0.07 0.045", pos=_s(lamp + [0.14, -0.08, 0.38]), euler="0 0.6 0", material="dark")
    G(world, type="sphere", size="0.025", pos=_s(lamp + [0.15, -0.09, 0.34]), material="glow")
    ET.SubElement(world, "light", {"name": "lamp", "pos": _s(lamp + [0.17, -0.1, 0.33]), "dir": "0.5 -0.4 -1",
                                   "diffuse": "1 0.72 0.4", "specular": "0.15 0.1 0.05", "cutoff": "55",
                                   "exponent": "4", "attenuation": "1 0 0.4", "castshadow": "true"})
    ET.SubElement(world, "light", {"name": "screenglow", "pos": f"{lx - 0.15} {ly} {dz + STAND_H + 0.15}",
                                   "dir": "-1 0 -0.3", "diffuse": "0.12 0.2 0.4", "specular": "0 0 0",
                                   "cutoff": "70", "castshadow": "false"})
    ET.SubElement(world, "light", {"name": "moon", "directional": "true", "pos": "5 -3 6", "dir": "-0.5 0.35 -0.8",
                                   "diffuse": "0.28 0.32 0.45", "specular": "0.05 0.05 0.08", "castshadow": "true"})
    ET.SubElement(world, "light", {"name": "room", "pos": "-2 0 2.4", "dir": "0 0 -1", "diffuse": "0.12 0.11 0.1",
                                   "specular": "0 0 0", "cutoff": "80", "castshadow": "false"})
    chair = np.array([-2.35, 1.35, 0.0])
    for pos, size in (((0, 0, 0.22), (0.32, 0.32, 0.12)), ((-0.28, 0, 0.55), (0.06, 0.32, 0.32)),
                      ((0, 0.3, 0.4), (0.3, 0.05, 0.14)), ((0, -0.3, 0.4), (0.3, 0.05, 0.14))):
        G(world, type="box", size=_s(size), pos=_s(chair + pos), rgba="0.35 0.1 0.1 1")
    sleeper = ET.SubElement(world, "body", {"name": "sleeper", "mocap": "true", "pos": _s(chair)})
    skin, black, white, jeans = "0.86 0.67 0.55 1", "0.06 0.06 0.07 1", "0.92 0.92 0.9 1", "0.2 0.26 0.4 1"
    for k in range(6):
        G(sleeper, type="capsule", size="0.15", fromto=_s([-0.06 - 0.025 * k, 0, 0.42 + 0.075 * k,
                                                           -0.085 - 0.025 * k, 0, 0.47 + 0.075 * k]),
          rgba=black if k % 2 == 0 else white)
    G(sleeper, name="sleeper_head", type="ellipsoid", size="0.095 0.085 0.11", pos="-0.17 0 0.98", euler="0 -0.35 0.3",
      rgba=skin)
    G(sleeper, type="ellipsoid", size="0.1 0.09 0.065", pos="-0.16 0 1.06", euler="0 -0.35 0.3", rgba=black)
    G(sleeper, type="ellipsoid", size="0.099 0.089 0.03", pos="-0.16 0 0.99", euler="0 -0.35 0.3", rgba=black)
    for sy in (-1, 1):
        G(sleeper, type="capsule", size="0.065", fromto=_s([0.0, sy * 0.09, 0.38, 0.42, sy * 0.1, 0.4]), rgba=jeans)
        G(sleeper, type="capsule", size="0.055", fromto=_s([0.42, sy * 0.1, 0.4, 0.5, sy * 0.1, 0.06]), rgba=jeans)
        G(sleeper, type="box", size="0.12 0.055 0.04", pos=f"0.56 {sy * 0.1} 0.04", rgba=black)
        G(sleeper, type="capsule", size="0.045", fromto=_s([-0.1, sy * 0.19, 0.8, 0.12, sy * 0.24, 0.52]), rgba=black)
        G(sleeper, type="sphere", size="0.05", pos=f"0.16 {sy * 0.25} 0.5", rgba=skin)
    G(sleeper, type="ellipsoid", size="0.12 0.14 0.16", pos="0.08 -0.45 0.12", rgba="0.6 0.5 0.32 1")
    G(sleeper, name="snore", type="sphere", size="0.012", pos="-0.06 0.05 0.97", rgba=white)
    # ---- the pincer arms (mocap bodies posed from the head by their IK): upper arm, forearm, two fingers each
    for arm in ("g", "c"):
        up = ET.SubElement(world, "body", {"name": f"arm_{arm}_up", "mocap": "true", "pos": "0 0 -5"})
        G(up, type="sphere", size="0.0075", material="metal")
        G(up, type="capsule", size="0.0048", fromto=f"0 0 0 {ARM['upper']} 0 0", material="cleo:leg")
        fo = ET.SubElement(world, "body", {"name": f"arm_{arm}_fore", "mocap": "true", "pos": "0 0 -5"})
        G(fo, type="sphere", size="0.0065", material="metal")
        G(fo, type="capsule", size="0.004", fromto=f"0 0 0 {ARM['fore']} 0 0", material="cleo:leg")
        G(fo, type="box", size="0.006 0.007 0.006", pos=f"{ARM['fore']} 0 0", material="dark")
        for k in (1, 2):
            fi = ET.SubElement(world, "body", {"name": f"arm_{arm}_f{k}", "mocap": "true", "pos": "0 0 -5"})
            if arm == "g":                                                # the gripper: toothed fingers
                G(fi, type="box", size=f"{ARM['finger'] / 2} 0.0022 0.0042", pos=f"{ARM['finger'] / 2} 0 0",
                  material="dark")
                for j in range(3):
                    G(fi, type="box", size="0.0018 0.0012 0.0035",
                      pos=f"{0.006 + 0.0065 * j} {(1 if k == 1 else -1) * -0.0025} 0", material="metal")
            else:                                                         # the cutter: blades
                G(fi, type="box", size=f"{ARM['finger'] / 2} 0.0012 0.0045", pos=f"{ARM['finger'] / 2} 0 0",
                  material="blade")
                G(fi, type="box", size="0.004 0.0016 0.005", pos="0.002 0 0", material="dark")
    # ---- the pendrive (a mocap body: held, plugged in, held again)
    usb = ET.SubElement(world, "body", {"name": "usb", "mocap": "true", "pos": "0 0 -5"})
    G(usb, type="box", size="0.0225 0.008 0.0042", rgba="0.1 0.3 0.85 1")
    G(usb, type="box", size="0.006 0.006 0.0024", pos="0.0285 0 0", material="metal")
    G(usb, type="sphere", size="0.0022", pos="-0.012 0 0.0042", material="usb_led")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return model, {"robot": robot, "mr": mr}


# --------------------------------------------------------------------------------------------- the crawl
def zyx(R):
    """(yaw, pitch, roll) with R = Rz(yaw) · Ry(pitch) · Rx(roll)."""
    return (math.atan2(R[1, 0], R[0, 0]), math.asin(max(-1.0, min(1.0, -R[2, 0]))), math.atan2(R[2, 1], R[2, 2]))


def simulate(model, info, render=None):
    import mujoco
    mr = info["mr"]
    p = mr.CleopatraParams()
    data = mujoco.MjData(model)
    P, S, T, U, s_start = robot_path()
    C, _ = duct_centreline()
    Ctan, _ = _rmf(C)
    fa = model.joint("head:free").qposadr[0]
    legs = [(i, key) for i in (1, 2, 3) for key in mr.LEG_KEYS]
    jadr = {lg: [model.joint(j).qposadr[0] for j in mr.leg_joints(*lg)] for lg in legs}
    badr = {(i, a): model.joint(f"body {i}-{i + 1} {a}").qposadr[0] for i in (1, 2) for a in ("yaw", "pitch", "roll")}
    seg_id = {i: model.body(f"segment {i}").id for i in (1, 2, 3)}
    head = model.body("head").id
    usb = model.body("usb").mocapid[0]
    slats = {}                                                            # k: (mocap id, half width, height)
    for k in range(13):
        try:
            b_ = model.body(f"slat{k}")
        except KeyError:
            continue
        z = 0.022 * (k - 6)
        slats[k] = (b_.mocapid[0], math.sqrt(HOLE_R ** 2 - z * z), Z_IN + z)
    arm_ids = {(a_, k): model.body(f"arm_{a_}_{k}").mocapid[0] for a_ in ("g", "c") for k in ("up", "fore", "f1", "f2")}
    sleeper = model.body("sleeper").mocapid[0]
    bar = model.geom("bar").id
    snore = model.geom("snore").id
    led = model.material("usb_led").id
    sleeper_p0 = data.mocap_pos[sleeper].copy()
    hip = {key: np.array(mr.hip_position(p, key)) for key in mr.LEG_KEYS}
    nominal = {key: hip[key] + np.array([0.0, mr.LEG_SIDES[key][1] * p.foot_out, -p.foot_drop]) for key in mr.LEG_KEYS}
    f_len, t_len, r_pad = p.femur, p.tibia, p.foot_diameter / 2
    lim = {"yaw": mr.LIMIT_YAW, "pitch": mr.LIMIT_PITCH, "roll": mr.LIMIT_ROLL}
    order = ["front-left", "rear-right", "front-right", "rear-left"]
    groups = [[(i, order[(g + i) % 4]) for i in (1, 2, 3)] for g in range(4)]
    behind = {1: 0.160, 2: 0.160 + p.pitch, 3: 0.160 + 2 * p.pitch}

    def frame_at(sc):
        k = float(np.interp(sc, S, np.arange(len(S))))
        i = min(int(k), len(S) - 2)
        w = k - i
        pos = (1 - w) * P[i] + w * P[i + 1]
        t = (1 - w) * T[i] + w * T[i + 1]
        t /= np.linalg.norm(t)
        u = (1 - w) * U[i] + w * U[i + 1]
        u -= (u @ t) * t
        u /= np.linalg.norm(u)
        return pos, np.column_stack([t, np.cross(u, t), u])

    def place_body(sc):
        """The head on the path at arc sc, the segments' body joints from the path's frames behind it."""
        pos, R = frame_at(sc)
        data.qpos[fa:fa + 3] = pos
        data.qpos[fa + 3:fa + 7] = mat2quat(R)
        Rs = {1: R}
        for i in (2, 3):
            Rs[i] = frame_at(sc - behind[i])[1]
        for i in (1, 2):
            yw, pt, rl = zyx(Rs[i].T @ Rs[i + 1])
            for a, v in (("yaw", yw), ("pitch", pt), ("roll", rl)):
                data.qpos[badr[(i, a)]] = float(np.clip(v, -lim[a], lim[a]))
        mujoco.mj_kinematics(model, data)

    def seg_pose(i):
        return data.xpos[seg_id[i]].copy(), data.xmat[seg_id[i]].reshape(3, 3).copy()

    def duct_hit(x):
        """(closest centre-line point, tangent, radial distance) if x is inside the duct, else None."""
        d = np.linalg.norm(C - x, axis=1)
        k = int(np.argmin(d))
        c, t = C[k], Ctan[k]
        if (k == 0 and (x - c) @ t < -0.002) or (k == len(C) - 1 and (x - c) @ t > 0.002):
            return None
        r = x - c - ((x - c) @ t) * t
        return (c + ((x - c) @ t) * t, t, float(np.linalg.norm(r))) if np.linalg.norm(r) < R_DUCT + 0.01 else None

    def surface_z(x, y):
        dx0, dx1, dy0, dy1, dz = DESK
        if dx0 <= x <= dx1 and dy0 <= y <= dy1:
            return dz
        return 0.0

    def foothold(lg, pos_R):
        """Where leg lg wants its foot (pad centre) with its segment at pos_R, and the surface normal there."""
        key = lg[1]
        ps, Rs = pos_R
        nom = ps + Rs @ nominal[key]
        h_w = ps + Rs @ hip[key]
        dh = duct_hit(h_w)
        if dh is not None:                                                # on the pipe's wall, radially out
            c, t, _ = duct_hit(nom) or dh
            r = nom - c
            r -= (r @ t) * t
            if np.linalg.norm(r) < 1e-6:
                r = -Rs[:, 2]
            r /= np.linalg.norm(r)
            return c + r * (R_DUCT - r_pad), -r
        q = ps + Rs @ (hip[key] + np.array([0.0, mr.LEG_SIDES[key][1] * p.foot_out, 0.0]))
        z = surface_z(q[0], q[1])
        return np.array([q[0], q[1], z + r_pad]), np.array([0.0, 0.0, 1.0])

    def leg_ik(lg, foot_w):
        i, key = lg
        ps, Rs = seg_pose(i)
        v = Rs.T @ (foot_w - ps) - hip[key]
        sy = mr.LEG_SIDES[key][1]
        yaw = math.atan2(v[0], sy * v[1])
        r, h = math.hypot(v[0], v[1]), -v[2]
        d = min(math.hypot(r, h), f_len + t_len - 1e-4)
        ck = (d * d - f_len ** 2 - t_len ** 2) / (2 * f_len * t_len)
        kr = math.acos(max(-1.0, min(1.0, ck)))
        ph = math.atan2(h, r) - math.atan2(t_len * math.sin(kr), f_len + t_len * math.cos(kr))
        for a, val in zip(jadr[lg], (yaw, ph, kr)):
            data.qpos[a] = val
        for a, (lo, hi) in zip(jadr[lg], (mr.LEG_RANGES["hip_yaw"], mr.LEG_RANGES["hip_pitch"], mr.LEG_RANGES["knee"])):
            data.qpos[a] = float(np.clip(data.qpos[a], lo, hi))

    st = {"t": 0.0, "sc": s_start, "usb": "held", "copy": 0.0, "snips": {},
          "arm": {"g": (ARM_TUCK["g"].copy(), 0.0), "c": (ARM_TUCK["c"].copy(), 0.3)}, "grip": None}
    F, N = {}, {}
    place_body(st["sc"])
    for lg in legs:
        F[lg], N[lg] = foothold(lg, seg_pose(lg[0]))

    def arm_pose(arm, tip_h):
        """The arm's world points with the head where it is: (base, elbow, tip, forearm direction, lateral) from its
        pincer tip target in the head frame (2-link IK in the vertical plane through the target, the elbow up)."""
        hp, hR = data.xpos[head].copy(), data.xmat[head].reshape(3, 3)
        sy = 1.0 if arm == "g" else -1.0
        b = np.array([ARM["base_x"], sy * ARM["base_y"], ARM["base_z"]])
        v = np.asarray(tip_h) - b
        yaw = math.atan2(v[1], v[0])
        r, h = math.hypot(v[0], v[1]), v[2]
        L1, L2 = ARM["upper"], ARM["fore"]
        d = min(max(math.hypot(r, h), abs(L1 - L2) + 1e-3), L1 + L2 - 1e-4)
        e = math.acos(max(-1.0, min(1.0, (d * d - L1 ** 2 - L2 ** 2) / (2 * L1 * L2))))
        th = math.atan2(h, r) + math.atan2(L2 * math.sin(e), L1 + L2 * math.cos(e))
        el = b + L1 * np.array([math.cos(th) * math.cos(yaw), math.cos(th) * math.sin(yaw), math.sin(th)])
        dvec = b + d * np.array([math.cos(math.atan2(h, r)) * math.cos(yaw), math.cos(math.atan2(h, r)) * math.sin(yaw),
                                 math.sin(math.atan2(h, r))]) - el
        dvec /= np.linalg.norm(dvec)
        tip = el + L2 * dvec
        W = lambda x: hp + hR @ x
        lat = hR @ np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        return W(b), W(el), W(tip), hR @ dvec, lat

    def frame_x(d, lat):
        d = d / np.linalg.norm(d)
        lat = lat - (lat @ d) * d
        lat /= np.linalg.norm(lat)
        return np.column_stack([d, lat, np.cross(d, lat)])

    def props():
        t = st["t"]
        hR = data.xmat[head].reshape(3, 3)
        # the arms and their pincers
        for arm in ("g", "c"):
            tip_h, opening = st["arm"][arm]
            b, el, tip, d, lat = arm_pose(arm, tip_h)
            data.mocap_pos[arm_ids[(arm, "up")]] = b
            data.mocap_quat[arm_ids[(arm, "up")]] = mat2quat(frame_x(el - b, lat))
            data.mocap_pos[arm_ids[(arm, "fore")]] = el
            Rf = frame_x(d, lat)
            data.mocap_quat[arm_ids[(arm, "fore")]] = mat2quat(Rf)
            ang = 0.08 + 0.5 * opening
            for k, sg in (("f1", 1.0), ("f2", -1.0)):
                data.mocap_pos[arm_ids[(arm, k)]] = tip + Rf[:, 1] * sg * 0.004
                data.mocap_quat[arm_ids[(arm, k)]] = mat2quat(Rf @ rot_z(sg * ang))
            if arm == "g":
                st["grip"] = (tip + d * GRIP_IN, d)
        if st["usb"] == "held":                                           # in the gripper, pointing ahead
            g, _ = st["grip"]
            data.mocap_pos[usb] = g + hR[:, 0] * (USB_HALF - 0.006)
            data.mocap_quat[usb] = data.xquat[head]
        # the cut: a slat cut at one end sags a little; cut at both, it drops onto the desk and settles there
        for k, (mid, w, z0) in slats.items():
            gx0 = X_END - 0.006
            n_cut, t_free = st["snips"].get(k, (0, None))
            if t_free is None:
                tilt = 0.06 if n_cut == 1 else 0.0
                data.mocap_pos[mid] = [gx0, 0.0, z0 - 0.003 * (n_cut == 1)]
                data.mocap_quat[mid] = mat2quat(rot_y(-tilt * 0.3) @ np.eye(3))
                continue
            dt_ = t - t_free
            rest = DESK[4] + 0.0045 + 0.009 * (k % 3)
            zc = max(z0 - 0.5 * 9.81 * dt_ * dt_, rest)
            u = min(1.0, (z0 - zc) / max(z0 - rest, 1e-6))
            data.mocap_pos[mid] = [gx0 - 0.05 * u - 0.012 * (k % 4), 0.015 * ((k % 5) - 2) * u, zc]
            data.mocap_quat[mid] = mat2quat(rot_z(0.25 * ((k % 3) - 1) * u) @ rot_y(-1.5708 * u))
        # the sleeper breathes; the copy's progress bar; the pendrive's LED
        data.mocap_pos[sleeper] = sleeper_p0 + np.array([0, 0, 0.006 * math.sin(t * 2 * math.pi / 4.2)])
        model.geom_size[snore][0] = 0.006 + 0.01 * max(0.0, math.sin(t * 2 * math.pi / 4.2))
        fill = 0.1 * st["copy"]
        model.geom_size[bar][1] = max(fill, 0.0005)
        model.geom_pos[bar][1] = -0.1 + fill
        on = st["usb"] == "port" and st["copy"] < 1.0 and (t * 6) % 1.0 < 0.5
        model.mat_emission[led] = 1.0 if on else 0.0

    def tick(phase, caption, speed=1.0):
        props()
        mujoco.mj_forward(model, data)
        if render is not None:
            render(st, data, phase, caption, speed)

    def freegait(phase, caption, s_to, v=0.12, t_swing=0.16, reach=0.085, look=0.06, speed=1.0, settle=0.3):
        sgn = 1.0 if s_to >= st["sc"] else -1.0
        step = (speed / render.fps) if render is not None else 1.0 / 30
        swing, last, blocked, quiet = {}, None, 0.0, 0.0
        for _ in range(int(400 / step)):
            sc = st["sc"]
            place_body(sc)
            poses = {i: seg_pose(i) for i in (1, 2, 3)}
            if not swing:
                s_a = float(np.clip(sc + sgn * look, min(st["sc"], s_to), max(st["sc"], s_to)))
                place_body(s_a)
                poses_a = {i: seg_pose(i) for i in (1, 2, 3)}
                place_body(sc)
                want_a = {lg: foothold(lg, poses_a[lg[0]]) for lg in legs}
                need = {k: max(np.linalg.norm(F[lg] - want_a[lg][0]) for lg in g) for k, g in enumerate(groups)}
                k = max((k for k in need if k != last), key=lambda k: need[k])
                if need[k] > 0.015 or blocked > 0.0:
                    if blocked > 0.0 and need[k] <= 0.015:
                        k = max(range(4), key=lambda k: max(np.linalg.norm(F[lg] - foothold(lg, poses[lg[0]])[0])
                                                            for lg in groups[k]))
                        want_a = {lg: foothold(lg, poses[lg[0]]) for lg in legs}
                    last = k
                    for lg in groups[k]:
                        swing[lg] = (st["t"], F[lg].copy(), want_a[lg][0].copy(), N[lg].copy(), want_a[lg][1].copy())
                    quiet = 0.0
                elif abs(s_to - sc) < 1e-6:
                    quiet += step
                    if quiet >= settle:
                        break
            # the body advances while every planted foot reaches
            if abs(s_to - sc) > 1e-6:
                s_try = sc + sgn * min(v * step, abs(s_to - sc))
                place_body(s_try)
                ok = all(np.linalg.norm(F[lg] - foothold(lg, seg_pose(lg[0]))[0]) < reach for lg in legs
                         if lg not in swing)
                if ok or blocked > 0.4:
                    st["sc"], blocked = s_try, 0.0
                else:
                    blocked += step
            place_body(st["sc"])
            feet = {}
            for lg in legs:
                if lg in swing:
                    t0, a, b, na, nb = swing[lg]
                    u = min((st["t"] + step - t0) / t_swing, 1.0)
                    n = na + nb
                    n /= np.linalg.norm(n) + 1e-9
                    feet[lg] = a + sm(u) * (b - a) + n * 0.03 * math.sin(math.pi * u)
                    if u >= 1.0:
                        F[lg], N[lg] = b, nb
                        del swing[lg]
                else:
                    feet[lg] = F[lg]
            for lg in legs:
                leg_ik(lg, feet[lg])
            st["t"] += step
            tick(phase, caption, speed)
        return st

    def hold(phase, caption, dur, fn=None, speed=1.0):
        step = (speed / render.fps) if render is not None else 1.0 / 30
        t0 = st["t"]
        while st["t"] < t0 + dur - 1e-9:
            if fn:
                fn((st["t"] - t0) / dur)
            st["t"] += step
            place_body(st["sc"])
            for lg in legs:
                leg_ik(lg, F[lg])
            tick(phase, caption, speed)

    s_mouth = float(S[np.argmin(np.linalg.norm(P - C[0], axis=1))])
    s_end = float(S[-1])
    hold("yard", "", 1.5)
    freegait("yard", "ACROSS THE YARD TO THE AIR-CONDITIONING INTAKE", s_mouth - 0.05, v=0.16)
    freegait("duct", "INTO THE DUCT — TWELVE FEET ON THE PIPE WALL", s_mouth + 1.15, v=0.13)
    s_cut = float(S[np.argmin(np.abs(P[:, 0] - (X_END + 0.07)) + 10 * (P[:, 2] < Z_IN - 0.05))])
    freegait("riser", "UP THE RISER AND THROUGH THE WALL", s_cut, v=0.11)

    def head_frame(w):
        hp, hR = data.xpos[head].copy(), data.xmat[head].reshape(3, 3)
        return hR.T @ (np.asarray(w) - hp)

    def move_arm(arm, phase, caption, dur, target_h, opening=None, lift=0.03, speed=1.0):
        a0, o0 = st["arm"][arm]
        a0 = a0.copy()
        o1 = o0 if opening is None else opening

        def fn(u):
            w = sm(u)
            st["arm"][arm] = (a0 + w * (np.asarray(target_h) - a0) + np.array([0, 0, lift * math.sin(math.pi * u)]),
                              o0 + w * (o1 - o0))
        hold(phase, caption, dur * speed, fn, speed=speed)
        st["arm"][arm] = (np.asarray(target_h, float), o1)

    # ---- the cut: the cutter snips each slat where the hole's edge crosses it, left end then right end; a slat cut
    # at both ends drops onto the desk. Top to bottom, so a falling slat clears the ones below
    gxc = X_END - 0.004
    cap = "THE CUTTER SNIPS THE GRILLE'S SLATS, ONE BY ONE"
    for n, k in enumerate(sorted(slats, reverse=True)):
        _, w, z0 = slats[k]
        speed = 1.0 if n < 3 else 3.0
        for side, sy in enumerate((1.0, -1.0)):
            pt = head_frame(np.array([gxc, sy * (w + 0.002), z0]))
            move_arm("c", "cut", cap, 0.45, pt + [0.012, 0, 0], opening=1.0, lift=0.01, speed=speed)   # open, reach
            move_arm("c", "cut", cap, 0.15, pt, lift=0.0, speed=speed)                                # in round it
            move_arm("c", "cut", cap, 0.12, pt, opening=0.0, lift=0.0, speed=speed)                   # snip
            st["snips"][k] = (side + 1, st["t"] if side == 1 else None)
    hold("cut", "THE SLATS DROP ONTO THE DESK", 0.8)
    move_arm("c", "cut", "THE SLATS DROP ONTO THE DESK", 0.8, ARM_TUCK["c"], opening=0.3)
    freegait("vent", "THROUGH THE HOLE, DOWN ONTO THE DESK", s_end - 0.12, v=0.09)
    freegait("plug", "TO THE LAPTOP", s_end, v=0.05)
    # ---- the plug: the gripper reaches down to the back port, lets go, comes back for it after the copy
    hR = data.xmat[head].reshape(3, 3)
    hx = hR[:, 0]
    usb_c = port_point() - hx * (USB_PLUG - 0.010)                        # the plug 10 mm in
    tip_w = usb_c - hx * (USB_HALF - 0.006) - hx * GRIP_IN
    move_arm("g", "plug", "THE GRIPPER PLUGS THE PENDRIVE IN", 1.8, head_frame(tip_w), lift=0.04)
    st["usb"] = "port"
    move_arm("g", "plug", "THE GRIPPER PLUGS THE PENDRIVE IN", 0.4, st["arm"]["g"][0], opening=1.0, lift=0.0)
    move_arm("g", "copy", "COPYING", 1.0, ARM_TUCK["g"], lift=0.0)
    hold("copy", "COPYING", 6.5, fn=lambda u: st.update(copy=min(1.0, u * 1.1)))
    move_arm("g", "plug", "COPY DONE — UNPLUG", 1.0, head_frame(tip_w), lift=0.0)
    move_arm("g", "plug", "COPY DONE — UNPLUG", 0.4, head_frame(tip_w), opening=0.0, lift=0.0)
    st["usb"] = "held"
    move_arm("g", "plug", "COPY DONE — UNPLUG", 1.2, ARM_TUCK["g"], lift=0.02)
    freegait("retreat", "BACK OUT THE WAY IT CAME", s_mouth - 0.4, v=0.13, speed=SPEEDUP)
    hold("out", "", 2.0)
    return st


# --------------------------------------------------------------------------------------------- the video
def run(out, width=1280, height=720, fps=30, cache=None, frames_dir=None, text=False):
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    cache = Path(cache or ROOT / "runs" / "cleopatra" / "mission_cache")
    cache.mkdir(parents=True, exist_ok=True)
    model, info = build(width, height, cache)
    renderer = mujoco.Renderer(model, height, width)
    cam = tm.Cam(45.0, width, height)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
    opt.sitegroup[:] = [0] * 6
    frames_dir = Path(frames_dir or (cache / "frames"))
    shutil.rmtree(frames_dir, ignore_errors=True)
    frames_dir.mkdir(parents=True)
    try:
        import matplotlib
        fp = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans-Bold.ttf")
        f_cap = ImageFont.truetype(fp, 22)
    except (ImportError, OSError):                                        # no matplotlib: Pillow's own font
        f_cap = ImageFont.load_default()
    head = model.body("head").id
    segs = [model.body(f"segment {i}").id for i in (1, 2, 3)]
    bar = int(height * 0.1)
    shot = {"name": None}
    count = {"n": 0}

    def render(st, data, phase, caption, speed):
        h = data.xpos[head].copy()
        mid = data.xpos[segs[1]].copy()
        if shot["name"] != phase:
            cam.cut()
            shot["name"] = phase
        if phase == "yard":
            cam.set(0.5 * (h + mid) + [0, 0, 0.05], 1.6, 205, -22, 0.08)
        elif phase == "duct":
            cam.set(mid + [0.0, 0, 0.02], 1.15, 250, -18, 0.08)
        elif phase == "riser":
            cam.set(mid + [0.0, 0, 0.05], 1.25, 235, -6, 0.08)
        elif phase == "cut":
            cam.set(np.array([X_END - 0.03, 0.0, Z_IN - 0.02]), 0.62, 80, -12, 0.1)
        elif phase == "vent":
            cam.set(h + [-0.05, 0, 0.0], 0.95, 20, -16, 0.08)
        elif phase == "plug":
            cam.set(0.5 * (port_point() + h) + [0, 0, 0.02], 0.62, 90, -12, 0.08)
        elif phase == "copy":
            cam.set(np.array([-1.3, 0.5, 0.75]), 1.5, 126, -18, 0.05)
        elif phase == "retreat":
            cam.set(0.5 * (h + mid), 1.3, 270, -14, 0.06)
        else:
            cam.set(np.array([X_MOUTH, 0.0, 0.15]), 2.2, 210, -16, 0.05)
        renderer.update_scene(data, camera=cam.c, scene_option=opt)
        img = Image.fromarray(renderer.render())
        dr = ImageDraw.Draw(img)
        dr.rectangle([0, 0, width, bar], fill=(0, 0, 0))
        dr.rectangle([0, height - bar, width, height], fill=(0, 0, 0))
        if text and caption:
            dr.text((24, height - bar + bar / 2 - 12), caption + (f"   ▶▶ ×{speed:g}" if speed > 1 else ""),
                    font=f_cap, fill=(240, 240, 235))
            dr.text((24, bar / 2 - 12), "OPERATION PIPE ENTRY", font=f_cap, fill=(255, 196, 64))
        img.save(frames_dir / f"f_{count['n']:05d}.jpg", quality=92)
        count["n"] += 1
        if count["n"] % 150 == 0:
            print(f"  {count['n']} frames, t = {st['t']:6.2f} s, {phase}", flush=True)

    render.fps = fps
    st = simulate(model, info, render=render)
    print(f"cleopatra_mission: {count['n']} frames, {st['t']:.1f} s of story", flush=True)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps), "-i", str(frames_dir / "f_%05d.jpg"),
                        "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags",
                        "+faststart", str(out)], check=True)
        print("cleopatra_mission:", out)
    else:
        print("cleopatra_mission: no ffmpeg — the frames are in", frames_dir)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cleopatra.sh mission", description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(ROOT / "runs" / "cleopatra" / "cleopatra_mission.mp4"))
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--text", action="store_true", help="captions in the letterbox")
    a = ap.parse_args(argv)
    run(a.out, a.width, a.height, a.fps, text=a.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
