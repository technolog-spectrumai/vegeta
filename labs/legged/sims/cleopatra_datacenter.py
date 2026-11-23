"""Cleopatra, "Operation Pipe Entry" (``./cleopatra.sh mission``): a night-time MuJoCo short. Cleopatra — the
three-segment, twelve-legged Myropod (``designs/myropod_robot.py``, notebook 18), 6.2 kg, here with two small pincer
arms under its head (a gripper, which carries the pendrive, and a cutter, as TARANTULA has) and a camera-and-lidar
turret on its head — crosses the street outside the house of Operation Silent Drop to the intake of the bandits' air duct (a 260 mm pipe,
on the ground along the building, through its wall) carrying a pendrive in its gripper and a fuse-timed gift clamped
under its tail. At the round return vent inside the bandits' datacenter the cutter snips the grille's slats free one
by one; Cleopatra climbs out, walks down the cold aisle between the racks and over the cable bundles to the last rack,
and the gripper plugs the pendrive into the server's front port. After the copy it unplugs, leaves the gift by the
server (its fuse timer blinking) and backs out the way it came (time-lapsed). Back in the datacenter: KABOOM —
confetti, smoke, and the dozing guard soot-black with his hair on end.

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
R_BEND = 0.45                                                             # its bends' radius (the body joints: ~21°)
X_MOUTH, X_END, Z_IN = 1.70, -0.42, R_DUCT                                # the intake, the indoor vent, the axis
AISLE_Y = -1.2                                                            # the datacenter's cold aisle (its centre)
WALL = (-0.25, 0.0)                                                       # the building's wall (x from, to)
RACK_W, RACK_D, RACK_H = 0.6, 1.0, 2.0                                    # a server rack
RACK_X0 = -0.9                                                            # the first rack's near edge
N_RACKS = 4                                                               # racks per row
FRONT_L, FRONT_R = AISLE_Y - 0.45, AISLE_Y + 0.45                         # the two rows' front faces
TARGET_X = RACK_X0 - 3.5 * RACK_W                                         # the target rack's centre (the last)
GIFT = (0.03, 0.025, 0.02)                                                # the gift's half sizes [m]
GIFT_UNDER = 0.035 + 0.022                                                # carried under segment 3's belly
N_CONFETTI, N_SMOKE = 180, 10
PORT_Z = 0.09                                                             # its bottom server's front USB port
CABLES = ((-0.78, 0.022, (0.1, 0.25, 0.75)), (-1.12, 0.018, (0.9, 0.75, 0.15)), (-1.42, 0.024, (0.08, 0.08, 0.09)),
          (-1.66, 0.016, (0.85, 0.2, 0.2)))                                # bundles across the aisle: x, radius, colour
SPEEDUP = 5.0                                                             # the retreat, time-lapsed
COLORS = {"shell": (0.36, 0.38, 0.41), "head": (0.27, 0.29, 0.32), "leg": (0.14, 0.15, 0.16),
          "pad": (0.06, 0.06, 0.06)}


STANCE = (0.065, 0.095)                                                   # the crawl's stance: foot out, down [m]


def hip_h():
    """The body's height over the ground in the crawl's stance (a crouch: the legs keep ~20 mm of reach in hand)."""
    import myropod_robot as mr
    return STANCE[1] + mr.cleopatra2_params().foot_diameter / 2


# --------------------------------------------------------------------------------------------- the CAD as meshes
def cleo_meshes(cache: Path):
    """{part: stl path} — Cleopatra v2's CAD (``myropod.Myropod`` at ``myropod_robot.CLEO2_MM``) cut onto the Chiron
    links, each in its link frame: the head with its turret and a segment (centred at hip height), the femur and the
    hooked tibia of a left and a right leg (straight out sideways, as the links are at q = 0), and the arms' parts
    (the upper arm and forearm along +x, a flat gripper jaw and a scissor blade, each in both hands)."""
    import myropod
    import myropod_robot as mr
    key = hashlib.sha1((LAB / "designs" / "myropod.py").read_bytes() + repr(mr.CLEO2_MM).encode()).hexdigest()[:12]
    out = cache / f"cleo2_{key}"
    index = out / "index.json"
    if index.exists():
        return json.loads(index.read_text())
    out.mkdir(parents=True, exist_ok=True)
    print("cleopatra_datacenter: tessellating the CAD (once; cached in", out, ")", flush=True)
    d = myropod.Myropod()
    p = d.resolve(**mr.CLEO2_MM)
    zc = p["seg_height"] / 2
    femur, tibia = d._femur(p), d._tibia(p)
    jaw, blade = d._gripper_jaw(p), d._cutter_blade(p)
    parts = {"head": d._head(p).translate((0, 0, -zc)), "turret": d._turret(p).translate((0, 0, p["seg_height"] - zc)),
             "segment": d._segment(p).translate((0, 0, -zc)),
             "femur_l": femur, "tibia_l": tibia, "femur_r": femur.mirror("XZ"), "tibia_r": tibia.mirror("XZ"),
             "arm_upper": d._arm_bar(p, p["arm_upper"], p["arm_diameter"]),
             "arm_fore": d._arm_bar(p, p["arm_fore"], p["arm_diameter"] * 0.85),
             "jaw_1": jaw, "jaw_2": jaw.mirror("XZ"), "blade_1": blade, "blade_2": blade.mirror("XZ")}
    idx = {}
    for name, w in parts.items():
        Vs, Fs, n0 = [], [], 0
        for sol in w.vals():
            vs, fs = sol.tessellate(0.2, 0.25)
            Vs.append(np.array([v.toTuple() for v in vs]) * 1e-3)
            Fs.append(np.array(fs) + n0)
            n0 += len(vs)
        path = out / f"{name}.stl"
        tv._stl(path, np.vstack(Vs), np.vstack(Fs))
        idx[name] = str(path)
    idx["arm"] = {k: p[k] * 1e-3 for k in ("arm_upper", "arm_fore", "jaw_length")}
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
    """The duct's centre line, on the ground all the way: from the intake towards the building, a bend to run along
    its wall, a bend back in, and through the wall to the datacenter's floor-level return vent."""
    a = np.linspace(0, math.pi / 2, 24)
    x1, y0 = 1.05, 0.0                                                    # a straight run between the two bends
    c1 = np.array([x1, y0 - R_BEND, R_DUCT])
    c2 = np.array([x1 - 2 * R_BEND, AISLE_Y + R_BEND, R_DUCT])           # (the run along the wall: c1 → c2)
    pts = [np.array([X_MOUTH, y0, R_DUCT]), np.array([x1, y0, R_DUCT])]
    pts += [c1 + R_BEND * np.array([-math.sin(t), math.cos(t), 0.0]) for t in a[1:]]
    pts += [c2 + R_BEND * np.array([math.cos(t), -math.sin(t), 0.0]) for t in a]
    pts.append(np.array([X_END, AISLE_Y, R_DUCT]))
    P, s = _resample(np.array(pts), 0.004)
    return P, s


def robot_path():
    """The head's path (centre at hip height): a lead-in behind the start (for the trailing segments), the yard, in at
    the intake, the duct's centre line, out of the vent down the cold aisle over the cables, and a turn in to the
    target rack's front, stopping where the gripper reaches its port."""
    H = hip_h()
    C, _ = duct_centreline()
    start = np.array([3.55, 1.0, H])
    pre = start + np.array([0.6, 0.3, 0.0])
    a0 = np.array([X_MOUTH + 0.35, 0.0, H])
    yard = [pre, start]
    for u in np.linspace(0, 1, 60)[1:]:                                   # a cubic from the start to the intake
        b = (1 - u) ** 3 * start + 3 * (1 - u) ** 2 * u * (start + np.array([-0.7, -0.35, 0])) \
            + 3 * (1 - u) * u ** 2 * (a0 + np.array([0.6, 0, 0])) + u ** 3 * a0
        yard.append(b)
    lead = [a0 + (C[0] - a0) * u for u in np.linspace(0, 1, 20)[1:]]      # down into the duct's axis
    r_t = 0.25
    port = port_point()
    head_y_end = port[1] + PLUG_REACH                                     # the gripper reaches the rest of the way
    x_turn = TARGET_X + r_t
    room = [np.array([X_END - 0.02, AISLE_Y, H]), np.array([x_turn, AISLE_Y, H])]
    ct = np.array([x_turn, AISLE_Y - r_t, H])
    room += [ct + r_t * np.array([-math.sin(t), math.cos(t), 0.0]) for t in np.linspace(0, math.pi / 2, 30)[1:]]
    room.append(np.array([TARGET_X, head_y_end, H]))
    P, s = _resample(np.array(yard + lead + list(C[1:]) + room), 0.004)
    T, U = _rmf(P)
    s_start = float(np.linalg.norm(pre[:2] - start[:2]))
    return P, s, T, U, s_start


# ---- Cleopatra v2's pincer arms (``myropod.Myropod`` ``arms``), head frame [m]: a shoulder (yaw + pitch) under the
# head's front, an upper arm, an elbow, a forearm and a pincer — the flat gripper on the left (it carries the pendrive),
# the scissor cutter on the right
ARM = {"base_x": 0.055, "base_y": 0.033, "base_z": -0.042, "upper": 0.115, "fore": 0.110, "finger": 0.022}
#: (``myropod_robot.ARM_V2``: the robot's jointed arms; the video poses their joints)
ARM_TUCK = {"g": np.array([0.11, 0.035, -0.045]), "c": np.array([0.105, -0.035, -0.045])}
HOLE_R = R_DUCT + 0.004                                                   # the hole the cutter cuts: the full bore
CARRY_AHEAD, CARRY_DROP = 0.20, 0.035                                    # the cutter carried at the path 0.2 m ahead
USB_HALF = 0.0225                                                         # the pendrive body's half length
USB_PLUG = USB_HALF + 0.012                                               # its centre to the plug's tip
GRIP_IN = 0.010                                                           # the gripper's tip to its grip point
PLUG_REACH = 0.19                                                         # the head stops this far from the port


def port_point():
    """The target server's front USB port (the bottom server in the third rack of the left row)."""
    return np.array([TARGET_X, FRONT_L + 0.002, PORT_Z])


# --------------------------------------------------------------------------------------------- the scene
def build(width, height, cache: Path):
    import mujoco
    import myropod_robot as mr
    from vegeta.chiron.robot import SimOptions
    robot = mr.cleopatra2("spring_damper")
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
    A("material", name="gift_led", rgba="1 0.12 0.08 1", emission="0")
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
        if name != "arm":
            A("mesh", name=f"m:{name}", file=path)
    A("texture", name="tiles", type="2d", builtin="checker", rgb1="0.42 0.43 0.45", rgb2="0.36 0.37 0.39",
      width="512", height="512")
    A("material", name="tiles", texture="tiles", texrepeat="1.67 1.67", texuniform="true", specular="0.25")
    A("material", name="dcwall", rgba="0.3 0.31 0.33 1", specular="0.05")
    A("material", name="rack", rgba="0.07 0.07 0.08 1", specular="0.4", shininess="0.4")
    A("material", name="server", rgba="0.16 0.16 0.18 1", specular="0.5", shininess="0.5")
    A("material", name="led_g", rgba="0.2 1 0.35 1", emission="1")
    A("material", name="led_b", rgba="0.25 0.55 1 1", emission="1")
    A("material", name="led_a", rgba="1 0.6 0.1 1", emission="1")
    A("material", name="striplight", rgba="0.75 0.85 1 1", emission="1")
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
    G(bodies["head"], type="mesh", mesh="m:turret", material="dark", mass="0")                    # the turret (CAD)
    G(bodies["head"], type="cylinder", size="0.0065 0.0012", pos="0.0345 0 0.05", zaxis="1 0 0", material="lens",
      mass="0")
    G(bodies["head"], type="cylinder", size="0.0132 0.0015", pos="-0.008 0 0.064", material="lidar", mass="0")
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
    for sy in (-1, 1):
        G(bodies["head"], type="cylinder", size="0.0045 0.002", pos=f"0.066 {sy * 0.031} 0", zaxis="1 0 0",
          material="headlamp", mass="0")
    ET.SubElement(bodies["head"], "light", {"name": "headlamp", "pos": "0.07 0 0.005", "dir": "1 0 -0.08",
                                            "diffuse": "0.9 0.88 0.8", "specular": "0.2 0.2 0.2", "cutoff": "38",
                                            "exponent": "6", "attenuation": "1 0 2", "castshadow": "false"})
    # ---- the building: the wall (brick outside), the datacenter inside, the roof
    x0, x1 = WALL
    G(world, name="wall_out", type="box", size="0.004 3 1.2", pos=f"{x1 - 0.004} 0 1.2", material="brick")
    G(world, name="wall", type="box", size=f"{(x1 - x0) / 2 - 0.004} 3 1.2", pos=f"{(x0 + x1) / 2 - 0.002} 0 1.2",
      material="dcwall")
    G(world, type="box", size="0.05 3 1.2", pos="-4.05 0 1.2", material="dcwall")
    for sy in (-1, 1):
        G(world, type="box", size="1.92 0.05 1.2", pos=f"-2.08 {sy * 3.0} 1.2", material="dcwall")
    G(world, type="box", size="2.1 3.05 0.045", pos="-2.0 0 2.355", material="dark")
    G(world, type="box", size="0.27 3.05 0.07", pos=f"{x1 - 0.03} 0 2.43", material="stone")   # the coping
    G(world, type="box", size="0.012 3.05 0.008", pos=f"{x1 + 0.228} 0 2.352", material="stone")
    G(world, type="box", size="1.9 2.95 0.003", pos="-2.13 0 0.0", material="tiles")         # raised-floor tiles
    for k, y in enumerate((-2.3, 1.25)):                                  # two windows, one lit (as before)
        G(world, type="box", size="0.004 0.26 0.36", pos=f"{x1 + 0.003} {y} 1.35",
          material="window" if k else "dark")
        G(world, type="box", size="0.04 0.3 0.02", pos=f"{x1 + 0.04} {y} 0.91", material="stone")
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
    gx = X_END - 0.006                                                    # the vent's grille: round, a ring and slats
    for k in range(32):                                                   # the ring (the duct's end flange)
        a_ = 2 * math.pi * k / 32
        G(world, type="box", size="0.005 0.0115 0.008",
          pos=f"{gx} {AISLE_Y + (HOLE_R + 0.009) * math.cos(a_):.4f} {Z_IN + (HOLE_R + 0.009) * math.sin(a_):.4f}",
          euler=f"{a_ + math.pi / 2:.4f} 0 0", rgba="0.75 0.76 0.78 1")
    for k in range(-5, 6):                                                # the slats, across the circle
        z = 0.022 * k
        w = math.sqrt(max(HOLE_R ** 2 - z * z, 0.0))
        if w > 0.03:                                                      # cut free at both ends, they drop
            sl = ET.SubElement(world, "body", {"name": f"slat{k + 6}", "mocap": "true",
                                               "pos": f"{gx} {AISLE_Y} {Z_IN + z:.4f}"})
            G(sl, type="box", size=f"0.004 {w:.4f} 0.0045", rgba="0.7 0.71 0.74 1")
    # ---- the yard: the AC unit by the duct
    G(world, type="box", size="0.3 0.25 0.38", pos="0.95 1.35 0.38", material="metal")
    G(world, type="cylinder", size="0.18 0.005", pos="1.26 1.35 0.42", zaxis="1 0 0", material="dark")
    # ---- the datacenter: two rows of racks either side of the cold aisle, servers with blinking LEDs, cable bundles
    # across the aisle's floor, cable trays overhead, blue strip lights; the target rack's status panel
    rng = np.random.default_rng(3)
    for row, front in ((-1, FRONT_L), (1, FRONT_R)):
        yc = front + row * RACK_D / 2
        for r in range(N_RACKS):
            xc = RACK_X0 - (r + 0.5) * RACK_W
            G(world, name=f"rack{row}_{r}", type="box", size=f"{RACK_W / 2 - 0.01} {RACK_D / 2} {RACK_H / 2}",
              pos=f"{xc} {yc} {RACK_H / 2}", material="rack")
            for u in range(0, 38, 2):                                     # 2U servers, a status LED each
                z = 0.05 + 0.089 * u / 2 + 0.04
                if z > RACK_H - 0.08:
                    break
                fy = front - row * 0.003
                G(world, type="box", size=f"{RACK_W / 2 - 0.035} 0.003 0.036", pos=f"{xc} {fy} {z}", material="server")
                led = ("led_g", "led_b", "led_g", "led_a")[int(rng.integers(0, 4))]
                G(world, name=f"led{row}_{r}_{u}", type="box", size="0.004 0.002 0.003",
                  pos=f"{xc + RACK_W / 2 - 0.06} {fy - row * 0.002} {z + 0.015}", material=led)
                G(world, type="box", size="0.012 0.002 0.002", pos=f"{xc - RACK_W / 2 + 0.08} {fy - row * 0.002} {z}",
                  material="led_b")
    port = port_point()
    G(world, type="box", size="0.0075 0.002 0.0035", pos=_s(port - [0, 0.0015, 0]), material="dark")
    scr = ET.SubElement(world, "body", {"name": "screen", "pos": f"{TARGET_X} {FRONT_L + 0.008} 1.2",
                                        "euler": "0 0 -1.5708"})            # the target rack's status panel
    G(scr, type="box", size="0.005 0.165 0.11", pos="0.004 0 0", material="metal", rgba="0.25 0.26 0.28 1")
    G(scr, name="display", type="box", size="0.0008 0.152 0.097", pos="-0.0015 0 0", material="screen")
    G(scr, name="bar_bg", type="box", size="0.0008 0.1 0.008", pos="-0.0025 0 -0.02", material="dark")
    G(scr, name="bar", type="box", size="0.0008 0.0005 0.0065", pos="-0.0032 -0.1 -0.02", material="bar")
    for x_c, r_c, col in CABLES:                                          # cable bundles across the aisle
        for j in range(3):
            G(world, type="capsule", size=f"{r_c * 0.55:.4f}",
              fromto=_s([x_c + 0.012 * (j - 1), FRONT_L - 0.05, r_c * 0.55, x_c + 0.012 * (j - 1) + 0.02 * (j - 1),
                         FRONT_R + 0.05, r_c * 0.55]), rgba=f"{col[0]} {col[1]} {col[2]} 1")
    for yy in (FRONT_R - 0.06,):                                          # a bundle along the right row's feet
        G(world, type="capsule", size="0.018", fromto=f"{RACK_X0} {yy} 0.018 {RACK_X0 - N_RACKS * RACK_W} {yy} 0.018",
          rgba="0.15 0.2 0.6 1")
    for row, front in ((-1, FRONT_L), (1, FRONT_R)):                      # the overhead trays and strip lights
        G(world, type="box", size=f"{N_RACKS / 2 * RACK_W} 0.15 0.02", pos=f"{RACK_X0 - N_RACKS / 2 * RACK_W} {front + row * 0.4} 2.2",
          material="metal")
    G(world, type="box", size=f"{N_RACKS / 2 * RACK_W} 0.025 0.008", pos=f"{RACK_X0 - N_RACKS / 2 * RACK_W} {AISLE_Y} 2.33",
      material="striplight")
    for k in range(3):
        ET.SubElement(world, "light", {"name": f"aisle{k}", "pos": f"{RACK_X0 - (0.5 + 1.0 * k) * RACK_W * 1.6} {AISLE_Y} 2.28",
                                       "dir": "0 0 -1", "diffuse": "0.32 0.38 0.5", "specular": "0.1 0.1 0.12",
                                       "cutoff": "60", "exponent": "3", "attenuation": "1 0 0.15",
                                       "castshadow": "true" if k == 1 else "false"})
    ET.SubElement(world, "light", {"name": "moon", "directional": "true", "pos": "5 -3 6", "dir": "-0.5 0.35 -0.8",
                                   "diffuse": "0.28 0.32 0.45", "specular": "0.05 0.05 0.08", "castshadow": "true"})
    chair = np.array([RACK_X0 - N_RACKS * RACK_W - 0.45, AISLE_Y, 0.0])          # the guard, at the aisle's end
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
    # ---- the jointed arms (the robot's links): their CAD on them
    for arm, tool in (("gripper", "jaw"), ("cutter", "blade")):
        G(bodies[f"{arm} upper arm"], type="mesh", mesh="m:arm_upper", material="cleo:leg", mass="0")
        G(bodies[f"{arm} forearm"], type="mesh", mesh="m:arm_fore", material="cleo:leg", mass="0")
        for k in (1, 2):
            G(bodies[f"{arm} jaw {k}"], type="mesh", mesh=f"m:{tool}_{k}", mass="0",
              material="dark" if arm == "gripper" else "blade")
    # ---- the gift (a fuse-timed box: carried under the tail, left by the server), the blast's props
    gb = ET.SubElement(world, "body", {"name": "gift", "mocap": "true", "pos": "0 0 -5"})
    gx_, gy_, gz_ = GIFT
    G(gb, name="gift", type="box", size=f"{gx_} {gy_} {gz_}", rgba="0.78 0.06 0.08 1")
    G(gb, type="box", size=f"{gx_ + 0.0006} 0.005 {gz_ + 0.0006}", rgba="0.98 0.78 0.2 1")
    G(gb, type="box", size=f"0.005 {gy_ + 0.0006} {gz_ + 0.0006}", rgba="0.98 0.78 0.2 1")
    for sy_ in (-1, 1):
        G(gb, type="ellipsoid", size="0.007 0.011 0.006", pos=f"0 {sy_ * 0.009} {gz_ + 0.004}", rgba="0.98 0.78 0.2 1")
    G(gb, type="sphere", size="0.0035", pos=f"{gx_ - 0.008} {gy_ - 0.006} {gz_ + 0.001}", material="gift_led")
    fb = ET.SubElement(world, "body", {"name": "flashball", "mocap": "true", "pos": "0 0 -5"})
    G(fb, name="flashball", type="sphere", size="0.01", rgba="1 0.92 0.55 0")
    rng2 = np.random.default_rng(7)
    for k in range(N_SMOKE):
        b_ = ET.SubElement(world, "body", {"name": f"smoke{k}", "mocap": "true", "pos": "0 0 -5"})
        G(b_, name=f"smoke{k}", type="sphere", size="0.05", rgba="0.3 0.3 0.3 0")
    palette = [(1, 0.2, 0.3), (1, 0.8, 0.1), (0.2, 0.75, 1), (0.4, 1, 0.35), (1, 0.45, 0.9), (1, 1, 1), (1, 0.55, 0.1)]
    for k in range(N_CONFETTI):
        b_ = ET.SubElement(world, "body", {"name": f"conf{k}", "mocap": "true", "pos": "0 0 -5"})
        c_ = palette[k % len(palette)]
        w_ = 0.006 + 0.006 * rng2.random()
        G(b_, type="box", size=f"{w_:.4f} {w_ * 0.6:.4f} 0.0004", rgba=f"{c_[0]} {c_[1]} {c_[2]} 1")
    ET.SubElement(world, "light", {"name": "flash", "pos": f"{TARGET_X} {AISLE_Y} 0.6", "dir": "0 0 -1",
                                   "diffuse": "0 0 0", "specular": "0 0 0", "cutoff": "90", "castshadow": "false",
                                   "attenuation": "1 0 0.3"})
    for k in range(7):                                                    # the guard's hair, after the blast
        a_ = -1.2 + 0.4 * k
        G(sleeper, name=f"hair{k}", type="capsule", size="0.006 0.04",
          pos=f"{-0.17 + 0.02 * math.sin(a_):.3f} {0.06 * math.sin(a_):.3f} {1.1 + 0.02 * math.cos(a_):.3f}",
          euler=f"{a_ * 0.8:.3f} {0.3 * math.cos(a_):.3f} 0", rgba="0.04 0.04 0.04 0")
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
    p = mr.cleopatra2_params()                                            # v2: the shorter legs
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
    for k in range(1, 13):
        try:
            b_ = model.body(f"slat{k}")
        except KeyError:
            continue
        z = 0.022 * (k - 6)
        slats[k] = (b_.mocapid[0], math.sqrt(HOLE_R ** 2 - z * z), Z_IN + z)                     # cut at the ring
    arm_q = {a_: [model.joint(j).qposadr[0] for j in mr.arm_joints(name)] for a_, name in (("g", "gripper"),
                                                                                               ("c", "cutter"))}
    sleeper = model.body("sleeper").mocapid[0]
    bar = model.geom("bar").id
    snore = model.geom("snore").id
    led = model.material("usb_led").id
    rack_leds = [model.material(n).id for n in ("led_g", "led_b", "led_a")]
    sleeper_p0 = data.mocap_pos[sleeper].copy()
    hip = {key: np.array(mr.hip_position(p, key)) for key in mr.LEG_KEYS}
    nominal = {key: hip[key] + np.array([0.0, mr.LEG_SIDES[key][1] * STANCE[0], -STANCE[1]]) for key in mr.LEG_KEYS}
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

    def behind_at(q, s_from, dist):
        """The arc s < s_from where the path is ``dist`` from the point q (the first such, going back)."""
        k0 = int(np.searchsorted(S, s_from)) - 1
        prev = None
        for k in range(max(k0, 1), 0, -1):
            d = float(np.linalg.norm(P[k] - q))
            if d >= dist:
                if prev is None:
                    return S[k]
                w = (dist - prev[1]) / max(d - prev[1], 1e-9)
                return prev[0] + w * (S[k] - prev[0])
            prev = (S[k], d)
        return S[0]

    def chord_frame(x, s_):
        """A frame with its x along x and its up as near the path's up at s_ as can be."""
        x = x / np.linalg.norm(x)
        u = frame_at(s_)[1][:, 2]
        u = u - (u @ x) * x
        u /= np.linalg.norm(u)
        return np.column_stack([x, np.cross(u, x), u])

    def place_body(sc):
        """Follow the leader: the head's centre on the path at arc sc, every segment's centre on the path behind it,
        each link's x along the chord to the next (segment 1 welded 0.160 m behind the head; the pins half a pitch
        from the segments' centres). The body joints are set from the links' relative rotations."""
        ph = frame_at(sc)[0]
        s1 = behind_at(ph, sc, behind[1])
        c1 = frame_at(s1)[0]
        R1 = chord_frame(ph - c1, 0.5 * (sc + s1))
        data.qpos[fa:fa + 3] = ph
        data.qpos[fa + 3:fa + 7] = mat2quat(R1)
        Rs, c_prev, s_prev = {1: R1}, c1, s1
        half = p.pitch / 2
        for i in (2, 3):
            pin = c_prev - half * Rs[i - 1][:, 0]
            s_i = behind_at(pin, s_prev, half)
            c_i = frame_at(s_i)[0]
            Rs[i] = chord_frame(pin - c_i, s_i)
            c_prev, s_prev = c_i, s_i
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

    def pipe_zone(x):
        """(centre-line point, tangent, radial) if x is within the duct's length and near its wall, else None."""
        d = np.linalg.norm(C - x, axis=1)
        k = int(np.argmin(d))
        c, t = C[k], Ctan[k]
        if (k == 0 and (x - c) @ t < 0) or (k == len(C) - 1 and (x - c) @ t > 0):
            return None
        r = x - c - ((x - c) @ t) * t
        return (c + ((x - c) @ t) * t, t, float(np.linalg.norm(r))) if np.linalg.norm(r) < R_DUCT + 0.08 else None

    def on_right_side(lg, foot):
        """The planted foot on the same side of the pipe's wall as its leg wants: inside on the wall if its segment
        is in the duct, else not inside the duct's cross-section (a leg never reaches through the wall)."""
        ps, Rs = seg_pose(lg[0])
        want_pipe = duct_hit(ps) is not None or pipe_zone(ps + Rs @ nominal[lg[1]]) is not None
        z = pipe_zone(foot)
        in_pipe = z is not None and z[2] < R_DUCT
        if min(np.linalg.norm(foot - C[0]), np.linalg.norm(foot - C[-1])) < R_DUCT + 0.07:
            return True                                                   # at an open end: reached through it
        return in_pipe == want_pipe or (z is None and not want_pipe)

    def surface_z(x, y):
        """The floor, or the top of a cable bundle the foot lands on."""
        z = 0.0
        if FRONT_L - 0.05 <= y <= FRONT_R + 0.05:
            for x_c, r_c, _ in CABLES:
                if abs(x - x_c) <= r_c * 0.55 + 0.035:                    # on top, never against its side
                    z = max(z, 2 * r_c * 0.55)
        if RACK_X0 - N_RACKS * RACK_W <= x <= RACK_X0 and abs(y - FRONT_R + 0.06) < 0.02:
            z = max(z, 0.036)
        return z

    def foothold(lg, pos_R):
        """Where leg lg wants its foot (pad centre) with its segment at pos_R, and the surface normal there."""
        key = lg[1]
        ps, Rs = pos_R
        nom = ps + Rs @ nominal[key]
        dh = duct_hit(ps) or pipe_zone(nom)                               # in (or at the mouth of) the duct: its wall
        if dh is not None:                                                # on the pipe's wall, radially out
            c, t, _ = duct_hit(nom) or dh
            r = nom - c
            r -= (r @ t) * t
            if np.linalg.norm(r) < 1e-6:
                r = -Rs[:, 2]
            r /= np.linalg.norm(r)
            return c + r * (R_DUCT - r_pad), -r
        q = ps + Rs @ (hip[key] + np.array([0.0, mr.LEG_SIDES[key][1] * STANCE[0], 0.0]))
        if st.get("gift") in ("dropping", "armed"):                       # never on or against the gift
            gp = data.mocap_pos[gift_m]
            dxy = q[:2] - gp[:2]
            clear = max(GIFT[0], GIFT[1]) + r_pad + 0.03
            if np.linalg.norm(dxy) < clear:
                q[:2] = gp[:2] + (dxy / (np.linalg.norm(dxy) + 1e-9)) * clear
        z = surface_z(q[0], q[1])
        return np.array([q[0], q[1], z + r_pad]), np.array([0.0, 0.0, 1.0])

    def leg_angles(lg, foot_w):
        """(hip yaw, hip pitch, knee) clamped to the joints' ranges, and the foot's miss [m] with them."""
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
        rng = (mr.LEG_RANGES["hip_yaw"], mr.LEG_RANGES["hip_pitch"], mr.LEG_RANGES["knee"])
        yaw, ph, kr = (float(np.clip(a_, lo, hi)) for a_, (lo, hi) in zip((yaw, ph, kr), rng))
        rr = f_len * math.cos(ph) + t_len * math.cos(ph + kr)
        hh = f_len * math.sin(ph) + t_len * math.sin(ph + kr)
        got = np.array([rr * math.sin(yaw), sy * rr * math.cos(yaw), -hh])
        return (yaw, ph, kr), float(np.linalg.norm(got - v))

    def knee_ok(lg, foot_w):
        """In the duct, the knee stays inside the bore (the leg bars never cross the pipe's wall)."""
        (yaw, ph, _), _ = leg_angles(lg, foot_w)
        i, key = lg
        ps, Rs = seg_pose(i)
        sy = mr.LEG_SIDES[key][1]
        kn = ps + Rs @ (hip[key] + f_len * np.array([math.cos(ph) * math.sin(yaw), sy * math.cos(ph) * math.cos(yaw),
                                                     -math.sin(ph)]))
        z = pipe_zone(kn)
        return z is None or z[2] < R_DUCT - 0.010 or not in_duct_foot(lg, foot_w)

    def in_reach(lg, foot_w):
        """The planted foot still reachable by its leg (within the joints' ranges; 2 mm tolerance), the knee inside
        the bore."""
        return leg_angles(lg, foot_w)[1] < 0.002 and knee_ok(lg, foot_w)

    def in_duct_foot(lg, foot):
        """The foot's in the duct: its segment is, or the foot itself is inside the pipe's bore."""
        z = pipe_zone(foot)
        return duct_hit(seg_pose(lg[0])[0]) is not None or (z is not None and z[2] < R_DUCT)

    def reachable_wall(lg, target):
        """In the duct: the point on the pipe's wall nearest ``target`` that leg lg reaches within its joints' ranges
        (``target`` itself if it does) — a foot is never left where its leg can't follow, pushed through the wall."""
        if in_reach(lg, target):
            return target
        z = pipe_zone(target) or duct_hit(seg_pose(lg[0])[0])
        if z is None:
            return target
        c, t, _ = z
        u = np.cross(t, [0.0, 0.0, 1.0])
        u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-6 else np.cross(t, [1.0, 0.0, 0.0])
        u /= np.linalg.norm(u)
        v = np.cross(t, u)
        best = None
        for off in np.linspace(-0.05, 0.05, 11):
            for a_ in np.radians(np.arange(0.0, 360.0, 6.0)):
                q_ = c + t * off + (R_DUCT - r_pad) * (math.cos(a_) * u + math.sin(a_) * v)
                if in_reach(lg, q_):
                    dd = float(np.linalg.norm(q_ - target))
                    if best is None or dd < best[0]:
                        best = (dd, q_)
        return target if best is None else best[1]

    def leg_ik(lg, foot_w):
        for adr, val in zip(jadr[lg], leg_angles(lg, foot_w)[0]):
            data.qpos[adr] = val

    st = {"t": 0.0, "sc": s_start, "usb": "held", "copy": 0.0, "snips": {},
          "arm": {"g": (ARM_TUCK["g"].copy(), 0.0), "c": (ARM_TUCK["c"].copy(), 0.3)}, "grip": None, "seat": 0.0,
          "usb_port": None, "gift": "carried", "gift_pos": None, "gift_t0": None, "t_boom": None, "blast": None}
    F, N = {}, {}
    place_body(st["sc"])
    for lg in legs:
        F[lg], N[lg] = foothold(lg, seg_pose(lg[0]))

    def arm_pose(arm, tip_h):
        """The arm's joint angles (shoulder yaw, shoulder pitch, elbow) for its pincer tip target in the head frame —
        both elbow solutions tried, each clamped to the joints' ranges, the closer kept — and the world points they
        give: (angles, base, elbow, tip, forearm direction)."""
        hp, hR = data.xpos[head].copy(), data.xmat[head].reshape(3, 3)
        sy = 1.0 if arm == "g" else -1.0
        b = np.array([ARM["base_x"], sy * ARM["base_y"], ARM["base_z"]])
        v = np.asarray(tip_h) - b
        L1, L2 = ARM["upper"], ARM["fore"]
        yaw = math.atan2(v[1], v[0])
        r, h = math.hypot(v[0], v[1]), v[2]
        d = min(max(math.hypot(r, h), abs(L1 - L2) + 1e-3), L1 + L2 - 1e-4)
        e = math.acos(max(-1.0, min(1.0, (d * d - L1 ** 2 - L2 ** 2) / (2 * L1 * L2))))
        rng = mr.ARM_RANGES
        best = None
        for sgn in (1.0, -1.0):                                           # elbow up, elbow down
            th = math.atan2(h, r) + sgn * math.atan2(L2 * math.sin(e), L1 + L2 * math.cos(e))
            ph = th - sgn * e                                             # the forearm's elevation
            q = [yaw, -th, th - ph]                                       # + pitch / elbow lower (about +y)
            q[2] = (q[2] + math.pi) % (2 * math.pi) - math.pi
            q = [float(np.clip(q[0], *rng["shoulder_yaw"])), float(np.clip(q[1], *rng["shoulder_pitch"])),
                 float(np.clip(q[2], *rng["elbow"]))]
            th_c, ph_c = -q[1], -q[1] - q[2]
            def dirn(a, y_=q[0]):
                return np.array([math.cos(a) * math.cos(y_), math.cos(a) * math.sin(y_), math.sin(a)])
            el = b + L1 * dirn(th_c)
            tip = el + L2 * dirn(ph_c)
            in_duct = duct_hit(hp) is not None                            # in the duct: elbow down (no ceiling
            err = float(np.linalg.norm(tip - np.asarray(tip_h))) + (0.005 if (sgn > 0) == in_duct else 0.0)  # room)
            if in_duct:                                                   # no part of the arm out through the wall
                for pt in [b + (el - b) * u_ for u_ in (0.5, 1.0)] + [el + (tip - el) * u_ for u_ in (0.5, 0.85)]:
                    zz = pipe_zone(hp + hR @ pt)
                    if zz is not None and zz[2] > R_DUCT - 0.008:
                        err += 1.0
            if best is None or err < best[0]:
                best = (err, q, el, tip, dirn(ph_c))
        _, q, el, tip, dv = best
        W = lambda x: hp + hR @ x
        return q, W(b), W(el), W(tip), hR @ dv

    def arm_fk(arm, q):
        """The pincer tip and forearm direction (world) of arm joint angles q (shoulder yaw, pitch, elbow)."""
        hp, hR = data.xpos[head].copy(), data.xmat[head].reshape(3, 3)
        sy = 1.0 if arm == "g" else -1.0
        b = np.array([ARM["base_x"], sy * ARM["base_y"], ARM["base_z"]])

        def dirn(a):
            return np.array([math.cos(a) * math.cos(q[0]), math.cos(a) * math.sin(q[0]), math.sin(a)])
        el = b + ARM["upper"] * dirn(-q[1])
        dv = dirn(-q[1] - q[2])
        return hp + hR @ (el + ARM["fore"] * dv), hR @ dv

    gift_m = model.body("gift").mocapid[0]
    gift_led = model.material("gift_led").id
    seg3 = model.body("segment 3").id
    conf = [model.body(f"conf{k}").mocapid[0] for k in range(N_CONFETTI)]
    smoke = [model.body(f"smoke{k}").mocapid[0] for k in range(N_SMOKE)]
    smoke_g = [model.geom(f"smoke{k}").id for k in range(N_SMOKE)]
    flash_m, flash_g, flash_l = model.body("flashball").mocapid[0], model.geom("flashball").id, model.light("flash").id
    sleeper_g = [g for g in range(model.ngeom) if model.geom_bodyid[g] == model.body("sleeper").id]
    hair_g = [model.geom(f"hair{k}").id for k in range(7)]
    rgba0 = {g: model.geom_rgba[g].copy() for g in sleeper_g}
    rng_b = np.random.default_rng(11)

    def gift_props(t):
        if st["gift"] == "carried":                                       # clamped under the tail segment
            R3 = data.xmat[seg3].reshape(3, 3)
            data.mocap_pos[gift_m] = data.xpos[seg3] + R3 @ np.array([0.0, 0.0, -GIFT_UNDER])
            data.mocap_quat[gift_m] = data.xquat[seg3]
        elif st["gift"] == "dropping":                                    # the clamp opens: it drops to the floor
            p0, z0, t0 = st["gift_pos"]
            zc = max(z0 - 0.5 * 9.81 * (t - t0) ** 2, GIFT[2])
            data.mocap_pos[gift_m] = [p0[0], p0[1], zc]
            if zc <= GIFT[2]:
                st["gift"] = "armed"
        if st["gift_t0"] is not None and st["gift"] != "gone":
            model.mat_emission[gift_led] = 1.0 if (t * 2.5) % 1.0 < 0.35 else 0.0
        if st["t_boom"] is not None and t >= st["t_boom"]:
            blast(t - st["t_boom"])

    def blast(s_):
        """KABOOM: the flash, confetti up and out, smoke; the guard soot-black, jolted, hair on end."""
        g0 = data.mocap_pos[gift_m].copy() if st["blast"] is None else st["blast"]["g0"]
        if st["blast"] is None:
            n = N_CONFETTI
            az = rng_b.uniform(0, 2 * math.pi, n)
            el = rng_b.uniform(0.35, 1.45, n)
            sp = rng_b.uniform(2.5, 6.5, n)
            st["blast"] = {"g0": g0, "cv": np.column_stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
                           * sp[:, None], "cp": np.tile(g0, (n, 1)), "cw": rng_b.normal(0, 14, (n, 3)),
                           "cq": np.tile([1.0, 0, 0, 0], (n, 1)), "last": 0.0,
                           "sv": rng_b.normal(0, 0.12, (N_SMOKE, 3)) + np.array([0, 0, 0.3])}
            st["gift"] = "gone"
            data.mocap_pos[gift_m] = [0, 0, -5]
            model.mat_emission[gift_led] = 0.0
        B = st["blast"]
        h = max(s_ - B["last"], 0.0)
        B["last"] = s_
        for _ in range(max(1, int(h / 0.004))):
            hh = h / max(1, int(h / 0.004))
            v = B["cv"]
            spd = np.linalg.norm(v, axis=1, keepdims=True)
            B["cv"] = v + hh * (np.array([0, 0, -9.81]) - 8.0 * spd * v + rng_b.normal(0, 4.0, v.shape))
            B["cp"] = B["cp"] + hh * B["cv"]
            B["cp"][:, 1] = np.clip(B["cp"][:, 1], FRONT_L + 0.005, FRONT_R - 0.005)   # the racks stop them
            low = B["cp"][:, 2] < 0.001
            B["cp"][low, 2] = 0.001
            B["cv"][low] = 0.0
            B["cw"][low] = 0.0
        for k in range(N_CONFETTI):
            w_ = B["cw"][k] * h
            ang = np.linalg.norm(w_)
            if ang > 1e-9:
                B["cq"][k] = tv._qmul(np.array([math.cos(ang / 2), *(math.sin(ang / 2) * w_ / ang)]), B["cq"][k])
            data.mocap_pos[conf[k]] = B["cp"][k]
            data.mocap_quat[conf[k]] = B["cq"][k]
        a_ = max(0.0, 1.0 - s_ / 0.45)
        data.mocap_pos[flash_m] = g0
        model.geom_size[flash_g][0] = 0.04 + 0.3 * sm(s_ / 0.1)
        model.geom_rgba[flash_g] = [1, 0.95 - 0.35 * min(s_ / 0.45, 1), 0.6 - 0.4 * min(s_ / 0.3, 1), 0.8 * a_]
        f_ = 6.0 * math.exp(-s_ / 0.18)
        model.light_diffuse[flash_l] = [f_, 0.8 * f_, 0.45 * f_]
        model.light_pos[flash_l] = g0 + np.array([0, 0, 0.4])
        for k in range(N_SMOKE):
            age = s_ - 0.03 * k
            if age < 0:
                continue
            data.mocap_pos[smoke[k]] = g0 + B["sv"][k] * age + np.array([0, 0, 0.05 * age * age])
            model.geom_size[smoke_g[k]][0] = 0.04 + 0.12 * (1 - math.exp(-age / 0.8))
            gg = 0.3 + 0.25 * min(age / 1.5, 1)
            model.geom_rgba[smoke_g[k]] = [gg, gg, gg * 1.02, 0.45 * math.exp(-age / 1.8)]
        k_ = sm(s_ / 0.15)                                                # the guard: soot, a jolt, hair on end
        for g in sleeper_g:
            if g in hair_g:
                model.geom_rgba[g] = [0.04, 0.04, 0.04, 1.0 * k_]
            else:
                model.geom_rgba[g] = (1 - k_) * rgba0[g] + k_ * np.array([0.05, 0.045, 0.045, rgba0[g][3]])
        jolt = 0.12 * math.exp(-s_ / 0.25) * abs(math.sin(s_ * 25))
        st["jolt"] = jolt

    def carry_target():
        """The carried cutter's target (head frame): the robot's path 0.2 m ahead of the head, 35 mm below it — in the
        duct the bore's axis ahead, so the arm follows the bends."""
        drop = CARRY_DROP if duct_hit(data.xpos[head]) is not None else -0.04    # in the room: well above the cables
        p_ = frame_at(min(st["sc"] + CARRY_AHEAD, S[-1]))[0] - np.array([0.0, 0.0, drop])
        hp, hR = data.xpos[head].copy(), data.xmat[head].reshape(3, 3)
        return hR.T @ (p_ - hp)

    def props():
        t = st["t"]
        if st.get("c_carry"):
            st["arm"]["c"] = (carry_target(), st["arm"]["c"][1])
        gift_props(t)
        hR = data.xmat[head].reshape(3, 3)
        # the arms and their pincers
        for arm in ("g", "c"):
            tip_h, opening = st["arm"][arm]
            q, _, _, tip, d = arm_pose(arm, tip_h)
            w = float(np.clip(np.linalg.norm(np.asarray(tip_h) - ARM_TUCK[arm]) / 0.04, 0.0, 1.0))
            if w < 1.0:                                                   # near the fold: the robot's folded pose
                qt = [mr.ARM_TUCK["shoulder_yaw"], mr.ARM_TUCK["shoulder_pitch"], mr.ARM_TUCK["elbow"]]
                q = [(1 - w) * a_ + w * b_ for a_, b_ in zip(qt, q)]
                tip, d = arm_fk(arm, q)
            jaw = float(np.clip(0.04 + 0.55 * opening, *mr.ARM_RANGES["jaw"]))
            for adr, val in zip(arm_q[arm], [*q, jaw, jaw]):                # the robot's arm joints
                data.qpos[adr] = val
            if arm == "g":
                st["grip"] = (tip + d * GRIP_IN, d)
        if st["usb"] == "held":                                           # in the gripper, pointing ahead
            g, _ = st["grip"]
            pos_h, quat_h = g + hR[:, 0] * (USB_HALF - 0.006), data.xquat[head].copy()
            w = sm(st["seat"])                                            # sliding level and straight into the port
            if w > 0.0:
                yaw = math.atan2(hR[1, 0], hR[0, 0])
                pos_h = (1 - w) * pos_h + w * st["usb_port"]
                quat_h = slerp(quat_h, mat2quat(rot_z(yaw)), w)
            data.mocap_pos[usb], data.mocap_quat[usb] = pos_h, quat_h
        # the cut: a slat cut at one end sags a little; cut at both, it drops onto the desk and settles there
        for k, (mid, w, z0) in slats.items():
            gx0 = X_END - 0.006
            n_cut, t_free = st["snips"].get(k, (0, None))
            if t_free is None:
                tilt = 0.06 if n_cut == 1 else 0.0
                data.mocap_pos[mid] = [gx0, AISLE_Y, z0 - 0.003 * (n_cut == 1)]
                data.mocap_quat[mid] = mat2quat(rot_y(-tilt * 0.3) @ np.eye(3))
                continue
            dt_ = t - t_free
            rest = 0.0045 + 0.009 * (k % 3)
            zc = max(z0 - 0.5 * 9.81 * dt_ * dt_, rest)
            u = min(1.0, (z0 - zc) / max(z0 - rest, 1e-6))
            data.mocap_pos[mid] = [gx0 - 0.05 * u - 0.012 * (k % 4), AISLE_Y + 0.015 * ((k % 5) - 2) * u, zc]
            data.mocap_quat[mid] = mat2quat(rot_z(0.25 * ((k % 3) - 1) * u) @ rot_y(-1.5708 * u))
        # the sleeper breathes; the copy's progress bar; the pendrive's LED
        breathe = 0.006 * math.sin(t * 2 * math.pi / 4.2) if st["t_boom"] is None or t < st["t_boom"] else 0.0
        data.mocap_pos[sleeper] = sleeper_p0 + np.array([0, 0, breathe + st.get("jolt", 0.0)])
        model.geom_size[snore][0] = 0.006 + 0.01 * max(0.0, math.sin(t * 2 * math.pi / 4.2))
        fill = 0.1 * st["copy"]
        model.geom_size[bar][1] = max(fill, 0.0005)
        model.geom_pos[bar][1] = -0.1 + fill
        on = st["usb"] == "port" and st["copy"] < 1.0 and (t * 6) % 1.0 < 0.5
        model.mat_emission[led] = 1.0 if on else 0.0
        for j, m_ in enumerate(rack_leds):                                # the servers' LEDs flicker
            model.mat_emission[m_] = 0.55 + 0.45 * (math.sin(t * (7.0 + 3.1 * j)) > -0.3)

    def tick(phase, caption, speed=1.0):
        props()
        mujoco.mj_forward(model, data)
        if render is not None:
            render(st, data, phase, caption, speed)

    def freegait(phase, caption, s_to, v=0.12, t_swing=0.16, reach=0.085, look=0.06, speed=1.0, settle=0.3):
        sgn = 1.0 if s_to >= st["sc"] else -1.0
        step = (1.0 / render.fps) if render is not None else 1.0 / 30        # the gait's own step: real time;
        every, n_tick = max(1, round(speed)), 0                         # a time-lapse renders every n-th
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
                        b_ = want_a[lg][0].copy()
                        if in_duct_foot(lg, b_):                                 # a step it can land
                            b_ = reachable_wall(lg, b_)
                        swing[lg] = (st["t"], F[lg].copy(), b_, N[lg].copy(), want_a[lg][1].copy())
                    quiet = 0.0
                elif abs(s_to - sc) < 1e-6:
                    quiet += step
                    if quiet >= settle:
                        break
            # the body advances while every planted foot reaches
            if abs(s_to - sc) > 1e-6:
                s_try = sc + sgn * min(v * step, abs(s_to - sc))
                place_body(s_try)
                planted = [lg for lg in legs if lg not in swing]
                ok = all(np.linalg.norm(F[lg] - foothold(lg, seg_pose(lg[0]))[0]) < reach and in_reach(lg, F[lg])
                         for lg in planted)
                wall_ok = all(on_right_side(lg, F[lg]) for lg in planted)                  # the hard rule
                if os.environ.get("CM_DEBUG") and int(st["t"] * 10) % 20 == 0:
                    bad = [lg for lg in planted if not on_right_side(lg, F[lg])]
                    print(f"    t {st['t']:.1f} sc {sc:.3f} ok {ok} swing {sorted(swing)} wrong side {bad} "
                          f"{[np.round(F[lg], 3).tolist() for lg in bad]}", flush=True)
                if wall_ok and (ok or blocked > 0.4):
                    st["sc"], blocked = s_try, 0.0
                else:
                    blocked += step
            place_body(st["sc"])
            for lg in legs:                                               # a planted foot out of reach: re-planted
                if lg not in swing and in_duct_foot(lg, F[lg]) and not in_reach(lg, F[lg]):
                    F[lg] = reachable_wall(lg, F[lg])
            feet = {}
            for lg in legs:
                if lg in swing:
                    t0, a, b, na, nb = swing[lg]
                    u = min((st["t"] + step - t0) / t_swing, 1.0)
                    n = na + nb
                    n /= np.linalg.norm(n) + 1e-9
                    lift = 0.055 if nb[2] > 0.9 else 0.012                # over cables; just off the pipe's wall
                    feet[lg] = a + sm(u) * (b - a) + n * lift * math.sin(math.pi * u)
                    if u >= 1.0:
                        F[lg], N[lg] = b, nb
                        del swing[lg]
                else:
                    feet[lg] = F[lg]
            for lg in legs:
                leg_ik(lg, feet[lg])
            st["feet_target"], st["swing"], st["planted"] = feet, swing, F
            st["t"] += step
            n_tick += 1  # noqa: SIM113
            if n_tick % every == 0:
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

    hold("yard", "", 1.0)
    move_arm("c", "yard", "THE CUTTER UNFOLDS — CARRIED AHEAD", 1.4, carry_target(), opening=0.1, lift=0.0)
    st["c_carry"] = True
    freegait("yard", "ACROSS THE YARD TO THE DATACENTER'S AIR INTAKE", s_mouth - 0.05, v=0.16)
    freegait("duct", "INTO THE DUCT — TWELVE HOOKED FEET ON THE PIPE WALL", s_mouth + 1.0, v=0.13, speed=3.0)
    s_cut = float(S[np.argmin(np.abs(P[:, 0] - (X_END + 0.07)) + 10 * (P[:, 2] < Z_IN - 0.05))])
    freegait("riser", "ALONG THE WALL AND THROUGH IT", s_cut, v=0.11, speed=4.0)

    # ---- the cut: the cutter snips each slat where the hole's edge crosses it, left end then right end; a slat cut
    # at both ends drops onto the desk. Top to bottom, so a falling slat clears the ones below
    gxc = X_END - 0.001                                                   # at the grille, just past the pipe's end
    cap = "THE CUTTER SNIPS THE GRILLE'S SLATS, ONE BY ONE"
    st["c_carry"] = False
    move_arm("c", "cut", cap, 0.8, head_frame(np.array([gxc - 0.03, AISLE_Y, Z_IN])), opening=0.2, lift=0.0)  # the middle
    for n, k in enumerate(sorted(slats, reverse=True)):
        _, w, z0 = slats[k]
        speed = 1.0 if n < 3 else 3.0
        for side, sy in enumerate((1.0, -1.0)):
            pt = head_frame(np.array([gxc, AISLE_Y + sy * (w - 0.003), z0]))
            move_arm("c", "cut", cap, 0.45, pt + [0.012, 0, 0], opening=0.22, lift=0.0, speed=speed)   # open, reach
            move_arm("c", "cut", cap, 0.15, pt, lift=0.0, speed=speed)                                # in round it
            move_arm("c", "cut", cap, 0.12, pt, opening=0.0, lift=0.0, speed=speed)                   # snip
            st["snips"][k] = (side + 1, st["t"] if side == 1 else None)
    hold("cut", "THE SLATS DROP TO THE FLOOR", 0.8)
    move_arm("c", "cut", "THE SLATS DROP TO THE FLOOR", 0.8, carry_target(), opening=0.1, lift=0.0)
    st["c_carry"] = True
    freegait("vent", "THROUGH THE HOLE INTO THE BANDITS' DATACENTER", s_cut + 0.75, v=0.07, speed=5.0)
    freegait("aisle", "DOWN THE COLD AISLE, OVER THE CABLES, BETWEEN THE RACKS", s_end - 0.12, v=0.1, speed=2.0)
    freegait("plug", "TO THE TARGET SERVER", s_end, v=0.05, speed=3.0)
    # ---- the plug: the gripper reaches down to the back port, lets go, comes back for it after the copy
    hR = data.xmat[head].reshape(3, 3)
    hx = hR[:, 0]
    usb_c = port_point() - hx * (USB_PLUG - 0.010)                        # the plug 10 mm in
    tip_w = usb_c - hx * (USB_HALF - 0.006) - hx * GRIP_IN
    st["usb_port"] = np.array([usb_c[0], usb_c[1], port_point()[2]])
    move_arm("g", "plug", "THE GRIPPER PLUGS THE PENDRIVE INTO THE SERVER", 1.8, head_frame(tip_w), lift=0.04)
    hold("plug", "THE GRIPPER PLUGS THE PENDRIVE IN", 0.4, lambda u: st.update(seat=u))
    st["seat"] = 1.0
    props()
    st["usb"] = "port"
    move_arm("g", "plug", "THE GRIPPER PLUGS THE PENDRIVE IN", 0.4, st["arm"]["g"][0], opening=1.0, lift=0.0)
    move_arm("g", "copy", "COPYING", 1.0, ARM_TUCK["g"], lift=0.0)
    hold("copy", "COPYING", 6.5, fn=lambda u: st.update(copy=min(1.0, u * 1.1)))
    move_arm("g", "plug", "COPY DONE — UNPLUG", 1.0, head_frame(tip_w), lift=0.0)
    move_arm("g", "plug", "COPY DONE — UNPLUG", 0.4, head_frame(tip_w), opening=0.0, lift=0.0)
    st["usb"] = "held"
    hold("plug", "COPY DONE — UNPLUG", 0.4, lambda u: st.update(seat=1.0 - u))
    st["seat"] = 0.0
    move_arm("g", "plug", "COPY DONE — UNPLUG", 1.2, ARM_TUCK["g"], lift=0.02)
    # the gift: the clamp under the tail opens, it drops by the server, its fuse timer starts
    g_now = data.mocap_pos[gift_m].copy()
    st["gift"], st["gift_pos"], st["gift_t0"] = "dropping", (g_now, float(g_now[2]), st["t"]), st["t"]
    hold("gift", "A GIFT FOR THE GUARD — THE FUSE TIMER STARTS", 2.2)
    freegait("retreat", "BACK OUT THE WAY IT CAME", s_mouth - 0.4, v=0.13, speed=SPEEDUP)
    st["c_carry"] = False
    move_arm("c", "out", "", 1.2, ARM_TUCK["c"], opening=0.3, lift=0.0)   # out of the duct: folded again
    hold("out", "", 0.8)
    st["t_boom"] = st["t"] + 2.4                                          # back in the datacenter: 3, 2, 1 …
    hold("boom", "", 2.4)
    hold("boom", "KABOOM", 4.5)
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
            cam.set(mid + [0.0, 0, 0.02], 1.2, 215, -14, 0.08)
        elif phase == "cut":
            cam.set(np.array([X_END - 0.03, AISLE_Y, Z_IN + 0.02]), 0.62, 20, -12, 0.1)
        elif phase in ("vent", "aisle"):
            cam.set(mid + [0, 0, 0.02], 0.95, 10, -14, 0.08)
        elif phase == "plug":
            cam.set(0.5 * (port_point() + h) + [0, 0, 0.03], 0.62, -15, -12, 0.08)
        elif phase == "copy":
            cam.set(np.array([TARGET_X - 0.5, AISLE_Y, 0.55]), 1.7, 180, -8, 0.05)
        elif phase == "gift":
            cam.set(np.array([TARGET_X + 0.1, AISLE_Y + 0.05, 0.08]), 0.75, -35, -14, 0.08)
        elif phase == "boom":
            cam.set(np.array([TARGET_X - 0.45, AISLE_Y, 0.45]), 1.7, 180, -6, 0.15)
        elif phase == "retreat":
            if mid[0] < WALL[0]:
                cam.set(mid + [0, 0, 0.02], 1.0, 10, -14, 0.06)
            else:
                cam.set(mid, 1.3, 250, -14, 0.06)
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
            dr.text((24, bar / 2 - 12), "OPERATION PIPE ENTRY — THE BANDITS' DATACENTER", font=f_cap,
                    fill=(255, 196, 64))
        img.save(frames_dir / f"f_{count['n']:05d}.jpg", quality=92)
        count["n"] += 1
        if count["n"] % 150 == 0:
            print(f"  {count['n']} frames, t = {st['t']:6.2f} s, {phase}", flush=True)

    render.fps = fps
    st = simulate(model, info, render=render)
    print(f"cleopatra_datacenter: {count['n']} frames, {st['t']:.1f} s of story", flush=True)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps), "-i", str(frames_dir / "f_%05d.jpg"),
                        "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags",
                        "+faststart", str(out)], check=True)
        print("cleopatra_mission:", out)
    else:
        print("cleopatra_datacenter: no ffmpeg — the frames are in", frames_dir)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cleopatra.sh mission", description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(ROOT / "runs" / "cleopatra" / "cleopatra_datacenter.mp4"))
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--text", action="store_true", help="captions in the letterbox")
    a = ap.parse_args(argv)
    run(a.out, a.width, a.height, a.fps, text=a.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
