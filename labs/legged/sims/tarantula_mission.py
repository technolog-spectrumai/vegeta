"""TARANTULA, "Operation Silent Drop" (``./tarantula.sh mission``): a night-time MuJoCo short. TARANTULA creeps tail
first across a roof to the parapet, sways back to drive its anchor hook into the parapet's timber nailer (§12), climbs
onto the coping, turns, goes over the edge head first and lets go, hanging on its dragline from the spool's fairlead.
The brake pays it down — no motor running, silently — to a terrace table, where its pedipalps' tarsal pincers set down
a gift; a chelicera's fang presses the gift's timer. The winch reels it back up (47 mm/s; the video runs that part
time-lapsed), it climbs back over and hides behind the parapet. An unsuspecting bandit strolls to the table, leans in
— the timer runs out: KABOOM, confetti, and a bandit as black as soot.

What is simulated (MuJoCo physics): the walk across the roof (the reversed wave gait on the robot's own servos and
feet), the sway planting the hook, and the whole time on the line — the robot hanging from the rope (a spatial tendon
wrapping the coping's corners, pulling only when taut: ``tarantula_video``'s rope law), the brake paying out and the
winch reeling in. Scripted (kinematic, the joints and the body moved along keyframes): the climb over the coping and
the hide; the bandit, the gift, the confetti and the smoke are props (mocap bodies).

    ./tarantula.sh mission                       # runs/tarantula/tarantula_mission.mp4 (1280 × 720, 30 fps)
    ./tarantula.sh mission --width 960 --height 540 --out my.mp4

Needs what ``tarantula_video`` needs (mujoco, cadquery, Pillow, ffmpeg).
"""
from __future__ import annotations

import argparse
import itertools
import math
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import tarantula_video as tv

ROOT = tv.ROOT
STREET = -2.4                                                             # the street below the roof [m]
TABLE_H = 0.75
SPEEDUP = 8.0                                                             # the time-lapse of the winch climb
N_CONFETTI = 320
N_SMOKE = 14
GIFT_HALF = (0.03, 0.062, 0.026)                                          # the gift box's half sizes, body frame [m]
GIFT_AT = (0.186, 0.0, -0.044)                                            # held between the tarsal pincers [m, body]


# --------------------------------------------------------------------------------------------- small maths
def sm(u):
    u = min(max(u, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * u)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


def mat2quat(R):
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
    return q


def quat2mat(q):
    import mujoco
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def slerp(q0, q1, u):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + u * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1 - u) * th) * q0 + math.sin(u * th) * q1) / math.sin(th)


def _s(v):
    return " ".join(f"{x:.5f}" for x in v)


# --------------------------------------------------------------------------------------------- the scene
def build(width, height, cache: Path, face_x=2.0, tip_z=0.09):
    """(MjModel, info): the roof, the parapet with its timber nailer (its face at ``face_x``, centred on the hook's
    height ``tip_z``), the facade, the terrace table, the lantern, the bandit, the gift and the explosion's props;
    TARANTULA dressed in its CAD (``tarantula_video.dress_robot``) and its dragline over the coping."""
    import motion_kit as mk
    import mujoco
    import tarantula_robot as tr
    from vegeta.chiron.robot import SimOptions
    robot = tr.tarantula()
    kin = mk.Kin(robot)
    opt = SimOptions(timestep=0.001, gravity=(0.0, 0.0, -9.81), cone="elliptic", impratio=3.0, noslip_iterations=2)
    root = ET.fromstring(robot.to_mjcf(options=opt))
    meshes = tv.cad_meshes(cache)
    tile = tv._brick_png(cache / "brick.png")
    vis = tv._child(root, "visual")
    tv._child(vis, "global").attrib.update({"offwidth": str(width), "offheight": str(height), "fovy": "40"})
    tv._child(vis, "quality").attrib.update({"shadowsize": "4096", "offsamples": "8"})
    tv._child(vis, "headlight").attrib.update({"ambient": "0.13 0.14 0.18", "diffuse": "0.12 0.12 0.14",
                                               "specular": "0.02 0.02 0.02"})
    tv._child(vis, "map").attrib.update({"znear": "0.004", "zfar": "60", "haze": "0.2"})
    asset = tv._child(root, "asset")
    ET.SubElement(asset, "texture", {"name": "sky", "type": "skybox", "builtin": "gradient", "rgb1": "0.05 0.07 0.15",
                                     "rgb2": "0.0 0.0 0.02", "mark": "random", "random": "0.0025",
                                     "markrgb": "0.9 0.9 1", "width": "800", "height": "800"})
    ET.SubElement(asset, "texture", {"name": "brick", "type": "2d", "file": str(cache / "brick.png")})
    ET.SubElement(asset, "material", {"name": "brick", "texture": "brick", "texrepeat": f"{1 / tile:.3f} {1 / tile:.3f}",
                                      "texuniform": "true", "roughness": "0.9", "specular": "0.05"})
    ET.SubElement(asset, "texture", {"name": "gravel", "type": "2d", "builtin": "flat", "rgb1": "0.33 0.33 0.34",
                                     "rgb2": "0.2 0.2 0.22", "mark": "random", "random": "0.25", "markrgb": "0.5 0.5 0.5",
                                     "width": "512", "height": "512"})
    ET.SubElement(asset, "material", {"name": "roof", "texture": "gravel", "texrepeat": "6 6", "texuniform": "true",
                                      "specular": "0.05"})
    ET.SubElement(asset, "texture", {"name": "paving", "type": "2d", "builtin": "checker", "rgb1": "0.27 0.26 0.25",
                                     "rgb2": "0.23 0.22 0.21", "width": "512", "height": "512"})
    ET.SubElement(asset, "material", {"name": "paving", "texture": "paving", "texrepeat": "4 4", "texuniform": "true",
                                      "specular": "0.08"})
    ET.SubElement(asset, "material", {"name": "stone", "rgba": "0.62 0.6 0.56 1", "specular": "0.05"})
    ET.SubElement(asset, "material", {"name": "wood", "rgba": "0.5 0.33 0.19 1", "specular": "0.1"})
    ET.SubElement(asset, "material", {"name": "darkwood", "rgba": "0.3 0.19 0.11 1", "specular": "0.2"})
    ET.SubElement(asset, "material", {"name": "glow", "rgba": "1 0.78 0.42 1", "emission": "1"})
    ET.SubElement(asset, "material", {"name": "window", "rgba": "0.95 0.72 0.4 1", "emission": "0.45"})
    ET.SubElement(asset, "material", {"name": "led", "rgba": "1 0.1 0.08 1", "emission": "0"})
    ET.SubElement(asset, "material", {"name": "metal", "rgba": "0.15 0.15 0.16 1", "specular": "0.5", "shininess": "0.5"})
    world = root.find("worldbody")
    for gm in world.findall("geom"):                                      # the Chiron ground plane: the street
        if gm.get("type") == "plane":
            gm.set("size", "12 12 0.1")
            gm.set("pos", f"0 0 {STREET}")
            gm.set("material", "paving")
            gm.attrib.pop("rgba", None)

    # the building: the roof at z = 0 (the robot's ground), the parapet's coping overhanging the facade
    xi = face_x + 0.015                                                   # the coping's inner face
    xf = xi + 0.06                                                        # the facade
    xo = xi + 0.30                                                        # the coping's outer edge
    zt = max(0.105, tip_z + 0.045)                                        # the coping's top
    zb = -0.045                                                           # its underside
    vis_g = {"contype": "0", "conaffinity": "0", "group": "1"}
    solid_g = {"contype": "1", "conaffinity": "2", "group": "1"}       # the roof's solid parts: the feet touch them
    ET.SubElement(world, "geom", {"name": "building", "type": "box", "size": f"2 3 {-STREET / 2:.3f}",
                                  "pos": f"{xf - 2:.4f} 0 {STREET / 2:.3f}", "contype": "1", "conaffinity": "2",
                                  "group": "3", "rgba": "0 0 0 0"})
    ET.SubElement(world, "geom", {"name": "roof", "type": "box", "size": "2 3 0.02", "pos": f"{xi - 2:.4f} 0 -0.02",
                                  "material": "roof", **vis_g})
    ET.SubElement(world, "geom", {"name": "facade", "type": "box", "size": f"0.2 3 {(-STREET + zb) / 2:.4f}",
                                  "pos": f"{xf - 0.2:.4f} 0 {(STREET + zb) / 2:.4f}", "material": "brick", **solid_g})
    ET.SubElement(world, "geom", {"name": "coping", "type": "box", "size": f"0.15 3 {(zt - zb) / 2:.4f}",
                                  "pos": f"{xi + 0.15:.4f} 0 {(zt + zb) / 2:.4f}", "material": "stone", **solid_g})
    ET.SubElement(world, "geom", {"name": "drip", "type": "box", "size": "0.012 3 0.008",
                                  "pos": f"{xo - 0.012:.4f} 0 {zb - 0.008:.4f}", "material": "stone", **vis_g})
    ET.SubElement(world, "geom", {"name": "nailer", "type": "box", "size": "0.0075 3 0.032",
                                  "pos": f"{face_x + 0.0075:.4f} 0 {tip_z - 0.008:.4f}", "material": "wood", **solid_g})
    for k, y in enumerate((-1.4, 1.25)):                                  # two windows, one lit
        ET.SubElement(world, "geom", {"name": f"window{k}", "type": "box", "size": "0.004 0.26 0.36",
                                      "pos": f"{xf + 0.003:.4f} {y} -1.05", "material": "window" if k else "metal", **vis_g})
        ET.SubElement(world, "geom", {"name": f"sill{k}", "type": "box", "size": "0.04 0.34 0.02",
                                      "pos": f"{xf + 0.04:.4f} {y} -1.49", "material": "stone", **vis_g})
    # the terrace: a table against the wall under a lantern, a bottle and a mug on it
    zt_tab = STREET + TABLE_H
    tx0, tx1 = xf + 0.06, xf + 0.78
    tcx = (tx0 + tx1) / 2
    ET.SubElement(world, "geom", {"name": "table", "type": "box", "size": f"{(tx1 - tx0) / 2:.4f} 0.42 0.018",
                                  "pos": f"{tcx:.4f} 0 {zt_tab - 0.018:.4f}", "material": "darkwood", **vis_g})
    for sx in (tx0 + 0.04, tx1 - 0.04):
        for sy in (-0.38, 0.38):
            ET.SubElement(world, "geom", {"type": "box", "size": f"0.02 0.02 {(TABLE_H - 0.036) / 2:.4f}",
                                          "pos": f"{sx:.4f} {sy} {STREET + (TABLE_H - 0.036) / 2:.4f}",
                                          "material": "darkwood", **vis_g})
    ET.SubElement(world, "geom", {"type": "cylinder", "size": "0.035 0.11", "pos": f"{tx1 - 0.12:.4f} -0.27 {zt_tab + 0.11:.4f}",
                                  "rgba": "0.1 0.35 0.15 0.85", **vis_g})
    ET.SubElement(world, "geom", {"type": "cylinder", "size": "0.013 0.04", "pos": f"{tx1 - 0.12:.4f} -0.27 {zt_tab + 0.25:.4f}",
                                  "rgba": "0.1 0.35 0.15 0.85", **vis_g})
    ET.SubElement(world, "geom", {"type": "cylinder", "size": "0.04 0.045", "pos": f"{tx1 - 0.2:.4f} 0.26 {zt_tab + 0.045:.4f}",
                                  "rgba": "0.85 0.85 0.8 1", **vis_g})
    lant = np.array([xf + 0.07, -0.55, -0.72])
    ET.SubElement(world, "geom", {"type": "box", "size": "0.035 0.01 0.01", "pos": _s(lant + [-0.035, 0, 0.09]),
                                  "material": "metal", **vis_g})
    ET.SubElement(world, "geom", {"type": "box", "size": "0.05 0.05 0.075", "pos": _s(lant), "rgba": "0.1 0.1 0.1 0.35",
                                  **vis_g})
    ET.SubElement(world, "geom", {"type": "sphere", "size": "0.028", "pos": _s(lant), "material": "glow", **vis_g})
    ET.SubElement(world, "geom", {"type": "box", "size": "0.055 0.055 0.01", "pos": _s(lant + [0, 0, 0.083]),
                                  "material": "metal", **vis_g})
    ET.SubElement(world, "light", {"name": "lantern", "pos": _s(lant + [0.06, 0.05, -0.05]), "dir": "0.45 0.35 -1",
                                   "diffuse": "1.0 0.66 0.32", "specular": "0.2 0.15 0.1", "cutoff": "75",
                                   "exponent": "2", "attenuation": "1 0 0.6", "castshadow": "true"})
    ET.SubElement(world, "light", {"name": "moon", "directional": "true", "pos": "4 -3 6", "dir": "-0.45 0.35 -0.82",
                                   "diffuse": "0.26 0.3 0.42", "specular": "0.08 0.08 0.1", "castshadow": "true"})
    ET.SubElement(world, "light", {"name": "roofspot", "pos": f"{xi - 0.6:.4f} -0.6 0.9", "dir": "0.6 0.5 -0.7",
                                   "diffuse": "0.32 0.34 0.42", "specular": "0 0 0", "cutoff": "40", "exponent": "8",
                                   "castshadow": "false"})
    ET.SubElement(world, "light", {"name": "flash", "pos": f"{tcx:.4f} 0 {zt_tab + 0.25:.4f}", "dir": "0 0 -1",
                                   "diffuse": "0 0 0", "specular": "0 0 0", "cutoff": "90", "castshadow": "false",
                                   "attenuation": "1 0 0.3"})

    # the robot, in its CAD, with its cords, fairlead and anchor hook
    dr = tv.dress_robot(root, kin, meshes)
    tendon_el = dr["tendon"]
    # the dragline's three paths (shown one at a time): eye → fairlead direct; over the coping's inner corner; over
    # both corners, hanging off the outer edge (the one that carries the robot, with the winch motor on it)
    r_w = 0.006
    c1 = np.array([xi, 0.0, zt])
    c2 = np.array([xo, 0.0, zt])
    for k, c in enumerate((c1, c2), 1):
        ET.SubElement(world, "geom", {"name": f"corner{k}", "type": "cylinder", "size": f"{r_w} 3",
                                      "pos": _s(c), "zaxis": "0 1 0", "contype": "0", "conaffinity": "0", "group": "3"})
        ET.SubElement(world, "site", {"name": f"side{k}", "pos": _s(c + [(-1 if k == 1 else 1) * 0.03, 0, 0.03]),
                                      "group": "4"})
    ET.SubElement(world, "site", {"name": "ridge", "pos": _s((c1 + c2) / 2 + [0, 0, r_w]), "group": "4"})
    for name in ("dl_short", "dl_over", "dragline"):
        el = ET.SubElement(tendon_el, "spatial", {"name": name, "width": "0.0008", "rgba": "0.96 0.96 0.9 0",
                                                  "group": "1"})
        ET.SubElement(el, "site", {"site": "eye"})
        if name != "dl_short":
            ET.SubElement(el, "geom", {"geom": "corner1", "sidesite": "side1"})
            ET.SubElement(el, "site", {"site": "ridge"})
        if name == "dragline":
            ET.SubElement(el, "geom", {"geom": "corner2", "sidesite": "side2"})
        ET.SubElement(el, "site", {"site": "muzzle"})
    act = root.find("actuator")
    ET.SubElement(act, "motor", {"name": "winch", "tendon": "dragline", "gear": "1", "ctrllimited": "false"})

    # the gift: a red box with a gold ribbon and bow, a timer's LED (a mocap body, carried, then set down)
    gb = ET.SubElement(world, "body", {"name": "gift", "mocap": "true", "pos": "0 0 -50"})
    hx, hy, hz = GIFT_HALF
    gift_geoms = [ET.SubElement(gb, "geom", {"name": "gift", "type": "box", "size": f"{hx} {hy} {hz}",
                                             "rgba": "0.78 0.06 0.08 1", **vis_g})]
    gold = "0.98 0.78 0.2 1"
    gift_geoms.append(ET.SubElement(gb, "geom", {"type": "box", "size": f"{hx + 0.0006} 0.007 {hz + 0.0006}",
                                                 "rgba": gold, **vis_g}))
    gift_geoms.append(ET.SubElement(gb, "geom", {"type": "box", "size": f"{hx + 0.0006} {hy + 0.0006} 0.007",
                                                 "rgba": gold, **vis_g}))
    for sy in (-1, 1):                                                    # the bow, on the box's top (body -x: up when
        gift_geoms.append(ET.SubElement(gb, "geom", {"type": "ellipsoid", "size": "0.007 0.014 0.008",  # hanging)
                                                     "pos": f"{-hx - 0.005} {sy * 0.012} 0", "rgba": gold, **vis_g}))
    gift_geoms.append(ET.SubElement(gb, "geom", {"name": "led", "type": "sphere", "size": "0.0045",
                                                 "pos": f"{-hx - 0.002} 0.034 {hz * 0.4:.4f}", "material": "led",
                                                 **vis_g}))
    # the explosion's props: a flash, smoke, confetti, the scorch mark
    fb = ET.SubElement(world, "body", {"name": "flashball", "mocap": "true", "pos": "0 0 -50"})
    ET.SubElement(fb, "geom", {"name": "flashball", "type": "sphere", "size": "0.01", "rgba": "1 0.92 0.55 0", **vis_g})
    rng = np.random.default_rng(7)
    for k in range(N_SMOKE):
        b = ET.SubElement(world, "body", {"name": f"smoke{k}", "mocap": "true", "pos": "0 0 -50"})
        ET.SubElement(b, "geom", {"name": f"smoke{k}", "type": "sphere", "size": "0.05", "rgba": "0.3 0.3 0.3 0",
                                  **vis_g})
    palette = [(1, 0.2, 0.3), (1, 0.8, 0.1), (0.2, 0.75, 1), (0.4, 1, 0.35), (1, 0.45, 0.9), (1, 1, 1), (1, 0.55, 0.1),
               (0.6, 0.35, 1)]
    for k in range(N_CONFETTI):
        b = ET.SubElement(world, "body", {"name": f"conf{k}", "mocap": "true", "pos": "0 0 -50"})
        c = palette[k % len(palette)]
        w = 0.006 + 0.006 * rng.random()
        ET.SubElement(b, "geom", {"type": "box", "size": f"{w:.4f} {w * 0.6:.4f} 0.0004", "rgba": f"{c[0]} {c[1]} {c[2]} 1",
                                  **vis_g})
    scorch = ET.SubElement(world, "body", {"name": "scorch", "mocap": "true", "pos": "0 0 -50"})
    ET.SubElement(scorch, "geom", {"type": "cylinder", "size": "0.11 0.0008", "rgba": "0.05 0.04 0.04 0.9", **vis_g})

    # the bandit: striped jumper, black eye mask, beanie (mocap bodies: the torso and head, two legs, two arms, the hat)
    skin, black, white, jeans = "0.86 0.67 0.55 1", "0.06 0.06 0.07 1", "0.92 0.92 0.9 1", "0.2 0.26 0.4 1"
    B = ET.SubElement(world, "body", {"name": "bandit", "mocap": "true", "pos": "4 2 -50"})
    bandit_geoms = {}

    def bg(body, name, **kw):
        bandit_geoms[name] = ET.SubElement(body, "geom", {"name": f"bandit:{name}", **kw, **vis_g})

    for k in range(7):                                                    # the jumper's stripes, 0.92 … 1.48 m
        z0 = 0.92 + 0.08 * k
        bg(B, f"stripe{k}", type="cylinder", size=f"{0.155 - 0.004 * abs(k - 3):.3f} 0.04", pos=f"0 0 {z0 + 0.04:.3f}",
           rgba=black if k % 2 == 0 else white)
    bg(B, "neck", type="capsule", size="0.045 0.03", pos="0 0 1.5", rgba=skin)
    bg(B, "head", type="ellipsoid", size="0.095 0.085 0.11", pos="0 0 1.62", rgba=skin)
    bg(B, "nose", type="sphere", size="0.022", pos="-0.093 0 1.6", rgba=skin)
    bg(B, "mask", type="ellipsoid", size="0.099 0.089 0.03", pos="0 0 1.645", rgba=black)
    for sy in (-1, 1):
        bg(B, f"eye{sy}", type="sphere", size="0.017", pos=f"-0.088 {sy * 0.036} 1.648", rgba=white)
        bg(B, f"pupil{sy}", type="sphere", size="0.008", pos=f"-0.103 {sy * 0.034} 1.646", rgba=black)
    bg(B, "moustache", type="capsule", size="0.01 0.035", pos="-0.09 0 1.575", euler="1.5708 0 0", rgba=black)
    bg(B, "mouth", type="ellipsoid", size="0.006 0.018 0.004", pos="-0.094 0 1.552", rgba="0.5 0.15 0.15 1")
    bg(B, "belt", type="cylinder", size="0.16 0.02", pos="0 0 0.92", rgba="0.15 0.1 0.06 1")
    bg(B, "sack", type="ellipsoid", size="0.12 0.14 0.16", pos="0.2 0 1.25", rgba="0.6 0.5 0.32 1")
    for k in range(7):                                                    # frazzled hair, shown after the blast
        a = -1.2 + 0.4 * k
        bg(B, f"hair{k}", type="capsule", size="0.006 0.04",
           pos=f"{0.02 * math.sin(a):.3f} {0.06 * math.sin(a):.3f} {1.74 + 0.02 * math.cos(a):.3f}",
           euler=f"{a * 0.8:.3f} {0.3 * math.cos(a):.3f} 0", rgba="0.06 0.06 0.07 0")
    limbs = {}
    for side, sy in (("l", 1), ("r", -1)):
        leg = ET.SubElement(world, "body", {"name": f"bandit_leg_{side}", "mocap": "true", "pos": "4 2 -50"})
        bg(leg, f"leg_{side}", type="capsule", size="0.065", fromto="0 0 0 0 0 -0.82", rgba=jeans)
        bg(leg, f"shoe_{side}", type="box", size="0.12 0.055 0.04", pos="-0.05 0 -0.87", rgba=black)
        arm = ET.SubElement(world, "body", {"name": f"bandit_arm_{side}", "mocap": "true", "pos": "4 2 -50"})
        bg(arm, f"arm_{side}", type="capsule", size="0.045", fromto="0 0 0 0 0 -0.52", rgba=black)
        bg(arm, f"hand_{side}", type="sphere", size="0.05", pos="0 0 -0.58", rgba=skin)
        limbs[side] = (leg, arm, sy)
    hat = ET.SubElement(world, "body", {"name": "bandit_hat", "mocap": "true", "pos": "4 2 -50"})
    bg(hat, "hat", type="ellipsoid", size="0.1 0.092 0.07", pos="0 0 0", rgba="0.1 0.1 0.12 1")
    bg(hat, "hatband", type="cylinder", size="0.101 0.02", pos="0 0 -0.03", rgba="0.2 0.2 0.24 1")
    bg(hat, "pompom", type="sphere", size="0.03", pos="0 0 0.07", rgba="0.85 0.1 0.1 1")

    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    W = robot.total_mass() * 9.81
    info = {"kin": kin, "robot": robot, "tr": tr, "eye_body": dr["to_body"](dr["eye_mm"]), "dir_body": np.asarray(dr["dir"]),
            "muzzle_body": dr["to_body"](dr["muzzle_mm"]), "W": W, "xi": xi, "xf": xf, "xo": xo, "zt": zt,
            "r_w": r_w, "table_z": zt_tab, "table_x": (tx0, tx1), "lantern": lant, "face_x": face_x,
            "solids": [(-9.0, xi, STREET, 0.0), (xi, xo, zb, zt), (-9.0, xf, STREET, zb),
                       (face_x, xi, tip_z - 0.04, tip_z + 0.024)],
            "bandit_geoms": [model.geom(f"bandit:{n}").id for n in bandit_geoms],
            "tip_len": (tr.S["winch"]["hook"]["front"] + tr.S["winch"]["hook"]["pick"] * 0.6) * 1e-3}
    return model, info


# --------------------------------------------------------------------------------------------- the camera
class Cam:
    """A free camera with low-passed targets; ``cut()`` jumps to the next target (a hard cut)."""

    def __init__(self, fovy, width, height):
        import mujoco
        self.c = mujoco.MjvCamera()
        self.c.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.fovy, self.w, self.h = fovy, width, height
        self.state = None

    def set(self, lookat, dist, az, el, k=0.1):
        tgt = np.array([*lookat, dist, az, el], float)
        if self.state is None:
            self.state = tgt
        else:
            self.state = self.state + k * (tgt - self.state)
        s = self.state
        self.c.lookat[:] = s[:3]
        self.c.distance, self.c.azimuth, self.c.elevation = s[3], s[4], s[5]

    def cut(self):
        self.state = None

    def project(self, p):
        """Pixel (x, y) of world point p, or None behind the camera."""
        az, el = math.radians(self.c.azimuth), math.radians(self.c.elevation)
        f = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        pos = np.array(self.c.lookat) - self.c.distance * f
        right = np.cross(f, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, f)
        v = np.asarray(p) - pos
        depth = float(v @ f)
        if depth <= 0.01:
            return None
        sc = (self.h / 2) / math.tan(math.radians(self.fovy) / 2)
        return self.w / 2 + sc * float(v @ right) / depth, self.h / 2 - sc * float(v @ up) / depth


# --------------------------------------------------------------------------------------------- the mission
def simulate(model, info, *, render=None, pre=False):
    """Run the mission. ``pre``: only the walk and the sway (returns the hook tip's position at full sway, to place
    the parapet). ``render(t, phase, data, extra)`` is called at each video frame."""
    import mujoco
    data = mujoco.MjData(model)
    tr, kin = info["tr"], info["kin"]
    import tarantula_gaits as tg
    motion = tg.gait("wave4", kin=kin)
    names = list(kin.names)
    qadr = np.array([model.joint(n).qposadr[0] for n in names])
    dadr = np.array([model.joint(n).dofadr[0] for n in names])
    act_servo = np.array([model.actuator(f"servo:{n}").id for n in names])
    a_winch = model.actuator("winch").id
    lines = {n: model.tendon(n).id for n in ("dl_short", "dl_over", "dragline")}
    sv = tr.drive()
    kp, kd, stall = sv.kp, sv.kd, sv.stall_torque
    T = motion.period
    ts_ = np.array(list(motion.t) + [T])
    Qk = np.vstack([motion.Q, motion.Q[:1]])

    def q_ref(t):
        tt = t % T
        i = min(int(np.searchsorted(ts_, tt, side="right")) - 1, len(ts_) - 2)
        w = (tt - ts_[i]) / (ts_[i + 1] - ts_[i])
        return (1 - w) * Qk[i] + w * Qk[i + 1]

    q_stand = q_ref(0.0)
    kin.set(q_stand)
    sway = 0.055
    targets = [(kin.geom_point(f)[0], kin.geom_point(f)[1], kin.data.geom_xpos[kin.model.geom(f).id] + np.array([sway, 0, 0]))
               for f in tr.LEGS]
    q_sw, _ = kin.ik(targets, q_init=q_stand, reg=1e-7)
    q_tuck = q_stand.copy()
    for k, n in enumerate(names):                                         # the legs folded in, hanging
        if n.endswith("_lift"):
            q_tuck[k] = math.radians(48)
        elif n.endswith("_knee"):
            q_tuck[k] = math.radians(70)
        elif n.endswith("_yaw"):
            q_tuck[k] = math.radians(18 if n[1] in "12" else -12)
    fa = model.joint("body:free").qposadr[0]
    fd = model.joint("body:free").dofadr[0]
    body = model.body("body").id
    hook = model.body("harpoon").mocapid[0]
    gift = model.body("gift").mocapid[0]
    muzzle_site = model.site("muzzle").id
    W = info["W"]
    dt = model.opt.timestep
    k_rope, c_rope, F_max = 6000.0, 40.0, 100.0
    v_in, v_out = 0.047, 0.25                                             # winch reel-in, brake pay-out [m/s]

    # the hanging pose: the fairlead straight above the centre of mass, the back to the street, head down
    data.qpos[fa:fa + 7] = [0, 0, 0, 1, 0, 0, 0]
    data.qpos[qadr] = q_tuck
    mujoco.mj_forward(model, data)
    com_b = data.subtree_com[body] - data.xpos[body]
    m_b = info["muzzle_body"]
    u = (m_b - com_b) / np.linalg.norm(m_b - com_b)
    a2 = np.array([0, 0, 1.0]) - u[2] * u
    a2 /= np.linalg.norm(a2)
    A = np.column_stack([u, a2, np.cross(u, a2)])
    Bw = np.column_stack([[0, 0, 1.0], [1.0, 0, 0], np.cross([0, 0, 1.0], [1.0, 0, 0])])
    R_hang = Bw @ A.T

    # start: facing away from the parapet (yaw 180°), standing on the roof
    R0 = rot_z(math.pi)
    data.qpos[:] = model.qpos0
    data.qpos[fa:fa + 3] = [0.0, 0.0, tr.STAND_Z]
    data.qpos[fa + 3:fa + 7] = mat2quat(R0)
    data.qpos[qadr] = q_stand
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    for n in lines:
        model.tendon_rgba[lines[n]][3] = 0.0

    def body_pose():
        return data.xpos[body].copy(), data.xmat[body].reshape(3, 3).copy()

    def hook_on_robot():
        p, R = body_pose()
        return p + R @ info["eye_body"], tv._qmul(data.xquat[body].copy(), tv._quat_from_x(info["dir_body"]))

    def hook_tip():
        p, q = data.mocap_pos[hook].copy(), data.mocap_quat[hook].copy()
        return p + quat2mat(q)[:, 0] * info["tip_len"]

    def gift_on_robot():
        p, R = body_pose()
        return p + R @ np.array(GIFT_AT), data.xquat[body].copy()

    def show_line(name):
        for n, i in lines.items():
            model.tendon_rgba[i][3] = 1.0 if n == name else 0.0

    def set_kin(pos, R, q):
        data.qpos[fa:fa + 3] = pos
        data.qpos[fa + 3:fa + 7] = mat2quat(R)
        data.qpos[qadr] = q
        data.qvel[:] = 0

    def servo(qr, qdr):
        q, qd = data.qpos[qadr].copy(), data.qvel[dadr].copy()
        tq = kp * (qr - q) + kd * (qdr - qd)
        lim = np.where(tq * qd < 0, stall, sv.torque_limit(qd))
        data.ctrl[act_servo] = np.clip(tq, -lim, lim) + kd * qd

    st = {"t": -0.6, "next_frame": -0.6, "fps_dt": None, "tension": 0.0, "L": None, "hook_fixed": None,
          "gift": "robot", "gift_pose": None, "timer_set": None, "t_boom": None, "phase": "", "caption": "",
          "speed": 1.0}
    extra = {}

    solid_ids = {model.geom(n).id for n in ("coping", "facade", "nailer", "building")}
    robot_b = {i for i in range(model.nbody) if model.body_rootid[i] == body}
    pen = {"worst": 0.0, "where": ""}

    def penetration():
        """The deepest overlap [m] of the robot's collision geoms with the stone, and which geom."""
        worst, where = 0.0, ""
        for c in data.contact[:data.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in solid_ids) == (g2 in solid_ids):
                continue
            g = g2 if g1 in solid_ids else g1
            if model.geom_bodyid[g] in robot_b and c.dist < worst:
                worst, where = float(c.dist), model.geom(g).name
        return worst, where

    def frame(phase, caption, speed=1.0, force=False):
        d_, w_ = penetration()
        if d_ < pen["worst"]:
            pen["worst"], pen["where"] = d_, w_
        if caption != st["caption"]:
            if st["caption"]:
                print(f"          worst overlap with the stone {-pen['worst'] * 1000:5.1f} mm ({pen['where']})", flush=True)
            pen["worst"], pen["where"] = 0.0, ""
            print(f"  {st['t']:6.2f} s  {caption or phase:70s} body {np.round(data.xpos[body] * 1000).astype(int)} mm"
                  f"  line {st['tension']:5.1f} N", flush=True)
        st["phase"], st["caption"] = phase, caption
        if render is None:
            return
        if st["t"] + 1e-9 >= st["next_frame"] or force:
            st["next_frame"] = st["t"] + speed / render.fps
            if st["gift"] == "robot":
                data.mocap_pos[gift], data.mocap_quat[gift] = gift_on_robot()
            render(st["t"], phase, caption, data, st, extra, speed)

    def physics(duration, phase, caption, ctrl, speed=1.0, until=None):
        t_end = st["t"] + duration
        while st["t"] < t_end - 1e-9:
            tau_frac = 1.0 - (t_end - st["t"]) / duration
            qr, qdr = ctrl(tau_frac, st["t"])
            servo(qr, qdr)
            if st["hook_fixed"] is None:
                data.mocap_pos[hook], data.mocap_quat[hook] = hook_on_robot()
            else:
                data.mocap_pos[hook], data.mocap_quat[hook] = st["hook_fixed"]
            if st["gift"] == "robot":
                data.mocap_pos[gift], data.mocap_quat[gift] = gift_on_robot()
            # the rope (on the dragline path only): pulls when taut, the brake / winch set the paid-out length L
            if st["L"] is not None:
                i = lines["dragline"]
                length, vel = float(data.ten_length[i]), float(data.ten_velocity[i])
                stretch = length - st["L"]
                st["tension"] = float(np.clip(k_rope * stretch + c_rope * max(vel, 0.0), 0, F_max)) if stretch > 0 else 0.0
                data.ctrl[a_winch] = -st["tension"]
            else:
                st["tension"] = 0.0
                data.ctrl[a_winch] = 0.0
            mujoco.mj_step(model, data)
            st["t"] += dt
            frame(phase, caption, speed)
            if until is not None and until():
                break

    # ---- the feet on the solid scenery in the scripted parts: the boxes (x0, x1, z0, z1), infinite along y
    boxes = info["solids"]
    foot_r = 0.010
    kin.set(q_stand)
    base_k = kin.data.xpos[kin.body("body")].copy()
    gid = {f: kin.model.geom(f).id for f in tr.LEGS}
    fz_stand = {f: float(kin.data.geom_xpos[gid[f]][2] - base_k[2]) for f in tr.LEGS}
    pts = {f: kin.geom_point(f) for f in tr.LEGS}

    def inside(p, eps=1e-6):
        return any(x0 + eps < p[0] < x1 - eps and z0 + eps < p[2] < z1 - eps for x0, x1, z0, z1 in boxes)

    def ray(p, d, L):
        """The first hit of p + t d (t ∈ [0, L]) on the solids: (point, normal) or None."""
        best = None
        for x0, x1, z0, z1 in boxes:
            tn, tf, nn = -np.inf, np.inf, None
            ok = True
            for ax, lo, hi in ((0, x0, x1), (2, z0, z1)):
                if abs(d[ax]) < 1e-9:
                    if not lo <= p[ax] <= hi:
                        ok = False
                    continue
                ta, tb = (lo - p[ax]) / d[ax], (hi - p[ax]) / d[ax]
                n_in = np.zeros(3)
                n_in[ax] = -1.0 if d[ax] > 0 else 1.0
                if ta > tb:
                    ta, tb = tb, ta
                if ta > tn:
                    tn, nn = ta, n_in
                tf = min(tf, tb)
            if not ok or nn is None or tn > tf or not 0.0 <= tn <= L:
                continue
            q_ = p + tn * d
            if inside(q_ + 1e-4 * nn):
                continue
            if best is None or tn < best[0]:
                best = (tn, q_, nn)
        return None if best is None else best[1:]

    def closest(p):
        """The nearest point on the solids' outer surface: (point, outward normal, distance)."""
        best = None
        for x0, x1, z0, z1 in boxes:
            if x0 <= p[0] <= x1 and z0 <= p[2] <= z1:                     # inside: out through the nearest face
                cands = [(p[0] - x0, 0, x0, -1), (x1 - p[0], 0, x1, 1), (p[2] - z0, 2, z0, -1), (z1 - p[2], 2, z1, 1)]
                for dist, ax, v, sg in sorted(cands):
                    c = p.copy()
                    c[ax] = v
                    n = np.zeros(3)
                    n[ax] = sg
                    if not inside(c + 1e-4 * n):
                        if best is None or -dist < best[2]:
                            best = (c, n, -dist)
                        break
                continue
            c = np.array([min(max(p[0], x0), x1), p[1], min(max(p[2], z0), z1)])
            v = p - c
            dist = float(np.linalg.norm(v))
            n = v / dist if dist > 1e-9 else np.array([0, 0, 1.0])
            if inside(c + 1e-4 * n):
                continue
            if best is None or dist < best[2]:
                best = (c, n, dist)
        return best

    cache = {}
    stone = {model.geom(n).id for n in ("coping", "facade", "nailer", "building")}
    foot_geoms = {model.geom(f).id for f in tr.LEGS}

    def depenetrate(pos, R, q, targets, rounds=5):
        """Push the legs' segments out of the stone: each leg geom MuJoCo finds overlapping it gets an extra IK
        target, its contact point moved out along the contact normal (the feet keep their ``targets``)."""
        for _ in range(rounds):
            set_kin(pos, R, q)
            mujoco.mj_forward(model, data)
            extra = []
            for c in data.contact[:data.ncon]:
                g1, g2 = int(c.geom1), int(c.geom2)
                if (g1 in stone) == (g2 in stone) or c.dist > -0.001:
                    continue
                g, n_out = (g2, np.array(c.frame[:3])) if g1 in stone else (g1, -np.array(c.frame[:3]))
                b = int(model.geom_bodyid[g])
                if g in foot_geoms or b == body or model.body_rootid[b] != body:
                    continue
                local = data.xmat[b].reshape(3, 3).T @ (c.pos - data.xpos[b])
                goal = c.pos + n_out * (-c.dist + 0.007)
                extra.append((kin.body(model.body(b).name), local, base_k + R.T @ (goal - pos)))
            if not extra:
                break
            q, _ = kin.ik(targets + extra, q_init=q, q_ref=q_stand, reg=1e-5, iters=60)
        return q

    def feet_on_solids(pos, R, q_nom, snap):
        """The joint angles with every foot on the solid surface under it (a ray down the body's own vertical; near
        an edge, the nearest face; never inside), the gait's swing lift kept off the surface."""
        key = (tuple(np.round(pos, 6)), tuple(np.round(R, 6).ravel()), tuple(np.round(q_nom, 6)), snap)
        if key in cache:
            return cache[key]
        kin.set(q_nom)
        targets = []
        for f in tr.LEGS:
            fb = kin.data.geom_xpos[gid[f]] - base_k
            lift = max(float(fb[2]) - fz_stand[f], 0.0)
            cand = pos + R @ fb
            tgt = None
            if snap > 0:
                p0 = pos + R @ np.array([fb[0], fb[1], 0.0])
                if not inside(p0):
                    hit = ray(p0, -R[:, 2], tr.STAND_Z + 0.08)
                    if hit is not None:
                        tgt = hit[0] + hit[1] * (foot_r + lift)
            if tgt is None:
                c, n, dist = closest(cand)
                if dist < max(snap, foot_r):
                    tgt = c + n * (foot_r + (lift if snap > 0 else 0.0))
                else:
                    tgt = cand
            targets.append((pts[f][0], pts[f][1], base_k + R.T @ (tgt - pos)))
        q, _ = kin.ik(targets, q_init=cache.get("last", q_nom), q_ref=q_nom, reg=1e-5, iters=80)
        cache["last"] = q
        q = depenetrate(pos, R, q, targets)
        cache[key] = q
        return q

    def kinematic(duration, phase, caption, path, speed=1.0, snap=0.07):
        """``path(u) -> (pos, R, q)`` for u ∈ [0, 1]; steps at the frame rate (or 20 ms without rendering). The feet
        are put on the solid scenery (``snap``: how far a foot reaches for a surface; 0: only kept out of it)."""
        t0 = st["t"]
        step = (speed / render.fps) if render is not None else 0.02
        while st["t"] < t0 + duration - 1e-9:
            st["t"] = min(st["t"] + step, t0 + duration)
            u = (st["t"] - t0) / duration
            pos, R, q = path(u)
            set_kin(pos, R, q if snap < 0 else feet_on_solids(np.asarray(pos, float), np.asarray(R, float), q, snap))
            if st["hook_fixed"] is not None:
                data.mocap_pos[hook], data.mocap_quat[hook] = st["hook_fixed"]
            mujoco.mj_forward(model, data)
            if st["gift"] == "robot":
                data.mocap_pos[gift], data.mocap_quat[gift] = gift_on_robot()
            st["next_frame"] = st["t"]
            frame(phase, caption, speed)

    # ---- the free gait of the scripted climbs: footholds fixed in the world, one pair of legs stepping at a time
    groups = [list(g) for g in tg.GAITS["wave4"][0]]
    fb_stand = {}
    kin.set(q_stand)
    for f in tr.LEGS:
        fb_stand[f] = kin.data.geom_xpos[gid[f]] - base_k
    foot_gid = {f: model.geom(f).id for f in tr.LEGS}

    def foothold(f, pos, R, reach=0.12):
        """Where leg f wants to stand with the body at (pos, R): under its nominal foot along the body's vertical,
        else the nearest surface within ``reach``, else in the air. (foot centre, surface normal or None)."""
        fb = fb_stand[f]
        p0 = pos + R @ np.array([fb[0], fb[1], 0.0])
        if not inside(p0):
            hit = ray(p0, -R[:, 2], tr.STAND_Z + 0.12)
            if hit is not None:
                return hit[0] + hit[1] * foot_r, hit[1]
        cand = pos + R @ fb
        c, n, dist = closest(cand)
        if dist < reach:
            return c + n * foot_r, n
        return cand, None

    def margin(feet):
        """The static stability margin [m]: the centre of mass's distance inside the planted feet's support polygon,
        seen along gravity (negative: outside)."""
        P = np.array([feet[f][:2] for f in feet])
        if len(P) < 3:
            return -1.0
        P = P[np.lexsort((P[:, 1], P[:, 0]))]
        cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        lo, hi = [], []
        for pt in P:
            while len(lo) >= 2 and cross(lo[-2], lo[-1], pt) <= 0:
                lo.pop()
            lo.append(pt)
        for pt in P[::-1]:
            while len(hi) >= 2 and cross(hi[-2], hi[-1], pt) <= 0:
                hi.pop()
            hi.append(pt)
        hull = lo[:-1] + hi[:-1]
        c = data.subtree_com[body][:2]
        m_ = np.inf
        for a, b in itertools.pairwise([*hull, hull[0]]):
            e = b - a
            m_ = min(m_, float(cross(a, b, c)) / (np.linalg.norm(e) + 1e-12))
        return m_

    def freegait(phase, caption, path, v=0.06, w=0.4, t_swing=0.3, reach=0.07, settle=0.5, look=0.45):
        """Walk the body along ``path(u) -> (pos, R)`` with a free gait: each foot stays where it stands; the body
        advances (``v`` m/s, ``w`` rad/s) only while every planted foot is within ``reach`` of where it wants to
        be; the wave gait's pairs step one pair at a time — the most strained pair first — lifting off along the
        surface normal and setting down on the foothold ahead. Returns the last joint angles."""
        us = np.linspace(0.0, 1.0, 241)
        poses = [path(u) for u in us]
        cost = [0.0]
        for (p0, R0), (p1, R1) in itertools.pairwise(poses):
            ang = math.acos(min(1.0, max(-1.0, (np.trace(R0.T @ R1) - 1) / 2)))
            cost.append(cost[-1] + max(np.linalg.norm(p1 - p0) / v, ang / w))
        cost = np.array(cost)
        C = cost[-1]

        def pose(sc):
            u = float(np.interp(sc, cost, us))
            return path(u)

        tick = (1.0 / render.fps) if render is not None else 0.033
        F = {f: data.geom_xpos[foot_gid[f]].copy() for f in tr.LEGS}
        Nn = {f: closest(F[f])[1] for f in tr.LEGS}
        swing, last, sc, quiet, blocked = {}, None, 0.0, 0.0, 0.0
        q = data.qpos[qadr].copy()

        def solve(pos, R, feet_now, q0):
            targets = [(pts[f][0], pts[f][1], base_k + R.T @ (feet_now[f] - pos)) for f in tr.LEGS]
            return kin.ik(targets, q_init=q0, q_ref=q_stand, reg=1e-5, iters=60)

        def stretch(f, pos, R):
            """How far foot f is from the best foothold it could have now, along the body's plane (heights are free:
            the legs take up a step)."""
            d = R.T @ (F[f] - foothold(f, pos, R)[0])
            return math.hypot(d[0], d[1])

        for _ in range(int(60 / tick)):
            pos, R = pose(sc)
            if not swing:                                                 # pick the next pair to step
                ahead = pose(min(C, sc + look))
                want_a = {f: foothold(f, *ahead) for f in tr.LEGS}
                need = {k: sum(np.linalg.norm(F[f] - want_a[f][0]) for f in g) for k, g in enumerate(groups)}
                k = max((k for k in need if k != last), key=lambda k: need[k])
                if need[k] <= 0.02 and blocked > 0.0:                    # stuck: the most stretched pair steps
                    k = max(range(len(groups)), key=lambda k: max(stretch(f, pos, R) for f in groups[k]))
                    want_a = {f: foothold(f, pos, R) for f in tr.LEGS}
                if need[k] > 0.02 or blocked > 0.0:
                    last = k
                    for f in groups[k]:
                        nb = want_a[f][1] if want_a[f][1] is not None else Nn[f]
                        swing[f] = (st["t"], F[f].copy(), want_a[f][0].copy(), Nn[f], nb)
                    quiet = 0.0
                elif sc >= C:
                    quiet += tick
                    if quiet >= settle:
                        break
            feet_now = {}
            for f in tr.LEGS:
                if f in swing:
                    t0, a, b, na, nb = swing[f]
                    u = min((st["t"] + tick - t0) / t_swing, 1.0)
                    n_mid = na + nb
                    n_mid = n_mid / (np.linalg.norm(n_mid) + 1e-9)
                    hgt = 0.028 + 0.03 * (1.0 - float(na @ nb))
                    pt = a + sm(u) * (b - a) + n_mid * hgt * math.sin(math.pi * min(u * 1.1, 1.0))
                    if inside(pt):
                        c, n, _ = closest(pt)
                        pt = c + n * foot_r
                    feet_now[f] = pt
                    if u >= 1.0:
                        F[f], Nn[f] = b, nb
                        del swing[f]
                else:
                    feet_now[f] = F[f]
            # the body advances if the planted legs still reach (the IK solves, no leg over-stretched)
            q_new = None
            if sc < C:
                sc_try = min(C, sc + tick)
                p_try, R_try = pose(sc_try)
                q_try, _ = solve(p_try, R_try, feet_now, q)
                planted = [f for f in tr.LEGS if f not in swing]
                if max(stretch(f, p_try, R_try) for f in planted) < reach or blocked > 1.5:
                    sc, pos, R, q_new, blocked = sc_try, p_try, R_try, q_try, 0.0
                else:
                    blocked += tick
            if q_new is None:
                q_new, _ = solve(pos, R, feet_now, q)
            q = depenetrate(pos, R, q_new, [(pts[f][0], pts[f][1], base_k + R.T @ (feet_now[f] - pos)) for f in tr.LEGS])
            strain = {f: stretch(f, pos, R) for f in tr.LEGS}
            st["t"] += tick
            set_kin(pos, R, q)
            if st["hook_fixed"] is not None:
                data.mocap_pos[hook], data.mocap_quat[hook] = st["hook_fixed"]
            mujoco.mj_forward(model, data)
            if st["gift"] == "robot":
                data.mocap_pos[gift], data.mocap_quat[gift] = gift_on_robot()
            if os.environ.get("TM_DEBUG") and round(st["t"] / tick) % 10 == 0:
                print(f"    fg t {st['t']:6.2f} s {sc:5.2f}/{C:5.2f} swing {sorted(swing)} strain "
                      + " ".join(f"{f}:{strain[f] * 1000:.0f}" for f in tr.LEGS), flush=True)
                if st["t"] > float(os.environ["TM_DEBUG"]):
                    raise SystemExit
            down = {f: F[f] for f in tr.LEGS if f not in swing}
            st["feet"] = (len(down), margin(down) if R[2, 2] > 0.8 else None)
            st["next_frame"] = st["t"]
            frame(phase, caption)
        st["feet"] = None
        return q

    # ---- 1. settle, then creep tail first to the parapet (the reversed wave gait)
    cyc = 3
    physics(0.6, "settle", "", lambda u, t: (q_stand, np.zeros_like(q_stand)))
    t_w = st["t"]
    physics(cyc * T, "walk", "STEP 1 · CREEP TO THE EDGE — tail first, on its own eight feet",
            lambda u, t: (q_ref(cyc * T - (t - t_w)), -(q_ref(cyc * T - (t - t_w) + dt) - q_ref(cyc * T - (t - t_w))) / dt))
    physics(0.4, "plant", "STEP 2 · SET THE ANCHOR HOOK", lambda u, t: (q_stand, np.zeros_like(q_stand)))
    physics(1.2, "plant", "STEP 2 · SET THE ANCHOR HOOK — sway back: the pick bites the timber",
            lambda u, t: (q_stand + sm(u / 0.85) * (q_sw - q_stand), np.zeros_like(q_stand)))
    if pre:
        return hook_tip(), body_pose()
    st["hook_fixed"] = (data.mocap_pos[hook].copy(), data.mocap_quat[hook].copy())
    show_line("dl_short")
    physics(0.4, "plant", "STEP 2 · SET THE ANCHOR HOOK — the cradle's latch opens", lambda u, t: (q_sw, np.zeros_like(q_sw)))
    physics(1.0, "plant", "STEP 2 · SET THE ANCHOR HOOK — sway forward: the hook stays",
            lambda u, t: (q_sw + sm(u) * (q_stand - q_sw), np.zeros_like(q_stand)))

    # ---- 2. over the coping (scripted): climb, turn round, over the outer edge head first, let go
    pA, RA = body_pose()
    h = tr.STAND_Z
    xi, xo, zt = info["xi"], info["xo"], info["zt"]
    y = pA[1]
    pB = np.array([xi + 0.11, y, zt + h])
    pC = np.array([xo - 0.13, y, zt + h])
    RC = rot_z(0.0) @ np.eye(3)
    Rface = rot_y(math.pi / 2)                                            # head down, belly to the coping's face
    pD = np.array([xo + h, y, zt - 0.16])
    m_hang = np.array([xo + info["r_w"], y, -0.11])                      # the fairlead, hanging clear of the coping
    pE = m_hang - R_hang @ m_b

    def bez(p0, p1, p2, u):
        return (1 - u) ** 2 * p0 + 2 * (1 - u) * u * p1 + u ** 2 * p2

    def bez3(p0, p1, p2, p3, u):
        return (1 - u) ** 3 * p0 + 3 * (1 - u) ** 2 * u * p1 + 3 * (1 - u) * u ** 2 * p2 + u ** 3 * p3

    d4 = 1.4
    show_line("dl_over")
    pA2 = pA + np.array([-0.17, 0.0, 0.0])                                # clear of the parapet, room to turn
    p_off = lambda u: (pA + sm(u) * (pA2 - pA), slerp_R(RA, rot_z(math.pi), sm(u)))
    p_turn = lambda u: (pA2, rot_z(math.pi * (1 - sm(u))))
    p_up = lambda u: (bez(pA2, np.array([xi - 0.07, y, zt + h + 0.02]), pB, sm(u)), rot_y(-0.35 * math.sin(math.pi * u)))
    p_across = lambda u: (pB + sm(u) * (pC - pB), RC)
    p_over = lambda u: (bez(pC, np.array([xo + 0.08, y, zt + h * 1.5]), pD, sm(u)), slerp_R(RC, Rface, sm(u)))
    freegait("over", "STEP 3 · OVER THE EDGE — step away from the parapet", p_off)
    freegait("over", "STEP 3 · OVER THE EDGE — turn round on the roof, a pair of feet at a time", p_turn)
    freegait("over", "STEP 3 · OVER THE EDGE — head first up onto the coping", p_up)
    freegait("over", "STEP 3 · OVER THE EDGE — across the coping", p_across)
    qD = freegait("over", "STEP 3 · OVER THE EDGE — head first down the face", p_over)
    show_line("dragline")
    kinematic(d4, "over", "STEP 3 · OVER THE EDGE — let go: hanging on the dragline",
              lambda u: (pD + sm(u) * (pE - pD) + np.array([0.09 * math.sin(math.pi * u), 0, 0]),
                         slerp_R(Rface, R_hang, sm(u)), qD + sm(u) * (q_tuck - qD)), snap=0.0)

    pH = np.array([info["face_x"] - 0.24, y, tr.STAND_Z - 0.021])
    p_back1 = lambda u: (bez(pD, np.array([xo + 0.08, y, zt + h * 1.5]), pC, sm(u)), slerp_R(Rface, RC, sm(u)))
    # back off the parapet tail first: along the top to the inner edge, then tail down onto the roof
    p_back2 = lambda u: (bez3(pC, np.array([xi + 0.02, y, zt + h + 0.01]), np.array([xi - 0.2, y, zt + h * 1.2]), pH,
                              sm(u)), rot_y(-0.3 * math.sin(math.pi * min(u * 1.25, 1.0))))

    def est(path, v=0.065, w=0.6):
        c, (p0, R0) = 0.0, path(0.0)
        for u in np.linspace(0, 1, 121)[1:]:
            p1, R1 = path(u)
            ang = math.acos(min(1.0, max(-1.0, (np.trace(R0.T @ R1) - 1) / 2)))
            c += max(np.linalg.norm(p1 - p0) / v, ang / w)
            p0, R0 = p1, R1
        return c

    # ---- 3. down the line on the brake (physics: the robot hangs from the rope)
    model.dof_damping[fd + 3:fd + 6] = 0.004                              # the air and the line's twist
    mujoco.mj_forward(model, data)
    st["L"] = float(data.ten_length[lines["dragline"]]) - W / k_rope        # taut: the rope takes the weight
    hold = lambda u, t: (q_tuck, np.zeros_like(q_tuck))
    print(f"tarantula_mission: hanging, {data.ncon} contacts", flush=True)
    physics(1.0, "hang", "STEP 4 · DROP ON THE LINE", hold)
    ztab = info["table_z"]

    def gift_low():
        return gift_on_robot()[0][2] - GIFT_HALF[0]                                        # body x is down: the box's half along x

    def pay_out(u, t):
        gap = gift_low() - ztab
        v = min(v_out, 0.6 * max(gap - 0.004, 0.0) + 0.004) if gap > 0.002 else 0.0
        v *= sm((t - t_desc) / 1.2)
        st["L"] += v * dt
        return q_tuck, np.zeros_like(q_tuck)

    t_desc = st["t"]
    physics(30.0, "descend", "STEP 4 · DROP ON THE LINE — the brake pays out, no motor running: silent",
            pay_out, until=lambda: gift_low() - ztab < 0.0025 and abs(data.qvel[fd + 2]) < 0.01)
    # ---- 4. the delivery: the pincers let go, the fang presses the timer
    p, _ = gift_on_robot()
    Rb = body_pose()[1]
    gy = Rb[:, 1] - Rb[2, 1] * np.array([0, 0, 1.0])
    gy /= np.linalg.norm(gy)
    gx = np.array([0, 0, -1.0])                                           # set down upright: the box's x down
    Rg = np.column_stack([gx, gy, np.cross(gx, gy)])
    st["gift"], st["gift_pose"] = "table", (np.array([p[0], p[1], ztab + GIFT_HALF[0]]), mat2quat(Rg))
    data.mocap_pos[gift], data.mocap_quat[gift] = st["gift_pose"]
    physics(1.0, "deliver", "STEP 5 · DELIVER THE GIFT — the tarsal pincers open", hold)
    dist_up = m_hang[2] - data.site_xpos[muzzle_site][2]
    t_up_est = dist_up / v_in * 1.03 + 1.0
    t_back = 1.4 + 1.6 * (est(p_back1) + est(p_back2)) + 1.5             # the climb back: the free gait's pace
    t_after = t_up_est + t_back + 1.0 + 5.0 + 2.6                          # climb, back over, hide, stroll, lean in
    st["timer_set"], st["t_boom"] = st["t"] + 0.6, st["t"] + 0.6 + math.ceil(t_after)
    physics(1.6, "timer", "STEP 5 · SET THE TIMER — a fang presses the button", hold)

    # ---- 5. back up on the winch (physics, time-lapsed)
    def reel(u, t):
        st["L"] -= v_in * dt
        return q_tuck, np.zeros_like(q_tuck)

    physics(120.0, "ascend", f"STEP 6 · WINCH BACK UP — 47 mm/s (×{SPEEDUP:g})", reel, speed=SPEEDUP,
            until=lambda: data.site_xpos[muzzle_site][2] >= m_hang[2] - 0.003)
    st["L"], st["tension"] = None, 0.0
    model.dof_damping[fd + 3:fd + 6] = 0.0
    # ---- 6. back over and hide (scripted), tail first, behind the parapet facing the edge
    pE2, RE2 = body_pose()
    qE2 = data.qpos[qadr].copy()
    kinematic(d4, "back", "STEP 7 · HIDE — up the face, tail first",
              lambda u: (bez(pE2, np.array([xo + 0.17, y, 0.5 * (pE2[2] + pD[2])]), pD, sm(u)), slerp_R(RE2, Rface, sm(u)),
                         qE2 + sm(u) * (qD - qE2)),
              snap=0.03)
    show_line("dl_over")
    freegait("back", "STEP 7 · HIDE — onto the coping, a pair of feet at a time", p_back1)
    show_line("dl_short")
    q_hide = freegait("back", "STEP 7 · HIDE — down behind the parapet", p_back2)
    pHide, RHide = body_pose()
    t_walk = max(st["t"] + 0.6, st["t_boom"] - 7.6)
    kinematic(t_walk - st["t"], "hide", "STEP 7 · HIDE — safe behind the parapet", lambda u: (pHide, RHide, q_hide),
              snap=-1)
    # ---- 7. the bandit (props), the timer, the blast
    extra["bandit_t0"] = st["t"]
    kinematic(st["t_boom"] - st["t"], "bandit", "TARGET APPROACHING", lambda u: (pHide, RHide, q_hide), snap=-1)
    extra["boom_t0"] = st["t"]
    pPeek = np.array([info["face_x"] - 0.17, y, tr.STAND_Z + 0.085])
    extra["peek"] = pPeek
    kinematic(3.6, "boom", "KABOOM", lambda u: (pHide, RHide, q_hide), snap=-1)
    t_peek = st["t"]
    q_peek = freegait("end", "MISSION ACCOMPLISHED",
                      lambda u: (pHide + sm(u) * (pPeek - pHide), slerp_R(RHide, rot_y(-0.62), sm(u))), v=0.05, w=0.5)
    pP, RP = body_pose()
    kinematic(max(5.2 - (st["t"] - t_peek), 2.0), "end", "MISSION ACCOMPLISHED", lambda u: (pP, RP, q_peek), snap=-1)
    return st


def slerp_R(R0, R1, u):
    return quat2mat(slerp(mat2quat(R0), mat2quat(R1), u))


def motion_crouch(tg, kin):
    """The crouched standing pose (the body 21 mm lower: ``tarantula_gaits.POSTURES``)."""
    return tg.gait("wave4", kin=kin, body_drop_mm=tg.POSTURES["crouched"]).Q[0]


# --------------------------------------------------------------------------------------------- the props
class Props:
    """The bandit's stroll, the gift's LED, the blast: flash, smoke, confetti, soot, the hat."""

    def __init__(self, model, info, seed=11):
        self.m, self.info = model, info
        self.rng = np.random.default_rng(seed)
        self.mid = {n: model.body(n).mocapid[0] for n in ("bandit", "bandit_leg_l", "bandit_leg_r", "bandit_arm_l",
                                                          "bandit_arm_r", "bandit_hat", "flashball", "scorch", "gift")}
        self.conf = [model.body(f"conf{k}").mocapid[0] for k in range(N_CONFETTI)]
        self.smoke = [model.body(f"smoke{k}").mocapid[0] for k in range(N_SMOKE)]
        self.smoke_g = [model.geom(f"smoke{k}").id for k in range(N_SMOKE)]
        self.flash_g = model.geom("flashball").id
        self.led = model.material("led").id
        self.light = model.light("flash").id
        self.gift_g = [i for i in range(model.ngeom) if model.geom_bodyid[i] == model.body("gift").id]
        self.hair = [model.geom(f"bandit:hair{k}").id for k in range(7)]
        self.eyes = [model.geom(f"bandit:eye{s}").id for s in (-1, 1)]
        self.bandit_rgba0 = {i: model.geom_rgba[i].copy() for i in info["bandit_geoms"]}
        x1 = info["table_x"][1]
        tcx = (info["table_x"][0] + x1) / 2
        self.stand = np.array([tcx - 0.05, 0.42 + 0.27, info["table_z"] - TABLE_H + 0.88])
        self.start = np.array([x1 + 1.3, 3.2, self.stand[2]])
        self.cv = None
        self.led_on(False)

    def bandit(self, data, t, t0, t_boom):
        """Stroll in (5 s), stop, lean in towards the gift; after the blast: stiff, soot-black, the hat up and back."""
        walk = 5.0
        u = min(max((t - t0) / walk, 0.0), 1.0)
        p = self.start + (0.3 * sm(u) + 0.7 * u) * (self.stand - self.start)
        d = self.stand - self.start
        h_walk = math.atan2(d[1], d[0])                                   # facing the way he walks …
        heading = h_walk + sm((u - 0.85) / 0.15) * (-math.pi / 2 - h_walk)  # … then the gift (-y)
        phase = (t - t0) * 2 * math.pi / 1.05
        swing = 0.42 * math.sin(phase) * (1.0 - sm((u - 0.88) / 0.12))
        bob = 0.018 * abs(math.sin(phase)) * (1.0 - sm((u - 0.88) / 0.12))
        lean = 0.22 * sm((t - t0 - walk) / 1.2)
        boom = t_boom is not None and t >= t_boom
        s = t - t_boom if boom else 0.0
        if boom:
            lean = 0.22 * (1 - sm(s / 0.25)) - 0.08 * sm(s / 0.25)
        Rh = rot_z(heading - math.pi)                                    # the figure faces -x at heading π
        Rl = Rh @ rot_y(-lean)                                           # leaning about the hip
        hip = p + np.array([0, 0, bob])
        data.mocap_pos[self.mid["bandit"]] = hip - Rl @ np.array([0, 0, 0.88])
        data.mocap_quat[self.mid["bandit"]] = mat2quat(Rl)
        arm_out = 0.55 * sm(s / 0.2) if boom else 0.0
        for side, sy in (("l", 1), ("r", -1)):
            sw = swing * sy
            data.mocap_pos[self.mid[f"bandit_leg_{side}"]] = hip + Rh @ np.array([0, sy * 0.08, 0.0])
            data.mocap_quat[self.mid[f"bandit_leg_{side}"]] = mat2quat(Rh @ rot_y(sw))
            c, s_ = math.cos(sy * arm_out), math.sin(sy * arm_out)
            Ra = Rl @ rot_y(-sw * 0.8) @ np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]])
            data.mocap_pos[self.mid[f"bandit_arm_{side}"]] = hip + Rl @ np.array([0, sy * 0.2, 0.58])
            data.mocap_quat[self.mid[f"bandit_arm_{side}"]] = mat2quat(Ra)
        head_top = hip + Rl @ np.array([0, 0, 0.83])
        hat_p, hat_R = head_top, Rl
        if boom:
            if s < 1.6:                                                   # up and back down, spinning
                hat_p = head_top + np.array([0.0, 0.0, max(4.5 * s - 6.9 * s * s, 0.0)])
                hat_R = Rl @ rot_z(9.0 * s) @ rot_y(0.4 * math.sin(6 * s))
            else:                                                         # landed askew
                hat_p = head_top + Rl @ np.array([0.01, 0.03, 0.0])
                hat_R = Rl @ rot_y(0.35)
            k = sm(s / 0.12)
            for i in self.info["bandit_geoms"]:                          # soot
                rgba = self.bandit_rgba0[i]
                tgt = np.array([0.92, 0.92, 0.9, rgba[3]]) if i in self.eyes else np.array([0.05, 0.045, 0.045, rgba[3]])
                self.m.geom_rgba[i] = (1 - k) * rgba + k * tgt
            for i in self.hair:
                self.m.geom_rgba[i] = [0.04, 0.04, 0.04, 1.0]
            blink = s > 1.0 and int(s * 10) % 17 == 0
            for i in self.eyes:
                self.m.geom_rgba[i][3] = 0.0 if blink else 1.0
        data.mocap_pos[self.mid["bandit_hat"]] = hat_p
        data.mocap_quat[self.mid["bandit_hat"]] = mat2quat(hat_R)

    def led_on(self, on):
        self.m.mat_emission[self.led] = 1.0 if on else 0.0
        self.m.mat_rgba[self.led] = [1, 0.12, 0.08, 1] if on else [0.35, 0.05, 0.04, 1]

    def blast(self, data, t, t_boom, gift_pos, dt):
        s = t - t_boom
        if s < 0:
            return
        if self.cv is None:                                               # the burst: confetti up and out
            n = N_CONFETTI
            az = self.rng.uniform(0, 2 * math.pi, n)
            el = self.rng.uniform(0.35, 1.45, n)
            sp = self.rng.uniform(2.5, 6.5, n)
            self.cv = np.column_stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]) * sp[:, None]
            self.cp = np.tile(gift_pos, (n, 1)) + self.rng.normal(0, 0.015, (n, 3))
            self.cw = self.rng.normal(0, 14, (n, 3))
            self.cq = np.tile([1.0, 0, 0, 0], (n, 1))
            self.sp0 = gift_pos + self.rng.normal(0, 0.05, (N_SMOKE, 3))
            self.sv = self.rng.normal(0, 0.12, (N_SMOKE, 3)) + np.array([0.08, 0, 0.3])
            self.t_last = s
            for i in self.gift_g:                                         # the gift is gone
                self.m.geom_rgba[i][3] = 0.0
            data.mocap_pos[self.mid["scorch"]] = gift_pos - np.array([0, 0, GIFT_HALF[0] - 0.0009])
        h = max(s - self.t_last, 0.0)
        self.t_last = s
        n_sub = max(1, int(h / 0.004))
        for _ in range(n_sub):                                            # drag: confetti flutters (~1.1 m/s terminal)
            hh = h / n_sub
            v = self.cv
            spd = np.linalg.norm(v, axis=1, keepdims=True)
            self.cv = v + hh * (np.array([0, 0, -9.81]) - 8.0 * spd * v + self.rng.normal(0, 4.0, v.shape))
            self.cp = self.cp + hh * self.cv
            floor = np.where((self.cp[:, 0] > self.info["table_x"][0]) & (self.cp[:, 0] < self.info["table_x"][1])
                             & (np.abs(self.cp[:, 1]) < 0.42) & (self.cp[:, 2] > self.info["table_z"] - 0.05),
                             self.info["table_z"] + 0.001, STREET + 0.001)
            self.cp[:, 0] = np.maximum(self.cp[:, 0], self.info["xf"] + 0.002)   # not through the facade
            low = self.cp[:, 2] < floor
            self.cp[low, 2] = floor[low]
            self.cv[low] = 0.0
            self.cw[low] *= 0.0
        for k in range(N_CONFETTI):
            w = self.cw[k] * h
            ang = np.linalg.norm(w)
            if ang > 1e-9:
                dq = np.array([math.cos(ang / 2), *(math.sin(ang / 2) * w / ang)])
                self.cq[k] = tv._qmul(dq, self.cq[k])
            data.mocap_pos[self.conf[k]] = self.cp[k]
            data.mocap_quat[self.conf[k]] = self.cq[k]
        # the flash and the light
        r = 0.04 + 0.3 * sm(s / 0.1)
        a = max(0.0, 1.0 - s / 0.45)
        data.mocap_pos[self.mid["flashball"]] = gift_pos
        self.m.geom_size[self.flash_g][0] = r
        self.m.geom_rgba[self.flash_g] = [1, 0.95 - 0.35 * min(s / 0.45, 1), 0.6 - 0.4 * min(s / 0.3, 1), 0.8 * a]
        f = 6.0 * math.exp(-s / 0.18)
        self.m.light_diffuse[self.light] = [f, 0.8 * f, 0.45 * f]
        # the smoke: puffs drifting up, growing, fading
        for k in range(N_SMOKE):
            age = s - 0.03 * k
            if age < 0:
                continue
            p = self.sp0[k] + self.sv[k] * age + np.array([0, 0, 0.05 * age * age])
            data.mocap_pos[self.smoke[k]] = p
            self.m.geom_size[self.smoke_g[k]][0] = 0.04 + 0.09 * (1 - math.exp(-age / 0.8))
            g = 0.3 + 0.25 * min(age / 1.5, 1)
            self.m.geom_rgba[self.smoke_g[k]] = [g, g, g * 1.02, 0.45 * math.exp(-age / 1.8)]


# --------------------------------------------------------------------------------------------- the video
def run(out, width=1280, height=720, fps=30, cache=None, frames_dir=None):
    import mujoco
    from PIL import Image, ImageDraw, ImageFont

    cache = Path(cache or ROOT / "runs" / "tarantula" / "video_cache")
    cache.mkdir(parents=True, exist_ok=True)
    # a dry run of the walk and the sway finds where the hook's pick will be: the parapet's nailer goes there
    model0, info0 = build(320, 240, cache)
    tip, _ = simulate(model0, info0, pre=True)
    face_x, tip_z = float(tip[0]) - 0.015, float(tip[2])
    print(f"tarantula_mission: the hook's pick at full sway at x {tip[0]:.4f}, z {tip[2]:.4f} m: the nailer's face at "
          f"x {face_x:.4f}", flush=True)
    model, info = build(width, height, cache, face_x=face_x, tip_z=tip_z)
    props = Props(model, info)
    renderer = mujoco.Renderer(model, height, width)
    cam = Cam(40.0, width, height)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
    opt.sitegroup[:] = [0, 0, 0, 0, 0, 0]
    opt.tendongroup[:] = [1, 1, 1, 1, 1, 1]
    try:
        import matplotlib
        fp = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf")
    except ImportError:
        fp = None
    font = (lambda n, bold=False: ImageFont.truetype(fp.replace("DejaVuSans.ttf", "DejaVuSans-Bold.ttf") if bold else fp, n)
            if fp else ImageFont.load_default())
    f_title, f_cap, f_small, f_boom = font(46, True), font(22, True), font(17), font(120, True)
    f_mono = ImageFont.truetype(fp.replace("DejaVuSans.ttf", "DejaVuSansMono-Bold.ttf"), 26) if fp else f_small
    frames_dir = Path(frames_dir or (cache / "mission_frames"))
    shutil.rmtree(frames_dir, ignore_errors=True)
    frames_dir.mkdir(parents=True)
    body = model.body("body").id
    gift_m = model.body("gift").mocapid[0]
    bar = int(height * 0.1)
    xf, ztab = info["xf"], info["table_z"]
    tcx = sum(info["table_x"]) / 2
    shot = {"name": None}
    count = {"n": 0}

    def render(t, phase, caption, data, st, extra, speed):
        b = data.xpos[body].copy()
        gp = data.mocap_pos[gift_m].copy()
        t_boom = st["t_boom"]
        # the props
        if "bandit_t0" in extra:
            props.bandit(data, t, extra["bandit_t0"], t_boom if t_boom is not None and t >= t_boom else None)
        if st["timer_set"] is not None and t >= st["timer_set"]:
            left = t_boom - t
            props.led_on(left > 0 and (t * (2 if left > 5 else 5)) % 1.0 < 0.35)
        if t_boom is not None and t >= t_boom:
            props.blast(data, t, t_boom, st["gift_pose"][0], 1.0 / fps)
        mujoco.mj_forward(model, data)
        # the shot
        name, k = phase, 0.12
        if phase in ("settle", "walk") and t < 2.4:
            name = "establish"
        if phase == "plant":
            name = "plant"
        if phase in ("deliver", "timer") or (phase == "descend" and gp[2] - ztab < 0.12):
            name = "deliver"
        if shot["name"] != name:
            cam.cut()
            shot["name"] = name
        if name == "establish":
            u = min(max((t + 0.6) / 3.0, 0), 1)
            cam.set((xf - 0.1, 0.2, -0.8 + 0.6 * u), 5.0 - 1.2 * u, 212 - 10 * u, -20 - 4 * u, k=1.0)
        elif name == "walk" or name == "settle":
            cam.set(b + np.array([0.08, 0.0, 0.0]), 1.0, 35, -22, k)
        elif name == "plant":
            tip = data.mocap_pos[model.body("harpoon").mocapid[0]]
            cam.set(0.5 * (tip + b) + np.array([0.04, 0, 0.02]), 0.62, 285, -15, 0.2)
        elif name == "over":
            cam.set(b + np.array([0.05, 0, -0.05]), 1.3, 250, -10, 0.08)
        elif name == "hang":
            cam.set(b + np.array([0, 0, 0.12]), 1.15, 205, 10, 0.1)
        elif name == "descend":
            look = np.array([b[0] + 0.05, b[1], 0.65 * b[2] + 0.35 * ztab + 0.2])
            cam.set(look, 1.6 + 0.4 * min(-b[2] / 1.5, 1), 222, 4, 0.08)
        elif name == "deliver":
            cam.set(gp + np.array([0.0, 0.0, 0.14]), 1.05, 222, -10, 0.12)
        elif name == "ascend":
            cam.set(np.array([b[0], b[1], 0.6 * b[2] + 0.4 * (ztab + 0.4)]), 1.9, 212, 3, 0.06)
        elif name in ("back", "hide"):
            cam.set(b + np.array([0.06, 0, 0.0]), 1.15, 250 if name == "back" else 20, -24, 0.08)
        elif name == "bandit":
            u = (t - extra["bandit_t0"]) / max(t_boom - extra["bandit_t0"], 1e-3)
            cam.set(np.array([tcx + 0.05, 0.4, ztab + 0.32 + 0.25 * (1 - u)]), 3.3 - 1.4 * sm(u), 150 - 8 * sm(u), -6, 0.06)
        elif name == "boom":
            s = t - t_boom
            shake = 0.04 * math.exp(-s / 0.35) * math.sin(70 * s)
            cam.set(np.array([tcx + 0.05, 0.4 + shake, ztab + 0.4 + shake]), 2.3 + 0.4 * sm(s / 0.4), 142, -4, 0.25)
        elif name == "end":
            cam.set(extra["peek"] + np.array([0.12, 0, 0.1]), 0.85, 215, -20, 0.1)
        renderer.update_scene(data, camera=cam.c, scene_option=opt)
        img = Image.fromarray(renderer.render())
        dr = ImageDraw.Draw(img, "RGBA")
        # letterbox, the mission's captions
        dr.rectangle([0, 0, width, bar], fill=(0, 0, 0, 255))
        dr.rectangle([0, height - bar, width, height], fill=(0, 0, 0, 255))
        dr.text((24, bar / 2 - 12), "OPERATION SILENT DROP", font=f_cap, fill=(255, 196, 64))
        clock = 23 * 3600 + 47 * 60 + max(t, 0.0)
        stamp = f"OLD TOWN · ROOF 4B · {int(clock // 3600) % 24:02d}:{int(clock // 60) % 60:02d}:{int(clock) % 60:02d}"
        dr.text((width - 24 - dr.textlength(stamp, font=f_small), bar / 2 - 9), stamp, font=f_small, fill=(210, 210, 205))
        if caption and phase not in ("boom", "end"):
            dr.text((24, height - bar + bar / 2 - 12), caption, font=f_cap, fill=(240, 240, 235))
        info_r = ""
        if phase in ("hang", "descend", "deliver", "timer", "ascend"):
            info_r = f"line {st['tension']:4.1f} N   height {b[2] - STREET:4.2f} m"
        if st.get("feet"):
            n_down, mg = st["feet"]
            info_r = f"feet down {n_down}/8" + (f"   stability margin {mg * 1000:+.0f} mm" if mg is not None else "")
        if speed > 1:
            info_r += f"   ▶▶ ×{speed:g}"
        if info_r:
            dr.text((width - 24 - dr.textlength(info_r, font=f_small), height - bar + bar / 2 - 9), info_r, font=f_small,
                    fill=(255, 196, 64))
        # the gift's timer, next to the gift on screen
        if st["timer_set"] is not None and t >= st["timer_set"] and (t_boom is None or t < t_boom):
            left = max(t_boom - t, 0.0)
            pix = cam.project(gp + np.array([0, 0, 0.05]))
            txt = f"{int(left // 60):02d}:{left % 60:05.2f}"
            if pix and bar < pix[1] < height - bar - 30 and 0 < pix[0] < width - 160:
                x0, y0 = pix[0] + 18, pix[1] - 40
                dr.rectangle([x0 - 8, y0 - 4, x0 + dr.textlength(txt, font=f_mono) + 8, y0 + 32], fill=(10, 10, 10, 190),
                             outline=(255, 60, 40, 255))
                dr.text((x0, y0), txt, font=f_mono, fill=(255, 70, 50))
        if phase == "bandit" and t_boom - t < 3.2:
            n = int(t_boom - t) + 1
            w_ = dr.textlength(str(n), font=f_boom)
            dr.text(((width - w_) / 2, height / 2 - 90), str(n), font=f_boom, fill=(255, 70, 50, 150))
        if phase == "boom" and t - t_boom < 1.2:
            s = t - t_boom
            sc = 0.6 + 0.3 * math.sin(min(s / 0.25, 1) * math.pi / 2)
            fb = font(int(120 * sc * height / 720), True)
            w_ = dr.textlength("KABOOM!", font=fb)
            x0, y0 = (width - w_) / 2, bar + height * 0.04
            for ox, oy in ((-4, 0), (4, 0), (0, -4), (0, 4), (5, 5)):
                dr.text((x0 + ox, y0 + oy), "KABOOM!", font=fb, fill=(60, 10, 0, 255))
            dr.text((x0, y0), "KABOOM!", font=fb, fill=(255, 205, 40, 255))
        if phase == "boom" and t - t_boom >= 1.6:
            dr.text((24, height - bar + bar / 2 - 12), "the bandit: soot-black, confetti everywhere · TARANTULA: unseen",
                    font=f_cap, fill=(240, 240, 235))
        if phase == "end":
            a = int(255 * sm((t - (t_boom + 4.8)) / 0.8))
            txt = "MISSION ACCOMPLISHED"
            w_ = dr.textlength(txt, font=f_title)
            yc = bar + 70
            dr.rectangle([0, yc - 50, width, yc + 50], fill=(0, 0, 0, int(a * 0.6)))
            dr.text(((width - w_) / 2, yc - 38), txt, font=f_title, fill=(255, 196, 64, a))
            sub = "TARANTULA · 1.5 kg · 24 tendon-driven joints · spine-comb feet · dragline pod"
            dr.text(((width - dr.textlength(sub, font=f_small)) / 2, yc + 18), sub, font=f_small,
                    fill=(235, 235, 230, a))
        if t < 2.0:                                                       # the title over the opening shot
            a = int(255 * (1 - sm((t - 1.2) / 0.8)))
            txt = "TARANTULA"
            w_ = dr.textlength(txt, font=f_title)
            dr.text(((width - w_) / 2, height / 2 - 60), txt, font=f_title, fill=(255, 255, 255, a))
            sub = "in  OPERATION SILENT DROP"
            dr.text(((width - dr.textlength(sub, font=f_cap)) / 2, height / 2), sub, font=f_cap, fill=(255, 196, 64, a))
        credit = "MuJoCo physics + renderer · CAD: labs/legged/designs/spiders.py"
        if phase == "end":
            dr.text((width - 24 - dr.textlength(credit, font=f_small), height - bar + bar / 2 - 9), credit, font=f_small,
                    fill=(200, 200, 195))
        img.convert("RGB").save(frames_dir / f"f_{count['n']:05d}.jpg", quality=92)
        count["n"] += 1
        if count["n"] % 90 == 0:
            print(f"  {count['n']} frames, t = {t:6.2f} s, {phase}", flush=True)

    render.fps = fps
    st = simulate(model, info, render=render)
    print(f"tarantula_mission: boom at t = {st['t_boom']:.1f} s, {count['n']} frames", flush=True)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps), "-i", str(frames_dir / "f_%05d.jpg"),
                        "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags",
                        "+faststart", str(out)], check=True)
        print("tarantula_mission:", out)
    else:
        print("tarantula_mission: no ffmpeg — the frames are in", frames_dir)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tarantula.sh mission", description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(ROOT / "runs" / "tarantula" / "tarantula_mission.mp4"))
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args(argv)
    run(a.out, a.width, a.height, a.fps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
