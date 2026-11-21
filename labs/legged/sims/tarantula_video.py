"""A MuJoCo video of TARANTULA on its dragline (``./tarantula.sh video``): MuJoCo's physics and MuJoCo's renderer, the
robot dressed in its CAD (``spiders.Tarantula``, split onto the simulated links), the cords as MuJoCo spatial tendons
wrapping the knee knuckle (so they follow the joints), the dragline as a rope (a spatial tendon that pulls only when
taut).

The scene: a vertical brick wall, a timber beam above. TARANTULA hangs head-down on its spine-comb feet, fires its
harpoon (spring launcher at the spinnerets, §12) into the beam's underside — the flight lasts 0.05 s, shown in slow
motion — takes up the slack, then walks down the wall head first with the spool's brake paying out line so the line
carries ``SHARE`` of its weight; it stops, and walks back up tail first while the winch reels in.

    ./tarantula.sh video                              # runs/tarantula/tarantula_dragline.mp4 (1280 × 720, 30 fps)
    ./tarantula.sh video --cycles 2 --width 960 --height 540 --out my.mp4

Needs mujoco, cadquery (the CAD → meshes, cached under runs/tarantula/video_cache), Pillow, and ffmpeg (else the
frames are left as JPEGs). Headless Linux: MUJOCO_GL=egl or osmesa (the script picks osmesa if no display).
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

SHARE = 0.6                                                               # of the weight the line carries
COLOR = {"shell": (0.42, 0.29, 0.21), "armour": (0.29, 0.2, 0.13), "turret": (0.15, 0.15, 0.17),
         "deck": (0.72, 0.74, 0.76), "jaw housing": (0.35, 0.24, 0.16), "jaws": (0.12, 0.12, 0.13),
         "blades": (0.87, 0.89, 0.91), "jaw drive": (0.72, 0.74, 0.76), "joints": (0.72, 0.74, 0.76),
         "carbon": (0.17, 0.17, 0.19), "springs": (0.85, 0.66, 0.16), "feet": (0.91, 0.54, 0.17),
         "spines": (0.87, 0.89, 0.91), "pod": (0.2, 0.42, 0.75), "line": (0.95, 0.95, 0.92), "harpoon": (0.8, 0.82, 0.85)}
METAL = {"deck", "blades", "jaw drive", "joints", "spines", "harpoon"}


# --------------------------------------------------------------------------------------------- the CAD as meshes
def _stl(path, V, F):
    """A binary STL of vertices V [m] and faces F."""
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    rec = np.zeros(len(F), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = n, tri
    with open(path, "wb") as fh:
        fh.write(b"tarantula".ljust(80, b" "))
        fh.write(np.uint32(len(F)).tobytes())
        fh.write(rec.tobytes())


def cad_meshes(cache: Path):
    """{link: [(group, stl path)]} and the harpoon's [(group, stl path)] — TARANTULA's CAD on the simulation's links
    (each in its link frame at the design pose, where the links are unrotated), the cords left out (they are
    tendons in the model). Cached by the CAD sources' hash."""
    import motion_kit as mk
    import spider_specs as ss
    import spiders as S
    import tarantula_robot as tr
    src = b"".join((LAB / "designs" / f).read_bytes() for f in ("spiders.py", "spider_specs.py", "tarantula_robot.py"))
    key = hashlib.sha1(src).hexdigest()[:12]
    out = cache / f"meshes_{key}"
    index = out / "index.json"
    if index.exists():
        return json.loads(index.read_text())
    out.mkdir(parents=True, exist_ok=True)
    print("tarantula_video: tessellating the CAD (once; cached in", out, ")", flush=True)
    d = S.Tarantula()
    s = d.spec
    kin = mk.Kin(tr.tarantula())
    kin.set(kin.q0)
    base = kin.data.xpos[kin.body("body")].copy()
    zc = ss.body_z(s)
    parts = {}

    def add(link, group, w, tol=0.4, ang=0.3):
        o = (kin.data.xpos[kin.body(link)] - base) * 1000 + np.array([0, 0, zc])
        for v in w.vals():
            vs, fs = v.tessellate(tol, ang)
            if fs:
                parts.setdefault(link, {}).setdefault(group, []).append(
                    ((np.array([x.toTuple() for x in vs]) - o) * 1e-3, np.array(fs)))

    for v in d.shells(s):
        add("body", "shell", v)
    for v in d.hip_sockets(s):
        add("body", "joints", v)
    for v in d.plates(s) + d.tergites(s):
        add("body", "armour", v)
    for v in d.turret(s):
        add("body", "turret", v)
    for v in d.deck(s):
        add("body", "deck", v)
    ch = d.chelicerae(s, 20.0)
    for k, g in (("housing", "jaw housing"), ("pins", "deck"), ("jaws", "jaws"), ("blades", "blades"),
                 ("drive", "jaw drive")):
        for v in ch[k]:
            add("body", g, v)
    pod = d.dragline_pod(s)
    for k, vs in pod.items():
        for v in vs:
            add("body", "line" if k == "line" else "pod", v)
    link_of = {"coxa": "hip", "femur": "femur", "knee": "femur", "knee_spring": "femur", "tibia": "tibia",
               "meta_joint": "tibia", "metatarsus": "tibia", "tarsus_joint": "tibia", "tarsus": "tibia", "foot": "tibia",
               "spines": "tibia"}
    group = {"coxa": "joints", "femur": "carbon", "knee": "joints", "knee_spring": "springs", "tibia": "carbon",
             "meta_joint": "joints", "metatarsus": "carbon", "tarsus_joint": "springs", "tarsus": "carbon", "foot": "feet",
             "spines": "spines"}
    for leg in S.LEGS:
        for k, v in S.leg_parts(s, detail=True).items():
            if k == "tendons":
                continue
            coarse = k in ("knee_spring", "tarsus_joint")
            add(f"{leg}_{link_of[k]}", group[k], S._place(v, leg, s), 1.2 if coarse else (0.1 if k == "spines" else 0.4),
                0.7 if coarse else 0.3)
    index_d = {}
    for link, groups in parts.items():
        index_d[link] = []
        for g, ps in groups.items():
            V = np.vstack([p[0] for p in ps])
            offs = np.cumsum([0] + [len(p[0]) for p in ps[:-1]])
            F = np.vstack([p[1] + o for p, o in zip(ps, offs)])
            path = out / f"{link}_{g.replace(' ', '_')}.stl"
            _stl(path, V, F)
            index_d[link].append([g, str(path)])
    hp = d.harpoon(s)
    hparts = []
    for k, v in (("harpoon", hp["shaft"]), ("harpoon", hp["eye"]), *(("harpoon", b) for b in hp["barbs"])):
        for sol in v.vals():
            vs, fs = sol.tessellate(0.1, 0.2)
            hparts.append((np.array([x.toTuple() for x in vs]) * 1e-3, np.array(fs)))
    V = np.vstack([p[0] for p in hparts])
    offs = np.cumsum([0] + [len(p[0]) for p in hparts[:-1]])
    F = np.vstack([p[1] + o for p, o in zip(hparts, offs)])
    _stl(out / "harpoon.stl", V, F)
    index_d["__harpoon__"] = [["harpoon", str(out / "harpoon.stl")]]
    index.write_text(json.dumps(index_d))
    return index_d


def _brick_png(path, px_per_mm=2):
    """A running-bond brick tile, 450 × 450 mm (two bricks across, six courses), the courses running along the image's
    v axis (MuJoCo maps u to the plane's x — here up and down the wall)."""
    from PIL import Image, ImageDraw
    bw, bh, m = 215, 65, 10
    W, H = 2 * (bw + m), 6 * (bh + m)
    im = Image.new("RGB", (W * px_per_mm, H * px_per_mm), (186, 178, 166))
    dr = ImageDraw.Draw(im)
    rng = np.random.default_rng(3)
    for r in range(6):
        off = (r % 2) * (bw + m) / 2
        for c in range(-1, 3):
            x0 = (c * (bw + m) + off + m / 2) * px_per_mm
            y0 = (r * (bh + m) + m / 2) * px_per_mm
            base = np.array([150, 62, 40]) * (0.85 + 0.3 * rng.random())
            dr.rectangle([x0, y0, x0 + bw * px_per_mm, y0 + bh * px_per_mm], fill=tuple(int(v) for v in np.clip(base, 0, 255)))
            for _ in range(160):
                sx, sy = x0 + rng.random() * bw * px_per_mm, y0 + rng.random() * bh * px_per_mm
                tone = np.clip(base * (0.75 + 0.5 * rng.random()), 0, 255)
                dr.rectangle([sx, sy, sx + 2, sy + 2], fill=tuple(int(v) for v in tone))
    im.transpose(Image.Transpose.ROTATE_90).save(path)
    return 0.45                                                            # tile size [m]


# --------------------------------------------------------------------------------------------- the scene
def build(width, height, cache: Path):
    """(MjModel, info): TARANTULA head-down on a vertical brick wall (gravity +x), its CAD meshes, cords, dragline and
    harpoon, the beam."""
    import motion_kit as mk
    import mujoco
    import spiders as S
    import tarantula_attachment as ta
    import tarantula_gaits as tg
    import tarantula_robot as tr
    import tarantula_sim as ts
    from vegeta.chiron.robot import SimOptions
    robot = tr.tarantula()
    kin = mk.Kin(robot)
    g = (9.81, 0.0, 0.0)                                                  # down the wall = +x: TARANTULA faces down
    opt = SimOptions(timestep=0.001, gravity=g, cone="elliptic", impratio=3.0, noslip_iterations=2)
    root = ET.fromstring(robot.to_mjcf(options=opt))
    meshes = cad_meshes(cache)
    tile = _brick_png(cache / "brick.png")
    vis = _child(root, "visual")
    _child(vis, "global").attrib.update({"offwidth": str(width), "offheight": str(height), "fovy": "40"})
    _child(vis, "quality").attrib.update({"shadowsize": "4096", "offsamples": "8"})
    _child(vis, "headlight").attrib.update({"ambient": "0.32 0.31 0.3", "diffuse": "0.35 0.34 0.33",
                                            "specular": "0.05 0.05 0.05"})
    _child(vis, "map").attrib.update({"znear": "0.005", "zfar": "40"})
    asset = _child(root, "asset")
    ET.SubElement(asset, "texture", {"name": "sky", "type": "skybox", "builtin": "gradient", "rgb1": "0.24 0.26 0.29",
                                     "rgb2": "0.06 0.07 0.08", "width": "256", "height": "256"})
    ET.SubElement(asset, "texture", {"name": "brick", "type": "2d", "file": str(cache / "brick.png")})
    ET.SubElement(asset, "material", {"name": "brick", "texture": "brick", "texrepeat": f"{8 / tile:.3f} {8 / tile:.3f}",
                                      "roughness": "0.9", "specular": "0.05"})
    ET.SubElement(asset, "material", {"name": "wood", "rgba": "0.52 0.36 0.21 1", "specular": "0.1"})
    for g_, (r, gg, b) in COLOR.items():
        ET.SubElement(asset, "material", {"name": f"mat:{g_}", "rgba": f"{r} {gg} {b} 1",
                                          "specular": "0.6" if g_ in METAL else "0.15",
                                          "shininess": "0.6" if g_ in METAL else "0.2"})
    for link, items in meshes.items():
        for g_, path in items:
            ET.SubElement(asset, "mesh", {"name": f"m:{link}:{g_}", "file": path, "inertia": "shell"})
    world = root.find("worldbody")
    for gm in world.findall("geom"):
        if gm.get("type") == "plane":
            gm.set("size", "4 4 0.1")
            gm.set("material", "brick")
            gm.attrib.pop("rgba", None)
    ET.SubElement(world, "light", {"name": "sun", "directional": "true", "pos": "-1.5 -1.2 2.5", "dir": "0.45 0.35 -0.82",
                                   "diffuse": "0.72 0.68 0.62", "specular": "0.15 0.15 0.15", "castshadow": "true"})
    anchor = np.array([-0.74, 0.0, 0.04])                                 # the beam's underside: the harpoon's mark
    ET.SubElement(world, "geom", {"name": "beam", "type": "box", "size": "0.06 1.6 0.035", "pos": "-0.80 0 0.035",
                                  "material": "wood", "contype": "0", "conaffinity": "0", "group": "1"})
    bodies = {b.get("name"): b for b in root.iter("body")}
    # the robot's own collision geoms out of sight (group 3); its CAD meshes in (group 1)
    for b in bodies.values():
        for gm in b.findall("geom"):
            if not gm.get("name", "").startswith("pm:"):
                gm.set("group", "3")
    for link, items in meshes.items():
        if link not in bodies:
            continue
        for g_, _ in items:
            ET.SubElement(bodies[link], "geom", {"type": "mesh", "mesh": f"m:{link}:{g_}", "contype": "0",
                                                 "conaffinity": "0", "group": "1", "mass": "0",
                                                 "material": f"mat:{g_}"})
    # the cords: spatial tendons between sites on the links, wrapping the knee knuckle (spiders.tendon_routes)
    tendon_el = _child(root, "tendon")
    kin.set(kin.q0)
    base = kin.data.xpos[kin.body("body")].copy()
    zc = tr.STAND_Z * 1000

    def local(link, p_leg, leg):
        (hx, hy), u, _ = tr.leg_frame(leg)
        ang = math.atan2(u[1], u[0])
        c_, s_ = math.cos(ang), math.sin(ang)
        cad = np.array([hx + c_ * p_leg[0] - s_ * p_leg[1], hy + s_ * p_leg[0] + c_ * p_leg[1], p_leg[2]])
        o = (kin.data.xpos[kin.body(link)] - base) * 1000 + np.array([0, 0, zc])
        return (cad - o) * 1e-3, (c_, s_)

    routes = S.tendon_routes(tr.S)
    cord_rgba = {"flexor": "0.82 0.14 0.12 1", "coupling": "0.95 0.45 0.12 1", "pushrod": "0.55 0.57 0.6 1"}
    for leg in tr.LEGS:
        for name, rt in routes.items():
            el = ET.SubElement(tendon_el, "spatial", {"name": f"cord:{leg}:{name}", "width": f"{rt['radius'] * 1e-3:.5f}",
                                                      "rgba": cord_rgba[name], "group": "1"})
            for k, (tag, p) in enumerate(rt["pts"]):
                if tag == "wrap":
                    c, R, _ = rt["wrap"]
                    link = f"{leg}_femur"
                    pc, (c_, s_) = local(link, c, leg)
                    axis = np.array([-s_, c_, 0.0])                       # the knee's axis (the leg frame's y)
                    cyl = f"pulley:{leg}:{name}"
                    nk = np.asarray(p) - np.asarray(c)
                    side_p, _ = local(link, np.asarray(c) + 2.0 * nk, leg)
                    ET.SubElement(bodies[link], "geom", {"name": cyl, "type": "cylinder", "size": f"{R * 1e-3:.5f} 0.004",
                                                         "pos": " ".join(f"{v:.5f}" for v in pc),
                                                         "zaxis": " ".join(f"{v:.5f}" for v in axis), "contype": "0",
                                                         "conaffinity": "0", "group": "3", "mass": "0"})
                    ET.SubElement(bodies[link], "site", {"name": f"side:{leg}:{name}",
                                                         "pos": " ".join(f"{v:.5f}" for v in side_p), "group": "4"})
                    ET.SubElement(el, "geom", {"geom": cyl, "sidesite": f"side:{leg}:{name}"})
                    continue
                link = f"{leg}_{tag}"
                pl, _ = local(link, p, leg)
                sname = f"s:{leg}:{name}:{k}"
                ET.SubElement(bodies[link], "site", {"name": sname, "pos": " ".join(f"{v:.5f}" for v in pl), "group": "4",
                                                     "size": "0.001"})
                ET.SubElement(el, "site", {"site": sname})
    # the dragline: from the launcher's muzzle on the body to the harpoon's eye (a mocap body, flown by the script)
    d = S.Tarantula()
    eye_mm, dvec = d.harpoon_pose(tr.S)
    _, _, T0 = d.pod_frame(tr.S)
    muzzle_mm = T0 + tr.S["winch"]["tube"][2] * dvec
    to_body = lambda p_mm: (np.asarray(p_mm) - np.array([0, 0, zc])) * 1e-3
    ET.SubElement(bodies["body"], "site", {"name": "muzzle", "pos": " ".join(f"{v:.5f}" for v in to_body(muzzle_mm)),
                                           "group": "4", "size": "0.002"})
    hb = ET.SubElement(world, "body", {"name": "harpoon", "mocap": "true"})
    hp_mesh = meshes["__harpoon__"][0][1]
    ET.SubElement(asset, "mesh", {"name": "m:harpoon", "file": hp_mesh, "inertia": "shell"})
    ET.SubElement(hb, "geom", {"type": "mesh", "mesh": "m:harpoon", "contype": "0", "conaffinity": "0", "group": "1",
                               "material": "mat:harpoon"})
    ET.SubElement(hb, "site", {"name": "eye", "pos": "-0.0022 0 0", "group": "4", "size": "0.002"})
    line = ET.SubElement(tendon_el, "spatial", {"name": "dragline", "width": "0.0008", "rgba": "0.96 0.96 0.9 1",
                                                "group": "1"})
    ET.SubElement(line, "site", {"site": "muzzle"})
    ET.SubElement(line, "site", {"site": "eye"})
    act = root.find("actuator")
    ET.SubElement(act, "motor", {"name": "winch", "tendon": "dragline", "gear": "1", "ctrllimited": "false"})
    pull, mu_att = ts._attach_params("spine:brick")
    for gm in root.iter("geom"):
        if gm.get("name", "") in tr.LEGS:
            gm.set("friction", f"{mu_att:.4f} 0.005 0.0001")
            gm.set("margin", "0.003")
    for f in tr.LEGS:
        ET.SubElement(act, "adhesion", {"name": f"adh:{f}", "body": f"{f}_tibia", "ctrlrange": "0 1", "gain": f"{pull:.5f}"})
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    info = {"kin": kin, "robot": robot, "anchor": anchor, "eye_body": to_body(eye_mm), "dir_body": np.asarray(dvec),
            "muzzle_body": to_body(muzzle_mm), "pull": pull, "attach": ta, "tg": tg, "tr": tr}
    return model, info


def _child(parent, tag):
    """The element's ``tag`` child, made if missing (an empty Element is falsy: never ``find(...) or ...``)."""
    el = parent.find(tag)
    return el if el is not None else ET.SubElement(parent, tag)


def _quat_from_x(d):
    """The quaternion turning +x onto the unit vector d."""
    d = np.asarray(d, float) / np.linalg.norm(d)
    x = np.array([1.0, 0.0, 0.0])
    v = np.cross(x, d)
    c = float(np.dot(x, d))
    if np.linalg.norm(v) < 1e-9:
        return np.array([1.0, 0, 0, 0]) if c > 0 else np.array([0.0, 0, 0, 1])
    q = np.array([1.0 + c, *v])
    return q / np.linalg.norm(q)


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


# --------------------------------------------------------------------------------------------- the run
def run(out, cycles=3, width=1280, height=720, fps=30, cache=None, frames_dir=None):
    import mujoco
    from PIL import Image, ImageDraw, ImageFont

    cache = Path(cache or ROOT / "runs" / "tarantula" / "video_cache")
    cache.mkdir(parents=True, exist_ok=True)
    model, info = build(width, height, cache)
    data = mujoco.MjData(model)
    tr, tg, kin = info["tr"], info["tg"], info["kin"]
    motion = tg.gait("wave4", kin=kin, body_drop_mm=tg.POSTURES["crouched"])
    names = list(kin.names)
    qadr = np.array([model.joint(n).qposadr[0] for n in names])
    dadr = np.array([model.joint(n).dofadr[0] for n in names])
    act_servo = np.array([model.actuator(f"servo:{n}").id for n in names])
    act_adh = {f: model.actuator(f"adh:{f}").id for f in tr.LEGS}
    a_winch = model.actuator("winch").id
    line_id = model.tendon("dragline").id
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

    groups, _ = tg.GAITS["wave4"]
    win = T / len(groups)
    group_of = {f: k for k, grp in enumerate(groups) for f in grp}

    def swing_u(f, tau):
        s = ((tau - group_of[f] * win) % T) / T
        return s * len(groups) if s < 1.0 / len(groups) else None

    # the script [s of sim time from the end of the settle]
    t_fire, t_fly = 0.8, 0.05
    t_set = t_fire + t_fly
    t_down = t_set + 0.9
    t_stop = t_down + cycles * T
    t_up = t_stop + 1.0
    t_end = t_up + cycles * T + 0.8
    settle = 0.6
    W = info["robot"].total_mass() * 9.81
    k_rope, c_rope, F_max = 6000.0, 40.0, 100.0
    v_in, v_out = 0.046, 0.25                                             # winch reel-in, brake pay-out [m/s]

    data.qpos[model.joint("body:free").qposadr[0] + 2] = tr.STAND_Z - tg.POSTURES["crouched"] * 1e-3
    data.qpos[qadr] = q_ref(0.0)
    mujoco.mj_forward(model, data)
    body = model.body("body").id
    mocap = model.body("harpoon").mocapid[0]
    anchor = info["anchor"]
    L = None                                                              # the line paid out [m] (None: free)
    tension = 0.0
    fired_from = None

    def harpoon_on_robot():
        R = data.xmat[body].reshape(3, 3)
        p = data.xpos[body] + R @ info["eye_body"]
        q_body = data.xquat[body].copy()
        return p, _qmul(q_body, _quat_from_x(info["dir_body"]))

    # the flight: from the muzzle to the beam's underside, tip first, gravity +x
    tip_len = (tr.S["winch"]["harpoon"][1] + tr.S["winch"]["harpoon"][2]) * 1e-3
    embed = 0.02

    renderer = mujoco.Renderer(model, height, width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
    opt.sitegroup[:] = [0, 0, 0, 0, 0, 0]
    opt.tendongroup[:] = [1, 1, 1, 1, 1, 1]
    font_path = None
    try:
        import matplotlib
        font_path = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf")
    except ImportError:                                                   # no matplotlib: Pillow's own font
        font_path = None
    f_big = ImageFont.truetype(font_path, 30) if font_path else ImageFont.load_default()
    f_small = ImageFont.truetype(font_path, 18) if font_path else ImageFont.load_default()
    f_mono = ImageFont.truetype(font_path.replace("DejaVuSans.ttf", "DejaVuSansMono.ttf"), 17) if font_path else f_small
    frames_dir = Path(frames_dir or (cache / "frames"))
    shutil.rmtree(frames_dir, ignore_errors=True)
    frames_dir.mkdir(parents=True)
    n_frames = 0
    next_frame = -settle
    look = None
    dt = model.opt.timestep
    t = -settle
    x_start = None
    while t < t_end:
        # the phase and the gait's clock τ
        if t < t_down:
            tau, phase = 0.0, "holding on, spines set" if t < t_fire else ("harpoon away" if t < t_set else "taking up the slack")
        elif t < t_stop:
            tau, phase = t - t_down, "lowering itself: brake paying out"
        elif t < t_up:
            tau, phase = cycles * T, "stopped on the line"
        elif t < t_up + cycles * T:
            tau, phase = cycles * T - (t - t_up), "pulling itself up: winch reeling in"
        else:
            tau, phase = 0.0, "back up, line taut"
        moving = t_down <= t < t_stop or t_up <= t < t_up + cycles * T
        sgn = 1.0 if t < t_up else -1.0
        qr = q_ref(tau)
        qdr = sgn * (q_ref(tau + dt) - qr) / dt if moving else np.zeros_like(qr)
        q, qd = data.qpos[qadr].copy(), data.qvel[dadr].copy()
        tq = kp * (qr - q) + kd * (qdr - qd)
        lim = np.where(tq * qd < 0, stall, sv.torque_limit(qd))
        data.ctrl[act_servo] = np.clip(tq, -lim, lim) + kd * qd
        for f in tr.LEGS:
            u = swing_u(f, tau) if moving else None
            data.ctrl[act_adh[f]] = 1.0 if u is None else (max(0.0, 1 - u / 0.1) if u < 0.1 else max(0.0, (u - 0.9) / 0.1))
        # the harpoon and the line
        if t < t_fire:
            p, qh = harpoon_on_robot()
            data.mocap_pos[mocap], data.mocap_quat[mocap] = p, qh
        elif t < t_set:
            if fired_from is None:
                p0, _ = harpoon_on_robot()
                tip0 = p0 + (data.xmat[body].reshape(3, 3) @ info["dir_body"]) * tip_len
                tip1 = anchor + np.array([-embed, 0.0, 0.0])
                gvec = np.array([9.81, 0.0, 0.0])
                v0 = (tip1 - tip0 - 0.5 * gvec * t_fly ** 2) / t_fly
                fired_from = (tip0, v0, gvec)
            tip0, v0, gvec = fired_from
            s_ = t - t_fire
            tip = tip0 + v0 * s_ + 0.5 * gvec * s_ ** 2
            vel = v0 + gvec * s_
            dvec = vel / np.linalg.norm(vel)
            data.mocap_pos[mocap], data.mocap_quat[mocap] = tip - dvec * tip_len, _quat_from_x(dvec)
        elif fired_from is not None and len(fired_from) == 3:
            tip0, v0, gvec = fired_from
            vel = v0 + gvec * t_fly
            dvec = vel / np.linalg.norm(vel)
            tip1 = anchor + np.array([-embed, 0.0, 0.0])
            data.mocap_pos[mocap], data.mocap_quat[mocap] = tip1 - dvec * tip_len, _quat_from_x(dvec)
            fired_from = (None,)
        length = float(data.ten_length[line_id])
        vel_l = float(data.ten_velocity[line_id])
        if t < t_set:
            L = None                                                      # slack: the spool turns freely
        else:
            if L is None:
                L = length + 0.005                                        # the harpoon is in: the brake goes on
            target = 0.3 * W if t < t_down else SHARE * W
            dL = np.clip(0.08 * (tension - target), -v_in, v_out)        # brake pays out / winch reels in
            if t < t_down and tension < target:
                dL = -v_in
            L += dL * dt
        stretch = length - L if L is not None else -1.0
        tension = float(np.clip(k_rope * stretch + c_rope * max(vel_l, 0.0), 0.0, F_max)) if stretch > 0 else 0.0
        data.ctrl[a_winch] = -tension
        mujoco.mj_step(model, data)
        t += dt
        if x_start is None and t >= 0:
            x_start = float(data.xpos[body][0])
        rate = 240.0 if t_fire - 0.02 <= t < t_set + 0.02 else fps     # slow motion for the shot
        if t >= next_frame:
            next_frame += 1.0 / rate
            bx = data.xpos[body].copy()
            # the robot in the lower middle of the frame, the line and the beam above it
            target_look = np.array([0.72 * bx[0] + 0.28 * anchor[0], bx[1], 0.04])
            look = target_look if look is None else look + 0.08 * (target_look - look)
            sep = abs(bx[0] - anchor[0])
            prog = min(max(t / t_end, 0.0), 1.0)
            cam.lookat[:] = look
            cam.distance = 0.78 + 0.55 * sep
            cam.azimuth = 180.0 + 30.0 * math.cos(math.pi * prog)
            cam.elevation = -52.0
            renderer.update_scene(data, camera=cam, scene_option=opt)
            img = Image.fromarray(renderer.render())
            dr = ImageDraw.Draw(img)
            dr.text((26, 18), "TARANTULA", font=f_big, fill=(245, 243, 238))
            dr.text((28, 56), "dragline pod: harpoon into a timber beam, then down and up a vertical brick wall",
                    font=f_small, fill=(232, 228, 220))
            slow = " · slow motion ×8" if rate > fps else ""
            dr.rectangle([20, height - 118, 600, height - 20], fill=(15, 17, 20))
            dr.text((32, height - 110), phase + slow, font=f_small, fill=(255, 177, 92))
            below = max(0.0, (bx[0] - anchor[0]))
            dr.text((32, height - 80), f"t {max(t, 0):5.1f} s   line tension {tension:5.1f} N ({tension / W:4.0%} of weight)",
                    font=f_mono, fill=(240, 240, 236))
            held = sum(1 for f in tr.LEGS if data.ctrl[act_adh[f]] > 0.5)
            dr.text((32, height - 54), f"below the anchor {below:4.2f} m   feet holding {held}/8",
                    font=f_mono, fill=(240, 240, 236))
            credit = "MuJoCo physics + renderer · CAD: labs/legged/designs/spiders.py"
            dr.text((width - 24 - dr.textlength(credit, font=f_small), height - 40), credit, font=f_small,
                    fill=(220, 220, 214))
            img.save(frames_dir / f"f_{n_frames:05d}.jpg", quality=92)
            n_frames += 1
            if n_frames % 60 == 0:
                print(f"  {n_frames} frames, t = {t:.1f} s", flush=True)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps), "-i", str(frames_dir / "f_%05d.jpg"),
                        "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags",
                        "+faststart", str(out)], check=True)
        print("tarantula_video:", out, f"({n_frames} frames)")
    else:
        print("tarantula_video: no ffmpeg — the frames are in", frames_dir)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tarantula.sh video", description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(ROOT / "runs" / "tarantula" / "tarantula_dragline.mp4"))
    ap.add_argument("--cycles", type=int, default=3, help="gait cycles down and up (2.4 s each)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args(argv)
    run(a.out, a.cycles, a.width, a.height, a.fps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
