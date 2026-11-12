"""Pictures of the teleop bench and its four task scenes, drawn with matplotlib from the compiled MuJoCo model's geoms
(boxes, cylinders, capsules, spheres at their world poses) — no OpenGL renderer needed.

    python manus_teleop_draw.py [out.png]
"""
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import manus_teleop as mt


def _box(c, R, h):
    s = np.array([[a, b, e] for a in (-1, 1) for b in (-1, 1) for e in (-1, 1)]) * h
    v = s @ R.T + c
    f = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
    return [v[k] for k in f]


def _cyl(c, R, r, hl, n=14, caps=True):
    t = np.linspace(0, 2 * np.pi, n + 1)
    ring = np.c_[r * np.cos(t), r * np.sin(t)]
    out = []
    for k in range(n):
        a, b = ring[k], ring[k + 1]
        q = np.array([[a[0], a[1], -hl], [b[0], b[1], -hl], [b[0], b[1], hl], [a[0], a[1], hl]])
        out.append(q @ R.T + c)
    if caps:
        for z in (-hl, hl):
            out.append(np.c_[ring, np.full(len(ring), z)] @ R.T + c)
    return out


def _sphere(c, r, n=8):
    out = []
    th, ph = np.linspace(0, np.pi, n + 1), np.linspace(0, 2 * np.pi, 2 * n + 1)
    for i in range(n):
        for j in range(2 * n):
            pts = [(th[i], ph[j]), (th[i + 1], ph[j]), (th[i + 1], ph[j + 1]), (th[i], ph[j + 1])]
            out.append(np.array([[r * math.sin(a) * math.cos(b), r * math.sin(a) * math.sin(b), r * math.cos(a)] for a, b in pts]) + c)
    return out


def scene_polys(lab, skip=(), fade=("table_top",)):
    """[(polygons, rgba)] for every drawn geom of the lab's model in its current state; ``skip`` leaves geoms out,
    ``fade`` draws them half-transparent (matplotlib does not sort depth across geoms: a big table hides small things)."""
    import mujoco
    m, d = lab.model, lab.data
    mujoco.mj_forward(m, d)
    out = []
    for i in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if name in skip or name.startswith("chiron_terrain"):
            continue
        t = m.geom_type[i]
        c, R, sz, rgba = d.geom_xpos[i], d.geom_xmat[i].reshape(3, 3), m.geom_size[i], m.geom_rgba[i].copy()
        if rgba[3] == 0:
            continue
        if name in fade:
            rgba[3] = 0.35
        if t == mujoco.mjtGeom.mjGEOM_BOX:
            out.append((_box(c, R, sz[:3]), rgba))
        elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
            out.append((_cyl(c, R, sz[0], sz[1]), rgba))
        elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            out.append((_cyl(c, R, sz[0], sz[1], caps=False) + _sphere(c + R[:, 2] * sz[1], sz[0], 4)
                        + _sphere(c - R[:, 2] * sz[1], sz[0], 4), rgba))
        elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
            out.append((_sphere(c, sz[0]), rgba))
    return out


def set_pose(lab, q: dict):
    import mujoco
    m = lab.model
    for n, v in q.items():
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        if j >= 0:
            lab.data.qpos[m.jnt_qposadr[j]] = v
    mujoco.mj_forward(m, lab.data)


def draw(ax, lab, lim=((0.5, 2.0), (-0.75, 0.75), (0.0, 1.6)), elev=28, azim=30, skip=()):
    for polys, rgba in sorted(scene_polys(lab, skip), key=lambda pr: -pr[1][3]):          # opaque first
        ax.add_collection3d(Poly3DCollection(polys, facecolors=[rgba], edgecolors=[rgba * [0.6, 0.6, 0.6, 1]],
                                             linewidths=0.2, shade=True))
    X, Y = np.meshgrid(np.linspace(*lim[0], 2), np.linspace(*lim[1], 2))
    ax.plot_surface(X, Y, np.zeros_like(X), color="#cfcfcf", alpha=0.3)
    ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1]); ax.set_zlim(*lim[2])
    ax.set_box_aspect([lim[0][1] - lim[0][0], lim[1][1] - lim[1][0], lim[2][1] - lim[2][0]])
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")


def main(out="manus_teleop_scenes.png"):
    fig = plt.figure(figsize=(18, 13))
    titles = {"pick_place": "pick and place: the block into the tray", "handover": "hand-over: left pincer to right",
              "rope": "pinch a 10 mm rope and lift it", "catch": "catch the swinging block at the end of its swing"}
    for k, name in enumerate(mt.TASKS):
        task = mt.task_scene(name)
        lab = mt.make_lab(task)
        root = lab._root_qadr
        lab.data.qpos[root:root + 3] = [0.0, 0.0, mt.base_height(lab.robot)]
        set_pose(lab, mt.start_pose(name))
        ax = fig.add_subplot(2, 2, k + 1, projection="3d")
        draw(ax, lab)
        ax.set_title(titles[name])
    fig.suptitle("Manus teleop bench: Onager Manus's two arms and pincers on a fixed base, the work table and the four tasks")
    fig.tight_layout()
    fig.savefig(out, dpi=95)
    # a close-up of the left pincer open over the block
    task = mt.task_scene("pick_place")
    lab = mt.make_lab(task)
    root = lab._root_qadr
    lab.data.qpos[root:root + 3] = [0.0, 0.0, mt.base_height(lab.robot)]
    q = mt.start_pose("pick_place")
    import onager_manus_robot as omr
    y = omr.arm_geometry()["y"]
    ik = omr.arm_ik((1.44, y, mt.grasp_pin_z(mt.TABLE_TOP + mt.BLOCK["size"] / 2) - mt.base_height(lab.robot)), -math.pi / 2, "L")
    q.update(dict(zip(omr.arm_joints("L")[:4], ik)))
    q.update({n: math.radians(12) for n in omr.arm_joints("L")[4:]})
    set_pose(lab, q)
    fig2 = plt.figure(figsize=(10, 9))
    ax = fig2.add_subplot(1, 1, 1, projection="3d")
    draw(ax, lab, lim=((1.24, 1.64), (0.08, 0.48), (0.50, 0.90)), elev=12, azim=75,
         skip=("table_top", "bench_block", "bench_column", "R_upper_arm", "R_forearm", "R_palm"))
    ax.set_title("left pincer open around the 50 mm block: straight 220 mm jaws, tips 5 mm above the table")
    out2 = out.replace(".png", "_pincer.png")
    fig2.tight_layout()
    fig2.savefig(out2, dpi=95)
    return out, out2


if __name__ == "__main__":
    print(main(*sys.argv[1:]))
