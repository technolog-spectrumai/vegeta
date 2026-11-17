"""The space game units' skeleton (matplotlib, from the CadQuery tessellation): the Marine's bare frame from the
front quarter, the side and the back - the S-curved spine of vertebrae, the ribcage and sternum, the clavicles and
scapulae, the pelvis's wings on the sacrum, the femurs, tibiae and fibulae, and the mechanical joint modules at
every joint (darker).    python space_skeleton_draw3d.py [out.png]"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import space_units as sun
import space_units_draw3d as dd

BONE, MODULE, HEAD = "#e6dcbb", "#4a4f57", "#8e969f"


def main(out="space_skeleton_3d.png"):
    d = sun.SpaceMarine()
    p = d.resolve()
    pieces = sun.skeleton_trunk(p)
    body = dd._tris(d.build(dict(p, part="skeleton")), 2.5)
    # the joint modules drawn over the body in their own colour: the hips, the knees, the ankles, the shoulders
    import math
    hh = sun.Humanoid.hip_height(p)
    a, L1 = math.radians(p["knee_bend_deg"]), p["thigh_length"]
    z0, Wt, Dt, Ht, zs = sun._torso_frame(p)
    modules = []
    for sy in (-1, 1):
        y = sy * p["hip_spacing"] / 2
        modules.append(sun.joint_module((0, y, hh), p, "y"))
        modules.append(sun.joint_module((L1 * math.sin(a), y, hh - L1 * math.cos(a)), p, "y", scale=0.9))
        modules.append(sun.joint_module((0, y, p["ankle_height"]), p, "y", scale=0.8))
        modules.append(sun.joint_module((0, sy * (Wt / 2 + 10), zs), p, "y", scale=p["arm_diameter"] * 1.3 / p["joint_module_diameter"]))
    mods = [t for m in modules for t in dd._tris(m, 2.5)]
    fig = plt.figure(figsize=(18, 10))
    for k, (el, az, title) in enumerate(((10, 32, "front quarter"), (4, 90, "left side"), (10, 180, "back"))):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        dd.add_pieces(ax, [(body, BONE, None), (mods, MODULE, None)])
        ax.set_xlim(-800, 800); ax.set_ylim(-800, 800); ax.set_zlim(0, 1950)
        ax.set_box_aspect((0.82, 0.82, 1), zoom=1.55); ax.view_init(elev=el, azim=az); ax.set_axis_off(); ax.set_title(title, fontsize=11)
    n = int(p["spine_segments"])
    fig.suptitle(f"The Marine's skeleton: an S-curved spine of {n} vertebrae set back in the chest, {int(p['rib_count'])} rib pairs to the sternum, "
                 "clavicles and scapulae, iliac wings on the sacrum, femurs, tibiae and fibulae - every joint a mechanical rotary module (dark)", fontsize=11)
    fig.subplots_adjust(left=0, right=1, top=0.94, bottom=0)
    fig.savefig(out, dpi=110)
    return out


if __name__ == "__main__":
    print(main(*sys.argv[1:]))
