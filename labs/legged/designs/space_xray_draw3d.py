"""The Marine with its armour transparent (matplotlib, from the CadQuery tessellation): the skeleton and the joint
modules solid, the linear actuators (the pistons, bronze), the lorica's bands, the shoulder guards, the arm lames, the
galea, the belt, the greaves and the boots as glass, the chest bay (the electronics) and the pack as
tinted casings, the weapons solid - from the front quarter, the side and the back.    python space_xray_draw3d.py [out.png]"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import space_units as sun
import space_units_draw3d as dd

BONE, MODULE, GLASS, CASING, WEAPON = (0.9, 0.86, 0.73, 1.0), (0.29, 0.31, 0.34, 1.0), (0.18, 0.29, 0.42, 0.25), (0.35, 0.38, 0.42, 0.45), (0.17, 0.17, 0.17, 1.0)
STRUT = (0.55, 0.3, 0.12, 1.0)


def main(out="space_xray_3d.png"):
    d = sun.SpaceMarine()
    p = d.resolve()
    pcs = [(dd._tris(d.build(dict(p, part="skeleton")), 3.0), BONE, None)]
    for name, shape in d.actuator_parts(p).items():
        pcs.append((dd._tris(shape, 4.0), STRUT, None))
    for name, shape in d.armour_parts(p).items():
        pcs.append((dd._tris(shape), CASING if name in ("chest_bay", "belly_bay", "backpack") else GLASS, (0.1, 0.16, 0.26, 0.9)))
    for name, shape in d.weapon_parts(p).items():
        pcs.append((dd._tris(shape), (0.33, 0.35, 0.38, 0.35) if name == "shield" else WEAPON, (0.1, 0.1, 0.12, 0.9) if name == "shield" else None))
    fig = plt.figure(figsize=(18, 10))
    for k, (el, az, title) in enumerate(((10, 32, "front quarter"), (4, 90, "left side"), (10, 180, "back"))):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        dd.add_pieces(ax, pcs)
        ax.set_xlim(-900, 900); ax.set_ylim(-900, 900); ax.set_zlim(0, 2000)
        ax.set_box_aspect((0.9, 0.9, 1), zoom=1.5); ax.view_init(elev=el, azim=az); ax.set_axis_off(); ax.set_title(title, fontsize=11)
    fig.suptitle("Belisario (the gorilla-human hybrid Marine) with the armour as glass: the skeleton, its joint modules and linear actuators (bronze),\n"
                 "the lorica segmentata's hoops, shoulder guards and arm lames, the galea, the chest and belly bays (electronics), the shield held in the fingers", fontsize=11)
    fig.subplots_adjust(left=0, right=1, top=0.94, bottom=0)
    fig.savefig(out, dpi=110)
    return out


if __name__ == "__main__":
    print(main(*sys.argv[1:]))
