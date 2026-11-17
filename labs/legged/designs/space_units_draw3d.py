"""The space game units in 3D (matplotlib, from the CadQuery tessellation): the Marine, the Breacher and the Heavy
from the front quarter, their armour and weapons coloured, and the numbers of their robot models.
    python space_units_draw3d.py [out.png]"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import space_unit_robots as sr
import space_units as sun

CASING = "#5a6068"
BODY, ARMOUR, WEAPON, STEEL, RED, GOLD = "#d9cfae", "#2e4a6b", "#2b2b2b", "#555a60", "#9b1c22", "#c9a227"


def _tris(shape, tol=3.0):
    out = []
    for s in shape.solids().vals():
        v, t = s.tessellate(tol, 0.5)
        v = np.array([(q.x, q.y, q.z) for q in v])
        out += [v[list(tr)] for tr in t]
    return out


EDGE = (0.12, 0.12, 0.14, 1.0)


def pieces(d, p):
    """[(triangles, colour, edge colour or None)] of a unit: the skeleton, the actuators, then the armour (its plates'
    edges drawn) and the weapons, coloured."""
    out = [(_tris(d.build(dict(p, part="skeleton")), 4.0), BODY, None)]
    for name, shape in d.actuator_parts(p).items():
        out.append((_tris(shape, 4.0), "#3b3f45", None))
    for name, shape in d.armour_parts(p).items():
        out.append((_tris(shape), RED if name == "crest" else CASING if name in ("chest_bay", "belly_bay", "backpack") else ARMOUR, EDGE))
    for name, shape in d.weapon_parts(p).items():
        out.append((_tris(shape), STEEL if name in ("barrel", "head", "cannon", "shield", "saw_bar") else RED if name == "banner_cloth"
                    else GOLD if name == "emblem" else WEAPON, EDGE if name == "shield" else None))
    return out


def add_pieces(ax, pcs, offset=(0.0, 0.0, 0.0)):
    """One collection for all the pieces (matplotlib sorts collections, not faces, against each other: separate
    collections hide the plates behind the frame), the faces depth-sorted together; a piece's edges in its edge colour."""
    off = np.asarray(offset)
    tris = [t + off for pc in pcs for t in pc[0]]
    cols = [pc[1] for pc in pcs for _ in pc[0]]
    edges = [(pc[2] if len(pc) > 2 and pc[2] is not None else pc[1]) for pc in pcs for _ in pc[0]]
    widths = [(0.35 if len(pc) > 2 and pc[2] is not None else 0.0) for pc in pcs for _ in pc[0]]
    ax.add_collection3d(Poly3DCollection(tris, facecolors=cols, edgecolors=edges, linewidths=widths, shade=True, zsort="average"))


def draw(ax, cls, elev, azim, title):
    d = cls()
    p = d.resolve()
    add_pieces(ax, pieces(d, p))
    ax.set_xlim(-1150, 1150); ax.set_ylim(-1150, 1150); ax.set_zlim(0, 2300)
    ax.set_box_aspect((1, 1, 1), zoom=1.6); ax.view_init(elev=elev, azim=azim); ax.set_axis_off(); ax.set_title(title, fontsize=10)


def main(out="space_units_3d.png"):
    fig = plt.figure(figsize=(21, 8))
    units = (("marine", sun.SpaceMarine, sr.marine), ("marine_saw", sun.SpaceMarineSaw, sr.marine_saw),
             ("centurion", sun.Centurion, sr.centurion), ("breacher", sun.Breacher, sr.breacher), ("heavy", sun.Heavy, sr.heavy))
    for k, (unit, cls, build) in enumerate(units):
        r = build()
        p = cls().resolve()
        carried = sr.carried_masses(unit)
        armour = sum(v for q, v in carried.items() if q in sr.ARMOUR)
        weapon = sum(v for q, v in carried.items() if q not in sr.ARMOUR)
        title = (f"{cls.NAME}: {sun.Humanoid.total_height(p) / 1000:.2f} m, {r.total_mass():.0f} kg\n"
                 f"armour {armour:.0f} kg, carried {weapon:.0f} kg")
        draw(fig.add_subplot(1, 5, k + 1, projection="3d"), cls, 16, 35, title)
    fig.suptitle("Space game units on Talia's frame (design stage), both hands busy: the Marine (carbine, combat shield), the Marine with a chainsaw "
                 "(carbine, chainsaw), the Centurion (carbine, the imperial banner, a feathered crest), the Breacher (tower shield, power maul), the Heavy (rotary cannon)\n"
                 "a skeleton frame (an S-curved spine, ribs, mechanical joint modules) in power armour of Ti-6Al-4V; modules: cycloidal 1600 Nm hips and knees, 800 Nm ankles, waist and spine, harmonic 150 Nm arms; 25 actuated joints each",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    return out


if __name__ == "__main__":
    print(main(*sys.argv[1:]))
