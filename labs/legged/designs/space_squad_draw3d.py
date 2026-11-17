"""A squad of the space game units in 3D (matplotlib, from the CadQuery tessellation): the Centurion leading at the
point of a wedge with the banner, two Marines with shields on its flanks, two Marines with chainsaws behind them,
the Breacher and the Heavy at the rear - each unit tessellated once and set down on the ground at its place in the
formation, facing the same way.    python space_squad_draw3d.py [out.png]"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import space_units as sun
import space_units_draw3d as dd

#: the formation: (unit class, x forward, y left) [mm]; the Centurion at the point
SQUAD = [(sun.Centurion, 1600.0, 0.0),
         (sun.SpaceMarine, 400.0, 1300.0), (sun.SpaceMarine, 400.0, -1300.0),
         (sun.SpaceMarineSaw, -900.0, 2400.0), (sun.SpaceMarineSaw, -900.0, -2400.0),
         (sun.Breacher, -2100.0, 1100.0), (sun.Heavy, -2100.0, -1100.0)]
ELEV, AZIM = 24.0, 12.0


def pieces(cls):
    """[(triangles, colour)] of one unit at the origin, coloured as the roster drawing."""
    d = cls()
    return dd.pieces(d, d.resolve())


def main(out="space_squad_3d.png"):
    cache = {}
    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    X, Y = np.meshgrid(np.linspace(-3600, 3200, 2), np.linspace(-3600, 3600, 2))
    ax.plot_surface(X, Y, np.zeros_like(X) - 2.0, color="#6f6a60", alpha=0.35)
    # matplotlib does not depth-sort between collections: draw the units farthest from the camera first
    cam = np.array([np.cos(np.radians(ELEV)) * np.cos(np.radians(AZIM)), np.cos(np.radians(ELEV)) * np.sin(np.radians(AZIM))])
    for cls, x, y in sorted(SQUAD, key=lambda u: u[1] * cam[0] + u[2] * cam[1]):
        if cls not in cache:
            cache[cls] = pieces(cls)
        dd.add_pieces(ax, cache[cls], (x, y, 0.0))
    ax.set_xlim(-3000, 2600); ax.set_ylim(-3100, 3100); ax.set_zlim(0, 2800)
    ax.set_box_aspect((5.6, 6.2, 2.8), zoom=1.4); ax.view_init(elev=ELEV, azim=AZIM); ax.set_axis_off()
    fig.suptitle("A squad of the space game units: the Centurion leading with the banner, two Marines with shields on its flanks, "
                 "two with chainsaws behind, the Breacher and the Heavy at the rear", fontsize=11)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.95, bottom=0.0)
    fig.savefig(out, dpi=130)
    return out


if __name__ == "__main__":
    print(main(*sys.argv[1:]))
