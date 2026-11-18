"""CAD of the recoilless net launcher (``net_drone``): the recoilless golf cannon with a net cartridge.

From ``golf_recoilless_cad`` (chamber, side inlet, diaphragm valve, 3 L tank, 600 mm tube of bore 42.7 mm, rear
venturi, mount): the venturi's throat is the net design's 36.3 mm; the ball, its seat and the hopper are replaced by:

- **the net cartridge** (150 g) in the breech: a 42.5 mm x 80 mm PE shell; at its front four pockets splayed 12 deg
  from the axis, each with a 20 g brass corner weight (13 mm x 16 mm, ~18 g brass); behind them the folded 2.4 m net (55 g);
  a cup at the back that seals the bore. The cartridge is loaded through a hinged breech door (the old port);
- a **second cartridge**, cut in half, beside the mount: the pockets, the weights and the net pack.

    python net_launcher_cad.py      # STEP, GLB, section GLB, cartridge GLB in output/net_launcher/cad
"""
from __future__ import annotations

import math
from pathlib import Path

import cadquery as cq

import golf_recoilless_cad as grc
from golf_gun_cad import cyl_x

grc.THROAT = 36.3
COLORS = dict(grc.COLORS, shell=(0.92, 0.92, 0.88, 1), weight=(0.80, 0.62, 0.20, 1), net=(0.20, 0.55, 0.30, 1),
              door=(0.35, 0.37, 0.42, 1))
BORE = grc.BORE
SPLAY = 12.0


def cartridge(x0=16.0, cut=False):
    """The net cartridge along +x from x0 (its back). Returns {name: (shape, colour)}."""
    L, R = 80.0, BORE / 2 - 0.1
    shell = cyl_x(R, L, x0, r_in=R - 1.5).union(cyl_x(R, 3, x0))                        # tube and back cup
    front = cyl_x(R, 22, x0 + L - 22)                                                    # the pocket block
    weights = None
    for k in range(4):
        ang = math.radians(90 * k)                     # on the axes: the section plane cuts two pockets
        off = 9.0
        c = cq.Vector(x0 + L - 20, off * math.cos(ang), off * math.sin(ang))
        dirv = cq.Vector(math.cos(math.radians(SPLAY)), math.sin(math.radians(SPLAY)) * math.cos(ang),
                         math.sin(math.radians(SPLAY)) * math.sin(ang))
        pocket = cq.Workplane("XY").add(cq.Solid.makeCylinder(7.0, 26, c, dirv))
        front = front.cut(pocket)
        w = cq.Workplane("XY").add(cq.Solid.makeCylinder(6.5, 16, c + dirv * 2, dirv))
        weights = w if weights is None else weights.union(w)
    net = cyl_x(R - 2.0, L - 30, x0 + 4)                                                 # the folded net pack
    # the lines from the weights back into the pack (four thin cords)
    parts = {"shell": shell.union(front), "weight": weights, "net": net}
    if cut:
        half = cq.Workplane("XY").box(1000, 500, 1000).translate((0, 250, 0))
        parts = {k: v.cut(half) for k, v in parts.items()}
    return {f"cart_{k}": (v, k) for k, v in parts.items()}


def assembly():
    p = grc.assembly()
    for k in ("ball", "seat", "hopper", "feed"):
        p.pop(k, None)
    p.update(cartridge(16.0))
    # the breech door over the loading port: a hinged plate along the tube's top
    od_b = BORE + 6
    door = (cq.Workplane("XY").box(110, BORE * 0.9, 4).translate((16 + 55, 0, od_b / 2 + 4))
            .union(cyl_x(3, 110, 16, 0 - BORE * 0.45 - 2, od_b / 2 + 4)))
    p["door"] = (door, "door")
    # the display cartridge, cut, standing beside the mount (axis up)
    for k, (s, c) in cartridge(0.0, cut=True).items():
        s = s.rotate((0, 0, 0), (0, 1, 0), -90).translate((300, 170, -260))
        p["show_" + k] = (s, c)
    return p


def export(root=Path("output/net_launcher/cad")):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    parts = assembly()
    asm = cq.Assembly(name="net_launcher")
    for name, (shape, col) in parts.items():
        asm.add(shape, name=name, color=cq.Color(*COLORS[col]))
    asm.export(str(root / "net_launcher.step"))
    asm.export(str(root / "net_launcher.glb"), tolerance=0.2, angularTolerance=0.2)
    half = cq.Workplane("XY").box(4000, 2000, 4000).translate((0, 1000, 0))
    sec = cq.Assembly(name="net_launcher_section")
    for name, (shape, col) in parts.items():
        if name != "jet" and not name.startswith("show_"):
            sec.add(shape.cut(half), name=name, color=cq.Color(*COLORS[col]))
    sec.export(str(root / "net_launcher_section.glb"), tolerance=0.2, angularTolerance=0.2)
    cart = cq.Assembly(name="cartridge")
    for name, (shape, col) in cartridge(0.0, cut=True).items():
        cart.add(shape, name=name, color=cq.Color(*COLORS[col]))
    cart.export(str(root / "net_cartridge.glb"), tolerance=0.05, angularTolerance=0.1)
    return root


if __name__ == "__main__":
    print(export())
