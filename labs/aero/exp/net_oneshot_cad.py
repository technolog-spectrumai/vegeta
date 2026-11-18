"""CAD of the extremely light, one-shot recoilless net interceptor (for bandit quadcopters and FPV drones).

One shot per fill: no HPA bottle, no regulator, no separate plenum. The charge vessel is the tank:

    one-shot carbon charge vessel (~80 mL, filled at the base) -> 25 mm burst-diaphragm holder (a puncture solenoid
    breaks the disc: full bore in ~0.5 ms) -> side inlet into the chamber -> the 70 g net cartridge in the breech ->
    0.35 m carbon tube, bore 32 mm -> muzzle;  chamber rear -> throat 27.2 mm -> 2:1 cone -> the jet.
    A single clamp bracket with a pivot to the Pekari's pan-tilt; the breech door on top.

The sizes come from the light interceptor (``net_light_design``: 45 m/s, throat 27.2 mm); the vessel's volume and
charge pressure are to be sized for the burst diaphragm (provisional here: 80 mL). Frame: x along the tube, z up,
origin at the bore's start. ``export()`` -> STEP, GLB, section GLB.
"""
from __future__ import annotations

import math
from pathlib import Path

import cadquery as cq

from golf_gun_cad import cyl_x, cyl_y, cyl_z
from golf_recoilless_cad import revolve_x

BORE, TUBE_OD, TUBE_L = 32.0, 35.0, 350.0
THROAT, EPS = 27.2, 2.0
VESSEL_ML = 80.0
SPLAY = 12.0
COLORS = {
    "tube": (0.15, 0.15, 0.17, 1), "chamber": (0.70, 0.72, 0.76, 1), "nozzle": (0.72, 0.45, 0.20, 1),
    "jet": (0.55, 0.75, 0.95, 0.25), "vessel": (0.12, 0.12, 0.14, 1), "burst": (0.80, 0.62, 0.20, 1),
    "solenoid": (0.60, 0.15, 0.15, 1), "door": (0.45, 0.47, 0.52, 1), "bracket": (0.35, 0.35, 0.38, 1),
    "shell": (0.92, 0.92, 0.88, 1), "weight": (0.80, 0.62, 0.20, 1), "net": (0.20, 0.55, 0.30, 1),
    "deck": (0.45, 0.40, 0.30, 1), "fill": (0.70, 0.70, 0.20, 1),
}


def cartridge(x0, cut=False):
    L, R = 70.0, BORE / 2 - 0.1
    shell = cyl_x(R, L, x0, r_in=R - 1.2).union(cyl_x(R, 2.5, x0))
    front = cyl_x(R, 18, x0 + L - 18)
    weights = None
    for k in range(4):
        ang = math.radians(90 * k)
        c = cq.Vector(x0 + L - 16, 7.0 * math.cos(ang), 7.0 * math.sin(ang))
        d = cq.Vector(math.cos(math.radians(SPLAY)), math.sin(math.radians(SPLAY)) * math.cos(ang),
                      math.sin(math.radians(SPLAY)) * math.sin(ang))
        front = front.cut(cq.Workplane("XY").add(cq.Solid.makeCylinder(5.0, 22, c, d)))
        w = cq.Workplane("XY").add(cq.Solid.makeCylinder(4.6, 12, c + d * 2, d))          # 8 g brass
        weights = w if weights is None else weights.union(w)
    net = cyl_x(R - 1.8, L - 24, x0 + 3)
    parts = {"shell": shell.union(front), "weight": weights, "net": net}
    if cut:
        half = cq.Workplane("XY").box(1000, 500, 1000).translate((0, 250, 0))
        parts = {k: v.cut(half) for k, v in parts.items()}
    return parts


def assembly():
    p = {}
    # carbon tube with a muzzle ring; the breech door over the loading port
    port = cq.Workplane("XY").box(74, BORE * 0.9, 12).translate((12 + 37, 0, TUBE_OD / 2))
    tube = cyl_x(TUBE_OD / 2, TUBE_L - 8, 8, r_in=BORE / 2).cut(port).union(cyl_x(TUBE_OD / 2 + 1.5, 12, TUBE_L - 12, r_in=BORE / 2))
    p["tube"] = (tube, "tube")
    p["door"] = (cq.Workplane("XY").box(78, BORE * 0.95, 2.5).translate((12 + 37, 0, TUBE_OD / 2 + 1.5))
                 .union(cyl_x(2, 78, 10, -BORE * 0.48, TUBE_OD / 2 + 1.5)), "door")
    # chamber (aluminium), side inlet below; rear venturi
    ch_in, ch_len = 40.0, 40.0
    p["chamber"] = (revolve_x([(-ch_len, ch_in / 2), (0, ch_in / 2), (0, BORE / 2), (8, BORE / 2), (8, TUBE_OD / 2 + 3),
                               (-ch_len - 5, ch_in / 2 + 3), (-ch_len - 5, ch_in / 2)])
                    .union(cyl_z(16, 12, -ch_len / 2, 0, -ch_in / 2 - 14)).cut(cyl_z(12.5, 40, -ch_len / 2, 0, -ch_in / 2 - 25)), "chamber")
    r_t, r_e = THROAT / 2, THROAT / 2 * math.sqrt(EPS)
    x0 = -ch_len - 5
    x_t = x0 - 10
    x_e = x_t - (r_e - r_t) / math.tan(math.radians(7.5))
    p["nozzle"] = (revolve_x([(x0, ch_in / 2), (x_t, r_t), (x_e, r_e), (x_e, r_e + 1.5), (x_t, r_t + 2.5), (x0, ch_in / 2 + 3)]), "nozzle")
    p["jet"] = (revolve_x([(x_e, 0), (x_e, r_e), (x_e - 600, r_e + 600 * math.tan(math.radians(12))), (x_e - 600, 0)]), "jet")
    # burst-diaphragm holder under the chamber, the puncture solenoid beside it; the one-shot vessel below
    zb = -ch_in / 2 - 14 - 10
    p["burst"] = (cyl_z(19, 20, -ch_len / 2, 0, zb - 10).cut(cyl_z(12.5, 24, -ch_len / 2, 0, zb - 12))
                  .union(cyl_z(21, 3, -ch_len / 2, 0, zb - 1.5)), "burst")
    p["solenoid"] = (cyl_y(9, 32, -ch_len / 2, zb).translate((0, 19 + 16, 0)), "solenoid")
    vr = 22.0                                                     # an 80 mL carbon vessel, 44 mm OD
    vl = VESSEL_ML * 1e3 / (math.pi * (vr - 2) ** 2) - 4 / 3 * (vr - 2)
    zv_top = zb - 10
    vessel = (cyl_z(vr, vl, -ch_len / 2, 0, zv_top - vr - vl).union(cq.Workplane("XY").sphere(vr).translate((-ch_len / 2, 0, zv_top - vr)))
              .union(cq.Workplane("XY").sphere(vr).translate((-ch_len / 2, 0, zv_top - vr - vl))))
    p["vessel"] = (vessel.union(cyl_z(10, 8, -ch_len / 2, 0, zv_top - 6)), "vessel")
    p["fill"] = (cyl_x(5, 14, -ch_len / 2 + vr - 2, 0, zv_top - vr - vl / 2), "fill")
    # the cartridge in the breech
    for k, v in cartridge(12.0).items():
        p["cart_" + k] = (v, k)
    # one clamp bracket at the balance point with a pivot, a small base plate
    xc = 110.0                                                   # forward of the vessel, near the loaded balance point
    clamp = cyl_x(TUBE_OD / 2 + 4, 24, xc, r_in=TUBE_OD / 2).union(
        cq.Workplane("XY").box(24, 8, 70).translate((xc + 12, 0, -TUBE_OD / 2 - 35)))
    pivot = cyl_y(6, 50, xc + 12, -TUBE_OD / 2 - 62)
    p["bracket"] = (clamp.union(pivot).union(cq.Workplane("XY").box(40, 50, 6).translate((xc + 12, 0, -TUBE_OD / 2 - 75))), "bracket")
    p["deck"] = (cq.Workplane("XY").box(160, 140, 6).translate((xc + 12, 0, -TUBE_OD / 2 - 82)), "deck")
    return p


def export(root=Path("output/net_oneshot/cad")):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    parts = assembly()
    asm = cq.Assembly(name="net_oneshot")
    for name, (shape, col) in parts.items():
        asm.add(shape, name=name, color=cq.Color(*COLORS[col]))
    asm.export(str(root / "net_oneshot.step"))
    asm.export(str(root / "net_oneshot.glb"), tolerance=0.08, angularTolerance=0.15)
    half = cq.Workplane("XY").box(4000, 2000, 4000).translate((0, 1000, 0))
    sec = cq.Assembly(name="net_oneshot_section")
    for name, (shape, col) in parts.items():
        if name != "jet":
            sec.add(shape.cut(half), name=name, color=cq.Color(*COLORS[col]))
    sec.export(str(root / "net_oneshot_section.glb"), tolerance=0.08, angularTolerance=0.15)
    return root


if __name__ == "__main__":
    print(export())
