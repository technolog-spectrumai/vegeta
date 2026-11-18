"""CAD of the recoilless 8 mm emitter (``bearing_recoilless``): the whole assembly and a half section.

    HPA bottle (0.8 L, 300 bar) + regulator under the rear -> HP hose -> 0.5 L plenum under the tube -> riser ->
    1/2" hammer valve (pilot solenoid) -> side inlet into the chamber -> the ball on its seat -> 1.0 m tube, 8 mm bore,
    with a muzzle crown;  chamber rear -> throat 6.9 mm -> 2:1 cone (exit 9.8 mm) -> the jet (a translucent cone:
    its first 0.5 m);  a gravity magazine (a 9 mm tube of balls) into the breech port;  clamps, cradle, pan-tilt,
    the basket lid.

Frame: x along the tube, z up, origin at the bore's start. ``export()`` -> STEP, GLB and the section GLB.
"""
from __future__ import annotations

import math
from pathlib import Path

import cadquery as cq

import bearing_recoilless as br
from golf_gun_cad import COLORS as GC, coupler, cyl_x, cyl_y, cyl_z, gauge, relief_valve, _hose
from golf_recoilless_cad import revolve_x

BORE = 8.0
BARREL = br.BARREL_M * 1e3
THROAT = 6.91
EPS = br.EPS
COLORS = dict(GC, chamber=(0.55, 0.57, 0.62, 1), nozzle=(0.72, 0.45, 0.20, 1), jet=(0.55, 0.75, 0.95, 0.25),
              magazine=(0.30, 0.65, 0.40, 1), balls=(0.85, 0.86, 0.90, 1))


def assembly():
    p = {}
    od = 16.0                                            # tube OD (a 16 x 4 honed hydraulic tube)
    # chamber block: 24 mm bore, 50 mm long, 10 mm wall; the side inlet below
    ch_in, ch_len = 24.0, 50.0
    ch = revolve_x([(-ch_len, ch_in / 2), (0, ch_in / 2), (0, BORE / 2), (10, BORE / 2), (10, od / 2 + 8),
                    (-ch_len - 8, ch_in / 2 + 10), (-ch_len - 8, ch_in / 2)])
    ch = ch.union(cyl_z(10, 14, -ch_len / 2, 0, -ch_in / 2 - 18)).cut(cyl_z(6, 40, -ch_len / 2, 0, -ch_in / 2 - 30))
    p["chamber"] = (ch, "chamber")
    # the venturi
    r_t, r_e = THROAT / 2, THROAT / 2 * math.sqrt(EPS)
    x0 = -ch_len - 8
    x_t = x0 - 14
    x_e = x_t - (r_e - r_t) / math.tan(math.radians(7.5))
    noz = revolve_x([(x0, ch_in / 2), (x_t, r_t), (x_e, r_e), (x_e, r_e + 3), (x_t, r_t + 5), (x0, ch_in / 2 + 10)])
    p["nozzle"] = (noz, "nozzle")
    p["jet"] = (revolve_x([(x_e, 0), (x_e, r_e), (x_e - 500, r_e + 500 * math.tan(math.radians(12))), (x_e - 500, 0)]), "jet")
    # the ball, the tube with its breech port and crown
    p["ball"] = (cq.Workplane("XY").sphere(4.0).translate((14, 0, 0)), "ball")
    port = cq.Workplane("XY").box(9, 8.5, 12).translate((14, 0, od / 2))
    tube = cyl_x(od / 2, BARREL - 10, 10, r_in=BORE / 2).cut(port).union(cyl_x(od / 2 + 3, 20, BARREL - 20, r_in=BORE / 2))
    p["barrel"] = (tube, "barrel")
    # gravity magazine: a 9 mm bore tube rising at 30 degrees, a few balls in it
    a = math.radians(30)
    ml = 160.0
    base = cq.Vector(14, 0, od / 2 + 2)
    d = cq.Vector(math.cos(a), 0, math.sin(a))           # rising toward the muzzle, clear of the chamber
    mag = cq.Workplane("XY").add(cq.Solid.makeCylinder(7, ml, base, d)).cut(cq.Workplane("XY").add(cq.Solid.makeCylinder(4.6, ml + 1, base, d)))
    p["magazine"] = (mag.union(cq.Workplane("XY").add(cq.Solid.makeCylinder(9, 12, base + d * ml, d))), "magazine")
    balls = None
    for k in range(1, 15):
        c = base + d * (8.2 * k + 4)
        b = cq.Workplane("XY").sphere(4.0).translate(c.toTuple())
        balls = b if balls is None else balls.union(b)
    p["balls"] = (balls, "balls")
    # hammer valve under the chamber, its pilot solenoid; riser to the plenum
    vb = 46.0
    z_v = -ch_in / 2 - 18 - vb / 2
    body = cq.Workplane("XY").box(vb, vb, vb).edges("|Z").fillet(3).translate((-ch_len / 2, 0, z_v))
    p["valve"] = (body, "valve")
    p["actuator"] = (cyl_x(16, 70, -ch_len / 2 - vb / 2 - 70, 0, z_v).union(cyl_z(9, 30, -ch_len / 2 - vb / 2 - 30, 0, z_v + 14)), "actuator")
    # the plenum: 0.5 L, 60 mm OD steel, under the tube's first half
    pr, pl = 30.0, 0.5e6 / (math.pi * 25 ** 2)
    z_p = z_v - vb / 2 - 30 - pr
    x_p0 = -ch_len / 2 - 20
    plen = (cyl_x(pr, pl, x_p0 + pr, 0, z_p).union(cq.Workplane("XY").sphere(pr).translate((x_p0 + pr, 0, z_p)))
            .union(cq.Workplane("XY").sphere(pr).translate((x_p0 + pr + pl, 0, z_p))))
    p["tank"] = (plen, "tank")
    riser_top = z_v - vb / 2
    p["pipe"] = (cyl_z(7, riser_top - (z_p + pr - 3), -ch_len / 2, 0, z_p + pr - 3).union(cyl_y(7, 40, -ch_len / 2, (riser_top + z_p + pr) / 2)), "pipe")
    zg = (riser_top + z_p + pr) / 2
    p["gauge"] = (cq.Workplane("XY").add(gauge((-ch_len / 2, 24, zg)).val()).rotate((-ch_len / 2, 20, zg), (1, 0, 0), -90), "gauge")
    p["relief"] = (relief_valve((x_p0 + pr + pl * 0.6, 0, z_p + pr - 3)), "relief")
    # HPA bottle + regulator under the plenum, the HP hose up to the plenum's front port
    rb, lb = 45.0, 280.0
    zb = z_p - pr - 25 - rb
    xb0 = x_p0 + 20
    bottle = (cyl_x(rb, lb - 2 * rb, xb0 + rb, 0, zb).union(cq.Workplane("XY").sphere(rb).translate((xb0 + rb, 0, zb)))
              .union(cq.Workplane("XY").sphere(rb).translate((xb0 + lb - rb, 0, zb))).union(cyl_x(14, 25, xb0 + lb - 5, 0, zb)))
    p["bottle"] = (bottle, "bottle")
    p["regulator"] = (cyl_x(20, 50, xb0 + lb + 20, 0, zb).union(gauge((xb0 + lb + 32, 0, zb + 16))), "regulator")
    x_front = x_p0 + 2 * pr + pl
    p["coupler"] = (coupler((x_front - 4, 0, z_p)), "coupler")
    p0 = cq.Vector(xb0 + lb + 70, 0, zb)
    p1 = cq.Vector(x_front + 36, 0, z_p)
    p["hose"] = (_hose(p0, cq.Vector(max(p0.x, p1.x) + 60, 0, (p0.z + p1.z) / 2), p1, 5.0), "hose")
    # clamps from the tube to the plenum, cradle, yoke, turntable, deck
    clamps = None
    for xc in (40.0, 260.0, 600.0, BARREL - 120):
        cl = cyl_x(od / 2 + 6, 18, xc, r_in=od / 2)
        if xc < x_front:
            cl = cl.union(cyl_x(pr + 5, 18, xc, 0, z_p, r_in=pr)).union(
                cq.Workplane("XY").box(18, 10, -od / 2 - (z_p + pr)).translate((xc + 9, 0, (-od / 2 + z_p + pr) / 2)))
        else:
            cl = cl.union(cq.Workplane("XY").box(18, 10, 40).translate((xc + 9, 0, -od / 2 - 20)))
        clamps = cl if clamps is None else clamps.union(cl)
    rail = cq.Workplane("XY").box(BARREL - 20, 10, 10).translate((BARREL / 2 + 30, 0, -od / 2 - 40))
    x_piv, z_piv = 260.0, z_p
    side = 2 * pr + 50
    plate = cq.Workplane("XY").box(260, 6, 160)
    cradle = (plate.translate((x_piv, side / 2, z_p)).union(plate.translate((x_piv, -side / 2, z_p))).union(cyl_y(10, side + 40, x_piv, z_piv)))
    p["clamp"] = (clamps.union(rail).union(cradle), "clamp")
    z_tt = zb - rb - 30
    yk = cq.Workplane("XY").box(70, 8, z_piv - z_tt + 20)
    yoke = (yk.translate((x_piv, side / 2 + 14, (z_piv + z_tt) / 2 + 10)).union(yk.translate((x_piv, -side / 2 - 14, (z_piv + z_tt) / 2 + 10)))
            .union(cq.Workplane("XY").box(80, side + 36, 10).translate((x_piv, 0, z_tt + 15))))
    p["mount"] = (yoke.union(cyl_z(70, 12, x_piv, 0, z_tt)).union(cyl_z(50, 10, x_piv, 0, z_tt - 10)), "mount")
    p["deck"] = (cq.Workplane("XY").box(380, 260, 8).translate((x_piv, 0, z_tt - 14)), "deck")
    return p


def export(root=Path("output/bearing_recoilless/cad")):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    parts = assembly()
    asm = cq.Assembly(name="bearing_recoilless")
    for name, (shape, col) in parts.items():
        asm.add(shape, name=name, color=cq.Color(*COLORS[col]))
    asm.export(str(root / "bearing_recoilless.step"))
    asm.export(str(root / "bearing_recoilless.glb"), tolerance=0.1, angularTolerance=0.2)
    half = cq.Workplane("XY").box(4000, 2000, 4000).translate((0, 1000, 0))
    sec = cq.Assembly(name="bearing_recoilless_section")
    for name, (shape, col) in parts.items():
        if name != "jet":
            sec.add(shape.cut(half), name=name, color=cq.Color(*COLORS[col]))
    sec.export(str(root / "bearing_recoilless_section.glb"), tolerance=0.1, angularTolerance=0.2)
    # the breech alone (chamber, nozzle, ball, the first 120 mm of tube), sectioned, for a close-up
    close = cq.Assembly(name="breech")
    win = cq.Workplane("XY").box(260, 200, 200).translate((-40, 0, 0))
    for name in ("chamber", "nozzle", "ball", "barrel", "valve", "magazine", "balls"):
        shape, col = parts[name]
        close.add(shape.intersect(win).cut(half), name=name, color=cq.Color(*COLORS[col]))
    close.export(str(root / "bearing_recoilless_breech.glb"), tolerance=0.05, angularTolerance=0.2)
    return root


if __name__ == "__main__":
    print(export())
