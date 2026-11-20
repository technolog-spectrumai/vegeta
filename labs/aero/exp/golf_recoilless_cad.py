"""The recoilless golf air cannon (``golf_recoilless``) as an assembly: the venturi breech and the rest of the gun.

    tank (under the barrel's front, clear of the back-blast) -> riser -> diaphragm valve -> side inlet into the
    chamber -> the ball on its seat at the bore's start -> 600 mm barrel;  the chamber's rear -> convergent cone ->
    throat -> 2:1 divergent cone -> the jet out of the back (a translucent cone shows its first metre).

Frame: x along the barrel (forward), z up, the origin at the bore's start (the chamber's front face). The design point:
the subsonic-fast case recoilless: 18.4 bar, throat 38 mm (``golf_recoilless.summary``). ``export()`` -> STEP + GLB.
"""
from __future__ import annotations

import math
from pathlib import Path

import cadquery as cq

import golf_gun as gg
from golf_gun_cad import COLORS as GC, coupler, cyl_x, cyl_y, cyl_z, gauge, relief_valve

BORE = gg.BORE * 1e3
R_BALL = gg.D_BALL * 500
BARREL = gg.BARREL * 1e3
THROAT = 38.0
EPS = 2.0

COLORS = dict(GC, chamber=(0.55, 0.57, 0.62, 1), nozzle=(0.72, 0.45, 0.20, 1), jet=(0.55, 0.75, 0.95, 0.25),
              seat=(0.95, 0.45, 0.10, 1))


def revolve_x(profile):
    """A solid of revolution about the x axis from an (x, r) outline (closed, r >= 0)."""
    w = cq.Workplane("XY").polyline(profile).close()
    return w.revolve(360, (0, 0, 0), (1, 0, 0))


def assembly():
    parts = {}
    od_b = BORE + 6
    # the chamber: a thick tube, the bore's start at x = 0, a side inlet at its bottom
    ch_in, ch_len, wall = 72.0, 90.0, 8.0
    chamber = revolve_x([(-ch_len, ch_in / 2), (0, ch_in / 2), (0, BORE / 2), (12, BORE / 2), (12, od_b / 2 + 10),
                         (-ch_len - 10, ch_in / 2 + wall), (-ch_len - 10, ch_in / 2)])
    chamber = chamber.union(cyl_z(30, 30, -ch_len / 2, 0, -ch_in / 2 - wall - 22)).cut(cyl_z(19, 60, -ch_len / 2, 0, -ch_in / 2 - 40))
    parts["chamber"] = (chamber, "chamber")
    # the rear nozzle: convergent from the chamber to the throat, then the 2:1 divergent cone (7.5 degree half angle)
    r_t, r_e = THROAT / 2, THROAT / 2 * math.sqrt(EPS)
    x0 = -ch_len - 10
    x_t = x0 - 30
    x_e = x_t - (r_e - r_t) / math.tan(math.radians(7.5))
    inner = [(x0, ch_in / 2), (x_t, r_t), (x_e, r_e)]
    outer = [(x_e, r_e + 4), (x_t, r_t + 6), (x0, ch_in / 2 + wall)]
    nozzle = revolve_x(inner + outer)
    flange = revolve_x([(x0, ch_in / 2), (x0 + 6, ch_in / 2), (x0 + 6, ch_in / 2 + wall + 10), (x0, ch_in / 2 + wall + 10)])
    parts["nozzle"] = (nozzle.union(flange), "nozzle")
    # the jet's first metre (display): a cone opening at ~12 degrees from the exit
    parts["jet"] = (revolve_x([(x_e, 0), (x_e, r_e), (x_e - 1000, r_e + 1000 * math.tan(math.radians(12))), (x_e - 1000, 0)]), "jet")
    # the ball on a seat ring at the bore's start, the barrel
    parts["seat"] = (cyl_x(BORE / 2, 4, 12, r_in=BORE / 2 - 2.5), "seat")
    parts["ball"] = (cq.Workplane("XY").sphere(R_BALL).translate((16 + R_BALL, 0, 0)), "ball")
    port = cq.Workplane("XY").box(BORE * 1.05, BORE * 0.95, 30).translate((16 + R_BALL, 0, od_b / 2))
    barrel = cyl_x(od_b / 2, BARREL - 12, 12, r_in=BORE / 2).cut(port).union(cyl_x(od_b / 2 + 4, 25, BARREL - 25, r_in=BORE / 2))
    parts["barrel"] = (barrel, "barrel")
    # hopper and feed into the port
    a = math.radians(35)
    fl = 120.0
    fx, fz = 16 + R_BALL + fl * math.sin(a), od_b / 2 + 2 + fl * math.cos(a)
    d = cq.Vector(-math.sin(a), 0, -math.cos(a))
    feed = cq.Workplane("XY").add(cq.Solid.makeCylinder(BORE / 2 + 2.5, fl, cq.Vector(fx, 0, fz), d)).cut(
        cq.Workplane("XY").add(cq.Solid.makeCylinder(BORE / 2 + 0.5, fl + 2, cq.Vector(fx, 0, fz), d)))
    parts["feed"] = (feed, "feed")
    parts["hopper"] = (cq.Workplane("XY").rect(BORE + 8, BORE + 8).workplane(offset=100).rect(220, 160).loft()
                       .faces(">Z").shell(-2.5).translate((fx, 0, fz - 5)), "hopper")
    # the valve under the chamber, its riser down to the tank; the tank under the barrel's front half
    vb = 72.0
    z_v = -ch_in / 2 - wall - 22 - vb / 2
    body = cq.Workplane("XY").box(vb, vb, vb).edges("|Z").fillet(4).translate((-ch_len / 2, 0, z_v))
    body = body.cut(cyl_z(19, vb + 2, -ch_len / 2, 0, z_v - vb / 2 - 1))
    parts["valve"] = (body, "valve")
    bon = cyl_y(vb * 0.55, 18, -ch_len / 2, z_v).translate((0, vb / 2 + 9, 0))
    parts["actuator"] = (bon.union(cyl_y(14, 40, -ch_len / 2 + 20, z_v).translate((0, vb / 2 + 38, 0))), "actuator")
    t_r, t_len = 56.5, 300.0                                   # 3 L, 16 bar rated (18.4 bar needs a 25 bar tank: see notes)
    z_t = z_v - vb / 2 - 40 - t_r
    x_t0 = -ch_len / 2 - 40
    tank = (cyl_x(t_r, t_len, x_t0 + t_r, 0, z_t).union(cq.Workplane("XY").sphere(t_r).translate((x_t0 + t_r, 0, z_t)))
            .union(cq.Workplane("XY").sphere(t_r).translate((x_t0 + t_r + t_len, 0, z_t))))
    parts["tank"] = (tank, "tank")
    riser_top = z_v - vb / 2
    parts["pipe"] = (cyl_z(24, riser_top - (z_t + t_r - 4), -ch_len / 2, 0, z_t + t_r - 4)
                     .union(cyl_y(24, 70, -ch_len / 2, (riser_top + z_t + t_r) / 2)), "pipe")
    zg = (riser_top + z_t + t_r) / 2
    parts["gauge"] = (cq.Workplane("XY").add(gauge((-ch_len / 2, 39, zg)).val()).rotate((-ch_len / 2, 35, zg), (1, 0, 0), -90), "gauge")
    parts["relief"] = (relief_valve((x_t0 + t_r + t_len * 0.7, 0, z_t + t_r - 3)), "relief")
    parts["coupler"] = (coupler((x_t0 + 2 * t_r + t_len + 4, 0, z_t)), "coupler")
    # clamps from the barrel to the tank, the cradle, trunnions, yoke, turntable, deck
    clamps = None
    for xc in (60.0, 300.0, BARREL - 120):
        cl = cyl_x(od_b / 2 + 8, 22, xc, r_in=od_b / 2)
        if xc < x_t0 + 2 * t_r + t_len:
            cl = cl.union(cyl_x(t_r + 6, 22, xc, 0, z_t, r_in=t_r)).union(
                cq.Workplane("XY").box(22, 14, -od_b / 2 - (z_t + t_r)).translate((xc + 11, 0, (-od_b / 2 + z_t + t_r) / 2)))
        clamps = cl if clamps is None else clamps.union(cl)
    x_piv, z_piv = 180.0, z_t + t_r * 0.3
    side = 2 * t_r + 50
    plate = cq.Workplane("XY").box(300, 6, 2 * t_r + od_b + 120)
    zc = (z_t - t_r + od_b / 2) / 2
    cradle = plate.translate((x_piv + 20, side / 2, zc)).union(plate.translate((x_piv + 20, -side / 2, zc))).union(cyl_y(12, side + 40, x_piv, z_piv))
    parts["clamp"] = (clamps.union(cradle), "clamp")
    z_tt = z_t - t_r - 50
    yk = cq.Workplane("XY").box(80, 8, z_piv - z_tt + 20)
    yoke = (yk.translate((x_piv, side / 2 + 14, (z_piv + z_tt) / 2 + 10)).union(yk.translate((x_piv, -side / 2 - 14, (z_piv + z_tt) / 2 + 10)))
            .union(cq.Workplane("XY").box(90, side + 36, 10).translate((x_piv, 0, z_tt + 15))))
    parts["mount"] = (yoke.union(cyl_z(80, 12, x_piv, 0, z_tt)).union(cyl_z(55, 10, x_piv, 0, z_tt - 10)), "mount")
    parts["deck"] = (cq.Workplane("XY").box(380, 260, 8).translate((x_piv, 0, z_tt - 14)), "deck")
    return parts


def export(root=Path("output/golf_recoilless/cad")):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    asm = cq.Assembly(name="golf_recoilless")
    for name, (shape, col) in assembly().items():
        asm.add(shape, name=name, color=cq.Color(*COLORS[col]))
    asm.export(str(root / "golf_recoilless.step"))
    asm.export(str(root / "golf_recoilless.glb"), tolerance=0.3, angularTolerance=0.3)
    # a half section through the breech and the nozzle, for the drawing
    half = cq.Workplane("XY").box(4000, 2000, 4000).translate((0, 1000, 0))
    sec = cq.Assembly(name="golf_recoilless_section")
    for name, (shape, col) in assembly().items():
        if name in ("jet",):
            continue
        sec.add(shape.cut(half), name=name, color=cq.Color(*COLORS[col]))
    sec.export(str(root / "golf_recoilless_section.glb"), tolerance=0.3, angularTolerance=0.3)
    return root


if __name__ == "__main__":
    print(export())
