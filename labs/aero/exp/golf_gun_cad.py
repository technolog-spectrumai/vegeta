"""The golf-ball air cannon as a whole assembly (Dedalus / CadQuery), one per case of ``golf_gun``.

    [near sonic: 300 bar HPA bottle -> regulator -> braided hose] -> fill coupler -> tank (gauge, relief valve, drain)
      -> riser pipe + elbows -> main valve (solenoid / diaphragm pulse valve / hammer valve) -> breech (feed port, the
      cup and the ball) -> 600 mm barrel -> muzzle;  hopper -> feed tube -> breech;  clamps -> cradle -> tilt trunnions
      -> yoke -> pan turntable -> the Pekari's basket lid

Frame: x along the barrel (forward), y left, z up; the origin is the main valve's outlet face on the barrel axis.
The parts are sized from the case (tank volume and bore, wall from the rated pressure, valve orifice); fittings are
catalogue-like sizes (BSPP 1/4", 1/2", 1", 1.5"). ``export(case)`` writes a coloured STEP and a GLB.
"""
from __future__ import annotations

import math
from pathlib import Path

import cadquery as cq

import golf_gun as gg

BORE = gg.BORE * 1e3
R_BALL = gg.D_BALL * 500
BARREL = gg.BARREL * 1e3

COLORS = {  # name -> (r, g, b, a)
    "barrel": (0.72, 0.74, 0.78, 1), "breech": (0.55, 0.57, 0.62, 1), "ball": (0.97, 0.97, 0.95, 1),
    "cup": (0.95, 0.45, 0.10, 1), "valve": (0.80, 0.62, 0.20, 1), "actuator": (0.15, 0.15, 0.18, 1),
    "tank": (0.20, 0.42, 0.75, 1), "pipe": (0.78, 0.60, 0.35, 1), "gauge": (0.92, 0.92, 0.92, 1),
    "relief": (0.85, 0.15, 0.12, 1), "coupler": (0.70, 0.70, 0.20, 1), "bottle": (0.12, 0.12, 0.12, 1),
    "regulator": (0.60, 0.15, 0.15, 1), "hose": (0.10, 0.10, 0.30, 1), "hopper": (0.30, 0.65, 0.40, 0.85),
    "feed": (0.30, 0.65, 0.40, 1), "mount": (0.35, 0.35, 0.38, 1), "clamp": (0.25, 0.25, 0.28, 1),
    "deck": (0.45, 0.40, 0.30, 1),
}

PIPE_OD = {12.0: 21.3, 25.0: 33.7, 38.0: 48.3}       # orifice -> the pipe's OD (1/2", 1", 1.5" schedule 40)


def cyl_x(r, length, x0=0.0, y=0.0, z=0.0, r_in=0.0):
    w = cq.Workplane("YZ").circle(r)
    if r_in:
        w = w.circle(r_in)
    return w.extrude(length).translate((x0, y, z))


def cyl_z(r, length, x=0.0, y=0.0, z0=0.0, r_in=0.0):
    w = cq.Workplane("XY").circle(r)
    if r_in:
        w = w.circle(r_in)
    return w.extrude(length).translate((x, y, z0))


def cyl_y(r, length, x=0.0, z=0.0):
    return cq.Workplane("XZ").circle(r).extrude(length / 2, both=True).translate((x, 0, z))


def elbow(r, r_bend, centre, plane="XZ", start_deg=0.0, sweep=90.0):
    """A 90-degree pipe bend: a torus segment (solid, OD only)."""
    t = cq.Solid.makeTorus(r_bend, r, pnt=cq.Vector(0, 0, 0), dir=cq.Vector(0, 1, 0), angleDegrees1=0, angleDegrees2=360)
    w = cq.Workplane("XY").add(t)
    wedge = (cq.Workplane("XZ").moveTo(0, 0).lineTo((r_bend + 2 * r) * 2, 0).lineTo((r_bend + 2 * r) * 2, (r_bend + 2 * r) * 2)
             .lineTo(0, (r_bend + 2 * r) * 2).close().extrude(2 * r + 2, both=True))
    w = w.intersect(wedge).rotate((0, 0, 0), (0, 1, 0), -start_deg)
    return w.translate(centre)


def gauge(at, up=True):
    x, y, z = at
    stem = cyl_z(5, 30, x, y, z)
    face = cq.Workplane("XZ").circle(26).extrude(9, both=True).translate((x, y, z + 30 + 26))
    return stem.union(face)


def relief_valve(at):
    x, y, z = at
    return cyl_z(9, 22, x, y, z).union(cyl_z(13, 30, x, y, z + 22)).union(cyl_z(3, 14, x, y, z + 52))


def coupler(at, axis="x"):
    x, y, z = at
    c = cyl_x(9, 25, x, y, z).union(cyl_x(12, 18, x + 25, y, z))
    return c


def assembly(case: gg.Case):
    """{name: (shape, colour key)} of the whole cannon for ``case``, and a dict of key dimensions."""
    t, v = case.tank, case.valve
    parts = {}
    od_pipe = PIPE_OD.get(v.orifice_mm, v.orifice_mm + 10)
    # --- barrel and breech --------------------------------------------------------------------------
    breech_len = 90.0
    od_b = BORE + 6
    breech = cyl_x(od_b / 2 + 6, breech_len, 0, r_in=BORE / 2)
    port = cq.Workplane("XY").box(BORE * 1.05, BORE * 0.95, 40).translate((20 + R_BALL, 0, od_b / 2 + 4))
    breech = breech.cut(port)
    flange = cyl_x(od_b / 2 + 18, 10, breech_len, r_in=BORE / 2)
    parts["breech"] = (breech.union(flange).union(cyl_x(od_b / 2 + 18, 10, -0.0 + breech_len + 10 - 10, r_in=BORE / 2)), "breech")
    barrel = cyl_x(od_b / 2, BARREL - breech_len, breech_len + 10, r_in=BORE / 2)
    barrel = barrel.union(cyl_x(od_b / 2 + 4, 25, BARREL - 25 + 10, r_in=BORE / 2))        # muzzle crown
    parts["barrel"] = (barrel, "barrel")
    x_ball = 20 + 8 + R_BALL
    parts["ball"] = (cq.Workplane("XY").sphere(R_BALL).translate((x_ball, 0, 0)), "ball")
    cup = cyl_x(BORE / 2 - 0.1, 10, 20).cut(cq.Workplane("XY").sphere(R_BALL).translate((x_ball, 0, 0)))
    parts["cup"] = (cup, "cup")
    # --- the main valve ------------------------------------------------------------------------------
    vb = v.orifice_mm + 34
    body = cq.Workplane("XY").box(vb, vb, vb).edges("|X").fillet(4).translate((-vb / 2, 0, 0))
    body = body.cut(cyl_x(v.orifice_mm / 2, vb, -vb))
    body = body.union(cyl_x(od_b / 2 + 18, 8, -8, r_in=BORE / 2))                             # outlet flange
    parts["valve"] = (body, "valve")
    if case.name == "slow":                          # direct solenoid: a coil and its can on top
        act = cyl_z(vb * 0.42, vb * 0.9, -vb / 2, 0, vb / 2).union(cyl_z(8, 12, -vb / 2, 0, vb / 2 + vb * 0.9))
        x_in = -vb
    elif case.name == "subsonic fast":               # diaphragm pulse valve: a domed bonnet and a pilot solenoid
        bon = cyl_z(vb * 0.62, 18, -vb / 2, 0, vb / 2).union(
            cq.Workplane("XY").sphere(vb * 0.62).intersect(cq.Workplane("XY").box(3 * vb, 3 * vb, 3 * vb).translate((0, 0, 1.5 * vb)))
            .translate((-vb / 2, 0, vb / 2 + 18 - vb * 0.25)))
        act = bon.union(cyl_z(14, 40, -vb / 2 + vb * 0.4, 0, vb / 2 + 18)).union(cyl_z(4, 14, -vb / 2 + vb * 0.4, 0, vb / 2 + 58))
        x_in = -vb
    else:                                            # hammer valve: an inline piston/hammer housing behind
        act = cyl_x(vb * 0.45, 110, -vb - 110).union(cyl_x(vb * 0.30, 20, -vb - 130)).union(
            cyl_z(14, 45, -vb - 60, 0, vb * 0.45))
        x_in = -vb                                   # the plenum feeds the valve from below (side inlet)
    parts["actuator"] = (act, "actuator")
    # --- tank under the barrel, the riser into the valve -------------------------------------------
    w = gg.wall_mm(t)
    r = t.bore_mm / 2 + w
    l_cyl = max(0.0, (t.volume_l * 1e6 - 4 / 3 * math.pi * (t.bore_mm / 2) ** 3) / (math.pi * (t.bore_mm / 2) ** 2))
    z_t = -(od_b / 2 + 30 + r)                       # tank axis
    x_t0 = -vb / 2 - r * 0.6                         # rear dome's start (the riser goes into its top near the rear)
    shell = (cyl_x(r, l_cyl, x_t0 + r, 0, z_t).union(cq.Workplane("XY").sphere(r).translate((x_t0 + r, 0, z_t)))
             .union(cq.Workplane("XY").sphere(r).translate((x_t0 + r + l_cyl, 0, z_t))))
    boss = cyl_z(od_pipe / 2 + 6, 14, -vb / 2, 0, z_t + r - 6)
    parts["tank"] = (shell.union(boss).union(cyl_x(14, 16, x_t0 + 2 * r + l_cyl - 4, 0, z_t)), "tank")
    # riser: from the tank's boss straight up into the valve's underside (a side inlet: a tee under the body)
    z_top = -vb / 2
    riser = cyl_z(od_pipe / 2, (z_top) - (z_t + r + 8), -vb / 2, 0, z_t + r + 8)
    union = cyl_z(od_pipe / 2 + 5, 16, -vb / 2, 0, (z_t + r + z_top) / 2 - 8)     # a union nut
    tee = cyl_y(od_pipe / 2, od_pipe + 30, -vb / 2, (z_t + r + z_top) / 2 + 20)
    parts["pipe"] = (riser.union(union).union(tee), "pipe")
    # body inlet opening from below
    parts["valve"] = (parts["valve"][0].cut(cyl_z(v.orifice_mm / 2, vb / 2 + 2, -vb / 2, 0, -vb / 2 - 1)), "valve")
    y_tee = (od_pipe + 30) / 2
    z_tee = (z_t + r + z_top) / 2 + 20
    parts["gauge"] = (cq.Workplane("XY").add(gauge((-vb / 2, y_tee + 4, z_tee - 0)).val()).rotate((-vb / 2, y_tee, z_tee), (1, 0, 0), -90)
                      .translate((0, 0, 0)), "gauge")
    parts["relief"] = (relief_valve((x_t0 + r + l_cyl * 0.6, 0, z_t + r - 3)), "relief")
    parts["drain"] = (cyl_z(6, 18, x_t0 + r + l_cyl * 0.5, 0, z_t - r - 15), "coupler")
    x_front = x_t0 + 2 * r + l_cyl + 12
    if case.name == "near sonic":
        # the HPA bottle under the tank's front, its regulator, and the braided hose into the plenum's front port
        rb, lb = 45.0, 300.0
        zb = z_t - r - 25 - rb
        xb0 = x_front - lb + 40
        bottle = cyl_x(rb, lb - 2 * rb, xb0 + rb, 0, zb).union(cq.Workplane("XY").sphere(rb).translate((xb0 + rb, 0, zb))) \
            .union(cq.Workplane("XY").sphere(rb).translate((xb0 + lb - rb, 0, zb)))
        parts["bottle"] = (bottle.union(cyl_x(14, 25, xb0 + lb - 5, 0, zb)), "bottle")
        reg = cyl_x(22, 55, xb0 + lb + 20, 0, zb).union(cyl_z(9, 30, xb0 + lb + 50, 0, zb + 18)).union(
            gauge((xb0 + lb + 32, 0, zb + 18)))
        parts["regulator"] = (reg, "regulator")
        p0 = cq.Vector(xb0 + lb + 75, 0, zb)
        p1 = cq.Vector(x_front + 40, 0, z_t)
        mid = cq.Vector(max(p0.x, p1.x) + 70, 0, (p0.z + p1.z) / 2)
        hose = _hose(p0, mid, p1, 6.0)
        parts["hose"] = (hose, "hose")
        parts["coupler"] = (coupler((x_front - 4, 0, z_t)), "coupler")
    else:
        parts["coupler"] = (coupler((x_front - 4, 0, z_t)), "coupler")
    # --- hopper and feed into the breech port ------------------------------------------------------------
    x_port = 20 + R_BALL
    feed_len = 140.0
    a = math.radians(35)
    fx, fz = x_port - feed_len * math.sin(a), od_b / 2 + 4 + feed_len * math.cos(a)
    feed = cq.Solid.makeCylinder(BORE / 2 + 2.5, feed_len, cq.Vector(fx, 0, fz), cq.Vector(math.sin(a), 0, -math.cos(a)))
    feed_in = cq.Solid.makeCylinder(BORE / 2 + 0.5, feed_len + 2, cq.Vector(fx, 0, fz), cq.Vector(math.sin(a), 0, -math.cos(a)))
    parts["feed"] = (cq.Workplane("XY").add(feed).cut(cq.Workplane("XY").add(feed_in)), "feed")
    hop = (cq.Workplane("XY").rect(BORE + 8, BORE + 8).workplane(offset=110).rect(230, 170).loft()
           .faces(">Z").shell(-2.5).translate((fx, 0, fz - 5)))
    parts["hopper"] = (hop, "hopper")
    # --- clamps, cradle, trunnions, yoke, turntable, deck ----------------------------------------------------
    x_piv = 120.0                                     # tilt axis along y, between the barrel and the tank
    z_piv = z_t + r * 0.15
    clamps = None
    for xc in (x_piv - 60, x_piv + 140, BARREL - 140):
        cl = cyl_x(od_b / 2 + 8, 22, xc, r_in=od_b / 2).union(cq.Workplane("XY").box(22, 14, 30).translate((xc + 11, 0, -od_b / 2 - 14)))
        if xc < BARREL - 200:
            cl = cl.union(cyl_x(r + 6, 22, xc, 0, z_t, r_in=r)).union(
                cq.Workplane("XY").box(22, 14, (-od_b / 2 - 14) - (z_t + r)).translate((xc + 11, 0, ((-od_b / 2 - 14) + (z_t + r)) / 2)))
        clamps = cl if clamps is None else clamps.union(cl)
    side = 2 * r + 50
    cradle = (cq.Workplane("XY").box(260, 6, r * 2 + od_b).translate((x_piv + 40, side / 2, (z_t + od_b / 2) / 2))
              .union(cq.Workplane("XY").box(260, 6, r * 2 + od_b).translate((x_piv + 40, -side / 2, (z_t + od_b / 2) / 2))))
    cradle = cradle.union(cyl_y(12, side + 40, x_piv, z_piv))
    parts["clamp"] = (clamps.union(cradle), "clamp")
    z_tt = z_t - r - (115 if case.name == "near sonic" else 45)
    yoke_h = z_piv - z_tt
    yoke = (cq.Workplane("XY").box(80, 8, yoke_h + 20).translate((x_piv, side / 2 + 14, z_tt + yoke_h / 2 + 10))
            .union(cq.Workplane("XY").box(80, 8, yoke_h + 20).translate((x_piv, -side / 2 - 14, z_tt + yoke_h / 2 + 10)))
            .union(cq.Workplane("XY").box(90, side + 36, 10).translate((x_piv, 0, z_tt + 15))))
    parts["mount"] = (yoke.union(cyl_z(80, 12, x_piv, 0, z_tt)).union(cyl_z(55, 10, x_piv, 0, z_tt - 10)), "mount")
    parts["deck"] = (cq.Workplane("XY").box(380, 260, 8).translate((x_piv, 0, z_tt - 14)), "deck")
    dims = dict(tank_length=2 * r + l_cyl, tank_od=2 * r, wall=w, valve_body=vb, pipe_od=od_pipe,
                overall_length=BARREL + 10 + (vb + 130 if case.name == "near sonic" else vb), pivot=(x_piv, z_piv))
    return parts, dims


def _hose(p0, mid, p1, r):
    pts = [p0, mid, p1]
    path = cq.Wire.assembleEdges([cq.Edge.makeSpline([cq.Vector(*q.toTuple()) for q in pts])])
    prof = cq.Wire.makeCircle(r, p0, (mid - p0).normalized())
    return cq.Workplane("XY").add(cq.Solid.sweep(prof, [], path, makeSolid=True, isFrenet=False))


def export(case: gg.Case, root=Path("output/golf_gun/cad")):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    parts, dims = assembly(case)
    asm = cq.Assembly(name=case.name.replace(" ", "_"))
    for name, (shape, col) in parts.items():
        asm.add(shape, name=name, color=cq.Color(*COLORS[col]))
    stem = root / f"assembly_{case.name.replace(' ', '_')}"
    asm.export(str(stem.with_suffix(".step")))
    asm.export(str(stem.with_suffix(".glb")), tolerance=0.3, angularTolerance=0.3)
    return stem, dims


if __name__ == "__main__":
    for c in gg.CASES:
        c = gg.size(c)
        print(export(c))
