"""The Pekari's grappling harpoon as a whole assembly (Dedalus / CadQuery), from ``harpoon``'s design point.

    compressor-air tank (3 L, 16 bar rated) -> riser -> 1.5" diaphragm pulse valve -> 700 mm barrel, bore 72 mm
      -> sabot cup -> the hook, folded, eye forward -> the leader out of the muzzle -> the line canister (40 m of
      6 mm Dyneema, flaked) under the muzzle;  clamps -> cradle -> tilt trunnions -> yoke -> pan turntable -> the
      Pekari's basket lid.  A second hook beside the mount shows the claws open (display only).

The hook: a titanium shank (eye at the front, the crown at the back), four 17-4PH claws hinged at the crown. Folded
they lie forward along the shank inside the bore; out of the muzzle torsion springs swing their tips out to 50
degrees. The tips point toward the line, so a pull on the line drags them into the parapet.

Frame: x along the barrel, z up, origin at the valve's outlet face on the barrel axis. ``export()`` writes a coloured
STEP and a GLB.
"""
from __future__ import annotations

import math
from pathlib import Path

import cadquery as cq

import harpoon as hp
from golf_gun_cad import COLORS as GOLF_COLORS, coupler, cyl_x, cyl_y, cyl_z, gauge, relief_valve, _hose

COLORS = dict(GOLF_COLORS, shank=(0.62, 0.64, 0.70, 1), claw=(0.30, 0.32, 0.36, 1), eye=(0.80, 0.62, 0.20, 1),
              line=(0.95, 0.80, 0.10, 1), canister=(0.18, 0.30, 0.22, 1), sabot=(0.95, 0.45, 0.10, 1),
              spring=(0.75, 0.20, 0.20, 1))


def hook_parts(h: hp.Hook = hp.HOOK, open_deg=None, at=(0.0, 0.0, 0.0), direction=1):
    """The hook along +x (eye forward) with its crown at x = 0; ``open_deg`` None = folded (in the bore)."""
    od, L, w = h.shank_od_mm, h.shank_len_mm, h.shank_wall_mm
    shank = cyl_x(od / 2, L, 0, r_in=od / 2 - w).union(cyl_x(od / 2 + 4, 14, 0, r_in=6))          # the crown ring
    eye = (cq.Workplane("XZ").circle(13).circle(7).extrude(5, both=True).translate((L + 16, 0, 0))
           .union(cyl_x(9, 16, L)))
    nose = cq.Workplane("XY").sphere(od / 2).intersect(cq.Workplane("XY").box(od, od + 2, od + 2).translate((od / 2, 0, 0))).translate((L - 2, 0, 0))
    nose = nose.cut(cyl_x(9.5, 40, L - 5))
    claws, springs = None, None
    ang = 4.0 if open_deg is None else open_deg
    for k in range(h.n_claws):
        roll = 45 + 90 * k
        b, d, cl = h.claw_width_mm, h.claw_depth_mm, h.claw_len_mm
        # a claw: a bar from its pin (at the crown, at the shank's surface) forward, a hooked-back tip with a point
        bar = cq.Workplane("XY").box(cl, b, d).translate((cl / 2, 0, 0))
        tip = (cq.Workplane("XZ").moveTo(cl - 4, -d / 2).lineTo(cl + 12, -d / 2 - 2).lineTo(cl + 4, d / 2 + 14)
               .lineTo(cl - 10, d / 2).close().extrude(b / 2, both=True))
        boss = cyl_y(d / 2 + 2, b + 4, 0, 0)
        claw = bar.union(tip).union(boss)
        claw = claw.rotate((0, 0, 0), (0, 1, 0), -ang).translate((14, 0, od / 2 + d / 2 + 1))
        spring = cq.Workplane("XZ").circle(d / 2 + 3).circle(d / 2 + 1).extrude(b / 2 + 4, both=True).translate((14, 0, od / 2 + d / 2 + 1))
        claw = claw.rotate((0, 0, 0), (1, 0, 0), roll)
        spring = spring.rotate((0, 0, 0), (1, 0, 0), roll)
        claws = claw if claws is None else claws.union(claw)
        springs = spring if springs is None else springs.union(spring)
    parts = {"shank": shank.union(nose), "eye": eye, "claw": claws, "spring": springs}
    out = {}
    for k, s in parts.items():
        if direction < 0:
            s = s.rotate((0, 0, 0), (0, 0, 1), 180)
        out[k] = s.translate(at)
    return out


def assembly():
    g, h = hp.GUN, hp.HOOK
    bore = g.bore_mm
    barrel_len = g.barrel_m * 1e3
    od_b = bore + 6
    parts = {}
    # barrel, breech, muzzle crown
    parts["barrel"] = (cyl_x(od_b / 2, barrel_len, 0, r_in=bore / 2)
                       .union(cyl_x(od_b / 2 + 4, 30, barrel_len - 30, r_in=bore / 2))
                       .union(cyl_x(od_b / 2 + 16, 10, 0, r_in=bore / 2)), "barrel")
    # sabot and the folded hook (crown at the back, on the sabot)
    x_crown = 40.0
    parts["sabot"] = (cyl_x(bore / 2 - 0.2, 12, x_crown - 12).cut(cyl_x(8, 14, x_crown - 13)), "sabot")
    for k, s in hook_parts(at=(x_crown, 0, 0)).items():
        parts["hook_" + k] = (s, k)
    # the leader from the eye out of the muzzle, down to the canister under it
    x_eye = x_crown + h.shank_len_mm + 22
    can_r, can_l = 62.0, 210.0
    x_can = barrel_len - can_l - 40
    z_can = -(od_b / 2 + 12 + can_r)
    leader = cyl_x(3, barrel_len - x_eye + 30, x_eye)
    p0 = cq.Vector(barrel_len + 28, 0, 0)
    drop = _hose(p0, cq.Vector(barrel_len + 70, 0, z_can / 2), cq.Vector(x_can + can_l + 6, 0, z_can), 3.0)
    parts["line"] = (leader.union(drop), "line")
    can = cyl_x(can_r, can_l, x_can, 0, z_can, r_in=can_r - 2).union(cyl_x(can_r, 3, x_can, 0, z_can))
    flake = None
    for i in range(7):                                   # the flaked line seen through the open front
        ring = cq.Workplane("YZ").circle(can_r - 6 - 6 * i).circle(can_r - 9 - 6 * i).extrude(can_l - 20).translate((x_can + 10, 0, z_can))
        flake = ring if flake is None else flake.union(ring)
    parts["canister"] = (can, "canister")
    parts["flake"] = (flake, "line")
    # the valve (diaphragm pulse valve) behind the breech, the tank under the barrel's rear, riser, gauge, relief
    vb = g.valve_orifice_mm + 34
    body = cq.Workplane("XY").box(vb, vb, vb).edges("|X").fillet(4).translate((-vb / 2, 0, 0)).cut(cyl_x(g.valve_orifice_mm / 2, vb, -vb))
    body = body.cut(cyl_z(g.valve_orifice_mm / 2, vb / 2 + 2, -vb / 2, 0, -vb / 2 - 1))
    parts["valve"] = (body.union(cyl_x(od_b / 2 + 16, 8, -8, r_in=bore / 2)), "valve")
    bon = cyl_z(vb * 0.62, 18, -vb / 2, 0, vb / 2)
    dome = cq.Workplane("XY").sphere(vb * 0.62).intersect(cq.Workplane("XY").box(3 * vb, 3 * vb, 3 * vb).translate((0, 0, 1.5 * vb)))
    act = bon.union(dome.translate((-vb / 2, 0, vb / 2 + 18 - vb * 0.25))).union(cyl_z(14, 40, -vb / 2 + vb * 0.4, 0, vb / 2 + 18))
    parts["actuator"] = (act, "actuator")
    t_bore, w = 110.0, 1.5
    r = t_bore / 2 + w
    l_cyl = (g.tank_l * 1e6 - 4 / 3 * math.pi * (t_bore / 2) ** 3) / (math.pi * (t_bore / 2) ** 2)
    z_t = -(od_b / 2 + 30 + r)
    x_t0 = -vb / 2 - r * 0.6
    shell = (cyl_x(r, l_cyl, x_t0 + r, 0, z_t).union(cq.Workplane("XY").sphere(r).translate((x_t0 + r, 0, z_t)))
             .union(cq.Workplane("XY").sphere(r).translate((x_t0 + r + l_cyl, 0, z_t))))
    parts["tank"] = (shell.union(cyl_z(30, 14, -vb / 2, 0, z_t + r - 6)), "tank")
    od_pipe = 48.3
    z_top = -vb / 2
    riser = cyl_z(od_pipe / 2, z_top - (z_t + r + 8), -vb / 2, 0, z_t + r + 8)
    z_tee = (z_t + r + z_top) / 2 + 6
    parts["pipe"] = (riser.union(cyl_y(od_pipe / 2, od_pipe + 30, -vb / 2, z_tee)), "pipe")
    y_tee = (od_pipe + 30) / 2
    parts["gauge"] = (cq.Workplane("XY").add(gauge((-vb / 2, y_tee + 4, z_tee)).val()).rotate((-vb / 2, y_tee, z_tee), (1, 0, 0), -90), "gauge")
    parts["relief"] = (relief_valve((x_t0 + r + l_cyl * 0.6, 0, z_t + r - 3)), "relief")
    parts["coupler"] = (coupler((x_t0 + 2 * r + l_cyl + 8, 0, z_t)), "coupler")
    # clamps and cradle, trunnions, yoke, turntable, deck
    x_piv, z_piv = 200.0, z_t + r * 0.2
    clamps = None
    for xc, both in ((60.0, True), (330.0, True), (barrel_len - 300, False)):
        cl = cyl_x(od_b / 2 + 8, 22, xc, r_in=od_b / 2)
        if both and xc < x_t0 + 2 * r + l_cyl:
            cl = cl.union(cyl_x(r + 6, 22, xc, 0, z_t, r_in=r)).union(
                cq.Workplane("XY").box(22, 14, (-od_b / 2) - (z_t + r)).translate((xc + 11, 0, ((-od_b / 2) + (z_t + r)) / 2)))
        clamps = cl if clamps is None else clamps.union(cl)
    cl = cyl_x(can_r + 6, 22, x_can + 60, 0, z_can, r_in=can_r).union(
        cq.Workplane("XY").box(22, 14, (-od_b / 2) - (z_can + can_r)).translate((x_can + 71, 0, ((-od_b / 2) + (z_can + can_r)) / 2)))
    clamps = clamps.union(cl).union(cyl_x(od_b / 2 + 8, 22, x_can + 60, r_in=od_b / 2))
    side = 2 * r + 50
    plate = cq.Workplane("XY").box(320, 6, 2 * r + od_b)
    cradle = (plate.translate((x_piv + 30, side / 2, (z_t + od_b / 2) / 2)).union(plate.translate((x_piv + 30, -side / 2, (z_t + od_b / 2) / 2)))
              .union(cyl_y(12, side + 40, x_piv, z_piv)))
    parts["clamp"] = (clamps.union(cradle), "clamp")
    z_tt = z_t - r - 45
    yoke_h = z_piv - z_tt
    yk = cq.Workplane("XY").box(80, 8, yoke_h + 20)
    yoke = (yk.translate((x_piv, side / 2 + 14, z_tt + yoke_h / 2 + 10)).union(yk.translate((x_piv, -side / 2 - 14, z_tt + yoke_h / 2 + 10)))
            .union(cq.Workplane("XY").box(90, side + 36, 10).translate((x_piv, 0, z_tt + 15))))
    parts["mount"] = (yoke.union(cyl_z(80, 12, x_piv, 0, z_tt)).union(cyl_z(55, 10, x_piv, 0, z_tt - 10)), "mount")
    parts["deck"] = (cq.Workplane("XY").box(380, 260, 8).translate((x_piv, 0, z_tt - 14)), "deck")
    # the display hook, claws open, standing on the deck beside the mount
    for k, s in hook_parts(open_deg=h.claw_open_deg, at=(0, 0, 0)).items():
        s = s.rotate((0, 0, 0), (0, 1, 0), -90).translate((x_piv - 120, 190, z_tt - 10 + 20))
        parts["open_" + k] = (s, k)
    return parts


def export(root=Path("output/harpoon/cad")):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    asm = cq.Assembly(name="pekari_harpoon")
    for name, (shape, col) in assembly().items():
        asm.add(shape, name=name, color=cq.Color(*COLORS[col]))
    asm.export(str(root / "harpoon_assembly.step"))
    asm.export(str(root / "harpoon_assembly.glb"), tolerance=0.3, angularTolerance=0.3)
    # the hook alone, open and folded, for the drawings and FEA
    for state, ang in (("open", hp.HOOK.claw_open_deg), ("folded", None)):
        a = cq.Assembly(name=f"hook_{state}")
        for k, s in hook_parts(open_deg=ang).items():
            a.add(s, name=k, color=cq.Color(*COLORS[k]))
        a.export(str(root / f"hook_{state}.step"))
    return root


if __name__ == "__main__":
    print(export())
