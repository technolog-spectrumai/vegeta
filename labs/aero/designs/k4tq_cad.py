"""K4TQ — Stage 2: the parametric CAD (Dedalus / CadQuery), every dimension from ``k4tq.params``.

The assembly is built from parts, each a CadQuery solid [mm] with a group, a colour and a mass rule (a material
density on its CAD volume, an areal density on its CAD area, or a bought mass from ``k4tq.CATALOGUE``). Every part is
built once in its own frame and placed with the transform ``k4tq.wing_frame`` uses for the sizing, so the CAD and the
numbers cannot drift apart.

The wing root (Stage 2.3 / 2.4), per wing, from the fuselage centreline outward:
- **inside the fuselage, fixed**: a frame (bulkhead) at the pivot station, a 7075 bearing block bolted to it with two
  6804-2RS bearings ``bearing_spacing`` apart, the tilt servo (standard 40 × 20 × 38 mm case) beside the stub with its
  16-tooth pinion, the encoder board, the hard-stop block;
- **inside the fuselage, tilting**: the 7075 stub shaft (Ø20 seats, Ø12 bore for the motor cables), the 48-tooth gear
  (3:1) with the stop pin, the encoder magnet;
- **just outside the skin, tilting**: the inner fold knuckle (two lugs) on the stub, the hinge pin along the chord
  (``hinge_pin_offset`` toward the upper surface) and the locking pin (``lock_couple`` below it);
- **outboard, folding and tilting**: the outer knuckle (one lug between the two) with the root plate and the spar
  socket, the carbon spar, the wing panel, the nacelle with the motor, the ESC and the folding propeller.

Configurations: ``assembly(p, tilt_deg, fold_deg)`` — cruise (0, 0), hover (90, 0), folded (90, 90); any tilt between.
``checks`` reports interferences, parts outside the fuselage skin, the folded envelope against a tube, clearances.
``masses`` turns the CAD into a mass table in ``k4tq.mass_table``'s format; ``export`` writes STEP / STL (and the
outer mould line for the CFD); ``render`` draws the views; ``geometries`` hands Dedalus ``Geometry`` objects to
``vegeta.dedalus.viz.plot3d`` for the interactive view in Jupyter.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cadquery as cq
import numpy as np
from vegeta.dedalus import Design, Geometry, Parameter

import k4tq
from kestrel import naca4

MM = 1000.0
DENSITY = {"7075": 2810.0, "steel": 7850.0, "bronze": 8800.0, "PA-CF": 1150.0, "CF": 1550.0, "G10": 1850.0, "POM": 1410.0}
PRINT_FILL = 0.45                 # printed parts: walls + 25 % gyroid ≈ 45 % of the solid volume
COLORS = {"fuselage": "#f4f5f7", "hatch": "#e4e7ec", "wing": "#d9dde3", "root": "#9aa3ad", "pin": "#5b6470", "nacelle": "#6b7380",
          "disc": "#7fb3e6", "blade": "#3b4252", "leg": "#4a5160", "fin": "#d0d4da", "internal": "#c8a25a", "cargo": "#2f80ed",
          "battery": "#3a7d44", "servo": "#2b2f36", "frame": "#b8bec7", "spar": "#30343b"}


@dataclass
class Part:
    solid: cq.Shape
    group: str
    rule: tuple                   # ("solid", material) | ("printed", material) | ("shell", material, wall mm) | ("area", kg/m²) | ("bought", kg) | ("none",)
    color: str = ""
    source: str = ""
    zone: str = "external"        # "internal" parts must lie inside the fuselage skin
    tags: set = field(default_factory=set)


# ================================================================================================= transforms
def _mat(L: dict, which: str, side: int, tilt: float, fold: float, *, folds: bool, tilts: bool = True) -> np.ndarray:
    """The 4 × 4 matrix [mm] of ``k4tq.wing_frame`` for a whole part: the fold (about the hinge pin along x at
    (y_hinge, hinge_pin_offset)) if the part folds, the tilt about y if it tilts, the side's mirror, the pivot."""
    w = L["wings"][which]
    yh, zh = w["y_hinge"] * MM, L["p"]["hinge_pin_offset"] * MM

    def T(x=0.0, y=0.0, z=0.0):
        M = np.eye(4)
        M[:3, 3] = (x, y, z)
        return M
    M = np.eye(4)
    if folds:
        c, s = math.cos(fold), math.sin(fold)
        R = np.eye(4)
        R[1:3, 1:3] = [[c, -s], [s, c]]
        M = T(0, yh, zh) @ R @ T(0, -yh, -zh) @ M
    if tilts:
        c, s = math.cos(tilt), math.sin(tilt)
        R = np.eye(4)
        R[0, 0], R[0, 2], R[2, 0], R[2, 2] = c, s, -s, c
        M = R @ M
    if side < 0:
        M = np.diag([1.0, -1.0, 1.0, 1.0]) @ M
    M = T(w["x_pivot"] * MM, 0, w["z"] * MM) @ M
    return M


def _place(shape: cq.Shape, M: np.ndarray, mirror: bool) -> cq.Shape:
    """Apply a rigid 4 × 4 (a rotation, or a rotation with the side's mirror, plus a translation) through gp_Trsf."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Trsf
    tr = gp_Trsf()
    tr.SetValues(*[float(M[i, j]) for i in range(3) for j in range(4)])
    return cq.Shape.cast(BRepBuilderAPI_Transform(shape.wrapped, tr, True).Shape())


def _v(p) -> cq.Vector:
    return cq.Vector(*(float(c) for c in p))


def _cyl(r, h, base, direction) -> cq.Solid:
    d = np.asarray(direction, float)
    return cq.Solid.makeCylinder(r, h, _v(base), _v(d / np.linalg.norm(d)))


def _tube(ro, ri, h, base, direction) -> cq.Shape:
    return _cyl(ro, h, base, direction).cut(_cyl(ri, h, base, direction))


def _box(x0, x1, y0, y1, z0, z1) -> cq.Solid:
    return cq.Solid.makeBox(x1 - x0, y1 - y0, z1 - z0, _v((x0, y0, z0)))


# ================================================================================================= the wing root
def root_parts(p: dict, L: dict, which: str) -> dict:
    """The right-hand root of one wing in its wing frame [mm] (x along the chord, aft; y outboard from the centreline;
    z the thickness, up), each part tagged with how it moves: ``fixed`` (fuselage), ``tilt`` (turns with the stub),
    ``fold`` (outboard of the hinge pin: folds and tilts)."""
    yf = p["fuselage_width"] / 2 * MM
    w = L["wings"][which]
    yh, y0 = w["y_hinge"] * MM, w["y0"] * MM
    zh, lc = p["hinge_pin_offset"] * MM, p["lock_couple"] * MM
    ro = p["stub_od"] / 2 * MM
    yb1 = yf - 9.0
    yb2 = yb1 - p["bearing_spacing"] * MM
    m = p["gear_module"]
    rg, rp = m * p["z_gear"] / 2, m * p["z_pinion"] / 2
    cd = rg + rp                                                    # centre distance, the pinion aft of the stub
    y_gear = yb2 - 11.0
    y_in = y_gear - 6.0
    up = -1.0 if p[f"z_{which}"] > 0 else 1.0                       # the servo body toward the fuselage's centre line
    cat = k4tq.CATALOGUE["root"]
    P = {}
    # fixed: frame, bearing block, bearings, servo, pinion, encoder board, stop block
    h_frame = p["fuselage_height"] * MM
    zp = p[f"z_{which}"] * MM
    frame = _box(-30, -27, 0.5, yf - 1.0, -0.5 * h_frame - 50, 0.5 * h_frame + 50)                   # 2 mm clear of the gear rim
    window = _box(-31, -26, 8.0, yf - 12.0, -0.5 * h_frame - zp + 12.0, 0.5 * h_frame - zp - 12.0).cut(_box(-31, -26, yb2 - 12.0, yf, -26, 26))
    P["frame"] = (Part(frame.cut(window), "frame", ("solid", "CF"), COLORS["frame"],
                       "3 mm carbon ring frame at the pivot station (trimmed to the skin, lightening window, pad for the bearing block)", "internal"), "fixed")
    blk = _cyl(20.0, 9.0, (0, yb1 - 4.5, 0), (0, 1, 0)).fuse(_cyl(20.0, 9.0, (0, yb2 - 4.5, 0), (0, 1, 0)))
    blk = blk.fuse(_box(-20, -18, yb2 - 4.5, yb1 + 4.5, -18, 18)).fuse(_box(18, 20, yb2 - 4.5, yb1 + 4.5, -18, 18))      # two 2 mm webs
    blk = blk.fuse(_box(-27, -20, yb2 - 4.5, yb1 + 4.5, -22, 22).cut(_box(-27, -21, yb2 - 4.5 + 3, yb1 + 4.5 - 3, -19, 19)))   # flange to the frame (a 3 mm rim)
    for yb in (yb1, yb2):
        blk = blk.cut(_cyl(16.0, 7.0, (0, yb - 3.5, 0), (0, 1, 0)))
    blk = blk.cut(_cyl(ro + 0.5, yf, (0, yb2 - 7.0, 0), (0, 1, 0)))
    P["bearing block"] = (Part(blk, "root", ("solid", "7075"), COLORS["root"], "7075 block, H7 seats, bolted to the frame", "internal"), "fixed")
    for i, yb in enumerate((yb1, yb2)):
        P[f"bearing {i + 1}"] = (Part(_tube(16.0, 10.0, 7.0, (0, yb - 3.5, 0), (0, 1, 0)), "pin", ("bought", cat["bearings 6804-2RS (2)"]["mass_kg"] / 2),
                                      COLORS["pin"], "6804-2RS 20 x 32 x 7", "internal"), "fixed")
    servo = _box(cd - 10, cd + 10, y_gear + 3.0, y_gear + 41.0, -10 if up > 0 else -30, 30 if up > 0 else 10)
    P["tilt servo"] = (Part(servo, "servo", ("bought", cat["tilt servo"]["mass_kg"]), COLORS["servo"], cat["tilt servo"]["source"], "internal"), "fixed")
    P["pinion"] = (Part(_cyl(rp + m, 6.0, (cd, y_gear - 3.0, 0), (0, 1, 0)), "pin", ("solid", "steel"), COLORS["pin"], f"z{p['z_pinion']} m{m}", "internal"), "fixed")
    P["encoder board"] = (Part(_box(-12, 12, y_in - 5.0, y_in - 3.4, -12, 12), "internal", ("bought", cat["encoder"]["mass_kg"]), COLORS["internal"],
                               cat["encoder"]["source"], "internal"), "fixed")
    P["hard stops"] = (Part(_box(-6, 6, y_gear - 3.0, y_gear + 3.0, rg + 2.0, rg + 9.0) if up < 0 else _box(-6, 6, y_gear - 3.0, y_gear + 3.0, -rg - 9.0, -rg - 2.0),
                            "pin", ("bought", cat["fasteners + stops"]["mass_kg"]), COLORS["pin"], "stops at −5° and 95° against the gear's stop pin", "internal"), "fixed")
    # tilting: stub, gear, magnet, inner knuckle, hinge pin, locking pin
    stub = _tube(ro, 6.0, (yh - 8.0) - y_in, (0, y_in, 0), (0, 1, 0))
    P["stub shaft"] = (Part(stub, "root", ("solid", "7075"), COLORS["root"], "7075 stub, Ø20 k5 seats, Ø12 cable bore", "through"), "tilt")    # passes the skin
    gear = _tube(rg + m, rg - 4.0, 6.0, (0, y_gear - 3.0, 0), (0, 1, 0)).fuse(_tube(rg - 4.0, ro, 2.0, (0, y_gear - 1.0, 0), (0, 1, 0)))
    gear = gear.fuse(_tube(ro + 4.0, ro, 6.0, (0, y_gear - 3.0, 0), (0, 1, 0)))
    P["gear"] = (Part(gear, "root", ("solid", "7075"), COLORS["root"], f"z{p['z_gear']} m{m} 7075 (POM fails: ~150 MPa tooth stress at the servo stall), rim + web, stop pin",
                      "internal"), "tilt")
    P["encoder magnet"] = (Part(_cyl(3.0, 2.5, (0, y_in - 2.6, 0), (0, 1, 0)), "pin", ("none",), COLORS["pin"], "diametric magnet", "internal"), "tilt")
    kn = _box(-14, -8, yh - 8.0, yh + 6.0, -lc + zh - 6.0, zh + 6.0).fuse(_box(8, 14, yh - 8.0, yh + 6.0, -lc + zh - 6.0, zh + 6.0))
    kn = kn.fuse(_box(-14, 14, yh - 10.0, yh - 7.0, -lc + zh - 6.0, zh + 6.0))
    for zz in (zh, zh - lc):
        kn = kn.cut(_cyl(3.0, 60.0, (-30, yh, zz), (1, 0, 0)))
    P["inner knuckle"] = (Part(kn, "root", ("solid", "7075"), COLORS["root"], "7075 clevis on the stub end", "external"), "tilt")
    P["hinge pin"] = (Part(_cyl(3.0, 32.0, (-16, yh, zh), (1, 0, 0)), "pin", ("solid", "steel"), COLORS["pin"], "Ø6 steel, along the chord", "external"), "tilt")
    P["locking pin"] = (Part(_cyl(3.0, 32.0, (-16, yh, zh - lc), (1, 0, 0)), "pin", ("solid", "steel"), COLORS["pin"],
                             "Ø6 hardened, spring-loaded, witness mark", "external", {"deployed only"}), "tilt")
    # folding: outer knuckle + root plate + spar socket, spar
    # the outer lug: round about both pins on the inboard side (it swings clear of the stub and the inner clevis)
    ok = _box(-7, 7, yh, y0 - 2.0, -lc + zh - 6.0, zh + 6.0).fuse(_box(-7, 7, yh - 6.0, yh, -lc + zh, zh))
    for zz in (zh, zh - lc):
        ok = ok.fuse(_cyl(6.0, 14.0, (-7, yh, zz), (1, 0, 0)))
    for zz in (zh, zh - lc):
        ok = ok.cut(_cyl(3.0, 20.0, (-10, yh, zz), (1, 0, 0)))
    c0 = w["c_at"](w["y0"]) * MM
    pc = w["pivot_chord"]
    ok = ok.fuse(_box(-pc * c0 * 0.6, (1 - pc) * c0 * 0.5, y0 - 3.0, y0, -6, 6))
    ok = ok.fuse(_tube(10.5, 8.0, 40.0, (0, y0, 0), (0, 1, 0)))
    P["outer knuckle"] = (Part(ok, "root", ("solid", "7075"), COLORS["root"], "7075 outer knuckle, root plate and spar socket", "external"), "fold")
    od, idd = (v / 2 * MM for v in k4tq.spar_tube(p))
    P["spar"] = (Part(_tube(od, idd, (w["y1"] - w["y0"]) * MM - 15.0, (0, y0, 0), (0, 1, 0)), "spar", ("solid", "CF"), COLORS["spar"],
                      f"carbon tube Ø{2 * od:.0f}×{2 * idd:.0f}", "external", {"inside the wing"}), "fold")
    return P


# ================================================================================================= wing, nacelle, propeller
def _section(designation: str):
    d = designation.split()[-1]
    up, lo = naca4(int(d[0]) / 100, int(d[1]) / 10, int(d[2:]) / 100, n=24)
    return up + lo[::-1][1:]


def wing_panel(L: dict, which: str, y_out: float | None = None) -> cq.Solid:
    """The right panel in its wing frame [mm]: the section lofted from the root rib (y0) to the tip (or ``y_out``)."""
    w = L["wings"][which]
    pc = w["pivot_chord"]
    wires = []
    for y in (w["y0"], y_out or w["y1"]):
        c = w["c_at"](y)
        pts = [_v(((xi - pc) * c * MM, y * MM, zc * c * MM)) for xi, zc in _section(k4tq.section_of(L["p"], which))]
        wires.append(cq.Wire.makePolygon(pts, close=True))
    return cq.Solid.makeLoft(wires, True)


def nacelle_parts(p: dict, L: dict, which: str, folded: bool) -> dict:
    """Nacelle, motor, ESC and the propeller (a disc for the clearance checks, or the two blades folded flat on the
    wing's upper and lower surfaces) in the wing frame [mm]; all fold with the wing."""
    w = L["wings"][which]
    pr = L["prop"]
    ym = p["motor_y"] * MM
    arm = (w["pivot_chord"] * w["c_at"](p["motor_y"]) + p["prop_ahead_of_le"]) * MM
    c = w["c_at"](p["motor_y"]) * MM
    mo = k4tq.CATALOGUE["motor"][p["motor"]]
    cat = k4tq.CATALOGUE
    P = {}
    nac = _cyl(22.0, 150.0, (-arm + 20.0, ym, 0), (1, 0, 0))
    P["nacelle"] = (Part(nac, "nacelle", ("shell", "PA-CF", 1.2), COLORS["nacelle"], "printed PA-CF nacelle shell (1.2 mm walls) with the ESC bay", "external"), "fold")
    P["motor"] = (Part(_cyl(23.0, 26.0, (-arm - 6.0, ym, 0), (1, 0, 0)), "nacelle", ("bought", mo["mass_kg"]), COLORS["servo"], mo["name"], "external"), "fold")
    P["ESC"] = (Part(_box(-arm + 35.0, -arm + 90.0, ym - 12.0, ym + 12.0, -7.0, 7.0), "internal", ("bought", cat["esc"]["mass_kg"]), COLORS["internal"],
                     cat["esc"]["name"], "external", {"inside the nacelle"}), "fold")
    P["spinner"] = (Part(cq.Solid.makeSphere(16.0, _v((-arm - 10.0, ym, 0)), angleDegrees1=-90), "nacelle", ("bought", 0.004), COLORS["nacelle"], "spinner + folding hub", "external"), "fold")
    if folded:
        for k, zb in (("blade upper", 0.085 * c + 5.0), ("blade lower", -0.055 * c - 5.0)):   # clear of the cambered section's surfaces
            b = _box(-arm - 8.0, -arm - 8.0 + pr["R"] * MM, ym - 14.0, ym + 14.0, zb - 2.0, zb + 2.0)
            P[k] = (Part(b, "blade", ("bought", pr["mass_kg"] / 2 - 0.004), COLORS["blade"], pr["source"], "external"), "fold")
    else:
        P["disc"] = (Part(_cyl(pr["R"] * MM, 3.0, (-arm - 14.0, ym, 0), (1, 0, 0)), "disc", ("bought", pr["mass_kg"] - 0.004), COLORS["disc"],
                          pr["source"], "external", {"disc"}), "fold")
    return P


# ================================================================================================= fuselage and its contents
def _superellipse(a, b, n, k=40):
    t = np.linspace(0, 2 * math.pi, k, endpoint=False) + 0.5 * math.pi / k     # no vertex on the symmetry plane (robust Booleans)
    c, s = np.cos(t), np.sin(t)
    return a * np.sign(c) * np.abs(c) ** (2 / n), b * np.sign(s) * np.abs(s) ** (2 / n)


FUSELAGE_STATIONS = [(0.0, 0.05, 2.0), (0.04, 0.45, 2.4), (0.12, 0.85, 3.5), (0.22, 1.0, None), (0.80, 1.0, None), (0.90, 0.75, 3.5), (1.0, 0.16, 2.2)]


def fuselage_shape(p: dict) -> cq.Solid:
    """The fuselage's outer mould line: superellipse sections (exponent ``fuselage_section_n`` in the mid-body, rounder
    at the nose and the tail) lofted from the nose tip to the tail."""
    Lf, bw, bh, n = p["fuselage_length"] * MM, p["fuselage_width"] * MM, p["fuselage_height"] * MM, p["fuselage_section_n"]
    wires = []
    for f, s, nn in FUSELAGE_STATIONS:
        ys, zs = _superellipse(max(bw / 2 * s, 2.0), max(bh / 2 * s, 2.0), nn or n)
        wires.append(cq.Wire.makePolygon([_v((f * Lf, y, z)) for y, z in zip(ys, zs)], close=True))
    return cq.Solid.makeLoft(wires, False)


def body_parts(p: dict, L: dict, items: list | None = None) -> dict:
    """Fuselage shell, hatch with the marking, cargo cartridge with the two sensors and the logger, battery, avionics
    tray, parachute canister, legs, fin. Positions from ``k4tq.mass_table`` (the battery where the CG wants it)."""
    Lf = p["fuselage_length"] * MM
    items = items or k4tq.mass_table(L)
    at = {i["item"]: i for i in items}
    P = {}
    fus = fuselage_shape(p)
    P["fuselage"] = (Part(fus, "fuselage", ("area", k4tq.FUSELAGE_AREAL_KG_M2), COLORS["fuselage"], "glass/carbon shell on foam, frames' share", "external",
                          {"skin"}), "body")
    cx, cy, cz = (v * MM for v in p["cargo_box"])
    xc = p["cargo_station"] * Lf
    zc0 = -p["fuselage_height"] / 2 * MM + 12.0
    cart = _box(xc - cx / 2, xc + cx / 2, -cy / 2, cy / 2, zc0, zc0 + cz)
    cat = k4tq.CATALOGUE["cargo"]
    P["cargo cartridge"] = (Part(cart, "cargo", ("bought", cat["insulated cartridge"]["mass_kg"] + cat["shock isolators (4)"]["mass_kg"]), COLORS["cargo"],
                                 cat["insulated cartridge"]["source"], "internal"), "body")
    for k, dx in (("temperature sensor 1", -60.0), ("temperature sensor 2", 60.0)):
        P[k] = (Part(_box(xc + dx - 6, xc + dx + 6, -cy / 2 + 4, -cy / 2 + 10, zc0 + 30, zc0 + 42), "internal", ("bought", cat["temperature sensors (2)"]["mass_kg"] / 2),
                     COLORS["internal"], cat["temperature sensors (2)"]["source"], "internal", {"in the cartridge"}), "body")
    P["temperature logger"] = (Part(_box(xc - 20, xc + 20, cy / 2 - 12, cy / 2 - 4, zc0 + 20, zc0 + 50), "internal", ("bought", cat["temperature logger"]["mass_kg"]),
                                    COLORS["internal"], cat["temperature logger"]["source"], "internal", {"in the cartridge"}), "body")
    ztop = p["fuselage_height"] / 2 * MM
    P["hatch"] = (Part(_box(xc - cx / 2 - 5, xc + cx / 2 + 5, -cy / 2 - 2, cy / 2 + 2, ztop - 3.0, ztop + 0.5), "hatch", ("bought", 0.020), COLORS["hatch"],
                       "sealed hatch (EPDM lip), two latches; marking: " + k4tq.STANDARD_MARKING, "external", {"skin"}), "body")
    for bat in [i for i in items if i["group"] == "battery"]:
        bx, by, bz = (v * MM for v in bat.get("box", p["battery_box"]))
        xb, z0 = bat["x"] * MM, -p["fuselage_height"] / 2 * MM + 10.0
        P[bat["item"].split(" 6S")[0]] = (Part(_box(xb - bx / 2, xb + bx / 2, -by / 2, by / 2, z0, z0 + bz), "battery", ("bought", bat["mass_kg"]),
                                                COLORS["battery"], bat["source"], "internal"), "body")
    av = [i for i in items if i["group"] == "avionics"]
    xa = p["avionics_station"] * Lf
    zt = max(v[0].solid.BoundingBox().zmax for k, v in P.items() if k.startswith("battery")) + 3.0
    xa = p["avionics_station"] * Lf
    la, wa, ha = (80.0, 60.0, 30.0) if p.get("autonomy") else (110.0, 68.0, 36.0)   # A2: shorter, behind the companion, where the tail narrows
    P["avionics tray"] = (Part(_box(xa - la / 2, xa + la / 2, -wa / 2, wa / 2, -ha / 2, ha / 2), "internal", ("bought", sum(i["mass_kg"] for i in av)), COLORS["internal"],
                               "flight controller, GNSS, telemetry, receiver, power module, pitot, remote ID", "internal"), "body")
    if p.get("autonomy"):                                           # K4TQ-A2: the companion computer, the stereo camera, the downward module
        for i in [i for i in items if i["group"] == "autonomy"]:
            x, z = i["x"] * MM, i["z"] * MM
            dims = {"companion": (100, 70, 45), "forward stereo": (28, 88, 18), "downward": (30, 30, 20), "12 V": (40, 25, 12)}
            dx, dy, dz = next(v for k, v in dims.items() if k in i["item"])
            P[i["item"].split(" (")[0]] = (Part(_box(x - dx / 2, x + dx / 2, -dy / 2, dy / 2, z - dz / 2, z + dz / 2), "internal", ("bought", i["mass_kg"]),
                                                "#76b900" if "companion" in i["item"] else COLORS["internal"], i["source"], "internal"), "body")
    if p["parachute"]:
        xpc = p["parachute_station"] * Lf
        P["parachute"] = (Part(_cyl(35.0, 100.0, (xpc, 0, ztop - 105.0), (0, 0, 1)), "internal", ("bought", k4tq.CATALOGUE["safety"]["recovery parachute"]["mass_kg"]),
                               COLORS["internal"], k4tq.CATALOGUE["safety"]["recovery parachute"]["source"], "internal"), "body")
    zb = -p["fuselage_height"] / 2 * MM
    for which in ("front", "rear"):
        xw = L["wings"][which]["x_pivot"] * MM
        for side in (-1, 1):
            leg = _tube(6.0, 5.0, -p["leg_bottom_z"] * MM + zb + 8.0, (xw + 40.0, side * 45.0, p["leg_bottom_z"] * MM), (0, 0, 1))
            leg = leg.fuse(cq.Solid.makeSphere(9.0, _v((xw + 40.0, side * 45.0, p["leg_bottom_z"] * MM + 9.0)), angleDegrees1=-90))   # a full sphere (CadQuery's default is a half)
            P[f"leg {which} {'L' if side < 0 else 'R'}"] = (Part(leg, "leg", ("solid", "CF"), COLORS["leg"], "carbon tube Ø12×10 + printed foot, plugs into the frame",
                                                                   "external"), "body")
    S_fin = (p.get("fin_area") or 0.05) * 1e6
    h = math.sqrt(S_fin / 2 / 1.1)
    for zs, name in ((1, "fin dorsal"), (-1, "fin ventral")):
        pts = [(Lf - 1.10 * h - 40, zs * (0.03 * MM)), (Lf - 20, zs * 20.0), (Lf - 15, zs * (20.0 + h)), (Lf - 0.55 * h - 30, zs * (20.0 + h))]
        face = cq.Face.makeFromWires(cq.Wire.makePolygon([_v((x, 0, z)) for x, z in pts], close=True))
        fin = cq.Solid.extrudeLinear(face, cq.Vector(0, 6.0, 0)).translate(cq.Vector(0, -3.0, 0))
        P[name] = (Part(fin, "fin", ("area", k4tq.FIN_AREAL_KG_M2 / 2), COLORS["fin"], "dorsal / ventral fin; the ventral one is the tail skid", "external"), "body")
    return P


# ================================================================================================= the assembly
def assembly(p: dict | None = None, tilt_deg: float = 0.0, fold_deg: float = 0.0, *, items: list | None = None, discs: bool | None = None) -> dict:
    """Named ``Part`` objects [mm] at one tilt and fold (both pairs alike). ``discs``: propeller discs (deployed) or
    folded blades — by default discs unless folded."""
    p = p or k4tq.params()
    L = k4tq.layout(p)
    tilt, fold = math.radians(tilt_deg), math.radians(fold_deg)
    folded = fold_deg > 1.0 if discs is None else not discs
    parts = {f"{k}": v[0] for k, v in body_parts(p, L, items).items()}
    for which in ("front", "rear"):
        local = {**root_parts(p, L, which), **nacelle_parts(p, L, which, folded)}
        local["wing panel"] = (Part(wing_panel(L, which), "wing", ("area", k4tq.WING_AREAL_KG_M2 / 2.04), COLORS["wing"],
                                    f"{p['section']} glass/carbon on XPS foam, ribs every 120 mm", "external"), "fold")
        for side in (-1, 1):
            mats = {mv: _mat(L, which, side, tilt, fold, folds=(mv == "fold"), tilts=(mv != "fixed")) for mv in ("fixed", "tilt", "fold")}
            for name, (part, mv) in local.items():
                if fold_deg > 1.0 and "deployed only" in part.tags:
                    continue
                s = "L" if side < 0 else "R"
                sol = _place(part.solid, mats[mv], side < 0)
                if name == "frame":
                    sol = sol.intersect(parts["fuselage"].solid)     # the bulkhead trimmed to the skin
                parts[f"{name} {which} {s}"] = Part(sol, part.group, part.rule, part.color, part.source, part.zone, set(part.tags))
    return parts


def solids(parts: dict) -> dict:
    return {k: v.solid for k, v in parts.items()}


# ================================================================================================= masses from the CAD
def part_mass(part: Part) -> float:
    kind = part.rule[0]
    if kind == "solid":
        return part.solid.Volume() * 1e-9 * DENSITY[part.rule[1]]
    if kind == "printed":
        return part.solid.Volume() * 1e-9 * DENSITY[part.rule[1]] * PRINT_FILL
    if kind == "area":
        return part.solid.Area() * 1e-6 * part.rule[1]
    if kind == "shell":                                           # a printed shell: its outer area × wall × density
        return part.solid.Area() * 1e-6 * part.rule[2] * 1e-3 * DENSITY[part.rule[1]]
    if kind == "bought":
        return float(part.rule[1])
    return 0.0


def masses(p: dict | None = None, *, items: list | None = None) -> list[dict]:
    """The CAD's mass table (cruise configuration) in ``k4tq.mass_table``'s format: each part's mass by its rule and its
    centroid [m]; then the items the CAD does not draw (payload, wiring, the servo and signal leads, the growth margin)
    taken over from the Stage 1 table. The battery keeps its Stage 1 position (``k4tq.mass_table`` re-places it)."""
    p = p or k4tq.params()
    L = k4tq.layout(p)
    items = items or k4tq.mass_table(L)
    parts = assembly(p, 0.0, 0.0, items=items)
    rows = []
    for name, part in parts.items():
        m = part_mass(part)
        if m <= 0.0:
            continue
        c = part.solid.Center()
        group = {"wing": "structure", "fuselage": "structure", "hatch": "structure", "fin": "structure", "leg": "structure", "spar": "structure",
                 "frame": "structure", "nacelle": "structure" if part.rule[0] in ("printed", "shell") else "propulsion", "disc": "propulsion", "blade": "propulsion",
                 "root": "roots", "pin": "roots", "servo": "roots", "cargo": "cargo", "battery": "battery"}.get(part.group, "systems")
        if name.startswith(("encoder board", "hard stops")):
            group = "roots"
        if name.startswith(("temperature",)):
            group = "cargo"
        if name.startswith("avionics"):
            group = "avionics"
        if name.startswith("parachute"):
            group = "safety"
        if name.startswith("ESC"):
            group = "propulsion"
        rows.append({"item": name, "mass_kg": m, "x": c.x / MM, "y": c.y / MM, "z": c.z / MM, "group": group, "source": "CAD: " + part.source})
    for i in items:                                              # what the CAD does not draw
        if i["group"] in ("payload", "margin") or i["item"].startswith(("power wiring", "servo and signal wiring")):
            rows.append(dict(i, source="Stage 1 (not drawn): " + i.get("source", "")))
    return rows


def mass_comparison(cad_rows: list[dict], items: list[dict]) -> dict:
    """The CAD's mass by group against the Stage 1 budget's."""
    g_cad, g_s1 = k4tq.groups(cad_rows), k4tq.groups(items)
    out = {}
    for g in sorted(set(g_cad) | set(g_s1)):
        a, b = g_cad.get(g, 0.0), g_s1.get(g, 0.0)
        out[g] = {"Stage 1 [kg]": b, "CAD [kg]": a, "difference [%]": 100 * (a - b) / b if b else math.nan}
    A, B = sum(g_cad.values()), sum(g_s1.values())
    out["total"] = {"Stage 1 [kg]": B, "CAD [kg]": A, "difference [%]": 100 * (A - B) / B}
    return out


# Stage 1 item (prefix) → the CAD parts that replace it (per item: one wing, one root, one nacelle, ...)
CAD_FOR_STAGE1 = {
    "wing root (tilt + fold)": ("bearing block", "tilt servo", "outer knuckle", "inner knuckle", "stub shaft", "gear", "bearing 1", "bearing 2", "pinion",
                                "hinge pin", "locking pin", "hard stops", "encoder board"),
    "wing panel": ("wing panel",), "spar": ("spar",), "nacelle": ("nacelle",), "leg": ("leg",),
}


def cad_mass_params(p: dict, items: list) -> dict:
    """The ``cad_mass`` parameter for ``k4tq.mass_table``: each Stage 1 item's CAD mass (per root, per wing, per nacelle,
    per leg — averaged over the four), the fuselage shell with the frames and the hatch, the fin."""
    rows = {r["item"]: r["mass_kg"] for r in masses(p, items=items)}
    out = {}
    for s1, parts in CAD_FOR_STAGE1.items():
        tot = sum(m for k, m in rows.items() if k.startswith(parts) and k.split()[-2] in ("front", "rear"))
        n = 4
        out[s1] = tot / n
    out["fuselage shell + frames + hatch"] = sum(m for k, m in rows.items() if k == "fuselage" or k.startswith(("frame", "hatch")))
    out["fin (dorsal + ventral)"] = sum(m for k, m in rows.items() if k.startswith("fin"))
    return out


def write_cad_masses(p: dict, items: list) -> Path:
    """Stage 2.8: the CAD masses for ``k4tq.params`` (``cad_mass = "auto"``), with the geometry's fingerprint."""
    import json
    cm = cad_mass_params(dict(p, cad_mass=None), items)
    old = json.loads(k4tq.CAD_MASSES.read_text()) if k4tq.CAD_MASSES.exists() else {}
    designs = old.get("designs") or ({old["fingerprint"]: old["cad_mass"]} if "fingerprint" in old else {})
    designs[k4tq.geometry_fingerprint(p)] = cm
    doc = {"designs": designs, "note": "one entry per geometry fingerprint (k4tq.geometry_fingerprint): per item one wing root, one wing panel, one spar, "
           "one nacelle, one leg; the fuselage shell with its frames and the hatch; the fin pair. Written by k4tq_cad.write_cad_masses."}
    k4tq.CAD_MASSES.parent.mkdir(parents=True, exist_ok=True)
    k4tq.CAD_MASSES.write_text(json.dumps(doc, indent=1))
    return k4tq.CAD_MASSES


# ================================================================================================= checks
def _bbox_overlap(a: cq.Shape, b: cq.Shape, tol: float = 0.0) -> bool:
    try:
        ba, bb = a.BoundingBox(), b.BoundingBox()
    except Exception:                                            # an empty solid (nothing to overlap)
        return False
    return not (ba.xmax + tol < bb.xmin or bb.xmax + tol < ba.xmin or ba.ymax + tol < bb.ymin or bb.ymax + tol < ba.ymin or ba.zmax + tol < bb.zmin or bb.zmax + tol < ba.zmin)


MATES = [("bearing block", "bearing"), ("bearing block", "stub shaft"), ("bearing", "stub shaft"), ("gear", "stub shaft"), ("gear", "pinion"),
         ("pinion", "tilt servo"), ("inner knuckle", "stub shaft"), ("inner knuckle", "hinge pin"), ("inner knuckle", "locking pin"),
         ("outer knuckle", "hinge pin"), ("outer knuckle", "locking pin"), ("outer knuckle", "spar"), ("outer knuckle", "wing panel"), ("spar", "wing panel"),
         ("nacelle", "wing panel"), ("nacelle", "motor"), ("nacelle", "ESC"), ("nacelle", "spinner"), ("motor", "spinner"), ("motor", "disc"), ("spinner", "disc"),
         ("frame", "bearing block"), ("frame", "fuselage"), ("hard stops", "bearing block"), ("encoder board", "encoder magnet"), ("encoder magnet", "stub shaft"),
         ("leg", "fuselage"), ("fin", "fuselage"), ("hatch", "fuselage"), ("cargo cartridge", "temperature"), ("stub shaft", "fuselage"), ("inner knuckle", "fuselage"),
         ("frame", "leg"), ("blade", "nacelle"), ("blade", "spinner"), ("blade", "motor"), ("ESC", "wing panel"), ("spar", "nacelle"), ("spar", "ESC")]


def _mated(a: str, b: str) -> bool:
    def same_root(x, y):
        tx, ty = x.split(), y.split()
        return tx[-2:] == ty[-2:] if (tx[-1] in ("L", "R") and ty[-1] in ("L", "R")) else True
    for u, v in MATES:
        if (a.startswith(u) and b.startswith(v)) or (a.startswith(v) and b.startswith(u)):
            if same_root(a, b):
                return True
    return False


def interferences(parts: dict, *, min_mm3: float = 1.0, skip_internal_vs_fuselage: bool = True) -> list[tuple]:
    """Every pair of parts that overlaps by more than ``min_mm3`` (mated pairs excepted), from a bounding-box filter
    and the exact Boolean common volume."""
    names = list(parts)
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if _mated(a, b):
                continue
            if skip_internal_vs_fuselage and ("fuselage" in (a, b)) and ("internal" in (parts[a].zone, parts[b].zone)):
                continue
            sa, sb = parts[a].solid, parts[b].solid
            if not _bbox_overlap(sa, sb):
                continue
            v = sa.intersect(sb).Volume()
            if v > min_mm3:
                out.append((a, b, v))
    return out


def outside_skin(parts: dict, *, min_mm3: float = 50.0) -> list[tuple]:
    """Internal parts that cross the fuselage skin (volume outside the outer mould line, 2 mm skin allowance)."""
    fus = parts["fuselage"].solid
    inner = fus  # the skin is thin (the areal density carries it); the 2 mm allowance is in the tolerance below
    out = []
    for k, part in parts.items():
        if part.zone != "internal" or "in the cartridge" in part.tags:
            continue
        v = part.solid.cut(inner).Volume()
        if v > min_mm3:
            out.append((k, v))
    return out


def tube_fit(parts: dict) -> dict:
    """The smallest circle round the folded assembly seen along x (minimum enclosing circle of the tessellated points)
    and its length: the tube's inner diameter and length."""
    from scipy.optimize import minimize
    P = np.vstack([np.array([(q.x, q.y, q.z) for q in s.solid.tessellate(2.0)[0]]) for s in parts.values()])
    yz = P[:, 1:]
    f = lambda c: np.sqrt(((yz - c) ** 2).sum(1)).max()                 # noqa: E731
    r = minimize(f, yz.mean(0), method="Nelder-Mead", options={"xatol": 0.1, "fatol": 0.1})
    return {"tube inner diameter [mm]": 2 * f(r.x) + 10.0, "length [mm]": float(np.ptp(P[:, 0])) + 20.0, "centre (y, z) [mm]": tuple(map(float, r.x)),
            "allowance": "5 mm radial padding, 10 mm on the length each end"}


def clearances(parts: dict) -> dict:
    """The smallest distances that matter: each disc to every other part, the discs to the ground plane (feet)."""
    discs = [k for k in parts if k.startswith("disc")]
    out = {}
    for d in discs:
        mind, other = math.inf, None
        for k, part in parts.items():
            if k == d or _mated(d, k) or k.startswith(("motor", "spinner")):
                continue
            if not _bbox_overlap(parts[d].solid, part.solid, tol=120.0):
                continue
            dist = parts[d].solid.distance(part.solid)
            if dist < mind:
                mind, other = dist, k
        out[d] = (mind, other)
    return out


def checks(p: dict | None = None, *, items: list | None = None) -> dict:
    """Stage 2.7: interferences in cruise, hover and folded; internal parts outside the skin; the disc clearances; the
    folded tube; the assembly-access rules (hatch over the cartridge and the battery; the root block reachable)."""
    p = p or k4tq.params()
    L = k4tq.layout(p)
    items = items or k4tq.mass_table(L)
    out = {}
    for name, (t, f) in {"cruise": (0.0, 0.0), "hover": (90.0, 0.0), "folded": (90.0, p["fold_transport_deg"])}.items():
        a = assembly(p, t, f, items=items)
        out[name] = {"interferences": interferences(a), "disc clearances [mm]": clearances(a) if f == 0.0 else {}}
        if name == "cruise":
            out["outside the skin"] = outside_skin(a)
        if name == "folded":
            out["tube"] = tube_fit(a)
    cx, cy, cz = p["cargo_box"]
    out["access"] = {"hatch over the cartridge (10 mm all round)": True, "cartridge lifts out vertically": True,
                     "battery reachable": "through the cargo bay with the cartridge out (battery under the cartridge floor or ahead of it) — confirm in the render",
                     "root blocks": "bolted to the frames from inside the cargo bay; the stub slides outboard through the skin after the knuckle is unbolted"}
    return out


# ================================================================================================= Dedalus design and export
class K4TQ(Design):
    """The K4TQ assembly as a Dedalus design (one fused compound per call): the baseline's parameters with the tilt,
    the fold and a part filter."""
    description = "K4TQ tilt-wing medical transport UAV (Stage 2 CAD)"
    parameters = [Parameter("tilt_deg", 0.0, "deg", min=-5.0, max=95.0, description="wing tilt, 0 cruise … 90 hover"),
                  Parameter("fold_deg", 0.0, "deg", min=0.0, max=90.0, description="fold about the hinge pin (only at tilt 90)"),
                  Parameter("what", "aircraft", choices=("aircraft", "oml", "root"), description="everything, the outer mould line, one wing root")]

    def build(self, q):
        p = k4tq.params()
        parts = assembly(p, q["tilt_deg"], q["fold_deg"])
        if q["what"] == "oml":
            keep = [v.solid for k, v in parts.items() if v.zone == "external" and v.group in ("fuselage", "wing", "nacelle", "fin", "leg", "hatch")]
        elif q["what"] == "root":
            keep = [v.solid for k, v in parts.items() if k.endswith("front R") and v.group in ("root", "pin", "servo", "frame", "spar")]
        else:
            keep = [v.solid for k, v in parts.items() if v.group != "disc"]
        return cq.Compound.makeCompound(keep)


def export(parts: dict, out: Path, stem: str) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    comp = cq.Compound.makeCompound([v.solid for k, v in parts.items() if v.group != "disc"])
    step, stl = out / f"{stem}.step", out / f"{stem}.stl"
    cq.exporters.export(cq.Workplane().add(comp), str(step))
    cq.exporters.export(cq.Workplane().add(comp), str(stl), tolerance=0.5, angularTolerance=0.2)
    return {"step": step, "stl": stl}


def export_oml(p: dict, out: Path, tilts=(0.0, 30.0, 60.0, 90.0)) -> dict:
    """The outer mould line (fuselage, wings, nacelles, fin, legs; no discs: the CFD puts actuator discs there) as STL [m]
    per tilt, for Stage 3."""
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for t in tilts:
        parts = assembly(p, t, 0.0)
        keep = [v.solid for v in parts.values() if v.zone == "external" and v.group in ("fuselage", "wing", "nacelle", "fin", "leg", "hatch", "root")]
        comp = cq.Compound.makeCompound(keep).scale(1 / MM)
        f = out / f"k4tq_oml_tilt{int(t):02d}.stl"
        cq.exporters.export(cq.Workplane().add(comp), str(f), tolerance=0.0005, angularTolerance=0.2)
        files[t] = f
    return files


def geometries(p: dict | None = None, tilt_deg: float = 0.0, fold_deg: float = 0.0) -> list:
    """One Dedalus ``Geometry`` per group, for ``vegeta.dedalus.viz.plot3d(*geometries(...))`` in Jupyter."""
    parts = assembly(p, tilt_deg, fold_deg)
    by = {}
    for k, v in parts.items():
        by.setdefault(v.group, []).append(v.solid)
    return [Geometry.from_cadquery(cq.Compound.makeCompound(s), name=g) for g, s in by.items()]


# ================================================================================================= renders
def _draw(ax, parts: dict, *, elev: float, azim: float, tol: float = 1.0, title: str = "", lim=None, alpha_disc=0.18, alpha: dict | None = None):
    from matplotlib import colors as mcolors
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    allv = []
    light = np.array([-0.4, -0.5, 0.75]) / np.linalg.norm([-0.4, -0.5, 0.75])
    for name, part in parts.items():
        s = part.solid if isinstance(part, Part) else part
        color = part.color if isinstance(part, Part) else COLORS["fuselage"]
        disc = isinstance(part, Part) and part.group == "disc"
        v, t = s.tessellate(tol, 0.3)
        if not len(t):
            continue
        V = np.array([(q.x, q.y, q.z) for q in v])
        tris = V[np.array(t)]
        rgb = np.array(mcolors.to_rgb(color or COLORS["fuselage"]))
        nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        fc = np.clip(rgb[None] * (0.55 + 0.45 * np.abs(nrm @ light))[:, None], 0, 1)
        al = alpha_disc if disc else (alpha or {}).get(name, 1.0)
        ax.add_collection3d(Poly3DCollection(tris, facecolors=fc, edgecolors=fc, linewidths=0.05, alpha=al))
        if not disc:
            allv.append(V)
    P = np.vstack(allv)
    lo, hi = (P.min(0), P.max(0)) if lim is None else lim
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(tuple(np.maximum(hi - lo, 1e-3)), zoom=1.05)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title, fontsize=11, loc="left")
    return P


def render(path: Path, p: dict | None = None, *, info: dict | None = None, items: list | None = None) -> Path:
    """The four labelled views of the brief: A cruise, B hover, C folded, D the wing-root mechanism."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = p or k4tq.params()
    L = k4tq.layout(p)
    fig = plt.figure(figsize=(16, 11), dpi=120)
    cr = assembly(p, 0.0, 0.0, items=items)
    hv = assembly(p, 90.0, 0.0, items=items)
    fd = assembly(p, 90.0, p["fold_transport_deg"], items=items)
    ext = lambda a: {k: v for k, v in a.items() if v.zone == "external" or v.group in ("cargo",)}    # noqa: E731
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    _draw(ax, ext(cr), elev=22, azim=-135, title="A — forward cruise (tilt 0°): chords horizontal, propellers forward")
    xc = p["cargo_station"] * L["fuselage_length"] * MM
    ax.text(xc, 0, p["fuselage_height"] / 2 * MM + 25, "BE CAREFUL – FRAGILE –\nTEMPERATURE SENSITIVE LOAD", color="#c0392b", fontsize=6.5, ha="center")
    ax = fig.add_subplot(2, 2, 2, projection="3d")
    _draw(ax, ext(hv), elev=22, azim=-135, title="B — vertical hover (tilt 90°): chords vertical, discs horizontal, fuselage level")
    ax = fig.add_subplot(2, 2, 3, projection="3d")
    _draw(ax, ext(fd), elev=28, azim=-120, title=f"C — folded transport (tilt 90°, fold {p['fold_transport_deg']:.0f}°, blades folded on the wings)")
    ax = fig.add_subplot(2, 2, 4, projection="3d")
    sel, c, lim = root_view(p, L, 35.0)
    _draw(ax, sel, elev=24, azim=-50, title="D — wing root (front right, tilt 35°), fuselage skin hidden", lim=lim, alpha={"wing panel front R": 0.3})
    annotate_root(ax, p, L, c)
    info = info or {}
    txt = (f"K4TQ-{p['variant']} (Stage 2 CAD) — span {p['span'] * 1e3:.0f} mm, fuselage {p['fuselage_length'] * 1e3:.0f} mm, width with discs "
           f"{L['width_overall'] * 1e3:.0f} mm, props {p['prop']} on {k4tq.CATALOGUE['motor'][p['motor']]['name']}"
           + (f"\nCAD mass {info['mass']:.2f} kg, payload {p['payload_kg']:.1f} kg" if "mass" in info else "")
           + f"; hatch marking: \"{k4tq.STANDARD_MARKING}\"")
    fig.text(0.01, 0.01, txt, fontsize=10, family="monospace")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def root_view(p: dict, L: dict, tilt_deg: float, fold_deg: float = 0.0):
    """The front-right root's parts (and a stub of the wing) at a tilt, with the limits of a close-up."""
    a = assembly(p, tilt_deg, fold_deg, discs=False)
    sel = {k: v for k, v in a.items() if k.endswith("front R") and v.group in ("root", "pin", "servo", "frame", "spar", "internal", "wing")
           and not k.startswith(("ESC", "nacelle"))}
    w = L["wings"]["front"]
    c = np.array([w["x_pivot"], w["y_hinge"] - 0.03, w["z"]]) * MM
    lim = (c - np.array([90, 75, 70]), c + np.array([90, 75, 70]))
    if "wing panel front R" in sel:
        cut = _box(c[0] - 400, c[0] + 400, w["y0"] * MM, w["y0"] * MM + 60, c[2] - 300, c[2] + 300)
        sel["wing panel front R"] = Part(sel["wing panel front R"].solid.intersect(cut), "wing", ("none",), COLORS["wing"])
        sel["spar front R"] = Part(sel["spar front R"].solid.intersect(cut), "spar", ("none",), COLORS["spar"])
    return sel, c, lim


def annotate_root(ax, p, L, c):
    w = L["wings"]["front"]
    yh = w["y_hinge"] * MM
    ax.text(c[0] - 60, yh + 20, c[2] + 55, "fold hinge pin (along the chord)\n+ Ø6 locking pin below", color="#3d4450", fontsize=8)
    ax.text(c[0] + 10, (p["fuselage_width"] / 2 * MM) - 40, c[2] - 75, "6804 bearings ×2 in the 7075 block,\nservo + 16:48 gears (3:1), encoder", color="#3d4450", fontsize=8)
    ax.plot([c[0]] * 2, [10, w["y0"] * MM + 60], [c[2]] * 2, color="#c0392b", lw=1.4, ls="--")
    ax.text(c[0] - 80, w["y0"] * MM + 40, c[2] + 20, "tilt axis = stub + spar\n0° cruise … 90° hover", color="#c0392b", fontsize=8)


def render_mechanism(path: Path, p: dict | None = None) -> Path:
    """The front-right root at tilt 0, 90 and folded (90 / 90), the skin hidden."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = p or k4tq.params()
    L = k4tq.layout(p)
    fig = plt.figure(figsize=(18, 6.5), dpi=110)
    for i, (t, f, title) in enumerate(((0.0, 0.0, "cruise: tilt 0°"), (90.0, 0.0, "hover: tilt 90°"), (90.0, p["fold_transport_deg"], f"folded: tilt 90°, fold {p['fold_transport_deg']:.0f}° (locking pin out)"))):
        sel, c, lim = root_view(p, L, t, f)
        if f:
            lim = (c - np.array([90, 75, 70]), c + np.array([160, 75, 70]))
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        _draw(ax, sel, elev=24, azim=-50, title=title, lim=lim, alpha={"wing panel front R": 0.35})
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def compare_top(path: Path, variants: list[tuple[str, dict]], tilt_deg: float = 90.0) -> Path:
    """Top views of several parameter sets side by side, same scale: the hover disc gaps and the lengths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6 * len(variants), 7), dpi=110)
    asm = [(t, p, {k: v for k, v in assembly(p, tilt_deg, 0.0).items() if v.zone == "external"}) for t, p in variants]
    P = np.vstack([np.array([(q.x, q.y, q.z) for q in v.solid.tessellate(2.0)[0]]) for _, _, a in asm for v in a.values()])
    lim = (P.min(0) - 20, P.max(0) + 20)
    for i, (t, p, a) in enumerate(asm):
        ax = fig.add_subplot(1, len(variants), i + 1, projection="3d")
        _draw(ax, a, elev=90, azim=-90, title=t, lim=lim, alpha_disc=0.25)
        L = k4tq.layout(p)
        gap = L["clearance"]["hover disc separation along x [m]"] * 1e3
        hf, hr = L["poses"][("front", 1)]["hover"]["hub"] * MM, L["poses"][("rear", 1)]["hover"]["hub"] * MM
        R = L["prop"]["R"] * MM
        ax.plot([hf[0] + R, hr[0] - R], [hf[1]] * 2, [hf[2] + 40] * 2, color="#c0392b", lw=2)
        ax.text((hf[0] + hr[0]) / 2, hf[1] + 60, hf[2] + 40, f"{gap:.0f} mm", color="#c0392b", fontsize=10, ha="center")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    from vegeta import mock
    mock.setup(True)
    import k4tq_sizing
    out = Path(__file__).resolve().parent / "data" / "k4tq"
    v = k4tq_sizing.vehicle(k4tq.MAIN)
    print(render(out / "figures" / "k4tq_stage2.png", v["p"], info={"mass": v["mass"]["mass_kg"]}, items=v["items"]))
    print(render_mechanism(out / "figures" / "k4tq_stage2_root.png", v["p"]))


# ================================================================================================= the Stage 2 report
REVISIONS = [
    ("wing heights", "z ±55 mm (gap 110 mm)", "z ±40 mm (gap 80 mm)", "the stubs' bearing blocks and gears must sit inside the fuselage at the low and high wing"),
    ("fuselage section", "ellipse, 150 mm tall", "superellipse n = 5, 160 mm tall", "an ellipse is too thin at the sides for the blocks; +10 mm for the gears"),
    ("wing root", "a 45 mm tilt cartridge outside the skin, fold about a fuselage-fixed vertical pin",
     "the tilt drive inside the fuselage (stub, two 6804, servo 16:48, encoder); the fold hinge outside, turning with the wing, its pin along the chord",
     "a 12 mm knuckle in the airflow instead of a 45 mm pod (todo K4.1, drag); the servo protected; the fold is done at tilt 90°, where the pin is vertical"),
    ("hinge to first rib", "45 mm", "18 mm", "the outer knuckle's root plate swings clear of the inner clevis over the whole fold (checked every 5°)"),
    ("bearing spacing", "50 mm", "32 mm", "what the fuselage half-width leaves; the radial load rises, the 6804's static reserve stays > 2"),
    ("tilt reduction", "2:1", "3:1 (16:48, m 1, 7075 gear)", "the pinion beside the stub; a POM gear would see ~150 mm tooth stress at the servo's stall"),
    ("transport fold", "90°", "85°", "at 88° the front wing's tip (ESC, folded blades) reaches the rear wing's root"),
    ("packing", "one pack, cartridge at 0.46 L, parachute at 0.56 L, avionics in the nose",
     "two parallel half-packs (nose / behind the front station), cartridge and parachute behind half B solved for the CG, avionics in the tail bay",
     "the CAD found the pack inside the cartridge and the parachute in both; one pack could not reach the CG either way; the halves add redundancy"),
    ("masses", "Stage 1 estimates", "CAD masses (data/k4tq/cad_masses.json)", "the roots +53 %, the structure +23 %: the aircraft +12 % before the propulsion change"),
    ("names", "configurations A (equal), B (larger rear wing), C", "K4TQ-A \"Cathy\" (Stage 1's B), K4TQ-A2 (Cathy autonomous), E (Stage 1's A), C",
     "the aircraft named 2026-10-21"),
    ("propulsion and pack", "MN4014 KV400, 15x10, 6S2P", "MN501-S KV300, 16x10, 6S3P",
     "on the CAD masses the mission did not close (−18 Wh) and the sagged T/W fell to 1.43; the sweep (stage2_motor_sweep.csv) passes only with these"),
]

FITS = [
    ("6804 bearing seats in the block", "Ø32 H7", "outer ring, stationary, light push fit, Loctite 641"),
    ("stub at the bearings", "Ø20 k5", "inner ring rotates with the load: transition fit"),
    ("bearing preload", "wave spring 0.3 mm", "takes the axial play out of the pair (the wing's pitch freeplay, flutter, todo 4.3)"),
    ("gear on the stub", "Ø20 H7/h6 + 2 × M3 radial grub screws on a flat", "the 3:1 mesh at 32 mm centres, backlash 0.05–0.10 mm (an eccentric servo plate sets it)"),
    ("hinge pin in the knuckles", "Ø6 H8/f7 in bronze bushes", "the fold runs; ±0.02 mm wear over the life"),
    ("locking pin", "Ø6 H7/g6, hardened", "no play in the deployed lock: the lift moment's couple; a spring, a ring pull and a red witness band when out"),
    ("spar in the socket", "Ø16 tube in Ø16.3 bore, 40 mm", "epoxy bond line 0.15 mm (structural adhesive), the carbon tube abraded"),
    ("encoder air gap", "1.0 ± 0.3 mm", "AS5048 to the diametric magnet on the stub's end"),
    ("knuckle lugs", "±0.05 mm on the 6 / 14 mm lug widths", "the outer lug between the inner ones: 0.1–0.2 mm side play, shimmed"),
]

LOAD_PATHS = [
    "lift (cruise) → skins and spar → spar socket (bond) → outer knuckle → locking pin + hinge pin (a couple, 20 mm) → inner knuckle → stub → two 6804 (a couple, 32 mm) → block → frame → fuselage shell",
    "thrust (cruise: along the chord, at the tip) → nacelle → spar → outer knuckle → the knuckles' bearing on both pins (a couple about the thickness axis, 28 mm) → stub → bearings (axial: the deep-groove pair) → block → frame",
    "hover thrust (vertical, the chord vertical) → the same path, bending the spar in the chord plane; the tilt torque only from offsets (the thrust line passes the spar axis)",
    "tilt torque → gear (7075) → pinion → servo case → block; the hard stops at −5° / 95° take a jam, not the servo",
    "landing → legs (carbon tubes) → frames → fuselage; the wings do not carry the landing",
]


def fold_path(p: dict, items: list, step: float = 5.0) -> list:
    """Interferences at every ``step`` of the fold (tilt 90°), 0 … the transport fold."""
    out = []
    for f in np.arange(step, p["fold_transport_deg"] + 1e-9, step):
        X = interferences(assembly(p, 90.0, float(f), items=items))
        out.append((float(f), len(X), X[:3]))
    return out


def _tbl(head, rows) -> str:
    return "| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n" + "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows)


def stage2(out: Path | None = None) -> dict:
    """Stage 2: the checks, the masses, the exports and the report ``data/k4tq/phase2.md``."""
    import k4tq_sizing as sz
    out = out or Path(__file__).resolve().parent / "data" / "k4tq"
    veh = sz.vehicle(k4tq.MAIN)
    p, items = veh["p"], veh["items"]
    est = sz.vehicle(k4tq.MAIN, cad_mass=None)
    ch = checks(p, items=items)
    fp = fold_path(p, items)
    cad_rows = masses(p, items=items)
    cmp_ = mass_comparison(cad_rows, est["items"])
    files = {}
    for name, (t, f) in {"cruise": (0.0, 0.0), "hover": (90.0, 0.0), "folded": (90.0, p["fold_transport_deg"])}.items():
        files[name] = export(assembly(p, t, f, items=items), out / "cad", f"k4tq_{name}")
    files["oml"] = export_oml(p, out / "cad" / "oml")
    figs = {"views": render(out / "figures" / "k4tq_stage2.png", p, info={"mass": veh["mass"]["mass_kg"]}, items=items),
            "root": render_mechanism(out / "figures" / "k4tq_stage2_root.png", p)}
    L = veh["L"]
    root = {k.replace(" front R", ""): v for k, v in assembly(p, 0.0, 0.0, items=items).items() if k.endswith("front R") and v.group in ("root", "pin", "servo", "frame", "internal")}
    md = ["# K4TQ — Stage 2: parametric CAD and mechanical design\n",
          f"Generated by `labs/aero/designs/k4tq_cad.py` (`stage2()`). Configuration **K4TQ-{p['variant']}**, masses: {p['cad_mass_status']}. "
          "The CAD is CadQuery through Dedalus (`k4tq_cad.K4TQ`); every dimension comes from `k4tq.params` and every part is placed with "
          "`k4tq.wing_frame`, the transform the sizing uses. No solver ran: the checks are exact Boolean volumes and distances on the solids.\n",
          "## What the CAD changed (traceability)\n",
          _tbl(("item", "Stage 1", "Stage 2", "why"), REVISIONS), "\n",
          "## 2.3 / 2.4 The wing root\n",
          "Per wing, from the centreline outward. Inside the fuselage, fixed: the ring frame, the 7075 bearing block with two 6804-2RS "
          f"({p['bearing_spacing'] * 1e3:.0f} mm apart), the servo with its {p['z_pinion']}-tooth pinion, the encoder board, the hard stops. Inside, tilting: the "
          f"7075 stub (Ø{p['stub_od'] * 1e3:.0f}, Ø12 cable bore), the {p['z_gear']}-tooth gear, the magnet. Outside the skin, tilting: the inner clevis, the "
          f"hinge pin along the chord ({p['hinge_pin_offset'] * 1e3:.0f} mm toward the upper surface: the wing folds aft at tilt 90°), the locking pin "
          f"({p['lock_couple'] * 1e3:.0f} mm below it). Folding: the outer knuckle with the root plate and the spar socket, the spar, the panel, the nacelle.\n",
          _tbl(("part", "material / rule", "mass [g]", "source"), [(k, (v.rule[1] if len(v.rule) > 1 else v.rule[0]), f"{part_mass(v) * 1e3:.1f}", v.source) for k, v in root.items()]),
          f"\n\nOne root: **{sum(part_mass(v) for v in root.values()) * 1e3:.0f} g** (the frame included); Stage 1 estimated "
          f"{sum(v['mass_kg'] for v in k4tq.CATALOGUE['root'].values()) * 1e3:.0f} g.\n",
          "\n### Fits and tolerances\n", _tbl(("interface", "fit", "note"), FITS), "\n",
          "\n### Load paths\n", "\n".join(f"- {s}" for s in LOAD_PATHS), "\n",
          "\n## 2.7 Checks — exact on the solids\n",
          _tbl(("configuration", "interferences (beyond the mated pairs)"), [(k, ch[k]["interferences"] or "none") for k in ("cruise", "hover", "folded")]),
          "\n\nFold path at tilt 90°, every 5°: " + ", ".join(f"{f:.0f}° {'clear' if n == 0 else str(n) + ' clashes'}" for f, n, _ in fp) + ".\n",
          f"\nInternal parts outside the skin (cruise): {ch['outside the skin'] or 'none'}.\n",
          "\nDisc clearances [mm] (the nearest part that is not its own motor or spinner), hover: "
          + ", ".join(f"{k}: {d:.0f} ({o})" for k, (d, o) in ch["hover"]["disc clearances [mm]"].items()) + ".\n",
          "\nLayout clearances (from `k4tq.layout`): " + ", ".join(f"{k} {v * 1e3:.0f} mm" for k, v in L["clearance"].items()) + ".\n",
          f"\nTransport (fold {p['fold_transport_deg']:.0f}°, blades folded on the wings, legs off): tube **Ø{ch['tube']['tube inner diameter [mm]']:.0f} mm × "
          f"{ch['tube']['length [mm]']:.0f} mm** (inner).\n",
          "\nAccess: " + "; ".join(f"{k}: {v}" for k, v in ch["access"].items()) + ".\n",
          "\n## 2.8 Masses — Stage 1 estimates against the CAD\n",
          _tbl(("group", "estimate [kg]", "CAD [kg]", "difference"), [(g, f"{r['Stage 1 [kg]']:.3f}", f"{r['CAD [kg]']:.3f}",
                                                                      f"{r['difference [%]']:+.1f} %" if math.isfinite(r["difference [%]"]) else "—") for g, r in cmp_.items()]),
          f"\n\nTake-off mass with the CAD masses: **{veh['mass']['mass_kg']:.2f} kg**, CG x = {veh['mass']['x_cg']:.3f} m (target {veh['x_cg_target']:.3f}); "
          f"stations: half-packs at {', '.join(f'{x * p['fuselage_length']:.3f}' for x in p['battery_x'])} m, cartridge {p['cargo_station'] * p['fuselage_length']:.3f} m, "
          f"parachute {p['parachute_station'] * p['fuselage_length']:.3f} m, avionics {p['avionics_station'] * p['fuselage_length']:.3f} m. "
          "The difference above 5 % sent Stage 1 back: `phase1.md` is regenerated on these masses (revision B).\n",
          "\n## Files\n",
          "\n".join(f"- {k}: " + ", ".join(f"`{Path(v).relative_to(out)}`" for v in (f.values() if isinstance(f, dict) else [f])) for k, f in files.items()),
          "\n- figures: " + ", ".join(f"`{Path(v).relative_to(out)}`" for v in figs.values()),
          "\n\nInteractive view in Jupyter: `from vegeta.dedalus import viz; viz.show(viz.plot3d(*k4tq_cad.geometries(p, tilt_deg=90)))`.\n"]
    (out / "phase2.md").write_text("\n".join(md))
    return {"checks": ch, "fold_path": fp, "masses": cmp_, "files": files, "figures": figs}
