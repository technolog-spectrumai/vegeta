"""GAVILAN-H with the one-shot recoilless net interceptor fitted (CAD).

The interceptor (``net_oneshot_cad``: 70 g net cartridge, 32 mm bore, rear venturi throat 27.2 mm, one-shot charge
vessel opened by a burst diaphragm), with the 0.25 m tube, fitted to GAVILAN-H (``gavilan.GavilanH``: 1.0 m span,
fuselage 64 mm x 860 mm, two 7-inch tractors at y +-200 mm, two all-moving tail halves in the propellers' wash):

- **the tube on the body axis**, its muzzle 10 mm out of the nose: the two tractors on the wing leave the nose free
  (GAVILAN-F's layout, ``weapon="emitter"``, now with the 32 mm net tube instead of the 8 mm one);
- **the chamber and the venturi inside the fuselage** under the wing root; a **38 mm jet pipe** runs on the axis to
  the fuselage's end, so the rear jet leaves behind the tail; GAVILAN-H's tail cone narrows below it, so a 47 mm
  **fairing** carries the pipe to the end (the fin and the tail halves sit on it);
- **the tail halves move out** (their roots 26 mm from the centre plane, not 14) and lose the through-shaft: each half
  turns on its own stub shaft and servo, so the jet pipe passes between them;
- **the charge vessel lies along x in a belly blister** under the wing root (the 64 mm fuselage cannot hold it beside
  the chamber), the burst-diaphragm holder between the vessel and the chamber's side inlet;
- the breech door becomes a **hatch on the fuselage's top**, ahead of the wing, for loading the cartridge.

Frame: GAVILAN's (``FixedWing``: the wing root's leading edge at the origin, x aft, y span, z up; the body axis at
z = -32 mm). ``export()`` writes the STEP, a GLB, and a GLB cut on the centre plane.
"""
from __future__ import annotations

import math
from pathlib import Path

import cadquery as cq

import gavilan as gv
import net_oneshot_cad as no
from golf_gun_cad import cyl_x, cyl_y, cyl_z

TUBE_L = 250.0
TAILERON_GAP = 26.0
JET_PIPE_OD = 41.0
COLORS = dict(no.COLORS, fuselage=(0.86, 0.87, 0.90, 0.30), wing=(0.80, 0.82, 0.86, 1), tail=(0.72, 0.74, 0.78, 1),
              props=(0.30, 0.30, 0.32, 0.35), pipe=(0.70, 0.72, 0.76, 1), blister=(0.86, 0.87, 0.90, 0.30),
              hatch=(0.45, 0.47, 0.52, 1), shafts=(0.25, 0.25, 0.28, 1))


def h_params():
    return gv.h_resolve({"weapon": "none", "show_props": True, "taileron_gap": TAILERON_GAP})


def tail_split(p):
    """GAVILAN-H's tail with the halves 26 mm out and a stub shaft in each (no shaft across the centre)."""
    L = gv.GavilanH.layout(p)
    r, tt = p["fuselage_diameter"] / 2, p["tail_thickness"]
    w = p["tail_span"] / 2 - p["taileron_gap"]
    half = cq.Workplane("XY").box(p["tail_chord"], w, tt, centered=(False, False, True)).translate((L["x_tail"], p["taileron_gap"], -r))
    half = half.rotate((L["x_hinge"], 0, -r), (L["x_hinge"], 1, -r), p["taileron_deg"])
    stub = cq.Workplane("XZ").circle(3.0).extrude(-18.0).translate((L["x_hinge"], p["taileron_gap"] - 6.0, -r))
    hstab = half.union(half.mirror("XZ"))
    shafts = stub.union(stub.mirror("XZ"))
    fin = (cq.Workplane("XY").box(p["tail_chord"], tt, p["fin_height"], centered=(False, True, False))
           .translate((L["x_tail"] - 0.5 * p["tail_chord"], 0, -r + JET_PIPE_OD / 2 + 2)))      # the fin sits on top of the pipe
    return hstab.union(fin), shafts


def launcher(p):
    """The interceptor's parts placed in GAVILAN's frame (the muzzle forward, -x)."""
    no.TUBE_L = TUBE_L
    raw = no.assembly()
    z_ax = -p["fuselage_diameter"] / 2
    x_muzzle = -p["nose_length"] - 10.0
    x_bore = x_muzzle + TUBE_L                       # the bore's start (the chamber's front face)

    def place(s):                                    # launcher frame (muzzle toward +x) -> aircraft frame (muzzle toward -x)
        return s.rotate((0, 0, 0), (0, 0, 1), 180).translate((x_bore, 0, z_ax))

    out = {}
    for k in ("tube", "chamber", "nozzle", "cart_shell", "cart_weight", "cart_net", "solenoid", "burst"):
        s, c = raw[k]
        out[k] = (place(s), c)
    # the jet: the venturi's exit continues as a pipe on the axis to the fuselage's end (the cone shows past it)
    x_exit = x_bore + 40 + 5 + 10 + (no.THROAT / 2 * math.sqrt(no.EPS) - no.THROAT / 2) / math.tan(math.radians(7.5))
    x_end = -p["nose_length"] + p["fuselage_length"] + 6.0
    out["pipe"] = (cyl_x(JET_PIPE_OD / 2, x_end - x_exit, x_exit, 0, z_ax, r_in=JET_PIPE_OD / 2 - 1.2), "pipe")
    # GAVILAN-H's tail cone narrows below the pipe: a 47 mm fairing over the pipe from the chamber to the tail's end
    out["fairing"] = (cyl_x(JET_PIPE_OD / 2 + 3, x_end - x_exit + 10, x_exit - 10, 0, z_ax, r_in=JET_PIPE_OD / 2 + 1.8), "fuselage")
    re = JET_PIPE_OD / 2 - 1.2
    out["jet"] = (no.revolve_x([(0, 0), (0, re), (600, re + 600 * math.tan(math.radians(12))), (600, 0)]).translate((x_end, 0, z_ax)), "jet")
    # the charge vessel lying along x in a belly blister, the burst holder between it and the chamber's inlet
    x_ch = x_bore + 20.0                             # the chamber's middle (its side inlet)
    vr, vl = 22.0, no.VESSEL_ML * 1e3 / (math.pi * 20.0 ** 2) - 4 / 3 * 20.0
    z_in = z_ax - 20 - 3 - 12                        # the chamber's inlet boss bottom
    burst = cyl_z(19, 20, x_ch, 0, z_in - 20).cut(cyl_z(12.5, 24, x_ch, 0, z_in - 22)).union(cyl_z(21, 3, x_ch, 0, z_in - 11.5))
    out["burst"] = (burst, "burst")
    out["solenoid"] = (cyl_y(9, 32, x_ch, z_in - 10).translate((0, 19 + 16, 0)), "solenoid")
    z_v = z_in - 20 - vr
    x_v0 = x_ch - 10.0
    vessel = (cyl_x(vr, vl, x_v0, 0, z_v).union(cq.Workplane("XY").sphere(vr).translate((x_v0, 0, z_v)))
              .union(cq.Workplane("XY").sphere(vr).translate((x_v0 + vl, 0, z_v))))
    out["vessel"] = (vessel.union(cyl_z(10, 8, x_ch, 0, z_v + vr - 6)), "vessel")
    out["fill"] = (cyl_x(5, 14, x_v0 + vl + vr - 2, 0, z_v), "fill")
    blister = (cq.Workplane("YZ").ellipse(vr + 9, (z_ax - z_v) + vr + 10).extrude(vl + 2 * vr + 70)
               .translate((x_v0 - vr - 35, 0, z_ax)))
    out["blister"] = (blister.intersect(cq.Workplane("XY").box(2000, 400, 400).translate((0, 0, z_ax - 200))), "blister")
    # the loading hatch on the fuselage's top over the breech
    x_cart = x_bore - 12 - 35                       # the cartridge's middle (ahead of the wing's leading edge at x = 0)
    out["hatch"] = (cq.Workplane("XY").box(80, 30, 2.5).translate((min(x_cart, -42.0), 0, 0.5)), "hatch")
    return out


def assembly():
    p = h_params()
    parts = {k: (v, k) for k, v in gv.GavilanH().parts(p).items()}
    tail, shafts = tail_split(p)
    parts["tail"] = (tail, "tail")
    parts["shafts"] = (shafts, "shafts")
    parts.update(launcher(p))
    return parts, p


def export(root=Path("output/gavilan_net/cad")):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    parts, p = assembly()
    asm = cq.Assembly(name="gavilan_h_net")
    for name, (shape, col) in parts.items():
        asm.add(shape, name=name, color=cq.Color(*COLORS[col]))
    asm.export(str(root / "gavilan_h_net.step"))
    asm.export(str(root / "gavilan_h_net.glb"), tolerance=0.3, angularTolerance=0.3)
    half = cq.Workplane("XY").box(4000, 2000, 4000).translate((0, 1000, 0))
    sec = cq.Assembly(name="gavilan_h_net_section")
    for name, (shape, col) in parts.items():
        if name in ("jet", "props"):
            continue
        sec.add(shape.cut(half), name=name, color=cq.Color(*COLORS[col]))
    sec.export(str(root / "gavilan_h_net_section.glb"), tolerance=0.3, angularTolerance=0.3)
    return root, p


if __name__ == "__main__":
    print(export())
