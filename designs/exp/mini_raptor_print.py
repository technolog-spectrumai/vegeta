"""UNSTABLE — an experiment, parked on 2026-10-19 (designs/exp/). The redesign of the briefs (Apus a one-motor plank,
Alcotan a twin, Gavilan the X cell) is unfinished: the CAD, the printed parts, notebook 49 and the tests still
describe the earlier layouts and fail. The designs may be re-examined later; they are not needed now. The races are
flown by the reference 5-inch quadcopter only (races.refquad).

The mini raptors as things to build (notebook 49 § 3b): every printed part split to a 250 × 210 × 210 mm bed, the
bought stock beside it (motors, propellers, carbon tubes, the pack and the stack), the in-house printability checks
that need no slicer — bed fit in the chosen orientation, the trailing edge against the nozzle, the overhangs that would
want support, the printed mass from the shell and the infill against the mass budget's structure — and the Mellonia
slice (PrusaSlicer) of each part when the slicer is on the PATH; when it is not, a NOT RUN table with the commands.

Printing rules assumed (``PLA`` for the pod, mounts and fittings; ``LW_PLA`` for the wings and fins, one perimeter and
no infill, the way foaming filament wings are printed): 0.4 mm nozzle, 0.2 mm layers, the trailing edge thickened to
``TE_MIN_MM`` (two LW-PLA lines) — the aerodynamic section is cut there, and ``drag_buildup`` does not know; a first
guess, not a validated print.
"""
from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import kestrel
import mini_raptor as mr

BED = (250.0, 210.0, 210.0)          # x, y, z of the example printer
NOZZLE = 0.4
TE_MIN_MM = 1.0                      # the printed trailing edge: two 0.45 mm LW-PLA lines and a little
WALL_MM = {"PLA": 1.2, "LW_PLA": 0.9}          # 3 and 2 perimeters of a 0.4 nozzle
INFILL = {"PLA": 0.08, "LW_PLA": 0.0}
DENSITY = {"PLA": 1.24, "LW_PLA": 0.60}      # g/cm³ as extruded (LW-PLA foamed)
SEGMENT_MAX_MM = 200.0               # a wing or fin segment stands on its root rib: its length is the print height
OVERHANG_DEG = 50.0                  # faces steeper than this below the horizontal want support
TUBE_OD = {"spar": 6.0, "post": 5.0, "foot": 4.0}


def settings():
    """The two print settings (Mellonia's PrintSettings): generic PLA and a foaming LW-PLA for the wings."""
    import dataclasses
    from vegeta.mellonia.examples import GENERIC_PLA_0_2MM
    pla = dataclasses.replace(GENERIC_PLA_0_2MM.replace(print={"fill_density": "8%", "perimeters": 3}), name="PLA 0.2 mm, 3 perimeters, 8 % (example)")
    lw = dataclasses.replace(GENERIC_PLA_0_2MM.replace(filament={"filament_density": 0.60, "temperature": 235, "first_layer_temperature": 235, "filament_cost": 60},
                                                       print={"perimeters": 1, "fill_density": "0%", "top_solid_layers": 2, "bottom_solid_layers": 2, "fill_pattern": "rectilinear"}),
                             name="LW-PLA 0.2 mm, one perimeter, no infill (example)")
    return {"PLA": pla, "LW_PLA": lw}


def te_cut_for(chord_mm: float, thickness: float, te_min_mm: float = TE_MIN_MM) -> float:
    """Where to cut a NACA-4 section so the trailing edge is ``te_min_mm`` thick (the printed section)."""
    for cut in np.arange(0.995, 0.80, -0.005):
        up, lo = kestrel.naca4(0.0, 0.3, thickness, n=60, te_cut=float(cut))
        if (up[0][1] - lo[0][1]) * chord_mm >= te_min_mm:
            return float(cut)
    return 0.80


def _section(c, x_le, camber, thickness, z0=0.0, te_cut=0.98):
    up, lo = kestrel.naca4(camber, 0.3, thickness, te_cut=te_cut)
    return [(x_le + c * u, z0 + c * v) for u, v in up] + [(x_le + c * u, z0 + c * v) for u, v in lo[::-1][1:]]


def printed_parts(variant: str) -> dict:
    """{name: {"solid", "count", "material", "orientation" (Mellonia Orientation), "print_height_axis", "note"}}: the pod
    in two halves (nose down and tail down), the wing in segments standing on a root rib with the spar bore, the fins
    (cruciform) the same, the motor mounts as cups on the spar; the carbon tubes and the bought parts are listed by
    ``stock``."""
    import cadquery as cq
    from vegeta.mellonia import Orientation
    p = {**{q.name: q.default for q in mr.MiniRaptor.parameters}, **mr.cad_params(variant)}
    L = mr.layout(variant)
    out = {}
    Lp, w, x0 = p["pod_length"], p["pod_width"], p["pod_nose_x"]
    pod = mr.parts(p)["pod"][0]
    x_split = p["wing_le_x"] + 0.30 * p["chord"]                      # the spar bulkhead
    front = pod.intersect(cq.Workplane("XY").box(x_split - x0, 4 * w, 4 * w).translate(((x0 + x_split) / 2, 0, 0)))
    rear = pod.intersect(cq.Workplane("XY").box(x0 + Lp - x_split, 4 * w, 4 * w).translate(((x_split + x0 + Lp) / 2, 0, 0)))
    out["pod, front half"] = {"solid": front, "count": 1, "material": "PLA", "orientation": Orientation(rotate_y=90), "note": "nose down on the bed, the bulkhead face up; the camera window and the battery hatch cut later"}
    out["pod, rear half"] = {"solid": rear, "count": 1, "material": "PLA", "orientation": Orientation(rotate_y=-90), "note": "tail down; the stack and the video transmitter inside"}
    c, x_le, t = p["chord"], p["wing_le_x"], p["thickness"]
    cut = te_cut_for(c, t)
    spar_x = x_le + 0.30 * c
    n_seg = max(1, math.ceil(p["span"] / SEGMENT_MAX_MM))
    seg = p["span"] / n_seg
    wing_profile = cq.Workplane("XZ").polyline(_section(c, x_le, p["camber"], t, 0.0, cut)).close()
    segment = wing_profile.extrude(seg).translate((0, seg, 0))          # from y = 0 to y = seg (XZ extrudes towards -y)
    bore = cq.Workplane("XZ").center(spar_x, 0).circle(TUBE_OD["spar"] / 2 + 0.15).extrude(seg + 2).translate((0, seg + 1, 0))
    segment = segment.cut(bore)
    wings = 2 if L["layout"] == "biplane" else 1
    n_mounts = L.get("n_motors", 4)
    out[f"wing segment ({seg:.0f} mm of {n_seg} per wing)"] = {"solid": segment, "count": n_seg * wings, "material": "LW_PLA", "orientation": Orientation(rotate_x=-90),
                                                              "note": f"stands on its root rib (span along z); trailing edge cut to {TE_MIN_MM:.1f} mm at {cut:.3f} c; the 6 mm spar tube through the bore glues the segments"}
    if L["layout"] == "cruciform":
        fc = p["fin_chord"]
        cutf = te_cut_for(fc, t)
        fin_len = p["arm"] - w / 2 + 10.0
        fin = cq.Workplane("XZ").polyline(_section(fc, x_le + 0.25 * c - 0.25 * fc, 0.0, t, 0.0, cutf)).close().extrude(fin_len).translate((0, fin_len, 0))
        fin = fin.cut(cq.Workplane("XZ").center(x_le + 0.25 * c - 0.25 * fc + 0.30 * fc, 0).circle(TUBE_OD["spar"] / 2 + 0.15).extrude(fin_len + 2).translate((0, fin_len + 1, 0)))
        out[f"fin ({fin_len:.0f} mm)"] = {"solid": fin, "count": 2, "material": "LW_PLA", "orientation": Orientation(rotate_x=-90), "note": "as the wing segments; the two fins' tubes pass through the pod's bulkhead"}
    rm = p["motor_diameter"] / 2
    cup = cq.Workplane("YZ").circle(rm + 1.6).extrude(14.0).faces(">X").workplane().hole(2 * rm - 3.0, 11.0)      # a 3 mm floor with the 16 × 16 bolt pattern, 11 mm of skirt
    boss = cq.Workplane("YZ").workplane(offset=14.0).rect(14.0, 9.0).extrude(x_le + 0.30 * c - p["prop_x"] - 14.0 + 12.0)
    mount = cup.union(boss).cut(cq.Workplane("YZ").workplane(offset=14.0).circle(TUBE_OD["spar"] / 2 + 0.15).extrude(200.0))
    out["motor mount"] = {"solid": mount, "count": n_mounts, "material": "PLA", "orientation": Orientation(rotate_y=-90), "note": "the cup takes the 2207's 16 × 16 bolt pattern (holes cut later); the strap slides on the spar tube to the station and is glued to the skin"}
    if L["layout"] == "biplane":
        fit = cq.Workplane("XY").box(16.0, 12.0, 12.0).faces(">Z").workplane().hole(TUBE_OD["post"] + 0.3, 9.0)
        out["post fitting"] = {"solid": fit, "count": 4, "material": "PLA", "orientation": Orientation(), "note": "glued to the wing at the post station, the 5 mm post tube in the hole"}
        sl = p["gap"] - 2 * 6.0
        sleeve = cq.Workplane("XY").polyline(_section(mr.POST_FAIRING_CHORD * 1e3, 0.0, 0.0, 0.35, 0.0, 0.95)).close().extrude(sl).cut(
            cq.Workplane("XY").center(0.30 * mr.POST_FAIRING_CHORD * 1e3, 0).circle(TUBE_OD["post"] / 2 + 0.2).extrude(sl))
        out[f"post fairing sleeve ({sl:.0f} mm)"] = {"solid": sleeve, "count": 2, "material": "PLA", "orientation": Orientation(), "note": "a 35 % thick streamlined sleeve printed standing, slid over the post tube"}
    return out


def stock(variant: str) -> pd.DataFrame:
    """The bought parts and the cut carbon tubes."""
    L = mr.layout(variant)
    pk = mr.PACKS[L["pack"]]
    nm = L.get("n_motors", 4)
    rows = [("motor 2207-class", nm, f"{mr.STACK['motor']['mass_kg'] * 1e3:.0f} g each", "off the shelf" + ("; two of the stack's four channels stay free" if nm == 2 else "")),
            (f"propeller 5x{mr.DRIVES[L['drive']]['pitch_in']:g} tri-blade", nm, "4 g each", "off the shelf"),
            ("4-in-1 ESC + flight controller", 1, "30 g", "off the shelf; a tailsitter firmware (ArduPilot / INAV)"),
            ("FPV camera + VTX + antenna", 1, "35 g", "off the shelf"), ("receiver + GNSS", 1, "12 g", "off the shelf"),
            (f"pack {L['pack']}", 1, f"{pk['mass_kg'] * 1e3:.0f} g", "off the shelf"),
            (f"carbon tube {TUBE_OD['spar']:.0f} mm, wing spar", 2 if L["layout"] == "biplane" else 1, f"{L['span'] * 1e3:.0f} mm", "cut from stock")]
    if L["layout"] == "biplane":
        rows.append((f"carbon tube {TUBE_OD['post']:.0f} mm, interplane post", 2, f"{L['gap'] * 1e3:.0f} mm", "cut from stock"))
    elif L["layout"] == "cruciform":
        rows.append((f"carbon tube {TUBE_OD['spar']:.0f} mm, fin spar", 1, f"{2 * L['arm'] * 1e3:.0f} mm", "cut from stock, through the pod"))
    rows.append((f"carbon rod {TUBE_OD['foot']:.0f} mm, feet", 4 if L["layout"] != "monoplane" else 2 + nm, f"{(L['x_feet'] - L['x_te']) * 1e3 + 10:.0f} mm", "cut from stock"))
    return pd.DataFrame(rows, columns=["item", "count", "size / mass", "source"])


def _oriented(solid, orientation):
    """The solid turned as PrusaSlicer will turn it (rotate-x, then y, then z, degrees), then set on z = 0."""
    s = solid
    if orientation.rotate_x:
        s = s.rotate((0, 0, 0), (1, 0, 0), orientation.rotate_x)
    if orientation.rotate_y:
        s = s.rotate((0, 0, 0), (0, 1, 0), orientation.rotate_y)
    if orientation.rotate_z:
        s = s.rotate((0, 0, 0), (0, 0, 1), orientation.rotate_z)
    bb = s.val().BoundingBox()
    return s.translate((-bb.xmin, -bb.ymin, -bb.zmin))


def check_part(name: str, part: dict, *, bed=BED, overhang_deg: float = OVERHANG_DEG) -> dict:
    """Bed fit in the orientation, the overhang area that would want support, the shell + infill mass estimate."""
    s = _oriented(part["solid"], part["orientation"])
    bb = s.val().BoundingBox()
    dims = (bb.xlen, bb.ylen, bb.zlen)
    fits_flat = (dims[0] <= bed[0] and dims[1] <= bed[1]) or (dims[1] <= bed[0] and dims[0] <= bed[1])
    fits_diag = math.hypot(dims[0], dims[1]) <= math.hypot(*bed[:2]) and min(dims[:2]) <= min(bed[:2])
    verts, tris = s.val().tessellate(0.5, 0.3)
    V = np.array([v.toTuple() for v in verts])
    F = V[np.array(tris)]
    n = np.cross(F[:, 1] - F[:, 0], F[:, 2] - F[:, 0])
    area = 0.5 * np.linalg.norm(n, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        nz = n[:, 2] / np.where(np.linalg.norm(n, axis=1) > 0, np.linalg.norm(n, axis=1), 1.0)
    zc = F[:, :, 2].mean(1)
    down = (nz < -math.sin(math.radians(overhang_deg))) & (zc > 0.5)         # facing down, steeper than the limit, not the bed face
    overhang = float(area[down].sum() / max(area.sum(), 1e-9))
    vol_cm3 = s.val().Volume() / 1e3
    surf_cm2 = s.val().Area() / 1e2
    mat = part["material"]
    shell_cm3 = min(vol_cm3, surf_cm2 * WALL_MM[mat] / 10)
    mass_g = DENSITY[mat] * (shell_cm3 + INFILL[mat] * max(vol_cm3 - shell_cm3, 0.0))
    return {"part": name, "count": part["count"], "material": mat, "orientation": f"x{part['orientation'].rotate_x:g} y{part['orientation'].rotate_y:g} z{part['orientation'].rotate_z:g}",
            "size on the bed [mm]": f"{dims[0]:.0f} × {dims[1]:.0f} × {dims[2]:.0f}", "fits the bed": bool(fits_flat or fits_diag) and dims[2] <= bed[2], "height ok": dims[2] <= bed[2],
            "overhang area wanting support [%]": 100 * overhang, "volume [cm³]": vol_cm3, "surface [cm²]": surf_cm2, "printed mass each [g]": mass_g, "printed mass all [g]": mass_g * part["count"], "note": part["note"]}


def check_all(variant: str, parts: dict | None = None) -> pd.DataFrame:
    parts = parts or printed_parts(variant)
    return pd.DataFrame([check_part(n, p) for n, p in parts.items()]).set_index("part")


def structure_against_budget(variant: str, checks: pd.DataFrame) -> dict:
    """The printed structure (+ the tubes) against what the mass budget assumed for the structure."""
    L = mr.layout(variant)
    tubes = (2 if L["layout"] == "biplane" else 1) * L["span"] * mr.SPAR_KG_M * 1e3 + (2 * L["gap"] * 0.014 * 1e3 if L["layout"] == "biplane" else 2 * L["arm"] * mr.SPAR_KG_M * 1e3 if L["layout"] == "cruciform" else 0.0) \
        + 4 * (L["x_feet"] - L["x_te"] + 0.01) * 0.010 * 1e3                                  # 6 mm tube 20 g/m, 5 mm 14 g/m, 4 mm rod 10 g/m
    printed = float(checks["printed mass all [g]"].sum())
    b = mr.budget(L)
    assumed = float(b.loc[b["group"] == "structure", "mass_g"].sum())
    return {"printed parts [g]": printed, "carbon tubes [g]": tubes, "structure as built [g]": printed + tubes, "structure in the budget [g]": assumed,
            "difference [g]": printed + tubes - assumed, "verdict": "the budget holds" if printed + tubes <= assumed * 1.1 else "the budget is light: carry the built mass into the next pass"}


def slicer_available(executable: str = "prusa-slicer") -> str | None:
    return shutil.which(executable) or shutil.which("PrusaSlicer") or shutil.which("prusa-slicer-console")


def slice_all(variant: str, workdir, parts: dict | None = None, *, executable: str | None = None) -> pd.DataFrame:
    """Mellonia on every printed part (one STL each, the orientation given) when PrusaSlicer is on the PATH; else a
    NOT RUN table with the command lines to run elsewhere."""
    from vegeta import mellonia
    parts = parts or printed_parts(variant)
    sets = settings()
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    exe = executable or slicer_available()
    rows = []
    tag = variant.lower().replace(" ", "_")
    for name, part in parts.items():
        stem = f"{tag}_{name.split(' (')[0].replace(' ', '_').replace(',', '')}"
        stl = workdir / f"{stem}.stl"
        part["solid"].val().exportStl(str(stl), 0.05, 0.2)
        o = part["orientation"]
        cmd = f"mellonia slice {stl.name} -s designs/mini_raptor_print.py:{part['material']} --rotate-x {o.rotate_x:g} --rotate-y {o.rotate_y:g} -o runs/{stem}"
        if exe is None:
            rows.append({"part": name, "status": "NOT RUN — PrusaSlicer not on the PATH", "command": cmd})
            continue
        res = mellonia.slice_stl(stl, sets[part["material"]], o, workdir / stem, executable=exe)
        m = res.metrics or {}
        rows.append({"part": name, "status": res.status, "estimated time [min]": (m.get("estimated_time_s") or 0) / 60, "filament [g]": m.get("filament_used_g"), "layers": m.get("layer_count"),
                     "max z [mm]": m.get("max_z"), "count": part["count"], "note": res.messages[-1][:120] if res.messages else "", "command": cmd})
    return pd.DataFrame(rows).set_index("part")


PLA = None
LW_PLA = None
try:                                                       # the CLI's `-s designs/mini_raptor_print.py:PLA`
    PLA, LW_PLA = settings()["PLA"], settings()["LW_PLA"]
except Exception:                                          # Mellonia missing: the checks still run
    pass
