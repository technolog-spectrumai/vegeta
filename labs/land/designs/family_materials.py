"""Materials and making of the Pekari family's parts — the table behind the materials notebook (notebook 41) and a
colour-coded CAD view of each machine.

``MATERIALS`` is the palette: each material with its colour and how the part is made (buy, machine, weld, cut,
mould, 3D print). ``parts(cls, platform)`` rebuilds a family machine's CAD (``pekari_2m._FamilyCAD``: Pekari-2M,
Pumba, Timon, Calydon) as separate solids, each tagged with its material by the rules of ``RULES`` — so a part's
colour says what it is made of. ``figure`` draws a machine (side and three-quarter views) with the legend.

The rules (from the loads the family's models give: the arm tubes see 7-13 MPa at their ratings x1.5, the heavy
machines' sprockets ~2 kN a tooth, a heavy road wheel ~0.8 kN with a 2.5 dynamic factor):

* steel where the load is concentrated, wears or strikes: pins, axles, heavy sprockets, the grab's jaws, the
  stabilizer legs and the spade, Calydon's subframe;
* aluminium where mass counts and the stress is low: hull sheet (5754), frames and arm tubes (6082-T6), machined
  flanges and pedestals (7075 / 6082);
* plastic for wheels, covers and light parts: PA6-GF30 moulded (production wheels, Pekari-2M's links and sprocket),
  printed PA-CF (prototype wheels, brackets), ASA (outdoor covers, the sensor box), TPU (pads, bumpers);
* bought: the rubber band tracks, joint modules, the saw.
"""
from __future__ import annotations

import math

try:
    import cadquery as cq
except ImportError:
    cq = None

import pekari_arms as pa

__all__ = ["MATERIALS", "RULES", "parts", "figure", "table"]

#: material -> (colour, how it is made, examples)
MATERIALS = {
    "Al 5754 sheet": ("#b8c4cc", "laser-cut and bent sheet", "hull shell, deck"),
    "Al 6082-T6 tube / extrusion": ("#8fa6b8", "cut and welded or bolted", "frames, arm tubes"),
    "Al 7075 / 6082 machined": ("#5f7d95", "CNC machined", "joint housings, pedestals, flanges"),
    "steel S355 / 42CrMo4": ("#4a4a4a", "laser-cut, welded, turned", "pins, axles, heavy sprockets, legs, subframe"),
    "Hardox wear plate": ("#8b2e2e", "laser-cut", "grab jaw tips, the spade"),
    "PA6-GF30 moulded": ("#e0b040", "injection moulded (production)", "road wheels, idlers, links, light sprockets"),
    "PA-CF 3D printed": ("#3f8f4f", "3D printed (SLS PA12 or FDM PA-CF)", "prototype wheels, brackets, mounts"),
    "ASA 3D printed": ("#e8e2d0", "3D printed, UV-stable", "sensor box, covers"),
    "TPU 3D printed": ("#c060a0", "3D printed, flexible", "jaw pads, bumpers"),
    "rubber band track (bought)": ("#1e1e1e", "bought", "Pumba's and Timon's tracks"),
    "bought module": ("#2f6fb0", "bought", "joint modules, the saw, drives, cylinders"),
}

#: The rules per machine size: what the track and the sprocket are.
HEAVY = ("Pumba", "Timon", "Calydon")
RULES = {
    "hull": "Al 5754 sheet", "deck": "Al 5754 sheet", "sensor box": "ASA 3D printed",
    "track frame": "Al 6082-T6 tube / extrusion", "axle": "steel S355 / 42CrMo4",
    "road wheel": "PA6-GF30 moulded", "idler": "PA6-GF30 moulded",
    "sprocket (light)": "PA6-GF30 moulded", "sprocket (heavy)": "steel S355 / 42CrMo4",
    "track (links)": "PA6-GF30 moulded", "track (band)": "rubber band track (bought)",
    "pedestal": "Al 7075 / 6082 machined", "arm tube": "Al 6082-T6 tube / extrusion",
    "joint housing": "bought module", "palm": "PA-CF 3D printed", "jaw": "steel S355 / 42CrMo4",
    "saw": "bought module", "stabilizer leg": "steel S355 / 42CrMo4", "stabilizer pad": "Hardox wear plate",
}


def _track_parts(d, p, platform):
    """The track module's parts by material, in the module's frame (as ``PekariRover._track_module``)."""
    out = []
    b, ft = p["track_width"], p["frame_thickness"]
    y_frame = -(b / 2 + 10.0 + ft / 2)
    zr = d.road_wheel_z(p)
    xs = d.road_wheel_x(p)
    x0, x1 = min(p["sprocket_x"], p["idler_x"]), max(p["sprocket_x"], p["idler_x"])
    z_lo, z_hi = zr + 12.0, min(p["return_roller_z"], p["sprocket_z"], p["idler_z"]) + 20.0
    frame = (cq.Workplane("XZ").center((x0 + x1) / 2, (z_lo + z_hi) / 2).rect(x1 - x0, z_hi - z_lo)
             .extrude(ft / 2, both=True).translate((0, y_frame, 0)))
    out.append(("track frame", frame))
    for x, z, dia in [(p["idler_x"], p["idler_z"], p["idler_diameter"]), (0.0, p["return_roller_z"], p["return_roller_diameter"])]:
        out.append(("idler", d._grooved_wheel(p, dia, p["axle_diameter"]).translate((x, 0, z))))
    for x in xs:
        out.append(("road wheel", d._grooved_wheel(p, p["road_wheel_diameter"], p["axle_diameter"]).translate((x, 0, zr))))
    heavy = platform.name in HEAVY
    out.append(("sprocket (heavy)" if heavy else "sprocket (light)", d._sprocket(p).translate((p["sprocket_x"], 0, p["sprocket_z"]))))
    for x, z in [(p["idler_x"], p["idler_z"]), (p["sprocket_x"], p["sprocket_z"])]:
        out.append(("axle", cq.Workplane("XZ").center(x, z).circle(p["axle_diameter"] / 2)
                    .extrude((abs(y_frame) + ft / 2 + 2.0) / 2, both=True).translate((0, y_frame / 2, 0))))
    out.append(("track (band)" if platform.band_track_kg_m else "track (links)", d._belt(p)))
    return out


def _arm_parts(arm, q):
    """An arm's parts by material, shoulder at the origin [mm] (as ``pekari_2m._FamilyCAD._arm``)."""
    k = 1000.0
    uw, ud = (v * k for v in arm.upper_section)
    fw, fd = (v * k for v in arm.fore_section)
    e1, e2 = -q["shoulder"], -(q["shoulder"] + q["elbow"])
    e3 = -(q["shoulder"] + q["elbow"] + q["wrist_pitch"])
    Lu, Lf, Lt = arm.upper * k, arm.fore * k, arm.tool * k
    ped_h, ped_r = arm.pedestal[0] * k, arm.pedestal[1] * k
    out = [("pedestal", cq.Workplane("XY").circle(ped_r).extrude(ped_h - 0.5 * ud).translate((0, 0, -ped_h)))]

    def tube(L, w, dd):
        return cq.Workplane("XY").box(L, w, dd).translate((L / 2, 0, 0))

    def boss(r, w):
        return cq.Workplane("XZ").circle(r).extrude(w, both=True)
    out.append(("arm tube", tube(Lu, uw, ud).rotate((0, 0, 0), (0, 1, 0), -math.degrees(e1))))
    out.append(("joint housing", boss(0.55 * ud, 0.6 * uw)))
    ex, ez = Lu * math.cos(e1), Lu * math.sin(e1)
    out.append(("arm tube", tube(Lf, fw, fd).rotate((0, 0, 0), (0, 1, 0), -math.degrees(e2)).translate((ex, 0, ez))))
    out.append(("joint housing", boss(0.55 * fd, 0.6 * fw).translate((ex, 0, ez))))
    wx, wz = ex + Lf * math.cos(e2), ez + Lf * math.sin(e2)
    t_mod = pa.flange(arm) * k
    tool = [("joint housing", boss(0.55 * fd, 0.55 * fw)),
            ("joint housing", cq.Workplane("YZ").circle(0.5 * fw).extrude(t_mod))]
    if arm.saw:
        from ogma import cutting
        bar = cutting.get_saw(arm.saw).bar_length * k
        body = max(Lt - t_mod - bar / 2, 50.0)
        tool.append(("saw", cq.Workplane("XY").box(body, max(1.2 * fw, 60), max(1.6 * fd, 80)).translate((t_mod + body / 2, 0, 0))))
        tool.append(("jaw", cq.Workplane("XZ").center(t_mod + body + bar / 2, 0).rect(bar, 70.0).extrude(2.5, both=True)))
    else:
        palm = max(Lt - arm.jaw_length * k - t_mod, 20.0)
        tool.append(("palm", cq.Workplane("XY").box(palm, 1.2 * fw, 1.2 * fd).translate((t_mod + palm / 2, 0, 0))))
        jl, jw, jd = arm.jaw_length * k, max(0.6 * fw, 20.0), max(0.25 * fd, 12.0)
        for s in (1, -1):
            tool.append(("jaw", cq.Workplane("XY").box(jl, jw, jd).translate((t_mod + palm + jl / 2, 0, s * (0.3 * fd + jd / 2)))))
    for name, sh in tool:
        out.append((name, sh.rotate((0, 0, 0), (0, 1, 0), -math.degrees(e3)).translate((wx, 0, wz))))
    return [(n, s.rotate((0, 0, 0), (0, 0, 1), math.degrees(q["yaw"]))) for n, s in out]


def parts(cls, pose: str = "deployed", stabilizers: str = "deployed", tool: str = "default") -> list:
    """[(rule, material, solid)] of a family machine (its CAD class) in mm, vehicle frame."""
    d = cls()
    p = d.resolve(arm_pose=pose, stabilizers=stabilizers, tool=tool)
    pl = d.platform()
    if tool != "default":
        import pekari_family as pf
        pl = pf.with_tool(pl, pl.mounts[0].name, tool)
    out = []
    hull = d._hull(p)
    out.append(("hull", hull))
    if d.has_basket(p):
        out.append(("deck", d._basket(p).translate((p["basket_x"], 0, p["ground_clearance"] + p["hull_height"]))))
    yc = d.track_center_y(p)
    for name, sh in _track_parts(d, p, pl):
        out.append((name, sh.translate((0, yc, 0))))
        out.append((name, sh.mirror("XZ").translate((0, -yc, 0))))
    roof = p["ground_clearance"] + p["hull_height"]
    for m in pl.mounts:
        q = pa.pose(m.arm, pose) | {"wrist_yaw": 0.0}
        for name, sh in _arm_parts(m.arm, q):
            out.append((name, sh.translate((m.x * 1000, m.y * 1000, roof + m.arm.pedestal[0] * 1000))))
    if stabilizers != "none":
        for s in pl.stabilizers:
            leg = d._stabilizer(s, p, stabilizers == "deployed")
            out.append(("stabilizer leg", leg))
    return [(n, RULES[n], sh) for n, sh in out]


def _tris(shape):
    import numpy as np
    shp = shape.val() if hasattr(shape, "val") else shape
    v, t = shp.tessellate(2.0, 0.5)
    v = np.array([(a.x, a.y, a.z) for a in v]) / 1000.0
    return v[np.array(t)]


def figure(machines, path=None):
    """Each machine (``[(label, cls, kwargs)]``) colour-coded by material: a side view and a three-quarter view,
    one row each, the legend of the materials used."""
    import matplotlib
    if "inline" not in matplotlib.get_backend().lower():         # a script: draw off screen; a notebook: inline
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(len(machines), 2, figsize=(15, 3.6 * len(machines)), squeeze=False)
    used = set()
    light = np.array([0.35, -0.75, 0.55]) / np.linalg.norm([0.35, -0.75, 0.55])
    for row, (label, cls, kw) in enumerate(machines):
        ps = parts(cls, **kw)
        for col, (az, el) in enumerate(((0.0, 0.0), (-35.0, 22.0))):
            ax = axes[row][col]
            a, e = math.radians(az), math.radians(el)
            R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
            E = np.array([[1, 0, 0], [0, math.cos(e), -math.sin(e)], [0, math.sin(e), math.cos(e)]])
            M = E @ R
            polys, cols, depth = [], [], []
            for _, mat, sh in ps:
                tri = _tris(sh) @ M.T                       # view: x right, z up, y into the page
                nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
                nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
                shade = 0.5 + 0.5 * np.clip(np.abs(nrm @ (M @ light)), 0, 1)
                base = np.array(matplotlib.colors.to_rgb(MATERIALS[mat][0]))
                polys.append(tri[:, :, [0, 2]])
                cols.append(np.clip(np.outer(shade, base), 0, 1))
                depth.append(tri[:, :, 1].mean(axis=1))
                used.add(mat)
            P, C, D = np.concatenate(polys), np.concatenate(cols), np.concatenate(depth)
            order = np.argsort(D)[::-1]                     # far first
            ax.add_collection(PolyCollection(P[order], facecolors=C[order], linewidths=0))
            ax.autoscale()
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(f"{label} — {'side' if col == 0 else 'three-quarter'}", fontsize=10)
    handles = [Patch(color=MATERIALS[m][0], label=f"{m}: {MATERIALS[m][1]}") for m in MATERIALS if m in used]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.05 + 0.01 * len(handles) / 3, 1, 1))
    if path:
        fig.savefig(path, dpi=110)
    return fig


def table():
    """The rules as a DataFrame: part, material, how it is made, the colour."""
    import pandas as pd
    return pd.DataFrame([{"part": k, "material": m, "made": MATERIALS[m][1], "colour": MATERIALS[m][0]} for k, m in RULES.items()])
