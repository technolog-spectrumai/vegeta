"""K4TQ — the first CAD (Stage 2 preliminary): the Stage 1 configuration as CadQuery solids, read from ``k4tq``.

Everything is positioned with ``k4tq.wing_point`` / ``k4tq.prop_pose`` — the same transforms the sizing used — so the
CAD and the numbers cannot drift apart. ``assembly(p, tilt_deg, fold_deg)`` returns named solids (millimetres);
``render`` draws the four views of the brief (A cruise, B hover, C folded, D wing-root detail) headless with
matplotlib from the tessellation; ``export`` writes STEP and STL.

Preliminary: the wing-root cartridge is drawn as its envelope (bearing housing, servo, fold hinge, locking pin) — the
mechanism itself is Stage 2.3 / 2.4. The fin is the Stage 1 directional-stability area split dorsal / ventral.
"""
from __future__ import annotations

import math
from pathlib import Path

import cadquery as cq
import numpy as np

import k4tq
from kestrel import naca4

MM = 1000.0
COLORS = {"fuselage": "#f4f5f7", "hatch": "#e4e7ec", "wing": "#d9dde3", "root": "#9aa3ad", "pin": "#5b6470", "nacelle": "#6b7380",
          "disc": "#7fb3e6", "leg": "#4a5160", "fin": "#d0d4da", "cartridge": "#c0392b", "light": "#2f80ed"}


def _v(p) -> cq.Vector:
    return cq.Vector(*(float(c) * MM for c in p))


def _section(designation: str):
    d = designation.split()[-1]
    up, lo = naca4(int(d[0]) / 100, int(d[1]) / 10, int(d[2:]) / 100, n=24)
    return up + lo[::-1][1:]                                    # TE → LE (upper) → TE (lower)


def _wing_solid(L: dict, which: str, side: int, tilt: float, fold: float, y_out: float | None = None) -> cq.Solid:
    """A panel lofted from the root rib to the tip rib, every point through ``k4tq.wing_point``'s transform (so the tilt
    turns it about the spar and the fold swings it about the hinge)."""
    w = L["wings"][which]
    sec = _section(L["p"]["section"])
    wires = []
    for y in (w["y0"], y_out or w["y1"]):
        c = w["c_at"](y)
        pts = []
        for xi, zc in sec:
            base = k4tq.wing_point(L, which, side, y, xi, tilt, fold)
            # the section's thickness/camber is normal to the chord: rotate it with the tilt (and the fold leaves z alone)
            n = np.array([math.sin(tilt) * math.cos(fold), side * -math.sin(tilt) * math.sin(fold) * 0.0, math.cos(tilt)])
            pts.append(_v(base + zc * c * n))
        wires.append(cq.Wire.makePolygon(pts, close=True))
    return cq.Solid.makeLoft(wires, True)


def _cyl(r, h, base, direction) -> cq.Solid:
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    return cq.Solid.makeCylinder(r * MM, h * MM, _v(base), cq.Vector(*d))


def _fuselage(p: dict) -> cq.Solid:
    Lf, bw, bh = p["fuselage_length"], p["fuselage_width"], p["fuselage_height"]
    stations = [(0.0, 0.04), (0.04, 0.45), (0.12, 0.85), (0.25, 1.0), (0.60, 1.0), (0.85, 0.70), (1.0, 0.18)]
    wires = []
    for f, s in stations:
        a, b = max(bw / 2 * s, 0.003), max(bh / 2 * s, 0.003)
        x = f * Lf
        pts = [_v((x, a * math.cos(t), b * math.sin(t) + 0.01 * (f - 0.5) * 0)) for t in np.linspace(0, 2 * math.pi, 28, endpoint=False)]
        wires.append(cq.Wire.makePolygon(pts, close=True))
    return cq.Solid.makeLoft(wires, False)


def assembly(p: dict | None = None, tilt_deg: float = 0.0, fold_deg: float = 0.0, *, discs: bool = True) -> dict:
    """Named solids [mm] at one tilt and fold (both pairs alike)."""
    p = p or k4tq.params()
    L = k4tq.layout(p)
    tilt, fold = math.radians(tilt_deg), math.radians(fold_deg)
    parts = {"fuselage": _fuselage(p)}
    bh = p["fuselage_height"]
    # cargo hatch outline (a thin plate on the top, the marking text goes on it in the renders)
    xc = p["cargo_station"] * L["fuselage_length"]
    parts["hatch"] = cq.Solid.makeBox(0.24 * MM, 0.09 * MM, 0.004 * MM, _v((xc - 0.12, -0.045, bh / 2 - 0.006)))
    for which in ("front", "rear"):
        w = L["wings"][which]
        for side in (-1, 1):
            s = "L" if side < 0 else "R"
            parts[f"wing {which} {s}"] = _wing_solid(L, which, side, tilt, fold)
            # root: fold hinge (vertical pin) at y_hinge, the tilt cartridge from the hinge to the first rib
            hinge = np.array([w["x_pivot"], side * w["y_hinge"], w["z"]])
            parts[f"hinge pin {which} {s}"] = _cyl(0.004, 0.06, hinge - [0, 0, 0.03], (0, 0, 1))
            yaxis = np.array([math.sin(fold) * 1.0, side * math.cos(fold), 0.0])
            parts[f"root cartridge {which} {s}"] = _cyl(0.021, p["root_length"], hinge, yaxis)
            pose = k4tq.prop_pose(L, which, side, tilt, fold)
            ax = pose["axis"]
            parts[f"nacelle {which} {s}"] = _cyl(0.022, 0.15, pose["hub"] - 0.15 * ax * 0.0 - 0.14 * ax, ax)
            parts[f"spinner {which} {s}"] = cq.Solid.makeSphere(0.016 * MM, _v(pose["hub"]))
            if discs and fold_deg == 0.0:
                parts[f"disc {which} {s}"] = _cyl(L["prop"]["R"], 0.003, pose["hub"] + 0.01 * ax, ax)
    # legs (the hover stance), fin (dorsal + ventral)
    for which in ("front", "rear"):
        x = L["wings"][which]["x_pivot"]
        for side in (-1, 1):
            parts[f"leg {which} {'L' if side < 0 else 'R'}"] = _cyl(0.006, -p["leg_bottom_z"] - bh / 2 + 0.01, (x + 0.03, side * 0.055, p["leg_bottom_z"]), (0, 0, 1))
    S_fin = p.get("fin_area") or 0.05
    h = math.sqrt(S_fin / 2 / 1.0)                               # each half: chord ≈ height
    xt = L["fuselage_length"]
    for zs, name in ((1, "fin dorsal"), (-1, "fin ventral")):
        pts = [_v((xt - 0.20, 0, zs * 0.03)), _v((xt - 0.02, 0, zs * 0.02)), _v((xt - 0.01, 0, zs * (0.02 + h))), _v((xt - 0.11, 0, zs * (0.02 + h)))]
        face = cq.Face.makeFromWires(cq.Wire.makePolygon(pts, close=True))
        parts[name] = cq.Solid.extrudeLinear(face, cq.Vector(0, 0.006 * MM, 0)).translate(cq.Vector(0, -0.003 * MM, 0))
    return parts


def group(name: str) -> str:
    for k, g in (("frame", "fuselage"), ("lug", "pin"), ("bearing", "pin"), ("servo", "leg"), ("spar", "nacelle"), ("housing", "root"),
                 ("fuselage", "fuselage"), ("hatch", "hatch"), ("wing", "wing"), ("root", "root"), ("pin", "pin"), ("nacelle", "nacelle"),
                 ("spinner", "nacelle"), ("disc", "disc"), ("leg", "leg"), ("fin", "fin")):
        if k in name:
            return g
    return "fuselage"


def export(parts: dict, out: Path, stem: str) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    comp = cq.Compound.makeCompound([s for k, s in parts.items() if not k.startswith("disc")])
    step, stl = out / f"{stem}.step", out / f"{stem}.stl"
    cq.exporters.export(cq.Workplane().add(comp), str(step))
    cq.exporters.export(cq.Workplane().add(comp), str(stl), tolerance=0.5, angularTolerance=0.2)
    return {"step": step, "stl": stl}


def _draw(ax, parts: dict, *, elev: float, azim: float, tol: float = 1.0, title: str = "", lim=None, alpha_disc=0.18, alpha: dict | None = None):
    from matplotlib import colors as matplotlib_colors
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    allv = []
    for name, s in parts.items():
        v, t = s.tessellate(tol, 0.3)
        V = np.array([(q.x, q.y, q.z) for q in v])
        if not len(t):
            continue
        g = group(name)
        tris = V[np.array(t)]
        disc = g == "disc"
        rgb = np.array(matplotlib_colors.to_rgb(COLORS[g]))
        nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        light = np.array([-0.4, -0.5, 0.75]) / np.linalg.norm([-0.4, -0.5, 0.75])
        k = 0.55 + 0.45 * np.abs(nrm @ light)
        fc = np.clip(rgb[None] * k[:, None], 0, 1)
        al = alpha_disc if disc else (alpha or {}).get(name, 1.0)
        col = Poly3DCollection(tris, facecolors=fc, edgecolors=fc, linewidths=0.05, alpha=al)
        ax.add_collection3d(col)
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


def root_detail(p: dict, L: dict, tilt_deg: float) -> tuple[dict, np.ndarray]:
    """The front-right wing root as separate solids for the close-up (envelopes of the Stage 2.3 / 2.4 parts)."""
    w = L["wings"]["front"]
    tilt = math.radians(tilt_deg)
    yh, y0 = w["y_hinge"], w["y0"]
    x, z = w["x_pivot"], w["z"]
    yf = p["fuselage_width"] / 2
    parts = {"fuselage side frame": cq.Solid.makeBox(0.17 * MM, 0.004 * MM, 0.13 * MM, _v((x - 0.085, yf - 0.010, z - 0.065)))}
    parts["hinge lug lower (fuselage)"] = cq.Solid.makeBox(0.030 * MM, 0.016 * MM, 0.010 * MM, _v((x - 0.015, yf - 0.006, z - 0.035)))
    parts["hinge lug upper (fuselage)"] = cq.Solid.makeBox(0.030 * MM, 0.016 * MM, 0.010 * MM, _v((x - 0.015, yf - 0.006, z + 0.025)))
    parts["hinge pin"] = _cyl(0.004, 0.075, (x, yh, z - 0.0375), (0, 0, 1))
    parts["locking pin"] = _cyl(0.003, 0.030, (x + 0.030, yh - 0.015, z), (0, 1, 0))
    parts["root housing"] = _cyl(0.021, p["root_length"] - 0.004, (x, yh + 0.004, z), (0, 1, 0))
    for i, yb in enumerate((yh + 0.008, yh + 0.008 + 0.030)):
        parts[f"bearing {i + 1}"] = _cyl(0.0165, 0.007, (x, yb, z), (0, 1, 0))
    parts["servo (inside the fuselage)"] = cq.Solid.makeBox(0.040 * MM, 0.020 * MM, 0.038 * MM, _v((x - 0.055, yf - 0.032, z - 0.019)))
    parts["spar"] = _cyl(0.008, (y0 + 0.14) - (yh + 0.002), (x, yh + 0.002, z), (0, 1, 0))
    parts["wing stub"] = _wing_solid(L, "front", 1, tilt, 0.0, y_out=y0 + 0.14)
    return parts, np.array([x, yh + 0.04, z]) * MM


def render(path: Path, p: dict | None = None, *, info: dict | None = None) -> Path:
    """The four labelled views of the brief in one PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = p or k4tq.params()
    L = k4tq.layout(p)
    fig = plt.figure(figsize=(16, 11), dpi=120)
    fig.patch.set_facecolor("white")
    cr = assembly(p, 0.0, 0.0)
    hv = assembly(p, 90.0, 0.0)
    fd = assembly(p, 90.0, 90.0, discs=False)
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    _draw(ax, cr, elev=22, azim=-135, title="A — forward cruise (tilt 0°): chords horizontal, propellers forward")
    xc = p["cargo_station"] * L["fuselage_length"] * MM
    ax.text(xc, 0, p["fuselage_height"] / 2 * MM + 25, "BE CAREFUL – FRAGILE –\nTEMPERATURE SENSITIVE LOAD", color="#c0392b", fontsize=6.5, ha="center")
    ax = fig.add_subplot(2, 2, 2, projection="3d")
    _draw(ax, hv, elev=22, azim=-135, title="B — vertical hover (tilt 90°): chords vertical, discs horizontal, fuselage level")
    ax = fig.add_subplot(2, 2, 3, projection="3d")
    _draw(ax, fd, elev=28, azim=-120, title="C — folded transport (tilt 90°, fold 90°, blades folded)")
    # D: the front-right root, close up: a wing stub, the spar, the two bearings, the servo, the fold hinge
    ax = fig.add_subplot(2, 2, 4, projection="3d")
    sel, c = root_detail(p, L, 35.0)
    lim = (c - np.array([110, 30, 80]), c + np.array([110, 150, 80]))
    _draw(ax, sel, elev=30, azim=-35, title="D — wing root (front right, tilt 35°): fold hinge, tilt cartridge, spar", lim=lim,
          alpha={"wing stub": 0.35, "fuselage side frame": 0.25})
    w = L["wings"]["front"]
    y = np.array([w["y_hinge"] - 0.005, w["y0"] + 0.13]) * MM
    ax.plot([c[0]] * 2, y, [c[2]] * 2, color="#c0392b", lw=1.6, ls="--")
    ax.text(c[0] + 10, y[1], c[2] + 30, "tilt axis = spar Ø16\n0° cruise … 90° hover", color="#c0392b", fontsize=8)
    ax.text(c[0] - 70, w["y_hinge"] * MM, c[2] + 75, "fold hinge: vertical pin\n+ Ø6 locking pin", color="#3d4450", fontsize=8)
    ax.text(c[0] + 60, w["y_hinge"] * MM + 30, c[2] - 80, "6804 bearings ×2 (50 mm apart)\nservo 30 kg·cm + 2:1 gears", color="#3d4450", fontsize=8)
    info = info or {}
    txt = (f"K4TQ-{p['variant']} (Stage 1 configuration, preliminary CAD) — span {p['span'] * 1e3:.0f} mm, fuselage {p['fuselage_length'] * 1e3:.0f} mm, "
           f"overall width with discs {L['width_overall'] * 1e3:.0f} mm, props {p['prop']} on {k4tq.CATALOGUE['motor'][p['motor']]['name']}"
           + (f"\nmass {info.get('mass', 0):.2f} kg, payload {p['payload_kg']:.1f} kg; hatch marking: \"{k4tq.STANDARD_MARKING}\"" if info else
              f"\nhatch marking: \"{k4tq.STANDARD_MARKING}\""))
    fig.text(0.01, 0.01, txt, fontsize=10, family="monospace")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "data" / "k4tq"
    p = k4tq.params()
    print(render(out / "figures" / "k4tq_first_cad.png", p))
