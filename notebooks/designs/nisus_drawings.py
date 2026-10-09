"""NISUS drawings from the same parameters the CAD and the analyses use (notebook 31): the dimensioned three-view,
the propeller's swept disk against the booms, the wing and the ground, the internal component placement of each
variant, and the exploded assembly (pyvista, off screen). Matplotlib figures; ``save`` writes PNGs."""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

import nisus
import nisus_systems as ns

GROUP_COLOR = {"wing/reinforcement/covering": "#9ecae1", "pod": "#d9d9d9", "booms/tail": "#525252", "joints/mounts/adhesive": "#fdae6b",
               "propulsion/ESC/wiring": "#e6550d", "battery/retention": "#fdd0a2", "servos/linkages": "#31a354",
               "flight controller/receiver/GNSS/wiring": "#3182bd", "camera/video": "#756bb1", "computer": "#de2d26", "construction allowance": "#bdbdbd"}


def _dim(ax, a, b, text, offset=(0, 0), color="k", fs=8):
    a, b = np.asarray(a, float), np.asarray(b, float)
    o = np.asarray(offset, float)
    ax.annotate("", xy=b + o, xytext=a + o, arrowprops=dict(arrowstyle="<->", color=color, lw=0.8))
    m = 0.5 * (a + b) + o
    ax.text(m[0], m[1], text, ha="center", va="center", fontsize=fs, color=color, bbox=dict(fc="white", ec="none", pad=0.5))
    for q in (a, b):
        if np.any(o):
            ax.plot([q[0], q[0] + o[0]], [q[1], q[1] + o[1]], color=color, lw=0.4, ls=":")


def three_view(p=None, figsize=(15, 11)):
    """Top, side and front views [mm] with the main dimensions; the propeller disc dashed, the control-surface hinge
    lines dotted, the ground line at rest in the side view."""
    p = nisus.resolve(p)
    o = nisus.outline(p)
    L = o["layout"]
    k = 1000.0
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.25], height_ratios=[1.5, 1.0])
    top, front, side = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    # --- top view: x right (aft), y up (right wing up)
    T = o["top"]
    for q in T["wing"]:
        top.fill(q[:, 0] * k, q[:, 1] * k, fc="#deebf7", ec="k", lw=0.7)
    for q in T["tail"]:
        top.fill(q[:, 0] * k, q[:, 1] * k, fc="#deebf7", ec="k", lw=0.7)
    top.fill(T["pod"][:, 0] * k, T["pod"][:, 1] * k, fc="#f0f0f0", ec="k", lw=0.8)
    for q in T["booms"] + T["fins"]:
        top.fill(q[:, 0] * k, q[:, 1] * k, fc="#636363", ec="k", lw=0.5)
    top.fill(T["motor"][:, 0] * k, T["motor"][:, 1] * k, fc="#252525")
    top.plot(T["prop"][:, 0] * k, T["prop"][:, 1] * k, "r--", lw=1.2)
    for h in T["hinges"]:
        top.plot(h[:, 0] * k, h[:, 1] * k, "k:", lw=0.8)
    b2 = p["span"] / 2
    _dim(top, (L["x_nose"], -b2 - 60), (L["tail_te"], -b2 - 60), f"overall length {L['overall_length']:.0f}")
    _dim(top, (L["x_nose"] - 40, -b2), (L["x_nose"] - 40, b2), f"span {p['span']:.0f}")
    _dim(top, (0, 30), (p["root_chord"], 30), f"root {p['root_chord']:.0f}", offset=(0, 40))
    xt = L["le_sweep_tip"]
    _dim(top, (xt, b2 + 25), (xt + p["tip_chord"], b2 + 25), f"tip {p['tip_chord']:.0f}")
    _dim(top, (L["tail_te"] + 40, -p["boom_y"]), (L["tail_te"] + 40, p["boom_y"]), f"booms {2 * p['boom_y']:.0f}")
    _dim(top, (L["tail_le"] - 45, -p["tail_span"] / 2), (L["tail_le"] - 45, p["tail_span"] / 2), f"tail {p['tail_span']:.0f}")
    _dim(top, (L["tail_le"], -p["tail_span"] / 2 - 25), (L["tail_te"], -p["tail_span"] / 2 - 25), f"{p['tail_chord']:.0f}")
    _dim(top, (L["x_nose"], b2 * 0.55), (L["x_pod_end"], b2 * 0.55), f"pod {p['pod_length']:.0f}")
    _dim(top, (p["boom_x0"], -p["boom_y"] - 45), (L["boom_x1"], -p["boom_y"] - 45), f"boom {p['boom_length']:.0f}")
    top.text(L["x_ac"], b2 * 0.82, f"S = {L['S_ref']:.3f} m², AR {L['AR']:.1f}, MAC {L['mac']:.0f}", fontsize=8)
    top.set_title("top view [mm] (x aft; propeller disc red dashed; hinge lines dotted)", fontsize=10)
    # --- side view
    S_ = o["side"]
    side.fill(S_["pod"][:, 0] * k, S_["pod"][:, 1] * k, fc="#f0f0f0", ec="k", lw=0.8)
    side.fill(S_["wing_section"][:, 0] * k, S_["wing_section"][:, 1] * k, fc="#deebf7", ec="k", lw=0.7)
    for key, fc in (("boom", "#636363"), ("fin", "#deebf7"), ("tail", "#deebf7"), ("motor", "#252525"), ("skid", "#fd8d3c")):
        q = S_[key]
        side.fill(q[:, 0] * k, q[:, 1] * k, fc=fc, ec="k", lw=0.6)
    side.plot(S_["prop"][:, 0] * k, S_["prop"][:, 1] * k, "r--", lw=1.2)
    for h in S_["hinges"]:
        side.plot(h[:, 0] * k, h[:, 1] * k, "k:", lw=0.8)
    # ground at rest: the keel's rear corner and the tail bumpers
    gx = np.array([p["skid_x0"] - 100, L["boom_x1"] + 60])
    z0, z1 = L["skid_lowest_z"], L["tail_bumper_z"]
    slope = (z1 - z0) / (L["boom_x1"] - p["skid_x1"])
    side.plot(gx, z0 + (gx - p["skid_x1"]) * slope, color="#8c6d31", lw=1.5)
    side.text(gx[0], z0 + (gx[0] - p["skid_x1"]) * slope - 18, f"ground at rest ({L['resting_pitch_deg']:.1f}° nose down)", fontsize=8, color="#8c6d31")
    _dim(side, (L["prop_x"] + 12, L["prop_lowest_z"]), (L["prop_x"] + 12, z0 + (L["prop_x"] - p["skid_x1"]) * slope),
         f"{L['prop_ground_margin_resting']:.0f}", color="r")
    _dim(side, (L["x_nose"] - 30, -p["pod_height"]), (L["x_nose"] - 30, 0), f"pod {p['pod_height']:.0f}")
    _dim(side, (L["tail_te"] + 30, L["skid_lowest_z"]), (L["tail_te"] + 30, p["boom_z"] + p["fin_height"]), f"height {L['height_over_skid']:.0f}")
    _dim(side, (L["fin_le"], p["boom_z"] + p["fin_height"] + 15), (L["tail_te"], p["boom_z"] + p["fin_height"] + 15), f"fin {p['fin_chord']:.0f}")
    _dim(side, (L["x_ac"], 70), (L["x_ac_tail"], 70), f"tail arm l_t {L['l_t']:.0f}")
    side.set_title(f"side view [mm] (wing incidence {p['wing_incidence_deg']:g}°, tail {p['tail_incidence_deg']:g}°, "
                   f"down-thrust {p['motor_downthrust_deg']:g}°)", fontsize=10)
    # --- front view
    F = o["front"]
    front.fill(F["wing"][:, 0] * k, F["wing"][:, 1] * k, fc="#deebf7", ec="k", lw=0.7)
    front.fill(F["pod"][:, 0] * k, F["pod"][:, 1] * k, fc="#f0f0f0", ec="k", lw=0.8)
    front.fill(F["stab"][:, 0] * k, F["stab"][:, 1] * k, fc="#deebf7", ec="k", lw=0.5)
    for q in F["booms"]:
        front.fill(q[:, 0] * k, q[:, 1] * k, fc="#636363", ec="k", lw=0.5)
    for q in F["fins"]:
        front.fill(q[:, 0] * k, q[:, 1] * k, fc="#deebf7", ec="k", lw=0.5)
    front.fill(F["skid"][:, 0] * k, F["skid"][:, 1] * k, fc="#fd8d3c", ec="k", lw=0.5)
    front.plot(F["prop"][:, 0] * k, F["prop"][:, 1] * k, "r--", lw=1.2)
    _dim(front, (L["prop_R"], p["motor_z"]), (p["boom_y"] - p["boom_od"] / 2, p["motor_z"]), f"{L['prop_clearance_boom']:.0f}", color="r", offset=(0, -40))
    _dim(front, (-L["prop_R"], -190), (L["prop_R"], -190), f"propeller Ø{p['prop_diameter']:.0f}")
    _dim(front, (-p["boom_y"], 160), (p["boom_y"], 160), f"{2 * p['boom_y']:.0f}")
    front.text(-b2, 60, f"dihedral {p['dihedral_deg']:g}°", fontsize=8)
    front.set_xlim(-b2 - 30, b2 + 30)
    front.set_ylim(-220, 200)
    front.set_title("front view [mm] (propeller disc red dashed)", fontsize=10)
    for ax in (top, side, front):
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
    fig.suptitle("NISUS — first engineering approximation (Nisus-OBS and Nisus-Zero share it)", fontsize=12)
    fig.tight_layout()
    return fig


def prop_clearance(p=None, tolerance_mm=1.5, boom_deflection_mm=None):
    """The propeller's swept disk in the plane of rotation against everything that crosses it: the two booms (with
    their lateral deflection under the fin load and the build tolerance), the pod's tail, the stabiliser (behind it),
    the ground at rest; a table and a front-view figure."""
    import pandas as pd
    p = nisus.resolve(p)
    L = nisus.Nisus.layout(p)
    bc = nisus.boom_check(p, tolerance_mm=tolerance_mm)
    d_def = bc["lateral_deflection_at_prop_mm"] if boom_deflection_mm is None else boom_deflection_mm
    rows = {
        "boom (nominal)": (L["prop_clearance_boom"], "distance from the disc's edge to the boom's surface in the propeller plane"),
        "boom (deflected + tolerance)": (L["prop_clearance_boom"] - d_def - tolerance_mm, f"fin side load deflects the boom {d_def:.2f} mm at the disc; ±{tolerance_mm} mm build"),
        "wing trailing edge to the disc (axial)": (L["prop_clearance_wing_te"] - 0.5 * 12, "propeller plane behind the root trailing edge, less half the hub"),
        "stabiliser leading edge behind the disc (axial)": (L["tail_le"] - L["prop_x"], "the disc sits ahead of the tail"),
        "pod top in the disc (radial)": (L["prop_R"] - (p["motor_diameter"] / 2 + 2.0), "the motor cup is inside the hub region (blades from r = 0.12 R)"),
        "ground at rest (vertical)": (L["prop_ground_margin_resting"], f"resting on the keel and the tail bumpers ({L['resting_pitch_deg']:.1f}°)"),
        "ground at a level belly touchdown (vertical)": (L["prop_ground_margin"], "keel flat on the ground, tail up"),
    }
    df = pd.DataFrame({k: {"clearance [mm]": v[0], "basis": v[1]} for k, v in rows.items()}).T
    fig, ax = plt.subplots(figsize=(7, 4.5))
    th = np.linspace(0, 2 * math.pi, 200)
    ax.fill(L["prop_R"] * np.cos(th), p["motor_z"] + L["prop_R"] * np.sin(th), fc="#fee0d2", ec="r", lw=1.2, label="swept disk")
    for s in (1, -1):
        c = (s * p["boom_y"], p["boom_z"])
        ax.add_patch(plt.Circle(c, p["boom_od"] / 2, color="#252525"))
        ax.add_patch(plt.Circle((c[0] - s * (d_def + tolerance_mm), c[1]), p["boom_od"] / 2, fill=False, ls="--", color="#252525"))
    ax.axhline(L["prop_lowest_z"] - L["prop_ground_margin_resting"], color="#8c6d31", lw=1.5, label="ground under the disc at rest")
    ax.add_patch(plt.Circle((0, p["motor_z"]), p["motor_diameter"] / 2, color="#636363", label="motor"))
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"propeller plane x = {L['prop_x']:.0f} mm: boom clearance {L['prop_clearance_boom']:.1f} mm nominal", fontsize=10)
    ax.set_xlabel("y [mm]"); ax.set_ylabel("z [mm]")
    return df, fig


def internal_layout(variant: str, battery_key: str = "gens-ace-3s-2200", p=None, figsize=(14, 7)):
    """The pod in side and top view with its bays and every mass-table item inside it as a marker sized by its mass,
    coloured by group; the CG, the neutral point and the battery's travel."""
    import nisus_flight as nf
    p = nisus.resolve(p)
    o = nisus.outline(p)
    t = ns.mass_table(variant, battery_key, p=p)
    ci = ns.cg_inertia(t, p)
    a = nf.aero(p)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    ax1.fill(o["side"]["pod"][:, 0] * 1000, o["side"]["pod"][:, 1] * 1000, fc="#f7f7f7", ec="k", lw=0.8)
    ax1.fill(o["side"]["wing_section"][:, 0] * 1000, o["side"]["wing_section"][:, 1] * 1000, fc="#deebf7", ec="k", lw=0.5, alpha=0.6)
    ax2.fill(o["top"]["pod"][:, 0] * 1000, o["top"]["pod"][:, 1] * 1000, fc="#f7f7f7", ec="k", lw=0.8)
    for name, (x0, x1, w, h) in o["bays"].items():
        ax1.add_patch(plt.Rectangle((x0 * 1000, -p["pod_height"] / 2 - h * 500), (x1 - x0) * 1000, h * 1000, fill=False, ls=":", ec="#636363"))
        ax2.add_patch(plt.Rectangle((x0 * 1000, -w * 500), (x1 - x0) * 1000, w * 1000, fill=False, ls=":", ec="#636363"))
        ax2.text(0.5 * (x0 + x1) * 1000, w * 500 + 4, name.split("(")[0].strip(), ha="center", fontsize=6.5, color="#636363")
    pod_x = (o["layout"]["x_nose"], o["layout"]["x_pod_end"] + 40)
    for item, r in t.iterrows():
        if not (pod_x[0] - 5 <= r["x [mm]"] <= pod_x[1]) or abs(r["y [mm]"]) > 60 or item.startswith(("wing:", "booms:", "tail:")):
            continue
        col = GROUP_COLOR.get(r["group"], "k")
        sz = 8 + 1.6 * r["mass [g]"]
        ax1.scatter(r["x [mm]"], r["z [mm]"], s=sz, c=col, ec="k", lw=0.3, zorder=3)
        ax2.scatter(r["x [mm]"], r["y [mm]"], s=sz, c=col, ec="k", lw=0.3, zorder=3)
        if r["mass [g]"] >= 9:
            ax1.annotate(item.split("(")[0][:26], (r["x [mm]"], r["z [mm]"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    lo, hi = t.attrs["battery_x_range"]
    ax2.plot([lo, hi], [-40, -40], color="#fdae6b", lw=4, label="battery travel (CG trim)")
    for ax in (ax1, ax2):
        ax.axvline(ci["x_cg_m"] * 1000, color="k", lw=1.2, label=f"CG {100 * ci['x_cg_frac_mac']:.1f} % MAC")
        ax.axvline(a["x_np"] * 1000, color="r", lw=1.0, ls="--", label=f"neutral point {100 * (a['x_np'] * 1000 - o['layout']['x_mac_le']) / o['layout']['mac']:.1f} % MAC")
        ax.set_aspect("equal"); ax.grid(alpha=0.25)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=g) for g, c in GROUP_COLOR.items() if g in set(t["group"])]
    ax1.legend(handles=handles, fontsize=6.5, loc="upper right", ncol=2)
    ax2.legend(fontsize=7, loc="lower right")
    ax1.set_title(f"Nisus-{variant}: internal placement, side view [mm] ({ci['mass_kg'] * 1000:.0f} g, battery {battery_key})", fontsize=10)
    ax2.set_title("top view [mm]", fontsize=10)
    ax2.set_xlabel("x [mm] (aft)")
    fig.tight_layout()
    return fig


def exploded_png(path, p=None, spread=1.0, size=(1600, 1000)):
    """The exploded assembly rendered off screen with pyvista (labels by colour group)."""
    import pyvista as pv
    from vegeta import dedalus
    from vegeta.dedalus import viz as dviz
    parts = nisus.exploded_parts(p, spread)
    pl = pv.Plotter(off_screen=True, window_size=list(size))
    pl.set_background("white")
    colors = {"wing": "#c6dbef", "spar": "#252525", "rear spar": "#252525", "pod": "#e0e0e0", "nose cone": "#bdbdbd", "boom": "#404040",
              "boom fitting": "#fd8d3c", "stabiliser": "#c6dbef", "fin": "#c6dbef", "tail fitting": "#fd8d3c", "motor mount": "#fd8d3c",
              "motor": "#636363", "propeller disc": "#fcbba1", "electronics tray": "#6baed6", "battery tray": "#fdd0a2", "belly skid": "#e6550d"}
    for name, wp in parts.items():
        g = dedalus.Geometry.from_cadquery(wp, name=name)
        mesh = dviz.to_pyvista(g, tolerance=0.3)
        key = next((k for k in colors if name.startswith(k)), None)
        pl.add_mesh(mesh, color=colors.get(key, "#cccccc"), smooth_shading=True, opacity=0.55 if name.startswith("propeller") else 1.0)
        c = np.asarray(mesh.center)
        pl.add_point_labels([c], [name], font_size=11, point_size=1, shape_opacity=0.35, always_visible=True)
    pl.camera_position = [(-900, -1500, 900), (250, 0, -40), (0, 0, 1)]
    pl.screenshot(str(path))
    pl.close()
    return path


__all__ = ["GROUP_COLOR", "three_view", "prop_clearance", "internal_layout", "exploded_png"]
