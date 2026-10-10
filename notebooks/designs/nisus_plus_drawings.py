"""NISUS+ drawings from the same parameters the CAD and the analyses use (notebook 33): NISUS's dimensioned three-view,
propeller disk and exploded view on NISUS+, plus the internal placement of the pack and the Orin, the crow
configuration (side and front), and the transport layout (the pieces against a backpack). Matplotlib figures."""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

import nisus_plus
import nisus_plus_systems as fs
from nisus_drawings import GROUP_COLOR, _dim


def three_view(p=None, figsize=(16, 11)):
    """Top, side and front views [mm] with the main dimensions; the propeller disc dashed, the hinge lines (ailerons,
    flaps, elevator, rudders) dotted, the panel joints dash-dotted, the ground line at rest in the side view."""
    p = nisus_plus.resolve(p)
    o = nisus_plus.outline(p)
    L = o["layout"]
    k = 1000.0
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.3], height_ratios=[1.6, 1.0])
    top, front, side = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    T = o["top"]
    for q in T["wing"] + T["tail"]:
        top.fill(q[:, 0] * k, q[:, 1] * k, fc="#deebf7", ec="k", lw=0.7)
    top.fill(T["pod"][:, 0] * k, T["pod"][:, 1] * k, fc="#f0f0f0", ec="k", lw=0.8)
    for q in T["booms"] + T["fins"]:
        top.fill(q[:, 0] * k, q[:, 1] * k, fc="#636363", ec="k", lw=0.5)
    top.fill(T["motor"][:, 0] * k, T["motor"][:, 1] * k, fc="#252525")
    top.plot(T["prop"][:, 0] * k, T["prop"][:, 1] * k, "r--", lw=1.2)
    for h in T["hinges"]:
        top.plot(h[:, 0] * k, h[:, 1] * k, "k:", lw=0.9)
    for j in T["joints"]:
        top.plot(j[:, 0] * k, j[:, 1] * k, color="#08519c", ls="-.", lw=1.2)
    b2 = p["span"] / 2
    _dim(top, (L["x_nose"], -b2 - 90), (L["tail_te"], -b2 - 90), f"overall length {L['overall_length']:.0f}")
    _dim(top, (L["x_nose"] - 60, -b2), (L["x_nose"] - 60, b2), f"span {p['span']:.0f}")
    _dim(top, (0, 40), (p["root_chord"], 40), f"root {p['root_chord']:.0f}", offset=(0, 60))
    xt = L["le_sweep_tip"]
    _dim(top, (xt, b2 + 40), (xt + p["tip_chord"], b2 + 40), f"tip {p['tip_chord']:.0f}")
    _dim(top, (L["tail_te"] + 60, -p["boom_y"]), (L["tail_te"] + 60, p["boom_y"]), f"booms {2 * p['boom_y']:.0f}")
    _dim(top, (L["tail_le"] - 60, -p["tail_span"] / 2), (L["tail_le"] - 60, p["tail_span"] / 2), f"tail {p['tail_span']:.0f}")
    _dim(top, (p["root_chord"] + 30, p["flap_y0"]), (p["root_chord"] + 30, p["flap_y1"]), "flap", color="#08519c")
    _dim(top, (p["root_chord"] + 30, L["y_aileron0"]), (p["root_chord"] + 30, b2), "aileron", color="#08519c")
    _dim(top, (-100, -p["wing_joint_y"]), (-100, p["wing_joint_y"]), f"centre piece {2 * p['wing_joint_y']:.0f}", color="#08519c")
    top.text(L["x_ac"] + 300, b2 * 0.85, f"S = {L['S_ref']:.2f} m², AR {L['AR']:.1f}, MAC {L['mac']:.0f}", fontsize=8)
    top.set_title("top view [mm] (x aft; propeller red dashed; hinges dotted; panel joints dash-dot)", fontsize=10)
    S_ = o["side"]
    side.fill(S_["pod"][:, 0] * k, S_["pod"][:, 1] * k, fc="#f0f0f0", ec="k", lw=0.8)
    side.fill(S_["wing_section"][:, 0] * k, S_["wing_section"][:, 1] * k, fc="#deebf7", ec="k", lw=0.7)
    for key, fc in (("boom", "#636363"), ("fin", "#deebf7"), ("tail", "#deebf7"), ("motor", "#252525"), ("skid", "#fd8d3c")):
        q = S_[key]
        side.fill(q[:, 0] * k, q[:, 1] * k, fc=fc, ec="k", lw=0.6)
    side.plot(S_["prop"][:, 0] * k, S_["prop"][:, 1] * k, "r--", lw=1.2)
    gx = np.array([p["skid_x0"] - 100, L["boom_x1"] + 60])
    z0, z1 = L["skid_lowest_z"], L["tail_bumper_z"]
    slope = (z1 - z0) / (L["boom_x1"] - p["skid_x1"])
    side.plot(gx, z0 + (gx - p["skid_x1"]) * slope, color="#8c6d31", lw=1.5)
    side.text(gx[0], z0 + (gx[0] - p["skid_x1"]) * slope - 25, f"ground at rest ({L['resting_pitch_deg']:.1f}° nose down)", fontsize=8, color="#8c6d31")
    _dim(side, (L["prop_x"] + 15, L["prop_lowest_z"]), (L["prop_x"] + 15, z0 + (L["prop_x"] - p["skid_x1"]) * slope), f"{L['prop_ground_margin_resting']:.0f}", color="r")
    _dim(side, (L["x_nose"] - 40, -p["pod_height"]), (L["x_nose"] - 40, 0), f"pod {p['pod_height']:.0f}")
    _dim(side, (L["tail_te"] + 40, L["skid_lowest_z"]), (L["tail_te"] + 40, p["boom_z"] + p["fin_height"]), f"height {L['height_over_skid']:.0f}")
    _dim(side, (L["x_ac"], 110), (L["x_ac_tail"], 110), f"tail arm l_t {L['l_t']:.0f}")
    side.set_title(f"side view [mm] (wing incidence {p['wing_incidence_deg']:g}°, tail {p['tail_incidence_deg']:g}°, down-thrust {p['motor_downthrust_deg']:g}°)", fontsize=10)
    F = o["front"]
    front.fill(F["wing"][:, 0] * k, F["wing"][:, 1] * k, fc="#deebf7", ec="k", lw=0.7)
    front.fill(F["pod"][:, 0] * k, F["pod"][:, 1] * k, fc="#f0f0f0", ec="k", lw=0.8)
    front.fill(F["stab"][:, 0] * k, F["stab"][:, 1] * k, fc="#deebf7", ec="k", lw=0.5)
    for q in F["fins"]:
        front.fill(q[:, 0] * k, q[:, 1] * k, fc="#deebf7", ec="k", lw=0.5)
    for q in F["booms"]:
        front.fill(q[:, 0] * k, q[:, 1] * k, fc="#636363", ec="k", lw=0.5)
    front.fill(F["skid"][:, 0] * k, F["skid"][:, 1] * k, fc="#fd8d3c", ec="k", lw=0.5)
    front.plot(F["prop"][:, 0] * k, F["prop"][:, 1] * k, "r--", lw=1.2)
    _dim(front, (L["prop_R"] * math.cos(0.23), p["motor_z"] - L["prop_R"] * math.sin(0.23)), (p["boom_y"] - p["boom_od"] / 2, p["boom_z"]),
         f"{L['prop_clearance_boom']:.0f}", color="r", offset=(0, -60))
    _dim(front, (-L["prop_R"], -260), (L["prop_R"], -260), f"propeller Ø{p['prop_diameter']:.0f}")
    front.text(-b2, 90, f"dihedral {p['dihedral_deg']:g}°", fontsize=8)
    front.set_xlim(-b2 - 40, b2 + 40)
    front.set_ylim(-300, 300)
    front.set_title("front view [mm] (propeller disc red dashed)", fontsize=10)
    for ax in (top, side, front):
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
    fig.suptitle("NISUS+ — first engineering approximation (Nisus+ Zero)", fontsize=12)
    fig.tight_layout()
    return fig


def prop_clearance(p=None, tolerance_mm=1.5):
    """NISUS's propeller-disk table on NISUS+ (booms with their deflection, the wing, the tail, the ground)."""
    import pandas as pd
    p = nisus_plus.resolve(p)
    L = nisus_plus.NisusPlus.layout(p)
    bc = nisus_plus.boom_check(p, tolerance_mm=tolerance_mm)
    d_def = bc["lateral_deflection_at_prop_mm"]
    rows = {"boom (nominal)": (L["prop_clearance_boom"], "disc edge to the boom's surface in the propeller plane"),
            "boom (deflected + tolerance)": (L["prop_clearance_boom"] - d_def - tolerance_mm, f"fin side load: {d_def:.2f} mm; ±{tolerance_mm} mm build"),
            "wing trailing edge to the disc (axial)": (L["prop_clearance_wing_te"] - 0.5 * 18, "less half the hub"),
            "stabiliser leading edge behind the disc (axial)": (L["tail_le"] - L["prop_x"], "the disc sits ahead of the tail"),
            "ground at rest (vertical)": (L["prop_ground_margin_resting"], f"resting on the keel and the tail bumpers ({L['resting_pitch_deg']:.1f}°)"),
            "ground at a level belly touchdown (vertical)": (L["prop_ground_margin"], "keel flat on the ground")}
    df = pd.DataFrame({k: {"clearance [mm]": v[0], "basis": v[1]} for k, v in rows.items()}).T
    fig, ax = plt.subplots(figsize=(7, 4.5))
    th = np.linspace(0, 2 * math.pi, 200)
    ax.fill(L["prop_R"] * np.cos(th), p["motor_z"] + L["prop_R"] * np.sin(th), fc="#fee0d2", ec="r", lw=1.2, label="swept disk")
    for s in (1, -1):
        ax.add_patch(plt.Circle((s * p["boom_y"], p["boom_z"]), p["boom_od"] / 2, color="#252525"))
    ax.add_patch(plt.Rectangle((-p["pod_width"] / 2, -p["pod_height"]), p["pod_width"], p["pod_height"], fill=False, ls=":", ec="#636363", label="pod section (max)"))
    ax.axhline(L["prop_lowest_z"] - L["prop_ground_margin_resting"], color="#8c6d31", lw=1.5, label="ground under the disc at rest")
    ax.add_patch(plt.Circle((0, p["motor_z"]), p["motor_diameter"] / 2, color="#636363", label="motor"))
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"propeller plane x = {L['prop_x']:.0f} mm: boom clearance {L['prop_clearance_boom']:.0f} mm", fontsize=10)
    ax.set_xlabel("y [mm]"); ax.set_ylabel("z [mm]")
    return df, fig


def internal_layout(battery_key: str = fs.DEFAULT_PACK, p=None, figsize=(15, 7)):
    """The pod in side and top view with its bays and every mass-table item inside it, the CG, the neutral point and the
    pack's travel (NISUS's figure)."""
    import nisus_plus_flight as ff
    p = nisus_plus.resolve(p)
    o = nisus_plus.outline(p)
    t = fs.mass_table(battery_key, p=p)
    ci = fs.cg_inertia(t, p)
    a = ff.aero(p)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    ax1.fill(o["side"]["pod"][:, 0] * 1000, o["side"]["pod"][:, 1] * 1000, fc="#f7f7f7", ec="k", lw=0.8)
    ax1.fill(o["side"]["wing_section"][:, 0] * 1000, o["side"]["wing_section"][:, 1] * 1000, fc="#deebf7", ec="k", lw=0.5, alpha=0.6)
    ax2.fill(o["top"]["pod"][:, 0] * 1000, o["top"]["pod"][:, 1] * 1000, fc="#f7f7f7", ec="k", lw=0.8)
    for name, (x0, x1, w, h) in o["bays"].items():
        ax1.add_patch(plt.Rectangle((x0 * 1000, -p["pod_height"] / 2 - h * 500), (x1 - x0) * 1000, h * 1000, fill=False, ls=":", ec="#636363"))
        ax2.add_patch(plt.Rectangle((x0 * 1000, -w * 500), (x1 - x0) * 1000, w * 1000, fill=False, ls=":", ec="#636363"))
        ax2.text(0.5 * (x0 + x1) * 1000, w * 500 + 5, name.split("(")[0].strip(), ha="center", fontsize=6.5, color="#636363")
    pod_x = (o["layout"]["x_nose"], o["layout"]["x_pod_end"] + 60)
    for item, r in t.iterrows():
        if not (pod_x[0] - 5 <= r["x [mm]"] <= pod_x[1]) or abs(r["y [mm]"]) > 80 or item.startswith(("wing:", "booms:", "tail:")):
            continue
        col = GROUP_COLOR.get(r["group"], "k")
        sz = 8 + 0.6 * r["mass [g]"]
        ax1.scatter(r["x [mm]"], r["z [mm]"], s=sz, c=col, ec="k", lw=0.3, zorder=3)
        ax2.scatter(r["x [mm]"], r["y [mm]"], s=sz, c=col, ec="k", lw=0.3, zorder=3)
        if r["mass [g]"] >= 25:
            ax1.annotate(item.split("(")[0][:28], (r["x [mm]"], r["z [mm]"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    lo, hi = t.attrs["battery_x_range"]
    ax2.plot([lo, hi], [-50, -50], color="#fdae6b", lw=4, label="pack travel (CG trim)")
    for ax in (ax1, ax2):
        ax.axvline(ci["x_cg_m"] * 1000, color="k", lw=1.2, label=f"CG {100 * ci['x_cg_frac_mac']:.1f} % MAC")
        ax.axvline(a["x_np"] * 1000, color="r", lw=1.0, ls="--", label=f"neutral point {100 * (a['x_np'] * 1000 - o['layout']['x_mac_le']) / o['layout']['mac']:.1f} % MAC")
        ax.set_aspect("equal"); ax.grid(alpha=0.25)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=g) for g, c in GROUP_COLOR.items() if g in set(t["group"])]
    ax1.legend(handles=handles, fontsize=6.5, loc="upper right", ncol=2)
    ax2.legend(fontsize=7, loc="lower right")
    ax1.set_title(f"Nisus+ Zero: internal placement, side view [mm] ({ci['mass_kg'] * 1000:.0f} g, {fs.pack(battery_key).name})", fontsize=10)
    ax2.set_title("top view [mm]", fontsize=10)
    ax2.set_xlabel("x [mm] (aft)")
    fig.tight_layout()
    return fig


def crow_view(p=None, flap_deg=55.0, aileron_deg=-25.0, figsize=(13, 4.5)):
    """The crow configuration: a chordwise section through the flap and one through the aileron with their surfaces
    deflected (side view), and the front view of the trailing edges."""
    from fixed_wing import naca4
    p = nisus_plus.resolve(p)
    d = nisus_plus.NisusPlus()
    L = nisus_plus.NisusPlus.layout(p)
    up, lo = naca4(p["camber"], p["camber_pos"], p["thickness"])
    sec = np.array(up + lo[::-1])
    fig, axes = plt.subplots(1, 3, figsize=figsize, gridspec_kw=dict(width_ratios=[1, 1, 1.3]))
    for ax, (name, y, frac, deg) in zip(axes[:2], (("flap", 0.5 * (p["flap_y0"] + p["flap_y1"]), 1 - p["flap_chord_frac"], flap_deg),
                                                   ("aileron", 0.5 * (L["y_aileron0"] + L["b2"]), 1 - p["aileron_chord_frac"], aileron_deg))):
        f = (y - L["yc"]) / (L["b2"] - L["yc"])
        c = p["root_chord"] + (p["tip_chord"] - p["root_chord"]) * f
        xs, zs = sec[:, 0] * c, sec[:, 1] * c
        xh, zh = d._hinge(p, y, frac)
        xh -= L["le_sweep_tip"] * f
        zh -= (y - L["yc"]) * math.tan(math.radians(p["dihedral_deg"]))
        fixed = xs <= xh
        ax.fill(xs[fixed], zs[fixed], fc="#deebf7", ec="k", lw=0.8)
        a = -math.radians(deg)
        mx, mz = xs[~fixed] - xh, zs[~fixed] - zh
        ax.fill(xh + mx * math.cos(a) - mz * math.sin(a), zh + mx * math.sin(a) + mz * math.cos(a), fc="#9ecae1", ec="#08519c", lw=0.8)
        ax.fill(xs[~fixed], zs[~fixed], fc="none", ec="#bdbdbd", lw=0.6, ls="--")
        ax.plot([xh], [zh], "ko", ms=3)
        ax.set_title(f"{name} at y = {y:.0f} mm: {deg:+.0f}°", fontsize=10)
        ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.set_xlabel("x [mm]")
    ax = axes[2]
    for s in (1, -1):
        for y0, y1, frac, deg, col in ((p["flap_y0"], p["flap_y1"], p["flap_chord_frac"], flap_deg, "#e6550d"),
                                       (L["y_aileron0"], L["b2"], p["aileron_chord_frac"], aileron_deg, "#3182bd")):
            for y in (y0, y1):
                c = p["root_chord"] + (p["tip_chord"] - p["root_chord"]) * (y - L["yc"]) / (L["b2"] - L["yc"])
                z0 = (y - L["yc"]) * math.tan(math.radians(p["dihedral_deg"]))
                ax.plot([s * y, s * y], [z0, z0 - frac * c * math.sin(math.radians(deg))], color=col, lw=1.5)
            ax.plot([s * y0, s * y1], [(y0 - L["yc"]) * math.tan(math.radians(p["dihedral_deg"])), (y1 - L["yc"]) * math.tan(math.radians(p["dihedral_deg"]))], color="k", lw=1)
    ax.plot([], [], color="#e6550d", label=f"flaps {flap_deg:+.0f}° (down)")
    ax.plot([], [], color="#3182bd", label=f"ailerons {aileron_deg:+.0f}° (up)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_title("front view: crow (trailing edges)", fontsize=10); ax.set_xlabel("y [mm]")
    fig.suptitle("NISUS+ in crow: the flaps down and the ailerons up — drag without lift loss (the fast descent)", fontsize=11)
    fig.tight_layout()
    return fig


def transport_view(p=None, backpack_mm=1150.0, figsize=(10, 4.5)):
    """The pieces NISUS+ comes apart into, drawn to scale beside a backpack's carrying length."""
    p = nisus_plus.resolve(p)
    tc = nisus_plus.transport_check(p, backpack_mm)
    fig, ax = plt.subplots(figsize=figsize)
    y = 0.0
    colors = {"centre section": "#9ecae1", "outer panel": "#c6dbef", "boom": "#525252", "pod": "#d9d9d9", "tail (stabiliser)": "#deebf7"}
    for name, Lp in tc["pieces_mm"].items():
        n = 2 if name in ("outer panel", "boom") else 1
        for i in range(n):
            ax.add_patch(plt.Rectangle((0, y), Lp, 50 if name != "boom" else 16, fc=colors.get(name, "#cccccc"), ec="k"))
            ax.text(Lp + 15, y + 8, f"{name}{' ' + str(i + 1) if n > 1 else ''}: {Lp:.0f} mm", fontsize=8, va="bottom")
            y += 70
    ax.axvline(backpack_mm, color="#a50f15", ls="--", lw=1.5)
    ax.text(backpack_mm + 10, y - 20, f"backpack ~{backpack_mm:.0f} mm", color="#a50f15", fontsize=8)
    ax.set_xlim(-20, backpack_mm + 450); ax.set_ylim(-20, y + 10)
    ax.set_yticks([]); ax.set_xlabel("length [mm]")
    ax.set_title(f"transport: longest piece {tc['longest_mm']:.0f} mm — {'fits' if tc['fits'] else 'does NOT fit'}", fontsize=10)
    fig.tight_layout()
    return fig


def exploded_png(path, p=None, spread=1.0, size=(1600, 1000)):
    """The exploded assembly rendered off screen with pyvista."""
    import pyvista as pv
    from vegeta import dedalus
    from vegeta.dedalus import viz as dviz
    parts = nisus_plus.exploded_parts(p, spread)
    pl = pv.Plotter(off_screen=True, window_size=list(size))
    pl.set_background("white")
    colors = {"wing": "#c6dbef", "spar": "#252525", "rear spar": "#252525", "pod": "#e0e0e0", "nose cone": "#bdbdbd", "boom": "#404040",
              "boom fitting": "#fd8d3c", "stabiliser": "#c6dbef", "fin": "#c6dbef", "tail fitting": "#fd8d3c", "motor mount": "#fd8d3c",
              "motor": "#636363", "propeller disc": "#fcbba1", "electronics tray": "#6baed6", "battery tray": "#fdd0a2", "belly skid": "#e6550d",
              "flap": "#9ecae1", "aileron": "#9ecae1", "spar joiner": "#08306b"}
    for name, wp in parts.items():
        g = dedalus.Geometry.from_cadquery(wp, name=name)
        mesh = dviz.to_pyvista(g, tolerance=0.5)
        key = next((k for k in colors if name.startswith(k)), None)
        pl.add_mesh(mesh, color=colors.get(key, "#cccccc"), smooth_shading=True, opacity=0.55 if name.startswith("propeller") else 1.0)
        pl.add_point_labels([np.asarray(mesh.center)], [name], font_size=11, point_size=1, shape_opacity=0.35, always_visible=True)
    pl.camera_position = [(-1400, -2300, 1400), (350, 0, -40), (0, 0, 1)]
    pl.screenshot(str(path))
    pl.close()
    return path


__all__ = ["three_view", "prop_clearance", "internal_layout", "crow_view", "transport_view", "exploded_png"]
