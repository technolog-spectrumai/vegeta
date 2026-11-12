"""FALCO drawings from the same parameters the CAD and the analyses use (notebook 35): NISUS+'s dimensioned three-view
and views on the tractor, the **parked-propeller clearance** (a blade straight down against a blade parked horizontal
over the ground line at touchdown), the internal placement, the crow configuration, the transport layout, the exploded
view. Matplotlib figures."""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

import falco
import falco_systems as fs
import nisus_plus_drawings as npd
from nisus_drawings import GROUP_COLOR, _dim


def three_view(p=None, figsize=(16, 11)):
    """Top, side and front views [mm] with the main dimensions: the propeller disc at the nose dashed, the parked
    propeller's plate in the side view, the hinge lines dotted, the panel joints dash-dotted, the ground line at rest."""
    p = falco.resolve(p)
    o = falco.outline(p)
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
    _dim(top, (L["tail_le"] - 60, -p["tail_span"] / 2), (L["tail_le"] - 60, p["tail_span"] / 2), f"tail {p['tail_span']:.0f}")
    _dim(top, (p["root_chord"] + 30, p["flap_y0"]), (p["root_chord"] + 30, p["flap_y1"]), "flap", color="#08519c")
    _dim(top, (p["root_chord"] + 30, L["y_aileron0"]), (p["root_chord"] + 30, b2), "aileron", color="#08519c")
    _dim(top, (-100, -p["wing_joint_y"]), (-100, p["wing_joint_y"]), f"centre piece {2 * p['wing_joint_y']:.0f}", color="#08519c")
    _dim(top, (L["prop_x"], -L["prop_R"] - 50), (L["prop_x"] + 0.01, L["prop_R"] + 50), f"propeller Ø{p['prop_diameter']:.0f}", color="r")
    top.text(L["x_ac"] + 300, b2 * 0.85, f"S = {L['S_ref']:.2f} m², AR {L['AR']:.1f}, MAC {L['mac']:.0f}", fontsize=8)
    top.set_title("top view [mm] (x aft; propeller red dashed; hinges dotted; panel joints dash-dot)", fontsize=10)
    S_ = o["side"]
    side.fill(S_["pod"][:, 0] * k, S_["pod"][:, 1] * k, fc="#f0f0f0", ec="k", lw=0.8)
    side.fill(S_["wing_section"][:, 0] * k, S_["wing_section"][:, 1] * k, fc="#deebf7", ec="k", lw=0.7)
    for key, fc in (("boom", "#636363"), ("fin", "#deebf7"), ("tail", "#deebf7"), ("motor", "#252525"), ("skid", "#fd8d3c"), ("prop_parked", "#a50f15")):
        q = S_[key]
        side.fill(q[:, 0] * k, q[:, 1] * k, fc=fc, ec="k", lw=0.6)
    side.plot(S_["prop"][:, 0] * k, S_["prop"][:, 1] * k, "r--", lw=1.2)
    gx = np.array([L["x_nose"] - 100, L["boom_x1"] + 60])
    z0, z1 = L["skid_lowest_z"], L["tail_bumper_z"]
    slope = (z1 - z0) / (L["boom_x1"] - 25.0 - p["skid_x1"])
    side.plot(gx, z0 + (gx - p["skid_x1"]) * slope, color="#8c6d31", lw=1.5)
    side.text(gx[0], z0 + (gx[0] - p["skid_x1"]) * slope - 25, f"ground at rest ({L['resting_pitch_deg']:.1f}° nose up)", fontsize=8, color="#8c6d31")
    z_line = z0 + (L["prop_x"] - p["skid_x1"]) * slope
    _dim(side, (L["prop_x"] + 25, L["prop_lowest_z"]), (L["prop_x"] + 25, z_line), f"blade down {L['prop_ground_margin_resting']:.0f}", color="r")
    _dim(side, (L["prop_x"] - 55, L["motor_z"] - p["hub_diameter"] / 2), (L["prop_x"] - 55, z_line), f"parked {L['prop_ground_margin_parked_resting']:.0f}", color="#a50f15")
    _dim(side, (L["x_nose"] - 40, -p["pod_height"]), (L["x_nose"] - 40, 0), f"fuselage Ø{p['pod_height']:.0f}")
    _dim(side, (L["tail_te"] + 40, L["skid_lowest_z"]), (L["tail_te"] + 40, L["tail_z"] + p["fin_height"]), f"height {L['height_over_skid']:.0f}")
    _dim(side, (L["x_ac"], 110), (L["x_ac_tail"], 110), f"tail arm l_t {L['l_t']:.0f}")
    _dim(side, (L["x_tube0"], p["boom_z"] - 60), (L["boom_x1"], p["boom_z"] - 60), f"tail tube {p['tail_tube_od']:g}/{p['tail_tube_id']:g} x {L['tube_length']:.0f}")
    side.set_title(f"side view [mm] (wing incidence {p['wing_incidence_deg']:g}°, tail {p['tail_incidence_deg']:g}°, down-thrust {p['motor_downthrust_deg']:g}°; "
                   f"the propeller parked horizontal for the landing)", fontsize=10)
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
    front.plot([-L["prop_R"], L["prop_R"]], [L["motor_z"], L["motor_z"]], color="#a50f15", lw=3, label="parked")
    _dim(front, (-L["prop_R"], -360), (L["prop_R"], -360), f"propeller Ø{p['prop_diameter']:.0f}")
    front.text(-b2, 90, f"dihedral {p['dihedral_deg']:g}°", fontsize=8)
    front.set_xlim(-b2 - 40, b2 + 40)
    front.set_ylim(-400, 460)
    front.set_title("front view [mm] (propeller disc red dashed; parked: the dark bar)", fontsize=10)
    for ax in (top, side, front):
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
    fig.suptitle("FALCO — first engineering approximation (Falco-Zero)", fontsize=12)
    fig.tight_layout()
    return fig


def park_clearance(p=None, pitch_deg=(0.0, 4.0, 8.0)):
    """The propeller against the ground at touchdown: the blade straight down and the blade parked horizontal (the hub
    and the spinner), over the ground line through the keel's rear corner at the body pitch angles ``pitch_deg``
    (level, the flare, the stall): the table and a side-view figure."""
    import pandas as pd
    p = falco.resolve(p)
    L = falco.Falco.layout(p)
    rows = {}
    for th in pitch_deg:
        t = math.radians(th)
        # the keel's rear corner on the ground; the propeller's lowest point rotates about it with the body pitch
        dx = L["prop_x"] - p["skid_x1"]
        z_blade = L["skid_lowest_z"] + (L["prop_lowest_z"] - L["skid_lowest_z"]) * math.cos(t) - dx * math.sin(t) - L["skid_lowest_z"]
        z_park = L["skid_lowest_z"] + (L["motor_z"] - p["hub_diameter"] / 2 - L["skid_lowest_z"]) * math.cos(t) - dx * math.sin(t) - L["skid_lowest_z"]
        rows[f"body pitch {th:+.0f}° (nose up)"] = {"blade straight down [mm]": z_blade, "parked horizontal: hub and spinner [mm]": z_park,
                                                    "verdict": "strike" if z_blade < 0 else "clears", "parked verdict": "clears" if z_park >= p["park_clearance_min"] else "too close"}
    rows["at rest (keel and tail bumper)"] = {"blade straight down [mm]": L["prop_ground_margin_resting"], "parked horizontal: hub and spinner [mm]": L["prop_ground_margin_parked_resting"],
                                             "verdict": "strike" if L["prop_ground_margin_resting"] < 0 else "clears",
                                             "parked verdict": "clears" if L["prop_ground_margin_parked_resting"] >= p["park_clearance_min"] else "too close"}
    df = pd.DataFrame(rows).T
    o = falco.outline(p)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    k = 1000.0
    S_ = o["side"]
    ax.fill(S_["pod"][:, 0] * k, S_["pod"][:, 1] * k, fc="#f0f0f0", ec="k", lw=0.8)
    ax.fill(S_["skid"][:, 0] * k, S_["skid"][:, 1] * k, fc="#fd8d3c", ec="k", lw=0.5)
    ax.fill(S_["motor"][:, 0] * k, S_["motor"][:, 1] * k, fc="#252525")
    ax.plot(S_["prop"][:, 0] * k, S_["prop"][:, 1] * k, "r--", lw=2, label="a blade straight down")
    ax.fill(S_["prop_parked"][:, 0] * k, S_["prop_parked"][:, 1] * k, fc="#a50f15", label="parked horizontal (the hub)")
    ax.axhline(L["skid_lowest_z"], color="#8c6d31", lw=1.5, label="the meadow at a level touchdown")
    ax.set_xlim(L["x_nose"] - 120, p["skid_x1"] + 200); ax.set_ylim(L["prop_lowest_z"] - 40, 60)
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"the propeller at touchdown: blade down {L['prop_ground_margin']:+.0f} mm, parked {L['prop_ground_margin_parked']:+.0f} mm above the keel's bottom", fontsize=10)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("z [mm]")
    return df, fig


def internal_layout(battery_key: str = fs.DEFAULT_PACK, p=None, figsize=(15, 7)):
    """The fuselage in side and top view with its bays and every mass-table item inside it, the CG, the neutral point
    and the pack's travel (NISUS+'s figure on FALCO)."""
    import falco_flight as ff
    p = falco.resolve(p)
    o = falco.outline(p)
    t = fs.mass_table(battery_key, p=p)
    ci = fs.cg_inertia(t, p)
    a = ff.aero(p)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    ax1.fill(o["side"]["pod"][:, 0] * 1000, o["side"]["pod"][:, 1] * 1000, fc="#f7f7f7", ec="k", lw=0.8)
    ax1.fill(o["side"]["wing_section"][:, 0] * 1000, o["side"]["wing_section"][:, 1] * 1000, fc="#deebf7", ec="k", lw=0.5, alpha=0.6)
    ax1.fill(o["side"]["motor"][:, 0] * 1000, o["side"]["motor"][:, 1] * 1000, fc="#252525")
    ax1.plot(o["side"]["prop"][:, 0] * 1000, o["side"]["prop"][:, 1] * 1000, "r--", lw=1)
    ax2.fill(o["top"]["pod"][:, 0] * 1000, o["top"]["pod"][:, 1] * 1000, fc="#f7f7f7", ec="k", lw=0.8)
    for name, (x0, x1, w, h) in o["bays"].items():
        ax1.add_patch(plt.Rectangle((x0 * 1000, -p["pod_height"] / 2 - h * 500), (x1 - x0) * 1000, h * 1000, fill=False, ls=":", ec="#636363"))
        ax2.add_patch(plt.Rectangle((x0 * 1000, -w * 500), (x1 - x0) * 1000, w * 1000, fill=False, ls=":", ec="#636363"))
        ax2.text(0.5 * (x0 + x1) * 1000, w * 500 + 5, name.split("(")[0].strip(), ha="center", fontsize=6.5, color="#636363")
    pod_x = (o["layout"]["x_nose"], o["layout"]["x_pod_end"] + 60)
    for item, r in t.iterrows():
        if not (pod_x[0] - 60 <= r["x [mm]"] <= pod_x[1]) or abs(r["y [mm]"]) > 80 or item.startswith(("wing:", "tail:")):
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
    ax1.set_title(f"Falco-Zero: internal placement, side view [mm] ({ci['mass_kg'] * 1000:.0f} g, {fs.pack(battery_key).name})", fontsize=10)
    ax2.set_title("top view [mm]", fontsize=10)
    ax2.set_xlabel("x [mm] (aft)")
    fig.tight_layout()
    return fig


def crow_view(p=None, flap_deg=55.0, aileron_deg=-25.0, figsize=(13, 4.5)):
    """NISUS+'s crow figure on FALCO's wing (the same wing, the flaps longer: no boom fitting in the way)."""
    fig = npd.crow_view(falco.resolve(p), flap_deg, aileron_deg, figsize, design=falco.Falco())
    fig.suptitle("FALCO in crow: the flaps down and the ailerons up — drag without lift loss (the fast descent)", fontsize=11)
    return fig


def transport_view(p=None, backpack_mm=1150.0, figsize=(10, 4.5)):
    """The pieces FALCO comes apart into, drawn to scale beside a backpack's carrying length."""
    p = falco.resolve(p)
    tc = falco.transport_check(p, backpack_mm)
    fig, ax = plt.subplots(figsize=figsize)
    y = 0.0
    colors = {"centre section": "#9ecae1", "outer panel": "#c6dbef", "fuselage": "#d9d9d9", "tail tube with the tail": "#525252", "tail (stabiliser)": "#deebf7"}
    for name, Lp in tc["pieces_mm"].items():
        n = 2 if name == "outer panel" else 1
        for i in range(n):
            ax.add_patch(plt.Rectangle((0, y), Lp, 50 if "tube" not in name else 20, fc=colors.get(name, "#cccccc"), ec="k"))
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
    parts = falco.exploded_parts(p, spread)
    pl = pv.Plotter(off_screen=True, window_size=list(size))
    pl.set_background("white")
    colors = {"wing": "#c6dbef", "spar": "#252525", "rear spar": "#252525", "fuselage": "#e0e0e0", "nose": "#bdbdbd", "tail tube": "#404040",
              "tail socket": "#fd8d3c", "stabiliser": "#c6dbef", "fin": "#c6dbef", "tail fitting": "#fd8d3c", "firewall": "#fd8d3c",
              "motor": "#636363", "propeller disc": "#fcbba1", "electronics tray": "#6baed6", "battery tray": "#fdd0a2", "belly skid": "#e6550d",
              "flap": "#9ecae1", "aileron": "#9ecae1", "spar joiner": "#08306b"}
    for name, wp in parts.items():
        g = dedalus.Geometry.from_cadquery(wp, name=name)
        mesh = dviz.to_pyvista(g, tolerance=0.5)
        key = next((k for k in colors if name.startswith(k)), None)
        pl.add_mesh(mesh, color=colors.get(key, "#cccccc"), smooth_shading=True, opacity=0.55 if name.startswith("propeller") else 1.0)
        pl.add_point_labels([np.asarray(mesh.center)], [name], font_size=11, point_size=1, shape_opacity=0.35, always_visible=True)
    pl.camera_position = [(-1500, -2300, 1400), (300, 0, -40), (0, 0, 1)]
    pl.screenshot(str(path))
    pl.close()
    return path


__all__ = ["three_view", "park_clearance", "internal_layout", "crow_view", "transport_view", "exploded_png"]
