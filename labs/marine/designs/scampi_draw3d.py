"""Scampi in 3D with matplotlib from the CadQuery model's tessellation (each part in its colour), the work pose and the
folded transit pose, and the sizing against the requirements.

    python scampi_draw3d.py [out.png]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from scampi import ARMS, EFFECTORS, REQUIREMENTS, Scampi

COLOURS = {"frame": "#e8e2d0", "fairing": "#f2b705", "foam": "#d9d9d9", "electronics": "#8a8f98", "battery": "#4f5560",
           "thruster": "#202020", "work_arm": "#c0392b", "hold_arm": "#2e86c1", "skid": "#333333", "tool_skid": "#27ae60",
           "gripper_l": "#1b4f72", "gripper_r": "#7b241c", "cutter": "#8e44ad", "tool_holder": "#145a32"}


def _mesh(shape, tol=4.0):
    verts, tris = shape.val().tessellate(tol, 0.5)
    v = np.array([(p.x, p.y, p.z) for p in verts])
    return [v[list(t)] for t in tris]


def parts(p):
    d = Scampi()
    out = {"frame": d._frame(p), "fairing": d._fairing(p), "electronics": d._housing(p, "electronics"),
           "battery": d._housing(p, "battery"),
           "work_arm": d._arm(p, p["arm_reach"], -0.22 * p["width"], d.WORK_LINKS, p["arm_pose"]),
           "hold_arm": d._arm(p, p["hold_reach"], 0.22 * p["width"], d.HOLD_LINKS, p["arm_pose"]),
           "tool_skid": d._tool_skid(p), "tool_holder": d._tool_holder(p),
           "gripper_l": d._gripper(p, "hold", p["jaw_opening"]), "cutter": d._cutter(p, p["cutter_on_arm"])}
    out["gripper_r"] = d._gripper_parked(p) if p["cutter_on_arm"] else d._gripper(p, "work", p["jaw_opening"])
    th = None
    for at, ax in d.thruster_layout(p):
        t = d._thruster(p, at, ax)
        th = t if th is None else th.union(t)
    out["thruster"] = th
    sk = None
    for sx in (1, -1):
        for sy in (1, -1):
            k = d._skid(p, sx, sy)
            sk = k if sk is None else sk.union(k)
    out["skid"] = sk
    return out


def draw(ax, p, elev=20, azim=-60, alpha_fairing=0.35, only=None):
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for name, shp in parts(p).items():
        if only and name not in only:
            continue
        tris = _mesh(shp)
        a = alpha_fairing if name == "fairing" else 1.0
        ax.add_collection3d(Poly3DCollection(tris, facecolors=COLOURS[name], edgecolors=COLOURS[name], linewidths=0.0, alpha=a, shade=True))
        pts = np.concatenate(tris)
        lo, hi = np.minimum(lo, pts.min(0)), np.maximum(hi, pts.max(0))
    c, r = (lo + hi) / 2, (hi - lo).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r * 0.6, c[2] + r * 0.6)
    ax.set_box_aspect((1, 1, 0.6))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_zlabel("z [mm]")


def main(out="scampi_3d.png"):
    p = {q.name: q.default for q in Scampi.parameters}
    s = Scampi.sizing(p)
    fig = plt.figure(figsize=(18, 14))
    ax = fig.add_subplot(3, 2, 1, projection="3d")
    draw(ax, p)
    ax.set_title("Scampi, work pose: hold arm (blue) + gripper, work arm (red) + gripper, cutter (purple) in its holder")
    ax3 = fig.add_subplot(3, 2, 3, projection="3d")
    draw(ax3, {**p, "cutter_on_arm": True, "jaw_opening": 100.0}, elev=15, azim=-35,
         only=("gripper_l", "cutter"))
    ax3.set_title("close-up: gripper open 100 mm (blue, hold arm), cutter on the work arm (purple)")
    ax2 = fig.add_subplot(3, 2, 5, projection="3d")
    draw(ax2, {**p, "arm_pose": "folded"}, elev=8, azim=-90)
    ax2.set_title("side view, arms folded for transit")
    tx = fig.add_subplot(1, 2, 2)
    tx.axis("off")
    he = s["housings"]["battery"]
    g, c, e = EFFECTORS["gripper"], EFFECTORS["cutter"], s["effectors"]
    rows = [
        ["market", REQUIREMENTS["market"], ""],
        ["mass in air", "60-120 kg", f"{s['mass_in_air_kg']:.0f} kg"],
        ["size (no arms)", "~1.5 m", f"{p['length'] / 1000:.1f} x {p['width'] / 1000:.1f} x {p['height'] / 1000:.2f} m"],
        ["depth rating", f"{REQUIREMENTS['depth_m']:.0f} m", f"housings x{1.5:g}: buckling margin {he['buckling_margin']:+.2f}"],
        ["dive", "6-8 h", f"{s['dive_hours']:.0f} h profile: {s['dive_energy_Wh'] / 1000:.2f} kWh of {s['battery_usable_Wh'] / 1000:.2f} usable"],
        ["station keeping", f"{REQUIREMENTS['current_m_s']:g} m/s current", f"{s['station_keeping_current_m_s']:.2f} m/s sideways, {s['v_max_m_s']['x']:.2f} m/s ahead"],
        ["buoyancy", "slightly positive", f"+{s['net_buoyancy_kg']:g} kg; foam {s['foam_L']:.0f} L; lifts {s['payload_in_water_kg']:.1f} kg"],
        ["arms", "6-DOF work + 4-DOF hold", f"reach {p['arm_reach'] / 1000:.2f} / {p['hold_reach'] / 1000:.2f} m"],
        ["autonomy", "NVIDIA Jetson", "AGX Orin 64GB, 15-60 W"],
        ["comms", "untethered", "acoustic modem; optional fibre micro-tether"],
        ["tool skids", "universal", "belly bay 0.6 x 0.42 x 0.08 m, wet-mate 48 V + Ethernet"],
        ["grippers x2", "grasp bars, handles, lids, ropes", f"two-jaw, 0-{g['jaw_opening_mm'][1]:.0f} mm, {g['grip_force_N']:.0f} N, "
         f"{g['mass_kg']:g} kg; wrist {ARMS['work']['wrist_torque_Nm']:g} / {ARMS['hold']['wrist_torque_Nm']:g} N m"],
        ["cutter", "cut rope, line, wire", f"{c['cut_force_kN']:g} kN, {c['cycle_s']:g} s, {c['mass_kg']:g} kg: line "
         f"{e['max_cut_mm']['soft_line']:.0f} / wire rope {e['max_cut_mm']['wire_rope']:.0f} / armour {e['max_cut_mm']['armour_wire']:.0f} mm"],
        ["effectors on Scampi", "within payload and trim", f"+{e['mass_kg']:.1f} kg air, {e['in_water_kg']:.1f} kg water (trim lead out); "
         f"cutter out: {e['cutter_out_pitch_deg']:.0f} deg pitch"],
    ]
    tab = tx.table(cellText=rows, colLabels=["", "requirement", "Scampi v0"], loc="upper center", cellLoc="left",
                   colWidths=[0.17, 0.25, 0.58])
    tab.auto_set_font_size(False); tab.set_fontsize(10); tab.scale(1, 1.9)
    y = 0.24
    tx.text(0.0, y, "dive energy profile", weight="bold", transform=tx.transAxes)
    for name, h, w in s["profile"]:
        y -= 0.035
        tx.text(0.0, y, f"{name}: {h:g} h at {w:.0f} W", transform=tx.transAxes, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    return out


if __name__ == "__main__":
    print(main(*sys.argv[1:]))
