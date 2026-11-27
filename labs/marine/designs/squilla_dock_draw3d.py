"""Squilla's dock on the jacket: the jacket's members as lines, the dock's envelope and approach corridor on leg A1 at
-80 m, the clearance survey on every leg and depth, and the numbers.   python squilla_dock_draw3d.py [out.png]"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import squilla_dock as sd
from oil_platform import Jacket

KIND = {"leg": ("#7d6608", 3.0), "brace": ("#b7950b", 1.2), "horizontal": ("#b7950b", 1.2), "conductor": ("#5d6d7e", 1.0),
        "riser": ("#1c2833", 1.5), "caisson": ("#5d6d7e", 1.0), "deck_leg": ("#7d6608", 2.0)}


def main(out="squilla_dock_3d.png"):
    g = Jacket.members()
    r = sd.report()
    fig = plt.figure(figsize=(20, 11))
    ax = fig.add_axes([0.0, 0.05, 0.48, 0.9], projection="3d")
    for m in g["members"]:
        c, w = KIND.get(m["kind"], ("#999", 1.0))
        ax.plot(*zip(m["a"], m["b"]), color=c, lw=w)
    ax.scatter(*zip(*[a["p"] for a in g["anodes"]]), s=6, color="#aab7b8")
    env = sd.envelope("A1", -80.0)
    for p, rad, what in env["points"]:
        col = {"garage": "#16a085", "corridor": "#e74c3c", "collar": "#2c3e50"}[what]
        ax.scatter(*p, s=40 if what != "corridor" else 25, color=col, alpha=0.8 if what != "corridor" else 0.4)
    ax.set_xlim(-45, 25); ax.set_ylim(-45, 25); ax.set_zlim(-110, 20); ax.set_box_aspect((1, 1, 1.85))
    ax.view_init(elev=12, azim=-120); ax.set_title("the dock on leg A1 at -80 m (green), its 12 m approach corridor (red), the collars (dark)")
    ax2 = fig.add_axes([0.53, 0.55, 0.43, 0.38])
    for leg, col in zip(sd.LEGS, ("#16a085", "#c0392b", "#2980b9", "#8e44ad")):
        rows = [x for x in sd.survey()["rows"] if x["leg"] == leg]
        ax2.plot([x["depth_m"] for x in rows], [min(x["clearance_m"], x["own_anode_gap_m"] + 0.3) for x in rows], "-o", color=col, label=f"leg {leg}")
    ax2.axhline(0.3, color="k", ls="--", lw=0.8); ax2.set_xlabel("depth [m]"); ax2.set_ylabel("least clearance [m]")
    ax2.set_title("clearance to the other members and anodes (fits above the dashed line); A2 carries the risers"); ax2.legend(); ax2.grid(alpha=0.3)
    tx = fig.add_axes([0.53, 0.04, 0.43, 0.45]); tx.axis("off")
    m, s, d = r["mounting"], r["services"], r["docking"]
    rows = [["where", f"leg A1 at -80 m (also B1, B2); depths that fit: {', '.join(str(int(-z)) for z in r['fitting_depths']['A1'])} m"],
            ["clamp", f"2 split collars, {m['collar_spacing_m']} m apart, liner bore {m['liner_bore_m']} m, {m['bolts_per_collar']} x M30 each; {m['collar_mass_kg']} kg"],
            ["loads", f"{m['weight_in_water_kN']} kN in water, {m['drag_kN']} kN drag at 1 m/s, lever {m['lever_m']} m: {m['normal_force_per_collar_kN']} kN a collar (mu 0.2, SF 2)"],
            ["charging", f"{s['charger']['kW']:g} kW inductive: Squilla full in {s['charger']['charge_h']} h; a patrol charge lasts {s['patrol']['hours_on_a_charge']} h ({s['patrol']['duty_out_of_dock']:.0%} out)"],
            ["data", "1 Gbit/s fibre up the leg (160 m hybrid cable); optical modem docked; acoustic + USBL away (~2 km)"],
            ["docking", f"dock {d['dock_s']} s, undock {d['undock_s']} s; cross current <= {d['cross_current_limit_m_s']} m/s, head 1.0 m/s; storm orbital at -80 m {d['orbital_at_depth_m_s']['-80m']} m/s"]]
    tab = tx.table(cellText=rows, colLabels=["", "Squilla's resident dock (class values, assumed)"], loc="center", cellLoc="left", colWidths=[0.13, 0.87])
    tab.auto_set_font_size(False); tab.set_fontsize(9); tab.scale(1, 1.9)
    fig.suptitle("Squilla's resident dock on the jacket: fit, clamp, power, data, docking", fontsize=13)
    fig.savefig(out, dpi=95)
    return out


if __name__ == "__main__":
    print(main(*sys.argv[1:]))
