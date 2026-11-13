"""The video's three measured trends, reproduced with the solved corona (``docs/plasma.md``, P3):

1. **serrated vs square teeth: +17 % velocity** — a sharper effective edge (the serration's tip) against a blunter
   one at the same voltage: the velocity goes as the square root of the thrust, so +17 % velocity is +37 % thrust;
2. **wider spacing and larger grounds on the Mark III: +26 % velocity** — the Mark III is a straight multi-stage
   thruster, its spacing is *axial*: the grounds' offset dz grown, each layout run at its own spark-limited voltage
   (a longer gap lets the supply go higher; thrust = I dz / mu grows with dz while the current, ~ V (V - V0) / d²,
   holds): the first test (a wider *radial* pitch) gave -10 % — the wrong geometry for that generation;
3. **the 4B's loss: a longer axial gap at more voltage got slower, twice** — the grounds' offset dz grown from 8 to
   12 to 16 mm at a supply's voltage ceiling: the current collapses faster than dz grows.

Coarse meshes (``lc``) so the three run in minutes; the 1-D numbers under ``vegeta.mock``.

    python halka_trends.py [out.png]
"""
from __future__ import annotations

import math
import sys

from vegeta.fulgur import DRY
from vegeta.fulgur.breakdown import propagation_voltage

import halka_fulgur as hf

G = 9.81


def velocity_ratio(t1: float, t0: float) -> float:
    return math.sqrt(max(t1, 0.0) / max(t0, 1e-12))


def teeth(preset: str = "mark4c", V: float = 23e3, sharp: float = 0.1e-3, blunt: float = 0.2e-3, lc: float = 2.0) -> dict:
    a = hf.solve_section(preset, V, edge_r=blunt, lc=lc)
    b = hf.solve_section(preset, V, edge_r=sharp, lc=lc)
    return {"square (blunt edge)": a, "serrated (sharp edge)": b, "velocity gain": velocity_ratio(b["thrust"], a["thrust"]) - 1.0, "video": 0.17}


def spacing(preset: str = "mark4c", dzs=(0.008, 0.012), grounds=((10.0, 6.0), (14.0, 10.0)), lc: float = 2.0) -> dict:
    """The Mark III's change in two halves: the axial spacing grown (each layout at its own spark-limited voltage,
    margin 1 on its gap), then the grounds enlarged (``grounds``: (width, height) mm) at that voltage."""
    out = {}
    for dz, (gw, gh) in zip(dzs, grounds):
        d = math.hypot(0.014, dz)                                    # the 4C's radial pitch is 14 mm
        V = propagation_voltage(d, DRY)
        if (gw, gh) != grounds[0]:
            r0 = hf.solve_section(preset, V, dz=dz, lc=lc, params={"ground_w": grounds[0][0], "ground_h": grounds[0][1]})
            out[f"dz {dz*1e3:.0f} mm at its limit {V/1e3:.1f} kV"] = r0
        r = hf.solve_section(preset, V, dz=dz, lc=lc, params={"ground_w": gw, "ground_h": gh})
        out[f"dz {dz*1e3:.0f} mm, grounds {gw:.0f}x{gh:.0f}" + ("" if (gw, gh) != grounds[0] else f" at {V/1e3:.1f} kV")] = r
    keys = list(out)
    out["velocity gain"] = velocity_ratio(out[keys[-1]]["thrust"], out[keys[0]]["thrust"]) - 1.0
    out["spacing alone"] = velocity_ratio(out[keys[1]]["thrust"], out[keys[0]]["thrust"]) - 1.0 if len(keys) > 2 else None
    out["video"] = 0.26
    return out


def offset_loss(preset: str = "mark4c", V_ceiling: float = 24e3, dzs=(0.008, 0.012, 0.016), lc: float = 2.0) -> dict:
    out = {f"dz {dz*1e3:.0f} mm": hf.solve_section(preset, V_ceiling, dz=dz, lc=lc) for dz in dzs}
    th = [out[k]["thrust"] for k in out]
    out["velocity steps"] = [velocity_ratio(b, a) - 1.0 for a, b in zip(th[:-1], th[1:])]
    out["video"] = "slower at each step"
    return out


def main(out="halka_trends.png", lc: float = 2.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, BLUE, ORANGE, GREY = "#1f2328", "#2f6db3", "#c8711a", "#9aa4ad"
    t, s, o = teeth(lc=lc), spacing(lc=lc), offset_loss(lc=lc)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (title, pairs, gain, video) in zip(axes, [
            ("1. serrated vs square teeth (edge 0.1 vs 0.2 mm, 23 kV)", [(k, v["thrust"]) for k, v in t.items() if isinstance(v, dict)], t["velocity gain"], "+17 %"),
            ("2. the Mark III: wider axial spacing at its spark limit, then larger grounds", [(k, v["thrust"]) for k, v in s.items() if isinstance(v, dict)], s["velocity gain"], "+26 %"),
            ("3. the 4B: a longer axial gap at the voltage ceiling (24 kV)", [(k, v["thrust"]) for k, v in o.items() if isinstance(v, dict)], None, "slower, twice")]):
        names, th = [p[0] for p in pairs], [p[1] / G * 1e3 for p in pairs]
        ax.bar(range(len(th)), th, color=[GREY] + [BLUE] * (len(th) - 1), width=0.6)
        ax.set_xticks(range(len(th)), names, fontsize=8)
        ax.set_ylabel("thrust [gf]"); ax.set_title(title, loc="left", fontsize=10, color=INK); ax.grid(axis="y", color="#e6e8eb"); ax.set_axisbelow(True)
        for i, v in enumerate(th):
            ax.text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=9, color=INK)
        note = f"velocity {gain:+.0%} (video {video})" if gain is not None else f"velocity steps {', '.join(f'{g:+.0%}' for g in o['velocity steps'])} (video: {video})"
        ax.text(0.02, 0.95, note, transform=ax.transAxes, fontsize=9, color=ORANGE, va="top")
    fig.suptitle("Halka: the video's three trends in the solved corona (thrust; velocity goes as its square root)", color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    return out, t, s, o


if __name__ == "__main__":
    out, t, s, o = main(*sys.argv[1:2])
    print("teeth: velocity gain", round(t["velocity gain"], 3), "video", t["video"])
    print("spacing: velocity gain", round(s["velocity gain"], 3), "(spacing alone", s["spacing alone"] and round(s["spacing alone"], 3), ") video", s["video"])
    print("offset: velocity steps", [round(x, 3) for x in o["velocity steps"]], "video", o["video"])
    print(out)
