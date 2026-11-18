"""Wind and the recoilless 8 mm emitter at 100 m (MuJoCo, ``bearing_target``): how wind moves the hits.

The emitter: ``bearing_recoilless`` (280 m/s, scatter 0.06 deg and 0.5 %). 20 shots per case at the lollipop target
(300 mm disc) 100 m away. The cases: calm; a steady crosswind of 2, 5 and 10 m/s with the aim NOT corrected; 5 m/s
with the aim corrected for the mean wind; 5 m/s gusting (± 2 m/s per shot) corrected; a 10 m/s head wind and tail
wind (uncorrected). Each case writes its own trajectory and energy plots; ``summary.png`` puts the target faces and
the hit counts side by side, with the drift each wind gives (hand: the point-mass model, no scatter).

    python bearing_wind.py [--shots 20] [--distance 100] [--out bearing_wind_results]
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

import bearing_recoilless as br
import bearing_target as bt

CASES = [
    ("calm", (0.0, 0.0), 0.0, False),
    ("cross 2 m/s", (0.0, 2.0), 0.0, False),
    ("cross 5 m/s", (0.0, 5.0), 0.0, False),
    ("cross 10 m/s", (0.0, 10.0), 0.0, False),
    ("cross 5, aim corrected", (0.0, 5.0), 0.0, True),
    ("cross 5 ±2 gusts, corrected", (0.0, 5.0), 2.0, True),
    ("head 10 m/s", (-10.0, 0.0), 0.0, False),
    ("tail 10 m/s", (10.0, 0.0), 0.0, False),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=20)
    ap.add_argument("--distance", type=float, default=100.0)
    ap.add_argument("--out", default="bearing_wind_results")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    v = br.TARGET_V
    e0, _, _ = bt.flight_aim(v, a.distance, bt.POLE_H)
    results = []
    for k, (name, wind, gust, corr) in enumerate(CASES):
        r = bt.run(a.shots, v, a.distance, seed=11, gap=0.1, scatter_speed=br.SCATTER_SPEED, scatter_deg=br.SCATTER_DEG,
                   wind=wind, gust=gust, aim_wind=corr)
        y, z = bt.path_to(v, e0, 0.0, a.distance, wind)                 # the drift of a calm-aimed shot, no scatter
        hits = sum(h is not None for h in r["hit"])
        tag = f"{k}_{name.replace(' ', '_').replace('/', '').replace(',', '').replace('±', 'pm')}"
        bt.plot(r, str(out / f"{tag}.png"))
        bt.plot_energy(r, str(out / f"{tag}_energy.png"))
        print(f"{name:30s} hits {hits:2d}/{a.shots}, points {sum(r['rings']):3d}, drift of an uncorrected shot "
              f"{y * 100:+6.1f} cm across, {(z - bt.POLE_H) * 100:+6.1f} cm up; aim az {r['az']:+.3f} deg", flush=True)
        results.append((name, wind, gust, corr, r, hits, y, z))
    summary(results, a.distance, out / "summary.png")


def summary(results, L, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    n = len(results)
    fig, axes = plt.subplots(2, (n + 1) // 2 + 1, figsize=(4.0 * ((n + 1) // 2 + 1), 8.4))
    faces = [ax for i, ax in enumerate(axes.flat) if i % axes.shape[1] != axes.shape[1] - 1]
    lim = 60.0
    for (name, wind, gust, corr, r, hits, y, z), ax in zip(results, faces):
        for rr in np.arange(bt.DISC_R, 0, -0.015):
            ax.add_patch(Circle((0, 0), rr * 100, fill=False, lw=0.4, color="k"))
        ax.add_patch(Circle((0, 0), bt.DISC_R * 100, color="goldenrod", alpha=0.25))
        for k, h in enumerate(r["hit"]):
            if h is not None:
                ax.plot(h[0] * 100, h[1] * 100, "o", color="tab:red", ms=5, mec="k")
        for k, tr in enumerate(r["tracks"]):
            if r["hit"][k] is None and len(tr) and tr[:, 1].max() >= L:
                j = np.searchsorted(tr[:, 1], L)
                ax.plot(tr[j, 2] * 100, (tr[j, 3] - bt.POLE_H) * 100, "x", color="tab:gray", ms=5)
                lim = max(lim, 1.1 * 100 * max(abs(tr[j, 2]), abs(tr[j, 3] - bt.POLE_H)))
        if wind[1]:
            ax.annotate("", xy=(-lim * 0.55, lim * 0.8), xytext=(-lim * 0.85, lim * 0.8),
                        arrowprops=dict(arrowstyle="->", color="tab:blue")) if wind[1] < 0 else \
                ax.annotate("", xy=(lim * 0.55, lim * 0.8), xytext=(lim * 0.85, lim * 0.8),
                            arrowprops=dict(arrowstyle="->", color="tab:blue"))
        ax.set_title(f"{name}\n{hits}/{len(r['hit'])} hits, {sum(r['rings'])} pts", fontsize=10)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    for ax in faces:
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.tick_params(labelsize=7)
    # the right column: hits per case, and drift against crosswind
    gs = axes[0, -1].get_gridspec()
    for ax in axes[:, -1]:
        ax.remove()
    axb = fig.add_subplot(gs[0, -1])
    axb.barh([x[0] for x in results][::-1], [x[5] for x in results][::-1], color="tab:red")
    axb.set_xlabel("hits / 20")
    axb.set_xlim(0, len(results[0][4]["hit"]))
    axb.tick_params(labelsize=8)
    axb.grid(alpha=0.3, axis="x")
    axd = fig.add_subplot(gs[1, -1])
    ws = np.linspace(0, 10, 11)
    e0, _, _ = bt.flight_aim(br.TARGET_V, L, bt.POLE_H)
    drift = [bt.path_to(br.TARGET_V, e0, 0.0, L, (0.0, w))[0] * 100 for w in ws]
    axd.plot(ws, drift, "o-", color="tab:blue")
    axd.axhline(bt.DISC_R * 100, color="goldenrod", ls="--", lw=1)
    axd.annotate("disc edge", (0.2, bt.DISC_R * 100 + 2), fontsize=8, color="goldenrod")
    axd.set_xlabel("crosswind [m/s]")
    axd.set_ylabel(f"drift at {L:.0f} m [cm]")
    axd.set_title("Crosswind drift (no aim correction)", fontsize=10)
    axd.grid(alpha=0.3)
    fig.suptitle(f"Recoilless 8 mm emitter, {br.TARGET_V:.0f} m/s, lollipop at {L:.0f} m: wind (target faces in cm, "
                 f"o hit, x miss; arrows: the wind's direction)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print("wrote", path)


if __name__ == "__main__":
    main()
