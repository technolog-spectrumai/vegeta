"""Speed distribution of the recoilless 8 mm emitter's shots: at the muzzle, along the flight and at the target.

Two sources on the same plots: the 20 MuJoCo shots of the calm 100 m test (``bearing_target.run``, as in
``bearing_wind``), and a Monte Carlo of the same point-mass model (2000 shots, the same scatter: 0.5 % in speed,
0.06 deg in direction; optionally gusts). The spread at the muzzle is the emitter's own (shot-to-shot pressure, ball
fit); along the flight the drag (Cd rising with Mach) narrows it in m/s terms.

    python bearing_speed_dist.py [--distance 100] [--n 2000] [--gust 0] [--out bearing_speed_dist.png]
"""
from __future__ import annotations

import argparse
import math

import numpy as np

import bearing_recoilless as br
import bearing_target as bt


def monte_carlo(n, distance, gust, seed=3):
    rng = np.random.default_rng(seed)
    v0 = br.TARGET_V
    e0, _, _ = bt.flight_aim(v0, distance, bt.POLE_H)
    xs = np.linspace(0, distance, 51)
    speeds = np.zeros((n, len(xs)))
    muzzle = np.zeros(n)
    for i in range(n):
        s = v0 * (1 + br.SCATTER_SPEED * rng.standard_normal())
        el = math.radians(e0 + br.SCATTER_DEG * rng.standard_normal())
        az = math.radians(br.SCATTER_DEG * rng.standard_normal())
        w = np.array([gust * rng.standard_normal(), gust * rng.standard_normal(), 0.0])
        p = bt.MUZZLE.copy()
        v = np.array([s * math.cos(el) * math.cos(az), s * math.cos(el) * math.sin(az), s * math.sin(el)])
        muzzle[i] = s
        j, dt = 0, 2e-4
        while j < len(xs):
            if p[0] >= xs[j]:
                speeds[i, j] = np.linalg.norm(v)
                j += 1
                continue
            u = v - w
            sp = np.linalg.norm(u)
            v = v + (-0.5 * bt.RHO * bt.cd(sp) * bt.AREA * sp / bt.MASS * u + np.array([0, 0, -9.81])) * dt
            p = p + v * dt
    return xs, muzzle, speeds


def mujoco_speeds(distance, gust):
    r = bt.run(20, br.TARGET_V, distance, seed=11, gap=0.1, scatter_speed=br.SCATTER_SPEED, scatter_deg=br.SCATTER_DEG,
               gust=gust)
    v0, vL, hit = [], [], []
    for k, tr in enumerate(r["tracks"]):
        if not len(tr):
            continue
        v0.append(tr[0, 4])
        if tr[:, 1].max() >= distance - 0.5:
            # the speed just before it reaches the target's plane (after a hit the track is the bounce)
            before = tr[tr[:, 1] < distance]
            vL.append(float(before[-1, 4]))
            hit.append(r["hit"][k] is not None)
    return np.array(v0), np.array(vL), np.array(hit), r


def plot(xs, muzzle, speeds, mj, distance, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = bt.MASS
    v0m, vLm, hitm, r = mj
    vL = speeds[:, -1]
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    a = ax[0, 0]
    a.hist(muzzle, bins=40, color="tab:blue", alpha=0.6, density=True, label=f"model, {len(muzzle)} shots")
    a.hist(v0m, bins=10, color="tab:red", alpha=0.6, density=True, label=f"MuJoCo, {len(v0m)} shots")
    a.set_title(f"Muzzle speed: mean {muzzle.mean():.1f} m/s, sd {muzzle.std():.2f} ({100 * muzzle.std() / muzzle.mean():.2f} %)")
    a.set_xlabel("m/s")
    a.legend()
    a = ax[0, 1]
    a.hist(vL, bins=40, color="tab:blue", alpha=0.6, density=True, label="model")
    a.hist(vLm[hitm], bins=8, color="tab:red", alpha=0.7, density=True, label=f"MuJoCo hits ({hitm.sum()})")
    if (~hitm).any():
        a.hist(vLm[~hitm], bins=5, color="tab:gray", alpha=0.6, density=True, label=f"MuJoCo misses ({(~hitm).sum()})")
    a.set_title(f"Speed at {distance:.0f} m: mean {vL.mean():.1f} m/s, sd {vL.std():.2f} ({100 * vL.std() / vL.mean():.2f} %)")
    a.set_xlabel("m/s")
    a.legend()
    a = ax[1, 0]
    p5, p50, p95 = np.percentile(speeds, [5, 50, 95], axis=0)
    a.fill_between(xs, p5, p95, color="tab:blue", alpha=0.25, label="model 5-95 %")
    a.plot(xs, p50, color="tab:blue", label="model median")
    for k, tr in enumerate(r["tracks"]):
        j = tr[:, 1] <= distance + 0.5
        a.plot(tr[j, 1], tr[j, 4], color="tab:red" if r["hit"][k] is not None else "tab:gray", lw=0.5, alpha=0.7)
    a.axhline(343.2 * 0.85, color="k", lw=0.5, ls=":")
    a.annotate("Mach 0.85", (1, 343.2 * 0.85 + 2), fontsize=8)
    a.set_title("Speed along the flight (lines: MuJoCo shots, red hit)")
    a.set_xlabel("distance [m]")
    a.set_ylabel("m/s")
    a.grid(alpha=0.3)
    a.legend()
    a = ax[1, 1]
    EL = 0.5 * m * vL ** 2
    a.hist(EL, bins=40, color="tab:orange", alpha=0.7, density=True)
    a.set_title(f"Energy at {distance:.0f} m: mean {EL.mean():.1f} J, 5-95 % {np.percentile(EL, 5):.1f}-{np.percentile(EL, 95):.1f} J"
                f" (muzzle {0.5 * m * muzzle.mean() ** 2:.0f} J)")
    a.set_xlabel("J")
    for b in ax.flat:
        b.grid(alpha=0.3)
    fig.suptitle(f"Recoilless 8 mm emitter ({br.TARGET_V:.0f} m/s, scatter {100 * br.SCATTER_SPEED:.1f} % / {br.SCATTER_DEG} deg): "
                 f"speed distribution", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--distance", type=float, default=100.0)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--gust", type=float, default=0.0)
    ap.add_argument("--out", default="bearing_speed_dist.png")
    a = ap.parse_args()
    xs, muzzle, speeds = monte_carlo(a.n, a.distance, a.gust)
    mj = mujoco_speeds(a.distance, a.gust)
    vL = speeds[:, -1]
    print(f"muzzle {muzzle.mean():.1f} ± {muzzle.std():.2f} m/s; at {a.distance:.0f} m {vL.mean():.1f} ± {vL.std():.2f} m/s "
          f"(5-95 %: {np.percentile(vL, 5):.1f}-{np.percentile(vL, 95):.1f}); MuJoCo: muzzle {mj[0].mean():.1f} ± {mj[0].std():.2f}, "
          f"at {a.distance:.0f} m {mj[1].mean():.1f} ± {mj[1].std():.2f}")
    print("wrote", plot(xs, muzzle, speeds, mj, a.distance, a.out))
