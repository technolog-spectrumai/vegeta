"""Compute-speed chart: Mancha on the CPU (measured) and on GPUs (estimated from memory bandwidth), and the FW-H
post-processing (measured), for the rotor-acoustics job of benchmarks/acoustics/rotor_fwh.py.

    python benchmarks/speed_chart.py [--out results/performance]

Measured numbers come from results/performance/numpy_*.json (benchmarks/performance.py) and
results/performance/fwh_speed.json; GPU numbers are estimates (docs/gpu.md) and are drawn hatched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# GPU estimates (docs/gpu.md): peak bandwidth [B/s]; assumed achieved fractions; bytes per node and step
GPUS = {"T4": 320e9, "RTX 4090": 1.0e12, "A100 80GB": 2.0e12, "H100 SXM": 3.35e12}
PLAIN = (0.5, 0.8, 152.0)
PROPELLER = (0.3, 0.6, 180.0)
NODES, STEPS = 108 * 72 * 72, 8000           # the D = 24 rotor job (steps: an estimate of convergence)
OBSERVERS = 14                               # 5 near-field probes + 9 far-field angles


def estimate(bw, case):
    lo, hi, b = case
    return bw * lo / b / 1e6, bw * hi / b / 1e6   # MLUPS range


def main(argv=None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="results/performance")
    out = Path(ap.parse_args(argv).out)
    cpu = {c: json.loads((out / f"numpy_{c}.json").read_text()) for c in ("periodic", "propeller")}
    cpu64 = {c: next(r["mlups"] for r in d["results"] if r["n"] == 64) for c, d in cpu.items()}
    fwh = json.loads((out / "fwh_speed.json").read_text())["360"]

    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5.2))
    x = np.arange(1 + len(GPUS))
    width = 0.38
    for k, (case_name, case, cpu_key, color) in enumerate((("plain BGK (periodic)", PLAIN, "periodic", "C0"),
                                                           ("propeller physics", PROPELLER, "propeller", "C3"))):
        vals = [cpu64[cpu_key]]
        err_lo, err_hi = [0.0], [0.0]
        for bw in GPUS.values():
            lo, hi = estimate(bw, case)
            mid = np.sqrt(lo * hi)
            vals.append(mid)
            err_lo.append(mid - lo)
            err_hi.append(hi - mid)
        pos = x + (k - 0.5) * width
        bars = a.bar(pos, vals, width, color=color, alpha=0.85, label=case_name,
                     yerr=[err_lo, err_hi], capsize=3, error_kw={"lw": 1})
        for i, bar in enumerate(bars):
            if i > 0:
                bar.set_hatch("//")
                bar.set_alpha(0.45)
            a.text(bar.get_x() + bar.get_width() / 2, vals[i] * (1.9 if i else 1.25),
                   f"{vals[i]:.1f}" if i == 0 else f"{vals[i]:,.0f}", ha="center", fontsize=7)
    a.set_yscale("log")
    a.set_xticks(x, ["CPU, NumPy\n4-core Xeon\nmeasured"] + [f"{g}\nestimated" for g in GPUS], fontsize=8)
    a.set_ylabel("throughput [MLUPS, million node updates / s]")
    a.set_title("Mancha throughput (solid: measured; hatched: estimate from bandwidth, range = whiskers)",
                fontsize=9)
    a.legend(fontsize=8, loc="upper left")
    a.grid(alpha=0.3, axis="y", which="both")

    # time to solution of the rotor acoustics job
    rows = [("Mancha flow solve\nCPU NumPy (measured rate)", NODES * STEPS / (cpu64["propeller"] * 1e6), None, False)]
    for g, bw in GPUS.items():
        lo, hi = estimate(bw, PROPELLER)
        t_hi, t_lo = NODES * STEPS / (lo * 1e6), NODES * STEPS / (hi * 1e6)
        rows.append((f"Mancha flow solve\n{g} (estimate)", np.sqrt(t_lo * t_hi), (t_lo, t_hi), True))
    rows.append((f"FW-H, {OBSERVERS} observers\nCPU (measured)", OBSERVERS * fwh["seconds_per_observer"], None,
                 False))
    rows.append(("FW-H, 1 observer\nCPU (measured)", fwh["seconds_per_observer"], None, False))
    y = np.arange(len(rows))[::-1]
    for yi, (name, val, rng, is_est) in zip(y, rows):
        color = "C2" if name.startswith("FW-H") else "C3"
        b.barh(yi, val, color=color, alpha=0.45 if is_est else 0.85, hatch="//" if is_est else None)
        if rng:
            b.errorbar(val, yi, xerr=[[val - rng[0]], [rng[1] - val]], fmt="none", color="k", capsize=3, lw=1)
        text = f"{val / 3600:.1f} h" if val > 3600 else (f"{val / 60:.1f} min" if val > 60 else f"{val:.2g} s")
        b.text((rng[1] if rng else val) * 1.25, yi, text, va="center", fontsize=8)
    b.set_xscale("log")
    b.set_yticks(y, [r[0] for r in rows], fontsize=8)
    b.set_xlabel("wall time [s]")
    b.set_title(f"Rotor acoustics job, time to solution\nflow: {NODES / 1e6:.2f} M nodes × ~{STEPS} steps; FW-H: "
                f"{fwh['elements']} surface elements × {fwh['source_samples']} samples per observer", fontsize=9)
    b.grid(alpha=0.3, axis="x", which="both")
    b.set_xlim(right=max(r[1] for r in rows) * 20)
    fig.suptitle("Compute speed: Mancha CPU vs GPU, and FW-H — GPU figures are estimates (no GPU run yet)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "speed_chart.png", dpi=140)
    print("->", out / "speed_chart.png")


if __name__ == "__main__":
    main()
