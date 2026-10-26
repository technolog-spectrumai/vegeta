"""Scientific validation benchmarks: refinement series against exact solutions and published data.

    python benchmarks/validate.py [CASE ...] [--backend numpy|triton] [--out results/validation] [--quick]

CASE is any of: uniform, taylor_green, poiseuille, channel, cavity, sphere (default: all). Each case writes
``<out>/<case>.json`` (every metric, the grids, the wall time, the platform) and plots where they help. The
automated tests (tests/) are regression guards on tiny grids; these runs are the record behind the numbers in
docs/validation.md. Nothing here is a claim beyond what the stated reference supports.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mancha import __version__  # noqa: E402
from mancha.diagnostics import observed_order  # noqa: E402
from mancha.validation import cavity, channel, poiseuille, references, sphere, taylor_green, uniform  # noqa: E402


def _clean(d: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in d.items():
        if k == "simulation":
            continue
        if isinstance(v, np.ndarray):
            v = v.tolist()
        elif isinstance(v, tuple):
            v = [x.tolist() if isinstance(x, np.ndarray) else x for x in v]
        elif isinstance(v, np.generic):
            v = v.item()
        out[k] = v
    return out


def _timed(fn, **kw) -> dict[str, Any]:
    t0 = time.perf_counter()
    r = fn(**kw)
    r["wall_time_s"] = time.perf_counter() - t0
    return r


def run_uniform(backend: str, quick: bool, out: Path) -> dict[str, Any]:
    n = 16 if quick else 32
    rest = _timed(uniform, shape=(n, n, n), steps=1000, backend=backend)
    flow = _timed(uniform, shape=(n, n, n), velocity=(0.05, 0.03, -0.02), steps=1000, backend=backend)
    return {"rest": _clean(rest), "flow": _clean(flow)}


def run_taylor_green(backend: str, quick: bool, out: Path) -> dict[str, Any]:
    ns = [16, 32, 64] if quick else [16, 32, 64, 128]
    rows = [_clean(_timed(taylor_green, n=n, viscosity=0.02, u0=0.04, backend=backend)) for n in ns]
    errors = [abs(r["decay_rate_error_rel"]) for r in rows]
    result = {"note": "2D Taylor-Green vortex at fixed tau and lattice velocity (diffusive scaling of the "
                      "time is not applied: Ma fixed, so the error is the spatial one plus a fixed Ma^2 part)",
              "runs": rows, "observed_order_decay_rate": observed_order(ns, errors),
              "observed_order_velocity_l2": observed_order(ns, [r["velocity_l2_error_rel"] for r in rows])}
    _plot_convergence(ns, errors, "Taylor-Green decay-rate error", out / "taylor_green_convergence.png")
    return result


def run_poiseuille(backend: str, quick: bool, out: Path) -> dict[str, Any]:
    ns = [8, 16, 32] if quick else [8, 16, 32, 64]
    rows = []
    for tau in (0.8, 1.0, 1.5):
        nu = (tau - 0.5) / 3.0
        for n in ns:
            r = _clean(_timed(poiseuille, ny=n, viscosity=nu, u_max=0.02, backend=backend,
                              max_steps=400000, tol=1e-10))
            r["tau"] = tau
            for k in ("profile", "exact", "y"):
                r.pop(k)
            rows.append(r)
    orders = {tau: observed_order(ns, [r["l2_error_rel"] for r in rows if r["tau"] == tau]) for tau in (0.8, 1.0, 1.5)}
    last = poiseuille(ny=ns[-1], viscosity=1 / 6, u_max=0.02, backend=backend, tol=1e-10, max_steps=400000)
    _plot_profile(last["y"] / ns[-1], last["profile"] / 0.02, last["exact"] / 0.02,
                  "Poiseuille (body force), tau = 1", out / "poiseuille_profile.png", "y / h", "u / u_max")
    return {"note": "halfway bounce-back + BGK: the wall slip depends on tau, so the error is second order "
                    "at fixed tau and vanishes at tau = 1/2 + sqrt(3/16)", "runs": rows,
            "observed_order_l2": {str(k): v for k, v in orders.items()}}


def run_channel(backend: str, quick: bool, out: Path) -> dict[str, Any]:
    ny = 16 if quick else 32
    r = _timed(channel, nx=8 * ny, ny=ny, viscosity=0.1, u_in=0.02, backend=backend, max_steps=400000)
    y = (np.arange(ny) + 0.5) / ny
    _plot_profile(y, r["profile"] / r["profile"].mean(), r["exact"] / r["profile"].mean(),
                  f"Inlet/outlet channel, profile at x = {r['station']}", out / "channel_profile.png",
                  "y / h", "u / mean(u)")
    return _clean({k: v for k, v in r.items() if k not in ("profile", "exact")})


def run_cavity(backend: str, quick: bool, out: Path) -> dict[str, Any]:
    ns = [32, 64] if quick else [32, 64, 128]
    rows = []
    for n in ns:
        r = _timed(cavity, n=n, reynolds=100.0, u_lid=0.05, backend=backend, tol=1e-8,
                   max_steps=400000)
        rows.append(r)
    best = rows[-1]
    s, u = best["u_centreline"]
    _plot_profile(s, u, None, f"Cavity Re=100, u on x = 0.5 (n = {ns[-1]})", out / "cavity_u.png", "y", "u / U",
                  ref=references.GHIA_RE100_U)
    s, v = best["v_centreline"]
    _plot_profile(s, v, None, f"Cavity Re=100, v on y = 0.5 (n = {ns[-1]})", out / "cavity_v.png", "x", "v / U",
                  ref=references.GHIA_RE100_V)
    keep = ("n", "reynolds", "steps", "residual", "vortex_centre", "vortex_centre_error", "ghia_u_max_abs_error",
            "ghia_v_max_abs_error", "mass", "wall_time_s")
    return {"reference": "Ghia, Ghia & Shin (1982), Tables I, II, V", "runs": [_clean({k: r[k] for k in keep})
                                                                                 for r in rows]}


def run_sphere(backend: str, quick: bool, out: Path) -> dict[str, Any]:
    cases = [(8, 20.0)] if quick else [(8, 20.0), (12, 20.0), (12, 100.0), (16, 20.0)]
    result: dict[str, Any] = {
        "note": "periodic lateral boundaries (a square array of spheres 4 D apart) and a voxel staircase: both "
                "raise the drag above the unconfined correlation; compare trends, not values",
        "reference": "Schiller & Naumann (1933) correlation for an unconfined sphere", "runs": []}
    for d, re in cases:
        result["runs"].append(_clean(_timed(sphere, diameter=d, reynolds=re, u_in=0.04, size=(8.0, 4.0, 4.0),
                                            backend=backend, tol=1e-6, max_steps=200000)))
        (out / "sphere.partial.json").write_text(json.dumps(result, indent=1))   # long runs: keep what finished
    return result


def _plot_convergence(ns, errors, title, path):
    try:
        from mancha.io.plots import convergence

        convergence(ns, errors, order=2.0, title=title, path=path)
    except ImportError:          # matplotlib is optional
        pass


def _plot_profile(x, num, exact, title, path, xlabel, ylabel, ref=None):
    try:
        import matplotlib.pyplot as plt

        from mancha.io import plots
    except ImportError:          # matplotlib is optional
        return
    fig = plots.velocity_profile(x, num, exact, xlabel=xlabel, ylabel=ylabel, title=title,
                                 reference_label="analytical")
    if ref is not None:
        ax = fig.axes[0]
        ax.plot(ref[:, 0], ref[:, 1], "s", mfc="none", ms=5, label="Ghia et al. (1982)")
        ax.legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)


CASES = {"uniform": run_uniform, "taylor_green": run_taylor_green, "poiseuille": run_poiseuille,
         "channel": run_channel, "cavity": run_cavity, "sphere": run_sphere}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cases", nargs="*", metavar="CASE", help=f"any of {', '.join(CASES)} (default: all)")
    ap.add_argument("--backend", default="numpy", choices=["numpy", "triton"])
    ap.add_argument("--out", default="results/validation")
    ap.add_argument("--quick", action="store_true", help="coarser series (minutes on a laptop CPU)")
    args = ap.parse_args(argv)
    unknown = sorted(set(args.cases) - set(CASES))
    if unknown:
        ap.error(f"unknown case(s) {unknown}; choose from {list(CASES)}")
    args.cases = args.cases or list(CASES)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.cases:
        t0 = time.perf_counter()
        result = CASES[name](args.backend, args.quick, out)
        record = {"case": name, "backend": args.backend, "quick": args.quick, "mancha_version": __version__,
                  "platform": {"python": platform.python_version(), "machine": platform.machine(),
                               "processor": platform.processor()},
                  "wall_time_s": time.perf_counter() - t0, "result": result}
        (out / f"{name}.json").write_text(json.dumps(record, indent=1))
        print(f"{name}: done in {record['wall_time_s']:.1f} s -> {out / (name + '.json')}", flush=True)


if __name__ == "__main__":
    main()
