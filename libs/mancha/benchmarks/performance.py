"""Throughput benchmark: MLUPS, wall time, memory and effective bandwidth.

    python benchmarks/performance.py [--backend triton|numpy] [--sizes 64 128 256] [--steps 200]
                                     [--block 128] [--case periodic|channel|sphere] [--out results/performance.json]

For every cubic size: build the case, run ``--warmup`` steps (kernel compilation and caches), then time
``--steps`` steps between device synchronisations. Reported per size:

* MLUPS — million lattice-node updates per second, ``N · steps / t / 1e6`` (solid nodes included);
* wall time of the timed steps;
* memory — solver state (``Backend.memory_bytes``) and, on CUDA, the peak allocated by PyTorch;
* effective bandwidth — ``MLUPS · 1e6 · B`` with ``B = 2 · 19 · 4 = 152`` bytes per node (read and write every
  population once; +4 with solids). It ignores the extra lines the shifted loads touch and the boundary
  re-reads, so it is a lower bound on the traffic; compare it with the GPU's peak bandwidth.

A size that does not fit in device memory is reported as ``skipped`` with the reason, and the run continues.
Results are only what this script measures on the machine it runs on; the GPU name is recorded with them.
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

from mancha import InsufficientMemoryError, Simulation, __version__  # noqa: E402
from mancha.boundaries import sphere  # noqa: E402

BYTES_PER_NODE = 2 * 19 * 4


def build(case: str, n: int, backend: str, block: int | None) -> Simulation:
    kw: dict[str, Any] = {"block_size": block} if backend == "triton" else {}
    sim = Simulation(shape=(n, n, n), viscosity=0.02, backend=backend, velocity=(0.02, 0, 0), **kw)
    if case in ("channel", "sphere"):
        sim.set_boundary("x-", "velocity", velocity=(0.02, 0, 0)).set_boundary("x+", "pressure", density=1.0)
        for face in ("y-", "y+", "z-", "z+"):
            sim.set_boundary(face, "wall")
    if case == "sphere":
        sim.set_solid(sphere(sim.shape, (n / 4, (n - 1) / 2, (n - 1) / 2), n / 8))
    return sim


def measure(case: str, n: int, backend: str, steps: int, warmup: int, block: int | None) -> dict[str, Any]:
    row: dict[str, Any] = {"case": case, "n": n, "nodes": n ** 3}
    torch = None
    if backend == "triton":
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    try:
        sim = build(case, n, backend, block)
    except InsufficientMemoryError as exc:
        return {**row, "skipped": str(exc)}
    sim.run(warmup)                       # compiles the kernel, warms caches; includes the health check
    sim.backend.synchronize()
    t0 = time.perf_counter()
    sim.backend.step(steps)               # no health checks inside the timed region
    sim.backend.synchronize()
    elapsed = time.perf_counter() - t0
    finite, umax = sim.backend.health()
    if not finite:
        raise RuntimeError(f"{case} {n}^3 became non-finite during the benchmark")
    mlups = n ** 3 * steps / elapsed / 1e6
    per_node = BYTES_PER_NODE + (4 if sim.solid is not None else 0)
    row.update(steps=steps, wall_time_s=elapsed, mlups=mlups, state_bytes=sim.backend.memory_bytes(),
               effective_bandwidth_gb_s=mlups * 1e6 * per_node / 1e9, max_speed=umax)
    if torch is not None and torch.cuda.is_available():
        row["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    del sim
    return row


def device_info(backend: str) -> dict[str, Any]:
    info: dict[str, Any] = {"python": platform.python_version(), "machine": platform.machine(),
                            "processor": platform.processor(), "numpy": np.__version__}
    if backend == "triton":
        import torch
        import triton

        info.update(torch=torch.__version__, triton=triton.__version__)
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info.update(gpu=p.name, gpu_memory_bytes=p.total_memory, capability=f"{p.major}.{p.minor}")
        else:
            info["gpu"] = "none (Triton interpreter: timings are meaningless)"
    return info


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--backend", default="triton", choices=["triton", "numpy"])
    ap.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--block", type=int, default=None, help="Triton block size (default: the backend's)")
    ap.add_argument("--case", default="periodic", choices=["periodic", "channel", "sphere"])
    ap.add_argument("--out", default="results/performance.json")
    args = ap.parse_args(argv)
    rows = []
    print(f"{'size':>6} {'MLUPS':>9} {'time [s]':>9} {'state [MiB]':>12} {'eff. BW [GB/s]':>15}")
    for n in args.sizes:
        row = measure(args.case, n, args.backend, args.steps, args.warmup, args.block)
        rows.append(row)
        if "skipped" in row:
            print(f"{n:>5}³ skipped: {row['skipped']}")
        else:
            print(f"{n:>5}³ {row['mlups']:9.1f} {row['wall_time_s']:9.3f} {row['state_bytes'] / 2**20:12.1f} "
                  f"{row['effective_bandwidth_gb_s']:15.1f}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"mancha_version": __version__, "backend": args.backend, "case": args.case,
                               "block": args.block, "device": device_info(args.backend), "results": rows},
                              indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
