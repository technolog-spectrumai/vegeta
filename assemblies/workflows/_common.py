"""What every workflow does with a node: add one whose parameters needed earlier results, solve FEA models onto it,
run a CFD case onto it. Solved cases are read back (``talos.solve_models``, ``aeromant.run_cases``)."""
from __future__ import annotations

from pathlib import Path

from vegeta import aeromant, talos

from ..vida import Assembly

FEA_KEEP = {"max_von_mises": "max_von_mises_MPa", "safety_factor_yield": "safety_factor",
            "max_displacement": "max_displacement_mm", "n_elements": "elements"}


def add_after(root: Assembly, node: Assembly, prior: Assembly | None, redo=()) -> Assembly:
    """Add ``node`` to ``root`` and take its results from the saved tree when its key is unchanged (``redo`` names
    nodes to compute again)."""
    root.add(node)
    if node.name not in redo:
        root.reuse(prior)
    return node


def solve_fea(node: Assembly, models: dict[str, talos.StructuralModel], out: Path, *, run: bool, threads: int = 1,
              progress=True) -> Assembly:
    """Solve the models in ``out/<name>`` and record ``stress`` (one row per model) and ``complete`` on ``node``."""
    out = Path(out)
    rs = talos.solve_models(list(models.values()), [out / k for k in models], threads=threads, run=run, progress=progress)
    rows = {}
    for name, r in zip(models, rs):
        m = r.metrics if r.ok else {}
        rows[name] = {"ok": bool(r.ok), **{v: m.get(k) for k, v in FEA_KEEP.items()}}
        if (out / name).exists():
            node.attach(name, out / name, "mesh")
        if not r.ok and run:
            node.not_run(f"{name}: {r.messages[-1] if r.messages else r.status}")
    if any(v["ok"] for v in rows.values()):
        node.record(stress=rows, complete=all(v["ok"] for v in rows.values()))
        if not run and not all(v["ok"] for v in rows.values()):
            node.not_run("run_fea=False")
    elif not run:
        node.not_run("run_fea=False")
    return node


def run_cfd(node: Assembly, case: aeromant.CFDCase, keep, *, run: bool, processors: int = 1, progress=True) -> Assembly:
    """Run (or read back) one case and record ``forces`` (the metrics named in ``keep``) and ``complete``."""
    r = aeromant.run_cases([case], processors=processors, run=run, progress=progress)[0]
    if case.workdir.exists():
        node.attach(case.workdir.name, case.workdir, "cases")
    if r.ok:
        node.record(complete=True, forces={k: r.metrics.get(k) for k in keep})
    else:
        node.not_run("run_cfd=False" if not run else f"failed: {r.messages[-1] if r.messages else r.status}")
    return node
