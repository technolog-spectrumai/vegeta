"""What every workflow does with a node: add one whose parameters needed earlier results, solve FEA models onto it,
run a CFD case onto it, solve the modes and unit load cases of a structure on one mesh. Solved cases are read back
(``talos.solve_models``, ``aeromant.run_cases``)."""
from __future__ import annotations

import shutil
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


def sub(tree: Assembly | None, path: str) -> Assembly | None:
    """The sub-assembly ``path`` of a saved tree, or None (no saved tree, or it has no such node)."""
    if tree is None:
        return None
    try:
        return tree.child(path)
    except KeyError:
        return None


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


def unit_fea(node: Assembly, base: talos.StructuralModel, unit: dict, out: Path, *, n_modes: int, run: bool,
             threads: int = 1, progress=True) -> dict:
    """The structure's modes and its unit load cases on one mesh (notebook 08 cells 82, 84, 90; 09b cells 8, 10, 15).

    ``base`` is the modal model (its point masses, no loads); ``unit`` maps a case name to ``(model, load_level)``,
    every model on the same geometry, regions and mesh settings. ``base`` is meshed once into ``out/mesh``, the meshed
    directory is copied for the modal solve and for each case, the cases are solved with ``talos.solve_models``.
    Records ``modes_hz``, ``unit`` (one row per case) and ``complete``; returns ``{name: (Result, load)}`` (the input
    ``talos.assess_fatigue`` takes), empty when nothing was solved. A complete node is not solved again: its unit
    results are read back from the case directories (``StructuralModel.ensure(run=False)``)."""
    out = Path(out)
    names = list(unit)
    if node.results.get("complete"):
        back = {k: (unit[k][0].ensure(out / k, run=False), unit[k][1]) for k in names}
        return back if all(r.ok for r, _ in back.values()) else {}
    if not run:
        node.not_run("run_fea=False")
        return {}
    mesh_dir = out / "mesh"
    if not base.mesh_is_current(mesh_dir):
        m = base.mesh(mesh_dir, progress=progress)
        if not m.ok:
            node.not_run(f"mesh: {m.messages[-1] if m.messages else m.status}")
            return {}

    def case_dir(name):                                   # a copy of the meshed directory per analysis
        d = out / name
        if not d.exists():
            shutil.copytree(mesh_dir, d)
        return d

    modes = base.solve_modes(case_dir("modal"), n_modes=n_modes, threads=threads, progress=progress)
    rs = talos.solve_models([unit[k][0] for k in names], [case_dir(k) for k in names], threads=threads, run=True,
                            progress=progress)
    rows = {}
    for k, r in zip(names, rs):
        m = r.metrics if r.ok else {}
        rows[k] = {"load": unit[k][1], "ok": bool(r.ok), "max_von_mises_MPa": m.get("max_von_mises"),
                   "max_displacement_mm": m.get("max_displacement"), "elements": m.get("n_elements")}
        node.attach(k, out / k, "mesh")
        if not r.ok:
            node.not_run(f"{k}: {r.messages[-1] if r.messages else r.status}")
    node.attach("modal", out / "modal", "mesh")
    if not modes.ok:
        node.not_run(f"modes: {modes.messages[-1] if modes.messages else modes.status}")
    complete = bool(modes.ok) and all(v["ok"] for v in rows.values())
    node.record(modes_hz=list(modes.metrics.get("frequencies_hz", [])) if modes.ok else [], unit=rows, complete=complete,
                point_mass_total=modes.metrics.get("point_mass_total") if modes.ok else None)
    return {k: (r, unit[k][1]) for k, r in zip(names, rs)} if complete else {}
