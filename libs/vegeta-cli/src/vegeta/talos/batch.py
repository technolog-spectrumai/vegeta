"""Many static analyses with one call: each meshed and solved only when needed, resumable."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Sequence

from . import cache as _cache
from .model import StructuralModel
from .result import Result


def solve_models(models: Sequence[StructuralModel], workdirs: Sequence[str | Path], *, threads: int = 1, run: bool = True,
                 executable: str = "ccx", timeout: float | None = None, progress=False,
                 cancel: threading.Event | None = None, cache=None) -> list[Result]:
    """``StructuralModel.ensure`` on every model in its own workdir; the results come back in the order given.

    Models already solved with the same inputs are read back (seconds); the others are meshed (only when their mesh
    is missing or out of date) and solved one after the other, CalculiX using ``threads`` threads each. Gmsh, which
    meshes and reads the meshes, is one process-wide state, so the models do not run side by side. A failed model is
    a failed result, never an exception. Interrupted: call again, the solved ones are read back. ``run=False``:
    nothing runs, unsolved models are NOT RUN results.

    ``cache`` (``vegeta.cache``): None (off), True (``<parent>/<workdir>`` names each model's entry) or one entry name (or None)
    per model. A model whose entry exists is loaded and not solved; a solved one is saved as its entry.
    """
    if len(models) != len(workdirs):
        raise ValueError("one workdir per model")
    names = _cache.entry_names(cache, [f"{w.parent.name}/{w.name}" for w in workdirs], "model")
    workdirs = [Path(w) for w in workdirs]
    if len({w.resolve() for w in workdirs}) != len(workdirs):
        raise ValueError("two models share a workdir; give each model its own directory")
    results: list[Result | None] = [None] * len(models)
    todo = []
    for i, (m, w) in enumerate(zip(models, workdirs)):
        r = m.ensure(w, run=False, cache=names[i])
        if r.ok or not run:
            results[i] = r
        else:
            todo.append(i)
    bar = None
    if progress and todo:
        from tqdm.auto import tqdm
        bar = tqdm(total=len(models), initial=len(models) - len(todo), desc="FEA models", unit="model")
    try:
        for i in todo:
            if cancel is not None and cancel.is_set():
                results[i] = Result(kind="talos.solve", metadata={"reused": False}).fail("cancelled before it started",
                                                                                      status="cancelled")
            else:
                results[i] = models[i].ensure(workdirs[i], executable=executable, threads=threads, timeout=timeout,
                                              cancel=cancel, cache=names[i])
            if bar is not None:
                bar.update(1)
    finally:
        if bar is not None:
            bar.close()
    return results  # type: ignore[return-value]
