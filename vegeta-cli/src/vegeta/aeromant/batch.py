"""Many cases with one call: each solved only when needed, several at a time, resumable."""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

from . import cache as _cache
from .case import CFDCase
from .result import Result


def run_cases(cases: Sequence[CFDCase], *, jobs: int = 1, processors: int = 1, run: bool = True, progress=False,
              cancel: threading.Event | None = None, timeout: float | None = None, cache=None) -> list[Result]:
    """``CFDCase.ensure`` on every case, in the order given; the results come back in that order.

    Cases already solved with the same inputs are read back first (seconds). The others run ``jobs`` at a time, each
    solver with ``processors`` MPI ranks (so ``jobs * processors`` cores in all). A failed case is a failed result in
    the list, never an exception, and does not stop the others. Interrupted: call again with the same cases, the
    solved ones are read back and the rest run. ``run=False``: nothing runs, unsolved cases are NOT RUN results.
    ``progress=True`` shows a bar over the cases (the solvers' own progress is off when ``jobs > 1``).
    ``cache`` (``vegeta.cache``): None (off), True (each case's workdir name is its entry) or one entry name (or None)
    per case. A case whose entry exists is loaded and not run; a finished one is saved as its entry.
    """
    if jobs < 1:
        raise ValueError("jobs must be >= 1")
    workdirs = [c.workdir.resolve() for c in cases]
    if len(set(workdirs)) != len(workdirs):
        raise ValueError("two cases share a workdir; give each case its own directory")
    names = _cache.entry_names(cache, [c.workdir.name for c in cases], "case")
    results: list[Result | None] = [None] * len(cases)
    todo = []
    for i, case in enumerate(cases):
        r = case.ensure(run=False, cache=names[i])
        if r.ok or not run:
            results[i] = r
        else:
            todo.append(i)
    bar = None
    if progress and todo:
        from tqdm.auto import tqdm
        bar = tqdm(total=len(cases), initial=len(cases) - len(todo), desc="CFD cases", unit="case")

    def one(i: int) -> None:
        if cancel is not None and cancel.is_set():
            res = Result(kind="aeromant.run", metadata={"case": cases[i].config(), "reused": False})
            results[i] = res.fail("cancelled before it started", status="cancelled")
        else:
            results[i] = cases[i].ensure(run=True, processors=processors, cancel=cancel, timeout=timeout,
                                         progress=bool(progress) and jobs == 1, cache=names[i])
        if bar is not None:
            bar.update(1)

    try:
        if jobs == 1:
            for i in todo:
                one(i)
        else:
            with ThreadPoolExecutor(max_workers=jobs) as pool:      # the solvers are external processes
                list(pool.map(one, todo))
    finally:
        if bar is not None:
            bar.close()
    return results  # type: ignore[return-value]
