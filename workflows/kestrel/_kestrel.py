"""What the Kestrel workflow scripts share: the paths, the case loop under a budget, the closing lines."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "workflows", ROOT / "notebooks" / "designs"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import runkit  # noqa: E402


def drive(run: runkit.Run, names: list[str], keys: dict, fn, *, budget: runkit.Budget, rerun_failed: bool = False, case_dirs: dict | None = None) -> str:
    """Run the cases in order under the budget; returns the run's end state: ``completed`` (every case attempted),
    ``partial`` (the budget stopped it with cases left) or ``cancelled``."""
    t0 = time.monotonic()
    started = 0
    for name in names:
        if run.cancel.is_set():
            left = "cancelled before it started"
        elif budget.max_cases is not None and started >= budget.max_cases:
            left = f"not started: --max-cases {budget.max_cases} reached"
        elif budget.max_minutes is not None and (time.monotonic() - t0) / 60 > budget.max_minutes:
            left = f"not started: --max-minutes {budget.max_minutes:g} reached"
        else:
            left = None
        if left:
            rec = run.records.get(name)
            if rec is None or rec.status not in runkit.FINISHED_CASE:
                run.records[name] = runkit.CaseRecord(name=name, key=keys[name], status="NOT RUN", note=left)
            continue
        prev = run.reusable(name, keys[name], rerun_failed=rerun_failed)
        if prev is None:
            started += 1
        run.run_case(name, keys[name], lambda d, cancel, n=name: fn(n, d, cancel), rerun_failed=rerun_failed,
                     case_dir=(case_dirs or {}).get(name))
    if run.cancel.is_set():
        return "cancelled"
    if any(r.status == "NOT RUN" and r.note.startswith("not started") for r in run.records.values()):
        return "partial"
    return "completed"


def closing(run: runkit.Run, code: int, *, hint: str) -> None:
    print(f"\nrun directory: {run.root}\n  status:  cat {run.root}/status.json\n  log:     tail -F {run.root}/run.log\n  {hint}")
    run.close()
    sys.exit(code)
