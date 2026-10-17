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
import library  # noqa: E402


def chapter_args(ap, book: str, kind: str) -> None:
    """``--chapter`` (``--campaign`` is kept as an alias): this script's campaigns, each a chapter of the book."""
    camps = [c["campaign"] for c in library.BOOKS[book]["chapters"].values() if c["kind"] == kind]
    ap.add_argument("--chapter", "--campaign", dest="chapter", default=camps[0], choices=tuple(camps),
                    help=f"the study (the {kind.upper()} chapters of book {book!r}: workflows/cli.py list)")


def prerequisites_or_exit(book: str, chapter: str, *, variant: str, expected: dict, ignore: bool) -> list[str]:
    missing = library.check_prerequisites(book, chapter, variant=variant, expected=expected)
    if missing and not ignore:
        print(f"chapter {library.chapters(book)[chapter]['code']} ({chapter}) of book {book!r} needs a completed compatible run of its prerequisite chapter(s) first:")
        for m in missing:
            print("  - " + m)
        print("(or --ignore-prerequisites, recorded in the manifest)")
        sys.exit(2)
    return missing


def drive(run: runkit.Run, names: list[str], keys: dict, fn, *, budget: runkit.Budget, rerun_failed: bool = False, case_dirs: dict | None = None,
          cache: runkit.Cache | None = None) -> str:
    """Run the cases in order under the budget; returns the run's end state: ``completed`` (every case attempted),
    ``partial`` (the budget stopped it with cases left) or ``cancelled``. With a ``cache`` a case this book solved
    in an earlier run (same key) is read back, not solved; a case solved here is put in it."""
    t0 = time.monotonic()
    started = 0
    for name in names:
        if cache is not None and run.reusable(name, keys[name], rerun_failed=rerun_failed) is None and not run.cancel.is_set():
            hit = cache.get(keys[name])
            if hit is not None:
                rec = runkit.CaseRecord(**{k: v for k, v in hit.items() if k in runkit.CaseRecord.__dataclass_fields__})
                rec.name, rec.key, rec.reused = name, keys[name], True
                rec.note = (f"from the cache: solved in run {hit.get('run_id')} on {hit.get('finished_at')}" + (f"; {rec.note}" if rec.note else ""))
                run.records[name] = rec
                d = run.case_dir(name) if not (case_dirs or {}).get(name) else Path(case_dirs[name])
                d.mkdir(parents=True, exist_ok=True)
                runkit.write_json_atomic(d / "case.json", rec.to_dict())
                c = dict(run.status().get("cases", {}))
                c["reused"] = c.get("reused", 0) + 1
                c["done"] = c.get("done", 0) + 1
                run.status(cases=c)
                run.log(f"case {name}: cache hit ({rec.status}, run {hit.get('run_id')}); native files in {rec.artifacts.get('case', '?')}")
                continue
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
        rec = run.run_case(name, keys[name], lambda d, cancel, n=name: fn(n, d, cancel), rerun_failed=rerun_failed,
                           case_dir=(case_dirs or {}).get(name))
        if cache is not None and not rec.reused:
            cache.put(keys[name], {**rec.to_dict(), "run_id": run.manifest["run_id"], "run": str(run.root)})
    if run.cancel.is_set():
        return "cancelled"
    if any(r.status == "NOT RUN" and r.note.startswith("not started") for r in run.records.values()):
        return "partial"
    return "completed"


def closing(run: runkit.Run, code: int, *, hint: str) -> None:
    print(f"\nrun directory: {run.root}\n  status:  cat {run.root}/status.json\n  log:     tail -F {run.root}/run.log\n  {hint}")
    run.close()
    sys.exit(code)
