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


class Progress:
    """Two tqdm bars on stderr: the chapter's cases (done / planned, with the cache hits and the case in hand) and,
    inside a case, the solver's own progress as the tool reports it (OpenFOAM's iteration out of the case's
    iteration count, Talos's stages). Off when stderr is not a terminal (a detached run: run.log has everything) or
    with ``--no-progress``; the run's log lines go through ``tqdm.write`` so they never break a bar."""

    def __init__(self, run: runkit.Run, total: int, *, enabled: bool = True, desc: str = "cases"):
        self.enabled = enabled and sys.stderr.isatty()
        self.outer = self.inner = None
        self.run = run
        self.hits = 0
        if not self.enabled:
            return
        from tqdm import tqdm
        self._tqdm = tqdm
        self.outer = tqdm(total=total, desc=desc, unit="case", dynamic_ncols=True, position=0, leave=True,
                          bar_format="{desc}: {n_fmt}/{total_fmt} |{bar}| {elapsed}<{remaining} {postfix}")
        run.echo_fn = lambda line: tqdm.write(line)

    def start_case(self, name: str) -> None:
        if self.outer is not None:
            self.outer.set_postfix_str(name[:50], refresh=True)

    def case_done(self, status: str, *, cached: bool = False) -> None:
        if self.outer is None:
            return
        if cached:
            self.hits += 1
        self.outer.update(1)
        self.outer.set_postfix_str(f"last {status}" + (f", {self.hits} from cache" if self.hits else ""), refresh=True)
        self.end_solver()

    def solver(self, stage: str, fraction, message: str = "") -> None:
        """A tool's progress callback ``(stage, fraction, message)`` → the inner bar (0–100 %)."""
        if not self.enabled:
            return
        if self.inner is None:
            self.inner = self._tqdm(total=100, desc="  solver", position=1, leave=False, dynamic_ncols=True,
                                    bar_format="{desc}: {percentage:3.0f}%|{bar}| {elapsed} {postfix}")
        self.inner.set_description_str(f"  {stage}"[:24])
        if fraction is not None:
            self.inner.update(max(0.0, min(100.0, fraction * 100)) - self.inner.n)
        if message:
            self.inner.set_postfix_str(message[:40], refresh=False)
        self.inner.refresh()

    def end_solver(self) -> None:
        if self.inner is not None:
            self.inner.close()
            self.inner = None

    def close(self) -> None:
        self.end_solver()
        if self.outer is not None:
            self.outer.close()
            self.run.echo_fn = print


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
          cache: runkit.Cache | None = None, progress: "Progress | None" = None) -> str:
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
                if progress:
                    progress.case_done(rec.status, cached=True)
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
        if progress:
            progress.start_case(name)
        rec = run.run_case(name, keys[name], lambda d, cancel, n=name: fn(n, d, cancel), rerun_failed=rerun_failed,
                           case_dir=(case_dirs or {}).get(name))
        if progress:
            progress.case_done(rec.status, cached=rec.reused)
        if cache is not None and not rec.reused:
            cache.put(keys[name], {**rec.to_dict(), "run_id": run.manifest["run_id"], "run": str(run.root)})
    if run.cancel.is_set():
        return "cancelled"
    if any(r.status == "NOT RUN" and r.note.startswith("not started") for r in run.records.values()):
        return "partial"
    return "completed"


def closing(run: runkit.Run, code: int, *, hint: str, progress: "Progress | None" = None) -> None:
    if progress:
        progress.close()
    print(f"\nrun directory: {run.root}\n  status:  cat {run.root}/status.json\n  log:     tail -F {run.root}/run.log\n  {hint}")
    run.close()
    sys.exit(code)
