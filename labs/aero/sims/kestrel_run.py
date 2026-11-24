#!/usr/bin/env python3
"""Kestrel-4Q-2N (the net launcher in the pod): every simulation from one script, under one progress bar
(``labs/aero/sims/kestrel.sh``). The phases, in order: the CFD campaigns (OpenFOAM), the FEA (CalculiX), the
reduced-order flight checks (no solver). One bar counts every CFD case, FEA case and stability job together, with
the time left; the chapter scripts' own case cache keeps finished work, so a restart continues.

    labs/aero/sims/kestrel.sh -j 8                    # everything on 8 cores: CFD (stall, sweep, cruise, transitions), FEA, stability
    labs/aero/sims/kestrel.sh -j 8 --cfd              # one phase: --cfd | --fea | --stability (any combination, always in that order)
    labs/aero/sims/kestrel.sh -j 8 --cfd --quality fine
    labs/aero/sims/kestrel.sh --dry-run               # list the cases and jobs, run nothing

``-j N`` is the cores: OpenFOAM's MPI ranks per CFD case and CalculiX's threads (``-n`` / ``--threads`` override one of
them). The CFD and FEA go through the side-labs chapter scripts (``labs/aero/campaigns/side_labs/kestrel/``), so every
run lands under ``side_labs/runs/kestrel/<chapter>/<run id>/`` with its manifest, log and summary, is restartable
(``--resume <run dir>``) and reuses solved cases from the book's cache. The stability phase is
``kestrel_netter_stability.py``. Each phase writes ONE timestamped file ``output/kestrel_netter/runs/kestrel_<phase>_<time>_<host>.json``
(the run's summary; paste them back). A phase that fails does not stop the next.
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import datetime as dt
import json
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DESIGNS = ROOT / "labs" / "aero" / "designs"
SIDE = ROOT / "labs" / "aero" / "campaigns" / "side_labs"
for _p in (str(SIDE), str(DESIGNS), str(ROOT / "labs" / "aero" / "exp"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT = HERE / "output" / "kestrel_netter"
BOOK = "kestrel"
GLOBAL_BAR = None


def _clean(x):
    return json.loads(json.dumps(x, default=str))


def _say(*x):
    print(" ".join(str(v) for v in x))


def _write(phase, doc) -> Path:
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    p = OUT / "runs" / f"kestrel_{phase}_{stamp}_{socket.gethostname()}.json"
    p.write_text(json.dumps(_clean(doc), indent=1) + "\n")
    return p


def _meta(phase, a):
    return {"aircraft": f"Kestrel-{a.variant}", "phase": phase, "started": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "host": socket.gethostname(), "commit": os.popen("git rev-parse --short HEAD 2>/dev/null").read().strip(), "cpu": a.cpu}


def _env():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(SIDE), str(DESIGNS), str(ROOT / "labs" / "aero" / "exp")] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return env


# ------------------------------------------------------------------------------------------------ the units
def units(a) -> dict:
    """How many bar units each phase has: CFD cases per campaign, FEA cases, stability jobs."""
    import kestrel_campaigns as kc
    import library
    n = {"cfd": 0, "fea": 0, "stability": 0, "cfd_by_campaign": {}}
    veh = None
    if "cfd" in a.phases or "fea" in a.phases:
        veh = kc.vehicle(a.variant)
    if "cfd" in a.phases:
        for c in a.campaigns:
            specs = kc.cfd_specs(a.variant, c, veh=veh)
            n["cfd_by_campaign"][c] = len(specs)
            n["cfd"] += len(specs)
            if a.dry_run:
                ch = library.chapter_of(BOOK, "cfd", c) or c
                _say(f"cfd  {c:12s} ({ch}): {len(specs)} case(s)  — " + ", ".join(list(specs)[:6]) + (" …" if len(specs) > 6 else ""))
    if "fea" in a.phases:
        cases = kc.fea_cases(a.variant, "all", tempfile.mkdtemp(prefix="kestrel_fea_count_"), veh=veh, element_size=a.element_size)
        n["fea"] = len(cases)
        if a.dry_run:
            _say(f"fea  frame-and-joint: {len(cases)} case(s)  — " + ", ".join(c.name for c in cases))
    if "stability" in a.phases:
        import kestrel_netter_stability as KS
        n["stability"] = KS.units(quick=a.quick, matrix=a.matrix) if a.variant == "4Q-2N" else 0
        if a.dry_run:
            _say(f"stability: {n['stability']} job(s)" + ("" if a.variant == "4Q-2N" else f"  (written for 4Q-2N; nothing for {a.variant})"))
    return n


# ------------------------------------------------------------------------------------------------ a chapter under the bar
def _chapter(a, kind: str, campaign: str, extra: list, total: int, label: str) -> dict:
    """Run one side-labs chapter as a subprocess with its own run directory; its log lines go above the bar, its
    status.json (cases done / reused) moves the bar; the run's summary is the result."""
    import library
    script = {"cfd": SIDE / "kestrel" / "airframe_cfd.py", "fea": SIDE / "kestrel" / "structure_fea.py"}[kind]
    chapter = library.chapter_of(BOOK, kind, campaign) or campaign
    out = library.run_dir(BOOK, chapter, f"{a.variant}_{a.quality}" if kind == "cfd" else a.variant)
    cmd = [a.python, str(script), "--variant", a.variant, "--chapter", campaign, "--out", str(out), "--no-progress"] + extra
    _say(f"---- {label}: {' '.join(cmd[1:])}")
    if a.dry_run:
        return {"dry_run": True}
    bar = GLOBAL_BAR
    base = bar.n if bar is not None else 0
    seen = 0
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        if line.strip():
            _say(f"  {line[:200]}")
        s = _read(out / "status.json")
        c = (s or {}).get("cases") or {}
        done = min(total, int(c.get("done", 0) or 0) + int(c.get("reused", 0) or 0))
        if bar is not None and done > seen:
            bar.update(done - seen)
            bar.set_postfix_str(f"{label}: {done}/{total}"[:70])
        seen = max(seen, done)
    rc = proc.wait()
    if bar is not None and bar.n < base + total:
        bar.update(base + total - bar.n)
    status, summary, manifest = _read(out / "status.json"), _read(out / "summary.json"), _read(out / "manifest.json")
    res = {"run_dir": str(out), "exit": rc, "state": (status or {}).get("state"), "cases": (status or {}).get("cases"),
           "summary": summary, "partial": _read(out / "summary.partial.json"), "minutes": round((time.time() - t0) / 60, 1),
           "design": ((manifest or {}).get("design") or {}).get("name")}
    _say(f"     {label}: {res['state']} ({res['minutes']} min), run {out.relative_to(ROOT)}")
    return res


def _read(p: Path):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


# ------------------------------------------------------------------------------------------------ the phases
def phase_cfd(a, n) -> int:
    results = {}
    for c in a.campaigns:
        results[c] = _chapter(a, "cfd", c, ["--quality", a.quality, "--processors", str(a.procs)], n["cfd_by_campaign"].get(c, 0), f"cfd {c}")
    if a.dry_run:
        return 0
    run = _write("cfd", {"run": {**_meta("cfd", a), "quality": a.quality, "campaigns": a.campaigns}, "results": results})
    _say("->", run)
    return 0 if all(r.get("state") == "completed" for r in results.values()) else 1


def phase_fea(a, n) -> int:
    res = _chapter(a, "fea", "all", ["--threads", str(a.threads), "--element-size", str(a.element_size)], n["fea"], "fea frame+joint")
    if a.dry_run:
        return 0
    run = _write("fea", {"run": _meta("fea", a), "results": {"frame-and-joint": res}})
    _say("->", run)
    return 0 if res.get("state") == "completed" else 1


def phase_stability(a, n) -> int:
    if a.variant != "4Q-2N":
        _say(f"stability: the reduced-order checks are written for 4Q-2N (kestrel_netter_stability.py); for {a.variant} run notebook 43 (kestrel_study)")
        return 0
    import kestrel_netter_stability as KS
    argv = ["--out", str(OUT)] + (["--quick"] if a.quick else []) + (["--matrix"] if a.matrix else [])
    _say(f"---- stability: kestrel_netter_stability.py {' '.join(argv)}")
    if a.dry_run:
        return 0
    bar = GLOBAL_BAR

    def progress(label):
        if bar is not None:
            bar.update(1)
            bar.set_postfix_str(f"stability: {label}"[:70])
    rc = KS.main(argv, progress=progress)
    doc = _read(OUT / "stability.json") or {}
    run = _write("stability", {"run": _meta("stability", a), "results": doc})
    _say("->", run)
    return rc


# ------------------------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="kestrel.sh", formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("-j", "--cpu", type=int, default=4, help="cores: MPI ranks per CFD case and CalculiX threads (default 4)")
    ap.add_argument("-n", "--processors", type=int, default=None, help="cfd: MPI ranks per case (default -j)")
    ap.add_argument("--threads", type=int, default=None, help="fea: CalculiX threads (default -j)")
    ap.add_argument("--cfd", action="store_true", help="the CFD campaigns")
    ap.add_argument("--fea", action="store_true", help="the FEA (frame + joint)")
    ap.add_argument("--stability", action="store_true", help="the reduced-order flight checks")
    ap.add_argument("--quality", default="screening", choices=("screening", "fine"), help="cfd: mesh quality")
    ap.add_argument("--campaigns", default="stall sweep cruise transitions", help="cfd: the campaigns, in order (default: stall sweep cruise transitions)")
    ap.add_argument("--element-size", type=float, default=2.0, help="fea: the frame's mesh [mm]")
    ap.add_argument("--variant", default="4Q-2N", choices=("4Q-2N", "4Q-1", "4Q-2", "4Q-3"))
    ap.add_argument("--quick", action="store_true", help="stability: the nominal transitions only")
    ap.add_argument("--matrix", action="store_true", help="stability: also the back transition's remedy matrix")
    ap.add_argument("--python", default=sys.executable, help="the interpreter for the chapter scripts")
    ap.add_argument("--dry-run", action="store_true", help="list the cases and jobs and stop")
    a = ap.parse_args(argv)
    a.phases = [ph for ph, on in (("cfd", a.cfd), ("fea", a.fea), ("stability", a.stability)) if on] or ["cfd", "fea", "stability"]
    a.campaigns = a.campaigns.split()
    a.procs = a.processors or a.cpu
    a.threads = a.threads or a.cpu
    os.environ["OMP_NUM_THREADS"] = str(a.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    _say(f"Kestrel-{a.variant}: {' → '.join(a.phases)}  (-j {a.cpu}: {a.procs} MPI ranks per CFD case, {a.threads} CalculiX threads; cfd {a.quality})")
    n = units(a)
    total = sum(n[ph] for ph in a.phases)
    fns = {"cfd": phase_cfd, "fea": phase_fea, "stability": phase_stability}
    if a.dry_run:
        for ph in a.phases:
            _say(f"---- {ph}: {n[ph]} unit(s)")
            fns[ph](a, n)
        _say(f"total: {total} units in {len(a.phases)} phase(s)  (dry run: nothing was executed)")
        return 0
    global GLOBAL_BAR
    plain = None
    try:
        from tqdm import tqdm
        import builtins
        GLOBAL_BAR = tqdm(total=total, desc=f"kestrel {a.variant}", unit="unit", dynamic_ncols=True, smoothing=0.05)
        plain = builtins.print
        builtins.print = lambda *x, **k: tqdm.write(" ".join(str(v) for v in x))       # lines above the bar, not through it
    except ImportError:
        _say("(tqdm is not installed in this Python: plain progress lines; pip install tqdm for the bar)")
    t0, ended = time.time(), []
    try:
        for i, ph in enumerate(a.phases, 1):
            _say(f"================ Kestrel-{a.variant} {i}/{len(a.phases)}: {ph}  ({n[ph]} units, -j {a.cpu}) ================")
            t = time.time()
            before = GLOBAL_BAR.n if GLOBAL_BAR is not None else 0
            try:
                rc = fns[ph](a, n)
            except Exception as exc:                                  # noqa: BLE001  a phase that fails does not stop the next
                rc = f"{type(exc).__name__}: {exc}"[:200]
            if GLOBAL_BAR is not None and GLOBAL_BAR.n < before + n[ph]:
                GLOBAL_BAR.update(before + n[ph] - GLOBAL_BAR.n)
            ended.append((ph, rc, time.time() - t))
    finally:
        if GLOBAL_BAR is not None:
            import builtins
            GLOBAL_BAR.close()
            builtins.print = plain
            GLOBAL_BAR = None
    _say(f"================ Kestrel-{a.variant}: done in {(time.time() - t0) / 60:.0f} min ================")
    for ph, rc, dt_ in ended:
        _say(f"  {ph:10s} {'ok' if rc == 0 else 'ended with ' + str(rc)}  ({dt_ / 60:.0f} min)")
    _say(f"  results: {OUT / 'runs'}/kestrel_*.json  (paste the newest of each phase back);  runs: {SIDE / 'runs' / BOOK}/")
    return 0 if all(rc == 0 for _, rc, _ in ended) else 1


if __name__ == "__main__":
    sys.exit(main())
