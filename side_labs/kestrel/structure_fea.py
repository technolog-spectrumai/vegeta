#!/usr/bin/env python3
"""Kestrel structural FEA — the carbon frame and the printed spar–post joint in Talos (Gmsh + CalculiX), one named
campaign per run, written as a run directory.

    .venv/bin/python side_labs/kestrel/structure_fea.py --variant 4Q-2 --chapter all --threads 4
    .venv/bin/python side_labs/kestrel/structure_fea.py --variant 4Q-1 --chapter frame --list
    .venv/bin/python side_labs/kestrel/structure_fea.py --variant 4Q-2 --chapter joint --prepare-only   # CAD + mesh, no solve
    .venv/bin/python side_labs/kestrel/structure_fea.py --resume side_labs/runs/kestrel/<chapter>/<run id>

Book ``kestrel`` (code K), the FEA chapters; campaigns (``kestrel_campaigns.FEA_CAMPAIGNS``, ``--chapter``): ``frame`` — the variant's bracing (X for 4Q-1, posts for 4Q-2 /
4Q-3) under A the 4 g pull-up, B the 75 % wing, C the heading manoeuvre, and the modes with the motor units;
``joint`` — the printed spar–post joint under the post's ultimate axial force; ``all``. A frame case is a few
minutes at 2 mm second-order elements; the whole ``all`` campaign under an hour on four threads.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _kestrel as wk  # noqa: E402
import runkit  # noqa: E402
import importlib  # noqa: E402

WORKFLOW = "structure_fea"
BOOK = "kestrel"
CAMPAIGNS = {"kestrel": "kestrel_campaigns", "kestrel-x": "kestrel_xc_campaigns"}     # the book → its campaign module (the same interface)
kc = None
KIND = "fea"


def main(argv=None, book: str = "kestrel") -> int:
    global BOOK, kc
    BOOK = book
    kc = importlib.import_module(CAMPAIGNS[book])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default=getattr(kc, "DEFAULT_VARIANT", next(iter(kc.VARIANTS))), choices=tuple(kc.VARIANTS))
    wk.chapter_args(ap, BOOK, KIND)
    ap.add_argument("--threads", type=int, default=4, help="CalculiX threads (OMP_NUM_THREADS)")
    ap.add_argument("--n-modes", type=int, default=6)
    ap.add_argument("--element-size", type=float, default=2.0, help="mm, the frame's mesh (second order)")
    runkit.add_common_arguments(ap)
    args = ap.parse_args(argv)

    if args.resume:
        cfg = (runkit.read_json(Path(args.resume) / "manifest.json") or {}).get("config") or {}
        if cfg:
            args.variant, args.chapter, args.element_size, args.n_modes = cfg.get("variant", args.variant), cfg.get("campaign", args.chapter), cfg.get("element_size", args.element_size), cfg.get("n_modes", args.n_modes)
    config = {"workflow": WORKFLOW, "book": BOOK, "kind": KIND, "chapter": wk.library.chapter_of(BOOK, KIND, args.chapter) or args.chapter, "campaign": args.chapter, "variant": args.variant, "element_size": args.element_size, "n_modes": args.n_modes}

    t0 = time.monotonic()
    veh = kc.vehicle(args.variant)
    fp = kc.fingerprint(args.variant, veh)
    if args.list:
        import tempfile
        cases = kc.fea_cases(args.variant, args.chapter, tempfile.mkdtemp(prefix="kestrel_fea_list_"), veh=veh, element_size=args.element_size)
        names = runkit.select_cases([c.name for c in cases], args.cases)
        print(f"{args.variant} / {args.chapter}: {len(names)} case(s) — {kc.FEA_CAMPAIGNS[args.chapter]}")
        for c in cases:
            if c.name in names:
                print(f"  {c.name:18s} {c.part:6s} {c.kind:7s} {c.description}")
        print(f"\nvehicle: {veh.name}, {veh.mass:.3f} kg; materials carbon {kc.kfe.CARBON['strength']} MPa, PETG {kc.kfe.PETG['strength']} / {kc.kfe.PETG['strength_z']} MPa across layers; ultimate ×{kc.kfe.ULTIMATE}")
        return 0

    CHAPTER = wk.library.chapter_of(BOOK, KIND, args.chapter) or args.chapter
    ignored = wk.prerequisites_or_exit(BOOK, CHAPTER, variant=args.variant, expected=fp, ignore=args.ignore_prerequisites)
    if args.resume:
        run = runkit.Run.open(args.resume, config=config, heartbeat_s=args.heartbeat, echo=not args.quiet)
    else:
        out = Path(args.out) if args.out else wk.library.run_dir(BOOK, CHAPTER, args.variant)
        run = runkit.Run.create(out, workflow=WORKFLOW, campaign=args.chapter, config=config, design=fp, planned_cases=[],
                                versions={"calculix": _ccx_version()}, heartbeat_s=args.heartbeat, echo=not args.quiet)
    run.manifest.update(book=BOOK, chapter=CHAPTER, kind=KIND, prerequisites_ignored=ignored, yanus={"needs": wk.GATE["needs"]})
    cache = runkit.Cache(BOOK, enabled=not args.no_cache)
    cases = kc.fea_cases(args.variant, args.chapter, run.root, veh=veh, element_size=args.element_size)   # the STEP files under <run>/cad
    byname = {c.name: c for c in cases}
    names = runkit.select_cases([c.name for c in cases], args.cases)
    bars = wk.Progress(run, len(names), enabled=not (args.no_progress or args.quiet), desc=f"{args.chapter} {args.variant}")
    if not run.manifest.get("planned_cases"):
        run.manifest["planned_cases"] = names
        runkit.write_json_atomic(run.root / "manifest.json", run.manifest)
        run.status(cases={**run.status().get("cases", {}), "planned": len(names)})
    run.log(f"{kc.VARIANTS[args.variant]}; {kc.FEA_CAMPAIGNS[args.chapter]}; {len(names)} case(s); CAD and models built in {time.monotonic() - t0:.0f} s")
    import shutil
    if not shutil.which("ccx") and not args.prepare_only:
        run.finish({"note": "CalculiX (ccx) not found"}, state="failed", note="CalculiX (ccx) not found on PATH")
        wk.closing(run, 1, hint="install CalculiX (docs/installation.md) or use --prepare-only (CAD + mesh)")
    keys = {n: kc.fea_case_key(byname[n]) for n in names}
    budget = runkit.Budget(args.max_cases, args.max_minutes, args.case_timeout)

    def one(name, case_dir, cancel):
        c = byname[name]
        prog = bars.solver
        if args.prepare_only:
            m = c.model.mesh(case_dir, progress=prog)
            if not m.ok:
                return kc.fea_outcome(c, m)
            return {"status": "NOT RUN", "process": None, "note": f"meshed only (--prepare-only): {m.metrics.get('nodes')} nodes, {m.metrics.get('elements')} elements",
                    "artifacts": {k: str(v) for k, v in m.artifacts.items()}}
        if c.kind == "static":
            res = c.model.ensure(case_dir, run=True, threads=args.threads, timeout=budget.case_timeout_s, cancel=cancel, progress=prog)
        else:
            if not c.model.mesh_is_current(case_dir):
                m = c.model.mesh(case_dir, progress=prog)
                if not m.ok:
                    return kc.fea_outcome(c, m)
            res = c.model.solve_modes(case_dir, n_modes=args.n_modes, threads=args.threads, timeout=budget.case_timeout_s, cancel=cancel, progress=prog)
        return kc.fea_outcome(c, res)

    with run.signals():
        run.status("running")
        state = wk.drive(run, names, keys, one, budget=budget, rerun_failed=args.rerun_failed, cache=None if args.prepare_only else cache, progress=bars)
    if args.prepare_only:
        state = "prepared"
    summary = kc.fea_summary(variant=args.variant, campaign=args.chapter, cases=[byname[n] for n in names], records=run.records, veh=veh, run_root=run.root)
    bars.close()
    code = run.finish(summary, state=state)
    import kestrel_keys
    pub = wk.publish(run, kestrel_keys.structure(args.variant, args.chapter, element_size=args.element_size, n_modes=args.n_modes,
                                                 mock=bool(run.manifest.get("mock")), fingerprint=summary["design"]))
    wk.closing(run, code, hint=wk.pin_hint(run) + "\n  " + pub)
    return code


def _ccx_version():
    import shutil
    import subprocess
    if not shutil.which("ccx"):
        return None
    try:
        out = subprocess.run(["ccx", "-v"], capture_output=True, text=True, timeout=10)
        return (out.stdout + out.stderr).strip().splitlines()[-1][:80]
    except Exception:
        return None


if __name__ == "__main__":
    sys.exit(main())
