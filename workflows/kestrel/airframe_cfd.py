#!/usr/bin/env python3
"""Kestrel airframe CFD — the whole aircraft with the four propellers as rotor disks (Aeromant / OpenFOAM), one
named campaign per run, written as a run directory the unified Kestrel notebook reads.

    .venv/bin/python workflows/kestrel/airframe_cfd.py --variant 4Q-2 --chapter stall --quality fine --processors 4
    .venv/bin/python workflows/kestrel/airframe_cfd.py --variant 4Q-2 --chapter sweep --list
    .venv/bin/python workflows/kestrel/airframe_cfd.py --variant 4Q-2 --chapter blown --prepare-only
    .venv/bin/python workflows/kestrel/airframe_cfd.py --resume workflows/runs/airframe_cfd/<run id>

Book ``kestrel_airframe``; chapters (``kestrel_campaigns.CFD_CAMPAIGNS``, ``--chapter``; ``--campaign`` still works): ``sweep`` the bare airframe 0–90° (10 cases), ``sweep-fine`` (13),
``stall`` 8–16° (5: CL max), ``cruise`` (2 blown), ``transitions`` (the 3 s transitions' points + hover, blown),
``blown`` (cruise + transitions), ``all`` (sweep + blown), ``sections`` (the two finalist sections as slabs, 8).
A run does exactly the cases it lists (``--list``) — filtered with ``--cases``, bounded with ``--max-cases`` /
``--max-minutes`` — and never more. Resolved propellers are prop_lab's (notebook 38), not this workflow's.

Order of magnitude on 4 MPI ranks: a screening case 10–20 min (0.3–0.5 M cells, 400 iterations), a fine case ~1 h
(×3 cells, 800 iterations): ``sweep`` at screening is an afternoon, ``all`` at fine is days — run it detached
(``nohup`` / ``tmux``, see workflows/README.md) and watch ``tail -F <run>/run.log``.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _kestrel as wk  # noqa: E402  (puts workflows/ and notebooks/designs/ on the path)
import runkit  # noqa: E402
import kestrel_campaigns as kc  # noqa: E402
import kestrel_cfd as kcfd  # noqa: E402

WORKFLOW = "airframe_cfd"
BOOK = "kestrel_airframe"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="4Q-2", choices=tuple(kc.VARIANTS))
    wk.chapter_args(ap, BOOK)
    ap.add_argument("--quality", default="screening", choices=tuple(kcfd.QUALITY))
    ap.add_argument("--processors", type=int, default=4, help="MPI ranks per case (meshing stays serial)")
    runkit.add_common_arguments(ap)
    args = ap.parse_args(argv)

    if args.resume:
        m = runkit.read_json(Path(args.resume) / "manifest.json") or {}
        cfg = m.get("config") or {}
        if cfg:
            args.variant, args.chapter, args.quality = cfg.get("variant", args.variant), cfg.get("chapter", cfg.get("campaign", args.chapter)), cfg.get("quality", args.quality)
    config = {"workflow": WORKFLOW, "book": BOOK, "chapter": args.chapter, "campaign": args.chapter, "variant": args.variant, "quality": args.quality}

    t0 = time.monotonic()
    veh = kc.vehicle(args.variant)
    specs = kc.cfd_specs(args.variant, args.chapter, veh=veh)
    names = runkit.select_cases(list(specs), args.cases)
    if not names:
        print(f"no case of campaign {args.chapter} matches --cases {args.cases!r}; the campaign has: " + ", ".join(specs))
        return 2
    if args.list:
        tab = kcfd.not_run_table({n: specs[n] for n in names if not specs[n].get("slab")})
        print(f"{args.variant} / {args.chapter} ({args.quality}): {len(names)} case(s) — {kc.CFD_CAMPAIGNS[args.chapter]}")
        if len(tab):
            print(tab.to_string())
        slabs = [n for n in names if specs[n].get("slab")]
        if slabs:
            print("section slabs:\n  " + "\n  ".join(slabs))
        print(f"\nvehicle: {veh.name}, {veh.mass:.3f} kg, CG x {veh.x_cg * 1000:.1f} mm; prop_lab calibration {'on file' if kc.ks.prop_lab_calibration()['on_file'] else 'not on file (generic blade section)'}")
        return 0

    fp = kc.fingerprint(args.variant, veh)
    ignored = wk.prerequisites_or_exit(BOOK, args.chapter, variant=args.variant, expected=fp, ignore=args.ignore_prerequisites)
    if args.resume:
        run = runkit.Run.open(args.resume, config=config, heartbeat_s=args.heartbeat, echo=not args.quiet)
    else:
        out = Path(args.out) if args.out else wk.library.run_dir(BOOK, args.chapter, f"{args.variant}_{args.quality}")
        run = runkit.Run.create(out, workflow=WORKFLOW, campaign=args.chapter, config=config, design=fp, planned_cases=names,
                                versions={"openfoam": _openfoam_version()}, heartbeat_s=args.heartbeat, echo=not args.quiet)
    run.manifest.update(book=BOOK, chapter=args.chapter, prerequisites_ignored=ignored)
    runkit.write_json_atomic(run.root / "manifest.json", run.manifest)
    cache = runkit.Cache(BOOK, enabled=not args.no_cache)
    run.log(f"cache {'off (--no-cache)' if args.no_cache else cache.dir}")
    run.log(f"{kc.VARIANTS[args.variant]}; {kc.CFD_CAMPAIGNS[args.chapter]}; quality {args.quality} {kcfd.QUALITY[args.quality]}; "
            f"{args.processors} MPI rank(s) per case; prop_lab {'on file' if fp['prop_lab']['on_file'] else 'not on file'}")
    budget = runkit.Budget(args.max_cases, args.max_minutes, args.case_timeout)

    env = None
    try:
        from vegeta import aeromant
        env = aeromant.OpenFOAMEnvironment.detect()
        run.log(f"OpenFOAM: {env.flavor()} {env.version()} via {env.bashrc or ' '.join(env.prefix) or 'PATH'}")
    except Exception as exc:
        run.log(f"OpenFOAM not found ({exc}); cases can be prepared, not run")
        if not args.prepare_only:
            run.finish({"note": "OpenFOAM not found"}, state="failed", note=f"OpenFOAM not found: {exc}")
            wk.closing(run, 1, hint="install OpenFOAM (docs/installation.md) or use --prepare-only")
        from vegeta import aeromant
        env = aeromant.OpenFOAMEnvironment(env={})          # the openfoam.com case files, written without a solver
    p = kc.params(args.variant)

    # the cases are built (STL exported, dictionaries written) one at a time inside the loop, so a --max-cases run
    # does not pay for the whole campaign's CAD
    cases, keys, dirs = {}, {}, {}
    run.log(f"preparing {len(names)} case(s) under {run.cases_dir}")
    for n in names:
        c = kc.cfd_case(n, specs[n], run.cases_dir, variant=args.variant, quality=args.quality, veh=veh, env=env)
        cases[n], keys[n], dirs[n] = c, kc.cfd_case_key(c, args.quality), c.workdir
    run.log(f"cases built in {time.monotonic() - t0:.0f} s")

    def one(name, case_dir, cancel):
        case, spec = cases[name], specs[name]
        prep = case.prepare(overwrite=True)
        if not prep.ok:
            return kc.cfd_outcome(name, spec, prep, p)
        run.log(f"case {name}: prepared ({case.template.name}, {case.workdir.relative_to(run.root)})")
        if args.prepare_only:
            return {"status": "NOT RUN", "process": None, "note": "prepared only (--prepare-only)", "artifacts": {"case": str(case.workdir)}}
        last = {"t": 0.0, "stage": None}

        def progress(stage, fraction=None, message=""):
            now = time.monotonic()
            if stage != last["stage"]:
                run.log(f"case {name}: stage {stage}", echo=False)
                last["stage"], last["t"] = stage, now
            elif message and now - last["t"] > 60:
                run.log(f"case {name}: {stage} {message}", echo=False)
                last["t"] = now
        res = case.run(progress=progress, cancel=cancel, timeout=budget.case_timeout_s, processors=args.processors)
        return kc.cfd_outcome(name, spec, res, p)

    with run.signals():
        run.status("running")
        state = wk.drive(run, names, keys, one, budget=budget, rerun_failed=args.rerun_failed, case_dirs=dirs, cache=None if args.prepare_only else cache)
    if args.prepare_only:
        state = "prepared"
    summary = kc.cfd_summary(variant=args.variant, campaign=args.chapter, quality=args.quality, specs={n: specs[n] for n in names}, records=run.records,
                             veh=veh, run_root=run.root)
    for k, mp in summary.get("maps", {}).items():
        run.add_artifact(k, run.root / mp["file"])
    code = run.finish(summary, state=state)
    wk.closing(run, code, hint=f'notebook: RUNS["airframe_cfd"]["{args.variant}"]["{args.chapter}"] = "{run.root.relative_to(wk.ROOT) if wk.ROOT in run.root.parents else run.root}"' + (f"  ({cache.hits} case(s) from the cache)" if cache.hits else ""))
    return code


def _openfoam_version():
    try:
        from vegeta import aeromant
        return aeromant.OpenFOAMEnvironment.detect().version()
    except Exception:
        return None


if __name__ == "__main__":
    sys.exit(main())
