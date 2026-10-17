#!/usr/bin/env python3
"""fenix — the FEniCSx chapters of Svarog's guard: the barrier plates under the stray beam, solved by finite elements
in a separate interpreter (dolfinx cannot live in the venv), one run directory per chapter like any CFD or FEA chapter.

    .venv/bin/python workflows/fenix/plate_thermal.py --chapter guard-ablation --variant 1kW --quality screening
    .venv/bin/python workflows/fenix/plate_thermal.py --chapter guard-steady --variant 2kW --list
    .venv/bin/python workflows/fenix/plate_thermal.py --resume workflows/runs/fenix/guard-ablation/<run id>

Book ``fenix`` (code F). Chapters: ``guard-steady`` — every barrier's steady state under the collimated beam (Newton on
the nonlinear radiation; seconds each); ``guard-ablation`` — the transient with melting and ablation for the pairs the
closed-form screening of ``svarog_industrial`` could not decide (the plastics, the brick, the thin steel); ``sheet-distance`` —
the reference 2 mm polycarbonate sheet (1200 kg/m³, 60 × 60 cm, its edge the cold boundary) under the collimated beam at
10, 20, 50, 100 and 200 m; ``shield-steady`` (Svarog III, notebook 47) — the nine shielding panels of ``svarog_commercial``
(polycarbonate clear/opaque/certified, aluminium bare/oxidised/coated, wood solid/plywood/fire-retardant) under a laser's
raw collimated beam at 5, 10, 20, 50 and 100 m, **steady state only**: the non-destructive response — no transient, no
burn-through time is ever computed for it. Variants: ``1kW`` and ``2kW``, the delivered power of notebook 46's devices, and
the four commercial lasers ``YLS-SM``, ``FL2000C``, ``FL2000CSM``, ``WFL-3000`` of notebook 47 (their delivered power from
the published limits and the head's optical efficiency). ``--quality screening`` halves the mesh and lengthens
the time step (auto mode); ``fine`` is 48 × 16 cells through the plate.

The worker is ``_worker.py``, launched with the fenicsx interpreter (``VEGETA_FENICSX_PYTHON``, else
``.fenicsx/env/bin/python`` as install_local.sh makes it); each case is a JSON spec in and a ``result.json`` out — the
process pipe between the two environments. No interpreter → every case NOT RUN with the reason, the run ``failed``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "workflows", ROOT / "notebooks" / "designs"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import runkit  # noqa: E402
import library  # noqa: E402

WORKFLOW = "plate_thermal"
BOOK = "fenix"
KIND = "thermal"
VARIANTS = {"1kW": 933.0, "2kW": 1811.0, "YLS-SM": None, "FL2000C": None, "FL2000CSM": None, "WFL-3000": None}   # 1kW/2kW: delivered to the head per notebook 46's report (these numbers when it is absent); the commercial lasers: svarog_commercial
SVAROG_VARIANTS = ("1kW", "2kW")
COMMERCIAL_VARIANTS = ("YLS-SM", "FL2000C", "FL2000CSM", "WFL-3000")
QUALITY = {"screening": {"nr": 24, "nz": 8, "dt_scale": 2.0, "t_max": 600.0}, "fine": {"nr": 48, "nz": 16, "dt_scale": 1.0, "t_max": 3000.0}}
WORKER = Path(__file__).resolve().parent / "_worker.py"


def fenicsx_python() -> Path | None:
    """The fenicsx interpreter: ``VEGETA_FENICSX_PYTHON``, else ``.fenicsx/env/bin/python``; None when neither exists."""
    cand = [os.environ.get("VEGETA_FENICSX_PYTHON", ""), str(ROOT / ".fenicsx" / "env" / "bin" / "python")]
    for c in cand:
        if c and Path(c).exists():
            return Path(c)
    return None


def probe() -> str | None:
    """Yanus's Tool probe: dolfinx's version through the fenicsx interpreter, None when there is none."""
    py = fenicsx_python()
    if py is None:
        return None
    try:
        out = subprocess.run([str(py), "-c", "import dolfinx; print(dolfinx.__version__)"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return f"dolfinx {out.stdout.strip()} ({py})" if out.returncode == 0 else None


def _commercial():
    import svarog_commercial
    return svarog_commercial


def power_of(variant: str) -> float:
    """The delivered power of the variant: notebook 46's record when it exists (else the recorded number) for the Svarog
    devices; the published power through the head's optical efficiency for the commercial lasers."""
    if variant in COMMERCIAL_VARIANTS:
        sc = _commercial()
        return float(sc.optics(sc.LASERS[variant])["delivered_W"])
    p = ROOT / "notebooks/designs/data/reports/46_svarog.json"
    if p.exists():
        d = json.loads(p.read_text()).get("industrial", {}).get(variant, {})
        if d.get("delivered_W"):
            return float(d["delivered_W"])
    return VARIANTS[variant]


def collimated_beam(variant: str, distance: float):
    """The variant's raw collimated beam at ``distance`` (an ``svarog_industrial.StrayBeam``)."""
    if variant in COMMERCIAL_VARIANTS:
        sc = _commercial()
        return sc.stray_beam(sc.LASERS[variant], distance, "collimated")
    import svarog_industrial as si
    delivery = si.DELIVERY_100 if variant == "2kW" else si.DELIVERY
    return si.stray_beams(power_of(variant), distance=distance, delivery=delivery)[0]


def fingerprint(variant: str) -> dict:
    import svarog_industrial as si
    beam = collimated_beam(variant, si.DISTANCE)
    barriers = [b.name for b in si.BARRIERS] if variant in SVAROG_VARIANTS else [s.name for lst in _commercial().SHIELDS.values() for s in lst]
    return {"variant": variant, "power_W": round(beam.power, 1), "beam_radius_m": round(beam.radius, 5), "barriers": barriers}


def compatible(expected: dict | None, found: dict | None, chapter=None) -> dict:
    if not expected or not found:
        return {"level": "warning", "reasons": ["no fingerprint to compare"]}
    reasons = []
    for k in ("power_W", "beam_radius_m"):
        a, b = expected.get(k), found.get(k)
        if a and b and abs(a - b) > 0.02 * abs(a):
            reasons.append(f"{k}: {b} in the run, {a} now")
    if expected.get("barriers") != found.get("barriers"):
        reasons.append("the barrier list changed")
    return {"level": "incompatible" if reasons else "compatible", "reasons": reasons}


def cases_of(campaign: str, variant: str, quality: str) -> list[dict]:
    """The chapter's cases: one JSON spec each (the worker's input)."""
    import svarog_industrial as si
    q = QUALITY[quality]
    out = []
    if campaign == "shield-steady":                                                   # Svarog III: every shielding panel, every distance, the steady state only
        sc = _commercial()
        for lst in sc.SHIELDS.values():
            for sh in lst:
                for d in sc.SHIELD_DISTANCES:
                    bm = collimated_beam(variant, d)
                    out.append({"name": f"{si_slug(sh.family + '_' + sh.variant)}_at_{int(d)}m", "mode": "steady", "material": sh.solid.name, "thickness": sh.thickness, "power": bm.power,
                                "radius": bm.radius, "shape": bm.shape, "h": 10.0, "nr": q["nr"], "nz": q["nz"], "r_max": max(8 * bm.radius, 12 * sh.thickness, 0.3), "quality": quality,
                                "distance_m": d, "family": sh.family, "variant": sh.variant})
        return out
    if variant not in SVAROG_VARIANTS:
        raise SystemExit(f"chapter {campaign} takes the Svarog variants {SVAROG_VARIANTS}; {variant} is a commercial laser of shield-steady")
    delivery = si.DELIVERY_100 if variant == "2kW" else si.DELIVERY
    beam = si.stray_beams(power_of(variant), delivery=delivery)[0]                 # the raw collimated beam: the worst case decides
    for b in si.BARRIERS:
        base = {"material": b.solid.name, "thickness": b.thickness, "power": beam.power, "radius": beam.radius, "shape": beam.shape, "h": 10.0,
                "nr": q["nr"], "nz": q["nz"] if b.thickness < 0.02 else q["nz"] + 8, "r_max": si._r_max(beam.radius, b.thickness), "quality": quality}
        if campaign == "guard-steady":
            out.append({"name": si_slug(b.name), "mode": "steady", **base})
        elif campaign == "guard-ablation":
            scr = si.screen(b, beam)
            if scr["decision"] == "simulate":
                out.append({"name": si_slug(b.name), "mode": "transient", "t_max": q["t_max"], "dt": None, **base, "screening": {k: v for k, v in scr.items() if k not in ("barrier", "beam")}})
    if campaign == "sheet-distance":                                                  # the reference 2 mm PC sheet, 60 × 60 cm, at every distance
        ref = si.REFERENCE_SHEET
        for d in si.DISTANCES:
            bm = si.stray_beams(power_of(variant), distance=d, delivery=delivery)[0]
            out.append({"name": f"pc2mm_at_{int(d)}m", "mode": "transient", "t_max": q["t_max"], "dt": None, "material": ref.solid.name, "thickness": ref.thickness, "power": bm.power,
                        "radius": bm.radius, "shape": bm.shape, "h": 10.0, "nr": q["nr"], "nz": max(4, q["nz"] // 2), "r_max": si.SHEET_HALF, "quality": quality, "distance_m": d})
    return out


def si_slug(name: str) -> str:
    return name.replace(" ", "_").replace("-", "_").replace(".", "p")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    camps = [c["campaign"] for c in library.BOOKS[BOOK]["chapters"].values() if c["kind"] == KIND]
    ap.add_argument("--chapter", "--campaign", dest="chapter", default=camps[0], choices=tuple(camps))
    ap.add_argument("--variant", default="1kW", choices=tuple(VARIANTS))
    ap.add_argument("--quality", default="fine", choices=tuple(QUALITY))
    runkit.add_common_arguments(ap)
    args = ap.parse_args(argv)
    if args.resume:
        cfg = (runkit.read_json(Path(args.resume) / "manifest.json") or {}).get("config") or {}
        if cfg:
            args.variant, args.chapter, args.quality = cfg.get("variant", args.variant), cfg.get("campaign", args.chapter), cfg.get("quality", args.quality)
    config = {"workflow": WORKFLOW, "book": BOOK, "kind": KIND, "chapter": library.chapter_of(BOOK, KIND, args.chapter) or args.chapter, "campaign": args.chapter,
              "variant": args.variant, "quality": args.quality}
    cases = cases_of(args.chapter, args.variant, args.quality)
    names = runkit.select_cases([c["name"] for c in cases], args.cases)
    if args.list:
        print(f"{args.variant} / {args.chapter} ({args.quality}): {len(names)} case(s) at {power_of(args.variant):.0f} W")
        for c in cases:
            if c["name"] in names:
                print(f"  {c['name']:36s} {c['mode']:10s} {c['material']} {c['thickness'] * 1e3:g} mm, {c['nr']}×{c['nz']} cells" + (f", t_max {c['t_max']:g} s" if c['mode'] == 'transient' else ""))
        py = fenicsx_python()
        print(f"\nfenicsx interpreter: {probe() or 'NONE (set VEGETA_FENICSX_PYTHON or run ./install_local.sh): every case would be NOT RUN'}")
        return 0
    fp = fingerprint(args.variant)
    t0 = time.monotonic()
    if args.resume:
        run = runkit.Run.open(args.resume, config=config, heartbeat_s=args.heartbeat, echo=not args.quiet)
    else:
        out = Path(args.out) if args.out else library.run_dir(BOOK, config["chapter"], f"{args.variant}_{args.quality}")
        run = runkit.Run.create(out, workflow=WORKFLOW, campaign=args.chapter, config=config, design=fp, planned_cases=names,
                                versions={"fenicsx": probe() or "none"}, heartbeat_s=args.heartbeat, echo=not args.quiet)
    run.signals()
    py = fenicsx_python()
    if py is None or probe() is None:
        run.log("no fenicsx interpreter (VEGETA_FENICSX_PYTHON / .fenicsx/env): every case NOT RUN")
        for n in names:
            run.records[n] = runkit.CaseRecord(name=n, key="none", status="NOT RUN", note="no fenicsx interpreter on this machine")
        return run.finish({"counts": {"n": len(names), "solved": 0, "not_run": len(names)}, "cases": {n: {"status": "NOT RUN"} for n in names}, "variant": args.variant,
                           "quality": args.quality, "power_W": power_of(args.variant)}, state="failed", note="no fenicsx interpreter")
    if args.prepare_only:
        for c in cases:
            d = run.case_dir(c["name"]); d.mkdir(parents=True, exist_ok=True); (d / "spec.json").write_text(json.dumps(c, indent=1))
        return run.finish({"counts": {"n": len(names), "solved": 0}, "variant": args.variant, "quality": args.quality}, state="prepared")
    budget = runkit.Budget(args.max_cases, args.max_minutes, args.case_timeout)
    started = 0
    results = {}
    for c in cases:
        if c["name"] not in names:
            continue
        if budget.max_cases is not None and started >= budget.max_cases:
            run.log("budget: max cases reached"); break
        if budget.max_minutes is not None and (time.monotonic() - t0) / 60 > budget.max_minutes:
            run.log("budget: max minutes reached"); break
        started += 1

        def solve(d: Path, cancel, spec=c):
            (d / "spec.json").write_text(json.dumps(spec, indent=1))
            cmd = [str(py), str(WORKER), str(d / "spec.json"), str(d)]
            env = {**os.environ, "PYTHONPATH": str(ROOT / "vegeta-cli" / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
            proc = subprocess.Popen(cmd, env=env, stdout=open(d / "worker.log", "w"), stderr=subprocess.STDOUT)
            while proc.poll() is None:
                if cancel.is_set():
                    proc.terminate()
                    return {"status": "cancelled", "process": "cancelled"}
                time.sleep(0.5)
            res = runkit.read_json(d / "result.json") or {}
            if proc.returncode != 0 or not res:
                return {"status": "FAILED", "process": "failed", "note": f"worker exit {proc.returncode}; read worker.log"}
            metrics = {k: res.get(k) for k in ("mode", "reaches_damage", "surface_damage_time", "failure_time", "failure_mode", "t_end", "seconds", "T_top_end", "T_bottom_end", "ablated_depth_end", "iterations")}
            metrics["T_max"] = res["T_max"] if res.get("mode") == "steady" else (res.get("T_max") or [None])[-1]     # the transient's T_max is a series: its end
            return {"status": res.get("status", "solved"), "process": "ok", "converged": res.get("converged"), "metrics": {k: v for k, v in metrics.items() if v is not None},
                    "artifacts": {"result": str(d / "result.json"), "spec": str(d / "spec.json")}}

        key = runkit.sha256_json({k: v for k, v in c.items() if k != "screening"})
        rec = run.run_case(c["name"], key, solve, rerun_failed=args.rerun_failed)
        results[c["name"]] = {"status": rec.status, **rec.metrics, "note": rec.note}
    solved = sum(1 for r in results.values() if r["status"] == "solved")
    left = [n for n in names if n not in results]
    state = "completed" if not left else "partial"
    summary = {"counts": {"n": len(names), "solved": solved, "failed": len(results) - solved}, "cases": results, "variant": args.variant, "quality": args.quality,
               "power_W": power_of(args.variant), "fenicsx": probe(), "seconds": round(time.monotonic() - t0, 1)}
    return run.finish(summary, state=state)


if __name__ == "__main__":
    sys.exit(main())
