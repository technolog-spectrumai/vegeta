#!/usr/bin/env python3
"""Book races (code R): one chapter per course — the reference quad's MuJoCo lap and its videos — under the run
contract (runkit): a run directory with the manifest, the cases, the summary the notebook reads.

    python workflows/races/race.py --chapter farm [--quality screening|fine] [--mock] [--list]

Cases: ``race`` (the lap: trajectory.npz, stats.json, map.png), then the videos ``video-chase``, ``video-onboard``,
``video-overview`` (MuJoCo camera frames through ffmpeg when an OpenGL context exists — set MUJOCO_GL=egl — else the
matplotlib replay to mp4, said so in the case record). ``--quality screening`` (auto mode) makes short, small videos
and a shorter timeout; ``--mock`` runs the point-mass flyer instead of MuJoCo (the smoke tests; the summary says
``mode: mock``). The summary: ``counts``, ``stats`` (lap time, completion, collisions, energy, speeds, powers),
``manoeuvre`` (``races.metrics``), ``gate_times``, ``videos`` (relative paths), ``course``, ``mode``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "workflows" / "kestrel"))
sys.path.insert(0, str(ROOT / "workflows"))
sys.path.insert(0, str(ROOT))
import _kestrel as wk  # noqa: E402
import runkit  # noqa: E402

WORKFLOW = "race"
BOOK = "races"
KIND = "race"
CAMERAS = ("chase", "onboard", "overview")


def main(argv=None) -> int:
    import numpy as np
    from races import courses, metrics, mock, refquad, replay, video
    from races.rules import Rules
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chapter", default="farm", choices=list(courses.COURSES), help="the course")
    ap.add_argument("--variant", default=None, help="ignored: one aircraft (kept for the shared plumbing)")
    ap.add_argument("--quality", default="screening", choices=("screening", "fine"), help="video length and size; the screening timeout")
    ap.add_argument("--mock", action="store_true", help="the point-mass mock flyer instead of MuJoCo")
    ap.add_argument("--collision", default="reset", choices=("reset", "count", "stop"))
    ap.add_argument("--safety", type=float, default=0.8)
    ap.add_argument("--v-cap", type=float, default=None)
    ap.add_argument("--no-video", action="store_true")
    runkit.add_common_arguments(ap)
    args = ap.parse_args(argv)
    if args.resume:
        cfg = (runkit.read_json(Path(args.resume) / "manifest.json") or {}).get("config") or {}
        if cfg:
            args.chapter, args.quality, args.mock = cfg.get("campaign", args.chapter), cfg.get("quality", args.quality), cfg.get("mock", args.mock)
    config = {"workflow": WORKFLOW, "book": BOOK, "kind": KIND, "chapter": args.chapter, "campaign": args.chapter, "variant": "refquad", "quality": args.quality, "mock": bool(args.mock),
              "collision": args.collision, "safety": args.safety, "v_cap": args.v_cap}
    course = courses.build(args.chapter)
    fp = {"aircraft": "reference 5-inch quad", "mass_kg": refquad.MASS_KG, "course": course.slug, "course_params": course.params, "zones": len(course.zones)}
    names = ["race"] + ([] if args.no_video else [f"video-{c}" for c in CAMERAS])
    names = runkit.select_cases(names, args.cases)
    if args.list:
        print(f"{course.name}: {course.length():.0f} m, {len(course.zones)} zones — {course.description}")
        for n in names:
            print(f"  {n:16s} {'the MuJoCo lap (or the mock with --mock)' if n == 'race' else 'video from the recorded lap'}")
        return 0
    ignored = wk.prerequisites_or_exit(BOOK, args.chapter, variant="refquad", expected=fp, ignore=args.ignore_prerequisites)
    if args.resume:
        run = runkit.Run.open(args.resume, config=config, heartbeat_s=args.heartbeat, echo=not args.quiet)
    else:
        out = Path(args.out) if args.out else wk.library.run_dir(BOOK, args.chapter, f"refquad_{args.quality}" + ("_mock" if args.mock else ""))
        versions = {}
        try:
            import mujoco
            versions["mujoco"] = mujoco.__version__
        except Exception:
            versions["mujoco"] = None
        run = runkit.Run.create(out, workflow=WORKFLOW, campaign=args.chapter, config=config, design=fp, planned_cases=names, versions=versions, heartbeat_s=args.heartbeat, echo=not args.quiet)
    run.manifest.update(book=BOOK, chapter=args.chapter, kind=KIND, prerequisites_ignored=ignored, yanus={"needs": wk.GATE["needs"]})
    runkit.write_json_atomic(run.root / "manifest.json", run.manifest)
    bars = wk.Progress(run, len(names), enabled=not (args.no_progress or args.quiet), desc=f"race {args.chapter}")
    rules = Rules(collision=args.collision, timeout_s=120.0 if args.quality == "screening" else 300.0)
    budget = runkit.Budget(args.max_cases, args.max_minutes, args.case_timeout)
    state = {"result": None}

    def one(name, case_dir, cancel):
        case_dir = Path(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)
        if name == "race":
            if args.prepare_only:
                (case_dir / "scene.xml").write_text(courses.to_mjcf(course))
                return {"status": "NOT RUN", "process": None, "note": "scene written only (--prepare-only)", "artifacts": {"scene": str(case_dir / "scene.xml")}}
            t0 = time.monotonic()
            if args.mock:
                res = mock.run_mock(course, rules=rules, safety=args.safety, v_cap=args.v_cap)
            else:
                os.environ.setdefault("MUJOCO_GL", "egl")
                from races import sim
                res = sim.run_mujoco(course, rules=rules, safety=args.safety, v_cap=args.v_cap)
                (case_dir / "scene.xml").write_text(res["xml"])
            state["result"] = res
            np.savez(case_dir / "trajectory.npz", history=res["history"])
            st = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in res["stats"].items()}
            man = metrics.from_history(res["history"], path=res["path"])
            (case_dir / "stats.json").write_text(json.dumps({"stats": st, "manoeuvre": man, "gate times": res["gate times"], "events": [[float(t), s] for t, s in res["events"]]}, indent=1, default=float))
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 1, figsize=(13, 11))
            replay.overview(course, res, ax=axes[0], view="plan"); replay.overview(course, res, ax=axes[1], view="elevation")
            fig.savefig(case_dir / "map.png", dpi=100, bbox_inches="tight"); plt.close(fig)
            ok = st["status"] == "finished"
            return {"status": "solved" if ok else "diverged", "process": "ok", "converged": ok, "note": f"{st['status']} {st.get('reason', '')}: lap {st['lap time [s]']} s, {st['collisions']} collision(s), {st['energy [Wh]']:.2f} Wh in {time.monotonic() - t0:.0f} s ({res['stats']['mode']})",
                    "metrics": {k: v for k, v in st.items() if isinstance(v, (int, float)) and v is not None}, "artifacts": {"trajectory": str(case_dir / "trajectory.npz"), "stats": str(case_dir / "stats.json"), "map": str(case_dir / "map.png")}}
        cam = name.split("-", 1)[1]
        res = state["result"]
        if res is None:                                                    # a resumed run: the lap from the earlier case
            tr = run.cases_dir / "race" / "trajectory.npz"
            if not tr.exists():
                return {"status": "NOT RUN", "process": None, "note": "no lap to film (the race case did not run)", "artifacts": {}}
            res = {"history": np.load(tr)["history"], "events": [], "stats": {"mode": "replay"}, "referee": None}
            res["events"] = [tuple(e) for e in (json.loads((run.cases_dir / "race" / "stats.json").read_text()).get("events") or [])]
        out = case_dir / f"{course.slug}_{cam}.mp4"
        made, how = None, ""
        if not args.mock:
            try:
                made = video.mujoco_video(course, res, out, camera=cam, quality=args.quality, work_dir=case_dir / "frames")
                how = "MuJoCo camera frames through ffmpeg"
            except Exception as e:                                         # no OpenGL context, no renderer: the replay instead
                run.log(f"{name}: MuJoCo rendering unavailable ({type(e).__name__}: {str(e)[:80]}); the matplotlib replay instead")
        if made is None:
            made = video.matplotlib_video(course, res, out, camera=cam, quality=args.quality)
            how = "the matplotlib replay (no MuJoCo renderer)"
        return {"status": "solved", "process": "ok", "converged": True, "note": f"{made.name}: {how}", "metrics": {"bytes": made.stat().st_size}, "artifacts": {"video": str(made)}}

    with run.signals():
        run.status("running")
        end_state = wk.drive(run, names, {n: f"{course.slug}:{n}:{args.quality}:{'mock' if args.mock else 'mujoco'}" for n in names}, one, budget=budget, rerun_failed=args.rerun_failed, cache=None, progress=bars)
    if args.prepare_only:
        end_state = "prepared"
    rec = run.records
    res = state["result"]
    st = json.loads((run.cases_dir / "race" / "stats.json").read_text()) if (run.cases_dir / "race" / "stats.json").exists() else {}
    videos = {n.split("-", 1)[1]: str(Path(r.artifacts["video"]).relative_to(run.root)) for n, r in rec.items() if n.startswith("video") and r.artifacts.get("video")}
    summary = {"schema": 1, "book": BOOK, "chapter": args.chapter, "course": {"name": course.name, "slug": course.slug, "kind": course.kind, "length_m": course.length(), "zones": [z.name for z in course.zones], "params": course.params},
               "aircraft": refquad.summary(), "mode": "mock" if args.mock else "mujoco", "quality": args.quality, "rules": {k: v for k, v in rules.__dict__.items() if k != "text"},
               "stats": st.get("stats"), "manoeuvre": st.get("manoeuvre"), "gate_times": st.get("gate times"), "videos": videos,
               "counts": {"n": len(names), "solved": sum(r.status == "solved" for r in rec.values()), "failed": sum(r.status == "FAILED" for r in rec.values())}}
    bars.close()
    race_rec = rec.get("race")
    if end_state == "completed" and (race_rec is None or race_rec.status in ("FAILED", "NOT RUN")) and not args.prepare_only:
        end_state = "failed"                                                   # no lap, no statistics: the gate must not hand this run to the notebook
        summary["note"] = f"the race case did not run: {getattr(race_rec, 'note', '') or 'see run.log'}"
    code = run.finish(summary, state=end_state, note=summary.get("note", ""))
    wk.closing(run, code, hint=wk.pin_hint(run))
    return code


if __name__ == "__main__":
    sys.exit(main())
