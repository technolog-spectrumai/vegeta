#!/usr/bin/env python3
"""One race lab run: a course's reference-quad lap (MuJoCo, or the mock flyer) and its videos, in a run directory
that ``python -m races ingest`` reads (manifest.json, status.json, run.log, cases/<case>/case.json + files,
summary.json).

    python races/lab/race.py --chapter farm [--quality screening|fine] [--mock] [--list] [--out DIR]

Cases: ``race`` (the lap: trajectory.npz, stats.json, map.png), then the videos ``video-chase``, ``video-onboard``,
``video-overview`` (MuJoCo camera frames through ffmpeg when an OpenGL context exists — set MUJOCO_GL=egl — else the
matplotlib replay to mp4, said so in the case record). ``--quality screening`` makes short, small videos and a
shorter timeout; ``--mock`` runs the point-mass flyer instead of MuJoCo (``mode: mock`` in the summary). The summary:
``counts``, ``stats`` (lap time, completion, collisions, energy, speeds, powers), ``manoeuvre`` (``races.metrics``),
``gate_times``, ``videos`` (relative paths), ``course``, ``mode``.

This was Vegeta's side-lab book R (``side_labs/races/race.py``, under the side labs' run contract); it graduated to
Piccolo with the racing quad and now writes its own run directories (``Run`` below) in the same layout.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")             # before the first `import mujoco`: the camera renderer's backend on a headless or Jupyter session (glfw/osmesa also work)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RUNS = ROOT / "races" / "runs" / "lab"                # races/ingest.py reads <RUNS>/<course>/<run>/

WORKFLOW = "race"
BOOK = "races"
KIND = "race"
CAMERAS = ("chase", "onboard", "overview")


def _write_json(path: Path, doc) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=str))
    os.replace(tmp, path)


class Run:
    """A run directory: the manifest, the state, a log, one folder per case with its record, the summary."""

    def __init__(self, root: Path, manifest: dict, echo: bool = True):
        self.root, self.echo = Path(root), echo
        self.cases_dir = self.root / "cases"
        self.cases_dir.mkdir(parents=True, exist_ok=False)
        self.manifest = {**manifest, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        _write_json(self.root / "manifest.json", self.manifest)
        self.status("running")

    def log(self, msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        with open(self.root / "run.log", "a") as f:
            f.write(line + "\n")
        if self.echo:
            print(line, flush=True)

    def status(self, state: str, **extra) -> None:
        _write_json(self.root / "status.json", {"state": state, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra})

    def case(self, name: str, fn) -> dict:
        d = self.cases_dir / name
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()
        try:
            rec = fn(name, d)
        except Exception as e:                                             # a failed case is recorded, the run goes on
            rec = {"status": "FAILED", "note": f"{type(e).__name__}: {e}"}
            self.log(traceback.format_exc())
        rec["elapsed_s"] = round(time.monotonic() - t0, 2)
        _write_json(d / "case.json", rec)
        self.log(f"{name}: {rec['status']} — {rec.get('note', '')}")
        return rec

    def finish(self, summary: dict, state: str) -> int:
        _write_json(self.root / "summary.json", {**summary, "state": state})
        self.status(state)
        return 0 if state == "completed" else 1


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
    ap.add_argument("--safety", type=float, default=None, help="the schedule's share of the corner speed the thrust vector allows (MuJoCo default 0.6, mock 0.8)")
    ap.add_argument("--v-cap", type=float, default=None, help="cap the schedule's top speed [m/s]")
    ap.add_argument("--lookahead", type=float, default=0.25, help="the autopilot's look-ahead [s of flight]")
    ap.add_argument("--omega-pos", type=float, default=2.5, help="the position loop's bandwidth [rad/s]")
    ap.add_argument("--omega-att", type=float, default=18.0, help="the attitude loop's bandwidth [rad/s]")
    ap.add_argument("--tilt-max", type=float, default=60.0, help="the largest tilt the position loop may ask for [deg]")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--list", action="store_true", help="list the cases and exit")
    ap.add_argument("--cases", default=None, help="comma-separated subset of the cases")
    ap.add_argument("--out", default=None, help="the run directory (default races/runs/lab/<course>/<time>_<quality>[_mock])")
    ap.add_argument("--prepare-only", action="store_true", help="write the scene, run nothing")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    config = {"workflow": WORKFLOW, "book": BOOK, "kind": KIND, "chapter": args.chapter, "campaign": args.chapter, "variant": "refquad", "quality": args.quality, "mock": bool(args.mock),
              "collision": args.collision, "safety": args.safety, "v_cap": args.v_cap, "lookahead": args.lookahead, "omega_pos": args.omega_pos, "omega_att": args.omega_att, "tilt_max": args.tilt_max}
    course = courses.build(args.chapter)
    fp = {"aircraft": "reference 5-inch quad", "mass_kg": refquad.MASS_KG, "course": course.slug, "course_params": course.params, "zones": len(course.zones)}
    names = ["race"] + ([] if args.no_video else [f"video-{c}" for c in CAMERAS])
    if args.cases:
        want = {c.strip() for c in args.cases.split(",")}
        names = [n for n in names if n in want]
    if args.list:
        print(f"{course.name}: {course.length():.0f} m, {len(course.zones)} zones — {course.description}")
        for n in names:
            print(f"  {n:16s} {'the MuJoCo lap (or the mock with --mock)' if n == 'race' else 'video from the recorded lap'}")
        return 0
    out = Path(args.out) if args.out else RUNS / args.chapter / (time.strftime("%Y%m%d_%H%M%S") + f"_{args.quality}" + ("_mock" if args.mock else ""))
    versions = {}
    try:
        import mujoco
        versions["mujoco"] = mujoco.__version__
    except Exception:
        versions["mujoco"] = None
    run = Run(out, {"workflow": WORKFLOW, "book": BOOK, "chapter": args.chapter, "campaign": args.chapter, "kind": KIND, "config": config,
                    "design": fp, "planned_cases": names, "tool_versions": versions}, echo=not args.quiet)
    rules = Rules(collision=args.collision, timeout_s=120.0 if args.quality == "screening" else 300.0)
    state = {"result": None}

    def one(name, case_dir):
        case_dir = Path(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)
        if name == "race":
            if args.prepare_only:
                (case_dir / "scene.xml").write_text(courses.to_mjcf(course))
                return {"status": "NOT RUN", "process": None, "note": "scene written only (--prepare-only)", "artifacts": {"scene": str(case_dir / "scene.xml")}}
            t0 = time.monotonic()
            if args.mock:
                res = mock.run_mock(course, rules=rules, safety=args.safety if args.safety is not None else 0.8, v_cap=args.v_cap)
            else:
                from races import sim
                res = sim.run_mujoco(course, rules=rules, safety=args.safety if args.safety is not None else 0.6, v_cap=args.v_cap, lookahead_s=args.lookahead,
                                     omega_pos=args.omega_pos, omega_att=args.omega_att, tilt_max_deg=args.tilt_max)
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

    rec = {}
    for n in names:
        rec[n] = run.case(n, one)
    end_state = "prepared" if args.prepare_only else "completed"
    res = state["result"]
    st = json.loads((run.cases_dir / "race" / "stats.json").read_text()) if (run.cases_dir / "race" / "stats.json").exists() else {}
    videos = {n.split("-", 1)[1]: str(Path(r["artifacts"]["video"]).relative_to(run.root)) for n, r in rec.items() if n.startswith("video") and r.get("artifacts", {}).get("video")}
    summary = {"schema": 1, "book": BOOK, "chapter": args.chapter, "course": {"name": course.name, "slug": course.slug, "kind": course.kind, "length_m": course.length(), "zones": [z.name for z in course.zones], "params": course.params},
               "aircraft": refquad.summary(), "mode": "mock" if args.mock else "mujoco", "quality": args.quality, "rules": {k: v for k, v in rules.__dict__.items() if k != "text"},
               "stats": st.get("stats"), "manoeuvre": st.get("manoeuvre"), "gate_times": st.get("gate times"), "videos": videos,
               "counts": {"n": len(names), "solved": sum(r["status"] == "solved" for r in rec.values()), "failed": sum(r["status"] == "FAILED" for r in rec.values())}}
    race_rec = rec.get("race")
    if end_state == "completed" and (race_rec is None or race_rec["status"] in ("FAILED", "NOT RUN")):
        end_state = "failed"                                                   # no lap, no statistics: the run must not count as a lap
        summary["note"] = f"the race case did not run: {(race_rec or {}).get('note') or 'see run.log'}"
    code = run.finish(summary, state=end_state)
    print(f"race {args.chapter}: {end_state} — {run.root}")
    return code


if __name__ == "__main__":
    sys.exit(main())
