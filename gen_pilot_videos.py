#!/usr/bin/env python3
"""MuJoCo videos of the tuned autopilot (races/autopilot.json), to validate it by eye: every course flown with its
tuned settings, rendered from the scene's cameras in real time with the recorded attitude and a HUD (time, speed,
altitude, zones, collisions). Real MuJoCo, no mocks, no solver.

    python gen_pilot_videos.py                          # all four courses, cameras chase, onboard and overview, the whole lap in real time
    python gen_pilot_videos.py -j 4 --courses hangar    # laps in 4 worker processes
    python gen_pilot_videos.py --quality screening      # short and small (seconds); --quality fine: every second row, full size
    python gen_pilot_videos.py --speed 2                # played twice as fast
    python gen_pilot_videos.py --held-out               # also the worst held-out lap of every family ./test_autopilot.sh did not pass
    python gen_pilot_videos.py --scenario "wind ≤ 2 m/s:3"   # one held-out scenario (family:index of races.validate, --seed)
    python gen_pilot_videos.py --settings races/runs/tune/autopilot_hangar_24s_knife_edge.json

Output: races/runs/videos/<course>/<stamp>_mujoco/ (timeseries, events, summary with provenance, one mp4 per camera)
and races/runs/videos/index.json. Without a working OpenGL backend (EGL, OSMesa, GLFW) it writes the matplotlib
replay instead, labelled as such. Exit 0 when every video was written, 1 if one failed, 2 when MuJoCo or the
settings are missing."""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "races" / "runs" / "videos"
CAMERAS = ("chase", "onboard", "overview")


def _write_atomic(path: Path, obj):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def held_out_worst(slug: str) -> list[tuple[str, int]]:
    """(family, index) of the worst held-out lap of every family not graded pass in the latest ./test_autopilot.sh
    run of the course: a DNF with the fewest zones, else the most collisions, else the slowest lap."""
    from races import validate
    f = ROOT / "races" / "runs" / "validate" / f"{slug}.csv"
    if not f.exists():
        return []
    rows = list(csv.DictReader(f.open()))
    picks = []
    for fam in dict.fromkeys(r["family"] for r in rows):
        R = [r for r in rows if r["family"] == fam]
        clean = [r["status"] == "finished" and int(r["collisions"] or 0) == 0 for r in R]
        if fam == "nominal" or validate.grade(sum(clean) / len(R)) == "pass":
            continue
        zones = lambda r: int(str(r["zones passed"] or "0 of").split(" ")[0] or 0)
        worst = min((r for r, c in zip(R, clean) if not c),
                    key=lambda r: (r["status"] == "finished", zones(r), -int(r["collisions"] or 0), -float(r["lap time [s]"] or 0)))
        picks.append((fam, int(worst["i"])))
    return picks


def scenario(family: str, i: int, seed: int) -> dict:
    """The held-out scenario ``family``:``i`` exactly as races.validate draws it for ``seed``."""
    from races import validate
    if family not in validate.FAMILIES:
        raise SystemExit(f"unknown scenario family {family!r}; one of: {', '.join(validate.FAMILIES)}")
    return next(s for s in validate.scenarios(i + 1, seed, (family,)) if s["i"] == i)


def fly_and_render(job: dict) -> dict:
    """One lap (a worker process): fly it, export it, render every camera. Returns the index entry and the log lines."""
    from races import courses, export, sim, validate, video
    log, t0 = [], time.time()
    c = courses.COURSES[job["course"]]()
    kw = {"control_dt": 0.004, **validate.apply(job["settings"], job["mult"])}
    lap = sim.run_mujoco(c, scenario=job["scenario"] or None, **kw)
    st = lap["stats"]
    fly_s = time.time() - t0
    run_dir = export.write_run(c, lap, simulator="mujoco", config={"autopilot": job["settings"], "label": job["label"], "mult": job["mult"],
                                                                   "scenario": job["scenario"]}, seed=None, timestep=kw["control_dt"], wall_s=fly_s, root=OUT)
    log.append(f"{job['course']} [{job['label']}]: {st['status']} {st['lap time [s]']:.2f} s, {st['collisions']} collisions, "
               f"{st['zones passed']} zones — flown in {fly_s:.0f} s → {run_dir}")
    videos, errors = {}, {}
    for cam in job["cameras"]:
        t1 = time.time()
        out = run_dir / f"{job['course']}_{cam}.mp4"
        try:
            if job["gl"]:
                f = video.mujoco_video(c, lap, out, camera=cam, quality=job["quality"], speed=job["speed"], hud=True, label=job["label"])
                if f is None:
                    raise RuntimeError("no mp4 written (ffmpeg missing on the PATH?)")
            else:
                f = video.matplotlib_video(c, lap, out, camera=cam, quality="fine" if job["quality"] == "lap" else job["quality"])
            videos[cam] = str(f)
            log.append(f"   {cam}: {f.name} ({time.time() - t1:.0f} s{'' if job['gl'] else ', matplotlib replay (no OpenGL)'})")
        except Exception as e:
            errors[cam] = f"{type(e).__name__}: {e}"
            log.append(f"   {cam}: RENDER FAILED — {errors[cam]} (the lap's data is kept in {run_dir})")
    s = json.loads((run_dir / "summary.json").read_text())
    s["artifacts"]["videos"] = {k: Path(v).name for k, v in videos.items()}
    s["artifacts"]["video_errors"] = errors
    s["artifacts"]["renderer"] = "mujoco" if job["gl"] else "matplotlib replay (no OpenGL)"
    (run_dir / "summary.json").write_text(json.dumps(s, indent=1, ensure_ascii=False))
    entry = {"course": job["course"], "label": job["label"], "status": st["status"], "lap time [s]": None if st["status"] != "finished" else round(st["lap time [s]"], 2),
             "collisions": st["collisions"], "zones passed": st["zones passed"], "run": str(run_dir.relative_to(ROOT)),
             "videos": {k: str(Path(v).relative_to(ROOT)) for k, v in videos.items()}, "errors": errors, "wall [s]": round(time.time() - t0, 1)}
    return {"entry": entry, "log": log, "expect": job.get("expect")}


def main(argv=None, gl: bool = True) -> int:
    from races import courses, validate, video
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--courses", nargs="*", default=list(courses.COURSES), choices=list(courses.COURSES))
    ap.add_argument("--cameras", nargs="*", default=list(CAMERAS), choices=("chase", "onboard", "overview", "grandstand"))
    ap.add_argument("--quality", default="lap", choices=list(video.QUALITY), help="lap (default: the whole lap in real time, 960×540, 30 fps), screening (short, small), fine")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed for --quality lap (default 1 = real time)")
    ap.add_argument("-j", "--jobs", type=int, default=1, help=f"laps in worker processes side by side (default 1; this machine has {os.cpu_count()} CPUs)")
    ap.add_argument("--settings", type=Path, default=None, help="an autopilot.json-format file instead of races/autopilot.json")
    ap.add_argument("--held-out", action="store_true", help="also the worst held-out lap of every family the latest ./test_autopilot.sh did not pass")
    ap.add_argument("--scenario", action="append", default=[], metavar="FAMILY:I", help="also one held-out scenario of races.validate (repeatable)")
    ap.add_argument("--seed", type=int, default=None, help="the held-out seed for --scenario / --held-out (default: the latest report's, else races.validate.SEED)")
    a = ap.parse_args(argv)

    stored = json.loads((a.settings or ROOT / "races" / "autopilot.json").read_text())
    report = ROOT / "races" / "runs" / "validate" / "report.json"
    seed = a.seed if a.seed is not None else (json.loads(report.read_text()).get("seed") if report.exists() else validate.SEED)
    jobs = []
    for slug in a.courses:
        settings = (stored.get(slug) or {}).get("settings")
        if not settings:
            print(f"== {slug}: no tuned settings in {a.settings or 'races/autopilot.json'} — skipped", flush=True)
            continue
        base = {"course": slug, "settings": settings, "cameras": a.cameras, "quality": a.quality, "speed": a.speed, "gl": gl}
        jobs.append({**base, "label": "nominal", "mult": {}, "scenario": {}})
        picks = [(f, i) for f, i in held_out_worst(slug)] if a.held_out else []
        picks += [(s.rsplit(":", 1)[0], int(s.rsplit(":", 1)[1])) for s in a.scenario]
        if a.held_out and not picks:
            print(f"== {slug}: no failing held-out family in races/runs/validate/{slug}.csv (run ./test_autopilot.sh first?)", flush=True)
        rows = {(r["family"], int(r["i"])): r for r in csv.DictReader((ROOT / "races" / "runs" / "validate" / f"{slug}.csv").open())} \
            if (ROOT / "races" / "runs" / "validate" / f"{slug}.csv").exists() else {}
        for fam, i in dict.fromkeys(picks):
            sc = scenario(fam, i, seed)
            jobs.append({**base, "label": f"held-out: {fam} #{i}", "mult": sc["mult"], "scenario": sc["scenario"], "expect": rows.get((fam, i))})
    if not jobs:
        print("gen_pilot_videos: nothing to fly", flush=True)
        return 2
    print(f"gen_pilot_videos: {len(jobs)} lap(s) × {len(a.cameras)} camera(s), quality {a.quality}, speed {a.speed}×, "
          f"{'MuJoCo frames' if gl else 'matplotlib replay (no OpenGL)'}; {a.jobs} worker(s) of {os.cpu_count()} CPUs", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    index = {"date": time.strftime("%Y-%m-%d %H:%M"), "settings file": str(a.settings or "races/autopilot.json"), "held-out seed": seed,
             "renderer": "mujoco" if gl else "matplotlib replay (no OpenGL)", "quality": a.quality, "speed": a.speed, "laps": []}
    t0, failed = time.time(), 0
    if a.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(min(a.jobs, len(jobs))) as ex:
            results = ex.map(fly_and_render, jobs)
            for r in results:
                failed += _report(r, index)
    else:
        for j in jobs:
            failed += _report(fly_and_render(j), index)
    print(f"gen_pilot_videos: {sum(len(e['videos']) for e in index['laps'])} video(s), {failed} failed, in {(time.time() - t0) / 60:.1f} min; "
          f"index {OUT / 'index.json'}", flush=True)
    return 1 if failed else 0


def _report(r: dict, index: dict) -> int:
    for line in r["log"]:
        print(line, flush=True)
    e, x = r["entry"], r["expect"]
    if x is not None:                                                    # a held-out lap: does it reproduce the validation run?
        same = x["status"] == e["status"] and (e["lap time [s]"] is None or abs(float(x["lap time [s]"] or 0) - e["lap time [s]"]) < 0.01)
        e["reproduces the validation lap"] = same
        if not same:
            print(f"   note: the validation run had {x['status']} {x['lap time [s]']} — the settings changed since ./test_autopilot.sh?", flush=True)
    index["laps"].append(e)
    _write_atomic(OUT / "index.json", index)
    return len(e["errors"])


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    import tune_autopilot
    tune_autopilot.say = lambda msg: print(f"gen_pilot_videos: {msg}", flush=True)
    boot = tune_autopilot.bootstrap(OUT, "videos", probes=(tune_autopilot.RENDER_PROBE, tune_autopilot.IMPORT_PROBE))
    if boot is None:
        sys.exit(2)
    sys.exit(main(gl=boot[1] == 0))
