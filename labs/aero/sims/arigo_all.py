"""ARIGO family: run every analysis, restartably, with one progress bar; one result file per run; charts.

WHAT IT RUNS (in this order; each step can be stopped and resumed)
  1. performance cards   speeds (min/stall, max level), turns (sustained and instantaneous over speed, corner speed),
                         climb (rate, angle), roll / pitch / yaw response, chase (run-down of a fleeing bandit,
                         zig-zag chase)                                             ~10 s per configuration
  2. stability           MuJoCo 6-DOF tests: modes, stall, gust, engine/fan out      ~5-10 s per configuration
  3. camera              ARIGO-0's propeller in the camera's view                    seconds
  4. videos              manoeuvre videos (needs OpenGL: MUJOCO_GL=egl or osmesa)    ~15 s per video
  5. CFD                 OpenFOAM angle-of-attack sweeps (needs OpenFOAM)            ~1.5 min a case (screening),
                                                                                    ~5-8 min a case (--full-cfd)
  Aircraft: ARIGO-0 (nose tractor; + its chin-strut version), ARIGO-1 (pusher), ARIGO-2 (two wing tractors) on the
  goshawk, zero and hybrid wings; ARIGO-20 (two ducted fans).

USAGE
  labs/aero/sims/arigo_all.sh                              everything, all four aircraft, 8 cores for the CFD
  labs/aero/sims/arigo_all.sh -j 16                        16 cores for the CFD (or --cpu 16)
  labs/aero/sims/arigo_all.sh ARIGO-0 ARIGO-1              only these aircraft
  labs/aero/sims/arigo_all.sh --stability-only             performance + stability + camera only (minutes)
  labs/aero/sims/arigo_all.sh --skip-video                 no videos (also: --skip-cfd)
  labs/aero/sims/arigo_all.sh --full-cfd -j 12 ARIGO-1     converged CFD (fine quality) into arigo*_fine_cfd.yaml
  labs/aero/sims/arigo_all.sh --fresh                      ignore saved results, recompute everything
  (SKIP_VIDEO=1, SKIP_CFD=1, FRESH=1 in the environment do the same as the flags.)

RESTARTING
  Ctrl-C any time; run the same command again and every step goes on where it stopped (a solved CFD case comes back
  from its OpenFOAM folder in seconds). Results computed by older code are set aside as *.stale.yaml.

OUTPUT
  ONE file per run, named by its start time and machine:
      labs/aero/sims/output/runs/arigo_run_<YYYYmmdd-HHMMSS>_<host>.yaml
  holding the run's settings (host, git commit, command, aircraft, steps) and every result its steps produced,
  rewritten after each step (an interrupted run keeps what it got), plus KPI scores. Next to it, the charts:
      ..._properties.png   bar charts of every property, all configurations
      ..._kpi.png          general and chase KPI scores
      ..._curves.png       sustained turn rate and climb rate over speed, per aircraft
  Several runs or machines -> ONE file:
      python labs/aero/sims/arigo_merge.py labs/aero/sims/output/runs/*.yaml -o all_runs.yaml
      python labs/aero/sims/arigo_charts.py all_runs.yaml          (charts for the merged file)
  The runners' own detailed output: labs/aero/sims/output/arigo_all.log (tail -f it). The step files
  (arigo*_mujoco.yaml, arigo*_cfd.yaml, arigo_perf.yaml...) stay in labs/aero/sims/output/ as before.

PASTE BACK the run file (and the PNGs if you want them discussed).
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from tqdm import tqdm

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
PY = sys.executable
W = ["--variants", "goshawk", "zero", "hybrid"]
# first-guess seconds per item, by kind (replaced by the measured mean once a few items of the kind are done)
GUESS = {"perf": 12.0, "stability": 6.0, "camera": 10.0, "video": 15.0, "cfd": 95.0, "cfd_fine": 400.0}
ITEM = re.compile(r"^\s*\[(\d+)/(\d+),")
PLAN = re.compile(r"(\d+) [\w ]+? already (?:done|solved) \(kept\), (\d+) to run")


def steps(want, a, np_):
    f = ["--fresh"] if a.fresh else []
    s = []
    def add(ac, kind, label, n, cmd, out):
        if ac in want:
            s.append({"aircraft": ac, "kind": kind, "label": label, "n": n, "cmd": cmd, "out": out})
    sfx = "_fine" if a.full_cfd else ""
    for ac, n in (("ARIGO-0", 6), ("ARIGO-1", 3), ("ARIGO-2", 3), ("ARIGO-20", 1)):
        add(ac, "perf", f"{ac} performance (speeds, turns, climb, roll/pitch/yaw, chase)", n, ["arigo_perf.py", ac, *f], "arigo_perf.yaml")
    add("ARIGO-0", "stability", "ARIGO-0 stability", 12, ["arigo1_mujoco.py", *W, "--propulsion", "tractor_lowcam", "tractor", "--name", "arigo0", *f], f"arigo0_mujoco.yaml")
    add("ARIGO-1", "stability", "ARIGO-1 stability", 6, ["arigo1_mujoco.py", *W, "--propulsion", "pusher", "--name", "arigo1", *f], f"arigo1_mujoco.yaml")
    add("ARIGO-2", "stability", "ARIGO-2 stability + engine out", 6, ["arigo2_mujoco.py", *f], f"arigo2_mujoco.yaml")
    add("ARIGO-20", "stability", "ARIGO-20 stability + fan out", 2, ["arigo20_mujoco.py", *f], f"arigo20_mujoco.yaml")
    add("ARIGO-0", "camera", "ARIGO-0 propeller in the camera", 1, ["arigo1_camera.py", "--layouts", "tractor_lowcam", "tractor", "pusher"], f"arigo0_camera.yaml")
    if not a.skip_video:
        add("ARIGO-0", "video", "ARIGO-0 videos", 36, ["arigo1_video.py", *W, "--propulsion", "tractor_lowcam", "tractor", "--name", "arigo0", *f], f"arigo0_video.yaml")
        add("ARIGO-1", "video", "ARIGO-1 videos", 18, ["arigo1_video.py", *W, "--propulsion", "pusher", "--name", "arigo1", *f], f"arigo1_video.yaml")
        add("ARIGO-2", "video", "ARIGO-2 videos", 21, ["arigo2_video.py", *f], f"arigo2_video.yaml")
        add("ARIGO-20", "video", "ARIGO-20 videos", 6, ["arigo20_video.py", *f], f"arigo20_video.yaml")
    if not a.skip_cfd:
        P = ["--processors", str(np_)]
        k, Q, lab = ("cfd_fine", ["--quality", "fine"], "full CFD") if a.full_cfd else ("cfd", [], "CFD")
        add("ARIGO-1", k, f"ARIGO-1 {lab}", 33, ["arigo1_cfd.py", *W, "--propulsion", "pusher", "--name", "arigo1" + sfx, *P, *Q, *f], f"arigo1{sfx}_cfd.yaml")
        add("ARIGO-0", k, f"ARIGO-0 {lab}", 66, ["arigo1_cfd.py", *W, "--propulsion", "tractor_lowcam", "tractor", "--name", "arigo0" + sfx, *P, *Q, *f], f"arigo0{sfx}_cfd.yaml")
        add("ARIGO-20", k, f"ARIGO-20 {lab}", 11, ["arigo20_cfd.py", "--name", "arigo20" + sfx, *P, *Q, *f], f"arigo20{sfx}_cfd.yaml")
        add("ARIGO-2", k, f"ARIGO-2 {lab}", 48, ["arigo2_cfd.py", "--name", "arigo2" + sfx, *P, *Q, *f], f"arigo2{sfx}_cfd.yaml")
    return s


def write_run(path, meta, todo, perf_keys):
    """ONE file for this run: its metadata and every result its steps produced (the step files' contents, under their
    names). Rewritten after each step, so an interrupted run still leaves what it got."""
    import json
    import yaml
    doc = {"run": {**meta, "updated": time.strftime("%Y-%m-%d %H:%M:%S")}, "results": {}}
    for st in todo:
        f = OUT / st["out"]
        if st.get("ran") and f.exists():
            try:
                d = yaml.safe_load(f.read_text())
            except Exception as exc:
                d = {"error": f"unreadable: {exc}"}
            name = f.stem
            if name == "arigo_perf" and isinstance(d, dict):    # the performance file holds every aircraft: keep this run's
                d = {**d, "cards": {k: v for k, v in (d.get("cards") or {}).items() if k.split("_")[0] in perf_keys}}
            doc["results"][name] = d
    tmp = path.with_suffix(".tmp")
    tmp.write_text(yaml.safe_dump(json.loads(json.dumps(doc, default=str)), sort_keys=False, width=140))
    tmp.replace(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="arigo_all.sh", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rest", nargs="*", help="[ARIGO-0 ARIGO-1 ARIGO-2 ARIGO-20] (default: all)")
    ap.add_argument("-j", "--cpu", type=int, default=8, help="cores for the CFD (default 8)")
    ap.add_argument("--stability-only", action="store_true")
    ap.add_argument("--full-cfd", action="store_true", help="CFD at fine quality (long) into arigo*_fine_cfd.yaml")
    ap.add_argument("--skip-video", action="store_true", default=bool(os.environ.get("SKIP_VIDEO")))
    ap.add_argument("--skip-cfd", action="store_true", default=bool(os.environ.get("SKIP_CFD")))
    ap.add_argument("--fresh", action="store_true", default=bool(os.environ.get("FRESH")))
    a = ap.parse_args(argv)
    if a.stability_only:
        a.skip_video = a.skip_cfd = True
    np_ = a.cpu
    want = []
    for r in a.rest:
        if r.isdigit():                                            # the old positional form, still accepted
            np_ = int(r)
        else:
            want.append(r.upper())
    want = want or ["ARIGO-0", "ARIGO-1", "ARIGO-2", "ARIGO-20"]
    todo = steps(want, a, np_)
    OUT.mkdir(parents=True, exist_ok=True)
    log = open(OUT / "arigo_all.log", "a")
    log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(sys.argv)}\n")
    import socket
    import subprocess as sp
    stamp = time.strftime("%Y%m%d-%H%M%S")
    host = socket.gethostname().split(".")[0]
    try:
        commit = sp.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=str(HERE)).stdout.strip()
    except Exception:
        commit = None
    runs = OUT / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    run_path = runs / f"arigo_run_{stamp}_{host}.yaml"
    meta = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "host": host, "commit": commit, "command": " ".join(sys.argv),
            "aircraft": want, "cpu": np_, "steps": [st["label"] for st in todo]}
    perf_keys = {w.lower() for w in want}
    pace = {k: [] for k in GUESS}                                  # measured seconds per item, by kind
    sec = lambda kind: (sum(pace[kind][-10:]) / len(pace[kind][-10:])) if pace[kind] else GUESS[kind]
    for st in todo:
        st["left"] = st["n"]                                       # items still to do in this step (revised by its plan line)
    total = lambda: sum(st["left"] * sec(st["kind"]) for st in todo)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if not a.skip_video:
        env.setdefault("MUJOCO_GL", "egl")
    status = 0
    bar = tqdm(total=round(total()), unit="s", unit_scale=False, dynamic_ncols=True,
               bar_format="{l_bar}{bar}| {percentage:3.0f}% [{elapsed}<{remaining}] {desc}")
    done_s = 0.0
    for i, st in enumerate(todo, 1):
        bar.set_description_str(f"{i}/{len(todo)} {st['label']}")
        tqdm.write(f"== [{i}/{len(todo)}] {st['label']}")
        t_item = time.time()
        p = subprocess.Popen([PY, str(HERE / st["cmd"][0]), *st["cmd"][1:]], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, env=env, cwd=str(HERE.parents[2]))
        for line in p.stdout:
            log.write(line)
            log.flush()
            m = PLAN.search(line)
            if m:
                st["left"] = int(m.group(2))
            elif ITEM.match(line):
                dt = time.time() - t_item
                t_item = time.time()
                pace[st["kind"]].append(dt)
                st["left"] = max(st["left"] - 1, 0)
                done_s += dt
                bar.set_postfix_str(line.strip()[:60], refresh=False)
            else:
                continue
            bar.total = round(done_s + total())                    # the remaining work, re-estimated at today's pace
            bar.n = round(done_s)
            bar.refresh()
        rc = p.wait()
        st["ran"] = True
        write_run(run_path, meta, todo, perf_keys)
        if st["kind"] == "camera":
            done_s += time.time() - t_item
        st["left"] = 0
        bar.total = round(done_s + total())
        bar.n = round(done_s)
        bar.refresh()
        if rc != 0:
            status = 1
            tqdm.write(f"   step failed (exit {rc}); going on — see {OUT / 'arigo_all.log'}")
    bar.close()
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["status"] = "ok" if status == 0 else "some steps failed (see arigo_all.log)"
    write_run(run_path, meta, todo, perf_keys)
    try:
        import arigo_charts
        pngs = arigo_charts.charts(run_path)
    except Exception as exc:
        pngs = []
        print(f"charts failed: {type(exc).__name__}: {exc}")
    for p in pngs:
        print(f"   chart: {p.relative_to(HERE.parents[2])}")
    print(f"All this run's results, in one file — paste it back:\n   {run_path.relative_to(HERE.parents[2])}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
