"""The engine behind ``arigo_cfd.sh`` and ``arigo_stability.sh`` (two separate phases, never mixed): runs the
restartable step runners as subprocesses with one progress bar, and writes ONE result file per run. See those two
scripts' ``--help``."""
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


def steps(want, a, np_, phase):
    f = ["--fresh"] if a.fresh else []
    s = []
    def add(ac, kind, label, n, cmd, out):
        if ac in want:
            s.append({"aircraft": ac, "kind": kind, "label": label, "n": n, "cmd": cmd, "out": out})
    sfx = "_fine" if getattr(a, "full_cfd", False) else ""
    if phase == "cfd":
        P = ["--processors", str(np_)]
        k, Q, lab = ("cfd_fine", ["--quality", "fine"], "full CFD") if a.full_cfd else ("cfd", [], "CFD")
        add("ARIGO-1", k, f"ARIGO-1 {lab}", 33, ["arigo1_cfd.py", *W, "--propulsion", "pusher", "--name", "arigo1" + sfx, *P, *Q, *f], f"arigo1{sfx}_cfd.yaml")
        add("ARIGO-0", k, f"ARIGO-0 {lab}", 66, ["arigo1_cfd.py", *W, "--propulsion", "tractor_lowcam", "tractor", "--name", "arigo0" + sfx, *P, *Q, *f], f"arigo0{sfx}_cfd.yaml")
        add("ARIGO-20", k, f"ARIGO-20 {lab}", 11, ["arigo20_cfd.py", "--name", "arigo20" + sfx, *P, *Q, *f], f"arigo20{sfx}_cfd.yaml")
        add("ARIGO-2", k, f"ARIGO-2 {lab}", 48, ["arigo2_cfd.py", "--name", "arigo2" + sfx, *P, *Q, *f], f"arigo2{sfx}_cfd.yaml")
        return s
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


def main(argv=None, phase="stability", doc="") -> int:
    ap = argparse.ArgumentParser(prog=f"arigo_{phase}.sh", description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rest", nargs="*", help="aircraft: ARIGO-0 ARIGO-1 ARIGO-2 ARIGO-20 (default: all)")
    if phase == "cfd":
        ap.add_argument("-j", "--cpu", type=int, default=8, help="cores for OpenFOAM (default 8)")
        ap.add_argument("--full-cfd", action="store_true", help="fine quality (converged; ~5-8 min a case) into arigo*_fine_cfd.yaml")
    else:
        ap.add_argument("--videos", action="store_true", help="also render the manoeuvre videos (needs OpenGL + ffmpeg)")
    ap.add_argument("--fresh", action="store_true", help="ignore the saved results and recompute")
    a = ap.parse_args(argv)
    a.skip_video = not getattr(a, "videos", False)
    np_ = getattr(a, "cpu", 8)
    want = []
    for r in a.rest:
        if r.isdigit():                                            # the old positional form, still accepted
            np_ = int(r)
        else:
            want.append(r.upper())
    want = want or ["ARIGO-0", "ARIGO-1", "ARIGO-2", "ARIGO-20"]
    todo = steps(want, a, np_, phase)
    OUT.mkdir(parents=True, exist_ok=True)
    log = open(OUT / f"arigo_{phase}.log", "a")
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
    run_path = runs / f"arigo_{phase}_{stamp}_{host}.yaml"
    meta = {"phase": phase, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "host": host, "commit": commit, "command": " ".join(sys.argv),
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
            tqdm.write(f"   step failed (exit {rc}); going on — see {OUT / f'arigo_{phase}.log'}")
    bar.close()
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["status"] = "ok" if status == 0 else "some steps failed (see the log)"
    write_run(run_path, meta, todo, perf_keys)
    try:
        import arigo_charts
        pngs = arigo_charts.charts(run_path) if phase == "stability" else []
    except Exception as exc:
        pngs = []
        print(f"charts failed: {type(exc).__name__}: {exc}")
    for p in pngs:
        print(f"   chart: {p.relative_to(HERE.parents[2])}")
    print(f"All this run's results, in one file — paste it back:\n   {run_path.relative_to(HERE.parents[2])}")
    return status


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[2:], sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] in ("cfd", "stability") else 2)
