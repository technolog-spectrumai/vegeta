"""The whole ARIGO family's analyses with one global progress bar (``arigo_all.sh`` runs this).

Each step is one of the restartable runners (``arigo1_mujoco.py``, ``arigo2_video.py``, ``arigo20_cfd.py``...), run
as a subprocess; its output goes to ``labs/aero/sims/output/arigo_all.log``, and its item lines ("[k/N, ...]") and
its "already done (kept), M to run" line drive the bar. The bar counts estimated seconds, not items — a CFD case
weighs ~100 stability runs — so its time left is a real estimate; it learns each kind's actual pace as it goes.

    python labs/aero/sims/arigo_all.py [processors] [ARIGO-0 ARIGO-1 ...] [--stability-only] [--full-cfd] [--skip-video] [--skip-cfd] [--fresh]

``--full-cfd``: the CFD at ``fine`` quality (finer mesh, 600 iterations, tighter residuals: converged numbers; several
times longer, ~5-8 min a case) into separate files ``arigo*_fine_cfd.yaml`` — the screening results are kept.
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
GUESS = {"stability": 6.0, "camera": 10.0, "video": 15.0, "cfd": 95.0, "cfd_fine": 400.0}
ITEM = re.compile(r"^\s*\[(\d+)/(\d+),")
PLAN = re.compile(r"(\d+) [\w ]+? already (?:done|solved) \(kept\), (\d+) to run")


def steps(want, a, np_):
    f = ["--fresh"] if a.fresh else []
    s = []
    add = lambda ac, kind, label, n, cmd: s.append({"aircraft": ac, "kind": kind, "label": label, "n": n, "cmd": cmd}) if ac in want else None
    add("ARIGO-0", "stability", "ARIGO-0 stability", 12, ["arigo1_mujoco.py", *W, "--propulsion", "tractor_lowcam", "tractor", "--name", "arigo0", *f])
    add("ARIGO-1", "stability", "ARIGO-1 stability", 6, ["arigo1_mujoco.py", *W, "--propulsion", "pusher", "--name", "arigo1", *f])
    add("ARIGO-2", "stability", "ARIGO-2 stability + engine out", 6, ["arigo2_mujoco.py", *f])
    add("ARIGO-20", "stability", "ARIGO-20 stability + fan out", 2, ["arigo20_mujoco.py", *f])
    add("ARIGO-0", "camera", "ARIGO-0 propeller in the camera", 1, ["arigo1_camera.py", "--layouts", "tractor_lowcam", "tractor", "pusher"])
    if not a.skip_video:
        add("ARIGO-0", "video", "ARIGO-0 videos", 36, ["arigo1_video.py", *W, "--propulsion", "tractor_lowcam", "tractor", "--name", "arigo0", *f])
        add("ARIGO-1", "video", "ARIGO-1 videos", 18, ["arigo1_video.py", *W, "--propulsion", "pusher", "--name", "arigo1", *f])
        add("ARIGO-2", "video", "ARIGO-2 videos", 21, ["arigo2_video.py", *f])
        add("ARIGO-20", "video", "ARIGO-20 videos", 6, ["arigo20_video.py", *f])
    if not a.skip_cfd:
        P = ["--processors", str(np_)]
        k, sfx, Q, lab = ("cfd_fine", "_fine", ["--quality", "fine"], "full CFD") if a.full_cfd else ("cfd", "", [], "CFD")
        add("ARIGO-1", k, f"ARIGO-1 {lab}", 33, ["arigo1_cfd.py", *W, "--propulsion", "pusher", "--name", "arigo1" + sfx, *P, *Q, *f])
        add("ARIGO-0", k, f"ARIGO-0 {lab}", 66, ["arigo1_cfd.py", *W, "--propulsion", "tractor_lowcam", "tractor", "--name", "arigo0" + sfx, *P, *Q, *f])
        add("ARIGO-20", k, f"ARIGO-20 {lab}", 11, ["arigo20_cfd.py", "--name", "arigo20" + sfx, *P, *Q, *f])
        add("ARIGO-2", k, f"ARIGO-2 {lab}", 48, ["arigo2_cfd.py", "--name", "arigo2" + sfx, *P, *Q, *f])
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rest", nargs="*", help="[processors] [ARIGO-0 ARIGO-1 ARIGO-2 ARIGO-20]")
    ap.add_argument("--stability-only", action="store_true")
    ap.add_argument("--full-cfd", action="store_true", help="CFD at fine quality (long) into arigo*_fine_cfd.yaml")
    ap.add_argument("--skip-video", action="store_true", default=bool(os.environ.get("SKIP_VIDEO")))
    ap.add_argument("--skip-cfd", action="store_true", default=bool(os.environ.get("SKIP_CFD")))
    ap.add_argument("--fresh", action="store_true", default=bool(os.environ.get("FRESH")))
    a = ap.parse_args(argv)
    if a.stability_only:
        a.skip_video = a.skip_cfd = True
    np_ = 8
    want = []
    for r in a.rest:
        if r.isdigit():
            np_ = int(r)
        else:
            want.append(r.upper())
    want = want or ["ARIGO-0", "ARIGO-1", "ARIGO-2", "ARIGO-20"]
    todo = steps(want, a, np_)
    OUT.mkdir(parents=True, exist_ok=True)
    log = open(OUT / "arigo_all.log", "a")
    log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(sys.argv)}\n")
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
    print("Paste back:")
    for f in sorted(OUT.glob("arigo*_*.yaml")):
        if ".stale" not in f.name:
            print("  ", f.relative_to(HERE.parents[2]))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
