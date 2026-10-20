#!/usr/bin/env python3
"""Plot the autopilot tuner's training history (races/trainviz.py): the loss history, the search projected on its
two principal components (with CMA-ES's sampling ellipses), the settings per step, and optionally an animation.

    python viz_autopilot_train_hist.py                       # every course that has a checkpoint or CSV in races/runs/tune/
    python viz_autopilot_train_hist.py --courses hangar --animate
    python viz_autopilot_train_hist.py --from races/runs/tune/hangar.ckpt.json      # a given checkpoint or CSV

Figures: races/runs/tune/train_hist/<course>_train_history.png, <course>_params.png (and <course>_search.gif). No
MuJoCo: it only reads what ./tune_autopilot.sh wrote."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main(argv=None) -> int:
    sys.path.insert(0, str(ROOT))
    from races import courses, trainviz
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--courses", nargs="*", default=None, choices=list(courses.COURSES), help="default: every course with a training history")
    ap.add_argument("--from", dest="source", type=Path, default=None, help="a checkpoint (.ckpt.json) or CSV (_converge.csv) to plot (one course)")
    ap.add_argument("--out", type=Path, default=trainviz.TUNE / "train_hist")
    ap.add_argument("--animate", action="store_true", help="also a GIF of the search plane, generation by generation")
    ap.add_argument("--dpi", type=int, default=110)
    a = ap.parse_args(argv)
    if a.source:
        jobs = [(a.source.name.split(".")[0].split("_converge")[0], a.source)]
    else:
        have = [s for s in courses.COURSES if (trainviz.TUNE / f"{s}.ckpt.json").exists() or (trainviz.TUNE / f"{s}_converge.csv").exists()]
        jobs = [(s, None) for s in (a.courses or have)]
    if not jobs:
        print(f"viz_autopilot_train_hist: no training history in {trainviz.TUNE} (run ./tune_autopilot.sh first)")
        return 2
    rc = 0
    for slug, src in jobs:
        try:
            h = trainviz.load(slug, src)
        except (OSError, ValueError) as e:
            print(f"== {slug}: no training history ({e})")
            rc = 1
            continue
        print(f"== {trainviz.summary(h)}")
        for f in trainviz.figures(h, a.out, animate=a.animate, dpi=a.dpi):
            print(f"   {f}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
