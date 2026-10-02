#!/usr/bin/env python3
"""Sikarian Lobster underwater — MuJoCo dynamics through Chiron, three movies (Nefri, or Ornatus with --variant).

    xvfb-run -a python3 scenarios/sikarian_lobster.py                  # Nefri, all three (pyvista needs a display)
    python3 scenarios/sikarian_lobster.py --only cut_and_enter --camera iso
    xvfb-run -a python3 scenarios/sikarian_lobster.py --variant ornatus   # the 15 kg Ornatus cuts the Ø12 mm cable

The scenario is ``notebooks/designs/lobster_scenario.py`` (the same one notebook 24 §9 runs), in fresh water
(MuJoCo's fluid drag, the ``lobster_robot.Water`` hook's buoyancy and vectored thrust):

* ``swim``           released 1 m above the bed, it swims on its tail thruster (the tail points the jet through the
                     CG, a depth loop sets its angle), eases off, lowers its legs and lands;
* ``cut_and_enter``  walks up to a Ø600 mm outfall pipe, cuts the Ø10 mm PP rope across its mouth with the right
                     pincer (the Onager Manus ``WireCutter`` hook: the rope parts at ``F_CUT`` of squeeze), stows the
                     arm and walks 1 m into the pipe;
* ``current``        a 0.5 m/s cross-current: it slides standing; the tail curls over its back, presses it down, and
                     it walks across the current.

Writes ``scenarios/output/sikarian_lobster_<job>.mp4`` and ``.json`` (mission log, events, the cut, the phase
summary); Ornatus's files are ``sikarian_lobster_ornatus_<job>.*``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks" / "designs"))

import lobster_scenario as ls  # noqa: E402

OUT = ROOT / "scenarios" / "output"
JOBS = ("swim", "cut_and_enter", "current")


def summary(kind: str, ep, variant: str = "nefri") -> dict:
    df = ls.timeseries(ep)
    doc = {"job": kind, "variant": variant, "mission": ep.log["mission"], "finished": ep.log["mission_finished"],
           "events": ep.log.get("events", []), "end_x_m": float(df.x.iloc[-1]), "end_y_m": float(df.y.iloc[-1]),
           "max_tilt_deg": float(df.tilt_deg.max())}
    if kind == "swim":
        doc["max_vx_m_s"] = float(df.vx.max())
        doc["max_height_m"] = float(df.z.max())
    if kind == "cut_and_enter":
        doc["cut_at_s"] = ep.log.get("cut_at")
        h = np.asarray(ep.log.get("cutter_history", np.zeros((0, 3)))).reshape(-1, 3)
        doc["peak_squeeze_N"] = float(h[:, 1:].min(axis=1).max()) if len(h) else 0.0
    if kind == "current":
        doc["drift_y_m"] = float(df.y.iloc[-1] - df.y.iloc[0])
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=JOBS, action="append", help="run only this job (repeatable)")
    ap.add_argument("--variant", choices=("nefri", "ornatus"), default="nefri", help="which Lobster (default Nefri)")
    ap.add_argument("--camera", default="follow", help="chiron.viz camera: follow, side, front, top, iso")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--no-movie", action="store_true", help="run the missions, write the json, skip rendering")
    args = ap.parse_args(argv)
    from vegeta.chiron import viz
    OUT.mkdir(parents=True, exist_ok=True)
    status = 0
    for kind in args.only or JOBS:
        t0 = time.time()
        scene = ls.Scene(args.variant)
        extra = dict(wind=(0.0, 0.5, 0.0), course_extent=(-2.0, 6.0, -2.0, 6.0)) if kind == "current" else {}
        lab = ls.make_lab(scene, kind, log_geoms=not args.no_movie, **extra)
        ep = ls.run(kind, scene, lab=lab)
        doc = summary(kind, ep, args.variant)
        stem = f"sikarian_lobster_{kind}" if args.variant == "nefri" else f"sikarian_lobster_{args.variant}_{kind}"
        if not args.no_movie:
            dt = float(ep.log["t"][1] - ep.log["t"][0])
            every = max(1, int(round(1.0 / (args.fps * dt))))
            imgs = viz.frames(ep, camera=args.camera, every=every, size=(args.width, args.height))
            doc["movie"] = str(viz.to_video(imgs, OUT / f"{stem}.mp4", fps=args.fps))
            doc["frames"] = len(imgs)
        (OUT / f"{stem}.json").write_text(json.dumps(doc, indent=1, default=float))
        status |= 0 if doc["finished"] else 1
        keys = [k for k in ("max_vx_m_s", "cut_at_s", "peak_squeeze_N", "drift_y_m") if k in doc]
        print(f"{args.variant} {kind}: finished={doc['finished']} end x={doc['end_x_m']:.2f} m, max tilt {doc['max_tilt_deg']:.0f}°, "
              + ", ".join(f"{k}={doc[k]}" for k in keys) + f" ({time.time() - t0:.0f} s) -> {doc.get('movie', '-')}")
    return status


if __name__ == "__main__":
    sys.exit(main())
