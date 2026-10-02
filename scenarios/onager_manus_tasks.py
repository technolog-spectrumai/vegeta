#!/usr/bin/env python3
"""Onager Manus clears a blocked track — MuJoCo dynamics through Chiron, one movie.

    xvfb-run -a python3 scenarios/onager_manus_tasks.py            # headless (pyvista needs a display)
    python3 scenarios/onager_manus_tasks.py --camera iso

The scenario is ``notebooks/designs/onager_manus_scenario.py`` (the same one notebook 22 §7 runs): a fence wire
across the track at 1.15 m (Ø 3.15 mm, 1200 MPa: 7.5 kN to cut) and a 14 kg log further on. The Manus stops at the
wire, puts its right pincer's cutter notch on it and closes — the wire parts only if both jaws squeeze it with the
cutting force — drives through, crouches at the log, takes it with its left pincer's hooked jaws, lifts it, swings
it over the side, puts it down and drives on. Writes ``scenarios/output/onager_manus_tasks.mp4`` and
``onager_manus_tasks.json`` (mission log, events, phase table).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks" / "designs"))

import onager_manus_scenario as oms  # noqa: E402

OUT = ROOT / "scenarios" / "output"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", default="follow", help="chiron.viz camera: follow, side, front, top, iso")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--duration", type=float, default=66.0)
    args = ap.parse_args(argv)
    from vegeta.chiron import viz
    OUT.mkdir(parents=True, exist_ok=True)
    scene = oms.Scene()
    lab = oms.make_lab(scene, log_geoms=True)
    ep = oms.run(lab, scene, duration=args.duration)
    table = oms.phase_table(ep)
    dt = float(ep.log["t"][1] - ep.log["t"][0])
    every = max(1, int(round(1.0 / (args.fps * dt))))
    imgs = viz.frames(ep, camera=args.camera, every=every, size=(args.width, args.height))
    movie = viz.to_video(imgs, OUT / "onager_manus_tasks.mp4", fps=args.fps)
    doc = {"mission": ep.log["mission"], "finished": ep.log["mission_finished"], "events": ep.log["events"],
           "phases": table.round(3).to_dict(orient="index"), "movie": str(movie), "frames": len(imgs)}
    (OUT / "onager_manus_tasks.json").write_text(json.dumps(doc, indent=1, default=float))
    print(f"{movie} ({len(imgs)} frames); events {ep.log['events']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
