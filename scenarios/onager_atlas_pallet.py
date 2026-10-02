#!/usr/bin/env python3
"""Onager Atlas moves a pallet — MuJoCo dynamics through Chiron, one movie.

    xvfb-run -a python3 scenarios/onager_atlas_pallet.py           # headless (pyvista needs a display)
    python3 scenarios/onager_atlas_pallet.py --camera side --fps 25

The scenario is ``notebooks/designs/onager_atlas_scenario.py`` (the same one notebook 21 §7 runs): a Euro pallet
with a 175 kg crate on a gravel yard; the Atlas drives up with its forks at travel height, lowers them to the
openings, creeps in, lifts, tilts the mast back, carries the pallet 8 m, sets it down and backs out. Writes
``scenarios/output/onager_atlas_pallet.mp4`` and ``onager_atlas_pallet.json`` (the phase table).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks" / "designs"))

import onager_atlas_scenario as oas  # noqa: E402

OUT = ROOT / "scenarios" / "output"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", default="follow", help="chiron.viz camera: follow, side, front, top, iso")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--duration", type=float, default=50.0)
    args = ap.parse_args(argv)
    from vegeta.chiron import viz
    OUT.mkdir(parents=True, exist_ok=True)
    scene = oas.Scene()
    lab = oas.make_lab(scene, log_geoms=True)
    ep = oas.run(lab, scene, duration=args.duration)
    table = oas.phase_table(ep)
    dt = float(ep.log["t"][1] - ep.log["t"][0])
    every = max(1, int(round(1.0 / (args.fps * dt))))
    imgs = viz.frames(ep, camera=args.camera, every=every, size=(args.width, args.height))
    movie = viz.to_video(imgs, OUT / "onager_atlas_pallet.mp4", fps=args.fps)
    doc = {"mission": ep.log["mission"], "finished": ep.log["mission_finished"], "events": ep.log.get("events", []),
           "phases": table.round(3).to_dict(orient="index"), "movie": str(movie), "frames": len(imgs)}
    (OUT / "onager_atlas_pallet.json").write_text(json.dumps(doc, indent=1, default=float))
    print(f"{movie} ({len(imgs)} frames)\n{table.round(2).to_string()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
