#!/usr/bin/env python3
"""Onager Sweeper cleans a street — MuJoCo dynamics through Chiron, one movie.

    xvfb-run -a python3 scenarios/onager_sweeper_street.py            # headless (pyvista needs a display)
    python3 scenarios/onager_sweeper_street.py --camera iso

The scenario is ``notebooks/designs/onager_sweeper_scenario.py`` (the same one notebook 23 §7 runs): litter (cans,
packets) in the sweeping path, a 2.3 kg brick and a 0.6 kg box on the centre line. The Sweeper spins its disc broom,
runs the fan (the CFD's suction, ``onager_sweeper_cfd.SUCTION``, acts on the litter under the hood: what reaches the
duct is collected), sweeps at 1 m/s, stops at the brick, takes it with its left pincer and swings it back into the
basket on the roof, does the same with the box with the right pincer, and sweeps on. Writes
``scenarios/output/onager_sweeper_street.mp4`` and ``onager_sweeper_street.json`` (mission log, events, what was
vacuumed, what landed in the basket, the phase table).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks" / "designs"))

import onager_sweeper_scenario as oss  # noqa: E402

OUT = ROOT / "scenarios" / "output"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", default="follow", help="chiron.viz camera: follow, side, front, top, iso")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--duration", type=float, default=95.0)
    args = ap.parse_args(argv)
    from vegeta.chiron import viz
    OUT.mkdir(parents=True, exist_ok=True)
    scene = oss.Scene()
    lab = oss.make_lab(scene, log_geoms=True)
    ep = oss.run(lab, scene, duration=args.duration)
    table = oss.phase_table(ep)
    dt = float(ep.log["t"][1] - ep.log["t"][0])
    every = max(1, int(round(1.0 / (args.fps * dt))))
    imgs = viz.frames(ep, camera=args.camera, every=every, size=(args.width, args.height))
    movie = viz.to_video(imgs, OUT / "onager_sweeper_street.mp4", fps=args.fps)
    basket = {name: oss.in_basket(ep, name) for name, _, _ in scene.pickups}
    doc = {"mission": ep.log["mission"], "finished": ep.log["mission_finished"], "events": ep.log["events"],
           "vacuumed": ep.log["collected"], "in_basket": basket,
           "phases": table.round(3).to_dict(orient="index"), "movie": str(movie), "frames": len(imgs)}
    (OUT / "onager_sweeper_street.json").write_text(json.dumps(doc, indent=1, default=float))
    print(f"{movie} ({len(imgs)} frames); vacuumed {sorted(ep.log['collected'])}; in the basket {basket}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
