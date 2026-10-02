#!/usr/bin/env python3
"""Onager Sentinel on patrol — MuJoCo dynamics through Chiron with two partial failures, one movie per response.

    xvfb-run -a python3 onager/scenarios/onager_patrol.py          # headless (pyvista needs a display)
    python3 onager/scenarios/onager_patrol.py --response limp --fps 25

The scenario is ``onager/designs/onager_scenario.py`` (the same one the Sentinel notebook §7 runs): a gravel road
(RMS 15 mm) with a 120 mm speed bump at 3 m/s; at 6 s the front-left hub motor loses power, at 9 s the rear-right
wheel seizes. ``drag`` keeps driving on the braked tyre; ``limp`` is the three-wheel limp (hull shifted forward,
the seized wheel lifted, 1.5 m/s). Writes ``onager/scenarios/output/onager_patrol_<response>.mp4`` and
``onager_patrol.json`` (the phase tables).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "onager" / "designs"))

import onager_robot as orb  # noqa: E402
import onager_scenario as osc  # noqa: E402

OUT = ROOT / "onager" / "scenarios" / "output"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--response", choices=tuple(osc.RESPONSES), action="append", help="default: both")
    ap.add_argument("--camera", default="follow", help="chiron.viz camera: follow, side, front, top, iso")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--duration", type=float, default=osc.DURATION)
    args = ap.parse_args(argv)
    from vegeta.chiron import viz
    OUT.mkdir(parents=True, exist_ok=True)
    lab = orb.onager_lab(osc.gravel_road(), log_geoms=True)
    results = {}
    for response in args.response or list(osc.RESPONSES):
        ep = osc.run(lab, response, duration=args.duration)
        table = osc.phase_table(ep)
        dt = float(ep.log["t"][1] - ep.log["t"][0])
        every = max(1, int(round(1.0 / (args.fps * dt))))
        imgs = viz.frames(ep, camera=args.camera, every=every, size=(args.width, args.height))
        movie = viz.to_video(imgs, OUT / f"onager_patrol_{response}.mp4", fps=args.fps)
        results[response] = {"events": ep.log["events"], "phases": table.round(3).to_dict(orient="index"),
                             "movie": str(movie), "frames": len(imgs)}
        print(f"{response}: {movie} ({len(imgs)} frames)\n{table.round(2).to_string()}", flush=True)
    (OUT / "onager_patrol.json").write_text(json.dumps(results, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
