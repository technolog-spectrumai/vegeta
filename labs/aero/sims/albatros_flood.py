#!/usr/bin/env python3
"""Albatros over a flooded town: Émerillons bring medicine to the people on the roofs — MuJoCo, a movie and stills.

    MUJOCO_GL=glfw xvfb-run -a python3 labs/aero/sims/albatros_flood.py          # both payloads, movie of the saddle bags
    python3 labs/aero/sims/albatros_flood.py --no-movie                          # the numbers only (no display needed)
    python3 labs/aero/sims/albatros_flood.py --payload box --drones 4 --seed 3   # one variant, another town

The scenario is ``designs/albatros_flood.py`` (notebook 50, section 12): a river town of eastern Poland or Lithuania
in 2.4 m of water; Albatros flies one pass and releases its Émerillons; each flies to its roof and drops a saddle
bag across the ridge beside the people (``--payload box``: a single box, which slides off) or lands on a flat roof
with the medicine. Writes ``sims/output/albatros_flood.json`` (every delivery and the summary),
``albatros_flood_<payload>.mp4`` and four stills.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "labs" / "aero" / "designs"))

import albatros_flood as fl  # noqa: E402

OUT = ROOT / "labs" / "aero" / "sims" / "output"


def stills(run, stem: Path) -> list:
    """Four pictures of a run: the pass, the hover over a ridge, the bag on the roof, the carriers on the school."""
    import numpy as np
    from PIL import Image

    pl = run["pilots"][0]
    td = pl.events.get("bag released", pl.events.get("landed", run["time_s"]))
    tgt = run["scene"].recipients()[0]["target"]
    school = next(q for q in run["scene"].buildings if q.kind == "school")
    i = int(np.searchsorted(run["log"]["t"], run["release_times"][-1] + 0.6))
    shots = {"town": dict(t=run["release_times"][0] + 1.5),
             "release": dict(t=run["release_times"][-1] + 0.6, lookat=run["log"]["mocap"][min(i, len(run["log"]["t"]) - 1)], distance=45.0, azimuth=70.0, elevation=-8.0),
             "hover": dict(t=td - 0.3, lookat=tgt, distance=9.0, azimuth=40.0, elevation=-15.0),
             "bag": dict(t=td + 3.0, lookat=tgt, distance=5.0, azimuth=20.0, elevation=-25.0),
             "collected": dict(t=run["time_s"], lookat=(school.x + 5.0, school.y - 2.0, school.wall_h), distance=22.0, azimuth=200.0, elevation=-25.0)}
    paths = []
    for name, kw in shots.items():
        path = Path(f"{stem}_{name}.png")
        Image.fromarray(fl.frame(run, **kw)).save(path)
        paths.append(str(path))
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--payload", choices=("saddle", "box", "both"), default="both")
    ap.add_argument("--drones", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--speed", type=float, default=2.0, help="movie speed (x real time; the drop is shown at 1x)")
    ap.add_argument("--no-movie", action="store_true")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    doc = {}
    for payload in (("saddle", "box") if args.payload == "both" else (args.payload,)):
        run = fl.run(fl.Scene(seed=args.seed, drones=args.drones, payload=payload))
        s = fl.summary(run)
        doc[payload] = {"summary": s, "deliveries": run["deliveries"]}
        print(f"{payload}: {s['delivered']} of {s['roofs']} roofs, {s['people reached']} of {s['people waiting']} people, "
              f"{s['first delivery [s]']}-{s['last delivery [s]']} s after the release; {s['carriers landed']} carriers landed")
        if not args.no_movie:
            doc[payload]["stills"] = stills(run, OUT / f"albatros_flood_{payload}")
            if payload == "saddle" or args.payload != "both":
                doc[payload]["movie"] = fl.render_movie(run, OUT / f"albatros_flood_{payload}.mp4", speed=args.speed)
    (OUT / "albatros_flood.json").write_text(json.dumps(doc, indent=1, default=str))
    return 0 if all(d["summary"]["delivered"] for d in doc.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
