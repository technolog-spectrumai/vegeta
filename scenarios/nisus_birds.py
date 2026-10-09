#!/usr/bin/env python3
"""NISUS-Zero bird photography in MuJoCo: the mission software (vegeta.mission) tracks the birds, picks them one by
one and flies stern photo passes; the autopilot launches, climbs, returns and lands (notebook 32).

    xvfb-run -a python3 scenarios/nisus_birds.py                         # the six standard bird scenarios with videos
    xvfb-run -a python3 scenarios/nisus_birds.py --scenario "pigeon flock"
    python3 scenarios/nisus_birds.py --no-movie                          # the numbers only
    python3 scenarios/nisus_birds.py --list

YOLOX is not run: the detections are simulated from the truth with the latency and rates of the ASSUMED Jetson budget
(vegeta.mission.compute). Writes ``scenarios/output/<slug>.mp4``, ``<slug>_photos.csv`` and ``nisus_birds.json``
(the summary per scenario).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks" / "designs"))

import nisus_birds as nb  # noqa: E402

OUT = ROOT / "scenarios" / "output"


def _key(bs):
    return f"{bs.name} / {bs.device} / {bs.lens}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="all", help='a scenario key from --list (or its first words, e.g. "pigeon flock"), or "all"')
    ap.add_argument("--no-movie", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    all_s = nb.bird_scenarios()
    sel = [s for s in all_s if args.scenario == "all" or _key(s).startswith(args.scenario) or _key(s) == args.scenario]
    if args.list or not sel:
        for s in all_s:
            print(_key(s))
        return 0 if args.list else 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = {}
    for bs in sel:
        print(f"== {_key(bs)}", flush=True)
        ep = nb.run(bs)
        s = nb.summary(ep)
        rows[_key(bs)] = s
        for k, v in s.items():
            print(f"   {k}: {v}")
        nb.photo_table(ep).to_csv(out / f"{bs.slug}_photos.csv", index=False)
        if not args.no_movie:
            import nisus_birds_movie as nm
            p = nm.render_movie(ep, out / f"{bs.slug}.mp4")
            print(f"   movie: {p}", flush=True)
    (out / "nisus_birds.json").write_text(json.dumps(rows, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
