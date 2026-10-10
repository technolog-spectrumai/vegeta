#!/usr/bin/env python3
"""Falco-Zero chasing birds in the mountains: NISUS-Zero's mission software (vegeta.mission) on the tractor aircraft (the camera looks past the propeller's hub)
over the north face — golden eagles in the ridge lift, griffon vultures circling, a flock of alpine choughs, the
Jetson failing mid-hunt; the autopilot launches, climbs, returns, descends in crow and lands (notebook 35, Part 11b).

    xvfb-run -a python3 scenarios/falco_birds.py                         # the four bird scenarios with videos
    xvfb-run -a python3 scenarios/falco_birds.py --scenario "griffon vultures"
    python3 scenarios/falco_birds.py --no-movie                          # the numbers only
    python3 scenarios/falco_birds.py --list

YOLOX is not run: the detections are simulated from the truth (vegeta.mission.sim) with the ASSUMED Jetson budget.
Writes ``scenarios/output/<slug>.mp4``, ``<slug>_photos.csv`` and ``falco_birds.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks" / "designs"))

import falco_birds as fb  # noqa: E402

OUT = ROOT / "scenarios" / "output"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="all", help='a scenario name from --list (or its first words), or "all"')
    ap.add_argument("--no-movie", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    all_s = fb.bird_scenarios()
    sel = [s for s in all_s if args.scenario == "all" or s.name.startswith(args.scenario)]
    if args.list or not sel:
        for s in all_s:
            print(s.name)
        return 0 if args.list else 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, ok = {}, True
    for bs in sel:
        print(f"== {bs.name}", flush=True)
        ep = fb.run(bs, log_geoms=not args.no_movie)
        s = fb.summary(ep)
        rows[bs.name] = s
        ok &= bool(ep.outcome["success"])
        for k, v in s.items():
            print(f"   {k}: {v}")
        fb.photo_table(ep).to_csv(out / f"{bs.slug}_photos.csv", index=False)
        if not args.no_movie:
            import falco_birds as fm
            print(f"   movie: {fm.render_movie(ep, out / f'{bs.slug}.mp4')}", flush=True)
    (out / "falco_birds.json").write_text(json.dumps(rows, indent=2, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
