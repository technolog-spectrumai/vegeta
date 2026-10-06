#!/usr/bin/env python3
"""Drongo delivers a potato and a cup of cream to hungry people — MuJoCo through Chiron, a movie per variant.

    xvfb-run -a python3 scenarios/drongo_delivery.py                 # both variants (pyvista needs a display)
    xvfb-run -a python3 scenarios/drongo_delivery.py --variant drop  # into the net (at most 5 m above it)
    python3 scenarios/drongo_delivery.py --no-movie                  # the numbers only

The scenario is ``notebooks/designs/drongo_scenario.py`` (the one notebook 08b runs): notebook 08's quadcopter with skids
and a pincer takes a 200 g potato and then a 412 g cup of cream from the kitchen door to people 36 m away, dropping each
into their net (``drop``) or putting it down on the zone (``place``). Writes ``scenarios/output/drongo_<variant>.mp4``
and ``drongo_delivery.json`` (wait times, deliveries, outcome).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks" / "designs"))

import drongo_scenario as ds  # noqa: E402

OUT = ROOT / "scenarios" / "output"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=ds.VARIANTS + ("both",), default="both")
    ap.add_argument("--speed", type=float, default=2.0, help="movie speed (× real time)")
    ap.add_argument("--no-movie", action="store_true")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    doc = {}
    for v in (ds.VARIANTS if args.variant == "both" else (args.variant,)):
        scene = ds.Scene(v)
        lab = ds.make_lab(scene, log_geoms=not args.no_movie)
        ep = ds.run(lab, scene)
        doc[v] = {"wait": ds.wait_times(ep), "outcome": ep.outcome, "energy_Wh": ep.log["energy_Wh"],
                  "deliveries": ds.deliveries(ep).reset_index().to_dict("records")}
        if not args.no_movie:
            doc[v]["movie"] = str(ds.render_movie(ep, scene, OUT / f"drongo_{v}.mp4", speed=args.speed))
        w = doc[v]["wait"]
        print(f"{v}: potato {w['potato [s]']:.1f} s, cream {w['cream [s]']:.1f} s — {ep.outcome['reason']}: {ep.outcome['detail']}")
    (OUT / "drongo_delivery.json").write_text(json.dumps(doc, indent=1, default=str))
    return 0 if all(d["outcome"]["success"] for d in doc.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
