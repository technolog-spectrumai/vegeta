#!/usr/bin/env python3
"""Nisus+ Zero mountain survey missions in MuJoCo through Chiron — a bungee launch from the valley meadow (1200 m), a
spiral climb to 3650 m, the transit to the north face, the terrain-following survey legs, the return, the crow +
propeller-brake descent (with regeneration), the steep crow approach and the belly landing — with a movie per
scenario (notebook 33, Part 11).

    xvfb-run -a python3 scenarios/nisus_plus_mission.py                         # the six standard scenarios (pyvista needs a display)
    xvfb-run -a python3 scenarios/nisus_plus_mission.py --condition "storm: weather escape"
    python3 scenarios/nisus_plus_mission.py --no-movie                          # the numbers only
    python3 scenarios/nisus_plus_mission.py --list                              # the condition names

The scenarios are ``notebooks/designs/nisus_plus_scenario.standard_scenarios()``: calm, the ridge lift of a south wind, the
lee downdraft of a north wind, a hot day (ISA +20 °C), a storm warning (the return and the fastest descent) and the
Jetson's failure. Writes ``scenarios/output/<Scenario.slug>.mp4`` (e.g. ``nisus_plus_calm.mp4``), the telemetry ``.csv`` and
``nisus_plus_mission.json`` (outcome, phases, energy by phase, the comparison).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks" / "designs"))

import nisus_plus_scenario as fsc  # noqa: E402

OUT = ROOT / "scenarios" / "output"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", default="all", help='a condition name (see --list) or "all"')
    ap.add_argument("--speed", type=float, default=30.0, help="movie speed (× real time)")
    ap.add_argument("--no-movie", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    scns = [s for s in fsc.standard_scenarios() if args.condition in ("all", s.name)]
    if args.list or not scns:
        for s in fsc.standard_scenarios():
            print(f'--condition "{s.name}"')
        return 0 if args.list else 2
    OUT.mkdir(parents=True, exist_ok=True)
    doc, eps = {}, []
    for scn in scns:
        lab = fsc.make_lab(scn, log_geoms=not args.no_movie)
        ep = fsc.run(lab, scn)
        eps.append(ep)
        fsc.timeseries(ep).to_csv(OUT / f"{scn.slug}.csv", index=False)
        doc[scn.label] = {"outcome": ep.outcome, "phases": fsc.phase_table(ep).reset_index().to_dict("records"),
                          "energy": fsc.energy_table(ep).reset_index().to_dict("records"), "events": ep.log["events"]}
        if not args.no_movie:
            doc[scn.label]["movie"] = str(fsc.render_movie(ep, scn, OUT / f"{scn.slug}.mp4", speed=args.speed))
        print(f"{scn.label:50s} {ep.outcome['reason']:22s} {ep.outcome['detail']}", flush=True)
    doc["comparison"] = fsc.compare(eps).reset_index().to_dict("records")
    (OUT / "nisus_plus_mission.json").write_text(json.dumps(doc, indent=1, default=str))
    return 0 if all(e.outcome["success"] for e in eps) else 1


if __name__ == "__main__":
    sys.exit(main())
