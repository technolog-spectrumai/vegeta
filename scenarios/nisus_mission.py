#!/usr/bin/env python3
"""NISUS survey missions in MuJoCo through Chiron — launch, climb, outbound, survey, energy-triggered return, approach
and landing — with a movie per scenario (notebook 31, Part 11).

    xvfb-run -a python3 scenarios/nisus_mission.py                              # the six standard scenarios (pyvista needs a display)
    xvfb-run -a python3 scenarios/nisus_mission.py --variant Zero --condition "Jetson failure"
    python3 scenarios/nisus_mission.py --no-movie                               # the numbers only
    python3 scenarios/nisus_mission.py --list                                   # the scenario names

The scenarios are ``notebooks/designs/nisus_scenario.standard_scenarios()`` (the ones notebook 31 flies): Nisus-OBS and
Nisus-Zero in calm air and in a 6 m/s headwind with turbulence, Zero in gusts and with a Jetson failure (its power
branch drops at 300 s; the flight controller returns on its own). Writes ``scenarios/output/<Scenario.slug>.mp4`` (e.g. ``nisus_Zero_headwind_6_ms.mp4``),
the telemetry ``.csv`` and ``nisus_mission.json`` (outcome, phases, energy by phase, the comparison).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks" / "designs"))

import nisus_scenario as nsc  # noqa: E402

OUT = ROOT / "scenarios" / "output"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=("OBS", "Zero", "both"), default="both")
    ap.add_argument("--condition", default="all", help='a condition name ("calm", "headwind 6 m/s", "gusty", "Jetson failure") or "all"')
    ap.add_argument("--speed", type=float, default=25.0, help="movie speed (× real time)")
    ap.add_argument("--no-movie", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    scns = [s for s in nsc.standard_scenarios() if args.variant in ("both", s.variant) and args.condition in ("all", s.name)]
    if args.list or not scns:
        for s in nsc.standard_scenarios():
            print(f'--variant {s.variant} --condition "{s.name}"')
        return 0 if args.list else 2
    OUT.mkdir(parents=True, exist_ok=True)
    doc, eps = {}, []
    for scn in scns:
        lab = nsc.make_lab(scn, log_geoms=not args.no_movie)
        ep = nsc.run(lab, scn)
        eps.append(ep)
        stem = scn.slug
        nsc.timeseries(ep).to_csv(OUT / f"{stem}.csv", index=False)
        doc[scn.label] = {"outcome": ep.outcome, "phases": nsc.phase_table(ep).reset_index().to_dict("records"),
                          "energy": nsc.energy_table(ep).reset_index().to_dict("records"), "events": ep.log["events"]}
        if not args.no_movie:
            doc[scn.label]["movie"] = str(nsc.render_movie(ep, scn, OUT / f"{stem}.mp4", speed=args.speed))
        print(f"{scn.label:40s} {ep.outcome['reason']:22s} {ep.outcome['detail']}")
    doc["comparison"] = nsc.compare(eps).reset_index().to_dict("records")
    (OUT / "nisus_mission.json").write_text(json.dumps(doc, indent=1, default=str))
    return 0 if all(e.outcome["success"] for e in eps) else 1


if __name__ == "__main__":
    sys.exit(main())
