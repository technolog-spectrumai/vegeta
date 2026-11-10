#!/usr/bin/env python3
"""The mixed-fleet civilian response scenario — two Pekari-2M, Pumba and a survey drone at an industrial incident
(``designs/fleet_scenario.py``, deterministic). Prints the task table and the events that matter, writes
``sims/output/fleet_response.json`` and the shared map ``fleet_response.png``.

    python3 labs/land/sims/fleet_response.py
    python3 labs/land/sims/fleet_response.py --no-drone --no-fault --no-estop
    python3 labs/land/sims/fleet_response.py --docs                 # also copy the figure to docs/figures/
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "labs" / "land" / "designs"))

import fleet_scenario as fs  # noqa: E402

OUT = ROOT / "labs" / "land" / "sims" / "output"
QUIET = {"task_added", "explored", "no_path", "link_lost", "reconnected"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-drone", action="store_true")
    ap.add_argument("--no-fault", action="store_true")
    ap.add_argument("--no-estop", action="store_true")
    ap.add_argument("--t-end", type=float, default=2400.0)
    ap.add_argument("--docs", action="store_true")
    a = ap.parse_args(argv)
    fleet = fs.build(with_drone=not a.no_drone, fault=not a.no_fault, estop=not a.no_estop)
    res = fleet.run(a.t_end)
    tb = fs.table(res)
    print(tb[~tb.index.str.startswith("explore")].to_string())
    n_exp = tb.index.str.startswith("explore").sum()
    print(f"\n{n_exp} exploration tasks; coverage {res['coverage']:.0%}; finished at {res['t_end']:.0f} s\n")
    for t, ev, kw in res["events"]:
        if ev in QUIET or str(kw.get("task", "")).startswith("explore"):
            continue
        print(f"{t:7.0f} s  {ev:20s} {kw}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fleet_response.json").write_text(json.dumps(res, indent=1, default=str))
    fs.plot(fleet, OUT / "fleet_response.png")
    if a.docs:
        (ROOT / "docs" / "figures").mkdir(parents=True, exist_ok=True)
        shutil.copy(OUT / "fleet_response.png", ROOT / "docs" / "figures" / "fleet_response.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
