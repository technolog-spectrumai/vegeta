"""MERLIN's propulsor race (notebook 27, branch dev_rave): can a reasonable ducted fan beat the propellers on the same battery?

    python scenarios/merlin_race.py                 # the whole design space: 108 fans + 24 propellers, 3 powers (~5 min on 4 cores)
    python scenarios/merlin_race.py --small         # a corner of it (a check, ~1 min)

Builds every ducted fan and propeller in ``merlin_race.EDF_SPACE`` / ``PROP_SPACE``, races each at 1900, 2700 and 3500 W
on the same 6S 8000 mAh pack — 5 km and back, 10 km reach, 10 km and back — and prints the fastest of each kind.
Writes ``scenarios/output/merlin_race.csv`` (every configuration and power) and ``merlin_race.json`` (the winners).
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks" / "designs"))
import merlin_race as mr  # noqa: E402

OUT = ROOT / "scenarios" / "output"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--small", action="store_true", help="two fan sizes and two propellers only")
    ap.add_argument("-j", type=int, default=None, help="processes (default: all cores)")
    a = ap.parse_args(argv)
    edf, props = dict(mr.EDF_SPACE), dict(mr.PROP_SPACE)
    if a.small:
        edf.update(diameter_mm=(90.0, 120.0), pitch_ratio=(1.78,), exit_area_ratio=(0.8,))
        props.update(diameter_in=(10.0,), pitch_ratio=(1.0,))
    cfgs = mr.configs(edf, props)
    t0 = time.time()
    units = mr.tabulate(cfgs, processes=a.j, progress=True)
    failed = {c.label: n["error"] for c, (u, n) in units.items() if u is None}
    table = mr.study(units)
    print(f"{len(cfgs)} configurations x {len(mr.POWERS_W)} powers in {time.time() - t0:.0f} s"
          + (f"; {len(failed)} could not be built: {failed}" if failed else ""))
    import pandas as pd
    pd.set_option("display.width", 220)
    best = {}
    for race in mr.RACES:
        w = mr.winners(table, race)
        best[race] = w[["config", race, "top speed [m/s]", "mass [kg]"]].reset_index().to_dict("records")
        print(f"\n{race}: the fastest of each kind at each power")
        print(w[["config", race, "top speed [m/s]", "mass [kg]"]].round(1).to_string())
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "merlin_race.csv", index=False)
    (OUT / "merlin_race.json").write_text(json.dumps({"winners": best, "failed": failed}, indent=1, default=float))
    print("\nwritten", OUT / "merlin_race.csv", "and merlin_race.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
