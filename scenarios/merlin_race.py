"""MERLIN's propulsor race (notebook 27, branch dev_rave): can a reasonable ducted fan beat the propellers on the same battery?

    python scenarios/merlin_race.py                 # every map of the library, 3 powers (~1 min)

Reads notebook 25's propulsor library (``notebooks/designs/data/propulsor_maps.json``: 324 ducted fans with one to three
stages, 48 propellers of 2, 6 and 12 blades as tractor and pusher; rebuild it with ``scenarios/propulsor_maps.py``), races each at 1900, 2700
and 3500 W on the same 6S 8000 mAh pack — 5 km and back, 10 km reach, 10 km and back — and prints the fastest of each kind.
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
    ap.add_argument("--library", type=Path, default=None, help="propulsor library (default: the committed one)")
    a = ap.parse_args(argv)
    import propulsor_maps as pm
    t0 = time.time()
    lib = pm.load(a.library or pm.LIBRARY)
    table = mr.study(lib)
    failed = {e["id"]: e["error"] for e in lib["entries"] if "error" in e}
    print(f"{len(table)} races (maps x layouts x powers) in {time.time() - t0:.0f} s" + (f"; {len(failed)} maps failed" if failed else ""))
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
