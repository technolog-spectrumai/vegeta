"""Build the propulsor library — every fan and propeller of notebook 25's design space as (airspeed x rpm) maps — and write
``notebooks/designs/data/propulsor_maps.json`` for the reduced flight models (MERLIN, notebooks 26 and 27). The same as
notebook 25 §16, headless.

    python scenarios/propulsor_maps.py              # 324 ducted fans (1-3 stages), 36 lossless bound fans, 48 propellers (2, 6, 12 blades) (~12 min on 4 cores)
    python scenarios/propulsor_maps.py -j 8 --out /tmp/maps.json
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks" / "designs"))
import propulsor_maps as pm  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-j", type=int, default=None, help="processes (default: all cores)")
    ap.add_argument("--out", type=Path, default=pm.LIBRARY)
    a = ap.parse_args(argv)
    t0 = time.time()
    lib = pm.build_library(processes=a.j, progress=True)
    bad = [e for e in lib["entries"] if "error" in e]
    path = pm.save(lib, a.out)
    print(f"{len(lib['entries']) - len(bad)} maps in {time.time() - t0:.0f} s -> {path} ({path.stat().st_size / 1e6:.1f} MB)"
          + (f"; {len(bad)} failed: {[(e['id'], e['error']) for e in bad]}" if bad else ""))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
