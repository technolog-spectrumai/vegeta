"""Build the propulsor libraries — every propeller and ducted fan of the propulsion notebooks' design spaces as (airspeed x
rpm) maps — and write them to ``notebooks/designs/data/`` for the reduced flight models (MERLIN, notebooks 26b and 27). The
same as the last section of each notebook, headless:

    python scenarios/propulsor_maps.py                      # all three (~12 min on 4 cores)
    python scenarios/propulsor_maps.py --kind propellers    # notebook 25: 2 and 3 blades, 9-12 inch, 4 pitches (32 maps, ~1 min)
    python scenarios/propulsor_maps.py --kind exotic        # notebook 25c: 6 and 12 blades (32 maps, ~1 min)
    python scenarios/propulsor_maps.py --kind edf -j 8      # notebook 25b: 324 fans (1-3 stages) + 36 lossless bound fans
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
    ap.add_argument("--kind", choices=tuple(pm.LIBRARIES) + ("all",), default="all")
    ap.add_argument("-j", type=int, default=None, help="processes (default: all cores)")
    a = ap.parse_args(argv)
    status = 0
    for kind in (pm.LIBRARIES if a.kind == "all" else (a.kind,)):
        t0 = time.time()
        lib = pm.build_library(kind, processes=a.j, progress=True)
        bad = [e for e in lib["entries"] if "error" in e]
        path = pm.save(lib)
        print(f"{kind}: {len(lib['entries']) - len(bad)} maps in {time.time() - t0:.0f} s -> {path} ({path.stat().st_size / 1e6:.2f} MB)"
              + (f"; {len(bad)} failed: {[(e['id'], e['error']) for e in bad]}" if bad else ""), flush=True)
        status |= bool(bad)
    return status


if __name__ == "__main__":
    sys.exit(main())
