"""Re-run ÉMERILLON's trade — every build the catalogue allows, its top speed and its price — and write it where
notebook 37 reads it (``labs/aero/designs/data/emerillon_trade.json``). About a minute and a half; no solver.

    .venv/bin/python labs/aero/sims/emerillon_trade.py            # the whole catalogue
    .venv/bin/python labs/aero/sims/emerillon_trade.py --quick    # a dozen builds, nothing written (the mocks do this)
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "designs"))
import emerillon_flight as ef  # noqa: E402
import emerillon_market as mk  # noqa: E402

OUT = HERE.parent / "designs" / "data" / "emerillon_trade.json"


def main(quick: bool) -> int:
    t0 = time.time()
    configs = ef.candidates()
    if quick:
        configs = ef.candidates(motors=mk.MOTORS[:1] + mk.MOTORS[-1:], propellers=mk.PROPELLERS[3:5], batteries=mk.BATTERIES[2:3])
    rows = ef.trade(configs)
    for label, limit in (("any rpm", False), ("inside the propeller makers' limits", True)):
        pk = ef.picks(rows, limit)
        print(f"-- {label}: {len(pk.get('front', []))} builds on the front")
        for r in pk.get("front", []):
            mark = {id(pk["budget"]): "budget", id(pk["knee"]): "knee", id(pk["fastest"]): "fastest"}.get(id(r), "")
            print(f"  {r['price']:7.0f} USD  {r['kmh']:5.0f} km/h  {r['mass_g']:5.0f} g  {r['label']}  {mark}")
    print(f"{len(rows)} builds, {sum(r['feasible'] for r in rows)} feasible, {time.time() - t0:.0f} s")
    if not quick:
        head = json.dumps({"priced_on": mk.PRICED_ON, "drag_margin": ef.DRAG_MARGIN, "print_factor": ef.PRINT_FACTOR})[:-1]
        OUT.write_text(head + ', "rows": [\n' + ",\n".join(json.dumps(r) for r in ef.export(rows)) + "\n]}\n")   # a row a line
        print("written:", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main("--quick" in sys.argv or os.environ.get("VEGETA_DEBUG") == "1"))
