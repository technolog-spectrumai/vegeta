"""The ARIGO family's performance cards (``designs/arigo_perf.py``): speeds, turns, climb, roll / pitch / yaw, chase.
Seconds per configuration. Writes ``labs/aero/sims/output/arigo_perf.yaml``."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "labs" / "aero" / "designs"))
sys.path.append(str(Path(__file__).resolve().parent))
import arigo1_resume as rs  # noqa: E402
import arigo_perf as P  # noqa: E402

rs.GLOBS = ["arigo*.py"]
OUT = ROOT / "labs" / "aero" / "sims" / "output"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("aircraft", nargs="*", help="ARIGO-0 ARIGO-1 ARIGO-2 ARIGO-20 (default: all)")
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args(argv)
    want = [x.upper() for x in a.aircraft] or ["ARIGO-0", "ARIGO-1", "ARIGO-2", "ARIGO-20"]
    path = OUT / "arigo_perf.yaml"
    OUT.mkdir(parents=True, exist_ok=True)
    doc = rs.load(path, a.fresh) or {"what": "performance cards", "note": P.__doc__.split("\n\n")[1].replace("\n", " "), "cards": {}}
    cfgs = [c for c in P.configurations() if c[1].split()[0] in want]
    todo = [c for c in cfgs if c[0] not in doc["cards"] or "error" in doc["cards"][c[0]]]
    print(f"  {len(cfgs) - len(todo)} cards already done (kept), {len(todo)} to run")
    t0 = time.time()
    for n, (key, name, Cf, sf, man, V) in enumerate(todo, 1):
        try:
            doc["cards"][key] = json.loads(json.dumps(P.card(name, Cf(), sf, man, V), default=float))
        except Exception as exc:
            doc["cards"][key] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
        print(f"  {rs.eta(n, len(todo), t0)} {name}: {doc['cards'][key].get('summary', doc['cards'][key].get('error'))}")
        rs.save(path, doc)
    rs.save(path, doc)
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
