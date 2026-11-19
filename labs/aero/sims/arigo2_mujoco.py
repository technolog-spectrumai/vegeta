"""ARIGO-2 stability in MuJoCo, run by ``arigo2_all.sh``: ARIGO-1's tests on the twin (separate motors) plus engine
out (open loop, rudder only, with the throttle protection; 16 and 24 m/s) and differential-thrust yaw. Each wing at
both post-stall bounds. Restartable.

Writes ``labs/aero/sims/output/arigo2_mujoco.yaml`` — THE FILE TO PASTE BACK.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "labs" / "aero" / "designs"))
sys.path.append(str(Path(__file__).resolve().parent))
import arigo1_resume as rs  # noqa: E402
import arigo2_mujoco as m2j  # noqa: E402

rs.GLOBS = ["arigo1*.py", "arigo2*.py"]
OUT = ROOT / "labs" / "aero" / "sims" / "output"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wings", nargs="*", default=["goshawk", "zero", "hybrid"], choices=("goshawk", "zero", "hybrid"))
    ap.add_argument("--speed", type=float, default=18.0)
    ap.add_argument("--bounds", type=float, nargs="*", default=[1.0, 0.35], help="post-stall Cm factors")
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    import mujoco
    path = OUT / "arigo2_mujoco.yaml"
    old = rs.load(path, a.fresh)
    if old and old.get("speed_m_s") != a.speed:
        old = None
    doc = old or {"aircraft": "ARIGO-2", "mujoco": mujoco.__version__, "speed_m_s": a.speed,
                  "note": "aerodynamics from arigo1_model (twin) + two separate motors (arigo2_model); hand estimates marked there", "runs": {}}
    keys = [(w, k, f"{w}_cm{k:g}") for w in a.wings for k in a.bounds]
    todo = [x for x in keys if x[2] not in doc["runs"] or "error" in doc["runs"][x[2]]]
    print(f"  {len(keys) - len(todo)} runs already done (kept), {len(todo)} to run")
    t0 = time.time()
    for n, (w, k, key) in enumerate(todo, 1):
        t1 = time.time()
        try:
            doc["runs"][key] = m2j.stability_tests(w, a.speed, post_stall_cm_factor=k)
        except Exception as exc:
            doc["runs"][key] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
        print(f"  {rs.eta(n, len(todo), t0)} {key}: {time.time() - t1:.1f} s  {doc['runs'][key].get('verdict', doc['runs'][key].get('error'))}")
        rs.save(path, doc)
    rs.save(path, doc)
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
