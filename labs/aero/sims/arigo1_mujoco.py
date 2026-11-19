"""ARIGO-1 stability in MuJoCo — run by ``arigo1_mujoco.sh`` (notebook 39). Seconds per configuration on one core.

For each wing variant x propulsion: the model's linear modes, and in MuJoCo (open loop, controls at trim unless
said): hands-off 20 s, elevator doublet (short period, phugoid), rudder doublet (Dutch roll), aileron pulse (roll,
spiral), full-aileron roll, 3 m/s vertical gust, power-off stall (released at once, and after 1.5 s of full up
elevator), a closed-loop 60 degree turn. The post-stall pitching moment is a hand estimate, so every case runs at
both bounds (``post_stall_cm_factor`` 1.0 optimistic, 0.35 pessimistic) until the CFD's Cm curve replaces it.

Writes ``labs/aero/sims/output/arigo1_mujoco.yaml`` — THE FILE TO PASTE BACK.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "labs" / "aero" / "designs"))
import arigo1_resume as rs  # noqa: E402
import arigo1_mujoco as amj  # noqa: E402

OUT = ROOT / "labs" / "aero" / "sims" / "output"


def eta(done, total, t0):
    """'k/N, elapsed, about X left' from the mean time per finished item."""
    import time as _t
    el = _t.time() - t0
    left = el / done * (total - done) if done else float("nan")
    f = lambda s: "?" if s != s else (f"{s / 3600:.1f} h" if s >= 3600 else f"{s / 60:.0f} min" if s >= 90 else f"{s:.0f} s")
    return f"[{done}/{total}, {f(el)} elapsed, ~{f(left)} left]"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="*", default=["goshawk", "zero", "hybrid"], choices=("goshawk", "zero", "hybrid"))
    ap.add_argument("--propulsion", nargs="*", default=["pusher", "tractor", "twin", "tractor_lowcam"], choices=("pusher", "tractor", "twin", "tractor_lowcam"))
    ap.add_argument("--speed", type=float, default=18.0)
    ap.add_argument("--bounds", type=float, nargs="*", default=[1.0, 0.35], help="post-stall Cm factors")
    ap.add_argument("--name", default="arigo1", help="the result files' stem (arigo0, arigo1)")
    ap.add_argument("--fresh", action="store_true", help="ignore the saved results and start over")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    import mujoco
    path = OUT / f"{a.name}_mujoco.yaml"
    old = rs.load(path, a.fresh)
    if old and old.get("speed_m_s") != a.speed:
        print("  saved results are for another speed: starting over"); old = None
    doc = old or {"aircraft": {"arigo0": "ARIGO-0", "arigo1": "ARIGO-1"}.get(a.name, a.name), "mujoco": mujoco.__version__, "speed_m_s": a.speed,
                  "note": "aerodynamics from arigo1_model (lattice + hand estimates); MuJoCo integrates the rigid body", "runs": {}}
    keys = [f"{v}_{p}_cm{k:g}" for v in a.variants for p in a.propulsion for k in a.bounds]
    ok = lambda k: k in doc["runs"] and "error" not in doc["runs"][k]
    total = sum(not ok(k) for k in keys)
    print(f"  {len(keys) - total} runs already done (kept), {total} to run")
    t0, n = time.time(), 0
    for v in a.variants:
        for p in a.propulsion:
            for k in a.bounds:
                key = f"{v}_{p}_cm{k:g}"
                if ok(key):
                    continue
                t1 = time.time()
                try:
                    doc["runs"][key] = amj.stability_tests(v, p, a.speed, post_stall_cm_factor=k)
                except Exception as exc:
                    doc["runs"][key] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
                n += 1
                print(f"  {eta(n, total, t0)} {key}: {time.time() - t1:.1f} s  {doc['runs'][key].get('verdict', doc['runs'][key].get('error'))}")
                rs.save(path, doc)
    rs.save(path, doc)
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
