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
import arigo1_mujoco as amj  # noqa: E402

OUT = ROOT / "labs" / "aero" / "sims" / "output"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="*", default=["goshawk", "zero", "hybrid"], choices=("goshawk", "zero", "hybrid"))
    ap.add_argument("--propulsion", nargs="*", default=["pusher", "tractor", "twin"], choices=("pusher", "tractor", "twin"))
    ap.add_argument("--speed", type=float, default=18.0)
    ap.add_argument("--bounds", type=float, nargs="*", default=[1.0, 0.35], help="post-stall Cm factors")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    import mujoco
    doc = {"aircraft": "ARIGO-1", "mujoco": mujoco.__version__, "speed_m_s": a.speed,
           "note": "aerodynamics from arigo1_model (lattice + hand estimates); MuJoCo integrates the rigid body", "runs": {}}
    t0 = time.time()
    for v in a.variants:
        for p in a.propulsion:
            for k in a.bounds:
                key = f"{v}_{p}_cm{k:g}"
                t1 = time.time()
                try:
                    doc["runs"][key] = amj.stability_tests(v, p, a.speed, post_stall_cm_factor=k)
                except Exception as exc:
                    doc["runs"][key] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
                print(f"  {key}: {time.time() - t1:.1f} s  {doc['runs'][key].get('verdict', doc['runs'][key].get('error'))}")
                doc["elapsed_s"] = round(time.time() - t0, 1)
                (OUT / "arigo1_mujoco.yaml").write_text(yaml.safe_dump(json.loads(json.dumps(doc, default=float)), sort_keys=False, width=140))
    print(f"-> {OUT / 'arigo1_mujoco.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
