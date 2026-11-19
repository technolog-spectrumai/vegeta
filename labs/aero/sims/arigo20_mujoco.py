"""ARIGO-20 stability in MuJoCo at 30 m/s: ARIGO-1's tests plus one fan out and the power pitch (the high thrust
line), both post-stall bounds. Restartable. Writes ``labs/aero/sims/output/arigo20_mujoco.yaml`` — PASTE BACK."""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "labs" / "aero" / "designs"))
sys.path.append(str(Path(__file__).resolve().parent))
import arigo1_resume as rs  # noqa: E402
import arigo20_mujoco as m20j  # noqa: E402

rs.GLOBS = ["arigo1*.py", "arigo20*.py"]
OUT = ROOT / "labs" / "aero" / "sims" / "output"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bounds", type=float, nargs="*", default=[1.0, 0.35])
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    import mujoco
    path = OUT / "arigo20_mujoco.yaml"
    doc = rs.load(path, a.fresh) or {"aircraft": "ARIGO-20", "mujoco": mujoco.__version__, "speed_m_s": m20j.V_TEST,
                                     "note": "arigo20_model: lattice + hand estimates (no CFD yet)", "runs": {}}
    todo = [k for k in a.bounds if f"cm{k:g}" not in doc["runs"] or "error" in doc["runs"][f"cm{k:g}"]]
    print(f"  {len(a.bounds) - len(todo)} runs already done (kept), {len(todo)} to run")
    t0 = time.time()
    for n, k in enumerate(todo, 1):
        try:
            doc["runs"][f"cm{k:g}"] = m20j.stability_tests(post_stall_cm_factor=k)
        except Exception as exc:
            doc["runs"][f"cm{k:g}"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
        print(f"  {rs.eta(n, len(todo), t0)} ARIGO-20 cm{k:g}: {doc['runs'][f'cm{k:g}'].get('verdict', doc['runs'][f'cm{k:g}'].get('error'))}")
        rs.save(path, doc)
    rs.save(path, doc)
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
