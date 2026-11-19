"""ARIGO-20's CFD sweep (fans off, ducts open) at 30 m/s; restartable. Writes ``labs/aero/sims/output/arigo20_cfd.yaml``
— PASTE BACK."""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "labs" / "aero" / "designs"))
sys.path.append(str(Path(__file__).resolve().parent))
import arigo1_resume as rs  # noqa: E402
import arigo20_cfd as c20  # noqa: E402

rs.GLOBS = ["arigo1*.py", "arigo20*.py"]
OUT = ROOT / "labs" / "aero" / "sims" / "output"
RUNS = ROOT / "labs" / "aero" / "_runs" / "arigo20_cfd"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--alphas", type=float, nargs="*", default=list(c20.ALPHAS))
    ap.add_argument("--quality", choices=tuple(c20.QUALITY), default="screening")
    ap.add_argument("--processors", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=3 * 3600.0)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--workdir", type=Path, default=RUNS)
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "arigo20_cfd.yaml"
    doc = rs.load(path, a.fresh) or {"aircraft": "ARIGO-20", "speed_m_s": c20.SPEED, "quality": a.quality, "rows": []}
    ref = c20.reference()
    doc["reference"] = {**ref, "cg_mm": list(ref["cg_mm"])}
    rows = doc["rows"] = [r for r in doc.get("rows", []) if r.get("status") == "solved"]
    done = {float(r["alpha_deg"]) for r in rows}
    todo = [al for al in a.alphas if float(al) not in done]
    print(f"ARIGO-20 CFD: {len(done)} cases already solved (kept), {len(todo)} to run")
    t0, ok = time.time(), 0
    for n, al in enumerate(todo, 1):
        err = ""
        try:
            res = c20.case(a.workdir, alpha_deg=al, quality=a.quality).ensure(run=True, processors=a.processors, timeout=a.timeout)
        except Exception as exc:
            res, err = None, f"{type(exc).__name__}: {exc}"
        r = c20.row(al, c20.SPEED, res)
        if res is None:
            r["messages"] = [err[:300]]
        rows.append(r)
        rows.sort(key=lambda x: x["alpha_deg"])
        doc["summary"] = c20.summary(rows)
        ok += r["status"] == "solved"
        print(f"  {rs.eta(n, len(todo), t0)} ARIGO-20 alpha {al:+5.1f}: {r['status']}" + (f"  Cl {r['Cl']:+.3f} Cd {r['Cd']:.4f} Cm {r['Cm_cg']:+.3f}" if r["status"] == "solved" else f" {err[:100]}"))
        rs.save(path, doc)
    rs.save(path, doc)
    print(f"-> {path}")
    return 0 if ok == len(todo) else 1


if __name__ == "__main__":
    raise SystemExit(main())
