"""ARIGO-2's CFD — power off (airframe and nacelles) and power on (both propellers as rotor disks), run by
``arigo2_all.sh`` on a machine with OpenFOAM. Restartable: solved cases are kept in the YAML and skipped.

Writes ``labs/aero/sims/output/arigo2_cfd.yaml`` — THE FILE TO PASTE BACK.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "labs" / "aero" / "designs"))
sys.path.append(str(Path(__file__).resolve().parent))
import arigo1_resume as rs  # noqa: E402
import arigo2_cfd as c2  # noqa: E402

rs.GLOBS = ["arigo1*.py", "arigo2*.py"]
OUT = ROOT / "labs" / "aero" / "sims" / "output"
RUNS = ROOT / "labs" / "aero" / "_runs" / "arigo2_cfd"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wings", nargs="*", default=["goshawk", "zero", "hybrid"], choices=("goshawk", "zero", "hybrid"))
    ap.add_argument("--alphas", type=float, nargs="*", default=list(c2.ALPHAS), help="power-off angles")
    ap.add_argument("--blown-alphas", type=float, nargs="*", default=list(c2.BLOWN_ALPHAS), help="power-on angles (none: skip)")
    ap.add_argument("--rpm", type=float, default=c2.RPM)
    ap.add_argument("--speed", type=float, default=c2.SPEED)
    ap.add_argument("--quality", choices=tuple(c2.QUALITY), default="screening")
    ap.add_argument("--processors", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=3 * 3600.0)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="ignore the saved results and start over")
    ap.add_argument("--workdir", type=Path, default=RUNS)
    a = ap.parse_args(argv)
    jobs = [(w, False, al) for w in a.wings for al in a.alphas] + [(w, True, al) for w in a.wings for al in a.blown_alphas]
    print(f"ARIGO-2 CFD: {len(jobs)} cases at {a.speed:g} m/s ({a.rpm:.0f} rpm when blown), quality {a.quality}")
    if a.list:
        for w, b, al in jobs:
            print(f"  {w:8s} {'blown' if b else 'aero ':5s} alpha {al:+5.1f}")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "arigo2_cfd.yaml"
    old = rs.load(path, a.fresh)
    if old and (old.get("speed_m_s"), old.get("quality"), old.get("rpm")) != (a.speed, a.quality, a.rpm):
        print("  saved results are for another speed/quality/rpm: starting over"); old = None
    doc = old or {"aircraft": "ARIGO-2", "speed_m_s": a.speed, "quality": a.quality, "rpm": a.rpm, "configurations": {}}
    doc["processors"] = a.processors
    key = lambda w, b: f"{w}_{'blown' if b else 'aero'}"
    done = {(k, float(r["alpha_deg"])) for k, c in doc["configurations"].items() for r in c.get("rows", []) if r.get("status") == "solved"}
    todo = [j for j in jobs if (key(j[0], j[1]), float(j[2])) not in done]
    print(f"  {len(jobs) - len(todo)} cases already solved (kept), {len(todo)} to run")
    t0, n, n_ok = time.time(), 0, 0
    for w, b, al in todo:
        k = key(w, b)
        cfg = doc["configurations"].setdefault(k, {})
        ref = c2.reference(w)
        cfg["reference"] = {**ref, "cg_mm": list(ref["cg_mm"])}
        rows = cfg["rows"] = [r for r in cfg.get("rows", []) if r.get("status") == "solved"]
        t1, err = time.time(), ""
        try:
            res = c2.case(a.workdir / k, w, alpha_deg=al, blown=b, V=a.speed, rpm=a.rpm, quality=a.quality).ensure(
                run=True, processors=a.processors, timeout=a.timeout)
        except Exception as exc:
            res, err = None, f"{type(exc).__name__}: {exc}"
        r = c2.row(w, al, a.speed, res, b)
        if res is None:
            r["messages"] = [err[:300]]
        r["seconds"] = round(time.time() - t1, 1)
        rows.append(r)
        rows.sort(key=lambda x: x["alpha_deg"])
        cfg["summary"] = c2.summary(rows)
        n += 1
        n_ok += r["status"] == "solved"
        print(f"  {rs.eta(n, len(todo), t0)} {k} alpha {al:+5.1f}: {r['status']}" +
              (f"  Cl {r['Cl']:+.3f} Cd {r['Cd']:.4f} Cm {r['Cm_cg']:+.3f}" if r["status"] == "solved" else f" {err[:120]}"))
        rs.save(path, doc)
    rs.save(path, doc)
    print(f"{n_ok} of {n} run now solved -> {path}")
    return 0 if n_ok == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
