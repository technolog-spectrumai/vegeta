"""ARIGO-1's stall sweeps in CFD — run by ``arigo1.sh`` on a machine with OpenFOAM (notebook 39).

Writes ``labs/aero/sims/output/arigo1_cfd.yaml`` after every case (stop and restart freely; a solved case is read
back, not run again): per configuration the reference, the model's prediction beside each solved case and a stall
summary. THE FILE TO PASTE BACK.
"""
import argparse
import sys
import time
from pathlib import Path

import json

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "labs" / "aero" / "designs"))
import arigo1_resume as rs  # noqa: E402
import arigo1_cfd as ac  # noqa: E402

OUT = ROOT / "labs" / "aero" / "sims" / "output"
RUNS = ROOT / "labs" / "aero" / "_runs" / "arigo1_cfd"


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
    ap.add_argument("--propulsion", nargs="*", default=["pusher"], choices=("pusher", "tractor", "twin"))
    ap.add_argument("--alphas", type=float, nargs="*", default=list(ac.ALPHAS))
    ap.add_argument("--speed", type=float, default=ac.SPEED)
    ap.add_argument("--quality", choices=tuple(ac.QUALITY), default="screening")
    ap.add_argument("--processors", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=3 * 3600.0)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--workdir", type=Path, default=RUNS)
    ap.add_argument("--fresh", action="store_true", help="ignore the saved results and start over")
    a = ap.parse_args(argv)
    configs = [(v, p) for v in a.variants for p in a.propulsion]
    print(f"ARIGO-1 CFD: {len(configs)} configurations x {len(a.alphas)} angles at {a.speed:g} m/s, quality {a.quality}")
    if a.list:
        for v, p in configs:
            r = ac.reference(v, p)
            print(f"  {v:8s} {p:8s} S {r['S']:.4f} m2  c {r['c']:.3f} m  CG x {r['cg_mm'][0]:.0f} mm  alphas {a.alphas}")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "arigo1_cfd.yaml"
    old = rs.load(path, a.fresh)
    if old and (old.get("speed_m_s"), old.get("quality")) != (a.speed, a.quality):
        print("  saved results are for another speed/quality: starting over"); old = None
    doc = old or {"aircraft": "ARIGO-1", "speed_m_s": a.speed, "quality": a.quality, "configurations": {}}
    doc["processors"] = a.processors
    done = {(k, float(r["alpha_deg"])) for k, c in doc["configurations"].items() for r in c.get("rows", []) if r.get("status") == "solved"}
    todo = sum((f"{v}_{p}", float(al)) not in done for v, p in configs for al in a.alphas)
    print(f"  {len(done)} cases already solved (kept), {todo} to run")
    t0, n_ok, n = time.time(), 0, 0
    for v, p in configs:
        ref = ac.reference(v, p)
        key = f"{v}_{p}"
        cfg = doc["configurations"].setdefault(key, {})
        cfg["reference"] = {**ref, "cg_mm": list(ref["cg_mm"])}
        rows = [r for r in cfg.get("rows", []) if r.get("status") == "solved"]      # failed / not-run cases are retried
        cfg["rows"] = rows
        for al in a.alphas:
            if (key, float(al)) in done:
                continue
            t1, err = time.time(), ""
            try:
                res = ac.case(a.workdir / key, v, p, alpha_deg=al, V=a.speed, quality=a.quality).ensure(run=True, processors=a.processors, timeout=a.timeout)
            except Exception as exc:
                res, err = None, f"{type(exc).__name__}: {exc}"
                print(f"  {key} alpha {al:+.0f}: {type(exc).__name__}: {exc}")
            r = ac.row(v, p, al, a.speed, res)
            if res is None:
                r["messages"] = [err[:300]]
            r["seconds"] = round(time.time() - t1, 1)
            rows.append(r)
            rows.sort(key=lambda x: x["alpha_deg"])
            n += 1
            n_ok += r["status"] == "solved"
            print(f"  {eta(n, todo, t0)} {key} alpha {al:+5.1f}: {r['status']}" + (f"  Cl {r['Cl']:+.3f} Cd {r['Cd']:.4f} Cm {r['Cm_cg']:+.3f}" if r["status"] == "solved" else ""))
            cfg["summary"] = ac.summary(rows)
            doc["elapsed_s"] = doc.get("elapsed_s", 0) + round(time.time() - t1)
            rs.save(path, doc)
    rs.save(path, doc)
    print(f"{n_ok} of {n} run now solved, {len(done) + n_ok} solved in all -> {path}")
    return 0 if n_ok == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
