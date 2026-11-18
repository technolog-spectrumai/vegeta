"""GAVILAN's stall sweeps in CFD — the launcher you run on a machine with OpenFOAM (notebook 38).

    source activate.sh
    python labs/aero/sims/gavilan_cfd.py --list                       # the cases, nothing run
    python labs/aero/sims/gavilan_cfd.py --design A --processors 8    # Gavilan-A, 0-90 degrees at 20 m/s (13 cases)
    python labs/aero/sims/gavilan_cfd.py --design F --processors 8    # the fixed wing, -4 to 24 degrees (11 cases)
    python labs/aero/sims/gavilan_cfd.py --design A --alphas 0 12 30 90 --quality fine

Each case is a steady RANS run (``designs/gavilan_cfd.py``): at the ``screening`` quality minutes per case on four
cores, at ``fine`` several times that. A solved case is read back, not run again, so the sweep can be stopped and
restarted. Writes, per design, into ``labs/aero/sims/output/``:

    gavilan_cfd_<design>.json     every case: Cl, Cd, Cm about the CG, forces, the pitching moment against what the
                                  motors can hold, convergence, mesh size — THE FILE TO SEND BACK
    gavilan_cfd_<design>.csv      the same as a table
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "labs" / "aero" / "designs"))
import gavilan_cfd as gc  # noqa: E402

OUT = ROOT / "labs" / "aero" / "sims" / "output"
RUNS = ROOT / "labs" / "aero" / "_runs" / "gavilan_cfd"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", choices=("A", "B", "F"), default="A", help="A: the tailsitter, B: the side panels, F: the fixed wing")
    ap.add_argument("--alphas", type=float, nargs="*", default=None, help="angles of attack [deg] (default: the design's sweep)")
    ap.add_argument("--speed", type=float, default=gc.SPEED, help="airspeed [m/s]")
    ap.add_argument("--quality", choices=tuple(gc.QUALITY), default="screening")
    ap.add_argument("--processors", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=3 * 3600.0, help="seconds allowed per case")
    ap.add_argument("--list", action="store_true", help="print the cases and stop")
    ap.add_argument("--workdir", type=Path, default=RUNS)
    a = ap.parse_args(argv)
    alphas = a.alphas if a.alphas else gc.ALPHAS[a.design]
    ref = gc.reference(a.design)
    print(f"Gavilan-{a.design}: {len(alphas)} cases at {a.speed:g} m/s, quality {a.quality}; S_ref {ref['S']:.4f} m2, c_ref {ref['c']:.3f} m, "
          f"mass {ref['mass']:.2f} kg, CG (x, z) {ref['cg_mm'][0]:.0f}, {ref['cg_mm'][1]:.0f} mm; pitch authority {gc.pitch_authority(a.design, a.speed):.2f} N m")
    if a.list:
        for al in alphas:
            print(f"  alpha {al:+5.1f} deg")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    for i, al in enumerate(alphas):
        t1 = time.time()
        try:
            res = gc.case(a.workdir / a.design, a.design, alpha_deg=al, V=a.speed, quality=a.quality).ensure(
                run=True, processors=a.processors, timeout=a.timeout)
        except Exception as exc:                                       # one failed case must not lose the others
            res = None
            print(f"  alpha {al:+5.1f}: {type(exc).__name__}: {exc}")
        r = gc.row(a.design, al, a.speed, res)
        r["quality"], r["seconds"] = a.quality, round(time.time() - t1, 1)
        rows.append(r)
        if r["status"] == "solved":
            print(f"  [{i + 1}/{len(alphas)}] alpha {al:+5.1f}: Cl {r['Cl']:+.3f}  Cd {r['Cd']:.3f}  Cm {r['Cm_cg']:+.3f}  "
                  f"moment {r['pitch_moment_Nm']:+.2f} N m  converged {r['converged']}  ({r['seconds']:.0f} s)")
        else:
            print(f"  [{i + 1}/{len(alphas)}] alpha {al:+5.1f}: {r['status']} {r.get('messages', '')}")
        doc = {"design": a.design, "speed_m_s": a.speed, "quality": a.quality, "reference": {**ref, "cg_mm": list(ref["cg_mm"])},
               "pitch_authority_Nm": gc.pitch_authority(a.design, a.speed), "rows": rows}
        (OUT / f"gavilan_cfd_{a.design.lower()}.json").write_text(json.dumps(doc, indent=1, default=str))   # after every case
    keys = sorted({k for r in rows for k in r if k != "messages"})
    with open(OUT / f"gavilan_cfd_{a.design.lower()}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    solved = sum(r["status"] == "solved" for r in rows)
    print(f"{solved} of {len(rows)} solved in {time.time() - t0:.0f} s -> {OUT / f'gavilan_cfd_{a.design.lower()}.json'}")
    return 0 if solved == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
