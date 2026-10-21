#!/usr/bin/env python3
"""Validate the tuned MuJoCo autopilot on held-out scenarios (validation in the machine-learning sense): the tuner fit
the settings in races/autopilot.json on the nominal lap only; this flies them on conditions it never saw — the
settings ±2 %, the start moved, the mass, the thrust, a wind, all combined (races/validate.py) — and reports the
generalisation gap. Real MuJoCo, no mocks, no solver.

    python test_autopilot.py                     # all four courses, 8 held-out laps per family (6 families + the nominal lap)
    python test_autopilot.py -j 8 -n 20          # 8 worker processes, 20 laps per family
    python test_autopilot.py --courses hangar --families nominal "wind ≤ 2 m/s"
    python test_autopilot.py --settings races/runs/tune/autopilot_hangar_24s_knife_edge.json   # validate other settings

Report: races/runs/validate/report.json (succinct, rewritten after every course) and <course>.csv (every lap).
Exit 0 when every course passes, 1 when one is weak or fails, 2 when MuJoCo or the settings are missing. The same
seed and options give the same laps whatever -j is."""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # the Piccolo repository
OUT = ROOT / "races" / "runs" / "validate"


def _write_atomic(path: Path, text: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def dumps(report: dict) -> str:
    """Indented JSON with every family on one line (the report stays a page long)."""
    lines, rep = {}, json.loads(json.dumps(report))
    for c in rep["courses"].values():
        for fam, f in (c.get("families") or {}).items():
            key = f"@@{len(lines)}@@"
            lines[key] = json.dumps(f, ensure_ascii=False)
            c["families"][fam] = key
    text = json.dumps(rep, indent=1, ensure_ascii=False)
    for key, line in lines.items():
        text = text.replace(f'"{key}"', line)
    return text + "\n"


def main(argv=None) -> int:
    from races import courses, validate
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--courses", nargs="*", default=list(courses.COURSES))
    ap.add_argument("-n", type=int, default=8, help="held-out laps per family (default 8)")
    ap.add_argument("-j", "--jobs", type=int, default=1, help=f"worker processes (default 1; this machine has {os.cpu_count()} CPUs)")
    ap.add_argument("--seed", type=int, default=validate.SEED, help=f"the held-out set's seed (default {validate.SEED}; the tuner never flies it)")
    ap.add_argument("--families", nargs="*", default=list(validate.FAMILIES), choices=validate.FAMILIES)
    ap.add_argument("--settings", type=Path, default=None, help="an autopilot.json-format file to validate instead of races/autopilot.json")
    ap.add_argument("--out", type=Path, default=OUT / "report.json")
    a = ap.parse_args(argv)

    stored = json.loads((a.settings or ROOT / "races" / "autopilot.json").read_text())
    sc = validate.scenarios(a.n, a.seed, a.families)
    print(f"test_autopilot: {len(sc)} laps per course ({a.n} per family; families: {', '.join(a.families)}), seed {a.seed}, "
          f"{a.jobs} worker(s) of {os.cpu_count()} CPUs", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    report = {"date": time.strftime("%Y-%m-%d %H:%M"), "settings file": str(a.settings or "races/autopilot.json"), "seed": a.seed,
              "laps per family": a.n, "grades": "per family on the clean rate (finished, 0 collisions): pass ≥ 0.8, weak ≥ 0.5, else fail",
              "verdict": None, "courses": {}}
    t_all = time.time()
    for slug in a.courses:
        entry = stored.get(slug) or {}
        settings = entry.get("settings")
        if not settings:
            print(f"== {slug}: no tuned settings in {a.settings or 'races/autopilot.json'} — skipped", flush=True)
            report["courses"][slug] = {"verdict": "no settings"}
            continue
        print(f"== {slug}: {settings}", flush=True)
        t0 = time.time()
        rows = validate.fly(slug, settings, sc, jobs=a.jobs)
        s = validate.summarise(rows)
        lap = entry.get("lap time [s]")
        s = {"verdict": s["verdict"], "nominal lap [s]": s["nominal lap [s]"], "stored lap [s]": round(lap, 2) if lap else None,
             "reproduced": (s["nominal lap [s]"] is not None and lap is not None and abs(s["nominal lap [s]"] - lap) < 0.01) if "nominal" in a.families else None,
             **{k: v for k, v in s.items() if k not in ("verdict", "nominal lap [s]")}, "wall [min]": round((time.time() - t0) / 60, 2)}
        report["courses"][slug] = s
        for fam, f in s["families"].items():
            print(f"   {fam:14s} {f['grade']:4s}  clean {f['clean rate']:.2f}  finish {f['finish rate']:.2f}  median {f['median lap [s]']}  gap {f['gap [s]']}", flush=True)
        print(f"   → {slug}: {s['verdict'].upper()} (held-out clean rate {s['held-out clean rate']}, weakest: {s['weakest family']}; "
              f"nominal {s['nominal lap [s]']} s vs stored {s['stored lap [s]']} s; {s['wall [min]']} min)", flush=True)
        with (OUT / f"{slug}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            for r in rows:
                w.writerow({**r, "mult": json.dumps(r["mult"]), "scenario": json.dumps(r["scenario"])})
        verdicts = [c["verdict"] for c in report["courses"].values()]
        report["verdict"] = "pass" if all(v == "pass" for v in verdicts) else ("fail" if "fail" in verdicts else "weak")
        report["wall [min]"] = round((time.time() - t_all) / 60, 2)
        _write_atomic(a.out, dumps(report))
    print(f"test_autopilot: {str(report['verdict']).upper()} in {(time.time() - t_all) / 60:.1f} min; report {a.out}", flush=True)
    return 0 if report["verdict"] == "pass" else 1


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "races" / "scripts"))      # tune_autopilot, its neighbour
    import tune_autopilot
    tune_autopilot.say = lambda msg: print(f"test_autopilot: {msg}", flush=True)
    if tune_autopilot.bootstrap(OUT, "validate") is None:
        sys.exit(2)
    sys.exit(main())
