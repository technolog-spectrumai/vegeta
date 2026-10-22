#!/usr/bin/env python3
"""Persephone's benchmark: spring-only vs spring–damper body joints × baseline vs load-feedback control, on the
hearth (flat, soot-rough, small bumps), in MuJoCo through Chiron. Trial lists in experiments.py.

NOT YET VALIDATED — run the smoke scale first:

    python3 benchmark/persephone/full_benchmark.py --scale smoke        # ~10 runs: does Persephone walk at all?
    python3 benchmark/persephone/full_benchmark.py --scale pilot -j 8
    python3 benchmark/persephone/full_benchmark.py                      # full (long; resumable)
    python3 benchmark/persephone/full_benchmark.py --plots-only

Writes ``results.json``, ``<experiment>_runs.csv``, ``all_runs.csv``, ``configs.csv``, ``raw/`` (per-run CSV),
``plots/`` and ``report.md`` to ``--out`` (default ``benchmark/persephone/results/<scale>/``). Runs are cached
in ``<out>/cache``; an interrupted benchmark resumes.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import pandas as pd  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

import _common as cm  # noqa: E402
import experiments as pe  # noqa: E402

LIMITATIONS = [
    "Persephone's dynamics model and gait were written at the close of the build and had not been run: if the smoke "
    "scale shows falls or stalls on flat ground, tune kp/kd/armature in persephone_robot.py and the gait in "
    "experiments.controller() before reading anything into the comparison.",
    "No flue: the 150 mm pipe, its 90° knee and the vertical climb (notebook 17) need static pipe geometry in Chiron and "
    "a bracing controller, which do not exist yet. Only the hearth is benchmarked.",
    "The worm servos' self-locking is not modelled (a PD servo with the torque–speed line); their reflected gear "
    "inertia is an estimate (armature 5e-4 kg·m²).",
    "Simulation only (MuJoCo 3, soft contacts, 0.25 ms step not convergence-checked for Persephone).",
    "The contact-force feasibility LP is off by default (48 feet); --feasibility-every N turns it on.",
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", choices=tuple(pe.SCALES), default="full")
    ap.add_argument("--processes", "-j", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--experiments", default=",".join(pe.EXPERIMENTS))
    ap.add_argument("--plots-only", action="store_true")
    ap.add_argument("--no-raw", action="store_true")
    ap.add_argument("--raw-every", type=int, default=2)
    ap.add_argument("--feasibility-every", type=int, default=0)
    a = ap.parse_args(argv)
    exps = [e for e in a.experiments.split(",") if e]
    out = Path(a.out or HERE / "results" / a.scale)
    out.mkdir(parents=True, exist_ok=True)
    if not a.plots_only:
        n = sum(len(pe.trials(e, scale=a.scale)) for e in exps)
        print(f"Persephone benchmark, scale {a.scale}: {n} runs in {exps} on {a.processes} processes -> {out}",
              flush=True)
        meta = cm.metadata("persephone", a.scale, a.processes, exps)
        tables = {}
        for name in tqdm(exps, desc="experiments", unit="exp"):
            t0 = time.time()
            tables[name] = pe.run_experiment(name, out, scale=a.scale, processes=a.processes, raw_every=a.raw_every,
                                             write_raw=not a.no_raw, feasibility_every=a.feasibility_every)
            tqdm.write(f"{name}: {len(tables[name])} runs, {time.time() - t0:.0f} s")
        pd.concat(tables.values(), ignore_index=True).to_csv(out / "all_runs.csv", index=False)
        cm.save_results_json(out / "results.json", meta, pe.configs_table(), tables)
    import matplotlib
    matplotlib.use("Agg")
    import stability_plots as sp
    doc, tables = cm.load_results_json(out / "results.json")
    book = cm.PlotBook(out / "plots")
    sections = []
    keys = ["info.terrain", "info.level", "v_target", "info.k_yaw", "controller", "treatment"]
    for name, df in tables.items():
        sections.append((f"Outcomes — {name} (all runs)", cm.md_table(cm.outcome_table(df, keys))))
    if "hearth" in tables:
        df = tables["hearth"]
        for terrain in ("rough", "long_bumps"):
            if (df.get("terrain.kind") == terrain).any():
                book(f"hearth_success_{terrain}", sp.success_vs_roughness, df, terrain)
                book(f"hearth_motion_slip_{terrain}_all", sp.angular_motion_and_slip_vs_roughness, df, terrain, "all")
        book("hearth_cost_vs_speed", sp.cost_vs_speed, df)
        for metric in ("success", "distance_m", "payload_tilt_p95_deg", "slip_per_m", "cot_mech"):
            t = book(f"hearth_paired_{metric}", sp.paired_difference_forest, df, metric)
            if t is not None and len(t):
                sections.append((f"Paired spring_damper − spring: {metric} (all runs)", cm.md_table(t)))
    if "speeds" in tables:
        book("speeds_achieved_vs_commanded", sp.achieved_vs_commanded_speed, tables["speeds"])
    if "onset" in tables:
        book("onset_undulation_vs_speed", sp.undulation_vs_speed, tables["onset"])
    report = cm.write_report(out / "report.md", "Persephone on the hearth: spring-only vs spring–damper body joints",
                             doc["metadata"], sections, book, LIMITATIONS)
    print(f"report: {report}\nresults: {out / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
