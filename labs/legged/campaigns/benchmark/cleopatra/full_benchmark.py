#!/usr/bin/env python3
"""Cleopatra's full benchmark: spring-only vs spring–damper body joints × baseline vs load-feedback control, in
MuJoCo through Chiron (docs/myropod_stability.md, Amendment D §12; the trial lists are in experiments.py).

    python3 benchmark/cleopatra/full_benchmark.py                       # full protocol (hours; resumable)
    python3 benchmark/cleopatra/full_benchmark.py --scale smoke         # ~40 runs, minutes: does everything work?
    python3 benchmark/cleopatra/full_benchmark.py --scale pilot -j 8    # a few seeds of every experiment
    python3 benchmark/cleopatra/full_benchmark.py --experiments main,onset
    python3 benchmark/cleopatra/full_benchmark.py --plots-only          # re-make plots/report from results.json

Writes to ``--out`` (default ``benchmark/cleopatra/results/<scale>/``):

- ``results.json`` — metadata (git commit, versions, machine, scale, times), configurations and every run's row;
- ``<experiment>_runs.csv`` (one row per run: configuration, seed, terrain parameters, outcome, every metric),
  ``all_runs.csv``, ``configs.csv``, ``raw/<experiment>/<run>.csv.gz`` (each run's time series, 50 Hz);
- ``plots/*.png`` (+ the table behind each plot as CSV) and ``report.md``.

Runs are cached in ``<out>/cache``: an interrupted benchmark resumes where it stopped (Ctrl-C is safe). Progress
bars: one over the experiments, one over each experiment's runs (tqdm).
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
import experiments as ce  # noqa: E402

LIMITATIONS = [
    "Simulation only (MuJoCo 3 through Chiron): soft contacts with MuJoCo's default solref/solimp, pyramidal "
    "friction cones, implicitfast at a 0.25 ms step (the §12.5 convergence result). No experiment on hardware.",
    "Leg servos follow §1's symmetric torque–speed bound: beyond the no-load speed a servo gives no torque, also "
    "when braking (a real DC servo brakes); knees exceed the no-load speed in a third of samples (Amendment E).",
    "The gait saturates the leg servos above about 0.146 m/s (Amendment E §13.3): results at 0.2–0.3 m/s include "
    "late touchdowns; actuator saturation is reported with the metrics.",
    "The load-feedback controller has a phase-rate floor (Amendment E §13.1) that is not in the original §9.1 rule.",
    "Controllers are terrain-blind and held fixed (§7.6 equal tuning not run): a treatment that needs a different "
    "gait to show its benefit is not given one.",
    "Spring-only does not mean lossless: contacts, friction, servo damping and the integrator dissipate energy "
    "(§12.2).",
]


def run(args) -> dict:
    out = Path(args.out)
    meta = cm.metadata("cleopatra", args.scale, args.processes, args.experiments)
    tables = {}
    bar = tqdm(args.experiments, desc="experiments", unit="exp")
    for name in bar:
        bar.set_postfix_str(f"{name}: {len(ce.trials(name, scale=args.scale))} runs")
        t0 = time.time()
        tables[name] = ce.run_experiment(name, out, scale=args.scale, processes=args.processes,
                                         raw_every=args.raw_every, write_raw=not args.no_raw,
                                         feasibility_every=args.feasibility_every, include_steps=args.steps)
        tqdm.write(f"{name}: {len(tables[name])} runs, {time.time() - t0:.0f} s, "
                   f"{int(tables[name]['success'].fillna(False).astype(bool).sum())} successes")
    all_runs = pd.concat(tables.values(), ignore_index=True)
    all_runs.to_csv(out / "all_runs.csv", index=False)
    cm.save_results_json(out / "results.json", meta, ce.configs_table(), tables)
    return {"meta": meta, "tables": tables}


def plots_and_report(out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import stability_plots as sp
    doc, tables = cm.load_results_json(out / "results.json")
    meta = doc["metadata"]
    book = cm.PlotBook(out / "plots")
    sections = []
    keys = ["info.terrain", "info.level", "v_target", "info.impulse_Ns", "info.k_yaw", "info.c_all",
            "info.roll_axis", "controller", "treatment"]
    for name, df in tables.items():
        sections.append((f"Outcomes — {name} (all runs)", cm.md_table(cm.outcome_table(df, keys))))
    if "main" in tables:
        df = tables["main"]
        for terrain in ("long_bumps", "alt_bumps", "cross_slope", "rough", "steps"):
            if (df.get("terrain.kind") == terrain).any():
                book(f"main_success_vs_roughness_{terrain}", sp.success_vs_roughness, df, terrain)
                book(f"main_motion_slip_{terrain}_all", sp.angular_motion_and_slip_vs_roughness, df, terrain, "all")
                book(f"main_motion_slip_{terrain}_success_only", sp.angular_motion_and_slip_vs_roughness, df,
                     terrain, "success")
        book("main_cost_vs_speed", sp.cost_vs_speed, df)
        verdicts = []
        for metric in ("success", "distance_m", "achieved_speed_m_s", "payload_tilt_p95_deg",
                       "worst_roll_rate_rms_rad_s", "slip_per_m", "belly_contact_fraction", "cot_mech"):
            t = book(f"main_paired_{metric}", sp.paired_difference_forest, df, metric)
            if t is not None and len(t):
                verdicts.append(t.assign(metric=metric))
        if verdicts:
            v = pd.concat(verdicts, ignore_index=True)
            vcol = next((c for c in ("verdict", "classification", "class") if c in v.columns), None)
            if vcol:
                summary = v.groupby(["metric", vcol]).size().unstack(fill_value=0).reset_index()
                sections.append(("Spring–damper vs spring-only: verdict counts over terrain × level × controller "
                                 "(§11.6)", cm.md_table(summary)))
            sections.append(("Paired differences spring_damper − spring (all runs)", cm.md_table(v)))
        for terrain in ("rough", "alt_bumps"):
            for level in (0.15, 0.25):
                if ((df.get("terrain.kind") == terrain) & (df.get("terrain.level") == level)).any():
                    book(f"main_factorial_{terrain}_{level}", sp.factorial_interaction, df, terrain, level)
        # one representative pair, re-simulated with full logs (two runs; seconds to a minute)
        try:
            from vegeta.chiron import experiments as cx
            tl = [t for t in ce.trials("main", scale="smoke") if t.terrain.get("kind") == "alt_bumps"
                  and t.info["controller_name"] == "fixed"]
            if len(tl) >= 2:
                a = cx.rerun_with_log(tl[0])
                b = cx.rerun_with_log(tl[1])
                book("traces_alt_bumps_pair", sp.synchronized_traces, a, b, ("spring", "spring_damper"))
        except Exception as exc:  # noqa: BLE001
            book.failed.append(("traces_alt_bumps_pair", f"{type(exc).__name__}: {exc}"))
    if "speed_maps" in tables:
        book("speed_maps_achieved_vs_commanded", sp.achieved_vs_commanded_speed, tables["speed_maps"])
        book("speed_maps_cost_vs_speed", sp.cost_vs_speed, tables["speed_maps"])
        for terrain in ("rough", "alt_bumps"):
            book(f"speed_maps_success_{terrain}", sp.success_vs_roughness, tables["speed_maps"], terrain)
    if "onset" in tables:
        book("onset_undulation_vs_speed", sp.undulation_vs_speed, tables["onset"])
    if "disturbance" in tables:
        book("disturbance_recovery", cm.recovery_plot, tables["disturbance"])
    if "damping" in tables:
        book("damping_sensitivity", cm.sensitivity_plot, tables["damping"], "info.c_all", "c on every axis [N·m·s/rad]")
    if "roll" in tables:
        book("roll_sensitivity", cm.sensitivity_plot, tables["roll"], "info.roll_axis", "roll hinge")
    return cm.write_report(out / "report.md", "Cleopatra: spring-only vs spring–damper body joints", meta, sections,
                           book, LIMITATIONS)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", choices=tuple(ce.SCALES), default="full")
    ap.add_argument("--processes", "-j", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--out", default=None, help="output directory (default benchmark/cleopatra/results/<scale>)")
    ap.add_argument("--experiments", default=",".join(ce.EXPERIMENTS), help="comma-separated subset")
    ap.add_argument("--plots-only", action="store_true", help="re-make plots and report from results.json")
    ap.add_argument("--no-raw", action="store_true", help="do not write each run's raw CSV")
    ap.add_argument("--raw-every", type=int, default=2, help="raw CSV decimation of the 100 Hz log (2 = 50 Hz)")
    ap.add_argument("--feasibility-every", type=int, default=10,
                    help="contact-force LP every n-th log sample (0 = skip; the LP is the slowest metric)")
    ap.add_argument("--steps", action="store_true", help="add the steps terrain to 'main' (protocol extra)")
    args = ap.parse_args(argv)
    args.experiments = [e for e in args.experiments.split(",") if e]
    unknown = sorted(set(args.experiments) - set(ce.EXPERIMENTS))
    if unknown:
        ap.error(f"unknown experiments {unknown}; known: {ce.EXPERIMENTS}")
    args.out = args.out or str(HERE / "results" / args.scale)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    if not args.plots_only:
        n = sum(len(ce.trials(e, scale=args.scale)) for e in args.experiments)
        print(f"Cleopatra benchmark, scale {args.scale}: {n} runs in {args.experiments} on {args.processes} "
              f"processes -> {args.out}", flush=True)
        run(args)
    report = plots_and_report(Path(args.out))
    print(f"report: {report}\nresults: {Path(args.out) / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
