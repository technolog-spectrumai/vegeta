# Myropod benchmarks

Physics benchmarks of the Sikarian Myropods in MuJoCo through Chiron (`vegeta.chiron`). The study is
pre-registered in [docs/myropod_stability.md](../docs/myropod_stability.md) (Amendments A–F). Nothing here has been
run at scale: the full benchmarks are for you to run.

| folder | robot | what is compared | status |
|---|---|---|---|
| `cleopatra/` | Cleopatra, 3 segments, 6.17 kg | spring-only vs spring–damper body joints × baseline vs load-feedback control; flat, longitudinal bumps, alternating bumps, cross-slope, rough; speed sweep, pushes, undulation onset, damping and roll sensitivity | model and physics checks tested (78 + 153 tests); one 20-run smoke of `main` ran end to end |
| `persephone/` | Persephone, 12 segments, 10.96 kg | the same treatments and controllers on the hearth (flat, soot-rough, small bumps), speed sweep, undulation onset | written at the close, **not run** — start with the smoke scale; no flue scenario yet (Chiron has no pipe geometry) |

## Running

```bash
./user_tests.sh                                   # env, unit tests, physics checks, both smoke benchmarks
./user_tests.sh cleopatra-pilot -j 8              # a few seeds of every Cleopatra experiment
python3 benchmark/cleopatra/full_benchmark.py     # the full Cleopatra study (6440 runs; hours; resumable)
python3 benchmark/persephone/full_benchmark.py --scale smoke
python3 benchmark/cleopatra/full_benchmark.py --experiments main,onset -j 16
python3 benchmark/cleopatra/full_benchmark.py --plots-only        # redo plots and report from results.json
```

Each run is cached in `results/<scale>/cache/`: Ctrl-C and re-run to resume. Progress bars (tqdm) over the
experiments and over each experiment's runs.

## Outputs (`<robot>/results/<scale>/`)

- `results.json` — metadata (git commit, dirty flag, versions, machine, scale, start and end), configurations, every
  run's row per experiment;
- `<experiment>_runs.csv`, `all_runs.csv` — one row per run: configuration (treatment, per-axis k/c/q0/limits,
  controller and σ, terrain kind, level, obstacle height and spacing, slope or RMS, seed, speed, push), outcome
  (success / fall / stall / off_course / timeout, distance) and every metric of `vegeta.chiron.metrics`;
- `configs.csv` — treatments, controllers, physics settings, failure rules;
- `raw/<experiment>/<run>.csv.gz` — each run's time series at 50 Hz: segment attitude and angular velocity, COM,
  per-foot normal and tangential force, loaded flag and position, per-joint angle, speed, torque and saturation
  flags, body-joint deflection (column units in `vegeta/chiron/export.py`);
- `plots/*.png` (+ the table behind each plot as CSV) and `report.md` — outcome tables per cell (failures by reason),
  the paired spring_damper − spring differences with the helps / hurts / no-clear-difference verdicts, plots that
  failed (with their errors) and the limitations.

## Rules the analysis follows

Runs, not frames, are the samples. Treatments differ only in the joint damping c, and every cell of an experiment
shares seeds, terrain and initial gait phase. Failed runs are always counted; any metric restricted to successful
runs says so. Damping is not assumed to win: the verdict per terrain, difficulty and controller is *helps*,
*hurts* or *no clear difference* by the rule in §11.6 of the protocol.

## Code

`_common.py` (metadata, results JSON, plot book, recovery and sensitivity plots, the report) is shared;
`notebooks/designs/stability_plots.py` makes the comparative plots; `cleopatra/experiments.py` and
`persephone/experiments.py` hold the trial lists (`chiron.experiments.Trial`), the per-run metrics with the raw
export, and the configuration tables; `persephone/persephone_robot.py` builds Persephone from Cleopatra's blocks
(`notebooks/designs/myropod_robot.py`).
