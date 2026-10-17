# Workflows — explicit, file-based long-running analyses

(The short how-to for engineers is `manual_workflows.md` at the repository root; this file is the contract.)

A **workflow** is a thin script under `workflows/<project>/` that prepares explicit inputs, runs the selected solver
cases, gathers the artifacts, parses the results and saves them into one **run directory**. It is launched by an
engineer from a shell, runs unattended for as long as it needs, and leaves a record a notebook can read, check and
cite. The engineering logic stays in the design modules (`notebooks/designs/*.py`) and the Vegeta tools; the
scripts only drive them. There is no scheduler, service, database, DAG engine or optimiser here, and none is planned:
a workflow is Python, its configuration is Python, its results are files.

## The rule

> **If an analysis is expected to take more than about 15–30 minutes, prefer an explicit workflow over notebook
> execution.** The estimate is for the whole sweep or campaign, not for one case.

What follows from it, for every notebook in the repository:

- A notebook **may require** workflow outputs produced in advance, and must then say exactly which run it reads,
  check that the run is finished and compatible with the design as the notebook builds it, and name the command
  that produces it when it is missing — it never substitutes an invented result.
- A notebook **never launches a solver silently**: not behind an import, not inside a plotting call, not in a
  "load or compute" helper. Expensive work that a notebook does itself (a Boreas table, a NeuralFoil polar — tens
  of seconds) is announced in the cell that does it.
- A result that is missing, stale, incompatible, running or failed is **NOT RUN** in the notebook: the analytical
  estimate stays visible beside it, labelled as such; optional validation may stay NOT RUN, required inputs stop
  the notebook with an actionable exception.
- Compatible results are **reused**, never recomputed: a run carries the fingerprint of the design it was made for.

The first workflows are Kestrel's (`workflows/kestrel/`); `runkit.py` is the shared run-directory bookkeeping the
next long-running analysis (a thermal case, another aircraft's CFD) reuses as it is.

## Layout

```
workflows/
  README.md                    this file
  cli.py                       the entry point: list / info / run / runs / status / tail / stop / resume / notebook, or interactive
  runkit.py                    the run-directory contract: manifest, status, run.log, heartbeat, cancel, resume, summary
  kestrel/
    airframe_cfd.py            the whole aircraft with the propellers as rotor disks (Aeromant / OpenFOAM), one campaign per run
    structure_fea.py           the carbon frame and the printed joint (Talos: Gmsh + CalculiX)
    import_legacy.py           the first CFD campaign's results file as run directories
    _kestrel.py                what the Kestrel scripts share (paths, the case loop under a budget)
  tests/                       the contract on a fake solver; the campaigns; prepare-only smoke cases
  runs/<workflow>/<run id>/    every run (gitignored): see the contract below
```

The engineering content of the Kestrel workflows — the variants and their fingerprints, the campaign catalogue, the
case builders, the summary and the loaders — is `notebooks/designs/kestrel_campaigns.py`, shared with the notebook.
Resolved-propeller CFD is **not** a workflow: it stays in notebook `38_prop_lab`, whose exported calibration
(`designs/data/prop_lab.json`) the airframe cases and the drive tables read when it passed its gates.

## Running (Ubuntu)

All commands from the repository root, with the virtual environment `install_local.sh` made (`.venv`). The quickest
way is `workflows/cli.py` (no arguments: interactive; `list`, `info`, `run --detach`, `runs`, `status`, `tail`, `stop`,
`resume`, `notebook`); the scripts below are what it calls.

```bash
# what a campaign would do, without writing anything
.venv/bin/python workflows/kestrel/airframe_cfd.py --variant 4Q-2 --campaign stall --list
.venv/bin/python workflows/kestrel/structure_fea.py --variant 4Q-2 --campaign all --list

# prepare only: the cases written (STL, dictionaries / STEP, mesh), no solver — the run says 'prepared'
.venv/bin/python workflows/kestrel/airframe_cfd.py --variant 4Q-2 --campaign stall --prepare-only

# foreground
.venv/bin/python workflows/kestrel/airframe_cfd.py --variant 4Q-2 --campaign stall --quality fine --processors 4

# detached with nohup (survives the terminal and Jupyter closing)
nohup .venv/bin/python workflows/kestrel/airframe_cfd.py --variant 4Q-2 --campaign sweep --processors 4 \
      > workflows/runs/airframe_cfd_sweep.out 2>&1 &
echo $!                                   # the PID (also in <run>/status.json)

# detached in tmux (attach later with: tmux attach -t kestrel)
tmux new -d -s kestrel '.venv/bin/python workflows/kestrel/structure_fea.py --variant 4Q-2 --campaign all --threads 4'

# watching a run
tail -F workflows/runs/airframe_cfd/<run id>/run.log
cat workflows/runs/airframe_cfd/<run id>/status.json
tail -n 3 workflows/runs/airframe_cfd/<run id>/cases/<case>/log.simpleFoam        # the native solver log

# stopping, resuming
kill -INT <pid>        # or Ctrl-C in the foreground: the running solver is stopped, the run ends 'cancelled', finished cases are kept
.venv/bin/python workflows/kestrel/airframe_cfd.py --resume workflows/runs/airframe_cfd/<run id>
```

Closing Jupyter never stops a workflow: the scripts are separate processes that write to disk; the notebook only
reads the run directory, whenever it is next executed.

Common options (every script): `--out DIR` (default `workflows/runs/<workflow>/<timestamp>_<campaign>`),
`--resume RUN_DIR`, `--rerun-failed`, `--prepare-only`, `--list`, `--cases TEXT[,TEXT]` (a name filter: a run does
the cases it lists and never more), `--max-cases N`, `--max-minutes M` (no new case after that; the run ends
`partial`, resumable), `--case-timeout S`, `--heartbeat S` (default 30), `--quiet`. CFD: `--quality screening|fine`,
`--processors N` (MPI ranks per case; meshing is serial). FEA: `--threads N`, `--n-modes`, `--element-size`.

Orders of magnitude (one 4-core workstation): a screening CFD case 10–20 min, a fine one about an hour; the `sweep`
campaign at screening an afternoon, `all` at fine days. An FEA frame case a few minutes; the whole `all` campaign
under an hour. Measure your own: `run.log` records every case's duration.

## Reading a run: the logs

Engineers must be able to tell a healthy long run from a stalled or failed one.

- **`run.log`** is written continuously (line-buffered, flushed): the run's creation and command, the configuration
  hash, the tool versions found, every case's start and finish with its directory, status and duration, the solver
  stages as the tools report them, and a **heartbeat** every `--heartbeat` seconds while a solver runs:
  `heartbeat: <case> alive (pid P, children [...]), 412 s into the case; cases/<dir>/log.simpleFoam 812 kB, advancing — Time = 213`.
  "alive" is the process; "advancing" is the solver's log having grown since the last beat; "no new output for N s"
  is reported, never acted on — a snappyHexMesh layer step or a CalculiX factorisation can be quiet for minutes. No
  percentage is invented.
- **`status.json`** (rewritten atomically): `state` ∈ `prepared | running | completed | partial | failed | cancelled`,
  the case counts (planned, done, reused, failed), the case in progress, the last heartbeat, `started_at`,
  `elapsed_s`, the PID. Numerical convergence is **not** in the state: a case whose solver exited cleanly without
  reaching its residual target is `process: ok`, `converged: false` in its record.
- The **native solver logs** stay where the tools write them (`cases/<case>/log.simpleFoam`, `log.snappyHexMesh`,
  `ccx.log`, Gmsh's log) — read them for residuals and errors; `run.log` only points at them.
- The exit code: `0` completed (or prepared), `3` partial (stopped by `--max-cases` / `--max-minutes`), `1` failed,
  `4` completed but nothing solved, `130` cancelled.

## The results contract

```
<run>/manifest.json         schema version, run id, created_at, host, pid, command, cwd, config + config_hash,
                            inputs (paths + sha256), design (the fingerprint), tool_versions (python, vegeta-cli,
                            openfoam / calculix, git commit), planned_cases, resumed[], artifacts{}
<run>/status.json           see above
<run>/run.log               see above
<run>/summary.json          written ONLY when the run completed its case list (atomically). A prepared, partial,
                            failed or cancelled run writes summary.partial.json instead: inspectable, never a
                            completed usable result.
<run>/cases/<case>/         the native case (OpenFOAM directory / mesh.msh + .inp + .frd + .dat), its logs, and
                            case.json (name, key, status, process, converged, timings, metrics, artifacts)
<run>/*.csv                 sampled curves and maps (e.g. bare_sweep.csv, blown_points.csv) indexed from summary.json
```

`summary.json` holds scalars with their units, the **reference quantities and frame** of the coefficients (S_ref,
chord, the CG the moments are about, wind axes), every case's status / process / convergence / failure reason, the
sampled maps as file references with their columns and **valid interpolation domain** (α, V, the slipstream ratio μ)
and the interpolation the notebook may do inside it — never a dense map manufactured from sparse samples, never an
extrapolation; and a `validation_note` saying what a solved-but-unconverged case is. Numbers are stored at full
precision; rounding is for presentation. Spatial fields stay in the native formats (OpenFOAM time directories,
CalculiX `.frd`) for ParaView; JSON indexes them, it never contains them.

**Design identity.** Each run's `design` is the fingerprint of the variant it was made for: the resolved CAD
parameters and their hash, the mass and CG the cases were set up for, the propeller calibration the rotor disks and
drive tables carried, the source hashes of the design modules. The notebook compares it with the design it builds:
a different CAD parameter, a CG more than 2 mm away or a mass more than 2 % off is **incompatible** (the run is not
used, even though it exists); a different propeller calibration is incompatible for rotor-disk cases and a warning
for the bare airframe; changed sources are warnings. The imported legacy campaign carries an *assumed* fingerprint
and says so.

**Resume.** `--resume` reopens the directory, refuses a different configuration, reuses every case whose record is
finished (`solved`, `diverged`, `FAILED`) with the same case key (template, values, geometry hash), and reruns the
rest from scratch. There is no solver checkpoint restart: an interrupted case starts again.

## Room for the next workflows

A thermal analysis (a motor's or battery's heat path), another aircraft's CFD, a durability campaign: a new script
under `workflows/<project>/` that imports `runkit`, builds its cases from the project's design module, runs them in
the same loop and writes the same files. Nothing in `runkit` is Kestrel-specific.
