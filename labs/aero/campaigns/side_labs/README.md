# Side labs — explicit, file-based long-running analyses

Side labs (formerly *workflows*) run the optional, expensive calculations and fill the Porunga cache; they never
name a notebook (`side_labs/tests/test_separation.py`). A completed Kestrel run is published with `porunga.save`
under the key of `designs/kestrel_keys.py`; a notebook derives the same key and loads it.

(The short how-to for engineers is `docs/manual_workflows.md`; this file is the contract.)

A **workflow** is a thin script under `side_labs/<project>/` that prepares explicit inputs, runs the selected solver
cases, gathers the artifacts, parses the results and saves them into one **run directory**. It is launched by an
engineer from a shell, runs unattended for as long as it needs, and leaves a record a notebook can read, check and
cite. The engineering logic stays in the design modules (`designs/*.py`) and the Vegeta tools; the
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
  of seconds) is announced in the cell that does it. The one loud exception: a notebook flagged `yanus_auto`
  (`docs/yanus.md` § 16) runs the chapters it is missing in **screening quality**, in the foreground, through its
  Yanus gate — the gate says what it started and the results are labelled indicative.
- A result that is missing, stale, incompatible, running or failed is **NOT RUN** in the notebook: the analytical
  estimate stays visible beside it, labelled as such; optional validation may stay NOT RUN, required inputs stop
  the notebook with an actionable exception.
- Compatible results are **reused**, never recomputed: a run carries the fingerprint of the design it was made for.

The first workflows are Kestrel's (`side_labs/kestrel/`); `runkit.py` is the shared run-directory bookkeeping the
next long-running analysis (a thermal case, another aircraft's CFD) reuses as it is.

## Books and chapters

A **book** is a design — a part or assembly we simulate: `kestrel` is the Kestrel aircraft, code **K**. Its
**chapters** are the numbered studies on it, of every kind (CFD and FEA alike), chosen by code, case-insensitive:
`K1`, `k2`, `K10`. `wf list` prints the tree:

```
kestrel  (K)  Kestrel-4Q — the zero-stagger biplane quadrotor tailsitter
├── CFD — the airframe with the propellers as rotor disks (Aeromant / OpenFOAM)
│   ├── K1   airframe-sweep         the bare airframe 0–90°: the model polar's reference
│   ├── K2   airframe-stall         8–16°: CL max and the stall speed
│   ├── K3   airframe-cruise        the cruise and the stall point with the rotor disks          [after K1]
│   ├── K4   airframe-transitions   the transition points with the rotor disks, the hover        [after K1]
│   ├── K5   airframe-blown         cruise + transitions                                          [after K1]
│   ├── K6   airframe-sweep-fine    13 angles, 8–16° resolved
│   ├── K7   airframe-all           sweep + blown
│   ├── K8   sections               the two finalist sections as slabs
│   └── K12  wing-interference      one wing / the pair / the equal monoplane at 0, 4, 8°: the interference measured
└── FEA — the carbon frame and the printed spar–post joint (Talos: Gmsh + CalculiX)
    ├── K9   frame                  A, B, C, modes
    ├── K10  joint                                                                                [after K9]
    └── K11  frame-and-joint
kestrel-x  (X)  Kestrel-X-C — the shallow X-wing with the payload module (variants X-C, X-C45, X-CX)
├── CFD — the bare X-C (Aeromant / OpenFOAM)
│   ├── X1   airframe-sweep         the bare X-C 0–90° at 10 m/s: the lattice's cos² φ rule, Cd0, the deep stall
│   └── X2   airframe-stall         8–16°: the panels' CL max at their dihedral
└── FEA — one panel's carbon spar in its socket (Talos: Gmsh + CalculiX)
    └── X3   panel                  the 4 g pull-up with the tip motor's inertia, the first modes
```
(the races book R graduated to Piccolo with the racing quad: `races/lab/race.py` there; docs/MOVED.md)

A second book reuses the same scripts: `side_labs/kestrel_x/airframe_cfd.py` and `panel_fea.py` call the Kestrel
scripts' `main(book="kestrel-x")`, which load the design adapter named in `CAMPAIGNS` (`kestrel_xc_campaigns`, the
same interface as `kestrel_campaigns`); `library.kind_of(book, kind)` is what the CLI, the batch runner and the GUI
read, so a chapter code carries its book.

The exported files follow the structure: every run is `side_labs/runs/<book>/<chapter>/<run id>/` (e.g.
`side_labs/runs/kestrel/airframe-stall/20261017-101500_4Q-2_fine/`), every cached case
`side_labs/cache/<book>/<case key>.json`. A chapter's **prerequisites are chapters of the same book only**: a chapter
does not start until each prerequisite has a completed run for the same variant with a compatible fingerprint
(`--ignore-prerequisites` overrides and is recorded in the manifest). The registry is `side_labs/flow.py` (`side_labs/registry.py`): the books, each chapter in order with its kind, the
script's campaign it maps to and its prerequisite chapters (`takes=[Run(...)]`), each kind's script and what it needs
on the machine (Ozaru's Yanus need types). `side_labs/library.py` generates `BOOKS` / `KINDS` from it for `wf` and
the GUI. A new chapter is one line there, a new book a new entry and a new script.

Notebooks never read run directories: a completed run publishes its summary into the Porunga cache under a key
(`designs/kestrel_keys.py`, `docs/cache_identities.md`) and a notebook loads it by the same key.

**Cache.** A case is identified by its key (template, every value, the geometry file's hash, the quality). When a
run of a book asks for a case any earlier run of that book already solved, the case record is read back from
`side_labs/cache/<book>/` — status, convergence, coefficients, and the path of the run that holds the native files —
and logged as a cache hit; nothing is solved twice. Only `solved` and `diverged` cases are cached (a `FAILED` one is
retried). `--no-cache` solves everything; `side_labs/cli.py cache` shows the cache, `clean-cache [--book B]` empties
it (the runs keep their native results). A cache hit is still subject to the run's design fingerprint, because the
key contains the geometry and the values.

## Layout

```
side_labs/
  README.md                    this file
  gui/                         the desktop interface (PyQt6): app.py the window, runner.py launching and following a run, models.py the runs table, theme.py
  cli.py                       the entry point: list / info / run / runs / status / tail / stop / resume / notebook / cache / clean-cache, or interactive
  library.py                   the books (designs) and their numbered chapters (studies), prerequisites, the tree, the run folders
  runkit.py                    the run-directory contract: manifest, status, run.log, heartbeat, cancel, resume, summary
  kestrel/
    airframe_cfd.py            the whole aircraft with the propellers as rotor disks (Aeromant / OpenFOAM), one campaign per run
    structure_fea.py           the carbon frame and the printed joint (Talos: Gmsh + CalculiX)
    import_legacy.py           the first CFD campaign's results file as run directories
    _kestrel.py                what the Kestrel scripts share (paths, the case loop under a budget)
  tests/                       the contract on a fake solver; the campaigns; prepare-only smoke cases
  runs/<book>/<chapter>/<run id>/   every run (gitignored): see the contract below
  cache/<book>/<key>.json           the solved cases reused across runs (gitignored)
```

The engineering content of the Kestrel workflows — the variants and their fingerprints, the campaign catalogue, the
case builders, the summary and the loaders — is `designs/kestrel_campaigns.py`, shared with the notebook.
Resolved-propeller CFD is **not** a workflow: it stays in notebook `38_prop_lab`, whose exported calibration
(`designs/data/prop_lab.json`) the airframe cases and the drive tables read when it passed its gates.

## Running (Ubuntu)

All commands from the repository root, with the virtual environment `install_local.sh` made (`.venv`); `source activate.sh`
activates it and defines `wf` (= `side_labs/cli.py`) and `wfrun <book> …` (= the book's script). The quickest
way is `side_labs/cli.py` (or its PyQt6 window: `./side_labs.sh` / `wf gui`, `side_labs/requirements-gui.txt`) (no arguments: interactive; `list`, `info`, `run --detach`, `runs`, `status`, `tail`, `stop`,
`resume`, `notebook`); the scripts below are what it calls.

```bash
# what a campaign would do, without writing anything
.venv/bin/python side_labs/kestrel/airframe_cfd.py --variant 4Q-2 --chapter stall --list
.venv/bin/python side_labs/kestrel/structure_fea.py --variant 4Q-2 --chapter all --list

# prepare only: the cases written (STL, dictionaries / STEP, mesh), no solver — the run says 'prepared'
.venv/bin/python side_labs/kestrel/airframe_cfd.py --variant 4Q-2 --chapter stall --prepare-only

# foreground
.venv/bin/python side_labs/kestrel/airframe_cfd.py --variant 4Q-2 --chapter stall --quality fine --processors 4

# detached with nohup (survives the terminal and Jupyter closing)
nohup .venv/bin/python side_labs/kestrel/airframe_cfd.py --variant 4Q-2 --chapter sweep --processors 4 \
      > side_labs/runs/airframe_cfd_sweep.out 2>&1 &
echo $!                                   # the PID (also in <run>/status.json)

# detached in tmux (attach later with: tmux attach -t kestrel)
tmux new -d -s kestrel '.venv/bin/python side_labs/kestrel/structure_fea.py --variant 4Q-2 --chapter all --threads 4'

# watching a run
tail -F side_labs/runs/kestrel/<chapter>/<run id>/run.log
cat side_labs/runs/kestrel/<chapter>/<run id>/status.json
tail -n 3 side_labs/runs/kestrel/<chapter>/<run id>/cases/<case>/log.simpleFoam        # the native solver log

# stopping, resuming
kill -INT <pid>        # or Ctrl-C in the foreground: the running solver is stopped, the run ends 'cancelled', finished cases are kept
.venv/bin/python side_labs/kestrel/airframe_cfd.py --resume side_labs/runs/kestrel/<chapter>/<run id>
```

Closing Jupyter never stops a workflow: the scripts are separate processes that write to disk; the notebook only
reads the run directory, whenever it is next executed.

Common options (every script): `--chapter` (`--campaign` still accepted), `--out DIR` (default `side_labs/runs/<book>/<chapter>/<timestamp>_<label>`), `--no-cache`, `--ignore-prerequisites`,
`--resume RUN_DIR`, `--rerun-failed`, `--prepare-only`, `--list`, `--cases TEXT[,TEXT]` (a name filter: a run does
the cases it lists and never more), `--max-cases N`, `--max-minutes M` (no new case after that; the run ends
`partial`, resumable), `--case-timeout S`, `--heartbeat S` (default 30), `--quiet`, `--no-progress`. CFD: `--quality screening|fine`,
`--processors N` (MPI ranks per case; meshing is serial). FEA: `--threads N`, `--n-modes`, `--element-size`.

Orders of magnitude (one 4-core workstation): a screening CFD case 10–20 min, a fine one about an hour; the `sweep`
campaign at screening an afternoon, `all` at fine days. An FEA frame case a few minutes; the whole `all` campaign
under an hour. Measure your own: `run.log` records every case's duration.

## A batch of chapters (`batch.yaml`)

`wf batch file.yaml` runs an ordered list of chapters from one file — the way to hand the heavy runs to a cloud box:
the GUI's Batch tab (or `wf batch --example`) writes it, any machine with the repository and the solvers runs it. The
format and the folder it leaves behind (`side_labs/runs/_batches/<stamp>_<name>/batch.json`: every step's state, exit
code, run directory and summary file) are in `side_labs/batch.py` and `docs/manual_workflows.md` § 2b. Each step is the
chapter's script with an explicit `--out`, so a batch never changes what a run is; a batch is an engineer's explicit
list, not a scheduler (`docs/exploration.md` Part 3).

## Reading a run: the logs

Engineers must be able to tell a healthy long run from a stalled or failed one.

- **`run.log`** is written continuously (line-buffered, flushed): the run's creation and command, the configuration
  hash, the tool versions found, every case's start and finish with its directory, status and duration, the solver
  stages as the tools report them, and a **heartbeat** every `--heartbeat` seconds while a solver runs:
  `heartbeat: <case> alive (pid P, children [...]), 412 s into the case; cases/<dir>/log.simpleFoam 812 kB, advancing — Time = 213`.
  "alive" is the process; "advancing" is the solver's log having grown since the last beat; "no new output for N s"
  is reported, never acted on — a snappyHexMesh layer step or a CalculiX factorisation can be quiet for minutes. No
  percentage is invented.
- **Progress bars** (a terminal only): the chapter's cases as one bar (done / planned, the case in hand, the cache hits)
  and the solver inside a case as a second one — OpenFOAM's iteration out of the case's iteration count, Talos's mesh
  and solve stages — as the tools report them through their progress callbacks. Off when the output is not a terminal
  (a detached run) or with `--no-progress`; `run.log` carries the same information either way.
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
under `side_labs/<project>/` that imports `runkit`, builds its cases from the project's design module, runs them in
the same loop and writes the same files. Nothing in `runkit` is Kestrel-specific.

The laser books `fenix` (F) and `svarog` (S) and their FEniCSx interpreter moved to
[Alabastron](https://github.com/technolog-spectrumai/alabastron) on 2026-10-18.
