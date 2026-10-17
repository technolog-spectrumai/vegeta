# Running the Kestrel workflows — a short manual

This is the plain version of `workflows/README.md`: what to type, in which order, and what you get back. All
commands are run from the repository root on Ubuntu, with the virtual environment that `install_local.sh` made
(`.venv`). OpenFOAM is needed for the CFD, CalculiX (`ccx`) and Gmsh for the FEA — `./test.sh --quick` tells you what
is installed.

## 0. Set the shell up

```bash
source activate.sh        # activates .venv, defines  wf  (= workflows/cli.py)  and  wfrun <book> …  (= the book's script)
wf                   # interactive
```
Everything below also works as `.venv/bin/python workflows/cli.py …` without sourcing anything.

## 0b. The desktop interface

```bash
pip install PyQt6            # once (or: sudo apt install python3-pyqt6 libegl1 libxkbcommon-x11-0 libxcb-cursor0)
./workflows.sh               # or: wf gui  (= .venv/bin/python workflows/gui)
```

One window: the **tree** of books and chapters on the left, each chapter marked with what happened to it so far —
✔ completed, ▶ running, ◐ partial or cancelled, ○ prepared, ✘ failed, — not run — per variant from its latest run
(double-click a chapter, or type its code — `K2` — in the box); the chapter's **options** on the right (variant, quality or threads, mode: run / prepare only / list,
cache on or off, limits) with the exact command shown and a big **Run chapter** button; below, the **log** of the
current run with a progress bar (cases done / planned) and the heartbeat in the status bar, and the **runs on
disk** table with Follow / Resume / Stop / Notebook line / Open folder. A run starts detached, so closing the window
never stops it. The toolbar has Refresh, the runs folder, the legacy import, the cache (with Clean), light / dark.
Everything the window does, the command line does too; it computes nothing itself.

## 1. The idea in one paragraph

A **workflow** is a script that runs the slow solver cases (CFD, FEA) and writes everything into one folder, the
**run directory**. The **notebook** (`notebooks/43_kestrel_study.ipynb`) never runs a solver: it reads run
directories you point it at, checks that they are finished and belong to the current design, and shows the numbers
beside its own quick estimates. Workflows are organised in **books** (a design: `kestrel`, code **K**) and numbered **chapters** (the studies on
it, CFD and FEA alike: K1 `airframe-sweep`, K2 `airframe-stall`, K3 `airframe-cruise`, K4 `airframe-transitions`,
… K9 `frame`, K10 `joint`); you pick a chapter by its code, `K2` or `k2`. Results land in
`workflows/runs/<book>/<chapter>/<run id>/`. Some chapters need another chapter of the same book first (K3, K4 and
K5 need a completed K1; K10 needs K9) — the tool tells you. Cases a book
already solved are **cached** and reused by later runs (`wf cache`, `wf clean-cache`, `--no-cache` to force). You can use the workflows without the notebook (the run directory is readable on its
own) and the notebook without the workflows (a screening pass with the solver results marked NOT RUN).

## 2. The easy way: `workflows/cli.py`

One tool lists, explains, launches and watches every workflow. With no arguments it asks you what to run:

```bash
wf                                       # interactive: book → chapter → variant/quality → run / prepare only / list; cache yes/no; confirm; detached or not
wf list                                  # the books, their chapters (with prerequisites), runs and cache
wf info K2 --variant 4Q-2     # the exact cases a chapter would run
wf K2 --detach --variant 4Q-2 --quality fine [--no-cache]
wf cache                                 # the cached cases per book
wf clean-cache [--book kestrel] # remove them (the runs keep their native results)
wf runs                                  # every run on disk: state, cases done, age
wf status <run>                          # one run: heartbeat, last log lines, and what to do next
wf tail <run>                            # follow its run.log
wf stop <run>                            # stop it cleanly (finished cases are kept)
wf resume <run>                          # continue it
wf notebook <run>                        # the yn pin command that makes a notebook read this run
yn status | yn why 43 | yn doctor         # Yanus: what is ready, what blocks a notebook and why, what this machine can run
```

`<run>` is the run id (the directory name) or its path. `--detach` starts the script in its own session with its
terminal output in `workflows/runs/<workflow>_<time>.out`, so logging out or closing Jupyter does not stop it.
Everything below works without the tool too.

## 3. Without a notebook: run a campaign and read the result

```bash
# 1. see what a campaign contains (nothing is written)
.venv/bin/python workflows/kestrel/airframe_cfd.py --variant 4Q-2 --chapter stall --list

# 2. run it — in the foreground …
.venv/bin/python workflows/kestrel/airframe_cfd.py --variant 4Q-2 --chapter stall --quality fine --processors 4

#    … or detached, so you can log out (nohup), or in a tmux window
nohup .venv/bin/python workflows/kestrel/airframe_cfd.py --variant 4Q-2 --chapter stall --quality fine --processors 4 \
      > workflows/runs/stall.out 2>&1 &
tmux new -d -s kestrel '.venv/bin/python workflows/kestrel/structure_fea.py --variant 4Q-2 --chapter all --threads 4'

# 3. watch it (in a terminal a run shows two progress bars: the cases, and the solver inside the case)
tail -F workflows/runs/kestrel/airframe-stall/<run id>/run.log      # one line per case start/finish, a heartbeat every 30 s
cat workflows/runs/kestrel/airframe-stall/<run id>/status.json      # state, case counts, the case in progress

# 4. stop it if you must (Ctrl-C in the foreground, or:)
kill -INT <pid>                                           # the pid is in status.json; finished cases are kept

# 5. continue a stopped run
.venv/bin/python workflows/kestrel/airframe_cfd.py --resume workflows/runs/kestrel/airframe-stall/<run id>
```

The script prints the run directory at the end. Inside it:

| file | what it tells you |
|---|---|
| `status.json` | `prepared`, `running`, `completed`, `partial`, `failed` or `cancelled`; how many cases are done; the last heartbeat |
| `run.log` | the history: every case, its duration and status, the heartbeats (`alive … advancing` = the solver is writing; `no new output for N s` = quiet, not necessarily stuck) |
| `summary.json` | **only exists when the run completed**: every case's status and numbers (CL, CD, Cm or stresses, frequencies), the reference quantities, convergence per case |
| `summary.partial.json` | the same for a run that stopped early — inspect it, do not treat it as a result |
| `bare_sweep.csv`, `blown_points.csv` | the sampled curves (CFD) |
| `cases/<case>/` | the native OpenFOAM case or CalculiX files and their logs — open in ParaView, read with the tools |
| `manifest.json` | what the run was made for: command, settings, design fingerprint, tool versions, git commit |

Reading the heartbeat: a healthy long run shows `advancing` most of the time and `Time = N` climbing in
`run.log`; a stalled one shows `no new output for` growing to many minutes with the solver's own log unchanged
(`tail -n 3 cases/<case>/log.simpleFoam`); a failed one ends with `run failed` or a case `FAILED — <reason>`.

What the campaigns are:

| code | chapter | cases | purpose | rough time (4 cores) |
|---|---|---|---|---|
| K1 | `airframe-sweep` | 10 bare airframe angles 0–90° | checks the lattice: lift slope, Cd0, Cmα | an afternoon (screening) |
| K2 | `airframe-stall` | 5 bare angles 8–16° | CL max → stall speed → wing area | 1–2 h screening, ~5 h fine |
| K3 | `airframe-cruise` | 2 with rotor disks | level cruise and the stall point, blown (after K1) | ~1 h |
| K4 | `airframe-transitions` | ~15 with rotor disks | the points the transitions fly (after K1) | a day |
| K5–K8 | `airframe-blown`, `airframe-sweep-fine`, `airframe-all`, `sections` | combinations / the section slabs | | |
| K9 | `frame` | 4 (A, B, C, modes) | the carbon frame at ultimate load | minutes each |
| K10 | `joint` | 1 | the printed spar–post joint (after K9) | minutes |
| K11 | `frame-and-joint` | 5 | both | under an hour |

Useful options: `--cases "α 12°"` runs only matching cases; `--max-cases 2` / `--max-minutes 90` bound a run (it ends
`partial`, resumable); `--prepare-only` writes the cases without solving (to inspect a mesh setup); `--quiet` keeps
the terminal clean (everything is still in `run.log`).

Variants: `4Q-1` (round struts, X bracing), `4Q-2` (elevons, faired posts, the reflexed section — the main one),
`4Q-3` (4Q-2 with crow-brake elevons and reverse-capable drives).

## 4. With the notebook

1. Start JupyterLab with `./jupyter.sh` and open `43_kestrel_study.ipynb`.
2. The notebook reads, for every chapter and variant, the **newest successful run** on disk (completed and compatible
   with the current design) unless one is **pinned** in `notebooks/pins/43_kestrel_study.json`; the table says which. Pin a run with `yn pin 43 K1 --variant 4Q-2 workflows/runs/kestrel/airframe-sweep/<run id>` — the
   workflow prints the exact command when it finishes, `yn pin 43 --suggest` lists the usable runs on disk (it writes
   nothing), and `yn why 43` shows the whole gate from the shell before you open the notebook. Which inputs are
   required and which optional is the notebook's contract in `workflows/flow.py` (`docs/yanus.md`).
3. Run the notebook. The prerequisite table shows, for every input: required or optional, which run, where it came
   from, whether it is compatible with the current design, and whether it is usable.
   - A **required** input that is missing, unfinished, failed or made for another design stops the notebook with an
     error that contains the command to run. Run it (section 2), pin the run directory it prints (`yn pin …`), run again.
   - An **optional** input that is missing shows **NOT RUN** and the notebook's own estimate stands beside it,
     labelled as an estimate.
4. For a quick look without any solver result: `KESTREL_SCREENING=1 ./jupyter.sh` (or set the variable in the
   kernel's environment). Everything becomes optional and the final assessment is marked *unvalidated*.
5. Section 10 of the notebook lists which campaigns to run next and why, with the commands ready to copy.

The notebook takes about two minutes (it prints its measured runtime at the end). Closing it never stops a workflow
you started from a shell; re-run the notebook when a workflow has finished.

## 5. The first time

The first CFD campaign (October 2026, screening quality, unconverged) lives in
`notebooks/designs/data/kestrel_cfd_results.json`. Turn it into run directories once, so the notebook can read it:

```bash
.venv/bin/python workflows/kestrel/import_legacy.py
```

It prints two directories under `workflows/runs/kestrel/airframe-sweep/` (chapter K1) — the ones the notebook's pins point at by
default. Re-pin (`yn pin 43 K1 --variant 4Q-2 <run>`) as real workflow runs arrive.

## 6. Order of work recommended now

1. `import_legacy.py` (seconds) and the notebook in screening mode (two minutes): the baseline numbers.
2. `wf K2 --variant 4Q-2 --quality fine` (hours): decides CL max, the stall speed and
   whether the wing must grow.
3. `wf K11 --variant 4Q-2` (under an hour): the frame's and the joint's margins.
4. Notebook again in its default mode with both runs pinned (`yn pin 43 K2 …`, `yn pin 43 K11 …`): the validated assessment.
   For a quick *what if* on the design (span, chord, section, propeller, motor, battery, spar) without any solver,
   `44_kestrel_fast.ipynb` compares candidates in about a minute (Zephyr sections, Boreas points, beam rules).
5. Then `cruise` and `transitions` with the rotor disks, and `38_prop_lab` at fine quality for the propeller.
