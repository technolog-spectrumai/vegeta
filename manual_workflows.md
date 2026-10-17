# Running the Kestrel workflows — a short manual

This is the plain version of `workflows/README.md`: what to type, in which order, and what you get back. All
commands are run from the repository root on Ubuntu, with the virtual environment that `install_local.sh` made
(`.venv`). OpenFOAM is needed for the CFD, CalculiX (`ccx`) and Gmsh for the FEA — `./test.sh --quick` tells you what
is installed.

## 1. The idea in one paragraph

A **workflow** is a script that runs the slow solver cases (CFD, FEA) and writes everything into one folder, the
**run directory**. The **notebook** (`notebooks/43_kestrel_study.ipynb`) never runs a solver: it reads run
directories you point it at, checks that they are finished and belong to the current design, and shows the numbers
beside its own quick estimates. You can use the workflows without the notebook (the run directory is readable on its
own) and the notebook without the workflows (a screening pass with the solver results marked NOT RUN).

## 2. Without a notebook: run a campaign and read the result

```bash
# 1. see what a campaign contains (nothing is written)
.venv/bin/python workflows/kestrel/airframe_cfd.py --variant 4Q-2 --campaign stall --list

# 2. run it — in the foreground …
.venv/bin/python workflows/kestrel/airframe_cfd.py --variant 4Q-2 --campaign stall --quality fine --processors 4

#    … or detached, so you can log out (nohup), or in a tmux window
nohup .venv/bin/python workflows/kestrel/airframe_cfd.py --variant 4Q-2 --campaign stall --quality fine --processors 4 \
      > workflows/runs/stall.out 2>&1 &
tmux new -d -s kestrel '.venv/bin/python workflows/kestrel/structure_fea.py --variant 4Q-2 --campaign all --threads 4'

# 3. watch it
tail -F workflows/runs/airframe_cfd/<run id>/run.log      # one line per case start/finish, a heartbeat every 30 s
cat workflows/runs/airframe_cfd/<run id>/status.json      # state, case counts, the case in progress

# 4. stop it if you must (Ctrl-C in the foreground, or:)
kill -INT <pid>                                           # the pid is in status.json; finished cases are kept

# 5. continue a stopped run
.venv/bin/python workflows/kestrel/airframe_cfd.py --resume workflows/runs/airframe_cfd/<run id>
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

| workflow | campaign | cases | purpose | rough time (4 cores) |
|---|---|---|---|---|
| `airframe_cfd.py` | `sweep` | 10 bare airframe angles 0–90° | checks the lattice: lift slope, Cd0, Cmα | an afternoon (screening) |
| | `stall` | 5 bare angles 8–16° | CL max → stall speed → wing area | 1–2 h screening, ~5 h fine |
| | `cruise` | 2 with rotor disks | level cruise and the stall point, blown | ~1 h |
| | `transitions` | ~15 with rotor disks | the points the transitions fly | a day |
| | `blown`, `all`, `sweep-fine`, `sections` | combinations / the section slabs | | |
| `structure_fea.py` | `frame` | 4 (A, B, C, modes) | the carbon frame at ultimate load | minutes each |
| | `joint` | 1 | the printed spar–post joint | minutes |
| | `all` | 5 | both | under an hour |

Useful options: `--cases "α 12°"` runs only matching cases; `--max-cases 2` / `--max-minutes 90` bound a run (it ends
`partial`, resumable); `--prepare-only` writes the cases without solving (to inspect a mesh setup); `--quiet` keeps
the terminal clean (everything is still in `run.log`).

Variants: `4Q-1` (round struts, X bracing), `4Q-2` (elevons, faired posts, the reflexed section — the main one),
`4Q-3` (4Q-2 with crow-brake elevons and reverse-capable drives).

## 3. With the notebook

1. Start JupyterLab with `./jupyter.sh` and open `43_kestrel_study.ipynb`.
2. The **first code cell** has a dictionary `RUNS` with the run directories the notebook reads, for example
   `RUNS["airframe_cfd"]["4Q-2"]["sweep"] = "workflows/runs/airframe_cfd/<run id>"`. Put your run directories there
   (the workflow prints the exact line to paste when it finishes).
3. Run the notebook. The prerequisite table shows, for every input: required or optional, which run, where it came
   from, whether it is compatible with the current design, and whether it is usable.
   - A **required** input that is missing, unfinished, failed or made for another design stops the notebook with an
     error that contains the command to run. Run it (section 2), paste the run directory into `RUNS`, run again.
   - An **optional** input that is missing shows **NOT RUN** and the notebook's own estimate stands beside it,
     labelled as an estimate.
4. For a quick look without any solver result: `KESTREL_SCREENING=1 ./jupyter.sh` (or set the variable in the
   kernel's environment). Everything becomes optional and the final assessment is marked *unvalidated*.
5. Section 10 of the notebook lists which campaigns to run next and why, with the commands ready to copy.

The notebook takes about two minutes (it prints its measured runtime at the end). Closing it never stops a workflow
you started from a shell; re-run the notebook when a workflow has finished.

## 4. The first time

The first CFD campaign (October 2026, screening quality, unconverged) lives in
`notebooks/designs/data/kestrel_cfd_results.json`. Turn it into run directories once, so the notebook can read it:

```bash
.venv/bin/python workflows/kestrel/import_legacy.py
```

It prints two directories (`legacy_…_4Q-1_…`, `legacy_…_4Q-2_…`) — the ones `RUNS` in the notebook points at by
default. Replace them as real workflow runs arrive.

## 5. Order of work recommended now

1. `import_legacy.py` (seconds) and the notebook in screening mode (two minutes): the baseline numbers.
2. `airframe_cfd.py --variant 4Q-2 --campaign stall --quality fine` (hours): decides CL max, the stall speed and
   whether the wing must grow.
3. `structure_fea.py --variant 4Q-2 --campaign all` (under an hour): the frame's and the joint's margins.
4. Notebook again in its default mode with both run directories in `RUNS`: the validated assessment.
5. Then `cruise` and `transitions` with the rotor disks, and `38_prop_lab` at fine quality for the propeller.
