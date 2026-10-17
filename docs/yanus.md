# Yanus — the door keeper (design, v0 — for review)

> Status: **v1 implemented** on `dev_yanus` (plan: `docs/yanus_plan.md`; package: `vegeta-yanus/README.md`).
> Kestrel first: K1–K11 and notebooks 38, 43, 44. See § 12 for where the implementation differs from this design.

Yanus guards the doors between Vegeta's notebooks and workflows. Like Janus it has two faces: one looks at what
comes **in** (tools on the machine, prerequisite runs, imported files), one at what goes **out** (the files a
notebook or workflow promised to export). It **guards, never drives**: it checks, blocks, explains and names the
command that would fix things; it never launches a solver, a notebook or a chain of them (`rejected.md`: no
automatic chaining, no DAG engine, no scheduler, no database, no YAML; `workflows/README.md`: a notebook never
launches a solver silently).

## 1. The problem today

| What should be guarded | How it is done now | Gap |
|---|---|---|
| Tools and packages (OpenFOAM, ccx, gmsh, vegeta.*) | `activate.sh`, `test.sh`, `VEGETA_SKIP_OPENFOAM`, try/except around `detect()` | spread out; no unit says what *it* needs; a `wf K3` finds a missing OpenFOAM only when it runs |
| Prerequisite runs | `library.check_prerequisites` (chapter→chapter, same book); `kestrel_campaigns.check_run`/`require` + `kestrel_study.prerequisites` (notebook 43) | two implementations; the chapter map exists twice (`library.py`, `kestrel_campaigns.py:41`); notebook→notebook not covered |
| Selecting which run a notebook reads | the hand-edited `RUNS = {...}` dict in notebook 43, `wf notebook <run>` prints a line to paste | selection is code inside the notebook; easy to point at a run that does not exist |
| Imported files (`designs/data/*.json`, `prop_lab.json`, dossiers) | ad hoc `load_design()` → `None` when missing | no producer, no schema version, no staleness check |
| Exported files | written wherever; runkit runs are well-formed | nothing checks a notebook actually wrote what the next one reads |
| Scaffolding | the header (DEBUG / `sys.path` / `OUT`) is copied by hand (~41 of 48 notebooks) | new notebooks and chapters start from copy-paste |
| One view of the flow | `wf list` (chapters only) | notebooks are invisible; nobody can see "what is red and why" |

## 2. Principles

1. **Guard only.** Yanus reads files and the environment; its only writes are stamps, pins and scaffolds — never results.
2. **One registry, in Python.** Every unit (a notebook or a workflow chapter) is declared once, with its contract.
   No second copy of the chapter map anywhere.
3. **Explicit selection stays explicit.** Yanus may *suggest* the newest compatible run; the notebook reads the
   **pinned** one. Nothing is picked silently (notebook 43's rule).
4. **All-at-once errors.** A gate checks everything, then raises one error listing *every* missing item with the
   exact command that produces it — the pattern of `kestrel_study.prerequisites`, generalised.
5. **Required vs optional, and screening.** A missing optional input is NOT RUN (the analytic estimate stays
   visible); a missing required one closes the gate. A screening mode turns required into optional, and says so.
6. **Files are the truth.** Runs keep runkit's contract (`manifest.json`, `status.json`, `summary.json`); other
   exports get a small sidecar stamp. No database.
7. **Standalone package.** `vegeta-yanus` imports no other vegeta package (the import-isolation rule); it reads
   runkit's files from disk and takes project-specific checks (fingerprint compatibility) as plain callables.

## 3. Vocabulary

| Term | Meaning |
|---|---|
| **Unit** | a notebook (`43_kestrel_study`) or a workflow chapter (`K3` = `kestrel/airframe-cruise`) |
| **Contract** | what a unit `needs`, `takes` and `gives` |
| **Need** | something on the machine: a `Tool` (`ccx`, OpenFOAM), a `Package` (`vegeta.aeromant`, `gmsh`), an `Env` var |
| **Take** | an input produced by another unit: a `Run` of a chapter, or a `File` another unit gives |
| **Give** | an output the unit promises: a `File` (JSON/CSV/STEP/…) or, for a chapter, its run |
| **Gate** | the check at a unit's door: `enter` (needs + takes) and `leave` (gives) |
| **Pin** | the explicit choice of which run/file version a unit reads |
| **Stamp** | the sidecar record of an exported file: producer, schema, hash, design fingerprint, time, git commit |

## 4. The contract (Python, in the registry)

```python
# workflows/flow.py — the one registry (replaces library.BOOKS' chapter map and kestrel_campaigns.CHAPTERS)
from vegeta.yanus import Flow, Notebook, Chapter, Tool, Package, Env, Run, File

flow = Flow("vegeta")

kestrel = flow.book("kestrel", code="K", variants=("4Q-1", "4Q-2", "4Q-3"),
                    compatible=lambda expected, found: kestrel_compare(expected, found))   # project callable

cfd = dict(kind="cfd", script="workflows/kestrel/airframe_cfd.py",
           needs=[Tool.openfoam(), Package("vegeta.aeromant")])
fea = dict(kind="fea", script="workflows/kestrel/structure_fea.py",
           needs=[Tool("ccx", version_cmd="ccx -v"), Package("gmsh"), Package("vegeta.talos")])

kestrel.chapter("airframe-sweep", campaign="sweep", **cfd)                                   # K1
kestrel.chapter("airframe-cruise", campaign="cruise", takes=[Run("airframe-sweep")], **cfd)  # K3
...
kestrel.chapter("joint", campaign="joint", takes=[Run("frame")], **fea)                      # K10

flow.notebook("38_prop_lab",
              needs=[Package("vegeta.boreas")],
              gives=[File("designs/data/prop_lab.json", schema=2)])

flow.notebook("43_kestrel_study",
              needs=[Package("vegeta.dedalus"), Package("vegeta.boreas")],
              takes=[Run("kestrel/airframe-sweep", variant="4Q-2", required=True),
                     Run("kestrel/frame-and-joint", variant="4Q-2", required=True),
                     Run("kestrel/airframe-sweep", variant="4Q-1"),          # optional: NOT RUN shown
                     File("38_prop_lab:designs/data/prop_lab.json", min_schema=2)],
              gives=[File("data/reports/43_kestrel_study.json"), File("designs/data/kestrel_dossier.json")],
              screening_env="KESTREL_SCREENING")
```

`library.py` keeps its public functions (`chapters`, `by_code`, `tree`, `runs_of`, `check_prerequisites`) so `wf`
and the scripts do not change; they become thin views over `flow`. Chapter codes (`K1…`) stay derived from order.

## 5. The two faces

### Entry — `yanus.enter(unit)`

```python
# first code cell of a notebook (generated by the scaffold)
from vegeta import yanus
gate = yanus.enter("43_kestrel_study")      # checks needs + takes against pins; displays the gate table
SUMM = gate.inputs                          # the usable summaries / file paths, keyed like the contract
```

* **Needs**: each tool/package/env checked, version recorded (for the stamp and the run manifest).
* **Takes / Run**: the pinned run exists → has `manifest.json` → workflow matches → `status.json` is `completed`
  with `summary.json` → the book's `compatible(expected, manifest["design"])` is not `incompatible`.
* **Takes / File**: the file exists → its stamp names the expected producer → schema ≥ `min_schema` → (optional)
  fingerprint compatible; a file stamped before its producer's design last changed is **stale** (warn, or fail if
  required — open question Q3).
* Result: a table (unit, input, role, status, provenance, compatibility, fix-command) shown in the notebook and
  in the shell; then either `gate.inputs` or one `GateClosed` (subclass of today's `PrerequisiteError`) listing
  every required failure with its command (`wf K1 --variant 4Q-2`, `jupyter … 38_prop_lab`, `wf import_legacy`).
* Workflow scripts call the same gate before preparing cases, so `wf K3` fails in a second when OpenFOAM is
  absent or K1 has no compatible run (replaces `prerequisites_or_exit`; `--ignore-prerequisites` stays and is
  recorded in the manifest as now).

### Exit — `gate.leave()`

* Last cell of a notebook (scaffolded). Checks every `gives` file was written **in this execution** (mtime after
  `enter`), writes its stamp `<file>.yanus.json`:
  `{schema, producer, unit_kind, written_at, sha256, design_fingerprint, git, tool_versions, inputs: [pins used]}`.
* A missing promised output fails the notebook run (`run_notebooks.sh` turns red) instead of failing the *next*
  notebook later.
* For chapters, runkit's `summary.json` already is the exit record; `leave` only verifies it.

## 6. Pins — explicit selection, out of the notebook code

`notebooks/pins/<unit>.json` (data, committed or not — Q2):

```json
{"kestrel/airframe-sweep@4Q-2": "workflows/runs/kestrel/airframe-sweep/20261017-101500_4Q-2_fine"}
```

* `yanus pin 43 K1 --variant 4Q-2 <run>` writes it (`wf notebook <run>` becomes this).
* `yanus why 43` shows: *pinned run is incompatible; newest compatible is … — `yanus pin 43 K1 …` to use it*.
* Nothing is ever picked automatically; an unpinned optional input is NOT RUN, an unpinned required one closes the
  gate with the suggestion.

## 7. Scaffold

* `vegeta yanus new notebook 45_thermal --book kestrel` → `notebooks/45_thermal.ipynb` with the standard header
  (logo, title, DEBUG/mock cell, imports + `sys.path`, `OUT = output/45_thermal`, `gate = yanus.enter(...)`),
  a section skeleton, the `gate.leave()` cell; and a stub entry in `workflows/flow.py` to fill in.
* `vegeta yanus new chapter kestrel/thermal --kind fea` → a runkit script skeleton under `workflows/kestrel/`
  (argparse with `--variant/--resume/--no-cache/--ignore-prerequisites`, `Run.create`, the gate, the case loop)
  and its registry entry.
* `vegeta yanus check --fix-header <nb>` (later): reports notebooks whose header drifted from the scaffold.

## 8. The one view

```
$ vegeta yanus status                       # or: wf status --flow
kestrel (K)
  K1  airframe-sweep         ✓ 4Q-1 4Q-2   · 4Q-3 not run          needs ✓ OpenFOAM v2406
  K3  airframe-cruise        ✗ blocked: K1 4Q-3                      needs ✗ OpenFOAM missing
  K9  frame                  · not run                               needs ✓ ccx 2.21, gmsh
notebooks
  38_prop_lab                ✓ gives prop_lab.json (schema 2, 2026-10-12)
  43_kestrel_study           ✗ K9 4Q-2 required, not pinned  → wf K9 --variant 4Q-2; yanus pin 43 K9 …
  44_kestrel_fast            ✓ no takes
$ vegeta yanus why 43                       # the full gate table for one unit, with every fix command
$ vegeta yanus doctor                       # needs only: what this machine can run (replaces activate.sh's line)
```

`status --json` for CI; `test.sh` gets an `optional` row that runs `yanus doctor`.

## 9. Package shape

```
vegeta-yanus/
  pyproject.toml              # deps: none besides stdlib (pandas optional for the table); entry point vegeta.commands: yanus
  src/vegeta/yanus/
    contract.py               # Need/Tool/Package/Env, Take/Run/File, Give, Unit, Book, Flow
    gate.py                   # enter / leave, GateReport, GateClosed
    runs.py                   # reads runkit run dirs (manifest/status/summary) — file contract only, no import of runkit
    stamps.py                 # sidecar stamps, staleness
    pins.py                   # pin files
    scaffold.py + templates/  # notebook and chapter skeletons
    cli.py                    # status, why, doctor, pin, new, stamp
  tests/
```

Project side: `workflows/flow.py` (the registry), `library.py` and `kestrel_campaigns.py` reduced to views over it,
notebook 43 switched to `yanus.enter`/pins, scripts call the gate. Registry found via `VEGETA_YANUS_FLOW`
(default `workflows/flow.py`, set by `activate.sh`).

## 10. v1 scope (Kestrel first)

In: the package; `flow.py` with book `kestrel` (K1–K11) and notebooks 38, 43, 44; entry/exit gates in 43 and the
two Kestrel scripts; pins replacing `RUNS`; `prop_lab.json` stamped by 38 and checked by 43; `status/why/doctor/pin`;
the two scaffolds; tests. Removing the duplicated chapter map; fixing the stale docstrings found on the way.

Out (later): the other 45 notebooks and `designs/data/*.json` links; header drift check; `notebooks/old/`;
any runner (never — guard only).

## 11. Open questions

* **Q1 — File schemas.** Is a `schema` integer + required top-level keys enough for exported JSON, or do you want
  field-level validation (types, ranges)? *Proposed: integer + required keys in v1.*
* **Q2 — Pins in git?** Commit `notebooks/pins/*.json` (reproducible intent, but runs are gitignored so pins may
  dangle on another machine) or keep them local? *Proposed: committed; a dangling pin is reported as NOT RUN.*
* **Q3 — Staleness.** When an input file's producer design changed after the file was stamped: warn or fail?
  *Proposed: required → fail, optional → NOT RUN with "stale".*
* **Q4 — Exit gate strictness.** Should a notebook that misses a promised output fail `run_notebooks.sh`?
  *Proposed: yes.*
* **Q5 — Where the registry lives.** `workflows/flow.py` (proposed) vs a top-level `flow.py`.
* **Q6 — Name in the CLI.** `vegeta yanus …` only, or also a short `yn` alias next to `wf`?

## 12. As implemented (v1) — where it differs from the design above

* **Q1–Q6.** These took the proposed answers:
  * Schema: an integer schema plus the required keys.
  * Pins: committed.
  * Stale input: closes the gate when required; NOT RUN when optional.
  * A missing promised output fails the run.
  * Registry: `workflows/flow.py`.
  * Command line: `yn` and `vegeta yanus`.
* **Staleness of files.** There is no design-fingerprint check. A file is stale when it changed after it was stamped,
  or when a stamped file it was made from was rewritten after it. Files written before Yanus are accepted as
  *unstamped*, and the gate says so.
* **Chapter prerequisites** take the newest usable run of the prerequisite chapter, as before. Only notebooks pin.
* **Notebook 43 shows two tables.** The first is the gate table: needs, pins, the prop_lab file. The second is the
  design's own prerequisite table, built from the pins by `kestrel_study.runs_from_gate`. It has the drive tables,
  the section polars and the compatibility levels. Everything after § 1 is unchanged.
  * A screening run of 43 gave a report identical to the committed one, apart from runtimes and the export time.
* **Gives.** `kestrel_dossier.json` is not declared as a give. Every notebook re-stitches it, so it has no single
  producer.
* **`doctor` finds packages; it does not import them.** A module whose import fails on a missing system library
  (gmsh without libGLU) still reads as found.
* **The committed exports are not stamped yet.** Stamping them needs 38 to run with OpenFOAM: run without CFD, it
  rewrites `prop_lab.json` with a NOT RUN listing. Until then the gate reports them as *unstamped* (todo Y.4).
