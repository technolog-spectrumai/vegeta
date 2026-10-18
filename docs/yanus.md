# Yanus — finding, reading and checking saved workflow runs

Yanus (`vegeta-yanus`, `from vegeta import yanus`) is a small library: a notebook asks it for a saved workflow run,
and Yanus finds it, reads it and checks it. **It never starts a computation**: no solver, no workflow, no other
notebook.

## Where calculations happen

- **In the notebook:** short calculations, written as visible calls. This includes real solver calls that take
  minutes (a single FEA case, a CalculiX modal check, a NeuralFoil polar). A long one is announced in its cell.
- **In a workflow:** expensive analyses (`workflows/`, `wf`). A campaign takes roughly 15–30 minutes in total, or
  longer: the CFD sweeps, the FEA campaigns. A workflow writes a runkit run directory: `manifest.json`, `status.json`,
  `summary.json`, `run.log` and the native solver cases.
- **Yanus** reads those run directories, nothing else.

## The API

```python
from vegeta import yanus
sel = yanus.find("K1", variant="4Q-2", fingerprint=FP, quality="fine")      # the newest usable run
sel = yanus.find("K1", variant="4Q-2", fingerprint=FP,
                 run="workflows/runs/kestrel/airframe-sweep/<run id>")       # exactly this run, or a diagnostic
display(sel)              # which run was chosen (or why none) and the wf command that would produce one
sel.ok, sel.summary       # summary.json as a dict when usable, else None
sel.require()             # the summary, or yanus.RunUnusable with the reasons and the command
sel.provenance()          # a JSON-ready record for the report: the request, the run, its state, quality, compatibility
yanus.table(selections)   # several selections as one DataFrame
```

**The chapter.** A chapter is given by its code (`"K1"`) or by `book/chapter` (`"kestrel/airframe-sweep"`); the
registry `workflows/flow.py` says which script and run folder it is.

**Selection modes.** There are two:
- **latest:** the newest run under `workflows/runs/<book>/<chapter>/` that passes every check. Each newer run that
  was skipped is listed with its reason.
- **explicit:** `run=` names one directory. Only that directory is checked. If it fails, `find` reports why and
  **never selects another run**. A missing run made by an import script (`legacy_…`) points at that script
  (`wf import_legacy`).

**The checks.** `find` uses runkit's files only:
1. `manifest.json` is readable, uses runkit schema 1, and comes from the chapter's workflow.
2. The manifest's `config.variant` is the requested variant.
3. `status.json` says `completed`, and `summary.json` is present and readable JSON. Failed, partial, cancelled, dead or
   prepared runs are refused, each with the next step.
4. If a `quality` was requested, `config.quality` must be one of the requested values. A run that records no quality
   is refused when one was requested.
5. The run was **not made under the mocks**: no Mimesis `"mock": true` (runkit writes it when `VEGETA_DEBUG` / `vegeta.mock` is on), and no `debug`/`mock` quality. A runs folder ending in `-mock` (the mocks' folder) is refused as a whole. See `docs/mimesis.md`.
6. Compatibility uses the book's hook, which for Kestrel is `kestrel_campaigns.compare`. It compares the caller's
   `fingerprint` with the manifest's `design` and returns one of:
   - `incompatible`: the run is refused.
   - `warnings`: the run is accepted, with the reasons shown.
   - `unverified`: no fingerprint was given, or the book has no hook. The run is accepted and the label stays visible; it is never reported as compatible.

**Missing or unusable.** `find` does not raise for a missing or unusable run. The notebook decides:
- A required input calls `require()`, which stops with the reasons and the `wf` command.
- An optional one shows NOT RUN, and the notebook keeps its separately labelled analytical estimate.

Yanus never makes up numbers in place of a missing run.

## Notebook 43, the one that reads runs

`kestrel_study.INPUTS` lists the inputs as `(chapter, variant) → required`: K1 for every variant, K11 and K12 for
4Q-2. The required ones are K1 and K11 for 4Q-2, and `KESTREL_SCREENING=1` makes everything optional. The notebook's
first code cell carries `RUN`, the explicit run directories (at present the imported legacy sweeps).
`kstu.select_runs(FP, RUN)` calls `find` for every input, and `kstu.prerequisites` shows one table: the notebook's own
inputs, then the runs. It raises one `PrerequisiteError` (a `RunUnusable`) that lists every missing required run.
The JSON report's `meta.runs` records `sel.provenance()` for each input.

Notebook 45 reads notebook 43's report directly as a file, and warns loudly when it is missing. The other notebooks
read no workflow runs.

## Command line

```
yn find K1 --variant 4Q-2 [--quality fine] [--run DIR] [--json]   what a notebook would read, or why none (exit 2)
yn doctor [--json]                                                 the chapters' needs (tools, packages) on this machine
yn validate                                                        the registry (CI)
yn new chapter kestrel/thermal --kind fea [--new-kind] [--after frame]   a registry entry (+ a runkit script skeleton)
```

`wf notebook <run>` prints the `yanus.find(..., run=...)` line for a run and whether Yanus accepts it. A workflow
script ends with that line.

## What else is in the package

- **The workflow registry** (`contract.py`, `registry.py`). It holds the books, kinds, chapters, each chapter's needs and its prerequisite chapters. `wf`, the GUI, `batch.py` and the scripts read it.
- **`check_chapter`.** A script's entry check: its needs are reported, and each prerequisite chapter needs a usable run, checked with `find`. Without one the script exits 2, unless `--ignore-prerequisites` is given, which the manifest records.
- **`check_need`.** The machine diagnostics. It finds a tool, a package or an environment variable, and starts at most a 2-second `--version` probe.

The package depends only on the standard library and imports no other Vegeta package. A test checks that no module
except the version probe can start a process. Another test runs `find`, `require`, the CLI and `check_chapter` with
every process-starting function patched to fail.

## Removed (October 2026)

- **Removed from Yanus:**
  - notebook contracts (`flow.notebook`, `needs/takes/gives`, `File`, `Data`)
  - the entry and exit gates (`yanus.enter`, `GATE.leave()`, `GateClosed`)
  - pins (`notebooks/pins/`, `yn pin`) and stamps (`<file>.yanus.json`)
  - auto mode (`yanus_auto`, `YANUS_AUTO`, `YANUS_AUTO_MINUTES`) and `YANUS_STRICT_PINS`
  - the commands `yn status|why|check|stamp|auto|new notebook`
- **Removed from the notebooks:**
  - the gate cells in every notebook
  - `kestrel_campaigns.check_run/require/find_runs`, which duplicated `find`
- **Alabastron** keeps its own copy of the old Yanus and migrates separately.

Remaining to remove: nothing in Vegeta calls the old API. Alabastron does not import Vegeta's `vegeta-yanus` (it carries its own `alabastron.yanus`).
