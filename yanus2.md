# Yanus 2: a small library for finding, reading and checking saved workflow runs


## Context
Yanus (`vegeta-yanus`, ~2000 lines) grew from a reader of saved results into an orchestrator. It now has notebook
contracts (needs/takes/gives), gates, pins, stamps, a recursive **auto mode** that launches chapters and
other notebooks, and a gate cell in every notebook. Only two notebooks actually read anything through it:
43 reads workflow runs and 45 reads 43's report. The other ~38 notebooks just run `yanus.enter(...)` and `GATE.leave()` as boilerplate.

Rule after this change: notebooks may run short calculations explicitly, including real solver calls. Expensive
campaigns (each campaign about 15–30 min in total) are `wf` workflows. Yanus only **finds, reads and checks** their saved runs. It never launches
a solver, a workflow or a notebook.

## The new API (`vegeta.yanus`)
```python
from vegeta import yanus
sel = yanus.find("K1", variant="4Q-2", fingerprint=FP["4Q-2"], quality="fine")   # newest usable run
sel = yanus.find("K1", variant="4Q-2", run="workflows/runs/kestrel/airframe-sweep/<id>")  # an explicit run
display(sel)            # chosen run, state, quality, compatibility, or why it was refused plus the wf command
sel.ok, sel.summary     # summary.json dict when usable, else None
sel.require()           # raises yanus.RunUnusable (reasons and command) when the notebook needs it
sel.provenance()        # dict for the report's meta (request, run dir/id, created, git, quality, compatibility)
yanus.table(sels)       # optional: one DataFrame of several selections
```
How a run is checked. Everything is read from the existing runkit files (`manifest.json`, `status.json`,
`summary.json`). No new format and no stamps:
1. The run directory exists, has a schema-1 manifest, and its workflow matches the chapter's kind.
2. `status.json` state is `completed` and `summary.json` is readable. Partial, failed or malformed runs are refused with the reason, and a dead `running` run counts as failed.
3. The variant matches `config.variant`.
4. Quality: if one was requested, `config.quality` must match (a string or a tuple of accepted values). A run that does not record its quality is refused when a quality was requested.
5. Mock/debug runs are always refused when `mimesis.is_mock(manifest)` is true or `config.quality` is `debug`/`mock`.
   runkit already writes that flag, both for runs in `runs-mock` and for resumed ones.
   Yanus reads the real runs folder: a `VEGETA_WORKFLOW_RUNS` ending in `-mock` makes `find` refuse everything with the
   message "mock run folder".
6. Compatibility uses the book's existing `compatible` hook (`kestrel_campaigns.compare`). `incompatible` → refused.
   `warnings` → accepted and labelled. If no fingerprint was given, or the book has no hook, the run is accepted and
   shown as **unverified**. It is never shown as compatible.
- Latest mode returns the newest run that passes, and lists why each newer run was skipped.
- Explicit `run=` checks only that directory. If it fails, the result is a clear diagnostic. It never falls back to another run.
- Unusable result → reasons plus the command that would produce it, e.g. `wf K1 --variant 4Q-2 --quality fine`.

## Package after the change
Keep:
- `contract.py`: `Flow`/`Book`/`Kind`/`Chapter`/`Import`/`Run` (prerequisites)/needs. `wf`, the GUI and `kestrel_campaigns` need these.
- `registry.py`
- `needs.py`: diagnostics only
- `runs.py`: runkit file reader, plus mock detection

Add:
- `select.py`: `find`, `Selection`, `RunUnusable`, `table` (about 150 lines)
- `check_chapter` rewritten on top of `find`, for the workflow scripts' prerequisite check

Delete:
- `gate.py`, `auto.py`, `pins.py`, `stamps.py`, the notebook scaffold
- `Notebook`, `File`, `Data`, the `takes`/`gives` of notebooks, `Book.fingerprint`
- `notebooks/pins/`
- `YANUS_AUTO`, `YANUS_AUTO_MINUTES`, `YANUS_STRICT_PINS`, `screening_env`

CLI `yn`:
- `find <chapter> [--variant] [--quality] [--run DIR]`: read-only, exit 0 or 2
- `doctor`: the chapters' needs
- `validate`: the registry
- `new chapter`: scaffold

## Callers and migration (audit done; each one is verified before moving on)
- `workflows/flow.py`: remove all `flow.notebook(...)` entries, the `auto`/`screening` arguments and `fingerprint=`.
- `workflows/library.py`:
  - `pin_commands` → `read_snippets(run_dir)`, which prints the `yanus.find(..., run=...)` line.
  - Update its callers: `cli.py cmd_notebook`, `gui/runner.py`, `_kestrel.pin_hint`.
  - `latest_completed`/`check_prerequisites` reuse `yanus.find`.
- `workflows/kestrel/_kestrel.py`: `check_chapter` keeps the same return shape, so the manifest's `yanus` entry, resume and the native artifacts stay unchanged.
- `notebooks/designs/kestrel_campaigns.py`:
  - `PrerequisiteError` subclasses `RunUnusable`.
  - Delete `check_run`/`require`/`_load`, which duplicate `find`. Their tests move to the Yanus tests.
- `notebooks/designs/kestrel_study.py`:
  - Delete `runs_from_gate`.
  - `prerequisites(selections, required)` builds the one table: prop_lab, drive tables and sections rows, plus one row per selection. It raises a list of all missing required runs with their commands.
- **43_kestrel_study**:
  - The setup cell declares `SELECT = {("K1","4Q-2"): <legacy path from the old pin>, ("K1","4Q-1"): <legacy path>, others: None}` (None = latest).
  - Required: K1 and K11 for 4Q-2, except with `KESTREL_SCREENING=1`.
  - Call `yanus.find` for each entry, show the table, then `require()` the required ones.
  - Drop the gate, the auto text and the duplicate run listing. Rewrite the markdown cell.
  - Keep the physics checks and figures. `kr.export` meta gets `"runs": {key: sel.provenance()}`.
- **45_kestrel_x**: read 43's report file directly. Keep the existing `kx.warn_missing` fallback, which is the labelled analytical estimate. Drop the gate.
- **38, 44 and the ~38 boilerplate notebooks (08–37, demo/06, 07, 10, 15)**:
  - Delete the `yanus.enter`/`GATE.leave()` cells. File hand-offs between them were always optional and each notebook already reads its own files.
  - Read each notebook first and check that nothing else uses `GATE`/`INPUTS`/`STAMPS`.
- Other uses:
  - `activate.sh` and `test.sh`: update the `yn` line.
  - `install_local.sh`: no change.
  - `scripts/test_all.sh`: unchanged.
- No compatibility adapters are planned, because no remaining Vegeta caller needs one. Alabaster stays on its own copy and will migrate separately; this is noted in the docs.
- Track anything left to remove in a "Remaining" section of `docs/yanus.md`.

## Docs
- Rewrite `docs/yanus.md` and `vegeta-yanus/README.md` around `find` and `Selection`.
- Delete `docs/yanus_plan.md`.
- Fix the yanus/auto/pin mentions in `README.md`, `workflows/README.md`, `todo.md`, `manual_workflows.md` and `docs/philosophy.md`.

## Verification
- Rewrite `vegeta-yanus/tests` to cover:
  - newest compatible run chosen, newer incompatible run skipped with the reason
  - wrong quality, missing quality
  - failed, partial, running-dead, missing summary and malformed JSON runs
  - schema mismatch
  - explicit good path, explicit bad path with a diagnostic and no fallback
  - mock manifest refused
  - no fingerprint → unverified
  - `require()` message contains the `wf` command
  - **no-computation test**: patch `subprocess.*`, `os.system` and `os.exec*` to raise, then run `find`/`yn find`
- `workflows/tests`: update `test_flow_registry` and `test_kestrel_workflows` (snippets, `check_prerequisites`).
- Notebooks: validate the JSON with nbformat, grep that no `GATE`/`yanus.enter` remains, compile every code cell.
  Do not execute campaigns.
- Known limits:
  - Talos, Aeromant, Kestrel and the workflow tests import and run here.
  - Mocked chapters (`VEGETA_DEBUG=1`) can produce real-shaped mock runs. Use them to show that `find` refuses them,
    and copy one with `"mock"` flipped to false as a fixture for the selection path.
  - Real CalculiX/OpenFOAM campaigns are not run.
  - Notebooks 43/45:
    - Run their selection cells headless in a scratch copy with `VEGETA_DEBUG=1`. Expect no usable runs → the screening path, the labelled analytical estimate and the printed `wf` commands.
    - Full 43 execution may be too heavy; report it if so.
  - Other notebooks are checked statically.
