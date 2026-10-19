# Yanus — implementation plan (v1, Kestrel first) — for approval

Design: `docs/yanus.md`. Base: `dev_yanus` with `dev_workflows` merged (the PyQt6 GUI, tqdm progress, per-variant
run states in `library.chapter_status`). Nothing below is implemented yet.

**Assumed answers to the design's open questions** (change any before approval and the plan follows):
Q1 schema integer + required top-level keys · Q2 pins committed, a dangling pin = NOT RUN · Q3 stale: required →
gate closed, optional → NOT RUN "stale" · Q4 a missing promised output fails the notebook run · Q5 registry in
`side_labs/flow.py` · Q6 `vegeta yanus …` plus `yn` (a script like `wf`).

**Hard constraints the plan keeps**

* Guard only: no code path in Yanus starts a solver, a script or a notebook (a test greps for `subprocess`/`Popen`
  in the package; only `doctor`'s version probes may run `<tool> -v`/`--version`, with a timeout).
* `vegeta-yanus` imports no other `vegeta.*` package and nothing from `side_labs/` or `designs/`
  (an independence test, as in `vegeta-ai/tests/test_independence.py`). Stdlib only; pandas optional for the table.
* Existing callers keep working unchanged: `wf`, the GUI (`library.BOOKS[..]["chapters"][..]["campaign"]`,
  `library.KINDS[kind]["needs"|"title"|"time"|"script"]`, `chapters`, `by_code`, `chapter_status`,
  `check_prerequisites`, `all_runs`), the two Kestrel scripts, `kestrel_campaigns.check_run/require`.
* Python configuration, file-based state, no YAML, no database.

---

## Phase 0 — groundwork (½ day)

1. `vegeta-yanus/` skeleton following `vegeta-ai`: `pyproject.toml` (setuptools>=64, `package-dir={""="src"}`,
   `packages=["vegeta.yanus"]`, `Private :: Do Not Upload`, `requires-python>=3.10`, version 0.1.0, extras
   `test=["pytest"]`, `table=["pandas"]`, entry point `[project.entry-points."vegeta.commands"] yanus =
   "vegeta.yanus.cli:main"`, pytest `--import-mode=importlib`), `README.md`, `src/vegeta/yanus/__init__.py`, `tests/`.
2. Wire it in: `install_local.sh` (`-e "$ROOT/vegeta-yanus[test]"`), `activate.sh` (src on PYTHONPATH,
   `VEGETA_YANUS_FLOW=$ROOT/side_labs/flow.py`, `yn` alias next to `wf`), `test.sh` (`optional` row like ai/fidia),
   `scripts/test_all.sh` (its tests).
3. `tests/test_independence.py`: importing `vegeta.yanus` pulls in no `vegeta.<other>` module; no `subprocess`
   use outside `needs.py`.

*Done when* `pip install -e vegeta-yanus[test]` + `pytest vegeta-yanus` pass (one trivial test) and
`vegeta yanus --help` prints.

## Phase 1 — the contract model (1 day)

`src/vegeta/yanus/contract.py` — plain dataclasses, no I/O:

| Class | Fields | Notes |
|---|---|---|
| `Need` (base) → `Tool`, `Package`, `Env` | `name`, `version_cmd`/`min_version`, `why` | `Tool.openfoam()` factory: `foamVersion` / `$WM_PROJECT_VERSION`, or a probe callable from the project |
| `Take` (base) → `Run`, `File` | `Run(chapter, variant=None, required=False, states=("completed",))`; `File(ref, min_schema=None, keys=(), required=False)` | `ref` = `"<producer unit>:<path>"` |
| `Give` → `File` | `path`, `schema`, `keys` | the same `File` used on the giving side |
| `Unit` → `Notebook`, `Chapter` | `name`, `needs`, `takes`, `gives`, `screening_env` (notebook), `kind/campaign/script/workflow` (chapter) | |
| `Kind` | `name`, `title`, `script`, `workflow`, `needs`, `time`, `options` | becomes `library.KINDS` |
| `Book` | `name`, `code`, `title`, `design`, `variants`, `compatible` (callable `(expected, found) -> {"level","reasons"}`), `fingerprint` (callable `variant -> dict`), chapters in order | codes `K1…` derived from order |
| `Flow` | books, notebooks, imports; `unit(code_or_name)`, `book.chapter(...)`, `notebook(...)`, `validate()` | |

`Flow.validate()` (run at load, and as `yanus check`): unknown chapter/unit names in `takes`, prerequisites outside
the book, cycles (an error — Yanus refuses a cycle rather than becoming a DAG engine; it only needs "is this
input there"), a `File` take whose producer does not declare that `gives`, duplicate names/codes.

`src/vegeta/yanus/registry.py` — `load_flow(path=None)`: `path` or `$VEGETA_YANUS_FLOW`, else walk up from cwd for
`side_labs/flow.py`; imports it by file path (like `designs/side_labs_runkit.py` does) and returns its
`flow`.

*Tests*: building a small flow; codes; every `validate()` error with its message; loading by path and by env var.

## Phase 2 — the Kestrel registry, one source of truth (1 day)

1. `side_labs/flow.py`: the `kestrel` book with K1–K11 exactly as `library.BOOKS` lists them today (same order,
   campaigns, prerequisites as `Run(...)` takes), the two kinds with their needs (OpenFOAM + `vegeta.aeromant`;
   `ccx`, `gmsh`, `vegeta.talos`), `compatible`/`fingerprint` delegating lazily to `kestrel_campaigns.compare` /
   `fingerprint` (lazy import keeps `wf list` fast); `IMPORTS["import_legacy"]`; notebooks 38, 43, 44 with their
   contracts (43's takes = today's `RUNS`/`REQUIRED` + `prop_lab.json`, drive tables and section polars as optional
   `File` takes; 38 gives `prop_lab.json`; 43 gives `data/reports/43_kestrel_study.json` and `kestrel_dossier.json`).
2. `side_labs/library.py`: `KINDS`, `BOOKS` and `IMPORTS` become **generated** from `flow` in the same dict shape
   (so the GUI and `cli.py` keep working); `chapters`, `by_code`, `chapter_of` read the flow. Everything else in
   `library.py` (runs, `chapter_status`, `tree`) stays.
3. `designs/kestrel_campaigns.py:41-48`: `CHAPTERS`, `SCRIPTS`, `PYTHON` derived from the flow (loaded by
   path, like `side_labs_runkit.py`) instead of a second copy.
4. Fix the stale docstrings found in review (`library.py` header numbering K8/K9; `airframe_cfd.py:8`,
   `structure_fea.py:8` run paths).

*Tests*: a regression test that the generated `library.BOOKS`/`KINDS`/`chapters("kestrel")` equal a frozen copy of
today's values (captured before the change); existing `side_labs/tests/*` (cli, runkit, kestrel workflows, gui)
pass unchanged.

## Phase 3 — reading the world (1½ days)

* `needs.py` — `check_need(need) -> NeedResult(ok, version, detail, fix)`: `shutil.which` + version probe with a
  2 s timeout (the only subprocess use); `importlib.util.find_spec` + `importlib.metadata.version` for packages
  (no import of heavy modules); env vars. Results cached per process. Fix hints: `install_local.sh`, `apt …`,
  `source activate.sh`.
* `runs.py` — the runkit **file contract** only (schema 1): `read_run(dir) -> RunView(manifest, status, summary,
  partial, state)` with runkit's `running`-but-dead-pid → `failed` rule (as `chapter_status` does);
  `runs_of(runs_root, book, chapter, variant)`; `latest_compatible(...)` for suggestions. Fails clearly on
  `schema != 1`.
* `stamps.py` — sidecar `<file>.yanus.json`: `{yanus: 1, schema, producer, kind, written_at, sha256, size,
  design_fingerprint, git: {commit, dirty}, needs: {name: version}, inputs: [{take, pin, sha256|run_id}]}`;
  `write_stamp`, `read_stamp`, `verify(file, stamp)` (hash), `is_stale(stamp, current_fingerprint, compatible)`.
  JSON exports may also carry `"schema"` inside; the sidecar wins when both exist.
* `pins.py` — `notebooks/pins/<unit>.json` `{"<take key>": "<path or run dir>"}` with take key
  `kestrel/airframe-sweep@4Q-2` or `38_prop_lab:designs/data/prop_lab.json`; `get/set/clear`, relative paths
  stored from the repo root; atomic writes.

*Tests*: fake run directories built in `tmp_path` (completed / running with a dead pid / partial / failed /
wrong schema / wrong workflow); stamps round-trip, hash mismatch, staleness via a fake `compatible`; pins
round-trip; needs with a fake tool on PATH and a missing one.

## Phase 4 — the gates (1½ days)

`gate.py`:

* `enter(unit, *, flow=None, screening=None, variant=None) -> Gate`. Checks **all** needs and takes, never stops
  at the first failure, then:
  * builds `GateReport` rows `{input, role (required/optional/need), selection (pin), status, provenance,
    compatibility, usable, fix}` — the same columns as today's `kestrel_study.prerequisites` table;
  * screening (`screening=True` or the unit's `screening_env` set to `1`) turns required takes into optional
    ones and says so in the report;
  * on any required failure raises `GateClosed` (a `RuntimeError`; `kestrel_campaigns.PrerequisiteError` becomes
    its subclass so existing `except` clauses keep working) listing every failure with its fix command: `wf K1
    --variant 4Q-2` (from the flow), `wf import_legacy` for a legacy pin that is missing, `jupyter nbconvert
    --execute 38_prop_lab.ipynb` for a file take, `yanus pin 43 K1 --variant 4Q-2 <run>` when a compatible run
    exists but is not pinned;
  * returns `Gate(report, inputs, started_at)`; `gate.inputs[take_key]` = the run's summary (`dict`) or the file
    `Path`; `gate.table()` → pandas DataFrame when pandas is there, else text; `_repr_html_` for notebooks.
* `Gate.leave()` — each `gives` file exists, was modified after `started_at`, has the declared `keys`/`schema`;
  writes its stamp (with the pins and needs used); raises `GateClosed` listing every missing/old output.
* `check_chapter(book, chapter, *, variant, expected, ignore=False)` — the script-side entry gate (needs +
  prerequisite runs): returns the ignored list for the manifest, or exits 2 with the report, exactly like
  `prerequisites_or_exit` today.

*Tests*: every row status on the fake runs/files; all failures listed at once; screening; the subclass
relationship; `leave()` fails for a missing and for a stale (pre-`enter`) output and stamps correctly.

## Phase 5 — the command line (1 day)

`cli.py` (`vegeta yanus …` and `yn`):

| Command | Does |
|---|---|
| `status [--json] [book\|unit]` | every chapter (per variant, using `chapter_status` marks) and notebook: ✔ ready / ✘ blocked (first reason) / · not run, plus a needs column |
| `why <unit> [--variant]` | the full gate report for one unit with every fix command (the same table `enter` shows) |
| `doctor [--json]` | the needs of every unit against this machine: which chapters/notebooks this machine can run |
| `pin <unit> <take> [--variant] <path>` / `pin <unit> --suggest` / `unpin` | write/clear pins; `--suggest` prints the newest compatible runs, never writes them |
| `check` | `Flow.validate()` + every stamp's hash + dangling pins (for CI) |
| `stamp <unit> <file>` | stamp an export made outside a notebook (rare; e.g. a hand-imported file) |
| `new notebook <name> [--book]` / `new chapter <book>/<name> --kind cfd\|fea` | Phase 6 |

`side_labs/cli.py`: `wf notebook <run>` prints the `yanus pin …` command (and still the old `RUNS` line for one
release). `wf run` calls the chapter gate before launching, so a missing tool fails at once and not in the
detached run.

*Tests*: each command on a fake flow + fake runs; `--json` shapes; exit codes (0 ok, 2 blocked).

## Phase 6 — the scaffolds (1 day)

`scaffold.py` + `templates/`:

* `new notebook 45_name --book kestrel`: writes `notebooks/45_name.ipynb` (nbformat JSON written directly, no
  nbformat dependency) with the header the notebooks share today — logo cell, title, DEBUG/`mock.setup` cell,
  imports with `sys.path.insert(0, "designs")`, `OUT = Path("output/45_name")`, the `gate = yanus.enter(...)`
  cell with `display(gate.table())`, empty numbered sections, the final `gate.leave()` cell — and a
  `flow.notebook("45_name", needs=[], takes=[], gives=[])` stub appended to `side_labs/flow.py`. Refuses to
  overwrite.
* `new chapter kestrel/thermal --kind fea`: if the kind exists, only the registry entry (the kind's script takes
  `--chapter`); with `--new-kind`, a runkit script skeleton under `side_labs/<book>/` modelled on
  `structure_fea.py` (argparse with `runkit.add_common_arguments`, `Run.create`/`Run.open`, the chapter gate,
  `_kestrel.drive` with `Progress`, `closing`) plus the kind and chapter entries.

*Tests*: generated notebook is valid JSON that `nbformat` (if installed) validates; the generated script passes
`python -m py_compile` and `--help`; the flow still validates after the stub; refusing to overwrite.

## Phase 7 — migrate Kestrel onto the gates (1½ days)

1. `side_labs/kestrel/_kestrel.py`: `prerequisites_or_exit` → `yanus.check_chapter` (same signature/exit code,
   needs added); both scripts unchanged otherwise. Manifest gains `yanus: {needs: {...}}` alongside
   `tool_versions`.
2. Notebook **43**: the prerequisites cell becomes `gate = yanus.enter("43_kestrel_study")`; `RUNS` moves to
   `notebooks/pins/43_kestrel_study.json` with today's two legacy pins; `kestrel_study.prerequisites` becomes a
   thin adapter that builds today's `(table, summaries)` from `gate` so §§ 2–N of the notebook do not change;
   `KESTREL_SCREENING=1` behaves as now; last cell `gate.leave()`.
3. Notebook **38**: last cell `gate.leave()` stamps `prop_lab.json` (its `GATES` acceptance stays the notebook's
   own physics check — Yanus checks presence/schema/staleness, not physics).
4. Notebook **44**: `enter`/`leave` with no takes (proves the empty case and scaffold header).
5. GUI (small): the chapter panel shows the needs check (✔/✘ per tool) from `yanus doctor`, and the Run button's
   prerequisite message uses the gate report text. No new GUI features beyond that.
6. Docs: `docs/yanus.md` updated to "implemented", `vegeta-yanus/README.md` (how to declare a unit, the gates,
   pins, CLI), `side_labs/README.md` + `manual_workflows.md` (pins replace `RUNS`; `yn status`), `README.md`
   package list, `todo.md` entries for the roll-out (other notebooks, `designs/data/*.json`).

*Verification*: workflow tests + `vegeta-yanus` tests green; notebook 43 executed in screening mode via
`scripts/run_notebooks.sh 43_kestrel_study` (the legacy import run first) gives the same prerequisite table and the
same figures/reports as before (diff of `data/reports/43_kestrel_study.json` ignoring timestamps); 38 and 44
executed; `yn status` / `yn why 43` / `yn doctor` output pasted into the PR description.

---

## Commits and order

One commit (or a few) per phase on `dev_yanus`, each with its tests green, pushed after each phase:
0 skeleton → 1 contract → 2 registry (+ regression test) → 3 readers → 4 gates → 5 CLI → 6 scaffolds →
7 migration + docs. Phases 0–5 change no behaviour of existing code except Phase 2's (tested-equal) generation of
`library` dicts; Phase 7 is the only one that changes notebooks.

**Estimate**: ~9 working days of focused work; Phases 0–4 (the core) ~5.

## Risks and how they are handled

| Risk | Handling |
|---|---|
| The GUI/CLI depend on `library` dict shapes | generated in the same shape; frozen-equality regression test before any other change |
| `kestrel_campaigns` imports are heavy (CAD) and would slow `yn status` | `compatible`/`fingerprint` imported lazily, only when a run's compatibility is actually checked; `status` without `--deep` checks state only |
| Notebook 43 output drifts after migration | adapter keeps `(table, summaries)` identical; before/after report diff |
| No `.venv`/pytest in this cloud container | install the editable packages into a venv here for the tests; solver-dependent tests stay behind the existing `requires_*` markers; notebook 43 runs in screening mode (no OpenFOAM/ccx needed) |
| Scope creep into a runner | the no-subprocess test; `pin --suggest` never writes; no `--with-prereqs` flag |

## Not in v1

The other 45 notebooks and the `designs/data/*.json` links (roll-out tracked in `todo.md`); header drift check;
`notebooks/old/`; field-level schema validation; any execution of prerequisites.
