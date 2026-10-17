# vegeta-yanus — the door keeper

Yanus guards the doors between Vegeta's notebooks and workflows. Every **unit** (a notebook, a workflow chapter)
declares once, in the project's registry `workflows/flow.py`, what it **needs** on the machine, what it **takes** from
other units and what it **gives**. Two gates check the declarations: one when the unit starts, one when it ends.
Yanus **guards**: it starts no solver, script or notebook; it prints the command that would, and an engineer types it.
The one exception is **auto mode** (§ below): a notebook flagged `auto=True` runs what it is missing by itself, in
screening quality, and never throws. Design: `docs/yanus.md`.

Stdlib only (pandas, when present, for the gate table). It imports no other vegeta package; a test enforces it.

## Declaring a unit (`workflows/flow.py`)

```python
from vegeta.yanus import File, Flow, Package, Run, Tool

flow = Flow("vegeta", root=ROOT)
kestrel = flow.book("kestrel", "K", variants=("4Q-1", "4Q-2", "4Q-3"),
                    fingerprint=lambda v: kestrel_campaigns.fingerprint(v),      # the design as it is now
                    compatible=lambda expected, found, chapter: ...)             # {"level", "reasons"} — the project's physics
kestrel.kind("fea", title="FEA …", script="workflows/kestrel/structure_fea.py", workflow="structure_fea",
             needs=[Tool("ccx", version_cmd=("ccx", "-v")), Package("gmsh")])
kestrel.chapter("frame", kind="fea", campaign="frame")                         # K9  (codes follow the order)
kestrel.chapter("joint", kind="fea", campaign="joint", takes=[Run("frame")])   # K10, after K9

flow.notebook("38_prop_lab", gives=[File("notebooks/designs/data/prop_lab.json", keys=("meta", "calibration"))])
flow.notebook("43_kestrel_study", needs=[Package("vegeta.boreas")],
              takes=[Run("kestrel/airframe-sweep", variant="4Q-2", required=True),
                     File("38_prop_lab:notebooks/designs/data/prop_lab.json")],
              gives=[File("notebooks/designs/data/reports/43_kestrel_study.json")],
              screening_env="KESTREL_SCREENING")
```

`yn check` validates the registry. It reports unknown units, cycles, a file taken that nobody gives, and pins to
takes that do not exist.

## The gates

```python
from vegeta import yanus
GATE = yanus.enter("43_kestrel_study")   # every need and take checked; GateClosed lists every fix when one required is missing
display(GATE.table())                    # input · role · selection · status · provenance · compatibility · usable · fix
GATE.inputs["kestrel/airframe-sweep@4Q-2"]   # the pinned run's summary.json (None = NOT RUN)
...
GATE.leave()                             # every promised file written by this execution → <file>.yanus.json stamp
```

- **Needs.** `Tool` (on the PATH, or a project `probe`), `Package` (found, not imported) and `Env`.
- **Run takes.**
  - A notebook reads the run **pinned** in `notebooks/pins/<notebook>.json`. Nothing is picked by itself.
  - A chapter's prerequisite takes the newest usable run.
  - A run is usable when it has a manifest and the right workflow and variant, is `completed` with a `summary.json`,
    and the book's `compatible` does not return `incompatible`.
- **File takes.** The file must exist and its stamp must name the producer. Its schema must be at least `min_schema`
  and the declared keys must be present. A file is **stale** when it changed after it was stamped, or when one of
  its stamped inputs was rewritten after it. An unstamped file is usable, and the gate says it is unstamped.
- **Screening** (`screening=True` or the notebook's `screening_env=1`) turns every required take optional.
- **DEBUG** (`enter(..., debug=DEBUG)`, or `VEGETA_DEBUG=1` as `vegeta.mock` reads it):
  - The numbers are mocked, so `leave()` stamps the outputs `debug` and only reports a missing one.
  - A gate that is not in DEBUG refuses a debug-stamped input.
  - `File(path, optional=True)` declares a give that a mode may skip without failing.
- **Auto mode** (`yanus_auto`). `flow.notebook(…, auto=True)`, or `enter(…, auto=True)`, or `YANUS_AUTO=1` from the
  shell (`YANUS_AUTO=0` switches a flagged notebook off): every missing take is **produced first** — the chapters whose
  runs it takes (their prerequisites first) with the kind's `screening` arguments (`--quality screening`, a coarse
  FEA mesh), a producer notebook headlessly with `nbconvert` — one after the other in the foreground, their output in
  the cell; then the gate opens with every take optional. It **never raises**: a chapter whose solver is not on the
  machine, or whose script failed, is NOT RUN with the reason in the table and in `gate.auto` (`yanus.auto.report`).
  `YANUS_AUTO_MINUTES=30` caps each chapter run. DEBUG runs nothing. `yn auto 43 --dry-run` prints the plan.
- **Scripts** call `yanus.check_chapter(...)`. It reports missing needs (the script may still `--prepare-only`) and
  exits 2 on a missing prerequisite unless the script passes `--ignore-prerequisites`.

## The command line (`yn`, or `vegeta yanus`)

```
yn status [unit|book] [--deep] [--json]   every chapter (latest run per variant) and notebook: ready / blocked, needs
yn why 43 [--deep]                        one unit's gate with every fix command
yn doctor                                 every need on this machine and which units it can run
yn pin 43 K1 --variant 4Q-2 <run dir>     pin the run a notebook reads   ·   yn pin 43 --suggest   ·   yn unpin 43 K1 --variant 4Q-2
yn check [--strict]                       registry, pins, stamps (CI)
yn stamp <unit> <file>                    stamp a given file made outside the gate
yn new notebook 45_name [--title …]       the common header and both gates, plus the registry entry
yn new chapter kestrel/thermal --kind fea [--new-kind] [--after frame]
yn auto 43 [--dry-run]                    the yanus_auto plan of a notebook (chapters in screening quality, producer notebooks); runs it without --dry-run
```

`--deep` builds the design fingerprints, which loads the CAD and takes a few seconds. Without it, `status` and `why`
check the run states only.

## Tests

`pytest` in this directory. The tests build a small flow with fake run directories in a temporary folder, so no
solver is needed. `workflows/tests/test_flow_registry.py` checks the real registry against a frozen copy of the
earlier `library` dicts.
