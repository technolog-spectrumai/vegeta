# Vegeta Core — workspaces, revisions, evaluations

`vegeta.core` (distribution `vegeta-core`) organises the engineer's work on top of the four tools.
It stores everything as plain files in a workspace directory — no database, no server — and it never
runs anything by itself: geometry is generated and each analysis is run only when you call it.
Analyses that were not run on a revision are shown as **NOT RUN**.

## Concepts
| concept | what it is |
|---------|------------|
| workspace | a directory (`workspace.json`, `designs/`, `revisions/`) |
| design | a Dedalus design registered by spec: `vegeta.dedalus.examples:Bracket` or `path/file.py:Plate` |
| revision | one immutable design state: full parameters, SHA-256 + snapshot of the design source, note, environment |
| branch | a new revision from an existing one with some parameters changed (parent is recorded) |
| geometry | STEP + STL made once by `generate()` into the revision |
| evaluation | one named analysis (FEA, CFD, print) on one revision's geometry, recorded write-once |
| label / note | human decisions (`preferred`, `rejected`, `reference`, `unclassified`) kept as history |

## Python
```python
from vegeta import core, talos, mellonia
from vegeta.mellonia.examples import GENERIC_PLA_0_2MM

ws = core.Workspace.create("runs/bracket_study")          # later: core.Workspace.open(...)
bracket = ws.add_design("bracket", "vegeta.dedalus.examples:Bracket")

r1 = bracket.new_revision(thickness=6.0, note="baseline")  # records only; nothing runs
r1.generate()                                              # Dedalus: r1/geometry/Bracket.step + .stl

def static(rev):                                           # analysis factory: explicit Python
    return talos.StructuralModel(
        rev.step, "mm-N-MPa", talos.Material("Al 6061-T6", 68900, 0.33, yield_strength=276),
        regions=[talos.SurfacesOnPlane("clamped", "x", -40.0), talos.SurfacesOnPlane("loaded", "x", 40.0)],
        supports=[talos.FixedSupport("clamped")], loads=[talos.Force("loaded", fz=-200.0)],
        mesh_settings=talos.MeshSettings(3.0))

r1.run_fea("static", static)                               # Talos on r1's STEP, recorded
r2 = r1.branch(thickness=8.0, note="stiffer")              # new revision; nothing generated or run
r2.label("preferred", note="candidate, needs FEA")
r2.generate()
r2.run_fea("static", static)
r2.run_print("flat", GENERIC_PLA_0_2MM, mellonia.Orientation())
ws.status()
```
Real output (from `notebooks/06_core_revisions.ipynb`):
```
rev  design   parent  label         changes        CAD          fea:static               print:flat
---  -------  ------  ------------  -------------  -----------  -----------------------  -----------------
r1   bracket  -       unclassified  -              V=1.832e+04  u=0.765 vM=130 SF=2.13   NOT RUN
r2   bracket  r1      preferred     thickness=8.0  V=2.443e+04  u=0.329 vM=75.7 SF=3.65  1h 29m 46s 13.8 g
```
(`changes` is relative to the parent, or to the design defaults for a first revision; 6 mm is the default.)

CFD: `rev.run_cfd(name, case_factory, steps=None)` where `case_factory(rev, workdir)` returns an
`aeromant.CFDCase` on `rev.stl` in `workdir`.

## Rules the core enforces
- **Immutability.** `revision.json` and `evaluation.json` are created write-once. Re-using an evaluation
  name fails; changed parameters or source mean a new revision.
- **Source identity.** A revision stores the SHA-256 of the design's build code and a snapshot of the source
  file. If the design file changed since, `generate()` refuses and asks for a new revision (`branch()`).
- **Provenance.** Analysis factories must use the revision's own STEP/STL. Each evaluation stores the
  tool configuration, the factory's source text, tool versions, every command run, timestamps and the
  environment; the tool's native files stay in `revisions/<rid>/evaluations/<kind>-<name>/`.
- **No automatic chaining.** `new_revision`/`branch` generate nothing, `generate` runs no analysis,
  `run_*` before `generate` is refused. Failed attempts are recorded as FAILED.

## Command line
`vegeta ws` and `vegeta rev` are added to the `vegeta` command when `vegeta-core` is installed.
```
vegeta ws -w runs/study init
vegeta ws -w runs/study add-design bracket vegeta.dedalus.examples:Bracket
vegeta rev -w runs/study new bracket -p thickness=6 --note baseline
vegeta rev -w runs/study generate r1
vegeta rev -w runs/study fea r1 --model analyses.py:static --name static
vegeta rev -w runs/study print r1 --settings vegeta.mellonia.examples:GENERIC_PLA_0_2MM --name flat
vegeta rev -w runs/study branch r1 -p thickness=8 --note stiffer
vegeta rev -w runs/study label r2 preferred --note "SF > 2 needed"
vegeta ws -w runs/study status
vegeta rev -w runs/study show r2
```

## Workspace layout
```
workspace.json
designs/<name>/design.json
revisions/<rid>/revision.json            write-once
revisions/<rid>/source/<file>.py         source snapshot
revisions/<rid>/geometry/                STEP, STL, summary.json
revisions/<rid>/evaluations/<kind>-<name>/evaluation.json + native tool files
revisions/<rid>/annotations.jsonl        labels, notes, generate attempts (append-only)
```
