# Assemblies — proven product code, notebook-free

A notebook is where something is tried. `assemblies/` is where what was proven is kept as plain Python, run without
Jupyter, and saved as `.vida` files that other workflows read instead of computing again. Code is promoted by copying
it here; duplication with the notebooks is accepted. The copy here is the one that is tested and run unattended.

```
assemblies/
  vida.py          Assembly (the tree node) and the .vida file: save / load / reuse
  _cli.py          the command line every workflow shares
  components/      proven parts with their geometry, models and CFD/FEA case builders
  workflows/       one module per product: builds the tree, runs the long calculations, saves, exports
  data/            what other workflows and notebooks read: <product>.vida (results level) and JSON exports
  tests/
runs/assemblies/<product>/   solver cases, meshes, STEP/STL (gitignored, readable names)
```

## Rules
- **Imports.** Components import the Vegeta tools and other components. Workflows import components, other
  workflows and `vida`. Nothing in `assemblies/` imports from `notebooks/`; a notebook may import anything from here.
- **Explicit.** A workflow's `run()` reads top to bottom: build the tree, reuse what the saved `.vida` has, compute
  what is missing, save, export. A workflow that needs another product's result loads that product's `.vida` from
  `data/` (or calls its workflow) by name, visibly. Nothing runs because something else changed.
- **Reuse, one rule.** A node's `key` is the hash of its kind, its parameters and its sub-assemblies' keys. A node whose
  key equals the saved one keeps its results. A solver case whose inputs are unchanged is read back from its directory
  (`CFDCase.ensure`, `StructuralModel.ensure`). When code changed but parameters did not, say so explicitly:
  `--redo <node>` for one sub-assembly, `--force` for everything.
- **Fidelity** (`smoke | quick | full`) is a parameter of every node that runs a solver: mesh levels, iterations,
  how many points. It is part of the key, so smoke results never stand in for full ones.
- **NOT RUN is shown, never hidden.** A node without results says why (`--no-cfd`, a failed case). Exports carry a
  `source` block (workflow, fidelity, git commit, what was complete).
- **Proven** means: the tests here pass, `run()` completes with the solvers off at `smoke`, and a full run was done on a
  workstation with the results-level `.vida` committed in `data/`.

## The `.vida` file
A zip archive (`unzip -l` shows it): `manifest.json` (format version, git commit, versions, include level), `tree.json`
(every node: name, kind, params, key, meta, files listed with path and SHA-256), and per node `results.json` +
`arrays.npz` (numpy, no pickle) and the included files. `include` chooses the weight:

| include | adds | typical size | use |
|---|---|---|---|
| `results` (default) | parameters, tables, maps, metrics | kB–MB | commit to `data/`, hand to other workflows |
| `geometry` | STEP, STL | MB | someone needs the parts |
| `mesh` | meshes, FEA results (`.msh`, `.frd`) | 10–100s of MB | inspect the FEA elsewhere |
| `cases` | whole solver case directories | GB | move a product to another machine |

Files of the levels left out are listed with their path and hash; `node.file(name)` returns the file when it is still
on disk unchanged, or extracts it from the archive, or says it is not there.

```python
from assemblies import vida
M = vida.load("assemblies/data/microjet.vida")
print(M)                                   # the tree with computed / reused / NOT RUN per node
M.table()                                  # the same as a DataFrame
M.child("compressor").results["speedline"]
plane = vida.Assembly("my_plane", "aircraft", params={...})
plane.add(M.copy("engine"))                # a saved product as a sub-assembly
plane.save("runs/my_plane.vida", include="geometry")
```

## Workflows
```
python -m assemblies.workflows.microjet --fidelity quick -j 4        # compute what is missing, save, export
python -m assemblies.workflows.microjet --no-cfd --no-fea            # without solvers: those nodes are NOT RUN
python -m assemblies.workflows.microjet --redo compressor            # this sub-assembly again
python -m assemblies.workflows.microjet --show                       # the saved tree
```
Options for all: `--fidelity`, `-j` (cores per CFD solver), `--jobs` (CFD cases side by side), `--threads` (CalculiX),
`--no-cfd`, `--no-fea`, `--redo PATH`, `--force`, `--out`, `--vida`, `--include`, `--no-export`, `--show`.
Interrupted runs resume: solved cases are read back.

| workflow | from | tree | writes |
|---|---|---|---|
| `microjet` | notebook 28 | microjet → parts, compressor (CFD speed line), wheels (FEA) | `data/microjet.vida`, `data/microjet.json` (read by notebook 29) |
| `aguya` | notebook 29 (not its race against MERLIN) | aguya → engine (= `microjet.vida`, grafted), airframe (sizing, missions), jet_cfd (CFD at the dash), wing (gust FEA, modes) | `data/aguya.vida`, `data/aguya.json` |

`aguya` does not compute the engine: it loads `data/microjet.vida` (or `--engine PATH`) as its `engine`
sub-assembly and stops with the command to run when it is missing. A new engine changes the engine node's key and
so the keys of the nodes built on it; nodes whose own parameters did not change are still reused.

The `.vida` and JSON files committed in `data/` were written with the solvers off (`--no-cfd --no-fea`, fidelity
`full`): the cycle, the parts, AGUYA's sizing and missions are computed, the CFD and FEA nodes say NOT RUN. A full run
on a workstation fills them in and is committed in their place. Full fidelity needs a workstation: AGUYA's wing at
4 mm elements takes more than 10 GB of memory in CalculiX, the microjet wheels at 1 mm more still.

## Components
| component | from | gives |
|---|---|---|
| `turbojet.py` | `notebooks/designs/turbojet.py` | the impeller, turbine wheel and engine CAD; the compressor passage STLs; sizing; masses |
| `turbojet_parts.py` | notebook 28 §4 | the three STEP files with masses and sizes (kept when the parameters are unchanged) |
| `cycle.py` | notebook 28 §1–3, 5, 7 | the catalogue engine, calibration from a speed line, design point, map, export |
| `impeller.py` | notebook 28 §5 | the compressor's `compressor_mrf` cases per fidelity, the speed line |
| `wheels.py` | notebook 28 §6 | the turbine and impeller FEA models (spin, hot, overspeed), their results |
| `fixed_wing.py`, `aguya.py`, `aguya_flight.py` | `notebooks/designs/` | the wing sections, AGUYA's CAD and CFD surfaces, the jet unit, airframe, missions with fuel burn, tank sizing |
| `aguya_jet.py` | notebook 29 §7 | the dash point, the free-jet estimate, the `jet_external` case and its results |
| `aguya_wing.py` | notebook 29 §9 | Pratt's gust numbers, the wing's gust and 6 g FEA, the vibration modes |

## Tests
`python -m pytest assemblies/tests -m "not slow"` (about two minutes: CAD and the workflows with the CFD batches
replaced by known results); `-m slow` runs the wheels' and the wing's FEA for real (CalculiX, a few minutes). Also in `scripts/test_all.sh` and
`./user_tests.sh microjet`.
