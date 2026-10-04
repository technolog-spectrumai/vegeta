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
| `microjet` | notebook 28 | microjet → parts, compressor (CFD speed line), wheels (FEA) | `microjet.vida`, `microjet.json` (read by notebook 29) |
| `aguya` | notebook 29 | aguya → engine (= `microjet.vida`, grafted), airframe (sizing, missions), jet_cfd, wing (gust FEA, modes), merlin_race (from `merlin.vida` and `propulsors.vida`) | `aguya.vida`, `aguya.json` |
| `quadcopter` | notebook 08 | quadcopter → frame (CAD, masses), frame_fea, canopy_cfd, propeller (5x4.3 tri-blade, BEMT points), rotor_cfd, blade_fea, life → unit_fea (point masses, 8 modes, unit cases), fatigue (missions, damage, life) | `quadcopter.vida`, `quadcopter.json` (boreas' format, as 08's `quad_5x43.json`), `quadcopter_life.json` (08's `life.json`) |
| `fixed_wing` | notebooks 09a, 09b | fixed_wing → airframe (planform, polar), propeller (9x6), wing_fea, aero_cfd (4°), rotor_cfd, blade_fea, installed_cfd (rotor disks), wing_life → unit_fea (NACA 2415), unit_fea_naca2412, fatigue; fuselage_life → unit_fea, fatigue | `fixed_wing.vida`, `fixed_wing.json` (the keys 09b reads), `fixed_wing_propulsion.json`, `fixed_wing_life.json` (09b's `life.json` + `fuselage_life.json`) |
| `propulsors` | notebooks 25, 25b, 25c (last cells) | propulsors → propellers, exotic, edf: the libraries over airspeed × rpm | `propulsors.vida`; at `full` also `*_maps.json` (only a full run writes them) |
| `merlin` | notebooks 26, 26b, 27 | merlin → airframe (3 propulsions), wing_fea, polar_cfd (3 aircraft), propulsors (grafted), race, mission | `merlin.vida`, `merlin_design.json` (kept, not replaced, by a run that solved neither the wing FEA nor the polar CFD) |
| `boat` | notebook 12 | boat → hull (CAD, masses, hydrostatics), propeller (60 mm marine), hull_cfd (double body), hull_fea, bracket_fea, rotor_cfd, blade_fea, scene_cfd | `boat.vida`, `boat.json` |
| `submarine` | notebooks 13, 14 | submarine → hull (buoyancy, trim), propeller (120 mm), hull_cfd, pressure_hull_fea, rotor_cfd, blade_fea, scene_cfd | `submarine.vida`, `submarine.json` |
| `rover` | notebook 11 | rover → parts (CAD), terrains (quarter car, ISO 8608 roads with rocks), arm_fea, chassis_fea; pins and wheel loads on the root | `rover.vida` |
| `onager` | notebooks 20–23 | onager → sentinel, atlas, manus, sweeper; each → body (mass budget, CAD drift), scene (MuJoCo episode); sentinel also → leg_fea (walk, braking, drop, cornering) | `onager.vida` |
| `walkers` | notebooks 16–19 | walkers → dog (body, gaits), cleopatra (body), persephone (CAD), apheloria (body, pack, unpack in MuJoCo) | `walkers.vida` |

All files go to `data/`. A workflow that needs another product's result loads its `.vida` and stops with the command
to run when it is missing: `aguya` grafts `microjet.vida` (or `--engine PATH`) as its `engine`; `merlin` grafts
`propulsors.vida`; `aguya`'s race reads both `merlin.vida` and `propulsors.vida`. A new engine changes the engine
node's key and so the keys of the nodes built on it; nodes whose own parameters did not change are still reused.
`--show` prints a saved tree. The ground workflows also take `--no-sim` (no MuJoCo scenes) and `--variant`/`--machine`.

The `.vida` and JSON files committed in `data/` were written with the solvers off (`--no-cfd --no-fea`, `--no-sim`,
fidelity `full`): CAD, cycles, masses, sizing, hydrostatics, the quarter car, the propulsor libraries, the races and
missions are computed; the CFD, FEA and MuJoCo nodes say NOT RUN. A full run on a workstation fills them in and is
committed in their place. Full fidelity needs a workstation: AGUYA's wing at 4 mm elements takes more than 10 GB of
memory in CalculiX, the microjet wheels at 1 mm more still.

The life parts are two nodes each: `unit_fea` (the point masses, the modal solve and the unit load cases on one mesh,
`_common.unit_fea`, behind `--no-fea`) and `fatigue` (missions → spectra → `talos.assess_fatigue` → `chronos.simulate_life`,
post-processing only, NOT RUN until its unit cases are solved). 09b's wing life is on 09a's preferred NACA 2415 wing
(compared with the NACA 2412), whatever the flown aircraft's default.

What stayed in the notebooks: life and fatigue of the boat and the submarine (12 §5–6, 14 §10–11), the leg torque tables of 18 and
19, the Onager's standing and rock-strike leg cases (20), the gait simulations of 16 and 17. The delta wing is not
here yet (`wing.WingSpec` refuses sweep).

## Data for notebooks
The split: a workflow runs what is long (CFD, FEA, MuJoCo, the libraries) and hands over what it computed as data; a
notebook keeps its own analysis code in view and imports those results instead of running the solvers again. Notebook
analysis is not moved into the workflows: the point of a notebook is to see in one place what you run.

Every workflow writes `data/<name>_results.json` next to its `.vida` when it exports (`--no-export` skips it):
plain JSON with every node's kind, status (computed / reused / NOT RUN and why), parameters and results; arrays as
lists, NaN as null. Grafted trees (AGUYA's engine, MERLIN's propulsors) are left to their own product's file, and a
single result over 1 MB (the ducted-fan maps) stays in the `.vida` as a reference that `results.load` resolves.
`python -m assemblies.results all` writes the files again from the saved `.vida` files without running anything.

```python
import sys; sys.path.insert(0, "..")                 # from notebooks/: the repository root
from assemblies import results
q = results.load("quadcopter")
q.status()                                           # one row per node: kind, status, why NOT RUN, when
q["life/fatigue"]["hours_to_failure"]                # a node's results
q.table("life/fatigue", "life")                      # a table-like result as a pandas DataFrame
results.load("merlin").table("race", "table")        # every race of notebook 27
```
Without `assemblies`: `json.load(open("../assemblies/data/quadcopter_results.json"))["nodes"]`. The product exports
(`quadcopter.json` in boreas' format, `fixed_wing.json` = the hand-off 09b reads, `microjet.json`, `merlin_design.json`,
`*_maps.json`, `*_life.json`) stay as they were.

## Components
| component | from | gives |
|---|---|---|
| `propeller.py` | 08, 09a, 12, 13, 14, 25, `scenarios/run_scenario.py` | `PropellerSpec` and the `CATALOGUE` (5x4.3 tri-blade, 9x6, 10x6, 60 mm and 120 mm marine), polar and blade tables, stamped CAD, rotor CFD cases per fidelity, blade loads and FEA |
| `wing.py` | 09a, 09b, 26, `aguya_wing` | `WingSpec` (rectangular and straight-tapered; sweep refused), planform, lift slope, Pratt gust, lift pressure, lower skins, root and motor regions, the wing FEA model |
| `hull.py` | 12, 13, 14, `run_scenario` | ITTC-57 friction, boat and body-of-revolution resistance, the double body, the nose upstream |
| `leg.py` | 11, 16, 18, 19, 20 | pin bending and shear, foot peak force, two-link wheel-leg torques, the plate-leg FEA models |
| `road_wheel.py` | 11, 20, 21 | ISO 8608 roads, rocks and drops, the quarter car, rock-strike force, rolling resistance, hub traction |
| `pincer.py` | 22, 23 | the Manus arm, its IK/FK, actuators, stow pose and parts |
| `quad_frame_analysis.py` | 08 cells 13, 29 | the frame FEA (full thrust, hard landing) and the canopy CFD case |
| `life.py` | 08 cells 88–104, 09b cells 12–25 | margins to the excitations, spectra, damage per mission, static re-check, damage rate, usage life, balanced propellers, nacelle amplitude |
| `quad_life.py` | 08 cells 80–104 | the frame with motors and stack as point masses, unit cases, the three missions, the PETG-CF curve, usage mixes |
| `fixed_wing_life.py` | 09b cells 4–42 | the wing with nacelle masses and the fuselage with the nose contents, unit cases, missions, the gust survey, the LW-PLA curve |
| `_cad.py` | — | `export_kept`: STEP/STL kept while the parameters are unchanged (stable FEA mesh keys) |
| `turbojet.py` | `notebooks/designs/turbojet.py` | the impeller, turbine wheel and engine CAD; the compressor passage STLs; sizing; masses |
| `turbojet_parts.py` | notebook 28 §4 | the three STEP files with masses and sizes (kept when the parameters are unchanged) |
| `cycle.py` | notebook 28 §1–3, 5, 7 | the catalogue engine, calibration from a speed line, design point, map, export |
| `impeller.py` | notebook 28 §5 | the compressor's `compressor_mrf` cases per fidelity, the speed line |
| `wheels.py` | notebook 28 §6 | the microjet's turbine and impeller FEA models (not the road wheel: that is `road_wheel.py`) |
| `aguya_jet.py`, `aguya_wing.py` | notebook 29 §7, §9 | the dash point and jet case; Pratt's gust numbers, the wing's gust and 6 g FEA, the modes |
| `apheloria_pack.py` | `scenarios/apheloria_pack.py` | the scripted pack/unpack controller and its summary (no movie) |

Copied verbatim from `notebooks/designs/` (imports made package-relative, nothing else changed; their tests copied
too): `fixed_wing`, `aguya`, `aguya_flight`, `quad_frame`, `air_propeller`, `ducted_fan`, `merlin`, `merlin_flight`,
`propulsor_maps`, `merlin_race`, `survey_boat`, `submarine`, `gait`, `actuators`, `rover`, `robot_dog*`, `myropod*`,
`apheloria*`, `onager*`. The notebooks keep their own copies; `benchmark/` keeps pointing at the notebook modules.

## Tests
`./assembly_tests.sh` (env + the fast tests; `--help` lists the sections) or directly:
`python -m pytest assemblies/tests -m "not slow"` (about eight minutes: CAD and the workflows with the CFD batches
replaced by known results, the simulations and FEA off; lifted snippets checked against the notebook cells themselves);
`-m slow` runs the FEA at smoke mesh for real (CalculiX), the MuJoCo scenes and a propulsor library.
`./assembly_tests.sh smoke` runs every workflow with the solvers on at `smoke` into `runs/assemblies_smoke/`
(nothing in `data/` changes); `./assembly_tests.sh air|water|ground|everything` are the real, resumable runs that
write `data/`.

### Solver mocks
Every test runs with the solver run functions mocked (`tests/stubs.py`, installed by `tests/conftest.py` for every
test): `talos.solve_models`, `StructuralModel.mesh` and `.solve_modes`, `talos.assess_fatigue` and `talos.read_frd`,
`aeromant.run_cases`, the OpenFOAM lookup and the ground workflows' `run_scene` are
`unittest.mock` objects autospecced from the real functions (a wrong call fails), each call recorded with the real
models and cases a workflow built, answered by stubs: fixed, plausible results in the real `Result` types and metric
names, `run=False` still NOT RUN. A test takes the `solvers` fixture to look at the calls (`solvers.fea`, `.cfd`,
`.sim`, or `solvers.solve_models.call_args_list`) or to swap an answer (`solvers.run_cases.side_effect = ...`). Tests
that need the real tools are marked `@pytest.mark.real_solvers` (and `slow`).

With the mocks, `test_stubbed_workflows.py` runs every workflow end to end with all its solvers on: nothing may stay
NOT RUN, a second run calls no solver and reuses every node of its own, and `stubs.check_calls` checks that what was
handed to the solvers is what the real ones take (real `StructuralModel`/`CFDCase` objects whose `key` computes, one
workdir per model, `progress` passed through). `test_cli_api.py` sends every command-line option of every workflow to
an autospecced mock of its `run()`.

Progress bars: the FEA and CFD batches, the propulsor libraries, MERLIN's race (one step per power), the Onager
variants and the walkers show tqdm bars; `progress=False` turns them off.
