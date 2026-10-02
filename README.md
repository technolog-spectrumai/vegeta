<p align="center"><img src="logo_with_text.png" alt="Vegeta logo" width="320"></p>

# Vegeta

A Python-first, human-in-the-loop engineering workbench built from small,
independent engineering tools, shipped together as one installable package, **`vegeta-cli`**
(`from vegeta import dedalus, talos, aeromant, mellonia`):

| Package | Job | Wraps | Consumes → Produces |
|---------|-----|-------|---------------------|
| **Dedalus** | parametric CAD | CadQuery | Python design → STEP, STL, measurements |
| **Talos** | linear static FEA | Gmsh + CalculiX | STEP + explicit config → mesh, `.inp`, `.frd`/`.dat`, summary |
| **Aeromant** | aerodynamics / CFD | OpenFOAM | STL + template case + explicit values → case, logs, Cd/Cl/Cm |
| **Mellonia** | 3D-print manufacturability | PrusaSlicer | STL + explicit print settings → G-code, time, material |
| **Vegeta Core** | workbench | the four above | workspaces, immutable revisions, recorded evaluations, NOT RUN |
| **Fidia** | AI modelling | Dedalus + `vegeta-ai` (Claude) | prompt → checked 3D model (glTF/GLB, OBJ, STEP); design copilot; campaigns |

Dedalus designs. Talos tests structures. Aeromant tests aerodynamics.
Mellonia tests manufacturability. Fidia sculpts from a prompt. Vegeta organizes the engineer's work.

Vegeta Core, Fidia and the AI connection are their own distributions next to `vegeta-cli`:
`vegeta-core`, **`vegeta-fidia`** and `vegeta-ai` (the provider layer only: keys, model, structured
output, images, cancellation, usage).

## Principles
- The four engineering tools are **independent** subpackages (`vegeta.dedalus`, ...): none imports
  another or any other part of Vegeta (enforced by tests). `vegeta` is a namespace package so more
  distributions can join it later.
- Each is a plain Python library, usable from a script or Jupyter, with a thin CLI on top
  (`vegeta <tool> ...`, or the shortcuts `dedalus`, `talos`, `aeromant`, `mellonia`).
  No server, database or background service. A GUI comes later, on top of the same API.
- Configuration happens in **Python** (dataclasses / keyword arguments), not config files.
- Packages compose through **files**: STEP, STL, meshes, CalculiX decks, OpenFOAM cases, G-code.
- Native solver artifacts are always kept; `summary.json` is a convenience, not a replacement.
- Nothing is guessed: supports, loads, materials, CFD values and print settings are explicit.
- Nothing runs automatically: every analysis is started by the engineer.

## Status
Stages 1–9 are implemented: the four independent tools (Python API, CLI, validated tests, docs,
notebooks) and Vegeta Core (workspaces, immutable revisions, branching, labels, recorded
evaluations, NOT RUN status; [docs/core.md](docs/core.md), `06_core_revisions`). Next: comparison and
sweeps, then a GUI — see [`todo.md`](todo.md).

| package | CLI | validation | guide | notebook |
|---------|-----|------------|-------|----------|
| Dedalus | `vegeta dedalus params/generate/measure` | exact box/hole volumes, STEP round trip | [docs/dedalus.md](docs/dedalus.md) | `01_dedalus_cad` |
| Talos | `vegeta talos inspect/mesh/solve/results` | cantilever vs beam theory (−0.6 %), bar F/A, gravity | [docs/talos.md](docs/talos.md) | `02_talos_fea` |
| Aeromant | `vegeta aeromant templates/prepare/run/results` | sphere Re=100 Cd 1.100 vs 1.092 | [docs/aeromant.md](docs/aeromant.md) | `03_aeromant_cfd` |
| Mellonia | `vegeta mellonia slice/parse` | layer counts, solid cube volume (+0.9 %) | [docs/mellonia.md](docs/mellonia.md) | `04_mellonia_print` |
| Boreas | `vegeta boreas point/for-thrust/map` | T~n², P~n³, momentum limit, APC 10x4.7 static point | [docs/boreas.md](docs/boreas.md) | propeller parts of `08_`, `09_` |
| Chronos | `vegeta chronos spectrum/life` | ASTM E1049 rainflow example, DAF = 1/2ζ at resonance, Miner sums | [docs/chronos.md](docs/chronos.md) | life parts of `08_`, `09_` |
| Chiron | `vegeta chiron info/run/metrics/render` | foot forces = m·g (1 %), contact decoding = `mj_contactForce` (1e-9), impulse Δv = J/m (1 %), servo torque–speed line, foot Jacobians = `mj_jac` (1e-12), bitwise-reproducible paired trials | [docs/chiron.md](docs/chiron.md) | `benchmark/cleopatra`, `benchmark/persephone` |
| Fidia | `vegeta fidia run/resume/show/propose/accept` | sandbox walls (timeout, memory, no keys), watertight/winding checks, glTF/GLB/OBJ re-import with trimesh and VTK | [docs/fidia.md](docs/fidia.md) | `15_fidia_prompt_to_3d`, `07_`, `10_` |

Product-level notebooks — one notebook per machine, the whole workflow on one design with revisions,
visualisation and a JSON record: `08_quadcopter` (printed X-frame: load cases, three revisions, drag with a
canopy, slicing; then the propeller and drive — BEMT, a rotating-frame CFD check, excitations, noise, a
blade FEA with a stress video; then modes, Campbell diagram, three mission types, rainflow spectra,
fatigue on the FEA stress fields and a fleet-usage life) and the twin-motor fixed wing in two notebooks:
`09a_fixed_wing_design` (wing pull-up and engine-out cases, whole-aircraft RANS, then the propeller on the
resulting drag polar with the same CFD/noise/blade/video treatment, then the whole aircraft in flight with its
propellers as rotor disks; it ends with a hand-off file) and `09b_fixed_wing_durability` (wing modes, missions
and life, then the whole fuselage over a long full-battery mission; it starts from 09a's hand-off, or recorded
values). `07_ai_design_copilot`
iterates a design with Claude; `10_agentic_design` lets Claude run a bounded parameter campaign (FEA on
every candidate, criteria, budget, approval policy) in a workspace; `15_fidia_prompt_to_3d` goes from a
prompt to an exported 3D model (plan, sandboxed build, checks, renders, review, feedback) and runs offline
with a scripted agent when no API key is set. Shared design files live in
`notebooks/designs/`. Every movie a notebook makes is written to `notebooks/output/<notebook>/`. The OpenFOAM cells run when you run them (`VEGETA_SKIP_OPENFOAM=1` skips them).
`11_rover_mechanics` (branch `dev_land`) is a small rugged-terrain rover: terrain profiles → quarter-car wheel
loads → arm, chassis and pin stresses over the operating cases (rocky peak, hill climbing, mud, heavy payload),
arm modes and a speed–frequency diagram, rainflow fatigue, life, printing.
`12_boat_at_sea` (branch `dev_sea`) takes a 1 m survey boat from hull lines through hydrostatics and the GZ
curve, resistance and propulsion in water, slamming and thrust structure, three sea states and fatigue life, then
its propeller in detail: performance and cavitation, a rotating-frame CFD check with a flow video, blade stress
with a video and modes, blade-passing noise, and one frequency diagram for bracket and blade. `13_submarine`
(`dev_sea`) is a 1.2 m AUV: buoyancy, trim lead and stability, resistance with a submerged RANS check, thruster
endurance and range, the pressure hull at its rated depth against thin-shell theory and collapse pressures,
the propeller (CFD, noise and cavitation, blade modes, Campbell), a dive simulation with a propulsion loss,
and dive profiles as pressure cycles. `14_submarine_propeller` (`dev_sea`) takes that propeller apart: pressure
against speed and depth down to cavitation, the fin wake (a model, or the hull's own CFD wake), rotor CFD at three
operating points, the flow in 3D with streamlines and tracer particles, OpenCV movies of the flow at each operating
point, blade FEA with its weak points, load harmonics with their frequencies and amplitudes, a frequency diagram
against blade and hull modes, blade response and fatigue, the noise spectrum, and design updates proposed by rules,
evaluated side by side, with a recommendation; then the recommendation (7 blades in nickel-aluminium bronze) made
quieter one measure at a time — skew, an anti-singing trailing edge, a damping alloy, a larger slower propeller,
fins moved upstream — each kept only if it is quieter without breaking a criterion.

`16_robot_dog` (branch `dev_sikarian`) is a 13 kg quadruped for patrols and stairs: CAD with a payload deck,
foot forces per gait (duty factor), joint torques over a stance, stairs and slopes with a payload, a drop from a
step, FEA of both leg links and the pins, lower-leg modes against stride harmonics, rainflow fatigue over three
missions, the trot as a sequence of leg angles with a movie, printing. Actuators, motors and joints for the dog and
the Myropods come from one shared catalogue, `notebooks/designs/actuators.py` (explicit entries with sources, a
selection helper and holding power). Walking is simulated by one shared stepping machine,
`notebooks/designs/gait.py` (terrain, body pose from the ground, foot stepping, inverse and forward kinematics and joint
torques for dog legs and Myropod legs), written for the dog and reused for Cleopatra and Persephone (whose feet land on a pipe
wall: `PipeContact`, `path_pose`). The **Myropods** (`dev_sikarian`) are a family of segmented walkers from one design file —
four legs on every segment, so each segment stands and braces on its own: `17_myropod_persephone` is the chimney
crawler (twelve 60 mm segments, 48 legs, a tether) in three versions — v1 the crawler (fit in flues and elbows,
bracing on soot and the servo class it needs, climb power and the tether, leg and shell FEA, a thermal limit in a
warm flue, the comms and power links, gait dynamics, leg fatigue), v2 transport into the chimney (coiled in a carry
case around its tether drum, the flue-mouth insertion guide, the push-in / anchor-and-pull / emergency-retreat
forces, case FEA), v3 two pincers on the head (grip, head moment, pincer FEA), and its gait from the hearth into the flue: 48 legs joint by
joint on the pipe wall through a 90° knee, with a step table and a movie; `18_myropod_cleopatra` is the
three-segment 12-legged walker (gaits, actuator torques, limb loss and the support polygon on three legs, leg FEA,
endurance, and its gait on open terrain: twelve legs joint by joint, the body joints between segments, a movie); `19_myropod_apheloria` (`designs/apheloria.py`) is the modular pill millipede that rolls into a ball
(configurations and module masses, ball geometry, rolling and a drop, the curl-up joint moment, plate and leg FEA,
endurance).

Project Onager — land robots (branch `dev_sikarian`): `20_onager_sentinel` (`designs/onager.py`) is the Onager
Sentinel SX-1, a 380 kg wheel-leg reconnaissance unit: CAD of the hull, turret, mast and four wheel-legs against
the datasheet envelope; mass budget and CG; wheel mode (resistance vs the hub motors' torque–speed line, grades,
the 0–38 km/h run, range and endurance); walking mode (stance torques, the joint modules and the speed they
allow); ISO 8608 terrains through a quarter car with the active leg as suspension, hull vibration PSDs, pitch,
leg and mast modes from Talos and a frequency diagram; static FEA of both leg plates, stub axle and knee pin; the
actuators under Chiron's servo model (a stand-up from the crouch, currents and the battery); and in MuJoCo through
ChironLab a patrol on gravel with a speed bump where a hub motor fails and a wheel seizes, two responses compared —
drag vs the three-wheel limp — with a movie each. `designs/onager_robot.py`, `onager_controller.py` and
`onager_scenario.py` are the Sentinel in ChironLab; `scenarios/onager_patrol.py` (or `./user_tests.sh onager`)
re-runs the patrol and writes the movies. `21_onager_atlas` (`designs/onager_atlas.py`) is the forklift: the Sentinel
chassis in a logistics stance with a tilting mast, lift carriage and forks (load chart and tipping, the stance's
knee torques, lift and tilt screws, fork FEA and mast modes, a quarter car with the pallet, and in MuJoCo a pallet
job — approach, lift, carry 8 m, set down — with a movie). `22_onager_manus` (`designs/onager_manus.py`) has two
manipulator arms with pincers (workspace and joint torques, the cutting envelope, why the jaws have hooked tips,
jaw and arm FEA, and in MuJoCo a track blocked by a wire it cuts — the wire parts only when both jaws squeeze it
with the cutting force — and a log it lifts off the track, with a movie). `scenarios/onager_atlas_pallet.py` and
`scenarios/onager_manus_tasks.py` (or `./user_tests.sh onager-atlas` / `onager-manus`) re-run them. `23_onager_sweeper`
(`designs/onager_sweeper.py`) is the street cleaner: the chassis low in a fixed stance, a bulky rounded body, a disc
broom with a suction hood behind it, the Manus arms and a basket on the roof (mass budget, broom and fan sizing,
power and endurance, the arms' reach to the road and the basket; the suction in CFD — Aeromant's new `suction_hood`
template: the depression, the lip inflow, which litter the hood lifts, a litter-particle movie; basket and hood FEA,
the broom's and fan's unbalance against their modes; and in MuJoCo a street with litter the hood vacuums — the CFD's
drag on Chiron props — and a brick and a box the pincers load into the basket, with a movie);
`scenarios/onager_sweeper_street.py` (or `./user_tests.sh onager-sweeper`) re-runs it.

Benchmarks (`benchmark/`, branch `dev_sikarian`): `benchmark/cleopatra/full_benchmark.py` runs Cleopatra's pre-registered body-joint study in MuJoCo through Chiron — spring-only vs spring–damper intersegment joints × baseline vs load-feedback control over flat, bumpy, cross-slope and rough ground, speed sweeps, pushes, the undulation onset and damping/roll sensitivity (docs/myropod_stability.md) — and saves results.json, CSV, raw time series, plots and a report; `benchmark/persephone/full_benchmark.py` does the same for Persephone on the hearth (not yet validated; no flue). `./user_tests.sh` runs the test suites, the physics checks and the smoke benchmarks and prints a report to paste back.

Scenarios (scripts, not notebooks): `scenarios/air_video.sh [-j N]` builds the whole fixed-wing aircraft, runs one
CFD case with both propellers as rotor disks and writes a particle movie to `scenarios/output/`; on `dev_sea`
also `scenarios/sub_video.sh` (submarine) and `scenarios/boat_video.sh` (boat, double body, no free surface) — see
[scenarios/scenarios.md](scenarios/scenarios.md).

More: [philosophy](docs/philosophy.md) · [result shape](docs/result-shape.md) ·
[installation](docs/installation.md) · [composition through files](docs/composition.md) ·
[Vegeta Core](docs/core.md) · [Fidia: AI modelling](docs/fidia.md) · [AI provider layer](docs/ai.md).

### The names
Every tool is a figure from Greek myth, one word, easy to say in a meeting:
Dedalus the craftsman (CAD), Talos the bronze giant (structure), Aeromant "reader of the air" (CFD),
Mellonia the goddess of bees and their wax (3D printing), Boreas the north wind (propellers and
rotors), Chronos time (missions, cyclic loads, life), Chiron the centaur who trained the heroes (legged robots
walking in MuJoCo; ChironLab is where they train), Fidia — Phidias, the sculptor of the Parthenon —
(AI modelling from a prompt). The workbench itself is Vegeta.

## Layout
```
vegeta-cli/        the vegeta-cli package: src/vegeta/{cli,dedalus,talos,aeromant,mellonia,boreas,chronos,chiron}, tests/<tool>/
vegeta-core/       the vegeta-core package: src/vegeta/core (workspaces, revisions), adds `vegeta ws|rev`
vegeta-ai/         the vegeta-ai package: src/vegeta/ai (connection to the AI provider), adds `vegeta ai check|models`
vegeta-fidia/      the vegeta-fidia package: src/vegeta/fidia (prompt-to-3D, copilot, campaigns), adds `vegeta fidia`
notebooks/         one notebook per package, the workflow, core, AI copilot and two product designs
docs/              philosophy, result shape, installation, composition, per-package guides
examples/cli/      input files for the CLI demo (Talos model, Aeromant case, Mellonia settings)
install_local.sh   creates .venv, installs all dependencies and vegeta-cli
test.sh            checks the installation (components + a short working run)
test_openfoam.sh   tests the CFD tool against every OpenFOAM installation found (or given), both flavours
jupyter.sh         starts JupyterLab from .venv
scripts/           test_all.sh, run_notebooks.sh, demo_cli.sh
```

## Quick start
```bash
./install_local.sh                  # .venv first, then apt tools, OpenFOAM, vegeta-cli, JupyterLab
./test.sh                           # check the installation; prints OK/MISSING/FAILED per part
./test_openfoam.sh                  # test Aeromant against the OpenFOAM installation(s) on this machine (.com and .org)
./jupyter.sh                        # JupyterLab from the venv; start with notebooks/00_smoke_test.ipynb
source .venv/bin/activate           # for the vegeta command and the scripts below
scripts/test_all.sh                 # each package tested on its own
scripts/run_notebooks.sh            # execute the notebooks headlessly
scripts/demo_cli.sh                 # CAD -> FEA -> print -> CFD using only the CLIs
```

## Licence
Proprietary — **internal use only** (see `LICENSE`). External tools keep their own licences, see
`THIRD_PARTY_LICENSES.md`.
