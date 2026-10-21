# K4TQ — implementation plan

Compact convertible tilt-wing medical transport UAV: four independently rotating tandem wings, one propeller near each
wingtip, a level fuselage in every mode. It replaces the Kestrel tailsitter line (Kestrel failed on the back transition's
balloon, an undersized wing against the measured CL max, and no solver validation).

**Objective.** The best of two worlds — quadcopter and fixed-wing aeroplane — with compromises accepted as the nature of
the design. The study must say plainly whether K4TQ beats a quadcopter of the same size (or larger), and why or why not.

Legend: `[ ]` open · `[x]` done · `[-]` deferred. Result labels everywhere: **ESTIMATE** (hand calculation), **CALC**
(numerical model), **MOCK** (mocked solver, pipeline check only), **SOLVER** (a real CFD/FEA run), **TEST** (hardware).
Nothing is called validated until a SOLVER or TEST result says so.

## Requirements (as agreed, 2026-10-21)

| # | requirement | value |
|---|---|---|
| R1 | fuselage / wingspan | ~1.0 m / ~1.4 m; tandem wings at ~30 % and ~75 % of the fuselage |
| R2 | payload | 1 kg on the standard mission; 0.5–0.75 kg acceptable on longer missions (payload–range trade) |
| R3 | standard mission | 10 km radius (20 km out and back), 2 min hover in total, 20 % energy reserve, 8 m/s wind |
| R4 | modes | cruise (wing chords horizontal, propellers forward), hover (chords vertical, discs horizontal), fuselage level in both |
| R5 | transport | wings fold to pack into a tube; the fold need not be 100 % (partial fold accepted) |
| R6 | launch | vertical from a hand (hand-release), from the transport tube (opened case as the pad), from a hospital roof — each checked |
| R7 | temperature logging | **mandatory**: redundant sensors, continuous timestamped log on board, survives power loss, streamed by telemetry |
| R8 | marking | text only, no symbol, no red cross: **"BE CAREFUL – FRAGILE – TEMPERATURE SENSITIVE LOAD"** |
| R9 | directional stability | a vertical fin is allowed if the analysis shows it is needed |
| R10 | transition | checked with CFD (cases prepared here, run manually) |
| R11 | propulsion | exactly four motors and four propellers, no extra lift rotors; parts available in Poland / the EU preferred |
| R12 | early phases | everything mocked (`DEBUG = True`, `VEGETA_DEBUG=1`); no expensive solver runs without approval |

## Where things go

`labs/aero/designs/k4tq*.py` (design modules), `labs/aero/designs/data/k4tq/` (spec JSON, drive tables, BOM, STEP/STL,
reports), `labs/aero/tests/test_k4tq*.py`, `labs/aero/campaigns/side_labs/` (book `k4tq`), notebook
`labs/aero/exp/K4TQ.ipynb` → graduation to `labs/aero/notebooks/K4TQ.ipynb` on approval only.

## Stage 1 — Engineering design and feasibility
*First-principles sizing of the whole aircraft, compared with three alternatives and a same-size quadcopter, before any CAD.*

- [ ] 1.1 Requirements, mission and operations (R1–R12) as data in `k4tq.py`; the component catalogue (motors, propellers, cells, avionics, actuators, bearings) with masses, prices and sources marked "to verify"
- [ ] 1.2 Parameters and layout: fuselage, the two wing pairs (stations, heights, sweep, chord), pivot and fold-hinge axes, nacelles, the propeller poses in cruise / hover / folded
- [ ] 1.3 Mass budget and CG (`og.mass`), the battery placed for the CG; growth margin
- [ ] 1.4 Wing loading, required lift, stall speed; section choice from NeuralFoil polars (recorded constants under the mocks)
- [ ] 1.5 Tandem interference: lattice (`og.aero.vlm`) — front downwash, rear-wing efficiency, lift split, neutral point, static margin, stall order
- [ ] 1.6 Drag build-up, cruise thrust and power, range and endurance; two-motor cruise (rear pair stopped, blades folded) against four
- [ ] 1.7 Hover: disc loading, induced power (actuator disc), BEMT figure of merit, hover endurance, T/W, motor-out note
- [ ] 1.8 Propeller and motor selection: BEMT sweep of candidate propellers on candidate motors (hover against cruise compromise)
- [ ] 1.9 Transition corridor: wing tilt against speed in level flight and descent, the blown and unblown wing angles, stall margin, thrust, power, pitch trim by thrust split
- [ ] 1.10 Wing-root loads: bending, torsion, bearing loads, actuator torque (aero, inertia, thrust offset, gyroscopic)
- [ ] 1.11 Safety factors and material requirements; first spar and bearing sizes
- [ ] 1.12 Directional stability: Cnβ without a fin, the fin needed (R9)
- [ ] 1.13 Launch and landing checks (R6): hand release, tube case as the pad, hospital roof (wind speed-up, gusts, downwash, ground hold)
- [ ] 1.14 Mission energy (R3) and the payload–range trade (R2) with the standard and extended packs
- [ ] 1.15 Configurations A / B / C compared; the same-size quadcopter references Q-13 and Q-best; the selection
- [ ] 1.16 Feasibility verdict, assumptions, uncertainties, initial BOM; report `data/k4tq/phase1.md` + spec JSON; tests

## Stage 2 — Parametric CAD and mechanical design
*The selected configuration as a Dedalus/CadQuery assembly in three configurations, with the wing-root tilt-and-fold assembly designed as real hardware.*

- [ ] 2.1 `k4tq_cad.py`: fuselage loft with the cargo bay, sealed hatch, battery and avionics bays (all dimensions from `k4tq.py`)
- [ ] 2.2 Wing panels with real airfoil sections, spar tubes, nacelles, motor mounts, simplified propeller discs (folding blades)
- [ ] 2.3 Wing-root cartridge (highest priority): spar in thin-section bearings, tilt drive (servo + reduction), hard stops 0° / 90°, encoder, wiring service loop through the hollow spar
- [ ] 2.4 Fold hinge: vertical-axis pin in bushings, spring latch, fold with the wings at the hover angle (edge-on), partial-fold angle from the clearance check
- [ ] 2.5 Landing legs, fin (if Stage 1 needs it), medical cartridge (insulated, shock-isolated, two temperature sensors), marking text
- [ ] 2.6 Assembly with `config = cruise | hover | folded` and any intermediate tilt
- [ ] 2.7 Checks: clearances, propeller separation, ground strike, folded envelope against the tube, tolerances, assembly access, load paths
- [ ] 2.8 STEP and STL export (Yax mass properties), CAD mass against the Stage 1 budget (revise Stage 1 if it differs)
- [ ] 2.9 Interactive 3D views in Jupyter; tests

## Stage 3 — Aerodynamics, CFD and performance optimisation
*Lightweight numerical aerodynamics on the CAD geometry, a few optimisation iterations, and CFD cases prepared (and mocked) for the transition and cruise.*

- [ ] 3.1 Section polars at the flight Reynolds numbers (Zephyr), wing and fuselage drag on the CAD wetted areas
- [ ] 3.2 Tandem interference and lift distribution on the CAD geometry (lattice with the tilt angles)
- [ ] 3.3 Propeller slipstream on the wings (actuator discs), cruise efficiency, rotor interference in the hover
- [ ] 3.4 Transition aerodynamics at intermediate tilt; longitudinal stability and control authority through the corridor
- [ ] 3.5 3–5 optimisation iterations (wing geometry, separation, incidence, twist, propeller diameter, fuselage proportions), logged
- [ ] 3.6 Comparison with Stage 1, discrepancies explained
- [ ] 3.7 CFD: cases, meshes and scripts for cruise and transition (Aeromant, actuator discs), the `k4tq` side-lab book; mocked runs only
- [ ] 3.8 The list of full simulations to run manually; tests

## Stage 4 — Structural FEA and manufacturing with Mellonia
*Load cases through Talos (mocked FEA), the make/buy plan and printability through Mellonia, and an EU-sourced BOM — revising the design where a part fails.*

- [ ] 4.1 Load cases: cruise gust, manoeuvre, hover with one motor at maximum, transition, landing (traceable to Stage 1 / 3 numbers)
- [ ] 4.2 Wing bending and torsion, spar and bearing loads, rotary joint and fold hinge, actuator torque, locks, motor mounts, fuselage
- [ ] 4.3 Buckling, modes (Campbell against the rotor lines), freeplay and flutter risk of the tilting wing, fatigue (Chronos)
- [ ] 4.4 Talos models and the FEA book chapters (mocked here, real runs prepared)
- [ ] 4.5 Make/buy classification: buy, print, composite, machine, hybrid (the wing roots)
- [ ] 4.6 Mellonia: printability, orientation, tolerances, materials, supports, print time and cost (mocked slicer)
- [ ] 4.7 BOM with Polish / EU suppliers, prices, masses, materials, make-or-buy; rejected options listed
- [ ] 4.8 Revisions of CAD and assumptions forced by the structure; tests

## Stage 5 — Validation, documentation and graduation
*The experimental notebook, the regression tests, the manual simulation checklist, and a graduation proposal awaiting approval.*

- [ ] 5.1 `labs/aero/exp/K4TQ.ipynb` organised by Stages 1–4 (bootstrap, `DEBUG = True`)
- [ ] 5.2 Smoke and regression tests: CAD generation, transformations, physics, load consistency, simulation inputs, exports (mocks)
- [ ] 5.3 Manual simulation checklist: exact commands, expected outputs, compute requirements
- [ ] 5.4 Engineering conclusions, unresolved risks, K4TQ against the quadcopter in one table
- [ ] 5.5 Graduation proposal to `labs/aero/notebooks/K4TQ.ipynb` — **on approval only**

## Key risks (tracked through the stages)

1. Wing-root mechanism mass, freeplay and torque — the "tilt-wing tax" against the quadcopter.
2. Directional stability without a fin (R9 allows one).
3. Transition: partial stall of the unblown wing at intermediate tilt, rear wing in the front wing's wake.
4. Hover-optimised propellers cruise badly; motors sized for the hover idle expensively in cruise.
5. Gusty rooftop operation: vertical wings present area to the wind in the hover and on the ground.
6. Motor-out in the hover (four rotors only) — a safety case item (parachute in the budget).
7. No solver in the container: every result before the manual runs is ESTIMATE, CALC or MOCK.
