# Scenarios — whole-machine flow simulations and particle movies

A scenario is a **script, not a notebook**: it builds a whole machine, runs one CFD case with its propeller(s)
as rotor disks, and writes a short movie of tracer particles in the converged flow, with the numbers next to it.
It runs unattended from a shell for tens of minutes; the notebooks stay interactive.

For now every scenario uses **preconfigured values** (hardcoded presets in `run_scenario.py`, taken from the
notebooks' current designs). Reading a design exported by a notebook (JSON) comes later.

```
scenarios/
  scenarios.md          this plan
  run_scenario.py       the one runner; PRESETS = {"air": ..., "sub": ..., "boat": ...}
  air_video.sh          fixed-wing aircraft with both propellers, in air          (every branch)
  sub_video.sh          submarine with its propeller, submerged                    (dev_sea, dev_demo)
  boat_video.sh         boat hull with its propeller, no free surface              (dev_sea, dev_demo)
  output/<scenario>/    everything a run writes (gitignored)
```

## Usage
```bash
scenarios/air_video.sh                 # 1 core (default)
scenarios/air_video.sh -j 8            # 8 cores: the solver runs with MPI on 8 subdomains
scenarios/air_video.sh --dry-run       # build the geometry, prepare the case, print the commands; no OpenFOAM
scenarios/air_video.sh --movie-only    # re-render the movie from the solved case
```
Options (all scripts): `-j N` cores (default 1), `--out DIR` (default `scenarios/output/<scenario>`),
`--iterations N`, `--seconds S` and `--fps F` for the movie, `--movie-only`, `--force` (start again from an empty
case), `--dry-run`. Progress: tqdm bars for the stages, the OpenFOAM steps (solver iterations read live from its
log) and the movie frames.

## What each run does
1. **Geometry** — build the Dedalus design with the preset parameters, write the STL (mm).
2. **Case** — `aeromant.CFDCase(template, stl, parameters)`, `prepare()`.
3. **Mesh** — blockMesh, feature edges, snappyHexMesh, checkMesh (serial).
4. **Solve** — `-j 1`: the solver runs serial. `-j N`: Aeromant writes `numberOfSubdomains N`, runs
   `decomposePar`, `mpirun -np N <solver> -parallel`, `reconstructPar -latestTime`.
5. **Movie** — tracer particles seeded upstream and carried by the converged velocity field, coloured by speed,
   the Vegeta logo in the corner: `output/<scenario>/<scenario>.mp4`.
6. **Summary** — `summary.json`: preset, template, cells, iterations, convergence, forces and coefficients,
   time per stage, files written.

A finished stage is not repeated when the script is started again; `--force` starts over.

## The three machines
| | air | submarine | boat (no free surface) |
|---|---|---|---|
| body | whole aircraft at the cruise angle of attack | hull with fins | hull below the waterline, mirrored about it ("double body") |
| propellers | 2 rotor disks, counter-rotating | 1 rotor disk behind the tail | 1 rotor disk + its mirror image (opposite sense) |
| fluid | air | sea water | sea water |
| template | `aircraft_rotor_disks` | `hull_rotor_disk` (one disk) | `aircraft_rotor_disks` (the mirror disk is disk 2) |
| results | airframe lift and drag in the slipstream | hull drag in the propeller's inflow | half the double-body drag: viscous + form drag, no wave drag |

The boat without a free surface has no waves: the waterline is a symmetry plane, so wave-making resistance, sinkage
and trim are missing. A free-surface scenario (interFoam, VOF) is a later step.

## Run times (estimates; replaced by measured times after the first real runs)
| | cells | 1 core | 8 cores |
|---|---|---|---|
| air | ~0.4–0.6 M | 25–50 min | 10–15 min |
| submarine | ~0.3–0.6 M | 20–45 min | 10–20 min |
| boat, double body | ~0.3–0.6 M | 20–45 min | 10–15 min |

## Work plan
- [x] Aeromant: `CFDCase.run(processors=N)` — decomposePar / mpirun / reconstructPar in both dialects; tests on
      the generated steps and commands (no OpenFOAM in the tests)
- [x] Aeromant: `animate_particles(progress=True)` — tqdm over the frames
- [x] `run_scenario.py` + `air_video.sh` with the air preset (09a's aircraft, propeller and cruise point)
- [x] Aeromant template `hull_rotor_disk` (one disk)
- [x] `sub_video.sh` with the submarine preset (`13_submarine`)
- [x] `boat_video.sh`, double body, with the boat preset (`12_boat_at_sea`)
- [ ] First real runs on a machine with OpenFOAM: disk thrust sign, mesh quality, measured run times → this file

Later: read the design from a JSON exported by the notebooks instead of the presets; a free-surface boat
(interFoam); resolved rotating propellers instead of disks (`todo.md` 4.12).

## Apheloria packs and unpacks (MuJoCo)

`xvfb-run -a python3 scenarios/apheloria_pack.py` (or `./user_tests.sh apheloria-movies`) simulates Apheloria
(`notebooks/designs/apheloria_robot.py`: head + 8 segments, 33.1 kg, 96 leg servos, 8 active body pitch joints of
60 N·m stall) in MuJoCo through Chiron and writes `scenarios/output/apheloria_pack.mp4` (stand → tuck the legs →
curl into the ball, neck first) and `apheloria_unpack.mp4` (from the ball → open tail first → stand), plus
`apheloria_pack_unpack.json` with the body joints' final angles, peak torques and saturation. A scripted controller
sets only servo targets; whether the ball closes is decided by gravity, contacts and the actuators' limits.

## Onager Sentinel on patrol (MuJoCo)

`xvfb-run -a python3 scenarios/onager_patrol.py [--response drag|limp]` (or `./user_tests.sh onager`) drives the
Onager Sentinel SX-1 (`notebooks/designs/onager_robot.py`: a 408 kg wheel-leg hybrid, eight 800 N·m joint
modules as position servos, four 3 kW hub motors as velocity servos on their torque–speed lines) over a gravel
road with a 120 mm speed bump at 3 m/s; at 6 s the front-left hub motor loses power (it freewheels), at 9 s the
rear-right wheel seizes. Both responses slow to 1.5 m/s, one movie each: `onager_patrol_drag.mp4` (keep driving on
three motors, the braked tyre skids) and `onager_patrol_limp.mp4` (the three-wheel limp: the hull shifts 0.4 m
forward on the three good legs, the seized wheel lifts 100 mm); `onager_patrol.json` has the per-phase tables
(speed, heading, tilt, corner loads, wheel torques and power, the seized wheel's drag). The scenario is
`notebooks/designs/onager_scenario.py`, the same one notebook 20 §7 runs.

## Onager Atlas moves a pallet (MuJoCo)

`xvfb-run -a python3 scenarios/onager_atlas_pallet.py` (or `./user_tests.sh onager-atlas`): the forklift Onager
(`notebooks/designs/onager_atlas_robot.py`, 726 kg, four-quadrant drives) picks a Euro pallet with a 175 kg crate
off a gravel yard (forks at travel height, stand-off, lower to the openings, creep in, lift, tilt back), carries it
8 m at 1 m/s, sets it down and backs out. The pallet is a Chiron prop; whether it rides on the forks is contact
physics. Writes `onager_atlas_pallet.mp4` and `onager_atlas_pallet.json` (the phase table: pallet height and tilt,
lift force, tilt torque, front knees against their stall, the lightest rear wheel).

## Onager Manus clears a blocked track (MuJoCo)

`xvfb-run -a python3 scenarios/onager_manus_tasks.py` (or `./user_tests.sh onager-manus`): the two-arm Onager
(`notebooks/designs/onager_manus_robot.py`, 458 kg) stops at a fence wire across the track (Ø 3.15 mm, 1200 MPa:
7.5 kN to cut), puts its right pincer's cutter notch on it and closes — a scene hook releases the wire's weld only
when both jaws squeeze it with the cutting force — drives through, crouches at a 14 kg log, takes it with the left
pincer's hooked jaws, lifts it, swings it over the side, puts it down and drives on. Writes
`onager_manus_tasks.mp4` (the wire drawn 4× thicker) and `onager_manus_tasks.json`.

## Onager Sweeper cleans a street (MuJoCo)

`xvfb-run -a python3 scenarios/onager_sweeper_street.py` (or `./user_tests.sh onager-sweeper`): the street-cleaning
Onager (`notebooks/designs/onager_sweeper_robot.py`, 590 kg: the chassis low in a fixed stance, a disc broom, a
suction hood, two Manus arms, a basket on the roof) spins its broom, runs the fan — the suction is the CFD's air
drag on the litter under the hood (`onager_sweeper_cfd.SUCTION`), what reaches the duct is collected — sweeps at
1 m/s over cans and packets, stops at a 2.3 kg brick, takes it with its left pincer's hooked jaws and swings it
back into the basket, does the same with a 0.6 kg box with the right pincer, and sweeps on. Writes
`onager_sweeper_street.mp4` and `onager_sweeper_street.json` (what was vacuumed, what landed in the basket, the
phase table).

## MERLIN races to a wildfire and samples its smoke

`python scenarios/merlin_mission.py [--propulsion fastest|edf|tractor|pusher] [--distance-km 20] [--heat-mw 40] [--wind 5]
[--wind-from 225] [--race-only] [--quick]` (or `./user_tests.sh merlin`). Builds the three MERLINs of notebook 26 (ducted fan,
tractor, pusher on one battery and one power limit), prints the time to a fire 10, 20 and 30 km out for each, flies the fastest
(or the chosen one) through a Gaussian smoke plume — climb, dash, crosswind passes at four heights, home, belly landing — and
estimates the fire's heat release from the passes. Writes `merlin_<propulsion>_<km>km.mp4` and `.json` (race table, mission
summary, phases, source estimate).

