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
