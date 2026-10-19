# Races — four drone race environments in MuJoCo for the reference 5-inch quad

Four courses, one reference aircraft, one autopilot, one set of rules, a mock flyer for development, a thin CLI
and one notebook per course in `notebooks/races/` (`01_hangar`, `02_industrial`, `03_farm`, `04_quarry`) with
`race_design_factors` combining their exports. MuJoCo does the rigid body and the contacts; the aircraft's
aerodynamics and propulsion are the existing models (the Boreas drive table of the 6S 5x4.3 2207 drive, the frame's
drag area) and the flight controller is Drongo's geometric controller (`designs/drongo_controller.Flight`).
Nothing is redesigned, no CFD, no FEA, no learning.

| course | kind | length | zones | what it is |
|---|---|---|---|---|
| `hangar` | technical | ~170 m | 18 | a 40 × 18 × 12 m hangar, three floors 4 m apart, a staircase, floor openings, open windows, roof beams, a door |
| `industrial` | technical | ~485 m | 18 | seven large buildings with 10–15 m gaps, two covered passages to fly under, stacks and a water tower |
| `farm` | open | ~350 m | 20 | a cottage, a barn to fly through, a silo, a 60 m straight over the field, a hedge, a slalom down the tree line |
| `quarry` | open | ~520 m | 21 | a pit with a 90 m floor and three 6 m benches, winding corridors between spoil heaps, a haul ramp, a crusher, a conveyor |

## The reference aircraft (`races/refquad.py`)
A carbon X-frame 5-inch racing quad: four **2207 1750 KV** motors on **6S** (22.2 V nominal; a 2400 KV 2207 is a 4S
motor), 5x4.3 three-blade propellers, a 4-in-1 ESC, a flight controller, an FPV camera and video transmitter, a
6S 1300 mAh pack. All-up mass **571 g** from the component budget (`refquad.BUDGET`). Propulsion: the Boreas table
`designs/data/mini_raptor_drive_6S_5x4p3.json` (thrust limit and electrical power against axial inflow;
18.4 N static per motor, thrust-to-weight 13), with an analytic fallback that says so. Drag: 0.016 m² at racing tilt.

## Zones, rules, statistics (`races/rules.py`)
Every course is an ordered list of **zones**: boxes the racer must enter — the start, gates, checkpoints, the finish.
- zones count **only in order**; a later zone entered first is a missed gate: the racer must go back (the autopilot does);
- every **checkpoint** (and the start) is a reset point;
- a **collision** is any contact of the aircraft with the scene once airborne; the rule is `reset` (back to the last
  checkpoint, +5 s), `count` (fly on) or `stop` (DNF); more than 5 collisions is a DNF; 300 s is a timeout;
- the **lap** is complete when the finish is entered after every other zone, in order; the lap time runs from leaving
  the start zone to entering the finish, penalties included.
Recorded: lap time, completion, zones passed, collisions, resets, missed gates, energy (Wh, Wh/km), peak and mean
power, maximum and mean speed, the gate splits, the trajectory.

## Scenes (`races/scenes/*.xml`, `races/courses.py`)
Each course is a Python builder with parameters (`hangar(length, width, height, floor_gap, gate_half)`,
`industrial(scale, gate_half)`, `farm(gate_half)`, `quarry(floor_r, bench_h, bench_w, n_benches, gate_half)`) that
gives collision geometry (boxes, cylinders, spheres), the zones (transparent, non-colliding boxes in the scene), a
starting grid of four pads, landmarks (a red tank, a yellow gantry, a red stack, a water tower, a chimney, a silo, a
crusher, an excavator), the extent, and recovery notes (every zone is reachable on foot, by the staircase or the
haul ramp). `to_mjcf` writes the MJCF with the quad (`refquad.mjcf_body`) and four cameras: `overview` (top-down),
`grandstand`, `onboard` (FPV, 30° up, 110°), `chase` (tracking the quad). `clearance_check` lists every zone's
distance to the nearest obstacle. The racing line is the polyline through the zone centres; `races/path.py` gives it
the speed schedule the aircraft's limits allow (the corner speed each zone's clearance permits at the thrust
vector's lateral acceleration, braking and acceleration between zones, the top speed) and the autopilot's moving
reference (`carrot`: short look-ahead, no corner cut larger than the zone, a return to a missed zone).

## As workflow chapters (Yanus): book `races`, code R
`wf R1`–`R4` (`side_labs/races/race.py --chapter <course>`) run a course under the run contract: a run directory
under `side_labs/runs/races/<course>/` with the manifest, the case `race` (trajectory, stats, map) and the cases
`video-chase`, `video-onboard`, `video-overview` (MuJoCo camera frames through ffmpeg when an OpenGL context exists —
`MUJOCO_GL=egl` — else the matplotlib replay, said so in the case record), and `summary.json` with the statistics,
the manoeuvre metrics and the video paths. Notebook 50 **takes** these runs through its gate: it shows the videos and
the statistics of the runs on file; in auto mode (`YANUS_AUTO=1`) it runs the missing chapters itself in screening
quality (short, small videos) first. `--quality fine` gives full-length videos; `--mock` runs the point-mass flyer
under the same contract (the smoke tests).

## Manoeuvrability statistics (`races/metrics.py`)
From every recorded lap: peak acceleration and deceleration along the path, peak lateral acceleration, the load
factor carried, the least turn radius above 3 m/s and the peak turn rate, peak roll / pitch / yaw rates and angular
accelerations (the recorded body rates in MuJoCo; the thrust vector's slew and the heading in the mock), time to
speed, tilt, the share of the lap near full thrust, the tracking error to the racing line. `envelope()` is what the
reference quad could do from its drive table and an inertia estimate; `LITERATURE` the published reference values
(Foehn et al. 2021: 1–20 m/s², 10 rad/s, thrust-to-weight 3.3–4; Swift, Nature 2023: over 100 km/h). Peaks are 99th
percentiles, since differentiated samples spike at the zone hand-overs.

## Digesting the results: `python -m races ingest`
`races/ingest.py` walks `side_labs/runs/races/`, takes the newest completed lap per course and writes one
`ingest_race.json`: the run's manifest essentials and tool versions, the referee's statistics, the manoeuvre metrics
recomputed from the trajectory, the per-zone splits (time, distance, mean and peak speed and power, energy, altitude
band), the events, the collisions, the closest approach to any obstacle, a 2 Hz downsampled trajectory (with tilt and
body rates when recorded), the videos' names and sizes, the tail of run.log, every other run's state and failure
note, the mock baseline, and a cross-course comparison. That file is what to hand over for analysis; the videos and
scenes stay where they are.

## The race notebooks (`notebooks/races/`)
Each course notebook sets the course up (its builder's parameters in `COURSE_PARAMS`), flies the mock lap (always,
headless, seconds) and, with `GENERATE_VIDEO = True`, the MuJoCo lap and its videos (overview, chase, onboard: the
recorded trajectory through the scene cameras). Every run is written by `races/export.py` to
`races/runs/notebook/<course>/<stamp>_<simulator>/` (ignored by git): `timeseries.csv` (units in the summary),
`events.json`, `summary.json` — status (finished or DNF and why), collisions, resets, lap time, elapsed simulated
time and wall time, the provenance (course, model and configuration SHA-256, seed, timestep, simulator and version,
git commit) and the artifacts. The run is written before any rendering, so a failed render keeps the data and is
recorded under `artifacts.video_errors`.

`race_design_factors` reads only those exports (`export.runs`, `export.read_lap`) and `races/requirements.py` keeps
three views apart: **per course** (measured, no margins), **pooled** (all valid laps end to end, time-weighted) and
the **worst-case envelope** (worst valid course × a stated margin = the requirement). DNF runs are listed and excluded;
rows no run supports (hover thrust / weight, an endurance requirement) are `unavailable`, not numbers. It writes
`designs/data/races/race_design_factors.json` and `race_design_factors_comparison.csv`. The earlier hand-over
(`designs/data/race_design_factors.json`, notebook 51, now removed) is kept as it was; of its four MuJoCo laps only the industrial one
finished (hangar, farm and quarry were DNF, six collisions each), yet their elapsed simulated time is listed as lap
time and their statistics feed the requirements; its "all courses" column is the worst case, and its "hover thrust /
weight required" is the manoeuvre load factor × 1.3 — read it with that in mind. `races/analysis.py`
(`lap_report`, the plots, `design_factors`) is unchanged.

## Tuning the MuJoCo autopilot
The first real laps (the hangar, 2026-10-18) showed the geometric controller lagging the racing line by metres where
the mock lagged by 0.25 m, so the MuJoCo defaults are conservative: schedule `--safety 0.6`, `--tilt-max 60`,
`--lookahead 0.25`, `--omega-pos 2.5`, `--omega-att 18`. Raise them per course once a lap is clean
(`wf R3 --safety 0.8 --tilt-max 70`); the manifest records them and the ingest file carries them under
`referee.controller`.

## Running
```
python -m races list                              # the courses
python -m races quad                              # the reference quad's budget and drive
python -m races build                             # write races/scenes/<course>.xml
python -m races run --course farm --mock --gif    # the mock flyer: stats, map, replay GIF under races/runs/
python -m races run --course farm                 # MuJoCo (pip install mujoco): the same stats and files
python -m races run --course hangar --param floor_gap=5 --collision count
```
The **mock** (`races/mock.py`) is a point mass with the same autopilot reference, the same referee, the same drive
table for the energy, and sphere-against-geometry collisions; it exists to develop and test the courses, the rules
and the notebook without MuJoCo. The **MuJoCo run** (`races/sim.py`) applies the four rotor thrusts (lag, inflow
limit, drag torques) and the frame's drag through `xfrc_applied`, flies Drongo's controller on the same reference,
reads contacts from `data.contact`, and teleports the free joint on a reset. It was written against the scene files
but **not executed in the session that produced it**: expect to tune `Flight`'s gains and the schedule's safety
factor on the first real runs. `races/sim.render` makes camera frames from a recorded trajectory (needs an OpenGL
context: EGL or GLFW); `races/replay.animate` makes a GIF replay from any run without a renderer.

## Smoke tests
`python -m pytest races/tests -q`: the scenes parse, every zone clears its surroundings, the mock completes every
course with no collision, the rules do what they say (a forced crash resets with a penalty, `stop` ends the race),
the CLI lists and builds. Full races are run by hand with the commands above.
