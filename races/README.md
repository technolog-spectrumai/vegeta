# Races — four drone race environments in MuJoCo for the reference 5-inch quad

Four courses, one reference aircraft, one autopilot, one set of rules, a mock flyer for development, a thin CLI
and notebook **50** (`notebooks/50_four_races.ipynb`). MuJoCo does the rigid body and the contacts; the aircraft's
aerodynamics and propulsion are the existing models (the Boreas drive table of the 6S 5x4.3 2207 drive, the frame's
drag area) and the flight controller is Drongo's geometric controller (`notebooks/designs/drongo_controller.Flight`).
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
`notebooks/designs/data/mini_raptor_drive_6S_5x4p3.json` (thrust limit and electrical power against axial inflow;
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
`wf R1`–`R4` (`workflows/races/race.py --chapter <course>`) run a course under the run contract: a run directory
under `workflows/runs/races/<course>/` with the manifest, the case `race` (trajectory, stats, map) and the cases
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
