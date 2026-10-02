# Onager — land robots

The **Onager series** are wheeled and wheel-leg land robots on one modular architecture (swappable mission
modules, high autonomy, encrypted and jamming-resistant links, self-diagnostics and redundancy). Everything
wheeled lives here, in its own category, apart from the walkers of `notebooks/` (the robot dog and the Myropods):

```
onager/
  01_onager_sentinel.ipynb   the Onager Sentinel SX-1: design, wheel and walking modes, terrain and vibration, FEA,
                             the actuators, and the MuJoCo patrol with partial failures (movies)
  designs/onager.py          the Sentinel as a Dedalus design (hull, turret, mast, four wheel-legs, the wheel)
  designs/onager_robot.py    the Sentinel in ChironLab: Chiron building blocks, hub motors as velocity servos
  designs/onager_controller.py   wheel-mode drive, heading hold, gravity feed-forward, partial failures, the limp
  designs/onager_scenario.py     the patrol scenario: gravel road + speed bump, two failures, two responses
  designs/tests/             pytest: CAD consistency, standing, driving, failures, the three-wheel limp
  scenarios/onager_patrol.py the patrol as a script: two movies and the per-phase tables
  _runs/, output/, scenarios/output/   what the notebook and the script write (gitignored)
```

The shared libraries stay where they are: the actuator / motor / joint catalogue `notebooks/designs/actuators.py`
(the Onager entries are the industrial joint modules with holding brakes and the 3 kW hub motor) and the two-link
kinematics of `notebooks/designs/gait.py`; the design modules put that folder on `sys.path` themselves.

Run the notebook from this folder (`cd onager && jupyter lab`, or `../jupyter.sh` and open it here). Tests:
`python3 -m pytest -q onager/designs/tests`. The patrol movies: `xvfb-run -a python3 onager/scenarios/onager_patrol.py`
or `./user_tests.sh onager`.

## The Sentinel on patrol (MuJoCo)

`onager/scenarios/onager_patrol.py [--response drag|limp]` drives the Sentinel (`designs/onager_robot.py`: a 408 kg
wheel-leg hybrid, eight 800 N·m joint modules as position servos, four 3 kW hub motors as velocity servos on their
torque–speed lines) over a gravel road with a 120 mm speed bump at 3 m/s; at 6 s the front-left hub motor loses
power (it freewheels), at 9 s the rear-right wheel seizes. Two responses, one movie each:
`onager_patrol_drag.mp4` (keep driving on three motors, the braked tyre skids) and `onager_patrol_limp.mp4` (the
three-wheel limp: the hull shifts 0.4 m forward on the three good legs, the seized wheel lifts 100 mm, speed down
to 1.5 m/s); `onager_patrol.json` has the per-phase tables (speed, heading, tilt, corner loads, wheel torques and
power, the seized wheel's drag). The scenario is `designs/onager_scenario.py`, the same one the notebook §7 runs.
