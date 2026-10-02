# Design files shared by the notebooks

`quad_frame.py` (`QuadFrame`, notebook 08) and `fixed_wing.py` (`FixedWing`, notebook 09) are ordinary
Dedalus designs. The notebooks copy them into their `_runs/...` folder before use, so a copilot
proposal (notebook 07) or a campaign (notebook 10) edits the copy, never this original.

`robot_dog.py` (`RobotDog`, notebook 16) is the quadruped. `myropod.py` (`Myropod`, notebooks 17 and 18) is
the Myropod family's segmented walker — its defaults are Persephone (12 segments, chimney crawler); Cleopatra is
the same class with three large segments (parameters in notebook 18). `apheloria.py` (`Apheloria`, notebook 19)
is the modular Myropod that curls into a ball. The wheeled machines of the Onager series live in their own category, `onager/` (`onager/designs/`).
`actuators.py` is the shared actuator / motor / joint catalogue
(`import actuators as act` from `notebooks/designs`): `act.table()`, `act.get(key)`, `act.select(torque, sf)`,
`Actuator.holding_power()`; it also carries the Onager series' industrial joint modules and hub motor (`onager/`). `gait.py` is the shared gait simulation (`import gait`): `make_terrain`, `body_pose`,
`simulate_steps`, `path_pose`, `PipeContact`, `ik_dog` / `ik_myropod`, `fk_dog` / `fk_myropod`, `torques_dog` / `torques_myropod`, `gait_diagram`.
All on branch `dev_sikarian`.
