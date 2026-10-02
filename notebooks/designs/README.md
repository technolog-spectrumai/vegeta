# Design files shared by the notebooks

`quad_frame.py` (`QuadFrame`, notebook 08) and `fixed_wing.py` (`FixedWing`, notebook 09) are ordinary
Dedalus designs. The notebooks copy them into their `_runs/...` folder before use, so a copilot
proposal (notebook 07) or a campaign (notebook 10) edits the copy, never this original.

`robot_dog.py` (`RobotDog`, notebook 16) is the quadruped. `myropod.py` (`Myropod`, notebooks 17 and 18) is
the Myropod family's segmented walker — its defaults are Persephone (12 segments, chimney crawler); Cleopatra is
the same class with three large segments (parameters in notebook 18). `apheloria.py` (`Apheloria`, notebook 19)
is the modular Myropod that curls into a ball. `onager.py` (`OnagerSentinel`, notebook 20) is the Onager series'
wheel-leg reconnaissance unit; `onager_robot.py` / `onager_controller.py` / `onager_scenario.py` are the Sentinel
in ChironLab (the robot from Chiron's building blocks with velocity-servo hub motors, the wheel-mode drive with a
heading hold, gravity and contact-force feed-forward and scripted partial failures — motor off, seized wheel,
locked knee — and the patrol scenario with its two responses; `PhasedMission` runs a list of phases on top of the
drive). `onager_atlas.py` / `onager_atlas_robot.py` / `onager_atlas_scenario.py` are the forklift (notebook 21) and
`onager_manus.py` / `onager_manus_robot.py` / `onager_manus_controller.py` / `onager_manus_scenario.py` the two-arm
variant with pincers (notebook 22); `onager_sweeper.py` / `onager_sweeper_robot.py` / `onager_sweeper_controller.py` /
`onager_sweeper_scenario.py` / `onager_sweeper_cfd.py` the street cleaner (notebook 23: disc broom, suction hood with
its Aeromant `suction_hood` CFD case and litter classes, the Manus arms loading a basket; the scene's `Vacuum` hook
applies the CFD's air drag to the litter). All reuse the chassis through `onager_robot.onager(...)`'s variant options
and their scenes use Chiron's props, welds and hooks. `actuators.py` is the shared actuator / motor / joint catalogue
(`import actuators as act` from `notebooks/designs`): `act.table()`, `act.get(key)`, `act.select(torque, sf)`,
`Actuator.holding_power()`; it also carries the Onager series' industrial joint modules, hub motor, arm and jaw drives, and (``LINEAR``, ``get_linear``) the Atlas's lift and tilt screws. `gait.py` is the shared gait simulation (`import gait`): `make_terrain`, `body_pose`,
`simulate_steps`, `path_pose`, `PipeContact`, `ik_dog` / `ik_myropod`, `fk_dog` / `fk_myropod`, `torques_dog` / `torques_myropod`, `gait_diagram`.
All on branch `dev_sikarian`.

`velutina.py` (`Velutina`, notebook 24, branch `dev_velutina`) is the mountain medical courier: a slim body with the ogive
nose as the removable medical capsule (with a grab handle for the hand-over), four short arms with pusher propellers near
the tail, four fins, the parachute bay in the tail cone (`part` = aircraft/body/arm/capsule/fin, `angle_of_attack_deg` for
the CFD). `velutina_flight.py` is its reduced flight model (`import velutina_flight as vf`): `Terrain` (an abstract mountain
valley: depot, rescue site, the rock face's direction), `Aircraft` (masses, drag areas, the Boreas hover point and thrust
limit, navigation error and loop response), `Wind` (steady with altitude shear plus a gust process), `Plan` (set-down or
hand-over, clearances, holds), `simulate` → `Episode` (phase table, events, energy, station-keeping error),
`touchdown_statistics`, `parachute_descent` (analytic), `render_movie` (matplotlib 3D frames → MP4 with the watermark),
`profile_figure`. `scenarios/velutina_mission.py` re-runs the mission from the notebook's recorded design. `velutina_inspection.py` (`import velutina_inspection as vi`) is Velutina v2: `Turbine` (a half-real parked
onshore turbine: tower, nacelle, three blades with a chord distribution, parked azimuths, `blade_point`, `blade_frame`),
`SiteWind` (shear and the tower's wind shadow on top of `Wind`), `Camera` (pixel size on the blade, largest standoff,
scan speed from blur and overlap, dwell time from a station-keeping error series), `InspectionPlan` and `inspection_path`
(four passes per blade with suspect points), `follow_path` → `InspectionEpisode` (the same point-mass model following
waypoints with dwells; table, track error), `station_keeping_at_blade`, `flights_needed`, `render_movie`, `wind_figure`.

`air_propeller.py` (`PropPod`, notebook 25, branch `dev_crazy_prop`) is a propeller's motor pod with its pylon in the CFD frame of
Aeromant's rotor templates (axis +x through the origin, the pylon up along +y), placed for `layout` = `tractor` (pod behind the
disc) or `pusher` (pod ahead); `outline` gives it as polygons for the particle movies, `pylon_wake` the pylon's wake alone as
a Boreas `WakeField` (Silverstein's airfoil wake behind a pusher's pylon), `installation_wake` the whole inflow the blades work
in — the potential flow of the closed bodies (pod + propeller hub as slender-body line sources, the pylon as a closed source
sheet) for both layouts, plus the viscous wakes behind them (the pusher) — and `installation_drag` the drag the running
propeller adds to pod and pylon (the thrust deduction: the pressure field of a loaded actuator disc from the blade-element
thrust distribution, exact disc solid angles in `disc_solid_angle`, on the pod's profile and the pylon's thickness, plus the
scrubbing of the faster air); `synthesize` / `write_wav` a propeller's tones and broadband as sound files on one common scale.

`ducted_fan.py` (`EDFRotor`, `EDFHousing`, notebook 25, branch `dev_crazy_prop`) is the electric ducted fan the open propellers are
compared with (90 mm, 12 blades, hub/tip 0.45, 7 stator vanes). `EDFRotor` is `dedalus.examples.Propeller` in its own CAD frame
(axis Z, upstream -Z) with the blade tips trimmed to the cylinder `r = D/2` and an elliptic spinner on the upstream hub face;
turned +90° about +Y it stands in the CFD frame of Aeromant's rotor templates. `EDFHousing` is the standing part in that frame
(axis +x, the rotor centre at the origin, flow along +x, mm) as one solid: the duct (a thick rounded lip, convex outside and
no higher than the nacelle, with a `lip_radius` bell-mouth inside; the shroud `D/2 + tip_clearance` over the rotor and the
stators; an area-ruled convergent nozzle whose flow area falls to the exit, `exit_area_ratio` of the fan annulus, so the exit is
its throat; a uniform wall outside), the centre body (the motor housing, `axial_gap` of free space behind the rotor — the hub
face or the blade-root stub behind it — then a pointed tail cone) and the radial NACA 00xx stator vanes `stator_gap` behind the
rotor's trailing-edge plane (`rotor_trailing_edge`, from the blade stations), cut to the annulus between the bodies so they sit in
both at any stagger. The housing takes the rotor's blade parameters under the same names (`SHARED`). `housing_geometry` gives the
numbers the Boreas ducted-fan and fan-noise models need (fan, exit and throat areas as built, shroud and nacelle radii, wetted
areas, lengths, stator and rotor positions); `outline` the housing as polygons for the particle movies.
