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

`merlin.py` (`Merlin`, notebook 26, branch `dev_merlin`) is the wildfire sampler: `FixedWing` subclassed (its `_wing`, `_tail`
and sections), a fuselage of its own and `propulsion` = `edf` (25's duct and stators from `EDFHousing.duct_profile` /
`vane_section`, the fuselage running through the nozzle: an annular jet), `tractor` or `pusher`; `part` = aircraft / wing /
nose; `layout`, `fuselage_profile`, `wetted_areas`, `drag_buildup` (component build-up), `edf_housing_params`, `outline`.
`merlin_flight.py` (`import merlin_flight as mf`) is its race and mission: `Airframe` (parabolic polar), `Unit` (net thrust and
shaft power tabulated over airspeed × rpm; full throttle at a shared electrical limit) from `open_propeller` (BEMT with an
effective wake and thrust deduction) or `ducted_fan_unit` (`boreas.ducted` less nacelle friction and jet scrubbing),
`installation` / `installation_pod` (MERLIN's fuselage, wing and tail as `air_propeller`'s pod and pylon), `build_merlins`
(the three aircraft, the pitch picked per race), `performance`, `race`, `Plume` (Gaussian smoke: Briggs rise,
Pasquill–Gifford spread, CO from the heat release, `particles` for drawing), `source_estimate` (the fire's size from the
passes), `Forest`, `fly` → `Episode`, `profile_figure`, `render_movie` (the smoke as particles, a map and the sensor trace).

`propulsor_maps.py` (`import propulsor_maps as pm`, branch `dev_rave`) is how the propulsion notebooks hand their results to the
reduced flight models, which then solve nothing: `build_library(kind)` (parallel) solves a design space over airspeed × rpm —
`propellers` (notebook 25: two and three blades, 9–12 inch, pitch 0.6–1.2 D, free-stream maps plus MERLIN's installation w
and t as tractor and pusher), `exotic` (notebook 25c: six and twelve blades) or `edf` (notebook 25b: 70–120 mm, pitch
1.4–2.2 D, nozzle exit 0.65–0.9, catalogue or well-made duct, one to three stages, plus the lossless bound); `save` writes
`data/<kind>_maps.json` (committed; rebuild with `scenarios/propulsor_maps.py`); `load` merges the files that exist;
`unit(entry, power, layout=...)` gives a `merlin_flight.Unit` by interpolation (the installation applied, the jet scrubbing
MERLIN's fairing taken off); `unit_mass_kg`, `merlin_airframe` (from notebook 26's design export when given), `fan_object`,
`prop_object`, `find`. `merlin_race.py` (notebook 27) races the libraries on MERLIN: `round_trip`, `study`, `winners`.
`merlin.Merlin` takes the fan's `edf_diameter`, `edf_pitch` and `edf_exit_area_ratio`; `merlin_flight.load_design` reads notebook
26's export (`data/merlin_design.json`) and `build_merlins(design=...)` puts the library's propulsors on that airframe.

`peregrine.py` (`Peregrine`, notebook 28, branch `dev_peregrine`) is the falcon-inspired farm drone: MERLIN's tractor body
(`Merlin` subclassed) with `FixedWing`'s wing, three wings in `VARIANTS` (A large fixed, B small fixed, C hinged), and for the
hinged wing `fold_deg` (each outer panel rotated aft about a vertical hinge at `hinge_y_frac` of the half span, on the spar at
`hinge_x_frac` of the local chord, with a hinge knuckle); `part` = aircraft/wing/hinge (the printed hinge lug)/nose (the motor
fairing). Beside the CAD: `resolve({"variant": ...})`, `planform` (the lifting surfaces as quads for the lattice, the folded
span, the hidden part of a folded panel), `fold_limit` (the fold at which the panels collide), `exposed_wing_area`,
`wetted_areas`, `drag_buildup` (MERLIN's Raymer build-up on the folded geometry plus a hinge allowance) and `outline` (side and
top views). `peregrine_flight.py` (`import peregrine_flight as pf`) is the rest: `vlm` (a vortex lattice in streamwise strips,
Trefftz-plane induced drag), `aero`, `cl_max`, `fold_polar`; `BOM_ITEMS`, `mass_table`, `bom`, `ENGINEERING_HOURS`,
`engineering_cost`; `stability` (CG, neutral point, margin, trim); `drive` (a `merlin_flight.Unit` from notebook 25's library
with a fixed no-load + avionics power), `Aircraft` / `make_aircraft` (an `Airframe` per fold), `turn`, `envelope`,
`envelope_table`; `BIRDS`, `bird_table`; `stoop`, `terminal_speed`; `gap_passage`; `prop_hang`, `hand_throw`,
`min_throw_speed`, `roof_drop`, `perched_wind`; `hinge_loads`; `Farm`, `Flock` (boids with a fear radius), `herd`, `mission`
→ `Episode`, `profile_figure`, `render_movie`; `decision_table`, `weight_sensitivity`.

`turbojet.py` (`Turbojet`, notebook 28, branch `dev_jet`) is a model turbojet's parts:
- the impeller: radial-element blades with an inducer lean, on a hub of revolution with a spinner and a shaft bore;
- the turbine wheel: a cast disc with twisted blades;
- the engine's outside: bell-mouth, casing, nozzle and tail cone.

All are sized from a `vegeta.boreas.microjet.Microjet` by `sized(engine)`. `compressor_surfaces(p, dir)` writes the four
STLs of Aeromant's `compressor_mrf`: impeller, shroud (inlet duct, casing over the tips with the gap, vaneless diffuser),
inlet face and outlet face. It returns a `location_in_mesh` in the passage. `passage_profile`, `masses`.

`aguya.py` (`Aguya`, notebook 29) is the turbojet sampler: `FixedWing`'s wing, an ogive sensor nose, a fuselage tank, the
engine pod on a pylon and a V-tail. `for_engine(engine)` sizes the pod and nozzle from the cycle. Also: `tank_volume_l`,
`drag_buildup` (Raymer, with Mach corrections), `cfd_surfaces(p, dir)` (the body, intake-face and nozzle-face STLs of
`jet_external`) and `outline`.

`aguya_flight.py` (`import aguya_flight as F`) is the mission with fuel burn:
- `JetUnit` (thrust and fuel flow over airspeed × shaft speed), from the cycle (`jet_unit`) or notebook 28's export
  (`jet_unit_from_export`);
- `JetAirframe` / `airframe(p, unit)`;
- `top_speed`;
- `fly`: start, catapult, climb, accelerate, dash, 3 min of passes, return, parachute;
- `fuel_for`: the smallest fuel load that keeps the reserve;
- `size_for`: the thinnest fuselage whose tank holds it;
- `race_table`.

`pekari_rover.py` (`PekariRover`, notebook 30, branch `dev_track`) is the small tracked rover: the hull with a sensor
block and the payload basket on its roof (the default configuration), two track modules (side frame, rear drive
sprocket of two toothed discs, front idler, road wheels in pairs on bogies, a return roller, the belt as a band), and
the parts on their own — sprocket, idler, road wheel, one hinged track link (knuckles, pin bores, grouser, guide horn)
and the gearbox's final-stage sun pinion (involute teeth). Static methods keep CAD and mechanics in step:
`track_center_y`, `track_gauge`, `sprocket_pitch_radius`, `road_wheel_x/z`, `contact_length`, `belt_circles`,
`belt_length` (the convex hull of the wrapped circles), `link_count` (links and the idler's take-up), `overall`.

`pekari_rover_robot.py` (`import pekari_rover_robot as prr`) holds the design's numbers: `DESIGN`, `CAD`, `MATERIALS`,
`PARTS_KG`, `ASSUMPTIONS`, `DRIVE` / `MOTOR` (catalogue keys), `PAYLOAD_KG`; `geometry` (SI), `mass_budget`, `masses`
(each mass with its position, the battery placed to balance the CG over the road wheels), `cg` (empty and loaded).
`pekari()` builds it for Chiron (MuJoCo): each track is a row of 11 rollers (sprocket, idler, 7 ground rollers every
60 mm, one on each inclined run), the road wheels on bogies with passive hinges, and every roller a velocity servo at
its side's belt speed. `pekari_lab`, `rollers`, `roller_joints` and `LAB_OPTIONS` go with it.

`pekari_controller.py` drives it (notebook 30 §7):
- `TrackDrive(legs)`: skid steering with a heading hold, over `Leg("straight", m)` / `Leg("turn", deg, radius=…)`;
- `MISSION`: 3 m, a 90° turn on 1 m, 2 m;
- `uneven_ground()`: rough soil with a log and stones;
- `run`, `timeseries` (path, heading, tilt, side belt forces, slip, power) and `leg_table`;
- the trials of §7.2 (`TRIALS`, `run_trial`, `trial_rules`, `trial_table`): `micro_hills` (an egg-crate of hills the
  rover's radius, sized to 15° slopes; `max_slope_deg`), `steep_hill` (up a 30° ramp and down), `mud_flat` with
  `MudHook` (the rollers' friction drops to the mud's, Bekker compaction and viscous drag on the hull, a sideways
  drift of the soft layer); each runs under `FailureRules`, so the episode and its movie end at the failing frame,
  and `end_card` writes the outcome on the last frame.

`terramechanics.py` (`import terramechanics as tm`) is the soil: `SOILS` (Bekker–Wong values from Wong's tables and
handbook μ / C_rr for hard ground), `pressure_sinkage` / `sinkage`, `track_sinkage`, `compaction_resistance_track`,
`thrust_track` (Janosi–Hanamoto, Wong's closed form), `max_thrust_track`, `drawbar_pull_track`, `mmp_rowland`, and the
rigid wheel's `wheel_sinkage`, `wheel_compaction_resistance`, `thrust_wheel`.

`gears.py` is the drive train: `involute_profile` (the CAD pinion's construction), `planetary` (ratio, assembly and
neighbour conditions), `select_ratio` (top speed against the continuous climbing torque on a motor's line),
`lewis_bending`, `contact_stress` (Hertz, external and internal meshes), `stage_loads`, `MATERIALS` (ISO 6336 class
allowables), `chain_efficiency`, `sn_cycles`, `sprocket_chordal` (the polygon effect) and `capstan`.

`tracks.py` is the track: `ground_pressure`, `resistance` (compaction or rolling, internal, grade, drag), `tractive_limit`,
`road_wheel_loads` (rigid frame or bogies), `sag`, `derail_tension`, `drive_tension`, `skid_steer` (Wong's turning
resistance; clutch-brake and regenerative power), `min_turn_radius`, `stability` (tip-over, step, trench).

`drongo.py` (`Drongo`, notebook 08b, branch `dev_potato`) is notebook 08's `QuadFrame` with a landing gear of two skids and
a parallel pincer (servo housing, rail, two jaws with TPU pads at `grip_height` above the ground when landed); `part` =
drongo / frame / gear / gripper / jaw / pad. `drongo_robot.py` (`import drongo_robot as dr`) holds its numbers (`DRONGO`,
`CAD`, `mass_budget`), the items (`POTATO`, `CREAM`: masses, sizes, friction, squeeze / impact / net-arrest limits), the
`NET`, `GRIP_N`, the grip and net arithmetic (`grip_needed`, `grip_holds`, `net_catch`), the propulsion (`propulsion()`:
notebook 08's Boreas export when it exists, else `ASSUMED_PROPULSION`), the Chiron robot (`drongo()`: the jaws as servo
slides, the skids as feet) and `Rotors` (a scene hook: rotor thrust with spin-up lag and limits, moments, drag torque,
airframe drag, energy). `drongo_controller.py` is the flight controller (`Flight`: position loop, SO(3) attitude, X
mixer with saturation; `Profile`) and the mission (`Mission`, `Plan`, `delivery(scene, plan)`: land over an item, grip,
climb, drop into the net or lower onto the zone until the descent stalls, home). `drongo_scenario.py` is the garden
(`Scene` with the supply basket — an open cube Drongo lands in over each item —, `scenery`, `item_props`), the judge (`Watch`: squeeze, slip, release, net catch with the 5 m rule, impacts,
the people taking each item to the table), `make_lab`, `run`, `timeseries`, `phase_table`, `deliveries`, `wait_times`,
`time_budget`, `sweep` and `render_movie` / `stills`.

`nisus.py` (`Nisus`, notebook 31, branch `dev_nisus`) is the twin-boom single-pusher survey drone: `FixedWing`'s wing on a
lofted elliptic pod, a main spar and a rear carry-through tube, carbon booms in printed root fittings that clamp both tubes,
an H-tail with two fins, the motor mount, trays and a keel skid (`part` = aircraft/wing/spar/rear_spar/pod/nose/boom/boom_fitting/
tail/tail_fitting/motor_mount/tray/skid/battery_tray, and the FEA's `*_fea` variants); `Nisus.layout` holds the stations, areas,
volumes and clearances, `bays` the component bays; `planform`, `wetted_areas`, `drag_buildup`, `outline`, `exploded_parts` and
`boom_check` the rest. `nisus_systems.py` is the two variants' components (with sources), mass tables, CG and inertia, electrical
loads, the propulsion requirement and map (`data/nisus_propulsion.json`), batteries and the mission energy; `nisus_flight.py` the
lattice aerodynamics, the tagged derivatives, trim, CG range, envelope and flight checks; `nisus_structure.py` the load cases, hand
checks and Talos models; `nisus_cfd.py` the Aeromant cases; `nisus_drawings.py` the drawings; `nisus_robot.py` /
`nisus_controller.py` / `nisus_scenario.py` the aircraft in ChironLab (the `Aero` hook: forces from the derivative table, the
propulsion map, actuators, wind), its flight controller (attitude, speed/height, line following, the energy manager, the OBS
scripted pilot, the Zero waypoint stream and its failure) and the missions with their movies. `nisus_birds.py` (notebook 32)
is the seam between that aircraft and the mission software `vegeta.mission`: `BirdMission` (a Chiron hook: the
navigation state out, the birds and the mission stack stepped, the guidance in for the controller's mode 'birds'),
`BirdScenario` / `bird_scenarios`, `run`, `summary`, `photo_table`; `nisus_birds_movie.py` the bird-mission video.

`nisus_plus.py` (`NisusPlus(Nisus)`, notebook 33, branch `dev_falco`; formerly FALCO) is the bigger mountain aircraft in NISUS's style: a 2.4 m
two-piece wing with a spar joiner, cut flaps and ailerons (posed for crow by `flap_deg` / `aileron_deg`), `spar_joiner`,
`flap` and `aileron` parts, `transport_check` and `planform_split`. `nisus_plus_systems.py` holds the atmosphere (ISA + ΔT,
Sutherland), the Li-ion pack model, the signed drive map with the windmill and brake region (`NisusPlusDrive`,
`data/nisus_plus_propulsion.json`) and the mission energy; `nisus_plus_flight.py` the aerodynamics with crow, the envelope against
altitude, the descent and regeneration tables; `nisus_plus_structure.py`, `nisus_plus_cfd.py`, `nisus_plus_drawings.py` as NISUS's;
`nisus_plus_robot.py` (`Massif` terrain, `MountainWind`, `NisusPlusAero`), `nisus_plus_controller.py` (TECS, crow and brake, terrain
following) and `nisus_plus_scenario.py` the mountain missions and movies; `nisus_plus_birds.py` / `nisus_plus_birds_movie.py` the bird hunt.
`frame_study.py` (notebook 34) compares NISUS+'s twin booms with a single tail for FALCO, the tractor redesign:
`frames` (the variants: the booms moved apart for 18-22" pushers, the single tube), `hand_table`, `wing_loading`,
`fea_models` / `solve_cases` / `fea_table` / `modes_table` (Talos), `aero_table` (the aerodynamic efficiency of each
variant), `performance_table` (speed and acceleration), `drive_for_prop` (the larger propellers' maps,
`data/frame_study_drive_*.json`), `verdict`, `sketch`.
`falco.py` (`Falco(NisusPlus)`, notebook 35) is the tractor built on the study: MERLIN's round fuselage in NISUS+'s
section rows (`pod_profile`), the propeller at the nose, one tail tube in a keel socket (`tail_tube`, `tail_socket`,
`tail_fitting` parts), one dorsal fin, the parked-propeller and blade-down ground clearances in `layout`, `outline`
with `side.prop_parked`. `falco_systems.py`: `COMPONENTS` (the AT5220-A KV220 with its bench table, the APC 20x15E, the
Hall position sensor), `PACKS` with the 8S3P, `motor_model` (fitted to the published point), `drive_options`,
`drive` (`data/falco_propulsion.json`), `structure_items`, `mass_table` / `cg_inertia` through NISUS+'s seams.
`falco_flight.py`: `aero` / `derivatives` / `cg_range` (one fin), `installation` / `installation_table` (the tractor's
wake fraction and thrust deduction), `airframe`, `propeller_trade`, `compare_with_nisus_plus`. `falco_structure.py`:
`load_cases` (+ gyroscopic, fin torsion, blade strike), `joints_hand` (the socket, the tube in bending and torsion,
the stabiliser spar, the fin rod, the firewall), `tail_frame` / `tail_tube_cases` / `solve_tail_tube` (the frame study's
FEA on FALCO's tube), `tail_socket_case`, `motor_mount_case`, `models`. `falco_cfd.py` (the nose rotor disk),
`falco_drawings.py` (`three_view`, `park_clearance`, `internal_layout`, `crow_view`, `transport_view`, `exploded_png`),
`falco_robot.py` (`falco_robot`, `FalcoAero`: the `park` command channel and the `parked` log column),
`falco_controller.py` (`FalcoController`: the flare parks the propeller), `falco_scenario.py` (`SIM`, `scenario`,
`standard_scenarios`, `run`, `outcome` with the blade-strike check, `park_table`), `falco_birds.py`.

`kestrel_study.py` (notebook **43** `43_kestrel_study`, the recommended Kestrel entry point): `prerequisites` (the table of
required / optional inputs — prop_lab, the drive tables, the sections, the airframe CFD and FEA runs by explicit
path — with provenance, compatibility and status; one `PrerequisiteError` naming every missing required input's
command), `build` / `build_all` (a variant's cheap numbers: mass table, polar, neutral point, hover, cruise table,
transition vehicle), `variants_table`, `hover_analytic`, `envelope_vs_tools`, `biplane_check`, `correlations_table`
(`CORRELATIONS`: source, applicability, uncertainty of every empirical input), `cfd_sweep` (a run read against the
polar, labelled `validated` only when converged), `stall_decision`, `trim_table`, `control_table`, `structure_screen`,
`fea_table` / `fea_against_frame`, `transitions` (the decision-relevant episodes only), `deceleration`, `performance`,
`energy_budget`, `recommended_campaigns`, `assessment`, `Stopwatch` (the measured runtime). `kestrel_campaigns.py`
(shared by the notebook and `workflows/kestrel/`): `VARIANTS`, `params` / `vehicle`, `fingerprint` / `compare` (the
design identity a run is made for and the tolerance it is read with), `CFD_CAMPAIGNS` / `cfd_specs` / `cfd_case` /
`cfd_outcome` / `cfd_summary`, `FEA_CAMPAIGNS` / `fea_cases` / `fea_outcome` / `fea_summary`, `import_legacy`,
`check_run` / `require` / `command` / `find_runs`, `records_frame` / `airframe_correction`; nothing in it runs a solver
(`workflows/tests` checks). `workflows_runkit.py` re-exports `workflows/runkit.py` for the design modules.
`kestrel_compare.py`: the other Vegeta machines (FALCO, NISUS+, MERLIN, PEREGRINE, from their design files) scaled to
Kestrel's mass and flown at its cruise speed, and Kestrel as a monoplane (same span and area; same span and chord) —
biplane against monoplane in one table (`compare`, `verdict`); `kestrel_flight.interference` / `prandtl_sigma`: the wing-interference
measure (the lattice's gap sweep, Prandtl's σ, the monoplane equivalents), the CFD chapter K12 (`wing-interference`:
one wing, the pair, the equal monoplane) measures it with RANS and `kestrel_cfd.interference_factors` reads the run.
`kestrel_fast.py` (notebook **44** `44_kestrel_fast`, standalone): `Candidate` (span, chord, gap, section, prop, motor,
battery, bracing, spar…) → `params`, `section_polar` (Zephyr, cached), `propulsion` (Boreas points), `spar_rules` /
`post_rules` (beam and Euler rules of thumb), `evaluate` / `compare` / `ratios` / `verdicts` / `plot_ratios`,
`default_candidates`; no solver. Notebooks 39–42 are in `notebooks/old/` (unchanged; see its README).

`kestrel.py` (`Kestrel4Q`, notebook 40, branch `dev_kestrel`) is the zero-stagger biplane quadrotor tailsitter: the
CAD (parts `aircraft`, `airframe` — the CFD body without the propellers —, `wing`, `pod`, `truss`, `nacelle`, `tail`;
`angle_of_attack_deg` turns the build about `rotation_x`), `layout` (metres, no CAD kernel needed), `checks` (the
clearance rules), `rotate` (a point turned like the build). `kestrel_systems.py`: `COMPONENTS`, `mass_table` (the
battery placed for the CG), `cg_inertia`, `propulsion` (Boreas), `drive_table` (`data/kestrel_drive.json`: full thrust,
κ = Q/T and power against the inflow), `rpm_for`, `hover_point`. `kestrel_control.py`: `SPINS`, `allocation`,
`rank_table`, `attainable` (a linear programme per axis), `hover_authority`, `motor_out`, `hover_wind`,
`roll_authority`. `kestrel_flight.py`: `aero` (the biplane in `peregrine_flight.vlm`), `neutral_point`,
`prop_cn_alpha`, `drag_buildup` (`faired=`), `Polar` (the full-range polar), `polar`, `cruise`, `stall_speed`.
`kestrel_transition.py`: `make_vehicle`, `slipstream`, `airframe_forces`, `simulate` (forward / back, `t_tr`,
`headwind`, `gust_up`, `V_entry`, `floor`), `matrix`, `verdict`, `CRITERIA`. `kestrel_cfd.py`: `points` (from the
transitions), `case` / `plan` (Aeromant `aircraft_quad_rotor_disks`, the bare sweep on `rans_ksst_external`), `table`,
`not_run_table`, `CFDAirframe` (the CFD's correction for the transition model; `from_results` / `from_records`
return `None` when nothing usable solved, `source="sweep"` corrects in α from the bare airframe), `table` with the
statuses `solved` / `diverged` (`SANE`) / `FAILED` / `NOT RUN` and a `note`, `summary`, `sweep_comparison` /
`sweep_numbers` (the bare sweep against the model: lift slope, CL max, Cd0, static margin from Cmα), `save_records` /
`load_records` / `plug_in` / `records_table` (the campaigns in `data/kestrel_cfd_results.json`, keyed `4Q-1`, `4Q-2`,
`sections`; a pass that solved nothing never erases a result), `section_table` / `save_section_table` (the slabs).
`kestrel_loop.py` (the design loop's bookkeeping): `envelope` (iteration 0 from the spec and the layout), `TARGETS`,
`check` / `check_table` / `failing`, `IterationLog`. Notebook **39** (`old/39_kestrel_tests`, historical) is the CFD campaign of every
case for both aircraft and the sections, failures accepted; `kestrel_cfd.flight_points` adds the level cruise and the
stall to its points. `prop_lab.py` (notebook **38**, `38_prop_lab`): the 9×6 alone in CFD — `geometry` / `cad` / `stl`
(Dedalus's propeller on the rotor frame), `bemt`, `configurations` (hover, axial, oblique on `rotor_mrf_static` /
`rotor_mrf` / `rotor_mrf_oblique`), `plan`, `table` (beside BEMT and De Young, statuses `solved` / `diverged` /
`FAILED` / `NOT RUN`), `summary`, `fit_section` (`cl_scale`, `cd_scale`), `fit_normal_force` (`cn_factor`), `export` /
`load` / `calibration` (`data/prop_lab.json`); `kestrel_systems.prop_lab_calibration` / `blade_airfoil` apply it to
`propulsion()` (the drive key stamps it), `kestrel_flight.prop_cn_alpha(calibrated=)` the normal force; `noise` /
`noise_table` (Gutin's tones on the measured loads, the broadband, dB(A) at `LISTENER`), `signal` / `signal_rotors` /
`signal_sweep` / `write_wav` / `render` (the WAV files: one rotor, the four rotors beating in the hover, the
transition glide). `kestrel_fea.py` (Talos, notebook 40 § 8b / 40): `frame_cad` / `frame_step` (the carbon frame as
hollow tubes per bracing), `joint_cad` (the printed spar–post clamp), `lift_segments`, `frame_cases` (A / B / C /
modes), `joint_cases`, `cases`, `solve(run=)`, `table` (statuses, nominal and peak stress, margins, across-layer
tension, frequencies), `summary`, `plane_frame_reference`, `save` (`data/kestrel_fea_results.json`).
`kestrel_report.py`: `export(name, title, sections)` — a notebook's tables, series, dicts and verdicts as JSON
(`data/reports/<name>.json`: `plain`, `write_json`, `dossier_json` → `data/kestrel_dossier.json`, `load_json`) and
as rounded, trimmed Markdown (`write`, `dossier` → `data/kestrel_dossier.md`, `paste(name)`). `kestrel2.py` (notebook
42): `vehicles` (4Q-1 / 4Q-2 / 4Q-3 from 38's recipe), `REMEDIES` / `remedy_matrix` / `remedy_verdict` (the back
transition's remedies: `kestrel_transition.simulate(airbrake_deg=, overshoot_deg=, reverse_N=, descent_entry=)`),
`decelerate_level`, `performance` / `pursuit` (endurance, range, V max, climb, dive, acceleration, turns, authority),
`propose_4q3`. `kestrel_structure.py`: `frame`,
`compare_bracing`, `stiffness_and_modes` (per bracing: the motor stations' symmetric and antisymmetric stiffness and
first modes), `ground_stance`. `kestrel_drawings.py`: `front_view`, `side_view`, `episode_figure`, `path_figure`,
`polar_figure`, `fairing_figure`, `front_view_4q2`, `xwing_front_view`. Part II: the pod's `nose_length` / `tail_length` / `pod_wall` and the payload bay (`bay_*`; `Kestrel4Q.pod_profile`,
`bays`, `bay_checks`, part `capsule`; `kestrel_systems.mass_table(capsule=)`, `kestrel_drawings.internal_layout`); the CAD's `elevons` / `elevon_chord` /
`elevon_inner_y` / `elevon_deg`, `strut_fairing` (`fairing_section`), `bracing` (`x` / `posts` / `none`);
`kestrel_control.elevon_hinge` / `elevon_authority`; `kestrel_transition.Vehicle.panels` (`biplane_panels`: the lifting
panels, each at its dihedral, height and rotor — the model flies any arrangement of them), `z_rotor`, `elevon_y_in`,
`lift_slope_vertical`; `kestrel_section.py` (on `vegeta.zephyr`: `CANDIDATES`, `table` → `data/kestrel_sections.json`,
`summary_frame`, `trim_table`, `cfd_specs`) and `kestrel_cfd.section_stl` / `section_case` / `section_not_run_table`;
`kestrel_xwing.py` (notebook 41: `VARIANTS`, `layout`, `checks`, `mass_table`, `quads` / `aero` on the lattice,
`drag_buildup`, `cantilever` — the spar sized on stress, deflection and the first mode —, `stance`, `make_vehicle`,
`cruise`).

The laser design modules (`svarog*.py`, `kimla*.py`) moved to Alabastron (https://github.com/technolog-spectrumai/alabastron) on 2026-10-18.