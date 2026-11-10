# Pekari-2M and Pumba — the Pekari family of civilian response platforms

Two interoperable tracked platforms on the Pekari Rover's architecture, for confined-space inspection, industrial
handling, debris removal, logistics and emergency response — and a deterministic heterogeneous-fleet layer for them.
No weapon mounts. The colours are rescue/industrial: safety orange (Pekari-2M) and signal yellow (Pumba) arms on
graphite hulls.

| | **Pekari-2M** (light, dual-arm) | **Pumba** (heavy, dual-arm, stabilizers) |
|---|---|---|
| module | `labs/land/designs/pekari_2m_robot.py` | `labs/land/designs/pumba_robot.py` |
| CAD | `pekari_2m.Pekari2M` | `pumba.Pumba` |
| mass empty / with nominal deck load | 55 kg / 85 kg (30 kg) | 444 kg / 594 kg (150 kg) |
| envelope, arms folded (L × W × H) | 0.90 × 0.58 × 0.42–0.45 m | 1.65–1.73 × 1.15 × 0.83–0.90 m |
| deck payload (target) | 30 kg nominal (20–40 kg) | 150 kg nominal (100–200 kg) |
| arms | 2 × 5-DOF + gripper, 0.97 m reach, 5 kg at 0.5 m | 6-DOF heavy arm (1.92 m, 60 kg at 1.2 m) + 5-DOF aux arm (1.23 m, 8 kg at 0.7 m) |
| tracks | 100 mm, 6 road wheels a side | 200 mm, 8 road wheels a side |
| top speed (flat) | 1.4 m/s (5 km/h, design) | 2.2 m/s (8 km/h) |
| battery / endurance (mixed profile) | 48 V 0.96 kWh / ~4.8 h | 48 V 4.8 kWh / ~3.4 h |
| 20° grade loaded | yes (peak drive), traction ok | yes (peak drive), traction ok |
| parts cost (class estimate) | €20–47 k | €68–143 k |

The ranges in the envelope row are the CAD's measurement and the conservative estimate in `capabilities()`.
Every number above comes from code in this repository, and `python3 labs/land/sims/pekari_family_checks.py`
regenerates it (see §8 for the open validations).

![The two platforms to scale](figures/pekari_family_scale.png)

---

## 1. Architecture assessment (what existed, what is reused)

**Implemented before this work** (inspected on branch `dev_pumba`, forked from the newest branch tip):

* **Pekari Rover** (`pekari_rover.py`, `pekari_rover_robot.py`, `pekari_controller.py`, notebook 30). It is a ~20 kg
  tracked rover with a 5 kg payload: 560 × 300 × 140 mm hull, 80 mm tracks of hinged PA6 links, a rear drive
  sprocket, a front idler and 4 road wheels on 2 bogies, two 100 W 16:1 track drives, a stereo camera and a 2D
  lidar. It has a parametric Dedalus/CadQuery design, a mass budget and CG from CAD × density, track, gear and
  terrain calculations from Ogma (`ogma.tracks`, `ogma.gears`, `ogma.soil`), and a Chiron (MuJoCo) model with
  tracks as rows of velocity-servoed rollers. Its track-drive controller (`TrackDrive`) runs missions over uneven
  terrain, hills and mud. It is registered for export (YX2/USD/MJCF through Yax). **It has no manipulator**: the
  basket on the roof is its payload deck.
* **The variant pattern on top of Pekari.** `prr.pekari(...)` takes keyword overrides: parameters, CAD numbers,
  parts, drive, extra children under the hull, extra masses and their moment in the battery placement, and the
  payload position. **Perseus** (`perseus_robot.py`, ~180 kg branch cutter) and **Perseus Mini** already build on
  it, so a heavier Pekari derivative with a boom was a solved pattern.
* **Manipulators.** The Onager Manus (`onager_manus_robot.py`) has two 4-joint arms with pincers: analytic IK/FK,
  catalogue joint modules and Chiron links. Perseus has a boom driven by electric cylinders, with static torques,
  workspace, tip-over and track-load checks. The conventions (yaw on a pedestal, pitch joints about y, elevation
  = −q) are reused unchanged.
* **Standard parts** in `libs/ogma` (actuator catalogue, tracks, motors, batteries), **Chiron** (MuJoCo
  robots, labs, terrains, episode logs), **Dedalus** (parametric CAD), **Yax export** (`designs/machines.py`), and
  `tools/lab.py`.

**Not present (confirmed by searching the repository):** any multi-robot, fleet, task-allocation, shared-map,
communication or supervision layer. No WOTAN, MIMIR, Wagenburg or similar subsystem exists. Chiron simulates
**one** robot per MuJoCo world. There was no aerial-drone ground-fleet integration and no ROS interface in this
repository (Piccolo, the ROS-side project, is separate and receives standard files only).

**Reused, not replaced:**
* `pekari_rover_robot` for both chassis: geometry, mass budget, battery placement, roller-track model, Chiron
  builder.
* `PekariRover` CAD as the base class of both CAD models.
* `pekari_controller.TrackDrive` drives both new machines unchanged.
* The Manus arm conventions.
* `ogma.actuators` (four entries added), `ogma.tracks`, Chiron, Dedalus and the Yax export.

The Pekari Rover's own files are untouched; the test `test_pekari_rover_is_untouched` checks its numbers.

**Added:**

| file | what |
|---|---|
| `designs/pekari_arms.py` | one arm description for the family: kinematics (5–7 DOF), statics, rated load vs reach, workspace, power, the Chiron link tree |
| `designs/pekari_family.py` | `Platform` (chassis + mounts + stabilizers + parts), mass/CG in 3-D, support polygon and tip-over, lift capacity, drive check, power/endurance, capabilities, BOM, the Chiron robot, a MuJoCo stand test |
| `designs/pekari_2m_robot.py`, `pumba_robot.py` | the two platforms (parameters, stored CAD numbers, parts, assumptions) |
| `designs/pekari_2m.py`, `pumba.py` | their CAD (Dedalus): arms in named poses, stabilizers stowed / deployed |
| `designs/fleet.py` | heterogeneous fleet coordination (deterministic) |
| `designs/fleet_scenario.py` | the mixed-fleet civilian response scenario |
| `sims/pekari_family_checks.py`, `sims/fleet_response.py` | runners: numbers, figures, MuJoCo stand tests, the scenario |
| `tests/test_pekari_family.py`, `tests/test_fleet.py` | 51 tests on mocks (MuJoCo and CadQuery ones skip without them) |
| `libs/ogma/src/ogma/actuators.py` | `harmonic 12 Nm, brake`, `harmonic 30 Nm, brake`, `cycloidal 1600 Nm, brake`, `pumba track drive 4 kW, 48 V` (class values, `assumed`) |

## 2. The reference: Stohid Technology WILK

Manufacturer-reported values, treated as such and not verified here:

| | value |
|---|---|
| mass empty | ~180 kg |
| payload | 200 kg |
| top speed | 8 km/h |
| size (L × W × H) | 1700 × 1170 × 700 mm |

They are in `pumba_robot.WILK_REFERENCE`.

Public sources found for this work are trade-press articles in Polish (defence24.pl, MSPO coverage). They describe
WILK as a light, modular, battery-electric UGV for carrying ammunition, equipment or water forward. It is a
Polish-built development of a system used in Ukraine, with a "China-free" component set.

**The sources disagree about its running gear.** One article describes it as wheeled. Another, about the newer
RYŚ, contrasts RYŚ as "developed as a wheeled vehicle, unlike WILK". The brief treats WILK as tracked. I could not
check photographs or a datasheet in this session, so the analysis below is about principles, not WILK's geometry.

Principles taken (no proprietary geometry copied):

* **A low, wide chassis.** Width 0.69 × length and height 0.41 × length give a low CG over a wide track gauge.
  Pumba has 1.15 / 1.65 = 0.70 and a hull roof at 0.50 m. Its CG is 0.35 m empty and 0.41 m loaded, and the
  folded tip angle is 52°.
* **The battery low between the tracks**, placed to balance the machine. Pumba's is placed by
  `prr.masses` to put the empty, folded machine's CG over the road wheels' centre. It is two swappable 50 Ah
  modules.
* **A flat equipment deck with standard fixing points**, the payload over the contact patch's centre. Pumba's
  600 × 380 mm deck sits behind the arms' pedestals, so 150 kg on it moves the CG only 0.08 m back.
* **Sensors high at the front, protected:** a sensor box on the nose, plus cameras at the wrists.
* **Maintenance access from the top:** roof hatches over the battery and electronics, and swappable battery
  modules. The track tensioner is on the idler slide, as on Pekari.

Pumba is heavier than WILK (444 kg against ~180 kg empty) for three reasons:
* **Arms and stabilizers, 150 kg.** WILK carries neither.
* **Running gear.** Pekari's architecture uses moulded PA6 wheels and links (~100 kg of running gear at this
  size), where WILK's lightness suggests a lighter band track.
* **The steel subframe under the main arm.**

The chassis alone is ~294 kg. The track and frame masses are the first numbers to bring down (§8).

## 3. Pekari-2M — the light response platform

Pekari-2M is the dual-arm version of Pekari; this work does not build a single-arm Pekari version. Pekari itself
(20 kg, 80 mm tracks, 100 W drives) is the wrong size for 20–40 kg plus two arms:
* Its 300 mm-wide roof has no room for two pedestals beside a deck.
* Loaded to ~64 kg, its two 16:1 drives reach their continuous rating at about 11° (`ogma.tracks.resistance`,
  gravel).
* The ground pressure under its 80 mm tracks rises to ~9 kPa.

So Pekari-2M is Pekari's architecture one size up, with the parameters in `pekari_2m_robot.CHASSIS`:
* 700 × 340 × 160 mm hull
* 100 mm tracks with 36 mm pitch
* 6 road wheels on 3 bogies a side
* the Perseus Mini's 20:1 track drives

* **Arms** (`pekari_arms.PEKARI_ARM`, two). Each has 5 DOF: yaw, shoulder, elbow, wrist pitch and wrist roll,
  on strain-wave modules with brakes (60 / 30 / 30 / 30 / 12 N m). The gripper is a parallel-jaw gripper on an
  ISO 9409 quick-change flange; other tools (hook, suction cup, probe, cutter) fit the same flange.
  - Reach is 0.97 m; each arm weighs 7.1 kg.
  - **5.6 kg at 0.5 m** at peak torque / 1.5 (rated 5 kg) and 2.3 kg at 0.8 m. Continuously (the rated torque,
    moving) it carries about half of that; holding still, the brakes carry the load.
  - Folded, the upper arm points back along the deck side and the forearm folds forward on top of it, grippers
    forward: 0.42 m high (CAD).
* **Sensors:** RGB-D at the front and at each wrist. A 3D lidar and a thermal camera are options (their mass is
  budgeted either way). The IMU is in the electronics.
* **Payload:** 30 kg on the deck. The drives hold it on a 20° grade at peak torque; continuously the limit is
  about 16°.
* **Confined spaces:** 0.58 m wide (a 0.8 m door) and 0.45 m high folded (a 0.55 m pipe rack). It turns on the
  spot (skid steer), and the ground pressure is 5.9 kPa loaded.
* **Interfaces kept:** the Pekari joint names (`L_sprocket` … `R_ramp_front`, bogies), `pekari_controller`, the
  CAD part names and the export path. Arm joints are `L_yaw … L_grip_a/b` and `R_…`.

## 4. Pumba — the heavy derivative

Pumba uses the same software, 48 V bus, arm conventions, joint naming, controllers, CAD base and export as
Pekari-2M (the test `test_shared_interfaces`). The chassis (`pumba_robot.CHASSIS`):
* 1300 × 700 × 330 mm hull
* 200 mm tracks with 60 mm pitch
* 8 road wheels on 4 bogies a side
* 15-tooth sprockets (r = 144 mm)
* a steel subframe under the arm pedestals

Two 4 kW track drives (class values) give 8 km/h.

**Arms** (two by default):
* **main** (`PUMBA_MAIN_ARM`): 6 DOF (yaw, shoulder, elbow, wrist pitch, **wrist yaw**, wrist roll) on cycloidal
  and strain-wave modules (800 / 1600 / 800 / 400 / 150 / 60 N m), with a two-jaw heavy grab on a quick coupler.
  - Reach is 1.92 m; the arm weighs 80 kg.
  - **60 kg at 1.2 m** and 76 kg up to 1.0 m at peak / 1.5; 37 kg at 1.6 m and 25 kg at full reach.
  - The joint torques at the rating are 1059 N m at the shoulder (stall 1600), 503 at the elbow (800) and 216 at
    the wrist (400).
* **aux** (`PUMBA_AUX_ARM`): the light arm one size up, 5 DOF, 1.23 m reach, 14 kg; 8 kg at 0.7 m and 15 kg at
  0.5 m. It holds, steadies or cuts while the main arm lifts.
* The arms sit at the front corners and fold back along the deck's sides.

**Stabilizers:** four swing-out legs with 12 kN electric cylinders, pads at (+0.80 / −0.75, ±0.80) m, 14 kg each.
They are deployed for lifts at reach or on soft ground.

### Calculations

**Unvalidated assumptions.** Masses come from tube lengths and catalogue entries, not weighed CAD or real
products. The actuator entries added here are class values. The stability analysis is static, on level ground.
Each assumption is listed in `Platform.assumptions` and `ArmSpec.assumptions`.

| quantity | Pumba | how |
|---|---|---|
| mass | 444 kg empty, 594 kg with 150 kg | `pf.mass_budget`: chassis 294 kg (CAD × density + parts), main arm 80, aux arm 14, stabilizers 56 |
| CG (x, y, z) | (0.00, 0.035, 0.35) m empty; (−0.08, 0.03, 0.41) loaded | `pf.cg`; y ≠ 0: the main arm is heavier than the aux |
| support | contact 1.32 × 1.15 m; with pads a 1.55 × 1.60 m polygon | `pf.support_polygon` |
| stability folded, 200 kg on the deck | margin 0.55 m, tip angle 52° | `pf.stability` |
| lift, main arm sideways, 1.2 m | arm 61 kg; tips at 235 kg (474 kg with pads) | `pf.lift_capacity`; capacity = min(arm, 0.75 × tipping) |
| lift, main arm sideways, 1.6 m | arm 37 kg; tips at 150 kg (279 kg with pads) | the arm limits at every reach; the pads add margin for dynamics and slopes |
| drive | 20° loaded: 2350 N needed, 5410 N peak, 3260 N traction; continuous 2190 N (≈ 18°) | `pf.drive_check` (`ogma.tracks.resistance`, gravel) |
| ground pressure | 11 kPa loaded | soft-soil class (WILK's would be lower) |
| power | 390 W driving at 0.8 m/s, ~3 kW both arms working, 120 W electronics; 1.2 kW average on the mixed profile | `pf.power` (50 % driving, 30 % manipulating, 20 % idle) |
| endurance | ~3.4 h on 4.8 kWh (85 % usable) | `pf.endurance` |

![Lift capacity](figures/pekari_family_lift.png)

![Workspaces](figures/pekari_family_workspace.png)

## 5. Heterogeneous fleet (`designs/fleet.py`)

There was nothing to reuse, so `fleet.py` is new. It is small and **deterministic**: the same inputs give the same
allocation, events and result, so the scenario is a regression test. Learned multi-agent policies are not used;
they would be measured against this baseline.

* **Capability-based task allocation.** Agents carry a `Profile`: capability tags, deck payload, arm lift, width,
  folded height, speed, endurance and sensor range. Pekari-2M and Pumba profiles come straight from
  `pekari_family.capabilities`. A task names its tags, payload, lift, dependencies (`after`) and whether it needs
  approval. Allocation is a sequential single-item auction: the cheapest eligible (agent, task) pair is assigned
  first, the cost is travel time plus work time divided by priority, and ties break on ids. The path is planned
  lazily only for the winning pair.
* **Width- and height-aware planning.** A* runs on a 0.2 m grid. A robot passes where the free width (from a
  Euclidean distance transform) is at least its width + 5 cm and the headroom is at least its height + 5 cm. That
  is how "only Pekari-2M fits under the pipe rack" and "only Pumba lifts the debris" come out of the numbers, not
  rules.
* **Shared mapping and situational awareness.** Each agent maps with range sensing (rays stop at walls; the drone
  maps outdoors from above). The coordinator merges connected agents' maps cell by cell, the newest observation
  winning (time stamps), so cleared debris propagates. Exploration targets frontier clusters, and frontiers no live
  agent can reach are given up and retried when a route opens.
* **Coordinated exploration and logistics:** explore tasks are generated from the shared map. Transports go pickup
  → work → deliver. A clear-debris task is raised when debris is seen, and dependent tasks wait for it.
* **Dynamic reassignment:**
  - A failed agent's task is released after `T_RELEASE` (90 s) of silence and re-auctioned.
  - A blocked path makes the agent replan, then release the task.
  - A task an agent dropped out of contact is freed when it reconnects.
* **Communication-loss recovery:**
  - The mesh has a link when distance + 6 m per wall cell is within 40 m, and agents relay.
  - An agent out of contact finishes its task and keeps the result. After `T_RETURN` (60 s) idle it goes back to
    its last connected point, then home.
  - On reconnection it reports work done offline. If the coordinator had handed that task to someone else, the
    duplicate is cancelled.
* **Human supervision and emergency override:** the `Supervisor` commands are applied before anything else each
  tick:
  - emergency stop (all or one agent; latched until reset)
  - hold / release
  - approvals, including pre-approval of a task not yet reported
  - add or cancel a task

  The e-stop travels on a **separate safety radio**, modelled as independent of the mesh, as industrial radio
  e-stops are.
* **Future agents:**
  - `drone_profile()` is a survey quadcopter that maps and inspects from above, outdoors only.
  - `third_party_profile()` maps another maker's capability vocabulary onto the fleet's tags; unknown names are
    dropped, never invented.
  - The coordinator does not care what kind of agent it is.

## 6. Simulation and visualisation

* **Chiron (MuJoCo) models of both platforms** (`pf.robot`): the Pekari roller tracks with the platform's drives
  and bogies, the arms as servoed joint trees with jaw grippers, and Pumba's stabilizers as slides with pads. Arm
  poses are `folded` / `deployed` / `reach`. Collision geometry: hull boxes, roller cylinders (tracks), arm tube
  boxes, palm and jaws (the pedestals and beams are visual). Each model's mass equals its budget (tested).
* **Stand tests** (`pf.stand_test`, MuJoCo, 1–1.5 s):

  | case | tilt | weight on the tracks | arm joint error |
  |---|---|---|---|
  | Pekari-2M, folded | 0.004° | 100 % | 0.6° |
  | Pekari-2M, deployed, 5 kg in each gripper | 0.007° | 100 % | 1.4° |
  | Pumba, folded | 0.001° | 100 % | 0.3° |
  | Pumba, deployed, 60 kg + 8 kg | 0.002° | 100 % | 0.9° |
  | Pumba, as above, pads down | 0.002° | 73 % (the pads take the rest) | 0.9° |

  The arm errors are without gravity feed-forward.
* **Driving:** both drive straight at 0.8 m/s with Pekari's `TrackDrive`, unchanged (tested). Pumba's roller
  gain is scaled with its mass.
* **CAD** (Dedalus): `Pekari2M` and `Pumba` with `arm_pose` (folded / deployed / reach) and `stabilizers` (none /
  stowed / deployed). The figure at the top shows them to scale from the CAD.
* **Mixed-fleet civilian response scenario** (`fleet_scenario.py`, `sims/fleet_response.py`). The fleet is two
  Pekari-2M, one Pumba and a survey drone at a collapsed industrial hall. The run takes 392 s and covers 100 % of
  the building:
  - Pekari-2M **P1** delivers 25 kg of first aid and water to a casualty (48 s).
  - The drone surveys the roof and yard (127 s).
  - Pekari-2M **P2** goes under the 0.55 m pipe rack into the control room for the thermal inspection, and has a
    drive fault at 40 s. After 90 s of silence its task is released, and P1 is assigned it as soon as it is back
    in contact (221 s); P1 finishes at 329 s.
  - The operator pre-approves the debris removal (60 s). Pumba clears the collapsed doorway (133 s), closes the
    plant-room valve with two arms out of radio range (done offline at 202 s, reported at 221 s), then carries the
    120 kg pump in (391 s).
  - The operator's emergency stop (240–270 s) freezes every agent (tested).
  - Without the drone the ground fleet still finishes.

![The shared map at the end of the scenario](figures/fleet_response.png)

**What the simulation does not do yet:**
* The fleet scenario is kinematic (1 s ticks on a grid). It is not one MuJoCo world with all robots, because
  Chiron simulates a single robot per world.
* Failed robots do not block passages.
* Grasping in MuJoCo is not exercised; the loads are point masses in the palms.

## 7. Incremental development plan

Each step keeps the previous ones working and adds tests; ✔ = done in this work.

1. **Pekari-2M manipulator integration**
   - ✔ Shared arm module: IK/FK, statics, rated load, workspace, Chiron links.
   - ✔ Two arms on the 2M, CAD poses, MuJoCo stand and hold tests.
   - Next: gravity feed-forward in an arm controller (the Manus/Perseus controllers' `qfrc_bias`); a pick-and-place
     mission with real MuJoCo contacts (grasping a box from the deck); wrist-camera models.
2. **Pumba mechanical and simulation model**
   - ✔ Chassis, arms, stabilizers, mass/CG/stability/lift/drive/power, CAD, MuJoCo stand and drive tests.
   - Next: the deflection of the main arm's tubes and the subframe (Talos FEA); track and pin stresses at 600 kg (a
     steel-rubber or band track decision); a MuJoCo tip test on a side slope with the arm at reach; brake-holding
     under power loss.
3. **Shared robot interfaces**
   - ✔ One 48 V bus.
   - ✔ The same joint naming and arm conventions.
   - ✔ `capabilities()` as the fleet contract.
   - ✔ The export of both through Yax.
   - Next: message schemas (status, assignment, map tile, e-stop) as dataclasses with a ROS 2 mapping when
     Piccolo takes the machines; a tool-flange ID for interchangeable grippers.
4. **Multi-robot coordination**
   - ✔ Deterministic allocation, shared map, exploration, reassignment, comms recovery, supervision, drone and
     third-party profiles.
   - Next:
     - battery-aware scheduling with charging
     - multi-robot tasks (two robots on one load)
     - robots as obstacles (deconfliction)
     - a learned policy only as a challenger to this baseline
5. **Simulation scenarios and automated tests**
   - ✔ The response scenario and its variants.
   - ✔ 51 tests on mocks (`python tools/lab.py test land`).
   - Next:
     - a notebook (Pekari-2M / Pumba, like notebook 30 / 37)
     - scenario sweeps (fault times, comm ranges)
     - the grasping and tip scenarios in MuJoCo
6. **Hardware feasibility, BOM and costs**
   - ✔ `pf.bom`: €20–47 k for a Pekari-2M and €68–143 k for a Pumba (class prices, parts only, no labour).
   - Next: quotes for the strain-wave and cycloidal modules (60 % of the arm cost), the track drive, the battery;
     a bench test of one arm joint at its rated torque.

## 8. Validation status and open items

**Checked here.**
* Code consistency is tested: CAD ↔ parameters, the stored CAD ↔ a rebuild, the budget ↔ the MuJoCo masses, IK ↔
  FK, ratings ↔ drive limits, the support polygon ↔ hand geometry.
* The physics is consistent with Ogma's track, soil and drive models.
* MuJoCo statics and driving are in the tables above.

**Not validated.** Nothing is measured. Each item below is unvalidated:
* the new actuator entries (class values)
* the arm tube masses (no weighed CAD)
* the deflection and strength of the arms and Pumba's frame
* track wear and pins at Pumba's load
* the internal track loss (Pekari's own open item)
* the power profile and the 85 % usable battery
* the WILK figures (manufacturer-reported)
* the BOM prices (estimates)

**Pumba is heavier than its WILK reference.** Reducing the running gear and the frame is the first design
iteration, if the 180 kg class matters more than the second arm and the stabilizers.
