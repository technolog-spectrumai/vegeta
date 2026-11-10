# pinna_launch: MuJoCo clips of the launches from Pinna

Pinna, the unmanned drone-carrier catamaran (`labs/marine/designs/pinna.py`), launches three machines. These
clips simulate each launch in MuJoCo with the CAD as meshes, the cards' masses and thrusts, a controller, and the
carrier's deck moving in a simple sea and wind:

| clip | sim window | what it shows |
|---|---|---|
| `gannet_launch` | 0-18 s | Gannet (16.4 kg with its 5 kg crate) locked on the starboard forward pad by the harpoon probe, rotors spinning up, lock released, lift-off following the pad, vertical climb to 30 m, transition to a 15 m/s cruise |
| `gannet_landing` | 60-82 s of the same flight | after a circuit (out 110 m, a turn to port, a downwind leg, a turn to final) the approach from astern into the wind, a hover 5 m over the pad tracking its heave and roll, the descent at 0.6 m/s relative to the deck, touch-down and the harpoon lock |
| `fulmar_launch` | 0-15 s | Fulmar (20.2 kg, 3.6 m) on the catapult's cradle, full throttle, the 2.76 m stroke at 6 g to 18 m/s, release, the climb-out at 10 deg and a turn towards the shore |
| `fulmar_net` | 0-16.5 s | the net recovery: 130 m astern at 1.15 V_stall (16.6 m/s air, 10.6 m/s closing in 6 m/s of wind), the glide path aimed at the net as it moves with the deck, the hook catching, the net's sprung slide absorbing the speed |
| `walrus_launch` | 0-25 s | Walrus hoisted off its deck chocks by the aft port davit robot, the boom luffing outboard, the winch lowering it through the splash zone, the hook releasing it under water, its 8 thrusters diving it away from the hull |
| `overview` | 0-20 s | all three at once: Gannet off its pad, Fulmar off the catapult at 7 s, Walrus over the side |

Run (from `labs/marine/sims`, after `source activate.sh`; `python -m pinna_launch` also finds the Vegeta libraries
from the source tree itself):

    python -m pinna_launch all --out /some/dir             # every clip: <clip>.mp4, <clip>.json, PNG stills
    python -m pinna_launch gannet_launch walrus_launch     # some; default out: labs/marine/sims/output/pinna_launch/videos
    python -m pinna_launch fulmar_net --no-meshes          # boxes instead of the CAD

The videos are 1280x720, 30 fps, H.264 (yuv420p); each clip renders in 1-2 minutes on 4 CPU cores (software GL).
Rendering needs an offscreen OpenGL: `MUJOCO_GL=egl` (Mesa's surfaceless EGL works: `apt install libegl1`) or
`osmesa` (`libosmesa6`); without `MUJOCO_GL` the first that works is used. The JSON next to each video has the run's
numbers (touch-down speed, release speed, peak g, catch speed, net travel, cable pull, depth).

## The model

* `meshes.py`: each design's CAD parts (Pinna's hulls, deck, house, flight deck, pads, mast, robots; Gannet's body,
  arms, rotor discs, skids, crate; Fulmar's FixedWing airframe and pod; Walrus's tracks, frame, housing, gantry,
  basket, thrusters, lights, arms) tessellated from CadQuery to binary STL (mm), cached in
  `labs/marine/sims/output/pinna_launch/meshes` (ignored by git) by a hash of the design's source and parameters.
* `world.py`: the carrier is a body on heave/roll/pitch joints whose position and velocity are prescribed every
  step from a sum of sines (heave +/-0.52 m, roll +/-3.2 deg, pitch +/-1.5 deg; Hs about 1.2 m), so contacts and
  welds see the deck's velocity. The meshes are drawn; simple boxes and cylinders collide (flight deck, pads, main
  deck, hangar, house, hulls, the mast). The water is a translucent slab heaving with the swell, the seabed is at
  10 m. The wind is 6 m/s from the bow (the carrier on DP heads into it) with Ornstein-Uhlenbeck gusts (1.2 m/s).
  The catapult rail, the net (a panel on a sprung slide: 700 N/m, 160 N s/m) and the davits are drawn from
  primitives.
* `gannet_sim.py`: one rigid body, the card's mass, four rotor thrusts along the body z at the CAD's rotor
  positions (first-order motor lag 0.07 s, the card's thrust/weight 3.55, reaction torque 0.018 m x thrust), frame
  drag with the card's CdA; the lab's geometric (Lee) position/attitude controller (as `drongo_controller` and
  `races.flight`) with a mixer; the guidance is minimum-jerk lift-off and climb, a speed-ramp transition, a circuit,
  a decelerating approach and the pad-tracking descent (the pad's motion comes from the carrier's INS). The harpoon
  lock is a weld, on at the start, off at release, made again at touch-down.
* `fulmar_sim.py`: one rigid body, the card's mass and wing (S 1.15 m2, AR 11.3, CD0 0.032, e 0.8, CL max 1.35),
  a Helmbold lift slope with a sharp post-stall blend, induced drag, class-value stability derivatives (assumed),
  elevator/aileron/rudder, two motors' thrust 120 N static falling to zero at 45 m/s (assumed). The autopilot: pitch
  with rate damping (gain-scheduled), flight path from altitude with the lift equation's AoA (not the measured one:
  that runs away near the stall), heading to roll, a yaw damper, throttle on airspeed. The catapult's shuttle runs
  a prescribed constant-acceleration stroke (v^2 / 2s from the card) and the cradle's weld opens at its end; the
  net's hook is a weld made on contact with the net panel.
* `walrus_sim.py`: one rigid body, the card's vehicle mass and its volume from the card's sink weight (2.9 kg heavy
  in sea water), buoyancy times the submerged fraction at a centre of buoyancy above the CG, quadratic drag per axis
  and rotational damping, a 0.2 m/s current; the cable is a MuJoCo tendon with a length limit (a rope) the winch
  sets every step from the boom's sheave, the boom's luffing angle is prescribed (the davit robot's trajectory). The
  eight thrusters are the card's (`walrus.thruster_layout`, layout C) and the controller allocates the PD wrench
  with the card's allocation matrix (`walrus.allocation`, pseudo-inverse), each thrust clipped to its maximum.
* `video.py`: the free camera, a PIL overlay (title, clock, the HUD lines each agent writes, the sea and wind), a
  blue fog and tint when the camera is under water, H.264 through ffmpeg. `clips.py`: the clips and their cameras.

Test: `labs/marine/tests/test_pinna_launch.py` builds each model, steps it and checks the lock, the climb, the
catapult's launch speed, the davit's lift and the approach (no rendering).

## What the simulation found, and what it leaves out

* **The catapult points the Fulmar at the autonomy mast.** In `pinna.py` the rail lies on the centreline and the
  mast's pole stands on the centreline 8.4 m ahead of the rail's end, 4.7-10.2 m high; the Fulmar leaves the rail at
  7 m and cannot climb over it in 0.5 s; yawed 12-15 deg its starboard wing still meets the radar bar. The sim yaws
  the rail 18 deg to port (`world.CATAPULT_YAW_DEG`); the CAD should move the mast off the centreline or turn the rail.
* The tether reels are drawn 0.8 m further inboard than the CAD's (where the Walrus's deck cradle would be).
* The catapult's peak is 6.9 g against the stroke's 6.0 g: the cradle's weld is a stiff spring and rings at the
  start of the stroke.
* Not modelled: rotor inflow and ground effect over the deck, the carrier's air wake, the propellers' gyroscopic and
  H-forces, the Walrus's added mass and slamming, waves (the sea surface heaves flat with the swell), the net's
  mesh (a rigid panel on a spring), the cable's elasticity and drag. The carrier's motion is prescribed, not solved.
