# Science — the methods behind Vegeta's numbers, their assumptions and their limits

One page per topic would be too long; this is the map. Each section says what the method is, what it assumes,
how it was checked, and where it stops being true. The per-tool documents (`docs/talos.md`, `docs/aeromant.md`,
`docs/boreas.md`, `docs/zephyr.md`, `docs/chronos.md`, `docs/chiron.md`) hold the details.

A rule across all of them: **every number is labelled by where it came from**: analytic, empirical, solver,
validated, or MOCK. A mocked number (DEBUG, `vegeta.mock`) is never a result. Its `mock` flag is part of every cache
key (`docs/cache_identities.md`), so it can never stand in for a real one.

---

## 1. Structures: finite elements (Talos: Gmsh + CalculiX)

- **Elements.** Gmsh tetrahedra are written to CalculiX as linear **C3D4** or quadratic **C3D10**
  (`vegeta-cli/src/vegeta/talos/mesh.py`). Quadratic is the default: linear tetrahedra lock in bending and are far too
  stiff.
- **Mesh.**
  - The size is set by `element_size` (Gmsh `MeshSizeMax`), with an optional minimum.
  - The 3-D algorithm is Delaunay by default (Frontal and HXT are available).
  - The mesh is optimised, and the high-order nodes are corrected so curved elements do not invert.
  - Curvature-based sizing (`curvature_points`, elements per 2π) refines curved surfaces when asked.
  - There is no adaptive refinement.
- **Quality gate.**
  - Meshing fails if any element is inverted (minimum scaled Jacobian ≤ 0) and warns below 0.1.
  - A tiny element almost always means a CAD feature (a short edge), not a meshing parameter.
- **Mesh dependence, stated honestly.** Peak nodal stresses at supports, point loads and re-entrant corners depend
  on the mesh (they are singular in the continuum).
  - Read the stresses away from those points, or compare two element sizes.
  - There is no automatic convergence study.
  - The element size is part of the result's cache key, so results at different resolutions are never confused.
- **Checked against.**
  - A rotating rod: the centrifugal body load ρω²r against the closed-form reaction mω²r_cg.
  - A steel cantilever's first mode: 209.6 Hz against beam theory.
- **Not modelled:** creep, contact, plasticity, the anisotropy of printed parts (material constants are isotropic).

## 2. Aerodynamics: CFD (Aeromant: OpenFOAM)

- **Mesh.** `blockMesh` (a background box) → `surfaceFeatureExtract` (sharp edges) → `snappyHexMesh` (hex-dominant,
  snapped to the STL, refinement by level near the surface and in the wake) → `checkMesh`.
  - Refinement levels are template parameters: `surface_level`, `rotor_level`, `wake_level`, and the domain size in
    body diameters.
  - There are no prism layers.
- **Physics.**
  - `laminar_external` for low Reynolds numbers and the validation case.
  - `rans_ksst_external`: steady RANS with the k-ω SST model and wall functions.
  - Without prism layers the SST case gives **trends, not absolute drag**.
- **Rotors.** The Multiple Reference Frame (MRF) templates (`rotor_mrf`, `_static`, `_oblique`, `_installed`,
  `aircraft_rotor_disks`) rotate a cell zone around the blades in a steady solve.
  - It is a **frozen rotor**: blade-passing interaction is not captured.
  - The hover case needs a small residual inflow (2 % of tip speed) for the steady solver to settle.
- **Convergence.** A run stops when every initial residual is below `residual_target` (1e-4 to 1e-5 by template).
  The result records `iterations`, `converged` and the final residuals.
  - An unconverged run is reported as such and kept out of corrections.
  - Beyond the stall, a steady RANS result is a time average at best.

## 3. Propellers, motors and batteries (Boreas)

- **Blade-element momentum theory (BEMT).**
  - Per annulus, the induced velocity is iterated to the balance of blade-element and momentum thrust, with swirl
    from the torque; hover is included.
  - Prandtl's tip loss F = (2/π)·arccos(e^−f) is clipped to [0.05, 1] (`vegeta-cli/src/vegeta/boreas/bemt.py`).
- **Motor and battery.**
  - The motor is a first-order brushless model (Kv, winding resistance R, no-load current I0, current limit Imax).
  - The battery is cells × capacity.
  - A point above the current limit is flagged (`current_limited`), never clipped silently.
- **Slipstream:** momentum theory.
- **Skin friction:** Prandtl–Schlichting, Cf = 0.455 / (log10 Re)^2.58.
- **Calibration.** Where CFD of the propeller exists (prop_lab), section-lift and normal-force scales are fitted and
  fed back to the drive tables.
- **Not modelled:** compressor and turbine maps for the microjet, surge, unsteady loads (beyond harmonic estimates).

## 4. Wing sections (Zephyr)

- **Backends:**
  - **NeuralFoil**, a neural surrogate of XFOIL with a confidence per point;
  - **XFOIL**, Drela's panel method with a coupled viscous boundary layer;
  - **thin-airfoil theory** for the zero-lift angle, Cm and flap effectiveness τ.
- **Checked against:**
  - NACA 23012: α₀ −1.09°, Cm −0.013;
  - the reflexed 230/240/250 lines: Cm_c/4 = 0 ± 0.0025;
  - τ(c_f) against the closed form;
  - XFOIL and NeuralFoil agree to ΔCl 0.05 and ΔCd 0.003 in the attached range.
- **Limits:**
  - NeuralFoil past the stall is a trend only.
  - XFOIL's default transition criterion (ncrit 9, a clean tunnel) is optimistic for a built-up small-UAV wing
    (nearer 4–6).

## 5. Stability and control (aircraft)

- **Vortex lattice** (`designs/peregrine_flight.py`): horseshoe vortices on flat panels, Kutta–Joukowski forces,
  induced drag in the Trefftz plane.
  - It gives the lift slope CLα, the induced factor k = CDi/CL², the span efficiency e and the wings' neutral point.
  - The biplane uses 20 spanwise × 4 chordwise panels.
- **Static stability.** The aircraft's neutral point adds the fuselage (Raymer's term) and the propellers' normal
  force to the wings'.
  - Static margin = (neutral point − CG) / mean chord.
  - Positive is stable; it is reported in % MAC.
- **Tailsitter transitions** (`vegeta-cli/src/vegeta/aeromant/transitions.py`, `designs/kestrel_transition.py`):
  - a pitch-plane simulation of hover ↔ wing-borne flight through the stall (α ≈ 90°);
  - the slipstream raises the dynamic pressure on the blown span;
  - rotor thrust has limits and lag;
  - a pitch controller's gains are scheduled on airspeed;
  - a transition passes when pitch is tracked within 10° without tail slide or a height loss beyond the limit;
  - roll and yaw are not simulated (6 DOF is open).
- **Attitude control (multirotors):** a geometric controller on SO(3) (Lee, Leok & McClamroch 2010,
  `designs/drongo_controller.py`). It needs no Euler angles and has no singularity at large tilt.

## 6. Buoyancy, hydrostatics and boat stability

**Not modelled yet.**
- The hulls (`designs/survey_boat.py`, a hard-chine V-bottom; `designs/submarine.py`, a Myring body of revolution)
  can export the outer solid "for hydrostatics". Nothing in the repository yet computes:
  - displacement against waterline;
  - the centre of buoyancy, the metacentric height GM, the righting arm GZ;
  - trim and sinkage.
- The intended method is the textbook one: integrate the submerged volume of the CAD solid at a waterline and
  heel, then GM = KB + BM − KG with BM = I_wp / ∇.
- Free-surface resistance needs an interFoam template (todo 5e.2).
- (The word "buoyant" in `designs/merlin_flight.py` is the Briggs rise of a smoke plume, not hydrostatics.)

## 7. Fatigue and life (Chronos, Talos)

- **Load spectrum** by rainflow counting of the stress history.
- **Damage** by Miner's linear rule on an S-N curve σ = σ_f (2N)^b, with Goodman's mean-stress correction against
  the ultimate strength, and an endurance limit if the material has one. Example: PETG-CF with σ_f 80 MPa,
  b −0.11, ultimate 55 MPa.
- **Assumptions:**
  - linear structure, load patterns acting one at a time;
  - dynamic amplification bounded by one mode;
  - a structural mode within ±20 % of an excitation line is a design problem, not a number.
- **Not modelled:** crack growth, load-sequence effects, temperature and humidity. Stresses at holes are
  mesh-dependent (section 1), so an absolute life must be confirmed by a test.

## 8. Legged robots and the race courses (MuJoCo)

- **Rigid bodies and contact:** MuJoCo, SI units, z up, soft contacts with friction cones.
- **Actuators (Chiron):** servo torque limited by a linear torque–speed line, |τ| ≤ τ_stall (1 − |ω|/ω₀).
- **Races** (`races/`): one reference 5-inch quad (571 g, 2207 1750 KV on 6S, 5x4.3 tri-blade, 1300 mAh).
  - Thrust and power come from its Boreas drive table, and the frame's drag area is 0.016 m².
  - The geometric controller (section 5) follows a racing line.
  - The line's speed schedule comes from the corner speed each zone's clearance allows, braking, acceleration and
    the top speed.
  - A look-ahead "carrot" is the reference point.
- **Rules (the referee, `races/rules.py`):**
  - zones count only in order;
  - checkpoints are reset points;
  - a collision resets to the last checkpoint with a 5 s penalty;
  - more than five collisions, or 300 s, is a **DNF** (did not finish);
  - lap time includes penalties.
- **The mock flyer** (`races/mock.py`) is a point mass with a saturated position–velocity loop under the same referee
  and drive table. It is for developing courses and notebooks, not for numbers.
- **Design requirements** (`races/requirements.py`) keep three views apart:
  - per course (measured, no margins);
  - pooled (all valid laps end to end, percentiles weighted by time);
  - the worst-case envelope × stated margins: speed 1.1, acceleration 1.2, lateral g 1.2, load factor 1.3, body
    rates 1.3, power 1.2, energy 1.25.

  DNF laps never feed a requirement. A requirement no run supports (hover thrust/weight, endurance) is
  "unavailable", not a number.

## 9. Tuning the race autopilot: generalized pattern search

`./tune_autopilot.sh` → `races/converge.py`.

- **Problem.** Minimise a lap score f(x) over the controller's knobs x:

  | knob | meaning |
  |---|---|
  | `safety` | share of the corner speed flown |
  | `lookahead_s` | carrot distance in time |
  | `omega_pos` | position-loop bandwidth |
  | `omega_att` | attitude-loop bandwidth |
  | `tilt_max_deg` | tilt limit |
  | `v_cap` | speed cap |

  Each knob is scaled to [0, 1] between physical bounds. f is a black box: one MuJoCo lap per evaluation,
  deterministic, with no gradient.
- **Score:**
  - a finished lap scores lap time + 2 s per collision;
  - a DNF scores 1000 − 20 × zones passed, so more of the course is better before any lap finishes;
  - a crash of the simulation scores 2000.
- **Method.** Generalized pattern search (GPS: Torczon 1997, *SIAM J. Optim.* 7(1); Audet & Dennis 2003, *SIAM J.
  Optim.* 13(3)). At each iteration:
  1. Poll the 2n points x ± Δ·eᵢ (a positive spanning set of directions) and take the first that lowers f.
  2. On success, move there and expand the mesh: Δ ← min(2Δ, 0.25).
  3. On failure, x is a mesh local optimum: refine, Δ ← Δ/2.
- **Convergence criterion:** stop when Δ < Δ_min (default 1/64 of every knob's range).
  - Torczon's theorem: for a continuously differentiable f, the mesh local optima of the refining subsequence
    converge to a stationary point.
  - For a Lipschitz f (Audet & Dennis), the limit is Clarke-stationary in the polled directions.
  - Our f is **piecewise**: a lap that turns into a DNF jumps. The honest guarantee is therefore a **mesh local
    optimum at resolution Δ_min**. It is not a global optimum, and a different start can find a different one.
  - `--max-laps` is a safety budget only, and is reported if it is what stopped the search. `--patience` is an
    optional heuristic early stop, off by default.
- **Bookkeeping.** MuJoCo is deterministic, so every setting is flown once and remembered. The search starts from
  the course's stored settings (`races/autopilot.json`), and only a faster finished lap replaces them.
- **First result.** Industrial went from 37.7 s (the best of the grids) to **32.0 s with no collision**, converging
  at Δ = 1/32 in 99 laps. `lookahead_s` ended near its upper bound (0.99 of 1.0 s), so a wider bound may still pay.

## 10. Mocks and solver guards

- `vegeta.mock` (on with `DEBUG = True` in every notebook, or `VEGETA_DEBUG=1`) replaces the slow tools with fixed or
  coarse numbers, labelled MOCK:
  - Talos (mesh, solve, modes, fatigue), Aeromant (CFD), Mellonia (slicing), Gmsh meshes;
  - MuJoCo episodes cut short;
  - BEMT, vortex lattice and transitions run coarse;
  - a thin-airfoil polar replaces NeuralFoil.
- Mocks prove that the code paths and the plots run. They never produce a result.
- `VEGETA_NO_SOLVERS=1` (set by `user_tests.sh`) refuses to start CalculiX, OpenFOAM, PrusaSlicer or XFOIL at all.
