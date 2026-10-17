# Kestrel — the design study (notebook 43)

*43_kestrel_study — exported 2026-10-17 09:36*

| key | value |
|---|---|
| main | 4Q-2 |
| mass [kg] | 1.65 |
| runtime_s | 130 |
| screening | yes |
| CFD | {'4Q-1': 'solver (unconverged: indicative)', '4Q-2': 'solver (unconverged: indicative)'… |
| FEA | NOT RUN |


## prerequisites

| input | role | selection | status | provenance | compatibility | usable |
|---|---|---|---|---|---|---|
| prop_lab calibration (notebook 38) | optional | notebooks/designs/data/prop_lab.json | NOT RUN or rejected: Boreas's generic blade section and De Young's normal force | 38_prop_lab |  | no |
| drive tables (Boreas) | required | notebooks/designs/data/kestrel_drive.json | on file for this motor, propeller, battery and calibration | kestrel_systems.compute_drive_table | keyed on motor, propeller, battery, calibration | yes |
| section polars (Zephyr / NeuralFoil) | optional | notebooks/designs/data/kestrel_sections.json | NeuralFoil polars on file | kestrel_section.table | keyed on candidates, flap chord, n_crit | yes |
| airframe CFD, 4Q-1, sweep | optional | workflows/runs/kestrel/airframe-sweep/legacy_20261016_4Q-1_39_kestrel_tests | completed: 29 of 29 solved, 0 converged | imported from kestrel_cfd_results.json (39_kestrel_tests, 2026-10-16, screening, 400 it) | warnings: fingerprint assumed from the current source: the legacy results file recorded… | yes |
| airframe CFD, 4Q-2, sweep | optional | workflows/runs/kestrel/airframe-sweep/legacy_20261016_4Q-2_39_kestrel_tests | completed: 29 of 29 solved, 1 converged | imported from kestrel_cfd_results.json (39_kestrel_tests, 2026-10-16, screening, 400 it) | warnings: fingerprint assumed from the current source: the legacy results file recorded… | yes |
| airframe CFD, 4Q-3, sweep | optional | — | not selected |  |  | no |


## parameters that differ

|  | Kestrel-4Q-1 | Kestrel-4Q-2 | Kestrel-4Q-3 |
|---|---|---|---|
| bracing | — | posts | posts |
| strut_fairing | — | yes | yes |
| elevons | — | yes | yes |
| elevon_chord | — | — | 0.32 |
| elevon_inner_y | — | — | 250 |


## layout [m, m²]

| layout [m, m²] | value |
|---|---|
| span | 1 |
| chord | 0.14 |
| gap | 0.28 |
| h | 0.14 |
| x_le | 0.2 |
| x_te | 0.34 |
| x_ac | 0.235 |
| x_spar | 0.242 |
| S_wing | 0.14 |
| S_ref | 0.28 |
| AR_wing | 7.14 |
| gap_chord | 2 |
| gap_span | 0.28 |
| R_prop | 0.114 |
| A_disc | 0.041 |
| motor_y | 0.39 |
| prop_x | 0.125 |
| prop_ahead_of_le | 0.075 |
| prop_gap | 0.0514 |
| prop_to_pod | 0.221 |
| prop_beyond_tip | 0.0043 |
| pod_length | 0.47 |
| pod_width | 0.11 |
| pod_height | 0.28 |
| post_length | 0.28 |
| diagonal_length | 0.437 |
| x_feet | 0.54 |
| fin_height | 0.19 |
| ventral_height | 0.15 |
| fin_tip_chord | 0.09 |
| fin_root_x | 0.28 |
| foot_rod_length | 0.256 |
| elevons | yes |
| elevon_cf | 0.28 |
| elevon_y_in | 0.275 |
| elevon_span_side | 0.225 |
| x_hinge | 0.301 |
| S_elevon | 0.0353 |
| diagonals | 0 |
| posts | 2 |


## mass groups [kg]

|  | propulsion | structure | systems | payload | margin |
|---|---|---|---|---|---|
| Kestrel-4Q-1 | 0.736 | 0.625 | 0.16 | 0.105 | 0.06 |
| Kestrel-4Q-2 | 0.736 | 0.542 | 0.208 | 0.105 | 0.06 |
| Kestrel-4Q-3 | 0.736 | 0.542 | 0.208 | 0.105 | 0.06 |


## mass, CG, inertia (main)

| key | value |
|---|---|
| mass_kg | 1.65 |
| x_cg | 0.215 |
| y_cg | 5.25e-19 |
| z_cg | -0.00458 |
| Ixx | 0.113 |
| Iyy | 0.0232 |
| Izz | 0.104 |
| battery_x | 0.22 |
| battery_clipped | yes |


## hover (Boreas)

|  | thrust_N | rpm | current_A | electrical_W | throttle | figure_of_merit | T_max_N | T_W | kappa_m | total_W |
|---|---|---|---|---|---|---|---|---|---|---|
| Kestrel-4Q-1 | 4.13 | 6.71e+03 | 6.65 | 55.2 | 0.56 | 0.565 | 12 | 2.89 | 0.0162 | 221 |
| Kestrel-4Q-2 | 4.05 | 6.64e+03 | 6.53 | 53.5 | 0.554 | 0.565 | 12 | 2.95 | 0.0162 | 214 |
| Kestrel-4Q-3 | 4.05 | 6.64e+03 | 6.53 | 53.5 | 0.554 | 0.565 | 12 | 2.95 | 0.0162 | 214 |


## hover (momentum theory)

| key | value |
|---|---|
| thrust per rotor [N] | 4.05 |
| disc loading [N/m²] | 98.6 |
| induced velocity [m/s] | 6.34 |
| ideal power, one rotor [W] | 25.7 |
| electrical at FM 0.6, η 0.8, four rotors [W] | 214 |
| Boreas FM | 0.565 |
| Boreas electrical, four rotors [W] | 214 |


## correlations

| correlation | used for | source | applicability | uncertainty | section |
|---|---|---|---|---|---|
| Prandtl's biplane factor | induced drag of the two wings against one monoplane of the same span | Prandtl, NACA TR 116 (1924); Munk's stagger theorem | equal spans, zero stagger, gap/span 0.1–0.5 (here 0.28); the lattice checks it | ±5 % on the induced drag | § 5 |
| momentum theory with a figure of merit | hover power per rotor: P = T^{3/2} / sqrt(2 ρ A) / FM | Glauert; FM from Boreas BEMT (generic blade section, prop_lab calibration when on file) | uniform inflow, no interference between the four discs, no ground effect | ±15 % on the hover power (the FM of a 9-inch propeller is 0.5–0.65) | § 4 |
| drag build-up | Cd0 from wetted areas: turbulent flat-plate skin friction with form factors, cylinders … | Hoerner, Fluid-Dynamic Drag; Prandtl–Schlichting Cf; Cd 1.1 for a tube at Re ~5e3, 0.12… | Re 5e4–5e5, attached flow; no gaps, no protuberances | ±20 % on Cd0 (the campaign's bare sweep read 0.8 × the model's) | § 5 |
| Oswald efficiency | e = 0.90 × the lattice's inviscid span efficiency | a rule of thumb for built-up small wings | attached flow | ±10 % on the induced drag | § 5 |
| Viterna post-stall extension | the wing's forces beyond the stall, blended to a flat plate at 90° | Viterna & Corrigan (1981), the flat-plate normal force with cd90 from the aspect ratio | time-averaged post-stall loads of a straight wing; **not** a model of stall onset, hyst… | ±30 % beyond 20°; the transition verdicts inherit it | § 5, § 9 |
| section polars (NeuralFoil) | cl_max (×0.9 for a built-up wing), α₀, the elevon's flap effectiveness τ, the section d… | NeuralFoil surrogate of XFOIL (Sharpe, 2024) at n_crit 6 | Re 5e4–1e6, smooth sections; the reflexed line's confidence 0.84–0.91, the 0012's 0.96 | ±0.1 on cl_max, ±15 % on cd; XFOIL or a slab CFD is the check | § 5 |
| De Young's propeller normal force | the propellers' destabilising normal-force slope in the neutral point | De Young, J. Aircraft 2 (1965); prop_lab's cn_factor when on file | advance ratio below ~1.5, small inflow angles | ±30 %; it moves the static margin by ~0.01 chord | § 6 |
| Glauert's actuator disk | the slipstream velocity the blown spans see in the transition model | momentum theory, uniform jet, fully developed at the wing | a wing close behind the disk sees less than the fully developed Δv | ±20 % on the blown lift | § 9 |
| plane frame and Euler buckling | spar stress and deflection, strut forces, buckling margins | beam and truss statics; Euler columns pin-ended | linear elastic, the joints ideal; the FEA case checks it | the joint stiffness is not in it: ±25 % on the deflection | § 8 |
| material allowables | carbon tube 400 MPa (compression, knock-down 0.8), PETG 45 MPa in plane / 27 MPa across… | supplier data and printed-coupon literature; no coupon test of these parts | room temperature, dry; printed parts at 100 % infill | ±25 % until coupons are tested | § 8 |
| vortex lattice (PEREGRINE) | the lift slope, the span efficiency, the neutral point of the wing pair | a first-order lattice, inviscid | attached flow, small angles: it knows nothing of the stall | ±5 % on the lift slope (the campaign's sweep read within 2 %) | § 5, § 6 |


## biplane check

| key | value |
|---|---|
| gap / span | 0.28 |
| Prandtl σ | 0.391 |
| induced drag / monoplane, Prandtl | 0.695 |
| induced drag / monoplane, lattice | 0.707 |
| lift slope / two isolated wings, lattice | 0.895 |


## drag areas [cm²]

|  | 4Q-1 round struts, X | 4Q-2 faired posts |
|---|---|---|
| wings (2) | 24.5 | 24.5 |
| pod (between the wings) | 6.85 | 6.85 |
| fin plate | 4.09 | 4.09 |
| nacelles (4) | 6.44 | 6.44 |
| posts (2) | 49.3 | 5.38 |
| diagonals (4) | 73.9 | 8.06 |
| foot rods (4), axial | 0.226 | 0.226 |
| camera dome | 4.07 | 4.07 |


## polars

|  | cl_alpha | cd0 | k | cl_max | cd90 | alpha0 | width | source |
|---|---|---|---|---|---|---|---|---|
| Kestrel-4Q-1 | 4.08 | 0.0665 | 0.0666 | 0.81 | 1.24 | 0 | 0.0262 | model |
| Kestrel-4Q-2 | 4.08 | 0.0257 | 0.0666 | 1.03 | 1.33 | -0.00169 | 0.0262 | NACA 23112 (k2/k1 0.03) (Zephyr) |
| Kestrel-4Q-3 | 4.08 | 0.0257 | 0.0666 | 1.03 | 1.33 | -0.00169 | 0.0262 | NACA 23112 (k2/k1 0.03) (Zephyr) |


## CFD sweep against the model

| key | value |
|---|---|
| 4Q-1 |                              α [deg]    CL CFD  CL model       ΔCL    CD CFD  CD model … |
| 4Q-2 |                              α [deg]    CL CFD  CL model       ΔCL    CD CFD  CD model … |


## CFD sweep numbers

|  | CFD | model |
|---|---|---|
| 4Q-1 | {'CLα [1/rad]': 3.181756901390271, 'CL max': 0.638042087, 'α at CL max [deg]': 45.0, 'C… | {'CLα [1/rad]': 3.3559647175391194, 'CL max': 0.6845668227392617, 'α at CL max [deg]': … |
| 4Q-2 | {'CLα [1/rad]': 3.251482958546284, 'CL max': 0.629399789, 'α at CL max [deg]': 45.0, 'C… | {'CLα [1/rad]': 3.9157028406675565, 'CL max': 0.7494424472949883, 'α at CL max [deg]': … |


## CFD labels

|  | label | bare cases | converged |
|---|---|---|---|
| 4Q-1 | solver (unconverged: indicative) | 13 | 0 |
| 4Q-2 | solver (unconverged: indicative) | 13 | 1 |
| 4Q-3 | NOT RUN | 0 | 0 |


## stall decision

|  | CL max, polar | stall speed, polar [m/s] | cruise / stall, polar | CL max, CFD sweep (< 20°) | at α [deg] | stall speed, CFD [m/s] | cruise / stall, CFD | CFD label | CFD converged | wing area for cruise = 1.3 × stall on the CFD's CL max [m²] | wing area now [m²] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Kestrel-4Q-1 | 0.81 | 10.9 | 1.47 | 0.529 | 14 | 13.5 | 1.19 | solver (unconverged: indicative) | 0 of 13 | 0.337 | 0.28 |
| Kestrel-4Q-2 | 1.03 | 9.56 | 1.67 | 0.528 | 14 | 13.4 | 1.2 | solver (unconverged: indicative) | 1 of 13 | 0.33 | 0.28 |
| Kestrel-4Q-3 | 1.03 | 9.56 | 1.67 | — | — | — | — | NOT RUN | — | — | — |


## neutral point

|  | x_np_wings | x_np | Cma_wings | Cma_pod | Cma_props | Cma | static_margin | x_cg |
|---|---|---|---|---|---|---|---|---|
| Kestrel-4Q-1 | 0.233 | 0.228 | -0.53 | 0.0665 | 0.0647 | -0.399 | 0.0955 | 0.215 |
| Kestrel-4Q-2 | 0.233 | 0.228 | -0.514 | 0.0665 | 0.065 | -0.383 | 0.0916 | 0.215 |
| Kestrel-4Q-3 | 0.233 | 0.228 | -0.519 | 0.0665 | 0.0649 | -0.388 | 0.0927 | 0.215 |


## trim (main)

| V [m/s] | CL | L/D | upper rotor [N] | lower rotor [N] | trim by thrust | electrical [W] | endurance [min] | range [km] | feasible |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 0.944 | 11.1 | -0.0179 | 0.747 | no | 41.1 | 57 | 34.2 | yes |
| 12 | 0.656 | 12.1 | -0.0461 | 0.717 | no | 45.1 | 52 | 37.4 | yes |
| 14 | 0.482 | 11.7 | -0.0359 | 0.728 | no | 51.2 | 45.8 | 38.5 | yes |
| 16 | 0.369 | 10.6 | -0.00132 | 0.765 | no | 59.1 | 39.7 | 38.1 | yes |
| 20 | 0.236 | 8.02 | 0.118 | 0.892 | no | 92.2 | 25.4 | 30.5 | yes |
| 25 | 0.151 | 5.54 | 0.336 | 1.13 | yes | 169 | 13.9 | 20.9 | yes |


## allocation rank

|  | senses (M1..M4) | mirror-symmetric | rank | σ (N, N·m) | condition | never produced | verdict |
|---|---|---|---|---|---|---|---|
| diagonal pairs (M1, M4 / M2, M3) | (1, -1, -1, 1) | yes | 4 | 2, 0.78, 0.28, 0.0323 | 61.9 | — | full control (thrust + 3 moments) |
| side columns alike (M1, M3 / M2, M4) | (1, -1, 1, -1) | yes | 3 | 2, 0.781, 0.28, 1.95e-16 | ∞ | +1.00·L + -0.04·N | NOT controllable in the hover |
| rows alike (M1, M2 / M3, M4) | (1, 1, -1, -1) | no | 3 | 2, 0.78, 0.282, 6.97e-17 | ∞ | +0.00·T + -1.00·L + +0.12·M | NOT controllable in the hover |
| all the same way | (1, 1, 1, 1) | no | 3 | 2, 0.78, 0.28, 5.39e-17 | ∞ | -0.02·T + -1.00·L | NOT controllable in the hover |


## hover authority

|  | most negative [N·m] | most positive [N·m] | inertia [kg·m²] | angular acceleration [rad/s²] |
|---|---|---|---|---|
| about the thrust axis (hover heading / cruise roll) | -0.237 | 0.237 | 0.113 | 2.09 |
| about y (pitch-over / cruise pitch) | -2.2 | 2.05 | 0.0232 | 88.4 |
| about z (hover tilt sideways / cruise yaw) | -5.72 | 5.72 | 0.104 | 55 |


## pursuit

|  | Kestrel-4Q-1 | Kestrel-4Q-2 | Kestrel-4Q-3 |
|---|---|---|---|
| mass [kg] | 1.69 | 1.65 | 1.65 |
| Cd0 | 0.0665 | 0.0257 | 0.0257 |
| stall speed [m/s] | 10.9 | 9.56 | 9.56 |
| V max, thrust = drag [m/s] | 30.1 | 33.3 | 33.3 |
| accelerate 13 → 25 m/s [s] | 0.95 | 0.8 | 0.8 |
| accelerate 13 → 0.9 V max [s] | 1.34 | 1.57 | 1.57 |
| best climb rate [m/s] | 28.2 | 30.9 | 30.9 |
| at climb speed [m/s] | 16.5 | 18 | 18 |
| steepest climb [deg] | 90 | 90 | 90 |
| time to climb 100 m [s] | 3.55 | 3.23 | 3.23 |
| vertical climb (hover thrust margin) [m/s²] | 18.6 | 19.2 | 19.2 |
| dive at 30° (idle) [m/s] | 28.3 | 45.3 | 45.3 |
| dive at 60° (idle) [m/s] | 36.5 | 58.3 | 58.3 |
| terminal dive (idle) [m/s] | 39.1 | 62.4 | 62.4 |
| corner speed [m/s] | 21.8 | 19.1 | 19.1 |
| turn rate at the corner (4 g) [deg/s] | 99.7 | 114 | 114 |
| turn radius at the corner [m] | 12.5 | 9.62 | 9.62 |
| time for 180° at the corner [s] | 1.8 | 1.58 | 1.58 |
| sustained load factor at 16 m/s | 2.15 | 2.8 | 2.8 |
| sustained turn rate at 16 m/s [deg/s] | 66.9 | 91.9 | 91.9 |
| pitch at 16 m/s, total [rad/s²] | 65.8 | 51.5 | 57.1 |
| steady roll rate at 16 m/s [deg/s] | 413 | 414 | 443 |
| yaw at 16 m/s, rotors [rad/s²] | 8.57 | 1.82 | 1.83 |
| endurance at 16 m/s [min] | 21 | 35.9 | 35.9 |
| best endurance [min] | 38 | 55.5 | 55.5 |
| at [m/s] | 11 | 10 | 10 |
| best range [km] | 25.1 | 37.5 | 37.5 |
| at range speed [m/s] | 11 | 13 | 13 |


## energy budget (main)

| key | value |
|---|---|
| usable energy [Wh] | 39.1 |
| hover, 1.5 min (take-off, landing, transitions) [Wh] | 5.35 |
| hover power [W] | 214 |
| cruise power at 16 m/s [W] | 59.1 |
| cruise time on the rest [min] | 34.2 |
| best-endurance speed [m/s] | 10 |
| power there [W] | 41.1 |
| loiter time on the rest [min] | 49.2 |
| range at 16 m/s on the rest [km] | 32.9 |


## plane frame, even lift

|  | spar peak moment [N·m] | spar stress [MPa] | spar reserve | tip deflection, upper [mm] | tip deflection, lower [mm] | motor station, upper [mm] | motor station, lower [mm] | propeller gap change [mm] | propeller gap under load [mm] | least buckling margin | truss mass [g] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| posts + X diagonals | 0.646 | 11.1 | 35.9 | -0.089 | -0.089 | 0.0339 | 0.0339 | 6.78e-18 | 51.4 | 85 | 127 |
| posts only | 1.18 | 20.4 | 19.6 | 1.8 | 1.8 | 1.36 | 1.36 | -6.51e-16 | 51.4 | ∞ | 40.8 |
| no truss (two cantilevers) | 1.18 | 20.4 | 19.6 | 1.8 | 1.8 | 1.36 | 1.36 | 0 | 51.4 | ∞ | 0 |


## plane frame, 75 % wing

|  | spar peak moment [N·m] | spar stress [MPa] | spar reserve | tip deflection, upper [mm] | tip deflection, lower [mm] | motor station, upper [mm] | motor station, lower [mm] | propeller gap change [mm] | propeller gap under load [mm] | least buckling margin | truss mass [g] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| posts + X diagonals | 0.711 | 12.3 | 32.6 | -0.143 | -0.0354 | 0.0338 | 0.0341 | -3.75e-04 | 51.4 | 587 | 127 |
| posts only | 1.5 | 25.9 | 15.5 | 1.73 | 1.87 | 1.36 | 1.35 | 5.74e-04 | 51.4 | ∞ | 40.8 |
| no truss (two cantilevers) | 2.91 | 50.2 | 7.98 | 6.1 | -2.51 | 4.43 | -1.72 | 6.15 | 57.6 | ∞ | 0 |


## stiffness and modes

|  | k symmetric [N/mm] | k antisymmetric [N/mm] | first mode, symmetric [Hz] | first mode, antisymmetric [Hz] |
|---|---|---|---|---|
| posts + X diagonals | 327 | 1.3e+04 | 261 | 1.64e+03 |
| posts only | 1.17 | 1.26e+04 | 15.6 | 1.62e+03 |
| no truss (two cantilevers) | 1.17 | 1.17 | 15.6 | 15.6 |


## ground stance

| key | value |
|---|---|
| CG height above the ground [m] | 0.325 |
| least tip-over angle [deg] | 39.4 |
| wind that tips it, broadside [m/s] | 8.81 |
| wind that tips it, edge-on [m/s] | 20.9 |
| touchdown load [N] | 119 |
| per foot, two feet first [N] | 59.7 |
| foot rod Euler load [N] | 766 |
| foot rod margin, two feet first | 12.8 |


## FEA

NOT RUN


## transitions

| vehicle · direction · case | height loss [m] | height gain [m] | pitch error max [deg] | time stalled, free stream [s] | least pitch margin | peak rotor thrust / T_max | elevon max [deg] | roll demand / authority | energy [Wh] | pass | method |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Kestrel-4Q-1 · forward · as is, still air | 0.108 | 0.388 | 2.46 | 0.954 | 0.823 | 0.554 | 4.19 | 0.0661 | 0.534 | yes | reduced-order model (solver) |
| Kestrel-4Q-1 · back · as is, still air | 0.553 | 2.44 | 4.65 | 1.5 | 0.768 | 0.438 | 5.76 | 0.277 | 0.528 | yes | reduced-order model (solver) |
| Kestrel-4Q-1 · back · as is, headwind 6 m/s | 0.187 | 2.44 | 4.65 | 1.5 | 0.768 | 0.467 | 5.76 | 0.277 | 0.51 | yes | reduced-order model (solver) |
| Kestrel-4Q-2 · forward · as is, still air | 0.126 | 0.414 | 2.45 | 0.814 | 0.843 | 0.531 | 3.77 | 0.0289 | 0.454 | yes | reduced-order model (solver) |
| Kestrel-4Q-2 · back · as is, still air | 0.172 | 12.3 | 3.44 | 1.35 | 0.816 | 0.365 | 4.55 | 0.116 | 0.441 | no | reduced-order model (solver) |
| Kestrel-4Q-2 · back · as is, headwind 6 m/s | 0.172 | 12.3 | 3.44 | 1.35 | 0.816 | 0.365 | 4.55 | 0.116 | 0.409 | no | reduced-order model (solver) |
| Kestrel-4Q-3 · forward · as is, still air | 0.136 | 0.409 | 2.46 | 0.818 | 0.846 | 0.529 | 3.68 | 0.0271 | 0.453 | yes | reduced-order model (solver) |
| Kestrel-4Q-3 · back · as is, still air | 0.181 | 11.9 | 3.55 | 1.35 | 0.821 | 0.367 | 4.42 | 0.12 | 0.441 | no | reduced-order model (solver) |
| Kestrel-4Q-3 · back · as is, headwind 6 m/s | 0.181 | 11.9 | 3.55 | 1.35 | 0.821 | 0.37 | 4.42 | 0.12 | 0.41 | no | reduced-order model (solver) |
| Kestrel-4Q-2 · back · reverse thrust 3 N per rotor | 0.442 | 0.242 | 4.1 | 1.32 | 0.849 | 0.41 | 3.71 | 0.144 | 0.543 | yes | reduced-order model (solver) |
| Kestrel-4Q-2 · back · crow brake ±20° + reverse thrust | 0.407 | 0.215 | 3.97 | 1.35 | 0.849 | 0.405 | 3.71 | 0.259 | 0.54 | yes | reduced-order model (solver) |
| Kestrel-4Q-3 · back · reverse thrust 3 N per rotor | 0.465 | 0.206 | 4.12 | 1.33 | 0.853 | 0.41 | 3.55 | 0.136 | 0.543 | yes | reduced-order model (solver) |
| Kestrel-4Q-3 · back · crow brake ±20° + reverse thrust | 0.42 | 0.183 | 3.97 | 1.36 | 0.852 | 0.404 | 3.55 | 0.252 | 0.539 | yes | reduced-order model (solver) |
| Kestrel-4Q-2 · forward · as is, CFD sweep correction | 0.503 | 0.398 | 2.5 | 0.846 | 0.81 | 0.536 | 4.54 | 0.064 | 0.519 | yes | reduced-order model on the CFD's airframe correction |
| Kestrel-4Q-2 · back · as is, CFD sweep correction | 0.397 | 0.274 | 3.83 | 1.39 | 0.836 | 0.417 | 4.05 | 0.28 | 0.497 | yes | reduced-order model on the CFD's airframe correction |


## level deceleration

| vehicle · means | from [m/s] | to [m/s] | time [s] | distance [m] | energy [Wh] | reached | mean deceleration [m/s²] |
|---|---|---|---|---|---|---|---|
| Kestrel-4Q-1 · rotors at idle | 16 | 13 | 2.68 | 38.6 | 0.032 | yes | 1.12 |
| Kestrel-4Q-2 · rotors at idle | 16 | 13 | 12 | 172 | 0.143 | yes | 0.251 |
| Kestrel-4Q-2 · crow brake ±20° | 16 | 13 | 1.86 | 26.8 | 0.0222 | yes | 1.61 |
| Kestrel-4Q-2 · reverse thrust 3 N | 16 | 13 | 0.37 | 5.35 | 0.0322 | yes | 8.11 |
| Kestrel-4Q-3 · rotors at idle | 16 | 13 | 12 | 172 | 0.143 | yes | 0.251 |
| Kestrel-4Q-3 · crow brake ±20° | 16 | 13 | 1.7 | 24.5 | 0.0203 | yes | 1.76 |
| Kestrel-4Q-3 · reverse thrust 3 N | 16 | 13 | 0.37 | 5.35 | 0.0322 | yes | 8.11 |


## recommended workflows

| order | workflow | variant | campaign | quality | why | command |
|---|---|---|---|---|---|---|
| 2 | airframe_cfd | 4Q-2 | stall | fine | CL max decides the stall speed and the wing area: the polar stalls at 14°; the screenin… | .venv/bin/python workflows/kestrel/airframe_cfd.py --variant 4Q-2 --chapter stall --qua… |
| 3 | airframe_cfd | 4Q-2 | cruise | screening | the level cruise and the stall point with the rotor disks on: the blown spans' lift and… | .venv/bin/python workflows/kestrel/airframe_cfd.py --variant 4Q-2 --chapter cruise --qu… |
| 4 | structure_fea | 4Q-2 | all | — | the frame's margins at ultimate and the joint's across-layer tension: no FEA on file | .venv/bin/python workflows/kestrel/structure_fea.py --variant 4Q-2 --chapter all |
| 5 | airframe_cfd | 4Q-2 | transitions | screening | the transition points with the disks (the model's least-known regime: blown, stalled); … | .venv/bin/python workflows/kestrel/airframe_cfd.py --variant 4Q-2 --chapter transitions… |


## final checks

| check | value | target | holds | note |
|---|---|---|---|---|
| take-off mass [kg] | 1.65 | ≤ 2 | ✔ | mass table: component data + assumed structure densities (estimate) |
| hover thrust / weight | 2.95 | ≥ 2 | ✔ | Boreas BEMT, generic section unless prop_lab is on file (solver) |
| static margin [chords] | 0.0916 | 0.05 – 0.15 | ✔ | lattice + De Young propellers (solver + empirical) |
| stall speed / cruise speed | 0.597 | ≤ 0.8 | ✔ | polar's CL max 1.03 (NeuralFoil ×0.9, empirical) |
| endurance at the cruise [min] | 39.7 | ≥ 20 | ✔ | drag build-up + BEMT (solver + empirical, ±20 % on Cd0) |
| forward transitions pass (nominal, still air) | yes | == yes | ✔ | reduced-order model (solver); Viterna post-stall |
| back transitions pass as is (still air and headwind) | no | == yes | ✘ | reduced-order model (solver) |
| a back-transition remedy passes | yes | == yes | ✔ | reverse thrust / crow brake in the model (solver); the drives' reverse thrust is an ass… |
| CFD sweep: lift slope / lattice | 0.83 | 0.85 – 1.15 | ✘ | solver (unconverged: indicative); 1 of 13 converged |
| CFD sweep: CL max (< 20°) / polar | 0.511 | ≥ 0.8 | ✘ | solver (unconverged: indicative): a 5° grid's lower bound; the fine stall campaign decides |
| structural FEA on file | no | == yes | ✘ | NOT RUN: run workflows/kestrel/structure_fea.py; the plane frame's margins stand in (an… |


## verdict

Kestrel-4Q-2: 7 of 11 checks hold (screening pass: unvalidated — CFD and FEA labelled NOT RUN where missing).


Mass 1.65 kg, static margin 0.092 c, hover T/W 2.95 at 214 W; cruise 16 m/s at 59 W for 40 min (L/D 10.6); stall 9.6 m/s on the polar's CL max 1.03.


The CFD sweep (solver (unconverged: indicative)) reads CL max 0.53 below 20° against the polar's 1.03: if it holds, the stall is 13.4 m/s and the cruise only 1.20 × it — the fine stall campaign decides the wing area.


Transitions (reduced-order model): forward pass; back as is FAIL; with a remedy pass.


FEA: NOT RUN — the plane frame's spar reserve stands in.


Pursuit: V max 33.3 m/s (thrust-limited), corner radius 9.6 m, best climb 30.9 m/s.



## variants

|  | take-off mass [kg] | CG x [mm] | static margin [chords] | Cd0 | CL max (polar) | stall speed [m/s] | stall / cruise | hover T/W | hover electrical [W] | L/D at 16 m/s | power at 16 m/s [W] | endurance at 16 m/s [min] | best endurance [min] | trim by thrust alone at the cruise |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Kestrel-4Q-1 | 1.69 | 215 | 0.0955 | 0.0665 | 0.81 | 10.9 | 0.682 | 2.89 | 221 | 4.96 | 110 | 21.3 | 40.2 | yes |
| Kestrel-4Q-2 | 1.65 | 215 | 0.0916 | 0.0257 | 1.03 | 9.56 | 0.597 | 2.95 | 214 | 10.6 | 59.1 | 39.7 | 57 | no |
| Kestrel-4Q-3 | 1.65 | 215 | 0.0927 | 0.0257 | 1.03 | 9.56 | 0.597 | 2.95 | 214 | 10.6 | 59.3 | 39.5 | 56.8 | no |

*4 more columns not shown: section, elevons, bracing, struts*


## runtime [s]

|  | seconds |
|---|---|
| 1 prerequisites | 1.7 |
| 2 geometry | 13.5 |
| 3 mass | 0 |
| 4 propulsion | 0.4 |
| 5 aerodynamics | 0.4 |
| 6 trim | 0.8 |
| 7 performance | 0.8 |
| 8 structure | 0.1 |
| 9 transitions | 112 |
| 10 recommendations | 0 |
| 11 assessment | 0 |
| total | 130 |


## status

| key | value |
|---|---|
| screening | yes |
| required |  |

