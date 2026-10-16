# Kestrel — the design dossier

*stitched 2026-10-16 12:35 from 4 report(s)*

| report | lines | exported |
|---|---|---|
| 38_kestrel | 917 | 38_kestrel — exported 2026-10-16 12:22 |
| 39_xwing | 172 | 39_xwing — exported 2026-10-16 12:34 |
| 40_kestrel_tests | 164 | 40_kestrel_tests — exported 2026-10-16 12:30 |
| 41_prop_lab | 172 | 41_prop_lab — exported 2026-10-16 12:30 |


---

## Kestrel-4Q — notebook 38, the design loop (Parts I and II)

*38_kestrel — exported 2026-10-16 12:22*

| key | value |
|---|---|
| mass [kg] | 1.69 |
| 4Q-2 mass [kg] | 1.65 |
| CFD | NOT RUN here |
| FEA | NOT RUN (no CalculiX here; notebook 40) |


## targets

| Kestrel-4Q — the targets this study sets | value |
|---|---|
| configuration | electric quadrotor tailsitter, two load-bearing wings |
| wings | zero stagger, equal span, straight |
| rotors | exactly four, fixed, thrust axes parallel to the pod |
| take-off mass (this study's target) [kg] | ≤ 2.0 |
| thrust-to-weight in the hover | ≥ 2 |
| cruise (survey) [m/s] | 16 |
| endurance at cruise [min] | ≥ 20 |
| transitions | both ways, height within 3 m (10 m gain going back), attitude within 10° |
| wind for operations [m/s] | 8 |


## parameters

| key | value |
|---|---|
| span | 1e+03 |
| chord | 140 |
| gap | 280 |
| wing_le_x | 200 |
| thickness | 0.12 |
| camber | 0 |
| camber_pos | 0.4 |
| spar_x | 0.3 |
| pod_length | 470 |
| pod_width | 110 |
| camera_diameter | 36 |
| nose_length | 150 |
| tail_length | 100 |
| pod_wall | 6 |
| bay_length | 160 |
| bay_width | 72 |
| bay_height | 72 |
| bay_x | 152 |
| bay_z | 10 |
| motor_y | 390 |
| prop_diameter | 229 |
| prop_x | 125 |
| nacelle_diameter | 34 |
| prop_clearance | 25 |
| post_diameter | 8 |
| diagonal_diameter | 6 |
| bracing | x |
| strut_fairing | no |
| fairing_tc | 0.33 |
| fairing_wall | 0.8 |
| elevons | no |
| elevon_chord | 0.28 |
| elevon_inner_y | 275 |
| elevon_gap | 1 |
| elevon_deg | 0 |
| x_feet | 540 |
| fin_height | 190 |
| ventral_height | 150 |
| fin_tip_chord | 90 |
| fin_thickness | 4 |
| foot_rod_diameter | 6 |


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
| elevons | no |
| elevon_cf | 0.28 |
| elevon_y_in | 0.275 |
| elevon_span_side | 0.225 |
| x_hinge | 0.301 |
| S_elevon | 0.0353 |
| diagonals | 4 |
| posts | 2 |


## mass groups [kg]

| kg | value |
|---|---|
| propulsion | 0.736 |
| structure | 0.625 |
| systems | 0.16 |
| payload | 0.105 |
| margin | 0.06 |


## mass, CG, inertia

| key | value |
|---|---|
| mass_kg | 1.69 |
| x_cg | 0.215 |
| y_cg | 5.14e-19 |
| z_cg | -0.00448 |
| Ixx | 0.106 |
| Iyy | 0.0221 |
| Izz | 0.098 |
| battery_x | 0.22 |
| battery_clipped | yes |


## hover point (one drive)

| key | value |
|---|---|
| thrust_N | 4.13 |
| rpm | 6.71e+03 |
| current_A | 6.65 |
| electrical_W | 55.2 |
| throttle | 0.56 |
| figure_of_merit | 0.565 |
| T_max_N | 12 |
| T_W | 2.89 |
| kappa_m | 0.0162 |
| total_W | 221 |


## allocation rank per sense of rotation

|  | senses (M1..M4) | mirror-symmetric | rank | σ (N, N·m) | condition | never produced | verdict |
|---|---|---|---|---|---|---|---|
| diagonal pairs (M1, M4 / M2, M3) | (1, -1, -1, 1) | yes | 4 | 2, 0.78, 0.28, 0.0323 | 61.9 | — | full control (thrust + 3 moments) |
| side columns alike (M1, M3 / M2, M4) | (1, -1, 1, -1) | yes | 3 | 2, 0.781, 0.28, 2.31e-17 | ∞ | +1.00·L + -0.04·N | NOT controllable in the hover |
| rows alike (M1, M2 / M3, M4) | (1, 1, -1, -1) | no | 3 | 2, 0.78, 0.282, 2.62e-17 | ∞ | +0.00·T + -1.00·L + +0.12·M | NOT controllable in the hover |
| all the same way | (1, 1, 1, 1) | no | 3 | 2, 0.78, 0.28, 7.82e-18 | ∞ | +0.02·T + +1.00·L | NOT controllable in the hover |


## hover authority

|  | most negative [N·m] | most positive [N·m] | inertia [kg·m²] | angular acceleration [rad/s²] |
|---|---|---|---|---|
| about the thrust axis (hover heading / cruise roll) | -0.242 | 0.242 | 0.106 | 2.28 |
| about y (pitch-over / cruise pitch) | -2.25 | 2.1 | 0.0221 | 95 |
| about z (hover tilt sideways / cruise yaw) | -5.85 | 5.85 | 0.098 | 59.7 |


## motor out

|  | thrusts of the others [N] | within limits | unbalanced torque about the axis [N·m] | hover |
|---|---|---|---|---|
| M1 out | 8.00, 8.27, 0.26 | yes | 0.259 | flies on one diagonal pair (the partner idles): holds thrust and tilt, spins about its … |
| M2 out | 8.00, 0.26, 8.27 | yes | -0.259 | flies on one diagonal pair (the partner idles): holds thrust and tilt, spins about its … |
| M3 out | 8.27, -0.26, 8.53 | no | -0.276 | cannot hold the weight level |
| M4 out | -0.26, 8.27, 8.53 | no | 0.276 | cannot hold the weight level |


## hover in wind

|  | wing drag [N] | lean into the wind [deg] | thrust per rotor [N] | T_max per rotor [N] | pitch moment left [N·m] |
|---|---|---|---|---|---|
| 2 m/s | 0.823 | 2.85 | 4.14 | 12 | 2.1 |
| 4 m/s | 3.29 | 11.3 | 4.22 | 12 | 2.14 |
| 6 m/s | 7.41 | 24.1 | 4.53 | 12 | 2.32 |
| 8 m/s | 13.2 | 38.5 | 5.29 | 12 | 2.72 |
| 10 m/s | 20.6 | 51.2 | 6.6 | 12 | 2.88 |


## biplane factors

| key | value |
|---|---|
| lift slope vs two isolated wings | 0.895 |
| induced drag vs the monoplane | 0.707 |
| gap / chord | 2 |
| gap / span | 0.28 |


## drag build-up [cm²]

|  | round tubes | faired struts |
|---|---|---|
| wings (2) | 24.5 | 24.5 |
| pod (between the wings) | 6.85 | 6.85 |
| fin plate | 4.09 | 4.09 |
| nacelles (4) | 6.44 | 6.44 |
| posts (2) | 49.3 | 5.38 |
| diagonals (4) | 73.9 | 8.06 |
| foot rods (4), axial | 0.226 | 0.226 |
| camera dome | 4.07 | 4.07 |


## neutral point

| key | value |
|---|---|
| x_np_wings | 0.233 |
| x_np | 0.228 |
| Cma_wings | -0.53 |
| Cma_pod | 0.0665 |
| Cma_props | 0.0647 |
| Cma | -0.399 |
| static_margin | 0.0955 |
| x_cg | 0.215 |


## polar

| key | value |
|---|---|
| cl_alpha | 4.08 |
| cd0 | 0.0665 |
| k | 0.0666 |
| cl_max | 0.81 |
| cd90 | 1.24 |
| alpha0 | 0 |
| width | 0.0262 |
| source | model |


## cruise, round tubes

| V [m/s] | CL | feasible | note | α [deg] | L/D | drag [N] | Cm about CG | upper rotor [N] | lower rotor [N] | trim by thrust | electrical [W] | endurance [min] | range [km] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 12 | 0.67 | yes | the upper pair below T_min: thrust alone cannot trim | 9.41 | 6.95 | 2.38 | -0.064 | 0.181 | 1.01 | no | 64.6 | 36.3 | 26.1 |
| 14 | 0.492 | yes |  | 6.91 | 5.95 | 2.78 | -0.047 | 0.277 | 1.11 | yes | 83.2 | 28.2 | 23.7 |
| 16 | 0.377 | yes |  | 5.29 | 4.96 | 3.34 | -0.036 | 0.412 | 1.26 | yes | 110 | 21.3 | 20.4 |
| 20 | 0.241 | yes |  | 3.39 | 3.42 | 4.83 | -0.023 | 0.774 | 1.64 | yes | 185 | 12.6 | 15.2 |
| 25 | 0.154 | yes |  | 2.17 | 2.26 | 7.3 | -0.0147 | 1.37 | 2.28 | yes | 330 | 7.11 | 10.7 |


## cruise, faired struts

| V [m/s] | CL | feasible | note | α [deg] | L/D | drag [N] | Cm about CG | upper rotor [N] | lower rotor [N] | trim by thrust | electrical [W] | endurance [min] | range [km] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 12 | 0.67 | yes | the upper pair below T_min: thrust alone cannot trim | 9.41 | 12.6 | 1.32 | -0.064 | -0.0765 | 0.734 | no | 45.7 | 51.2 | 36.9 |
| 14 | 0.492 | yes | the upper pair below T_min: thrust alone cannot trim | 6.91 | 12.4 | 1.33 | -0.047 | -0.0733 | 0.738 | no | 51.6 | 45.5 | 38.2 |
| 16 | 0.377 | yes | the upper pair below T_min: thrust alone cannot trim | 5.29 | 11.5 | 1.44 | -0.036 | -0.0456 | 0.767 | no | 59.2 | 39.6 | 38 |
| 20 | 0.241 | yes | the upper pair below T_min: thrust alone cannot trim | 3.39 | 8.83 | 1.87 | -0.023 | 0.0583 | 0.878 | no | 85.8 | 27.3 | 32.8 |
| 25 | 0.154 | yes |  | 2.17 | 6.17 | 2.68 | -0.0147 | 0.254 | 1.09 | yes | 155 | 15.2 | 22.7 |


## stall speed [m/s]

| key | value |
|---|---|
| V_stall | 10.9 |


## roll authority in cruise

| V | torque lever [N·m] | ailerons [N·m] | ratio | aileron span per side [m] |
|---|---|---|---|---|
| 12 | 0.422 | 2.17 | 5.13 | 0.224 |
| 16 | 0.394 | 3.91 | 9.91 | 0.224 |
| 20 | 0.347 | 6.18 | 17.8 | 0.224 |


## transition matrix (model polar)

| vehicle · direction · t_tr [s] · headwind [m/s] | height loss [m] | height gain [m] | pitch error max [deg] | time stalled, free stream [s] | α blown max while stalled [deg] | moment shortfall [s] | least pitch margin | peak rotor thrust / T_max | elevon max [deg] | roll demand / authority | roll demand / rotors' authority | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| elevons · forward · 1.5 · 0 | 3.26 | 0.383 | 7.86 | 0.278 | 5.54 | 0 | 0.736 | 1 | 6.33 | 0.0945 | 1.29 | no |
| elevons · forward · 1.5 · 6 | 0.308 | 0.0873 | 4.52 | 1.12 | 19.3 | 0 | 0.827 | 1 | 4.06 | 0.0687 | 1.15 | yes |
| elevons · forward · 3 · 0 | 0.108 | 0.389 | 2.46 | 0.952 | 8.97 | 0 | 0.822 | 0.555 | 4.22 | 0.0681 | 1.12 | yes |
| elevons · forward · 3 · 6 | 0.453 | 0.0946 | 1.91 | 2.5 | 19.4 | 0 | 0.807 | 0.555 | 4.63 | 0.0547 | 0.786 | yes |
| elevons · forward · 5 · 0 | 1.12 | 0.369 | 1.55 | 2.55 | 15.1 | 0 | 0.769 | 0.565 | 5.61 | 0.0707 | 1.33 | yes |
| elevons · forward · 5 · 6 | 1.44 | 0.433 | 1.31 | 4.47 | 20.8 | 0 | 0.804 | 0.639 | 4.82 | 0.0681 | 1.13 | yes |
| elevons · back · 1.5 · 0 | 0.313 | 1.6 | 7.51 | 0.978 | 75.4 | 0 | 0.659 | 0.476 | 8.49 | 0.435 | 1.47 | yes |
| elevons · back · 1.5 · 6 | 0.185 | 1.6 | 7.51 | 0.978 | 75.4 | 0 | 0.659 | 0.398 | 8.49 | 0.435 | 1.47 | yes |
| elevons · back · 3 · 0 | 0.564 | 2.49 | 4.65 | 1.5 | 73.5 | 0 | 0.768 | 0.438 | 5.77 | 0.28 | 1.36 | yes |
| elevons · back · 3 · 6 | 0.185 | 2.49 | 4.65 | 1.5 | 73.5 | 0 | 0.768 | 0.466 | 5.77 | 0.28 | 1.36 | yes |
| elevons · back · 5 · 0 | 0.927 | 3.4 | 2.35 | 1.84 | 61.5 | 0 | 0.828 | 0.422 | 4.25 | 0.233 | 1.28 | yes |
| elevons · back · 5 · 6 | 0.456 | 3.4 | 2.35 | 1.84 | 61.5 | 0 | 0.828 | 0.498 | 4.25 | 0.233 | 1.28 | yes |
| rotors only · forward · 1.5 · 0 | 0.794 | 0.289 | 7.84 | 0.27 | 5.51 | 0 | 0.557 | 1 | 0 | 1.65 | 1.65 | no |
| rotors only · forward · 1.5 · 6 | 0.182 | 0.137 | 4.43 | 1.1 | 19.1 | 0 | 0.71 | 1 | 0 | 1.12 | 1.12 | no |
| rotors only · forward · 3 · 0 | 0.108 | 0.431 | 2.44 | 0.922 | 8.8 | 0 | 0.762 | 0.586 | 0 | 0.502 | 0.502 | yes |
| rotors only · forward · 3 · 6 | 0.217 | 0.137 | 1.96 | 2.45 | 19.1 | 0 | 0.748 | 0.566 | 0 | 0.734 | 0.734 | yes |
| rotors only · forward · 5 · 0 | 0.707 | 0.412 | 1.69 | 2.44 | 14.8 | 0 | 0.716 | 0.578 | 0 | 0.993 | 0.993 | yes |
| rotors only · forward · 5 · 6 | 1.04 | 0.387 | 1.45 | 4.39 | 20.6 | 0 | 0.788 | 0.647 | 0 | 1.08 | 1.08 | no |
| rotors only · back · 1.5 · 0 | 0.535 | 2.68 | 6.79 | 1.06 | 74.6 | 0 | 0.612 | 0.422 | 0 | 1.72 | 1.72 | no |
| rotors only · back · 1.5 · 6 | 0.164 | 2.68 | 6.79 | 1.06 | 74.6 | 0 | 0.612 | 0.404 | 0 | 1.72 | 1.72 | no |
| rotors only · back · 3 · 0 | 1.55 | 5.49 | 4.51 | 1.47 | 72.3 | 0 | 0.683 | 0.424 | 0 | 1.8 | 1.8 | no |
| rotors only · back · 3 · 6 | 1.24 | 5.49 | 4.51 | 1.47 | 72.3 | 0 | 0.683 | 0.438 | 0 | 1.8 | 1.8 | no |
| rotors only · back · 5 · 0 | 2.03 | 8.13 | 2.3 | 1.56 | 63.4 | 0 | 0.696 | 0.426 | 0 | 1.36 | 1.36 | no |
| rotors only · back · 5 · 6 | 1.6 | 8.13 | 2.3 | 1.56 | 63.4 | 0 | 0.696 | 0.456 | 0 | 1.36 | 1.36 | no |
| 2.0 kg, CG aft · forward · 1.5 · 0 | 4 | -1.12e-06 | 16.9 | 0.452 | 25.4 | 0 | 0.601 | 1 | 11 | 0.171 | 1.49 | no |
| 2.0 kg, CG aft · forward · 1.5 · 6 | 0.32 | 0.105 | 4.53 | 1.13 | 18.4 | 0 | 0.814 | 1 | 4.41 | 0.074 | 1.24 | yes |
| 2.0 kg, CG aft · forward · 3 · 0 | 0.108 | 0.4 | 2.42 | 0.986 | 8.57 | 0 | 0.839 | 0.635 | 3.82 | 0.0714 | 1.21 | yes |
| 2.0 kg, CG aft · forward · 3 · 6 | 0.336 | 0.106 | 1.73 | 2.51 | 18.5 | 0 | 0.826 | 0.607 | 4.17 | 0.0717 | 1.27 | yes |
| 2.0 kg, CG aft · forward · 5 · 0 | 0.984 | 0.405 | 1.31 | 2.62 | 14.3 | 0 | 0.776 | 0.571 | 5.46 | 0.076 | 1.44 | yes |
| 2.0 kg, CG aft · forward · 5 · 6 | 1.35 | 0.4 | 1.09 | 4.53 | 20.1 | 0 | 0.815 | 0.677 | 4.58 | 0.0676 | 1.32 | yes |
| 2.0 kg, CG aft · back · 1.5 · 0 | 0.209 | 0.634 | 7.32 | 0.932 | 76.5 | 0 | 0.65 | 0.595 | 8.72 | 0.553 | 1.46 | yes |
| 2.0 kg, CG aft · back · 1.5 · 6 | 0.183 | 0.634 | 7.32 | 0.932 | 76.5 | 0 | 0.65 | 0.469 | 8.72 | 0.553 | 1.46 | yes |
| 2.0 kg, CG aft · back · 3 · 0 | 0.186 | 0.666 | 4.54 | 1.54 | 60.9 | 0 | 0.757 | 0.497 | 6.05 | 0.375 | 1.34 | yes |
| 2.0 kg, CG aft · back · 3 · 6 | 0.183 | 0.666 | 4.54 | 1.54 | 60.9 | 0 | 0.757 | 0.516 | 6.05 | 0.375 | 1.34 | yes |
| 2.0 kg, CG aft · back · 5 · 0 | 0.197 | 0.6 | 2.28 | 2.43 | 45.4 | 0 | 0.801 | 0.467 | 4.94 | 0.292 | 1.3 | yes |
| 2.0 kg, CG aft · back · 5 · 6 | 0.197 | 0.6 | 2.28 | 2.43 | 45.4 | 0 | 0.801 | 0.518 | 4.94 | 0.292 | 1.3 | yes |


## verdict (model)

|  | passes | of | failures |
|---|---|---|---|
| 2.0 kg, CG aft | 11 | 12 | ['forward 1.5 s, headwind 0 m/s: fails holds height, holds attitude'] |
| elevons | 11 | 12 | ['forward 1.5 s, headwind 0 m/s: fails holds height'] |
| rotors only | 3 | 12 | ['forward 1.5 s, headwind 0 m/s: fails roll holds', 'forward 1.5 s, headwind 6 m/s: fai… |


## back-transition strategies

| effectors · entry speed [m/s] · collective floor | height loss [m] | height gain [m] | pitch error max [deg] | time stalled, free stream [s] | α blown max while stalled [deg] | moment shortfall [s] | least pitch margin | peak rotor thrust / T_max | elevon max [deg] | roll demand / authority | roll demand / rotors' authority | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| elevons · 16 · 0 | 0.564 | 2.49 | 4.65 | 1.5 | 73.5 | 0 | 0.768 | 0.438 | 5.77 | 0.28 | 1.36 | yes |
| rotors only · 16 · 0 | 1.55 | 5.49 | 4.51 | 1.47 | 72.3 | 0 | 0.683 | 0.424 | 0 | 1.8 | 1.8 | no |
| elevons · 16 · 0.5 | 0.185 | 11.8 | 3.83 | 1.95 | 33.5 | 0 | 0.78 | 0.366 | 5.14 | 0.129 | 1.83 | no |
| rotors only · 16 · 0.5 | 0.115 | 15.7 | 3.27 | 1.74 | 27.7 | 0 | 0.627 | 0.434 | 0 | 2.36 | 2.36 | no |
| elevons · 13 · 0 | 0.625 | 0.0759 | 4.3 | 1.83 | 43.9 | 0 | 0.784 | 0.419 | 5.09 | 0.224 | 0.905 | yes |
| rotors only · 13 · 0 | 0.255 | 0.997 | 4.41 | 1.51 | 62 | 0 | 0.742 | 0.421 | 0 | 1.06 | 1.06 | no |
| elevons · 13 · 0.5 | 1.43 | 5.15 | 4.21 | 1.92 | 41.9 | 0 | 0.781 | 0.424 | 5.12 | 0.169 | 1.18 | yes |
| rotors only · 13 · 0.5 | 2.03 | 7.63 | 4 | 1.95 | 37.7 | 0 | 0.742 | 0.426 | 0 | 1.39 | 1.39 | no |


## gusts ±3 m/s

|  | height loss [m] | height gain [m] | pitch error max [deg] | time stalled, free stream [s] | α blown max while stalled [deg] | moment shortfall [s] | least pitch margin | peak rotor thrust / T_max | elevon max [deg] | roll demand / authority | roll demand / rotors' authority | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| forward, gust +3 m/s | 0.108 | 0.371 | 2.4 | 0.728 | 19 | 0 | 0.823 | 0.558 | 4.18 | 0.0678 | 1.12 | yes |
| forward, gust -3 m/s | 0.18 | 0.103 | 2.79 | 0.718 | 9.98 | 0 | 0.837 | 0.599 | 3.88 | 0.0415 | 0.567 | yes |
| back, gust +3 m/s | 0.816 | 3.48 | 5.33 | 1.58 | 72.4 | 0 | 0.683 | 0.422 | 7.88 | 0.318 | 1.36 | yes |
| back, gust -3 m/s | 0.402 | 1.71 | 4.7 | 1.44 | 73.3 | 0 | 0.768 | 0.428 | 5.77 | 0.297 | 1.36 | yes |


## CFD cases (this run)

|  | status | α [deg] | V [m/s] | source |
|---|---|---|---|---|
| forward 3 s: α 53°, 1.5 m/s, T 4.38/4.85 N | NOT RUN | 52.9 | 1.5 | forward 3 s |
| forward 3 s: α 45°, 2.1 m/s, T 4.47/5.02 N | NOT RUN | 44.9 | 2.1 | forward 3 s |
| forward 3 s: α 30°, 3.8 m/s, T 4.64/5.33 N | NOT RUN | 30 | 3.82 | forward 3 s |
| forward 3 s: α 20°, 5.8 m/s, T 4.58/5.36 N | NOT RUN | 20 | 5.81 | forward 3 s |
| forward 3 s: α 15°, 7.3 m/s, T 4.28/5.08 N | NOT RUN | 15 | 7.32 | forward 3 s |
| forward 3 s: α 10°, 9.3 m/s, T 3.10/3.94 N | NOT RUN | 9.99 | 9.29 | forward 3 s |
| back 3 s: α 80°, 6.0 m/s, T 0.27/1.66 N | NOT RUN | 80.1 | 6.03 | back 3 s |
| back 3 s: α 60°, 7.7 m/s, T 0.25/2.08 N | NOT RUN | 60 | 7.69 | back 3 s |
| back 3 s: α 45°, 9.9 m/s, T 0.25/2.45 N | NOT RUN | 45.1 | 9.94 | back 3 s |
| back 3 s: α 30°, 13.0 m/s, T 0.25/2.07 N | NOT RUN | 30 | 13 | back 3 s |
| back 3 s: α 20°, 14.7 m/s, T 0.25/1.21 N | NOT RUN | 20 | 14.7 | back 3 s |
| back 3 s: α 15°, 15.2 m/s, T 0.25/1.19 N | NOT RUN | 15 | 15.2 | back 3 s |
| back 3 s: α 10°, 15.5 m/s, T 0.25/1.41 N | NOT RUN | 10 | 15.5 | back 3 s |
| hover: α 90°, 1.0 m/s, T 4.13/4.13 N | NOT RUN | 90 | 1 | hover |
| cruise: α 5°, 16.0 m/s, T 0.83/0.83 N | NOT RUN | 5.29 | 16 | cruise |
| stall: α 10°, 11.5 m/s, T 0.58/0.58 N | NOT RUN | 9.67 | 11.5 | stall |
| bare airframe, α 0°, 10 m/s | NOT RUN | 0 | 10 | sweep |
| bare airframe, α 5°, 10 m/s | NOT RUN | 5 | 10 | sweep |
| bare airframe, α 10°, 10 m/s | NOT RUN | 10 | 10 | sweep |
| bare airframe, α 15°, 10 m/s | NOT RUN | 15 | 10 | sweep |
| bare airframe, α 20°, 10 m/s | NOT RUN | 20 | 10 | sweep |
| bare airframe, α 30°, 10 m/s | NOT RUN | 30 | 10 | sweep |
| bare airframe, α 45°, 10 m/s | NOT RUN | 45 | 10 | sweep |
| bare airframe, α 60°, 10 m/s | NOT RUN | 60 | 10 | sweep |
| bare airframe, α 75°, 10 m/s | NOT RUN | 75 | 10 | sweep |
| bare airframe, α 90°, 10 m/s | NOT RUN | 90 | 10 | sweep |


## CFD campaign on file

| key | value |
|---|---|
| solved | 10 |
| diverged | 2 |
| FAILED | 12 |
| NOT RUN | 0 |
| n | 24 |
| solved with rotor disks | 0 |
| solved bare | 10 |
| with rotor disks | 14 |


## bare sweep: CFD against the model

|  | α [deg] | CL CFD | CL model | ΔCL | CD CFD | CD model | ΔCD | Cm CFD | Cm model | ΔCm | converged | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bare airframe, α 0°, 10 m/s | 0 | -0.0488 | 0 | -0.0488 | 0.0532 | 0.0665 | -0.0133 | 0.021 | 0.00213 | 0.0189 | no |  |
| bare airframe, α 5°, 10 m/s | 5 | 0.319 | 0.352 | -0.0328 | 0.0808 | 0.075 | 0.00579 | -0.0304 | -0.0498 | 0.0194 | no |  |
| bare airframe, α 10°, 10 m/s | 10 | 0.508 | 0.564 | -0.0569 | 0.134 | 0.101 | 0.033 | -0.0705 | -0.0844 | 0.0139 | no |  |
| bare airframe, α 15°, 10 m/s | 15 | 0.541 | 0.38 | 0.161 | 0.198 | 0.152 | 0.0459 | -0.0882 | -0.0749 | -0.0132 | no |  |
| bare airframe, α 20°, 10 m/s | 20 | 0.571 | 0.421 | 0.15 | 0.297 | 0.219 | 0.0781 | -0.133 | -0.1 | -0.0327 | no |  |
| bare airframe, α 30°, 10 m/s | 30 | 0.572 | 0.576 | -0.00446 | 0.46 | 0.399 | 0.0606 | -0.12 | -0.173 | 0.0523 | no |  |
| bare airframe, α 45°, 10 m/s | 45 | 0.615 | 0.685 | -0.0698 | 0.752 | 0.751 | 0.00132 | -0.203 | -0.292 | 0.0891 | no |  |
| bare airframe, α 60°, 10 m/s | 60 | 0.45 | 0.606 | -0.156 | 0.936 | 1.12 | -0.18 | -0.242 | -0.403 | 0.161 | no |  |
| bare airframe, α 75°, 10 m/s | 75 | 0.208 | 0.354 | -0.147 | 0.857 | 1.39 | -0.532 | -0.212 | -0.481 | 0.27 | no |  |
| bare airframe, α 90°, 10 m/s | 90 | 0.0604 | 8.71e-17 | 0.0604 | 0.768 | 1.49 | -0.722 | -0.229 | -0.51 | 0.28 | no |  |


## sweep numbers

|  | CLα [1/rad] | CL max | α at CL max [deg] | CL max below 20° | α of it [deg] | CD at the lowest α | Cmα [1/rad] | static margin from Cmα [chords] |
|---|---|---|---|---|---|---|---|---|
| CFD | 3.19 | 0.615 | 45 | 0.541 | 15 | 0.0532 | -0.524 | 0.164 |
| model | 3.23 | 0.685 | 45 | 0.564 | 10 | 0.0665 | -0.496 | 0.153 |


## verdict (CFD correction)

|  | passes | of | failures |
|---|---|---|---|
| elevons, CFD | 12 | 12 | [] |
| rotors only, CFD | 1 | 12 | ['forward 1.5 s, headwind 0 m/s: fails roll holds', 'forward 1.5 s, headwind 6 m/s: fai… |


## truss: even lift

|  | spar peak moment [N·m] | spar stress [MPa] | spar reserve | tip deflection, upper [mm] | tip deflection, lower [mm] | motor station, upper [mm] | motor station, lower [mm] | propeller gap change [mm] | propeller gap under load [mm] | least buckling margin | truss mass [g] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| posts + X diagonals | 0.667 | 11.5 | 34.8 | -0.0909 | -0.0909 | 0.0351 | 0.0351 | -2.03e-17 | 51.4 | 77.4 | 127 |
| posts only | 1.25 | 21.6 | 18.5 | 1.98 | 1.98 | 1.49 | 1.49 | -2.17e-16 | 51.4 | ∞ | 40.8 |
| no truss (two cantilevers) | 1.25 | 21.6 | 18.5 | 1.98 | 1.98 | 1.49 | 1.49 | 0 | 51.4 | ∞ | 0 |


## truss: upper wing 75 %

|  | spar peak moment [N·m] | spar stress [MPa] | spar reserve | tip deflection, upper [mm] | tip deflection, lower [mm] | motor station, upper [mm] | motor station, lower [mm] | propeller gap change [mm] | propeller gap under load [mm] | least buckling margin | truss mass [g] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| posts + X diagonals | 0.734 | 12.7 | 31.6 | -0.146 | -0.0361 | 0.0349 | 0.0353 | -3.83e-04 | 51.4 | 575 | 127 |
| posts only | 1.58 | 27.3 | 14.7 | 1.91 | 2.05 | 1.49 | 1.49 | 5.86e-04 | 51.4 | ∞ | 40.8 |
| no truss (two cantilevers) | 3.02 | 52.1 | 7.68 | 6.38 | -2.42 | 4.63 | -1.66 | 6.28 | 57.7 | ∞ | 0 |


## ground stance

| key | value |
|---|---|
| CG height above the ground [m] | 0.325 |
| least tip-over angle [deg] | 39.3 |
| wind that tips it, broadside [m/s] | 8.91 |
| wind that tips it, edge-on [m/s] | 21.1 |
| touchdown load [N] | 122 |
| per foot, two feet first [N] | 61 |
| foot rod Euler load [N] | 766 |
| foot rod margin, two feet first | 12.6 |


## FEA (Talos) cases

|  | part | bracing | kind | status | description | note |
|---|---|---|---|---|---|---|
| frame_x_A | frame | x | static | NOT RUN | 4 g pull-up, the lift elliptic on both wings, the motors' and wings' inertia against it… | NOT RUN: frame_x_A in _runs/kestrel/fea/frame_x_A (run=False) |
| frame_x_B | frame | x | static | NOT RUN | 4 g with the upper wing carrying 75 % (one wing stalling first) (ultimate ×1.5) | NOT RUN: frame_x_B in _runs/kestrel/fea/frame_x_B (run=False) |
| frame_x_C | frame | x | static | NOT RUN | the heading manoeuvre in the hover: M1 and M4 at 12.0 N, M2 and M3 at 0.25 N (the diago… | NOT RUN: frame_x_C in _runs/kestrel/fea/frame_x_C (run=False) |
| frame_x_modes | frame | x | modes | NOT RUN | the first modes with the motor units as point masses at the stations |  |
| frame_posts_A | frame | posts | static | NOT RUN | 4 g pull-up, the lift elliptic on both wings, the motors' and wings' inertia against it… | NOT RUN: frame_posts_A in _runs/kestrel/fea/frame_posts_A (run=False) |
| frame_posts_B | frame | posts | static | NOT RUN | 4 g with the upper wing carrying 75 % (one wing stalling first) (ultimate ×1.5) | NOT RUN: frame_posts_B in _runs/kestrel/fea/frame_posts_B (run=False) |
| frame_posts_C | frame | posts | static | NOT RUN | the heading manoeuvre in the hover: M1 and M4 at 12.0 N, M2 and M3 at 0.25 N (the diago… | NOT RUN: frame_posts_C in _runs/kestrel/fea/frame_posts_C (run=False) |
| frame_posts_modes | frame | posts | modes | NOT RUN | the first modes with the motor units as point masses at the stations |  |
| joint_post | joint |  | static | NOT RUN | the post's 5.5 N (ultimate) into the printed socket, the spar bore held; build axis z | NOT RUN: joint_post in _runs/kestrel/fea/joint_post (run=False) |


## plane-frame reference for the FEA

|  | bracing | case | spar stress, limit [MPa] | spar stress, ultimate [MPa] | tip deflection, upper, limit [mm] | tip deflection, ultimate [mm] | spar reserve (limit) | first mode, symmetric [Hz] | first mode, antisymmetric [Hz] |
|---|---|---|---|---|---|---|---|---|---|
| frame_x_A | x | A | 11.5 | 17.3 | -0.0909 | -0.136 | 34.8 | — | — |
| frame_x_B | x | B | 12.7 | 19 | -0.146 | -0.219 | 31.6 | — | — |
| frame_x_modes | x | modes | — | — | — | — | — | 261 | 1.64e+03 |
| frame_posts_A | posts | A | 21.6 | 32.4 | 1.98 | 2.97 | 18.5 | — | — |
| frame_posts_B | posts | B | 27.3 | 40.9 | 1.91 | 2.86 | 14.7 | — | — |
| frame_posts_modes | posts | modes | — | — | — | — | — | 15.6 | 1.62e+03 |
| frame_none_A | none | A | 21.6 | 32.4 | 1.98 | 2.97 | 18.5 | — | — |
| frame_none_B | none | B | 52.1 | 78.1 | 6.38 | 9.57 | 7.68 | — | — |
| frame_none_modes | none | modes | — | — | — | — | — | 15.6 | 15.6 |


## Part II — payload bay

|  | mass [kg] | CG x [mm] | battery x [mm] | battery at its stop | static margin | hover T/W | hover power [W] | stall speed [m/s] | power at 16 m/s [W] | endurance at 16 m/s [min] | endurance best [min] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| no capsule (Part I) | 1.69 | 215 | 220 | yes | 0.0955 | 2.89 | 221 | 10.9 | 110 | 21.3 | 40.2 |
| capsule, empty | 1.78 | 216 | 220 | yes | 0.0893 | 2.75 | 238 | 11.2 | 112 | 21 | 35.3 |
| capsule, 0.3 kg (nominal) | 2.08 | 218 | 220 | yes | 0.0725 | 2.35 | 299 | 12.1 | 117 | 20 | 28.9 |
| capsule, 0.5 kg (full) | 2.28 | 219 | 220 | yes | 0.0638 | 2.14 | 343 | 12.7 | 120 | 19.5 | 27.3 |


## capsule transitions

|  | passes | of | failures |
|---|---|---|---|
| capsule 0.3 kg | 4 | 4 | [] |
| capsule 0.5 kg | 4 | 4 | [] |


## elevons

| the elevons | value |
|---|---|
| elevon chord [mm] | 39.2 |
| span each [mm] | 225 |
| area, four [m²] | 0.0353 |
| share of the wing area | 0.126 |
| hinge x [mm] | 301 |
| mass added (4 servos + linkages) [g] | 48 |


## elevon hinge moments

|  | V [m/s] | elevon area, one [m²] | elevon chord [m] | hinge moment [N·m] | hinge moment, hover slipstream [N·m] | slipstream speed in the hover [m/s] | servo torque rating [N·m] | servo margin, cruise | servo margin, hover | time to 20° [s] |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 m/s, α 5°, δ 20° | 16 | 0.00882 | 0.0392 | -0.0121 | -0.00671 | 12.8 | 0.17 | 14.1 | 25.3 | 0.04 |
| 20 m/s, α 3°, δ 25° | 20 | 0.00882 | 0.0392 | -0.0219 | -0.00839 | 12.8 | 0.17 | 7.77 | 20.3 | 0.04 |
| 25 m/s, α 2°, δ 25° | 25 | 0.00882 | 0.0392 | -0.0334 | -0.00839 | 12.8 | 0.17 | 5.09 | 20.3 | 0.04 |


## elevon authority

|  | one elevon ΔL [N] | roll / heading from elevons [N·m] | pitch from elevons [N·m] | rotors: about the thrust axis [N·m] | rotors: pitch [N·m] | elevons / rotors, about the axis |
|---|---|---|---|---|---|---|
| hover (slipstream only) | 2.63 | 4.07 | 0.44 | 0.242 | 2.1 | 16.8 |
| cruise 16 m/s | 4.09 | 6.34 | 0.684 | 0.394 | 2.1 | 16.1 |


## elevon transitions

|  | passes | of | failures |
|---|---|---|---|
| 4Q + elevons | 11 | 12 | ['forward 1.5 s, headwind 0 m/s: fails holds height'] |
| 4Q rotors only | 3 | 12 | ['forward 1.5 s, headwind 0 m/s: fails roll holds', 'forward 1.5 s, headwind 6 m/s: fai… |


## faired struts

| faired struts | value |
|---|---|
| sleeve volume [cm³] | 236 |
| sleeve skin [cm²] | 1.23e+03 |
| mass, foam 30 kg/m³ + film [g] | 10.8 |
| mass, printed LW-PLA 0.45 mm [g] | 36.1 |
| mass in the table (SLEEVE_KG_M) [g] | 10.3 |
| Cd0 round | 0.0665 |
| Cd0 faired | 0.0234 |
| truss drag at 16 m/s, round [N] | 2.12 |
| truss drag at 16 m/s, faired [N] | 0.232 |
| broadside area of the struts, round [m²] | 0.015 |
| broadside area of the struts, faired [m²] | 0.0564 |
| wings' planform (the broadside reference) [m²] | 0.28 |


## faired transitions

|  | passes | of | failures |
|---|---|---|---|
| faired + elevons | 10 | 12 | ['back 5 s, headwind 0 m/s: fails holds height', 'back 5 s, headwind 6 m/s: fails holds… |


## section study (Zephyr)

|  | t/c | α₀ [deg] | cm₀ | cm (thin) | cl max (cruise Re) | α cl max | cl max (transition Re) | cd at cruise cl | L/D at cruise cl | L/D max | cl kept 4° past stall | x_ac | elevon τ | elevon x_cp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NACA 0012 | 0.12 | -7.44e-15 | -2.43e-17 | 0 | 1.06 | 13 | 0.964 | 0.0124 | 30.5 | 42.4 | 0.654 | 0.235 | 0.465 | 0.405 |
| NACA 2412 | 0.12 | -1.9 | -0.0506 | -0.0531 | 1.2 | 15 | 1.16 | 0.0115 | 33.1 | 58.7 | 0.732 | 0.215 | 0.541 | 0.395 |
| NACA 22112 | 0.12 | -0.952 | 8.13e-04 | -0.00134 | 1.25 | 14 | 1.11 | 0.0128 | 29.7 | 44.2 | 0.668 | 0.237 | 0.488 | 0.404 |
| NACA 23112 | 0.12 | -1.17 | -2.51e-04 | 0.00113 | 1.26 | 13 | 1.15 | 0.0134 | 28.3 | 47.5 | 0.871 | 0.237 | 0.496 | 0.405 |
| NACA 24112 | 0.12 | -1.4 | -0.00525 | 5.81e-05 | 1.23 | 12 | 1.17 | 0.0137 | 27.6 | 51.8 | 0.91 | 0.238 | 0.513 | 0.405 |
| NACA 23112 (k2/k1 0.015) | 0.12 | -0.751 | 0.017 | 0.0221 | 1.22 | 13 | 1.1 | 0.0142 | 26.7 | 45.9 | 0.856 | 0.25 | 0.482 | 0.409 |
| NACA 23112 (k2/k1 0.02) | 0.12 | -0.536 | 0.026 | 0.0348 | 1.19 | 13 | 1.07 | 0.0148 | 25.8 | 44.9 | 0.842 | 0.251 | 0.478 | 0.411 |
| NACA 23112 (k2/k1 0.025) | 0.12 | -0.319 | 0.035 | 0.0475 | 1.17 | 13 | 1.04 | 0.0152 | 24.9 | 43.9 | 0.822 | 0.26 | 0.478 | 0.412 |
| NACA 23112 (k2/k1 0.03) | 0.12 | -0.0971 | 0.0438 | 0.0603 | 1.15 | 13 | 1 | 0.0157 | 24.2 | 42.9 | 0.795 | 0.267 | 0.482 | 0.414 |
| NACA 24112 (k2/k1 0.06) | 0.12 | -0.931 | 0.0137 | 0.0241 | 1.17 | 12 | 1.11 | 0.0145 | 26.2 | 50.2 | 0.922 | 0.253 | 0.494 | 0.408 |
| NACA 24112 (k2/k1 0.08) | 0.12 | -0.736 | 0.0233 | 0.0403 | 1.15 | 12 | 1.08 | 0.0151 | 25.2 | 49.2 | 0.915 | 0.261 | 0.49 | 0.411 |
| NACA 23110 (k2/k1 0.02) | 0.1 | -0.536 | 0.022 | 0.0348 | 1.14 | 12 | 1.06 | 0.0137 | 27.7 | 45.5 | 0.904 | 0.232 | 0.522 | 0.41 |
| NACA 23110 (k2/k1 0.025) | 0.1 | -0.333 | 0.0312 | 0.0475 | 1.12 | 12 | 1.03 | 0.0142 | 26.7 | 44.5 | 0.892 | 0.241 | 0.513 | 0.412 |

*1 more columns not shown: confidence (attached)*


## section trim

|  | cm needed | cm of the section | elevon [deg] | Δcd (elevon) | cd cruise | trim drag / section drag | cl lost to the elevon | trims within ±10° |
|---|---|---|---|---|---|---|---|---|
| NACA 0012 | 0.0363 | -0.0124 | -5.49 | 0.00292 | 0.0124 | 0.234 | -0.314 | yes |
| NACA 2412 | 0.0363 | -0.0616 | -13.1 | 0.0109 | 0.0115 | 0.948 | -0.674 | no |
| NACA 22112 | 0.0363 | -0.00943 | -5.51 | 0.00208 | 0.0128 | 0.163 | -0.297 | yes |
| NACA 23112 | 0.0363 | -0.0117 | -5.77 | 0.00268 | 0.0134 | 0.2 | -0.309 | yes |
| NACA 24112 | 0.0363 | -0.0196 | -6.83 | 0.00418 | 0.0137 | 0.304 | -0.361 | yes |
| NACA 23112 (k2/k1 0.015) | 0.0363 | 0.00381 | -3.67 | 0.0018 | 0.0142 | 0.126 | -0.205 | yes |
| NACA 23112 (k2/k1 0.02) | 0.0363 | 0.0134 | -2.5 | 0.00146 | 0.0148 | 0.0987 | -0.143 | yes |
| NACA 23112 (k2/k1 0.025) | 0.0363 | 0.0231 | -1.4 | 9.53e-04 | 0.0152 | 0.0625 | -0.0815 | yes |
| NACA 23112 (k2/k1 0.03) | 0.0363 | 0.0328 | -0.356 | 2.79e-04 | 0.0157 | 0.0177 | -0.0211 | yes |
| NACA 24112 (k2/k1 0.06) | 0.0363 | -0.0023 | -4.41 | 0.00252 | 0.0145 | 0.174 | -0.244 | yes |
| NACA 24112 (k2/k1 0.08) | 0.0363 | 0.00919 | -2.96 | 0.00204 | 0.0151 | 0.136 | -0.169 | yes |
| NACA 23110 (k2/k1 0.02) | 0.0363 | 0.0103 | -2.94 | 0.00139 | 0.0137 | 0.101 | -0.163 | yes |
| NACA 23110 (k2/k1 0.025) | 0.0363 | 0.0197 | -1.81 | 9.99e-04 | 0.0142 | 0.0702 | -0.103 | yes |


## section pick

| key | value |
|---|---|
| pick | NACA 23112 (k2/k1 0.03) |


## stiffness and modes per bracing

|  | k symmetric [N/mm] | k antisymmetric [N/mm] | first mode, symmetric [Hz] | first mode, antisymmetric [Hz] |
|---|---|---|---|---|
| posts + X diagonals | 327 | 1.3e+04 | 261 | 1.64e+03 |
| posts only | 1.17 | 1.26e+04 | 15.6 | 1.62e+03 |
| no truss (two cantilevers) | 1.17 | 1.17 | 15.6 | 15.6 |


## bracing: mass and drag

| bracing · struts | mass [kg] | Δ mass vs Part I [g] | bracing drag area [cm²] | Cd0 |
|---|---|---|---|---|
| posts + X diagonals · round | 1.69 | 0 | 136 | 0.0665 |
| posts + X diagonals · faired | 1.7 | 10.3 | 14.8 | 0.0234 |
| posts only · round | 1.6 | -86.4 | 54.2 | 0.0375 |
| posts only · faired | 1.6 | -83.1 | 5.91 | 0.0203 |
| no truss · round | 1.56 | -127 | 0 | 0.0181 |
| no truss · faired | 1.56 | -127 | 0 | 0.0181 |


## Kestrel-4Q-2 against 4Q

|  | mass [kg] | static margin | Cd0 | L/D at 16 m/s | L/D best | power at 16 m/s [W] | endurance at 16 m/s [min] | endurance best [min] | trim by thrust at 16 m/s | section | elevons | bracing | struts | transitions passed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Kestrel-4Q (Part I) | 1.69 | 0.0955 | 0.0665 | 4.96 | 7.32 | 110 | 21.3 | 40.2 | yes | NACA 0012 | no | x | round | 11 of 12 |
| Kestrel-4Q-2 | 1.65 | 0.0916 | 0.0257 | 10.6 | 12.1 | 59.1 | 39.7 | 57 | no | NACA 23112 (k2/k1 0.03) (Zephyr) | yes | posts | faired | 8 of 12 (6 of 6 back with the 13 m/s entry) |

*1 more columns not shown: hover pitch [rad/s²]*


## 4Q-2 cruise

| V [m/s] | CL | feasible | note | α [deg] | L/D | drag [N] | Cm about CG | upper rotor [N] | lower rotor [N] | trim by thrust | electrical [W] | endurance [min] | range [km] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 12 | 0.656 | yes | the upper pair below T_min: thrust alone cannot trim | 9.11 | 12.1 | 1.34 | -0.06 | -0.0461 | 0.717 | no | 45.1 | 52 | 37.4 |
| 14 | 0.482 | yes | the upper pair below T_min: thrust alone cannot trim | 6.67 | 11.7 | 1.38 | -0.0441 | -0.0359 | 0.728 | no | 51.2 | 45.8 | 38.5 |
| 16 | 0.369 | yes | the upper pair below T_min: thrust alone cannot trim | 5.08 | 10.6 | 1.53 | -0.0338 | -0.00132 | 0.765 | no | 59.1 | 39.7 | 38.1 |
| 20 | 0.236 | yes | the upper pair below T_min: thrust alone cannot trim | 3.22 | 8.02 | 2.02 | -0.0216 | 0.118 | 0.892 | no | 92.2 | 25.4 | 30.5 |
| 25 | 0.151 | yes |  | 2.03 | 5.54 | 2.92 | -0.0138 | 0.336 | 1.13 | yes | 169 | 13.9 | 20.9 |


## 4Q-2 transitions

|  | passes | of | failures |
|---|---|---|---|
| Kestrel-4Q-2 | 8 | 12 | ['back 3 s, headwind 0 m/s: fails holds height', 'back 3 s, headwind 6 m/s: fails holds… |


## 4Q-2 back transition vs entry speed

| entry speed [m/s] · headwind [m/s] · t_tr [s] | height loss [m] | height gain [m] | pitch error max [deg] | time stalled, free stream [s] | α blown max while stalled [deg] | moment shortfall [s] | least pitch margin | peak rotor thrust / T_max | elevon max [deg] | roll demand / authority | roll demand / rotors' authority | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 · 0 · 1.5 | 1.07 | 4.81 | 6.8 | 0.958 | 63.8 | 0 | 0.627 | 0.406 | 9.28 | 0.3 | 1.86 | yes |
| 16 · 0 · 3 | 0.172 | 12.3 | 3.44 | 1.35 | 62.6 | 0 | 0.816 | 0.365 | 4.55 | 0.116 | 1.53 | no |
| 16 · 0 · 5 | 0.172 | 16 | 1.73 | 1.02 | 57 | 0 | 0.849 | 0.356 | 3.71 | 0.0789 | 0.766 | no |
| 16 · 6 · 1.5 | 0.685 | 4.77 | 6.8 | 0.958 | 63.8 | 0 | 0.627 | 0.383 | 9.28 | 0.3 | 1.86 | yes |
| 16 · 6 · 3 | 0.172 | 12.3 | 3.44 | 1.35 | 62.6 | 0 | 0.816 | 0.365 | 4.55 | 0.116 | 1.53 | no |
| 16 · 6 · 5 | 0.172 | 16 | 1.73 | 1.02 | 57 | 0 | 0.839 | 0.425 | 3.71 | 0.0789 | 0.766 | no |
| 13 · 0 · 1.5 | 0.314 | 1.99 | 7.24 | 0.9 | 74.8 | 0 | 0.68 | 0.448 | 7.96 | 0.308 | 1.2 | yes |
| 13 · 0 · 3 | 1.32 | 5.08 | 4.34 | 1.3 | 71.8 | 0 | 0.816 | 0.407 | 4.56 | 0.186 | 1.15 | yes |
| 13 · 0 · 5 | 1.96 | 9.21 | 2.06 | 1.25 | 63.8 | 0 | 0.842 | 0.409 | 3.91 | 0.13 | 0.908 | yes |
| 13 · 6 · 1.5 | 0.161 | 1.99 | 7.24 | 0.9 | 74.8 | 0 | 0.68 | 0.383 | 7.96 | 0.308 | 1.2 | yes |
| 13 · 6 · 3 | 0.914 | 5.08 | 4.34 | 1.3 | 71.8 | 0 | 0.816 | 0.395 | 4.56 | 0.186 | 1.15 | yes |
| 13 · 6 · 5 | 1.33 | 9.21 | 2.06 | 1.25 | 63.8 | 0 | 0.84 | 0.428 | 3.91 | 0.13 | 0.908 | yes |
| 12 · 0 · 1.5 | 0.196 | 1.21 | 7.33 | 0.86 | 75.1 | 0 | 0.703 | 0.469 | 7.39 | 0.295 | 1.01 | yes |
| 12 · 0 · 3 | 0.676 | 3.12 | 4.34 | 1.3 | 72.1 | 0 | 0.81 | 0.403 | 4.72 | 0.198 | 0.995 | yes |
| 12 · 0 · 5 | 1.96 | 6.52 | 2.13 | 1.36 | 63.6 | 0 | 0.84 | 0.409 | 3.94 | 0.15 | 0.896 | yes |
| 12 · 6 · 1.5 | 0.157 | 1.21 | 7.33 | 0.86 | 75.1 | 0 | 0.703 | 0.385 | 7.39 | 0.295 | 1.01 | yes |
| 12 · 6 · 3 | 0.157 | 3.12 | 4.34 | 1.3 | 72.1 | 0 | 0.81 | 0.401 | 4.72 | 0.198 | 0.995 | yes |
| 12 · 6 · 5 | 1.41 | 6.52 | 2.13 | 1.36 | 63.6 | 0 | 0.84 | 0.429 | 3.94 | 0.15 | 0.896 | yes |


## 4Q-2 with the slow-entry rule

| key | value |
|---|---|
| forward | 6 of 6 |
| back, entered at 13 m/s | 6 of 6 |


## gusts, 4Q-2

| key | value |
|---|---|
| forward, gust +3 m/s | yes |
| forward, gust -3 m/s | yes |
| back, gust +3 m/s | no |
| back, gust -3 m/s | no |
| back, gust +3 m/s, entered at 13 m/s | yes |
| back, gust -3 m/s, entered at 13 m/s | yes |


## drag vs speed

|  | V stall [m/s] | V of least drag [m/s] | least drag [N] | best L/D | drag at 16 m/s [N] | parasite at 16 [N] | induced at 16 [N] | power at 16 m/s [W] | least power [W] | V of least power [m/s] | V max, thrust = drag [m/s] | full thrust at 16 m/s [N] | thrust / drag at 16 m/s | power at V max [W] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Kestrel-4Q-1 (round struts, X, no elevons) | 10.9 | 11 | 2.26 | 7.32 | 3.34 | 2.92 | 0.415 | 111 | 61.7 | 11 | 30.1 | 32.4 | 9.72 | 643 |
| Kestrel-4Q-2 (elevons, faired posts, no X) | 9.56 | 12.5 | 1.34 | 12.1 | 1.53 | 1.13 | 0.398 | 65.3 | 42.2 | 10 | 33.3 | 32.4 | 21.2 | 1.49e+03 |


## manoeuvrability at 16 m/s

|  | CL | thrust per rotor [N] | ΔT available [N] | pitch: rotors [rad/s²] | pitch: elevons [rad/s²] | pitch: total [rad/s²] | yaw: rotors [rad/s²] | roll: rotors' torque [N·m] | roll: elevons [N·m] | roll acceleration [rad/s²] | steady roll rate [deg/s] | load factor available | turn rate [deg/s] | turn radius [m] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Kestrel-4Q-1 | 0.377 | 0.834 | 0.584 | 14.8 | 0 | 14.8 | 8.57 | 0.394 | 0 | 3.71 | 24.2 | 2.15 | 66.9 | 13.7 |
| Kestrel-4Q-2 | 0.369 | 0.382 | 0.132 | 3.18 | 48.3 | 51.5 | 1.82 | 0.394 | 6.34 | 59.5 | 414 | 2.8 | 91.9 | 9.98 |


## the iteration log

| iteration | what changed | source | take-off mass [kg] | static margin [chords] | hover thrust / weight | stall speed [m/s] | Cd0 | L/D at the cruise | cruise power [W] | endurance at the cruise [min] | transitions passing | checks holding | failing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 — envelope | — | spec + layout + 5 assumptions | 2 | 0.12 | 2 | 11.3 | — | 6 | 108 | 21.7 | — | 6 / 6 | — |
| 1 — CAD, Boreas, lattice, drag build-up | the CG target at 10 % of the chord (battery station) | kestrel_systems, Boreas, PEREGRINE lattice | 1.69 | 0.0955 | 2.89 | 10.9 | 0.0665 | 4.96 | 110 | 21.3 | — | 8 / 9 | battery inside its bay (not at a stop) |
| 2 — transition model | elevons; pitch-over ≥ 3 s; back entry 13 m/s | kestrel_transition (model polar) | 1.69 | 0.0955 | 2.89 | 10.9 | 0.0665 | 4.96 | 110 | 21.3 | 8 of 8 (≥ 3 s) | 5 / 7 | rotors only, every schedule, cruise entry: passing, elevons, every schedule: passing |
| 3 — CFD plugged in | the polar read against the CFD (sweep correction, 10 solved cases) | 38_kestrel run on an OpenFOAM machine (uploaded notebook), Part I points, screening qua… | 1.69 | 0.164 | 2.89 | 13.3 | 0.0532 | 4.96 | 110 | 21.3 | 12 of 12 | 5 / 7 | CL max below 20°, CFD / the model's polar peak, blown cases solved |
| 4 — payload bay (Part II) | pod 430 → 470 mm, ogive nose, the capsule on the axis | CAD bays + mass table + transitions | 2.28 | 0.0638 | 2.14 | 12.7 | 0.0665 | — | 120 | 19.5 | 8 of 8 | 5 / 6 | endurance at the cruise, full [min] |
| 5 — elevons (Part II) | four 28 % surfaces from 275 mm to the tips, τ from Zephyr | CAD + kestrel_control + transitions | 1.73 | 0.0955 | 2.89 | 10.9 | 0.0665 | — | — | — | 11 of 12 | 3 / 4 | transitions, rotors only: passing |
| 6 — faired struts (Part II) | teardrop sleeves on posts and diagonals | drag build-up + CAD + transitions | 1.74 | 0.0955 | 2.89 | 10.9 | 0.0234 | 11.5 | 59.2 | 39.6 | 10 of 12 | 4 / 5 | trim by thrust alone, faired |
| 7 — reflexed section (Part II, Zephyr) | NACA 23112 (k2/k1 0.03) instead of NACA 0012 | vegeta.zephyr / NeuralFoil | 1.69 | 0.0955 | — | 9.66 | — | — | — | — | — | 4 / 5 | CFD slab cases of the finalists solved |
| 8 — bracing (Part II) | the X diagonals dropped, posts kept | kestrel_structure (frame, modes) | 1.6 | — | — | — | 0.0203 | — | — | — | — | 4 / 4 | — |
| 9 — Kestrel-4Q-2 | elevons + faired posts + no X + the section; the slow-entry rule (≤ 13 m/s) | all of the above | 1.65 | 0.0916 | 2.95 | 9.56 | 0.0257 | 10.6 | 59.1 | 39.7 | 12 of 12 (back at 13 m/s) | 8 / 9 | back transitions entered at the cruise speed: passing |


## CHK0

| check | value | target | holds | note |
|---|---|---|---|---|
| take-off mass [kg] | 2 | ≤ 2 | ✔ |  |
| endurance at the cruise [min] | 21.7 | ≥ 20 | ✔ |  |
| static margin [chords] | 0.12 | 0.05 – 0.15 | ✔ |  |
| stall speed / cruise speed | 0.705 | ≤ 0.8 | ✔ |  |
| cruise CL / CL max | 0.496 | ≤ 0.6 | ✔ |  |
| hover power / battery → hover time [min] | 10.3 | ≥ 5 | ✔ | the hover is for take-off, landing and the transitions |


## CHK1

| check | value | target | holds | note |
|---|---|---|---|---|
| layout rules broken | 0 | ≤ 0 | ✔ |  |
| take-off mass [kg] | 1.69 | ≤ 2 | ✔ |  |
| hover thrust / weight | 2.89 | ≥ 2 | ✔ |  |
| battery inside its bay (not at a stop) | no | == yes | ✘ | at its forward stop: the margin is the bay's position, not the battery's (§ 11) |
| static margin [chords] | 0.0955 | 0.05 – 0.15 | ✔ |  |
| stall speed / cruise speed | 0.682 | ≤ 0.8 | ✔ |  |
| cruise CL / CL max | 0.465 | ≤ 0.6 | ✔ |  |
| endurance at the cruise [min] | 21.3 | ≥ 20 | ✔ |  |
| trim by thrust alone at the cruise | yes | == yes | ✔ | fails with faired struts: elevons (§ 6, Part II) |


## CHK2

| check | value | target | holds | note |
|---|---|---|---|---|
| rotors only, every schedule, cruise entry: passing | 0.25 | ≥ 1 | ✘ | 3 of 12: fails on roll |
| elevons, every schedule: passing | 0.917 | ≥ 1 | ✘ | 11 of 12: the 1.5 s pitch-over sinks |
| elevons, schedules ≥ 3 s: passing | 1 | ≥ 1 | ✔ | 8 of 8 |
| height loss in a transition [m] | 0.564 | ≤ 3 | ✔ | 3 s schedules |
| back transition entered at 13 m/s: height gain [m] | 0.0759 | ≤ 10 | ✔ |  |
| back at 13 m/s: blown span's α while stalled [deg] | 43.9 | ≤ 45 | ✔ | the regime the model knows least |
| gusts ±3 m/s, both ways: passing | 1 | ≥ 1 | ✔ |  |


## CHK3

| check | value | target | holds | note |
|---|---|---|---|---|
| lift slope, CFD / model | 0.986 | 0.85 – 1.15 | ✔ |  |
| Cd0, CFD / model | 0.8 | ≤ 1.2 | ✔ | the model may be conservative, not optimistic |
| CL max below 20° on the sweep's angles, CFD / model | 0.959 | ≥ 0.8 | ✔ | a 5° grid: the peak itself waits for the fine sweep (notebook 40) |
| CL max below 20°, CFD / the model's polar peak | 0.668 | ≥ 0.8 | ✘ | the model's 0.81 against the CFD's 0.54 at 15°: the stall speed and the entry rule hang… |
| static margin from the CFD's Cmα [chords] | 0.164 | 0.05 – 0.2 | ✔ |  |
| blown cases solved | 0 | ≥ 1 | ✘ | the transition points with rotor disks: the question this CFD was set up for |
| transitions on the CFD's aerodynamics (elevons): passing | 1 | ≥ 0.666667 | ✔ | all but the 1.5 s pitch-over |


## CHK4

| check | value | target | holds | note |
|---|---|---|---|---|
| bay rules broken | 0 | ≤ 0 | ✔ |  |
| take-off mass, full capsule [kg] | 2.28 | ≤ 2.3 | ✔ | 2.0 + the 0.3 kg the bay adds over the spec's payload |
| hover thrust / weight, full | 2.14 | ≥ 2 | ✔ |  |
| static margin, full [chords] | 0.0638 | 0.05 – 0.15 | ✔ |  |
| endurance at the cruise, full [min] | 19.5 | ≥ 20 | ✘ | on 4Q's round struts; 4Q-2's drag makes it (§ 16) |
| transitions with the capsule: passing | 1 | ≥ 1 | ✔ |  |


## CHK5

| check | value | target | holds | note |
|---|---|---|---|---|
| servo margin (0.17 N·m / hinge moment), least | 5.09 | ≥ 2 | ✔ |  |
| hover heading authority, elevons / rotors | 16.8 | ≥ 2 | ✔ |  |
| transitions, 4Q + elevons: passing | 0.917 | ≥ 0.666667 | ✔ | 11 of 12 (all but the 1.5 s pitch-over) |
| transitions, rotors only: passing | 0.25 | ≥ 1 | ✘ | 3 of 12 — the change is needed |


## CHK6

| check | value | target | holds | note |
|---|---|---|---|---|
| Cd0 faired / round | 0.352 | ≤ 0.5 | ✔ |  |
| endurance at the cruise, faired [min] | 39.6 | ≥ 20 | ✔ |  |
| sleeve mass [g] | 10.8 | ≤ 40 | ✔ |  |
| trim by thrust alone, faired | no | == yes | ✘ | never: the elevons trim |
| transitions, faired + elevons, cruise entry: passing | 0.833 | ≥ 0.666667 | ✔ | 10 of 12: the low-drag aircraft balloons (§ 16: the slow entry) |


## CHK7

| check | value | target | holds | note |
|---|---|---|---|---|
| the pick trims the cruise within [deg] | 0.356 | ≤ 3 | ✔ | NACA 0012 needs -5.5° |
| cl max, pick / NACA 0012 | 1.08 | ≥ 0.97 | ✔ |  |
| section drag at the cruise cl, pick / 0012 (incl. the 0012's trim elevon) | 1.02 | ≤ 1.1 | ✔ |  |
| surrogate confidence on the attached branch | 0.911 | ≥ 0.8 | ✔ |  |
| CFD slab cases of the finalists solved | 0 | ≥ 1 | ✘ | NOT RUN here: notebook 40 |


## CHK8

| check | value | target | holds | note |
|---|---|---|---|---|
| posts only: first mode / pitch loop bandwidth | 8.17 | ≥ 3 | ✔ | the structure must not be in the loop |
| posts only: first mode / rotor 1P | 0.142 | ≤ 0.5 | ✔ | clear below the 1P |
| mass saved by dropping the X [g] | 93.4 | ≥ 50 | ✔ |  |
| spar stress reserve at 4 g, one wing 75 % (posts only) | 14.7 | ≥ 3 | ✔ |  |


## CHK8b

| check | value | target | holds | note |
|---|---|---|---|---|
| FEA cases solved | 0 | ≥ 9 | ✘ | CalculiX: notebook 40 on a machine that has it |


## CHK9

| check | value | target | holds | note |
|---|---|---|---|---|
| take-off mass [kg] | 1.65 | ≤ 2 | ✔ |  |
| hover thrust / weight | 2.95 | ≥ 2 | ✔ |  |
| static margin [chords] | 0.0916 | 0.05 – 0.15 | ✔ |  |
| stall speed / cruise speed | 0.597 | ≤ 0.8 | ✔ |  |
| endurance at the cruise [min] | 39.7 | ≥ 20 | ✔ |  |
| forward transitions: passing | 1 | ≥ 1 | ✔ | 6 of 6 |
| back transitions entered at the cruise speed: passing | 0.333 | ≥ 1 | ✘ | balloons past 10 m: the rule below |
| back transitions entered at 13 m/s: passing | 1 | ≥ 1 | ✔ | 6 of 6 |
| gusts ±3 m/s with the slow entry: passing | 1 | ≥ 1 | ✔ |  |


## status

| key | value |
|---|---|
| CFD | NOT RUN here |
| FEA (Talos): frame x / posts, joint | NOT RUN (no CalculiX here; notebook 40) |
| CAD | run |
| mass table | run |
| propulsion (Boreas tables) | run |
| hold point (allocation) | run |
| vortex lattice + drag build-up + polar | run |
| cruise | run |
| transitions (model polar) | run (36 in the matrix, 25 pass) |
| truss frame + ground stance | run |
| Part II: elevons, faired struts, bracing, 4Q-2 | run (8 of 12 4Q-2 transitions pass) |
| Part II: reflexed section (Zephyr) | run |
| X-wing | its own notebook, 39_xwing |


## prop_lab calibration

| key | value |
|---|---|
| cl_scale | 1 |
| cd_scale | 1 |
| cn_factor | 1 |
| on_file | no |
| meta | {'source': '41_prop_lab', 'quality': 'screening', 'date': '2026-10-16', 'propeller': "9… |



---

## X-wing — notebook 39 (X-A and X-B against Kestrel-4Q and 4Q-2)

*39_xwing — exported 2026-10-16 12:34*

## variants

|  | panel_length | chord | motor_r |
|---|---|---|---|
| X-A (the biplane's wing area) | 0.5 | 0.14 | 0.46 |
| X-B (twice the wing area) | 0.7 | 0.2 | 0.66 |


## layout

|  | panel_length | chord | motor_r | span | span_projected | height | S_ref | AR_wing | h | motor_y | prop_gap | prop_to_pod | prop_beyond_tip | elevon_y_in |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| X-A (the biplane's wing area) | 0.5 | 0.14 | 0.46 | 1.11 | 0.785 | 0.785 | 0.28 | 7.93 | 0.325 | 0.46 | 0.422 | 0.291 | 0.0193 | 0.346 |
| X-B (twice the wing area) | 0.7 | 0.2 | 0.66 | 1.51 | 1.07 | 1.07 | 0.56 | 7.55 | 0.467 | 0.66 | 0.705 | 0.491 | 0.0193 | 0.546 |

*1 more columns not shown: elevon_span_side*


## masses

|  | X-A (the biplane's wing area) | X-B (twice the wing area) | Kestrel-4Q |
|---|---|---|---|
| propulsion | 0.736 | 0.736 | 0.736 |
| structure | 0.478 | 0.769 | — |
| systems | 0.208 | 0.208 | — |
| payload | 0.105 | 0.105 | — |
| margin | 0.06 | 0.06 | — |


## spars

|  | n | panel lift [N] | root moment [N·m] | spar OD [mm] | spar stress [MPa] | spar reserve | tip deflection [mm] | tip deflection / length | first bending, with the tip motor [Hz] | spar mass per panel [g] | sized by |
|---|---|---|---|---|---|---|---|---|---|---|---|
| X-A (the biplane's wing area) | 4 | 22 | 2.31 | 9 | 50.9 | 7.85 | 10.3 | 0.0206 | 12.3 | 19.5 | first mode |
| X-B (twice the wing area) | 4 | 26 | 3.49 | 13.5 | 30.5 | 13.1 | 9.23 | 0.0132 | 12.3 | 42.6 | first mode |


## hold point

|  | CLα vertical [1/rad] | CLα / the biplane's | induced k | e (projected span) | AR projected | x_np [mm] | Cd0 | CdA [cm²] |
|---|---|---|---|---|---|---|---|---|
| X-A (the biplane's wing area) | 1.88 | 0.461 | 0.145 | 0.996 | 2.2 | 233 | 0.0344 | 96.4 |
| X-B (twice the wing area) | 1.88 | 0.462 | 0.144 | 1.09 | 2.04 | 247 | 0.0303 | 170 |


## aerodynamics

|  | CLa | k | e | AR_projected | S | x_np | CLa_flat_estimate |
|---|---|---|---|---|---|---|---|
| X-A (the biplane's wing area) | 1.88 | 0.145 | 0.996 | 2.2 | 0.28 | 0.233 | — |
| X-B (twice the wing area) | 1.88 | 0.144 | 1.09 | 2.04 | 0.56 | 0.247 | — |


## cruise tables

| key | value |
|---|---|
| X-A (the biplane's wing area) |          CL vertical  panel cl  feasible              note    α [deg]       L/D  drag [… |
| X-B (twice the wing area) |          CL vertical  panel cl  feasible note    α [deg]       L/D  drag [N]  upper rot… |


## section (Zephyr)

|  | section cl | section cd | trim elevon [deg] | Δcd of the trim | cd with the trim | section L/D (trimmed) | cl max | margin to stall | cl lost to the elevon |
|---|---|---|---|---|---|---|---|---|---|
| X-A (0012, panels) | 0.501 | 0.0133 | -9.27 | 0.00764 | 0.021 | 23.9 | 1.06 | 2.11 | -0.531 |
| X-B (0012, panels) | 0.297 | 0.0119 | -5.6 | 0.00305 | 0.015 | 19.8 | 1.06 | 3.58 | -0.321 |
| Kestrel-4Q (0012) | 0.38 | 0.0124 | -5.49 | 0.00292 | 0.0154 | 24.7 | 1.06 | 2.79 | -0.314 |
| Kestrel-4Q-2 (NACA 23112 (k2/k1 0.03)) | 0.38 | 0.0157 | -0.356 | 2.79e-04 | 0.016 | 23.7 | 1.15 | 3.02 | -0.0211 |


## capsule

|  | pod section inside the walls [mm] | capsule fits the section | wall left beside the capsule [mm] | length needed in line [mm] | pod length [mm] | fits in line |
|---|---|---|---|---|---|---|
| X-A (the biplane's wing area) | 98 | yes | 13 | 462 | 430 | no |
| X-B (twice the wing area) | 98 | yes | 13 | 462 | 430 | no |


## standing on the tips

|  | CG height above the ground [m] | least tip-over angle [deg] | footprint half-width [m] | wind that tips it [m/s] |
|---|---|---|---|---|
| X-A (the biplane's wing area) | 0.226 | 60.1 | 0.392 | 16.4 |
| X-B (twice the wing area) | 0.28 | 62.3 | 0.534 | 13.6 |


## transitions

| vehicle · direction · t_tr [s] · headwind [m/s] | height loss [m] | height gain [m] | pitch error max [deg] | time stalled, free stream [s] | α blown max while stalled [deg] | moment shortfall [s] | least pitch margin | peak rotor thrust / T_max | elevon max [deg] | roll demand / authority | roll demand / rotors' authority | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| X-A (the biplane's wing area) · forward · 1.5 · 0 | 1.2 | 0.469 | 7.32 | 0.384 | 5.03 | 0 | 0.809 | 1 | 4.84 | 0.119 | 2.11 | yes |
| X-A (the biplane's wing area) · forward · 1.5 · 6 | 0.509 | 0.196 | 5.21 | 1.25 | 17.4 | 0 | 0.887 | 0.977 | 2.86 | 0.122 | 2.02 | yes |
| X-A (the biplane's wing area) · forward · 3 · 0 | 0.141 | 0.142 | 2.43 | 1.18 | 8.83 | 0 | 0.945 | 0.542 | 1.39 | 0.118 | 2.07 | yes |
| X-A (the biplane's wing area) · forward · 3 · 6 | 0.494 | 0.0512 | 1.86 | 2.52 | 17.3 | 0 | 0.933 | 0.522 | 1.7 | 0.122 | 2.41 | yes |
| X-A (the biplane's wing area) · forward · 5 · 0 | 0.573 | 0.13 | 1.25 | 2.47 | 13.9 | 0 | 0.92 | 0.401 | 2.01 | 0.123 | 2.36 | yes |
| X-A (the biplane's wing area) · forward · 5 · 6 | 1.33 | 0.338 | 1.11 | 4.19 | 19 | 0 | 0.865 | 0.451 | 3.4 | 0.121 | 2.35 | yes |
| X-A (the biplane's wing area) · back · 1.5 · 0 | 0.567 | 0.0705 | 5.47 | 0.842 | 58.8 | 0 | 0.804 | 0.495 | 4.75 | 0.639 | 2.1 | yes |
| X-A (the biplane's wing area) · back · 1.5 · 6 | 0.567 | 0.282 | 5.47 | 0.842 | 58.8 | 0 | 0.833 | 0.461 | 4.04 | 0.639 | 2.1 | yes |
| X-A (the biplane's wing area) · back · 3 · 0 | 1.14 | 0.169 | 2.42 | 1.85 | 42.7 | 0 | 0.766 | 0.477 | 5.69 | 0.569 | 2.1 | yes |
| X-A (the biplane's wing area) · back · 3 · 6 | 1.14 | 0.404 | 2.42 | 1.85 | 42.7 | 0 | 0.852 | 0.402 | 3.57 | 0.569 | 2.1 | yes |
| X-A (the biplane's wing area) · back · 5 · 0 | 1.69 | 0.334 | 1 | 4.42 | 33.5 | 0 | 0.759 | 0.439 | 5.88 | 0.278 | 2.1 | yes |
| X-A (the biplane's wing area) · back · 5 · 6 | 1.69 | 0.633 | 1 | 4.42 | 33.5 | 0 | 0.801 | 0.418 | 5.1 | 0.278 | 2.1 | yes |
| X-B (twice the wing area) · forward · 1.5 · 0 | 0.729 | 0.216 | 7.71 | 0.33 | 4.19 | 0 | 0.455 | 1 | 13.7 | 0.255 | 7.62 | yes |
| X-B (twice the wing area) · forward · 1.5 · 6 | 0.534 | 0.193 | 5.29 | 1.18 | 13.2 | 0 | 0.602 | 1 | 9.99 | 0.235 | 7.78 | yes |
| X-B (twice the wing area) · forward · 3 · 0 | 0.25 | 0.0955 | 2.43 | 1.05 | 7.55 | 0 | 0.802 | 0.697 | 4.98 | 0.0899 | 3.21 | yes |
| X-B (twice the wing area) · forward · 3 · 6 | 0.52 | 0.264 | 1.67 | 2.39 | 13.2 | 0 | 0.837 | 0.72 | 4.09 | 0.101 | 3.86 | yes |
| X-B (twice the wing area) · forward · 5 · 0 | 0.647 | 0.271 | 1.15 | 2.25 | 11.9 | 0 | 0.889 | 0.581 | 2.8 | 0.119 | 4.51 | yes |
| X-B (twice the wing area) · forward · 5 · 6 | 1.66 | 0.655 | 0.9 | 3.96 | 15.7 | 0 | 0.894 | 0.742 | 2.66 | 0.124 | 5.14 | yes |
| X-B (twice the wing area) · back · 1.5 · 0 | 0.259 | 1.15 | 5.67 | 0.812 | 71.8 | 0 | 0.723 | 0.61 | 6.83 | 1.05 | 7.84 | no |
| X-B (twice the wing area) · back · 1.5 · 6 | 0.0537 | 1.15 | 5.67 | 0.812 | 71.8 | 0 | 0.838 | 0.446 | 4.06 | 1.05 | 7.84 | no |
| X-B (twice the wing area) · back · 3 · 0 | 0.535 | 2.53 | 2.66 | 1.31 | 64.2 | 0 | 0.631 | 0.577 | 9.11 | 0.86 | 6.91 | yes |
| X-B (twice the wing area) · back · 3 · 6 | 0.0537 | 2.53 | 2.66 | 1.31 | 64.2 | 0 | 0.699 | 0.538 | 7.61 | 0.86 | 6.91 | yes |
| X-B (twice the wing area) · back · 5 · 0 | 0.658 | 3.1 | 1.17 | 1.66 | 54.2 | 0 | 0.757 | 0.503 | 5.98 | 0.737 | 6.34 | yes |
| X-B (twice the wing area) · back · 5 · 6 | 0.359 | 3.1 | 1.17 | 1.66 | 54.2 | 0 | 0.728 | 0.551 | 6.89 | 0.737 | 6.34 | yes |


## verdict

|  | passes | of | failures |
|---|---|---|---|
| X-A (the biplane's wing area) | 12 | 12 | [] |
| X-B (twice the wing area) | 10 | 12 | ['back 1.5 s, headwind 0 m/s: fails roll holds', 'back 1.5 s, headwind 6 m/s: fails rol… |


## back entry speeds

|  | height loss [m] | height gain [m] | pitch error max [deg] | time stalled, free stream [s] | α blown max while stalled [deg] | moment shortfall [s] | least pitch margin | peak rotor thrust / T_max | elevon max [deg] | roll demand / authority | roll demand / rotors' authority | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| X-A, back entered at 16 m/s | 1.14 | 0.169 | 2.42 | 1.85 | 42.7 | 0 | 0.766 | 0.477 | 5.69 | 0.569 | 2.1 | yes |
| X-A, back entered at 13 m/s | 4.34 | 1.36 | 1.88 | 3 | 38.7 | 0 | 0.743 | 0.901 | 6.27 | 0.515 | 1.44 | no |
| X-A, back entered at 11 m/s | 6.08 | 1.9 | 2.11 | 3 | 59.5 | 0 | 0.742 | 0.991 | 6.29 | 0.475 | 1.24 | no |
| X-B, back entered at 16 m/s | 0.535 | 2.53 | 2.66 | 1.31 | 64.2 | 0 | 0.631 | 0.577 | 9.11 | 0.86 | 6.91 | yes |
| X-B, back entered at 13 m/s | 1.2 | 0.148 | 2.54 | 1.78 | 42.7 | 0 | 0.636 | 0.562 | 9.01 | 0.845 | 4.75 | yes |
| X-B, back entered at 11 m/s | 3.89 | 1.13 | 2.09 | 3 | 29.8 | 0 | 0.643 | 0.699 | 8.83 | 0.448 | 3.7 | no |


## against Kestrel-4Q and 4Q-2

|  | mass [kg] | wing area [m²] | projected span [m] | CLα vertical | induced k | Cd0 | CdA [cm²] | α at 16 m/s [deg] | panel cl at 16 m/s | power at 16 m/s [W] | endurance best [min] | at [m/s] | hover pitch [rad/s²] | hover heading [rad/s²] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| X-A (the biplane's wing area) | 1.59 | 0.28 | 0.785 | 1.88 | 0.145 | 0.0344 | 96.4 | 10.8 | 0.501 | 87.6 | 29.1 | 15 | 67 | 1.81 |
| X-B (twice the wing area) | 1.88 | 0.56 | 1.07 | 1.88 | 0.144 | 0.0303 | 170 | 6.38 | 0.297 | 110 | 33.1 | 12 | 50.6 | 0.916 |
| Kestrel-4Q (Part I) | 1.69 | 0.28 | 1 | 4.08 | 0.0599 | 0.0665 | 186 | — | — | 110 | 40.2 | — | 95 | 2.35 |
| Kestrel-4Q-2 | 1.65 | 0.28 | 1 | 4.08 | 0.0599 | 0.0257 | 72.1 | — | — | 59.1 | 57 | — | 88.4 | — |

*5 more columns not shown: spar OD [mm], tip-over [deg], wind that tips it [m/s], transitions passed, bracing*


## drag vs speed, all four

|  | V stall [m/s] | V of least drag [m/s] | least drag [N] | best L/D | drag at 16 m/s [N] | parasite at 16 [N] | induced at 16 [N] | power at 16 m/s [W] | least power [W] | V of least power [m/s] | V max, thrust = drag [m/s] | full thrust at 16 m/s [N] | thrust / drag at 16 m/s | power at V max [W] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Kestrel-4Q-1 (round struts, X, no elevons) | 10.9 | 11 | 2.26 | 7.32 | 3.34 | 2.92 | 0.415 | 111 | 61.7 | 11 | 30.1 | 32.4 | 9.72 | 643 |
| Kestrel-4Q-2 (elevons, faired posts, no X) | 10.8 | 12.5 | 1.28 | 12.7 | 1.43 | 1.03 | 0.398 | 61.8 | 42.3 | 11 | 33.5 | 32.4 | 22.7 | 1.49e+03 |
| X-A | 12.6 | 14 | 2.32 | 6.71 | 2.4 | 1.51 | 0.891 | 90.5 | 72.8 | 13 | 32.5 | 32.4 | 13.5 | 1.49e+03 |
| X-B | 9.68 | 11 | 2.56 | 7.18 | 3.28 | 2.66 | 0.617 | 110 | 63.2 | 10 | 30.5 | 32.4 | 9.88 | 616 |


## manoeuvrability at 16 m/s, all four

|  | CL | thrust per rotor [N] | ΔT available [N] | pitch: rotors [rad/s²] | pitch: elevons [rad/s²] | pitch: total [rad/s²] | yaw: rotors [rad/s²] | roll: rotors' torque [N·m] | roll: elevons [N·m] | roll acceleration [rad/s²] | steady roll rate [deg/s] | load factor available | turn rate [deg/s] | turn radius [m] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Kestrel-4Q-1 (round struts, X, no elevons) | 0.377 | 0.834 | 0.584 | 14.8 | 0 | 14.8 | 8.57 | 0.394 | 0 | 3.71 | 24.2 | 2.15 | 66.9 | 13.7 |
| Kestrel-4Q-2 (elevons, faired posts, no X) | 0.369 | 0.357 | 0.107 | 2.57 | 46.7 | 49.3 | 1.47 | 0.394 | 6.13 | 57.7 | 401 | 2.2 | 68.7 | 13.4 |
| X-A | 0.355 | 0.6 | 0.35 | 6.5 | 9.57 | 16.1 | 3.54 | 0.394 | 6.11 | 50.4 | 352 | 1.62 | 44.6 | 20.6 |
| X-B | 0.21 | 0.82 | 0.57 | 6.65 | 8.58 | 15.2 | 3.48 | 0.394 | 12.6 | 42.6 | 190 | 2.73 | 89.2 | 10.3 |



---

## Kestrel — notebook 40, the CFD and FEA campaign

*40_kestrel_tests — exported 2026-10-16 12:30*

| key | value |
|---|---|
| date | 2026-10-16 |
| quality | screening |


## settings

| key | value |
|---|---|
| CFD | NOT RUN |
| quality | screening |
| jobs | 1 |
| processors | 4 |
| campaigns | 4Q-1, 4Q-2, sections |
| sweep only | no |
| FEA | NOT RUN |


## aircraft

|  | mass [kg] | CG [mm] | Cd0 | transition points |
|---|---|---|---|---|
| 4Q-1 | 1.69 | 215 | 0.0665 | 16 |
| 4Q-2 | 1.65 | 215 | 0.0234 | 16 |


## prop_lab calibration

| key | value |
|---|---|
| cl_scale | 1 |
| cd_scale | 1 |
| cn_factor | 1 |
| on_file | no |
| meta | {'source': '41_prop_lab', 'quality': 'screening', 'date': '2026-10-16', 'propeller': "9… |


## case lists

| key | value |
|---|---|
| 4Q-1 |                                              status    α [deg]    V [m/s]              … |
| 4Q-2 |                                              status    α [deg]    V [m/s]              … |


## section slabs

|  | status | α [deg] | V [m/s] | Re (chord) | AR | template | reads against |
|---|---|---|---|---|---|---|---|
| NACA 0012, α 0°, 16 m/s, AR 8 slab | NOT RUN | 0 | 16 | 1.53e+05 | 8 | rans_ksst_external | the slab's lift slope ≈ AR/(AR+2) = 0.80 of the section's |
| NACA 0012, α 4°, 16 m/s, AR 8 slab | NOT RUN | 4 | 16 | 1.53e+05 | 8 | rans_ksst_external | the slab's lift slope ≈ AR/(AR+2) = 0.80 of the section's |
| NACA 0012, α 8°, 16 m/s, AR 8 slab | NOT RUN | 8 | 16 | 1.53e+05 | 8 | rans_ksst_external | the slab's lift slope ≈ AR/(AR+2) = 0.80 of the section's |
| NACA 0012, α 12°, 16 m/s, AR 8 slab | NOT RUN | 12 | 16 | 1.53e+05 | 8 | rans_ksst_external | the slab's lift slope ≈ AR/(AR+2) = 0.80 of the section's |
| NACA 23112 (k2/k1 0.03), α 0°, 16 m/s, AR 8 slab | NOT RUN | 0 | 16 | 1.53e+05 | 8 | rans_ksst_external | the slab's lift slope ≈ AR/(AR+2) = 0.80 of the section's |
| NACA 23112 (k2/k1 0.03), α 4°, 16 m/s, AR 8 slab | NOT RUN | 4 | 16 | 1.53e+05 | 8 | rans_ksst_external | the slab's lift slope ≈ AR/(AR+2) = 0.80 of the section's |
| NACA 23112 (k2/k1 0.03), α 8°, 16 m/s, AR 8 slab | NOT RUN | 8 | 16 | 1.53e+05 | 8 | rans_ksst_external | the slab's lift slope ≈ AR/(AR+2) = 0.80 of the section's |
| NACA 23112 (k2/k1 0.03), α 12°, 16 m/s, AR 8 slab | NOT RUN | 12 | 16 | 1.53e+05 | 8 | rans_ksst_external | the slab's lift slope ≈ AR/(AR+2) = 0.80 of the section's |


## campaign tables

| key | value |
|---|---|
| 4Q-1 |                                              status    α [deg]    V [m/s]       source … |
| 4Q-2 |                                              status    α [deg]    V [m/s]       source … |


## slab results

|  | status | α [deg] | V [m/s] | Re (chord) | AR | template | reads against |
|---|---|---|---|---|---|---|---|
| NACA 0012, α 0°, 16 m/s, AR 8 slab | NOT RUN | 0 | 16 | 1.53e+05 | 8 | rans_ksst_external | the slab's lift slope ≈ AR/(AR+2) = 0.80 of the section's |
| NACA 0012, α 4°, 16 m/s, AR 8 slab | NOT RUN | 4 | 16 | 1.53e+05 | 8 | rans_ksst_external | the slab's lift slope ≈ AR/(AR+2) = 0.80 of the section's |
| NACA 0012, α 8°, 16 m/s, AR 8 slab | NOT RUN | 8 | 16 | 1.53e+05 | 8 | rans_ksst_external | the slab's lift slope ≈ AR/(AR+2) = 0.80 of the section's |
| NACA 0012, α 12°, 16 m/s, AR 8 slab | NOT RUN | 12 | 16 | 1.53e+05 | 8 | rans_ksst_external | the slab's lift slope ≈ AR/(AR+2) = 0.80 of the section's |
| NACA 23112 (k2/k1 0.03), α 0°, 16 m/s, AR 8 slab | NOT RUN | 0 | 16 | 1.53e+05 | 8 | rans_ksst_external | the slab's lift slope ≈ AR/(AR+2) = 0.80 of the section's |
| NACA 23112 (k2/k1 0.03), α 4°, 16 m/s, AR 8 slab | NOT RUN | 4 | 16 | 1.53e+05 | 8 | rans_ksst_external | the slab's lift slope ≈ AR/(AR+2) = 0.80 of the section's |
| NACA 23112 (k2/k1 0.03), α 8°, 16 m/s, AR 8 slab | NOT RUN | 8 | 16 | 1.53e+05 | 8 | rans_ksst_external | the slab's lift slope ≈ AR/(AR+2) = 0.80 of the section's |
| NACA 23112 (k2/k1 0.03), α 12°, 16 m/s, AR 8 slab | NOT RUN | 12 | 16 | 1.53e+05 | 8 | rans_ksst_external | the slab's lift slope ≈ AR/(AR+2) = 0.80 of the section's |


## FEA (Talos)

|  | part | bracing | kind | status | description | note |
|---|---|---|---|---|---|---|
| frame_x_A | frame | x | static | NOT RUN | 4 g pull-up, the lift elliptic on both wings, the motors' and wings' inertia against it… | NOT RUN: frame_x_A in _runs/kestrel_tests/fea/frame_x_A (run=False) |
| frame_x_B | frame | x | static | NOT RUN | 4 g with the upper wing carrying 75 % (one wing stalling first) (ultimate ×1.5) | NOT RUN: frame_x_B in _runs/kestrel_tests/fea/frame_x_B (run=False) |
| frame_x_C | frame | x | static | NOT RUN | the heading manoeuvre in the hover: M1 and M4 at 12.0 N, M2 and M3 at 0.25 N (the diago… | NOT RUN: frame_x_C in _runs/kestrel_tests/fea/frame_x_C (run=False) |
| frame_x_modes | frame | x | modes | NOT RUN | the first modes with the motor units as point masses at the stations |  |
| frame_posts_A | frame | posts | static | NOT RUN | 4 g pull-up, the lift elliptic on both wings, the motors' and wings' inertia against it… | NOT RUN: frame_posts_A in _runs/kestrel_tests/fea/frame_posts_A (run=False) |
| frame_posts_B | frame | posts | static | NOT RUN | 4 g with the upper wing carrying 75 % (one wing stalling first) (ultimate ×1.5) | NOT RUN: frame_posts_B in _runs/kestrel_tests/fea/frame_posts_B (run=False) |
| frame_posts_C | frame | posts | static | NOT RUN | the heading manoeuvre in the hover: M1 and M4 at 12.0 N, M2 and M3 at 0.25 N (the diago… | NOT RUN: frame_posts_C in _runs/kestrel_tests/fea/frame_posts_C (run=False) |
| frame_posts_modes | frame | posts | modes | NOT RUN | the first modes with the motor units as point masses at the stations |  |
| joint_post | joint |  | static | NOT RUN | the post's 5.5 N (ultimate) into the printed socket, the spar bore held; build axis z | NOT RUN: joint_post in _runs/kestrel_tests/fea/joint_post (run=False) |


## plane-frame reference

|  | bracing | case | spar stress, limit [MPa] | spar stress, ultimate [MPa] | tip deflection, upper, limit [mm] | tip deflection, ultimate [mm] | spar reserve (limit) | first mode, symmetric [Hz] | first mode, antisymmetric [Hz] |
|---|---|---|---|---|---|---|---|---|---|
| frame_x_A | x | A | 11.5 | 17.3 | -0.0909 | -0.136 | 34.8 | — | — |
| frame_x_B | x | B | 12.7 | 19 | -0.146 | -0.219 | 31.6 | — | — |
| frame_x_modes | x | modes | — | — | — | — | — | 261 | 1.64e+03 |
| frame_posts_A | posts | A | 21.6 | 32.4 | 1.98 | 2.97 | 18.5 | — | — |
| frame_posts_B | posts | B | 27.3 | 40.9 | 1.91 | 2.86 | 14.7 | — | — |
| frame_posts_modes | posts | modes | — | — | — | — | — | 15.6 | 1.62e+03 |
| frame_none_A | none | A | 21.6 | 32.4 | 1.98 | 2.97 | 18.5 | — | — |
| frame_none_B | none | B | 52.1 | 78.1 | 6.38 | 9.57 | 7.68 | — | — |
| frame_none_modes | none | modes | — | — | — | — | — | 15.6 | 15.6 |


## verdicts, model against CFD

|  | CFD (sweep) | model |
|---|---|---|
| 4Q-1 | {'passes': 8, 'of': 8, 'failures': []} | {'passes': 8, 'of': 8, 'failures': []} |


## campaigns on file

|  | solved | diverged | FAILED | NOT RUN | n | solved with rotor disks | solved bare | with rotor disks | source | quality | date |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4Q-1 | 10 | 2 | 12 | 0 | 24 | 0 | 10 | 14 | 38_kestrel run on an OpenFOAM machine (uploaded notebook), Part I points, screening qua… | screening | 2026-10-16 |
| 4Q-2 | 0 | 0 | 0 | 29 | 29 | 0 | 0 | 16 | 40_kestrel_tests | screening | 2026-10-16 |
| sections | 0 | 0 | 0 | 8 | — | — | — | — | 40_kestrel_tests | screening | 2026-10-16 |


## cases that did not solve

|  | status | note | rotor disks | α [deg] | V [m/s] |
|---|---|---|---|---|---|
| 4Q-1: forward 3 s: α 53°, 1.5 m/s, T 4.63/4.64 N | diverged | diverged: /coefficients/ beyond 1e90 | yes | 52.9 | 1.51 |
| 4Q-1: forward 3 s: α 45°, 2.1 m/s, T 4.73/4.80 N | FAILED | step solver failed (log not in the notebook) | yes | 45 | 2.1 |
| 4Q-1: forward 3 s: α 30°, 3.8 m/s, T 4.91/5.11 N | FAILED | step solver failed (log not in the notebook) | yes | 29.9 | 3.83 |
| 4Q-1: forward 3 s: α 20°, 5.8 m/s, T 4.85/5.14 N | diverged | diverged: /coefficients/ beyond 1e90 | yes | 19.9 | 5.82 |
| 4Q-1: forward 3 s: α 15°, 7.3 m/s, T 4.54/4.88 N | FAILED | step solver failed (log not in the notebook) | yes | 15 | 7.35 |
| 4Q-1: forward 3 s: α 10°, 9.3 m/s, T 3.29/3.79 N | FAILED | step solver failed (log not in the notebook) | yes | 9.99 | 9.31 |
| 4Q-1: back 3 s: α 80°, 6.1 m/s, T 0.29/1.58 N | FAILED | step solver failed (log not in the notebook) | yes | 80.1 | 6.13 |
| 4Q-1: back 3 s: α 60°, 7.8 m/s, T 0.25/1.98 N | FAILED | step solver failed (log not in the notebook) | yes | 60.1 | 7.78 |
| 4Q-1: back 3 s: α 45°, 10.0 m/s, T 0.25/2.35 N | FAILED | step solver failed (log not in the notebook) | yes | 45 | 10 |
| 4Q-1: back 3 s: α 30°, 13.0 m/s, T 0.25/2.02 N | FAILED | step solver failed (log not in the notebook) | yes | 30 | 13 |
| 4Q-1: back 3 s: α 20°, 14.6 m/s, T 0.25/1.24 N | FAILED | step solver failed (log not in the notebook) | yes | 20 | 14.6 |
| 4Q-1: back 3 s: α 15°, 15.2 m/s, T 0.25/1.16 N | FAILED | step solver failed (log not in the notebook) | yes | 15.1 | 15.2 |
| 4Q-1: back 3 s: α 10°, 15.5 m/s, T 0.25/1.39 N | FAILED | step solver failed (log not in the notebook) | yes | 10 | 15.5 |
| 4Q-1: hover: α 90°, 1.0 m/s, T 4.15/4.15 N | FAILED | step solver failed (log not in the notebook) | yes | 90 | 1 |


## checks

| check | value | target | holds | note |
|---|---|---|---|---|
| campaigns on file | 3 | ≥ 3 | ✔ |  |
| FEA cases solved (Talos) | 0 | ≥ 9 | ✘ |  |
| 4Q-1: blown cases solved | 0 | ≥ 1 | ✘ |  |
| 4Q-1: bare sweep solved | 10 | ≥ 10 | ✔ |  |
| 4Q-2: blown cases solved | 0 | ≥ 1 | ✘ |  |
| 4Q-2: bare sweep solved | 0 | ≥ 10 | ✘ |  |
| sections: slabs solved | 0 | ≥ 8 | ✘ |  |



---

## prop_lab — notebook 41, the 9x6 alone

*41_prop_lab — exported 2026-10-16 12:30*

| key | value |
|---|---|
| date | 2026-10-16 |
| quality | screening |


## settings

| key | value |
|---|---|
| CFD | NOT RUN |
| quality | screening |
| kinds | hover, axial, oblique |


## the 9x6

| key | value |
|---|---|
| diameter | 229 |
| pitch | 152 |
| blades | 2 |
| hub_diameter | 16 |
| hub_height | 10 |
| bore | 5 |
| chord_root | 14 |
| chord_max | 21 |
| chord_tip | 8 |
| thickness | 0.1 |
| camber | 0.04 |
| stations | 12 |


## blade rows (Boreas)

|  | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r [m] | 0.0137 | 0.0229 | 0.032 | 0.0411 | 0.0503 | 0.0594 | 0.0686 | 0.0777 | 0.0869 | 0.096 | 0.105 | 0.114 |
| twist [deg] | 60.5 | 46.7 | 37.2 | 30.5 | 25.7 | 22.2 | 19.5 | 17.3 | 15.6 | 14.2 | 13 | 12 |
| chord [m] | 0.014 | 0.0158 | 0.0176 | 0.0195 | 0.021 | 0.0202 | 0.0189 | 0.0172 | 0.0153 | 0.0131 | 0.0106 | 0.008 |


## rpm

| key | value |
|---|---|
| hover | 6.72e+03 |
| cruise | 6.56e+03 |


## cases: CFD against BEMT

|  | kind | rpm | V [m/s] | angle [deg] | J | BEMT T [N] | BEMT Q [N·m] | BEMT P [W] | BEMT η / FM | De Young CNα [1/rad] | status | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hover, 5000 rpm | hover | 5e+03 | 0 | 0 | 0 | 2.3 | 0.0371 | 19.4 | 0.565 | — | NOT RUN |  |
| hover, 6723 rpm (the hover) | hover | 6.72e+03 | 0 | 0 | 0 | 4.15 | 0.067 | 47.2 | 0.565 | — | NOT RUN |  |
| hover, 9000 rpm | hover | 9e+03 | 0 | 0 | 0 | 7.44 | 0.12 | 113 | 0.565 | — | NOT RUN |  |
| axial, 4 m/s, 8000 rpm | axial | 8e+03 | 4 | 0 | 0.131 | 5.44 | 0.0979 | 82 | 0.265 | — | NOT RUN |  |
| axial, 8 m/s, 8000 rpm | axial | 8e+03 | 8 | 0 | 0.262 | 4.7 | 0.0955 | 80 | 0.47 | — | NOT RUN |  |
| axial, 12 m/s, 8000 rpm | axial | 8e+03 | 12 | 0 | 0.394 | 3.68 | 0.0854 | 71.6 | 0.616 | — | NOT RUN |  |
| axial, 16 m/s, 8000 rpm | axial | 8e+03 | 16 | 0 | 0.525 | 2.41 | 0.0659 | 55.2 | 0.697 | — | NOT RUN |  |
| axial, 20 m/s, 8000 rpm | axial | 8e+03 | 20 | 0 | 0.656 | 1.01 | 0.0372 | 31.1 | 0.649 | — | NOT RUN |  |
| axial, 24 m/s, 8000 rpm | axial | 8e+03 | 24 | 0 | 0.787 | -0.663 | -0.00662 | -5.54 | 0 | — | NOT RUN |  |
| axial, 12 m/s, 10000 rpm | axial | 1e+04 | 12 | 0 | 0.315 | 6.76 | 0.145 | 151 | 0.536 | — | NOT RUN |  |
| axial, 20 m/s, 10000 rpm | axial | 1e+04 | 20 | 0 | 0.525 | 3.76 | 0.103 | 108 | 0.697 | — | NOT RUN |  |
| cruise, 16 m/s, 6562 rpm | axial | 6.56e+03 | 16 | 0 | 0.64 | 0.8 | 0.0278 | 19.1 | 0.671 | — | NOT RUN |  |
| oblique, 8 m/s at 15°, 8000 rpm | oblique | 8e+03 | 8 | 15 | 0.254 | 4.76 | 0.0959 | 80.3 | 0.458 | 0.319 | NOT RUN |  |
| oblique, 8 m/s at 30°, 8000 rpm | oblique | 8e+03 | 8 | 30 | 0.227 | 4.93 | 0.0968 | 81.1 | 0.421 | 0.323 | NOT RUN |  |
| oblique, 8 m/s at 45°, 8000 rpm | oblique | 8e+03 | 8 | 45 | 0.186 | 5.17 | 0.0977 | 81.8 | 0.357 | 0.328 | NOT RUN |  |
| oblique, 8 m/s at 60°, 8000 rpm | oblique | 8e+03 | 8 | 60 | 0.131 | 5.44 | 0.0979 | 82 | 0.265 | 0.333 | NOT RUN |  |
| oblique, 8 m/s at 90°, 8000 rpm | oblique | 8e+03 | 8 | 90 | 1.61e-17 | 5.88 | 0.0949 | 79.5 | 3.62e-17 | 0.342 | NOT RUN |  |
| oblique, 14 m/s at 15°, 8000 rpm | oblique | 8e+03 | 14 | 15 | 0.444 | 3.21 | 0.079 | 66.2 | 0.656 | 0.214 | NOT RUN |  |
| oblique, 14 m/s at 30°, 8000 rpm | oblique | 8e+03 | 14 | 30 | 0.398 | 3.64 | 0.085 | 71.2 | 0.62 | 0.219 | NOT RUN |  |
| oblique, 14 m/s at 45°, 8000 rpm | oblique | 8e+03 | 14 | 45 | 0.325 | 4.25 | 0.0918 | 76.9 | 0.547 | 0.227 | NOT RUN |  |
| oblique, 14 m/s at 60°, 8000 rpm | oblique | 8e+03 | 14 | 60 | 0.23 | 4.91 | 0.0967 | 81 | 0.424 | 0.234 | NOT RUN |  |


## campaign

| key | value |
|---|---|
| solved | 0 |
| diverged | 0 |
| FAILED | 0 |
| NOT RUN | 21 |
| n | 21 |
| by kind | {'hover': 0, 'axial': 0, 'oblique': 0} |


## section fit

| key | value |
|---|---|
| cl_scale | — |
| cd_scale | — |
| n_cases | 0 |
| rms_log_ratio_before | — |
| rms_log_ratio_after | — |


## normal-force fit

| key | value |
|---|---|
| cn_factor | — |
| n_cases | 0 |


## calibration on file

| key | value |
|---|---|
| cl_scale | 1 |
| cd_scale | 1 |
| cn_factor | 1 |
| on_file | no |
| meta | {'source': '41_prop_lab', 'quality': 'screening', 'date': '2026-10-16', 'propeller': "9… |


## noise at the listener

|  | source | rpm | V [m/s] | angle [deg] | BPF [Hz] | tone 1 [dB] | tone 2 [dB] | tone 3 [dB] | tonal [dB] | broadband [dB] | broadband peak [Hz] | total [dB] | total [dB(A)] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hover, 5000 rpm | BEMT | 5e+03 | 0 | 0 | 167 | 26.9 | -1.16 | -30.6 | 26.9 | 28.8 | 4.14e+03 | 31 | 29.8 |
| hover, 6723 rpm (the hover) | BEMT | 6.72e+03 | 0 | 0 | 224 | 34.9 | 11.9 | -12.4 | 35 | 36.5 | 5.57e+03 | 38.8 | 37.1 |
| hover, 9000 rpm | BEMT | 9e+03 | 0 | 0 | 300 | 40.7 | 22.7 | 3.28 | 40.8 | 44.1 | 7.45e+03 | 45.8 | 43.8 |
| axial, 4 m/s, 8000 rpm | BEMT | 8e+03 | 4 | 0 | 267 | 40.5 | 20.5 | -0.923 | 40.5 | 41.1 | 6.64e+03 | 43.8 | 41.4 |
| axial, 8 m/s, 8000 rpm | BEMT | 8e+03 | 8 | 0 | 267 | 41.5 | 21.5 | 0.0971 | 41.6 | 41.2 | 6.67e+03 | 44.4 | 41.6 |
| axial, 12 m/s, 8000 rpm | BEMT | 8e+03 | 12 | 0 | 267 | 41.6 | 21.6 | 0.187 | 41.7 | 41.5 | 6.73e+03 | 44.6 | 41.8 |
| axial, 16 m/s, 8000 rpm | BEMT | 8e+03 | 16 | 0 | 267 | 40.3 | 20.3 | -1.08 | 40.4 | 41.8 | 6.81e+03 | 44.1 | 41.9 |
| axial, 20 m/s, 8000 rpm | BEMT | 8e+03 | 20 | 0 | 267 | 36.6 | 16.6 | -4.81 | 36.7 | 42.2 | 6.91e+03 | 43.2 | 41.9 |
| axial, 24 m/s, 8000 rpm | BEMT | 8e+03 | 24 | 0 | 267 | -506 | -506 | -506 | -506 | 42.6 | 7.04e+03 | 42.6 | 42.1 |
| axial, 12 m/s, 10000 rpm | BEMT | 1e+04 | 12 | 0 | 333 | 47.1 | 30.8 | 13.2 | 47.2 | 47.1 | 8.37e+03 | 50.2 | 47 |
| axial, 20 m/s, 10000 rpm | BEMT | 1e+04 | 20 | 0 | 333 | 46.6 | 30.4 | 12.8 | 46.7 | 47.6 | 8.51e+03 | 50.2 | 47.2 |
| cruise, 16 m/s, 6562 rpm | BEMT | 6.56e+03 | 16 | 0 | 219 | 31.1 | 7.69 | -17.1 | 31.1 | 36.9 | 5.66e+03 | 38 | 37.3 |
| oblique, 8 m/s at 15°, 8000 rpm | BEMT | 8e+03 | 8 | 15 | 267 | 41.5 | 21.5 | 0.0543 | 41.5 | 41.2 | 6.67e+03 | 44.4 | 41.6 |
| oblique, 8 m/s at 30°, 8000 rpm | BEMT | 8e+03 | 8 | 30 | 267 | 41.3 | 21.3 | -0.0947 | 41.4 | 41.2 | 6.67e+03 | 44.3 | 41.6 |
| oblique, 8 m/s at 45°, 8000 rpm | BEMT | 8e+03 | 8 | 45 | 267 | 41 | 21 | -0.403 | 41.1 | 41.2 | 6.67e+03 | 44.2 | 41.6 |
| oblique, 8 m/s at 60°, 8000 rpm | BEMT | 8e+03 | 8 | 60 | 267 | 40.5 | 20.5 | -0.923 | 40.5 | 41.2 | 6.67e+03 | 43.9 | 41.5 |
| oblique, 8 m/s at 90°, 8000 rpm | BEMT | 8e+03 | 8 | 90 | 267 | 38.8 | 18.8 | -2.63 | 38.8 | 41.2 | 6.67e+03 | 43.2 | 41.3 |
| oblique, 14 m/s at 15°, 8000 rpm | BEMT | 8e+03 | 14 | 15 | 267 | 41.3 | 21.3 | -0.116 | 41.3 | 41.6 | 6.77e+03 | 44.5 | 41.9 |
| oblique, 14 m/s at 30°, 8000 rpm | BEMT | 8e+03 | 14 | 30 | 267 | 41.6 | 21.6 | 0.17 | 41.6 | 41.6 | 6.77e+03 | 44.6 | 41.9 |
| oblique, 14 m/s at 45°, 8000 rpm | BEMT | 8e+03 | 14 | 45 | 267 | 41.7 | 21.7 | 0.276 | 41.7 | 41.6 | 6.77e+03 | 44.7 | 41.9 |
| oblique, 14 m/s at 60°, 8000 rpm | BEMT | 8e+03 | 14 | 60 | 267 | 41.3 | 21.3 | -0.08 | 41.4 | 41.6 | 6.77e+03 | 44.5 | 41.9 |


## sound files

|  | path | seconds | rms_db | full_scale_pa | total_db | total_dba | rpms | beat_hz |
|---|---|---|---|---|---|---|---|---|
| one_rotor_hover | output/41_prop_lab/sound/one_rotor_hover.wav | 3 | 38.7 | 0.0173 | 38.8 | 37.1 | — | — |
| one_rotor_cruise_16ms | output/41_prop_lab/sound/one_rotor_cruise_16ms.wav | 3 | 37.9 | 0.0173 | 38 | 37.3 | — | — |
| one_rotor_transition_8ms_45deg | output/41_prop_lab/sound/one_rotor_transition_8ms_45deg.wav | 3 | 44.1 | 0.0173 | — | — | — | — |
| four_rotors_hover | output/41_prop_lab/sound/four_rotors_hover.wav | 3 | 44.2 | 0.0173 | 44.8 | — | [6759.653306224799, 6659.007687140006, 6659.007687140006, 6759.653306224799] | [3.3548539694931208] |
| transition_hover_to_cruise | output/41_prop_lab/sound/transition_hover_to_cruise.wav | 6 | 43.3 | 0.0173 | — | — | — | — |


## checks

| check | value | target | holds | note |
|---|---|---|---|---|
| hover cases solved | 0 | ≥ 2 | ✘ |  |
| axial cases solved | 0 | ≥ 5 | ✘ |  |
| oblique cases solved | 0 | ≥ 4 | ✘ |  |
| section fit: rms log ratio after | nan | ≤ 0.1 | ✘ | thrust and torque within ~10 % together |
| cl scale within | 0 | ≤ 0.3 | ✔ | further than this and the generic section is the wrong model, not a scaled one |
| cn factor within | 0 | ≤ 0.5 | ✔ |  |

