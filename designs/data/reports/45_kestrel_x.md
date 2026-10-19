# Kestrel-X — notebook 45 (X-A and X-B against Kestrel-4Q, notebook 43)

*45_kestrel_x — exported 2026-10-19 21:05*

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

|  | X-A (the biplane's wing area) | X-B (twice the wing area) |
|---|---|---|
| propulsion | 0.736 | 0.736 |
| structure | 0.478 | 0.769 |
| systems | 0.208 | 0.208 |
| payload | 0.105 | 0.105 |
| margin | 0.06 | 0.06 |


## spars

|  | n | panel lift [N] | root moment [N·m] | spar OD [mm] | spar stress [MPa] | spar reserve | tip deflection [mm] | tip deflection / length | first bending, with the tip motor [Hz] | spar mass per panel [g] | sized by |
|---|---|---|---|---|---|---|---|---|---|---|---|
| X-A (the biplane's wing area) | 4 | 22 | 2.31 | 9 | 50.9 | 7.85 | 10.3 | 0.0206 | 12.3 | 19.5 | first mode |
| X-B (twice the wing area) | 4 | 26 | 3.49 | 13.5 | 30.5 | 13.1 | 9.23 | 0.0132 | 12.3 | 42.6 | first mode |


## hold point

|  | CLα vertical [1/rad] | CLα / the biplane's | induced k | e (projected span) | AR projected | x_np [mm] | Cd0 | CdA [cm²] |
|---|---|---|---|---|---|---|---|---|
| X-A (the biplane's wing area) | 2.07 | 0.477 | 0.119 | 1.22 | 2.2 | 235 | 0.0344 | 96.4 |
| X-B (twice the wing area) | 2.08 | 0.48 | 0.117 | 1.34 | 2.04 | 250 | 0.0303 | 170 |


## aerodynamics

|  | CLa | k | e | AR_projected | S | x_np | CLa_flat_estimate |
|---|---|---|---|---|---|---|---|
| X-A (the biplane's wing area) | 2.07 | 0.119 | 1.22 | 2.2 | 0.28 | 0.235 | — |
| X-B (twice the wing area) | 2.08 | 0.117 | 1.34 | 2.04 | 0.56 | 0.25 | — |


## cruise tables

| key | value |
|---|---|
| X-A (the biplane's wing area) |          CL vertical  panel cl  feasible              note   α [deg]       L/D  drag [N… |
| X-B (twice the wing area) |          CL vertical  panel cl  feasible note    α [deg]       L/D  drag [N]  upper rot… |


## section (Zephyr)

|  | section cl | section cd | trim elevon [deg] | Δcd of the trim | cd with the trim | section L/D (trimmed) | cl max | margin to stall | cl lost to the elevon |
|---|---|---|---|---|---|---|---|---|---|
| X-A (0012, panels) | 0.501 | 0.0133 | -10.2 | 0.0089 | 0.0222 | 22.6 | 1.06 | 2.11 | -0.583 |
| X-B (0012, panels) | 0.297 | 0.0119 | -6.14 | 0.00372 | 0.0156 | 19 | 1.06 | 3.58 | -0.351 |
| Kestrel-4Q (0012) | 0.38 | 0.0124 | -5.49 | 0.00291 | 0.0154 | 24.7 | 1.06 | 2.79 | -0.314 |
| Kestrel-4Q-2 (NACA 23112 (k2/k1 0.03)) | 0.38 | 0.0157 | -0.355 | 2.79e-04 | 0.016 | 23.7 | 1.15 | 3.02 | -0.0211 |


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
| X-A (the biplane's wing area) · forward · 1.5 · 0 | 4.17 | -5.80e-05 | 6.89 | 0.4 | 4.84 | 0 | 0.841 | 1 | 4.03 | 0.2 | 1.63 | no |
| X-A (the biplane's wing area) · forward · 1.5 · 6 | 0.699 | -4.14e-05 | 5.64 | 1.16 | 18.9 | 0 | 0.866 | 1 | 3.39 | 0.109 | 1.93 | yes |
| X-A (the biplane's wing area) · forward · 3 · 0 | 0.223 | 0.129 | 2.62 | 1.22 | 8.56 | 0 | 0.928 | 0.533 | 1.82 | 0.109 | 2.11 | yes |
| X-A (the biplane's wing area) · forward · 3 · 6 | 0.576 | -4.14e-05 | 2.39 | 2.5 | 19 | 0 | 0.951 | 0.588 | 1.23 | 0.111 | 2.58 | yes |
| X-A (the biplane's wing area) · forward · 5 · 0 | 1.15 | 0.123 | 1.53 | 2.58 | 13.9 | 0 | 0.919 | 0.513 | 2.03 | 0.11 | 2.55 | yes |
| X-A (the biplane's wing area) · forward · 5 · 6 | 1.64 | -4.14e-05 | 1.44 | 4.12 | 19 | 0 | 0.871 | 0.576 | 3.23 | 0.082 | 2.59 | yes |
| X-A (the biplane's wing area) · back · 1.5 · 0 | 0.629 | 3.98e-05 | 5.3 | 0.88 | 61.7 | 0 | 0.843 | 0.481 | 3.76 | 0.684 | 2.1 | yes |
| X-A (the biplane's wing area) · back · 1.5 · 6 | 0.629 | 0.152 | 5.3 | 0.88 | 61.7 | 0 | 0.828 | 0.464 | 4.16 | 0.684 | 2.1 | yes |
| X-A (the biplane's wing area) · back · 3 · 0 | 1.32 | 0.11 | 2.43 | 1.92 | 42.2 | 0 | 0.813 | 0.458 | 4.53 | 0.615 | 2.1 | yes |
| X-A (the biplane's wing area) · back · 3 · 6 | 1.32 | 0.368 | 2.43 | 1.92 | 42.2 | 0 | 0.86 | 0.407 | 3.37 | 0.615 | 2.1 | yes |
| X-A (the biplane's wing area) · back · 5 · 0 | 1.85 | 0.369 | 1.08 | 4.52 | 33.8 | 0 | 0.798 | 0.426 | 4.93 | 0.265 | 2.1 | yes |
| X-A (the biplane's wing area) · back · 5 · 6 | 1.85 | 0.655 | 1.08 | 4.52 | 33.8 | 0 | 0.839 | 0.403 | 4.15 | 0.265 | 2.1 | yes |
| X-B (twice the wing area) · forward · 1.5 · 0 | 0.684 | -7.59e-05 | 7.26 | 0.34 | 4.03 | 0 | 0.511 | 1 | 12.3 | 0.204 | 6.4 | yes |
| X-B (twice the wing area) · forward · 1.5 · 6 | 0.659 | -5.63e-05 | 5.76 | 1.22 | 16.6 | 0 | 0.582 | 1 | 10.5 | 0.203 | 7.01 | yes |
| X-B (twice the wing area) · forward · 3 · 0 | 0.319 | 0.0881 | 2.6 | 1.12 | 7.37 | 0 | 0.777 | 0.709 | 5.6 | 0.184 | 7.21 | yes |
| X-B (twice the wing area) · forward · 3 · 6 | 0.758 | -5.63e-05 | 2.19 | 2.5 | 16.6 | 0 | 0.791 | 0.83 | 5.24 | 0.0897 | 4.17 | yes |
| X-B (twice the wing area) · forward · 5 · 0 | 1.44 | 0.139 | 1.49 | 2.44 | 12.3 | 0 | 0.816 | 0.825 | 4.61 | 0.111 | 5.56 | yes |
| X-B (twice the wing area) · forward · 5 · 6 | 2.14 | -5.63e-05 | 1.7 | 4.26 | 16.6 | 0 | 0.825 | 1 | 4.39 | 0.127 | 6.03 | yes |
| X-B (twice the wing area) · back · 1.5 · 0 | 0.27 | 1.21 | 5.38 | 0.84 | 70.4 | 0 | 0.778 | 0.567 | 5.44 | 0.99 | 7.97 | yes |
| X-B (twice the wing area) · back · 1.5 · 6 | 0.0601 | 1.21 | 5.38 | 0.84 | 70.4 | 0 | 0.847 | 0.43 | 3.83 | 0.99 | 7.97 | yes |
| X-B (twice the wing area) · back · 3 · 0 | 0.575 | 2.85 | 2.69 | 1.36 | 67.1 | 0 | 0.689 | 0.542 | 7.69 | 0.886 | 7.2 | yes |
| X-B (twice the wing area) · back · 3 · 6 | 0.108 | 2.85 | 2.69 | 1.36 | 67.1 | 0 | 0.752 | 0.488 | 6.28 | 0.886 | 7.2 | yes |
| X-B (twice the wing area) · back · 5 · 0 | 0.883 | 3.72 | 1.2 | 1.7 | 57.3 | 0 | 0.778 | 0.502 | 5.46 | 0.724 | 6.69 | yes |
| X-B (twice the wing area) · back · 5 · 6 | 0.388 | 3.72 | 1.2 | 1.7 | 57.3 | 0 | 0.778 | 0.534 | 5.66 | 0.724 | 6.69 | yes |


## verdict

|  | passes | of | failures |
|---|---|---|---|
| X-A (the biplane's wing area) | 11 | 12 | ['forward 1.5 s, headwind 0 m/s: fails holds height'] |
| X-B (twice the wing area) | 12 | 12 | [] |


## back entry speeds

|  | height loss [m] | height gain [m] | pitch error max [deg] | time stalled, free stream [s] | α blown max while stalled [deg] | moment shortfall [s] | least pitch margin | peak rotor thrust / T_max | elevon max [deg] | roll demand / authority | roll demand / rotors' authority | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| X-A, back entered at 16 m/s | 1.32 | 0.11 | 2.43 | 1.92 | 42.2 | 0 | 0.813 | 0.458 | 4.53 | 0.615 | 2.1 | yes |
| X-A, back entered at 13 m/s | 2.89 | 0.565 | 2.32 | 3 | 37.8 | 0 | 0.809 | 0.451 | 4.64 | 0.277 | 1.55 | yes |
| X-A, back entered at 11 m/s | 2.93 | 0.611 | 2.22 | 3 | 37.2 | 0 | 0.804 | 0.694 | 4.75 | 0.273 | 1.52 | yes |
| X-B, back entered at 16 m/s | 0.575 | 2.85 | 2.69 | 1.36 | 67.1 | 0 | 0.689 | 0.542 | 7.69 | 0.886 | 7.2 | yes |
| X-B, back entered at 13 m/s | 1.18 | 0.0933 | 2.59 | 1.72 | 45.2 | 0 | 0.689 | 0.551 | 7.68 | 0.901 | 4.89 | yes |
| X-B, back entered at 11 m/s | 2.77 | 0.496 | 2.52 | 3 | 29.9 | 0 | 0.681 | 0.559 | 7.89 | 0.292 | 3.45 | yes |


## against Kestrel-4Q (notebook 43)

|  | mass [kg] | wing area [m²] | projected span [m] | CLα vertical | induced k | Cd0 | CdA [cm²] | α at 16 m/s [deg] | panel cl at 16 m/s | power at 16 m/s [W] | endurance best [min] | at [m/s] | hover pitch [rad/s²] | hover heading [rad/s²] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| X-A (the biplane's wing area) | 1.59 | 0.28 | 0.785 | 2.07 | 0.119 | 0.0344 | 96.4 | 9.81 | 0.501 | 82.4 | 31.2 | 15 | 67 | 1.81 |
| X-B (twice the wing area) | 1.88 | 0.56 | 1.07 | 2.08 | 0.117 | 0.0303 | 170 | 5.77 | 0.297 | 108 | 35.5 | 12 | 50.6 | 0.916 |
| Kestrel-4Q-1 | 1.69 | 0.28 | 1 | 4.34 | 0.0527 | 0.0665 | 186 | — | — | 110 | 40.2 | — | — | — |
| Kestrel-4Q-2 | 1.65 | 0.28 | 1 | 4.34 | 0.0527 | 0.0257 | 72.1 | — | — | 59.1 | 57 | — | 88.4 | 2.09 |
| Kestrel-4Q-3 | 1.65 | 0.28 | 1 | 4.34 | 0.0527 | 0.0257 | 72.1 | — | — | 59.3 | 56.8 | — | — | — |

*5 more columns not shown: spar OD [mm], tip-over [deg], wind that tips it [m/s], transitions passed, bracing*


## drag vs speed, all four

|  | V stall [m/s] | V of least drag [m/s] | least drag [N] | best L/D | drag at 16 m/s [N] | parasite at 16 [N] | induced at 16 [N] | power at 16 m/s [W] | least power [W] | V of least power [m/s] | V max, thrust = drag [m/s] | full thrust at 16 m/s [N] | thrust / drag at 16 m/s | power at V max [W] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Kestrel-4Q-1 (round struts, X, no elevons) | 10.9 | 11 | 2.15 | 7.68 | 3.29 | 2.92 | 0.365 | 110 | 59.8 | 11 | 30.1 | 32.4 | 9.87 | 642 |
| Kestrel-4Q-2 (elevons, faired posts, no X) | 10.8 | 12 | 1.2 | 13.5 | 1.38 | 1.03 | 0.35 | 60.2 | 40.1 | 11 | 33.5 | 32.4 | 23.5 | 1.49e+03 |
| X-A | 12.6 | 13.5 | 2.1 | 7.42 | 2.24 | 1.51 | 0.727 | 86.8 | 68 | 13 | 32.5 | 32.4 | 14.5 | 1.49e+03 |
| X-B | 9.68 | 10.5 | 2.31 | 7.98 | 3.16 | 2.66 | 0.5 | 108 | 58.3 | 10 | 30.5 | 32.4 | 10.3 | 615 |


## manoeuvrability at 16 m/s, all four

|  | CL | thrust per rotor [N] | ΔT available [N] | pitch: rotors [rad/s²] | pitch: elevons [rad/s²] | pitch: total [rad/s²] | yaw: rotors [rad/s²] | roll: rotors' torque [N·m] | roll: elevons [N·m] | roll acceleration [rad/s²] | steady roll rate [deg/s] | load factor available | turn rate [deg/s] | turn radius [m] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Kestrel-4Q-1 (round struts, X, no elevons) | 0.377 | 0.822 | 0.572 | 14.5 | 0 | 14.5 | 8.39 | 0.394 | 0 | 3.71 | 22.8 | 2.15 | 66.9 | 13.7 |
| Kestrel-4Q-2 (elevons, faired posts, no X) | 0.369 | 0.345 | 0.0945 | 2.28 | 49.7 | 52 | 1.3 | 0.394 | 6.52 | 61.2 | 399 | 2.2 | 68.7 | 13.4 |
| X-A | 0.355 | 0.56 | 0.31 | 5.74 | 10.6 | 16.3 | 3.12 | 0.394 | 6.73 | 55.3 | 350 | 1.62 | 44.6 | 20.6 |
| X-B | 0.21 | 0.791 | 0.541 | 6.31 | 9.49 | 15.8 | 3.3 | 0.394 | 14 | 47 | 189 | 2.73 | 89.2 | 10.3 |

