# X-wing — notebook 41 (X-A and X-B against Kestrel-4Q and 4Q-2)

*41_xwing — exported 2026-10-16 12:34*

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

