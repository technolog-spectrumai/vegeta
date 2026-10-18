# Mini raptors — Apus Mini, Alcotan, Gavilan: three first-guess designs on a 5-inch racing stack

*49_mini_raptor — exported 2026-10-18 03:38*

| key | value |
|---|---|
| method | first principles and empirical correlations; no CFD, no FEA; conceptual CAD |
| cad | {'Apus Mini': {'aircraft.step': '/home/user/vegeta/notebooks/designs/data/cad/mini_rapt… |


## drives

|  | 4S 5x4.3 | 4S 5x4.3 light | 6S 5x4.3 | 6S 5x5 |
|---|---|---|---|---|
| pack | 4S 1500 | 4S 1300 | 6S 1300 | 6S 1300 |
| pack mass [kg] | 0.17 | 0.15 | 0.21 | 0.21 |
| usable energy [Wh] | 17.8 | 15.4 | 23.1 | 23.1 |
| propeller | 5x4.3 three-blade | 5x4.3 three-blade | 5x4.3 three-blade | 5x5 three-blade |
| KV | 2.4e+03 | 2.4e+03 | 1.75e+03 | 1.75e+03 |
| static thrust per motor [N] | 12.9 | 12.9 | 17.7 | 16.2 |
| thrust at 20 m/s [N] | 10.4 | 10.4 | 14.7 | 13.4 |
| thrust at 40 m/s [N] | 7.7 | 7.7 | 12.3 | 11.2 |
| thrust at 60 m/s [N] | 2.48 | 2.48 | 5.85 | 9.03 |
| power at 1.2 N static [W] | 17.4 | 17.4 | 18.2 | 19.4 |
| power at 0.5 N, 20 m/s [W] | 20.2 | 20.2 | 21.7 | 20.1 |
| electrical ceiling per motor [W] | 592 | 592 | 888 | 888 |


## racing quad

|  | racing quad |
|---|---|
| layout | quad |
| drive | 4S 5x4.3 |
| mass [kg] | 0.5 |
| hover thrust / weight | 10.5 |
| vertical climb, thrust-limited [m/s] | 36.5 |
| power in that climb [W] | 2.27e+03 |
| vertical climb at the sustainable power [m/s] | 22 |
| initial vertical acceleration from the hover [g] | 9.53 |
| end-on drag area [m²] | 0.0352 |
| zoom climb from the top speed [m] | 69.8 |
| cross-section climbing [m] | 0.28 × 0.28 |
| smallest square hatch [m] | 0.382 |
| smallest slot [m] | 0.382 |
| smallest round chimney [m] | 0.499 |
| hover power [W] | 77.6 |
| hover endurance [min] | 13.7 |
| pitch acceleration in the hover [rad/s²] | 1.76e+03 |
| roll acceleration in the hover [rad/s²] | 1.76e+03 |
| yaw acceleration in the hover [rad/s²] | 16.5 |
| power at 20 m/s [W] | 169 |
| Wh/km at 20 m/s | 2.34 |
| range at 20 m/s [km] | 7.57 |
| best range [km] | 8.83 |
| at [m/s] | 15 |
| least Wh/km | 2.01 |
| best endurance [min] | 13.7 |
| top speed [m/s] | 37 |
| top speed [km/h] | 133 |
| power at top speed [W] | 827 |
| full-throttle endurance at top speed [s] | 77.3 |
| sustained power [W] | 666 |
| sustained top speed [m/s] | 34 |
| sustained top speed [km/h] | 122 |
| n at 20 m/s (banked) | 2.92 |
| turn radius at 20 m/s (banked) [m] | 14.8 |
| turn radius at 20 m/s, 45° bank [m] | 40.8 |
| turn radius at 40 m/s (banked) [m] | 59.4 |
| least banked turn radius [m] | 0 |
| span / gate | 0.167 |
| energy [Wh] | 17.8 |
| stall speed [m/s] | 0 |
| brake deceleration at 20 m/s [g] | 3.55 |
| brake 20→4 m/s distance [m] | 6.18 |
| brake 20→4 m/s time [s] | 0.535 |
| hover pitch-over 90° [s] | 0.0844 |
| hover yaw 90° [s] | 0.874 |
| transitions | none needed |


## designs

|  | Apus Mini | Alcotan | Gavilan |
|---|---|---|---|
| layout | monoplane | cruciform | biplane |
| title | economy: one wing a third of a gate wide, two of the stack's motors, the hover pitch by… | the chaser: sustained speed on 6S and the 5x5, the motors at the tips of a + | tight spaces: the racing frame's motor square in a short cell, the wing as a brake |
| drive | 4S 5x4.3 | 6S 5x5 | 4S 5x4.3 light |
| pack | 4S 1500 | 6S 1300 | 4S 1300 |
| span | 0.5 | 0.42 | 0.31 |
| chord | 0.17 | 0.075 | 0.1 |
| S_ref | 0.085 | 0.0315 | 0.062 |
| AR | 2.94 | 5.6 | 1.55 |
| motor_y | 0.14 | 0.14 | 0.0765 |
| motor_z | 0 | 0.14 | 0.0775 |
| disc_gap_span | 0.153 | 0.071 | 0.026 |
| disc_gap_vertical | ∞ | 0.071 | 0.028 |
| disc_to_pod | 0.0515 | 0.0515 | ∞ |
| span_over_gate | 0.333 | 0.28 | 0.207 |
| Re_20 | 2.33e+05 | 1.03e+05 | 1.37e+05 |
| cl_max | 1 | 0.95 | 1.1 |
| n_struct | 8 | 10 | 8 |
| section | thin cambered section (3% camber, 10 % thick); the elevons' reflex trims it | thin cambered section (2% camber, 10 % thick); the elevons' reflex trims it | thin cambered section (4% camber, 10 % thick); the elevons' reflex trims it |
| rules | no vertical thrust lever: the hover pitch comes from the elevons in the slipstream only… | all hold | all hold |


## comparison

|  | 5-inch racing quad | Apus Mini | Alcotan | Gavilan | Apus Mini (as printed) | Alcotan (as printed) | Gavilan (as printed) |
|---|---|---|---|---|---|---|---|
| layout | quad | monoplane (2 motors) | cruciform | biplane | monoplane (2 motors) | cruciform | biplane |
| mass [kg] | 0.5 | 0.46 | 0.572 | 0.524 | 0.532 | 0.664 | 0.59 |
| span / gate | 0.167 | 0.333 | 0.28 | 0.207 | 0.333 | 0.28 | 0.207 |
| top speed [km/h] | 133 | 223 | 295 | 230 | 223 | 295 | 230 |
| power at top speed [W] | 827 | 558 | 1.6e+03 | 921 | 558 | 1.6e+03 | 922 |
| full-throttle endurance at top speed [s] | 77.3 | 115 | 51.9 | 60.2 | 115 | 51.9 | 60.1 |
| sustained power [W] | 666 | 651 | 866 | 577 | 651 | 866 | 577 |
| sustained top speed [km/h] | 122 | 223 | 263 | 205 | 223 | 263 | 205 |
| power at 20 m/s [W] | 169 | 26 | 34.5 | 38.7 | 27.6 | 38.1 | 41.3 |
| Wh/km at 20 m/s | 2.34 | 0.361 | 0.479 | 0.538 | 0.383 | 0.529 | 0.574 |
| least Wh/km | 2.01 | 0.348 | 0.477 | 0.521 | 0.376 | 0.529 | 0.566 |
| range at 20 m/s [km] | 7.57 | 49.2 | 48.2 | 28.6 | 46.4 | 43.7 | 26.8 |
| best range [km] | 8.83 | 51 | 48.4 | 29.5 | 47.3 | 43.7 | 27.2 |
| at [m/s] | 15 | 17 | 19 | 18 | 18 | 20 | 18 |
| best endurance [min] | 13.7 | 58.4 | 44.5 | 32.2 | 50.9 | 38 | 28.3 |
| stall speed [m/s] | 0 | 9.31 | 17.5 | 11.1 | 10 | 18.9 | 11.8 |
| hover thrust / weight | 10.5 | 5.72 | 11.5 | 10 | 4.94 | 9.92 | 8.92 |
| hover power [W] | 77.6 | 87.5 | 100 | 82 | 106 | 120 | 94.4 |
| hover endurance [min] | 13.7 | 12.2 | 13.8 | 11.3 | 10.1 | 11.5 | 9.79 |
| n at 20 m/s (banked) | 2.92 | 4.61 | 1.31 | 3.25 | 3.99 | 1.13 | 2.89 |
| turn radius at 20 m/s (banked) [m] | 14.8 | 9.06 | 48.6 | 13.2 | 10.6 | 78.9 | 15 |
| turn radius at 40 m/s (banked) [m] | 59.4 | 20.5 | 31.8 | 20.5 | 20.5 | 37.1 | 20.5 |
| least banked turn radius [m] | 0 | 8.92 | 31.4 | 12.6 | 10.3 | 36.4 | 14.2 |
| brake deceleration at 20 m/s [g] | 3.55 | 7.23 | 4.79 | 4.75 | 6.27 | 4.51 | 4.26 |
| brake 20→4 m/s distance [m] | 6.18 | 4.64 | 5.2 | 5.41 | 4.94 | 5.4 | 5.56 |
| brake 20→4 m/s time [s] | 0.535 | 0.43 | 0.465 | 0.48 | 0.45 | 0.48 | 0.49 |
| vertical climb, thrust-limited [m/s] | 36.5 | 55.5 | 59.5 | 59.5 | 54 | 59.5 | 59 |
| vertical climb at the sustainable power [m/s] | 22 | 53.5 | 56.5 | 42.5 | 51 | 54 | 41 |
| power in that climb [W] | 2.27e+03 | 721 | 947 | 1.11e+03 | 728 | 1.02e+03 | 1.18e+03 |
| initial vertical acceleration from the hover [g] | 9.53 | 4.72 | 10.5 | 9.05 | 3.94 | 8.92 | 7.92 |
| zoom climb from the top speed [m] | 69.8 | 196 | 343 | 209 | 196 | 343 | 209 |
| cross-section climbing [m] | 0.28 × 0.28 | 0.50 × 0.13 | 0.42 × 0.41 | 0.31 × 0.28 | 0.50 × 0.13 | 0.42 × 0.41 | 0.31 × 0.28 |
| smallest square hatch [m] | 0.382 | 0.6 | 0.52 | 0.41 | 0.6 | 0.52 | 0.41 |
| smallest slot [m] | 0.382 | 0.227 | 0.507 | 0.382 | 0.227 | 0.507 | 0.382 |
| smallest round chimney [m] | 0.499 | 0.616 | 0.685 | 0.519 | 0.616 | 0.685 | 0.519 |
| hover pitch-over 90° [s] | 0.0844 | 0.226 | 0.0909 | 0.124 | 0.211 | 0.0916 | 0.125 |
| pitch acceleration in the hover [rad/s²] | 1.76e+03 | 245 | 1.52e+03 | 813 | 284 | 1.5e+03 | 802 |
| hover pitch by | — | elevons in the wash | differential thrust (+ elevons) | differential thrust (+ elevons) | elevons in the wash | differential thrust (+ elevons) | differential thrust (+ elevons) |
| thrust / weight at 1.2 Vs | — | 5.08 | 9.43 | 8.71 | 4.35 | 8 | 7.67 |
| pull-up speed [m/s] | — | 17 | 25 | 18 | 18 | 27 | 18 |
| pull-up radius into the hover [m] | — | 12.6 | 61.2 | 20.2 | 14.8 | 70.7 | 24.7 |
| transitions | none needed | both with thrust to spare; the hover pitch by elevons only | both with thrust to spare | both with thrust to spare | both with thrust to spare; the hover pitch by elevons only | both with thrust to spare | both with thrust to spare |
| energy [Wh] | 17.8 | 17.8 | 23.1 | 15.4 | 17.8 | 23.1 | 15.4 |


## targets

|  | target | value | met |
|---|---|---|---|
| Apus Mini: less energy at the quad's 20 m/s | Wh/km at 20 m/s < half the quad's | 0.36 vs 2.34 | yes |
| Apus Mini: comparable speed | sustained top speed ≥ the quad's | 223 vs 122 km/h | yes |
| Apus Mini: hover pitch with elevons only | a 90° pitch-over in the hover under 0.5 s (the quad: ~0.1 s) | 0.23 s at 245 rad/s² | yes |
| Apus Mini: comparable manoeuvre | banked turn radius at 20 m/s ≤ the quad's at 45° bank (41 m) | 9 m | yes |
| Alcotan: faster sustained | sustained top speed (pack 30C, motors 22 A) ≥ 1.25 × the quad's | 263 vs 122 km/h | yes |
| Alcotan: burst speed | thrust-limited top speed ≥ 1.25 × the quad's | 295 vs 133 km/h (52 s of pack) | yes |
| Alcotan: a chase at 40 m/s | endurance at 40 m/s ≥ the quad's at its top speed, with a turn | 8.3 min, r = 32 m | yes |
| Gavilan: span for a hangar | span ≤ a fifth of a gate (0.30 m) | 0.31 m | yes |
| Gavilan: the quad's hover agility | pitch acceleration in the hover ≥ the quad's | 813 vs 1764 rad/s² | no |
| Gavilan: brakes harder than the quad | brake distance 20→4 m/s ≤ the quad's | 5.4 vs 6.2 m | yes |
| Gavilan: flies slower than the quad can | stall speed ≤ 12 m/s (a hangar's straights) | 11.1 m/s | yes |


## manoeuvre

|  | 5-inch racing quad | Apus Mini | Alcotan | Gavilan |
|---|---|---|---|---|
| n at 20 m/s (banked) | 2.92 | 4.61 | 1.31 | 3.25 |
| turn radius at 20 m/s (banked) [m] | 14.8 | 9.06 | 48.6 | 13.2 |
| turn radius at 40 m/s (banked) [m] | 59.4 | 20.5 | 31.8 | 20.5 |
| least banked turn radius [m] | 0 | 8.92 | 31.4 | 12.6 |
| pitch acceleration in the hover [rad/s²] | 1.76e+03 | 245 | 1.52e+03 | 813 |
| hover pitch-over 90° [s] | 0.0844 | 0.226 | 0.0909 | 0.124 |
| brake deceleration at 20 m/s [g] | 3.55 | 7.23 | 4.79 | 4.75 |
| brake 20→4 m/s distance [m] | 6.18 | 4.64 | 5.2 | 5.41 |
| brake 20→4 m/s time [s] | 0.535 | 0.43 | 0.465 | 0.48 |
| stall speed [m/s] | 0 | 9.31 | 17.5 | 11.1 |
| thrust / weight at 1.2 Vs | — | 5.08 | 9.43 | 8.71 |
| pull-up speed [m/s] | — | 17 | 25 | 18 |
| pull-up radius into the hover [m] | — | 12.6 | 61.2 | 20.2 |
| transitions | none needed | both with thrust to spare; the hover pitch by elevons only | both with thrust to spare | both with thrust to spare |
| hover yaw 90° [s] (quad: torque) | 0.874 | — | — | — |
| hover roll 90° [s] | — | 0.117 | 0.0992 | 0.126 |


## sensitivity

| key | value |
|---|---|
| Apus Mini |                                     as modelled  Cd0 + 30 %  cl max − 10 %  mass + 7 % … |
| Alcotan |                                     as modelled  Cd0 + 30 %  cl max − 10 %  mass + 7 % … |
| Gavilan |                                     as modelled  Cd0 + 30 %  cl max − 10 %  mass + 7 % … |


## uncertainty

|  | relative |
|---|---|
| structure mass | 0.2 |
| Cd0 | 0.3 |
| cl_max | 0.1 |
| drive power | 0.15 |
| quad CdA | 0.3 |


## biplane question

|  | Apus Mini (twin monoplane, as designed) | Alcotan (cruciform, as designed) | Gavilan (biplane, as designed) | Apus as a biplane cell | Apus with the span a gate forbids | Apus flat: four motors in a row | Apus cruciform: four motors on wing and fins | Alcotan twin: two motors on one wing | X-C mini for the hangar | X-C mini for the race | Gavilan on one wing (cruciform) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| layout | monoplane (2 motors) | cruciform | biplane | biplane | cruciform | monoplane | cruciform | monoplane (2 motors) | xwing | xwing | cruciform |
| span [m] | 0.5 | 0.42 | 0.31 | 0.5 | 0.71 | 0.64 | 0.5 | 0.42 | 0.396 | 0.517 | 0.36 |
| span / gate | 0.333 | 0.28 | 0.207 | 0.333 | 0.473 | 0.427 | 0.333 | 0.28 | 0.264 | 0.345 | 0.24 |
| mass [kg] | 0.46 | 0.572 | 0.524 | 0.57 | 0.561 | 0.544 | 0.557 | 0.476 | 0.507 | 0.562 | 0.51 |
| wing area [m²] | 0.085 | 0.0315 | 0.062 | 0.085 | 0.0852 | 0.0832 | 0.085 | 0.0315 | 0.0432 | 0.0738 | 0.0612 |
| aspect ratio | 2.94 | 5.6 | 1.55 | 2.94 | 5.92 | 4.92 | 2.94 | 5.6 | 3.63 | 3.62 | 2.12 |
| induced k | 0.12 | 0.0632 | 0.14 | 0.0814 | 0.0598 | 0.0718 | 0.12 | 0.0632 | 0.115 | 0.115 | 0.167 |
| Cd0 at 20 m/s | 0.0181 | 0.0583 | 0.0372 | 0.0308 | 0.0262 | 0.0242 | 0.0249 | 0.0399 | 0.0608 | 0.046 | 0.0306 |
| CdA at 20 m/s [cm²] | 15.4 | 18.4 | 23.1 | 26.2 | 22.4 | 20.1 | 21.2 | 12.6 | 26.2 | 34 | 18.7 |
| L/D at 20 m/s | 9.14 | 7.93 | 6.35 | 7.32 | 8.68 | 8.99 | 7.91 | 9.6 | 5.46 | 5.37 | 6.78 |
| power at 20 m/s [W] | 26 | 34.5 | 38.7 | 36.9 | 31.6 | 30 | 33.9 | 25.6 | 42.8 | 47.4 | 35.8 |
| Wh/km at 20 m/s | 0.361 | 0.479 | 0.538 | 0.512 | 0.44 | 0.417 | 0.471 | 0.355 | 0.595 | 0.659 | 0.497 |
| best range [km] | 51 | 48.4 | 29.5 | 39.4 | 46 | 46.8 | 39.5 | 65.1 | 26.9 | 30.5 | 31.1 |
| stall speed [m/s] | 9.31 | 17.5 | 11.1 | 10.4 | 10.3 | 10.2 | 10.2 | 16 | 14.7 | 11.9 | 11 |
| turn radius at 20 m/s (banked) [m] | 9.06 | 48.6 | 13.2 | 11.4 | 11.2 | 11.1 | 11.1 | 33.7 | 26.4 | 15.3 | 13 |
| hover thrust / weight | 5.72 | 11.5 | 10 | 9.23 | 9.38 | 9.67 | 9.45 | 6.92 | 10.4 | 9.37 | 10.3 |
| hover power [W] | 87.5 | 100 | 82 | 90.8 | 89.1 | 85.9 | 88.3 | 98.3 | 79 | 89.1 | 79.5 |
| hover pitch by | elevons in the wash | differential thrust (+ elevons) | differential thrust (+ elevons) | differential thrust (+ elevons) | differential thrust (+ elevons) | elevons in the wash | differential thrust (+ elevons) | elevons in the wash | differential thrust (+ elevons) | differential thrust (+ elevons) | differential thrust (+ elevons) |
| pitch acceleration in the hover [rad/s²] | 245 | 1.52e+03 | 813 | 782 | 1.06e+03 | 174 | 922 | 85.7 | 902 | 793 | 875 |
| hover pitch-over 90° [s] | 0.226 | 0.0909 | 0.124 | 0.127 | 0.109 | 0.269 | 0.117 | 0.383 | 0.118 | 0.126 | 0.12 |
| sustained top speed [km/h] | 223 | 263 | 205 | 212 | 220 | 223 | 216 | 288 | 198 | 198 | 212 |
| top speed [km/h] | 223 | 295 | 230 | 227 | 230 | 230 | 230 | 292 | 227 | 223 | 230 |
| disc rules | no vertical thrust lever: the hover pitch comes from the elevons in the slipstream only… | all hold | all hold | all hold | all hold | no vertical thrust lever: the hover pitch comes from the elevons in the slipstream only… | all hold | no vertical thrust lever: the hover pitch comes from the elevons in the slipstream only… | all hold | all hold | the discs on one wing are 14 mm apart; the upper and lower discs are 14 mm apart: two 5… |
| parts to build | pod, 1 wing, 2 mounts, big elevons | pod, 1 wing, 2 fins, 4 mounts | pod, 2 wings, 2 posts + sleeves, 4 fittings, 4 mounts | pod, 2 wings, 2 posts + sleeves, 4 fittings, 4 mounts | pod, 1 wing, 2 fins, 4 mounts | pod, 1 wing, 4 mounts, big elevons | pod, 1 wing, 2 fins, 4 mounts | pod, 1 wing, 2 mounts, big elevons | pod, 4 panels, 4 mounts | pod, 4 panels, 4 mounts | pod, 1 wing, 2 fins, 4 mounts |


## as built

|  | Apus Mini | Alcotan | Gavilan |
|---|---|---|---|
| printed parts [g] | 187 | 205 | 189 |
| carbon tubes [g] | 12.8 | 16.8 | 19.1 |
| structure as built [g] | 200 | 222 | 208 |
| structure in the budget [g] | 128 | 130 | 142 |
| difference [g] | 71.9 | 91.5 | 66 |
| verdict | the budget is light: the as-printed rows below carry the built mass | the budget is light: the as-printed rows below carry the built mass | the budget is light: the as-printed rows below carry the built mass |
| source | Mellonia (PrusaSlicer) | Mellonia (PrusaSlicer) | Mellonia (PrusaSlicer) |

