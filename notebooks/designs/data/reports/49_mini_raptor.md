# Mini raptors — Apus Mini, Alcotan, Gavilan: three first-guess designs on a 5-inch racing stack

*49_mini_raptor — exported 2026-10-18 03:21*

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
| layout | cruciform | cruciform | biplane |
| title | economy: the quad's speed and manoeuvre for a third of the power — one wing a third of … | the chaser: sustained speed on 6S and the 5x5, the motors at the tips of a + | tight spaces: the racing frame's motor square in a short cell, the wing as a brake |
| drive | 4S 5x4.3 | 6S 5x5 | 4S 5x4.3 light |
| pack | 4S 1500 | 6S 1300 | 4S 1300 |
| span | 0.5 | 0.42 | 0.31 |
| chord | 0.17 | 0.075 | 0.1 |
| S_ref | 0.085 | 0.0315 | 0.062 |
| AR | 2.94 | 5.6 | 1.55 |
| motor_y | 0.14 | 0.14 | 0.0765 |
| motor_z | 0.14 | 0.14 | 0.0775 |
| disc_gap_span | 0.071 | 0.071 | 0.026 |
| disc_gap_vertical | 0.071 | 0.071 | 0.028 |
| disc_to_pod | 0.0515 | 0.0515 | ∞ |
| span_over_gate | 0.333 | 0.28 | 0.207 |
| Re_20 | 2.33e+05 | 1.03e+05 | 1.37e+05 |
| cl_max | 1 | 0.95 | 1.1 |
| n_struct | 8 | 10 | 8 |
| section | thin cambered section (3% camber, 10 % thick); the elevons' reflex trims it | thin cambered section (2% camber, 10 % thick); the elevons' reflex trims it | thin cambered section (4% camber, 10 % thick); the elevons' reflex trims it |
| rules | all hold | all hold | all hold |


## comparison

|  | 5-inch racing quad | Apus Mini | Alcotan | Gavilan | Apus Mini (as printed) | Alcotan (as printed) | Gavilan (as printed) |
|---|---|---|---|---|---|---|---|
| layout | quad | cruciform | cruciform | biplane | cruciform | cruciform | biplane |
| mass [kg] | 0.5 | 0.557 | 0.572 | 0.524 | 0.655 | 0.664 | 0.59 |
| span / gate | 0.167 | 0.333 | 0.28 | 0.207 | 0.333 | 0.28 | 0.207 |
| top speed [km/h] | 133 | 230 | 295 | 230 | 230 | 295 | 230 |
| power at top speed [W] | 827 | 888 | 1.6e+03 | 921 | 889 | 1.6e+03 | 922 |
| full-throttle endurance at top speed [s] | 77.3 | 72 | 51.9 | 60.2 | 71.9 | 51.9 | 60.1 |
| sustained power [W] | 666 | 666 | 866 | 577 | 666 | 866 | 577 |
| sustained top speed [km/h] | 122 | 216 | 263 | 205 | 216 | 263 | 205 |
| power at 20 m/s [W] | 169 | 33.9 | 34.5 | 38.7 | 36.6 | 38.1 | 41.3 |
| Wh/km at 20 m/s | 2.34 | 0.471 | 0.479 | 0.538 | 0.508 | 0.529 | 0.574 |
| least Wh/km | 2.01 | 0.45 | 0.477 | 0.521 | 0.497 | 0.529 | 0.566 |
| range at 20 m/s [km] | 7.57 | 37.7 | 48.2 | 28.6 | 35 | 43.7 | 26.8 |
| best range [km] | 8.83 | 39.5 | 48.4 | 29.5 | 35.7 | 43.7 | 27.2 |
| at [m/s] | 15 | 17 | 19 | 18 | 18 | 20 | 18 |
| best endurance [min] | 13.7 | 45.2 | 44.5 | 32.2 | 38.2 | 38 | 28.3 |
| stall speed [m/s] | 0 | 10.2 | 17.5 | 11.1 | 11.1 | 18.9 | 11.8 |
| hover thrust / weight | 10.5 | 9.45 | 11.5 | 10 | 8.04 | 9.92 | 8.92 |
| hover power [W] | 77.6 | 88.3 | 100 | 82 | 108 | 120 | 94.4 |
| hover endurance [min] | 13.7 | 12.1 | 13.8 | 11.3 | 9.88 | 11.5 | 9.79 |
| n at 20 m/s (banked) | 2.92 | 3.81 | 1.31 | 3.25 | 3.24 | 1.13 | 2.89 |
| turn radius at 20 m/s (banked) [m] | 14.8 | 11.1 | 48.6 | 13.2 | 13.2 | 78.9 | 15 |
| turn radius at 40 m/s (banked) [m] | 59.4 | 20.5 | 31.8 | 20.5 | 20.5 | 37.1 | 20.5 |
| least banked turn radius [m] | 0 | 10.8 | 31.4 | 12.6 | 12.7 | 36.4 | 14.2 |
| brake deceleration at 20 m/s [g] | 3.55 | 6.75 | 4.79 | 4.75 | 5.76 | 4.51 | 4.26 |
| brake 20→4 m/s distance [m] | 6.18 | 4.82 | 5.2 | 5.41 | 5.15 | 5.4 | 5.56 |
| brake 20→4 m/s time [s] | 0.535 | 0.445 | 0.465 | 0.48 | 0.465 | 0.48 | 0.49 |
| hover pitch-over 90° [s] | 0.0844 | 0.117 | 0.0909 | 0.124 | 0.118 | 0.0916 | 0.125 |
| pitch acceleration in the hover [rad/s²] | 1.76e+03 | 922 | 1.52e+03 | 813 | 902 | 1.5e+03 | 802 |
| thrust / weight at 1.2 Vs | — | 8.29 | 9.43 | 8.71 | 6.97 | 8 | 7.67 |
| pull-up speed [m/s] | — | 17 | 25 | 18 | 18 | 27 | 18 |
| pull-up radius into the hover [m] | — | 16.8 | 61.2 | 20.2 | 20.3 | 70.7 | 24.7 |
| transitions | none needed | both with thrust to spare | both with thrust to spare | both with thrust to spare | both with thrust to spare | both with thrust to spare | both with thrust to spare |
| energy [Wh] | 17.8 | 17.8 | 23.1 | 15.4 | 17.8 | 23.1 | 15.4 |


## targets

|  | target | value | met |
|---|---|---|---|
| Apus Mini: less energy at the quad's 20 m/s | Wh/km at 20 m/s < half the quad's | 0.47 vs 2.34 | yes |
| Apus Mini: comparable speed | sustained top speed ≥ the quad's | 216 vs 122 km/h | yes |
| Apus Mini: comparable manoeuvre | banked turn radius at 20 m/s ≤ the quad's at 45° bank (41 m) | 11 m | yes |
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
| n at 20 m/s (banked) | 2.92 | 3.81 | 1.31 | 3.25 |
| turn radius at 20 m/s (banked) [m] | 14.8 | 11.1 | 48.6 | 13.2 |
| turn radius at 40 m/s (banked) [m] | 59.4 | 20.5 | 31.8 | 20.5 |
| least banked turn radius [m] | 0 | 10.8 | 31.4 | 12.6 |
| pitch acceleration in the hover [rad/s²] | 1.76e+03 | 922 | 1.52e+03 | 813 |
| hover pitch-over 90° [s] | 0.0844 | 0.117 | 0.0909 | 0.124 |
| brake deceleration at 20 m/s [g] | 3.55 | 6.75 | 4.79 | 4.75 |
| brake 20→4 m/s distance [m] | 6.18 | 4.82 | 5.2 | 5.41 |
| brake 20→4 m/s time [s] | 0.535 | 0.445 | 0.465 | 0.48 |
| stall speed [m/s] | 0 | 10.2 | 17.5 | 11.1 |
| thrust / weight at 1.2 Vs | — | 8.29 | 9.43 | 8.71 |
| pull-up speed [m/s] | — | 17 | 25 | 18 |
| pull-up radius into the hover [m] | — | 16.8 | 61.2 | 20.2 |
| transitions | none needed | both with thrust to spare | both with thrust to spare | both with thrust to spare |
| hover yaw 90° [s] (quad: torque) | 0.874 | — | — | — |
| hover roll 90° [s] | — | 0.112 | 0.0992 | 0.126 |


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

|  | Apus Mini (one wing, as designed) | Gavilan (biplane, as designed) | Apus as a biplane cell | Apus with the span a gate forbids | Gavilan on one wing (cruciform) |
|---|---|---|---|---|---|
| layout | cruciform | biplane | biplane | cruciform | cruciform |
| span [m] | 0.5 | 0.31 | 0.5 | 0.71 | 0.36 |
| span / gate | 0.333 | 0.207 | 0.333 | 0.473 | 0.24 |
| mass [kg] | 0.557 | 0.524 | 0.57 | 0.561 | 0.51 |
| wing area [m²] | 0.085 | 0.062 | 0.085 | 0.0852 | 0.0612 |
| aspect ratio | 2.94 | 1.55 | 2.94 | 5.92 | 2.12 |
| induced k | 0.12 | 0.14 | 0.0814 | 0.0598 | 0.167 |
| Cd0 at 20 m/s | 0.0249 | 0.0372 | 0.0308 | 0.0262 | 0.0306 |
| CdA at 20 m/s [cm²] | 21.2 | 23.1 | 26.2 | 22.4 | 18.7 |
| L/D at 20 m/s | 7.91 | 6.35 | 7.32 | 8.68 | 6.78 |
| power at 20 m/s [W] | 33.9 | 38.7 | 36.9 | 31.6 | 35.8 |
| Wh/km at 20 m/s | 0.471 | 0.538 | 0.512 | 0.44 | 0.497 |
| best range [km] | 39.5 | 29.5 | 39.4 | 46 | 31.1 |
| stall speed [m/s] | 10.2 | 11.1 | 10.4 | 10.3 | 11 |
| turn radius at 20 m/s (banked) [m] | 11.1 | 13.2 | 11.4 | 11.2 | 13 |
| pitch acceleration in the hover [rad/s²] | 922 | 813 | 782 | 1.06e+03 | 875 |
| sustained top speed [km/h] | 216 | 205 | 212 | 220 | 212 |
| disc rules | all hold | all hold | all hold | all hold | the discs on one wing are 14 mm apart; the upper and lower discs are 14 mm apart: two 5… |
| parts to build | pod, 1 wing, 2 fins, 4 mounts | pod, 2 wings, 2 posts + sleeves, 4 fittings, 4 mounts | pod, 2 wings, 2 posts + sleeves, 4 fittings, 4 mounts | pod, 1 wing, 2 fins, 4 mounts | pod, 1 wing, 2 fins, 4 mounts |


## as built

|  | Apus Mini | Alcotan | Gavilan |
|---|---|---|---|
| printed parts [g] | 234 | 205 | 189 |
| carbon tubes [g] | 18.4 | 16.8 | 19.1 |
| structure as built [g] | 253 | 222 | 208 |
| structure in the budget [g] | 155 | 130 | 142 |
| difference [g] | 97.8 | 91.5 | 66 |
| verdict | the budget is light: the as-printed rows below carry the built mass | the budget is light: the as-printed rows below carry the built mass | the budget is light: the as-printed rows below carry the built mass |
| source | Mellonia (PrusaSlicer) | Mellonia (PrusaSlicer) | Mellonia (PrusaSlicer) |

