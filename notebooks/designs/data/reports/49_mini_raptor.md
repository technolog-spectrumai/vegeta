# Mini raptors — Apus Mini, Alcotan, Gavilan: three first-guess designs on a 5-inch racing stack

*49_mini_raptor — exported 2026-10-18 03:58*

| key | value |
|---|---|
| method | first principles and empirical correlations; no CFD, no FEA; conceptual CAD |
| cad | {'Apus Mini': {'aircraft.step': '/home/user/vegeta/notebooks/designs/data/cad/mini_rapt… |


## drives

|  | 4S 5x4.3 | 4S 5x4.3 light | 6S 5x4.3 | 6S 5x5 |
|---|---|---|---|---|
| pack | 4S 1500 | 4S 1300 | 6S 1300 | 6S 1300 |
| pack mass [kg] | 0.19 | 0.165 | 0.23 | 0.23 |
| usable energy [Wh] | 17.8 | 15.4 | 23.1 | 23.1 |
| propeller | 5x4.3 three-blade | 5x4.3 three-blade | 5x4.3 three-blade | 5x5 three-blade |
| KV | 2.4e+03 | 2.4e+03 | 1.75e+03 | 1.75e+03 |
| static thrust per motor [N] | 13.4 | 13.4 | 18.4 | 16.8 |
| thrust at 20 m/s [N] | 10.9 | 10.9 | 15.4 | 14 |
| thrust at 40 m/s [N] | 8.58 | 8.58 | 13 | 11.7 |
| thrust at 60 m/s [N] | 2.88 | 2.88 | 6.38 | 9.92 |
| power at 1.2 N static [W] | 18.6 | 18.6 | 20 | 21.2 |
| power at 0.5 N, 20 m/s [W] | 22.1 | 22.1 | 24.3 | 22.4 |
| electrical ceiling per motor [W] | 622 | 622 | 932 | 932 |


## racing quad

|  | racing quad |
|---|---|
| layout | quad |
| drive | 4S 5x4.3 |
| mass [kg] | 0.546 |
| hover thrust / weight | 10 |
| vertical climb, thrust-limited [m/s] | 37.5 |
| power in that climb [W] | 2.42e+03 |
| vertical climb at the sustainable power [m/s] | 22 |
| initial vertical acceleration from the hover [g] | 9.03 |
| end-on drag area [m²] | 0.0352 |
| zoom climb from the top speed [m] | 73.6 |
| cross-section climbing [m] | 0.28 × 0.28 |
| smallest square hatch [m] | 0.382 |
| smallest slot [m] | 0.382 |
| smallest round chimney [m] | 0.499 |
| hover power [W] | 91.2 |
| hover endurance [min] | 11.7 |
| pitch acceleration in the hover [rad/s²] | 1.58e+03 |
| roll acceleration in the hover [rad/s²] | 1.58e+03 |
| yaw acceleration in the hover [rad/s²] | 15.4 |
| power at 20 m/s [W] | 180 |
| Wh/km at 20 m/s | 2.5 |
| range at 20 m/s [km] | 7.1 |
| best range [km] | 7.93 |
| at [m/s] | 15 |
| least Wh/km | 2.24 |
| best endurance [min] | 11.7 |
| top speed [m/s] | 38 |
| top speed [km/h] | 137 |
| power at top speed [W] | 891 |
| full-throttle endurance at top speed [s] | 71.8 |
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
| brake deceleration at 20 m/s [g] | 3.48 |
| brake 20→4 m/s distance [m] | 6.25 |
| brake 20→4 m/s time [s] | 0.54 |
| hover pitch-over 90° [s] | 0.0891 |
| hover yaw 90° [s] | 0.904 |
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
| mass [kg] | 0.546 | 0.493 | 0.618 | 0.565 | 0.565 | 0.71 | 0.631 |
| span / gate | 0.167 | 0.333 | 0.28 | 0.207 | 0.333 | 0.28 | 0.207 |
| top speed [km/h] | 137 | 223 | 295 | 230 | 223 | 295 | 230 |
| power at top speed [W] | 891 | 609 | 1.99e+03 | 1.06e+03 | 610 | 1.99e+03 | 1.06e+03 |
| full-throttle endurance at top speed [s] | 71.8 | 105 | 41.8 | 52.5 | 105 | 41.8 | 52.4 |
| sustained power [W] | 666 | 651 | 866 | 577 | 651 | 866 | 577 |
| sustained top speed [km/h] | 122 | 223 | 245 | 194 | 223 | 245 | 194 |
| power at 20 m/s [W] | 180 | 31.5 | 45.6 | 49.3 | 33.4 | 49.9 | 52.4 |
| Wh/km at 20 m/s | 2.5 | 0.438 | 0.634 | 0.685 | 0.463 | 0.694 | 0.728 |
| least Wh/km | 2.24 | 0.417 | 0.628 | 0.656 | 0.451 | 0.694 | 0.71 |
| range at 20 m/s [km] | 7.1 | 40.6 | 36.4 | 22.5 | 38.3 | 33.3 | 21.1 |
| best range [km] | 7.93 | 42.6 | 36.8 | 23.5 | 39.4 | 33.3 | 21.7 |
| at [m/s] | 15 | 17 | 19 | 17 | 18 | 20 | 18 |
| best endurance [min] | 11.7 | 49.3 | 32.3 | 26 | 43.1 | 27.7 | 22.9 |
| stall speed [m/s] | 0 | 9.64 | 18.2 | 11.5 | 10.3 | 19.5 | 12.2 |
| hover thrust / weight | 10 | 5.55 | 11.1 | 9.7 | 4.85 | 9.66 | 8.69 |
| hover power [W] | 91.2 | 98.1 | 117 | 94.7 | 117 | 139 | 108 |
| hover endurance [min] | 11.7 | 10.9 | 11.8 | 9.75 | 9.1 | 9.98 | 8.57 |
| n at 20 m/s (banked) | 2.92 | 4.3 | 1.21 | 3.02 | 3.76 | 1.05 | 2.7 |
| turn radius at 20 m/s (banked) [m] | 14.8 | 9.74 | 60.1 | 14.3 | 11.3 | 124 | 16.3 |
| turn radius at 40 m/s (banked) [m] | 59.4 | 20.5 | 34.5 | 20.5 | 20.5 | 39.9 | 20.5 |
| least banked turn radius [m] | 0 | 9.55 | 33.9 | 13.6 | 10.9 | 38.9 | 15.2 |
| brake deceleration at 20 m/s [g] | 3.48 | 6.75 | 4.64 | 4.42 | 5.91 | 4.39 | 4.16 |
| brake 20→4 m/s distance [m] | 6.25 | 4.79 | 5.31 | 5.52 | 5.08 | 5.47 | 5.66 |
| brake 20→4 m/s time [s] | 0.54 | 0.44 | 0.475 | 0.49 | 0.46 | 0.485 | 0.5 |
| vertical climb, thrust-limited [m/s] | 37.5 | 55.5 | 59.5 | 59.5 | 54.5 | 59.5 | 59 |
| vertical climb at the sustainable power [m/s] | 22 | 51 | 52.5 | 40 | 48.5 | 50.5 | 38.5 |
| power in that climb [W] | 2.42e+03 | 781 | 1.09e+03 | 1.26e+03 | 798 | 1.16e+03 | 1.31e+03 |
| initial vertical acceleration from the hover [g] | 9.03 | 4.55 | 10.1 | 8.7 | 3.85 | 8.66 | 7.69 |
| zoom climb from the top speed [m] | 73.6 | 196 | 343 | 209 | 196 | 343 | 209 |
| cross-section climbing [m] | 0.28 × 0.28 | 0.50 × 0.13 | 0.42 × 0.41 | 0.31 × 0.28 | 0.50 × 0.13 | 0.42 × 0.41 | 0.31 × 0.28 |
| smallest square hatch [m] | 0.382 | 0.6 | 0.52 | 0.41 | 0.6 | 0.52 | 0.41 |
| smallest slot [m] | 0.382 | 0.227 | 0.507 | 0.382 | 0.227 | 0.507 | 0.382 |
| smallest round chimney [m] | 0.499 | 0.616 | 0.685 | 0.519 | 0.616 | 0.685 | 0.519 |
| hover pitch-over 90° [s] | 0.0891 | 0.217 | 0.0931 | 0.126 | 0.203 | 0.0937 | 0.127 |
| pitch acceleration in the hover [rad/s²] | 1.58e+03 | 266 | 1.45e+03 | 786 | 305 | 1.43e+03 | 776 |
| hover pitch by | — | elevons in the wash | differential thrust (+ elevons) | differential thrust (+ elevons) | elevons in the wash | differential thrust (+ elevons) | differential thrust (+ elevons) |
| thrust / weight at 1.2 Vs | — | 4.92 | 9.05 | 8.39 | 4.26 | 7.76 | 7.45 |
| pull-up speed [m/s] | — | 17 | 26 | 17 | 18 | 28 | 18 |
| pull-up radius into the hover [m] | — | 14 | 66.1 | 25 | 16.2 | 75.2 | 27.8 |
| transitions | none needed | both with thrust to spare; the hover pitch by elevons only | both with thrust to spare | both with thrust to spare | both with thrust to spare; the hover pitch by elevons only | both with thrust to spare | both with thrust to spare |
| energy [Wh] | 17.8 | 17.8 | 23.1 | 15.4 | 17.8 | 23.1 | 15.4 |


## targets

|  | target | value | met |
|---|---|---|---|
| Apus Mini: less energy at the quad's 20 m/s | Wh/km at 20 m/s < half the quad's | 0.44 vs 2.50 | yes |
| Apus Mini: comparable speed | sustained top speed ≥ the quad's | 223 vs 122 km/h | yes |
| Apus Mini: hover pitch with elevons only | a 90° pitch-over in the hover under 0.5 s (the quad: ~0.1 s) | 0.22 s at 266 rad/s² | yes |
| Apus Mini: comparable manoeuvre | banked turn radius at 20 m/s ≤ the quad's at 45° bank (41 m) | 10 m | yes |
| Alcotan: faster sustained | sustained top speed (pack 30C, motors 22 A) ≥ 1.25 × the quad's | 245 vs 122 km/h | yes |
| Alcotan: burst speed | thrust-limited top speed ≥ 1.25 × the quad's | 295 vs 137 km/h (42 s of pack) | yes |
| Alcotan: a chase at 40 m/s | endurance at 40 m/s ≥ the quad's at its top speed, with a turn | 6.2 min, r = 34 m | yes |
| Gavilan: span for a hangar | span ≤ a fifth of a gate (0.30 m) | 0.31 m | yes |
| Gavilan: the quad's hover agility | pitch acceleration in the hover ≥ the quad's | 786 vs 1584 rad/s² | no |
| Gavilan: brakes harder than the quad | brake distance 20→4 m/s ≤ the quad's | 5.5 vs 6.2 m | yes |
| Gavilan: flies slower than the quad can | stall speed ≤ 12 m/s (a hangar's straights) | 11.5 m/s | yes |


## manoeuvre

|  | 5-inch racing quad | Apus Mini | Alcotan | Gavilan |
|---|---|---|---|---|
| n at 20 m/s (banked) | 2.92 | 4.3 | 1.21 | 3.02 |
| turn radius at 20 m/s (banked) [m] | 14.8 | 9.74 | 60.1 | 14.3 |
| turn radius at 40 m/s (banked) [m] | 59.4 | 20.5 | 34.5 | 20.5 |
| least banked turn radius [m] | 0 | 9.55 | 33.9 | 13.6 |
| pitch acceleration in the hover [rad/s²] | 1.58e+03 | 266 | 1.45e+03 | 786 |
| hover pitch-over 90° [s] | 0.0891 | 0.217 | 0.0931 | 0.126 |
| brake deceleration at 20 m/s [g] | 3.48 | 6.75 | 4.64 | 4.42 |
| brake 20→4 m/s distance [m] | 6.25 | 4.79 | 5.31 | 5.52 |
| brake 20→4 m/s time [s] | 0.54 | 0.44 | 0.475 | 0.49 |
| stall speed [m/s] | 0 | 9.64 | 18.2 | 11.5 |
| thrust / weight at 1.2 Vs | — | 4.92 | 9.05 | 8.39 |
| pull-up speed [m/s] | — | 17 | 26 | 17 |
| pull-up radius into the hover [m] | — | 14 | 66.1 | 25 |
| transitions | none needed | both with thrust to spare; the hover pitch by elevons only | both with thrust to spare | both with thrust to spare |
| hover yaw 90° [s] (quad: torque) | 0.904 | — | — | — |
| hover roll 90° [s] | — | 0.124 | 0.105 | 0.132 |


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
| mass [kg] | 0.493 | 0.618 | 0.565 | 0.616 | 0.607 | 0.59 | 0.603 | 0.509 | 0.548 | 0.608 | 0.551 |
| wing area [m²] | 0.085 | 0.0315 | 0.062 | 0.085 | 0.0852 | 0.0832 | 0.085 | 0.0315 | 0.0432 | 0.0738 | 0.0612 |
| aspect ratio | 2.94 | 5.6 | 1.55 | 2.94 | 5.92 | 4.92 | 2.94 | 5.6 | 3.63 | 3.62 | 2.12 |
| induced k | 0.12 | 0.0632 | 0.14 | 0.0814 | 0.0598 | 0.0718 | 0.12 | 0.0632 | 0.115 | 0.115 | 0.167 |
| Cd0 at 20 m/s | 0.0213 | 0.0755 | 0.046 | 0.0372 | 0.0326 | 0.0307 | 0.0313 | 0.0485 | 0.0733 | 0.0534 | 0.0395 |
| CdA at 20 m/s [cm²] | 18.1 | 23.8 | 28.5 | 31.6 | 27.8 | 25.5 | 26.6 | 15.3 | 31.7 | 39.4 | 24.2 |
| L/D at 20 m/s | 8.37 | 6.86 | 5.64 | 6.59 | 7.62 | 7.78 | 6.93 | 8.63 | 4.94 | 5 | 5.89 |
| power at 20 m/s [W] | 31.5 | 45.6 | 49.3 | 46.5 | 40.5 | 38.8 | 43.6 | 31.9 | 54.1 | 58.6 | 46.5 |
| Wh/km at 20 m/s | 0.438 | 0.634 | 0.685 | 0.645 | 0.563 | 0.539 | 0.606 | 0.443 | 0.751 | 0.813 | 0.645 |
| best range [km] | 42.6 | 36.8 | 23.5 | 32.1 | 37.3 | 37.6 | 31.3 | 52.3 | 21.5 | 24.9 | 24.2 |
| stall speed [m/s] | 9.64 | 18.2 | 11.5 | 10.8 | 10.7 | 10.7 | 10.7 | 16.5 | 15.3 | 12.3 | 11.5 |
| turn radius at 20 m/s (banked) [m] | 9.74 | 60.1 | 14.3 | 12.4 | 12.1 | 12.1 | 12.1 | 37.9 | 29.6 | 16.8 | 14.1 |
| hover thrust / weight | 5.55 | 11.1 | 9.7 | 8.89 | 9.02 | 9.28 | 9.09 | 6.74 | 9.99 | 9.02 | 9.94 |
| hover power [W] | 98.1 | 117 | 94.7 | 105 | 103 | 99.5 | 102 | 111 | 91.6 | 103 | 92.2 |
| hover pitch by | elevons in the wash | differential thrust (+ elevons) | differential thrust (+ elevons) | differential thrust (+ elevons) | differential thrust (+ elevons) | elevons in the wash | differential thrust (+ elevons) | elevons in the wash | differential thrust (+ elevons) | differential thrust (+ elevons) | differential thrust (+ elevons) |
| pitch acceleration in the hover [rad/s²] | 266 | 1.45e+03 | 786 | 764 | 1.02e+03 | 185 | 890 | 94.4 | 878 | 752 | 865 |
| hover pitch-over 90° [s] | 0.217 | 0.0931 | 0.126 | 0.128 | 0.111 | 0.261 | 0.119 | 0.365 | 0.12 | 0.129 | 0.121 |
| sustained top speed [km/h] | 223 | 245 | 194 | 202 | 209 | 209 | 205 | 288 | 191 | 191 | 198 |
| top speed [km/h] | 223 | 295 | 230 | 227 | 230 | 230 | 230 | 288 | 227 | 223 | 230 |
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

