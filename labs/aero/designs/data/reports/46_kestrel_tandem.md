# Kestrel-Q4T — the tandem-wing Kestrel (Q4T1N-10 against Q4T1P-10, Q4T1N-12 and Kestrel-4Q-2)

*46_kestrel_tandem — exported 2026-10-17 22:33*

| key | value |
|---|---|
| main | Q4T1N-10 |
| biplane_reference | 43_kestrel_study.json (2026-10-17T09:37) |
| debug | no |


## variants

|  | span | chord | gap | stagger_c | sign | prop | battery_ah | title |
|---|---|---|---|---|---|---|---|---|
| Q4T1N-10 | 1 | 0.14 | 0.28 | 3 | -1 | 10x7 | 3.3 | front wing low, rear high — the design |
| Q4T1P-10 | 1 | 0.14 | 0.28 | 3 | 1 | 10x7 | 3.3 | front wing high, rear low — the mirror |
| Q4T1N-12 | 1.2 | 0.14 | 0.28 | 3 | -1 | 10x7 | 3.3 | the design at span 1.2 m |
| Q4T1-10 (flat) | 1 | 0.14 | 0 | 3 | 0 | 10x7 | 3.3 | both wings in one plane — the control case |


## layout

|  | span | chord | gap | stagger | x_le_front | x_le_rear | z_front | z_rear | S_ref | AR_wing | AR_total | pod_length | pod_height | hover_pitch_lever |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Q4T1N-10 | 1 | 0.14 | 0.28 | 0.42 | 0.2 | 0.62 | -0.14 | 0.14 | 0.28 | 7.14 | 3.57 | 0.86 | 0.34 | 0.23 |
| Q4T1P-10 | 1 | 0.14 | 0.28 | 0.42 | 0.2 | 0.62 | 0.14 | -0.14 | 0.28 | 7.14 | 3.57 | 0.86 | 0.34 | 0.23 |
| Q4T1N-12 | 1.2 | 0.14 | 0.28 | 0.42 | 0.2 | 0.62 | -0.14 | 0.14 | 0.336 | 8.57 | 4.29 | 0.86 | 0.34 | 0.23 |
| Q4T1-10 (flat) | 1 | 0.14 | 0 | 0.42 | 0.2 | 0.62 | 0 | 0 | 0.28 | 7.14 | 3.57 | 0.86 | 0.11 | 0 |

*3 more columns not shown: prop_ground_clearance, prop_to_pod, elevon_y_in*


## masses

|  | Q4T1N-10 | Q4T1P-10 | Q4T1N-12 | Q4T1-10 (flat) |
|---|---|---|---|---|
| propulsion | 0.752 | 0.752 | 0.752 | 0.752 |
| structure | 0.617 | 0.617 | 0.67 | 0.633 |
| systems | 0.219 | 0.219 | 0.219 | 0.219 |
| payload | 0.105 | 0.105 | 0.105 | 0.105 |
| margin | 0.06 | 0.06 | 0.06 | 0.06 |


## spars

|  | n | half-wing lift [N] | root moment [N·m] | spar | spar stress [MPa] | spar reserve | tip deflection [mm] | tip deflection / half-span [%] | first bending, with the motor [Hz] | spar mass per wing [g] |
|---|---|---|---|---|---|---|---|---|---|---|
| Q4T1N-10 | 4 | 22.7 | 2.45 | carbon Ø10×8 | 42.3 | 9.46 | 6.36 | 1.43 | 23.6 | 43.8 |
| Q4T1P-10 | 4 | 22.7 | 2.45 | carbon Ø10×8 | 42.3 | 9.46 | 6.36 | 1.43 | 23.6 | 43.8 |
| Q4T1N-12 | 4 | 23.4 | 3.3 | carbon Ø10×8 | 57 | 7.02 | 13.7 | 2.52 | 23.6 | 52.6 |
| Q4T1-10 (flat) | 4 | 22.9 | 2.49 | carbon Ø10×8 | 43 | 9.31 | 6.44 | 1.45 | 23.6 | 43.8 |


## lattice

|  | CLα [1/rad] | CLα one wing | k = CDi/CL² | e (span) | k / the biplane's | x_np from the front LE [c] | front share at α 4° | front slope / alone | rear slope / alone (downwash) | CL at α 0 (decalage) |
|---|---|---|---|---|---|---|---|---|---|---|
| Q4T1N-10 | 4.23 | 4.58 | 0.0607 | 1.47 | 1.01 | 1.55 | 0.674 | 1.02 | 0.821 | 0.0659 |
| Q4T1P-10 | 4.23 | 4.58 | 0.0607 | 1.47 | 1.01 | 1.61 | 0.674 | 1.02 | 0.821 | 0.0659 |
| Q4T1N-12 | 4.43 | 4.8 | 0.0526 | 1.41 | 0.877 | 1.54 | 0.675 | 1.02 | 0.819 | 0.0688 |
| Q4T1-10 (flat) | 3.65 | 4.58 | 0.0833 | 1.07 | 1.39 | 1.28 | 0.816 | 1.04 | 0.551 | 0.0442 |


## Munk

| Munk: the gap decides the induced drag, not the stagger | value |
|---|---|
| biplane k (lattice, gap 0.28, no stagger) | 0.0599 |
| tandem flat k (lattice) | 0.0833 |
| tandem N k (lattice) | 0.0607 |
| flat / N (lattice) | 1.37 |
| 1 / ((1+σ)/2) at gap/span 0.28 (Prandtl) | 1.44 |
| N / biplane (lattice) | 1.01 |


## stall rule

|  | front share at trim | stalls first | CL max (aircraft) | rear wing cl at the front's stall | static margin [c] | static margin [stagger] | nose-down at the stall |
|---|---|---|---|---|---|---|---|
| Q4T1N-10 | 0.667 | front | 0.978 | 0.651 | 0.3 | 0.1 | yes |
| Q4T1P-10 | 0.647 | front | 1.01 | 0.711 | 0.3 | 0.1 | yes |
| Q4T1N-12 | 0.669 | front | 0.976 | 0.647 | 0.3 | 0.1 | yes |
| Q4T1-10 (flat) | 0.725 | front | 0.9 | 0.494 | 0.21 | 0.0699 | yes |


## drag areas

|  | wings (2) | pod | nacelles (4) | pylons (2, faired) | skid legs (2, faired) and runners | fin | camera dome | pylons (4, faired) |
|---|---|---|---|---|---|---|---|---|
| Q4T1N-10 | 24.5 | 11 | 6.44 | 2.16 | 2.4 | 0.611 | 4.07 | — |
| Q4T1P-10 | 24.5 | 11 | 6.44 | 2.16 | 2.4 | 0.611 | 4.07 | — |
| Q4T1N-12 | 29.4 | 11 | 6.44 | 2.16 | 2.4 | 0.611 | 4.07 | — |
| Q4T1-10 (flat) | 24.5 | 5.08 | 6.44 | — | 2.4 | 0.611 | 4.07 | 4.32 |


## elevator

|  | Cm_delta rear pair [1/rad] | Cm_delta front pair [1/rad] | rear arm [c] | front arm [c] | Cm_delta both, opposed [1/rad] |
|---|---|---|---|---|---|
| Q4T1N-10 | -0.309 | 0.106 | 2.35 | -0.648 | -0.415 |
| Q4T1P-10 | -0.301 | 0.115 | 2.29 | -0.708 | -0.416 |
| Q4T1N-12 | -0.366 | 0.125 | 2.36 | -0.644 | -0.492 |
| Q4T1-10 (flat) | -0.223 | 0.0787 | 2.53 | -0.474 | -0.301 |


## hover authority

| key | value |
|---|---|
| Q4T1N-10 |                                                      most negative [N·m]  most positive… |
| Q4T1P-10 |                                                      most negative [N·m]  most positive… |
| Q4T1N-12 |                                                      most negative [N·m]  most positive… |
| Q4T1-10 (flat) |                                                      most negative [N·m]  most positive… |


## cruise tables

| key | value |
|---|---|
| Q4T1N-10 |                CL  feasible note   α [deg]        L/D  drag [N]  elevator trim [deg]  Δ… |
| Q4T1P-10 |                CL  feasible note   α [deg]        L/D  drag [N]  elevator trim [deg]  Δ… |
| Q4T1N-12 |                CL  feasible note   α [deg]        L/D  drag [N]  elevator trim [deg]  Δ… |
| Q4T1-10 (flat) |                CL  feasible note    α [deg]        L/D  drag [N]  elevator trim [deg]  … |


## horizontal start

|  | lifts off | stall speed [m/s] | lift-off speed [m/s] | ground roll [m] | time [s] | static T/W | CL on the ground | rolling friction |
|---|---|---|---|---|---|---|---|---|
| Q4T1N-10 | yes | 10.1 | 11.6 | 2.74 | 0.42 | 3.35 | 0.287 | 0.08 |
| Q4T1P-10 | yes | 9.97 | 11.5 | 2.74 | 0.42 | 3.35 | 0.287 | 0.08 |
| Q4T1N-12 | yes | 9.39 | 10.8 | 2.43 | 0.4 | 3.25 | 0.301 | 0.08 |
| Q4T1-10 (flat) | yes | 10.6 | 12.2 | 2.96 | 0.44 | 3.32 | 0.235 | 0.08 |


## transitions

| vehicle · direction · t_tr [s] · headwind [m/s] | height loss [m] | height gain [m] | pitch error max [deg] | time stalled, free stream [s] | α blown max while stalled [deg] | moment shortfall [s] | least pitch margin | peak rotor thrust / T_max | elevon max [deg] | roll demand / authority | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Q4T1N-10 · forward · 1.5 · 0 | 1.03 | 0.0126 | 9.54 | 0.248 | 7.03 | 0 | 0.493 | 1 | 12.6 | 0.0064 | yes |
| Q4T1N-10 · forward · 1.5 · 6 | 0.853 | 0.0655 | 7.46 | 1.03 | 21.4 | 0 | 0.554 | 1 | 11.2 | 0.0122 | yes |
| Q4T1N-10 · forward · 3 · 0 | 0.686 | 0.295 | 4.59 | 0.844 | 11.4 | 0 | 0.532 | 0.65 | 11.7 | 0.0178 | yes |
| Q4T1N-10 · forward · 3 · 6 | 0.804 | 0.0321 | 4.24 | 2.14 | 21.4 | 0 | 0.51 | 0.614 | 12.2 | 0.0233 | yes |
| Q4T1N-10 · forward · 5 · 0 | 0.962 | 0.457 | 4.63 | 2.07 | 16.3 | 0 | 0.489 | 0.615 | 12.8 | 0.0274 | yes |
| Q4T1N-10 · forward · 5 · 6 | 1.98 | 0.415 | 5.86 | 3.95 | 24.3 | 0 | 0.444 | 0.973 | 13.9 | 0.0312 | yes |
| Q4T1N-10 · back · 1.5 · 0 | 0.675 | 18.2 | 7.55 | 0.736 | 33.2 | 0 | 0.292 | 0.607 | 17.7 | 0.153 | no |
| Q4T1N-10 · back · 1.5 · 6 | 0.675 | 19 | 7.55 | 0.736 | 33.2 | 0 | 0.292 | 0.607 | 17.7 | 0.153 | no |
| Q4T1N-10 · back · 3 · 0 | 0.676 | 44.4 | 4.89 | 0 | 0 | 0 | 0.489 | 0.523 | 12.8 | 0 | no |
| Q4T1N-10 · back · 3 · 6 | 0.676 | 43.8 | 4.89 | 0 | 0 | 0 | 0.489 | 0.523 | 12.8 | 0.017 | no |
| Q4T1N-10 · back · 5 · 0 | 0.676 | 52 | 2.91 | 0 | 0 | 0 | 0.56 | 0.455 | 11 | 0 | no |
| Q4T1N-10 · back · 5 · 6 | 0.676 | 52.4 | 2.91 | 0 | 0 | 0 | 0.56 | 0.455 | 11 | 0.0188 | no |
| Q4T1P-10 · forward · 1.5 · 0 | 1.08 | 0.011 | 9.25 | 0.244 | 6.34 | 0 | 0.563 | 1 | 10.9 | 0.00636 | yes |
| Q4T1P-10 · forward · 1.5 · 6 | 0.824 | -2.89e-07 | 5.96 | 1.04 | 22.4 | 0 | 0.641 | 1 | 8.94 | 0.0137 | yes |
| Q4T1P-10 · forward · 3 · 0 | 0.63 | 0.274 | 3.94 | 0.838 | 9.82 | 0 | 0.603 | 0.586 | 9.9 | 0.0185 | yes |
| Q4T1P-10 · forward · 3 · 6 | 0.889 | -2.89e-07 | 2.83 | 2.27 | 22.5 | 0 | 0.602 | 0.587 | 9.93 | 0.0293 | yes |
| Q4T1P-10 · forward · 5 · 0 | 1.29 | 0.404 | 2.2 | 2.27 | 18.1 | 0 | 0.581 | 0.533 | 10.5 | 0.0356 | yes |
| Q4T1P-10 · forward · 5 · 6 | 1.67 | 0.438 | 2.5 | 3.87 | 24.3 | 0 | 0.577 | 0.543 | 10.6 | 0.0389 | yes |
| Q4T1P-10 · back · 1.5 · 0 | 0.67 | 17.9 | 8.81 | 0.688 | 32.5 | 0 | 0.442 | 0.547 | 13.9 | 0.117 | no |
| Q4T1P-10 · back · 1.5 · 6 | 0.67 | 19 | 8.81 | 0.688 | 32.5 | 0 | 0.442 | 0.547 | 13.9 | 0.117 | no |
| Q4T1P-10 · back · 3 · 0 | 0.67 | 39 | 5.58 | 0 | 0 | 0 | 0.556 | 0.457 | 11.1 | 0 | no |
| Q4T1P-10 · back · 3 · 6 | 0.67 | 38.7 | 5.58 | 0 | 0 | 0 | 0.556 | 0.457 | 11.1 | 0.0161 | no |
| Q4T1P-10 · back · 5 · 0 | 0.67 | 45.1 | 3.14 | 0 | 0 | 0 | 0.623 | 0.393 | 9.4 | 0 | no |
| Q4T1P-10 · back · 5 · 6 | 0.67 | 45.4 | 3.14 | 0 | 0 | 0 | 0.623 | 0.393 | 9.4 | 0.00925 | no |
| Q4T1N-12 · forward · 1.5 · 0 | 1.11 | 0.00818 | 9.76 | 0.252 | 6.86 | 0 | 0.542 | 1 | 11.4 | 0.00412 | yes |
| Q4T1N-12 · forward · 1.5 · 6 | 0.941 | 0.0187 | 7.67 | 1.03 | 20.3 | 0 | 0.583 | 1 | 10.4 | 0.00775 | yes |
| Q4T1N-12 · forward · 3 · 0 | 0.794 | 0.246 | 4.92 | 0.894 | 11.3 | 0 | 0.531 | 0.661 | 11.7 | 0.0121 | yes |
| Q4T1N-12 · forward · 3 · 6 | 1.11 | -2.02e-07 | 4.84 | 2.23 | 20.3 | 0 | 0.489 | 0.689 | 12.8 | 0.015 | yes |
| Q4T1N-12 · forward · 5 · 0 | 1.6 | 0.399 | 5.71 | 2.24 | 16.4 | 0 | 0.425 | 0.83 | 14.4 | 0.0174 | yes |
| Q4T1N-12 · forward · 5 · 6 | 2.78 | 0.684 | 6.07 | 4.07 | 23.7 | 0 | 0.365 | 1 | 15.9 | 0.0192 | yes |
| Q4T1N-12 · back · 1.5 · 0 | 0.724 | 19.8 | 7.5 | 0.662 | 28.4 | 0 | 0.455 | 0.544 | 13.6 | 0.0604 | no |
| Q4T1N-12 · back · 1.5 · 6 | 0.724 | 20.5 | 7.5 | 0.662 | 28.4 | 0 | 0.455 | 0.544 | 13.6 | 0.0604 | no |
| Q4T1N-12 · back · 3 · 0 | 0.725 | 41.1 | 4.97 | 0 | 0 | 0 | 0.547 | 0.466 | 11.3 | 0 | no |
| Q4T1N-12 · back · 3 · 6 | 0.725 | 40.6 | 4.97 | 0 | 0 | 0 | 0.547 | 0.466 | 11.3 | 0.0114 | no |
| Q4T1N-12 · back · 5 · 0 | 0.726 | 47.4 | 2.99 | 0 | 0 | 0 | 0.61 | 0.406 | 9.73 | 0 | no |
| Q4T1N-12 · back · 5 · 6 | 0.726 | 47.8 | 2.99 | 0 | 0 | 0 | 0.61 | 0.406 | 9.73 | 0.00873 | no |
| Q4T1-10 (flat) · forward · 1.5 · 0 | 3.81 | 17.5 | 8.72 | 0 | 0 | 0 | 0 | 1 | 18.5 | 0.0278 | no |
| Q4T1-10 (flat) · forward · 1.5 · 6 | 3.81 | 17.7 | 9.57 | 0 | 0 | 0 | 0 | 1 | 21 | 0.0194 | no |
| Q4T1-10 (flat) · forward · 3 · 0 | 3.69 | 28.4 | 4.35 | 0 | 0 | 0 | 0 | 1 | 12.7 | 0 | no |
| Q4T1-10 (flat) · forward · 3 · 6 | 3.69 | 28.6 | 5.14 | 0 | 0 | 0 | 0 | 1 | 21 | 0.0194 | no |
| Q4T1-10 (flat) · forward · 5 · 0 | 2.31 | 49.2 | 1.54 | 0 | 0 | 0 | 0 | 1 | 12.1 | 0 | no |
| Q4T1-10 (flat) · forward · 5 · 6 | 2.4 | 48.8 | 5.14 | 0 | 0 | 0 | 0 | 1 | 21 | 0.0194 | no |
| Q4T1-10 (flat) · back · 1.5 · 0 | 0.694 | 154 | 6.42 | 0 | 0 | 0 | 0 | 1 | 9.87 | 0 | no |
| Q4T1-10 (flat) · back · 1.5 · 6 | 0.694 | 152 | 6.42 | 0 | 0 | 0 | 0 | 1 | 9.87 | 0 | no |
| Q4T1-10 (flat) · back · 3 · 0 | 0.695 | 180 | 3.35 | 0 | 0 | 0 | 0 | 1 | 7.6 | 0 | no |
| Q4T1-10 (flat) · back · 3 · 6 | 0.695 | 178 | 3.35 | 0 | 0 | 0 | 0 | 1 | 7.6 | 0 | no |
| Q4T1-10 (flat) · back · 5 · 0 | 0.696 | 211 | 1.59 | 0 | 0 | 0 | 0 | 1 | 6.46 | 0 | no |
| Q4T1-10 (flat) · back · 5 · 6 | 0.696 | 209 | 1.59 | 0 | 0 | 0 | 0 | 1 | 6.46 | 0 | no |


## verdict

|  | passes | of | failures |
|---|---|---|---|
| Q4T1-10 (flat) | 0 | 12 | ['forward 1.5 s, headwind 0 m/s: fails holds height', 'forward 1.5 s, headwind 6 m/s: f… |
| Q4T1N-10 | 6 | 12 | ['back 1.5 s, headwind 0 m/s: fails holds height', 'back 1.5 s, headwind 6 m/s: fails h… |
| Q4T1N-12 | 6 | 12 | ['back 1.5 s, headwind 0 m/s: fails holds height', 'back 1.5 s, headwind 6 m/s: fails h… |
| Q4T1P-10 | 6 | 12 | ['back 1.5 s, headwind 0 m/s: fails holds height', 'back 1.5 s, headwind 6 m/s: fails h… |


## pull-up remedies

|  | height loss [m] | height gain [m] | pitch error max [deg] | time stalled, free stream [s] | α blown max while stalled [deg] | moment shortfall [s] | least pitch margin | peak rotor thrust / T_max | elevon max [deg] | roll demand / authority | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 s schedule | 0.676 | 44.4 | 4.89 | 0 | 0 | 0 | 0.489 | 0.523 | 12.8 | 0 | no |
| 5 s schedule | 0.676 | 52 | 2.91 | 0 | 0 | 0 | 0.56 | 0.455 | 11 | 0 | no |
| 3 s, descent 3 m/s into it | 0.00595 | 45.3 | 5.02 | 0 | 0 | 0 | 0.47 | 0.541 | 13.2 | 0.0561 | no |
| 5 s, descent 3 m/s into it | 0.00595 | 52.8 | 3.01 | 0 | 0 | 0 | 0.546 | 0.469 | 11.3 | 0.0561 | no |
| 3 s, entered at 12 m/s | 0.503 | 34.4 | 4.89 | 0 | 0 | 0 | 0.519 | 0.493 | 12 | 0 | no |


## the fall out of the hover

|  | height loss [m] | height gain [m] | pitch error max [deg] | α free-stream max in transition [deg] | time stalled, free stream [s] | α blown max while stalled [deg] | moment shortfall [s] | collective given way [s] | least pitch margin | peak rotor thrust / T_max | elevon max [deg] | roll demand / authority | roll demand / rotors' authority | speed at the end [m/s] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Q4T1N-10, 1.5 s | 1.03 | 0.0126 | 9.54 | 40.2 | 0.248 | 7.03 | 0 | 6.19 | 0.493 | 1 | 12.6 | 0.0064 | 0.0943 | 29.1 |
| Q4T1N-10, 3.0 s | 0.686 | 0.295 | 4.59 | 49.7 | 0.844 | 11.4 | 0 | 6 | 0.532 | 0.65 | 11.7 | 0.0178 | 0.178 | 29.6 |
| Q4T1P-10, 1.5 s | 1.08 | 0.011 | 9.25 | 40 | 0.244 | 6.34 | 0 | 6.18 | 0.563 | 1 | 10.9 | 0.00636 | 0.0951 | 27.9 |
| Q4T1P-10, 3.0 s | 0.63 | 0.274 | 3.94 | 49.5 | 0.838 | 9.82 | 0 | 6 | 0.603 | 0.586 | 9.9 | 0.0185 | 0.186 | 28.3 |
| Q4T1N-12, 1.5 s | 1.11 | 0.00818 | 9.76 | 39.7 | 0.252 | 6.86 | 0 | 6.19 | 0.542 | 1 | 11.4 | 0.00412 | 0.095 | 27.8 |
| Q4T1N-12, 3.0 s | 0.794 | 0.246 | 4.92 | 49.4 | 0.894 | 11.3 | 0 | 6 | 0.531 | 0.661 | 11.7 | 0.0121 | 0.176 | 28.3 |
| Q4T1-10 (flat), 1.5 s | 3.81 | 17.5 | 8.72 | 13.7 | 0 | 0 | 0 | 8.99 | 0 | 1 | 18.5 | 0.0278 | ∞ | 38.5 |
| Q4T1-10 (flat), 3.0 s | 3.69 | 28.4 | 4.35 | 4.1 | 0 | 0 | 0 | 10.5 | 0 | 1 | 12.7 | 0 | 0 | 40.9 |

*9 more columns not shown: distance [m], energy [Wh], duration [s], hover reached at [s], holds height, holds attitude, roll holds, diverged at [s]…*


## against Kestrel-4Q-2 and between the variants

|  | mass [kg] | wing area [m²] | span [m] | CLα [1/rad] | induced k | Cd0 | CL max (aircraft) | stall speed [m/s] | static margin [c] | front share at trim | L/D at 16 m/s | power at 16 m/s [W] | endurance best [min] | at [m/s] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Q4T1N-10 | 1.75 | 0.28 | 1 | 4.23 | 0.0607 | 0.0201 | 0.978 | 10.1 | 0.3 | 0.667 | 12.7 | 58.8 | 52.7 | 11 |
| Q4T1P-10 | 1.75 | 0.28 | 1 | 4.23 | 0.0607 | 0.0201 | 1.01 | 9.97 | 0.3 | 0.647 | 12.6 | 59.1 | 53 | 10 |
| Q4T1N-12 | 1.81 | 0.336 | 1.2 | 4.43 | 0.0526 | 0.0184 | 0.976 | 9.39 | 0.3 | 0.669 | 13 | 59.3 | 60 | 10 |
| Q4T1-10 (flat) | 1.77 | 0.28 | 1 | 3.65 | 0.0833 | 0.0186 | 0.9 | 10.6 | 0.21 | 0.725 | 10.9 | 67.1 | 46.5 | 12 |
| Kestrel-4Q-2 (notebook 43) | 1.65 | 0.28 | 1 | — | — | 0.0257 | — | — | 0.0916 | — | — | 59.1 | — | — |

*9 more columns not shown: elevator trim at 16 [deg], ground roll [m], hover pitch [rad/s²], hover heading [rad/s²], spar reserve, transitions passed, fall out of the hover, 3 s [m], balloon into the hover, 3 s [m]…*


## drag vs speed

|  | V stall [m/s] | V of least drag [m/s] | least drag [N] | best L/D | drag at 16 m/s [N] | parasite at 16 [N] | induced at 16 [N] | power at 16 m/s [W] | least power [W] | V of least power [m/s] | V max, thrust = drag [m/s] | full thrust at 16 m/s [N] | thrust / drag at 16 m/s | power at V max [W] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Kestrel-4Q-2 (biplane, faired posts) | 10.8 | 12.5 | 1.28 | 12.7 | 1.43 | 1.03 | 0.398 | 61.8 | 42.3 | 11 | 33.5 | 32.4 | 22.7 | 1.49e+03 |
| Q4T1N-10 | 10.1 | 13.5 | 1.27 | 13.6 | 1.34 | 0.883 | 0.454 | 58.3 | 42.2 | 10.5 | — | 41.6 | 31.2 | — |
| Q4T1P-10 | 9.97 | 13.5 | 1.27 | 13.6 | 1.34 | 0.883 | 0.454 | 58.3 | 42.1 | 10 | — | 41.6 | 31.2 | — |
| Q4T1N-12 | 9.39 | 12.5 | 1.16 | 15.3 | 1.32 | 0.967 | 0.348 | 57.6 | 37.4 | 9.5 | — | 41.6 | 31.7 | — |


## runtime [s]

| key | value |
|---|---|
| total | 428 |

