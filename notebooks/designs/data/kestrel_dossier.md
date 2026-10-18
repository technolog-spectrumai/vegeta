# Kestrel — the design dossier

*stitched 2026-10-18 03:11 from 8 report(s)*

| report | lines | exported |
|---|---|---|
| 38_prop_lab | 172 | 38_prop_lab — exported 2026-10-16 12:30 |
| 39_kestrel_tests | 164 | 39_kestrel_tests — exported 2026-10-16 12:30 |
| 40_kestrel | 917 | 40_kestrel — exported 2026-10-16 12:22 |
| 41_xwing | 172 | 41_xwing — exported 2026-10-16 12:34 |
| 43_kestrel_study | 467 | 43_kestrel_study — exported 2026-10-17 09:37 |
| 46_kestrel_tandem | 256 | 46_kestrel_tandem — exported 2026-10-17 22:33 |
| 49_mini_raptor | 192 | 49_mini_raptor — exported 2026-10-18 03:11 |
| 47_kestrel_benchmark | 133 | 47_kestrel_benchmark — exported 2026-10-18 01:49 |


---

## prop_lab — notebook 38, the 9x6 alone

*38_prop_lab — exported 2026-10-16 12:30*

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
| meta | {'source': '38_prop_lab', 'quality': 'screening', 'date': '2026-10-16', 'propeller': "9… |


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
| one_rotor_hover | output/38_prop_lab/sound/one_rotor_hover.wav | 3 | 38.7 | 0.0173 | 38.8 | 37.1 | — | — |
| one_rotor_cruise_16ms | output/38_prop_lab/sound/one_rotor_cruise_16ms.wav | 3 | 37.9 | 0.0173 | 38 | 37.3 | — | — |
| one_rotor_transition_8ms_45deg | output/38_prop_lab/sound/one_rotor_transition_8ms_45deg.wav | 3 | 44.1 | 0.0173 | — | — | — | — |
| four_rotors_hover | output/38_prop_lab/sound/four_rotors_hover.wav | 3 | 44.2 | 0.0173 | 44.8 | — | [6759.653306224799, 6659.007687140006, 6659.007687140006, 6759.653306224799] | [3.3548539694931208] |
| transition_hover_to_cruise | output/38_prop_lab/sound/transition_hover_to_cruise.wav | 6 | 43.3 | 0.0173 | — | — | — | — |


## checks

| check | value | target | holds | note |
|---|---|---|---|---|
| hover cases solved | 0 | ≥ 2 | ✘ |  |
| axial cases solved | 0 | ≥ 5 | ✘ |  |
| oblique cases solved | 0 | ≥ 4 | ✘ |  |
| section fit: rms log ratio after | nan | ≤ 0.1 | ✘ | thrust and torque within ~10 % together |
| cl scale within | 0 | ≤ 0.3 | ✔ | further than this and the generic section is the wrong model, not a scaled one |
| cn factor within | 0 | ≤ 0.5 | ✔ |  |



---

## Kestrel — notebook 39, the CFD and FEA campaign

*39_kestrel_tests — exported 2026-10-16 12:30*

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
| meta | {'source': '38_prop_lab', 'quality': 'screening', 'date': '2026-10-16', 'propeller': "9… |


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
| 4Q-1 | 10 | 2 | 12 | 0 | 24 | 0 | 10 | 14 | 40_kestrel run on an OpenFOAM machine (uploaded notebook), Part I points, screening qua… | screening | 2026-10-16 |
| 4Q-2 | 0 | 0 | 0 | 29 | 29 | 0 | 0 | 16 | 39_kestrel_tests | screening | 2026-10-16 |
| sections | 0 | 0 | 0 | 8 | — | — | — | — | 39_kestrel_tests | screening | 2026-10-16 |


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

## Kestrel-4Q — notebook 40, the design loop (Parts I and II)

*40_kestrel — exported 2026-10-16 12:22*

| key | value |
|---|---|
| mass [kg] | 1.69 |
| 4Q-2 mass [kg] | 1.65 |
| CFD | NOT RUN here |
| FEA | NOT RUN (no CalculiX here; notebook 39) |


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
| 3 — CFD plugged in | the polar read against the CFD (sweep correction, 10 solved cases) | 40_kestrel run on an OpenFOAM machine (uploaded notebook), Part I points, screening qua… | 1.69 | 0.164 | 2.89 | 13.3 | 0.0532 | 4.96 | 110 | 21.3 | 12 of 12 | 5 / 7 | CL max below 20°, CFD / the model's polar peak, blown cases solved |
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
| CL max below 20° on the sweep's angles, CFD / model | 0.959 | ≥ 0.8 | ✔ | a 5° grid: the peak itself waits for the fine sweep (notebook 39) |
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
| CFD slab cases of the finalists solved | 0 | ≥ 1 | ✘ | NOT RUN here: notebook 39 |


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
| FEA cases solved | 0 | ≥ 9 | ✘ | CalculiX: notebook 39 on a machine that has it |


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
| FEA (Talos): frame x / posts, joint | NOT RUN (no CalculiX here; notebook 39) |
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
| X-wing | its own notebook, 41_xwing |


## prop_lab calibration

| key | value |
|---|---|
| cl_scale | 1 |
| cd_scale | 1 |
| cn_factor | 1 |
| on_file | no |
| meta | {'source': '38_prop_lab', 'quality': 'screening', 'date': '2026-10-16', 'propeller': "9… |



---

## X-wing — notebook 41 (X-A and X-B against Kestrel-4Q and 4Q-2)

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



---

## Kestrel — the design study (notebook 43)

*43_kestrel_study — exported 2026-10-17 09:37*

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
| 1 prerequisites | 2.9 |
| 2 geometry | 12.6 |
| 3 mass | 0 |
| 4 propulsion | 0.4 |
| 5 aerodynamics | 0.4 |
| 6 trim | 0.5 |
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



---

## Kestrel-Q4T — the tandem-wing Kestrel (Q4T1N-10 against Q4T1P-10, Q4T1N-12 and Kestrel-4Q-2)

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



---

## Mini raptors — Apus Mini, Alcotan, Gavilan: three first-guess designs on a 5-inch racing stack

*49_mini_raptor — exported 2026-10-18 03:11*

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
| layout | biplane | cruciform | biplane |
| title | economy: the quad's speed and manoeuvre for a third of the power | the chaser: sustained speed on 6S and the 5x5, the motors at the tips of a + | tight spaces: the racing frame's motor square in a short cell, the wing as a brake |
| drive | 4S 5x4.3 | 6S 5x5 | 4S 5x4.3 light |
| pack | 4S 1500 | 6S 1300 | 4S 1300 |
| span | 0.5 | 0.42 | 0.31 |
| chord | 0.085 | 0.075 | 0.1 |
| S_ref | 0.085 | 0.0315 | 0.062 |
| AR | 2.94 | 5.6 | 1.55 |
| motor_y | 0.171 | 0.14 | 0.0765 |
| motor_z | 0.08 | 0.14 | 0.0775 |
| disc_gap_span | 0.216 | 0.071 | 0.026 |
| disc_gap_vertical | 0.033 | 0.071 | 0.028 |
| disc_to_pod | 0.083 | 0.0515 | ∞ |
| span_over_gate | 0.333 | 0.28 | 0.207 |
| Re_20 | 1.16e+05 | 1.03e+05 | 1.37e+05 |
| cl_max | 1 | 0.95 | 1.1 |
| n_struct | 8 | 10 | 8 |
| section | thin cambered section (3% camber, 10 % thick); the elevons' reflex trims it | thin cambered section (2% camber, 10 % thick); the elevons' reflex trims it | thin cambered section (4% camber, 10 % thick); the elevons' reflex trims it |
| rules | all hold | all hold | all hold |


## comparison

|  | 5-inch racing quad | Apus Mini | Alcotan | Gavilan |
|---|---|---|---|---|
| layout | quad | biplane | cruciform | biplane |
| mass [kg] | 0.5 | 0.564 | 0.572 | 0.518 |
| span / gate | 0.167 | 0.333 | 0.28 | 0.207 |
| top speed [km/h] | 133 | 220 | 295 | 223 |
| power at top speed [W] | 827 | 1.1e+03 | 1.6e+03 | 1.22e+03 |
| full-throttle endurance at top speed [s] | 77.3 | 58.1 | 51.9 | 45.3 |
| sustained power [W] | 666 | 666 | 866 | 577 |
| sustained top speed [km/h] | 122 | 191 | 263 | 187 |
| power at 20 m/s [W] | 169 | 48.9 | 34.5 | 50.2 |
| Wh/km at 20 m/s | 2.34 | 0.679 | 0.479 | 0.698 |
| least Wh/km | 2.01 | 0.537 | 0.477 | 0.629 |
| range at 20 m/s [km] | 7.57 | 26.1 | 48.2 | 22.1 |
| best range [km] | 8.83 | 33.1 | 48.4 | 24.5 |
| at [m/s] | 15 | 14 | 19 | 16 |
| best endurance [min] | 13.7 | 46.5 | 44.5 | 29.6 |
| stall speed [m/s] | 0 | 10.3 | 17.5 | 11 |
| hover thrust / weight | 10.5 | 9.33 | 11.5 | 10.2 |
| hover power [W] | 77.6 | 89.6 | 100 | 80.9 |
| hover endurance [min] | 13.7 | 11.9 | 13.8 | 11.4 |
| n at 20 m/s (banked) | 2.92 | 3.76 | 1.31 | 3.29 |
| turn radius at 20 m/s (banked) [m] | 14.8 | 11.2 | 48.6 | 13 |
| turn radius at 40 m/s (banked) [m] | 59.4 | 20.5 | 31.8 | 20.5 |
| least banked turn radius [m] | 0 | 10.9 | 31.4 | 12.5 |
| brake deceleration at 20 m/s [g] | 3.55 | 6 | 4.79 | 4.8 |
| brake 20→4 m/s distance [m] | 6.18 | 5.02 | 5.2 | 5.4 |
| brake 20→4 m/s time [s] | 0.535 | 0.455 | 0.465 | 0.48 |
| hover pitch-over 90° [s] | 0.0844 | 0.127 | 0.0909 | 0.124 |
| pitch acceleration in the hover [rad/s²] | 1.76e+03 | 783 | 1.52e+03 | 815 |
| thrust / weight at 1.2 Vs | — | 8.17 | 9.43 | 8.82 |
| pull-up speed [m/s] | — | 15 | 25 | 16 |
| pull-up radius into the hover [m] | — | 20.6 | 61.2 | 23.6 |
| transitions | none needed | both with thrust to spare | both with thrust to spare | both with thrust to spare |
| energy [Wh] | 17.8 | 17.8 | 23.1 | 15.4 |


## targets

|  | target | value | met |
|---|---|---|---|
| Apus Mini: less energy at the quad's 20 m/s | Wh/km at 20 m/s < half the quad's | 0.68 vs 2.34 | yes |
| Apus Mini: comparable speed | sustained top speed ≥ the quad's | 191 vs 122 km/h | yes |
| Apus Mini: comparable manoeuvre | banked turn radius at 20 m/s ≤ the quad's at 45° bank (41 m) | 11 m | yes |
| Alcotan: faster sustained | sustained top speed (pack 30C, motors 22 A) ≥ 1.25 × the quad's | 263 vs 122 km/h | yes |
| Alcotan: burst speed | thrust-limited top speed ≥ 1.25 × the quad's | 295 vs 133 km/h (52 s of pack) | yes |
| Alcotan: a chase at 40 m/s | endurance at 40 m/s ≥ the quad's at its top speed, with a turn | 8.3 min, r = 32 m | yes |
| Gavilan: span for a hangar | span ≤ a fifth of a gate (0.30 m) | 0.31 m | yes |
| Gavilan: the quad's hover agility | pitch acceleration in the hover ≥ the quad's | 815 vs 1764 rad/s² | no |
| Gavilan: brakes harder than the quad | brake distance 20→4 m/s ≤ the quad's | 5.4 vs 6.2 m | yes |
| Gavilan: flies slower than the quad can | stall speed ≤ 12 m/s (a hangar's straights) | 11.0 m/s | yes |


## manoeuvre

|  | 5-inch racing quad | Apus Mini | Alcotan | Gavilan |
|---|---|---|---|---|
| n at 20 m/s (banked) | 2.92 | 3.76 | 1.31 | 3.29 |
| turn radius at 20 m/s (banked) [m] | 14.8 | 11.2 | 48.6 | 13 |
| turn radius at 40 m/s (banked) [m] | 59.4 | 20.5 | 31.8 | 20.5 |
| least banked turn radius [m] | 0 | 10.9 | 31.4 | 12.5 |
| pitch acceleration in the hover [rad/s²] | 1.76e+03 | 783 | 1.52e+03 | 815 |
| hover pitch-over 90° [s] | 0.0844 | 0.127 | 0.0909 | 0.124 |
| brake deceleration at 20 m/s [g] | 3.55 | 6 | 4.79 | 4.8 |
| brake 20→4 m/s distance [m] | 6.18 | 5.02 | 5.2 | 5.4 |
| brake 20→4 m/s time [s] | 0.535 | 0.455 | 0.465 | 0.48 |
| stall speed [m/s] | 0 | 10.3 | 17.5 | 11 |
| thrust / weight at 1.2 Vs | — | 8.17 | 9.43 | 8.82 |
| pull-up speed [m/s] | — | 15 | 25 | 16 |
| pull-up radius into the hover [m] | — | 20.6 | 61.2 | 23.6 |
| transitions | none needed | both with thrust to spare | both with thrust to spare | both with thrust to spare |
| hover yaw 90° [s] (quad: torque) | 0.874 | — | — | — |
| hover roll 90° [s] | — | 0.142 | 0.0992 | 0.126 |


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



---

## Kestrel benchmark — the family on one table and the mission scores (from notebooks 43, 45, 46)

*47_kestrel_benchmark — exported 2026-10-18 01:49*

## benchmark

|  | family | notebook | mass [kg] | wing area [m²] | span [m] | Cd0 | induced k | CL max | stall speed [m/s] | L/D at 16 m/s | power at 16 m/s [W] | best endurance [min] | hover T/W | hover pitch [rad/s²] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Kestrel-4Q-1 | biplane (tail-sitter) | 43 | 1.69 | 0.28 | 1 | 0.0665 | 0.0599 | 0.81 | 10.9 | 4.96 | 110 | 40.2 | 2.89 | — |
| Kestrel-4Q-2 | biplane (tail-sitter) | 43 | 1.65 | 0.28 | 1 | 0.0257 | 0.0599 | 1.03 | 9.56 | 10.6 | 59.1 | 57 | 2.96 | 88.4 |
| Kestrel-4Q-3 | biplane (tail-sitter) | 43 | 1.65 | 0.28 | 1 | 0.0257 | 0.0599 | 1.03 | 9.56 | 10.6 | 59.3 | 56.8 | 2.96 | — |
| Kestrel-X-A | X-wing (tail-sitter) | 45 | 1.59 | 0.28 | 0.785 | 0.0344 | 0.145 | — | 12.6 | — | 87.6 | 29.1 | — | 67 |
| Kestrel-X-B | X-wing (tail-sitter) | 45 | 1.88 | 0.56 | 1.07 | 0.0303 | 0.144 | — | 9.68 | — | 110 | 33.1 | — | 50.6 |
| Kestrel-Q4T1N-10 | tandem (horizontal start) | 46 | 1.75 | 0.28 | 1 | 0.0201 | 0.0607 | 0.978 | 10.1 | 12.7 | 58.8 | 52.7 | — | 23 |
| Kestrel-Q4T1P-10 | tandem (horizontal start) | 46 | 1.75 | 0.28 | 1 | 0.0201 | 0.0607 | 1.01 | 9.97 | 12.6 | 59.1 | 53 | — | 23 |
| Kestrel-Q4T1N-12 | tandem (horizontal start) | 46 | 1.81 | 0.336 | 1.2 | 0.0184 | 0.0526 | 0.976 | 9.39 | 13 | 59.3 | 60 | — | 22 |
| Kestrel-Q4T1-10 (flat) | tandem (horizontal start) | 46 | 1.77 | 0.28 | 1 | 0.0186 | 0.0833 | 0.9 | 10.6 | 10.9 | 67.1 | 46.5 | — | 11.3 |

*6 more columns not shown: hover heading [rad/s²], transitions passed, bracing, ground handling, best L/D, balloon into the hover [m]*


## ranks

|  | mass [kg] | Cd0 | induced k | CL max | stall speed [m/s] | L/D at 16 m/s | power at 16 m/s [W] | best endurance [min] | hover T/W | hover pitch [rad/s²] | hover heading [rad/s²] | measures ranked | mean rank |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Kestrel-4Q-2 | 2 | 5 | 2 | 1 | 2 | 5 | 3 | 2 | 1 | 1 | 1 | 11 | 2.27 |
| Kestrel-4Q-3 | 2 | 5 | 2 | 1 | 2 | 5 | 4 | 3 | 1 | — | — | 9 | 2.78 |
| Kestrel-Q4T1N-12 | 8 | 1 | 1 | 5 | 1 | 1 | 5 | 1 | — | 6 | 5 | 10 | 3.4 |
| Kestrel-Q4T1N-10 | 5 | 3 | 5 | 4 | 6 | 2 | 1 | 5 | — | 4 | 2 | 10 | 3.7 |
| Kestrel-Q4T1P-10 | 5 | 3 | 5 | 3 | 5 | 3 | 2 | 4 | — | 5 | 3 | 10 | 3.8 |
| Kestrel-Q4T1-10 (flat) | 7 | 2 | 7 | 6 | 7 | 4 | 6 | 6 | — | 7 | — | 9 | 5.78 |
| Kestrel-4Q-1 | 4 | 9 | 2 | 7 | 8 | 7 | 8 | 7 | 3 | — | — | 9 | 6.11 |
| Kestrel-X-A | 1 | 8 | 9 | — | 9 | — | 7 | 9 | — | 2 | 4 | 8 | 6.12 |
| Kestrel-X-B | 9 | 7 | 8 | — | 4 | — | 9 | 8 | — | 3 | 6 | 8 | 6.75 |


## anchors

|  | measure | 1 point at | 10 points at | why |
|---|---|---|---|---|
| cruise economy | power at 16 m/s [W] | 120 | 50 | electrical power in level flight at the 16 m/s survey speed |
| endurance | best endurance [min] | 20 | 60 | best endurance on 80 % of the 4S 3300 pack |
| slow flight | stall speed [m/s] | 14 | 8 | the stall speed: the launch, the landing and the entry into the hover |
| clean airframe | Cd0 | 0.07 | 0.018 | zero-lift drag on the wing area: struts, pods, junctions |
| hover authority | hover pitch [rad/s²] | 0 | 90 | pitch acceleration available in the hover (the weakest axis of a quad tail-sitter) |
| transitions | transition pass fraction | 0 | 1 | share of the modelled hover↔cruise transitions that pass as is |
| structure | structure score | 0 | 1 | no bracing and a spar reserve ≥ 2 score full; posts 0.6, an X truss 0.3; a reserve belo… |
| ground handling | ground score | 0 | 1 | tail-sitters: tip-over angle and the wind that topples them; the tandem: skids and a sh… |
| payload | payload score | 0 | 1 | does the medical capsule (72×72×160 mm) fit the pod near the CG |
| evidence | evidence score | 0 | 1 | how much of the row rests on solver results (CFD, FEA) rather than the model |
| confined launch and recovery | launch score | 0 | 1 | can it leave and come back to a vessel's deck or a shore without a strip: vertical take… |


## mission weights

|  | mountain survey (cruise) | medical delivery (hover at the site) | pursuit (fast response) | field trials (build it first) | 3-civilian-missions: maritime rescue (primary) | 3-civilian-missions: urgent industrial logistics | 3-civilian-missions: emergency medical delivery |
|---|---|---|---|---|---|---|---|
| cruise economy | 0.25 | 0.05 | 0.15 | 0.1 | 0.15 | 0.15 | 0.15 |
| endurance | 0.25 | 0.05 | 0.1 | 0.05 | 0.25 | 0.15 | 0.1 |
| slow flight | 0.1 | 0.1 | 0.05 | 0.05 | 0.1 | 0.05 | 0.05 |
| clean airframe | 0.05 | 0 | 0.2 | 0 | 0 | 0.05 | 0 |
| hover authority | 0.05 | 0.2 | 0.1 | 0.1 | 0.1 | 0.05 | 0.1 |
| transitions | 0.1 | 0.2 | 0.15 | 0.15 | 0.1 | 0.1 | 0.1 |
| structure | 0.05 | 0.05 | 0.1 | 0.25 | 0 | 0.05 | 0.05 |
| ground handling | 0.05 | 0.1 | 0.1 | 0.1 | 0 | 0 | 0 |
| payload | 0.05 | 0.2 | 0 | 0 | 0.05 | 0.2 | 0.25 |
| evidence | 0.05 | 0.05 | 0.05 | 0.2 | 0.05 | 0.05 | 0.05 |
| confined launch and recovery | 0 | 0 | 0 | 0 | 0.2 | 0.15 | 0.15 |


## scores 1-10

|  | cruise economy | endurance | slow flight | clean airframe | hover authority | transitions | structure | ground handling | payload | evidence | confined launch and recovery |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Kestrel-4Q-1 | 2.26 | 5.54 | 5.63 | 1.6 | — | 10 | 3.7 | 4.6 | 10 | 3.7 | 7.75 |
| Kestrel-4Q-2 | 8.83 | 9.33 | 7.66 | 8.66 | 9.84 | 7.43 | 6.4 | 4.6 | 10 | 3.7 | 7.75 |
| Kestrel-4Q-3 | 8.8 | 9.28 | 7.66 | 8.66 | — | 6.4 | 6.4 | 4.6 | 10 | 3.7 | 7.75 |
| Kestrel-X-A | 5.16 | 3.04 | 3.12 | 7.16 | 7.7 | 10 | 10 | 8.65 | 3.7 | 2.35 | 9.55 |
| Kestrel-X-B | 2.23 | 3.94 | 7.48 | 7.87 | 6.06 | 8.5 | 10 | 8.65 | 3.7 | 2.35 | 9.55 |
| Kestrel-Q4T1N-10 | 8.87 | 8.35 | 6.81 | 9.64 | 3.3 | 5.5 | 10 | 7.3 | 8.2 | 2.35 | 3.25 |
| Kestrel-Q4T1P-10 | 8.84 | 8.43 | 7.04 | 9.64 | 3.3 | 5.5 | 10 | 7.3 | 8.2 | 2.35 | 3.25 |
| Kestrel-Q4T1N-12 | 8.8 | 10 | 7.91 | 9.94 | 3.2 | 5.5 | 10 | 7.3 | 8.2 | 2.35 | 3.25 |
| Kestrel-Q4T1-10 (flat) | 7.8 | 6.96 | 6.09 | 9.89 | 2.13 | 1 | 10 | 7.3 | 8.2 | 2.35 | 3.25 |


## weighted by mission

|  | mountain survey (cruise) | mountain survey (cruise) · coverage | medical delivery (hover at the site) | medical delivery (hover at the site) · coverage | pursuit (fast response) | pursuit (fast response) · coverage | field trials (build it first) | field trials (build it first) · coverage | 3-civilian-missions: maritime rescue (primary) | 3-civilian-missions: maritime rescue (primary) · coverage | 3-civilian-missions: urgent industrial logistics | 3-civilian-missions: urgent industrial logistics · coverage | 3-civilian-missions: emergency medical delivery | 3-civilian-missions: emergency medical delivery · coverage |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Kestrel-4Q-1 | 4.94 | 0.95 | 7.23 | 0.8 | 4.45 | 0.9 | 4.9 | 0.9 | 6.14 | 0.9 | 6.38 | 0.95 | 6.9 | 0.9 |
| Kestrel-4Q-2 | 8.21 | 1 | 8.09 | 1 | 7.76 | 1 | 6.63 | 1 | 8.39 | 1 | 8.44 | 1 | 8.53 | 1 |
| Kestrel-4Q-3 | 8 | 0.95 | 7.39 | 0.8 | 7.34 | 0.9 | 6.1 | 0.9 | 8.09 | 0.9 | 8.25 | 0.95 | 8.27 | 0.9 |
| Kestrel-X-A | 5.34 | 1 | 6.48 | 1 | 6.92 | 1 | 6.93 | 1 | 5.83 | 1 | 5.92 | 1 | 5.98 | 1 |
| Kestrel-X-B | 5.07 | 1 | 6.19 | 1 | 6.54 | 1 | 6.51 | 1 | 5.74 | 1 | 5.64 | 1 | 5.53 | 1 |
| Kestrel-Q4T1N-10 | 7.58 | 1 | 6.29 | 1 | 7.44 | 1 | 6.5 | 1 | 6.16 | 1 | 6.87 | 1 | 6.54 | 1 |
| Kestrel-Q4T1P-10 | 7.61 | 1 | 6.31 | 1 | 7.45 | 1 | 6.51 | 1 | 6.19 | 1 | 6.88 | 1 | 6.55 | 1 |
| Kestrel-Q4T1N-12 | 8.09 | 1 | 6.46 | 1 | 7.7 | 1 | 6.62 | 1 | 6.66 | 1 | 7.17 | 1 | 6.74 | 1 |
| Kestrel-Q4T1-10 (flat) | 6.39 | 1 | 4.96 | 1 | 6.36 | 1 | 5.5 | 1 | 5.01 | 1 | 5.97 | 1 | 5.64 | 1 |


## 3-civilian-missions

|  | 3-civilian-missions |   maritime rescue (primary) |   urgent industrial logistics |   emergency medical delivery |
|---|---|---|---|---|
| Kestrel-4Q-2 | 8.44 | 8.39 | 8.44 | 8.53 |
| Kestrel-4Q-3 | 8.17 | 8.09 | 8.25 | 8.27 |
| Kestrel-Q4T1N-12 | 6.81 | 6.66 | 7.17 | 6.74 |
| Kestrel-Q4T1P-10 | 6.46 | 6.19 | 6.88 | 6.55 |
| Kestrel-Q4T1N-10 | 6.43 | 6.16 | 6.87 | 6.54 |
| Kestrel-4Q-1 | 6.39 | 6.14 | 6.38 | 6.9 |
| Kestrel-X-A | 5.89 | 5.83 | 5.92 | 5.98 |
| Kestrel-X-B | 5.66 | 5.74 | 5.64 | 5.53 |
| Kestrel-Q4T1-10 (flat) | 5.41 | 5.01 | 5.97 | 5.64 |


## 3-civilian-missions definitions

|  | situation | payload | outcome | what the score weighs |
|---|---|---|---|---|
| maritime rescue | a person is reported overboard or in difficulty offshore | compact flotation equipment, search camera, optional thermal camera | locate the person, deliver accessible flotation and help rescuers maintain their position | endurance left after the transit (0.25), cruise economy (0.15), launch and recovery fro… |
| urgent industrial logistics | a vessel, offshore installation or remote facility needs a small part to resume work | protected cargo pod with a spare part, tool or electronic component | deliver the correct item intact, avoiding a longer boat or road journey | the cargo pod (payload 0.20), the transit (cruise 0.15, endurance 0.15), a deck at both… |
| emergency medical delivery | a clinic, vessel or remote team urgently needs medical supplies or sample transport | secured medical container, insulated and temperature-monitored where required | deliver the supplies within their required time and handling conditions | the secured container near the CG (payload 0.25), the transit (cruise 0.15, endurance 0… |


## notes

| key | value |
|---|---|
| 0 | 43: 43_kestrel_study.json exported 2026-10-17T09:37; the hover authority is the main va… |
| 1 | 45: kestrel_x_design.json missing — notebook 41's xwing_design.json (the same model, an… |
| 2 | 45: xwing_design.json; the X's stall speed from its drag-vs-speed table (panel cl max),… |
| 3 | 46: kestrel_tandem_design.json; main variant Q4T1N-10 |

