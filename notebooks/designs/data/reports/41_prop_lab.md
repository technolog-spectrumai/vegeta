# prop_lab — notebook 41, the 9x6 alone

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

