# Kestrel — notebook 39, the CFD and FEA campaign

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

