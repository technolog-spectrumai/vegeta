# Kestrel benchmark — the family on one table and the mission scores (from notebooks 43, 45, 46)

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

