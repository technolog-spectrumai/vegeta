# ARIGO product lines: KPI definitions (draft 1)

ARIGO-1 hybrid has graduated to **ARIGO-1H**, the family's baseline airframe. Four product lines start from the
ARIGO family. Their names are **mission identities, not airframe configurations**: which airframe, wing, pack and kit
serves each one is what the exploration has to find.

| Line | Bird | Mission | Priorities |
|---|---|---|---|
| **BUTEO** | buzzard: a watchful perch hunter | point defence from a fixed perch or launch station | readiness, weather tolerance, reliable deployment, straightforward recovery |
| **ARPELLA** | harrier: sustained patrol over open terrain | area defence, mission radius up to 10 km | endurance, efficient coverage, dependable communications, return reserves |
| **ASTOR** | goshawk: a powerful, agile pursuit hunter | fast response; arrival time before patrol time | quick deployment, reliable prepared launch, rapid turnaround, predictable recovery |
| **SPARFELL** | sparrowhawk (Breton *sparfell*) | portable field platform: backpack-carried, hand-launched by one person | low mass, compact packing, quick assembly, minimal equipment, rugged handling |

## How to read this

**Requirements are proposed, not agreed.** Every threshold and objective below is a proposal, to be confirmed with
the client.
- **Gate:** pass or fail. A design that fails a gate is not a candidate for that line.
- **Threshold:** the least the client accepts. It scores 0.
- **Objective:** the point where more stops paying. It scores 100.

Scores are linear between threshold and objective and clipped at both ends. A line's score is the mean of its scored
KPIs. Because the scale is absolute, a score moves only when the design changes.

**Evidence: what is shown, and how.** Nothing in this document is flight-demonstrated or test-demonstrated. The
"ARIGO-1H today" column says where each number comes from:

| Mark | Meaning |
|---|---|
| **SIM** | Computed with ARIGO-1H's flight model: MuJoCo 6-DOF or point mass, on screening CFD (RANS, propellers off). It is a model result, not a measurement. |
| **EST** | Computed, but driven by hand-estimated inputs: structure masses ±20 %, efficiencies, the radio link budget, kit masses and times, the sensor's cue range. |
| **PROP** | Depends on a kit or feature that is proposed but not yet designed in CAD. Examples: the spring perch push-off, the charging perch, the parachute, the quick-release wing. |
| **—** | Not yet assessed: no model exists. These need a Monte Carlo study, a test or a trial (see the last section). |

**The configuration behind "ARIGO-1H today."** Each line's starting spec is ARIGO-1H plus that line's kit:

| Line | Starting spec |
|---|---|
| BUTEO | ARIGO-1H + perch kit (spring push-off) + charging perch |
| ARPELLA | ARIGO-1H + long-range radio, hand launch |
| ASTOR | ARIGO-1H + rail + parachute. ARIGO-20 on a rail with a parachute is shown beside it. |
| SPARFELL | ARIGO-1H, two-piece wing, 3S1P pack |

The values are from the current models, in a container test run, not from your own stability run. The status column
reads ✅ when the objective is met, ✔ when the threshold is met, and ✖ when the threshold is not met.

**Common scenario assumptions (EST, proposed).**
- The site's sensor cues a target at 3 km. The alert from cue to launch takes 2 s from a perch or launcher, and 10 s when a person launches by hand.
- A low target flies at 50 m.
- The perch is 15 m high.
- A hand throw is 12 m/s.
- The defended target flies straight at the site. Keep-out is the distance from the site at which the two meet.

---

## BUTEO — point defence from a perch or launch station

**Gate**
- **B-G1.** It launches from its station with no person present, and the launch succeeds in the sim. ARIGO-1H with the perch kit: **pass**, SIM + PROP.

| ID | KPI (measure) | Priority | Threshold | Objective | ARIGO-1H today | Evidence | Status |
|---|---|---|---|---|---|---|---|
| B1 | Cue to safe flight (s): alert + launch to 10 m at 1.3 × stall | readiness | 12 | 3 | 5.0 | SIM + PROP | ✔ |
| B2 | Watch time on station (h), camera and computer on, motor off | readiness | 2 | 24 (continuous) | 7.7 on its pack; 24+ with a charging perch | EST / PROP | ✔ / ✅ |
| B3 | Perch height to spare (m): perch height − height the drop needs | reliable deployment | 0 | 6 | 4.3 with an 8 m/s spring push; 1.1 without | SIM + PROP | ✔ |
| B4 | Launch success: Monte Carlo over steady wind 0–12 m/s with 1.5× gusts, wind from any direction, push speed ±15 %, perch attitude ±5° (%) | reliable deployment | 95 | 99.5 | 75 on a fixed perch; 100 on a turning perch | SIM (16 draws, ±11 %) + PROP | ✖ / ✅ |
| B5 | Vertical gust that stalls it at loiter speed (m/s), quasi-steady (conservative). **Reported, not scored** (decision 2026-11-20): B6's turbulence model measures gust tolerance in flight. | weather | (3) | (6) | 2.5 | SIM | info |
| B6 | Operating wind: the strongest steady wind (gusts 1.5×) at which launch and recovery both succeed ≥ 95 % (m/s) | weather | 12 / 18 | 15 / 22 | 4.8 / 7.2 on a fixed perch; 12.8 / 19.2 on a turning perch | SIM (16 launch and 4 landing draws per wind) + PROP | ✖ / ✔ |
| B7 | Wind that lifts it off the perch untethered (m/s). A perch latch makes this a design requirement instead. | weather | 15 | 30 | 31.7 | EST | ✅ |
| B8 | Keep-out vs a 30 m/s drone cued at 3 km (m) | effect | 1000 | 2000 | 1748 | SIM + EST | ✔ |
| B9 | Urban PN chase: time to close to 5 m (s) | effect | 15 | 3 | 10.5 | SIM | ✔ |
| B10 | Touchdown energy (J), belly landing on the approach path | recovery | 300 | 40 | 174 | SIM | ✔ |
| B11 | Miss from the landing aim point (m), calm air | recovery | 30 | 3 | 4.5 | SIM | ✔ |
| B12 | Landed to ready again (s): pack, check, back on the perch | recovery | 600 | 120 | 120 | EST | ✅ |

**Reading.** ARIGO-1H meets most BUTEO thresholds today. **The station matters as much as the aircraft.** On a
fixed perch, with the wind free to come from any direction, the drop fails in tailwinds (it hits the ground) and in
crosswind gusts (it rolls past 60°). Launch success is about 75 % and the operating wind only 4.8 m/s. A
**perch that turns into the wind** (the `turning_station` kit, ground equipment) gives 100 % launch success and a
12.8 m/s operating wind, and recovery becomes the limit at 16 m/s. The other clear gap is **gust tolerance (B5)**: at 18 m/s loiter,
a 2.5 m/s vertical gust reaches the stall angle in the quasi-steady bound. The levers are:
- a faster loiter, which trades away endurance (a minor priority for BUTEO);
- a higher stall angle or a lower wing loading;
- an autopilot gust margin.

B4 and B6 are now modelled (`designs/raptors_robust.py`). The numbers here come from small samples; the run scores them with more draws.

---

## ARPELLA — area patrol, out to 10 km

**Gate**
- **A-G1.** It meets a target 10 km out, chases at full power for 60 s, and comes home with at least 15 % of its usable pack, in a 10 m/s steady wind. ARIGO-1H: **pass** (wind limit 21 m/s), EST.

| ID | KPI (measure) | Priority | Threshold | Objective | ARIGO-1H today | Evidence | Status |
|---|---|---|---|---|---|---|---|
| A1 | Endurance at best speed (min) | endurance | 45 | 120 | 74 | EST | ✔ |
| A2 | Pack left after a 10 km mission, calm (%) | return reserves | 15 | 50 | 57.7 | EST | ✅ |
| A3 | Steady wind in which the 10 km mission still returns with 15 % (m/s) | return reserves / weather | 8 | 15 | 21.2 | EST | ✅ |
| A4 | Area covered per sortie (km²): a 60° camera from 100 m, at best-endurance speed | efficient coverage | 4 | 15 | 7.9 | EST | ✔ |
| A5 | Transit: time to 10 km (s) | efficient coverage | 600 | 250 | 197 | SIM | ✅ |
| A6 | Telemetry link margin at 10 km (dB), free space | communications | 10 | 20 | 33.5 | EST (radio values, not the airframe) | ✅ |
| A7 | Link availability along the mission with terrain and clutter (%) | communications | 95 | 99.9 | — | — | — |
| A8 | Open-field PN chase: time to close to 5 m (s) | effect | 20 | 8 | 11.8 | SIM | ✔ |
| A9 | Touchdown energy (J), belly landing | recovery | 300 | 60 | 170 | SIM | ✔ |
| A10 | Operating wind: the top of the band where the hand launch into the wind and the belly landing both succeed ≥ 95 % (m/s) | weather | 10 | 15 | 13.3 | SIM (run 1, 12 launches and 6 landings per wind) | ✔ |
| A11 | Calm air: launch and landing success (%) | weather / dependable | 90 | 99 | 58: a 12 m/s throw is below the 12.4 m/s stall | SIM (run 1) | ✖ |

**Reading.** On paper, ARIGO-1H already over-delivers ARPELLA's 10 km radius: about 14 km going out at dash speed,
about 27 km at economy speed. Its real gaps are **coverage per sortie** and **communications**. Coverage is mostly a
sensor matter (swath and altitude). Communications is an avionics and relay matter, and A7 needs a terrain model.
This line can trade some reserve for a lighter pack or a slower, quieter loiter.

---

## ASTOR — fast response from a prepared launcher

**Gate**
- **S-G1.** It launches from a prepared rail with no hand involved, and the launch succeeds in the sim. ARIGO-1H on a rail: **pass**, SIM + PROP. ARIGO-20 on a rail: **pass**, SIM + PROP.

| ID | KPI (measure) | Priority | Threshold | Objective | ARIGO-1H today | ARIGO-20 today | Evidence |
|---|---|---|---|---|---|---|---|
| S1 | Cue to safe flight from the launcher (s) | quick deployment | 12 | 3 | 4.2 ✔ | 5.0 ✔ | SIM + PROP |
| S2 | Arrival: time to 3 km (s) | arrival | 90 | 35 | 62.7 ✔ | 52.3 ✔ | SIM |
| S3 | Arrival: time to 5 km (s) | arrival | 150 | 60 | 100.7 ✔ | 82.4 ✔ | SIM |
| S4 | Time to 150 m height (s) | arrival | 20 | 6 | 10.8 ✔ | 11.1 ✔ | SIM |
| S5 | Keep-out vs a 50 m/s fixed wing cued at 3 km (m) | effect | 1000 | 2000 | 1342 ✔ | 1452 ✔ | SIM + EST |
| S6 | Can it tail-chase a 50 m/s target (top speed > 1.15 × 50 m/s)? | effect | no | yes | no | yes | SIM |
| S7 | Full power on one pack (min) | arrival / pursuit | 1 | 6 | 5.7 ✔ | 2.8 ✔ | EST |
| S8 | Launch success from the rail: Monte Carlo over steady wind 0–12 m/s with 1.5× gusts, exit speed ±5 %, pitch ±2°, fixed or turning rail (%) | reliable prepared launch | 98 | 99.9 | in the run | in the run | SIM + PROP |
| S9 | Landed to ready again (s): pack + canopy repack | rapid turnaround | 900 | 180 | 360 ✔ | 360 ✔ | EST + PROP |
| S10 | Recovery miss (m), parachute in a 6 m/s wind | predictable recovery | 30 | 3 | 15 ✔ | 15 ✔ | EST + PROP |
| S11 | Touchdown energy under the parachute (J) | predictable recovery | 300 | 40 | 23 ✅ | 32 ✅ | EST + PROP |

**Reading.** On arrival time, ARIGO-20 beats ARIGO-1H by only about 10–20 %, because the sensor cue and the alert
time dominate. Its unique capability is **S6**: it is the only ARIGO that can chase a 50 m/s target from behind. It
pays for that with half the full-power time and a heavier launcher and recovery set. Whether ASTOR needs S6 is a
**client question**: what is the fastest target it must catch from behind?

---

## SPARFELL — one person, a backpack, a throw

**Gates**
- **P-G1.** One person can hand-launch it: mass ≤ 2 kg and the 12 m/s throw succeeds in the sim. ARIGO-1H on 3S1P: **pass**, SIM + EST.
- **P-G2.** No launcher and no recovery equipment are needed. ARIGO-1H: **pass**.

| ID | KPI (measure) | Priority | Threshold | Objective | ARIGO-1H today | Evidence | Status |
|---|---|---|---|---|---|---|---|
| P1 | Take-off mass (kg) | low mass | 1.8 | 0.9 | 1.57 | EST | ✔ |
| P2 | Longest piece to carry (mm) | compact packing | 900 | 500 | 720 (two-piece wing) | EST + PROP | ✔ |
| P3 | Assembly: from the bag to ready to throw (s) | quick assembly | 180 | 30 | 60 | EST + PROP | ✔ |
| P4 | Throw margin: throw speed ÷ stall speed | minimal equipment | 0.9 | 1.3 | 1.03 | SIM + EST | ✔ |
| P5 | Throw to safe flight (s) | quick deployment | 5 | 1.5 | 1.9 | SIM | ✔ |
| P6 | Keep-out vs a 15 m/s quadcopter seen at 1 km, 10 s to throw (m) | effect | 200 | 700 | 601 | SIM + EST | ✔ |
| P7 | Urban PN chase: time to close to 5 m (s) | effect | 15 | 3 | 17.3 | SIM | ✖ |
| P8 | Endurance (min) | effect | 15 | 40 | 43 | EST | ✅ |
| P9 | Touchdown energy (J), belly landing | rugged handling | 200 | 40 | 128 | SIM | ✔ |
| P10 | Propulsor in a belly landing (folds clear = 100, folding strike = 50, fixed strike = 0) | rugged handling | 0 | 100 | 50: the folding propeller reaches 82 mm below the belly | EST (CAD) | ✔ |
| P11 | Survives a 1 m drop onto a corner, and five belly landings without repair | rugged handling | pass | pass | — | — | — |
| P12 | Throw success: Monte Carlo over steady wind 0–12 m/s with 1.5× gusts, thrown into the wind ±30°, 12 ± 2 m/s, pitch ±5° (%) | minimal equipment / rugged | 90 | 99 | 100; its operating wind is about 16 m/s | SIM (16 draws) | ✅ |
| P13 | Calm air: throw and landing success (%) | minimal equipment | 90 | 99 | in run 2 | SIM | — |

**Reading.** ARIGO-1H is a heavy SPARFELL. It is light enough to throw, but 1.57 kg and 720 mm are far from the
objectives. On the small pack it is also slower in the urban chase (**P7 fails**). The levers are lighter avionics, a
smaller airframe and a propeller that clears the belly. The earlier test showed that a smaller airframe alone raises
the stall speed, because the avionics do not shrink with it. SPARFELL is the line where ARIGO-1H is furthest from the
objective.

---

## Not yet assessed: what each needs

| KPI | What it needs |
|---|---|
| ~~B4, S8, B6~~ | **Done:** `designs/raptors_robust.py` (MuJoCo with steady wind plus Gauss–Markov turbulence, launch and landing Monte Carlo). Scored by `sims/raptors_explore.sh`. |
| A7: link availability | A terrain and clutter line-of-sight model plus the radio's link budget. Outside the airframe models. |
| P11: drop and landing survival | Physical tests on a built airframe. Until then a structural FEA of the belly and joints. |
| B2, S9, P3: times | Trials with users and prototypes. The values above are estimates. |

## Next

1. **Agree the KPIs**: the thresholds and objectives above, which client questions need answering first (e.g. S6, B6), and the evidence each requires.
2. **Explore** (`labs/aero/sims/raptors_explore.sh -j N`, ready to run, branch `dev_raptors_exp`): for each line, sweep the levers named above from ARIGO-1H, scored with the launch-success and operating-wind models. Take the winners to CFD and back.

---

## Exploration run 1 (2026-11-20, `raptors_explore.sh`, 116 specs, all on the lattice model)

| Line | Baseline (ARIGO-1H on the lattice) | Best spec | Best score | What moved it |
|---|---|---|---|---|
| BUTEO | 51.7 | ARIGO-1H b1050 c300 + perch + charging perch + **turning station** | 66.6 | launch success 87.5 → 100 % (B4); operating wind 2.4 → 12.6 m/s (B6); urban chase |
| ARPELLA | 66.7 | ARIGO-1H b1250 c260, **3S4P** | 74.3 | coverage 7.1 → 13.4 km²; endurance 80 → 137 min. Calm-air landing energy rises to 257 J. |
| ASTOR | 55.6 | ARIGO-1H **b800** c260, 3S3P, **1200 W** motor, rail + parachute | 81.4 | 65 m/s top speed: it can tail-chase a 50 m/s target (S6), as fast as ARIGO-20 (best 72.1) with 5.7 min at full power against 2.3 |
| SPARFELL | 56.1 | ARIGO-1H (×1.0) 3S1P + quick joiner + **lite avionics** | 60.4 | mass 1.57 → 1.49 kg; assembly 60 → 30 s. Scaling the whole airframe down raises the stall (×0.7: 15.5 m/s) and costs the throw. |

**Model findings that changed the tools for run 2:**
- **The PN chase is chaotic.** The same airframe caught in about 10 s, about 18 s, or never. Each scenario is now flown from five perturbed starts and scored by the median time, with a catch rate (`arigo_perf.CHASE_STARTS`).
- **Calm air is a limit of its own.** It fails for hand throws slower than the stall, and for heavy landings with no headwind to cut the ground speed. Operating wind is now the top of the working band, and calm-air success is a separate KPI (A11, P13).

**Proposals for the KPIs (yours to decide):**
- **B5**, the quasi-steady gust stall, is about 1.6 m/s for every ARIGO-1H variant on the lattice model. It is a very conservative bound: it ignores gust alleviation and the autopilot, and the turbulence model behind B6 already measures gust tolerance dynamically. Proposal: keep B5 as information and drop it from the score, or lower its threshold.

**Next design levers (run 2):**
- SPARFELL: shrink the fuselage separately from the wing. The ×1.0 fuselage alone sets the 720 mm packed length, and the wing has to stay large for the throw.
- ASTOR: a CFD check of the b800 wing, a structural check at 65 m/s (dynamic pressure ≈ 2.6 kPa) and a propulsor-map check at 1200 W.
- ARPELLA: 3S3P against 3S4P with the calm-air landing energy in the score.

## Run 2 (`raptors_explore2.sh`, branch `dev_raptors_exp`)

**KPI change:** B5 is reported, not scored. Run 1 re-scored with `--rescore` (no simulation): BUTEO's best rises from
66.6 to 72.7; ARPELLA (74.3), ASTOR (81.4) and SPARFELL (60.4) are unchanged.

**Specs.** 90 in run 2, with its own cache and run files, so nothing from run 1 is repeated. Per line it scores two
references, the lattice baseline and run 1's best, re-scored under the run-2 models. Then the run-2 levers
(`arigo_products.SPACE2`):

| Line | Run-2 levers |
|---|---|
| BUTEO | span 1000 / 1050 / 1100 × chord 300 / 320 × the perch spring's push 6 / 8 / 10 m/s, with a turning station |
| ARPELLA | span 1100 / 1250 × chord 260 / 280 × 3S3P / 3S4P, with and without a parachute (against the heavy calm-air landing) |
| ASTOR | span 700 / 750 / 800 × chord 240 / 260 × motor 1000 / 1200 / 1400 W × 3S2P / 3S3P, with rail, turning station and parachute; plus run 1's best ARIGO-20 |
| SPARFELL | wing scale 0.9 / 1.0 × **fuselage scale 0.75 / 0.80 / 0.85**, set separately from the wing (the tail grows by √(1/fuselage scale) to keep its volume; 0.65 does not fit the tail), 3S1P, quick joiner + lite avionics, with and without the rugged skin |

**Not checked yet:**
- whether the avionics and the 3S1P pack fit the shorter fuselages (×0.75: a 540 mm fuselage);
- the structure and propulsor maps at 1400 W;
- CFD for any of the new geometries.
