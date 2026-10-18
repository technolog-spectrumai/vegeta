# Raptors redesign — from the four races

Input: `data/race_design_factors.json` (real MuJoCo laps, 2026-10-18: hangar DNF 13/18,
industrial 42.0 s with 1 collision, farm DNF, quarry DNF; reference 5-inch quad, 571 g).
This is a gut-feeling design note, not an analysis. Numbers are quoted only where they decide something.

## 1. What the laps actually say (and what they do not)

Hard facts (course geometry, independent of the flyer):
- openings 1.2–1.5 m, footprint allowed 0.9 m; 10 openings under 2 m in the hangar
- corners: 24 per lap, sharpest 152° (hangar), 90° outdoors; least radius 1.3 m at ~6 m/s
- legs: hangar median 9 m, outdoors 15–25 m, longest 39–50 m
- vertical 18–39 m per lap, descent up to 8 m/s, climb 5 m/s

Soft facts (they describe the quad and its untuned autopilot, not the course):
- 92 % of hangar time below 12 m/s, 22 % hovering; outdoors cruise p50 ~15 m/s, top needed ≤ 24 m/s
- hover T/W 6.5, lateral 4.3 g, 747 °/s: inflated by six crash/reset cycles per course; treat as upper bounds
- energy 1.4–2.3 Wh per lap, 8.7 Wh for a race: the 23 Wh pack is 2.7× oversize
- the DNFs were all contact (closest approach 0.11 m), never power, never speed

The last two lines kill most of the old assumptions.

## 2. Assumptions thrown away

1. "A raptor is a plane that can hover." No: on these courses it is a rotorcraft that may borrow a wing.
   On the 39–50 m straights a 2 g machine reaches 20 m/s in 20 m; the wing has 20 m left to earn its keep.
2. "Economy wins races." Energy is not binding anywhere. The pack is the biggest free mass in the whole design.
3. "Speed wins." Hangar needs 11 m/s, outdoors 18–20. Every design already has that. Contact and corners decide.
4. "Three winged variants." The missions split differently than the old names did.

## 3. Missions (re-cut)

- **M-A indoor technical** (hangar): 0–11 m/s, 1.2 m holes, 150° corners, descents, 22 % hover, contact likely.
- **M-B outdoor technical** (industrial, farm, quarry): 15–20 m/s cruise, 90° corners, 25–50 m legs, 1.5 m gates.
- **M-C open range** (not among the four courses): the only mission where a real wing pays. Kept as a derivative.

## 4. Embodiments

### Gavilan → "Gavilan-D", a ducted 4-inch for M-A (unorthodox: smaller, slower, armoured)
- 4-inch tri-blades in four printed ducts (PETG/CF-PETG, 1.2 mm wall) that double as bumpers; duct ring
  adds low-speed static thrust, which is the only regime that matters indoors.
- 1504/1804 motors on 4S 850 mAh (9 Wh usable: 3 laps + margin); target 330–360 g, footprint 0.24 m.
- No wing. The old X-wing surfaces survive only as the duct struts.
- Why it wins the hangar: the reference quad lost it to six contacts, not to speed. A glancing contact at
  3–6 m/s against a 1.2 m frame becomes a bounce instead of a reset.
- Trade-offs: top speed ~17 m/s (irrelevant on M-A, loses M-B), duct drag above 12 m/s, duct mass ≈ 40 g.
- Cost ≈ €215 (4×€14 motors, €55 stack, €55 FPV, €15 pack, €8 print, €30 RX/GNSS).

### Apus → "Apus-L", the light big-disc quad for M-B economy at equal pace
- Same 5-inch stack as the reference, pack cut to 6S 850 (−70 g), 6-inch props on 2207 if the gates allow 0.33 m
  (they do: 1.5 m). Disc loading drops a third; hover power −20–25 %; all thrust-limited accelerations gain 15 %.
- Printed plate frame instead of carbon: +25 g, −€35; the plate is shaped as a low-AR lifting body (no span
  penalty, no servos). It will not fly on it, but at 25° nose-down tilt it adds a few newtons at 15 m/s for free.
- Trade-offs: 6-inch props slow the rate response (~15 % lower angular acceleration), a printed plate flexes.
- Cost ≈ €255. Beats the reference on Wh/lap by ~35 % and on lap time by a few percent through mass alone.

### Alcotan → "Hobo-W", quad with a clip-on incidence wing for M-B/M-C (the one wing we keep; a vagrant quad that hitches a wing)
- Reference quad geometry, a single plank wing 0.8 m × 0.14 m (fits the 0.9 m footprint) clipped between the
  arms at +30° incidence to the frame: at 25° nose-down forward tilt the wing sees ~5°, lifts ~2/3 of the
  weight at 18 m/s, cuts cruise power roughly in half on the 50 m legs; at hover it is a vertical fence and costs
  nothing but crosswind. Zero servos, zero transitions; the rotors stay the only controls.
- Wing is a hot-wire EPP core or a two-piece print with a 6 mm carbon spar: €12 and 60 g.
- Trade-offs: roll inertia up ~2× (747 °/s is a hangar number, outdoors 300–400 °/s suffice); the wing
  cannot pass the 1.2 m hangar holes except sideways; in a 90° corner at 15 m/s it adds lift-vector help
  only if the autopilot banks instead of skidding (the current controller does not, see R.2).
- M-C derivative: 1.2 m span, 6S 1300, same airframe; range and sustained speed from the old Alcotan brief.
- Cost ≈ €310 with the clip-on wing; the wing is the cheapest part of the aircraft.

### What stays on the shelf
- Single-motor plank (old Apus): cannot make 4+ hover T/W cheaply; fails M-A outright.
- Twin monoplane / dart / biplane cells: span or depth kills them at the 1.2 m holes, and the energy they save
  is not needed. The biplane cell returns only for M-C if a 0.9 m span ever becomes a rule.
- Tailsitters: the 22 % hover time and 152° corners are exactly where they are worst.

## 5. Required parameters per mission (design targets, not laps)

| | M-A Gavilan-D | M-B Apus-L | M-B/C Hobo-W |
|---|---|---|---|
| footprint [m] | ≤ 0.30 | ≤ 0.35 | ≤ 0.90 (wing), passes 1.5 m gates |
| hover T/W | ≥ 4 | ≥ 4 | ≥ 3.5 |
| top speed [m/s] | ≥ 15 | ≥ 25 | ≥ 25, sustained ≥ 20 |
| roll/pitch rate [°/s] | ≥ 800 | ≥ 700 | ≥ 400 |
| least turn radius at 6 m/s [m] | ≤ 1.5 | ≤ 2 | ≤ 3 |
| descent / climb [m/s] | 8 / 5 | 8 / 5 | 6 / 5 |
| usable energy for 3 laps [Wh] | ≥ 6 | ≥ 8 | ≥ 8 (M-C: 20) |
| contact tolerance | bounce at 6 m/s | none | none |
| mass [g] | 330–360 | 430–470 | 520–560 |
| BOM [EUR] | ~215 | ~255 | ~310 |

## 6. Cost to manufacture (per unit, small batch, printed on Mellonia)

- Reference quad ≈ €300 (carbon frame €45). Every raptor above is within ±€15 of it; the only expensive part
  anywhere is the FPV link (€55) and the 4-in-1 stack (€55–75).
- Print time: ducts 4 × 40 min, Apus plate 2 h, Hobo-W wing halves 2 × 1.5 h; all under 60 g per part, no supports
  if the wing is printed on its trailing edge with a 6 mm spar hole.
- Servos were the cost and failure item of the old winged designs; none of the three uses one.

## 7. What to verify next (first principles only)

1. Gavilan-D: duct static thrust gain vs duct mass at 4-inch, and the 4S 850 endurance at hover power.
2. Apus-L: 6-inch drive table (new cache), rate response vs reference.
3. Hobo-W: lift at 18 m/s and 5° for a 0.8 × 0.14 plank (Re ~1.7e5), roll inertia, and whether the
   autopilot banks in corners; the wing's hover-fence drag in the industrial crosswind corridors.
4. Re-run the four races with each embodiment once the autopilot survives the hangar stair (todo R.2);
   until then the "required" rates and load factors are upper bounds, not targets.

## 8. Hypothesis: one core, three overlapping configurations (recommended over three specialists)

Lap time is thrust/mass, DNFs are contact; both are set by the shared parts (stack, motors, pack, arms).
What separates indoor from outdoor is bolt-on: bumper rings, prop size, pack size, a wing.

- **Hobo (core, universal):** 5-inch arms, 2207 stack, 6S 1000 (covers 6 Wh indoors / 8 Wh outdoors),
  printed plate with pack bay and front/rear rails, ~430 g bare, footprint 0.30 m (fits 1.2 m holes with
  0.45 m each side). Finishes all four courses as is.
- **Gavilan kit (indoor):** four printed bumper rings / half-ducts on the arms (+40 g), 4S 850 in the same
  bay (−60 g). Replaces the separate 4-inch Gavilan-D; keep that only for openings under ~0.8 m.
- **Apus kit (outdoor):** 6-inch props, lightest pack that makes the race, rings off (−40 g, +44 % disc area).
- **W kit:** the 0.8 m clip-on incidence plank on the rails (outdoor courses, open-range derivative).

Cost of universality by first principles: rings carried outdoors ≈ +9 % mass ≈ +4 % lap time (√ law) plus
ring drag above 15 m/s (~+2 %); the big pack carried indoors ≈ +3 %. Specialists win each course by 3–6 %;
one DNF outweighs all of it over a series. Scored across mixed courses → the family wins; swap-per-race → specialists.

Manufacturing: one plate, one ring, one wing, one rail; ~80 % of cost common. Family ≈ €300 core + €20–40 per
kit vs ≈ €780 for three specialists.

Weakest point: whether a ring on a 5-inch turns a 6 m/s glancing hit into a bounce or a tumble (ring stiffness,
hit location) — the first thing to test in MuJoCo, since contact decides these courses; and the rails must be
quick-change without loosening or nobody swaps between heats.
