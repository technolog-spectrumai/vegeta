# Kestrel-Q4T — the tandem-wing Kestrel (design brief, for approval)

> Status: **approved — Q4T1N-10 is the design**, compared with Q4T1P-10, Q4T1N-12 and Kestrel-4Q-2. Built in
> `kestrel_tandem.py` and notebook **46 `46_kestrel_tandem`** (the plan's steps 1–3 and 5); the workflows (step 4, a
> book `kestrel-tandem`, code T) are todo 5r.24. Where the notebook's numbers differ from the table in § 4 (its
> drive points, its drag build-up with the pod stretched over both wings), the notebook is the reference. Numbers below: first principles,
> the project's correlations and the fast comparator's drive and spar rules (`kestrel_fast`, `kestrel_flight`,
> Boreas); none of it is a solver result. Companion briefs: `kestrel3.md` (4Q-3), `K4Q_vs_X.md` (the X-wing).

## 1. The three airframes

Four propellers on two equal wings, one behind the other, a pod between them. The names say where the wings sit:

| code | wings | what it is |
|---|---|---|
| **Q4T1** | both at the same height | the plain tandem: two wings in one plane, the rear one a stagger behind the front |
| **Q4T1P** | front wing **high**, rear wing **low** | positive stagger in the biplane sense (the upper wing leads) |
| **Q4T1N** | front wing **low**, rear wing **high** | negative stagger: the Rutan Quickie / Proteus arrangement |

Everything else is Kestrel's: 2216 880 KV motors, 9×6 or 10×7 propellers, a 4S 3300 (or 4000) mAh pack, carbon
Ø10×8 spars, printed wings and pod, the camera dome in the nose, elevons on both wings (the rear pair as the
elevator, both pairs as ailerons in cruise and as roll/yaw paddles in the slipstream in the hover).

## 2. The requirements, as read

1. **Horizontal start.** The aircraft takes off in the aeroplane attitude — a short ground roll on belly skids (or a
   hand launch), four propellers blowing over both wings. It does not need to stand on its tail on the ground.
2. **Hover.** It can pull up from forward flight into a nose-up hover and hold it (thrust-to-weight ≥ 1.5 at the
   hover with authority in all three axes), like Kestrel-4Q.
3. **Back to horizontal with a big fall accepted.** The hover → forward-flight transition may be a nose-over and a
   dive: a large altitude loss is allowed. (Kestrel-4Q's unresolved problem was the opposite transition's balloon;
   here the demanding leg is the pull-up into the hover, and the forgiving one is the way out.)
4. Not required (say so if wrong): a vertical take-off from the ground, a vertical landing. A horizontal (belly)
   landing in the aeroplane attitude is assumed; the hover is a mission state, not the way the aircraft is handled.

Two consequences come first, before any number:

- **The span is free.** Kestrel-4Q's 1.0 m span was bounded by the tail-sitting footprint. A horizontal start and
  landing remove that bound; what remains is transport and the printed-part length. The conventional monoplanes in
  `kestrel_compare` buy their L/D with 1.3–1.6 m of span at this mass; the tandem may take some of that.
- **The hover's pitch control needs a vertical gap between the two wings' propellers.** In the hover the body x axis
  is vertical and the four thrust lines are parallel to it; a pitching moment (about body y) needs propellers at
  different *z*. In Q4T1 all four propellers are at one height: differential thrust gives roll (about x, by
  torque) and yaw (about z, left against right) but **no pitch**; pitch would come from the elevons in the
  slipstream alone (what two-motor tail-sitters do). In Q4T1P and Q4T1N the gap is the lever, exactly as the
  biplane's gap was.

## 3. First principles and correlations

**Induced drag — Munk's stagger theorem.** For a given spanwise lift distribution the induced drag of a system of
wings does not depend on stagger, only on the vertical gap. So:
- **Q4T1 (no gap)** carries its lift like one wing of the same span: k = 1/(π · b²/S · e). The lattice gives
  k = 0.084 at b 1.0 m, S 0.28 m² — **1.42 ×** the biplane's 0.059 (the same ratio as Prandtl's 1/((1+σ)/2) at
  gap/span 0.28, § 5 of notebook 43).
- **Q4T1P / Q4T1N (gap 0.28 m)** recover the biplane's figure: lattice k = 0.060, Prandtl's (1+σ)/2 = 0.695 of the
  single wing's. Stagger adds nothing to it and costs nothing.

**Lift and the rear wing's downwash.** The rear wing flies in the front wing's downwash: with no gap it keeps about
60 % of its isolated lift slope (lattice: CLα 3.65/rad for the pair, the front wing carrying 65 % of the lift at
α = 4° even without decalage). With the 0.28 m gap the rear wing is out of the worst of it (CLα 4.23/rad, front
share 55 %) — slightly *better* than the biplane's 4.10, because the staggered wings also shade each other less.

**Longitudinal stability comes from the stagger, not from reflex.** The rear wing is the tail. The neutral point of
the equal pair sits 1.57 c behind the front wing's leading edge at a stagger of 3 c (1.13 c at 2 c, 2.03 c at 4 c),
ahead of the mid-point because the rear wing's slope is reduced. The CG goes 0.3 c ahead of that (a static margin
of 10 % of the stagger, ~0.3 chord): the front wing then carries **~66 %** of the lift at trim and stalls first —
the one rule a tandem must obey, and it gives a clean nose-drop into the "big fall" of requirement 3. The wing
sections need no reflex: a cambered NACA 2412-class section on both wings (front with +2° decalage) instead of
the biplane's reflexed 23112, so the section cl_max rises from ~1.15 to ~1.45.

**CL max of the whole aircraft is the tandem's known cost.** When the front wing reaches its cl_max the rear wing
is at about half of it (the trim share), so the aircraft's CL max is 0.9 · 1.45 · (S_f/S) / 0.66 ≈ **0.99** on the
total area — about the biplane's 1.03, not better, despite the better section. A larger front wing or a drooped
rear elevator at low speed buys a little; equal wings are kept here for one printed part and one spar.

**Pitch authority is large.** The rear elevons act at ~2 c behind the CG (the biplane's elevons act at a fraction of
a chord): roughly four times the pitching moment per degree, and the front pair can be deflected the other way.
The pull-up into the hover and the recovery from the stall have the control power Kestrel-4Q lacked.

**P against N.** The lattice (flat wake along x) cannot separate them; the physics can:
- At cruise α (3–5°) both rear wings are a gap away from the front wake; the difference is second order.
- At high α (the pull-up, 20–60°) the front wing's wake, separated or not, trails **downward**. In **Q4T1P** the low
  rear wing — the elevator — flies into it: the pitch control is weakest exactly when the transition needs it.
  In **Q4T1N** the high rear wing stays in clean air through the stall. This is why the Quickie puts the rear
  wing high and why Q4T1N is the primary candidate here.
- On the ground one wing is the low one: its propellers need clearance (a 9-inch propeller on a wing 0.14 m above
  the skids leaves ~25 mm; 10-inch needs the skids 30 mm taller or the low wing's motors raised on short pylons).
  N puts the low wing in front, where a belly landing touches first — the front propellers are the ones at
  risk; P puts the risk at the back. Not decisive; it costs a pylon either way.

**Hover geometry.** Front propellers at (x_f, ±0.35, z_f), rear at (x_f + stagger, ±0.35, z_r): in the hover the
stagger lies along the thrust axis and carries no moment; the thrust rectangle is 0.70 m × 0.28 m, the biplane's.
The CG must sit at mid-gap in z (the pod between the wings, as before) and the thrust split front/rear trims the
pitch. Yaw (about body z) from left against right as before; roll (about x) from the reaction torques plus the
four elevons in the slipstream, which also give the pitch when a motor is lost.

**Structure.** The wings are cantilevers (no posts, no diagonals: a tandem has nothing to brace against), one
Ø10×8 carbon tube each, through the pod. The beam rules give an ultimate reserve of 3.6 at b 1.0 m and 2.9 at
1.2 m, tip deflection 2.6–3.9 % of the semi-span at 4 g: fine. The pod grows by the stagger (+0.42 m at 3 c) and
carries both spar sockets and the skids; that is where the mass goes (+60 g assumed, to be checked in CAD).

**Drag.** Without the bracing the build-up gives Cd0 ≈ 0.021 on 0.28 m² (the biplane with faired posts: 0.023, with
round struts 0.067), the longer pod included. Both airframes are the same propellers, nacelles and dome.

## 4. The candidates and their numbers

Mass from the Kestrel mass table on each layout (+60 g pod and skids), drive from Boreas (the 2216 880 KV on 4S),
spar from the beam rules; CL max 0.99 as derived above (0.94 for the flat tandem, its rear wing deeper in the wake);
k from the lattice; cruise at 16 m/s; usable energy 80 % of the pack.

| | T-flat Q4T1 | T-10 Q4T1N/P | T-10p Q4T1N/P | T-12 Q4T1N/P | T-12w Q4T1N/P |
|---|---|---|---|---|---|
| span × chord [m], gap [m] | 1.0 × 0.14, 0 | 1.0 × 0.14, 0.28 | 1.0 × 0.14, 0.28 | 1.2 × 0.14, 0.28 | 1.2 × 0.16, 0.28 |
| propeller, battery | 9×6, 3300 | 9×6, 3300 | 10×7, 3300 | 10×7, 3300 | 10×7, 4000 |
| mass [kg] | 1.69 | 1.69 | 1.70 | 1.79 | 1.90 |
| wing loading [N/m²] | 59 | 59 | 60 | 52 | 48 |
| Cd0 / k | 0.021 / 0.084 | 0.021 / 0.060 | 0.021 / 0.060 | 0.019 / 0.052 | 0.018 / 0.052 |
| stall [m/s], cruise/stall | 10.1, 1.58 | 9.9, 1.62 | 9.9, 1.61 | 9.3, 1.72 | 8.9, 1.79 |
| L/D at 16 m/s (max) | 11.4 (11.8) | 12.7 (14.0) | 12.7 (14.0) | 13.3 (15.8) | 13.6 (16.5) |
| cruise electrical [W], endurance [min] | 70, 34 | 67, 35 | 66, 35 | 66, 35 | 67, 42 |
| hover electrical [W], time [min] | 233, 10 | 233, 10 | 223, 10.5 | 239, 9.8 | 260, 10.9 |
| static T/W | 2.9 | 2.9 | 3.5 | 3.3 | 3.1 |
| V max (pitch-speed limited) [m/s] | 30 | 30 | 34 | 34 | 34 |
| spar reserve (ultimate), tip deflection at 4 g | 3.7, 2.6 % | 3.7, 2.6 % | 3.6, 2.6 % | 2.9, 3.9 % | 2.7, 4.1 % |
| hover pitch lever | **none** (slipstream elevons only) | 0.28 m | 0.28 m | 0.28 m | 0.28 m |

For scale, Kestrel-4Q-2 on the same rules: L/D at 16 m/s 11.7, stall 9.6 m/s (CL max 1.03 on 0.28 m²), hover
233 W, cruise ~59 W in notebook 43's full model (the comparator's drive points run a few watts higher than it).

## 5. Reading the table

- **The flat tandem Q4T1 loses on both counts that matter**: +42 % induced drag (L/D 11.4 against 12.7) and no
  pitch lever in the hover. It stays in the study as the control case the lattice and one CFD chapter can
  measure, not as a candidate to build.
- **Q4T1N and Q4T1P are the same aircraft on paper** until the flow separates; the high-α argument of § 3 says N.
  P is worth one test (the transition cases with the rotor disks) to see whether the low rear wing really loses
  its elevator in the pull-up — if it does not, P has the easier belly landing (clean front propellers).
- **The span is the lever the tandem is allowed to pull.** 1.0 → 1.2 m gives +5 % L/D, −6 % stall speed, a
  cruise/stall of 1.7, for 90 g and a stiffer-needed spar (still 2.9 reserve). The gain over the biplane is not
  the tandem arrangement itself (Munk) but the span the horizontal start permits and the bracing it does away with.
- **Endurance is pack-bound, not airframe-bound**: every 3300 mAh row sits at 35 min; the 4000 mAh buys 7 min for
  110 g. The hover is 10 min on any of them — the hover is a manoeuvre, as the requirement says.
- **Stall speed and CL max** are the biplane's. The horizontal start therefore needs the same ~10 m/s and a ground
  roll of roughly 8–12 m at T/W 3 on grass; a hand launch works at any of them.

## 6. The shortlist proposed for the notebook

Three airframes to carry through the study (lattice, mass and CG, trim, the transition model, the fast comparator,
then the workflows), one control case:

1. **Q4T1N-10** — span 1.0, chord 0.14, gap 0.28, stagger 0.42 (3 c), 10×7, front wing low, decalage +2°, CG 1.27 c
   behind the front leading edge. *The baseline: Kestrel-4Q's size with the tandem's trim and control.*
2. **Q4T1N-12** — the same at span 1.2. *What the freed span is worth.*
3. **Q4T1P-10** — the mirror of 1 with the front wing high. *The one test of the stagger sign, in the high-α cases.*
4. *(control)* **Q4T1-10** — the flat tandem, analytic and one CFD chapter only.

Fixed by argument, not varied: equal wings (one printed part, one spar type), stagger 3 c (2 c puts the neutral
point too close to the front wing for a sensible CG range; 4 c costs pod length and mass for 0.4 c of arm),
gap 0.28 m (the biplane's hover lever and σ), motors at ±0.35 m, cantilever spars, cambered 2412-class sections.

## 7. Decisions to take before the notebook (for approval)

| question | proposal | alternative |
|---|---|---|
| ground handling | belly skids under the pod and the low wing's tips; horizontal take-off and landing only | a tail-sitting stance as well (feet on the rear wing's trailing edge + fin) — restores the 1.0 m span bound |
| low wing's propellers | the low wing's motors on 30 mm pylons (clearance for a 10-inch propeller on the skids) | 9×6 propellers on the low wing only (asymmetric drive; rejected unless the pylons cost too much drag) |
| elevator | the rear elevons (both pairs mix in the hover) | a separate elevator on the rear wing's centre section — more parts, no gain |
| sections | NACA 2412 both wings, front +2° | 23012 rear (lower cm) if the trim drag of the cambered rear wing shows in the lattice |
| the hover entry | a pull-up from cruise to a near-vertical climb, thrust up, the elevator holding the nose (the transition model's forward leg, run in reverse) | not studied: a vertical take-off (needs the stance above) |
| the way out | nose-over to −30…−60° and a dive to 1.3 × stall, pull-out — altitude budget 40 m | a powered slow transition as 4Q tried (not needed per requirement 3) |

## 8. The plan after approval

1. **Design module** `designs/kestrel_tandem.py`: parameters (span, chord, gap, stagger sign and length,
   decalage, pylons), `layout`, the lattice quads with stagger (`peregrine_flight.vlm` — it already takes staggered
   quads; the P/N difference appears only in CFD), mass table and CG from `kestrel_systems`, the trim and neutral
   point, the polar (front-first stall as the CL max rule above), the transition model from `kestrel_transition`
   with the rear elevator's arm, the fast-comparator rows (`kestrel_fast.Candidate` gains `layout="tandem"`,
   `stagger`, `stagger_sign`), the take-off roll.
2. **CAD** in `kestrel.py` or a sibling `KestrelQ4T` (Dedalus): the pod lengthened by the stagger, the two wings at
   their heights, pylons, skids, the fin; `part="airframe"` for the CFD STL as today.
3. **Notebook 46 `46_kestrel_tandem`** (`yn new notebook --auto`): the three airframes and the control case against
   Kestrel-4Q-2 from notebook 43's report (as 45 does for the X-wing); sections: requirements, geometry, mass and
   CG, the lattice and the interference measure (§ 5 of 43 reused: the flat tandem's +42 %), trim and control
   authority, the hover geometry, performance, the take-off roll, the two transitions, structure (cantilever spar
   rules), the verdict and the workflows to run next. Under the 15–30 min rule, no solver in the notebook.
4. **Workflows**: a book `kestrel-tandem` (code **T**) in `workflows/flow.py` reusing `airframe_cfd.py` and
   `structure_fea.py` through a design key: T1 `airframe-sweep` per variant (N, P, flat), T2 `high-alpha` (20–60°
   bare: where P's elevator lives), T3 `transitions` with the rotor disks, T4 `spar-and-pod` (FEA of the cantilever
   spar in its pod socket at 4 g). All three run from one `batch.yaml`.
5. **Tests and docs**: the design tests as `test_kestrel2.py` (Munk's ratio within 10 % of the lattice, front wing
   stalls first, the hover lever zero for the flat tandem), the designs README, todo entries.

Estimated effort: module + CAD + notebook two sessions; the workflows a third once the CAD exports.
