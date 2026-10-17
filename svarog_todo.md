# Svarog — the road to the industrial laser (branch `dev_svarog`)

Svarog exists to arrive at **one device: a 1 kW fibre laser for welding metal, with the guard that contains it.**
Everything in this file is read against that device. Hyperion (`vegeta.hyperion`) is the optics, laser, fibre, heat and
welding library it is designed on; the physics and the solvers are `docs/hyperion_physics.md`, `docs/hyperion_solvers.md`,
`docs/svarog_physics.md`, `docs/svarog_solvers.md`.

## Done — the chain that leads to the device

- [x] **46 the rod** (`svarog.py`): end-pumped Nd:YAG, the thermal lens, Magni's rule, zone I/II cavities — where pumping,
  heat and modes were learnt; kept as the reference, not the source.
- [x] **47 the fibre laser** (`svarog_fiber.py`, `hyperion.fibre`): the Yb double-clad oscillator between gratings and the
  first amplifier — the architecture the device uses.
- [x] **48 the industrial laser** (`svarog_industrial.py`, `hyperion.welding`): the 30/250 power stage to 1 kW, its heat
  budget, the process fibre and head, the welds in steel and aluminium with their process windows.
- [x] **51 the 2 kW variant**: two modules beam-combined (the industrial route, a 200 µm spot) against one stage pumped
  harder (the 100 µm spot, 220 W on one spool), the welds and the guard at 2 kW.
- [x] **49 the guard** (`hyperion.thermal`): the stray kilowatt at 20 m against plastics, brick, aluminium, steel — closed
  forms, the steady solve, the ablation transient only where it decides, verdicts per IEC 60825-4 class; the image
  `designs/data/figures/svarog_beam_vs_plastic_box.png`.

## MVP — what the device still needs before it is a design one could order parts for

The line below is the MVP: the items above it make 48/49's numbers trustworthy enough to buy against; nothing below it is
needed for that.

- [ ] **M.1 The verdict's weakest input, measured or bounded**: the barriers' absorptivity at 1 µm (bare, painted, oxidised
  steel and aluminium — 0.12 to 0.8) as a sweep in notebook 49, so each verdict carries its margin; the clear-sheet
  transmission likewise. One afternoon; it decides whether the enclosure is 1.5 mm steel or needs a liner.
- [ ] **M.2 The enclosure as parts** (Dedalus): the steel panels sized by 49, the interlocked head cover (the collimated
  beam must not exist outside it), the stripper block and the water-cooled spool from 48's heat budget; the parts list.
- [ ] **M.3 The weld trial plan**: 48's process windows as a test matrix (material, thickness, speed, focus position),
  the measured penetration to recalibrate `melting_efficiency` and the keyhole absorptivity — the two stated numbers the
  windows hang on.
- [ ] **M.4 The welding head's optics as built**: ray tracing of the real 100/200 doublets (spherical aberration at
  NA 0.06, the focus shift from the lenses' 2 W of heating), the cover slide, the fibre's end cap — the spot and its
  depth of focus as delivered, against the paraxial 100 µm.
- [x] **M.5 The field ablation check** (`dev_svarog_advanced`, smoke-tested with dolfinx 0.11: steady 1222 vs 1232 K on the
  guard's steel, polycarbonate burn-through 26 vs 30 s): the same 10 mm polycarbonate and 1.5 mm steel
  cases in FEniCSx (enthalpy method, element removal) → XDMF → PyVista, against `hyperion.thermal`'s cell model — the
  validation of the cheap solver that 49's verdicts rest on.

———————————————————————————— MVP line ————————————————————————————

## Later — useful, not needed for the device

- [ ] L.1 The weld pool beyond the energy balance: a moving-source transient (the plate solver with a moving beam) for the
  pool shape and cooling rate; keyhole stability (porosity, spatter) from the literature.
- [ ] L.2 Pulsed variants for cutting and marking (Q-switch / pulsed MOPA), with SRS and self-focusing as the walls.
- [ ] L.3 The rod's thermal FEA, stress and fracture limits (Talos heat transfer) — the fibre device has no rod.
- [ ] L.4 FFT physical optics (apertures, Fox–Li, the aberrated thermal lens) — a multimode 50 µm process beam does not
  need it; a single-mode marking laser would.
- [ ] L.5 Other gain media (Yb:YAG, Nd:YVO₄, Ti:sapphire, CO₂) and the rod's folded cavities — other products.
- [ ] L.6 The rod's mounts, printing and frame FEA (Dedalus, Mellonia, Talos) — the rod is not being built.
- [ ] L.7 Fibre details outside the MVP: ASE and the noise figure of a weak-seed preamplifier, the few-mode coil's
  higher-mode stripping budget, the coupling of a free-space beam into a delivery fibre.

Skipped on purpose: a structural analysis of the laser itself (a coil on a cooled spool, a combiner, a catalogue head —
nothing load-bearing); the enclosure's panels are sheet metal sized by 49, M.2.

## Svarog III — the commercial lasers (notebook 47, branch `dev_svarog_iii`)

- [x] **III.1 The sheets as published**: IPG YLS-SM, Coherent FL2000C / FL2000CSM, nLIGHT WFL-3000 in
  `designs/data/svarog_commercial_specs.json` — limits, not operating values; unknowns kept as null and never filled.
- [x] **III.2 Optics, energy, welds, shielding** (`designs/svarog_commercial.py`): BPP ⇄ M² and the fibre each implies, the
  core imaged against the embedded Gaussian through Svarog's head; the energy balance as (low, nominal, high) on named
  assumptions with a sensitivity table; the six sheet cases per laser on the thin-plate Rosenthal forms now in
  `hyperion.welding`; the 60-case base shielding matrix (180 with the variants, ×2 with the reflected beam) with the MPE,
  NOHD, OD ladder, non-destructive bounds, wood's fire risk, seams and degraded sheets; verdicts PASS / FAIL / INCOMPLETE.
- [x] **III.2b Endurance and the euro**: how long black PC, steel and aluminium at 1, 2, 3 mm stand the collimated beam
  at 5–100 m (46's screen → transient → verdict route), recorded by `svarog_commercial.py --endurance` and read by the
  notebook — a design number outside the verdicts; on it and sourced economic parameters (October 2026: listings for the
  source, head and chiller, Eurostat electricity and labour, argon, diode life, sheet prices; the first top-of-head
  allowances kept only as a record): capex, cost per hour and per metre of weld, the shield a 1 / 3 / 5 s escape needs
  per k€ of source, escape seconds per euro of panel; the sheet thickness a 1 / 3 / 5 / 10 s hold needs at 2–200 m from
  the closed-form bounds alone (`required_thickness`, no transient).
- [x] **III.3 fenix F4 `shield-steady`**: the nine panels' steady states by finite elements for each commercial laser,
  steady only, run on request (`wf run F4 --variant FL2000C`); the notebook takes the runs as optional inputs.
- [ ] **III.4 The MPE constants** (`mpe_eye_1070`, `mpe_skin_1070`) checked against the current IEC 60825-1 edition, and
  EN 207's D-mode ladder / EN 12254's AB levels against their current editions — the notebook says "verify" until this is
  ticked. *Not done: no safety finding of notebook 47 is a permission.*
- [ ] **III.5 Barrier evidence**: an EN 12254 protection level or an IEC 60825-4 test report per panel for the irradiance
  and exposure class — the only route from INCOMPLETE to PASS (`evidence = "certified"` entered by hand with the report).
- [ ] **III.6 The sheets' unknowns** (YLS-SM and WFL-3000 electrical numbers, the Coherent pair's M²) from the makers'
  current data; until then the intervals stand.
- [ ] **III.7 The weld trials** of M.3 for the four lasers' six sheet cases: no penetration, cooling or strength of
  notebook 47 is quoted before them.
- [ ] **III.8 F4 on a fenicsx machine** for the four lasers, and the FEM steady states against the closed-form bounds.
- [ ] **III.9 A moving-source sheet-weld chapter** (the thin-plate field by finite elements with the keyhole as a cavity:
  L.1's first use) — deferred; the closed forms serve the comparison.

## Brokar — Svarog in the field and on Perseus (planned, not implemented)

The next study after Svarog III, planned in a session on 2026-10-17 and **deferred**: **Brokar Field** (the 1 kW / 2 kW
Svarog on a truck with a generator — generator sizing from the wall-plug budget and the chiller's inrush, air against
liquid cooling, mass, IP, transport and maintenance, a MIG/MAG set as the economic baseline) and **Brokar Robot** (the
laser cabinet stationary on the plant's mains, the **existing Perseus rover** carrying only the welding head, sensors and
the protected process fibre: no new robot — Perseus's geometry, payload, arm and power as the source of truth, the head
as a tool entry in `cutting.SAWS` and `replace(pbr.PERSEUS, saw=…)` so no Perseus module changes; CG, tip-over, arm
torques, reach and the drive from `perseus_robot` unchanged; the cable's route, bend radius and damage detection; an
integration-requirements table for what Perseus lacks — tool mount, 48 V tap, laser-enable/shutter/interlock as an
independent hardware loop outside the Chiron controller, telemetry, strain relief — documented, not invented). Both
platforms welding 1–3 mm steel and aluminium on the Svarog III forms, a commercial single-source laser against the
two-module route, costs as dated assumptions with sources, and a laser-safety assessment (MPE, NOHD, OD, the audit of
46's distance table citing `si.EXPOSURE` / `si.verdict`, PASS / FAIL / INCOMPLETE). Modules `brokar.py`,
`brokar_safety.py`, `brokar_drawings.py`, notebook 48, report `reports/48_brokar.json`; a test matrix for enclosed trials.

- [ ] **B.1** Brokar as above — not started.

## M.5 as built — book fenix

The field solver is a workflow book (`workflows/fenix/`, code F): F1 steady, F2 ablation, F3 the reference sheet at
10–200 m; notebooks 49 and 51 take its runs; `docs/fenics.md`. The advanced branch and notebook 50 are retired into it.
