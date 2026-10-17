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
- [ ] **M.5 The field ablation check** (`dev_svarog_advanced`, not run here): the same 10 mm polycarbonate and 1.5 mm steel
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
