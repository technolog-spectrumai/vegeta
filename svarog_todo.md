# Svarog and Hyperion — what comes next (branch `dev_svarog`)

Hyperion is Vegeta's general optics library (`vegeta.hyperion`: materials, paraxial ABCD elements, Gaussian beams,
resonators, a four-level gain model); Svarog is the laser designed and analysed with it (`notebooks/46_svarog`,
`notebooks/designs/svarog.py`). The first increment is a CW diode-end-pumped Nd:YAG at 1064 nm in a two-mirror cavity:
the rod and its pump, the thermal lens, the cavity's mode and stability window over pump power, the beam out and a
delivery train, a JSON report. Everything paraxial, pure numpy, nothing run on the machine.

The items below are deliberately deferred, in the order they will be needed: each one answers the question the one
before it raises. A stage is picked up cold from its text; the numbers of the first increment are the inputs.

## S.0 — The first increment (this branch)
- [ ] S.0.1 `vegeta.hyperion`: `materials` (Sellmeier n(λ), dn/dT, κ: BK7, fused silica, Nd:YAG, sapphire, air), `elements`
  (free space, thin and thick lens, curved and flat mirror at incidence, interface, slab, Brewster plate, thermal lens; a `Train`
  with its product matrix), `beam` (`GaussianBeam`: q, w(z), R(z), M², through a train, mode overlap), `resonator` (round-trip
  matrix, stability, eigenmode, spot sizes on every element, Gouy phase, FSR, finesse, linewidth, a parameter sweep), `gain`
  (four-level end-pumped rod: absorption, threshold, slope, output vs pump, Rigrod's optimum coupler, the thermal lens); JSON
  export, `Result`, CLI `vegeta hyperion material/lens/beam/cavity/laser`; tests against Kogelnik & Li, the lensmaker, Koechner;
  `docs/hyperion.md`
- [ ] S.0.2 `46_svarog`: the rod and pump, the thermal lens vs pump, the cavity on Hyperion with three revisions compared, the
  stability window, the beam out and its delivery train, `designs/data/svarog_design.json` and `reports/46_svarog.json`;
  the Yanus contract in `workflows/flow.py`

## S.1 — Ray tracing (the check on the paraxial numbers)
Needed first: the moment the cavity has real curved mirrors at an angle and the delivery train has a real focusing lens,
the paraxial matrices stop saying how big the aberrations are. Exact meridional (then skew) ray tracing through
spherical surfaces, pure numpy, in `hyperion.rays`.
- [ ] S.1.1 Sequential surfaces (sphere, plane; refraction by Snell, reflection) traced exactly; a fan of rays → the paraxial
  focus in the small-height limit (the test), the spherical aberration of a singlet and of a curved mirror off axis (astigmatism
  = the tangential/sagittal split the ABCD matrices already assume)
- [ ] S.1.2 Spot diagrams and the geometric spot size at the work spot of Svarog's delivery train; the wavefront error of the
  focusing lens against λ/4; which lens orientation and which bending minimise it
- [ ] S.1.3 Misalignment: the 3×3 (augmented) ABCD matrices for tilted and decentred elements, the cavity axis walk and the
  mode displacement per mirror tilt — the alignment tolerance S.3's mounts must hold

## S.2 — Thermal analysis of the rod
Needed next: the thermal lens of S.0 is a closed form (uniform pump absorption, radial conduction, a cooled barrel). Pushing
the pump up asks what the temperature, the stress and the fracture limit really are, and where the closed form stops.
- [ ] S.2.1 Analytic first, in `hyperion.gain`: the radial temperature profile of an end-pumped rod with the measured absorption
  length, the thermally induced stress and the stress-fracture limit (Koechner's figures of merit), the stress-induced
  birefringence and its depolarisation loss, the bulging end faces as a second lens term; the pump power at which each
  limit is reached in Svarog's rod
- [ ] S.2.2 The thermal lens with a non-uniform (Gaussian, top-hat fibre) pump: the aberrated lens as a radial map, its
  paraxial part for the cavity and its residual as a phase screen (the input S.6's FFT propagation reads)
- [ ] S.2.3 Talos heat transfer: a steady conduction case (CalculiX `*HEAT TRANSFER`) on the Dedalus rod and holder of S.3 —
  the heat load from `gain`, convection or a cooled copper barrel as the boundary — against S.2.1's analytic profile
  (the validation), then the coupled thermal-stress run of the same model; Talos grows a `thermal` case kind for this

## S.3 — Mechanical support: Dedalus mounts, Mellonia printing
Needed once the cavity is frozen and its alignment tolerance (S.1.3) is known: the parts that hold the cavity at that
tolerance, and how they are made.
- [ ] S.3.1 `svarog_mechanics.py` on Dedalus: the base plate / rail, the mirror mounts (kinematic: two adjustment screws and a
  pivot), the rod holder with its cooling barrel, the pump-fibre collimator holder, the output beam's lens mounts; STEP and STL
  for every part, measurements, the parts list and mass
- [ ] S.3.2 Mellonia: the printable parts (mounts, holders, covers) sliced with explicit settings — time, material, the walls
  and infill that matter for stiffness; which parts stay metal (the rod barrel, the mirror cells) and why
- [ ] S.3.3 A `47_svarog_build` notebook: the mechanical design, printing, the assembly order and the alignment procedure
  written from S.1.3's sensitivities (which screw moves what, by how much per turn)

## S.4 — FEA of the structure (Talos)
Needed after S.3 exists: whether the printed frame holds the cavity within S.1.3's tolerance under its own weight, the
screws' preload, handling and a temperature change.
- [ ] S.4.1 Linear static on the Dedalus base and mounts: deflection of each mirror under gravity and preload → the mirror
  tilt → the mode displacement through S.1.3; the stiffness ranking of the candidate frames
- [ ] S.4.2 Thermal expansion of the frame (Talos thermal, from S.2.3): the cavity-length drift per kelvin, the frequency drift
  and the mode-hopping rate; aluminium vs printed vs an Invar rail
- [ ] S.4.3 Modes of the frame (Chronos's modal path on the Talos model): the first bending mode against the pump-diode fan
  and bench vibration; the pointing jitter it implies

## S.5 — New laser types
Needed when Svarog's first laser works on paper: the same library on a different gain medium, cavity and regime. Each
one adds what Hyperion lacks for it, in this order of growing distance from the CW Nd:YAG.
- [ ] S.5.1 Q-switched Nd:YAG: the pulse build-up (the rate equations in time), pulse energy, width and peak power vs
  repetition rate (an acousto-optic or passive Cr:YAG switch); the damage thresholds of the coatings against the peak fluence;
  intracavity SHG to 532 nm (a KTP/LBO crystal as a `Slab` with phase matching and conversion efficiency)
- [ ] S.5.2 Other crystals: Yb:YAG (quasi-three-level: reabsorption, the temperature-dependent threshold), Nd:YVO₄ (anisotropic,
  a polarised output without a Brewster plate), Ti:sapphire (a broadband four-level medium, a Brewster-cut crystal, the
  folded astigmatically compensated cavity S.1 and the tilted-mirror elements already describe)
- [ ] S.5.3 A gas laser: CO₂ at 10.6 µm (a long tube, ZnSe optics, a large mode against the tube bore — the hard aperture S.6
  computes), its discharge as a volume pump: Hyperion takes the gain profile as a `GainMedium` map and designs the
  resonator on it
- [x] S.5.4 The fibre laser as the source (done first, out of order, as Svarog-fiber — notebook 47, `hyperion.fibre`): a
  step-index fibre's modes and bend loss, Yb:silica, the pump/signal power evolution of an FBG oscillator and of a seeded
  amplifier, SRS and the core's temperature, the beam out and the same delivery train, against the rod. Still open: the
  coupling of Svarog's rod beam into a delivery fibre (S.0's overlap), ASE and the noise figure of a weak-seed preamplifier,
  the higher-mode stripping budget of a few-mode fibre's coil

## S.6 — FFT physical optics
Needed last, when the paraxial description stops explaining: hard apertures, diffraction losses, the real mode of an
aperture-limited cavity, the aberrated thermal lens of S.2.2 and the M² of the beam that leaves it.
- [ ] S.6.1 `hyperion.wave`: angular-spectrum and Fresnel propagation of a sampled field (numpy FFT; scipy optional), apertures
  and phase screens, lenses as phase, the propagation through a `Train`; validated against the Gaussian beam's own w(z) and
  the Airy pattern of a circular aperture
- [ ] S.6.2 Fox–Li iteration on Svarog's cavity: the lowest-loss mode against S.0's eigenmode, the diffraction loss per round
  trip vs mirror size, the transverse-mode discrimination an aperture buys
- [ ] S.6.3 The output beam's M² and beam-quality budget: the aberrated thermal lens (S.2.2) and the lens train's aberrations
  (S.1.2) propagated as fields, the second-moment widths along z, M² against the ISO 11146 fit — the number the paraxial model
  took as an input
