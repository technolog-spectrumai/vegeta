# Svarog — how the notebooks compute

Each Svarog notebook is a design module (`notebooks/designs/svarog*.py`) that composes Hyperion, and a notebook that
reads it. This page says what each one runs, in what order, with which rules and at what cost — so a number can be
traced to its solver. The solvers themselves are in [hyperion_solvers.md](hyperion_solvers.md).

## 46 — the rod (`svarog.py`)

- **The pump in the rod**: the fibre's beam (M² from π r NA/λ) traced through the collimator and the focusing lens to
  the rod's face, then inside the rod with n_YAG; the effective pump radius is the absorption-weighted mean of w_p(z)²
  (trapezoid, 400 points). The magnification is chosen from a table over 1.5–6 by the thermal lens and Magni's cap.
- **The cavities**: a `Train` per revision with the rod split at the pump's absorption centroid around a `ThermalLens`;
  the thermal lens at each pump power from Innocenti's form with a stated 20 % stress/bulge share. The operating point
  re-solves the eigenmode at each pump (both planes), reads the mode radius at the lens plane and feeds it into the
  four-level laser's threshold and output — **self-consistent power curves**, 61 pump points per revision in seconds.
- **The stability windows**: `Resonator.sweep` over 0–45 W in both planes; every contiguous stable run is a window.
- **The choice rule**: the revision whose window covers the 15–30 W work range with the largest mode-to-pump ratio (D);
  B kept as the fallback that lases from cold. Output powers are all within a watt — the rule is about the mode.

## 47 — the fibre laser (`svarog_fiber.py`)

- **Doping from the datasheet**: N = α/(Γ_p σ_a) from the cladding absorption at 976 nm (1.7 dB/m for 10/125, 5 dB/m
  for 20/125) — never a free number.
- **The length**: a sweep of 2–20 m at the design pump (19 solves, n = 150, ~4 s); the rule is *the shortest fibre within
  3 % of the best output* (cheaper, stiffer against SRS and bending, less unabsorbed pump to strip).
- **The coupler**: six reflectivities solved at the chosen length; 10 % chosen where the curve is flat.
- **The amplifier**: the same length rule on 1–10 m with the oscillator's output as seed; counter- and co-pumped solved
  for the same length to show the inversion and heat profiles.
- **The coil**: Marcuse's loss on 106 radii; the spool radius is the smallest whose total loss over the length is
  ≤ 0.05 dB. **The limits**: SRS by Smith's criterion; the core's temperature at the hottest metre for four h values.
- **The comparison with the rod** reads notebook 46's report through the Yanus take (optional: the table shows the fibre
  alone when it is missing).

## 48 — the industrial laser (`svarog_industrial.py`, notebook 48)

- **The power stage**: 30/250 fibre doped to 2 dB/m, seeded with 47's 59 W (from its report when present), pumped by
  the six modules' 1254 W maximum for the length sweep (4–20 m, rule: shortest within 3 %: 10 m), then the pump *for the
  1 kW target* by bisection (40 solves): 1077 W.
- **The thermal budget** is bookkeeping on the solution: the active fibre's heat (the solver's integral of the quantum
  defect along z) and its peak per metre into `core_temperature` for three coolings; the unabsorbed pump plus 0.1 dB of
  splice loss into the stripper; 0.3 dB in the process fibre; 0.2 % absorbed per lens into Carslaw & Jaeger's steady
  Gaussian rise on BK7. No FEA: the fibre is a coil on a cooled spool and the head is a catalogue part; a structural
  analysis would be of the enclosure, which is sheet metal.
- **The head**: `DeliveryFibre` (BPP, M²) and `WeldHead` (spot, cone, Rayleigh length, the collimated beam's radius —
  the number the guard study needs) — closed forms.
- **The welds**: `keyhole_threshold`, `penetration_curve` over 11 speeds and `process_window` over the thicknesses for
  steel (keyhole absorptivity 0.7) and aluminium (0.5); the melting efficiency 0.4 is stated once and carried in the
  record. Milliseconds.

## 49 — the guard (`svarog_industrial.py`, notebook 49)

The ROI rule: **the one simulation per barrier that decides, and none where a closed form already has**.

1. **The beams** at 20 m from the head's numbers: the collimated radius plus its divergence (r_core/f_c) times the
   distance; the cone's half-angle times the distance; a fifth of the power in the latter.
2. **The screening** per (barrier, beam), closed forms only: the transmitted share (> 1 % → *fails: transparent*); the
   semi-infinite time to surface damage; the lumped lower bound on burn-through (no lateral losses) — above 30 000 s →
   *survives: energy bound*; the thin-plate Green's-function upper bound on the steady rise (no losses but conduction,
   P_abs (ln(R/w) + ½)/(2πkd); the semi-infinite Gaussian rise for a slab thicker than the spot) — below the damage point
   → *survives: steady bound*.
3. **The steady solve** (`steady_plate`, Jacobi, < 1 s) for what is left: with convection and radiation in, does the
   hottest point ever reach the damage temperature? No → *survives: steady solve*. Yes → one transient.
4. **The transient** (`heat_plate`) with the beam's real profile, 24 × 5 cells (8 through a brick), capped at 3000 s:
   stops at burn-through, or when the melt/ablation front has stalled (steady), else the front's speed over the last
   fifth of the run extrapolates the time to cross. Seconds for plastics and brick; a minute for 1.5 mm steel.
5. **The verdict** per IEC 60825-4 class: holds longer than 10 / 100 / 30 000 s; the worst beam decides the barrier.

Of 33 (barrier, beam) pairs, 6 are simulated; the others are decided by the bounds, and the notebook prints which
bound decided each. The absorptivity of the barrier at 1 µm (bare aluminium 0.12, painted or oxidised up to 0.5) is the
input to vary before trusting a verdict near a class boundary; the standard's test on the panel is the confirmation.

## Yanus: the chain runs itself (auto mode)

Notebooks 47, 48, 49 and 51 carry the `yanus_auto` flag with `SVAROG_SCREENING` as their screening switch: opening one
whose take is missing (48 without 47's report, 49 or 51 without 48's) makes the gate run the producer notebook
headlessly first, with `SVAROG_SCREENING=1` — the design modules then solve with a third of the RK4 steps
(`N_FINE` 100, `N_SWEEP` 40) and the guard on half the grid (`NR_GUARD` 12) — and then open with what is there; a
producer that fails is NOT RUN, never a crash. `yn auto 48 --dry-run` prints the plan; `yn status` marks the four
`(auto)`. Tested: 48 opened with 47's report removed ran 47 in 0.2 min and read its 58.9 W seed. The batch mechanism
(`wf batch batch.yaml`) is for workflow chapters, which Svarog has none of — every step is seconds inside a notebook.

## fenix — the finite-element chapters (book F)

The same plates in FEniCSx, as workflow chapters (`workflows/fenix/plate_thermal.py`, `docs/fenics.md`): F1 the steady
states, F2 the ablation transients, F3 the reference 2 mm PC sheet at 10–200 m; variants 1kW and 2kW; a dolfinx worker
per case in the fenicsx interpreter, a run directory per chapter. Notebooks 49 and 51 take them (optional; auto mode
makes them in screening quality) and tabulate FEM against the cell model. The air's loss on the way
(`hyperion.atmosphere`, Kruse–Kim on the visibility) is in the same sections: ≤ 1 % on a clear day at 200 m, 13 % in a
dusty town, 93 % in fog.

## Runtimes (this machine, numpy only)

| notebook | what runs | time |
|---|---|---|
| 46 | 4 revisions × 61-point power curves, 4 stability sweeps × 2 planes | ~10 s |
| 47 | length sweep, coupler table, power curve, amplifier sweep, two profiles | ~20 s |
| 48 | power-stage sweep, pump bisection, budget, welds | ~15 s |
| 49 | 33 screenings, ~10 steady solves, ~6 transients | ~1–2 min |
