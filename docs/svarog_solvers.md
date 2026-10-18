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

## 47 — Svarog III (`svarog_commercial.py`, notebook 47)

Closed forms only, under a minute; nothing is optimised and nothing is destroyed.

1. **The specs** from `designs/data/svarog_commercial_specs.json` into frozen `Spec`s; `None` is Unknown and is never
   filled; consistency checks (M² ⇄ BPP against the mode; cooling capacity against the electrical maximum's waste heat).
2. **Optics** per laser (`optics`): the fibre class by étendue, the spot two ways (core imaged / embedded Gaussian) with
   the regime by M² ≤ 1.5, the collimated beam's radius and growth, the optical efficiency to the work as a range, the
   stability band. `beam_radius_at(opt, d, kind)` is what the shielding uses.
3. **Energy** (`energy_balance`): (low, nominal, high) triples from the extreme combinations of the named `ASSUMPTIONS`
   (diode, driver and optical efficiencies, auxiliaries, COP, power factor), capped by the published maximum; a
   sensitivity table moves one assumption at a time.
4. **Welds** (`weld_case` × 2 materials × 3 thicknesses): `wd.keyhole_threshold`, `wd.speed_for_penetration` at 1.2 ×
   the thickness (the speed's interval over the absorptivity and melting-efficiency ranges), `wd.rosenthal_thin_plate`
   (K₀ in numpy, A&S 9.8.5–9.8.6; the isotherms by bisection), `wd.cooling_time_t85`, `wd.cooling_rate_at`,
   `wd.haz_width`, the distortion index, the humping and marginal-keyhole flags; every row `validated = False`.
5. **Shielding** (`shield_cases`: 4 lasers × 5 distances × 3 families = 60 base rows; 180 with the variants; the
   reflected beam doubles them): per row the irradiance against `mpe_eye_1070` / `mpe_skin_1070`, `nohd`, `od_required`
   (EN 207's LB ladder), `thermal_response` (the semi-infinite rise at T1 and T2, the lumped plate at T1,
   `si.plate_steady_bound` — the smallest over the damage rise as `bound_ratio`), `fire_risk` for wood, `seam_leak`,
   the evidence the panel would need; `verdict` → PASS only with `evidence == "certified"`, FAIL for > 1 % transmission
   or a seam, INCOMPLETE otherwise. No `heat_plate`, no burn-through time: a simulation cannot certify a barrier.
6. **Endurance** (`endurance`, 46 Part IV §3's route for the commercial lasers: `si.screen` → `th.heat_plate` only
   where undecided → `si.verdict`): black PC, mild steel and aluminium at 1, 2, 3 mm against 5–100 m, `holds_s` and the
   T1/T2/T3 classes — minutes per laser, so `python designs/svarog_commercial.py --endurance` records it in
   `designs/data/svarog_commercial_endurance.json` and the notebook only reads it (NOT RUN otherwise). A design number,
   kept out of the verdicts. On it and the sourced economic parameters (`economics` in the specs file: capex ranges from
   listings, Eurostat electricity and labour, argon, diode life, sheet prices — each with its source or marked assumed):
   `capex`, `opex_per_hour`, `cost_per_metre` (arc-on time over the model speed) and `shield_economics` (the thinnest
   recorded sheet that holds 1, 3, 5 s, its cost per m² and per k€ of source; escape seconds per euro of panel).
   `required_thickness` turns it round without any transient: the sheet a 1 / 3 / 5 / 10 s hold needs at 2–200 m from
   the lossless energy bound (d = q·t / enthalpy to gone) and, for the metals, the steady bound's never-melting thickness.
7. **F4** (`fenix` chapter `shield-steady`): the same panels' steady states by finite elements, run only by
   `wf run F4 --variant <laser>`; the notebook cross-references the runs when they exist and says NOT RUN otherwise.
8. **The record** `reports/47_svarog_iii.json` (specs verbatim, assumptions, standards with "verify the edition",
   every section, `validation_open`) and the matrices as CSV under `output/47_svarog_iii/`.

## 48 — KIMLA as a welding platform (`kimla.py`, notebook 48)

The same closed forms on five power tiers (3–30 kW) with per-tier multimode optics: `weld_case` asks `wd.speed_for_penetration`
for the joint's depth (butt 1.2 t; fillet 1.2 √2 t), takes the lower of that, the humping onset and the gantry's feed as the speed
(the cap named, the surplus depth stated), then the thin-plate field, t8/5 / cooling rate, HAZ and the distortion index, and lists
the model's limitations per row (a pool narrower than the sheet, the thick-plate regime, conduction mode, a keyhole aspect ratio
past the balance). `throughput` adds editable handling overheads per part, seam and metre; `opex_per_hour` and `capex` reuse
notebook 47's sourced parameters with KIMLA's "> 30 %" wall-plug and an assumed gantry price; `stray_cases` evaluates the beam past
the focus and specular reflections at 20–200 m against `sc.mpe_eye_1070` / `sc.mpe_skin_1070` for stated durations with no
aversion response, `collimated_fault` apart; `barrier_assessment` applies 47's verdict rule. `shield_endurance` asks 46's
question of four shields (Al 1 and 3 mm, black PC 3 mm, steel 3 mm): in the notebook `screen_bounds` (46's closed forms without
the steady solve: transparent / steady bound / energy bound / undecided) and `fire_risk` (PC's absorbed flux against its
ignition fluxes, the surface's time to the ignition temperature; metals do not burn); the steady solve and the transient only
in `python designs/kimla.py --endurance`, recorded in `data/kimla_endurance.json` and read back; in debug mode
`mock_endurance_record` (3 × the lower bound where undecided, labelled MOCK) stands in so the read-back path runs. ~80 s in the notebook.

## One notebook, and Yanus

`46_svarog.ipynb` holds the series per topic: Part I the rod, Part II the fibre laser, Part III the industrial lasers —
the 1 kW and the 2 kW devices (`si.Device`: the 2 kW as two 1 kW modules beam-combined, route B shown where it differs)
analysed one after the other in each topic (stage, heat budget, delivery and head, welds) — Part IV their guard (stray
beams, barriers, distance on the reference sheet, the air, the fenix runs), Part V the record: `svarog_design.json`,
`svarog_fiber_design.json`, `svarog_industrial_design.json` and one `reports/46_svarog.json` (Part I's keys at the top,
then `fibre`, `industrial` per device, `guard` per device, `air`), which the fenix book's `power_of` reads. The one
contract takes the fenix runs (F1–F3, 1kW and 2kW; optional) and gives those four files.

## Yanus: the chain runs itself (auto mode)

The notebook carries the `yanus_auto` flag with `SVAROG_SCREENING` as their screening switch: opening one
whose take is missing (48 without 47's report, 49 or 51 without 48's) makes the gate run the producer notebook
headlessly first, with `SVAROG_SCREENING=1` — the design modules then solve with a third of the RK4 steps
(`N_FINE` 100, `N_SWEEP` 40) and the guard on half the grid (`NR_GUARD` 12) — and then open with what is there; a
producer that fails is NOT RUN, never a crash. `yn auto 48 --dry-run` prints the plan; `yn status` marks the four
`(auto)`. Tested: 48 opened with 47's report removed ran 47 in 0.2 min and read its 58.9 W seed. The batch mechanism
(`wf batch batch.yaml`) is for workflow chapters, which Svarog has none of — every step is seconds inside a notebook.

## fenix — the finite-element chapters (book F)

The same plates in FEniCSx, as workflow chapters (`workflows/fenix/plate_thermal.py`, `docs/fenics.md`): F1 the steady
states, F2 the ablation transients, F3 the reference 2 mm PC sheet at 10–200 m, F4 Svarog III's nine shielding panels at 5–100 m (steady only); variants 1kW and 2kW, and the four commercial lasers of notebook 47 for F4; a dolfinx worker
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
| 47 (Svarog III) | 4 optics and energy balances, 24 thin-plate weld cases, 360 shielding rows | ~30 s |
