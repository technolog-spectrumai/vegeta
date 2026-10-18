# Hyperion — the solvers

How each quantity in `vegeta.hyperion` is computed: the algorithm, its discretisation, its stopping rule, its cost, and
what checks it. The physics is in [hyperion_physics.md](hyperion_physics.md). Everything is numpy; nothing calls an
external solver.

## Matrices and beams (`elements.py`, `beam.py`)

- A train's matrix is the ordered product of its elements' 2 × 2 matrices (`Train.matrix`); composite elements expand
  into their surfaces and spaces (`parts`) so a beam trace can sample inside them. The media are checked to join.
- A beam trace walks the expanded parts, applies q′ = (Aq + B)/(Cq + D) at each, samples the spot inside every space
  (40 points by default) and detects a waist where Re q crosses zero inside a space; the index changes at interfaces, so
  the spot is read as w² = −M²λ/(π n Im(1/q)) with the local n. Cost: microseconds.
- `lens_for_waist` inverts Self's formula: a quadratic in f whose discriminant says whether the waist is reachable.

## Eigenmodes and stability (`resonator.py`)

- The round trip of a linear cavity is the forward train, then the reversed train (every element's `reverse`, which flips
  an interface's media and a thick lens's radii), then the first mirror; a ring's is its loop. The reference plane sits just
  after the first mirror.
- The self-consistent q solves C q² + (D − A) q − B = 0 in the form 1/q = (D − A)/(2B) − i √(4 − (A+D)²)/(2|B|): a
  Gaussian when |A + D| < 2 (stable), none at the edges, every beam when B = 0 (degenerate, flagged). The mode is then
  traced along the forward train for the spot on every element, in both planes (tangential/sagittal) when any element
  is tilted.
- `sweep(builder, values)` rebuilds the resonator for each parameter value and solves it — a stability window costs
  milliseconds per point.
- Checked against Kogelnik & Li's closed forms for five two-mirror cavities (1e-9), the ring/linear equivalence, the
  edges and the lens-in-a-cavity window 0 < 1/f < 4/L with its degenerate point.

## The four-level laser (`gain.py`)

Closed forms only (threshold, slope, Rigrod's line, the thermal lens); the optimum coupler is a scan of 600 transmissions
against Rigrod's √(g₀L) − L (they agree to 0.002). `effective_pump_radius` integrates w_p(z)² weighted by the pump's
absorption e^{−αz} with the trapezoid rule on 400 points.

## Fibres (`fibre.py`)

- **Bessel functions by quadrature**, so that numpy suffices: Jₙ(x) = (1/π)∫₀^π cos(nτ − x sin τ) dτ on 2001 points and
  Kₙ(x) = ∫₀^∞ e^{−x cosh t} cosh(nt) dt on 6001 points to t = 14 (the integrand is below 1e-300 long before for x ≥ 0.1).
  Both at 1e-9 against the tables at x = 1.
- **The LP₀₁ eigenvalue** u J₁(u)/J₀(u) = w K₁(w)/K₀(w), u² + w² = V²: bisection on u in (0, min(V, 2.405)) — the function
  is negative at u → 0 and positive at the upper end — 80 halvings. Marcuse's bend loss then needs κ = u/a, γ = w/a,
  β and K₁(γa).
- **The power evolution.** State (P_p, P_s⁺, P_s⁻) along z. At each point the inversion is the algebraic steady state
  of the local rates, N₂/N = (R_p σ_ap + R_s σ_as)/(R_p(σ_ap + σ_ep) + R_s(σ_as + σ_es) + 1/τ) with R = P Γ/(hν A_core),
  and the gains per metre follow; the three ODEs are marched with classical RK4 on n = 300–400 steps (negative powers
  clipped at zero). The backward wave is integrated *forward* in z as an ODE — fine for the one-way gains of a few dB
  an oscillator sees; it would be stiff for a 30 dB preamplifier, which is outside the model anyway (no ASE).
  - *Oscillator:* the two unknowns at z = 0 are tied by the HR (P_s⁺(0) = R_hr P_s⁻(0)), leaving one, the circulating
    power x = P_s⁺(0). The residual P_s⁻(L) − R_oc P_s⁺(L) is negative for a tiny x (the gain is unsaturated and the
    waves grow) and positive for a huge one, so a bisection in log x (60 halvings between 1e-9 W and ten times the
    pump) finds the operating point. Threshold: a bisection on the pump for the small-signal round trip
    R_hr R_oc exp(2∫(g_s − α_s)dz) = 1. Below it the solve returns zero output. One solve ≈ 0.7 s at n = 400.
  - *Amplifier:* co-pumped is a single forward march. Counter-pumped: the pump's power at z = 0 (its *exit*) is the
    unknown; it is shot by bisection so that the marched pump reaches its launched value at z = L. 60 iterations.
  - Checked: Rigrod's invariant P⁺P⁻ = const along the fibre (1e-6, since both waves see the same net gain), the
    gratings' boundaries, pump absorbed + unabsorbed = launched, the quantum limit, zero below threshold, an interior
    optimum length at a reabsorbing wavelength (1030 nm), co- vs counter-pumping within 5 %, the counter-pump shot to
    1e-6, saturation.
- `length_sweep` and `power_curve` are loops over solves; a 19-point length sweep at n = 150 takes ~4 s.

## Heat in a plate (`thermal.py`)

Two solvers on one grid, used in that order by the guard study: the steady one answers *whether*, the transient *when*.

**The grid.** Axisymmetric r–z: `nr` rings of width dr out to `r_max` (default the larger of 8 spot radii and 12
thicknesses — far enough to hold T0 at the edge), `nz` layers of dz through the thickness. A ring cell's volume is
2π r_c dr dz; its radial faces have areas 2π r_f dz, its axial faces 2π r_c dr. Fluxes between cells are
k ΔT/distance through those faces; the axis is insulated by having no inner face at r = 0; the far edge conducts to T0.

**The transient (`heat_plate`).** The state is the enthalpy per volume H of each cell, which keeps the latent heats
honest: T(H) is linear below ρc(T_melt − T0), flat at T_melt across the latent band ρL_melt, linear again above, and
capped at T_fail (a boiling or decomposition plateau) until the cell has taken ρ(c(T_fail − T0) + L_melt + L_fail), at
which point it is removed — the ablation. The beam enters the first live cell of each column (or is spread down the
column with Beer–Lambert weights when the material has an absorption depth); the live surface and the back face lose by
convection h(T − T0) and radiation εσ(T⁴ − T0⁴). Explicit forward Euler at 80 % of the stability limit
dt < 1/(2κ(1/dr² + 1/dz²)), further capped so that no surface cell takes more than 5 % of its way to failure in one
step (an insulator's stability limit is lax; this keeps the ablation front resolved). Stops at failure — the melt or
ablation front reaching the back face at the centre (a metal's melt runs or is blown away: `failure="melt"`; a plastic
must decompose: `"ablate"`) — at t_max, or once steady: the hottest point moves less than 0.02 K/s and the front would
not cross a tenth of the plate in 30 000 s. Cost: the step count is t_max/dt — seconds for insulators and brick, tens of
seconds for a thin aluminium plate (κ = 8e-5 m²/s forces dt ~ 1e-4 s), which is why the steady solve goes first.

**The steady state (`steady_plate`).** The same grid without phase change: Jacobi iteration of the conduction balance
with the face losses linearised about the current temperature (h_eff = h + εσ(T² + T0²)(T + T0)) until the largest
change is below 1e-3 K. Reports T_max against the damage point. Milliseconds to a second.

**Checked** against Carslaw & Jaeger's semi-infinite rise (8 %, the grid's own resolution), the lumped thin-plate line
(3 %) and its melt-through time (8 %), the melting plateau cell by cell, the steady solve against the transient's settled
state (3 K) and the thin-plate Green's-function bound, the boiling cap, ablation through a sheet, the transmission of a
clear one.

## Welding (`welding.py`)

Closed forms and two root finds: Rosenthal's melting isotherm is found by bisection along the axis (ahead and behind
the source) and, for the width, a bisection in y at each of 2000 stations in x; the speed for a given penetration is a
bisection in log v on the monotonic depth-speed curve (60 halvings). The keyhole threshold is Carslaw & Jaeger's steady
Gaussian spot reaching the boiling point. Checked against their own closed forms, the monotonicity of the curves and the
textbook's orders of magnitude (1 kW at 1 m/min in steel: a few millimetres; the keyhole threshold ~1 MW/cm² for steel
and ten times that for aluminium).

## What the numbers are good for

Paraxial optics and the cavity modes are exact within the paraxial approximation — the measured mode differs only by
what the model leaves out (apertures, aberrations). The laser models (four-level, fibre rate equations) are the standard
first models and land within tens of per cent of a built device; their inputs (η_heat, cross sections, absorptivities)
carry the uncertainty, and the notebooks state each. The plate solver's own discretisation error is a few per cent;
the materials' absorptivity at 1 µm is the number that can be off by two, which is why a guard verdict near a class
boundary needs the standard's test on the real panel.
