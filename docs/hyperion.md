# Hyperion — optics (materials, ABCD elements, Gaussian beams, resonators, lasers, fibres, beam combining, heat, welding)

The physics behind each module, with references: [hyperion_physics.md](hyperion_physics.md); how each is solved and
checked: [hyperion_solvers.md](hyperion_solvers.md). The device series built on it: [svarog_physics.md](svarog_physics.md),
[svarog_solvers.md](svarog_solvers.md).

Hyperion is the optics library: what a laser's cavity and beam are made of, on paper. Pure numpy; nothing is run on
the machine and nothing is picked for you — the elements, their spacings, the wavelength, the pump, the mode, the
coupler and the losses are inputs. Lengths in metres, angles in degrees at the elements' construction, radians inside.

```python
from vegeta import hyperion as hy

hy.YAG.n(1064e-9)                                        # Sellmeier index; dn/dT, κ, α, ν on the Material
lens = hy.ThickLens(0.1, -0.1, 0.02, "BK7")              # radii, thickness, material
lens.cardinal_points(633e-9)                             # f, principal planes, BFL/FFL, the lensmaker's f for comparison
b = hy.GaussianBeam(1064e-9, w0=200e-6, m2=1.2)          # a beam at its waist; w(z), R(z), rayleigh_range, divergence
tr = b.through([hy.FreeSpace(0.5), hy.ThinLens(0.1, name="L1"), hy.FreeSpace(0.3)])
tr.spot_at("L1"), tr.waists, tr.out                      # the spot on every surface, every waist inside, the beam at the end
hy.lens_for_waist(b.at(0.5), 50e-6)                      # the lens that focuses this beam to 50 µm (Self's formula)

cav = hy.Resonator(hy.Train([hy.CurvedMirror(0.5, name="HR"), hy.FreeSpace(0.05), hy.Slab(0.01, "Nd:YAG", name="rod"),
                             hy.FreeSpace(0.14), hy.FlatMirror(0.95, name="OC")]), 1064e-9, loss=0.01)
m = cav.mode()                                           # stable, (A+D)/2, w0, z0, the spot on every element, the round-trip Gouy phase
cav.spectrum()                                           # FSR, photon lifetime, finesse, linewidth, Q, transverse mode spacing
hy.Resonator.sweep(lambda f: build(f), fs, elements=["rod"])      # the stability window over a thermal lens, a spacing, ...

rod = hy.Rod(hy.ND_YAG, radius=1.5e-3, length=0.01)
laser = hy.Laser(rod, w_pump=200e-6, w_mode=200e-6, T=0.05, loss=0.01)
laser.threshold(), laser.slope_efficiency(), laser.output(10.0), laser.optimum_coupler(10.0)
hy.thermal_lens(rod, 10.0, 200e-6)                       # a ThermalLens element for the cavity, f ∝ κ w_p² / (η_h P dn/dT)
hy.save_design("runs/svarog.json", {"cavity": cav, "beam": b})    # the hand-off; load_design reads it back
```

**Elements** (`elements.py`): `FreeSpace(L, n)`, `ThinLens(f)`, `ThermalLens(f, pump_W)`, `FlatMirror(reflectivity)`,
`CurvedMirror(R, angle, reflectivity)` (tangential f = R cos θ/2, sagittal R/2 cos θ), `Interface(n1, n2, R)`, `Slab(t, n)`,
`BrewsterPlate(t, n)` (Kogelnik's effective lengths t√(n²+1)/n⁴ and /n²), `ThickLens(r1, r2, t, material)`. Rays are
(height, geometric angle): an interface's determinant is n₁/n₂, a train that returns to its medium has determinant 1.
A `Train` checks that its media join, gives the product matrix (`matrix(wavelength, plane, start, stop)`), the positions
and the optical length, `reverse()` for a return trip, `to_dict`/`from_dict`. A radius is positive when its centre
lies beyond the surface along the beam (concave mirror R > 0; biconvex lens R₁ > 0, R₂ < 0).

**Beams** (`beam.py`): `GaussianBeam(wavelength, w0, z0, m2, n)` — q transforms by the ABCD law with the embedded
Gaussian's wavelength M²λ/n; `through(train)` → a `BeamTrace` (stations, waists, the sampled envelope `z`/`w` for
drawing), `after(train)`, `overlap(other)` (the TEM₀₀ power coupling), `from_fibre(λ, core_radius, NA)` (M² = π r NA/λ).

**Resonators** (`resonator.py`): a linear cavity is the train from one end mirror to the other; a ring is the loop in
order. `mode(plane)` solves the self-consistent q from the round trip (unstable when |(A+D)/2| ≥ 1, degenerate when B = 0),
traces it along the forward path and returns the spot on every element (parts of a composite too: `"rod in"`, `"rod out"`);
`modes()` both planes (a folded cavity's astigmatism); `g_parameters()` for a bare two-mirror cavity; `spectrum()`;
`Resonator.sweep(builder, values)`; `two_mirror(R1, R2, L, λ)` and `kogelnik_li(...)` (the closed forms, for checks).

**Gain** (`gain.py`): `GainMedium` (σ, τ, λ_l, λ_p, α_pump, η_q, η_heat, the host Material), `ND_YAG` as the named case;
`Rod(medium, radius, length)`; `Laser(rod, w_pump, w_mode, T, loss)` — the four-level, end-pumped steady state through
the effective area π(w_p² + w_l²)/2: `threshold()` = π hν_p (w_p² + w_l²)(T + L)/(4 σ τ η_abs η_q), `slope_efficiency()`
= η_abs η_q (λ_p/λ_l) T/(T + L), `output(P)` (Rigrod's low-gain line), `small_signal_gain(P)`, `optimum_coupler(P)` (a scan,
and Rigrod's √(g₀L) − L), `table(P)`; `thermal_lens(rod, P, w_p, stress_and_bulge)` (Innocenti's end-pumped dn/dT lens;
the stress and end-bulging share as a stated fraction — of the order of 0.2 in Nd:YAG — not a default),
`rod_temperature_rise`, `effective_pump_radius(w(z), rod)` (a diverging pump's absorption-weighted radius).

**Fibres** (`fibre.py`): `Fibre(core_radius, NA, clad_radius, coating_radius, clad_NA, alpha_signal, alpha_pump)` — `V`,
`cutoff_wavelength`, `single_mode`, `mode_radius` (Marcuse's fit, w/a = 0.65 + 1.619 V^−1.5 + 2.879 V^−6), `mode_area`,
`overlap_core` Γ_s = 1 − exp(−2a²/w²), `overlap_clad` Γ_p = (a/b)², `beam(λ)` (the LP₀₁ as a `GaussianBeam`), `lp01(λ)` (the
eigenvalue u J₁(u)/J₀(u) = w K₁(w)/K₀(w), Bessel functions by quadrature so numpy suffices), `bend_loss(R, λ)` (Marcuse 1976
with the 1.28 stress-optic factor; an order of magnitude, as the catalogue curves). `Dopant` (a (λ, σ_a, σ_e) table, τ, N),
`YB_SILICA` (Paschotta's shape), `ions_for_absorption(fibre, dB/m, λ)` — the datasheet's absorption sets N. `ActiveFibre(fibre,
dopant, length)` is the gain medium alone. `FibreLaser(active, λ_p, λ_s, R_hr, R_oc)` (alias `FibreOscillator`) composes it with the two
reflectors' boundary conditions — the fibre is the guided path, the gratings the resonator: the pump and both signal waves along z from the two-level rate
equations with the overlap factors (Kelson & Hardy 1998, no ASE), RK4 and a shooting on the circulating power so the waves meet
the gratings' reflectivities; `solve(P)` → a `FibreSolution` (profiles, the inversion, the heat per metre, output, absorbed and
unabsorbed pump), `threshold()` (the small-signal round trip reaching 1), `power_curve`, `length_sweep`. `FibreAmplifier(active,
λ_p, λ_s, "co" | "counter")`: one pass of a seed, the counter-pump shot to its launched power. `srs_threshold` (Smith's
16 A_eff/(g_R L_eff)), `core_temperature` (radial conduction through core, cladding, coating, and h to the sink).

**Beam combining** (`combining.py`): the outputs of complete fibre lasers (not pump diodes — that is a fibre combiner's
étendue sum, `svarog_industrial.SignalCombiner`) put together on a grid. `Channel(beam, power, x, y, phase, loss)` — a
`GaussianBeam` at its waist (wavelength, w0, M²), its power before the combiner, its centre, phase and the fraction lost on
the way in; `channels(n, λ, P, w0, pitch, layout="ring"|"line"|"hex"|"overlap", m2, phases, losses, wavelengths)` makes a
set (neighbours 3.5 w0 apart by default: a 0.2 % field overlap). `Grid(size, n)` / `Grid.for_channels` (the window holds
every channel with 12 waists of margin so the far-field pixel λ/size is under 1/7 of a channel's half-angle; the sampling
resolves the smallest waist). `far_field(E, grid, λ)` — Fraunhofer by FFT, scaled dx²/λ so Parseval holds: ∫|E_far|² dΩ =
∫|E|² dA. Three combiners, each `combine(chans, grid=None, n=256)` → `Combined`:
`IncoherentCombiner(throughput)` — intensities add (different lasers, no phase control), every channel's far field widened
by its M²; `CoherentCombiner(throughput, aperture="auto"|"filled"|"tiled")` — fields add at one wavelength and M² = 1
(two wavelengths or a multimode channel raise): *filled* (every channel on one axis through a lossless N-port, the output
Σ E_i/√N — in phase it carries ΣP, phase errors send the rest out of the other ports, `extra["port_loss"]`), *tiled* (side by
side: the single-tile envelope times the array factor, N² on axis in a lobe N times narrower, the side lobes keep the
rest, the touching tiles' cross term reported as `overlap`); `SpectralCombiner(min_separation, dispersion,
efficiency_per_channel)` — distinct wavelengths overlapped by a grating or dichroics: intensities add on one axis, each
colour's far field resampled onto one angle grid, a residual pointing `dispersion` × (λ − λ_centre) per colour, the element's
efficiency per channel. `Combined` carries `I_near` [W/m²], `I_far` [W/sr] on `theta_x/y`, the accounting (`power_in`,
`power_out`, `power_near`, `power_far`, `conservation` = the Parseval check, `overlap` = the near field against `power_out`),
the second moments `sigma_x/y`, `sigma_tx/ty`, `bpp_x/y` = σ_x σ_θx, `m2_x/y` = 4π σ_x σ_θx/λ (one wavelength only, else
nan: there is no M² across colours), `brightness` = P/(π·2σ_x·2σ_y·π·2σ_θx·2σ_θy) [W/(m²·sr)], `I_far_axis`, `efficiency`
(incoherent and spectral: power out / power in; coherent: the power in the array's central lobe — the cone of a
diffraction-limited Gaussian of the array's near-field size — against what such a Gaussian puts there; 1 for one beam,
less for an array or for phase errors) and `extra` (`axis_gain` against the same channels in phase, `axis_vs_single`,
`cone_half_angle`, `port_loss`, `residual_pointing_max`); `cross_section`, `to_dict`. `equivalent_gaussian(P, λ, w0, grid)` —
the one beam of the same power every result is held against; `phase_sensitivity(chans, combiner, sigmas, trials)` — the
on-axis gain and central-lobe efficiency under Gaussian random phase errors, with the closed form 1/N + (1 − 1/N) e^(−σ²);
`power_table`. Power is not brightness: every method delivers the channels' power (minus its own loss); only the filled
coherent and the spectral combiner keep one channel's étendue.

**Heat** (`thermal.py`): `Solid` (ρ, c, k, T_melt/L_melt, T_fail/L_fail, absorptivity at 1 µm, emissivity, an absorption
depth for a partly transparent material, the failure mode melt/ablate) with `MILD_STEEL`, `STAINLESS`, `ALUMINIUM`,
`BLACK_PC`, `CLEAR_PC`, `LASER_PC`, `BLACK_PE`, `BRICK`; the closed forms `semi_infinite_surface_rise`, `thin_plate_time_to`,
`gaussian_steady_rise`; `steady_plate` (Jacobi on the r–z grid with convection and linearised radiation) and `heat_plate`
(explicit enthalpy-method transient with ablation, the surface-damage and burn-through times, `HeatingResult.survives(t)`).

**Welding** (`welding.py`): `DeliveryFibre` (BPP, M², its beam), `WeldHead` (spot, cone, collimated radius, Rayleigh
length), `keyhole_threshold`, `rosenthal_pool`, `penetration` / `penetration_curve` / `speed_for_penetration` /
`process_window`.

**Atmosphere** (`atmosphere.py`): `Atmosphere(visibility, extra)` with `ATMOSPHERES` (clear day, wet day, dusty town, foggy
autumn morning), `aerosol_attenuation` (Kruse–Kim on the visibility), `transmission_table(distances, λ)`.

## Validation
- Materials against the catalogues: N-BK7 n_d 1.5168, fused silica 1.4570 at 633 nm and 1.4496 at 1064 nm (Malitson), YAG
  1.816 at 1064 nm (Zelmon), sapphire 1.760 at 800 nm, air 1.000277.
- The thick lens against the lensmaker's equation with its thickness term (1e-12) and the principal planes
  h = −f (n−1) t/(n R₂) (Hecht); the thin limit as t → 0; every element's determinant 1 in both planes, an interface's n₁/n₂;
  a flat interface crossed and crossed back is the identity, a curved one crossed twice has twice the power.
- Gaussian beams: w(z), R(z) and the Rayleigh range closed forms (1e-12 along a trace); the Gaussian lens formula
  (Self 1983: w₀′ = w₀ f/√((z−f)² + z_R²), its waist position) and its inverse `lens_for_waist` (1e-9); the geometric
  limit of a collimated beam; a collimated M² beam focuses M² times larger; the spot continuous through a slab.
- Resonators against Kogelnik & Li (1966) for five two-mirror cavities (waist, its position, both mirror spots, 1e-9);
  (A+D)/2 = 2g₁g₂ − 1; the round-trip Gouy phase 2 arccos √(g₁g₂); the eigen-q reproduces itself; the edges
  (concentric, plane–plane, convex–flat unstable; the confocal point degenerate, m = −1, w₀² → λL/2π beside it with the
  transverse modes degenerate at FSR/2); a ring equals its unfolded linear twin; a folded cavity's tangential spot
  smaller than its sagittal; a flat–flat cavity with a lens stable for 0 < 1/f < 4/L except the imaging point 2/L.
- The laser: the threshold formula, η_s, the output line, g₀(P_th) = T + L, the scan's optimum coupler against Rigrod's
  closed form (±0.002), the Nd:YAG numbers in Koechner's ranges (I_sat ≈ 2.9 kW/cm², a 100–300 mW threshold for a 200 µm
  spot, a thermal lens of tens of mm at 10 W); f ∝ 1/P and ∝ w_p².
- Fibres: J₀, J₁, K₀, K₁ at 1 to 1e-9; SMF-28's MFD 10.4 µm at 1550 nm (±0.2) and its cutoff; Marcuse's w/a = 1.10 at the
  cutoff; the bend loss exponential in R, tens of dB/m at R = 10 mm and nothing at 30 mm for SMF-28, less at 1310 nm. The
  oscillator: Rigrod's P⁺P⁻ constant along the fibre (1e-6), both gratings' boundaries met, pump absorbed + unabsorbed =
  launched, the output under the quantum limit (λ_p/λ_s × absorbed), nothing below the threshold and light above it, the slope
  between 0.7 and λ_p/λ_s, an interior optimum length where the signal is reabsorbed (1030 nm). The amplifier: energy
  conservation, the counter-pump shot to its launched power (1e-6), co- and counter-pumping within 5 % in output with the
  stronger inversion at the co-pump's weak-signal end, the gain falling with the seed (saturation).
- Beam combining: a Gaussian's second moments (σ = w0/2, σ_θ = λ/(2π w0), M² = 1, BPP = λ/4π) and Parseval (1e-12) on the
  FFT far field; the same channel identical across the three methods; incoherent: power N×, the far field a channel's, M² up,
  brightness down, losses and throughput take power only; coherent filled: ΣP in phase and N × a channel's on-axis
  intensity, brightness N × one channel's, cos²(Δφ/2) for two channels (the π case: nothing out, all of it in
  `port_loss`); coherent tiled: ΣP kept (the 3.5 w0 pitch's cross term under 1 %), N² on axis, the central lobe's share
  below 1; two tiles reproduce Young's envelope × cos²(π d θ/λ) (1e-6) with the nulls at λ/2d and the π case dark on
  axis; two wavelengths refused by the coherent combiner, too-close ones by the spectral; spectral: 0.95 × ΣP, one channel's
  σ_θ and σ_x, brightness ∝ power, M² nan, a residual dispersion widens θ only; a channel's M² widens its far field by M²
  (M² read back ±2 %, brightness /M⁴) and the coherent combiner refuses it; `phase_sensitivity` against the closed form (±0.05).

## CLI
```bash
vegeta hyperion materials                                                   # the named materials and their indices
vegeta hyperion material --name Nd:YAG --wavelength 1064e-9
vegeta hyperion lens --r1 0.1 --r2 -0.1 --thickness 0.02 --material BK7 --wavelength 633e-9
vegeta hyperion beam --wavelength 1064e-9 --w0 200e-6 --lens 0.1 --at 0.5 --after 0.3 [--m2 1.3] [--z 0 1 0.1]
vegeta hyperion cavity --r1 0.5 --r2 inf --length 0.2 [--lens 0.3 --at 0.05] [--reflectivity 1 0.95 --loss 0.01]
vegeta hyperion laser --length 0.01 --pump 10 --wp 200e-6 --wl 200e-6 --t 0.05 [--loss 0.01] [--bulge 0.2] [--medium Nd:YAG]
vegeta hyperion fibre --core 5e-6 --na 0.075 --clad 62.5e-6 --wavelength 1070e-9 [--bend 0.03 0.05]
vegeta hyperion fibre-laser --core 5e-6 --na 0.075 --coating 125e-6 --length 8 --absorption 1.7 --pump 20 [--r-oc 0.1]   # an oscillator
vegeta hyperion fibre-laser --core 10e-6 --na 0.06 --length 5 --absorption 5 --pump 40 --seed 15 [--direction counter]     # an amplifier
vegeta hyperion combine --n 4 --power 250 --w0 12e-6 --method coherent-filled|coherent-tiled|incoherent|spectral [--layout ring --pitch 4e-5 --phases 0 0 0 0 --losses ... --spacing 3e-9 --dispersion 0 --m2 1.3 --json]
```
Exit codes: 0 success (`cavity`: 1 when the cavity is unstable; `fibre-laser`: 1 when nothing comes out; `combine`: 1 when nothing leaves the output port), 2 invalid input.

## What it is not
Paraxial, Gaussian and steady: no apertures, no diffraction loss, no aberration — a tilted mirror's astigmatism and a
Brewster plate's are in, a lens's spherical aberration and a hard-edged mirror's diffraction are not (the ray-tracing and
physical-optics stages of `svarog_todo.md`). The thermal lens is the end-pumped closed form for a top-hat pump: the first
number, not the measured one; the laser model is the four-level low-gain steady state with the standard overlap area,
good to the tens of per cent a design decision needs, not the per cent a measurement gives. The fibre laser is the
two-level rate equations with overlap factors and no ASE: right for an oscillator and a seeded amplifier above a watt, not for
a preamplifier or the noise figure; the bend loss is Marcuse's for the pure bend of LP₀₁ (no mode coupling, no
higher-mode stripping budget), and the cross sections are the literature's shape — the fibre's datasheet decides N.
Pulses, Q-switching and frequency conversion come with the new laser types (S.5).
