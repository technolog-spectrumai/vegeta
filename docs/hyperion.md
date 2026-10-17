# Hyperion — optics (materials, ABCD elements, Gaussian beams, resonators, a four-level laser)

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

## CLI
```bash
vegeta hyperion materials                                                   # the named materials and their indices
vegeta hyperion material --name Nd:YAG --wavelength 1064e-9
vegeta hyperion lens --r1 0.1 --r2 -0.1 --thickness 0.02 --material BK7 --wavelength 633e-9
vegeta hyperion beam --wavelength 1064e-9 --w0 200e-6 --lens 0.1 --at 0.5 --after 0.3 [--m2 1.3] [--z 0 1 0.1]
vegeta hyperion cavity --r1 0.5 --r2 inf --length 0.2 [--lens 0.3 --at 0.05] [--reflectivity 1 0.95 --loss 0.01]
vegeta hyperion laser --length 0.01 --pump 10 --wp 200e-6 --wl 200e-6 --t 0.05 [--loss 0.01] [--bulge 0.2] [--medium Nd:YAG]
```
Exit codes: 0 success (`cavity`: 1 when the cavity is unstable), 2 invalid input.

## What it is not
Paraxial, Gaussian and steady: no apertures, no diffraction loss, no aberration — a tilted mirror's astigmatism and a
Brewster plate's are in, a lens's spherical aberration and a hard-edged mirror's diffraction are not (the ray-tracing and
physical-optics stages of `svarog_todo.md`). The thermal lens is the end-pumped closed form for a top-hat pump: the first
number, not the measured one; the laser model is the four-level low-gain steady state with the standard overlap area,
good to the tens of per cent a design decision needs, not the per cent a measurement gives. Pulses, Q-switching, frequency
conversion and quasi-three-level media come with the new laser types (S.5).
