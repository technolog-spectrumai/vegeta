# Hyperion — the physics

What each module of `vegeta.hyperion` computes and from which theory, in the order the code is organised. The
companion [hyperion_solvers.md](hyperion_solvers.md) says *how* each is solved numerically. References are at the end;
`[K&L]` and the like point into that list.

## 1. Materials (`materials.py`)

The refractive index of a glass or crystal against wavelength is Sellmeier's dispersion formula, a sum of resonances
fitted to measurements, n²(λ) = 1 + Σ Bᵢ λ²/(λ² − Cᵢ) with λ in µm `[Sellmeier]`; the coefficients are the catalogues'
(Schott for N-BK7, Malitson for fused silica and sapphire, Zelmon for YAG `[Malitson, Zelmon]`). Air is Ciddor's two-term
form `[Ciddor]`. Each material also carries what a thermal lens is made of: the temperature coefficient of the index dn/dT,
the thermal conductivity κ, the expansion α_T and Poisson's ratio ν `[Koechner]`.

## 2. Paraxial optics — ABCD matrices (`elements.py`)

A ray near the axis is a height and an angle, (y, θ). Every element maps that pair linearly, so it is a 2 × 2 matrix
and a train of elements is a product of matrices `[Siegman ch. 15, K&L]`: free space [[1, L], [0, 1]], a thin lens
[[1, 0], [−1/f, 1]], a spherical interface from n₁ to n₂ of radius R [[1, 0], [(n₁ − n₂)/(R n₂), n₁/n₂]], a curved
mirror the thin lens with f = R/2. Hyperion uses the *geometric* angle, so an interface's determinant is n₁/n₂ and a
train that returns to its medium has determinant 1. Two consequences the code leans on:

- A thick lens is two interfaces and a slab; from its product matrix come the effective focal length f = −1/C and the
  cardinal points (principal planes, back and front focal lengths), which reduce to the lensmaker's equation with its
  thickness term `[Hecht §6.1]`.
- A mirror struck at an angle θ has different focal lengths in the plane of incidence (tangential, f cos θ) and across
  it (sagittal, f / cos θ); a plate at Brewster's angle has different effective lengths in the two planes. This is the
  astigmatism of a folded cavity, and Kogelnik's 1972 paper shows how a Brewster-cut crystal and a tilted mirror can
  cancel each other `[Kogelnik 1972]`. Hyperion carries both planes through every train.

## 3. Gaussian beams (`beam.py`)

A laser beam is not a ray but a Gaussian: its radius w(z) = w₀ √(1 + (z/z_R)²), wavefront curvature R(z) = z (1 +
(z_R/z)²), Rayleigh range z_R = π n w₀²/λ, far-field half-angle λ/(π n w₀) `[Siegman ch. 17, K&L]`. The whole beam is
one complex number, q = z + i z_R, and the ABCD law q′ = (Aq + B)/(Cq + D) carries it through any paraxial element —
the same matrices as the rays. A beam of quality M² > 1 (a multimode beam) propagates like a Gaussian with the wavelength
replaced by M² λ: Siegman's embedded-Gaussian rule `[Siegman 1990]`. Self's closed form for a thin lens focusing a
Gaussian (the new waist w₀ f/√((z − f)² + z_R²) and where it sits) is used as the check and inverted to pick a lens for
a wanted spot `[Self]`. The mode overlap of two coaxial Gaussians (the coupling into a fibre or a cavity) is the
standard overlap integral in closed form `[Joyce & DeLoach]`.

## 4. Resonators (`resonator.py`)

Round the cavity once and the beam must come back to itself: q = (Aq + B)/(Cq + D) with the round-trip matrix. That
quadratic has a Gaussian solution only when |(A + D)/2| < 1 — the stability criterion, which for two mirrors is
0 < g₁g₂ < 1 with gᵢ = 1 − L/Rᵢ `[K&L, Siegman ch. 19]`. The solution gives the waist, its position and the spot on every
element; Kogelnik & Li's 1966 closed forms for the two-mirror case are Hyperion's reference. The phase a Gaussian
accumulates beyond a plane wave (the Gouy phase, arccos((A + D)/2) per round trip) sets the transverse-mode frequency
spacing; the free spectral range c/L_rt, the photon lifetime, the finesse and the linewidth follow from the round-trip
survival of the power `[Siegman ch. 11, 19]`. A ring cavity is the same algebra on the loop.

A thermal lens inside a cavity moves it across the stability diagram. Magni's 1986 analysis shows what the code's sweeps
show numerically: the size of the mode in the lens is set by the dioptric *width* of the stability zone, w² = 2λ/(π ΔD),
not by the mirrors; a cavity meant to stay stable from cold to full pump (zone I) therefore has a small mode, and the
high-power design opens a second zone (zone II, L > R_oc) above a threshold pump `[Magni]`.

## 5. The four-level laser (`gain.py`)

An end-pumped Nd:YAG rod is the classic four-level system: pump photons fill the upper level, stimulated emission
empties it, the rate equations in steady state give a threshold and a slope. With a top-hat pump of radius w_p and a
Gaussian mode of radius w_l the overlap integral reduces to an effective area π(w_p² + w_l²)/2 `[Fan & Byer, Laporta &
Brussard]`: P_th = π hν_p (w_p² + w_l²)(T + L)/(4 σ τ η_abs η_q) and η_slope = η_abs η_q (λ_p/λ_l) T/(T + L). The output
against the coupler's transmission T is Rigrod's — the circulating intensity that saturates the gain to the loss —
in its low-gain form, with the optimum √(g₀ L) − L `[Rigrod]`. The pump's heat makes a lens: the radial temperature
profile of a rod cooled at its barrel, through dn/dT, gives 1/f = η_h P_abs (dn/dT)/(π κ w_p²) for an end-pumped rod
`[Innocenti]`; the stress-induced index change and the end-face bulging add of the order of 20 % in Nd:YAG `[Koechner
ch. 7]`, a stated fraction in Hyperion, not a default.

## 6. Fibres and the fibre laser (`fibre.py`)

A step-index fibre guides by total internal reflection; its single parameter is V = 2π a NA/λ, single-mode below
2.405 (the first zero of J₀). The fundamental mode LP₀₁ is close to a Gaussian whose radius Marcuse fitted as
w/a = 0.65 + 1.619 V^−1.5 + 2.879 V^−6 `[Marcuse 1978]`; its exact eigenvalue is u J₁(u)/J₀(u) = w K₁(w)/K₀(w) with
u² + w² = V² `[Snyder & Love]`. A coiled fibre radiates: Marcuse's 1976 pure-bend loss formula, exponential in the radius,
with the stress-optic correction of silica (an effective radius 1.28 times the geometric one) `[Marcuse 1976, Schermer &
Cole]`. In a double-clad fibre the pump fills the inner cladding and crosses the core with probability (a/b)², the
signal sits in the core with Γ_s = 1 − exp(−2a²/w²).

Yb³⁺ in silica is a quasi-three-level ion: absorption and emission cross sections overlap `[Paschotta]`, so the signal
is reabsorbed where the pump is weak and a fibre can be too long. The steady-state power evolution of pump and signal
along the fibre comes from the local two-level rate equations with the overlap factors — the model of Kelson & Hardy
`[Kelson & Hardy]`, here without amplified spontaneous emission: for an oscillator both signal directions between two
fibre Bragg gratings, for an amplifier one pass of a seed with a co- or counter-propagating pump. The limits: stimulated
Raman scattering above Smith's threshold 16 A_eff/(g_R L_eff) `[Smith, Agrawal]`, and the core's temperature from
radial conduction through core, cladding and coating to the heat sink `[Brown & Hoffman]`.

## 7. Heat in a solid (`thermal.py`)

What a beam does to a plate is heat conduction with a source at the surface (or, for a partly transparent material,
Beer–Lambert deposition in the bulk), losses by convection and radiation at the faces, and phase changes: melting at
T_melt with its latent heat, and leaving — vaporisation of a metal, decomposition of a polymer — at T_fail with its own.
The closed forms are Carslaw & Jaeger's: a uniform flux on a semi-infinite solid raises the surface by 2q√(t/πρck); a
Gaussian spot in steady state raises the centre by P_abs/(√(2π) k w); a thin plate heated uniformly without losses is a
lumped capacitance `[Carslaw & Jaeger]`. Burn-through of a metal barrier is a melt-through (the melt runs or is blown
away); of a polymer, decomposition; IEC 60825-4 names the exposure times a laser guard must hold (T1 10 s, T2 100 s, T3
30 000 s) and asks for a test on the real panel — the calculation says where to look `[IEC 60825-4]`. The material numbers
are handbook values; the absorptivity at 1 µm is the least certain (a bare metal's can double with oxide or roughness).

## 8. Welding (`welding.py`)

A multimode fibre's beam is characterised by its beam parameter product r·NA (the invariant through any lens) and
M² = π BPP/λ; a welding head images the fibre's core onto the work with the focus-to-collimator ratio and the beam's
cone is the fibre's NA divided by it `[Steen & Mazumder ch. 2]`. When the absorbed intensity is enough to boil the
surface the vapour opens a keyhole and the beam is absorbed deep in it — the keyhole threshold, estimated here as the
power at which Carslaw & Jaeger's steady Gaussian spot reaches the boiling point, which gives the textbook's ~1 MW/cm²
for steel `[Steen & Mazumder ch. 4]`. Below it, conduction-mode welding: Rosenthal's moving point source, the
temperature field of a heat source travelling over a thick plate, whose melting isotherm is the pool `[Rosenthal]`.
Above it, the keyhole weld's depth is an energy balance — the absorbed power that ends in the melt, divided by the
speed, the weld width and the energy to melt a unit volume — with a melting efficiency that Swift-Hook & Gick showed
cannot exceed 0.48 for a line source at speed `[Swift-Hook & Gick]`, and a keyhole absorptivity raised by the
multiple reflections inside it (0.6–0.8 for steel, ~0.5 for aluminium) `[Steen & Mazumder, Kaplan]`.

## References

- **[Sellmeier]** W. Sellmeier, "Zur Erklärung der abnormen Farbenfolge im Spectrum einiger Substanzen", Ann. Phys. 219, 272 (1871).
- **[Malitson]** I. H. Malitson, "Interspecimen comparison of the refractive index of fused silica", J. Opt. Soc. Am. 55, 1205 (1965); "Refraction and dispersion of synthetic sapphire", J. Opt. Soc. Am. 52, 1377 (1962).
- **[Zelmon]** D. E. Zelmon, D. L. Small, R. Page, "Refractive-index measurements of undoped yttrium aluminum garnet from 0.4 to 5.0 µm", Appl. Opt. 37, 4933 (1998).
- **[Ciddor]** P. E. Ciddor, "Refractive index of air: new equations for the visible and near infrared", Appl. Opt. 35, 1566 (1996).
- **[K&L]** H. Kogelnik, T. Li, "Laser beams and resonators", Appl. Opt. 5, 1550 (1966).
- **[Kogelnik 1972]** H. W. Kogelnik, E. P. Ippen, A. Dienes, C. V. Shank, "Astigmatically compensated cavities for CW dye lasers", IEEE J. Quantum Electron. 8, 373 (1972).
- **[Siegman]** A. E. Siegman, *Lasers*, University Science Books (1986); **[Siegman 1990]** "New developments in laser resonators", Proc. SPIE 1224, 2 (1990) — the M² embedded Gaussian.
- **[Hecht]** E. Hecht, *Optics*, 5th ed., Pearson (2017).
- **[Self]** S. A. Self, "Focusing of spherical Gaussian beams", Appl. Opt. 22, 658 (1983).
- **[Joyce & DeLoach]** W. B. Joyce, B. C. DeLoach, "Alignment of Gaussian beams", Appl. Opt. 23, 4187 (1984).
- **[Magni]** V. Magni, "Resonators for solid-state lasers with large-volume fundamental mode and high alignment stability", Appl. Opt. 25, 107 (1986).
- **[Koechner]** W. Koechner, *Solid-State Laser Engineering*, 6th ed., Springer (2006).
- **[Fan & Byer]** T. Y. Fan, R. L. Byer, "Diode laser-pumped solid-state lasers", IEEE J. Quantum Electron. 24, 895 (1988).
- **[Laporta & Brussard]** P. Laporta, M. Brussard, "Design criteria for mode size optimization in diode-pumped solid-state lasers", IEEE J. Quantum Electron. 27, 2319 (1991).
- **[Rigrod]** W. W. Rigrod, "Saturation effects in high-gain lasers", J. Appl. Phys. 36, 2487 (1965).
- **[Innocenti]** M. E. Innocenzi, H. T. Yura, C. L. Fincher, R. A. Fields, "Thermal modeling of continuous-wave end-pumped solid-state lasers", Appl. Phys. Lett. 56, 1831 (1990).
- **[Marcuse 1978]** D. Marcuse, "Gaussian approximation of the fundamental modes of graded-index fibers", J. Opt. Soc. Am. 68, 103 (1978) — the w/a fit; **[Marcuse 1976]** "Curvature loss formula for optical fibers", J. Opt. Soc. Am. 66, 216 (1976).
- **[Schermer & Cole]** R. T. Schermer, J. H. Cole, "Improved bend loss formula verified for optical fiber by simulation and experiment", IEEE J. Quantum Electron. 43, 899 (2007).
- **[Snyder & Love]** A. W. Snyder, J. D. Love, *Optical Waveguide Theory*, Chapman & Hall (1983).
- **[Paschotta]** R. Paschotta, J. Nilsson, A. C. Tropper, D. C. Hanna, "Ytterbium-doped fiber amplifiers", IEEE J. Quantum Electron. 33, 1049 (1997).
- **[Kelson & Hardy]** I. Kelson, A. A. Hardy, "Strongly pumped fiber lasers", IEEE J. Quantum Electron. 34, 1570 (1998).
- **[Smith]** R. G. Smith, "Optical power handling capacity of low loss optical fibers as determined by stimulated Raman and Brillouin scattering", Appl. Opt. 11, 2489 (1972); **[Agrawal]** G. P. Agrawal, *Nonlinear Fiber Optics*, 6th ed., Academic Press (2019).
- **[Brown & Hoffman]** D. C. Brown, H. J. Hoffman, "Thermal, stress, and thermo-optic effects in high average power double-clad silica fiber lasers", IEEE J. Quantum Electron. 37, 207 (2001).
- **[Carslaw & Jaeger]** H. S. Carslaw, J. C. Jaeger, *Conduction of Heat in Solids*, 2nd ed., Oxford (1959).
- **[IEC 60825-4]** IEC 60825-4, *Safety of laser products — Part 4: Laser guards* (2006, amended 2011).
- **[Steen & Mazumder]** W. M. Steen, J. Mazumder, *Laser Material Processing*, 4th ed., Springer (2010).
- **[Rosenthal]** D. Rosenthal, "The theory of moving sources of heat and its application to metal treatments", Trans. ASME 68, 849 (1946).
- **[Swift-Hook & Gick]** D. T. Swift-Hook, A. E. F. Gick, "Penetration welding with lasers", Welding J. 52, 492s (1973).
- **[Kaplan]** A. Kaplan, "A model of deep penetration laser welding based on calculation of the keyhole profile", J. Phys. D 27, 1805 (1994).

Codes that do the same physics at greater depth (not dependencies; what to reach for when Hyperion's first number is
not enough): **LightPipes** (physical-optics beam propagation, Fox–Li, Python, BSD), **RayOptics** (sequential ray tracing
and paraxial design, Python), **OpticStudio / Zemax** and **Code V** (commercial sequential/non-sequential optics),
**LASCAD** (thermal lensing of pumped crystals with FEA, commercial), **RP Fiber Power** (fibre amplifier and laser rate
equations with ASE, commercial), **pyfiberamp** (Yb/Er fibre amplifier rate equations with ASE, Python, MIT), **Elmer**
and **CalculiX** (conduction with phase change for the barrier and the weld pool, open FEA — the Talos path),
**OpenFOAM** (melt-pool flow, the Aeromant path).
