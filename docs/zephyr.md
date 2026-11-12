# Zephyr — 2-D wing sections (geometry, thin-airfoil theory, polars)

Zephyr is the section tool: the shape of a wing section, what thin-airfoil theory says about it, and its viscous
polar from one of two backends — **NeuralFoil** (a neural surrogate of XFOIL, `pip install neuralfoil`, MIT; seconds
for hundreds of points, a *confidence* per point) or **XFOIL** itself (Drela's panel method with a viscous boundary
layer; the executable on the `PATH` or `XFOIL=/path/to/xfoil`). Pure numpy otherwise; nothing is picked for you: the
section, the Reynolds number, the transition criterion and the flap are inputs.

```python
from vegeta import zephyr

s = zephyr.naca("23112", reflex_k21=0.02)      # a reflexed 5-digit section with more reflex than the standard line
zephyr.thin_airfoil(s)                           # alpha0_deg, cm_c4, alpha_ideal_deg (Glauert's integrals on the camber line)
p = zephyr.analyze(s, alphas=range(-6, 20), Re=1.5e5)        # Polar: alpha_deg, cl, cd, cm, confidence, source
p.summary(cl_design=0.4)                         # cl_max and its angle, alpha0, lift slope, cd_min, L/D max, cm0, x_ac, ...
zephyr.check_against_thin_airfoil(p, s)          # the backend's alpha0 and cm0 against theory
zephyr.flap_sweep(s, cf=0.28, Re=1.5e5)          # a plain flap: tau (and thin-airfoil's), the flap's lift centre, drag per deflection
zephyr.save_polars("runs/sections.json", [p])    # the hand-off; load_polars reads it back
```
`zephyr.naca` takes 4-digit (`2412`) and 5-digit designations, plain (`23012`) and **reflexed** (`23112`: the third
digit 1; `reflex_k21` raises the aft reflex above the standard k₂/k₁ for a nose-up moment — a tailless aircraft's
trim). `from_dat` reads Selig and Lednicer files, `from_coordinates` any outline; `Section.flapped(cf, delta)` turns a
plain flap; `Section.coordinates(n, te_cut)` gives a CAD-ready outline (`te_cut` < 1 for a finite trailing edge).

## Validation
- Thin-airfoil theory against Abbott & von Doenhoff: NACA 2412 α₀ = −2.08°, Cm_c/4 = −0.053; 4412 −4.15°, −0.106;
  23012 −1.09°, −0.013; the standard reflexed 230-/240-/250-series lines come out at Cm_c/4 = 0 (±0.0025), which is
  what they were drawn for. A plain flap's thin-airfoil effectiveness τ(c_f) against the closed form.
- Each backend against theory: α₀ within 0.6°, Cm₀ within 0.012, the lift slope 0.85–1.05 × 2π; NACA 0012 antisymmetric
  in lift and symmetric in drag; XFOIL and NeuralFoil within ΔCl 0.05 and ΔCd 0.003 in the attached range (when both
  are present). Tests skip a backend that is not installed.

## CLI
```bash
vegeta zephyr backends                                              # what this machine has
vegeta zephyr section --naca 23112 --k21 0.02 -o s.dat              # thin-airfoil numbers, a .dat file
vegeta zephyr polar --naca 2412 --re 1.5e5 --alpha -6 18 1 --cl 0.4 -o polar.json [--method xfoil] [--ncrit 6]
vegeta zephyr flap --naca 0012 --cf 0.28 --re 1.5e5
```
Exit codes: 0 success, 1 the analysis failed (the Result says why), 2 invalid input.

## What it is not
NeuralFoil's polar is a surrogate's: trust it where its confidence is high (attached flow, ordinary shapes), read it as
a trend past the stall; XFOIL is the reference at low Reynolds numbers but not an oracle either (free transition,
2-D). The section an aircraft is built on deserves a tunnel or a 2-D CFD run (Aeromant) at the end. `n_crit` 9 is a
clean tunnel; a built-up small-UAV wing in turbulent air is nearer 4–6, and the polar moves with it.
