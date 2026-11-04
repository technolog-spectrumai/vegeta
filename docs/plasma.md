# Fulgur and the bladeless turbine — plan (v0)

Two things, in order:

1. **`vegeta-fulgur`** (`libs/vegeta-fulgur`, imported as `vegeta.fulgur`): a reusable cold-plasma toolkit. Its
   first physics is the corona discharge in air and the thrust it makes (electrohydrodynamic, "ionic" thrust). Later
   physics (dielectric-barrier discharge actuators for flow control, plasma igniters, ozone, sterilisation) reuses
   the same field, chemistry and transport pieces.
2. **A bladeless turbine** for electric, noiseless transport. We begin by **recreating the small prototype** from
   Plasma Channel's video *I Built A Bladeless Turbine For Silent Flight* (the transcript and the screenshots in this
   session), in CAD and in Fulgur, and check Fulgur's numbers against what the video measured. Only when Fulgur
   reproduces the small one do we scale up.

The toolkit is tested against published measurements first, the replica second: the video's numbers are from a
garage (a vane anemometer, 78 % humidity, an unstated voltage), so they bound Fulgur, they do not calibrate it.

## What the video gives us

Four generations; all ring electrodes, radial symmetry, air pulled through the rings along the axis.

| model | layout | mass | velocity (anemometer, at the focal point) | power | thrust |
|---|---|---|---|---|---|
| Mark III (2023) | multi-stage: 3 sections × 3 segments, straight | — | 2.6–2.7 m/s; 3.3 m/s after wider electrode spacing and larger grounds (+26 %) | — | — |
| Mark IV | single stage, **dual convergence**: 3 copper-plated printed ground rings + 2 serrated bronze emitter rings; outer and inner streams converge at the rim and again ~100 mm downstream | 209 g | 2.7 / 3.2 / 3.5 m/s at low / medium / full power (78 % RH); **4.1 m/s** at 50 % RH | **50 W** | — |
| Mark 4B | 3× the flow area of IV, as large as the Form 4 prints (~200 mm): 4 nickel-plated ground rings, 2 plain bronze loop emitters with longer, sharper teeth, hollow-bolt threading to adjust the gap | 513 g | 2.5 / 2.8 / 3.1 / 3.6 m/s; focal point 150 mm back | — | — |
| Mark 4C | 4B without the threading, less restriction, lighter | 484 g | 2.7 m/s low, **3.7 m/s** max | — | **~25 gf** (pendulum: 499 g at 470 mm, 24 mm deflection) |

Measured sensitivities worth reproducing:

* serrated emitter edge vs square teeth: **+17 %** velocity;
* wider electrode spacing and larger grounds on the Mark III: **+26 %**;
* **more gap + more voltage made it worse** on the 4B (two steps, each slower), ending in a short that melted the
  frame — the opposite of the 1-D theory, so something else limits it (the sheath of ions reaching the next ring,
  the frame's surface conduction, the supply's current limit);
* humidity: shorts at 78 % RH; the same thruster gave 3.5 → 4.1 m/s going from 78 % to 50 % RH.

What the screenshots add (estimates, to be refined as parameters, never hard-coded):

* Mark IV: 5 concentric rings of ~6 mm tall strip on a green printed base, outer diameter ~150 mm (it sits on a
  kitchen scale), 6 printed supports between rings, clear resin, 4 mm-ish screws; the emitters are bronze strip,
  teeth ~1 mm pitch, bent to rings and brazed;
* Mark 4B: a blue printed frame, 6 spokes, clip seats for 4 ground rings of 3 mm round section, outer ring ~200 mm,
  a central cone (hub) with an open 20 mm bore, emitter loops of ~2 mm bronze; the Onshape model "Ionic Thrust Wing"
  shows the grounds as D-section rings and the convergence as the inner rings sitting downstream of the outer ones;
* Form 4 print: "Mark 4B ground 4", 3 h 43 min, 36.6 mL of resin, printed on edge.

## The physics Fulgur must get right

A corona at the sharp emitter ionises air; ions drift to the ground ring through the gap `d` at mobility
`μ ≈ 2.0×10⁻⁴ m²/(V·s)` (positive ions, dry air), colliding with neutrals on the way; the neutrals are the jet.

* one-dimensional limit: thrust `T = I·d/μ`, thrust per watt `T/P = d/(μ·V)` — longer gaps and lower voltages are
  *more* efficient per watt, which is why the Mark III gained from spacing and why the 4B's loss is interesting;
* the current: `I = C·V·(V − V₀)` (Townsend/Warburg), onset `V₀` from Peek's law with the emitter's tip radius; the
  teeth lower `V₀` and raise `C`, which is the serration's +17 %;
* the jet: momentum flux `T = ρ·A·v²` at the exit area; for the 4C, 0.245 N at 3.7 m/s needs `A ≈ 0.015 m²`
  (a 140 mm equivalent disc, consistent with a ~200 mm ring with a hub) — a first check that the velocity and
  thrust the video reports agree with each other;
* the power the 4C must have drawn: with `d ≈ 20 mm` and `V ≈ 30 kV`, `T/P ≈ 3.3 mN/W`, so 0.245 N ≈ **75 W**; the
  Mark IV's stated 50 W at 4.1 m/s over a smaller area sits on the same curve. Fulgur's first result is this curve;
* humidity: water lowers the mobility slightly and the breakdown field a lot (surface conduction on the resin);
  Fulgur models it as a mobility and onset correction plus a surface-flashover limit on the printed supports;
* what it cannot be: an EHD thruster's `T/P` is ~2–20 mN/W; a vehicle needs thrust ≥ weight, so at 5 mN/W a 1 kg
  craft needs 2 kW and a supply lighter than that. The plan keeps this limit visible from the first number.

## `vegeta-fulgur` — what it holds

numpy + scipy only (sparse solvers), no OpenFOAM; the CFD step is *prepared* for Aeromant and run elsewhere, as
Drako's CFD is.

| module | what | first version |
|---|---|---|
| `gas` | air: density, ion mobility (species, temperature, pressure, humidity), Peek onset field, breakdown field | tables and the standard fits |
| `corona` | 0-D/1-D: onset voltage for a tooth/wire/point, the `C·V·(V−V₀)` current law, 1-D thrust and `T/P`, velocity at an exit area | analytic; validated on wire-to-cylinder data |
| `ehd` (solve) | on FiPy: Poisson `∇²φ = −ρ_q/ε₀` (electrodes Dirichlet, the frame a dielectric) coupled to the ions' drift–diffusion `∇·(ρ_q μ E − D ∇ρ_q) = 0`, the Kaptzov condition at the emitter (the field stays at onset; the injected charge adjusts), iterated until the current converges; Gmsh meshes of the ring section | the coaxial benchmark first (exact), then the rings |
| `ehd` | the body force `f = ρ_q·E` and the flow it drives: first an integral jet model (momentum balance per ring pair, losses through the rings), then an `fvOptions` body-force export for Aeromant/OpenFOAM | integral model; the export prepared with mocks |
| `thruster` | ring-electrode builders: concentric emitter/ground rings with radius, axial offset (convergence), gap, tooth pitch, section → the 2-D axisymmetric section Fulgur solves, and the 3-D CAD inputs the lab's design uses | Mark IV, 4B, 4C presets |
| `validate` | the published benchmarks as tests: Peek's onset; wire-to-cylinder `I(V)`; Masuyama & Barrett (2013) `T/P` vs `d/V`; Moreau et al. (2013) velocity profiles | each a test with a tolerance |
| `mock` | fixed numbers under `vegeta.mock`, so the lab's tests run without the solvers | — |

`RISK.md` from the start: high voltage (tens of kV), ozone and NOx production, fire on resin frames (the 4B melted),
RF noise; the toolkit reports the supply current and the ozone rate with every result.

## The lab: `labs/plasma`

A new lab, *what pushes air without blades*, registered in `tools/labs.py`. Stage by stage, as the others:

* `designs/halka.py` — the parametric ring thruster in CadQuery (dedalus): printed frame with spokes and clip
  seats, N plated ground rings, M emitter rings with teeth, hub cone, the convergence as axial offsets; presets
  `mark4`, `mark4b`, `mark4c` from the table above; `bom()` with masses (checked against 209 / 513 / 484 g) and
  the plating areas; `draw3d` pictures;
* `designs/halka_fulgur.py` — the study: for each preset, Fulgur's onset voltage, `I(V)`, thrust, exit velocity
  and power against the video's columns; the serrated/square and the gap/voltage experiments reproduced;
* `designs/halka_cfd.py` — the Aeromant case with the EHD body force, prepared, run where OpenFOAM is;
* `tests/` — every part builds; the presets' masses; Fulgur's numbers within the stated tolerance of the video
  (loose: ±30 % on velocity, the power within a factor of 2 of 50 W for the Mark IV);
* later `sims/`: the thruster on a vehicle in MuJoCo with the thrust curve as the actuator.

The design is **Halka** (ion ring; decided in round 1).

## Phases

| phase | what | done when |
|---|---|---|
| **P0** | this plan; the branch `dev_plasma` | pushed |
| **P1** ✓ | `vegeta-fulgur` `gas`, `climate`, `corona`, `breakdown`, `supply`, `thruster`: the 0-D/1-D toolkit against the literature | 17 benchmark tests pass; `docs/plasma_p1.png` |
| **P2** ✓ | `labs/plasma/designs/halka.py`: the Mark IV / 4B / 4C replicas in CAD, masses, pictures | 21 tests; masses within 25 % of the video (under: the frames are the least certain part); drawings sent |
| **P3** ✓ | `ehd` on FiPy: the corona solved on the ring section; `jet`: the integral jet and the OpenFOAM sources; `halka_fulgur.py`, `halka_cfd.py`, `halka_trends.py` | the coaxial benchmark within 3 % of the exact solution; the Mark 4C solved (thrust = I dz/μ); the three trends have the right sign (+8 % teeth vs +17 %, +20 % spacing-and-grounds vs +26 %, the 4B slower at each step); the CFD case prepared with the sources, run where OpenFOAM is |
| **P4** | the scaling study: thrust, power and mass versus diameter, ring count, stages and voltage; humidity and the surface-flashover limit; the `fvOptions` export | a design chart: thrust per kg and per W reachable, honest about the T/W ≥ 1 wall |
| **P5** | the bigger designs: a multi-stage, multi-ring thruster sized from P4, its HV supply's mass, the vehicle concept and MuJoCo sim | the next plan |

## Decisions taken in v0

* the toolkit is `vegeta-fulgur`, a separate distribution like `vegeta-bionic`, numpy + scipy, no CFD inside;
* 2-D axisymmetric is the resolution: the rings are axisymmetric but for the teeth and spokes, which enter as
  effective parameters (tooth pitch sets the emitter's onset and current per length, spokes as blockage);
* the replica keeps the video's geometry as *parameters with estimated defaults*, each marked estimated until
  measured;
* validation order: literature → video; the video's numbers are checks, not fits.

## Decisions (round 1)

| question | decision |
|---|---|
| the design's name | **Halka** (ion ring): `labs/plasma/designs/halka.py`, the video's generations as its presets `mark4`, `mark4b`, `mark4c` |
| the P5 target | attempt a demonstrator; if the turbine is too weak, scale it up; if it does not work, no demonstrator — the scaling study (P4) decides, honestly |
| the HV supply | modelled simply: a voltage source with a series resistance, a current limit and a stored capacitance; mass and efficiency from a curve, not a circuit |
| humidity | two design conditions: Warsaw year-round (the monthly RH and temperature range) and dry indoor air |
| CFD | OpenFOAM via Aeromant: the P4 body-force export is run, not only prepared |
| the PDE solver | **not our own**: FiPy (NIST's finite-volume library; Poisson + drift-diffusion with implicit convection, axisymmetric grids, Gmsh meshes) for the field and the ions; Gmsh for the ring sections; OpenFOAM for the jet. Fulgur's `ehd` is the Kaptzov loop and the bookkeeping around them, not a solver |

## P1 done (2026-11-04)

`libs/vegeta-fulgur` (`vegeta.fulgur`): `gas` (Air: density, mobility, Peek, breakdown and stability fields, net
ionisation), `climate` (Warsaw's months, a rainy day, the dry room), `corona` (the 1-D corona: onset, the
Townsend current law per metre of edge, thrust `I d/μ`, the jet), `breakdown` (the four criteria → spark margin),
`supply` (operating point with fold-back, input power, mass, a spark's energy), `thruster` (ring thrusters, the
Mark IV / 4B / 4C presets with their geometry estimated), 17 tests, `RISK.md`, `docs/plasma_p1.png`.

The 1-D model has one fitted factor, `collection` = 0.6, set so the Mark IV draws 50 W at a spark margin above 1:
that gives 18 kV, 2.8 mA, 17 gf, a mean exit velocity of 3.0 m/s and 4.2 m/s at the focal point (the video: 4.1).
It then **predicts** the Mark 4C: 25 gf at 21 kV / 71 W with 3.6 m/s at the focal point (the video: 25 gf, 3.7 m/s),
at a spark margin of 1.06 — "as much power as I want before it self-destructs". Both run at ~1.5 MV/m mean field,
3.3–3.5 mN/W. Through Warsaw's year the 4C's limit is the air gap from March to September (~31 gf at 23 kV) and the
wet resin surface from October to February (23–30 gf); a rainy day halves it (11 gf at 14 kV); the dry room gives
29 gf. A 50 W supply without a current trip delivers 1.5 kJ into a 30 s arc, seven times what softens a gram of
resin — the 4B's frame; a 2 mA-limited one under 1 J.

Estimated, not measured: the rings' radii and gaps (11 mm on the IV, 15 mm on the 4B/4C), the tooth tip radius
(40–50 µm), the creepage (14–18 mm), the focal gain (1.4), the surface limits; the video never states its voltage.

## P2 done (2026-11-04)

`labs/plasma` (registered in `tools/labs.py`): `designs/halka.py` — the parametric ion-ring thruster in CadQuery,
its ring layout read from `vegeta.fulgur.thruster`'s presets (one source of truth for CAD and physics), scaled by
`scale` for P4. Parts: the printed frame (the Mark IV's base plate with grooves and a rim; the 4B/4C's wall, six
spokes, hub cone with an open bore, snap seats for the grounds), the plated ground rings (the IV: 6 × 2 mm strips,
copper; the 4B/4C: 10 × 6 mm D-section, nickel — the section that reproduces the video's 36.6 mL "ground 4"
print), the bronze emitter strips with every tooth drawn on request (1 mm pitch, 1.5 mm tall, pointing downstream
at the grounds), the standoffs (the creepage path), screws and terminals. `masses()` from volumes and materials:
Mark IV 161 g (video 209), 4B 401 g (513), 4C 401 g (484) — all under, the frames being the least certain part; the
4C is not yet lighter than the 4B in the model. `halka_draw3d.py` → `halka_3d.png`, `halka_3d_detail.png`; 21 tests.

## P3 so far (2026-11-04): the corona solved

`vegeta.fulgur.ehd` on **FiPy**: Poisson with the ions' space charge and the ions' steady drift–diffusion
(upwind), the Kaptzov condition held by adjusting the injected charge (warm start, adaptive relaxation; the ions
settle, not only the current). Three geometries: FiPy's cylindrical grid, a planar section with the r weight in
the coefficients (Gmsh meshes: `ring_section` builds a preset's rings with the emitter as a rounded edge of
radius `edge_r`, the teeth's effective tip), and a plain slab.

**Benchmark**: the coaxial wire-to-cylinder corona has an exact space-charge solution; the solver matches its
current within 3 % from 1.3× to 10× onset on both the grid and the Gmsh section, the current is conserved across
the gap to 0.3 %, and `∫ρE dV = I d/μ` holds to 2 %. Townsend's 1914 formula sits 10–15 % under the exact one.

**The Mark 4C solved** (32 k cells, 10–30 s a voltage): onset 17.6 kV for the 0.1 mm modelled edge (the real
teeth start lower: the onset is the least certain output, ±30 %); 20 kV → 1.7 mA, 9.8 gf, 33 W; 23 kV → 3.2 mA,
17.8 gf, 73 W; 26 kV → 4.9 mA, 27.5 gf, 128 W. **The finding: thrust = I·dz/μ**, with dz the grounds' axial
offset (the convergence), within 5 % at every offset tried — the ions travel mostly radially between the rings,
the radial momentum cancels between the inner and outer grounds, and only the axial part is thrust. So the
convergence is not a refinement, it is the whole mechanism of a ring thruster: thrust per watt = dz/(μV) — at 20 kV,
2.0 mN/W at dz = 8 mm, 2.9 at 12, 3.9 at 16, 5.0 at 20 — while the current at a fixed voltage falls with the longer
gap (3.3 → 0.1 mA). The 1-D model now carries the axial projection; its presets' dz were raised to 10–12 mm,
which reconciles the video's thrust and power better than the 8 mm first read off the screenshots. `labs/plasma/
designs/halka_fulgur.py` (sweep, offset sweep, the figure `halka_fulgur.png`; the 1-D numbers under the mocks).

Lessons from the solver, kept as tests: FiPy's cell gradient on a cylindrical grid carries a φ/r term, its face
gradient does not; a constraint added in a loop accumulates; the ions need an outflow on every face but the
emitter's; Gmsh arcs stay under π and a line belongs to one physical group; the hub ring must not touch its
neighbour.

## P3 done (2026-11-04): the jet and the trends

**The jet.** `vegeta.fulgur.jet`: the integral model (mean exit velocity from the momentum flux through the open
exit area, the jet's convergence half-angle from the force field's radial over axial parts, a focal distance and a
capped focal gain — the estimate the CFD replaces) and the CFD export: `force_zones` bins the solved `ρ_q E` into
annular cell zones, `write_openfoam_sources` writes them as `fvOptions` momentum sources (`vectorSemiImplicitSource`,
absolute newtons per zone) with the `topoSetDict` that makes the zones and a `run.sh` with `topoSet` before the
solver. `labs/plasma/designs/halka_cfd.py` prepares the Aeromant case (`rans_ksst_external`, the frame and the
ground rings as the body with the thrust axis turned to +x, a residual 0.5 m/s inflow as the static rotor template
does) and writes the sources into it; the binning keeps the force to 5 %. Not run here (no OpenFOAM): `run.sh` in
the case, mocked in the tests.

**The video's three trends** (`halka_trends.py`, coarse meshes; velocity goes as the square root of thrust):

| trend | video | solved | note |
|---|---|---|---|
| serrated vs square teeth | +17 % velocity | **+8 %** | a 0.1 vs 0.2 mm effective edge at 23 kV: the right sign, half the size — the teeth as an axisymmetric edge is the coarsest part of the model |
| the Mark III's wider spacing and larger grounds | +26 % | **+20 %** | in two halves: the axial spacing grown at each layout's spark limit alone gives −2 % (thrust ∝ dz, current ∝ V(V−V0)/d², at the limit they cancel); the larger grounds give the rest (+22 %): a bigger cathode collects more current |
| the 4B: more axial gap at the voltage ceiling | slower, twice | **−9 %, then −16 %** | dz 8 → 12 → 16 mm at 24 kV: the current collapses faster than dz grows — the hollow-bolt threading moved the grounds the wrong way for a supply at its ceiling |

A fourth check, free: the first attempt at trend 2 moved the *radial* pitch and gave −10 %; the Mark III's
spacing is axial (a straight multi-stage thruster), and the model only agreed once the geometry was the right
one — a reminder that the preset's geometry, not the solver, is now the limit of the agreement.

## Arcs: predicted, not simulated

Fulgur solves the corona (steady, cold, at the teeth) and predicts the breakdown that ends it; it does not simulate
the arc channel itself (a thermal plasma at thousands of kelvin whose simulation tells a designer nothing). Per
design point and humidity it reports a **spark margin** = breakdown voltage / operating voltage, from:

1. **inception** — the Meek/Raether streamer criterion along field lines from the teeth: the integral of air's net
   ionisation coefficient exceeding ~18–20, on the field *with* the corona's space charge (which shields it: why
   30 kV over 20 mm can run without a spark);
2. **propagation** — the streamer stability field (positive streamers need a mean ~0.5 MV/m in dry air to cross),
   checked against standard point–plane breakdown curves as a test;
3. **surface flashover** — the tangential field and creepage distance along each printed support between rings,
   derated by humidity (surface conduction on wet resin: the video's 78 % day);
4. **the spark's energy** — from the supply model, ½CV² plus current × arc voltage × time until the supply folds
   back, against the resin's melting energy: what melted the 4B's frame, and what current limit would have saved it.

The thruster is designed to a margin, not to the edge.

## Sources

* Plasma Channel, *I Built A Bladeless Turbine For Silent Flight* (transcript and screenshots in this session).
* Masuyama, K., Barrett, S. R. H. (2013). On the performance of electrohydrodynamic propulsion. *Proc. R. Soc. A*.
* Moreau, E., Benard, N., Lan-Sun-Luk, J.-D., Chabriat, J.-P. (2013). Electrohydrodynamic force produced by a wire-to-cylinder DC corona discharge in air at atmospheric pressure. *J. Phys. D*.
* Peek, F. W. (1929). *Dielectric Phenomena in High Voltage Engineering* (the onset law).
