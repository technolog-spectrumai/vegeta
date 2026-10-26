# Validation

Two kinds of evidence, kept apart:

* **Regression tests** (`tests/`, run in CI): tiny grids, loose tolerances, seconds. They detect when something
  breaks; they are not accuracy claims.
* **Validation benchmarks** (`benchmarks/validate.py`, `benchmarks/openfoam/cube_in_duct.py`): refinement series
  against exact solutions, published data and an independent solver. The numbers below were measured with those
  scripts — NumPy backend, float64, on the development container's CPU (x86-64), Mancha 0.1.0 — and their JSON
  records are written to `results/` when the scripts run; the records behind this page are kept in
  [`records/`](records/). Nothing is quoted that was not measured.

Lattice units throughout unless stated. "Error" is relative unless stated.

## 1. Uniform rest state and uniform flow — exact steady solutions

32³ periodic box, 1000 steps, ν = 0.1.

| | max velocity error | max density error | mass drift |
|---|---|---|---|
| rest | 0 | 2.2e-16 | 0 |
| u = (0.05, 0.03, −0.02) | 6.2e-17 | 1.1e-16 | 0 |

Preserved to round-off: the equilibrium, streaming and collision are consistent.

## 2. Taylor–Green vortex — energy decay against the exact solution

Two-dimensional vortex `u = U sin kx cos ky`, `v = −U cos kx sin ky` (exact Navier–Stokes solution, energy
∝ exp(−2ν(kx²+ky²)t)), periodic n × n × 1, ν = 0.02, U = 0.04, run for one energy half-life.

| n | steps | decay-rate error | velocity L2 error | mass drift |
|---|---|---|---|---|
| 16 | 56 | 7.4e-2 | 2.5e-2 | −5.8e-15 |
| 32 | 225 | 1.8e-2 | 6.1e-3 | −2.3e-14 |
| 64 | 899 | 3.7e-3 | 1.4e-3 | −8.9e-14 |
| 128 | 3596 | 2.8e-4 | 5.7e-4 | −3.6e-13 |

Observed order: 2.6 (decay rate), 1.85 (velocity field). The lattice velocity and τ are held fixed while the
grid is refined, so the compressibility error (∝ Ma²) does not shrink; the decay-rate error at n = 128 drops
faster than second order, most likely because spatial and compressibility errors of opposite sign partly
cancel. Expect second order, not better.

## 3. Poiseuille channel — body force, halfway bounce-back walls

Walls at `y−`/`y+`, periodic x and z, Guo forcing, peak velocity 0.02, steady to a relative change < 1e-10.
L2 error of the profile against `u = F/(2ν) y(h − y)`, `h = ny`:

| ny | τ = 0.8 | τ = 1.0 | τ = 1.5 |
|---|---|---|---|
| 8 | 1.11e-2 | 7.13e-3 | 9.27e-2 |
| 16 | 2.78e-3 | 1.78e-3 | 2.32e-2 |
| 32 | 6.95e-4 | 4.46e-4 | 5.79e-3 |
| 64 | 1.74e-4 | 1.11e-4 | 1.45e-3 |
| observed order | 2.000 | 2.000 | 2.000 |

Exactly second order at every τ; the τ-dependence of the constant is the BGK bounce-back wall slip
([boundaries.md](boundaries.md)). At τ = ½ + √(3/16) the profile is exact to round-off (a regression test
checks this). Mass drift over the runs: below 6e-12.

## 4. Channel with velocity inlet and pressure outlet

Quasi-2D (one periodic z layer), ny = 32, length 256, ν = 0.1, u_in = 0.02, steady (27 000 steps).

* Profile at x = 192 against the parabola of the same mean: L2 error 4.5e-4; peak / mean = 1.498 (exact 1.5).
* Mass flux constant along the channel to 2.8e-8.
* Delivered flux 0.6333 against u_in · ny = 0.64: −1.0 % (one third of a cell of the 32), from the wall
  precedence at the inlet edges — the inlet condition imposes the velocity, not the flux
  ([boundaries.md](boundaries.md#limitations)).

## 5. Lid-driven cavity, Re = 100 — against Ghia, Ghia & Shin (1982)

Two-dimensional (one periodic z layer), lid speed 0.05, steady to a relative change < 1e-8. Maximum absolute
difference from Ghia's tabulated centreline velocities (normalised by the lid speed), and the distance of the
primary vortex centre from theirs:

| n | steps | max \|Δu\| (x = ½) | max \|Δv\| (y = ½) | vortex centre | distance |
|---|---|---|---|---|---|
| 32 | 22 000 | 0.0104 | 0.0061 | (0.609, 0.734) | 0.0078 |
| 64 | 42 000 | 0.0055 | 0.0078 | (0.617, 0.727) | 0.0078 |
| 128 | 79 000 | 0.0053 | 0.0088 | (0.613, 0.730) | 0.0056 |

Ghia et al.: vortex at (0.6172, 0.7344). The differences stop shrinking at about 0.5–0.9 % of the lid speed
from n = 64: that is the level of Ghia's own data (a 129 × 129 grid, tables to 4–5 digits) plus the fixed Ma²
error of the lid speed, so the comparison cannot resolve Mancha's error below it. The vortex centre is located on
the node grid (spacing 1/n). Mass is conserved to 1e-11 in the closed cavity.

## 6. Sphere — drag and wake (characterisation, not validation)

Velocity inlet, pressure outlet, laterally periodic box 8 D × 4 D × 4 D (a square array of spheres 4 D apart,
blockage 4.9 %), sphere voxelised from its centre test, Re = U D / ν.

| D [cells] | Re | steps | Cd (momentum exchange) | Schiller–Naumann (unconfined) | recirculation |
|---|---|---|---|---|---|
| 8 | 20 | 13 500 | 3.47 | 2.61 | none |

The finer runs of the series (D = 12, 16 and Re = 100) were stopped to free the machine for the OpenFOAM
comparison; `python benchmarks/validate.py sphere` completes them (about an hour per run on one CPU core with the
NumPy backend — a GPU job). The 33 % excess over the correlation is expected in sign and plausible in size: the
periodic neighbours and the lateral confinement raise the drag, and D = 8 is coarse. No separation at Re = 20 is
consistent with the onset of the wake near Re ≈ 20–24 for an unconfined sphere. Do not read Cd from this case as
a prediction.

## 7. Independent solver: OpenFOAM, cube in a duct

Steady flow past a wall-mounted cube in a duct (Re ≈ 39), compared point by point with OpenFOAM
(simpleFoam) on identical grids: velocity fields within 0.015 U_b RMS beyond the inlet region at n = 12,
pressure drop within 1.4 %, reattachment length within 2 %, drag 7 % higher than OpenFOAM at the same grid and
5 % above OpenFOAM's finest grid, with the two codes converging towards each other. Details: [openfoam.md](openfoam.md).

## 8. Rotating frame and LES

Measured with the NumPy backend (float64); the same cases are regression tests in `tests/test_frame_les.py`,
and `tests/test_triton.py` checks the Triton kernels against them (agreement to float32 round-off).

**Fluid at rest in the lab, seen from the rotating frame** (6 × 20 × 20, ω = 0.004, far-field velocity faces,
ν = 0.05, 2 000 steps): a rigid rotation up to 0.046 lattice units. Lab-frame velocity left: 8.1e-5; density
variation 9.7e-6. The density is the discriminating quantity — the Coriolis force of a rigid rotation is a
gradient that pressure can absorb: with the Coriolis sign flipped the density varied by 5.8e-3, without it by
2.9e-3 (300–600 × more).

**Co-rotating container** (a cylinder static in the frame, ω = 0.006): relative velocity 7e-9; the density
follows the centrifugal profile exp(3ω²r²/2) to 2.4e-7 across a 1.05 % variation.

**Uniform axial flow through the rotating frame** (inlet, outlet, far-field faces, U = 0.03): lab velocity
error below 1e-3.

**Regularized BGK, forced Poiseuille** (L2 error of the profile; BGK in brackets):

| τ | ny = 8 | ny = 16 | ny = 32 |
|---|---|---|---|
| 0.6 | 1.57e-2 (2.03e-2) | 3.92e-3 (5.06e-3) | 9.81e-4 (1.27e-3) |
| 0.8 | 4.28e-3 (1.11e-2) | 1.07e-3 (2.78e-3) | 2.67e-4 (6.95e-4) |
| 1.2 | 1.85e-2 (3.45e-2) | 4.64e-3 (8.63e-3) | 1.16e-3 (2.16e-3) |

Second order, 1.3–2.6 × smaller than BGK. Couette flow exact (≤ 2e-12) with both.

**Stability at τ → ½** (3D Taylor–Green, 32³, U = 0.05, ν = 1e-4, Re ≈ 2 500 on the box scale, 3 000 steps):
BGK blew up within the first few hundred steps; BGK + WALE later, but also blew up; regularized BGK ran stably
(E/E₀ = 0.195 at the end) and so did regularized BGK + WALE (E/E₀ = 0.167, the extra decay being the eddy
viscosity). This is a stability result, not a turbulence validation: the reference test for LES (the Re = 1 600
Taylor–Green dissipation curve at 128³–256³) needs GPU time.

## Findings that changed the code or the docs

* **Moving-wall density.** With the local density in the moving-wall correction, the closed cavity gained 0.8 %
  mass (n = 32) through uncancelled lid-corner terms. The reference density is used now; mass is conserved.
* **WALE and rotation.** WALE is not zero in solid-body rotation (only in pure shear). In a rotating frame the
  far field is a solid-body rotation, so WALE is evaluated on the lab-frame gradient; a test checks the far field
  gets no eddy viscosity.
* **Forces on parts of a body.** Momentum exchange sees the absolute pressure cs²ρ; on a part of a body (a blade
  on a hub, the cube on the floor) the unwetted contact face left it uncancelled — half a cube showed 33 × the
  whole cube's drag. Forces are now gauge (relative to the reference density); closed bodies are unchanged.
* **Regularized collision with a force** needed the first-order Hermite term (see numerics.md); found by the
  Poiseuille test, which did not converge without it.
* **A wrong exponent in the GPU WALE kernel** ((S^d:S^d)^{3/4} instead of ^{5/4}) was caught by the CPU/GPU
  agreement test; a direct test of the kernel's WALE viscosity now guards it.
* **Pressure outlet at low viscosity.** In a duct walled on all four sides, the anti-bounce-back outlet
  oscillates without settling (density at the outlet swinging between 0.63 and 1.38) at ν = 0.005 (τ = 0.515),
  4 cells per cube height; it is steady at ν ≥ 0.0075 in the same duct, and the quasi-2D channel is steady even at
  ν = 0.005. The oscillation stays below the speed of sound, so the instability check does not catch it; the
  steady-state loops then never converge. Keep τ ≳ 0.52 with a pressure outlet in walled ducts, or refine.

## Performance

No GPU was available during development: the Triton backend has been verified for correctness (interpreter,
ahead-of-time compilation for sm_80/sm_90) but **has no measured performance**. Run
`benchmarks/performance.py` on a GPU and record the results here with the GPU named.
