# Mancha — todo

GPU Lattice Boltzmann (D3Q19) for Basilisk Systems, named after La Mancha. The solver only: no OpenFOAM, neural
networks, optimisation, acoustics, web or external services. Lives in Vegeta at `libs/mancha/`, a standalone
package (imports nothing from Vegeta or Ogma).

Legend: `[ ]` open · `[x]` done · `[-]` deferred · `[~]` partly done (see note)

## Design decisions (fixed before code)
- State = **post-collision** populations `f*` in a structure-of-arrays block `f[q, x, y, z]` (C order, z fastest).
  One step = pull-stream + boundaries + moments + BGK collision, fused, into a second buffer; then swap.
- Boundaries are applied *during the pull*: the population arriving at fluid node `x` along `c_i` comes from
  `x − c_i`; if that source is a solid voxel or lies outside the domain on a non-periodic face, a boundary rule
  supplies it instead. Precedence when a source leaves through several faces (edges, corners):
  wall > moving wall > velocity > pressure > periodic. The same rule in both backends.
- Lattice units inside the solver; `mancha.units.UnitConverter` is the only place physical units appear.
- Collision models are objects (`CollisionModel`): the NumPy backend calls `collide(f, rho, u)`, the Triton
  backend looks up a kernel by `model.kind` and raises for kinds it has no kernel for.
- NumPy backend is the reference (float64 by default, float32 on request); Triton backend is FP32.
- Triton tests run on CUDA, or on the CPU through Triton's interpreter (`TRITON_INTERPRET=1`); otherwise skipped.

## Phase A — CPU reference (NumPy)
- [x] A1 package skeleton, `pyproject.toml`, conftest, `test.sh`
- [x] A2 `D3Q19` lattice: velocities, weights, opposites, isotropy checks
- [x] A3 second-order equilibrium, moments, BGK collision (`tau`/`viscosity` validation)
- [x] A4 pull streaming with periodic faces, double buffer
- [x] A5 boundaries: halfway bounce-back walls (stationary, moving), velocity inlet (bounce-back with wall
      velocity), pressure outlet (anti-bounce-back), voxel solids
- [x] A6 `Simulation` API, unit converter, NaN / Mach / tau checks, uniform body force (Guo forcing, needed for the Poiseuille reference)
- [x] A7 regression tests: rest state, uniform flow, Taylor–Green decay rate, Poiseuille profile, mass conservation

## Phase B — GPU solver (Triton)
- [ ] B1 fused pull-stream-collide kernel, SoA, configurable block size
- [ ] B2 boundaries in the kernel (same precedence), solid mask
- [ ] B3 GPU macroscopic fields kernel
- [ ] B4 CPU/GPU agreement tests (every boundary type), interpreter fallback
- [ ] B5 VRAM estimate and a clean error before allocating

## Phase C — Validation (scientific benchmarks, `benchmarks/`)
- [ ] C1 Taylor–Green grid refinement (decay-rate error, observed order)
- [ ] C2 Poiseuille channel (L2 error vs analytical, grid refinement)
- [ ] C3 lid-driven cavity Re=100 against Ghia et al. (1982) centreline data
- [ ] C4 sphere in a channel: drag by momentum exchange, wake length; compared with the Schiller–Naumann
      correlation (unconfined) and the blockage stated
- [ ] C5 uniform rest / uniform flow preservation reported with mass drift

## Phase D — Performance
- [ ] D1 `benchmarks/performance.py`: MLUPS, wall time, memory, effective bandwidth for 64³/128³/256³
- [ ] D2 Nsight profiling and tuning (needs an NVIDIA GPU — not available in the development container)
- [ ] D3 documented MLUPS (only measured numbers, with the GPU named)

## Phase E — Usability
- [ ] E1 NumPy output, JSON metadata, checkpoints (save/restore)
- [ ] E2 VTK export (`.vti` through PyVista when installed, a dependency-free writer otherwise)
- [ ] E3 plots: velocity profiles, convergence histories (matplotlib, optional)
- [ ] E4 README, docs (numerics, memory layout, boundaries and their limits, units, extension points)
- [ ] E5 examples (the README example runs as written), CI workflow

## Later (design hooks only, nothing implemented)
- [-] TRT / MRT / cumulant collision (CollisionModel kinds)
- [-] moving and rotating boundaries (interpolated bounce-back, per-link wall velocity)
- [-] turbulence modelling (Smagorinsky: a per-cell tau in the collision model)
- [-] aeroacoustics (probes, FW-H outside the solver)
