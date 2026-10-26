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
- [x] B1 fused pull-stream-collide kernel, SoA, configurable block size (default 128: one node per thread)
- [x] B2 boundaries in the kernel (same precedence), solid links as a per-node bit mask (one 4-byte load)
- [x] B3 GPU macroscopic fields kernel; health check (finite, max |u|) on the device
- [x] B4 CPU/GPU agreement tests (every boundary kind on every axis, mixed precedence, solids, force, block
      sizes, cross-backend checkpoints): `tests/test_triton.py`, run under Triton's interpreter here
- [x] B4b the kernels compile for sm_80 and sm_90 without a GPU, no register spills (`tests/test_triton_compile.py`)
- [x] B5 VRAM estimate before allocating (`InsufficientMemoryError`), OOM during allocation re-raised as such
- [ ] B6 run `tests/test_triton.py` on a real CUDA GPU (no GPU in the development container: only the
      interpreter and ahead-of-time compilation have been exercised)

## Phase C — Validation (scientific benchmarks, `benchmarks/`)
- [x] C1 Taylor–Green grid refinement (decay-rate error, observed order): n = 16–128, order 2.6 / 1.85
- [x] C2 Poiseuille channel (L2 error vs analytical, grid refinement): order 2.000 at τ = 0.8, 1, 1.5
- [x] C2b inlet/outlet channel: profile, flux conservation, delivered flux deficit measured
- [x] C3 lid-driven cavity Re=100 against Ghia et al. (1982): n = 32, 64, 128
- [~] C4 sphere in a periodic box: D = 8 at Re = 20 done; D = 12, 16 and Re = 100 stopped (CPU time) — rerun on
      a GPU (`benchmarks/validate.py sphere --backend triton`)
- [x] C5 uniform rest / uniform flow preservation reported with mass drift
- [x] C6 OpenFOAM cross-check: wall-mounted cube in a duct, Re ≈ 39, identical grids n = 8, 12, OpenFOAM alone
      at n = 16, 24 (`benchmarks/openfoam/cube_in_duct.py`, `docs/openfoam.md`)
- [ ] C7 Mancha cube series at n = 16, 24 on a GPU, to narrow the drag bracket (4.02–4.22)

## Phase D — Performance
- [x] D1 `benchmarks/performance.py`: MLUPS, wall time, memory, effective bandwidth for 64³/128³/256³ (skips
      sizes that do not fit, with the reason)
- [ ] D2 Nsight profiling and tuning (needs an NVIDIA GPU — not available in the development container)
- [ ] D3 documented MLUPS (only measured numbers, with the GPU named)

## Phase E — Usability
- [x] E1 NumPy output, JSON metadata, checkpoints (save/restore, also across backends)
- [x] E2 VTK export (`.vti`, a dependency-free writer; `to_pyvista`; read back through PyVista in the tests)
- [x] E3 plots: velocity profiles, convergence histories (matplotlib, optional)
- [x] E4 README, docs (numerics, memory layout, boundaries and their limits, GPU, validation, OpenFOAM, extending)
- [x] E5 examples (duct flow = the README example, physical units, cavity, sphere wake), CI workflow
      (`.github/workflows/mancha.yml`: NumPy tests, Triton interpreter + sm_80/sm_90 compilation, opt-in GPU job)
- [ ] E6 packaging beyond the editable install (wheel, version policy) — when a consumer needs it

## Later (design hooks only, nothing implemented)
- [-] TRT / MRT / cumulant collision (CollisionModel kinds)
- [-] moving and rotating boundaries (interpolated bounce-back, per-link wall velocity)
- [-] turbulence modelling (Smagorinsky: a per-cell tau in the collision model)
- [-] aeroacoustics (probes, FW-H outside the solver)
