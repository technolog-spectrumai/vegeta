# Mancha

A lattice Boltzmann solver for three-dimensional incompressible flow on NVIDIA GPUs, written in Python and
Triton, with a readable NumPy reference implementation. Named after La Mancha, Don Quixote's country of
windmills. Developed for Basilisk Systems as a standalone scientific library; long-term targets are propeller
aerodynamics, vortex dynamics, hydrodynamics and aeroacoustics.

The solver only: D3Q19, BGK and regularized BGK, static geometry, a rotating reference frame (geometry that
turns with the frame, e.g. an isolated propeller) and WALE LES. No moving geometry in the lab frame, grid
refinement, acoustics, OpenFOAM coupling or machine learning yet ([docs/extending.md](docs/extending.md)
lists the extension points).

```python
from mancha import Simulation, D3Q19

sim = Simulation(
    lattice=D3Q19(),
    shape=(128, 128, 128),
    viscosity=0.02,          # lattice units (dx²/dt): tau = 3·0.02 + 1/2 = 0.56
    backend="triton",        # or "numpy" (the CPU reference)
)

sim.set_boundary("x-", "velocity", velocity=(0.05, 0, 0))   # lattice units (dx/dt), Ma ≈ 0.09
sim.set_boundary("x+", "pressure", density=1.0)
sim.set_boundary("y-", "wall")
sim.set_boundary("y+", "wall")
sim.set_boundary("z-", "wall")
sim.set_boundary("z+", "wall")

sim.run(steps=10000)

velocity = sim.velocity()    # NumPy [3, nx, ny, nz]
density = sim.density()      # NumPy [nx, ny, nz]

sim.save("results/")         # density.npy, velocity.npy, metadata.json, fields.vti
```

This is `examples/duct_flow.py` (`--backend numpy --shape 32 32 32 --steps 2000` runs it on a CPU). Its
small-grid version is a test (`tests/test_simulation.py::test_readme_example_runs`).

**Units.** The solver works in lattice units (`dx = dt = 1`, reference density 1); `viscosity`, velocities
and densities above are lattice values. `UnitConverter` maps to and from SI and is the only place physical
units appear — see `examples/physical_units.py` and [docs/numerics.md](docs/numerics.md#units).

## Features

| | |
|---|---|
| Lattice | D3Q19, second-order equilibrium |
| Collision | BGK/SRT (`viscosity=`, `tau=` or `collision=`), regularized BGK (`RegularizedBGK`, stable as τ → ½); `CollisionModel` interface for TRT/MRT/cumulant |
| Forcing | uniform body force, Guo scheme (`force=`) |
| Rotating frame | `frame=RotatingFrame(omega, center)`: Coriolis + centrifugal forcing, lab-frame face velocities, `absolute_velocity()` |
| LES | `turbulence=WALE(c_w)`: WALE eddy viscosity (lab-frame gradient in a rotating frame) |
| Streaming | pull, fused with collision, double-buffered, structure-of-arrays `f[q, x, y, z]` |
| Boundaries | periodic; halfway bounce-back walls, stationary or moving (tangential); velocity inlet; pressure outlet (anti-bounce-back); static voxel solids — [docs/boundaries.md](docs/boundaries.md) |
| Backends | `numpy` (float64 or float32, the reference), `triton` (FP32, CUDA; CPU only under Triton's interpreter) |
| Safety | τ ≤ ½ refused, τ near ½ warned, Mach > 0.1 warned, ≥ cs refused; NaN/Inf or `|u| > cs` raise `NumericalInstabilityError`; VRAM checked before allocating (`InsufficientMemoryError`) |
| Output | NumPy arrays, `.npy`, VTK ImageData `.vti` (no dependency; `to_pyvista` with PyVista), JSON metadata, monitored history, checkpoints (`checkpoint` / `Simulation.restore`, also across backends), plots |
| Diagnostics | mass, kinetic energy, errors and observed order, momentum-exchange force and moment on solids or parts of them (gauge pressure) |

## Install and test

```bash
pip install -e libs/mancha            # numpy only: the CPU reference
pip install -e "libs/mancha[gpu]"     # + torch and triton for the GPU backend
pip install -e "libs/mancha[vtk,plot,test]"

libs/mancha/test.sh                   # all tests; Triton tests run on CUDA, are skipped otherwise
libs/mancha/test.sh -m "not slow"     # the fast subset (seconds)
libs/mancha/test.sh --interpret       # the Triton tests on the CPU through Triton's interpreter (minutes)
```

The tests are **regression tests** on tiny grids with tolerances that catch breakage. The **scientific
validation** — refinement series against exact solutions and published data — is
`benchmarks/validate.py`; its measured results are in [docs/validation.md](docs/validation.md). Throughput is
`benchmarks/performance.py` ([docs/gpu.md](docs/gpu.md)).

## Status

* NumPy reference: complete for the features above and validated (Taylor–Green, Poiseuille, channel, Re = 100
  cavity against Ghia et al., sphere drag characterised) — [docs/validation.md](docs/validation.md).
* Triton backend: agrees with the reference to FP32 round-off for every boundary kind, under Triton's
  interpreter; compiles for sm_80/sm_90 with no register spills. **It has not yet been run on a GPU**, so there
  are no GPU performance figures. Plan and progress: [todo.md](todo.md).

## Layout

```
src/mancha/
  lattice/d3q19.py        velocities, weights, opposites
  collision/              equilibrium, moments, CollisionModel, BGK (+ Guo forcing)
  boundaries/             face conditions, precedence rule, voxel masks
  backends/               numpy_backend.py (reference), triton_backend.py + triton_kernels.py
  simulation.py           the public API
  units.py                UnitConverter
  diagnostics.py          mass, energy, errors, convergence order, momentum-exchange force
  io/                     VTK writer, plots
  validation/             the benchmark cases and reference data (shared by tests and benchmarks)
benchmarks/               validate.py, performance.py
examples/                 duct_flow.py (the example above), physical_units.py, cavity.py, sphere_wake.py
docs/                     numerics, boundaries, gpu, validation, extending
```

Requires Python ≥ 3.11 and NumPy; PyTorch + Triton, PyVista and matplotlib are optional.
