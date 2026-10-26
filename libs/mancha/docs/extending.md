# Extension points

Mancha's first phase is the solver only. These are the places later work plugs in, without rewriting the
solver.

## Collision models (TRT, MRT, cumulant, LES)

Subclass `mancha.collision.CollisionModel`: `kind`, `viscosity`, `collide(f, rho, u, c, w, force)` (the NumPy
reference) and `parameters()`; register it in `mancha.collision._MODELS` (checkpoints rebuild it by `kind`).
For the GPU, write a fused kernel with the same signature as `stream_collide_bgk` and add it to
`triton_backend.KERNELS`. The pull/boundary helper `_pull` is shared, so only the relaxation part
(`_relax_store`) changes. TRT is the natural next step: it removes the τ-dependent wall slip of bounce-back.

Turbulence: WALE is implemented (`mancha.turbulence`, per-node τ from the velocity gradient); another eddy
viscosity model (Vreman, σ-model) is a function of the same gradient plus a branch in the kernel's `_wale`.
Regularized BGK exists; the cumulant model (D3Q27) is the next collision model for high Reynolds numbers.

## Boundaries

New face kinds get a code in `boundaries.Kind` (its value is its precedence), a branch in
`NumpyBackend._stream` and in `triton_kernels._pull`. Non-uniform inlet profiles need a per-node velocity
array instead of the per-face table.

**Rotating geometry in a rotating frame** is implemented (`mancha.frames`): enough for an isolated propeller
in uniform inflow. **Moving geometry in the lab frame** (rotor–stator, installed propellers) still needs: a per-link wall velocity in the bounce-back
(`+ 6 w_i ρ₀ c_i·u_w(x)`), refilling of nodes that become fluid, interpolated bounce-back (Bouzidi) for
second-order curved walls, and the force by momentum exchange in the moving frame. The link mask is where a
per-step update would go; the solid mask is static today by design.

## Output and coupling

`Simulation.populations()`, `density()`, `velocity()` return NumPy arrays; `Simulation.run(callback=...)`
calls back every `callback_interval` steps — the hook for probes (pressure time series for aeroacoustics, FW-H
outside the solver), averaging or adaptive stopping.

## Lattices

Only D3Q19. A D3Q27 lattice (better isotropy for cumulants and turbulence) would be a new dataclass with the
same fields plus a new kernel set; the NumPy backend is lattice-generic except for the D3Q19 check in
`Simulation`.
