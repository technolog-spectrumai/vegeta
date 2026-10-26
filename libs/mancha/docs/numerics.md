# Numerics

Everything inside the solver is in **lattice units**: `dx = dt = 1`, reference density `ρ₀ = 1` (configurable),
speed of sound `cs = 1/√3`. Physical units enter only through `mancha.UnitConverter` (see [units](#units)).

## Lattice — D3Q19

19 velocities `c_i`: the rest velocity (weight 1/3), the 6 face neighbours (1/18) and the 12 edge neighbours
(1/36). Order (`mancha.lattice.d3q19`): 0 rest; 1–6 = ±x, ±y, ±z; 7–18 the edges in the planes xy, xz, yz;
every moving velocity is followed by its opposite. The weights reproduce the isotropic moments up to fourth
order (`tests/test_lattice.py`), which is what the second-order equilibrium needs.

## Equilibrium and collision

    f_i^eq = w_i ρ [1 + 3 c_i·u + 9/2 (c_i·u)² − 3/2 u·u]

**BGK / SRT** (`mancha.collision.BGK`): `f_i* = f_i − (f_i − f_i^eq)/τ`, with the kinematic viscosity
`ν = cs² (τ − ½) = (τ − ½)/3`. `τ ≤ ½` is refused (`InvalidParameterError`); `τ < 0.51` warns (BGK is
close to its stability limit); `τ > 2` warns (the halfway-bounce-back wall location drifts with τ).

**Body force** (Guo, Zheng & Shi 2002): a uniform force density `F` (`Simulation(force=...)`) uses
`u = (Σ f_i c_i + F/2)/ρ` in the equilibrium and adds
`S_i = (1 − 1/2τ) w_i [3 (c_i − u) + 9 (c_i·u) c_i]·F`. It is second-order accurate and is what the
Poiseuille reference uses.

**Collision interface.** `CollisionModel` has `collide(f, rho, u, c, w, force)` (NumPy reference),
`viscosity` and `parameters()`; the Triton backend maps `model.kind` to a fused kernel. TRT, MRT, cumulant or an
LES model with a per-node τ are new subclasses plus a kernel each — streaming and boundaries do not change
([extending.md](extending.md)).

## Time step

The stored state is the **post-collision** population `f*` at time `t`. One step, for every node `x`:

1. **pull**: `f_i(x, t+1) = f_i*(x − c_i, t)` — or a boundary rule when `x − c_i` is solid or outside a
   non-periodic face ([boundaries.md](boundaries.md));
2. **moments**: `ρ = Σ f_i`, `u = (Σ f_i c_i + F/2)/ρ`;
3. **collide** and write `f*(x, t+1)` into the second buffer (double buffering); then swap the buffers.

Moments of the stored `f*` equal the pre-collision moments (BGK conserves mass and momentum), except that a
force adds `F/2` to the momentum; `Simulation.velocity()` subtracts it. Solid voxels hold a rest equilibrium and
report zero velocity; fluid nodes never read them.

## Memory layout

Structure of arrays: `f[i, x, y, z]`, C order (z fastest), one contiguous block of `N = nx·ny·nz` values per
direction, two such blocks of 19 (double buffering): `2 · 19 · 4 · N` bytes in FP32 (2.4 GiB at 256³). For
direction `i` the pull reads `f*[i, n − offset(c_i)]` with `offset = (cx·ny + cy)·nz + cz`: threads handling
consecutive nodes read consecutive addresses (a shifted but contiguous run), and write `f[i, n]` contiguously.
Both are coalesced; the shift costs at most one extra memory transaction per warp and direction. The NumPy
backend uses the same layout (`np.roll` per direction).

## Units

`UnitConverter(dx, dt, rho)` maps lattice quantities to SI: length `n·dx`, time `steps·dt`,
velocity `u·dx/dt`, viscosity `ν·dx²/dt`, gauge pressure `cs² (ρ − ρ₀) ρ_phys (dx/dt)²`, force
`F ρ_phys dx⁴/dt²`. `from_velocity(dx, U, u_lattice)` fixes the Mach number (keep `u_lattice ≲ 0.1`);
`from_viscosity(dx, ν, ν_lattice)` fixes τ. Either way the Reynolds number is the same in both systems.
`examples/physical_units.py` is a worked case.

## Accuracy and stability (what to expect)

* Second order in space for smooth flows at fixed τ and lattice velocity (Taylor–Green: observed order ≈ 2,
  [validation.md](validation.md)).
* Compressibility error ∝ Ma²: lattice velocities above 0.1 warn, at or above `cs` are refused.
* BGK becomes unstable as τ → ½ at finite velocity (high Reynolds number on a coarse grid). The run checks every
  `check_interval` steps and raises `NumericalInstabilityError` on non-finite values or `|u| > cs` — it never
  continues with a broken state. A blow-up between checks is detected at the next check.
* FP32 (Triton) loses about 5–6 significant digits of the populations; density deviations of order 1e-6 are at
  round-off. A "shifted" storage (`f − w_i ρ₀`) would recover digits for nearly incompressible flows — not
  implemented yet.
