# Boundary conditions and their limitations

Every face of the box (`x-`, `x+`, `y-`, `y+`, `z-`, `z+`) has one condition; all start periodic.

```python
sim.set_boundary("x-", "velocity", velocity=(0.05, 0, 0))   # lattice units
sim.set_boundary("x+", "pressure", density=1.0)
sim.set_boundary("y+", "wall", velocity=(0.1, 0, 0))         # moving lid (tangential only)
sim.set_boundary("y-", "wall")
sim.set_solid(mask)                                          # bool [nx, ny, nz], True = solid
```

All conditions act on the *pull*: the population arriving at node `x_b` along `c_i` would come from
`x_b − c_i`; when that source is solid or outside the box on a non-periodic face, the rule below supplies it
(`ī` is the opposite direction, `f*` the post-collision populations of the previous step). The walls are
**halfway**: half a cell beyond the last fluid node, so a channel of `ny` nodes has width `ny`.

| Kind | Rule | Notes |
|---|---|---|
| periodic | `f_i = f_i*(x_b − c_i)` wrapped | faces must be paired (`x-` periodic ⇔ `x+` periodic) |
| wall | `f_i = f_ī*(x_b)` | halfway bounce-back, no slip |
| moving wall | `f_i = f_ī* + 6 w_i ρ₀ c_i·u_w` | Ladd; `u_w` must be tangential (refused otherwise) |
| velocity | `f_i = f_ī* + 6 w_i ρ₀ c_i·u_in` | bounce-back with a wall velocity |
| pressure | `f_i = −f_ī* + 2 w_i ρ_out [1 + 9/2 (c_i·u)² − 3/2 u²]` | anti-bounce-back, `u = u*(x_b)` |
| solid voxel | `f_i = f_ī*(x_b)` | halfway bounce-back on the staircase |

## Edges and corners: precedence

A diagonal direction at an edge or corner node can leave the box through two or three faces at once. The
condition with the highest precedence supplies it:

    wall > moving wall > velocity > pressure > periodic      (ties: x before y before z)

Both backends implement exactly this (`mancha.boundaries.conditions.link_groups`, and the key computation in
`triton_kernels._pull`); `tests/test_boundaries.py` and `tests/test_triton.py` check it.

## Limitations

**Walls (halfway bounce-back).** Second order for straight walls aligned with the grid, but with BGK the
effective wall position depends on τ: Poiseuille flow is exact only at `τ = ½ + √(3/16) ≈ 0.933`, and the
slip error grows like `(τ − ½)²` (a viscosity-dependent wall shift; TRT with a "magic" parameter removes it).
Curved surfaces become staircases: first order in the geometry, with the effective size differing by up to half
a cell (`voxel_volume_ratio` in the sphere case reports the volume error).

**Moving wall.** Only tangential velocities (no mass flux through the wall). The density in the correction is
the reference `ρ₀`, not the local one — this keeps the mass of a closed cavity exactly constant (with the local
density, the uncancelled lid-corner terms changed the cavity mass by ~0.8 % at n = 32) at the cost of an
O(Ma²) error in the momentum injected. At the lid–side-wall corners the side wall wins (precedence), so the
velocity jumps there: the classic cavity corner singularity, resolved only to the grid.

**Velocity inlet.** The inlet velocity is imposed at the halfway plane `x = −½` through the bounce-back
correction, not at the node; the node velocity is close to but not equal to `u_in`. On the edges where an inlet
meets a wall, the wall wins, so the delivered mass flux is slightly below `u_in · area`: measured 2.1 % lower
for `ny = 16` in the quasi-2D channel (`flux_inlet` = 0.3133 vs 0.32), i.e. one third of a cell of the 16 in total.
Measure the flux (`validation.channel`) rather than assuming it. The incoming profile is uniform — no
prescribed profiles yet — and the condition is not non-reflecting: pressure waves reflect from it.

**Pressure outlet.** Anti-bounce-back fixes the density at the halfway plane to `ρ_out` using the velocity of
the boundary node from the previous step (no extrapolation): first-order in time for the velocity it uses. It is
reflecting for acoustic waves and assumes the flow leaves roughly normal to the face; backflow through the
outlet (a wake reaching it) is not stabilised and can blow up — keep the outlet several body sizes downstream.
At low viscosity it can also oscillate without blowing up: in a duct walled on all four sides it never settled at
τ = 0.515 and was steady from τ = 0.5225 up ([validation.md](validation.md#findings-that-changed-the-code-or-the-docs));
the oscillation stays subsonic, so the instability check does not flag it. Keep τ ≳ 0.52 there.
Where the outlet meets an inlet-free periodic face, the outlet applies; where it meets a wall, the wall applies.

**Inlet/outlet intersections.** A domain where the `x-` inlet and a `y-` pressure face share an edge resolves
the edge links as velocity (higher precedence). Such configurations are allowed but untested against reference
data; avoid them where accuracy at the edge matters.

**Solids.** Static only; no moving or rotating geometry yet (no per-link wall velocity, no interpolated
bounce-back). Solid voxels must not touch a non-periodic face if their force is to be measured
(`momentum_exchange_force` ignores links that leave the box). A one-voxel-thick plate blocks the flow but has no
interior: both sides bounce independently, which is correct for halfway bounce-back.

**Force on solids.** Momentum exchange `F = Σ 2 c_i f_i*(x)` over fluid→solid links (stationary body). With a
body force, the Guo correction to the exchanged momentum is not applied (O(F) error).
