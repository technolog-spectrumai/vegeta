"""Choosing lattice parameters from physical ones with UnitConverter: water in a 10 mm square duct.

    python examples/physical_units.py [--backend numpy|triton] [--cells 24]

The physical problem: water (ν = 1.0e-6 m²/s, ρ = 998 kg/m³) entering a 10 mm square duct at 5 mm/s,
Re = U W / ν = 50. The lattice: ``cells`` across the duct and a lattice inlet speed of 0.04 (Ma ≈ 0.07); that
fixes dx, dt and the lattice viscosity (ν_lu = 0.04 · cells / Re: 24 cells give tau ≈ 0.56). Fewer cells push
tau towards 1/2, where BGK goes unstable — the run then stops with NumericalInstabilityError. The pressure
drop along the duct is reported in pascals.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mancha import Simulation, UnitConverter  # noqa: E402
from mancha.units import reynolds  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--backend", default="numpy")
ap.add_argument("--cells", type=int, default=24)
ap.add_argument("--lengths", type=float, default=4.0, help="duct length in widths")
args = ap.parse_args()

width, speed, nu, rho = 0.010, 0.005, 1.0e-6, 998.0
units = UnitConverter.from_velocity(dx=width / args.cells, velocity=speed, lattice_velocity=0.04, rho=rho)
nu_lattice = units.to_lattice_viscosity(nu)
print(f"dx = {units.dx * 1e3:.3f} mm, dt = {units.dt * 1e6:.2f} µs, lattice ν = {nu_lattice:.4f} "
      f"(tau = {3 * nu_lattice + 0.5:.3f}), Re = {reynolds(speed, width, nu):.0f}")

n = args.cells
sim = Simulation(shape=(int(args.lengths * n), n, n), viscosity=nu_lattice, backend=args.backend, units=units,
                 velocity=(0.04, 0, 0))
sim.set_boundary("x-", "velocity", velocity=(0.04, 0, 0)).set_boundary("x+", "pressure", density=1.0)
for face in ("y-", "y+", "z-", "z+"):
    sim.set_boundary(face, "wall")
flow_through = int(sim.shape[0] / 0.04)
sim.run(3 * flow_through)

p = units.pressure(sim.density())                # gauge pressure [Pa]
u = units.velocity(sim.velocity())               # [m/s]
drop = p[1].mean() - p[-2].mean()
print(f"simulated {sim.time:.2f} s of flow in {sim.steps_done} steps ({sim.wall_time:.1f} s wall time)")
print(f"pressure drop over {units.length(sim.shape[0] - 3) * 1e3:.1f} mm: {drop:.3f} Pa; "
      f"peak velocity {u[0].max():.4f} m/s")
