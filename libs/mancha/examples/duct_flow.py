"""The README example: flow through a square duct, velocity inlet to pressure outlet, walls on the other faces.

    python examples/duct_flow.py                       # 128³ on the GPU, 10 000 steps (as in the README)
    python examples/duct_flow.py --backend numpy --shape 32 32 32 --steps 2000    # a CPU-sized run

Everything is in lattice units: the inlet speed 0.05 is Ma ≈ 0.09, the viscosity 0.02 gives tau = 0.56 and a
duct Reynolds number U·W/ν = 0.05 · 128 / 0.02 = 320.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mancha import D3Q19, Simulation  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--backend", default="triton")
ap.add_argument("--shape", type=int, nargs=3, default=(128, 128, 128))
ap.add_argument("--steps", type=int, default=10000)
ap.add_argument("--out", default="results/duct_flow")
args = ap.parse_args()

sim = Simulation(
    lattice=D3Q19(),
    shape=tuple(args.shape),
    viscosity=0.02,              # lattice units (dx²/dt): tau = 3ν + 1/2 = 0.56
    backend=args.backend,
)

sim.set_boundary("x-", "velocity", velocity=(0.05, 0, 0))   # lattice units (dx/dt)
sim.set_boundary("x+", "pressure", density=1.0)
sim.set_boundary("y-", "wall")
sim.set_boundary("y+", "wall")
sim.set_boundary("z-", "wall")
sim.set_boundary("z+", "wall")

sim.run(steps=args.steps)

velocity = sim.velocity()        # [3, nx, ny, nz] NumPy array
density = sim.density()          # [nx, ny, nz]

sim.save(args.out)
nx, ny, nz = sim.shape
mlups = nx * ny * nz * sim.steps_done / sim.wall_time / 1e6
print(f"{sim.steps_done} steps in {sim.wall_time:.1f} s ({mlups:.1f} MLUPS)")
print(f"centreline u at the outlet: {velocity[0, -1, ny // 2, nz // 2]:.4f}  (inlet 0.05)")
print(f"density range: {density.min():.5f} .. {density.max():.5f}; results in {args.out}/")
