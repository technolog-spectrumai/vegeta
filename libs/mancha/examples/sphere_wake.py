"""Flow past a sphere: drag by momentum exchange and a VTK file of the wake for ParaView.

    python examples/sphere_wake.py [--backend numpy|triton] [--diameter 10] [--reynolds 50]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mancha.validation import sphere  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--backend", default="numpy")
ap.add_argument("--diameter", type=float, default=10.0)
ap.add_argument("--reynolds", type=float, default=50.0)
ap.add_argument("--out", default="results/sphere_wake")
args = ap.parse_args()

r = sphere(diameter=args.diameter, reynolds=args.reynolds, backend=args.backend, tol=1e-6)
print(f"grid {r['shape']}, {r['steps']} steps, blockage {r['blockage_ratio']:.3f}")
print(f"Cd = {r['cd']:.3f} (unconfined Schiller–Naumann correlation: {r['cd_schiller_naumann']:.3f}; this box is "
      "periodic laterally, so expect more drag)")
print(f"recirculation length behind the sphere: {r['wake_length_over_d']:.2f} D")
r["simulation"].save(args.out)
print(f"fields in {args.out}/fields.vti (density, pressure, velocity, solid)")
