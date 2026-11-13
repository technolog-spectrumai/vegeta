"""Regenerate examples/cube.yax, the spec's test vector: a 10 mm aluminium cube (2700 kg/m^3) from the origin.

Exact values: volume 1000 mm^3, mass 0.0027 kg, centre (5, 5, 5) mm, Ixx = Iyy = Izz = m a^2 / 6 = 4.5e-8 kg m^2."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
import yax  # noqa: E402

A = 10.0
V = [(x, y, z) for x in (0.0, A) for y in (0.0, A) for z in (0.0, A)]          # index = 4*ix + 2*iy + iz
QUADS = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]   # outward winding


def cube():
    return [tri for a, b, c, d in QUADS for tri in ((V[a], V[b], V[c]), (V[a], V[c], V[d]))]


def make(out=HERE / "cube.yax"):
    tris = cube()
    stl_bytes = HERE / ".cube.stl"
    yax.stl.write(stl_bytes, tris)
    try:
        return yax.write(out, {
            "name": "cube",
            "producer": {"tool": "yax-example", "version": "1", "design_class": "Cube", "parameters": {"a_mm": A}},
            "units": {"length": "mm", "mass": "kg"},
            "frame": {"up": "+z", "forward": "+x"},
            **yax.describe(tris, "mm", density_kg_m3=2700.0),
        }, {"body.stl": stl_bytes})
    finally:
        stl_bytes.unlink()


if __name__ == "__main__":
    print(make().dir)
