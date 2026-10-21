# Yax

**Yax 2 (in progress): the format converter.** Platforms exchange standard files — Piccolo URDF/SDF, Vegeta MJCF and
JSBSim, CadQuery STEP, everyone USD — and Yax converts between them through a USD hub
([docs/converter.md](docs/converter.md)). Yax imports no platform's code; Vegeta and Piccolo vendor it.

## Yax 1 packages (still read)

Design interchange. A **package** is a directory with `yax.json` (units, frame, geometry,
mass properties, file hashes) and `body.stl`; a producer writes it, a consumer loads it, and neither imports the
other. Specification: [docs/spec.md](docs/spec.md).

```python
import yax

d = yax.load("drongo.yax")            # verifies hashes and the manifest, raises yax.Invalid on problems
d.mass["kg"], d.mass["inertia"]       # SI, about d.mass["com"]
tris = d.mesh()                       # triangles of body.stl, in d.length_unit
yax.units.LENGTH[d.length_unit]       # 0.001 for mm

yax.write("out/cube.yax", manifest, {"body.stl": "cube.stl"})   # producers: atomic, fills hashes and identity
yax.describe(triangles, "mm", density_kg_m3=2700)              # geometry + mass sections from a closed mesh
```

```bash
python -m yax check out/*.yax      # or the `yax` command once installed
python -m yax info out/cube.yax
```

* **Never** imports a producer or a consumer, numpy, a CAD kernel or a simulator.
* Exact mass properties of a closed mesh (`yax.stl.mass_properties`), so a consumer without CAD can verify a
  producer.
* `examples/cube.yax` is the test vector (`python3 examples/make_cube.py` regenerates it). Tests: `pytest` here.
