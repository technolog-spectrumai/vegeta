# Talos — linear static FEA (Gmsh + CalculiX)

Talos takes **any STEP file** plus explicit engineering configuration, meshes it with Gmsh and
solves it with CalculiX (`ccx`). It does not need Dedalus. Initial scope: linear static analysis
with quadratic (C3D10, default) or linear (C3D4) tetrahedra.

Talos never guesses supports, loads, material properties or strengths. A missing value is an error.

## Workflow
```python
from vegeta import talos

info = talos.inspect_step("bracket.step", units="mm-N-MPa")   # surfaces: tag, area, centroid, normal
print(info.table())

model = talos.StructuralModel(
    geometry="bracket.step",
    units="mm-N-MPa",                                          # explicit: mm-N-MPa or m-N-Pa
    material=talos.Material("S235", youngs_modulus=210000, poissons_ratio=0.3,
                            density=7.85e-9, yield_strength=235, source="EN 10025-2"),
    regions=[talos.SurfacesOnPlane("fixed", "x", 0.0),
             talos.Surfaces("holes", [7, 8]),
             talos.SurfacesInBox("pad", (70, -5, 2, 80, 5, 4))],
    supports=[talos.FixedSupport("fixed")],
    loads=[talos.Force("pad", fz=-500.0), talos.Pressure("holes", 2.0), talos.Acceleration(az=-9810)],
    mesh_settings=talos.MeshSettings(element_size=2.0, order=2),   # high_order_optimize=1 by default
)
mesh_result = model.mesh("runs/bracket")       # explicit step 1: Gmsh -> mesh.msh
result = model.solve("runs/bracket")           # explicit step 2: CalculiX (needs an up-to-date mesh)
result.metrics["max_von_mises"], result.metrics["safety_factor_yield"]
```

## Units
| system | length | force | stress | density | acceleration |
|--------|--------|-------|--------|---------|--------------|
| `mm-N-MPa` | mm | N | MPa | t/mm³ (steel 7.85e-9) | mm/s² (g = 9810) |
| `m-N-Pa` | m | N | Pa | kg/m³ (steel 7850) | m/s² (g = 9.81) |

The STEP geometry is converted to the chosen length unit by OpenCascade on import.

## Regions, supports and loads
- Regions name sets of surfaces: `Surfaces(name, tags)` (tags from `inspect_step`),
  `SurfacesOnPlane(name, axis, value, tol=None)`, `SurfacesInBox(name, box)`. A region that selects
  nothing is an error.
- Supports: `FixedSupport(region)`, `Displacement(region, ux=None, uy=None, uz=None)`.
- Loads: `Force(region, fx, fy, fz)` — total force distributed as a uniform traction with consistent
  nodal loads; `Pressure(region, value)` — positive pushes into the solid (CalculiX convention);
  `Acceleration(ax, ay, az)` — body load on the whole solid, needs `density`.

## Meshing notes
Second-order meshes place mid-side nodes on curved geometry (holes, fillets), which can invert an
element. `MeshSettings(high_order_optimize=1)` (default) runs Gmsh's high-order optimisation to fix
this; `mesh()` still fails loudly if any element remains inverted (minSICN ≤ 0) and warns below 0.1.

`MeshSettings(curvature_points=24)` sizes elements from curvature (Gmsh `MeshSizeFromCurvature`, elements
per full circle), so small holes and fillets get finer elements than `element_size` without a global
refinement; off by default.

A Gmsh "PLC Error: a segment and a facet intersect" almost always means a sliver in the geometry (a
bolt hole 0.1 mm from an edge, a 0.4 mm edge left by a union) rather than a meshing setting: check the
smallest edges of the CAD before changing the mesh (`notebooks/08_quadcopter` shows such a case).

## Results
`solve()` returns a `talos.Result` (see `docs/result-shape.md`). Metrics:

| metric | meaning |
|--------|---------|
| `max_displacement`, `_location`, `_node` | largest nodal displacement magnitude |
| `displacement_min` / `displacement_max` | per-component extremes |
| `max_von_mises`, `_location` | largest nodal von Mises stress (from CalculiX nodal stresses) |
| `reactions`, `reaction_total` | CalculiX `RF` totals per support region |
| `applied_force_total` | sum of `Force` loads |
| `safety_factor_yield` | `yield_strength / max_von_mises`, only when `yield_strength` is given |

Caveats reported in `messages`:
- Peak nodal stresses are mesh dependent at supports, point-like loads and re-entrant corners.
- CalculiX `RF` is the internal nodal force; external load applied directly on supported nodes
  (the body-force share of supported nodes, or a load region touching a support) is not included.

Artifacts kept in the work directory: `mesh.msh`, `gmsh.log`, `mesh_summary.json`, `model.inp`,
`model.frd`, `model.dat`, `model.sta`, `ccx.log`, `summary.json`.
Field data: `talos.read_frd(result.artifacts["frd"])` → nodes, displacement, stress, von Mises.
Plots: `plot_deformed`, `plot_von_mises_histogram`, `plot_along_axis` (matplotlib figures).

## CLI
```
talos inspect part.step --units mm-N-MPa [--json]
talos mesh  model.py -w runs/case          # model.py defines model = talos.StructuralModel(...)
talos solve model.py -w runs/case [--png] [--threads 4] [--ccx /path/to/ccx]
talos results runs/case                     # prints NOT RUN when nothing was solved there
```
`solve` never meshes implicitly: if the geometry, regions or mesh settings changed since the last
`mesh`, it fails and asks for a new mesh.

## Validation (vegeta-cli/tests/talos/test_integration.py)
- Cantilever 200×20×10 mm, tip load 100 N: tip deflection within 3 % of Euler–Bernoulli
  (observed −0.6 %), root stress within 10 % of Mc/I, reactions balance the load.
- Bar 100×10×10 mm in tension (force or pressure; C3D10 and C3D4) with symmetry supports:
  displacement and uniform stress within 0.5 % of F·L/(E·A) and F/A.
- Gravity: reaction equals weight up to the documented RF caveat.
