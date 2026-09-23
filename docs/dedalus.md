# Dedalus — parametric CAD

Dedalus wraps [CadQuery](https://cadquery.readthedocs.io). Its job ends with geometry and geometry
metadata; it knows nothing about FEA, CFD, slicing or Vegeta. Lengths are CadQuery's millimetres.

## Defining a design
```python
import cadquery as cq
from vegeta import dedalus
from vegeta.dedalus import Design, Parameter, design

class Plate(Design):
    parameters = [
        Parameter("width", 40.0, "mm", min=5, description="plate width"),
        Parameter("thickness", 3.0, "mm", min=0.5),
    ]
    def build(self, p):                       # p: dict of validated values
        return cq.Workplane().box(p["width"], 20, p["thickness"])

@design(parameters=[Parameter("r", 5.0, "mm", min=0.1)])  # or a plain function
def ball(p):
    return cq.Workplane().sphere(p["r"])
```
`Parameter` types follow the default (`float`, `int`, `bool`, `str`); `min`/`max`/`choices` are
checked. Unknown or out-of-range values raise `ValueError` — nothing is clamped or guessed.

## Generating and inspecting
```python
g = Plate().generate(width=60)      # -> Geometry (raises BuildError if build() fails)
g                                    # 3D view in Jupyter (CadQuery's own renderer)
g.measure()                          # bbox, dimensions, volume, surface area, center of mass, validity
g.show()                             # CadQuery's VTK viewer (desktop)
dedalus.plot_views(g)                # static matplotlib projections (e.g. for reports / CLI)
```
Volume and center of mass are reported only for valid closed solids; otherwise they are `None` and
`g.messages` explains why.

## Exporting
```python
res = g.export("out/plate")                     # STEP + STL + summary.json -> Result
res = Plate().run("out/plate", width=60)        # generate + export; failures -> failed Result
g.export_step("plate.step"); g.export_stl("plate.stl", tolerance=0.01)
```
The result follows the common shape (`docs/result-shape.md`); `metadata` records parameters,
source identity (module, qualname, SHA-256 of the `build` source) and the CadQuery version.

Existing STEP files from any CAD program: `dedalus.load_step("part.step").measure()`.

## CLI
```
dedalus params  vegeta.dedalus.examples:Bracket
dedalus generate my_design.py:Plate -p width=60 -p thickness=4 -o out/plate --png [--json]
dedalus measure out/plate/Plate.step [--json]
```
Exit codes: 0 success, 1 failed run (e.g. build error), 2 invalid input.

## Examples
`dedalus.examples`: `CantileverBeam`, `Bracket`, `StreamlinedBody`, `Cube` — illustrations only.
Notebook: `notebooks/01_dedalus_cad.ipynb`.
