# Assemblies — components arranged into machines (geometry only)

A component (`components/`) is one part's parametric CAD: a frame, a hull, a propeller, a motor. An assembly says
how components are put together to make a machine: which ones, with which parameters, and where each sits. Nothing
else. Masses, loads, FEA, CFD, flight and missions stay in the notebooks, where they are seen and run. A notebook may
use an assembly, use components directly, or build its own geometry; nothing here forces any of it.

```
assemblies/
  base.py          Assembly (a Dedalus Design whose parts(p) places components), component_parameters, pick, turned
  quadcopter.py    Quadcopter          QuadFrame + 4 Outrunner motors + 4 propellers (notebook 08)
  fixed_wing.py    FixedWingDrone      FixedWing airframe + 2 tractor propellers at the nacelles (notebook 09a)
  survey_boat.py   SurveyBoatAssembly  SurveyBoat (hull, deck, pod bracket) + the propeller behind the pod (notebook 12)
  submarine.py     SubmarineAssembly   Submarine vehicle + the propeller behind the tail (notebook 13)
  tests/           geometry only, quick: imports, placement, build by spec
```

The placements are the ones the notebooks and `scenarios/run_scenario.py` use (propeller plane 12 mm ahead of the
nacelle nose, 10 mm behind the boat's pod, 40 mm behind the submarine's tail tip; quad motors on the pads at 45° +
90°·k, diagonals turning the same way).

## Using one in a notebook
```python
import sys; sys.path.insert(0, "..")                 # notebooks/ -> the repository root
from assemblies.quadcopter import Quadcopter

quad = Quadcopter()
parts = quad.generate_parts(wheelbase=280, propeller="5x4.3 tri-blade")   # {"frame": Shape, "motor_1": ..., "propeller_4": ...}
geometry = quad.generate(wheelbase=280)               # one compound: a dedalus.Geometry (export, measure, show)
step = quad.assembly(wheelbase=280)                   # a cq.Assembly with one named child per part

from vegeta import dedalus                            # or by spec, as the notebooks load designs
quad = dedalus.load_design("assemblies.quadcopter:Quadcopter")
```
An assembly's parameters are its components' parameters (the `part` selector aside, which the assembly fixes) plus
its own: the propeller (a name from `components.propeller.CATALOGUE`) and the gaps that place it. The frame used for
FEA is still `components.quad_frame.QuadFrame` (or the notebook's own); the assembly is for the whole machine.

## Writing one
```python
from components.thing import Thing
from components.propeller import CATALOGUE, blade
from .base import Assembly, component_parameters, pick, turned

class ThingWithProp(Assembly):
    parameters = component_parameters(Thing) + [Parameter("propeller", "9x6 electric", choices=tuple(CATALOGUE))]

    def parts(self, p):
        body = Thing().generate(**pick(p, Thing, part="whole")).shape
        prop, _ = blade(p["propeller"])                # axis +Z, hub centre at the origin
        return {"body": body, "propeller": turned(prop, "-x").translate((-50, 0, 0))}
```
Rules (checked by `tests/test_assemblies.py`): import only geometry (cadquery, numpy, math, `vegeta.dedalus`,
`components`); put components in place and nothing more; add the module to the test's `SPECS`.

## Tests
```bash
python -m pytest -q components/tests assemblies/tests      # geometry only, about two minutes, no solver
./user_tests.sh geometry                                   # the same, in the report
```
