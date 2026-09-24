# Aeromant — external aerodynamics (OpenFOAM)

Aeromant runs OpenFOAM on **template cases**. It never invents a CFD case: the template holds the
engineering decisions (domain, patches, physics model, boundary conditions, meshing, solver
settings); the engineer supplies the geometry and the physical values. Aeromant accepts any STL and
does not need Dedalus.

## Templates
```
aeromant templates                      # or: aeromant.get_template(name).describe()
```
| template | physics | notes |
|----------|---------|-------|
| `laminar_external` | steady laminar | low Reynolds numbers; validation case |
| `rans_ksst_external` | steady RANS k-ω SST, wall functions | no prism layers: trends, not absolute drag |

Both use a box domain around the body bounding box, flow along **+x**, lift along **+z**, pitch
axis **+y**; patches `inlet` (fixed velocity), `outlet` (fixed pressure), `sides` (slip), `body`
(no-slip wall, forceCoeffs). Pipeline: `blockMesh → features → snappyHexMesh → checkMesh → restore0 → solver`.

Each template ships its case files in **both OpenFOAM dialects**, because the two forks use different
dictionaries and solvers:

| flavor | versions | case files | feature tool | solver |
|---|---|---|---|---|
| openfoam.com | v1912 and later (conda-forge `openfoam`, official `.com` packages) | `com/` (`constant/triSurface`, `transportProperties`, `turbulenceProperties`) | `surfaceFeatureExtract` | `simpleFoam` |
| openfoam.org | 12 and later (`/opt/openfoam14`, official `.org` packages) | `org/` (`constant/geometry`, `physicalProperties`, `momentumTransport`) | `surfaceFeatures` | `foamRun -solver incompressibleFluid` |

Templates: `laminar_external`, `rans_ksst_external` (a body in a free stream), `rotor_mrf`, `rotor_mrf_static` (a
propeller in a rotating frame, see below). `aeromant templates` lists them with their parameters.

`CFDCase` reads the installation's `WM_PROJECT_VERSION` (`v2412` → .com, `14` → .org) and picks the
matching files; `case.flavor` tells you which. `OpenFOAMEnvironment.detect()` prefers a .com install
when both exist (`detect(flavor="openfoam.org")` to prefer .org). The placeholders and the step names
are the same for both, so a case file `case.py` works unchanged on either fork.

Each template is a directory of native OpenFOAM files (`com/` and `org/`) plus a Python `TemplateSpec`
in `aeromant/templates/__init__.py`, which lists:
- **required values** (no default): `velocity`, `kinematic_viscosity`, `density`, `reference_area`,
  `reference_length`, `center_of_rotation`;
- **template decisions** with visible defaults (domain size in L_ref, refinement levels, iterations,
  residual target, inlet turbulence), which the engineer may override explicitly.

Unknown parameter names, missing required values, a body larger than the template allows
(`max_body_extent × reference_length`, usually a units mistake) or leftover placeholders are errors.

## Workflow
```python
import math
from vegeta import aeromant

env = aeromant.OpenFOAMEnvironment(bashrc="/usr/lib/openfoam/openfoam2406/etc/bashrc")
# or aeromant.OpenFOAMEnvironment.conda("/opt/foam")  or  aeromant.OpenFOAMEnvironment.detect()

case = aeromant.CFDCase(
    "laminar_external", "body.stl",
    dict(velocity=1.0, kinematic_viscosity=0.01, density=1.0,
         reference_area=math.pi / 4, reference_length=1.0, center_of_rotation=(0, 0, 0)),
    workdir="runs/sphere", geometry_units="m",          # STL has no units: say what they are
    environment=env,
)
case.prepare()                                  # copy template, scale + insert STL, fill values
case.run(steps=["blockMesh", "features", "snappyHexMesh", "checkMesh"])  # mesh only
res = case.run()                                # or the whole pipeline
res.metrics["Cd"], res.metrics["converged"]
aeromant.plot_coefficients("runs/sphere"); aeromant.plot_residuals("runs/sphere")
```
Nothing runs on construction; `run()` refuses an unprepared (or changed) case; `results()` only reads.

## Results
| metric | meaning |
|--------|---------|
| `Cd`, `Cl`, `Cm` | last value written by `forceCoeffs` (None if not available) |
| `Cd_mean_lastN`, `Cd_std_lastN` | mean / standard deviation over the last N iterations |
| `drag_force_N`, `lift_force_N` | `C × ½ρU² × A_ref` |
| `iterations`, `converged`, `final_residuals` | from `log.solver` |
| `mesh_cells`, `mesh_ok`, `mesh_failed_checks` | from `log.checkMesh` (failures are reported, not fatal) |
| `reynolds_number` | `U L_ref / ν` |

The whole case directory is kept: `inputs/` (original STL), `constant/triSurface/body.stl` (scaled to
metres), `system/`, `0.orig/`, `0/`, the mesh, `postProcessing/`, `log.*`, `aeromant_case.json`
(configuration and derived values) and `summary.json`.

## CLI
```
aeromant templates [NAME] [--json]
aeromant prepare case.py [--overwrite]        # case.py defines case = aeromant.CFDCase(...)
aeromant run case.py [--steps blockMesh,snappyHexMesh,checkMesh] [--timeout S]
aeromant results runs/sphere [--png] [--window 50]
```

## Rotating propellers and rotors (`rotor_mrf`, `rotor_mrf_static`)
A propeller in a rotating reference frame (MRF cell zone around the blades, steady k-ω SST):
`rotor_mrf` for axial inflow along +x (a propeller in flight or a marine propeller under way),
`rotor_mrf_static` for hover / static thrust (a residual inflow of 2 % of the tip speed keeps the steady
solution stable; open total-pressure boundaries diverged). Place the STL with its
axis on +x through `center` (default the origin); the rotor pushes fluid toward +x.
```python
case = aeromant.CFDCase("rotor_mrf_static", "prop.stl",
    dict(rpm=8200, diameter=0.127, kinematic_viscosity=1.5e-5, density=1.2, rotation=1),
    workdir="runs/prop_hover", geometry_units="mm", environment=env)
case.prepare(); res = case.run()
res.metrics["thrust_N"], res.metrics["torque_Nm"], res.metrics["power_W"], res.metrics["figure_of_merit"]
# rotor_mrf adds airspeed=…; its metrics include efficiency and advance_ratio; both give ct, cp
```
`rotation=1` turns the rotor by the right-hand rule about +x; a negative thrust means the blades are
handed the other way — rerun with `rotation=-1` (handedness does not change by rotating the CAD; only a
mirror does). `vegeta.dedalus.examples.Propeller`, rotated z→x, is right-handed for `rotation=1`. Forces come from the `forces` function object
(`postProcessing/forces/*/force.dat`, `moment.dat`), averaged over the last 50 iterations. Template
decisions you can override: domain size in diameters, MRF zone radius/length, refinement levels
(`surface_level` on the blades, `rotor_level` in the zone, `wake_level`), iterations, far-field turbulence.
Expect tens of percent against blade element theory (`vegeta.boreas`): no blade-passing unsteadiness,
no tip-vortex resolution, wall functions without prism layers. Compare trends, refine before trusting absolutes.

### Particle movies of a rotor case (`vegeta.aeromant.movie`)
One file, used from a notebook after a rotor case has run: a few tracer particles carried by the
converged velocity field, drawn with OpenCV in a side view and a view along the axis, the blades turning
in consistent slow motion (each frame advances the flow by the time the rotor takes to turn
`degrees_per_frame`). In an MRF case `U` is the absolute velocity, so the swirl the particles pick up is
the swirl the rotor puts into the flow. Air or water makes no difference: the field is what the case solved.
```python
from vegeta.aeromant import movie
movie.make_movie(case, "prop_particles.mp4", blades=3, n=40, seconds=12, fps=24, degrees_per_frame=10)
```
`RotorView.from_case` reads centre, diameter, rpm and sense of rotation from `aeromant_case.json`;
`openfoam_sampler(case)` reads the last solved time step once (pyvista) and looks velocities up by
inverse distance over the nearest cell centres (scipy); `Tracer` and `render_frame` are plain numpy and
OpenCV and take any `points -> (U, valid)` function, which is how the unit tests run on a stub field
(`tests/aeromant/test_movie.py`). A steady result shown as motion, not a transient simulation.

## OpenFOAM installations
- `./test_openfoam.sh` (repository root) runs both templates on every installation it finds, one per flavour,
  and reports which passed; use it after installing or upgrading OpenFOAM, or to validate the `org/` case files on
  an openfoam.org machine (`--bashrc /opt/openfoam14/etc/bashrc`).
- Official openfoam.com (`/usr/lib/openfoam/openfoamXXXX/etc/bashrc`) or openfoam.org packages: use
  `bashrc=`; `detect()` finds them.
- conda-forge `openfoam` (e.g. `micromamba create -p /opt/foam -c conda-forge openfoam=2412`):
  `OpenFOAMEnvironment.conda("/opt/foam")`.
- The Ubuntu 24.04 `openfoam` package (v1912) meshes correctly but **cannot start function objects**
  (`FOAM FATAL IO ERROR: error in IOstream "sha1"`), so no force coefficients. Aeromant reports this
  with a hint instead of returning numbers.

## Validation (vegeta-cli/tests/aeromant/test_integration.py)
Laminar sphere, Re = 100, ~49k cells: Cd = 1.100 vs Schiller–Naumann 1.092 (+0.7 %; test tolerance
10 %), |Cl| < 0.02, converged in ~105 iterations (OpenFOAM v2412). The test harness picks the
installation from `AEROMANT_TEST_OPENFOAM_PREFIX` / `AEROMANT_TEST_OPENFOAM_BASHRC` or `detect()`.
