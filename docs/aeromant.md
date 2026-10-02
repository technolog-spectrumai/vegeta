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
propeller in a rotating frame, see below), `aircraft_rotor_disks` (a whole aircraft with two propellers as
rotor disks, see below). `aeromant templates` lists them with their parameters.

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

## A whole aircraft with its propellers (`aircraft_rotor_disks`)
The external RANS case of an aircraft with **two propellers as rotor disks**: each disk is a thin cylindrical
cell zone (snappyHexMesh) carrying OpenFOAM's `rotorDisk` blade-element source (`system/fvOptions` for
openfoam.com, `constant/fvModels` for openfoam.org). The blade geometry and the section polar are tables, e.g.
from Boreas:
```python
case = aeromant.CFDCase("aircraft_rotor_disks", "aircraft.stl", dict(
    velocity=14, kinematic_viscosity=1.5e-5, density=1.2, reference_area=0.17, reference_length=0.25,
    center_of_rotation=(0.05, 0, 0),
    disk1_center=(-0.06, -0.3, 0.0), disk2_center=(-0.06, 0.3, 0.0), disk_axis=(-1, 0, 0),   # thrust direction
    diameter=prop.diameter, rpm=cruise.rpm, blades=prop.blades,
    blade=[[r, beta, c] for r, beta, c in zip(prop.r, prop.beta_deg, prop.chord)],          # m, deg, m
    polar=[[a, cd, cl] ...],                                                                 # deg, -180..180
    rotation1=1, rotation2=-1), workdir="runs/aircraft_disks", geometry_units="mm")
```
The forces (Cl, Cd, lift and drag) are the **airframe's in the slipstream**; the disks' own thrust is printed by
the source in the solver log. Momentum and swirl are right on average; blade passing and tip vortices are not
resolved (a fully resolved rotating-blade simulation of the whole aircraft is on the to-do list). Sign check: the
flow behind the disks must be faster than the free stream; if not, flip `disk_axis`. Notebook:
`09a_fixed_wing_design`, Part 3.

`hull_rotor_disk` is the same case with **one** disk (`disk1_center`, `rotation1`) around any body — a hull, a
robot. **Heavily loaded disks** (near bollard: a jet several times the free stream, e.g. a thruster at walking or
swimming speed in water) diverge with `rotorDisk`'s default local inflow — the blade elements see the cell velocity,
which the disk itself drives. `fixed_inflow=1` makes them see the free stream (`inletFlowType fixed`): a steady
source with no induction, so it over-predicts the thrust — set the rpm to the one whose no-induction thrust is the
thrust you want (`notebooks/designs/lobster_cfd.disk_calibration` does it from Boreas). Notebook:
`24_sikarian_lobster_nefri`, §5.

## A suction hood under a vehicle (`suction_hood`)
The Onager Sweeper's vacuum nozzle (notebook 23): the air between the road and the vehicle floor, a hood whose lips
stand a gap over the road, a square duct through the floor drawing the fan's flow. The domain is nine blockMesh
blocks between `z = 0` (the road, a no-slip wall moving at `-ground_speed`: the road passing under the machine) and
`floor_height` (the floor, a wall); the middle block is the duct's column — its top face is the `suction` patch
(fixed outflow `flow_rate / duct_inner²`), the other eight end on the `floor`; the `sides` are the atmosphere (p = 0,
`pressureInletOutletVelocity`). The STL carries the hood **and** its duct through the floor (the block boundary runs
through the duct's wall, so `duct_center`, `duct_inner` and `duct_wall` must match the STL); place it on the road.
```python
case = aeromant.CFDCase("suction_hood", "hood.stl",
    dict(flow_rate=0.35, duct_center=(-0.1, 0.0, 0.0), duct_inner=0.14, duct_wall=0.01, floor_height=0.30,
         kinematic_viscosity=1.5e-5, density=1.2, reference_length=0.5, ground_speed=1.0),
    workdir="runs/hood", geometry_units="mm", environment=env)
case.prepare(); res = case.run(processors=4)
res.metrics["fan_static_pressure_Pa"], res.metrics["flow_rate_m3_s"], res.metrics["air_power_W"]
```
Metrics come from `surfaceFieldValue` function objects: the depression at the duct (what the fan must supply, before
the losses downstream of it), the flow actually drawn (a check: it must equal `flow_rate`), the mean pressure on the
hood walls, the air power `Q × Δp`. The gap flow is resolved by the `gap_level` refinement box (road to half the
body's height). `notebooks/designs/onager_sweeper_cfd.QUALITY` holds two presets: `fast` (8 cells per L_ref,
levels 3/1/2, 150 iterations: ~110 k cells for the Sweeper's hood, a minute or two on 4 cores) and `fine` (12 cells,
levels 3/2/3, 400 iterations: ~0.9 M cells, ~15 min). `notebooks/designs/onager_sweeper_cfd.py` reads the lip inflow and the hood's upward speed off the
field, compares them with the litter classes' terminal speeds, and runs litter particles through the field for a
movie.

## OpenFOAM installations
- `./test_openfoam.sh` (repository root) runs both templates on every installation it finds, one per flavour,
  and reports which passed; use it after installing or upgrading OpenFOAM, or to validate the `org/` case files on
  an openfoam.org machine (`--bashrc /opt/openfoam14/etc/bashrc`).
- Official openfoam.com (`/usr/lib/openfoam/openfoamXXXX/etc/bashrc`) or openfoam.org packages: use
  `bashrc=`; `detect()` finds them.
- conda-forge `openfoam` (e.g. `micromamba create -p /opt/foam -c conda-forge openfoam=2412`):
  `OpenFOAMEnvironment.conda("/opt/foam")`. The 2412 build links its MPI Pstream through `lib/sys-mpich` but ships it
  as `lib/mpich-3.3`: without `ln -s mpich-3.3 /opt/foam/lib/sys-mpich` (which `install_local.sh` adds) every
  `processors > 1` run stops with "The dummy Pstream library cannot be used in parallel mode".
- The Ubuntu 24.04 `openfoam` package (v1912) meshes correctly but **cannot start function objects**
  (`FOAM FATAL IO ERROR: error in IOstream "sha1"`), so no force coefficients. Aeromant reports this
  with a hint instead of returning numbers.

## Validation (vegeta-cli/tests/aeromant/test_integration.py)
Laminar sphere, Re = 100, ~49k cells: Cd = 1.100 vs Schiller–Naumann 1.092 (+0.7 %; test tolerance
10 %), |Cl| < 0.02, converged in ~105 iterations (OpenFOAM v2412). The test harness picks the
installation from `AEROMANT_TEST_OPENFOAM_PREFIX` / `AEROMANT_TEST_OPENFOAM_BASHRC` or `detect()`.
