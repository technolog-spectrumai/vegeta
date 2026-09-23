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

## OpenFOAM installations
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
