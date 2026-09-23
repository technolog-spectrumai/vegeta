# Installation

Python ≥ 3.10. Each package installs on its own:

```bash
pip install -e packages/dedalus        # CadQuery CAD
pip install -e packages/talos          # needs gmsh (pip) + CalculiX `ccx` (system)
pip install -e packages/aeromant       # needs OpenFOAM (system)
pip install -e packages/mellonia       # needs PrusaSlicer (system)
```
Add `[pandas]` for `to_dataframe()` helpers and `[test]` for the test suite.

## External tools (Ubuntu 24.04)
`scripts/install_tools.sh` installs:

| Tool | Package | Check |
|------|---------|-------|
| CadQuery | `pip install cadquery` | `python -c "import cadquery"` |
| Gmsh | `pip install gmsh` (+ `libglu1-mesa libxrender1 libxcursor1 libxft2 libxinerama1`) | `python -c "import gmsh"` |
| CalculiX | `apt install calculix-ccx` | `ccx -v` |
| OpenFOAM | official openfoam.com/.org package, or conda-forge `openfoam` | `simpleFoam -help` |
| PrusaSlicer | `apt install prusa-slicer` | `prusa-slicer --help` |

### OpenFOAM notes
Tell Aeromant explicitly how to launch OpenFOAM:
- sourced installation: `OpenFOAMEnvironment(bashrc="/usr/lib/openfoam/openfoam2406/etc/bashrc")`
- conda environment: `micromamba create -p /opt/foam -c conda-forge openfoam=2412`, then
  `OpenFOAMEnvironment.conda("/opt/foam")`
- `OpenFOAMEnvironment.detect()` looks in the usual places.

Avoid the Ubuntu 24.04 `openfoam` apt package (v1912): it needs
`OpenFOAMEnvironment(env={"WM_PROJECT_DIR": "/usr/share/openfoam"})` and, more importantly, its
function objects fail (`IOstream "sha1"` error), so `forceCoeffs` never produces coefficients.

### PrusaSlicer notes
The CLI slices headless. If your build insists on a display, run it under `xvfb-run`
(`slice_stl(..., executable=["xvfb-run", "-a", "prusa-slicer"])`).

## Tests
`scripts/test_all.sh` runs every package's tests separately. Integration tests skip cleanly when a
tool is missing; markers: `requires_cadquery`, `requires_gmsh`, `requires_ccx`, `requires_openfoam`,
`requires_prusaslicer`, `slow`. Run only fast tests with `pytest -m "not slow"`.
