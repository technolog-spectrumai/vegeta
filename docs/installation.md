# Installation

On Ubuntu, one script installs everything (system tools, OpenFOAM, a `.venv` with all four
`vegeta-cli` and Jupyter) and checks each tool at the end:

```bash
./install_local.sh      # options: --venv DIR, --system-python, --skip-system, --no-openfoam, --openfoam-prefix DIR
source .venv/bin/activate
```

Manual installation — Python ≥ 3.10; one package provides all four tools:

```bash
pip install -e "vegeta-cli[pandas,test]"   # CadQuery and gmsh come from pip
```
The external programs are only needed by the tool that uses them: Talos needs CalculiX `ccx`,
Aeromant needs OpenFOAM, Mellonia needs PrusaSlicer. Missing programs give a failed result with an
explanation, never an import error.
Add `[pandas]` for `to_dataframe()` helpers and `[test]` for the test suite.

## External tools (Ubuntu 24.04)
`install_local.sh` first creates the virtual environment, then installs:

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
`scripts/test_all.sh` runs the tests of each tool (`vegeta-cli/tests/<tool>`) separately. Integration tests skip cleanly when a
tool is missing; markers: `requires_cadquery`, `requires_gmsh`, `requires_ccx`, `requires_openfoam`,
`requires_prusaslicer`, `slow`. Run only fast tests with `pytest -m "not slow"`.
