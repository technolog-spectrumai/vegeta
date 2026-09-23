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
| OpenFOAM | `apt install openfoam` (v1912, openfoam.com) | `WM_PROJECT_DIR=/usr/share/openfoam blockMesh -help` |
| PrusaSlicer | `apt install prusa-slicer` | `prusa-slicer --help` |

### OpenFOAM notes
The Ubuntu `openfoam` package puts executables in `/usr/bin` but they need `WM_PROJECT_DIR`
to find their `etc/` files. Aeromant takes this explicitly:
`OpenFOAMEnvironment(env={"WM_PROJECT_DIR": "/usr/share/openfoam"})`, or
`OpenFOAMEnvironment(bashrc="/opt/openfoam2406/etc/bashrc")` for a sourced installation.

### PrusaSlicer notes
The CLI slices headless. If your build insists on a display, run it under `xvfb-run`
(`slice_stl(..., executable=["xvfb-run", "-a", "prusa-slicer"])`).

## Tests
`scripts/test_all.sh` runs every package's tests separately. Integration tests skip cleanly when a
tool is missing; markers: `requires_cadquery`, `requires_gmsh`, `requires_ccx`, `requires_openfoam`,
`requires_prusaslicer`, `slow`. Run only fast tests with `pytest -m "not slow"`.
