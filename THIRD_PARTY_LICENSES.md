# Third-party tools and licences

Vegeta's own source code is MIT licensed (see `LICENSE`). It drives external
engineering tools that carry their own licences. Nothing from these tools is
bundled in this repository; you install them yourself.

| Tool | Used by | How it is used | Licence |
|------|---------|----------------|---------|
| CadQuery / OCP (OpenCascade) | Dedalus | Python library import | Apache-2.0 / LGPL-2.1 |
| Gmsh | Talos | Python API (`import gmsh`, dynamically links `libgmsh`) | GPL-2.0-or-later |
| CalculiX (`ccx`) | Talos | separate process via CLI | GPL-2.0-or-later |
| OpenFOAM | Aeromant | separate processes via CLI | GPL-3.0-or-later |
| PrusaSlicer | Mellonia | separate process via CLI | AGPL-3.0 |

Notes
- Calling a GPL/AGPL program as a separate process (CalculiX, OpenFOAM,
  PrusaSlicer) does not make Vegeta a derivative work; the MIT licence applies
  to Vegeta's code without conflict.
- Talos links to the Gmsh library through its Python API. MIT is
  GPL-compatible, so Talos' source stays MIT; however a *combined
  distribution* of Talos together with Gmsh (for example a bundled installer or
  container image) must comply with the GPL terms of Gmsh. Installing Gmsh
  separately with `pip install gmsh` keeps this the user's choice.
- OpenFOAM template cases shipped in Aeromant are written for this project and
  are MIT; they are inputs to OpenFOAM, not OpenFOAM code.
