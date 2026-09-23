# Third-party tools and licences

Vegeta's own code is proprietary and for internal use only (see `LICENSE`). It drives external
engineering tools that carry their own licences. Nothing from these tools is bundled in this
repository; they are installed separately.

| Tool | Used by | How it is used | Licence |
|------|---------|----------------|---------|
| CadQuery / OCP (OpenCascade) | `vegeta.dedalus` | Python library import | Apache-2.0 / LGPL-2.1 |
| Gmsh | `vegeta.talos` | Python API (`import gmsh`, dynamically links `libgmsh`) | GPL-2.0-or-later |
| CalculiX (`ccx`) | `vegeta.talos` | separate process via CLI | GPL-2.0-or-later |
| OpenFOAM | `vegeta.aeromant` | separate processes via CLI | GPL-3.0-or-later |
| PrusaSlicer | `vegeta.mellonia` | separate process via CLI | AGPL-3.0 |

Notes
- Internal use (running the tools inside the organisation) is not distribution, so the GPL/AGPL
  terms of these tools place no obligations on Vegeta's code.
- CalculiX, OpenFOAM and PrusaSlicer are only called as separate processes.
- `vegeta.talos` links the Gmsh library. If `vegeta-cli` were ever distributed outside the
  organisation together with Gmsh, the GPL terms of Gmsh would apply to that distribution; ask
  before doing so.
- The OpenFOAM template cases in `vegeta.aeromant` were written for this project and are part of
  Vegeta's proprietary code; they are inputs to OpenFOAM, not OpenFOAM code.

## Logo
`logo_with_text.png`, `logo_no_text.png` and `logo_small_mono.png` are for internal use only. The
figure is based on a character whose rights belong to Bird Studio/Shueisha and Toei Animation, so the
logo must not be used publicly or commercially.
