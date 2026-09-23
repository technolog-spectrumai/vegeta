# vegeta-cli

One installable package with the Vegeta engineering tools, each an independent subpackage:

| import | command | job |
|--------|---------|-----|
| `vegeta.dedalus` | `vegeta dedalus` / `dedalus` | parametric CAD on CadQuery → STEP/STL, measurements |
| `vegeta.talos` | `vegeta talos` / `talos` | linear static FEA: STEP → Gmsh → CalculiX |
| `vegeta.aeromant` | `vegeta aeromant` / `aeromant` | aerodynamics on OpenFOAM template cases: STL → Cd/Cl/Cm |
| `vegeta.mellonia` | `vegeta mellonia` / `mellonia` | 3D-print manufacturability with PrusaSlicer: STL → G-code |

The subpackages never import each other; they compose through files. `vegeta` is a namespace
package, so further distributions in this repository can add `vegeta.<name>` later.

```bash
pip install -e "vegeta-cli[pandas,test]"
python -c "from vegeta import dedalus, talos, aeromant, mellonia"
vegeta --help
```
Documentation: `../docs/`. Tests: `cd vegeta-cli && pytest` (tools that are not installed are skipped).
