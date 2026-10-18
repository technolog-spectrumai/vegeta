# Mimesis in Vegeta — how mock results are labelled

`mimesis/` is a small package with no dependencies outside the standard library (`mimesis-mock`, import `mimesis`;
its own README and tests). It was copied from Alabastron, which uses it for its laser-solver stubs.
It sets the convention for imitating expensive APIs: a stub result says it is one (`"mock": true`, the mimic's name,
a note), and a loud banner says when stubs are in force.

## In Vegeta: every FEA and CFD run can be mocked

One switch, `VEGETA_DEBUG=1`, or `DEBUG = True` in a notebook's first cell (`DEBUG = mock.setup(DEBUG)`), puts
`vegeta.mock` in place. Each stub is a **Mimesis mimic**, and `mimesis.available()` lists them:

| mimic | replaces | real cost |
|---|---|---|
| `talos.mesh`, `talos.read_mesh` | Gmsh meshing of a structural model | seconds to minutes |
| `talos.solve`, `talos.solve_modes`, `talos.read_frd` | CalculiX static and modal solves | minutes to an hour |
| `talos.assess_fatigue` | the fatigue pass over a load spectrum | minutes |
| `aeromant.run`, `aeromant.results`, `aeromant.read_case_results`, `aeromant.read_results`, `aeromant.openfoam_sampler` | the OpenFOAM pipeline (meshing and solving) and reading its results | 10 min to hours |
| `mellonia.slice_stl` | PrusaSlicer | seconds to minutes |

Every call to a solver process in Talos, Aeromant and Mellonia goes through these methods: `StructuralModel.solve` and
`solve_modes` for CalculiX, `CFDCase.run` for OpenFOAM, `slice_stl` for PrusaSlicer. So `ensure`, `solve_models`,
`run_cases`, `vegeta.core` and the workflow scripts are all covered. Case preparation (CAD, STL, the case files) stays real.

What a mock looks like:

- **Labelled.** Every mocked `Result` carries Mimesis's label in its `metadata`: `"mock": true`, the mimic's name and
  "not a result". The real metadata keys (model, keys) are kept beside it. `mock.label(d)` / `mock.is_mock(d)`
  do the same for any dict.
- **Announced.** A `mimesis.banner` appears when the mocks go in: an HTML box in Jupyter, a starred line in a terminal.
- **Kept apart.**
  - While the mocks are on, `VEGETA_WORKFLOW_RUNS` and `VEGETA_WORKFLOW_CACHE` point to `workflows/runs-mock` and `workflows/cache-mock` (`mimesis.isolate_env`); `uninstall` restores them.
  - The Talos/Aeromant result cache is off.
  - runkit writes `"mock": true` into the manifest and a MOCK line into `run.log`. A run resumed under the mocks becomes mock.
- **Workflows.** `VEGETA_DEBUG=1 wf K1 --variant 4Q-2` (or the script directly) runs the whole chapter against the
  mimics in seconds. A missing `ccx` or OpenFOAM no longer stops it. The tool version is recorded as `mock`, and the
  closing line says no notebook reads the run.
- **Never taken for a result.**
  - Yanus refuses any run whose manifest is mock.
  - `prop_lab.export` under the mocks writes `prop_lab-mock.json` (labelled) instead of the calibration Kestrel reads.
  - `kestrel_systems.prop_lab_calibration` ignores a mock-labelled record.

Remaining:

- **Unguarded notebook records.** The other notebooks that write records under `notebooks/designs/data/` (25, 26, 28, 29, 31, 33–36) still write them in DEBUG with mocked numbers in their solver-fed parts. Run DEBUG in a scratch copy, or restore those files with git.
- **No schema checks yet.** The stubs are not checked against recorded real schemas (`mimesis.record_schema` / `check`). Recording needs real CalculiX/OpenFOAM runs.

Install: `./install_local.sh` installs it (`-e mimesis[test]`). `activate.sh` puts `mimesis/src` on the path. Tests:
`scripts/test_all.sh` (or `pytest mimesis/tests`, `vegeta-cli/tests/mock`).
