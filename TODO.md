# TODO — microjet (notebook 28): spin movies, blade heat solve, performance maps

Requested for `notebooks/28_microjet.ipynb`:
- movies of the engine spinning: the rotor in its casing, the hot turbine, and the stress while spinning
- a thermal analysis of the turbine blades, with stated combustion assumptions, done as a new 3D steady heat-transfer solve in Talos
- a 3D map of the hottest metal temperature
- compressor and turbine maps, in a new section at the end of the notebook

## A. Talos: steady heat transfer (library)
- [ ] **`Material.conductivity`:** W/m·K, a constant or a table of (T, k); written to the deck as `*CONDUCTIVITY`.
- [ ] **New loads in `loads.py`** (SI inputs, converted to mm-N-MPa when the deck is written):
  - `Film(region, sink_temperature [K], coefficient [W/m²K])`
  - `FixedTemperature(region, T)`, written as `*BOUNDARY` on degree of freedom 11
  - `HeatFlux(region, q [W/m²])`
- [ ] **Heat deck in `ccx.py`:**
  - `*HEAT TRANSFER, STEADY STATE`
  - `*FILM` on element faces, resolved through the same face code the `Pressure` loads use
  - `*NODE FILE NT`
- [ ] **`StructuralModel.solve_heat(workdir, *, progress, cache=)`:**
  - a public wrapper around a private `_solve_heat`; results have kind `talos.heat`
  - metrics: `temperature_min`, `temperature_max`, `max_temperature_location`, and the heat flow through each film region
  - the cache keeps the deck, the `.frd` file, the mesh and the log
- [ ] **`FieldResults.temperature` in `frd.py`:** read from the NDTEMP block.
- [ ] **`NodalTemperature(source)` load:**
  - takes a heat Result or an `.frd` path and puts the 3D field into the stress solve on the same mesh
  - the yield safety factor is evaluated at each node's own temperature
- [ ] **`viz.py`:**
  - `plot_results(field="temperature")`, with `results_to_pyvista` carrying the nodal temperature
  - mark the hottest point
  - a slow-motion option for `animate`, `degrees_per_frame`: at 125 000 rpm the true angle aliases
- [ ] **Mock (`vegeta.mock`):**
  - mock `solve_heat`: hot at the tips, cool toward the bore, built from the films' sink temperatures, with the real metric keys
  - add a temperature field to the mocked FRD reader
  - make the new plot field a placeholder
  - fix the mock FEA's thermal keys to match the real ones: `temperature_min` and `temperature_max` (drop `max_temperature`), and `safety_factor_temperature` in K
- [ ] **Tests:**
  - deck text (no ccx), unit conversions, validation, cache kind, mock tests, `test_independence`
  - one real solve, marked slow, for the user to run
- [ ] **Docs:** a heat-transfer section in `docs/talos.md`.

## B. Movies
- [ ] **Helper:** `dedalus.viz.spin_movie(parts, path, *, rpm, axis=(1,0,0), rotating=..., cut_plane=..., scalars=..., degrees_per_frame=6, seconds, fps)`.
  - Renders off-screen with pyvista, writes frames with cv2, and states the slow-motion factor.
  - Mocked in DEBUG as a placeholder file.
- [ ] **Rotor in its casing:** the impeller and turbine at their real axial stations, inside the engine casing cut away along a plane.
- [ ] **Hot turbine spinning:** coloured by the solved temperature, with the hottest spot marked (`tviz.animate(field="temperature")`).
- [ ] **Stress while spinning:** the hot turbine and the impeller (`tviz.animate`, slow motion).
- [ ] **Output:** `output/28_microjet/`, shown with `IPython.display.Video`.

## C. Notebook 28: blade heat analysis
A new section after the existing wheel FEA, which stays as it is (cells 18–20).
- [ ] **Combustion and gas assumptions table:**
  - Jet-A1, LHV 43 MJ/kg
  - burner efficiency from the calibrated `ENGINE`
  - mean TIT from `mj.solve` at full speed
  - pattern factor ~0.25, giving peak gas temperature T4 + PF·(T4−T3)
  - rotor-relative total temperature
  - gas properties at the film temperature
  - radiation neglected; no blade cooling
- [ ] **Film boundaries, with h and sink temperatures shown in a table before solving:**
  - blades and rim: a blade Nusselt correlation, Nu ∝ Re^0.68 Pr^0.33 on the chord
  - disc faces: a rotating-disc correlation, with the sink at T3
  - bore: cooling air or a bearing temperature
- [ ] **Solves:** mesh the turbine once, run `solve_heat(cache=...)`, then a stress solve with `Centrifugal` + `NodalTemperature`.
- [ ] **3D temperature map:** hottest point marked, with its value and location.
- [ ] **Comparisons:** against `mj.metal_temperatures` and against the existing radial-profile result.
- [ ] **Limits flagged:** about 1100 K for short creep life, about 1255 K for incipient melting.
- [ ] **Sensitivity:** of the hottest metal temperature to pattern factor and h (±30 %).

## D. Notebook 28: performance maps
A new final section, before the export.

In `boreas.microjet`, pressure ratio depends only on rpm, so there is no constant-speed characteristic. The characteristic models go in visible notebook cells, anchored so that the running line from `mj.solve` (idle to 105 %) lies on them.
- [ ] **Compressor pressure-ratio map:** pressure ratio against corrected flow at constant corrected speed, with surge and choke lines and the running line. The CFD speed line is overlaid when it exists.
- [ ] **Compressor efficiency map:** efficiency islands.
- [ ] **Surge-margin / operating-margin map.**
- [ ] **Turbine swallowing-capacity map:** corrected flow against expansion ratio, with the NGVs choking at the `a4` throat.
- [ ] **Turbine efficiency map:** against expansion ratio and blade-speed ratio U/C₀, anchored to `eta_t`.
- [ ] **Turbine specific-work / power map.**
- [ ] **Turbine blade-loading map (Δh/U²):** with the zero-power-balance contour, where turbine power = compressor power / mechanical efficiency.

## E. Export and verification
- [ ] **`microjet.json`:**
  - add the hottest metal temperature and its location, the hot-stress safety factor, and map summaries
  - keep every existing key: notebook 29 reads them
- [ ] **Quick suites with mocks:** talos, mock and cache.
- [ ] **Debug run:** `scripts/debug_notebooks.sh 28_ 29_`.
- [ ] **Real solves (the user's):** `jupyter nbconvert --to notebook --execute 28_microjet.ipynb --output 28_microjet-run.ipynb`.
