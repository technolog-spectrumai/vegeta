<!-- Detailed design for TODO.md (notebook 28: heat solve, maps, movies). Read from the code, not yet implemented. -->
# Implementation plan: notebook 28 microjet (movies, 3D blade thermal analysis, compressor and turbine maps, 3D max-temperature map)

This plan comes from reading the code only. Nothing was edited. I had no Write tool, so the plan is in this message rather than in the plan file. Line numbers refer to the current tree.

## 0. What the code does today, and what that means for the design

- **Mesh key.** `StructuralModel` meshes and solves in one workdir (`model.py:129-255`). The mesh key is the geometry hash, units, the tagged regions and the mesh settings (`model.py:109-116`). Loads are not in it. So a heat model and a stress model share a mesh only if their regions and `MeshSettings` are identical. `solve()` writes `model.inp/.frd` and `summary.json`. A heat job can sit in the same directory if it uses its own job name, the way `_solve_modes` uses `"modes"` (`model.py:324`).
- **Element faces.** `element_faces()` (`mesh.py:175-188`, face numbers from `TET_FACES` at `mesh.py:19`) already maps surface triangles to tetrahedron faces. That mapping is proven for `*SURFACE ... S<n>` with `*DLOAD P` (`ccx.py:49-51,94`). `*FILM` uses the same face numbers with the label `F<n>`.
- **Temperature loads are RadialTemperature-only.** `write_inp` and `_summarise` only recognise `RadialTemperature` (`ccx.py:53-72,110-114`; `model.py:83-87,438-452`).
- **The .frd parser already handles any block name.** `frd.py:55-90` would read an `NDTEMP` block. `FieldResults` just has no temperature accessor.
- **Plots crash on heat results.** `viz.results_to_pyvista` assumes a displacement field (`viz.py:62`), so it fails on a heat-only .frd. `animate` uses the true shaft angle (`viz.py:203`), which aliases at 125 000 rpm.
- **Dedalus has no movie writer or logo.** It also must not import talos or aeromant (`tests/dedalus/test_independence.py`). So a "rotor in casing" helper in dedalus needs its own copy of `_watermark.py` and `assets/logo.png`, plus a package-data line. That is the existing convention (`talos/_watermark.py` docstring; `pyproject.toml:49-52`).
- **The CAD frames do not line up.**
  - The turbine disc is centred on x=0 (`turbojet.py:184`), and so is the impeller's inducer plane.
  - The engine part is a solid revolved down to the axis. Its intake lip is at x=0, with a recess to 0.06 L (`turbojet.py:235-248`).
  - So the movie needs explicit axial placements. Those are a picture assumption and stay visible in the notebook.
  - Tessellating the engine gives only its skin, so clipping it with a plane gives an open cut-away shell. That is good enough for the casing.
- **The cycle has no characteristic.** `_compressor` (`microjet.py:195-203`) makes pressure ratio and work depend on rpm only. The NGV is an orifice from P4 to √(P4·P5) (`microjet.py:256`), so the turbine's corrected flow against expansion ratio is a single curve for all speeds. That is the cycle's own turbine flow characteristic, and the map can show it exactly.
- **Notebook 29 only reads `engine` and `map` from `microjet.json`.** Adding keys is safe.
- **Mock mismatch.** `mock.FEA` (`mock/__init__.py:80-81`) emits `max_temperature` and `safety_factor_temperature=3.0` on every model.

## 1. Library changes

### 1a. Talos: units (`talos/units.py`)
- Add `UnitSystem` fields with defaults: `power` ("mW" / "W"), `conductivity` ("mW/(mm K)" / "W/(m K)") and `film` ("mW/(mm^2 K)" / "W/(m^2 K)").
- Add explicit conversions, so nothing is converted silently:
  - `SI_FACTORS = {"mm-N-MPa": {"conductivity": 1.0, "film_coefficient": 1e-3, "heat_flux": 1e-3, "power": 1e3, "specific_heat": 1e6, "density": 1e-12, "stress": 1e-6, "length": 1e3}, "m-N-Pa": {...all 1.0}}`
  - `from_si(value, quantity, units)` and `to_si(...)`.
- Worked examples: k in W/(m K) is the same number in mW/(mm K); 2870 W/(m² K) becomes 2.87 mW/(mm² K); 7910 kg/m³ becomes 7.91e-9 t/mm³.

### 1b. Talos: materials (`materials.py`)
- Add `conductivity: float | None = None` (model units).
- Add `conductivity_table: tuple | None = None`, rows `(T [K], k)` with T increasing.
- Add `specific_heat: float | None = None` (optional; steady state does not need it).
- Validate k > 0 and increasing T.
- Add `conductivity_at(T)`.
- In `StructuralModel.config()` (`model.py:100`), leave out these three keys when they are `None`. Then the solve keys of existing models do not change, and `ensure` does not re-solve old workdirs.

### 1c. Talos: regions (`regions.py`)
- **`SurfacesInCylinder(name, r_min, r_max, axial=None, axis="x", center=(0,0,0), tol=None)`.** Selects surfaces lying completely inside a cylindrical shell (and an optional axial range).
  - Axis-aligned boxes cannot pick blades all the way round, and the bounding box or centre of mass of a surface of revolution sits on the axis. So sample each face's boundary curves instead: `getBoundary([(2,tag)], oriented=False)`, then `getParametrizationBounds(1,c)` and `getValue(1,c,u)`, about 9 points per curve.
  - Fallback for faces with no curves: bounding-box corners.
  - Document the limit: a face whose interior bulges outside the radius range of its boundary (a dome) is misjudged.
- **`SurfacesExcept(name, region, minus: tuple)`.** Returns `region.select(...)` minus the union of `m.select(...)`. It is a generic composite, needed to keep the cooled disc faces apart from the bore.
- Extend the `Region` union (`regions.py:78`).

### 1d. Talos: loads (`loads.py`)
- **Heat boundaries:**
  - `Film(region, coefficient, sink_temperature: float | RadialTemperature)`. Coefficient in model units, sink in K. A `RadialTemperature` sink is evaluated at each element face's centroid, which gives the radial gas profile on the blades for free.
  - `FixedTemperature(region, temperature)`.
  - `HeatFlux(region, flux)`, in model power per area, positive into the solid.
  - All frozen dataclasses with validation (h > 0, T > 0).
- **`TemperatureField(frd: str | Path)`.** A solved nodal temperature field used as a load.
  - Frozen. `digest` is `field(init=False)`, set to the file's sha256 in `__post_init__`, so the solve key follows the data. Raise `FileNotFoundError` if the file is missing.
  - `classmethod from_result(res)` takes `res.artifacts["frd"]`.
- **One nodal-temperature interface:** add `nodal(node_ids, coords)` to both classes.
  - `RadialTemperature.nodal` returns `at(coords)`.
  - `TemperatureField.nodal` reads the .frd with `read_frd` imported inside the function, so the DEBUG patch applies. It looks up `NDTEMP` by node id. It raises `ValueError("temperature field does not cover node ... solved on another mesh?")` on missing ids, or when the coordinates of matching ids differ by more than 1e-6 of the model size.
- `THERMAL_LOADS = (RadialTemperature, TemperatureField)`. Add the new classes to `Load` and define `Boundary = Film | FixedTemperature | HeatFlux`.

### 1e. Talos: frd (`frd.py`)
- Add `FieldResults.temperature` returning `fields["NDTEMP"][:, 0]`.

### 1f. Talos: deck writer (`ccx.py`)
- **In `write_inp`:** use `isinstance(l, THERMAL_LOADS)` at `ccx.py:53` and `:111`, and `temperatures = load.nodal(mesh.node_ids, mesh.coords)`. The deck for `RadialTemperature` stays exactly as it is, so `test_thermal.py` keeps passing.
- **New `write_heat_inp(path, mesh, material, boundaries, initial_temperature) -> dict`:**
  ```
  *HEADING / Talos steady heat transfer
  *NODE, NSET=NALL ... ; *ELEMENT, TYPE=C3D10|C3D4, ELSET=EALL   (same writer code as write_inp; factor a _nodes_elements(lines, mesh) helper)
  *NSET, NSET=N_<R>          (FixedTemperature regions)
  *MATERIAL, NAME=MAT / *CONDUCTIVITY / "k, T" rows (or one "k")  [*DENSITY, *SPECIFIC HEAT only if given]
  *SOLID SECTION, ELSET=EALL, MATERIAL=MAT
  *INITIAL CONDITIONS, TYPE=TEMPERATURE / NALL, T0
  *STEP / *HEAT TRANSFER, STEADY STATE / 1., 1.
  *BOUNDARY / N_<R>, 11, 11, T
  *FILM / "<eid>, F<face>, <sink>, <h>"   one line per element face (explicit, robust)
  *DFLUX / "<eid>, S<face>, <q>"
  *NODE FILE / NT
  *END STEP
  ```
  - Face numbering reuses `element_faces`.
  - Raise `ValueError` when an element face belongs to two Film/HeatFlux boundaries.
  - Return a book with the face count and area per boundary.

### 1g. Talos: new `talos/thermal.py` with `ThermalModel`
A separate class, because `StructuralModel` demands supports and loads (`model.py:74-77`).

```python
@dataclass
class ThermalModel:
    geometry; units; material; regions; boundaries: Sequence[Film|FixedTemperature|HeatFlux]; mesh_settings
    name="talos_heat"; notes=""; initial_temperature: float | None = None
    def config(); def _mesh_key(); key (property)
    def mesh(workdir, progress=False, *, cache=None) -> Result      # kinds ("talos.mesh",), MESH_KEEP, restore_to=workdir
    def mesh_is_current(workdir)
    def solve(workdir, *, executable="ccx", threads=1, timeout=None, progress=False, cancel=None, cache=None) -> Result
        # _cache.cached(..., keep=SOLVE_KEEP, kinds=("talos.heat",)) around _solve
    def _solve(...): job "heat" (heat.inp/.frd, heat_summary.json; artifacts frd, mesh, inp, ccx_log, summary)
HEAT_METRICS = ("n_nodes","n_elements","element_type","temperature_min","temperature_max","temperature_mean",
  "max_temperature_node","max_temperature_location","min_temperature_location","region_temperatures",
  "region_area","film_heat_flow","heat_flow_in","heat_balance")
```

- **Checks in `__post_init__`:**
  - the material has conductivity or a conductivity table;
  - there is at least one Film or FixedTemperature (otherwise the problem is ill-posed);
  - every region referenced exists;
  - region names are unique;
  - the material's unit tag matches.
- **Share code in `model.py`, keeping the diff small:**
  - Move the body of `_mesh` into `_run_mesh(model, workdir, progress)` (it is duck-typed) and make `StructuralModel._mesh` call it.
  - Add `mesh_key_of(geometry, units, regions, mesh_settings)` and `_mesh_is_current(model, workdir)`.
  - Then a `ThermalModel` and a `StructuralModel` with the same regions and settings share one mesh by construction.
  - Add `copy_mesh(src_dir, dst_dir) -> Path`, which copies `mesh.msh` and `mesh_summary.json`, for further cases on the same mesh.
- **Summary (`_summarise_heat`):**
  - `read_frd(heat.frd).temperature` gives the min, max and mean, the hottest node and its location.
  - Per boundary region: nodal min, mean and max, and area (`mesh.triangle_areas`).
  - Film heat flow per region: Q = Σ h·A·(T_sink − mean corner T), positive into the solid. Then `heat_flow_in` and `heat_balance = ΣQ / Σ|Q|`.
  - Messages: steady, no radiation, film flows estimated from corner temperatures, units.
- **Exports:** add `ThermalModel, Film, FixedTemperature, HeatFlux, TemperatureField, SurfacesInCylinder, SurfacesExcept, copy_mesh, from_si, to_si` to `talos/__init__.py`.
- No talos docstring line may start with "from vegeta".

### 1h. Talos: `model.py` thermal handling
- In `__post_init__` (`:83-87`) and `_summarise` (`:438-452`), use `THERMAL_LOADS` and `temps = thermal[0].nodal(fr.node_ids, fr.coords)`.
- The metric keys stay the same: `temperature_min/max`, `safety_factor_yield/location/temperature/von_mises`.

### 1i. Talos: visualisation (`viz.py`)
- **`results_to_pyvista`:** write displacement only if a `DISP` field exists (otherwise zeros), and add `grid["temperature"]` when `NDTEMP` exists.
- **`plot_results`:** add `cmap="turbo"` and `mark_max=False` (a sphere and label at the field's maximum). Change the title when there is no displacement.
- **`animate`:** add `degrees_per_frame: float | None = None`, `mark_max=False`, `cmap="turbo"` and `title=None`.
  - With `degrees_per_frame`, the angle is k·dpf and the label reads "125 000 rpm shown 5 208× slower (6°/frame, 8.0 µs real per frame)". Factor = rpm·6/(fps·dpf).
  - Without it, behaviour is unchanged.
  - With `field == "temperature"` the scalar is never ramped (temperatures are not a load).
  - The max marker follows the rotated point each frame.
  - Optional speed-up: take `extract_surface()` once before the frame loop.

### 1j. Dedalus: `dedalus/viz.py`, plus `_watermark.py` and `assets/logo.png` copied from talos
Add package data `"vegeta.dedalus" = ["assets/*.png"]`.

```python
@dataclass
class MoviePart: shape; rotating: bool = True; offset=(0,0,0); color=None; opacity=1.0; scalars=None (array or callable(points)); cut=False; label=""
def slow_motion(rpm, degrees_per_frame, fps) -> dict   # {"factor", "real_time_per_frame_s", "video_deg_per_s"}
def spin_movie(path, parts: dict[str, MoviePart], *, rpm, axis=(1,0,0), center=(0,0,0), degrees_per_frame=6.0,
               seconds=6.0, fps=24, cut_normal=(0,0,1), size=(1280,720), cmap="inferno", clim=None, scalar_title="",
               camera=None, title=None, tolerance=0.05) -> Path
```
- Tessellate once with `to_pyvista` (pyvista meshes are taken as they are), then translate by `offset`.
- Clip the `cut` parts once with the plane through `center`.
- For each frame, call `rotate_vector(axis, k·dpf, point=center)` on the rotating parts.
- Overlay the slow-motion text. Write frames with cv2 and the watermark.
- The docstring says to keep `degrees_per_frame` below half the blade pitch to avoid wagon-wheel aliasing.

### 1k. Boreas: `microjet.py`
- Make `flow_function(p0, t0, p_back, g)` public, keep `_flow_function = flow_function` (the existing test and internal calls use it), and add it to `__all__`.
- The map assumptions stay in the notebook.

## 2. Mock changes (`vegeta/mock/__init__.py`)
- **Fix the existing mismatch.** Remove `max_temperature` and `safety_factor_temperature` from `FEA`. In `_fea_metrics`, when a thermal load is present:
  - for `RadialTemperature`, use the min and max of its temperatures;
  - for `TemperatureField`, use `frd_fields(load.frd)`;
  - then set `temperature_min/max`, `safety_factor_temperature = temperature_max`, `safety_factor_yield = material.yield_at(Tmax)/12` (otherwise ys/12 or None), `safety_factor_location` and `safety_factor_von_mises = 12`;
  - with no thermal load, emit none of these keys.
- **`_talos_heat_solve(self, workdir, ...)`:**
  - Take the sinks (floats, or min/max of a `RadialTemperature`) and fixed temperatures.
  - Set `t_min = lo + 0.05·(hi − lo)` and `t_max = hi − 0.05·(hi − lo)`.
  - Write `heat.frd` containing `vegeta.mock: heat {"t_min":..,"t_max":..}`.
  - Return `Result("talos.heat")` with exactly the `HEAT_METRICS` keys: region dictionaries for every boundary region, film flows balanced to a sum of 0, `heat_balance` 0.0, `max_temperature_location` equal to the last mock node, `[90, 0, 0]`. Save `heat_summary.json`.
- **`frd_fields(path)`:** if the file starts with `vegeta.mock: heat`, set NDTEMP to `linspace(t_min, t_max)`. Otherwise keep the current fields, so the fatigue test still holds.
- **`_spin_movie(path, *a, **k)`:** writes a placeholder file.
- **In `install()`:**
  - patch `talos.ThermalModel.mesh` with the existing `_talos_mesh` (same signature, and it uses `config()` / `_mesh_key()`);
  - patch `talos.ThermalModel.solve` with `_talos_heat_solve`;
  - patch `dedalus.viz.spin_movie` with `_spin_movie`.
  - `tviz.animate` and `plot_results` are already placeholders. `copy_mesh` and `TemperatureField` run for real on the mock files.
- Update the module docstring.

## 3. Tests (none run ccx, gmsh or OpenFOAM, except the marked ones)
- **`tests/talos/test_heat.py`** (no ccx), using the one-tet `MeshData` as in `test_thermal.py`:
  - the deck has `*HEAT TRANSFER, STEADY STATE`, conductivity rows `"10.9, 366"`, `"1, F1, 1000, 2.5"` for triangle [1,2,3], `N_FIX, 11, 11, 400`, `*NODE FILE\nNT` and the initial condition;
  - a radial sink is evaluated at the face centroid;
  - overlapping film regions raise;
  - the model checks (no conductivity, no Film or FixedTemperature, unknown region);
  - `ThermalModel._mesh_key() == StructuralModel._mesh_key()` for the same dummy geometry file, regions and settings;
  - `_summarise_heat` on a hand-written .frd (fixed-width node and NDTEMP blocks) gives the temperature, the location and the film flows.
- **`test_thermal.py` additions:**
  - `TemperatureField` digest and nodal lookup from a hand-written .frd;
  - a missing node or moved coordinates raise;
  - `write_inp` with a `TemperatureField` writes `*TEMPERATURE` values and `*EXPANSION`;
  - `RadialTemperature` plus `TemperatureField` gives "one temperature field";
  - `FieldResults.temperature`.
- **`test_units_materials_loads.py`:** `from_si` / `to_si` round trip and factors; conductivity-table validation; Film / HeatFlux validation.
- **Regions:** `SurfacesInCylinder.select` against a fake gmsh model (rings at r=10 and r=20, a blade spanning 18–30, an axial range); `SurfacesExcept`.
- **Visualisation** (pyvista via importorskip, off-screen; no mesher): `results_to_pyvista(FieldResults with only NDTEMP, mesh=tet())` has `temperature` and zero U. `animate(..., degrees_per_frame=6, field="temperature", mark_max=True)` writes a file with `seconds·fps` frames (importorskip cv2).
- **Dedalus `tests/dedalus/test_spin_movie.py`:** the `slow_motion` numbers (125 000 rpm, 6°, 24 fps gives about 5 208×); `spin_movie` with two pyvista primitives (frame count via cv2); a watermark test as in talos. The independence tests stay green.
- **Boreas:** `mj.flow_function is mj._flow_function`, and the NGV corrected flow is independent of P4 and T4.
- **`tests/mock/test_mock.py`, new `test_heat_through_the_mocks`:**
  - the metric key set equals `talos.thermal.HEAT_METRICS`;
  - `read_frd(heat frd).temperature` has shape (N,) and its min and max equal the metrics;
  - `TemperatureField` from the mock .frd, used in a solved `StructuralModel`, gives `temperature_min/max` and `safety_factor_temperature` inside the range, and no `max_temperature`;
  - a model without thermal load has no `"temperature_max"` key (use `in`, not indexing: `Metrics.__missing__` returns 1.0);
  - `dviz.spin_movie` and `tviz.animate` write placeholders;
  - `uninstall` restores `ThermalModel.solve` and `spin_movie`.
- **Slow, for the user** (`requires_gmsh`, `requires_ccx`, `slow`):
  - a 100×10×10 bar with fixed 400 K at x=0 and a Film at x=100: compare T(L) with 400 − qL/k, where q = ΔT/(L/k + 1/h), within 0.5 %, with `heat_balance` about 0;
  - a uniform 500 K field passed through `TemperatureField` into a stress solve: tip ux ≈ α·ΔT·L within 2 %;
  - film face labels on C3D10.

## 4. Docs
- **`docs/talos.md`:** a new section "Heat transfer (`ThermalModel`) and solved temperature fields (`TemperatureField`)" covering:
  - the API and the unit table (`from_si`) with the deck cards and F1–F4 face labels;
  - metrics and artifacts (`heat.inp/.frd`, `heat_summary.json`) and cache kind `talos.heat`;
  - sharing a mesh (same regions and settings; `copy_mesh`; when caching, also cache the mesh with `mesh(..., cache=)` so the stress solve finds it);
  - limits (steady, no radiation, h constant per region, sink may be radial).
  - Also: the new regions in "Regions", `temperature` / `mark_max` in Results, and `degrees_per_frame` in `animate`.
- **`docs/visualisation.md`:** rows for `animate(..., degrees_per_frame=)` and `dedalus.viz.spin_movie`.
- **`docs/dedalus.md`:** `spin_movie`.
- **`docs/boreas.md`:** `flow_function`.

## 5. Notebook 28, cell by cell (new numbering)
Keep cell ids, give new cells new ids, clear outputs.

- **Cell 1 (overview):** a 10-step list: maps (6), thermal (8), movies (4 and 9), export (10).
- **Cell 3:** add `Video` to the display import, and `MOVIES = Path("output")/"28_microjet"` (mkdir).
- **Cells 15–16, after the CAD cells: "Rotor in its casing".**
  - `LAYOUT` assumptions: inducer at x = 0.06 L + |x_spinner| + 2 mm, turbine disc at 0.70 L. Check that the turbine tip is inside the casing radius at that x (interpolate `engine_profile`).
  - `DEG_PER_FRAME = 6`, `FPS = 24`, with `assert DEG_PER_FRAME < 0.5*360/P["turbine_blades"]`.
  - `SM = dviz.slow_motion(...)`, printed as "slowed N×, one video second = x ms real".
  - Parts:
    - casing: `MoviePart(PARTS["engine"], rotating=False, cut=True, opacity=0.3)`
    - impeller: offset to its station
    - turbine: offset to its station
    - shaft: `pv.Cylinder` with radius bore/2
  - The md says the NGV and combustor are not drawn and that this is not a simulation.
  - `ROTOR_MOVIE = dviz.spin_movie(MOVIES/"rotor_in_casing.mp4", ...)`, then `display(Video(..., embed=False, width=960))`, inside try/except ImportError.
- **Cells 17–19:** the existing CFD cells.
- **Cells 20–22: "## 6. Compressor and turbine maps".** These go after the CFD so its speed line can be overlaid.
  - **Cell 20 (md):** the assumptions are generic, not measured, anchored on the cycle's running line, and not fed back into the cycle.
  - **Cell 21 (compressor):**
    - Corrected flow `mc = m√θ/δ` with θ = T2/288.15 and δ = P2/101325; corrected speed N/√θ.
    - Speeds n = 0.30…1.05.
    - The running point at each speed comes from `mj.solve`.
    - Explicit constants: `SURGE_RANGE=0.22`, choke range `0.20-0.12n`, `PR_AT_CHOKE=0.55`, `ETA_FALL=1.2`, `ETA_PEAK_SHIFT=0.03`.
    - Speed line on u = (mc − mc_s)/(mc_ch − mc_s): PR − 1 = (PR_op − 1)·[1 + B(u_op³ − u³)] with B = (1 − g_ch)/(1 − u_op³). It is smooth and passes through the running point.
    - Efficiency: η = η_op·[1 − k((mc − mc_pk)/(mc_ch − mc_s))²].
    - Plots: surge and choke lines; efficiency islands by `tricontour` on a dense set of lines; running line at static and at 150 m/s; the CFD speed line when `SPEEDLINE` is a DataFrame with ok rows (corrected flow column).
    - Table: surge margin along the running line.
  - **Cell 22 (turbine):**
    - m_gas = m(1 + f) with f = fuel/air; `mt = m_gas√T4/(P4/1e3)`; ER = P4/P5; N/√T4.
    - U at the mean radius `(r_rim+r_tip)/2` from `turbine_stations(P)`; C0 = √(2·cp_g·T4·(1 − ER^(−(γ−1)/γ))).
    - NGV curve: `a4·mj.flow_function(P4, T4, P4/√ER, G_GAS)·√T4/P4`. Assert the running points lie on it (rel 1e-6; a real check of the cycle).
    - Choke at ER = crit⁻².
    - η(U/C0) = η_t·[1 − K((ν − ν_opt)/ν_opt)²], with ν_opt at the design point (an assumption) and K = 1.1.
    - Two panels: flow against ER, and η contours on ER against N/√T4, with the running line.
    - Table: ν and η from the map against the cycle's constant η_t.
- **Cells 23–25:** the existing FEA cells, heading renumbered to "## 7.". Code unchanged.
- **Cells 26–35: "## 8. Turbine wheel temperature: combustion to a 3D heat-transfer solve".**
  - **Cell 26 (md):** the chain combustion → rotor-frame gas temperature → film coefficients → CalculiX `*HEAT TRANSFER` → stress.
  - **Cell 27 (combustion and gas path), shown as a DataFrame with columns quantity, value, unit, source/assumption:**
    - Jet-A1 LHV = `ENGINE.lhv`; η_b fitted; FAR and fuel flow from `HOT`; T3, T4, T5, P4, P5.
    - OTDF = 0.25, giving the NGV peak T4 + OTDF·(T4 − T3) (reported only).
    - RTDF = 0.10, with a parabolic radial profile peaking at 60 % span (the rotor sees the circumferential average).
    - NGV exit swirl from Euler: cθ = w_t/U_m with axial exit; free vortex.
    - T0,rel(r) = T4(r) − (2U_m·cθ_m − U(r)²)/(2cp_g). About 1028 K at the mean radius, against 1051 K from the `metal_temperatures` shortcut.
    - W1 and static density at √(P4·P5).
  - **Cell 28 (film coefficients and BC table):**
    - Gas properties at ~1000 K: k 0.068 W/(m K), μ 4.2e-5 Pa·s, Pr 0.72.
    - Blades: Re_c from W1 and the chord, Nu = 0.285·Re^0.68·Pr^(1/3) (an Ainley-type average, ±30 %). This gives about 2.9 kW/(m² K).
    - Rim: 0.7 × the blade value (endwall).
    - Disc faces: free-disc turbulent average Nu = 0.015·Re_φ^0.8 with b = r_rim and air at T3, P3. About 1.2 kW/(m² K).
    - Sinks: front face T3; back face T3 + 0.3(T5 − T3).
    - Bore: shaft at T3, h = 2000 W/(m² K) (contact assumption).
    - Radiation check: 4εσT³ ≈ 180 W/(m² K), under 7 % of the blade value, so neglected.
    - The table shows SI values next to `talos.from_si(h, "film_coefficient", "mm-N-MPa")`.
  - **Cell 29 (model and solve):**
    - `IN713C_K = replace(IN713C, conductivity_table=((366,10.9),(589,13.4),(811,17.0),(1033,20.9),(1144,23.0)))`, labelled as handbook values to replace.
    - Regions:
      - `bore`: the existing box
      - `rim`: `SurfacesInCylinder(r_rim-1-tol, r_rim+tol)`
      - `blades`: `SurfacesExcept(SurfacesInCylinder(r_rim-1-tol, r_tip+2), (rim,))`
      - `disc_front` / `disc_back`: `SurfacesExcept(SurfacesInCylinder(0, r_rim-1+tol, axial=(−∞,0) or (0,∞)), (bore,))`
    - Boundaries: a radial sink on the blades, the rim at the profile's hub value, the disc faces and the bore.
    - `THERMAL = talos.ThermalModel(r_turb.step, ..., MeshSettings(element_size=ELEMENT))`; `TW = ROOT/"turbine_thermal"`; `THERMAL.mesh(TW)`, then `HEAT = THERMAL.solve(TW, threads=CPU)`.
    - Summary table: T min and max, hottest point in (x, r, span %), per-region temperatures against `MT`, film flows and balance, sum of region areas against `PARTS["turbine"].surface_area` (printed, no asserts on solver numbers).
  - **Cell 30 (3D max-temperature map):**
    - `tviz.show(tviz.plot_results(HEAT, field="temperature", mark_max=True, cmap="inferno"))`, and a clipped view.
    - From `talos.read_frd(...).temperature`: a meridional projection with all nodes scattered in (x, r) sorted by ascending T, so the visible colour is the maximum over θ. This is robust with the mock's degenerate coordinates.
    - Histogram against the assumed limits: 1100 K (creep-limited, tens of hours) and 1255 K (short-time ceiling).
    - Correction to the brief: IN713C's solidus (incipient melting) is about 1530 K, not 1255 K. Label 1255 K as a service limit.
  - **Cells 31–32 (sensitivity):** `thermal_model(rtdf, k_gas, k_disc)` and `copy_mesh` into `TW/"sens_<case>"`.
    - Cases: baseline, RTDF 0.2, h_gas ±30 %, h_disc ±30 %. QUICK runs only baseline and h_gas ±30 %.
    - Table: max T, blade/rim/bore mean, Δ against baseline, and flags.
  - **Cells 33–35 (thermo-mechanical stress with the 3D field):**
    - `TFIELD = talos.TemperatureField.from_result(HEAT)`.
    - `turbine_hot_3d` (Centrifugal at rpm_max plus TFIELD) is solved in TW, next to the heat files.
    - `turbine_over_3d` (1.15×) and `turbine_hot_radial_same_mesh` (Centrifugal plus `TURBINE_T`) are solved in copied-mesh directories.
    - All use the same `REGIONS_T` and settings, so the mesh key matches.
    - Comparison table against `FEA["turbine_hot"]`: max von Mises, yield safety factor at T, temperature at that point, T range.
    - Plot von Mises with `mark_max`.
- **Cells 36–38: "## 9. Movies of the hot wheels".**
  - "Hot turbine spinning": `tviz.animate(HEAT, MOVIES/"turbine_hot_spinning.mp4", field="temperature", rpm=ENGINE.rpm_max, axis="x", degrees_per_frame=DEG_PER_FRAME, load_ramp=False, mark_max=True, cmap="inferno", seconds=6)`.
  - "Stress while spinning": `animate(S3D, ..., field="von_mises", load_ramp=True, mark_max=True)`, and the same for `FEA["impeller_hot"].tool_results[-1]`.
  - Each movie is guarded by `.ok` and shown with `Video`.
- **Cells 39–40: Export ("## 10.").** Keep all existing keys. Add:
  - `"maps": {"compressor": {...assumptions, surge margins}, "turbine": {...}}`
  - `"thermal": {"combustion": ..., "boundary_conditions": ..., "heat": {selected HEAT metrics}, "sensitivity": ..., "fea_3d": ...}`
  - Cast all values to plain float, int or list, because `mj.export` uses `json.dumps` without a `default`.

## 6. Verification
1. `pytest vegeta-cli/tests/talos vegeta-cli/tests/dedalus vegeta-cli/tests/boreas vegeta-cli/tests/mock -m "not slow"`. Solver tests skip without ccx or gmsh. The independence tests must pass.
2. `scripts/debug_notebooks.sh 28_` must finish ok.
3. For the user: the slow tests with ccx, then the real nbconvert run of notebook 28 (QUICK first).

## 7. Risks
- **`*FILM` with explicit F-labels on C3D10.** The face numbering is shared with the proven S-labels, so the risk is low. The slow 1D film test confirms it. Steady state with temperature-dependent k is nonlinear, but ccx iterates.
- **Material cards.** ccx may want `*ELASTIC` or `*SPECIFIC HEAT` even for steady heat transfer. Pass `specific_heat` in the notebook as a precaution: 440 J/(kg K) is 4.4e8 in mm-N-MPa units. The slow test will show which.
- **NT parsing.** The `NDTEMP` block name is parsed by the existing generic code. No `HFL` output, to stay simple; heat flows come from temperatures.
- **Mesh reuse.** Any difference in regions or `MeshSettings` between heat and stress fails with the existing "mesh out of date" message. `TemperatureField` also checks node coverage and coordinates. When using `cache=`, cache the mesh too: solve entries do not restore it.
- **Region selection.** A wrong tolerance silently moves faces between regions. The area table and the "unassigned area" print make this visible.
- **Movies under DEBUG.** All movie functions are mocked; only `pv.Cylinder` runs for real, with no rendering. Real runs need off-screen VTK (Xvfb or `PYVISTA_OFF_SCREEN`), and rendering the full 23-blade wheel for 144 frames takes minutes (use the `extract_surface` speed-up).
- **Mock shape.** Mock results must keep the exact real key sets (`HEAT_METRICS`, thermal FEA keys), or notebook cells pass in DEBUG and break in the real run. The mock test pins this down.

### Critical Files for Implementation
- /home/user/vegeta/vegeta-cli/src/vegeta/talos/model.py
- /home/user/vegeta/vegeta-cli/src/vegeta/talos/ccx.py
- /home/user/vegeta/vegeta-cli/src/vegeta/talos/viz.py
- /home/user/vegeta/vegeta-cli/src/vegeta/mock/__init__.py
- /home/user/vegeta/notebooks/28_microjet.ipynb

The other files touched: `talos/loads.py`, `materials.py`, `regions.py`, `frd.py`, `units.py`, new `talos/thermal.py`, `dedalus/viz.py` (with the new `dedalus/_watermark.py` and `assets/logo.png`), `boreas/microjet.py`, `vegeta-cli/pyproject.toml`, `docs/talos.md`, `docs/visualisation.md`, `docs/dedalus.md`, `docs/boreas.md`, and the tests under `vegeta-cli/tests/{talos,dedalus,boreas,mock}`.