# FEniCSx in Vegeta — where a field solver pays, and how to install it

FEniCSx (dolfinx) is a finite-element library: you write the weak form in UFL, it assembles and solves on an
unstructured mesh, in parallel, and writes XDMF/VTK that PyVista reads. It is the right tool where Hyperion's
closed forms and structured-grid solvers stop: real geometry, coupled fields, moving fronts. It is also a heavy
install (PETSc, MPI), so it lives on `dev_svarog_advanced` and nothing in the main branches imports it.

## Install

- **`./install_local.sh`** does it: a micromamba-made conda-forge environment at `.fenicsx/env` with dolfinx, PyVista and
  ipykernel, the Vegeta packages installed into it without dependencies, registered as the Jupyter kernel **"Python (fenicsx)"**;
  the fenix chapters run their worker in it; every notebook stays on "Python (vegeta)" and reads the runs. `--no-fenicsx` skips it.

- **conda-forge** (the supported way): `micromamba create -n fenicsx -c conda-forge fenics-dolfinx mpich pyvista
  h5py meshio gmsh python-gmsh` then `micromamba activate fenicsx` and `pip install -e vegeta-cli --no-deps`.
- **Docker**: `docker run -ti -v $PWD:/root/vegeta dolfinx/dolfinx:stable` (PyVista headless needs `xvfb` or
  `pyvista.OFF_SCREEN = True`).
- Without the interpreter (`VEGETA_FENICSX_PYTHON` unset and no `.fenicsx/env`) a fenix run is `failed` with every case
  NOT RUN, and the notebooks say so — the repository's convention for a solver that is not on the machine.

## The backend in Hyperion (`hyperion.thermal_fenicsx`)

`thermal.steady_plate(..., backend="auto")` (used in-process where the interpreter *is* the kernel; the notebooks go through fenix) and `thermal.heat_plate(..., backend="auto")` take FEniCSx when dolfinx is
importable and fall back to the numpy solvers otherwise (`backend="fenicsx"` raises when it is missing; `"numpy"` forces
the fallback). The FEniCSx steady solve is Newton on the nonlinear radiation; the transient is implicit Euler with the
apparent-heat-capacity enthalpy method, Picard, and cell deactivation for ablation — the same inputs and the same result
shapes, so notebook 49 and the guard study switch solvers by installing dolfinx. The test
`test_backend_dispatch_and_fenicsx_when_present` compares the two on an aluminium steady case (5 K) and a polycarbonate
burn-through (25 %) and skips (NOT RUN) without dolfinx.

**Smoke-tested here (dolfinx 0.11.0, conda-forge via micromamba):** the steady aluminium case 311 K (numpy Jacobi) against
315 K (FEniCSx Newton, 2 iterations); the guard's 1.5 mm steel under the collimated kilowatt 1222 K against 1232 K
(FEniCSx 0.5 s, Jacobi 3.6 s); the 10 mm black polycarbonate burn-through 26.2 s (cells, 24 × 10) against 30.4 s (FEM,
13 s of compute) — the cell model is 16 % fast, its grid's coarseness, inside the margin 49's verdicts use.
``backend="auto"`` takes FEniCSx for the steady solve and keeps numpy for the transient (whose stall detection the FEM
version does not have yet); ``backend="fenicsx"`` forces the FEM transient.

## The first use: the guard's ablation — book fenix (`workflows/fenix/`)

**fenix** is a book of the workflows layer (code F), the same mechanism as the CFD and FEA chapters: a chapter script
in the Vegeta venv (`workflows/fenix/plate_thermal.py`) that launches a dolfinx worker (`_worker.py`) under the fenicsx
interpreter for each case — a JSON spec in, a `result.json` out: the process pipe between the two environments — and
writes a run directory per the runkit contract (manifest, status, run.log, cases, summary.json). Chapters: F1
`guard-steady` (every barrier's steady state, Newton on the nonlinear radiation), F2 `guard-ablation` (the transients
the closed-form screening could not decide), F3 `sheet-distance` (the reference 2 mm PC sheet at 10–200 m). Variants
`1kW` and `2kW`. Notebooks 49 and 51 take the runs as optional inputs through Yanus; auto mode produces them in
screening quality when the interpreter exists; without it the run is `failed` with every case NOT RUN and the
notebook says so. `wf list` shows the book, `wf batch` can queue its chapters, `yn status` the notebooks that take them.

```bash
wf run F2 --variant 1kW --quality fine            # or: .venv/bin/python workflows/fenix/plate_thermal.py --chapter guard-ablation --variant 1kW
.venv/bin/python workflows/fenix/plate_thermal.py --chapter sheet-distance --variant 2kW --list
```

What it validates: the cell model 49's verdicts rest on — steady states within 1 % (steel 1222 vs 1232 K), burn-through
times within 10–20 % (the cell grid's coarseness), the same verdicts, except the marginal 2 kW / 200 m sheet where
the cell model's stall detection says steady and the FEM burns through at 136 s — a case for the standard's test.
Known difference: the FEM transient has no temperature cap at the failure point (the apparent-capacity band only), so a
brick that keeps absorbing after its surface boils reads a few hundred kelvin above T_fail where the cell model holds the
plateau; the ablation timing is unaffected (the removal is by enthalpy). What it will extend to: a 3-D panel with a rib, a beam at an angle, a panel with a liner (the enclosure, MVP M.2).

## Where else it pays, in the order Vegeta would use it

1. **Talos heat transfer** (todo S.2.3 / the rod, and the weld pool L.1): CalculiX has a thermal solver, but a coupled
   thermo-mechanical problem with temperature-dependent properties and phase change is cleaner in UFL — the pumped rod's
   temperature, stress and fracture margin, the weld pool as a moving source on a plate (the keyhole as a prescribed
   cavity), the fibre spool's cooling.
2. **The thermal lens as a field**: the rod's refractive index map n(r, z) from the FEniCSx temperature field, then
   the aberrated lens (S.2.2) as a phase screen for Hyperion's beams — Innocenti's closed form against the real pump
   profile.
3. **Aeromant's low-speed limit**: Stokes and laminar incompressible flow (the quadcopter's duct, the submarine's
   boundary layers) in a few lines of UFL where OpenFOAM is heavy; not a replacement for RANS.
4. **Chronos' vibration**: the frame's modes as an eigenproblem (SLEPc through dolfinx) on the same mesh Talos uses —
   today Talos asks CalculiX for them.
5. **Boreas / Zephyr**: a potential-flow or Euler section solver is a stretch; XFOIL and NeuralFoil remain the
   tools. FEniCSx does not help here.
6. **Fidia's checks**: a quick elastic or thermal sanity solve on a generated model (does the printed part hold?) —
   a FEniCSx worker behind the sandbox could replace the Gmsh + CalculiX round trip for small parts.

Pattern for each: a design module writes the mesh (Gmsh via Dedalus' STEP, or a built-in rectangle/box), the FEniCSx
script writes XDMF + a `result.json` with the numbers a notebook reads, PyVista renders, Yanus takes the `result.json`
as a `Run` so a missing solve is NOT RUN, not a crash.

## References and codes

- M. S. Alnæs et al., "The FEniCS Project Version 1.5", Archive of Numerical Software 3 (2015); I. A. Baratta et al.,
  "DOLFINx: the next generation FEniCS problem solving environment", Zenodo (2023).
- H. P. Langtangen, A. Logg, *Solving PDEs in Python — The FEniCS Tutorial I*, Springer (2016); J. S. Dokken, *The
  FEniCSx tutorial* (jsdokken.com/dolfinx-tutorial) — the heat equation with Picard/Newton, Robin boundaries, XDMF output.
- V. R. Voller, C. R. Swaminathan, "General source-based method for solidification phase change", Numer. Heat
  Transfer B 19, 175 (1991) — the enthalpy / apparent-heat-capacity methods.
- PyVista: C. B. Sullivan, A. Kaszynski, "PyVista: 3D plotting and mesh analysis through a streamlined interface for
  the Visualization Toolkit (VTK)", J. Open Source Software 4, 1450 (2019).
- Codes doing laser ablation / welding with FEM: **FEniCS-based laser heating examples** in the dolfinx tutorial;
  **Elmer** (thermal with phase change, open), **Abaqus/ANSYS element deletion** (commercial, the usual industrial route),
  **OpenFOAM** `laserbeamFoam` (melt-pool flow with a laser source, open).
