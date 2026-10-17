# FEniCSx in Vegeta — where a field solver pays, and how to install it

FEniCSx (dolfinx) is a finite-element library: you write the weak form in UFL, it assembles and solves on an
unstructured mesh, in parallel, and writes XDMF/VTK that PyVista reads. It is the right tool where Hyperion's
closed forms and structured-grid solvers stop: real geometry, coupled fields, moving fronts. It is also a heavy
install (PETSc, MPI), so it lives on `dev_svarog_advanced` and nothing in the main branches imports it.

## Install

- **conda-forge** (the supported way): `micromamba create -n fenicsx -c conda-forge fenics-dolfinx mpich pyvista
  h5py meshio gmsh python-gmsh` then `micromamba activate fenicsx` and `pip install -e vegeta-cli --no-deps`.
- **Docker**: `docker run -ti -v $PWD:/root/vegeta dolfinx/dolfinx:stable` (PyVista headless needs `xvfb` or
  `pyvista.OFF_SCREEN = True`).
- A cloud session of this repository cannot build it (no conda, no MPI); the advanced notebook therefore reads results
  from `advanced/runs/` when they exist and says NOT RUN otherwise — the repository's convention for a solver that is
  not on the machine.

## The backend in Hyperion (`hyperion.thermal_fenicsx`)

`thermal.steady_plate(..., backend="auto")` and `thermal.heat_plate(..., backend="auto")` take FEniCSx when dolfinx is
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

## The first use: the guard's ablation (`advanced/svarog_ablation_fenicsx.py`, notebook 50)

The barrier under the stray beam as a transient heat problem in r–z with the apparent-heat-capacity enthalpy method
(the latent heats of melting and of decomposition spread over ±15 K), implicit Euler with Picard iterations on the
capacity and the linearised radiation, and ablation by cell deactivation once a cell held at the failure point has
taken its latent heat of removal — the same physics as `hyperion.thermal.heat_plate`, on a finer mesh and
unconditionally stable in time. It validates the cheap solver 49's verdicts rest on (the burn-through time of the
10 mm polycarbonate, the steady state of 1.5 mm steel) and extends it where the structured grid cannot go: a 3-D
panel with a rib, a beam at an angle, a panel with a liner.

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
