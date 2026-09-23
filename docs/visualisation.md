# Visualisation

Every tool has a `viz` module. 3D views use **pyvista** (`pip install "vegeta-cli[viz]"`, done by
`install_local.sh`): interactive in a live Jupyter kernel (trame backend — rotate, zoom, pick), and a
static PNG when no interactive backend exists (headless execution, `PYVISTA_JUPYTER_BACKEND=static`).
2D sections are matplotlib figures. Nothing in `viz` runs a solver or changes data.

Show a 3D plot with `viz.show(plotter)` (or `plotter.show()`); save a PNG with `plotter.screenshot("x.png")`.

| tool | function | shows |
|---|---|---|
| `vegeta.dedalus.viz` | `plot3d(*geometries)` | interactive geometry, several parts with a legend |
| | `plot_sections(geometry, normal="z", n=4)` | 2D section curves through the part |
| | `section_polylines(geometry, normal, position)` | the curves as arrays (for your own plots) |
| `vegeta.talos.viz` | `plot_mesh(mesh, clip="y")` | the Gmsh mesh, optionally cut open |
| | `plot_problem(model, mesh)` | **problem statement**: regions coloured, supports, force/pressure/acceleration glyphs, material and units |
| | `plot_results(result, field="von_mises", clip=None)` | deformed shape (auto-scaled) coloured by von Mises, `\|U\|`, `Sxx` … over the undeformed wireframe |
| | `plot_section(result, normal="y", field=...)` | 2D contour section through the solid |
| | `export_vtu(result, "beam.vtu")` | ParaView file with mesh, displacement, stresses |
| | `animate(result, "blade.mp4", rpm=8000, axis="z")` | MP4 (OpenCV): the load ramps 0 → 100 % (stress and deformation with it) while the part turns at `rpm`; without `rpm` the camera orbits |
| `vegeta.aeromant.viz` | `plot_setup(case)` | **problem statement**: domain, refinement boxes, body, inflow arrows, Re and reference values |
| | `plot_mesh_slice(case, normal="y")` | snappyHexMesh section showing refinement levels |
| | `plot_field_slice(case, "U" or "p", normal="y")` | interactive field on a plane |
| | `plot_section(case, "U", zoom=2)` | 2D matplotlib contour around the body |
| | `plot_streamlines(case)` | streamlines seeded upstream, coloured by speed |
| | `plot_surface_pressure(case)` | pressure on the body |
| | `animate_particles(case, "flow.mp4", rpm=8000)` | MP4 (OpenCV): tracer particles advected through the converged field, coloured by speed, the body turned at `rpm` for the eye |
| `vegeta.mellonia.viz` | `plot_toolpath(result)` | 3D toolpath coloured by layer |
| | `plot_layer(result, layer)` / `plot_layer_grid(result, n=6)` | layers from above: perimeters, infill, travel moves |
| | `toolpath_to_pyvista(result)` | the moves as lines (for your own views) |

Aeromant reads the native case through pyvista's OpenFOAM reader (`case.foam` is created on demand);
Talos builds the grid from `mesh.msh` + `model.frd` (C3D10 elements map directly onto VTK quadratic
tetrahedra); Mellonia parses the G-code moves (`read_gcode(path, moves=True)`).

The two `animate` functions present **steady** results as motion (a linear static solution scaled with a load
ramp; particles carried by a converged velocity field): they are not transient analyses. They need OpenCV
(`opencv-python-headless`, in the `viz` extra) and write H.264-free `mp4v` files that any player opens;
in a notebook show them with `IPython.display.Video(path, embed=False)`.

Talos also draws mode shapes (`viz.plot_mode(modes_result, mode=1)`) and fatigue damage maps
(`viz.plot_damage(fatigue_result, mesh)`); Chronos draws mission profiles, spectra, Campbell diagrams and
cumulative-damage curves with matplotlib.

See the notebooks: `02_talos_fea`, `03_aeromant_cfd`, `04_mellonia_print`, `05_workflow`, and the two
product-level ones, `08_quadcopter` and `09_fixed_wing_drone`.
