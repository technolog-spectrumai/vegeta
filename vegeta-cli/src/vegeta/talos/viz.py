"""Visualisation for Talos: Gmsh meshes, the problem statement (regions, supports, loads) and results.

Interactive 3D uses pyvista (optional extra ``vegeta-cli[viz]``); in Jupyter the plots are interactive
(trame backend) and fall back to static images when no interactive backend is available. 2D sections
are matplotlib figures. Nothing here changes any data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .frd import FieldResults, read_frd
from .mesh import MeshData, read_mesh

VTK_TETRA, VTK_QUADRATIC_TETRA = 10, 24  # VTK quadratic tetra node order equals CalculiX C3D10 order


def _pv():
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover
        raise ImportError("3D visualisation needs pyvista: pip install 'vegeta-cli[viz]'") from exc
    return pv


def _mesh_of(obj) -> MeshData:
    if isinstance(obj, MeshData):
        return obj
    if hasattr(obj, "artifacts"):
        return read_mesh(obj.artifacts["mesh"])
    return read_mesh(Path(obj))


def mesh_to_pyvista(mesh) -> "pv.UnstructuredGrid":
    """Gmsh mesh (``mesh.msh`` path, mesh Result or MeshData) as a pyvista UnstructuredGrid."""
    pv = _pv()
    mesh = _mesh_of(mesh)
    pos = {int(n): i for i, n in enumerate(mesh.node_ids)}
    conn = np.vectorize(pos.get)(mesh.connectivity)
    n = conn.shape[1]
    cells = np.hstack([np.full((len(conn), 1), n), conn]).ravel()
    types = np.full(len(conn), VTK_QUADRATIC_TETRA if n == 10 else VTK_TETRA, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, types, mesh.coords.astype(float))
    grid.cell_data["element_id"] = mesh.element_ids
    for name, nodes in mesh.region_nodes.items():
        flag = np.zeros(len(mesh.node_ids), dtype=np.uint8)
        flag[[pos[int(k)] for k in nodes if int(k) in pos]] = 1
        grid.point_data[f"region:{name}"] = flag
    return grid


def results_to_pyvista(result, mesh=None) -> "pv.UnstructuredGrid":
    """Solved case (talos Result or .frd path) as a grid with U, |U|, stress components and von Mises."""
    fr = result if isinstance(result, FieldResults) else read_frd(result.artifacts["frd"] if hasattr(result, "artifacts") else result)
    if mesh is None and hasattr(result, "artifacts"):
        mesh = result.artifacts["mesh"]
    grid = mesh_to_pyvista(mesh)
    pos = {int(n): i for i, n in enumerate(fr.node_ids)}
    order = np.array([pos[int(n)] for n in _mesh_of(mesh).node_ids])
    grid.point_data["U"] = np.nan_to_num(fr.displacement[order])
    grid.point_data["|U|"] = np.nan_to_num(fr.displacement_magnitude[order])
    if "STRESS" in fr.fields:
        s = np.nan_to_num(fr.stress[order])
        grid.point_data["von_mises"] = np.nan_to_num(fr.von_mises[order])
        for i, name in enumerate(("Sxx", "Syy", "Szz", "Sxy", "Syz", "Szx")):
            grid.point_data[name] = s[:, i]
    return grid


def export_vtu(result, path: str | Path) -> Path:
    """Write results as a ParaView-readable ``.vtu`` (mesh, displacement, stresses, von Mises)."""
    path = Path(path)
    results_to_pyvista(result).save(str(path))
    return path


def plot_mesh(mesh, *, show_edges: bool = True, clip: str | None = None, color="#b8c9dc", plotter=None):
    """3D view of the mesh; ``clip="x"|"y"|"z"`` cuts it in half to show the interior."""
    pv = _pv()
    grid = mesh_to_pyvista(mesh)
    pl = plotter or pv.Plotter()
    body = grid.clip(normal=clip) if clip else grid
    pl.add_mesh(body, show_edges=show_edges, color=color, edge_color="#3d4f63", line_width=0.5)
    pl.add_text(f"{grid.n_cells} elements, {grid.n_points} nodes", font_size=10)
    pl.add_axes()
    return pl


def plot_problem(model, mesh, *, arrow_scale: float | None = None, plotter=None):
    """The problem statement: regions coloured on the mesh, supports and loads drawn as glyphs."""
    from .loads import Acceleration, Displacement, FixedSupport, Force, Pressure

    pv = _pv()
    grid = mesh_to_pyvista(mesh)
    md = _mesh_of(mesh)
    pl = plotter or pv.Plotter()
    pl.add_mesh(grid, color="#e8ecf0", opacity=0.35, show_edges=False)
    span = float(np.ptp(md.coords, axis=0).max()) or 1.0
    scale = arrow_scale or 0.15 * span
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
    supports = {s.region: s for s in model.supports}
    loads = {l.region: l for l in model.loads if hasattr(l, "region")}
    for i, region in enumerate(model.regions):
        pts = grid.points[grid.point_data[f"region:{region.name}"] == 1]
        if not len(pts):
            continue
        colour = palette[i % len(palette)]
        cloud = pv.PolyData(pts)
        pl.add_mesh(cloud, color=colour, point_size=6, render_points_as_spheres=True, label=region.name)
        centre = pts.mean(axis=0)
        text = region.name
        if region.name in supports:
            s = supports[region.name]
            text += " — fixed" if isinstance(s, FixedSupport) else f" — u=({s.ux},{s.uy},{s.uz})"
            pl.add_mesh(pv.Cube(center=centre, x_length=scale * 0.2, y_length=scale * 0.2, z_length=scale * 0.2),
                        color=colour, opacity=0.6)
        if region.name in loads:
            l = loads[region.name]
            if isinstance(l, Force):
                d = np.array(l.vector, dtype=float)
                d = d / (np.linalg.norm(d) or 1.0)
                pl.add_mesh(pv.Arrow(start=centre - d * scale, direction=d, scale=scale), color=colour)
                text += f" — F=({l.fx:g},{l.fy:g},{l.fz:g})"
            elif isinstance(l, Pressure):
                text += f" — p={l.value:g}"
        pl.add_point_labels([centre], [text], font_size=11, text_color=colour, shape=None, always_visible=True)
    for l in model.loads:
        if isinstance(l, Acceleration):
            d = np.array(l.vector, dtype=float)
            d = d / (np.linalg.norm(d) or 1.0)
            c = md.coords.mean(axis=0)
            pl.add_mesh(pv.Arrow(start=c, direction=d, scale=scale), color="#555555")
            pl.add_point_labels([c + d * scale], [f"a=({l.ax:g},{l.ay:g},{l.az:g})"], font_size=11, shape=None)
    pl.add_text(f"{model.name}: {model.material.name}, units {model.units}", font_size=10)
    pl.add_axes()
    return pl


def plot_results(result, *, field: str = "von_mises", deform_scale: float | None = None, clip: str | None = None,
                 show_edges: bool = False, plotter=None, mesh=None, clim=None):
    """Deformed shape coloured by ``field`` (von_mises, |U|, Sxx ...). ``deform_scale=None`` picks a
    magnification so the largest displacement is ~10 % of the model size (shown in the title)."""
    pv = _pv()
    grid = results_to_pyvista(result, mesh)
    span = float(np.ptp(grid.points, axis=0).max()) or 1.0
    umax = float(grid.point_data["|U|"].max()) or 1.0
    if deform_scale is None:
        deform_scale = 0.1 * span / umax
    warped = grid.warp_by_vector("U", factor=deform_scale)
    if clip:
        warped = warped.clip(normal=clip)
    pl = plotter or pv.Plotter()
    pl.add_mesh(grid.extract_surface(), style="wireframe", color="#9aa5b1", opacity=0.25, line_width=0.5)
    pl.add_mesh(warped, scalars=field, cmap="turbo", show_edges=show_edges, clim=clim,
                scalar_bar_args={"title": field})
    pl.add_text(f"{field} on deformed shape (x{deform_scale:.3g})", font_size=10)
    pl.add_axes()
    return pl


def plot_section(result, *, normal: str = "y", origin=None, field: str = "von_mises", ax=None, levels: int = 24,
                 mesh=None):
    """2D contour section through the results on a plane (matplotlib)."""
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    grid = results_to_pyvista(result, mesh)
    axis = "xyz".index(normal)
    if origin is None:
        origin = grid.center
    sl = grid.slice(normal=normal, origin=origin).triangulate()
    pts = sl.points
    keep = [i for i in range(3) if i != axis]
    x, y = pts[:, keep[0]], pts[:, keep[1]]
    tris = sl.faces.reshape(-1, 4)[:, 1:]
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    tri = mtri.Triangulation(x, y, tris)
    cs = ax.tricontourf(tri, sl.point_data[field], levels=levels, cmap="turbo")
    ax.triplot(tri, color="k", lw=0.15, alpha=0.4)
    ax.set_aspect("equal")
    ax.set_xlabel("xyz"[keep[0]])
    ax.set_ylabel("xyz"[keep[1]])
    ax.set_title(f"{field}, section {normal} = {origin[axis]:.3g}")
    ax.figure.colorbar(cs, ax=ax, label=field)
    return ax.figure


def show(plotter, **kwargs):
    """Show a plotter: interactive in a live Jupyter kernel, static image otherwise."""
    return _show(plotter, **kwargs)


def _show(plotter, **kwargs):
    import os

    backend = os.environ.get("PYVISTA_JUPYTER_BACKEND")
    if backend:
        return plotter.show(jupyter_backend=backend, **kwargs)
    try:
        return plotter.show(**kwargs)
    except Exception:  # no interactive backend (e.g. headless execution)
        return plotter.show(jupyter_backend="static", **kwargs)
