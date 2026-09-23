"""Visualisation for Aeromant: the problem statement (domain, refinement boxes, body, inflow) and the
OpenFOAM results (mesh sections, velocity/pressure fields, streamlines) via pyvista's OpenFOAM reader.

3D plots are interactive in Jupyter (pyvista, optional extra ``vegeta-cli[viz]``); 2D sections are
matplotlib. Nothing here runs OpenFOAM or changes the case.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _pv():
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover
        raise ImportError("3D visualisation needs pyvista: pip install 'vegeta-cli[viz]'") from exc
    return pv


def _case_dir(case) -> Path:
    return Path(case.workdir) if hasattr(case, "workdir") else Path(case)


def case_info(case) -> dict:
    p = _case_dir(case) / "aeromant_case.json"
    if not p.is_file():
        raise FileNotFoundError(f"{p.parent} is not a prepared Aeromant case")
    return json.loads(p.read_text())


def _vec(text: str) -> np.ndarray:
    return np.array([float(v) for v in text.split()])


def plot_setup(case, plotter=None):
    """Problem statement: domain box, near/wake refinement boxes, the body, inflow direction and values."""
    pv = _pv()
    info = case_info(case)
    d, p = info["derived"], info["config"]["parameters"]
    cdir = _case_dir(case)
    stl = next((cdir / sub / "body.stl" for sub in ("constant/triSurface", "constant/geometry")
                if (cdir / sub / "body.stl").is_file()), None)
    pl = plotter or pv.Plotter()
    lo = np.array([float(d["XMIN"]), float(d["YMIN"]), float(d["ZMIN"])])
    hi = np.array([float(d["XMAX"]), float(d["YMAX"]), float(d["ZMAX"])])
    pl.add_mesh(pv.Box(bounds=(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])), style="wireframe", color="#555555",
                line_width=1.5, label="domain")
    for name, colour in (("NEAR", "#2ca02c"), ("WAKE", "#ff7f0e")):
        a, b = _vec(d[f"{name}_MIN"]), _vec(d[f"{name}_MAX"])
        pl.add_mesh(pv.Box(bounds=(a[0], b[0], a[1], b[1], a[2], b[2])), style="wireframe", color=colour,
                    line_width=1.0, label=f"{name.lower()} refinement (level {d[name + '_LEVEL']})")
    if stl is not None:
        pl.add_mesh(pv.read(str(stl)), color="#9fb8d0", smooth_shading=True, label="body")
    span = float((hi - lo).max())
    u = float(p["velocity"])
    for y in np.linspace(lo[1] + 0.2 * (hi[1] - lo[1]), hi[1] - 0.2 * (hi[1] - lo[1]), 3):
        for z in np.linspace(lo[2] + 0.2 * (hi[2] - lo[2]), hi[2] - 0.2 * (hi[2] - lo[2]), 3):
            pl.add_mesh(pv.Arrow(start=(lo[0], y, z), direction=(1, 0, 0), scale=0.08 * span), color="#1f77b4")
    pl.add_point_labels([[lo[0], hi[1], hi[2]]], [f"inlet U = {u:g} m/s"], font_size=12, shape=None,
                        text_color="#1f77b4")
    pl.add_text(
        f"{info['config']['template']} ({info['config'].get('openfoam_flavor', '')})\n"
        f"Re = {u * p['reference_length'] / p['kinematic_viscosity']:.3g}, nu = {p['kinematic_viscosity']:g} m2/s, "
        f"rho = {p['density']:g} kg/m3\nA_ref = {p['reference_area']:.4g} m2, L_ref = {p['reference_length']:g} m, "
        f"background cells {d['NX']}x{d['NY']}x{d['NZ']}", font_size=10)
    pl.add_legend()
    pl.add_axes()
    return pl


def read_results(case, time: str | float | None = None):
    """The OpenFOAM case as a pyvista MultiBlock (``internalMesh`` + ``boundary``) at the last time."""
    pv = _pv()
    cdir = _case_dir(case)
    foam = cdir / "case.foam"
    foam.touch(exist_ok=True)
    reader = pv.OpenFOAMReader(str(foam))
    times = list(reader.time_values)
    if not times:
        raise FileNotFoundError(f"no time directories in {cdir}")
    reader.set_active_time_value(float(time) if time is not None else times[-1])
    reader.cell_to_point_creation = True
    return reader.read()


def plot_mesh_slice(case, normal: str = "y", origin=None, plotter=None):
    """Section through the snappyHexMesh mesh: shows the refinement levels around the body."""
    pv = _pv()
    mb = read_results(case)
    internal = mb["internalMesh"]
    origin = origin or internal.center
    sl = internal.slice(normal=normal, origin=origin)
    pl = plotter or pv.Plotter()
    pl.add_mesh(sl, style="wireframe", color="#3d4f63", line_width=0.6)
    _add_body(pl, case, pv)
    pl.add_text(f"mesh section {normal} = {origin['xyz'.index(normal)]:.3g}: {internal.n_cells} cells", font_size=10)
    _flat_view(pl, normal)
    return pl


def plot_field_slice(case, field: str = "U", normal: str = "y", origin=None, plotter=None, clim=None):
    """Velocity magnitude (``U``) or pressure (``p``) on a plane, interactive (pyvista)."""
    pv = _pv()
    mb = read_results(case)
    internal = mb["internalMesh"]
    origin = origin or internal.center
    sl = internal.slice(normal=normal, origin=origin)
    scalars = field
    if field == "U":
        sl.point_data["|U|"] = np.linalg.norm(sl.point_data["U"], axis=1)
        scalars = "|U|"
    pl = plotter or pv.Plotter()
    pl.add_mesh(sl, scalars=scalars, cmap="turbo", clim=clim, scalar_bar_args={"title": scalars + (" [m/s]" if field == "U" else " [m2/s2]")})
    _add_body(pl, case, pv)
    pl.add_text(f"{scalars}, section {normal} = {origin['xyz'.index(normal)]:.3g}", font_size=10)
    _flat_view(pl, normal)
    return pl


def plot_section(case, field: str = "U", normal: str = "y", origin=None, ax=None, levels: int = 30, zoom: float = 1.0):
    """2D matplotlib contour of |U| or p on a plane; ``zoom`` > 1 crops around the body."""
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    mb = read_results(case)
    internal = mb["internalMesh"]
    origin = origin or internal.center
    axis = "xyz".index(normal)
    sl = internal.slice(normal=normal, origin=origin).triangulate()
    keep = [i for i in range(3) if i != axis]
    pts = sl.points
    vals = np.linalg.norm(sl.point_data["U"], axis=1) if field == "U" else sl.point_data[field]
    tri = mtri.Triangulation(pts[:, keep[0]], pts[:, keep[1]], sl.faces.reshape(-1, 4)[:, 1:])
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.5))
    cs = ax.tricontourf(tri, vals, levels=levels, cmap="turbo")
    ax.figure.colorbar(cs, ax=ax, label="|U| [m/s]" if field == "U" else f"{field}")
    info = case_info(case)
    b = info["body_bbox_m"]
    if zoom > 1:
        c = [(b[0][k] + b[1][k]) / 2 for k in range(3)]
        half = max(b[1][k] - b[0][k] for k in range(3)) * zoom
        ax.set_xlim(c[keep[0]] - half, c[keep[0]] + 2 * half)
        ax.set_ylim(c[keep[1]] - half, c[keep[1]] + half)
    ax.set_aspect("equal")
    ax.set_xlabel("xyz"[keep[0]] + " [m]")
    ax.set_ylabel("xyz"[keep[1]] + " [m]")
    ax.set_title(f"{'|U|' if field == 'U' else field}, section {normal} = {origin[axis]:.3g} m")
    return ax.figure


def plot_streamlines(case, n: int = 60, plotter=None, normal_plane: str = "y"):
    """Streamlines seeded upstream of the body, coloured by speed, with the body."""
    pv = _pv()
    mb = read_results(case)
    internal = mb["internalMesh"]
    info = case_info(case)
    b = np.array(info["body_bbox_m"])
    size = (b[1] - b[0]).max()
    centre = (b[0] + b[1]) / 2
    seeds = pv.Plane(center=(b[0][0] - size, centre[1], centre[2]), direction=(1, 0, 0),
                     i_size=1.3 * size, j_size=1.3 * size, i_resolution=int(np.sqrt(n)), j_resolution=int(np.sqrt(n)))
    stream = internal.streamlines_from_source(seeds, vectors="U", max_length=50.0 * size, integration_direction="forward")
    stream.point_data["|U|"] = np.linalg.norm(stream.point_data["U"], axis=1)
    pl = plotter or pv.Plotter()
    pl.add_mesh(stream.tube(radius=0.01 * size), scalars="|U|", cmap="turbo", scalar_bar_args={"title": "|U| [m/s]"})
    _add_body(pl, case, pv, opacity=1.0)
    pl.add_text("streamlines", font_size=10)
    pl.add_axes()
    return pl


def _write_video(frames, path, fps: int) -> Path:
    try:
        import cv2
    except ImportError as e:  # pragma: no cover
        raise ImportError("video export needs OpenCV: pip install opencv-python-headless") from e
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        out.write(cv2.cvtColor(np.ascontiguousarray(f[:, :, :3]), cv2.COLOR_RGB2BGR))
    out.release()
    return path


def animate_particles(case, path, *, n: int = 400, seconds: float = 6.0, fps: int = 24, rpm: float | None = None,
                      axis: str = "x", lengths: float = 6.0, size=(960, 720), seed: int = 0, camera=None,
                      point_size: float = 7.0) -> Path:
    """MP4 (OpenCV) of tracer particles carried by the converged velocity field: seeded upstream of the
    body, advected through the steady field (Heun steps on the sampled ``U``), coloured by speed, and
    re-seeded once they leave the domain. ``lengths`` body sizes of travel fill the video. With ``rpm``
    the body is turned about ``axis`` at that speed for the eye — in an MRF case the field itself is
    steady in the rotating zone. Returns the file path."""
    pv = _pv()
    mb = read_results(case)
    internal = mb["internalMesh"]
    info = case_info(case)
    b = np.array(info["body_bbox_m"])
    extent = b[1] - b[0]
    size_m = float(extent.max())
    centre = (b[0] + b[1]) / 2
    rng = np.random.default_rng(seed)
    cross = np.array([extent[1], extent[2]]).max() * 0.8

    def seeds(k, spread: float = 0.0):
        r = cross * np.sqrt(rng.uniform(0, 1, k)); a = rng.uniform(0, 2 * np.pi, k)
        x = b[0][0] - rng.uniform(0.2 - spread, 1.2, k) * size_m
        return np.column_stack([x, centre[1] + r * np.cos(a), centre[2] + r * np.sin(a)])

    def velocity(pts):
        s = pv.PolyData(pts).sample(internal, tolerance=1e-6)
        u = np.asarray(s.point_data["U"], dtype=float)
        ok = np.asarray(s.point_data["vtkValidPointMask"]).astype(bool) if "vtkValidPointMask" in s.point_data else np.ones(len(pts), bool)
        return u, ok

    pts = seeds(n, spread=lengths * 0.5)            # a stream that is already flowing at the first frame
    u0, _ = velocity(pts)
    u_ref = float(np.linalg.norm(u0, axis=1).mean()) or 1.0
    n_frames = max(2, int(round(seconds * fps)))
    dt = lengths * size_m / u_ref / n_frames
    stl = None
    cdir = _case_dir(case)
    for sub in ("constant/triSurface", "constant/geometry"):
        if (cdir / sub / "body.stl").is_file():
            stl = pv.read(str(cdir / sub / "body.stl"))
    rot = {"x": "rotate_x", "y": "rotate_y", "z": "rotate_z"}[axis]
    pl = pv.Plotter(off_screen=True, window_size=list(size))
    frames = []
    speed_lim = [0.0, float(np.percentile(np.linalg.norm(internal.point_data["U"], axis=1), 99))]
    for k in range(n_frames):
        u1, ok = velocity(pts)
        mid = pts + 0.5 * dt * u1
        u2, ok2 = velocity(mid)
        pts = pts + dt * np.where(ok2[:, None], u2, u1)
        lost = ~(ok & ok2) | (pts[:, 0] > b[1][0] + 2.5 * size_m) | (np.abs(pts[:, 1] - centre[1]) > 4 * size_m) | (np.abs(pts[:, 2] - centre[2]) > 4 * size_m)
        if lost.any():
            pts[lost] = seeds(int(lost.sum()))
        cloud = pv.PolyData(pts)
        cloud.point_data["|U|"] = np.linalg.norm(u1, axis=1)
        pl.clear()
        pl.add_mesh(cloud, scalars="|U|", cmap="turbo", clim=speed_lim, point_size=point_size, render_points_as_spheres=True,
                    scalar_bar_args={"title": "|U| [m/s]"})
        if stl is not None:
            body = getattr(stl, rot)((rpm / 60.0 * 360.0 * k / fps) % 360.0, point=tuple(centre), inplace=False) if rpm else stl
            pl.add_mesh(body, color="#dddddd")
        pl.add_text("tracer particles in the converged flow" + (f", body at {rpm:.0f} rpm" if rpm else ""), font_size=10)
        if k == 0:
            if camera is not None:
                pl.camera_position = camera
            else:
                focus = centre + np.array([0.7 * size_m, 0.0, 0.0])          # flow left to right, seen from ahead-left and above
                pl.camera_position = [tuple(focus + size_m * np.array([-2.2, -3.0, 1.6])), tuple(focus), (0.0, 0.0, 1.0)]
                pl.reset_camera(bounds=[b[0][0] - 1.2 * size_m, b[1][0] + 2.5 * size_m, centre[1] - cross, centre[1] + cross, centre[2] - cross, centre[2] + cross])
                pl.camera.zoom(1.35)
            cam = pl.camera_position
        else:
            pl.camera_position = cam
        frames.append(pl.screenshot(return_img=True))
    pl.close()
    return _write_video(frames, path, fps)


def plot_surface_pressure(case, plotter=None):
    """Pressure (kinematic) on the body surface."""
    pv = _pv()
    mb = read_results(case)
    boundary = mb["boundary"]
    names = [boundary.get_block_name(i) for i in range(boundary.n_blocks)]
    pl = plotter or pv.Plotter()
    for i, name in enumerate(names):
        if name and name.startswith("body"):
            pl.add_mesh(boundary[i], scalars="p", cmap="coolwarm", scalar_bar_args={"title": "p [m2/s2]"})
    pl.add_text("surface pressure on the body", font_size=10)
    pl.add_axes()
    return pl


def _add_body(pl, case, pv, opacity=1.0):
    cdir = _case_dir(case)
    for sub in ("constant/triSurface", "constant/geometry"):
        stl = cdir / sub / "body.stl"
        if stl.is_file():
            pl.add_mesh(pv.read(str(stl)), color="#dddddd", opacity=opacity)
            return


def _flat_view(pl, normal):
    views = {"x": "yz", "y": "xz", "z": "xy"}
    pl.camera_position = views[normal]
    pl.enable_parallel_projection()
    pl.add_axes()


def show(plotter, **kwargs):
    """Show a plotter: interactive in a live Jupyter kernel, static image otherwise."""
    import os

    backend = os.environ.get("PYVISTA_JUPYTER_BACKEND")
    if backend:
        return plotter.show(jupyter_backend=backend, **kwargs)
    try:
        return plotter.show(**kwargs)
    except Exception:
        return plotter.show(jupyter_backend="static", **kwargs)
