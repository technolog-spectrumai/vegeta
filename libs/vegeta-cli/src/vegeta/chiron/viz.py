"""Off-screen rendering of an episode with pyvista, and movies with OpenCV.

MuJoCo's own renderer needs EGL or OSMesa; Chiron draws episodes with pyvista instead, from the log alone (no
model needed): run with ``ChironLab(..., log_geoms=True)`` so the log carries every geom's shape and pose under
``'geom_pose'`` (plus a coarse terrain grid). Without it, logged bodies and feet are drawn as markers.
On a headless machine run under ``xvfb-run -a`` with ``PYVISTA_OFF_SCREEN=true``.

    imgs = viz.frames(episode, camera="follow", every=4)      # viz = vegeta.chiron.viz
    viz.to_video(imgs, "walk.mp4", fps=25)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["frames", "to_video", "render"]

CAMERAS = ("follow", "side", "top", "front", "iso")


def _pv():
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover
        raise ImportError("vegeta.chiron.viz needs pyvista (pip install 'vegeta-cli[viz]')") from exc
    pv.OFF_SCREEN = True
    return pv


def _geom_mesh(pv, kind, size):
    if kind == "box":
        return pv.Box(bounds=(-size[0], size[0], -size[1], size[1], -size[2], size[2]))
    if kind == "sphere":
        return pv.Sphere(radius=size[0], theta_resolution=18, phi_resolution=12)
    if kind == "capsule":
        r, h = size[0], size[1]
        body = pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1), radius=r, height=2 * h, resolution=18, capping=False)
        a = pv.Sphere(radius=r, center=(0, 0, h), theta_resolution=18, phi_resolution=10)
        b = pv.Sphere(radius=r, center=(0, 0, -h), theta_resolution=18, phi_resolution=10)
        return body.merge(a).merge(b)
    if kind == "cylinder":
        return pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1), radius=size[0], height=2 * size[1], resolution=24)
    if kind == "ellipsoid":
        return pv.ParametricEllipsoid(size[0], size[1], size[2])
    return None


def _terrain_mesh(pv, log, centre, half):
    gp = log.get("geom_pose") or {}
    if "terrain_z" in gp and not bool(np.asarray(gp.get("terrain_plane", False))):
        Z = np.asarray(gp["terrain_z"], dtype=float)
        x0, x1, y0, y1 = np.asarray(gp["terrain_extent"], dtype=float)
        xs, ys = np.linspace(x0, x1, Z.shape[1]), np.linspace(y0, y1, Z.shape[0])
        X, Y = np.meshgrid(xs, ys)
        return pv.StructuredGrid(X, Y, Z)
    spec = log.get("terrain") or {"kind": "flat"}
    try:
        from .terrain import terrain_from_spec

        terrain = terrain_from_spec(spec)
    except Exception:
        terrain = None
    x0, x1 = centre[0] - half[0], centre[0] + half[0]
    y0, y1 = centre[1] - half[1], centre[1] + half[1]
    if terrain is None or terrain.is_flat:
        return pv.Plane(center=(centre[0], centre[1], 0.0), direction=(0, 0, 1), i_size=x1 - x0, j_size=y1 - y0,
                        i_resolution=1, j_resolution=1)
    Z, (x0, x1, y0, y1) = terrain.heightfield((x0, x1), (y0, y1), 0.02)
    X, Y = np.meshgrid(np.linspace(x0, x1, Z.shape[1]), np.linspace(y0, y1, Z.shape[0]))
    return pv.StructuredGrid(X, Y, Z)


def _camera(camera, com, scale):
    if callable(camera):
        return camera(com)
    if isinstance(camera, dict):
        return camera["position"], camera["focal_point"], camera.get("view_up", (0, 0, 1))
    c = np.asarray(com, dtype=float)
    d = 3.0 * scale
    offsets = {"follow": (-0.9, 1.1, 0.7), "iso": (-0.9, 1.1, 0.7), "side": (0.0, 1.5, 0.25),
               "front": (1.6, 0.0, 0.35), "top": (0.0, 0.0, 1.8)}
    if camera not in offsets:
        raise ValueError(f"camera must be one of {CAMERAS}, a dict or a callable")
    off = np.asarray(offsets[camera]) * d
    up = (1, 0, 0) if camera == "top" else (0, 0, 1)
    return tuple(c + off), tuple(c), up


def frames(episode, camera="follow", every=1, size=(960, 540), *, start=0, stop=None, show_terrain=True,
           show_forces=False, show_time=True, background="white", robot_color="#c8a24a", ground=None,
           scenery_range=None, ground_color="#b9b2a5"):
    """Render the episode's log samples ``start:stop:every`` to RGB arrays (H, W, 3) uint8.

    ``camera``: 'follow' (three-quarter view from the rear left, following the COM), 'side', 'front', 'top',
    'iso' (fixed at the first sample), a dict ``position/focal_point/view_up`` or a callable ``com -> (position,
    focal_point, view_up)``. ``show_forces`` draws each foot's contact force as an arrow (one robot size per
    robot weight). ``ground`` (x0, x1, y0, y1) [m]: the flat ground drawn over that extent (default: a few robot
    sizes around the run); ``scenery_range`` [m]: geoms farther than this from the COM are hidden (default 25 robot
    sizes); ``ground_color`` the terrain's colour.
    """
    pv = _pv()
    log = episode.log if hasattr(episode, "log") else episode
    t = np.asarray(log["t"])
    idx = np.arange(len(t))[start:stop:max(1, int(every))]
    com = np.asarray(log["com"])
    gp = log.get("geom_pose")
    pl = pv.Plotter(off_screen=True, window_size=[int(size[0]), int(size[1])])
    pl.set_background(background)
    actors = []
    if gp is not None:
        pos, mat = np.asarray(gp["pos"], dtype=float), np.asarray(gp["mat"], dtype=float)
        sizes, rgba = np.asarray(gp["size"], dtype=float), np.asarray(gp["rgba"], dtype=float)
        p0 = pos[0]
        if "robot" in gp and np.any(gp["robot"]):           # frame the robot, not the scenery around it
            p0 = p0[np.asarray(gp["robot"], dtype=bool)]
        scale = float(max(np.ptp(p0[:, 0]), np.ptp(p0[:, 1]), np.ptp(p0[:, 2]), 0.1)) if len(p0) else 0.5
        for g, kind in enumerate(gp["type"]):
            mesh = _geom_mesh(pv, kind, sizes[g])
            if mesh is None:
                actors.append(None)
                continue
            col = rgba[g, :3] if not np.allclose(rgba[g], [0.5, 0.5, 0.5, 1.0]) else None
            foot = bool(np.asarray(gp.get("foot", np.zeros(len(gp["type"]), bool)))[g])
            colour = "#333333" if foot else (col if col is not None else robot_color)
            actors.append(pl.add_mesh(mesh, color=colour, smooth_shading=True, opacity=float(rgba[g, 3]) or 1.0))
    else:
        pos = mat = None
        bp = np.asarray(log["body_pos"])
        scale = float(max(np.ptp(bp[0, :, 0]) if bp.shape[1] else 0.0, 0.3))
    half = (max(1.5, 4 * scale), max(1.0, 2 * scale))
    if show_terrain:
        centre = com[idx].mean(axis=0) if len(idx) else np.zeros(3)
        if ground is not None:
            x0, x1, y0, y1 = (float(v) for v in ground)
            centre, half = np.array([(x0 + x1) / 2, (y0 + y1) / 2, 0.0]), ((x1 - x0) / 2, (y1 - y0) / 2)
        elif gp is None or "terrain_z" not in gp:
            span = np.ptp(com[:, 0]) if len(com) else 0.0
            half = (max(half[0], span / 2 + 2 * scale), half[1])
        pl.add_mesh(_terrain_mesh(pv, log, centre, half), color=ground_color, smooth_shading=True)
    images = []
    marker_names = []
    first_cam = _camera(camera, com[idx[0]] if len(idx) else np.zeros(3), scale) if camera in ("iso",) else None
    weight = float(log.get("total_mass", 1.0)) * 9.81
    for i in idx:
        if gp is not None:
            far = 25.0 * scale if scenery_range is None else float(scenery_range)   # scenery parked far away
            for g, actor in enumerate(actors):               # is not drawn: it would stretch the camera's clipping range
                if actor is None:
                    continue
                M = np.eye(4)
                M[:3, :3] = mat[i, g].reshape(3, 3)
                M[:3, 3] = pos[i, g]
                actor.user_matrix = M
                actor.SetVisibility(bool(np.linalg.norm(pos[i, g] - com[i]) < far))
        else:
            for name in marker_names:
                pl.remove_actor(name)
            marker_names = []
            bp = np.asarray(log["body_pos"])[i]
            for b, p in enumerate(bp):
                n = f"body{b}"
                pl.add_mesh(pv.Sphere(radius=0.03 * max(scale / 0.3, 0.5), center=p), color=robot_color, name=n)
                marker_names.append(n)
            if "foot_pos" in log:
                for f, p in enumerate(np.asarray(log["foot_pos"])[i]):
                    n = f"foot{f}"
                    b = int(np.asarray(log["foot_body"])[f]) if "foot_body" in log else 0
                    line = pv.Line(bp[b] if len(bp) else p, p)
                    pl.add_mesh(line.tube(radius=0.006) if line.n_points else line, color="#555555", name=n + "l")
                    pl.add_mesh(pv.Sphere(radius=0.012, center=p), color="#333333", name=n)
                    marker_names += [n, n + "l"]
        if show_forces and "foot_force" in log:
            for name in [m for m in marker_names if m.startswith("force")]:
                pl.remove_actor(name)
            marker_names = [m for m in marker_names if not m.startswith("force")]
            for f, (p, F) in enumerate(zip(np.asarray(log["foot_pos"])[i], np.asarray(log["foot_force"])[i])):
                mag = float(np.linalg.norm(F))
                if mag <= 1e-9:
                    continue
                L = scale * mag / weight
                pl.add_mesh(pv.Arrow(start=p, direction=F / mag, scale=L), color="#d32f2f", name=f"force{f}")
                marker_names.append(f"force{f}")
        cam = first_cam or _camera(camera, com[i], scale)
        pl.camera_position = [cam[0], cam[1], cam[2]]
        if show_time:
            pl.add_text(f"t = {t[i]:.2f} s", position="upper_left", font_size=10, color="black", name="time")
        pl.render()                                          # screenshot() renders only the first time
        images.append(np.asarray(pl.screenshot(return_img=True))[..., :3].copy())
    pl.close()
    return images


def to_video(images, path, fps=25):
    """Write RGB frames to a video (mp4v codec for .mp4, MJPG for .avi) with OpenCV; returns the path."""
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not images:
        raise ValueError("no frames")
    h, w = images[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*("MJPG" if path.suffix.lower() == ".avi" else "mp4v"))
    out = cv2.VideoWriter(str(path), fourcc, float(fps), (w, h))
    try:
        for img in images:
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            out.write(cv2.cvtColor(np.ascontiguousarray(img, dtype=np.uint8), cv2.COLOR_RGB2BGR))
    finally:
        out.release()
    return path


def render(episode, path, *, speed=1.0, fps=25, camera="follow", size=(960, 540), **kwargs):
    """Frames at ``speed`` × real time and the movie in one call; returns the path."""
    log = episode.log if hasattr(episode, "log") else episode
    t = np.asarray(log["t"])
    dt = float(t[1] - t[0]) if len(t) > 1 else 1.0
    every = max(1, int(round(speed / (fps * dt))))
    return to_video(frames(episode, camera=camera, every=every, size=size, **kwargs), path, fps=fps)
