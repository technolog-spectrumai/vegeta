"""Multi-view renders of a model: what the reviewer looks at, and what the user sees per revision.

``render_views(parts, outdir)`` writes ``front.png``, ``right.png``, ``top.png``, ``iso.png``, ``rear_iso.png``
(``size`` px square, part colours, feature edges) and ``sheet.png`` (the five views and a legend with the overall
size and the part colours, 3 x 2 tiles: 1536 x 1024 px at the default size, about 2k image tokens). Rendering is
off-screen pyvista/VTK when it works here (probed once, in a subprocess, so a broken GL stack cannot crash the
caller) and matplotlib otherwise; ``FIDIA_RENDER=matplotlib`` forces the fallback.
"""
from __future__ import annotations

import functools
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .mesh import Part, bounds

# name: (direction from the model to the camera, view up, parallel projection)
VIEWS = {
    "front": ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0), True),
    "right": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), True),
    "top": ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), True),
    "iso": ((1.0, -1.0, 0.8), (0.0, 0.0, 1.0), False),
    "rear_iso": ((-1.0, 1.0, 0.8), (0.0, 0.0, 1.0), False),
}
BACKGROUND = (246, 247, 249)

_PROBE = """
import pyvista as pv
pl = pv.Plotter(off_screen=True, window_size=(64, 64))
pl.add_mesh(pv.Sphere(), color="red")
img = pl.screenshot(return_img=True)
pl.close()
assert img.shape[0] == 64 and img.std() > 1
"""


@functools.lru_cache(maxsize=1)
def pyvista_works() -> bool:
    """Whether off-screen VTK rendering works in this environment (probed once in a subprocess)."""
    if os.environ.get("FIDIA_RENDER", "").lower() == "matplotlib":
        return False
    try:
        import pyvista  # noqa: F401
    except ImportError:
        return False
    try:
        return subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, timeout=120).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _pyvista_views(parts: Sequence[Part], size: int) -> dict[str, np.ndarray]:
    import pyvista as pv

    meshes = []
    for p in parts:
        faces = np.hstack([np.full((len(p.triangles), 1), 3), p.triangles]).ravel()
        meshes.append((pv.PolyData(np.asarray(p.vertices, float), faces).clean(tolerance=1e-9), p))
    pl = pv.Plotter(off_screen=True, window_size=(size, size))
    pl.set_background([c / 255 for c in BACKGROUND])
    for m, p in meshes:
        pl.add_mesh(m, color=p.color[:3], opacity=p.color[3], smooth_shading=True, split_sharp_edges=True,
                    specular=0.2, ambient=0.25)
        edges = m.extract_feature_edges(boundary_edges=True, feature_edges=True, manifold_edges=False, feature_angle=35)
        if edges.n_cells:
            pl.add_mesh(edges, color=(0.15, 0.15, 0.18), line_width=1)
    pl.enable_anti_aliasing("ssaa")
    lo, hi = bounds(list(parts))
    centre = (lo + hi) / 2
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]) - centre
    out = {}
    for name, (direction, up, parallel) in VIEWS.items():
        d = np.asarray(direction) / np.linalg.norm(direction)
        u = np.asarray(up, float)
        pl.camera.focal_point = centre
        pl.camera.position = centre + d * float(np.linalg.norm(hi - lo) or 1.0) * 3
        pl.camera.up = up
        pl.camera.parallel_projection = parallel
        pl.reset_camera()
        if parallel:  # the same scale rule for every orthographic view: the projected box plus a margin
            r = np.cross(u, d)
            half = max(np.abs(corners @ u).max(), np.abs(corners @ r).max())
            pl.camera.parallel_scale = 1.22 * float(half or 1.0)
        else:
            pl.camera.zoom(1.15)
        pl.render()  # screenshot() alone may return the frame rendered by reset_camera(), before the scale change
        out[name] = np.asarray(pl.screenshot(return_img=True))[:, :, :3].copy()
    pl.close()
    return out


def _matplotlib_views(parts: Sequence[Part], size: int) -> dict[str, np.ndarray]:
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    lo, hi = bounds(list(parts))
    centre, span = (lo + hi) / 2, max(float((hi - lo).max()), 1e-9) / 2
    light = np.array([0.4, -0.6, 0.7])
    light /= np.linalg.norm(light)
    polys, colours = [], []
    for p in parts:
        tris = np.asarray(p.vertices, float)[p.triangles]
        n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        shade = 0.45 + 0.55 * np.abs(n @ light)
        colours.append(np.clip(np.outer(shade, p.color[:3]), 0, 1))
        polys.append(tris)
    polys_all, colours_all = np.concatenate(polys), np.concatenate(colours)
    angles = {"front": (0, -90), "right": (0, 0), "top": (90, -90), "iso": (30, -45), "rear_iso": (30, 135)}
    out = {}
    for name in VIEWS:
        fig = plt.figure(figsize=(size / 100, size / 100), dpi=100)
        fig.patch.set_facecolor([c / 255 for c in BACKGROUND])
        ax = fig.add_axes([0, 0, 1, 1], projection="3d")
        ax.set_facecolor([c / 255 for c in BACKGROUND])
        ax.add_collection3d(Poly3DCollection(polys_all, facecolors=colours_all, edgecolors="none", linewidths=0))
        for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), centre):
            setter(c - span, c + span)
        ax.set_box_aspect((1, 1, 1))
        ax.set_proj_type("ortho")
        ax.view_init(*angles[name])
        ax.set_axis_off()
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        out[name] = img
    return out


def _font(px: int):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=px)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def contact_sheet(images: dict[str, np.ndarray], parts: Sequence[Part], size: int, title: str = "") -> "Image":
    """Five views and a legend tile (overall size, part colours) on a 3 x 2 grid."""
    from PIL import Image, ImageDraw

    sheet = Image.new("RGB", (3 * size, 2 * size), BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    label = _font(max(12, size // 28))
    for k, name in enumerate(VIEWS):
        tile = Image.fromarray(images[name]).resize((size, size))
        x, y = (k % 3) * size, (k // 3) * size
        sheet.paste(tile, (x, y))
        draw.text((x + 8, y + 6), name.replace("_", " "), fill=(40, 40, 48), font=label)
        draw.rectangle([x, y, x + size - 1, y + size - 1], outline=(200, 202, 208))
    x, y = 2 * size, size
    lo, hi = bounds(list(parts))
    w, d, h = (hi - lo)
    big = _font(max(14, size // 22))
    lines = [(title[:40], big)] if title else []
    lines += [(f"size  {w:.0f} x {d:.0f} x {h:.0f} mm", big), (f"(x, y, z; Z up)  {len(parts)} part(s)", label)]
    ty = y + 14
    for text, font in lines:
        draw.text((x + 14, ty), text, fill=(30, 30, 36), font=font)
        ty += int(font.size * 1.5) if hasattr(font, "size") else 18
    row = max(14, size // 26)
    for p in parts[: max(1, (size - (ty - y) - 10) // (row + 4))]:
        draw.rectangle([x + 14, ty, x + 14 + row, ty + row], fill=tuple(p.rgba8[:3]), outline=(60, 60, 60))
        draw.text((x + 22 + row, ty - 1), p.name, fill=(30, 30, 36), font=label)
        ty += row + 4
    draw.rectangle([x, y, x + size - 1, y + size - 1], outline=(200, 202, 208))
    return sheet


def render_views(parts: Sequence[Part], outdir: str | Path, *, size: int = 512, backend: str = "auto",
                 title: str = "") -> dict[str, Any]:
    """Write the five views and ``sheet.png`` into ``outdir``; return file names and the backend used."""
    from PIL import Image

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    use = backend if backend != "auto" else ("pyvista" if pyvista_works() else "matplotlib")
    note = None
    if use == "pyvista":
        try:
            images = _pyvista_views(parts, size)
        except Exception as exc:  # fall back rather than lose the revision's pictures
            note, use = f"pyvista failed ({type(exc).__name__}: {exc}); used matplotlib", "matplotlib"
    if use == "matplotlib":
        images = _matplotlib_views(parts, size)
    views = {}
    for name, img in images.items():
        Image.fromarray(img).save(out / f"{name}.png")
        views[name] = f"{name}.png"
    contact_sheet(images, parts, size, title).save(out / "sheet.png")
    info = {"backend": use, "size": size, "views": views, "sheet": "sheet.png"}
    if note:
        info["note"] = note
    return info
