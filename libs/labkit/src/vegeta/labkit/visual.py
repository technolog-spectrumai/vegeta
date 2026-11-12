"""A machine's looks: its CAD design as a light triangle mesh that rides in the export as a visual shape.

A Chiron robot is boxes, cylinders and spheres (what a simulator needs); the CAD design is what the machine looks like.
``machines.register(..., visual="module:Class")`` names the design; ``tools/export.py`` builds it, makes it light
(``mesh``) and places it over the robot (``place``), as the root link's ``visual`` mesh (Yax system tag ``00geom``).

    pts, faces = mesh(design_or_geometry, max_triangles=6000)     # metres, the design's own frame
    pts = place(pts, machine)                                      # in the machine's root link frame

numpy only: the design is tessellated by CadQuery (the lab already has it), the simplification is vertex clustering.
"""
from __future__ import annotations

import importlib

import numpy as np

MAX_TRIANGLES = 6000


def build(spec):
    """A design from ``"module:Class"`` (imported from the lab's designs, on ``sys.path``), a class, a ``Design``
    instance or a callable; its geometry (``generate()``) or the object itself when it is already one."""
    if isinstance(spec, str):
        mod, name = spec.split(":")
        spec = getattr(importlib.import_module(mod), name)
    if isinstance(spec, type):
        spec = spec()
    elif callable(spec) and not hasattr(spec, "generate"):
        spec = spec()
    return spec.generate() if hasattr(spec, "generate") else spec


def tessellate(geometry, tolerance_mm: float = 0.5, angular: float = 0.3):
    """(points N×3 in metres, faces M×3) of a CadQuery shape, workplane or Vegeta geometry (millimetres)."""
    shape = geometry.val() if hasattr(geometry, "val") else geometry
    verts, tris = shape.tessellate(tolerance_mm, angular)
    pts = np.array([(v.x, v.y, v.z) for v in verts] if len(verts) and hasattr(verts[0], "x") else verts, float) * 1e-3
    return pts, np.asarray(tris, int).reshape(-1, 3)


def cluster(pts, faces, cell: float):
    """Vertex clustering on a ``cell``-metre grid: the vertices of a cell merge at their mean, collapsed and repeated
    triangles go. Shapes smaller than a cell flatten; the outline stays."""
    key = np.floor(pts / cell).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    n = int(inv.max()) + 1
    out = np.zeros((n, 3))
    np.add.at(out, inv, pts)
    out /= np.bincount(inv, minlength=n)[:, None]
    f = inv[faces]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    _, first = np.unique(np.sort(f, axis=1), axis=0, return_index=True)
    f = f[np.sort(first)]                                   # one of each triangle, in its first order and winding
    used = np.unique(f)
    remap = np.full(n, -1)
    remap[used] = np.arange(len(used))
    return out[used], remap[f]


def mesh(design, max_triangles: int = MAX_TRIANGLES, tolerance_mm: float = 0.5):
    """The design as at most ``max_triangles`` triangles, metres, in its own frame (the cell grows by 25 % until it
    fits; a mesh already small enough is kept as tessellated)."""
    pts, faces = tessellate(build(design), tolerance_mm)
    if len(faces) <= max_triangles:
        return pts, faces
    cell = float(np.max(np.ptp(pts, axis=0))) / 400
    while True:
        p, f = cluster(pts, faces, cell)
        if len(f) <= max_triangles:
            return p, f
        cell *= 1.25


def extent(machine):
    """(lo, hi) of the machine's primitive shapes in the machine frame at q = 0 (its meshes left out)."""
    from yax import hub
    los, his = [], []
    for lk in machine.links:
        Rl, pl = hub.R(lk.quat), np.asarray(lk.pos, float)
        for s in lk.shapes:
            if s.kind == "mesh":
                continue
            p, _ = hub.tessellate(s)
            w = pl + np.asarray(p, float) @ Rl.T
            los.append(w.min(axis=0))
            his.append(w.max(axis=0))
    if not los:
        raise ValueError(f"{machine.name}: no shapes to place the visual mesh over")
    return np.min(los, axis=0), np.max(his, axis=0)


def place(pts, machine):
    """The CAD mesh over the robot: its footprint centred on the primitives' footprint, its lowest point on theirs
    (the ground at q = 0), then in the root link's frame (the designs' x forward, z up, as Chiron's)."""
    from yax import hub
    lo, hi = extent(machine)
    plo, phi = pts.min(axis=0), pts.max(axis=0)
    shift = np.array([(lo[0] + hi[0] - plo[0] - phi[0]) / 2, (lo[1] + hi[1] - plo[1] - phi[1]) / 2, lo[2] - plo[2]])
    w = pts + shift
    root = machine.links[0]
    return (w - np.asarray(root.pos, float)) @ hub.R(root.quat)
