"""STL in and out, and the mass properties of a closed triangle mesh — standard library only.

A mesh is a list of triangles, each three ``(x, y, z)`` vertex tuples, in the units of the file (Yax: the manifest's
``units.length``). Mass properties use the divergence theorem (signed tetrahedra from the origin), so they are exact
for the mesh as written: a closed, consistently outward-wound surface.
"""
from __future__ import annotations

import struct
from pathlib import Path

Triangle = tuple  # ((x, y, z), (x, y, z), (x, y, z))


def read(path) -> list:
    """The triangles of a binary or ASCII STL file."""
    raw = Path(path).read_bytes()
    if len(raw) >= 84:
        n = struct.unpack_from("<I", raw, 80)[0]
        if 84 + 50 * n == len(raw):
            return [tuple(tuple(struct.unpack_from("<3f", raw, 84 + 50 * i + 12 + 12 * k)) for k in range(3))
                    for i in range(n)]
    text = raw.decode("ascii", errors="replace")
    if not text.lstrip().startswith("solid"):
        raise ValueError(f"{path}: neither binary STL (size does not match the triangle count) nor ASCII STL")
    vertices = [tuple(float(v) for v in line.split()[1:4]) for line in text.splitlines()
                if line.strip().startswith("vertex")]
    if len(vertices) % 3:
        raise ValueError(f"{path}: ASCII STL with {len(vertices)} vertices, not a multiple of three")
    return [tuple(vertices[i:i + 3]) for i in range(0, len(vertices), 3)]


def write(path, triangles, header: bytes = b"yax") -> Path:
    """A binary STL (normals from the winding)."""
    out = bytearray(header[:80].ljust(80, b" "))
    out += struct.pack("<I", len(triangles))
    for a, b, c in triangles:
        n = _cross(_sub(b, a), _sub(c, a))
        length = sum(x * x for x in n) ** 0.5 or 1.0
        out += struct.pack("<12fH", *(x / length for x in n), *a, *b, *c, 0)
    path = Path(path)
    path.write_bytes(bytes(out))
    return path


def bbox(triangles) -> list:
    """``[[xmin, ymin, zmin], [xmax, ymax, zmax]]``."""
    pts = [p for t in triangles for p in t]
    return [[min(p[i] for p in pts) for i in range(3)], [max(p[i] for p in pts) for i in range(3)]]


def mass_properties(triangles, density: float = 1.0) -> dict:
    """Volume, centre of mass and the inertia tensor about the centre of mass, for a uniform solid of ``density``
    (mass per unit volume in the mesh's units: with mm and density 1, ``inertia`` is in mm^5).

    ``{"volume": V, "com": [x, y, z], "inertia": [[Ixx, Ixy, Ixz], ...]}`` — the tensor with negated products."""
    volume = 0.0
    first = [0.0, 0.0, 0.0]
    second = [[0.0] * 3 for _ in range(3)]          # integral of x_i x_j dV
    for a, b, c in triangles:
        d = _dot(a, _cross(b, c))                   # 6 x the signed tetrahedron volume
        volume += d / 6.0
        for i in range(3):
            first[i] += d * (a[i] + b[i] + c[i]) / 24.0
            for j in range(3):
                second[i][j] += d * (2 * (a[i] * a[j] + b[i] * b[j] + c[i] * c[j])
                                     + a[i] * b[j] + a[j] * b[i] + a[i] * c[j] + a[j] * c[i]
                                     + b[i] * c[j] + b[j] * c[i]) / 120.0
    if volume <= 0:
        raise ValueError("mesh volume is not positive: the mesh is open or wound inwards")
    com = [f / volume for f in first]
    cov = [[second[i][j] - volume * com[i] * com[j] for j in range(3)] for i in range(3)]
    trace = cov[0][0] + cov[1][1] + cov[2][2]
    inertia = [[density * ((trace if i == j else 0.0) - cov[i][j]) for j in range(3)] for i in range(3)]
    return {"volume": volume, "com": com, "inertia": inertia}


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
