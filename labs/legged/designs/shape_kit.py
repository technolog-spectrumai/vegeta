"""CadQuery shapes the zoomorphic designs share: capsules between two points, egg-shaped bodies (lofted ellipses),
limb chains through a list of points, tapered segment chains (snakes, trunks, tentacles) and thin membranes.
All in mm; x forward, y left, z up."""
from __future__ import annotations

import math

import cadquery as cq

__all__ = ["capsule", "rod", "ball", "egg", "rounded_box", "polyline_limb", "planar_chain", "membrane", "disc",
           "union_all", "with_defaults"]


def _v(p):
    return cq.Vector(*map(float, p))


def rod(a, b, r):
    """A cylinder of radius ``r`` from point ``a`` to point ``b``."""
    a, b = _v(a), _v(b)
    d = b - a
    return cq.Workplane("XY").add(cq.Solid.makeCylinder(r, d.Length, a, d.normalized()))


def ball(c, r):
    return cq.Workplane("XY").sphere(r).translate(tuple(c))


def capsule(a, b, r):
    """A cylinder with round ends from ``a`` to ``b``."""
    return rod(a, b, r).union(ball(a, r)).union(ball(b, r))


def egg(length, width, height, center=(0, 0, 0), nose=0.55, tail=0.6):
    """A body along x: ellipses lofted tail → middle → nose (``nose``/``tail``: end sizes relative to the middle)."""
    L = length
    w = (cq.Workplane("YZ").workplane(offset=-L / 2).ellipse(width * tail / 2, height * tail / 2)
         .workplane(offset=L / 2).ellipse(width / 2, height / 2)
         .workplane(offset=L / 2).ellipse(width * nose / 2, height * nose / 2).loft(ruled=False))
    return w.translate(tuple(center))


def rounded_box(lx, ly, lz, center=(0, 0, 0), fillet=10.0):
    return cq.Workplane("XY").box(lx, ly, lz).edges().fillet(min(fillet, lx / 2 - 0.5, ly / 2 - 0.5, lz / 2 - 0.5)) \
        .translate(tuple(center))


def polyline_limb(points, radii):
    """A limb through ``points`` (3-D, mm): a capsule per segment, radius ``radii[i]`` on segment i (joints are
    balls of the larger radius)."""
    out = None
    for i in range(len(points) - 1):
        s = capsule(points[i], points[i + 1], radii[min(i, len(radii) - 1)])
        out = s if out is None else out.union(s)
    return out


def planar_chain(start, n, seg_len, r0, r1, heading_deg=180.0, bend_deg=0.0, plane="xy", wave=0.0):
    """``n`` segments from ``start``, radius tapering ``r0`` → ``r1``; each turns ``bend_deg`` from the last (a curl),
    plus a sine ``wave`` [deg] (a slither). ``plane``: 'xy' (on the ground) or 'xz' (hanging / rising). Returns
    (solid, points)."""
    pts = [tuple(start)]
    h = math.radians(heading_deg)
    for i in range(n):
        h += math.radians(bend_deg) + math.radians(wave) * math.sin(i * 1.1)
        x, y, z = pts[-1]
        if plane == "xy":
            pts.append((x + seg_len * math.cos(h), y + seg_len * math.sin(h), z))
        else:
            pts.append((x + seg_len * math.cos(h), y, z + seg_len * math.sin(h)))
    radii = [r0 + (r1 - r0) * i / max(n - 1, 1) for i in range(n)]
    return polyline_limb(pts, radii), pts


def membrane(points, thickness):
    """A flat panel through 3 or more coplanar-ish points (wing and ear skins): the polygon in the plane of its
    first three points, extruded ``thickness``."""
    p0, p1, p2 = (_v(p) for p in points[:3])
    n = (p1 - p0).cross(p2 - p0).normalized()
    xd = (p1 - p0).normalized()
    plane = cq.Plane(origin=p0, xDir=xd, normal=n)
    loc = [((_v(p) - p0).dot(xd), (_v(p) - p0).dot(n.cross(xd))) for p in points]
    return cq.Workplane(plane).polyline(loc).close().extrude(thickness / 2, both=True)


def disc(center, r, t, axis="z"):
    wp = {"z": "XY", "x": "YZ", "y": "XZ"}[axis]
    return cq.Workplane(wp).circle(r).extrude(t / 2, both=True).translate(tuple(center))


def union_all(parts):
    out = None
    for s in parts:
        if s is not None:
            out = s if out is None else out.union(s)
    return out


def with_defaults(parameters, **defaults):
    """``parameters`` (a Design's list) with some defaults replaced: a variant of a design (``Wolf`` from ``Dog``)."""
    import dataclasses
    names = {p.name for p in parameters}
    unknown = set(defaults) - names
    if unknown:
        raise KeyError(f"unknown parameters {sorted(unknown)}")
    return [dataclasses.replace(p, default=type(p.default)(defaults[p.name])) if p.name in defaults else p
            for p in parameters]
