"""Wing ribs from a section: the outline at the station's chord, the skin let in where the wing is sheeted, slots for
cap strips and stringers, the nose and tail cut back for leading- and trailing-edge stock, round holes for tube spars,
lightening holes that keep a wall, and the rib split at a hinge line into a wing piece and a control-surface piece
with a bevelled nose. Everything is stated in the ``RibSpec``; nothing is sized by Atropos."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .airfoil import section_outline
from .geometry import Part, Path, circle, clip_halfplane, clip_x, dedupe


@dataclass(frozen=True)
class Sheeting:
    """The skin between two chord fractions on ``side`` (``upper``, ``lower`` or ``both``): the rib is let in by the
    sheet's thickness there, so the sheeted surface comes out on the section line."""

    x0_frac: float
    x1_frac: float
    thickness_mm: float
    side: str = "both"

    def __post_init__(self):
        if not 0 <= self.x0_frac < self.x1_frac <= 1:
            raise ValueError("sheeting needs 0 <= x0_frac < x1_frac <= 1")
        if self.thickness_mm <= 0 or self.side not in ("upper", "lower", "both"):
            raise ValueError("sheeting needs a positive thickness and side upper/lower/both")


@dataclass(frozen=True)
class TubeHole:
    """A round hole for a spar tube at ``x_frac`` of the chord: ``diameter_mm`` plus ``clearance_mm`` on the diameter
    (the glue gap). ``z_mm`` places the centre (rib coordinates: x from the leading edge, z up); None puts it on the
    section's mid line. ``min_wall_mm`` of rib must remain around the hole or ``make_rib`` refuses."""

    x_frac: float
    diameter_mm: float
    clearance_mm: float = 0.0
    z_mm: float | None = None
    min_wall_mm: float = 0.0
    name: str = "spar"

    @property
    def bore(self) -> float:
        return self.diameter_mm + self.clearance_mm


@dataclass(frozen=True)
class Lightening:
    """A round lightening hole at ``x_frac``, at most ``diameter_mm`` across, shrunk to keep ``wall_mm`` to the
    surfaces and to the other holes, dropped below ``min_diameter_mm``."""

    x_frac: float
    diameter_mm: float
    wall_mm: float
    min_diameter_mm: float = 4.0


@dataclass(frozen=True)
class Notch:
    """A rectangular slot from the ``upper`` or ``lower`` surface at ``x_frac``: ``width_mm`` along the chord,
    ``depth_mm`` below the lower of the surface's two heights at the slot's edges (a cap strip or a stringer sits in it)."""

    x_frac: float
    side: str
    width_mm: float
    depth_mm: float

    def __post_init__(self):
        if self.side not in ("upper", "lower") or self.width_mm <= 0 or self.depth_mm <= 0:
            raise ValueError("notch needs side upper/lower and positive width and depth")


@dataclass
class RibSpec:
    """One rib. ``upper`` and ``lower`` are the section at chord 1 from the trailing edge to the leading edge (as
    ``airfoil.naca4`` returns them); everything else is in mm or chord fractions. ``le_cut_mm`` and ``te_cut_mm``
    remove the nose and the tail of the rib for leading- and trailing-edge stock. ``split_at_frac`` cuts the rib at a
    hinge line into ``<name>-fwd`` and ``<name>-aft`` with ``split_gap_mm`` between them; the aft piece's nose is a
    wedge of ``bevel_deg`` either side of the hinge point (``hinge_z_mm``, or the mid line) so the surface can move."""

    name: str
    chord_mm: float
    upper: list
    lower: list
    sheeting: list[Sheeting] = field(default_factory=list)
    tubes: list[TubeHole] = field(default_factory=list)
    lightening: list[Lightening] = field(default_factory=list)
    notches: list[Notch] = field(default_factory=list)
    le_cut_mm: float = 0.0
    te_cut_mm: float = 0.0
    split_at_frac: float | None = None
    split_gap_mm: float = 0.0
    bevel_deg: float = 0.0
    hinge_z_mm: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.chord_mm <= 0:
            raise ValueError(f"rib {self.name!r}: chord must be positive")
        if self.split_at_frac is not None and not 0 < self.split_at_frac < 1:
            raise ValueError(f"rib {self.name!r}: split_at_frac must be inside (0, 1)")
        if self.le_cut_mm < 0 or self.te_cut_mm < 0 or self.le_cut_mm + self.te_cut_mm >= self.chord_mm:
            raise ValueError(f"rib {self.name!r}: le_cut_mm/te_cut_mm must be >= 0 and leave some rib")


# ================================================================================================= helpers
def _tagged_outline(spec: RibSpec) -> tuple[np.ndarray, np.ndarray]:
    """The outline and a tag per vertex: 0 upper, 1 lower (the leading edge counts as upper, the trailing edge's
    upper corner as upper, its lower corner as lower)."""
    up = np.asarray(spec.upper, float) * spec.chord_mm
    pts = section_outline(spec.upper, spec.lower, spec.chord_mm)
    n_up = 0
    for i, p in enumerate(pts):
        if np.any(np.all(np.isclose(up, p, atol=1e-9), axis=1)):
            n_up = i + 1
    tags = np.array([0] * n_up + [1] * (len(pts) - n_up))
    return pts, tags


def _surface_z(pts: np.ndarray, tags: np.ndarray, side: int, x: float) -> float:
    """Height of one surface (0 upper, 1 lower) of a tagged outline at ``x`` (linear interpolation)."""
    s = pts[tags == side]
    order = np.argsort(s[:, 0])
    return float(np.interp(x, s[order, 0], s[order, 1]))


def _crossing(pts: np.ndarray, tags: np.ndarray, side: int, x: float) -> tuple[int, float] | None:
    """(edge index i, parameter t) where the ``side`` surface's edge i -> i+1 crosses ``x``."""
    n = len(pts)
    for i in range(n):
        j = (i + 1) % n
        if tags[i] != side and tags[j] != side:
            continue
        xi, xj = pts[i, 0], pts[j, 0]
        if (xi < x < xj) or (xj < x < xi):
            return i, (x - xi) / (xj - xi)
    return None


def _insert_boundary(pts: np.ndarray, tags: np.ndarray, side: int, x: float, twice: bool = True,
                     others: list[np.ndarray] | None = None):
    """Insert the point where the ``side`` surface crosses ``x`` (twice, so a region can start or stop with a step).
    ``others`` are polygons with the same vertices (offsets of ``pts``) that get the matching point too."""
    hit = _crossing(pts, tags, side, x)
    if hit is None:
        return (pts, tags) if others is None else (pts, tags, others)
    i, t = hit
    j = (i + 1) % len(pts)
    k = 2 if twice else 1

    def ins(a):
        p = a[i] + (a[j] - a[i]) * t
        return np.insert(a, j, [p] * k, axis=0)

    new_pts = ins(pts)
    tags = np.insert(tags, j, [side] * k)
    if others is None:
        return new_pts, tags
    return new_pts, tags, [ins(o) for o in others]


def _let_in(pts: np.ndarray, tags: np.ndarray, sheeting: list[Sheeting], chord: float) -> tuple[np.ndarray, np.ndarray]:
    """Move the sheeted stretches of the outline inwards by the sheet thickness (along the vertex normals)."""
    regions = []
    for sh in sheeting:
        for side in ((0, 1) if sh.side == "both" else ((0,) if sh.side == "upper" else (1,))):
            regions.append((side, sh.x0_frac * chord, sh.x1_frac * chord, sh.thickness_mm))
    thicknesses = sorted({r[3] for r in regions})
    insets = [Path(pts, True).offset(-t).points for t in thicknesses]     # on the clean outline, vertex for vertex
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    boundaries = []
    for side, x0, x1, _ in regions:
        for xb in (x0, x1):
            if x_min < xb < x_max:
                pts, tags, insets = _insert_boundary(pts, tags, side, xb, True, insets)
                boundaries.append((side, xb))
    n = len(pts)
    thick = np.zeros(n)
    is_boundary = np.zeros(n, bool)
    for i in range(n):
        for side, xb in boundaries:
            if tags[i] == side and abs(pts[i, 0] - xb) < 1e-9:
                is_boundary[i] = True
    for i in range(n):
        if is_boundary[i]:
            continue
        for side, x0, x1, t in regions:
            if tags[i] == side and x0 - 1e-9 <= pts[i, 0] <= x1 + 1e-9:
                thick[i] = max(thick[i], t)
    # a boundary pair (i, i+1): the first copy follows its previous neighbour, the second its next one
    i = 0
    while i < n:
        if is_boundary[i]:
            j = (i + 1) % n
            if is_boundary[j] and np.allclose(pts[i], pts[j]):
                thick[i] = thick[(i - 1) % n]
                thick[j] = thick[(j + 1) % n]
                i += 2
                continue
            thick[i] = thick[(i - 1) % n]
        i += 1
    out = pts.copy()
    for t, inset in zip(thicknesses, insets):
        sel = thick == t
        out[sel] = inset[sel]
    return out, tags


def _notch(pts: np.ndarray, tags: np.ndarray, notch: Notch, chord: float) -> tuple[np.ndarray, np.ndarray]:
    side = 0 if notch.side == "upper" else 1
    xa, xb = notch.x_frac * chord - 0.5 * notch.width_mm, notch.x_frac * chord + 0.5 * notch.width_mm
    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    if xa <= x_min or xb >= x_max:
        raise ValueError(f"notch at {notch.x_frac:g} c runs off the rib")
    za, zb = _surface_z(pts, tags, side, xa), _surface_z(pts, tags, side, xb)
    z_bottom = (min(za, zb) - notch.depth_mm) if side == 0 else (max(za, zb) + notch.depth_mm)
    pts, tags = _insert_boundary(pts, tags, side, xa)
    pts, tags = _insert_boundary(pts, tags, side, xb)
    keep = np.ones(len(pts), bool)
    new = pts.copy()
    for i in range(len(pts)):
        if tags[i] != side:
            continue
        x = pts[i, 0]
        if xa + 1e-9 < x < xb - 1e-9:
            keep[i] = False
    # the second copy of each boundary pair (in path order) is the one on the slot's bottom
    for xb_ in (xa, xb):
        idx = [i for i in range(len(pts)) if tags[i] == side and abs(pts[i, 0] - xb_) < 1e-9]
        if len(idx) == 2:
            # the inner copy is the one whose neighbour (towards the slot) is removed or the other copy
            i0, i1 = idx
            prev_removed = not keep[(i0 - 1) % len(pts)]
            inner = i0 if prev_removed else i1
            new[inner, 1] = z_bottom
    return new[keep], tags[keep]


def _z_range_at(poly: np.ndarray, x: float) -> tuple[float, float] | None:
    """(z_low, z_high) of a polygon on the vertical line ``x`` (None when the line misses it)."""
    zs = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        if (a[0] <= x <= b[0] or b[0] <= x <= a[0]) and a[0] != b[0]:
            zs.append(a[1] + (b[1] - a[1]) * (x - a[0]) / (b[0] - a[0]))
    if len(zs) < 2:
        return None
    return min(zs), max(zs)


def _check_hole(outline: Path, hole: Path, wall: float, what: str):
    for x, y in hole.points:
        if not outline.contains(x, y) or outline.distance(x, y) < wall - 1e-9:
            raise ValueError(f"{what} does not fit inside the rib (wall {wall:g} mm)")


# ================================================================================================= the rib
def make_rib(spec: RibSpec) -> list[Part]:
    """The rib's part(s): one, or the forward and aft pieces when ``split_at_frac`` is set."""
    c = spec.chord_mm
    pts, tags = _tagged_outline(spec)
    if spec.sheeting:
        pts, tags = _let_in(pts, tags, spec.sheeting, c)
    for notch in spec.notches:
        pts, tags = _notch(pts, tags, notch, c)
    poly = dedupe(pts, True)
    poly = clip_x(poly, spec.le_cut_mm if spec.le_cut_mm > 0 else None, c - spec.te_cut_mm if spec.te_cut_mm > 0 else None)
    if len(poly) < 3:
        raise ValueError(f"rib {spec.name!r}: nothing left after the leading/trailing edge cuts")

    tube_holes = []
    for tube in spec.tubes:
        x = tube.x_frac * c
        if tube.z_mm is None:
            zr = _z_range_at(section_outline(spec.upper, spec.lower, c), x)
            if zr is None:
                raise ValueError(f"rib {spec.name!r}: tube {tube.name!r} at {tube.x_frac:g} c is off the rib")
            z = 0.5 * (zr[0] + zr[1])
        else:
            z = tube.z_mm
        tube_holes.append((tube, circle(x, z, 0.5 * tube.bore)))

    pieces: list[tuple[str, np.ndarray, str]] = []
    if spec.split_at_frac is None:
        pieces.append((spec.name, poly, "rib"))
    else:
        xs = spec.split_at_frac * c
        g = 0.5 * spec.split_gap_mm
        fwd = clip_x(poly, None, xs - g)
        aft = clip_x(poly, xs + g, None)
        if len(fwd) < 3 or len(aft) < 3:
            raise ValueError(f"rib {spec.name!r}: the split at {spec.split_at_frac:g} c leaves an empty piece")
        if spec.bevel_deg > 0:
            zr = _z_range_at(poly, xs)
            zh = spec.hinge_z_mm if spec.hinge_z_mm is not None else 0.5 * (zr[0] + zr[1])
            tb = math.tan(math.radians(spec.bevel_deg))
            xa = xs + g
            # keep (x - xa) - tb (z - zh) >= 0 and (x - xa) + tb (z - zh) >= 0: a wedge opening aft from the hinge
            aft = clip_halfplane(aft, 1.0, -tb, -xa + tb * zh)
            aft = clip_halfplane(aft, 1.0, tb, -xa - tb * zh)
        pieces.append((f"{spec.name}-fwd", fwd, "wing"))
        pieces.append((f"{spec.name}-aft", aft, "surface"))

    parts = []
    for name, outline_pts, role in pieces:
        outline = Path(outline_pts, True)
        x0, _, x1, _ = outline.bbox()
        holes = []
        for tube, hole in tube_holes:
            cx = hole.centroid()[0]
            if x0 <= cx <= x1:
                _check_hole(outline, hole, tube.min_wall_mm, f"rib {spec.name!r}: the {tube.name} hole ({tube.bore:g} mm at {tube.x_frac:g} c)")
                holes.append(hole)
        for lt in spec.lightening:
            x = lt.x_frac * c
            if not x0 < x < x1:
                continue
            zr = _z_range_at(outline.points, x)
            if zr is None:
                continue
            d = min(lt.diameter_mm, (zr[1] - zr[0]) - 2 * lt.wall_mm)
            for h in holes:
                hc = h.centroid()
                rh = 0.5 * (h.bbox()[2] - h.bbox()[0])
                d = min(d, 2 * (math.hypot(hc[0] - x, hc[1] - 0.5 * (zr[0] + zr[1])) - rh - lt.wall_mm))
            d = min(d, 2 * (x - x0 - lt.wall_mm), 2 * (x1 - x - lt.wall_mm))
            # the surfaces slope: shrink until the wall holds everywhere around the hole
            while d >= lt.min_diameter_mm:
                hole = circle(x, 0.5 * (zr[0] + zr[1]), 0.5 * d)
                try:
                    _check_hole(outline, hole, lt.wall_mm, "lightening hole")
                    for h in holes:
                        if any(h.contains(px, py) or h.distance(px, py) < lt.wall_mm for px, py in hole.points):
                            raise ValueError("too close to another hole")
                except ValueError:
                    d -= 0.5
                    continue
                holes.append(hole)
                break
        meta = dict(spec.meta, chord_mm=c, role=role, rib=spec.name)
        parts.append(Part(name, outline, holes, meta=meta))
    return parts
