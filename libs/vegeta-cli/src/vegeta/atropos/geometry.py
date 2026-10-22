"""Flat parts in millimetres: closed and open paths in the XY plane, parts (an outer contour, holes, score and
engrave lines), offsets for the kerf, clipping by half-planes, point-in-polygon. Outer contours run counter-clockwise,
holes clockwise (``normalised()`` enforces it); ``offset(d)`` moves a contour away from the region it encloses."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np


def _pts(points) -> np.ndarray:
    a = np.asarray(points, float)
    if a.ndim != 2 or a.shape[1] != 2 or len(a) < 2:
        raise ValueError("a path needs at least two (x, y) points")
    return a


def dedupe(points: np.ndarray, closed: bool = True, tol: float = 1e-9) -> np.ndarray:
    """Drop consecutive repeats (and, for a closed path, a last point that repeats the first)."""
    a = _pts(points)
    keep = np.r_[True, np.linalg.norm(np.diff(a, axis=0), axis=1) > tol]
    a = a[keep]
    if closed and len(a) > 1 and np.linalg.norm(a[0] - a[-1]) <= tol:
        a = a[:-1]
    return a


def signed_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


@dataclass
class Path:
    """A polyline (``closed=False``) or a polygon (``closed=True``, the closing segment implied), with the operation
    that cuts it (``cut``, ``score``, ``engrave`` ...)."""

    points: np.ndarray
    closed: bool = True
    op: str = "cut"

    def __post_init__(self):
        self.points = dedupe(self.points, self.closed) if self.closed else dedupe(self.points, False)
        if self.closed and len(self.points) < 3:
            raise ValueError("a closed path needs at least three distinct points")

    # ---------------------------------------------------------------------------------------------- measures
    def length(self) -> float:
        seg = np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum()
        if self.closed:
            seg += float(np.linalg.norm(self.points[0] - self.points[-1]))
        return float(seg)

    def area(self) -> float:
        """Signed area: positive counter-clockwise, 0 for an open path."""
        return signed_area(self.points) if self.closed else 0.0

    def bbox(self) -> tuple[float, float, float, float]:
        (x0, y0), (x1, y1) = self.points.min(axis=0), self.points.max(axis=0)
        return float(x0), float(y0), float(x1), float(y1)

    def centroid(self) -> tuple[float, float]:
        if not self.closed:
            c = self.points.mean(axis=0)
            return float(c[0]), float(c[1])
        x, y = self.points[:, 0], self.points[:, 1]
        xn, yn = np.roll(x, -1), np.roll(y, -1)
        cross = x * yn - xn * y
        a = 0.5 * cross.sum()
        if abs(a) < 1e-12:
            c = self.points.mean(axis=0)
            return float(c[0]), float(c[1])
        return float(((x + xn) * cross).sum() / (6 * a)), float(((y + yn) * cross).sum() / (6 * a))

    # ---------------------------------------------------------------------------------------------- transforms
    def transformed(self, angle_deg: float = 0.0, dx: float = 0.0, dy: float = 0.0) -> "Path":
        """Rotate about the origin by ``angle_deg`` (counter-clockwise), then translate."""
        a = math.radians(angle_deg)
        c, s = math.cos(a), math.sin(a)
        r = np.array([[c, -s], [s, c]])
        return Path(self.points @ r.T + np.array([dx, dy]), self.closed, self.op)

    def reversed(self) -> "Path":
        return Path(self.points[::-1].copy(), self.closed, self.op)

    def normalised(self, ccw: bool = True) -> "Path":
        """The same polygon running counter-clockwise (``ccw=True``) or clockwise."""
        if not self.closed or (self.area() > 0) == ccw:
            return self
        return self.reversed()

    def offset(self, d: float) -> "Path":
        """The closed path moved by ``d`` away from the region it encloses (``d < 0``: into it), vertex by vertex
        along the bisector of the adjacent edge normals, by ``d / cos(θ/2)`` capped at ``2 d`` so a sharp corner
        (a trailing edge) does not run away. Exact for circles and for straight edges; the corners of a rectangle
        are cut square-ish (a chamfer by the cap), which a laser kerf rounds anyway."""
        if not self.closed:
            raise ValueError("offset needs a closed path")
        if d == 0:
            return Path(self.points.copy(), True, self.op)
        p = self.points
        sign = 1.0 if self.area() > 0 else -1.0       # CCW: outward normal is to the right of the edge
        e_next = np.roll(p, -1, axis=0) - p
        e_prev = p - np.roll(p, 1, axis=0)

        def normals(e):
            n = np.stack([e[:, 1], -e[:, 0]], axis=1) * sign
            return n / np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None]

        n1, n2 = normals(e_prev), normals(e_next)
        bis = n1 + n2
        norm = np.linalg.norm(bis, axis=1)
        bis = bis / np.maximum(norm, 1e-12)[:, None]
        cos_half = np.clip(np.einsum("ij,ij->i", bis, n1), 0.5, 1.0)   # cap: 1/cos <= 2
        return Path(p + bis * (d / cos_half)[:, None], True, self.op)

    # ---------------------------------------------------------------------------------------------- queries
    def contains(self, x: float, y: float) -> bool:
        """Point in polygon (ray casting; points on the boundary count as inside)."""
        if not self.closed:
            return False
        px, py = self.points[:, 0], self.points[:, 1]
        qx, qy = np.roll(px, -1), np.roll(py, -1)
        crosses = ((py > y) != (qy > y)) & (x < (qx - px) * (y - py) / np.where(qy != py, qy - py, 1e-300) + px)
        return bool(crosses.sum() % 2 == 1) or self.distance(x, y) < 1e-9

    def distance(self, x: float, y: float) -> float:
        """Distance from a point to the path's edges."""
        p = self.points
        q = np.roll(p, -1, axis=0) if self.closed else p[1:]
        a = p if self.closed else p[:-1]
        e = q - a
        t = np.clip(np.einsum("ij,ij->i", np.array([x, y]) - a, e) / np.maximum(np.einsum("ij,ij->i", e, e), 1e-300), 0, 1)
        proj = a + e * t[:, None]
        return float(np.min(np.hypot(proj[:, 0] - x, proj[:, 1] - y)))

    def to_dict(self) -> dict[str, Any]:
        return {"points": self.points.round(6).tolist(), "closed": self.closed, "op": self.op}

    @classmethod
    def from_dict(cls, d: dict) -> "Path":
        return cls(np.asarray(d["points"], float), bool(d.get("closed", True)), str(d.get("op", "cut")))


def circle(cx: float, cy: float, r: float, n: int = 64, op: str = "cut") -> Path:
    """A counter-clockwise circle."""
    if r <= 0:
        raise ValueError("circle radius must be positive")
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return Path(np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], axis=1), True, op)


def rect(x: float, y: float, w: float, h: float, op: str = "cut") -> Path:
    """A counter-clockwise rectangle from its lower-left corner."""
    if w <= 0 or h <= 0:
        raise ValueError("rectangle sides must be positive")
    return Path(np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]), True, op)


def polyline(points, op: str = "score") -> Path:
    return Path(points, False, op)


def clip_halfplane(points: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Sutherland–Hodgman: the part of the polygon where ``a x + b y + c >= 0``. Empty array when nothing is left."""
    p = _pts(points)
    out = []
    n = len(p)
    for i in range(n):
        cur, prev = p[i], p[i - 1]
        fc, fp = a * cur[0] + b * cur[1] + c, a * prev[0] + b * prev[1] + c
        if fc >= 0:
            if fp < 0:
                out.append(prev + (cur - prev) * (fp / (fp - fc)))
            out.append(cur)
        elif fp >= 0:
            out.append(prev + (cur - prev) * (fp / (fp - fc)))
    if len(out) < 3:
        return np.zeros((0, 2))
    return dedupe(np.array(out), True)


def clip_x(points: np.ndarray, x0: float | None = None, x1: float | None = None) -> np.ndarray:
    """The part of a polygon between two vertical lines (either may be None)."""
    p = _pts(points)
    if x0 is not None:
        p = clip_halfplane(p, 1.0, 0.0, -x0)
    if x1 is not None and len(p):
        p = clip_halfplane(p, -1.0, 0.0, x1)
    return p


@dataclass
class Part:
    """A flat part: the outer contour (cut), holes (cut), score lines (``score``) and engraved lines or text paths
    (``engrave``), with free metadata (station, piece, grain direction, material)."""

    name: str
    outer: Path
    holes: list[Path] = field(default_factory=list)
    scores: list[Path] = field(default_factory=list)
    engraves: list[Path] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.outer.closed:
            raise ValueError(f"part {self.name!r}: the outer contour must be closed")
        self.outer = Path(self.outer.points, True, "cut").normalised(True)
        self.holes = [Path(h.points, True, "cut").normalised(False) for h in self.holes]
        for s in self.scores:
            s.op = "score"
        for e in self.engraves:
            e.op = "engrave"

    # ---------------------------------------------------------------------------------------------- measures
    def paths(self) -> list[Path]:
        return [self.outer, *self.holes, *self.scores, *self.engraves]

    def bbox(self) -> tuple[float, float, float, float]:
        boxes = np.array([p.bbox() for p in self.paths()])
        return float(boxes[:, 0].min()), float(boxes[:, 1].min()), float(boxes[:, 2].max()), float(boxes[:, 3].max())

    def size(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox()
        return x1 - x0, y1 - y0

    def area(self) -> float:
        """Material area: the outer contour minus the holes [mm²]."""
        return abs(self.outer.area()) - sum(abs(h.area()) for h in self.holes)

    def cut_length(self) -> float:
        return self.outer.length() + sum(h.length() for h in self.holes)

    def length(self, op: str) -> float:
        return sum(p.length() for p in self.paths() if p.op == op)

    # ---------------------------------------------------------------------------------------------- transforms
    def transformed(self, angle_deg: float = 0.0, dx: float = 0.0, dy: float = 0.0) -> "Part":
        t = lambda ps: [p.transformed(angle_deg, dx, dy) for p in ps]   # noqa: E731
        return Part(self.name, self.outer.transformed(angle_deg, dx, dy), t(self.holes), t(self.scores), t(self.engraves),
                    dict(self.meta))

    def at_origin(self) -> "Part":
        """Moved so its bounding box starts at (0, 0)."""
        x0, y0, _, _ = self.bbox()
        return self.transformed(0.0, -x0, -y0)

    def kerf_compensated(self, kerf_mm: float) -> "Part":
        """The toolpaths that leave this part after a cut of width ``kerf_mm``: the outer contour moved out by half
        the kerf, the holes moved in (so the holes come out the size drawn). Score and engrave lines are unchanged."""
        if kerf_mm == 0:
            return self
        k = 0.5 * kerf_mm
        holes = [h.offset(-k) for h in self.holes]
        return Part(self.name, self.outer.offset(k), holes, list(self.scores), list(self.engraves), dict(self.meta))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "outer": self.outer.to_dict(), "holes": [h.to_dict() for h in self.holes],
                "scores": [s.to_dict() for s in self.scores], "engraves": [e.to_dict() for e in self.engraves],
                "meta": dict(self.meta)}

    @classmethod
    def from_dict(cls, d: dict) -> "Part":
        return cls(str(d["name"]), Path.from_dict(d["outer"]), [Path.from_dict(h) for h in d.get("holes", [])],
                   [Path.from_dict(s) for s in d.get("scores", [])], [Path.from_dict(e) for e in d.get("engraves", [])],
                   dict(d.get("meta", {})))


def parts_to_json(parts: Iterable[Part]) -> list[dict]:
    return [p.to_dict() for p in parts]


def parts_from_json(data: list[dict]) -> list[Part]:
    return [Part.from_dict(d) for d in data]
