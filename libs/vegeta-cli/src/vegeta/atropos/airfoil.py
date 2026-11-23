"""Aerofoil sections for ribs: a NACA 4-digit generator and the outline, area and arc lengths of any section given as
upper and lower point lists at chord 1 (both from the trailing edge to the leading edge, the convention of the
design files that call Atropos)."""
from __future__ import annotations

import math

import numpy as np


def naca4(camber: float, camber_pos: float, thickness: float, n: int = 40, te_cut: float = 0.97) -> tuple[list, list]:
    """NACA 4-digit section as (upper, lower) point lists from the trailing edge to the leading edge, chord 1,
    truncated at ``te_cut`` so the trailing edge has a small finite thickness (``te_cut=1.0``: sharp)."""
    m, p, t = camber, camber_pos, thickness
    xs = [te_cut * 0.5 * (1 - math.cos(math.pi * i / n)) for i in range(n + 1)]   # cosine spacing, LE dense
    upper, lower = [], []
    for x in xs:
        yt = 5 * t * (0.2969 * math.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
        if p > 0 and x < p:
            yc, dyc = m / p**2 * (2 * p * x - x**2), 2 * m / p**2 * (p - x)
        elif p > 0:
            yc, dyc = m / (1 - p)**2 * (1 - 2 * p + 2 * p * x - x**2), 2 * m / (1 - p)**2 * (p - x)
        else:
            yc, dyc = 0.0, 0.0
        th = math.atan(dyc)
        upper.append((x - yt * math.sin(th), yc + yt * math.cos(th)))
        lower.append((x + yt * math.sin(th), yc - yt * math.cos(th)))
    return upper[::-1], lower[::-1]          # both start at the trailing edge, end at the leading edge


def camber_z(camber: float, camber_pos: float, x_frac: float) -> float:
    """Height of a NACA 4-digit camber line at ``x_frac`` of the chord (chord 1)."""
    m, p, x = camber, camber_pos, x_frac
    if m == 0 or p <= 0:
        return 0.0
    if x < p:
        return m / p**2 * (2 * p * x - x**2)
    return m / (1 - p)**2 * (1 - 2 * p + 2 * p * x - x**2)


def section_outline(upper, lower, chord: float) -> np.ndarray:
    """The closed outline at ``chord`` mm as an (N, 2) array, counter-clockwise: from the trailing edge along the upper
    surface to the leading edge, back along the lower surface. The leading edge is at x = 0, z up, no duplicate
    points, the last point is not a repeat of the first."""
    up = np.asarray(upper, float) * chord
    lo = np.asarray(lower, float) * chord
    if up.ndim != 2 or lo.ndim != 2 or len(up) < 3 or len(lo) < 3:
        raise ValueError("upper and lower must be lists of (x, z) points")
    if up[0, 0] < up[-1, 0]:
        raise ValueError("upper and lower must run from the trailing edge to the leading edge")
    pts = np.vstack([up, lo[::-1]])
    keep = np.r_[True, np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-9]
    pts = pts[keep]
    if np.linalg.norm(pts[0] - pts[-1]) <= 1e-9:
        pts = pts[:-1]
    return pts


def section_area(upper, lower, chord: float) -> float:
    """Enclosed area [mm²] of the section at ``chord`` mm."""
    pts = section_outline(upper, lower, chord)
    x, z = pts[:, 0], pts[:, 1]
    return abs(0.5 * float(np.dot(x, np.roll(z, -1)) - np.dot(z, np.roll(x, -1))))


def thickness_at(upper, lower, chord: float, x_frac: float) -> tuple[float, float]:
    """(z_upper, z_lower) [mm] of the section at ``x_frac`` of the chord (linear interpolation on each surface)."""
    up = np.asarray(upper, float)[::-1]      # LE -> TE, x increasing
    lo = np.asarray(lower, float)[::-1]
    x = x_frac
    zu = float(np.interp(x, up[:, 0], up[:, 1])) * chord
    zl = float(np.interp(x, lo[:, 0], lo[:, 1])) * chord
    return zu, zl


def arc_length(curve, chord: float, x0_frac: float = 0.0, x1_frac: float = 1.0) -> float:
    """Length [mm] of one surface (``upper`` or ``lower`` as returned by ``naca4``) between two chord fractions:
    what a sheet glued over that surface must be wide."""
    c = np.asarray(curve, float)[::-1] * chord          # LE -> TE
    x0, x1 = sorted((x0_frac * chord, x1_frac * chord))
    xs = c[:, 0]
    inside = (xs >= x0) & (xs <= x1)
    pts = [np.array([x0, np.interp(x0, xs, c[:, 1])])] + [p for p in c[inside]] + [np.array([x1, np.interp(x1, xs, c[:, 1])])]
    pts = np.array(pts)
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
