"""Wing sections: NACA 4-digit, NACA 5-digit (plain and reflexed), coordinates from a file; plain flaps; thin-airfoil
theory (the zero-lift angle, the quarter-chord moment, a flap's effectiveness) as the first estimate and the check of
every backend.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# NACA 5-digit camber lines, design lift coefficient 0.3 (the second digit P: maximum camber at P/20 of the chord)
# plain (third digit 0): (r, k1); reflexed (third digit 1): (r, k1, k2/k1) — Abbott & von Doenhoff, NASA's naca456.
NACA5_PLAIN = {1: (0.0580, 361.4), 2: (0.1260, 51.64), 3: (0.2025, 15.957), 4: (0.2900, 6.643), 5: (0.3910, 3.230)}
NACA5_REFLEX = {2: (0.1300, 51.99, 0.000764), 3: (0.2170, 15.793, 0.00677), 4: (0.3180, 6.520, 0.0303), 5: (0.4410, 3.191, 0.1355)}


def thickness_distribution(x, t: float, closed: bool = True):
    """NACA 00xx half thickness at x/c (``closed``: the −0.1036 trailing-edge coefficient, zero thickness at x = 1;
    else the classic −0.1015, a 0.21 % open trailing edge)."""
    x = np.asarray(x, dtype=float)
    a4 = -0.1036 if closed else -0.1015
    return 5 * t * (0.2969 * np.sqrt(np.maximum(x, 0.0)) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 + a4 * x ** 4)


@dataclass(frozen=True)
class Section:
    """A section, chord 1: a camber line and a thickness, or given coordinates. Build one with ``naca``,
    ``from_coordinates`` or ``from_dat``; ``coordinates()`` is the closed outline in Selig order (trailing edge → upper
    surface → leading edge → lower surface → trailing edge)."""
    name: str
    kind: str                                   # naca4 | naca5 | naca5r | coordinates
    params: dict = field(default_factory=dict)
    points: tuple = ()                          # given coordinates (kind == coordinates)

    # ---- the camber line and the thickness
    def camber(self, x):
        x = np.asarray(x, dtype=float)
        p = self.params
        if self.kind == "naca4":
            m, pp = p["m"], p["p"]
            if m == 0:
                return np.zeros_like(x)
            return np.where(x < pp, m / pp ** 2 * (2 * pp * x - x ** 2), m / (1 - pp) ** 2 * (1 - 2 * pp + 2 * pp * x - x ** 2))
        if self.kind == "naca5":
            r, k1 = NACA5_PLAIN[p["P"]]
            y = np.where(x < r, k1 / 6 * (x ** 3 - 3 * r * x ** 2 + r ** 2 * (3 - r) * x), k1 * r ** 3 / 6 * (1 - x))
            return y * p["cl_design"] / 0.3
        if self.kind == "naca5r":
            r, k1, _ = NACA5_REFLEX[p["P"]]
            k21 = p["k21"]
            y = np.where(x < r, k1 / 6 * ((x - r) ** 3 - k21 * (1 - r) ** 3 * x - r ** 3 * x + r ** 3),
                         k1 / 6 * (k21 * (x - r) ** 3 - k21 * (1 - r) ** 3 * x - r ** 3 * x + r ** 3))
            return y * p["cl_design"] / 0.3
        raise ValueError("a section from coordinates has no analytic camber line; use camber_from_points")

    @property
    def thickness(self) -> float | None:
        return self.params.get("t")

    def coordinates(self, n: int = 100, te_cut: float = 1.0, closed: bool = True) -> np.ndarray:
        """The outline [N, 2] in Selig order, cosine-spaced with ``n`` points per surface; ``te_cut`` < 1 truncates the
        chord (a finite trailing edge for a CAD model) and rescales nothing."""
        if self.kind == "coordinates":
            return np.array(self.points, dtype=float)
        b = np.linspace(0.0, math.pi, n)
        x = te_cut * 0.5 * (1 - np.cos(b))
        yc = self.camber(x)
        dy = np.gradient(yc, x) if n > 2 else np.zeros_like(x)
        th = np.arctan(dy)
        yt = thickness_distribution(x, self.params["t"], closed)
        up = np.c_[x - yt * np.sin(th), yc + yt * np.cos(th)]
        lo = np.c_[x + yt * np.sin(th), yc - yt * np.cos(th)]
        return np.r_[up[::-1], lo[1:]]

    def flapped(self, cf: float, delta_deg: float, n: int = 120) -> "Section":
        """The same section with a plain flap of chord fraction ``cf`` turned ``delta_deg`` (trailing edge down
        positive) about a hinge on the camber line at x = 1 − cf: a section from coordinates."""
        if not 0 < cf < 1:
            raise ValueError("the flap chord fraction must be between 0 and 1")
        xy = self.coordinates(n)
        xh = 1 - cf
        zh = float(self.camber(xh)) if self.kind != "coordinates" else float(camber_from_points(xy, xh))
        a = math.radians(-delta_deg)
        out = xy.copy()
        aft = xy[:, 0] > xh
        dx, dz = xy[aft, 0] - xh, xy[aft, 1] - zh
        out[aft, 0] = xh + dx * math.cos(a) - dz * math.sin(a)
        out[aft, 1] = zh + dx * math.sin(a) + dz * math.cos(a)
        return Section(f"{self.name} flap {cf:.2f}c {delta_deg:+g}°", "coordinates", dict(self.params, flap_cf=cf, flap_deg=delta_deg),
                       tuple(map(tuple, out)))

    def write_dat(self, path: str | Path, n: int = 100) -> Path:
        """A Selig-format .dat file (the name, then x y per line)."""
        path = Path(path)
        xy = self.coordinates(n)
        path.write_text(self.name + "\n" + "\n".join(f"{x: .6f} {y: .6f}" for x, y in xy) + "\n")
        return path

    def to_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind, "params": dict(self.params), **({"points": [list(p) for p in self.points]} if self.points else {})}


def naca(designation: str, *, reflex_k21: float | None = None) -> Section:
    """``naca("2412")``, ``naca("NACA 0012")``, ``naca("23012")`` (5-digit plain), ``naca("23112")`` (5-digit
    reflexed; ``reflex_k21`` replaces the standard k2/k1 — more reflex, a positive moment)."""
    digits = re.sub(r"\D", "", designation)
    if len(digits) == 4:
        m, p, t = int(digits[0]) / 100, int(digits[1]) / 10, int(digits[2:]) / 100
        if m > 0 and not 0 < p < 1:
            raise ValueError(f"NACA {digits}: the camber position must be 1-9")
        return Section(f"NACA {digits}", "naca4", {"m": m, "p": p, "t": t})
    if len(digits) == 5:
        L, P, S, t = int(digits[0]), int(digits[1]), int(digits[2]), int(digits[3:]) / 100
        cl_design = 0.15 * L
        if S == 0:
            if P not in NACA5_PLAIN:
                raise ValueError(f"NACA {digits}: no plain 5-digit camber line for P = {P}")
            return Section(f"NACA {digits}", "naca5", {"L": L, "P": P, "t": t, "cl_design": cl_design})
        if S == 1:
            if P not in NACA5_REFLEX:
                raise ValueError(f"NACA {digits}: no reflexed 5-digit camber line for P = {P} (2-5)")
            k21 = NACA5_REFLEX[P][2] if reflex_k21 is None else float(reflex_k21)
            name = f"NACA {digits}" + (f" (k2/k1 {k21:g})" if reflex_k21 is not None else "")
            return Section(name, "naca5r", {"L": L, "P": P, "t": t, "cl_design": cl_design, "k21": k21})
        raise ValueError(f"NACA {digits}: the third digit is 0 (plain) or 1 (reflexed)")
    raise ValueError(f"{designation!r} is not a NACA 4- or 5-digit designation")


def from_coordinates(name: str, xy) -> Section:
    """A section from an outline [N, 2] in Selig order (closed or not). An outline not already on the unit chord
    (a chord in millimetres, a leading edge far from x = 0) is shifted and scaled to it; one within 2 % of the unit
    chord is kept as given (a cambered section's leading edge sits a little ahead of x = 0 and must stay there)."""
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 10:
        raise ValueError("coordinates must be [N, 2] with at least 10 points")
    x0, x1 = xy[:, 0].min(), xy[:, 0].max()
    if abs(x1 - x0 - 1.0) > 0.02 or x0 < -0.02 or x0 > 0.02:
        xy = (xy - [x0, 0.0]) / (x1 - x0)
    return Section(name, "coordinates", {}, tuple(map(tuple, xy)))


def from_dat(path: str | Path) -> Section:
    """A Selig .dat file (first line the name) or a Lednicer one (two surfaces from the leading edge, a count line)."""
    path = Path(path)
    lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
    name, rows = lines[0], []
    for l in lines[1:]:
        try:
            a, b = l.split()[:2]
            rows.append((float(a), float(b)))
        except ValueError:
            continue
    xy = np.array(rows)
    if len(xy) > 2 and xy[0, 0] > 1.5:                                      # Lednicer: the first row is the two counts
        n_up = int(xy[0, 0])
        up, lo = xy[1:1 + n_up], xy[1 + n_up:]
        xy = np.r_[up[::-1], lo[1:]]
    return from_coordinates(name, xy)


def camber_from_points(xy: np.ndarray, x):
    """The mean line of an outline at x/c (upper and lower surface interpolated)."""
    xy = np.asarray(xy, dtype=float)
    i_le = int(np.argmin(xy[:, 0]))
    up, lo = xy[:i_le + 1][::-1], xy[i_le:]
    return 0.5 * (np.interp(x, up[:, 0], up[:, 1]) + np.interp(x, lo[:, 0], lo[:, 1]))


# ================================================================================================= thin-airfoil theory
def thin_airfoil(section: Section, n: int = 6000) -> dict:
    """Glauert's integrals on the camber line: the zero-lift angle α₀ [deg], the moment about the quarter chord
    Cm_c/4 (independent of α), the lift slope 2π per rad, the ideal angle of attack."""
    th = np.linspace(1e-4, math.pi - 1e-4, n)
    x = 0.5 * (1 - np.cos(th))
    if section.kind == "coordinates":
        xy = np.array(section.points)
        dz = np.gradient(camber_from_points(xy, x), x)
    else:
        dz = np.gradient(section.camber(x), x)
    A0 = -np.trapezoid(dz * (np.cos(th) - 1), th) / math.pi      # α − α0 = A0 at zero lift
    A = lambda k: 2 / math.pi * np.trapezoid(dz * np.cos(k * th), th)
    A1, A2 = A(1), A(2)
    alpha_ideal = np.trapezoid(dz, th) / math.pi                     # the leading-edge term A0 vanishes: no suction peak
    return {"alpha0_deg": math.degrees(A0), "cm_c4": math.pi / 4 * (A2 - A1), "cl_alpha_per_rad": 2 * math.pi,
            "alpha_ideal_deg": math.degrees(alpha_ideal), "cl_ideal": math.pi * A1}


def flap_effectiveness(cf: float) -> float:
    """Thin-airfoil τ = dα₀/dδ of a plain flap of chord fraction ``cf``: 1 − (θ_h − sin θ_h)/π, θ_h = acos(2 cf − 1)."""
    th = math.acos(2 * cf - 1)
    return 1 - (th - math.sin(th)) / math.pi
