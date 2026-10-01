"""Terrains: analytic height functions z = h(x, y) [m] that ChironLab turns into a MuJoCo height field.

World frame: z up; a course runs along +x from x = 0, y is lateral (left positive). Every terrain is a pure,
vectorised function of (x, y) and is fully determined by its parameters and seed — two configurations that use
the same spec walk on the identical surface (paired trials). Nothing here knows about a robot: a study that
normalises heights by a leg length or spacings by a body pitch does that itself and passes metres.

``Terrain.spec()`` returns a plain dict that ``terrain_from_spec`` turns back into the same terrain, so the
surface is recorded with every episode.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

__all__ = ["Terrain", "Flat", "LongitudinalBumps", "AlternatingBumps", "CrossSlope", "Steps", "Rough", "Custom",
           "terrain_from_spec", "TERRAIN_KINDS"]


class Terrain:
    """Base class. Subclasses implement ``_height(x, y)`` on float arrays of equal shape."""

    kind = "terrain"
    #: True when the surface is the plane z = 0 everywhere (ChironLab may then use a MuJoCo plane).
    is_flat = False

    def height(self, x, y):
        """Terrain height [m] at (x, y) [m]; scalars or arrays (broadcast). Returns a float for scalar input."""
        xa, ya = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        z = np.asarray(self._height(xa, ya), dtype=float)
        if z.shape != xa.shape:
            z = np.broadcast_to(z, xa.shape)
        return float(z) if z.ndim == 0 else z

    def _height(self, x, y):  # pragma: no cover - abstract
        raise NotImplementedError

    def heightfield(self, x_range, y_range, cell):
        """Sample the surface on a regular grid of vertices ``cell`` [m] apart.

        Returns ``(Z, extents)``: ``Z[r, c]`` is the height at ``x = x0 + c·dx``, ``y = y0 + r·dy`` (rows run
        along +y, row 0 at ``y0``; this is MuJoCo's ``hfield_data`` order) and ``extents = (x0, x1, y0, y1)``.
        The grid covers the ranges exactly; ``dx``/``dy`` are the cell rounded so an integer number fits.
        """
        x0, x1 = map(float, x_range)
        y0, y1 = map(float, y_range)
        if not (x1 > x0 and y1 > y0 and cell > 0):
            raise ValueError("heightfield needs x1 > x0, y1 > y0 and cell > 0")
        ncol = int(round((x1 - x0) / cell)) + 1
        nrow = int(round((y1 - y0) / cell)) + 1
        xs = np.linspace(x0, x1, ncol)
        ys = np.linspace(y0, y1, nrow)
        X, Y = np.meshgrid(xs, ys)
        Z = np.asarray(self.height(X, Y), dtype=float).reshape(nrow, ncol)
        return Z, (x0, x1, y0, y1)

    def spec(self) -> dict:
        """A plain dict ``{'kind': ..., <parameters>}`` that ``terrain_from_spec`` rebuilds this terrain from."""
        return {"kind": self.kind}

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.spec().items() if k != "kind")
        return f"{type(self).__name__}({params})"


class Flat(Terrain):
    """The plane z = 0."""

    kind = "flat"
    is_flat = True

    def _height(self, x, y):
        return np.zeros_like(x)


def _jitter(seed, count, amplitude):
    """Deterministic position jitter, uniform in ±amplitude; the k-th value depends only on (seed, k)."""
    if amplitude == 0 or count == 0:
        return np.zeros(count)
    rng = np.random.default_rng(seed)
    return rng.uniform(-amplitude, amplitude, size=count)


def _half_sine(u, width):
    """Half-sine profile of unit height and base ``width`` centred at u = 0 (zero outside |u| ≤ width/2)."""
    u = np.asarray(u, dtype=float)
    inside = np.abs(u) <= width / 2
    return np.where(inside, np.cos(np.pi * np.where(inside, u, 0.0) / width), 0.0)


class LongitudinalBumps(Terrain):
    """Half-sine ridges across the path (each spans all y), one per ``spacing`` along +x after ``start``.

    Ridge k occupies its own interval ``[start + k·spacing, start + (k+1)·spacing]``: its centre is
    ``start + (k + ½)·spacing + u_k`` with ``u_k`` uniform in ±``jitter`` (from ``seed``), its base is ``width``
    wide and its crest ``height`` high: ``z = height · cos(π (x − x_k) / width)`` for ``|x − x_k| ≤ width/2``.
    Ridges that overlap (large jitter) combine by their maximum. ``end`` bounds the ridged region [m].
    """

    kind = "long_bumps"

    def __init__(self, height, spacing, width, start=0.0, jitter=0.0, seed=0, end=50.0):
        if spacing <= 0 or width <= 0:
            raise ValueError("spacing and width must be positive")
        self.height_m, self.spacing, self.width = float(height), float(spacing), float(width)
        self.start, self.jitter, self.seed, self.end = float(start), float(jitter), int(seed), float(end)
        n = max(0, int(math.ceil((self.end - self.start) / self.spacing)))
        self.centres = self.start + (np.arange(n) + 0.5) * self.spacing + _jitter(self.seed, n, self.jitter)

    def _height(self, x, y):
        z = np.zeros_like(x)
        if self.height_m == 0 or len(self.centres) == 0:
            return z
        k = np.floor((x - self.start) / self.spacing).astype(int)
        reach = int(math.ceil((self.jitter + self.width / 2) / self.spacing))
        for dk in range(-reach, reach + 1):
            kk = k + dk
            ok = (kk >= 0) & (kk < len(self.centres))
            u = np.where(ok, x - self.centres[np.clip(kk, 0, len(self.centres) - 1)], np.inf)
            np.maximum(z, self.height_m * _half_sine(u, self.width), out=z)
        return z

    def spec(self):
        return {"kind": self.kind, "height": self.height_m, "spacing": self.spacing, "width": self.width,
                "start": self.start, "jitter": self.jitter, "seed": self.seed, "end": self.end}


class AlternatingBumps(Terrain):
    """Half-sine bumps under one side of the path, then the other, alternating every ``spacing``.

    Bump k is centred at ``x_k = start + (k + ½)·spacing + u_k`` (``u_k`` uniform in ±``jitter`` from ``seed``)
    and ``y_k = +track_y`` for even k (left, first) and ``−track_y`` for odd k (right). Its profile is a half-sine
    in both directions: ``height · cos(π (x − x_k)/width) · cos(π (y − y_k)/track_width)`` inside a footprint
    ``width`` long and ``track_width`` wide (``track_width`` defaults to ``width``: a round bump). Overlaps
    combine by their maximum.
    """

    kind = "alt_bumps"

    def __init__(self, height, spacing, width, track_y, start=0.0, jitter=0.0, seed=0, track_width=None, end=50.0):
        if spacing <= 0 or width <= 0:
            raise ValueError("spacing and width must be positive")
        self.height_m, self.spacing, self.width = float(height), float(spacing), float(width)
        self.track_y = float(track_y)
        self.track_width = float(width if track_width is None else track_width)
        self.start, self.jitter, self.seed, self.end = float(start), float(jitter), int(seed), float(end)
        n = max(0, int(math.ceil((self.end - self.start) / self.spacing)))
        self.centres = self.start + (np.arange(n) + 0.5) * self.spacing + _jitter(self.seed, n, self.jitter)
        self.sides = np.where(np.arange(n) % 2 == 0, 1.0, -1.0) * self.track_y

    def _height(self, x, y):
        z = np.zeros_like(x)
        if self.height_m == 0 or len(self.centres) == 0:
            return z
        k = np.floor((x - self.start) / self.spacing).astype(int)
        reach = int(math.ceil((self.jitter + self.width / 2) / self.spacing))
        n = len(self.centres)
        for dk in range(-reach, reach + 1):
            kk = k + dk
            ok = (kk >= 0) & (kk < n)
            kc = np.clip(kk, 0, n - 1)
            u = np.where(ok, x - self.centres[kc], np.inf)
            v = y - self.sides[kc]
            np.maximum(z, self.height_m * _half_sine(u, self.width) * _half_sine(v, self.track_width), out=z)
        return z

    def spec(self):
        return {"kind": self.kind, "height": self.height_m, "spacing": self.spacing, "width": self.width,
                "track_y": self.track_y, "track_width": self.track_width, "start": self.start,
                "jitter": self.jitter, "seed": self.seed, "end": self.end}


class CrossSlope(Terrain):
    """The plane tilted about the route (x) axis by ``angle_deg``: z = tan(angle)·y (left side up for angle > 0).

    Flat before ``start``; the tilt ramps in linearly over ``transition`` metres after ``start`` (a warped
    surface ``z = tan(angle)·y·clip((x − start)/transition, 0, 1)``) so the start of the slope is not a step.
    ``transition = 0`` gives an abrupt change (a step whose height grows with |y|). The 0.3 m default is
    Chiron's choice, not a protocol value — set it explicitly in a study.
    """

    kind = "cross_slope"

    def __init__(self, angle_deg, start=0.0, transition=0.3):
        self.angle_deg, self.start, self.transition = float(angle_deg), float(start), float(transition)

    def _height(self, x, y):
        t = math.tan(math.radians(self.angle_deg))
        if self.transition > 0:
            w = np.clip((x - self.start) / self.transition, 0.0, 1.0)
        else:
            w = (x >= self.start).astype(float)
        return t * y * w

    def spec(self):
        return {"kind": self.kind, "angle_deg": self.angle_deg, "start": self.start, "transition": self.transition}


class Steps(Terrain):
    """Steps across the path every ``spacing`` after ``start``.

    Edge k sits at ``start + jitter + k·spacing + u_k`` (``u_k`` uniform in ±``jitter`` from ``seed``, so the
    first edge is never before ``start``). ``mode='alternate'`` (default): up by ``height`` at even edges, back
    down at odd edges (0, h, 0, h, ...); ``mode='stairs'``: up by ``height`` at every edge (a staircase).
    Edges are vertical in the function; a height field samples them over one cell.
    """

    kind = "steps"

    def __init__(self, height, spacing, start=0.0, jitter=0.0, seed=0, mode="alternate", end=50.0):
        if spacing <= 0:
            raise ValueError("spacing must be positive")
        if mode not in ("alternate", "stairs"):
            raise ValueError("mode must be 'alternate' or 'stairs'")
        self.height_m, self.spacing = float(height), float(spacing)
        self.start, self.jitter, self.seed = float(start), float(jitter), int(seed)
        self.mode, self.end = mode, float(end)
        n = max(0, int(math.ceil((self.end - self.start) / self.spacing)))
        self.edges = self.start + self.jitter + np.arange(n) * self.spacing + _jitter(self.seed, n, self.jitter)
        self.edges = np.maximum.accumulate(self.edges) if n else self.edges

    def _height(self, x, y):
        n_passed = np.searchsorted(self.edges, x, side="right")
        if self.mode == "alternate":
            return self.height_m * (n_passed % 2)
        return self.height_m * n_passed

    def spec(self):
        return {"kind": self.kind, "height": self.height_m, "spacing": self.spacing, "start": self.start,
                "jitter": self.jitter, "seed": self.seed, "mode": self.mode, "end": self.end}


class Rough(Terrain):
    """Isotropic Gaussian random field with a Gaussian spectrum, zero mean and RMS height ``rms`` [m].

    Correlation ``C(r) = rms² · exp(−r² / correlation_length²)``. The field is generated once on a grid over
    ``extent = (x0, x1, y0, y1)`` with ``cell`` spacing (default: ``correlation_length/8``, at most 5 mm) by
    filtering white noise from ``seed`` in the Fourier domain, then scaled so its RMS over the grid is exactly
    ``rms``; heights between grid points are bilinear. The surface is z = 0 before ``start`` and outside
    ``extent``; it fades in over ``ramp`` metres after ``start`` (default: one correlation length) so the start
    of the rough region is not a step.
    """

    kind = "rough"

    def __init__(self, rms, correlation_length, start=0.0, seed=0, extent=(-1.0, 5.0, -1.5, 1.5), cell=None,
                 ramp=None):
        if correlation_length <= 0:
            raise ValueError("correlation_length must be positive")
        self.rms, self.correlation_length = float(rms), float(correlation_length)
        self.start, self.seed = float(start), int(seed)
        self.extent = tuple(float(v) for v in extent)
        self.cell = float(cell) if cell is not None else min(0.005, self.correlation_length / 8)
        self.ramp = float(self.correlation_length if ramp is None else ramp)
        self._grid = None

    def _field(self):
        if self._grid is None:
            x0, x1, y0, y1 = self.extent
            nx = int(round((x1 - x0) / self.cell)) + 1
            ny = int(round((y1 - y0) / self.cell)) + 1
            rng = np.random.default_rng(self.seed)
            noise = rng.standard_normal((ny, nx))
            if self.rms == 0:
                self._grid = np.zeros((ny, nx))
            else:
                kx = 2 * np.pi * np.fft.rfftfreq(nx, d=self.cell)
                ky = 2 * np.pi * np.fft.fftfreq(ny, d=self.cell)
                k2 = ky[:, None] ** 2 + kx[None, :] ** 2
                # PSD of exp(-r²/ℓ²) ∝ exp(-k² ℓ² / 4); the filter is its square root.
                filt = np.exp(-k2 * self.correlation_length ** 2 / 8.0)
                field = np.fft.irfft2(np.fft.rfft2(noise) * filt, s=(ny, nx))
                field -= field.mean()
                field *= self.rms / np.sqrt(np.mean(field ** 2))
                self._grid = field
        return self._grid

    def _height(self, x, y):
        g = self._field()
        x0, x1, y0, y1 = self.extent
        ny, nx = g.shape
        fx = (x - x0) / self.cell
        fy = (y - y0) / self.cell
        inside = (fx >= 0) & (fx <= nx - 1) & (fy >= 0) & (fy <= ny - 1)
        fxc = np.clip(fx, 0, nx - 1)
        fyc = np.clip(fy, 0, ny - 1)
        i0 = np.minimum(np.floor(fxc).astype(int), nx - 2)
        j0 = np.minimum(np.floor(fyc).astype(int), ny - 2)
        tx, ty = fxc - i0, fyc - j0
        z = ((1 - tx) * (1 - ty) * g[j0, i0] + tx * (1 - ty) * g[j0, i0 + 1]
             + (1 - tx) * ty * g[j0 + 1, i0] + tx * ty * g[j0 + 1, i0 + 1])
        if self.ramp > 0:
            w = np.clip((x - self.start) / self.ramp, 0.0, 1.0)
        else:
            w = (x >= self.start).astype(float)
        return np.where(inside, z * w, 0.0)

    def spec(self):
        return {"kind": self.kind, "rms": self.rms, "correlation_length": self.correlation_length,
                "start": self.start, "seed": self.seed, "extent": list(self.extent), "cell": self.cell,
                "ramp": self.ramp}


class Custom(Terrain):
    """Any height function ``fn(x, y) -> z`` [m]; vectorised functions are used directly, others are
    wrapped with ``np.vectorize``. Its spec records ``name`` only (it cannot be rebuilt from a dict)."""

    kind = "custom"

    def __init__(self, fn: Callable, name: str = "custom", vectorized: bool = True):
        self.fn, self.name = fn, name
        self._vfn = fn if vectorized else np.vectorize(fn, otypes=[float])

    def _height(self, x, y):
        return self._vfn(x, y)

    def spec(self):
        return {"kind": self.kind, "name": self.name}


TERRAIN_KINDS = {
    "flat": Flat,
    "long_bumps": LongitudinalBumps,
    "longitudinal_bumps": LongitudinalBumps,
    "alt_bumps": AlternatingBumps,
    "alternating_bumps": AlternatingBumps,
    "cross_slope": CrossSlope,
    "steps": Steps,
    "rough": Rough,
}

_META_KEYS = ("kind", "level", "name", "params", "label", "difficulty")


def terrain_from_spec(spec: dict | Terrain) -> Terrain:
    """Build a terrain from ``{'kind': ..., <parameters>}`` (parameters flat or under ``'params'``).

    Bookkeeping keys (``level``, ``label``, ``difficulty``) are ignored: a study keeps its own normalised level
    next to the metric parameters. ``seed`` is passed to kinds that take one. A ``Terrain`` is returned as is.
    """
    if isinstance(spec, Terrain):
        return spec
    spec = dict(spec)
    kind = spec.get("kind")
    if kind not in TERRAIN_KINDS:
        raise ValueError(f"unknown terrain kind {kind!r}; known: {sorted(TERRAIN_KINDS)}")
    cls = TERRAIN_KINDS[kind]
    params = dict(spec.get("params") or {})
    params.update({k: v for k, v in spec.items() if k not in _META_KEYS})
    if cls is Flat:
        return Flat()
    if "seed" in params and params["seed"] is None:
        params.pop("seed")
    if cls is Rough and "extent" in params:
        params["extent"] = tuple(params["extent"])
    if cls is CrossSlope:
        params.pop("seed", None)
    return cls(**params)
