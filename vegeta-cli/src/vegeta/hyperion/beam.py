"""Gaussian beams: the complex beam parameter q, the spot size and wavefront along z, propagation through a
``Train`` by the ABCD law, the mode overlap of two beams. A beam of quality M² is carried as its embedded Gaussian
(Siegman): q transforms as a Gaussian's with the wavelength M²·λ₀/n, and the real spot is read from it."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .elements import Element, Train

C_LIGHT = 299792458.0


def q_from(w: float, R: float, wavelength: float, n: float = 1.0, m2: float = 1.0) -> complex:
    """The q parameter of a beam with spot ``w`` and wavefront radius ``R`` (inf: a waist) at one plane."""
    inv_r = 0.0 if math.isinf(R) else 1.0 / R
    return 1.0 / complex(inv_r, -m2 * wavelength / (math.pi * n * w * w))


def spot_from(q: complex, wavelength: float, n: float = 1.0, m2: float = 1.0) -> tuple[float, float]:
    """(w, R) at the plane of ``q``."""
    inv = 1.0 / q
    w = math.sqrt(-m2 * wavelength / (math.pi * n * inv.imag)) if inv.imag < 0 else math.nan
    R = math.inf if abs(inv.real) < 1e-300 else 1.0 / inv.real
    return w, R


def propagate_q(q: complex, m: np.ndarray) -> complex:
    (a, b), (c, d) = m
    return (a * q + b) / (c * q + d)


@dataclass(frozen=True)
class GaussianBeam:
    """A beam at a plane: vacuum ``wavelength`` [m], waist ``w0`` [m] located ``z0`` [m] from the plane (negative:
    before it), beam quality ``m2`` (1: a TEM₀₀), in a medium of index ``n``."""
    wavelength: float
    w0: float
    z0: float = 0.0
    m2: float = 1.0
    n: float = 1.0
    name: str = ""

    @property
    def rayleigh_range(self) -> float:
        return math.pi * self.n * self.w0 ** 2 / (self.m2 * self.wavelength)

    @property
    def divergence(self) -> float:
        """Far-field half-angle [rad] (1/e² radius)."""
        return self.m2 * self.wavelength / (math.pi * self.n * self.w0)

    @property
    def q(self) -> complex:
        """q at the plane (z = 0)."""
        return complex(-self.z0, self.rayleigh_range)

    def q_at(self, z) -> complex:
        return complex(z - self.z0, self.rayleigh_range)

    def w(self, z):
        """Spot radius at ``z`` (a number or an array) [m]."""
        z = np.asarray(z, dtype=float)
        return self.w0 * np.sqrt(1.0 + ((z - self.z0) / self.rayleigh_range) ** 2)

    def R(self, z):
        """Wavefront radius of curvature at ``z`` (inf at the waist)."""
        z = np.asarray(z, dtype=float)
        dz = z - self.z0
        with np.errstate(divide="ignore"):
            return np.where(dz == 0, np.inf, dz * (1.0 + (self.rayleigh_range / np.where(dz == 0, 1.0, dz)) ** 2))

    def gouy(self, z):
        return np.arctan((np.asarray(z, dtype=float) - self.z0) / self.rayleigh_range)

    def at(self, z: float) -> "GaussianBeam":
        """The same beam described from the plane ``z``."""
        return GaussianBeam(self.wavelength, self.w0, self.z0 - z, self.m2, self.n, self.name)

    @classmethod
    def from_q(cls, q: complex, wavelength: float, n: float = 1.0, m2: float = 1.0, name: str = "") -> "GaussianBeam":
        if q.imag <= 0:
            raise ValueError(f"q = {q} is not a beam (Im q must be positive)")
        w0 = math.sqrt(q.imag * m2 * wavelength / (math.pi * n))
        return cls(wavelength, w0, -q.real, m2, n, name)

    @classmethod
    def from_spot(cls, w: float, R: float, wavelength: float, n: float = 1.0, m2: float = 1.0, name: str = "") -> "GaussianBeam":
        """The beam that has spot ``w`` and wavefront radius ``R`` (inf: a waist) at the plane."""
        return cls.from_q(q_from(w, R, wavelength, n, m2), wavelength, n, m2, name)

    @classmethod
    def from_fibre(cls, wavelength: float, core_radius: float, NA: float, name: str = "fibre") -> "GaussianBeam":
        """The beam leaving a fibre end: the waist is the core radius, the far-field half-angle the numerical aperture,
        and M² = π·r·NA/λ ties them (a multimode pump fibre: M² in the tens; a single-mode fibre: M² ≈ 1)."""
        m2 = max(1.0, math.pi * core_radius * NA / wavelength)
        return cls(wavelength, core_radius, 0.0, m2, 1.0, name)

    def to_dict(self) -> dict:
        return {"wavelength": self.wavelength, "w0": self.w0, "z0": self.z0, "m2": self.m2, "n": self.n, "name": self.name,
                "rayleigh_range": self.rayleigh_range, "divergence": self.divergence}

    def through(self, train: Train | list, plane: str = "t", samples: int = 40) -> "BeamTrace":
        """The beam traced through a train: q after every element, the spot on every surface, a sampled envelope."""
        return trace(self, train if isinstance(train, Train) else Train(train), plane, samples)

    def after(self, train: Train | list, plane: str = "t") -> "GaussianBeam":
        """The beam at the plane after the train."""
        return self.through(train, plane, samples=2).out

    def overlap(self, other: "GaussianBeam") -> float:
        """Power coupling efficiency of two coaxial TEM₀₀ beams at the same plane (both M² = 1; the same
        wavelength and medium): 4 / ((w₁/w₂ + w₂/w₁)² + (π w₁ w₂ / λ)² (1/R₁ − 1/R₂)²)."""
        if abs(self.wavelength - other.wavelength) > 1e-15 or abs(self.n - other.n) > 1e-12:
            raise ValueError("overlap needs the same wavelength and medium")
        w1, w2 = float(self.w(0.0)), float(other.w(0.0))
        r1, r2 = float(self.R(0.0)), float(other.R(0.0))
        i1, i2 = (0.0 if math.isinf(r1) else 1.0 / r1), (0.0 if math.isinf(r2) else 1.0 / r2)
        lam = self.wavelength / self.n
        return 4.0 / ((w1 / w2 + w2 / w1) ** 2 + (math.pi * w1 * w2 / lam) ** 2 * (i1 - i2) ** 2)


@dataclass
class BeamTrace:
    """A beam through a train: ``stations`` (the beam at the plane after each part: z, w, R, n, name), the sampled
    envelope ``z``/``w`` for drawing, ``out`` the beam at the end, ``waists`` every waist inside the train."""
    beam: GaussianBeam
    train: Train
    plane: str
    stations: list[dict] = field(default_factory=list)
    z: np.ndarray = field(default_factory=lambda: np.zeros(0))
    w: np.ndarray = field(default_factory=lambda: np.zeros(0))
    out: GaussianBeam | None = None
    waists: list[dict] = field(default_factory=list)

    def spot_at(self, name: str) -> dict:
        for s in self.stations:
            if s["name"] == name:
                return s
        raise KeyError(name)

    @property
    def length(self) -> float:
        return self.train.length

    def to_dict(self) -> dict:
        return {"plane": self.plane, "beam": self.beam.to_dict(), "stations": self.stations, "waists": self.waists,
                "out": self.out.to_dict() if self.out else None}


def trace(beam: GaussianBeam, train: Train, plane: str = "t", samples: int = 40) -> BeamTrace:
    lam, m2 = beam.wavelength, beam.m2
    train.check(lam)
    if abs(train.n_in(lam) - beam.n) > 1e-9:
        raise ValueError(f"the beam is in n = {beam.n:.4f}, the train starts in n = {train.n_in(lam):.4f}")
    q, n, z = beam.q, beam.n, 0.0
    tr = BeamTrace(beam, train, plane)
    zs, ws = [0.0], [spot_from(q, lam, n, m2)[0]]
    tr.stations.append({"name": "start", "element": "", "z": 0.0, "w": ws[0], "R": spot_from(q, lam, n, m2)[1], "n": n})
    for element, part in train.expand(lam, plane):
        m = part.matrix(lam, plane)
        length = part.length
        if length > 0:                                           # sample the spot inside a space
            zz = np.linspace(0.0, length, max(2, samples))
            b = float(m[0, 1])
            for t in zz[1:]:
                qq = propagate_q(q, np.array([[1.0, b * t / length], [0.0, 1.0]]))
                zs.append(z + t)
                ws.append(spot_from(qq, lam, n, m2)[0])
            # a waist inside this space: q real part crosses zero
            if -q.real > 0 and -q.real < b:
                qw = propagate_q(q, np.array([[1.0, -q.real], [0.0, 1.0]]))
                tr.waists.append({"z": z + (-q.real) * length / b, "w0": spot_from(qw, lam, n, m2)[0], "in": element._label(), "n": n})
        q = propagate_q(q, m)
        n = part.n_out(lam)
        z += length
        w, R = spot_from(q, lam, n, m2)
        tr.stations.append({"name": part._label(), "element": element._label(), "z": z, "w": w, "R": R, "n": n})
        if length == 0:
            zs.append(z)
            ws.append(w)
    tr.z, tr.w = np.array(zs), np.array(ws)
    tr.out = GaussianBeam.from_q(q, lam, n, m2, beam.name) if q.imag > 0 else None
    return tr


def lens_for_waist(beam: GaussianBeam, w0_target: float, distance: float | None = None) -> dict:
    """The thin lens at the beam's plane that focuses it to a waist ``w0_target``: Self's formula on the beam's
    (w, R) there. ``distance`` (where the new waist should be) picks the lens when the target is reachable; else the
    lens for the nearest waist. Returns f, the waist position and the magnification."""
    w, R = float(beam.w(0.0)), float(beam.R(0.0))
    zr_in = beam.rayleigh_range
    z_in = -beam.z0                                              # distance from the input waist to the lens
    lam = beam.m2 * beam.wavelength / beam.n
    # the lens sees q_in; we want q_out with waist w0_target at distance d: 1/q_out = 1/q_in − 1/f
    # minimum attainable waist for a given f: w0' = w0 f / sqrt((z_in − f)² + zr²); solve for f (two roots, pick the real one)
    w0 = beam.w0
    k = (w0 / w0_target) ** 2
    # (z_in − f)² + zr² = k f²  →  (1 − k) f² − 2 z_in f + (z_in² + zr²) = 0
    a, b, c = (1.0 - k), -2.0 * z_in, z_in ** 2 + zr_in ** 2
    disc = b * b - 4 * a * c
    if abs(a) < 1e-15:
        roots = [-c / b]
    elif disc < 0:                                               # the largest waist a lens here can make is the spot on it
        raise ValueError(f"a waist of {w0_target:.3g} m is not reachable from this plane: larger than the spot on the lens ({w:.3g} m)")
    else:
        roots = [(-b + math.sqrt(disc)) / (2 * a), (-b - math.sqrt(disc)) / (2 * a)]
    out = []
    for f in roots:
        if f == 0:
            continue
        d = f + (z_in - f) * f * f / ((z_in - f) ** 2 + zr_in ** 2)
        out.append({"f": f, "waist_distance": d, "w0": w0_target, "magnification": w0_target / w0, "w_lens": w, "R_lens": R})
    if not out:
        raise ValueError("no lens")
    if distance is not None:
        out.sort(key=lambda r: abs(r["waist_distance"] - distance))
    else:
        out.sort(key=lambda r: -r["f"])
    return out[0]
