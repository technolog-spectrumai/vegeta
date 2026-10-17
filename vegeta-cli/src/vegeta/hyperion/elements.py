"""Paraxial optical elements as ray-transfer (ABCD) matrices, and a ``Train`` of them.

Conventions: lengths in metres, angles in radians, the ray vector is (height, geometric angle) — so free space in a
medium of index n is [[1, L], [0, 1]] and a flat interface from n₁ to n₂ is [[1, 0], [0, n₁/n₂]] (its determinant
n₁/n₂; a train that starts and ends in the same medium has determinant 1). A radius of curvature is positive when
the centre of curvature lies beyond the surface as the beam travels: a concave mirror has R > 0, a biconvex lens
R₁ > 0 and R₂ < 0. An element tilted by ``angle`` (a folding mirror, a Brewster plate) has different matrices in
the tangential (plane of incidence, ``plane="t"``) and sagittal (``"s"``) planes. Every element knows its physical
``length`` along the axis (for drawing), its optical length (for the cavity's free spectral range) and the media it
starts and ends in; ``parts`` splits a composite (a thick lens, a slab) into the surfaces and spaces a beam trace
samples; ``reverse`` is the element seen from the other side (a linear cavity's return trip)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .materials import Material, get as get_material, index_of

INF = math.inf


def _medium_dict(m) -> Any:
    return m.to_dict() if isinstance(m, Material) else m


def _medium_from(d) -> Any:
    return Material.from_dict(d) if isinstance(d, dict) else d


class Element:
    """What every element answers; the subclasses are frozen dataclasses."""
    name: str = ""

    def matrix(self, wavelength: float | None = None, plane: str = "t") -> np.ndarray:
        raise NotImplementedError

    @property
    def length(self) -> float:
        """Physical length along the axis [m]."""
        return 0.0

    def optical_length(self, wavelength: float | None = None) -> float:
        return 0.0

    def n_in(self, wavelength: float | None = None) -> float:
        return 1.0

    def n_out(self, wavelength: float | None = None) -> float:
        return 1.0

    def parts(self, wavelength: float | None = None, plane: str = "t") -> list["Element"]:
        return [self]

    def reverse(self) -> "Element":
        return self

    @property
    def kind(self) -> str:
        return type(self).__name__

    def to_dict(self) -> dict:
        d = {"kind": self.kind}
        for k, v in self.__dict__.items():
            d[k] = _medium_dict(v) if k in ("n", "n1", "n2", "material") else v
        return d

    def _label(self) -> str:
        return self.name or self.kind


@dataclass(frozen=True)
class FreeSpace(Element):
    """A length of a medium (air by default; ``n`` a number, a Material or its name). ``physical`` is the axis length
    for drawing when it differs from the matrix length (a Brewster plate's effective lengths)."""
    L: float
    n: Any = 1.0
    name: str = ""
    physical: float | None = None

    def matrix(self, wavelength=None, plane="t"):
        return np.array([[1.0, self.L], [0.0, 1.0]])

    @property
    def length(self):
        return self.L if self.physical is None else self.physical

    def optical_length(self, wavelength=None):
        return index_of(self.n, wavelength) * self.length

    def n_in(self, wavelength=None):
        return index_of(self.n, wavelength)

    n_out = n_in


@dataclass(frozen=True)
class ThinLens(Element):
    """A thin lens of focal length ``f`` [m] (negative: diverging)."""
    f: float
    name: str = ""

    def matrix(self, wavelength=None, plane="t"):
        return np.array([[1.0, 0.0], [-1.0 / self.f, 1.0]])


@dataclass(frozen=True)
class ThermalLens(Element):
    """The thermal lens of a pumped crystal: a thin lens of focal length ``f`` [m] at ``pump_W`` of pump power
    (``gain.thermal_lens`` makes one; ``f = inf`` is the cold crystal)."""
    f: float
    pump_W: float = 0.0
    name: str = "thermal lens"

    def matrix(self, wavelength=None, plane="t"):
        return np.array([[1.0, 0.0], [-1.0 / self.f if math.isfinite(self.f) else 0.0, 1.0]])


@dataclass(frozen=True)
class FlatMirror(Element):
    """A flat mirror; ``reflectivity`` (power) enters the cavity's losses, ``angle`` [deg] its drawing only."""
    reflectivity: float = 1.0
    angle: float = 0.0
    name: str = ""

    def matrix(self, wavelength=None, plane="t"):
        return np.eye(2)


@dataclass(frozen=True)
class CurvedMirror(Element):
    """A spherical mirror of radius ``R`` [m] (concave R > 0, convex R < 0) used at an incidence ``angle`` [deg]:
    f = R cos θ / 2 in the tangential plane, R / (2 cos θ) in the sagittal — the astigmatism of a folded cavity."""
    R: float
    angle: float = 0.0
    reflectivity: float = 1.0
    name: str = ""

    def focal_length(self, plane: str = "t") -> float:
        c = math.cos(math.radians(self.angle))
        if plane == "t":
            return self.R * c / 2.0
        if plane == "s":
            return self.R / (2.0 * c)
        raise ValueError("plane is 't' (tangential) or 's' (sagittal)")

    def matrix(self, wavelength=None, plane="t"):
        f = self.focal_length(plane)
        return np.array([[1.0, 0.0], [-1.0 / f if math.isfinite(f) and f != 0 else 0.0, 1.0]])


@dataclass(frozen=True)
class Interface(Element):
    """A spherical refracting surface of radius ``R`` [m] between the media ``n1`` → ``n2`` (R = inf: flat).
    Determinant n₁/n₂, the price of geometric angles."""
    n1: Any
    n2: Any
    R: float = INF
    name: str = ""

    def matrix(self, wavelength=None, plane="t"):
        n1, n2 = index_of(self.n1, wavelength), index_of(self.n2, wavelength)
        c = 0.0 if math.isinf(self.R) else (n1 - n2) / (self.R * n2)
        return np.array([[1.0, 0.0], [c, n1 / n2]])

    def n_in(self, wavelength=None):
        return index_of(self.n1, wavelength)

    def n_out(self, wavelength=None):
        return index_of(self.n2, wavelength)

    def reverse(self):
        return replace(self, n1=self.n2, n2=self.n1, R=-self.R if math.isfinite(self.R) else self.R)


@dataclass(frozen=True)
class Slab(Element):
    """A plane-parallel plate or crystal of ``thickness`` [m] and index ``n`` (a Material or a number) at normal
    incidence: [[1, t/n], [0, 1]] between two flat interfaces; optical length n·t."""
    thickness: float
    n: Any
    name: str = ""

    def matrix(self, wavelength=None, plane="t"):
        return np.array([[1.0, self.thickness / index_of(self.n, wavelength)], [0.0, 1.0]])

    @property
    def length(self):
        return self.thickness

    def optical_length(self, wavelength=None):
        return index_of(self.n, wavelength) * self.thickness

    def parts(self, wavelength=None, plane="t"):
        return [Interface(1.0, self.n, name=f"{self._label()} in"), FreeSpace(self.thickness, self.n, name=self._label()),
                Interface(self.n, 1.0, name=f"{self._label()} out")]


@dataclass(frozen=True)
class BrewsterPlate(Element):
    """A plate (or a Brewster-cut crystal) of ``thickness`` [m] and index ``n`` at Brewster's angle: the effective
    lengths t√(n²+1)/n⁴ (tangential) and t√(n²+1)/n² (sagittal) of Kogelnik's astigmatically compensated cavity;
    the beam's geometric path inside is t√(n²+1)/n, which is its axis length."""
    thickness: float
    n: Any
    name: str = ""

    def _n(self, wavelength):
        return index_of(self.n, wavelength)

    def effective_length(self, wavelength=None, plane="t") -> float:
        n = self._n(wavelength)
        s = math.sqrt(n * n + 1.0)
        if plane == "t":
            return self.thickness * s / n**4
        if plane == "s":
            return self.thickness * s / n**2
        raise ValueError("plane is 't' or 's'")

    def path_length(self, wavelength=None) -> float:
        n = self._n(wavelength)
        return self.thickness * math.sqrt(n * n + 1.0) / n

    def matrix(self, wavelength=None, plane="t"):
        return np.array([[1.0, self.effective_length(wavelength, plane)], [0.0, 1.0]])

    @property
    def length(self):
        # the axis length for drawing: a Material's index taken at 1 µm (the matrix uses the true wavelength)
        n = self.n if isinstance(self.n, (int, float)) else get_material(self.n).n(1e-6)
        return self.thickness * math.sqrt(n * n + 1.0) / n

    def optical_length(self, wavelength=None):
        return self._n(wavelength) * self.path_length(wavelength)

    def parts(self, wavelength=None, plane="t"):
        return [FreeSpace(self.effective_length(wavelength, plane), 1.0, name=self._label(), physical=self.path_length(wavelength))]


@dataclass(frozen=True)
class ThickLens(Element):
    """A real lens: surfaces of radii ``r1`` and ``r2`` [m], ``thickness`` [m] at the axis, made of ``material``
    (a Material, its name or a number). Two interfaces and the glass between them; ``cardinal_points`` gives the
    focal length, the principal planes and the back and front focal lengths from the vertices."""
    r1: float
    r2: float
    thickness: float
    material: Any
    name: str = ""

    def parts(self, wavelength=None, plane="t"):
        lab = self._label()
        return [Interface(1.0, self.material, self.r1, name=f"{lab} front"), FreeSpace(self.thickness, self.material, name=lab),
                Interface(self.material, 1.0, self.r2, name=f"{lab} back")]

    def matrix(self, wavelength=None, plane="t"):
        m = np.eye(2)
        for p in self.parts(wavelength, plane):
            m = p.matrix(wavelength, plane) @ m
        return m

    @property
    def length(self):
        return self.thickness

    def optical_length(self, wavelength=None):
        return index_of(self.material, wavelength) * self.thickness

    def reverse(self):
        return replace(self, r1=-self.r2, r2=-self.r1)

    def focal_length(self, wavelength=None) -> float:
        return -1.0 / self.matrix(wavelength)[1, 0]

    def cardinal_points(self, wavelength=None) -> dict:
        """``f`` (effective focal length), ``bfl``/``ffl`` (from the back/front vertex to the focal points), ``h1``/``h2``
        (the principal planes from the front/back vertex, positive into the lens), ``n`` and the lensmaker's ``f_lensmaker``
        for comparison."""
        (a, _), (c, d) = self.matrix(wavelength)
        n = index_of(self.material, wavelength)
        f = -1.0 / c
        inv = (n - 1.0) * (1.0 / self.r1 - 1.0 / self.r2 + (n - 1.0) * self.thickness / (n * self.r1 * self.r2))
        return {"f": f, "bfl": -a / c, "ffl": -d / c, "h1": (d - 1.0) / c, "h2": -(1.0 - a) / c, "n": n,
                "f_lensmaker": 1.0 / inv if inv else INF}


ELEMENT_KINDS = {k.__name__: k for k in (FreeSpace, ThinLens, ThermalLens, FlatMirror, CurvedMirror, Interface, Slab, BrewsterPlate, ThickLens)}


def element_from_dict(d: dict) -> Element:
    d = dict(d)
    kind = ELEMENT_KINDS[d.pop("kind")]
    for k in ("n", "n1", "n2", "material"):
        if k in d:
            d[k] = _medium_from(d[k])
    for k in ("R", "r1", "r2", "f"):
        if k in d and d[k] is None:
            d[k] = INF
    return kind(**d)


def is_mirror(e: Element) -> bool:
    return isinstance(e, (FlatMirror, CurvedMirror))


@dataclass(frozen=True)
class Train:
    """An ordered train of elements along one axis: the product matrix, each element's position, the optical length.
    The media must join (an element starts in the medium the one before it ends in)."""
    elements: tuple = field(default_factory=tuple)
    name: str = ""

    def __init__(self, elements=(), name: str = ""):
        object.__setattr__(self, "elements", tuple(elements))
        object.__setattr__(self, "name", name)
        for e in self.elements:
            if not isinstance(e, Element):
                raise TypeError(f"not an element: {e!r}")

    def __iter__(self):
        return iter(self.elements)

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, i):
        return self.elements[i]

    def __add__(self, other):
        return Train(self.elements + tuple(other.elements if isinstance(other, Train) else other), self.name)

    def check(self, wavelength: float | None = None) -> None:
        for a, b in zip(self.elements, self.elements[1:]):
            if abs(a.n_out(wavelength) - b.n_in(wavelength)) > 1e-9:
                raise ValueError(f"media do not join between {a._label()} (n = {a.n_out(wavelength):.4f}) and "
                                 f"{b._label()} (n = {b.n_in(wavelength):.4f})")

    def matrix(self, wavelength: float | None = None, plane: str = "t", start: int = 0, stop: int | None = None) -> np.ndarray:
        """The ABCD matrix from the plane before element ``start`` to the plane after element ``stop - 1``."""
        m = np.eye(2)
        for e in self.elements[start:stop]:
            m = e.matrix(wavelength, plane) @ m
        return m

    @property
    def length(self) -> float:
        return sum(e.length for e in self.elements)

    def optical_length(self, wavelength: float | None = None) -> float:
        return sum(e.optical_length(wavelength) for e in self.elements)

    def positions(self) -> list[float]:
        """The axis position of the start of each element (and the end of the train as the last entry)."""
        z, out = 0.0, []
        for e in self.elements:
            out.append(z)
            z += e.length
        out.append(z)
        return out

    def n_in(self, wavelength=None):
        return self.elements[0].n_in(wavelength) if self.elements else 1.0

    def n_out(self, wavelength=None):
        return self.elements[-1].n_out(wavelength) if self.elements else 1.0

    def reverse(self) -> "Train":
        return Train([e.reverse() for e in reversed(self.elements)], self.name)

    def expand(self, wavelength: float | None = None, plane: str = "t") -> list[tuple[Element, Element]]:
        """Every (element, part) a beam trace walks through — composites split into their surfaces and spaces."""
        return [(e, p) for e in self.elements for p in e.parts(wavelength, plane)]

    def to_dict(self) -> dict:
        return {"name": self.name, "elements": [e.to_dict() for e in self.elements]}

    @classmethod
    def from_dict(cls, d: dict) -> "Train":
        return cls([element_from_dict(e) for e in d["elements"]], d.get("name", ""))

    def table(self, wavelength: float | None = None) -> list[dict]:
        """One row per element: its kind, name, position, length, the matrix's entries."""
        rows, pos = [], self.positions()
        for e, z in zip(self.elements, pos):
            (a, b), (c, d) = e.matrix(wavelength)
            rows.append({"element": e._label(), "kind": e.kind, "z": z, "length": e.length, "A": a, "B": b, "C": c, "D": d})
        return rows


def abcd_power(m: np.ndarray) -> float:
    """The dioptric power −C of a matrix (1/f of the equivalent thin lens)."""
    return -float(m[1, 0])
