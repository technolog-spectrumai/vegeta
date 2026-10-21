"""Mass, centre of gravity and inertia of an assembly of parts: ``og.mass``.

An ``Item`` is a mass at a position [m], optionally with a ``shape`` whose own inertia about its centroid is added
(the shape's axes are the budget's axes). A ``Budget`` sums them: total mass, CG, and the inertia tensor about the
CG [kg m²] (the usual tensor, products negated). Units SI; any right-handed frame — the budget does not care which
way is forward, the caller states it.

The solid shapes' formulas are MuJoCo's (``inertiafromgeom``) so a model built from them and the MuJoCo body built
from the same geoms agree; the thin shapes (``Rod``, ``Plate``, ``Shell``) are the idealisations Vegeta's mass
budgets use. Characterisation: ``tests/test_mass.py`` reproduces Vegeta's ``kestrel_systems.cg_inertia`` and
``nisus_systems.cg_inertia`` on their real tables.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_AXES = {"x": 0, "y": 1, "z": 2}


# ------------------------------------------------------------------------------------------------------------ shapes
@dataclass(frozen=True)
class Rod:
    """A thin rod along ``axis`` ('x', 'y' or 'z'): m L²/12 about the two cross axes."""
    axis: str
    length: float

    def inertia(self, m: float) -> np.ndarray:
        k = m * self.length ** 2 / 12
        return k * (np.eye(3) - np.diag([float(self.axis == a) for a in "xyz"]))


@dataclass(frozen=True)
class Plate:
    """A thin plate of sides ``a`` (along ``axes[0]``) and ``b`` (along ``axes[1]``), normal along the third axis."""
    a: float
    b: float
    axes: str = "xy"

    def inertia(self, m: float) -> np.ndarray:
        i, j = _AXES[self.axes[0]], _AXES[self.axes[1]]
        n = 3 - i - j
        d = np.zeros(3)
        d[i] = m * self.b ** 2 / 12
        d[j] = m * self.a ** 2 / 12
        d[n] = m * (self.a ** 2 + self.b ** 2) / 12
        return np.diag(d)


@dataclass(frozen=True)
class Shell:
    """A thin ellipsoidal shell of semi-axes ``a``, ``b``, ``c`` (x, y, z): m/3 (b²+c², a²+c², a²+b²)."""
    a: float
    b: float
    c: float

    def inertia(self, m: float) -> np.ndarray:
        a, b, c = self.a, self.b, self.c
        return m / 3 * np.diag([b * b + c * c, a * a + c * c, a * a + b * b])


@dataclass(frozen=True)
class Box:
    """A solid box of full sides ``sx``, ``sy``, ``sz``."""
    sx: float
    sy: float
    sz: float

    def inertia(self, m: float) -> np.ndarray:
        x, y, z = self.sx ** 2, self.sy ** 2, self.sz ** 2
        return m / 12 * np.diag([y + z, x + z, x + y])


@dataclass(frozen=True)
class Sphere:
    radius: float

    def inertia(self, m: float) -> np.ndarray:
        return 0.4 * m * self.radius ** 2 * np.eye(3)


@dataclass(frozen=True)
class Cylinder:
    """A solid cylinder of ``radius`` and full ``length`` along ``axis``."""
    radius: float
    length: float
    axis: str = "z"

    def inertia(self, m: float) -> np.ndarray:
        r2, h2 = self.radius ** 2, self.length ** 2
        d = np.full(3, m * (3 * r2 + h2) / 12)
        d[_AXES[self.axis]] = 0.5 * m * r2
        return np.diag(d)


@dataclass(frozen=True)
class Capsule:
    """A solid capsule: a cylinder of ``radius`` and full ``length`` along ``axis`` with a hemisphere on each end
    (MuJoCo's ``inertiafromgeom``: the mass split by volume, the hemispheres moved out by parallel axes)."""
    radius: float
    length: float
    axis: str = "z"

    def inertia(self, m: float) -> np.ndarray:
        r, h = self.radius, self.length
        m_sph = m * 4 * r / (4 * r + 3 * h)                       # the two hemispheres' share of the volume
        m_cyl = m - m_sph
        i_sph = 2 * m_sph * r * r / 5
        d = np.full(3, m_cyl * (3 * r * r + h * h) / 12 + i_sph + m_sph * h * (3 * r + 2 * h) / 8)
        d[_AXES[self.axis]] = m_cyl * r * r / 2 + i_sph
        return np.diag(d)


@dataclass(frozen=True)
class Ellipsoid:
    """A solid ellipsoid of semi-axes ``a``, ``b``, ``c``: m/5 (b²+c², …)."""
    a: float
    b: float
    c: float

    def inertia(self, m: float) -> np.ndarray:
        a, b, c = self.a ** 2, self.b ** 2, self.c ** 2
        return m / 5 * np.diag([b + c, a + c, a + b])


# ------------------------------------------------------------------------------------------------------------ budget
@dataclass(frozen=True)
class Item:
    """One mass [kg] at ``pos`` [m]; ``shape`` adds its own inertia about ``pos``."""
    name: str
    kg: float
    pos: tuple = (0.0, 0.0, 0.0)
    shape: object = None
    group: str = ""
    source: str = ""


@dataclass(frozen=True)
class MassProperties:
    mass: float
    com: np.ndarray
    inertia: np.ndarray                        # about ``com`` [kg m²]

    def shifted(self, point) -> np.ndarray:
        """The inertia about ``point`` (parallel axes)."""
        d = self.com - np.asarray(point, dtype=float)
        return self.inertia + self.mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))

    def __add__(self, other: "MassProperties") -> "MassProperties":
        m = self.mass + other.mass
        com = (self.mass * self.com + other.mass * other.com) / m
        return MassProperties(m, com, self.shifted(com) + other.shifted(com))


@dataclass
class Budget:
    items: list = field(default_factory=list)

    def add(self, *items: Item) -> "Budget":
        self.items.extend(items)
        return self

    @property
    def mass(self) -> float:
        return float(sum(i.kg for i in self.items))

    def properties(self) -> MassProperties:
        """Mass, CG and the inertia about the CG: point masses (parallel-axis tensor) plus each shape's own."""
        m = np.array([i.kg for i in self.items], dtype=float)
        r = np.array([i.pos for i in self.items], dtype=float).reshape(-1, 3)
        M = m.sum()
        cg = (m[:, None] * r).sum(0) / M
        d = r - cg
        inertia = np.zeros((3, 3))
        for mi, di, it in zip(m, d, self.items):
            inertia += mi * (np.dot(di, di) * np.eye(3) - np.outer(di, di))
            if it.shape is not None:
                inertia += it.shape.inertia(mi)
        return MassProperties(float(M), cg, inertia)

    def by_group(self) -> dict:
        out = {}
        for i in self.items:
            out[i.group] = out.get(i.group, 0.0) + i.kg
        return out


def place_mass_for_cg(mass: float, moment: float, kg: float, target: float, bay: tuple | None = None) -> tuple:
    """Where an item of ``kg`` must sit for the CG of an assembly (``mass``, first ``moment`` = Σ m x along the axis,
    any consistent units) plus the item to be at ``target``, clipped to ``bay`` = (lo, hi) if given: returns
    (position, clipped). The algebra every Vegeta mass table repeated for its battery, for tables that keep their own
    sums (lists of dicts, DataFrames)."""
    x = (target * (mass + kg) - moment) / kg
    if bay is None:
        return float(x), False
    lo, hi = bay
    if lo <= x <= hi:
        return float(x), False
    return float(min(max(x, lo), hi)), True


def cg_with(mass: float, moment: float, kg: float, x: float) -> float:
    """The CG of an assembly (``mass``, ``moment``) once an item of ``kg`` sits at ``x``."""
    return (moment + kg * x) / (mass + kg)


def place_for_cg(budget: Budget, kg: float, target: float, bay: tuple, axis: int = 0) -> tuple:
    """Where along ``axis`` an item of ``kg`` must sit for the budget's CG to be at ``target``, clipped to ``bay`` =
    (lo, hi): returns (position, clipped)."""
    moment = float(sum(i.kg * i.pos[axis] for i in budget.items))
    return place_mass_for_cg(budget.mass, moment, kg, target, bay)


def point_inertia(masses, positions) -> np.ndarray:
    """The inertia about the origin of point masses (the tensor, products negated)."""
    out = np.zeros((3, 3))
    for m, p in zip(masses, np.asarray(positions, dtype=float)):
        out += m * (np.dot(p, p) * np.eye(3) - np.outer(p, p))
    return out


def solid_density_mass(volume_m3: float, density: float, fill: float = 1.0) -> float:
    """A printed or machined part's mass from its CAD volume."""
    return volume_m3 * density * fill


__all__ = ["Rod", "Plate", "Shell", "Box", "Sphere", "Cylinder", "Capsule", "Ellipsoid", "Item", "MassProperties", "Budget",
           "place_for_cg", "place_mass_for_cg", "cg_with", "point_inertia", "solid_density_mass"]
