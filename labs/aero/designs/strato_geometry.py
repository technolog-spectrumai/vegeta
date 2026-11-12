"""Strato — the hull of a stratospheric airship as numbers (SI: metres): the profile, its volume and area, and the
sizes of the parts that scale with the hull (Stage V8, ``todo.md``). Pure numpy; ``strato.py`` builds the CAD from
the same profile, so the analytic volume and area can be checked against the solid (tests/test_strato_cad.py).

Profile — a body of revolution in two segments that meet at the maximum radius R = D/2 at x_m = ``x_max`` · L:

- nose (0 ≤ x ≤ x_m): a super-ellipse quarter, r = R (1 − (1 − x/x_m)^n_nose)^(1/n_nose); n_nose = 2 is the ellipse
- tail (x_m ≤ x ≤ L): r = R (1 − ((x − x_m)/(L − x_m))^n_tail); n_tail = 2 is the parabolic tail of the classic
  low-drag hulls (Gertler series 58 family: maximum diameter at 0.4 L, fineness 4–6)

Both segments have zero slope at x_m, so the profile is smooth there. The tail ends in a point; the stern propeller's
hub covers it. Geometry is exact for this profile; whether the profile is a good airship hull is a design choice, not
verified physics.

Frame (shared with the CAD): x along the hull axis from the nose tip (+x aft), z up, y to starboard.
"""
from __future__ import annotations

import math
from functools import cached_property
from dataclasses import dataclass, field, replace

import numpy as np


@dataclass(frozen=True)
class Hull:
    length: float = 15.0              # m
    diameter: float = 3.5             # m, maximum
    x_max: float = 0.40               # station of the maximum diameter / length
    n_nose: float = 2.0               # super-ellipse exponent of the nose (2 = ellipse)
    n_tail: float = 2.0               # tail exponent (2 = parabolic)

    def __post_init__(self):
        problems = self.problems()
        if problems:
            raise ValueError("impossible hull: " + "; ".join(problems))

    def problems(self) -> list[str]:
        out = []
        if self.length <= 0 or self.diameter <= 0:
            out.append("length and diameter must be positive")
        if not 0.2 <= self.x_max <= 0.6:
            out.append(f"x_max = {self.x_max} outside [0.2, 0.6]")
        if self.n_nose < 1.2 or self.n_tail < 1.2:
            out.append("profile exponents below 1.2 give a blunt cusp")
        if self.length > 0 and self.diameter > 0 and not 2.0 <= self.fineness <= 12.0:
            out.append(f"fineness L/D = {self.fineness:.2f} outside [2, 12]")
        return out

    @property
    def fineness(self) -> float:
        return self.length / self.diameter

    @property
    def radius(self) -> float:
        return self.diameter / 2

    def r(self, x):
        """Radius [m] at station ``x`` [m] (vectorised; 0 outside the hull)."""
        x = np.asarray(x, dtype=float)
        L, R, xm = self.length, self.radius, self.x_max * self.length
        out = np.zeros_like(x)
        nose = (x >= 0) & (x <= xm)
        tail = (x > xm) & (x <= L)
        out[nose] = R * (1 - (1 - x[nose] / xm) ** self.n_nose) ** (1 / self.n_nose)
        out[tail] = R * (1 - ((x[tail] - xm) / (L - xm)) ** self.n_tail)
        return out

    def stations(self, n: int = 2001) -> np.ndarray:
        """Stations clustered at the nose and the tail (cosine spacing), where the radius changes fastest."""
        return 0.5 * self.length * (1 - np.cos(np.linspace(0.0, math.pi, n)))

    def profile(self, n: int = 2001) -> tuple[np.ndarray, np.ndarray]:
        x = self.stations(n)
        return x, self.r(x)

    @cached_property
    def volume(self) -> float:
        """Enclosed volume [m³]: ∫ π r² dx."""
        x, r = self.profile()
        return float(np.trapezoid(math.pi * r ** 2, x))

    @cached_property
    def surface_area(self) -> float:
        """Wetted (envelope) area [m²]: ∫ 2π r √(1 + r'²) dx, from the polyline (exact in the limit)."""
        x, r = self.profile()
        ds = np.hypot(np.diff(x), np.diff(r))
        return float(np.sum(math.pi * (r[1:] + r[:-1]) * ds))

    @cached_property
    def centre_of_buoyancy(self) -> float:
        """x of the volume's centroid [m] (the centre of buoyancy on the axis)."""
        x, r = self.profile()
        a = math.pi * r ** 2
        return float(np.trapezoid(a * x, x) / np.trapezoid(a, x))

    @cached_property
    def prismatic_coefficient(self) -> float:
        """V / (π R² L): 1 for a cylinder, 2/3 for an ellipsoid."""
        return self.volume / (math.pi * self.radius ** 2 * self.length)

    def top_area(self, x0: float, x1: float, half_angle_deg: float) -> float:
        """Envelope area [m²] between stations ``x0`` and ``x1`` within ±``half_angle_deg`` of the top (the solar
        array's footprint on the curved hull)."""
        x = np.linspace(x0, x1, 2001)
        r = self.r(x)
        ds = np.hypot(np.diff(x), np.diff(r))
        return float(np.sum(0.5 * (r[1:] + r[:-1]) * ds) * 2 * math.radians(half_angle_deg))

    def scaled(self, length: float, fineness: float | None = None) -> "Hull":
        """The same profile at another length; the fineness is kept unless given."""
        f = self.fineness if fineness is None else fineness
        return replace(self, length=length, diameter=length / f)


@dataclass(frozen=True)
class Layout:
    """Where the parts sit and how big they are, as fractions of the hull (so a 15 m and an 80 m ship are the same
    drawing at different scales). Starting values for the CAD, not sized by any physics yet: fin area, array area,
    gondola and propellers are resized by the Stage V8 studies."""
    # fins: four in an X (45° from vertical, clear of the ground on landing and of the array on top)
    fin_count: int = 4
    fin_roll_deg: float = 45.0
    fin_root_chord: float = 0.14      # / L
    fin_tip_chord: float = 0.07       # / L
    fin_span: float = 0.55            # / D, outward from the hull surface at the root's mid-chord
    fin_sweep_deg: float = 35.0       # leading edge
    fin_thickness: float = 0.12       # / root chord (thick inflated or foam-core fins)
    fin_trailing_edge: float = 0.86   # station of the root trailing edge / L
    # solar array: conformal on the top of the hull
    solar_x0: float = 0.18            # / L
    solar_x1: float = 0.72            # / L
    solar_half_angle_deg: float = 55.0
    # gondola (payload bay, batteries, avionics) under the hull at the centre of buoyancy
    gondola_length: float = 0.16      # / L
    gondola_width: float = 0.16       # / D
    gondola_height: float = 0.13      # / D
    gondola_drop: float = 0.05        # gap under the hull / D
    battery_fraction: float = 0.45    # of the gondola length, the forward part
    # propulsion: one stern pusher on the axis and two vectoring propellers on outriggers at the CB
    stern_prop_diameter: float = 0.38  # / D
    side_prop_count: int = 2
    side_prop_diameter: float = 0.22   # / D
    side_prop_station: float = 0.42    # / L
    side_prop_outboard: float = 0.30   # outrigger reach beyond the hull radius / D
    # structure: nose cone battens (stiffeners that take the dynamic pressure) and a bottom keel
    batten_count: int = 12
    batten_length: float = 0.10       # / L
    keel_x0: float = 0.25             # / L
    keel_x1: float = 0.62             # / L


@dataclass(frozen=True)
class Strato:
    hull: Hull = field(default_factory=Hull)
    layout: Layout = field(default_factory=Layout)

    def scaled(self, length: float, fineness: float | None = None) -> "Strato":
        return replace(self, hull=self.hull.scaled(length, fineness))

    @cached_property
    def solar_area(self) -> float:
        h, lay = self.hull, self.layout
        return h.top_area(lay.solar_x0 * h.length, lay.solar_x1 * h.length, lay.solar_half_angle_deg)

    @property
    def fin_planform_area(self) -> float:
        """One fin's exposed planform area [m²] (trapezoid)."""
        h, lay = self.hull, self.layout
        return 0.5 * (lay.fin_root_chord + lay.fin_tip_chord) * h.length * lay.fin_span * h.diameter

    def facets(self, n_x: int = 40, n_phi: int = 36) -> dict:
        """The envelope as ``n_x`` × ``n_phi`` flat facets: centre (body frame), outward unit normal (body frame),
        area [m²], and the fraction of the facet the solar array covers. Body frame: x aft from the nose tip, y starboard,
        z up; φ is measured from the top towards starboard."""
        h, lay = self.hull, self.layout
        xe = h.stations(n_x + 1)
        xc = 0.5 * (xe[1:] + xe[:-1])
        r_e = h.r(xe)
        ds = np.hypot(np.diff(xe), np.diff(r_e))
        rc = 0.5 * (r_e[1:] + r_e[:-1])
        drdx = np.diff(r_e) / np.diff(xe)
        phi = (np.arange(n_phi) + 0.5) * 2 * math.pi / n_phi
        X, PHI = np.meshgrid(xc, phi, indexing="ij")
        R = np.broadcast_to(rc[:, None], X.shape)
        slope = np.broadcast_to(drdx[:, None], X.shape)
        centre = np.stack([X, R * np.sin(PHI), R * np.cos(PHI)], -1).reshape(-1, 3)
        normal = np.stack([-slope, np.sin(PHI), np.cos(PHI)], -1)
        normal = (normal / np.linalg.norm(normal, axis=-1, keepdims=True)).reshape(-1, 3)
        area = (np.broadcast_to((rc * ds)[:, None], X.shape) * (2 * math.pi / n_phi)).reshape(-1)
        # the array's share of each facet (fractional at its edges, so the array area is exact to the facet sums)
        w_phi = 360.0 / n_phi
        dphi = np.degrees(np.minimum(PHI, 2 * math.pi - PHI))
        f_phi = np.clip((lay.solar_half_angle_deg - (dphi - 0.5 * w_phi)) / w_phi, 0.0, 1.0)
        x0, x1 = lay.solar_x0 * h.length, lay.solar_x1 * h.length
        f_x = np.clip((np.minimum(xe[1:], x1) - np.maximum(xe[:-1], x0)) / np.diff(xe), 0.0, 1.0)
        array = (f_phi * f_x[:, None]).reshape(-1)
        return {"centre": centre, "normal": normal, "area": area, "array": array}

    def summary(self) -> dict:
        h = self.hull
        return {"length_m": h.length, "diameter_m": h.diameter, "fineness": h.fineness, "volume_m3": h.volume,
                "envelope_area_m2": h.surface_area, "centre_of_buoyancy_m": h.centre_of_buoyancy,
                "prismatic_coefficient": h.prismatic_coefficient, "solar_area_m2": self.solar_area,
                "fin_area_m2": self.fin_planform_area * self.layout.fin_count}


def body_to_enu(heading_deg: float) -> np.ndarray:
    """Rotation matrix from the body frame (x aft, y starboard, z up) to east-north-up for a ship whose nose points
    to ``heading_deg`` (from north, clockwise), level."""
    p = math.radians(heading_deg)
    nose = np.array([math.sin(p), math.cos(p), 0.0])
    starboard = np.array([math.cos(p), -math.sin(p), 0.0])
    return np.column_stack([-nose, starboard, [0.0, 0.0, 1.0]])


DEMONSTRATOR = Strato()                                     # 15 m × 3.5 m
REFERENCE_LENGTHS = (15.0, 30.0, 50.0, 80.0)                 # the designs Stage V8 compares


def family(lengths=REFERENCE_LENGTHS, fineness: float | None = None) -> dict[float, Strato]:
    """The demonstrator's drawing at each length (fineness kept unless given)."""
    return {L: DEMONSTRATOR.scaled(L, fineness) for L in lengths}
