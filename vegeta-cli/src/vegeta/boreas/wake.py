"""A propeller in a non-uniform wake: the wake field, the load harmonics it causes, their tones, and
a slipstream velocity model.

- ``WakeField``: axial wake fraction ``w(r/R, phi)`` at the propeller plane (inflow = ship speed x (1 - w)),
  built uniform, from fin wakes (``fin_wake``) or from samples of a flow solution (``sampled_wake``).
- ``load_harmonics``: quasi-steady blade loads around one revolution (a blade-element solution at the local
  inflow of every angle), their Fourier orders per blade, and what reaches the shaft: thrust and torque at
  orders k·B only, side (bearing) forces at orders k·B ± 1 of the blade loads.
- ``unsteady_tones``: the shaft force harmonics as compact dipoles, the low-frequency tones of a propeller in
  a wake (they usually dominate the steady-loading Gutin tones under water).
- ``slipstream_sampler``: an axisymmetric momentum-theory velocity field (inflow, induced axial velocity and
  swirl from a blade-element solution, contraction by continuity) as a ``points -> (U, valid)`` function —
  the signature the particle movies of ``vegeta.aeromant.movie`` take — for when no CFD field exists yet.

Angles: ``phi`` is measured about +x from +y towards +z (the sense of ``rotation = +1``). Quasi-steady means no
unsteady-aerofoil lag (Sears): the harmonic amplitudes are upper estimates at higher orders.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .airfoil import Airfoil
from .bemt import OperatingPoint, solve
from .noise import P_REF_AIR, P_REF_WATER, Medium
from .propeller import Propeller


@dataclass(frozen=True)
class WakeField:
    """Axial wake fraction on a (radius fraction x angle) grid; ``phi_deg`` spans one revolution evenly."""

    r_frac: np.ndarray
    phi_deg: np.ndarray
    w: np.ndarray
    source: str = ""

    def __post_init__(self):
        r, p, w = (np.asarray(a, float) for a in (self.r_frac, self.phi_deg, self.w))
        if w.shape != (len(r), len(p)):
            raise ValueError(f"w must have shape (len(r_frac), len(phi_deg)) = {(len(r), len(p))}, got {w.shape}")
        if np.any(np.diff(r) <= 0) or np.any(np.diff(p) <= 0) or p[0] < 0 or p[-1] >= 360:
            raise ValueError("r_frac and phi_deg must increase; phi_deg within [0, 360)")
        object.__setattr__(self, "r_frac", r)
        object.__setattr__(self, "phi_deg", p)
        object.__setattr__(self, "w", w)

    def __call__(self, r_frac, phi_deg) -> np.ndarray:
        """Wake fraction at (r_frac, phi_deg), linear in radius (clamped), periodic linear in angle."""
        r, ph = np.broadcast_arrays(np.clip(np.asarray(r_frac, float), self.r_frac[0], self.r_frac[-1]),
                                    np.mod(np.asarray(phi_deg, float), 360.0))
        pp = np.concatenate([self.phi_deg, [self.phi_deg[0] + 360.0]])
        rows = np.array([np.interp(ph.ravel(), pp, np.concatenate([row, row[:1]])) for row in self.w])   # (m, n)
        if len(self.r_frac) == 1:
            return rows[0].reshape(ph.shape)
        j = np.clip(np.searchsorted(self.r_frac, r.ravel(), side="right") - 1, 0, len(self.r_frac) - 2)
        t = (r.ravel() - self.r_frac[j]) / (self.r_frac[j + 1] - self.r_frac[j])
        n = np.arange(rows.shape[1])
        return ((1 - t) * rows[j, n] + t * rows[j + 1, n]).reshape(ph.shape)

    def mean(self, r_frac: float = 0.7) -> float:
        """Circumferential mean wake fraction at ``r_frac``."""
        return float(np.mean(self(r_frac, np.arange(360.0))))

    def harmonics(self, r_frac: float = 0.7, orders: int = 12) -> np.ndarray:
        """Amplitudes of the circumferential Fourier orders 1..``orders`` of w at ``r_frac``."""
        x = self(r_frac, np.arange(360.0))
        X = np.fft.rfft(x) / len(x)
        return 2 * np.abs(X[1:orders + 1])


def uniform_wake(w: float = 0.0) -> WakeField:
    return WakeField(np.array([0.2, 1.0]), np.array([0.0, 180.0]), np.full((2, 2), float(w)), f"uniform w = {w}")


def fin_wake(n_fins: int, mean: float, depth: float, width_deg: float, phase_deg: float = 0.0, *,
             r_frac=(0.2, 0.5, 0.7, 0.9, 1.0), n_phi: int = 360) -> WakeField:
    """A mean wake plus ``n_fins`` Gaussian deficits (peak ``depth``, 1-sigma ``width_deg``) behind fins at
    ``phase_deg + k 360/n_fins``. The same at every radius (``r_frac`` only sets the grid)."""
    if n_fins < 0 or width_deg <= 0:
        raise ValueError("n_fins >= 0 and width_deg > 0")
    phi = np.arange(n_phi) * 360.0 / n_phi
    w = np.full_like(phi, float(mean))
    for k in range(n_fins):
        d = (phi - phase_deg - k * 360.0 / n_fins + 180.0) % 360.0 - 180.0
        w += depth * np.exp(-0.5 * (d / width_deg) ** 2)
    r = np.asarray(r_frac, float)
    return WakeField(r, phi, np.tile(w, (len(r), 1)),
                     f"mean {mean} + {n_fins} fin wakes (depth {depth}, width {width_deg} deg)")


def sampled_wake(sampler, center, radius: float, ship_speed: float, *, r_frac=(0.3, 0.5, 0.7, 0.9, 1.0),
                 n_phi: int = 72, source: str = "sampled flow") -> WakeField:
    """Wake fraction from a velocity field (``sampler(points) -> (U, valid)``, e.g. a hull CFD case read with
    ``vegeta.aeromant.movie.openfoam_sampler``) on the propeller plane ``x = center[0]``. Points the sampler
    marks invalid take the mean of the valid ones at that radius."""
    c = np.asarray(center, float)
    r = np.asarray(r_frac, float)
    phi = np.arange(n_phi) * 360.0 / n_phi
    a = np.radians(phi)
    w = np.empty((len(r), n_phi))
    for i, rf in enumerate(r):
        pts = c + np.column_stack([np.zeros(n_phi), rf * radius * np.cos(a), rf * radius * np.sin(a)])
        u, ok = sampler(pts)
        wi = 1.0 - np.asarray(u, float)[:, 0] / ship_speed
        if not ok.any():
            raise ValueError(f"no valid samples at r/R = {rf}: is the propeller plane inside the flow domain?")
        wi[~ok] = wi[ok].mean()
        w[i] = wi
    return WakeField(r, phi, w, source)


@dataclass
class LoadHarmonics:
    """Quasi-steady loads over one shaft revolution and their Fourier orders (order k at k x shaft rate)."""

    rpm: float
    blades: int
    theta_deg: np.ndarray                   # shaft angle
    blade_thrust: np.ndarray                # one blade, N
    blade_torque: np.ndarray                # one blade, N m
    shaft_thrust: np.ndarray                # all blades, N
    shaft_torque: np.ndarray
    side_force_y: np.ndarray                # bearing force from the blades' tangential forces, N
    side_force_z: np.ndarray
    orders: np.ndarray = field(repr=False)
    amplitudes: dict = field(repr=False)    # name -> amplitude per order (index = order)

    @property
    def shaft_hz(self) -> float:
        return self.rpm / 60.0

    def table(self, max_order: int = 16, threshold: float = 1e-9) -> list[dict]:
        """One row per order up to ``max_order``: frequency and the amplitude of every load."""
        rows = []
        for k in range(1, max_order + 1):
            row = {"order": k, "frequency_hz": k * self.shaft_hz}
            row.update({name: float(a[k]) if abs(a[k]) > threshold else 0.0 for name, a in self.amplitudes.items()})
            rows.append(row)
        return rows


def _amplitudes(x: np.ndarray, scale: float) -> np.ndarray:
    """Mean (index 0) and amplitude of every order; orders that cancel (round-off below 1e-12 of ``scale``) are 0."""
    X = np.fft.rfft(x) / len(x)
    a = 2 * np.abs(X)
    a[0] = X[0].real
    a[1:][a[1:] < 1e-12 * scale] = 0.0
    return a


def load_harmonics(prop: Propeller, airfoil: Airfoil, rpm: float, ship_speed: float, wake: WakeField, rho: float, *,
                   rotation: int = 1, r_frac: float = 0.7, n_angles: int | None = None) -> LoadHarmonics:
    """Blade and shaft load harmonics of ``prop`` at ``rpm`` behind ``wake`` (ship speed [m/s]): a blade-element
    solution at the inflow ``ship_speed (1 - w(r_frac, phi))`` of every angle (quasi-steady), the blade loads
    around the revolution, the shaft sums over the blades, their Fourier amplitudes. ``n_angles`` (a multiple of
    the blade count) defaults to the smallest multiple at or above 360."""
    B = prop.blades
    n_angles = n_angles or B * math.ceil(360 / B)
    if n_angles % B:
        raise ValueError(f"n_angles must be a multiple of the blade count {B}")
    if rotation not in (1, -1):
        raise ValueError("rotation must be +1 or -1")
    phi = np.arange(n_angles) * 360.0 / n_angles                              # spatial angle of a blade
    inflow = ship_speed * (1.0 - wake(r_frac, phi))
    cache: dict = {}
    T1, Q1 = np.empty(n_angles), np.empty(n_angles)
    for i, v in enumerate(inflow):
        key = round(float(v), 9)
        if key not in cache:
            op = solve(prop, airfoil, rpm, max(float(v), 0.0), rho)
            cache[key] = (op.thrust / B, op.torque / B)
        T1[i], Q1[i] = cache[key]
    shift = n_angles // B
    idx = (rotation * np.arange(n_angles)[:, None] + shift * np.arange(B)[None, :]) % n_angles   # (theta, blade)
    phib = np.radians(phi[idx])
    Ft = Q1[idx] / (r_frac * prop.radius)                                      # tangential force per blade
    # the fluid resists the rotation: force on the blade = -rotation * Ft * e_theta, e_theta = (-sin, cos) in (y, z)
    Fy = (rotation * Ft * np.sin(phib)).sum(1)
    Fz = (-rotation * Ft * np.cos(phib)).sum(1)
    blade_T, blade_Q = T1[idx[:, 0]], Q1[idx[:, 0]]
    shaft_T, shaft_Q = T1[idx].sum(1), Q1[idx].sum(1)
    sT = float(np.abs(shaft_T).max()) or 1.0
    sQ = float(np.abs(shaft_Q).max()) or 1.0
    sF = float(np.abs(Ft).max()) or 1.0
    amps = {"blade_thrust_N": _amplitudes(blade_T, sT), "blade_torque_Nm": _amplitudes(blade_Q, sQ),
            "shaft_thrust_N": _amplitudes(shaft_T, sT), "shaft_torque_Nm": _amplitudes(shaft_Q, sQ),
            "side_force_y_N": _amplitudes(Fy, sF), "side_force_z_N": _amplitudes(Fz, sF)}
    theta = np.arange(n_angles) * 360.0 / n_angles
    return LoadHarmonics(rpm, B, theta, blade_T, blade_Q, shaft_T, shaft_Q, Fy, Fz, np.arange(len(amps["shaft_thrust_N"])), amps)


def unsteady_tones(h: LoadHarmonics, distance: float, medium: Medium, angle_deg: float = 0.0, max_order: int = 30) -> list[dict]:
    """Tones of the shaft force harmonics as compact dipoles at ``distance`` [m] and ``angle_deg`` from the
    shaft axis: p = omega F / (4 pi c r) x directivity (cos for thrust, sin for side force). Levels in dB
    re 20 uPa (air) or 1 uPa (water); orders with no force are left out."""
    th = math.radians(angle_deg)
    cos_t, sin_t = (0.0 if abs(v) < 1e-12 else abs(v) for v in (math.cos(th), math.sin(th)))    # exact zeros on and across the axis
    p_ref = P_REF_AIR if medium.name == "air" else P_REF_WATER
    rows = []
    for k in range(1, min(max_order, len(h.orders) - 1) + 1):
        f = k * h.shaft_hz
        Fa = float(h.amplitudes["shaft_thrust_N"][k])
        Fs = float(math.hypot(h.amplitudes["side_force_y_N"][k], h.amplitudes["side_force_z_N"][k]) / math.sqrt(2))
        pa = 2 * math.pi * f * Fa / (4 * math.pi * medium.speed_of_sound * distance) * cos_t
        ps = 2 * math.pi * f * Fs / (4 * math.pi * medium.speed_of_sound * distance) * sin_t
        p_rms = math.hypot(pa, ps) / math.sqrt(2)
        if p_rms <= 0:                                  # no force at this order, or none that radiates in this direction
            continue
        rows.append({"order": k, "frequency_hz": f, "thrust_amplitude_N": Fa, "side_force_amplitude_N": Fs,
                     "p_rms_Pa": p_rms, "spl_db": 20 * math.log10(p_rms / p_ref)})
    return rows


def slipstream_sampler(prop: Propeller, op: OperatingPoint, center=(0.0, 0.0, 0.0), *, rotation: int = 1,
                       hub_frac: float = 0.12, extent: float = 20.0):
    """Velocity field of the propeller's slipstream from a blade-element solution (``op``): free stream
    ``op.airspeed`` along +x, induced axial velocity and swirl growing from 0 far upstream to their disc values
    at the disc and twice that far downstream (smoothly, over about a radius), inside a stream tube that
    contracts by continuity; radial velocity from continuity. A model for pictures and particle movies before
    a CFD field exists, not a flow solution. Returns ``points -> (U (n, 3), valid (n,))``."""
    c = np.asarray(center, float)
    R, V, s = prop.radius, float(op.airspeed), float(rotation)
    rr = np.asarray(op.r, float)
    vi_r = np.asarray(op.induced_velocity, float)
    # swirl velocity at the disc from the annulus angular momentum: dQ = 4 pi r^2 rho (V + vi) u_theta dr
    ut_r = np.asarray(op.dQ_dr, float) / (4 * math.pi * rr ** 2 * op.rho * np.maximum(V + vi_r, 1e-6))
    vi_m = float(np.sum(vi_r * rr) / np.sum(rr))                                # radius-weighted mean induced velocity
    U0 = max(V + vi_m, 1e-6)

    def g(xr):                                                                  # 0 upstream, 1 at the disc, 2 far downstream
        return 1.0 + np.tanh(2.0 * xr)

    def sample(points):
        p = np.atleast_2d(np.asarray(points, float)) - c
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        r = np.hypot(y, z)
        xr = x / R
        gx = g(xr)
        Ux = V + vi_m * gx
        r0 = r * np.sqrt(np.maximum(Ux, 1e-6) / U0)                            # radius at the disc of this stream tube
        inside = r0 <= R
        vi = np.interp(r0, rr, vi_r, left=vi_r[0], right=0.0)
        ut = np.interp(r0, rr, ut_r, left=ut_r[0], right=0.0) * gx * np.where(r > 1e-9, r0 / np.maximum(r, 1e-9), 1.0)
        ux = np.where(inside, V + vi * gx, V)
        dUdx = vi_m * 2.0 / R / np.cosh(2.0 * xr) ** 2
        ur = np.where(inside, -0.5 * r * dUdx, 0.0)                              # contraction; none outside the tube (model)
        ut = np.where(inside, ut, 0.0) * s
        with np.errstate(invalid="ignore", divide="ignore"):
            cy = np.where(r > 1e-12, y / np.maximum(r, 1e-12), 1.0)
            sz = np.where(r > 1e-12, z / np.maximum(r, 1e-12), 0.0)
        U = np.column_stack([ux, ur * cy - ut * sz, ur * sz + ut * cy])
        hub = (r < hub_frac * R) & (np.abs(x) < 0.15 * R)
        valid = ~hub & (np.abs(x) < extent * R) & (r < extent * R)
        return U, valid

    return sample
