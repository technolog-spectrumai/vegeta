"""Ffowcs Williams–Hawkings acoustic analogy, Farassat's formulation 1A (time domain, subsonic surfaces).

The sound at observer ``x`` and time ``t`` from a surface ``f = 0`` (solid — the body — or permeable — any closed
surface around the sources) is the sum of the thickness and loading terms, evaluated at the retarded (emission)
time τ of every surface element (Brentner & Farassat 2003)::

    4π p_T = ∫ [ρ0 (U̇_n + U_ṅ) / (r (1−M_r)²)]
           + ∫ [ρ0 U_n (r Ṁ_r + c (M_r − M²)) / (r² (1−M_r)³)]                              dS
    4π p_L = ∫ [L̇_r / (c r (1−M_r)²)] + ∫ [(L_r − L_M) / (r² (1−M_r)²)]
           + ∫ [L_r (r Ṁ_r + c (M_r − M²)) / (c r² (1−M_r)³)]                                 dS

with ``M = v/c`` the surface Mach vector, ``r`` from source to observer, ``U_i = (1 − ρ/ρ0) v_i + ρ u_i/ρ0`` and
``L_i = p' n_i + ρ u_i (u_n − v_n)`` (inviscid; on a solid surface ``u = v``, so ``U = v`` and ``L = p' n``).
Dots are derivatives in source time. Compact point forces are the loading term with ``∫ L dS → F`` (the force
the body exerts **on the fluid**). The quadrupole (volume) term is not included: use a permeable surface that
encloses the nonlinear region when it matters.

Algorithm: source-time dominant (Brentner) — every element's contribution is computed at its own source times,
placed at its arrival time ``t = τ + r/c``, interpolated onto one observer time grid and summed. Sources and
observers may share a uniform translation ``V`` (a propeller in flight, or — equivalently — a fixed propeller
and fixed microphones in a wind tunnel with stream ``−V``): retarded times then come from ``|d + V s| = c s``.
Units are whatever is consistent (SI, or lattice units with ``c = 1/√3``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SourceSurface:
    """A sampled surface, or a set of point forces, over uniform source times ``tau[nt]``.

    Arrays are ``[nt, ne, 3]`` (vectors) or ``[nt, ne]`` (scalars) for ``ne`` elements. ``y`` positions and ``v``
    velocities are relative to the translating frame (the translation ``V`` is given to :func:`fwh`), ``n`` unit
    normals into the fluid, ``dS[ne]`` areas. ``kind`` selects the data:

    * ``"solid"``: gauge pressure ``p`` on an impenetrable surface (``U = v + V``, ``L = p n``);
    * ``"permeable"``: ``rho``, ``u`` (medium-frame velocity) and gauge ``p`` on any closed surface;
    * ``"forces"``: compact forces ``force`` exerted on the fluid at ``y`` (loading only; ``dS`` = 1).

    Optional exact derivatives ``vdot`` and ``ndot`` replace finite differences in source time.
    """

    kind: str
    tau: np.ndarray
    y: np.ndarray
    v: np.ndarray
    dS: np.ndarray
    n: np.ndarray | None = None
    p: np.ndarray | None = None
    rho: np.ndarray | None = None
    u: np.ndarray | None = None
    force: np.ndarray | None = None
    vdot: np.ndarray | None = None
    ndot: np.ndarray | None = None

    def ddt(self, a: np.ndarray) -> np.ndarray:
        return np.gradient(a, self.tau, axis=0, edge_order=2)

    def acceleration(self) -> np.ndarray:
        return self.vdot if self.vdot is not None else self.ddt(self.v)

    def terms(self, V: np.ndarray, rho0: float):
        """``U_n``, ``U̇_n + U_ṅ`` (both ``None`` without a thickness term), ``L`` and ``L̇``."""
        if self.kind == "forces":
            L = np.asarray(self.force, dtype=np.float64)
            return None, None, L, self.ddt(L)
        n = self.n
        ndot = self.ndot if self.ndot is not None else self.ddt(n)
        vt = self.v + V
        p = np.asarray(self.p, dtype=np.float64)[..., None]
        if self.kind == "solid":
            U, Udot = vt, self.acceleration()
            L = p * n
        elif self.kind == "permeable":
            rr = np.asarray(self.rho, dtype=np.float64)[..., None] / rho0
            U = (1.0 - rr) * vt + rr * self.u
            Udot = self.ddt(U)
            un = (self.u * n).sum(-1, keepdims=True)
            vn = (vt * n).sum(-1, keepdims=True)
            L = p * n + rr * rho0 * self.u * (un - vn)
        else:
            raise ValueError(f"unknown surface kind {self.kind!r}")
        Un = (U * n).sum(-1)
        Udot_n = (Udot * n).sum(-1) + (U * ndot).sum(-1)
        return Un, Udot_n, L, self.ddt(L)


def _retarded(d: np.ndarray, V: np.ndarray, c: float) -> np.ndarray:
    """Propagation time ``s`` with ``|d + V s| = c s`` (``d``: observer minus source in the translating frame)."""
    v2 = float(V @ V)
    if v2 == 0.0:
        return np.linalg.norm(d, axis=-1) / c
    dv = d @ V
    return (dv + np.sqrt(dv * dv + (c * c - v2) * (d * d).sum(-1))) / (c * c - v2)


def fwh(surface: SourceSurface, observers, c: float, rho0: float, translation=(0.0, 0.0, 0.0),
        t_obs: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Acoustic pressure at ``observers[no, 3]`` (fixed in the translating frame).

    Returns ``t[nt_obs]`` and ``thickness``, ``loading``, ``total`` ``[no, nt_obs]``. The observer time grid
    defaults to the window every element covers, with the source-time step.
    """
    V = np.asarray(translation, dtype=np.float64)
    if V @ V >= c * c:
        raise ValueError("supersonic translation is not supported")
    obs = np.atleast_2d(np.asarray(observers, dtype=np.float64))
    tau = np.asarray(surface.tau, dtype=np.float64)
    M = (surface.v + V) / c                               # Mach vector in the medium frame
    Mdot = surface.acceleration() / c
    M2 = (M * M).sum(-1)
    Un, Udot_n, L, Ldot = surface.terms(V, rho0)
    arrivals, pts, pls = [], [], []
    for x in obs:
        d = x[None, None, :] - surface.y                  # [nt, ne, 3] in the translating frame
        s = _retarded(d, V, c)
        rvec = d + V * s[..., None]
        r = c * s
        rhat = rvec / r[..., None]
        Mr = (M * rhat).sum(-1)
        if np.any(Mr >= 1.0):
            raise ValueError("a source element moves supersonically towards the observer")
        D = 1.0 - Mr
        common = r * (Mdot * rhat).sum(-1) + c * (Mr - M2)
        pt = np.zeros_like(r)
        if Un is not None:
            pt = rho0 * Udot_n / (r * D ** 2) + rho0 * Un * common / (r * r * D ** 3)
        Lr = (L * rhat).sum(-1)
        pl = ((Ldot * rhat).sum(-1) / (c * r * D ** 2) + (Lr - (L * M).sum(-1)) / (r * r * D ** 2)
              + Lr * common / (c * r * r * D ** 3))
        arrivals.append(tau[:, None] + s)
        pts.append(pt * surface.dS / (4 * np.pi))
        pls.append(pl * surface.dS / (4 * np.pi))
    if t_obs is None:
        t0 = max(float(a[0].max()) for a in arrivals)
        t1 = min(float(a[-1].min()) for a in arrivals)
        if not t1 > t0:
            raise ValueError("the source record is shorter than the spread of propagation times")
        t_obs = np.arange(t0, t1, tau[1] - tau[0])
    thick = np.zeros((len(obs), len(t_obs)))
    load = np.zeros_like(thick)
    for o in range(len(obs)):
        a, pt, pl = arrivals[o], pts[o], pls[o]
        for e in range(a.shape[1]):
            thick[o] += np.interp(t_obs, a[:, e], pt[:, e])
            load[o] += np.interp(t_obs, a[:, e], pl[:, e])
    return {"t": t_obs, "thickness": thick, "loading": load, "total": thick + load}


def harmonics(t: np.ndarray, p: np.ndarray, frequency: float, count: int = 5) -> np.ndarray:
    """RMS amplitude of the first ``count`` multiples of ``frequency`` in ``p[..., nt]`` (uniform ``t``), from the
    longest whole number of periods in the record (a discrete Fourier projection, no windowing)."""
    dt = t[1] - t[0]
    n = int(np.floor((t[-1] - t[0] + dt) * frequency) / frequency / dt + 1e-6)
    if n < 2:
        raise ValueError("the record is shorter than one period")
    tt, pp = t[:n] - t[0], p[..., :n]
    out = [np.sqrt(2.0) * np.abs((pp * np.exp(-2j * np.pi * m * frequency * tt)).sum(-1)) / n
           for m in range(1, count + 1)]
    return np.stack(out, axis=-1)


def gutin(thrust: float, torque: float, omega: float, blades: int, effective_radius: float, distance: float,
          angle: float, c: float, harmonics_count: int = 5) -> np.ndarray:
    """Gutin's (1936) far-field steady-loading tones, RMS, for comparison:
    ``p_m = m B Ω / (2√2 π c r) |T cos θ − Q c / (Ω R_e²)| J_mB(m B Ω R_e sin θ / c)`` (θ from the axis, 0 =
    upstream on the axis). The same formula as ``vegeta.boreas.noise.gutin_harmonics``, kept here so Mancha has no
    dependency on Vegeta."""
    from math import factorial, gamma

    def jv(order: int, x: float) -> float:
        return sum((-1) ** k / (factorial(k) * gamma(k + order + 1)) * (x / 2) ** (2 * k + order) for k in range(80))

    out = []
    for m in range(1, harmonics_count + 1):
        amp = m * blades * omega / (2 * np.sqrt(2) * np.pi * c * distance)
        load = abs(thrust * np.cos(angle) - torque * c / (omega * effective_radius ** 2))
        out.append(amp * load * abs(jv(m * blades, m * blades * omega * effective_radius * np.sin(angle) / c)))
    return np.array(out)
