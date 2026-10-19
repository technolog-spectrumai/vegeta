"""Spur and planetary gears and the drive train from motor to track sprocket (notebook 30, todo 5j.2). Units: N, mm,
MPa, N·m where marked; every formula is the textbook one, named in its docstring.

* **geometry** — the involute flank (``involute_profile``, the same construction as the CAD's pinion), a simple
  planetary stage (sun driving, ring fixed, carrier out: i = 1 + z_r / z_s) with its assembly and neighbour
  conditions (``planetary``);
* **ratio** — ``select_ratio``: the overall reduction that lets a motor on its linear torque–speed line reach the top
  speed on the flat *and* hold the climbing torque at the climbing speed, within its continuous (rated) torque;
* **strength** — Lewis tooth-root bending σ_F = F_t / (b m Y) (Shigley, Y of the 20° full-depth form) and the Hertz
  contact stress σ_H = C_p √(F_t / (b d₁ I)) with I = cos α sin α / 2 · u / (u ± 1) (Shigley/AGMA, + external,
  − internal), allowable values of ``MATERIALS`` (ISO 6336 class values);
* **losses** — per-mesh efficiency and the chain product;
* **sprocket–track mesh** — the polygon (chordal) effect of a z-tooth sprocket and the capstan limit of a belt.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["LEWIS_Y", "MATERIALS", "GearMaterial", "lewis_y", "involute_profile", "Planetary", "planetary",
           "select_ratio", "lewis_bending", "elastic_coefficient", "contact_stress", "stage_loads", "MESH_EFFICIENCY",
           "chain_efficiency", "sprocket_chordal", "capstan", "sn_cycles"]

#: Lewis form factor Y for 20° full-depth teeth (Shigley, Table 14-2), by number of teeth.
LEWIS_Y = {12: 0.245, 13: 0.261, 14: 0.277, 15: 0.290, 16: 0.296, 17: 0.303, 18: 0.309, 19: 0.314, 20: 0.322,
           21: 0.328, 22: 0.331, 24: 0.337, 26: 0.346, 28: 0.353, 30: 0.359, 34: 0.371, 38: 0.384, 43: 0.397,
           50: 0.409, 60: 0.422, 75: 0.435, 100: 0.447, 150: 0.460, 300: 0.472, 400: 0.480}

#: Mesh efficiency of one planetary stage (sun–planet and planet–ring meshes plus the planet bearings), a
#: spur mesh, and the track drive (sprocket–link engagement and belt bending) — catalogue class values.
MESH_EFFICIENCY = {"planetary stage": 0.97, "spur mesh": 0.98, "sprocket and belt": 0.95}


@dataclass(frozen=True)
class GearMaterial:
    """Allowable stresses [MPa]: ``sigma_F_lim`` tooth-root bending endurance (ISO 6336 σ_Flim, the nominal stress
    of a unidirectionally loaded tooth for 3·10⁶ cycles), ``sigma_H_lim`` contact endurance (σ_Hlim, 5·10⁷ cycles);
    ``E`` [MPa], ``nu``; ``b_F`` and ``b_H`` the exponents of the finite-life lines (σ ∝ N^b)."""

    name: str
    sigma_F_lim: float
    sigma_H_lim: float
    E: float = 210000.0
    nu: float = 0.3
    b_F: float = -1 / 8.7          # ISO 6336-3 through-hardened and nitrided: N_L exponent 0.0174…0.115 — a mid value
    b_H: float = -1 / 13.2
    source: str = ""


MATERIALS = {
    "42CrMo4 nitrided": GearMaterial("42CrMo4 nitrided", 350.0, 1000.0, source="ISO 6336-5, nitrided through-hardening steel, MQ class"),
    "42CrMo4 QT": GearMaterial("42CrMo4 QT", 280.0, 700.0, source="ISO 6336-5, through-hardened alloy steel, MQ class"),
    "POM": GearMaterial("POM", 40.0, 60.0, E=2800.0, nu=0.35, source="plastic gear class value (VDI 2736), 23 °C"),
}


def lewis_y(z: int) -> float:
    """Y for ``z`` teeth (linear between the table's entries; the 12-tooth value below 12)."""
    ks = sorted(LEWIS_Y)
    if z <= ks[0]:
        return LEWIS_Y[ks[0]]
    if z >= ks[-1]:
        return LEWIS_Y[ks[-1]]
    return float(np.interp(z, ks, [LEWIS_Y[k] for k in ks]))


# ----------------------------------------------------------------------------------------------- geometry
def involute_profile(m: float, z: int, alpha_deg: float = 20.0, n_points: int = 8) -> list:
    """One tooth's outline [(x, y) mm] of a standard external spur gear (module ``m``, ``z`` teeth), centred on +x,
    from the root circle up one involute flank, across the tip and down the mirrored flank (the flank is the
    involute of the base circle between the base (or root) and the tip circle). The CAD pinion uses the same
    construction (``pekari_rover.PekariRover._tooth_points``)."""
    a = math.radians(alpha_deg)
    r = m * z / 2.0
    rb = r * math.cos(a)
    ra = r + m
    rf = r - 1.25 * m
    inv = lambda t: math.tan(t) - t                                  # noqa: E731
    half = math.pi / (2 * z) + inv(a)                                # half tooth angle at the base circle
    r0 = max(rb, rf)
    flank = []
    for k in range(n_points + 1):
        rr = r0 + (ra - r0) * k / n_points
        t = math.acos(min(1.0, rb / rr))
        th = half - inv(t)
        flank.append((rr * math.cos(th), rr * math.sin(th)))
    root = (rf * math.cos(half), rf * math.sin(half))
    upper = [root] + flank
    lower = [(x, -y) for x, y in reversed(upper)]
    return upper + lower


@dataclass(frozen=True)
class Planetary:
    z_sun: int
    z_planet: int
    z_ring: int
    n_planets: int
    ratio: float
    assembles: bool
    neighbours_clear: bool

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def planetary(z_sun: int, z_ring: int, n_planets: int = 3) -> Planetary:
    """A simple planetary stage, sun in, ring fixed, carrier out: i = 1 + z_r / z_s; the planet has
    (z_r − z_s) / 2 teeth; it assembles with equally spaced planets when (z_s + z_r) / n is a whole number; the
    planets clear each other when (z_s + z_p) sin(π / n) > z_p + 2 (tip circles, in modules)."""
    if (z_ring - z_sun) % 2:
        raise ValueError("z_ring − z_sun must be even (a whole number of planet teeth)")
    zp = (z_ring - z_sun) // 2
    return Planetary(z_sun, zp, z_ring, n_planets, 1.0 + z_ring / z_sun, (z_sun + z_ring) % n_planets == 0,
                     (z_sun + zp) * math.sin(math.pi / n_planets) > zp + 2)


# ----------------------------------------------------------------------------------------------- ratio
def select_ratio(*, stall_Nm: float, rated_Nm: float, no_load_rpm: float, tau_climb: float, omega_climb: float,
                 tau_flat: float, omega_top: float, efficiency: float, ratios=None) -> dict:
    """The feasible overall reductions for one motor (linear torque–speed line τ = τ_s (1 − n / n₀)):

    * top speed — at ``omega_top`` [rad/s] of the sprocket on the flat (load ``tau_flat`` [N·m] at the sprocket)
      the motor must turn at i ω_top while giving τ_flat / (i η): i ω_top ≤ ω₀ (1 − τ_flat / (i η τ_s));
    * climb — at ``omega_climb`` the motor must give τ_climb / (i η) ≤ its line *and* ≤ ``rated_Nm`` (continuous).

    Returns the bounds ``i_min`` (climb), ``i_max`` (top speed) and a table of ``ratios`` (default 3…40) with both
    margins."""
    w0 = no_load_rpm * 2 * math.pi / 60.0
    ratios = np.arange(3.0, 40.01, 0.5) if ratios is None else np.asarray(ratios, dtype=float)
    rows = []
    for i in ratios:
        tm_flat = tau_flat / (i * efficiency)
        w_flat = w0 * max(0.0, 1.0 - tm_flat / stall_Nm)
        v_ok = w_flat / i >= omega_top
        tm_climb = tau_climb / (i * efficiency)
        line = stall_Nm * max(0.0, 1.0 - i * omega_climb / w0)
        c_ok = tm_climb <= min(line, rated_Nm)
        rows.append({"ratio": float(i), "top_speed_rad_s": w_flat / i, "motor_torque_climb_Nm": tm_climb,
                     "climb_margin": min(line, rated_Nm) / tm_climb if tm_climb > 0 else np.inf,
                     "speed_ok": bool(v_ok), "climb_ok": bool(c_ok)})
    ok = [r["ratio"] for r in rows if r["speed_ok"] and r["climb_ok"]]
    return {"rows": rows, "i_min": min(ok) if ok else None, "i_max": max(ok) if ok else None}


# ----------------------------------------------------------------------------------------------- strength
def lewis_bending(Ft: float, m: float, b: float, z: int) -> float:
    """Lewis tooth-root bending stress [MPa]: F_t [N] / (b [mm] · m [mm] · Y(z))."""
    return Ft / (b * m * lewis_y(z))


def elastic_coefficient(E1: float, nu1: float, E2: float, nu2: float) -> float:
    """C_p [√MPa] = √(1 / (π ((1 − ν₁²)/E₁ + (1 − ν₂²)/E₂))) — 191 √MPa for steel on steel."""
    return math.sqrt(1.0 / (math.pi * ((1 - nu1 ** 2) / E1 + (1 - nu2 ** 2) / E2)))


def contact_stress(Ft: float, d1: float, b: float, u: float, *, alpha_deg: float = 20.0, internal: bool = False,
                   Cp: float = 191.0) -> float:
    """Hertz contact stress [MPa] at the pitch point: C_p √(F_t / (b d₁ I)), I = cos α sin α / 2 · u / (u ± 1)
    (``u`` = z₂ / z₁ ≥ 1, − for an internal mesh); ``d1`` the pinion's pitch diameter [mm]."""
    a = math.radians(alpha_deg)
    I = math.cos(a) * math.sin(a) / 2.0 * (u / (u - 1.0) if internal else u / (u + 1.0))
    return Cp * math.sqrt(Ft / (b * d1 * I))


def stage_loads(T_in: float, m: float, z_sun: int, n_planets: int, load_share: float = 1.15) -> dict:
    """Tangential force [N] on one sun–planet mesh of a planetary stage with ``T_in`` [N·m] on the sun: the torque
    shared by the planets with an unequal-sharing factor ``load_share`` (1.15: three planets, floating sun)."""
    r = m * z_sun / 2.0 / 1000.0
    return {"Ft_N": T_in / r / n_planets * load_share, "pitch_radius_m": r}


def chain_efficiency(stages: list) -> float:
    """Product of the ``MESH_EFFICIENCY`` entries named in ``stages``."""
    out = 1.0
    for s in stages:
        out *= MESH_EFFICIENCY[s]
    return out


def sn_cycles(sigma: float, sigma_lim: float, N_lim: float, b: float) -> float:
    """Cycles to failure [-] on a finite-life line through (N_lim, σ_lim) with exponent ``b`` (σ ∝ N^b); infinite
    below σ_lim."""
    if sigma <= sigma_lim:
        return math.inf
    return N_lim * (sigma / sigma_lim) ** (1.0 / b)


# ----------------------------------------------------------------------------------------------- sprocket and belt
def sprocket_chordal(z: int, pitch: float) -> dict:
    """The polygon effect of a ``z``-tooth sprocket driving links of ``pitch`` [mm]: pitch radius
    R = p / (2 sin(π/z)), the belt's speed varies between R cos(π/z) ω and R ω, i.e. by 1 − cos(π/z), ``z`` times per
    revolution."""
    R = pitch / (2.0 * math.sin(math.pi / z))
    return {"pitch_radius_mm": R, "speed_variation": 1.0 - math.cos(math.pi / z), "min_radius_mm": R * math.cos(math.pi / z)}


def capstan(mu: float, wrap_rad: float) -> float:
    """The largest tension ratio T_tight / T_slack a friction drive transmits before slipping: e^(μ θ)."""
    return math.exp(mu * wrap_rad)
