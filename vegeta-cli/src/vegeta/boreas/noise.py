"""Propeller noise and cavitation estimates: tonal blade-passing harmonics (Gutin's steady-loading
model), a broadband allowance, sound levels in air or water, and a cavitation-number check.

These are first estimates for comparing designs and operating points, not certification numbers:
Gutin's formula covers the steady thrust and torque loading only (no thickness noise, no unsteady
inflow, no installation effects), the broadband term is an empirical allowance, and cavitation
inception is judged from a section-minimum-pressure coefficient you supply.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .propeller import Propeller

P_REF_AIR = 20e-6      # Pa, sound pressure reference in air
P_REF_WATER = 1e-6     # Pa, reference in water


def spl(p_rms: float, medium: str = "air") -> float:
    """Sound pressure level in dB re 20 uPa (air) or 1 uPa (water)."""
    ref = {"air": P_REF_AIR, "water": P_REF_WATER}[medium]
    return 20 * math.log10(max(p_rms, 1e-30) / ref)


@dataclass(frozen=True)
class Medium:
    name: str
    density: float           # kg/m^3
    speed_of_sound: float    # m/s


AIR = Medium("air", 1.2, 340.0)
SEA_WATER = Medium("water", 1025.0, 1500.0)


def _bessel(order: int, x: np.ndarray) -> np.ndarray:
    try:
        from scipy.special import jv
        return jv(order, x)
    except ImportError:  # pragma: no cover - series fallback, adequate for x up to ~30
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x)
        for k in range(60):
            out += (-1) ** k / (math.factorial(k) * math.gamma(k + order + 1)) * (x / 2) ** (2 * k + order)
        return out


def gutin_harmonics(prop: Propeller, thrust: float, torque: float, rpm: float, distance: float, angle_deg: float,
                    medium: Medium = AIR, harmonics: int = 5, effective_radius: float = 0.8) -> dict:
    """Tonal noise of the steady blade loading (Gutin 1936): rms pressure and level of the first
    ``harmonics`` multiples of the blade-passing frequency at ``distance`` [m] and ``angle_deg`` from
    the rotor axis (0 = on the axis ahead, 90 = in the rotor plane).

    p_m = m B Ω / (2 sqrt2 π c r) * | T cos(theta) - Q c / (Ω R_e^2) | * J_{mB}(m B Ω R_e sin(theta) / c)
    """
    if thrust < 0 or torque < 0 or rpm <= 0 or distance <= 0:
        raise ValueError("thrust, torque >= 0; rpm and distance > 0")
    B, omega, c = prop.blades, rpm * 2 * math.pi / 60, medium.speed_of_sound
    re = effective_radius * prop.radius
    th = math.radians(angle_deg)
    m = np.arange(1, harmonics + 1)
    bpf = B * rpm / 60
    arg = m * B * omega * re * math.sin(th) / c
    amp = m * B * omega / (2 * math.sqrt(2) * math.pi * c * distance) * abs(thrust * math.cos(th) - torque * c / (omega * re ** 2))
    p = np.abs(amp * np.array([float(_bessel(int(mm * B), np.array([a]))[0]) for mm, a in zip(m, arg)]))
    levels = np.array([spl(x, medium.name) for x in p])
    total = spl(math.sqrt(float(np.sum(p ** 2))), medium.name)
    return {"blade_pass_hz": bpf, "harmonic": m.tolist(), "frequency_hz": (m * bpf).tolist(), "p_rms_pa": p.tolist(),
            "spl_db": levels.tolist(), "total_tonal_db": total, "distance_m": distance, "angle_deg": angle_deg,
            "reference": "20 uPa" if medium.name == "air" else "1 uPa"}


def broadband_level(prop: Propeller, thrust: float, rpm: float, distance: float, medium: Medium = AIR,
                    k_db: float | None = None) -> float:
    """Empirical broadband (vortex/trailing-edge) allowance: L = K + 10 log10(A_blade V_tip^6 / distance^2) with
    a conventional K for air (rotor blade area A_blade, tip speed V_tip) — a level to add in power to the tones."""
    vt = rpm * 2 * math.pi / 60 * prop.radius
    r, cch = np.asarray(prop.r), np.asarray(prop.chord)
    a_blade = prop.blades * float(np.trapezoid(cch, r))
    if k_db is None:
        k_db = 10.0 if medium.name == "air" else 60.0    # air: Hubbard-type constant (dB re 20 uPa); water: crude, re 1 uPa
    return k_db + 10 * math.log10(max(a_blade * vt ** 6 / distance ** 2, 1e-30))


def cavitation(prop: Propeller, rpm: float, airspeed: float, depth_m: float, medium: Medium = SEA_WATER, *,
               p_atm: float = 101325.0, p_vapour: float = 2300.0, cp_min: float = -1.0, radial_station: float = 0.7) -> dict:
    """Cavitation check at a blade station: cavitation number sigma = (p_static - p_v) / (0.5 rho V_rel^2)
    against the section's minimum pressure coefficient (``cp_min``, a section property you supply; -1 is a
    generic moderately loaded section). Cavitation is expected when sigma < -cp_min."""
    omega = rpm * 2 * math.pi / 60
    r = radial_station * prop.radius
    v_rel = math.hypot(airspeed, omega * r)
    p_static = p_atm + medium.density * 9.81 * depth_m
    sigma = (p_static - p_vapour) / (0.5 * medium.density * v_rel ** 2)
    return {"radial_station": radial_station, "relative_speed_m_s": v_rel, "static_pressure_pa": p_static,
            "cavitation_number": sigma, "cp_min": cp_min, "margin": sigma + cp_min,
            "cavitates": sigma < -cp_min,
            "rpm_at_inception": rpm * math.sqrt(sigma / (-cp_min)) if cp_min < 0 and sigma > 0 else None}
