"""A multirotor rotor's noise, better than steady Gutin tones + Hubbard vortex noise (which put a 5" quad at ~30 dB(A)
at 10 m; real ones are 55-65 dB(A)). Two parts, both from the CFD's thrust and torque and the blade geometry:

1. **Tones — steady + unsteady loading** (Lowson 1965; Lowson & Ollerhead): a rotating point force at 0.8 R whose
   thrust and drag vary around the turn with harmonics T_k, D_k. Harmonic m of the blade-passing frequency:

       p_mB = mBΩ / (2√2 π c r) · | Σ_k i^-(mB-k) J_{mB-k}(mB M sinθ) · (T_k cosθ − (mB−k)/(mB M) · D_k) |

   k = 0 is Gutin's steady term (weak: J_{mB} of a small argument). The k ≠ 0 terms are what one hears from a
   multirotor: low-order Bessel functions radiate well even at low tip Mach. The loading harmonics come from
   - **the arm under the rotor**: the blade loses ``arm_dip`` of its load as it passes over the arm, a Gaussian dip
     in azimuth of the arm's width plus twice its gap below the rotor, at 0.8 R -> T_k = −T0 · dip · σ/(2√π) · exp(−(kσ/2)²) for every k (an impulse:
     a wide comb of harmonics up to k ~ 2/σ);
   - **edgewise flight**: the advancing/retreating asymmetry, a once-per-rev load T_±1 = T0 · asym · μ (μ the advance
     ratio in the disc plane) and a weaker twice-per-rev one;
   - **ingested turbulence / the other rotors' wakes**: a slowly decaying random-phase comb, T_k = T0 · turb · |k|^-2.
2. **Broadband — turbulent boundary layer trailing-edge noise** (Brooks, Pope & Marcolini 1989, NASA RP-1218,
   simplified: tripped flat-plate δ*, both sides, the 1/3-octave shape A around St_peak = 0.02 M^-0.6, mean
   directivity), summed over 8 radial strips of every blade with the local speed (rotation + flight).

The empirical knobs have literature-typical defaults (arm dip 0.3, asym 1.0, turb 0.02) and are in ``NoiseParams``.
Accuracy: about ±5 dB(A) for small multirotors; the trend with rpm, speed and size is the point.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import jv

C0, RHO, NU, PREF = 343.0, 1.225, 1.46e-5, 20e-6
THIRD = 1000 * 2 ** (np.arange(-17, 14) / 3)          # 1/3-octave centres, 20 Hz - 20 kHz


@dataclass(frozen=True)
class NoiseParams:
    arm_width_m: float = 0.010     # what the blade passes over (arm / motor mount), at 0.8 R
    arm_gap_m: float = 0.010       # rotor plane to arm: the dip spreads over width + 2 x gap (potential field)
    arm_dip: float = 0.30          # the load the blade loses over it (fraction)
    asym: float = 1.0              # 1/rev load asymmetry per unit advance ratio
    turb: float = 0.02             # random loading harmonics at k = 1, decaying k^-2
    harmonics: int = 40            # of the blade-passing frequency
    k_max: int = 120               # loading harmonics summed
    strips: int = 8                # broadband radial strips
    seed: int = 1


def loading_harmonics(T0: float, Om: float, R: float, V_inplane: float, p: NoiseParams):
    """Complex thrust-loading harmonics T_k, k = -k_max..k_max (T_0 = T0)."""
    k = np.arange(-p.k_max, p.k_max + 1)
    sig = (p.arm_width_m + 2 * p.arm_gap_m) / (0.8 * R) / 2
    Tk = (-T0 * p.arm_dip * sig / (2 * math.sqrt(math.pi)) * np.exp(-(k * sig / 2) ** 2)).astype(complex)
    mu = V_inplane / max(Om * R, 1e-9)
    for kk, a in ((1, p.asym * mu), (2, 0.3 * p.asym * mu)):
        Tk[k == kk] += T0 * a / 2
        Tk[k == -kk] += T0 * a / 2
    rng = np.random.default_rng(p.seed)
    ka = np.maximum(np.abs(k), 1)
    Tk += T0 * p.turb * ka ** -2.0 * np.exp(1j * rng.uniform(0, 2 * np.pi, len(k))) * (k != 0)
    Tk[k == 0] = T0
    return k, Tk


def tones(prop, T: float, Q: float, rpm: float, distance: float, angle_deg: float, V_inplane: float = 0.0,
          p: NoiseParams = NoiseParams()) -> list[tuple[float, float]]:
    """[(frequency Hz, SPL dB)] of the first ``p.harmonics`` blade-passing harmonics."""
    B, R = prop.blades, prop.radius
    Om = rpm * 2 * math.pi / 60
    re = 0.8 * R
    M = Om * re / C0
    th = math.radians(angle_deg)
    k, Tk = loading_harmonics(T, Om, R, V_inplane, p)
    Dk = Tk * (Q / re) / max(T, 1e-12)                  # drag follows the thrust's fluctuations
    out = []
    for m in range(1, p.harmonics + 1):
        n = m * B - k
        terms = (1j ** (-n)) * jv(n, m * B * M * math.sin(th)) * (Tk * math.cos(th) - n / (m * B * M) * Dk)
        prms = m * B * Om / (2 * math.sqrt(2) * math.pi * C0 * distance) * abs(terms.sum())
        out.append((m * B * rpm / 60, 20 * math.log10(max(prms, 1e-12) / PREF)))
    return out


def _bpm_shape(st_ratio):
    """BPM's spectral shape A (dB) as a function of log10(St / St_peak) (the curve for large Re)."""
    a = np.abs(np.log10(st_ratio))
    return np.where(a < 0.204, np.sqrt(np.maximum(67.552 - 886.788 * a ** 2, 0.0)) - 8.219,
                    np.where(a < 0.244, -32.665 * a + 3.981, -142.795 * a ** 3 + 103.656 * a ** 2 - 57.757 * a + 6.006))


def broadband(prop, rpm: float, distance: float, V: float = 0.0, p: NoiseParams = NoiseParams()) -> np.ndarray:
    """1/3-octave TBL-TE levels (dB) at ``THIRD`` for the whole rotor."""
    r = np.asarray(prop.r); c = np.asarray(prop.chord)
    Om = rpm * 2 * math.pi / 60
    edges = np.linspace(r[0], prop.radius * 0.98, p.strips + 1)
    p2 = np.zeros(len(THIRD))
    for a, b in zip(edges[:-1], edges[1:]):
        rm = (a + b) / 2
        ch = float(np.interp(rm, r, c))
        U = math.hypot(Om * rm, V)
        Mloc = U / C0
        Re = max(U * ch / NU, 1e4)
        dstar = ch * 0.0464 * Re ** -0.2 if Re > 3e5 else ch * 10 ** (3.411 - 1.5397 * math.log10(Re) + 0.1059 * math.log10(Re) ** 2) * 0.6
        st_peak = 0.02 * Mloc ** -0.6
        st = THIRD * dstar / U
        k1 = 128.5 if Re > 8e5 else (-4.31 * math.log10(Re) + 156.3 if Re > 2.47e5 else 133.4) - 4.0
        L = 10 * math.log10(dstar * Mloc ** 5 * (b - a) * 1.0 / distance ** 2) + _bpm_shape(st / st_peak) + k1 - 3
        p2 += 2 * prop.blades * 10 ** (L / 10)          # suction + pressure side, every blade
    return 10 * np.log10(np.maximum(p2, 1e-30))


def rotor(prop, T, Q, rpm, distance, angle_deg, V=0.0, tilt_deg=0.0, p: NoiseParams = NoiseParams()) -> dict:
    vin = V * math.cos(math.radians(tilt_deg))          # the flight speed in the disc plane
    return {"tones": tones(prop, T, Q, rpm, distance, angle_deg, vin, p),
            "broadband": [[float(f), float(L)] for f, L in zip(THIRD, broadband(prop, rpm, distance, V, p))]}


def a_weight(f):
    f2 = np.asarray(f, float) ** 2
    ra = 12194 ** 2 * f2 ** 2 / ((f2 + 20.6 ** 2) * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) * (f2 + 12194 ** 2))
    return 20 * np.log10(np.maximum(ra, 1e-30)) + 2.0


def dba(res) -> float:
    L = [L + float(a_weight(f)) for f, L in res["tones"]] + [L + float(a_weight(f)) for f, L in res["broadband"]]
    return 10 * math.log10(sum(10 ** (x / 10) for x in L))
