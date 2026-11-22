"""NeuralFoil (Sharpe & Hansman 2024; a neural network trained on XFoil) as a Boreas airfoil and a boundary-layer source.

- ``NeuralFoilAirfoil``: drop-in for ``boreas.Airfoil`` (``coefficients(alpha_rad) -> (cl, cd)``) at one Reynolds number;
  the polar is tabulated once (-25..+25 deg) and interpolated, so the blade-element solver stays fast;
- ``te_dstar``: the displacement thickness at the trailing edge, suction and pressure side (theta x H at the last of
  NeuralFoil's 32 boundary-layer stations), for the BPM trailing-edge noise.

Optional: ``pip install neuralfoil`` (pulls AeroSandbox). Without it ``available()`` is False and the callers keep the
simple polar and the flat-plate boundary layer. NeuralFoil is a small numpy network: it runs on the CPU (a GPU does
not help); a polar is ~0.1 s.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

ALPHA = np.arange(-25.0, 25.01, 0.5)


def available() -> bool:
    try:
        import aerosandbox  # noqa: F401
        import neuralfoil  # noqa: F401
        return True
    except ImportError:
        return False


@lru_cache(maxsize=64)
def _airfoil(section: str):
    import aerosandbox as asb
    return asb.Airfoil(section)


@lru_cache(maxsize=512)
def _polar(section: str, re_round: float, model_size: str):
    import neuralfoil as nf
    r = nf.get_aero_from_airfoil(_airfoil(section), alpha=ALPHA, Re=re_round, model_size=model_size)
    return np.asarray(r["CL"], float), np.asarray(r["CD"], float), np.asarray(r["analysis_confidence"], float)


def _re_key(re):
    return float(f"{max(re, 1e3):.2g}")                      # 2 significant digits: the polars are cached


class NeuralFoilAirfoil:
    """Boreas-compatible airfoil from NeuralFoil at Reynolds number ``re``."""

    def __init__(self, section: str = "naca4409", re: float = 1e5, model_size: str = "large"):
        self.section, self.re, self.model_size = section, _re_key(re), model_size
        self.cl_tab, self.cd_tab, self.conf = _polar(section, self.re, model_size)
        self.name = f"{section} (NeuralFoil {model_size}, Re {self.re:.2g})"
        self.source = "NeuralFoil"
        self.cl_max = float(np.max(self.cl_tab))

    def coefficients(self, alpha_rad):
        a = np.degrees(np.asarray(alpha_rad, float))
        cl = np.interp(a, ALPHA, self.cl_tab)
        cd = np.interp(a, ALPHA, self.cd_tab)
        beyond = np.clip(np.abs(a) - 25.0, 0, None)            # outside the table: flat-plate-like drag growth
        return cl, cd + 0.02 * beyond

    @property
    def confidence(self) -> float:
        return float(np.mean(self.conf[(ALPHA >= -5) & (ALPHA <= 12)]))

    def describe(self) -> dict:
        return {"section": self.section, "re": self.re, "model": self.model_size, "cl_max": round(self.cl_max, 3),
                "confidence": round(self.confidence, 3)}


def te_dstar(section: str, alpha_deg: float, re: float, chord: float, model_size: str = "large") -> tuple[float, float]:
    """(suction-side, pressure-side) displacement thickness at the trailing edge [m]."""
    import neuralfoil as nf
    r = nf.get_aero_from_airfoil(_airfoil(section), alpha=float(np.clip(alpha_deg, -10, 15)), Re=_re_key(re),
                                 model_size=model_size)
    up = float(np.ravel(r["upper_bl_theta_31"])[0] * np.ravel(r["upper_bl_H_31"])[0])
    lo = float(np.ravel(r["lower_bl_theta_31"])[0] * np.ravel(r["lower_bl_H_31"])[0])
    return max(up, 1e-5) * chord, max(lo, 1e-5) * chord


def reynolds(prop, rpm: float, airspeed: float = 0.0, station: float = 0.75, nu: float = 1.46e-5) -> float:
    r = station * prop.radius
    c = float(np.interp(r, np.asarray(prop.r), np.asarray(prop.chord)))
    return math.hypot(rpm * 2 * math.pi / 60 * r, airspeed) * c / nu
