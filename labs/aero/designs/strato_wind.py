"""Strato — stratospheric winds over Europe by month, latitude and altitude (Stage V8.10). **Estimate.**

A parametric climatology, not data. It reproduces the features that decide whether a stratospheric airship can hold
station, as reported by reanalysis (ERA5/MERRA-2) studies of HAPS sites and by the standard climatologies of the
middle atmosphere (e.g. CIRA-86, Fleming et al. 1990):

- the tropospheric jet near the tropopause (~10–12 km), strongest in winter (~30–40 m/s monthly mean at 45–50°N),
  decaying above;
- a zonal-mean wind *minimum* in the lower stratosphere around 18–22 km, deepest in summer, where the winter
  westerlies turn into the summer easterlies (the "turnaround" that makes HAPS work);
- winter westerlies that grow with altitude towards the polar vortex (~15–25 m/s at 20 km at 50–60°N, more aloft),
  summer easterlies that grow with altitude above ~20 km;
- day-to-day variability much larger in winter (sudden warmings, vortex excursions) than in summer.

Use: the mean zonal (u, east +) and meridional (v, north +) wind, the day-to-day standard deviation ``sigma`` of each
component, and samples of daily winds from them (Gaussian components around the mean — a Rice distribution of speed).
Replace ``profile`` with ERA5 statistics for a site to make a study site-specific; nothing else changes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _season(month: int) -> float:
    """+1 in January (winter), −1 in July (summer), cosine in between (so April and October are alike: the real
    spring and autumn transitions differ by a few weeks, which this estimate ignores)."""
    return math.cos(2 * math.pi * (month - 1) / 12.0)


@dataclass(frozen=True)
class Climatology:
    jet_altitude: float = 11000.0         # m
    jet_width: float = 3500.0             # m, Gaussian half-width of the jet in altitude
    jet_mean: float = 27.0                # m/s, annual mean jet core speed at 47.5°N
    jet_amplitude: float = 10.0           # m/s, winter excess / summer deficit
    strat_base: float = 16000.0           # m, where the stratospheric (seasonal) wind starts to build
    strat_scale: float = 10000.0          # m, per which the stratospheric wind grows by ``winter`` / ``summer``
    winter_westerly: float = 30.0         # m/s per strat_scale above strat_base, mid-winter at 50°N
    summer_easterly: float = 12.0         # m/s per strat_scale, mid-summer
    meridional_fraction: float = 0.08     # mean v / mean |u| (the mean flow is nearly zonal)
    sigma_winter: float = 9.0             # m/s, day-to-day std of each component at 20 km in mid-winter
    sigma_summer: float = 3.0             # m/s, mid-summer
    sigma_jet: float = 8.0                # m/s, extra variability in the jet
    note: str = "estimate: parametric climatology tuned to the published features above, not ERA5 data"


CLIMATOLOGY = Climatology()


def profile(altitude_m, month: int, lat: float, c: Climatology = CLIMATOLOGY) -> dict:
    """Mean (u, v) [m/s] and day-to-day ``sigma`` [m/s] at ``altitude_m`` (vectorised) in ``month`` (1–12) at
    latitude ``lat`` [°N, 35–65]."""
    if not 1 <= month <= 12:
        raise ValueError("month must be 1–12")
    if not 35.0 <= lat <= 65.0:
        raise ValueError("the climatology covers 35–65°N")
    z = np.asarray(altitude_m, dtype=float)
    s = _season(month)                                     # +1 winter … −1 summer
    winter = 0.5 * (1 + s)
    summer = 0.5 * (1 - s)
    lat_jet = math.exp(-((lat - 47.5) / 12.0) ** 2)        # the jet is strongest at mid-latitudes
    lat_vortex = 0.6 + 0.4 * (lat - 40.0) / 20.0           # the winter vortex strengthens poleward
    jet = (c.jet_mean + c.jet_amplitude * s) * lat_jet * np.exp(-0.5 * ((z - c.jet_altitude) / c.jet_width) ** 2)
    rise = np.clip((z - c.strat_base) / c.strat_scale, 0.0, None)
    strat = (c.winter_westerly * lat_vortex * winter ** 1.5 - c.summer_easterly * summer ** 1.5) * rise
    u = jet + strat
    v = c.meridional_fraction * np.abs(u) * (1 if s >= 0 else -1)
    in_jet = np.exp(-0.5 * ((z - c.jet_altitude) / c.jet_width) ** 2)
    above = np.clip((z - 12000.0) / 8000.0, 0.0, 1.5)
    sigma = (c.sigma_summer + (c.sigma_winter - c.sigma_summer) * winter * lat_vortex) * np.maximum(above, 0.35) \
        + c.sigma_jet * in_jet
    return {"u": u, "v": v, "speed": np.hypot(u, v), "sigma": sigma}


def daily_samples(altitude_m: float, month: int, lat: float, n: int, seed: int = 0) -> np.ndarray:
    """``n`` daily mean wind vectors (u, v) [m/s] at one altitude: Gaussian components around the monthly mean."""
    p = profile(altitude_m, month, lat)
    rng = np.random.default_rng(seed)
    return np.column_stack([p["u"] + p["sigma"] * rng.standard_normal(n), p["v"] + p["sigma"] * rng.standard_normal(n)])


def speed_percentile(altitude_m: float, month: int, lat: float, q: float, n: int = 20000, seed: int = 0) -> float:
    """The ``q``-th percentile (0–100) of the daily wind speed (Monte Carlo over ``daily_samples``)."""
    w = daily_samples(altitude_m, month, lat, n, seed)
    return float(np.percentile(np.hypot(w[:, 0], w[:, 1]), q))


def exceedance(altitude_m: float, month: int, lat: float, speed: float, n: int = 20000, seed: int = 0) -> float:
    """Fraction of days whose wind exceeds ``speed`` [m/s]."""
    w = daily_samples(altitude_m, month, lat, n, seed)
    return float(np.mean(np.hypot(w[:, 0], w[:, 1]) > speed))


def best_altitude(month: int, lat: float, lo: float = 16000.0, hi: float = 24000.0, q: float = 95.0) -> float:
    """The altitude [m] in [lo, hi] with the lowest ``q``-th percentile wind speed (1 km grid, then 100 m)."""
    grid = np.arange(lo, hi + 1, 1000.0)
    k = int(np.argmin([speed_percentile(z, month, lat, q, 4000) for z in grid]))
    fine = np.arange(max(lo, grid[k] - 900), min(hi, grid[k] + 900) + 1, 100.0)
    return float(fine[int(np.argmin([speed_percentile(z, month, lat, q, 4000) for z in fine]))])
