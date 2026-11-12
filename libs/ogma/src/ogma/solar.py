"""og.solar — where the sun is, and how much of it reaches a surface high in the atmosphere.

- **Position** (verified): the NOAA Global Monitoring Laboratory solar calculator's algorithm (after Meeus,
  *Astronomical Algorithms*): declination, equation of time, hour angle, elevation and azimuth to ~0.01° for
  1950–2050 (no refraction). Azimuth is from north, clockwise.
- **Irradiance** (estimate): the total solar irradiance 1361 W/m² (Kopp & Lean 2011) scaled by the Earth–Sun distance,
  attenuated by the Meinel & Meinel (1976) clear-sky relation I = I0 · 0.7^(AM′^0.678) with the air mass AM′ =
  AM(z) · p/p0 (Kasten & Young 1989 air mass, scaled by the pressure above the vehicle — the usual altitude correction).
  At 20 km the air above is 5.5 % of the column and the beam is ~95 % of I0 at noon. Diffuse sky light above the
  tropopause is small; reflected light from clouds and ground (albedo) reaches surfaces that face down. The constants
  are documented assumptions, not measurements of a site.
- **Incidence**: cosine of the angle between the sun and any set of surface normals (east-north-up), so a curved
  array (a hull top) is a set of facets.

Times are UTC ``datetime``; longitudes east-positive, latitudes north-positive, degrees.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np

TSI = 1361.0                      # W/m², total solar irradiance at 1 AU (Kopp & Lean 2011)
P0 = 101325.0
R_EARTH_MEAN = 6371000.0


def julian_day(t: datetime) -> float:
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    t = t.astimezone(timezone.utc)
    return (t - datetime(2000, 1, 1, 12, tzinfo=timezone.utc)).total_seconds() / 86400.0 + 2451545.0


def _sun(t: datetime) -> dict:
    jc = (julian_day(t) - 2451545.0) / 36525.0
    L0 = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360.0
    M = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    e = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    Mr = math.radians(M)
    C = (math.sin(Mr) * (1.914602 - jc * (0.004817 + 0.000014 * jc)) + math.sin(2 * Mr) * (0.019993 - 0.000101 * jc)
         + math.sin(3 * Mr) * 0.000289)
    true_long, true_anom = L0 + C, M + C
    R = 1.000001018 * (1 - e * e) / (1 + e * math.cos(math.radians(true_anom)))
    omega = 125.04 - 1934.136 * jc
    app_long = true_long - 0.00569 - 0.00478 * math.sin(math.radians(omega))
    eps0 = 23 + (26 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60) / 60
    eps = eps0 + 0.00256 * math.cos(math.radians(omega))
    decl = math.degrees(math.asin(math.sin(math.radians(eps)) * math.sin(math.radians(app_long))))
    y = math.tan(math.radians(eps / 2)) ** 2
    L0r = math.radians(L0)
    eot = 4 * math.degrees(y * math.sin(2 * L0r) - 2 * e * math.sin(Mr) + 4 * e * y * math.sin(Mr) * math.cos(2 * L0r)
                           - 0.5 * y * y * math.sin(4 * L0r) - 1.25 * e * e * math.sin(2 * Mr))
    return {"declination_deg": decl, "equation_of_time_min": eot, "distance_au": R}


def declination(t: datetime) -> float:
    return _sun(t)["declination_deg"]


def equation_of_time(t: datetime) -> float:
    """Minutes (apparent minus mean solar time)."""
    return _sun(t)["equation_of_time_min"]


def distance_au(t: datetime) -> float:
    return _sun(t)["distance_au"]


def position(t: datetime, lat: float, lon: float) -> tuple[float, float]:
    """(elevation, azimuth) [deg] of the sun's centre seen from (``lat``, ``lon``) at ``t`` (geometric, no refraction)."""
    s = _sun(t)
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    tu = t.astimezone(timezone.utc)
    minutes = tu.hour * 60 + tu.minute + tu.second / 60 + tu.microsecond / 6e7
    tst = (minutes + s["equation_of_time_min"] + 4 * lon) % 1440.0
    H = math.radians(tst / 4 - 180.0)
    phi, d = math.radians(lat), math.radians(s["declination_deg"])
    cosz = math.sin(phi) * math.sin(d) + math.cos(phi) * math.cos(d) * math.cos(H)
    z = math.acos(max(-1.0, min(1.0, cosz)))
    az = math.degrees(math.atan2(math.sin(H), math.cos(H) * math.sin(phi) - math.tan(d) * math.cos(phi))) + 180.0
    return 90.0 - math.degrees(z), az % 360.0


def sun_vector(elevation_deg: float, azimuth_deg: float) -> np.ndarray:
    """Unit vector towards the sun, east-north-up."""
    e, a = math.radians(elevation_deg), math.radians(azimuth_deg)
    return np.array([math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e)])


def horizon_dip(altitude_m: float) -> float:
    """Degrees below the horizontal the geometric horizon lies when seen from ``altitude_m``: acos(R/(R+h))."""
    return math.degrees(math.acos(R_EARTH_MEAN / (R_EARTH_MEAN + max(altitude_m, 0.0))))


def air_mass(zenith_deg: float) -> float:
    """Kasten & Young (1989) relative optical air mass (1 at the zenith, ~38 at the horizon); clamped at 90°."""
    z = min(zenith_deg, 90.0)
    return 1.0 / (math.cos(math.radians(z)) + 0.50572 * (96.07995 - z) ** -1.6364)


def extraterrestrial(t: datetime) -> float:
    """W/m² normal to the sun outside the atmosphere."""
    return TSI / distance_au(t) ** 2


def beam_normal(t: datetime, elevation_deg: float, pressure_pa: float, altitude_m: float = 0.0) -> float:
    """Direct normal irradiance [W/m²] at a vehicle where the static pressure is ``pressure_pa`` (estimate: Meinel with
    pressure-scaled air mass). Zero when the sun is below the vehicle's horizon (``elevation`` < −dip)."""
    if elevation_deg < -horizon_dip(altitude_m):
        return 0.0
    am = air_mass(90.0 - elevation_deg) * pressure_pa / P0
    return extraterrestrial(t) * 0.7 ** (am ** 0.678)


def irradiance_on(normals, t: datetime, lat: float, lon: float, pressure_pa: float, altitude_m: float = 0.0,
                  albedo: float = 0.3, diffuse_fraction: float = 0.01) -> np.ndarray:
    """Irradiance [W/m²] on facets with unit ``normals`` (N×3, east-north-up): the beam on the sunlit side, plus an
    isotropic sky diffuse (``diffuse_fraction`` of the beam on a horizontal plane; ~1 % above the tropopause) on the
    upward-facing view factor, plus ``albedo`` × horizontal beam on the downward-facing view factor (clouds and
    ground). Estimate (see the module docstring)."""
    n = np.atleast_2d(np.asarray(normals, dtype=float))
    el, az = position(t, lat, lon)
    dni = beam_normal(t, el, pressure_pa, altitude_m)
    if dni == 0.0:
        return np.zeros(len(n))
    s = sun_vector(el, az)
    beam = dni * np.clip(n @ s, 0.0, None)
    ghi = dni * max(math.sin(math.radians(el)), 0.0)
    up = n[:, 2]
    sky = diffuse_fraction * ghi * (1 + up) / 2
    ground = albedo * ghi * (1 - up) / 2
    return beam + sky + ground


def day_length(date: datetime, lat: float, lon: float = 0.0, altitude_m: float = 0.0, step_min: float = 2.0) -> float:
    """Hours the sun's centre is above the vehicle's horizon on the UTC day of ``date`` (sampled every ``step_min``)."""
    t0 = datetime(date.year, date.month, date.day, tzinfo=timezone.utc)
    dip = horizon_dip(altitude_m)
    n = int(1440 / step_min)
    up = sum(position(t0 + timedelta(minutes=i * step_min), lat, lon)[0] > -dip for i in range(n))
    return up * step_min / 60.0
