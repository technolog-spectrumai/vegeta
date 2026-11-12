"""Closed-form engineering models for a LEO servicing spacecraft: orbit mechanics, rocket equation, eclipse, power,
thermal, drag and relative motion. SI units throughout. Every function is a textbook relation (references in the
docstrings) so that tests can check it against a hand calculation.

Constants are VERIFIED (CODATA / IERS / EGM96 values); models of the atmosphere and the solar array are ASSUMED and
carry the uncertainty the caller passes in."""
from __future__ import annotations

import math

import numpy as np

MU = 3.986004418e14            # m^3/s^2, Earth GM (EGM2008/IERS)
RE = 6378137.0                 # m, equatorial radius (WGS-84)
J2 = 1.08262668e-3             # EGM2008
G0 = 9.80665                   # m/s^2, standard gravity (definition)
SIGMA = 5.670374419e-8         # W/m^2/K^4 (SI definition)
S0 = 1361.0                    # W/m^2, total solar irradiance (Kopp & Lean 2011)
EARTH_IR = 237.0               # W/m^2, mean outgoing longwave radiation (Trenberth et al.)
ALBEDO = 0.30                  # mean Bond albedo of Earth
YEAR = 365.25 * 86400.0
OMEGA_SUN = 2 * math.pi / YEAR  # rad/s, mean Sun motion needed for sun-synchronous orbits (1 rev per tropical year, approx.)


# --- orbit mechanics --------------------------------------------------------------------------------------------
def radius(alt_km: float) -> float:
    return RE + alt_km * 1e3


def v_circ(r: float) -> float:
    return math.sqrt(MU / r)


def period(a: float) -> float:
    return 2 * math.pi * math.sqrt(a ** 3 / MU)


def mean_motion(a: float) -> float:
    return math.sqrt(MU / a ** 3)


def hohmann(r1: float, r2: float) -> dict:
    """Two-impulse transfer between coplanar circular orbits (Vallado, alg. 36). Returns dv1, dv2, total, time of flight."""
    at = 0.5 * (r1 + r2)
    dv1 = abs(math.sqrt(MU * (2 / r1 - 1 / at)) - v_circ(r1))
    dv2 = abs(v_circ(r2) - math.sqrt(MU * (2 / r2 - 1 / at)))
    return {"dv1": dv1, "dv2": dv2, "dv": dv1 + dv2, "tof": math.pi * math.sqrt(at ** 3 / MU)}


def perigee_lowering_dv(r_circ: float, r_perigee: float) -> float:
    """One tangential burn at apogee that lowers perigee of a circular orbit to ``r_perigee``."""
    at = 0.5 * (r_circ + r_perigee)
    return v_circ(r_circ) - math.sqrt(MU * (2 / r_circ - 1 / at))


def spiral_dv(r1: float, r2: float) -> float:
    """Low-thrust circular-to-circular transfer (Edelbaum, coplanar): |v1 - v2|."""
    return abs(v_circ(r1) - v_circ(r2))


def phasing_dv(r: float, dphi: float, n_orbits: int = 1) -> dict:
    """Catch up (dphi > 0: target is ahead by dphi rad) with a coplanar phasing orbit of ``n_orbits`` revolutions.
    Two tangential burns of equal size at the common apse. Valid while the phasing orbit stays above the atmosphere."""
    T = period(r)
    Tp = T * (1 - dphi / (2 * math.pi * n_orbits))
    a = (MU * (Tp / (2 * math.pi)) ** 2) ** (1 / 3)
    rp = 2 * a - r
    if rp < RE + 120e3:
        raise ValueError(f"phasing orbit perigee {(rp - RE) / 1e3:.0f} km is inside the atmosphere: use more orbits")
    dv = abs(math.sqrt(MU * (2 / r - 1 / a)) - v_circ(r))
    return {"dv": 2 * dv, "duration": n_orbits * Tp, "a": a, "perigee_alt_km": (rp - RE) / 1e3}


def plane_change_dv(v: float, di_rad: float) -> float:
    return 2 * v * math.sin(abs(di_rad) / 2)


def raan_rate_j2(a: float, inc_rad: float, e: float = 0.0) -> float:
    """Secular nodal precession [rad/s] (Vallado eq. 9-41)."""
    p = a * (1 - e * e)
    return -1.5 * J2 * (RE / p) ** 2 * mean_motion(a) * math.cos(inc_rad)


def sso_inclination_deg(alt_km: float) -> float:
    a = radius(alt_km)
    c = OMEGA_SUN / (-1.5 * J2 * (RE / a) ** 2 * mean_motion(a))
    return math.degrees(math.acos(c))


def differential_raan_rate_deg_per_day(alt_km: float, inc_deg: float, d_alt_km: float) -> float:
    """Nodal-rate difference [deg/day] between two circular orbits ``d_alt_km`` apart: the only free way to close a RAAN gap
    (a plane change costs ~130 m/s per degree at 550 km)."""
    a, i = radius(alt_km), math.radians(inc_deg)
    return math.degrees(raan_rate_j2(a + d_alt_km * 1e3, i) - raan_rate_j2(a, i)) * 86400


# --- propulsion -------------------------------------------------------------------------------------------------
def propellant_mass(m0: float, dv: float, isp: float) -> float:
    """Tsiolkovsky: mp = m0 (1 - exp(-dv / (Isp g0)))."""
    return m0 * (1 - math.exp(-dv / (isp * G0)))


def final_mass(m0: float, dv: float, isp: float) -> float:
    return m0 * math.exp(-dv / (isp * G0))


def burn_time(m0: float, dv: float, isp: float, thrust: float) -> float:
    """Constant-thrust burn duration: t = (m0 Isp g0 / F)(1 - exp(-dv/(Isp g0)))."""
    ve = isp * G0
    return m0 * ve / thrust * (1 - math.exp(-dv / ve))


def total_impulse(m0: float, dv: float, isp: float) -> float:
    return propellant_mass(m0, dv, isp) * isp * G0


def hall_thrust(power_w: float, isp: float, efficiency: float) -> float:
    """F = 2 eta P / (Isp g0)."""
    return 2 * efficiency * power_w / (isp * G0)


# --- eclipse, power ---------------------------------------------------------------------------------------------
def eclipse_fraction(alt_km: float, beta_deg: float) -> float:
    """Fraction of a circular orbit in Earth's cylindrical shadow (Wertz): 0 above the critical beta."""
    h = alt_km * 1e3
    a = RE + h
    beta = math.radians(beta_deg)
    if abs(beta) >= math.asin(RE / a):
        return 0.0
    return math.acos(math.sqrt(h * h + 2 * RE * h) / (a * math.cos(beta))) / math.pi


def beta_angle_deg(inc_deg: float, raan_minus_sun_ra_deg: float, sun_decl_deg: float = 0.0) -> float:
    """Sun-orbit-plane angle (Wertz). For a sun-synchronous orbit the nodal local time fixes raan - RA_sun."""
    i, w, d = map(math.radians, (inc_deg, raan_minus_sun_ra_deg, sun_decl_deg))
    return math.degrees(math.asin(math.cos(d) * math.sin(i) * math.sin(w) + math.sin(d) * math.cos(i)))


def array_area(p_eol: float, eff_bol: float, packing: float, degr_per_year: float, years: float,
               incidence_loss: float, temp_loss: float, other_loss: float = 0.0) -> float:
    """Panel area [m^2] that delivers ``p_eol`` after ``years`` of degradation: P = S0 A eta pf (1-d)^t cos-loss..."""
    flux = S0 * eff_bol * packing * (1 - degr_per_year) ** years * (1 - incidence_loss) * (1 - temp_loss) * (1 - other_loss)
    return p_eol / flux


def battery_energy_wh(p_eclipse_w: float, t_eclipse_s: float, dod: float, eff: float = 0.95) -> float:
    return p_eclipse_w * t_eclipse_s / 3600 / (dod * eff)


# --- thermal ---------------------------------------------------------------------------------------------------
def equilibrium_temperature(q_internal: float, a_sun: float, a_earth_ir: float, a_alb: float, a_total: float,
                            alpha: float, eps: float, albedo_view: float = 0.3, sun: float = S0,
                            earth_ir: float = EARTH_IR, albedo: float = ALBEDO) -> float:
    """Single-node (isothermal) steady state: absorbed solar + albedo + Earth IR + internal = eps sigma A T^4 radiated.
    Areas are projected areas toward the Sun, Earth, and the albedo-lit Earth, ``a_total`` the radiating area."""
    absorbed = alpha * sun * a_sun + alpha * albedo * albedo_view * sun * a_alb + eps * earth_ir * a_earth_ir + q_internal
    return (absorbed / (eps * SIGMA * a_total)) ** 0.25


# --- drag and perturbations -------------------------------------------------------------------------------------
# ASSUMED exponential-fit density [kg/m^3] vs altitude [km] for three solar-activity cases; order-of-magnitude values
# consistent with NRLMSISE-00 / Jacchia class models. Replace with a real model before any lifetime claim.
_ALT = np.array([400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0])
_RHO = {"min": np.array([1.5e-12, 4.5e-13, 1.5e-13, 5.0e-14, 1.8e-14, 7.0e-15, 3.0e-15]),
        "mod": np.array([5.0e-12, 1.5e-12, 5.0e-13, 1.8e-13, 7.0e-14, 3.0e-14, 1.3e-14]),
        "max": np.array([1.5e-11, 6.0e-12, 2.5e-12, 1.1e-12, 5.0e-13, 2.4e-13, 1.2e-13])}


def density(alt_km: float, activity: str = "mod") -> float:
    """Log-linear interpolation of the table above."""
    return float(np.exp(np.interp(alt_km, _ALT, np.log(_RHO[activity]))))


def drag_dv_per_year(alt_km: float, ballistic_cda_m: float, activity: str = "mod") -> float:
    """Δv/yr to hold altitude: a_drag t = 0.5 rho v^2 (Cd A/m) t. ``ballistic_cda_m`` in m^2/kg."""
    v = v_circ(radius(alt_km))
    return 0.5 * density(alt_km, activity) * v * v * ballistic_cda_m * YEAR


def decay_rate_m_per_day(alt_km: float, ballistic_cda_m: float, activity: str = "mod") -> float:
    """da/dt = -rho (Cd A/m) sqrt(mu a) for a circular orbit."""
    a = radius(alt_km)
    return -density(alt_km, activity) * ballistic_cda_m * math.sqrt(MU * a) * 86400


def lifetime_days(alt_km: float, ballistic_cda_m: float, activity: str = "mod", stop_km: float = 150.0) -> float:
    """Integrate the circular decay rate down to ``stop_km`` (coarse 1 km steps; circular orbits only, so it
    underestimates the life of an elliptical orbit that is eccentric at release)."""
    t, h = 0.0, alt_km
    while h > stop_km:
        rate = -decay_rate_m_per_day(h, ballistic_cda_m, activity) / 1e3     # km/day
        t += 1.0 / rate
        h -= 1.0
    return t


# --- relative motion (Clohessy-Wiltshire, circular reference) ---------------------------------------------------
def cw_propagate(x0, n: float, t: float):
    """Hill frame: x radial (up), y along-track, z cross-track. State = [x y z vx vy vz]; closed-form propagation."""
    x, y, z, vx, vy, vz = x0
    s, c = math.sin(n * t), math.cos(n * t)
    return np.array([
        (4 - 3 * c) * x + s / n * vx + 2 / n * (1 - c) * vy,
        6 * (s - n * t) * x + y - 2 / n * (1 - c) * vx + (4 * s - 3 * n * t) / n * vy,
        c * z + s / n * vz,
        3 * n * s * x + c * vx + 2 * s * vy,
        -6 * n * (1 - c) * x - 2 * s * vx + (4 * c - 3) * vy,
        -n * s * z + c * vz])


def cw_min_distance(x0, n: float, duration: float, steps: int = 2000) -> float:
    ts = np.linspace(0, duration, steps)
    return float(min(np.linalg.norm(cw_propagate(x0, n, t)[:3]) for t in ts))


def cw_drift_free_ic(y_along: float) -> np.ndarray:
    """Initial condition for a co-elliptic hold: no secular drift (vy = -2 n x ... zero here), pure along-track offset."""
    return np.array([0.0, y_along, 0.0, 0.0, 0.0, 0.0])
