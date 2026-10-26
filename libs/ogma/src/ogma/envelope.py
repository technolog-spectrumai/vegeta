"""og.envelope — membrane forces in a pressurised, thin-walled body of revolution (balloon, airship hull, tank).

Membrane theory (no bending stiffness: exact for fabric and film envelopes). For an axisymmetric shell under a uniform
internal overpressure Δp, with r the radius to the axis, r1 the meridional radius of curvature and r2 = r / cos(θ)
the second principal radius (along the normal, to the axis; θ the meridian's slope):

    N_meridional = Δp r2 / 2
    N_hoop       = Δp r2 (1 − r2 / (2 r1))

(cylinder: Δp r and Δp r/2; sphere: Δp r/2 both). Forces are line loads [N/m]; stress = N / thickness. A hull also
bends as a beam (lift and weight not in the same place, gusts): on a thin circular section the extreme meridional
line load from a bending moment M is M / (π r²). The fabric cannot take compression, so the section wrinkles when the
bending load exceeds the pressure's meridional pre-tension: the minimum superpressure is Δp ≥ 2 M / (π r³).

Verified physics (membrane theory, Timoshenko & Woinowsky-Krieger ch. 14). Material strengths are inputs.
"""
from __future__ import annotations

import math

import numpy as np


def principal_radii(x, r) -> tuple[np.ndarray, np.ndarray]:
    """(r1, r2) [m] along a meridian given as stations ``x`` and radii ``r`` (numerical derivatives). r1 is signed:
    positive where the meridian is convex (curving towards the axis), negative where it is concave, inf where
    straight."""
    x, r = np.asarray(x, float), np.asarray(r, float)
    dr = np.gradient(r, x)
    d2r = np.gradient(dr, x)
    s = np.sqrt(1 + dr ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        r1 = np.where(np.abs(d2r) > 1e-12, -s ** 3 / d2r, np.inf)
    r2 = r * s
    return r1, r2


def membrane_forces(x, r, dp: float) -> dict:
    """Meridional and hoop line loads [N/m] along the meridian under overpressure ``dp`` [Pa]."""
    r1, r2 = principal_radii(x, r)
    with np.errstate(divide="ignore", invalid="ignore"):
        hoop = dp * r2 * (1 - np.where(np.isfinite(r1), r2 / (2 * r1), 0.0))
    return {"x": np.asarray(x, float), "meridional": dp * r2 / 2, "hoop": hoop, "r1": r1, "r2": r2}


def cylinder(dp: float, radius: float) -> tuple[float, float]:
    """(meridional, hoop) line loads [N/m] of a long pressurised cylinder."""
    return dp * radius / 2, dp * radius


def sphere(dp: float, radius: float) -> float:
    return dp * radius / 2


def bending_line_load(moment: float, radius: float) -> float:
    """Extreme meridional line load [N/m] of a thin circular section of ``radius`` carrying bending ``moment`` [N m]."""
    return moment / (math.pi * radius ** 2)


def wrinkling_pressure(moment: float, radius: float) -> float:
    """The superpressure [Pa] below which a section of ``radius`` under ``moment`` wrinkles: 2 M / (π r³)."""
    return 2 * abs(moment) / (math.pi * radius ** 3)


def beam_loads(x, w) -> tuple[np.ndarray, np.ndarray]:
    """Shear [N] and bending moment [N m] along a free–free beam loaded by the net distributed load ``w`` [N/m]
    (upward positive) at stations ``x`` — the hull as a beam floating in equilibrium (∫w dx ≈ 0). Integrated from the
    nose; any residual (the load not exactly in equilibrium) shows at the tail."""
    x, w = np.asarray(x, float), np.asarray(w, float)
    shear = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(x))])
    moment = np.concatenate([[0.0], np.cumsum(0.5 * (shear[1:] + shear[:-1]) * np.diff(x))])
    return shear, moment


def required_strength(line_load: float, safety_factor: float) -> float:
    """N/m of strength a laminate needs to carry ``line_load`` with ``safety_factor`` (creep, seams, UV ageing and
    temperature knock-downs belong in the factor or in the strength the caller passes)."""
    return line_load * safety_factor
