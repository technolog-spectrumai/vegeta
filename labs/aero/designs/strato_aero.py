"""Strato — aerodynamics of the airship: drag, added mass, the Munk moment against the fins, propulsive power (V8.7).

Verified physics: Lamb's inertia coefficients of a prolate spheroid (exact potential flow), the Munk moment that
follows from them, actuator-disc (momentum) propeller efficiency, the skin-friction law (``og.aero.cf_turbulent``).
**Estimates**: the drag build-up (Hoerner's form factor for streamlined bodies of revolution, his fin form factor,
a gondola pressure-drag coefficient, an allowance for outriggers and interference), the fin lift slope (low-aspect-
ratio wing formula, a hull-wake factor), the propeller profile efficiency. Total hull drag coefficients of
streamlined airship hulls at Re 10⁶–10⁸ are reported at C_DV ≈ 0.02–0.035 (on V^(2/3)); the tests check the
build-up lands in that band, which is a plausibility check, not a validation (Stage V8.7 keeps an Aeromant RANS case
for that, NOT RUN without OpenFOAM).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import ogma.aero as oa
from ogma import stratosphere as st

import strato_catalogue as cat
import strato_geometry as sg

INTERFERENCE = 1.10               # estimate: hull–fin–gondola interference and surface roughness
GONDOLA_CD = 0.25                 # estimate: rounded box, on its frontal area
APPENDAGE_ALLOWANCE = 0.06        # estimate: outriggers, nacelles, stern hub, antennas: of the hull drag
FIN_WAKE_EFFICIENCY = 0.8         # estimate: dynamic pressure at the fins / free stream (hull boundary layer)


def lamb_k(fineness: float) -> tuple[float, float, float]:
    """Lamb's inertia coefficients (k1 axial, k2 transverse, k′ rotational) of a prolate spheroid of ``fineness``
    a/b (exact potential-flow solution; Lamb, *Hydrodynamics* §373; e.g. f = 4: k1 = 0.082, k2 = 0.860)."""
    if fineness <= 1.0:
        raise ValueError("a prolate spheroid needs fineness > 1")
    e = math.sqrt(1 - 1 / fineness ** 2)
    ln = math.log((1 + e) / (1 - e))
    a0 = 2 * (1 - e * e) / e ** 3 * (0.5 * ln - e)
    b0 = 1 / e ** 2 - (1 - e * e) / (2 * e ** 3) * ln
    k1 = a0 / (2 - a0)
    k2 = b0 / (2 - b0)
    kr = e ** 4 * (b0 - a0) / ((2 - e * e) * (2 * e * e - (2 - e * e) * (b0 - a0)))
    return k1, k2, kr


@dataclass(frozen=True)
class Drag:
    hull: float                   # N
    fins: float
    gondola: float
    appendages: float
    total: float
    cdv: float                    # total / (q V^(2/3))
    reynolds: float


def drag(ship: sg.Strato, speed: float, altitude: float, delta_t: float = 0.0) -> Drag:
    """Drag [N] at true airspeed ``speed`` [m/s] and ``altitude`` [m] (estimate: the build-up of the module docstring)."""
    h, lay = ship.hull, ship.layout
    air = st.state(altitude, delta_t)
    q = air.dynamic_pressure(speed)
    mach = speed / air.speed_of_sound
    re = air.reynolds(max(speed, 1e-6), h.length)
    cf = oa.cf_turbulent(re, mach, re_min=1e4)
    d_l = 1 / h.fineness
    hull = q * cf * (1 + 1.5 * d_l ** 1.5 + 7 * d_l ** 3) * h.surface_area
    c_mean = 0.5 * (lay.fin_root_chord + lay.fin_tip_chord) * h.length
    t_c = lay.fin_thickness * lay.fin_root_chord / (0.5 * (lay.fin_root_chord + lay.fin_tip_chord))
    cf_fin = oa.cf_turbulent(air.reynolds(max(speed, 1e-6), c_mean), mach, re_min=1e4)
    fins = q * cf_fin * (1 + 2 * t_c + 60 * t_c ** 4) * 2 * ship.fin_planform_area * lay.fin_count
    gondola = q * GONDOLA_CD * (lay.gondola_width * h.diameter) * (lay.gondola_height * h.diameter)
    appendages = APPENDAGE_ALLOWANCE * hull
    total = INTERFERENCE * (hull + fins + gondola + appendages)
    return Drag(hull, fins, gondola, appendages, total, total / (q * h.volume ** (2 / 3)) if q > 0 else 0.0, re)


def propeller_efficiency(thrust: float, speed: float, density: float, disc_area: float,
                         profile: float = cat.PROPULSION.profile_efficiency) -> float:
    """η = profile × actuator-disc ideal efficiency 2 / (1 + √(1 + T / (½ ρ V² A)))."""
    if thrust <= 0:
        return profile
    if speed <= 0:
        return 0.0
    ct = thrust / (0.5 * density * speed ** 2 * disc_area)
    return profile * 2 / (1 + math.sqrt(1 + ct))


def stern_disc_area(ship: sg.Strato) -> float:
    return math.pi * (0.5 * ship.layout.stern_prop_diameter * ship.hull.diameter) ** 2


def electric_power(ship: sg.Strato, speed: float, altitude: float, delta_t: float = 0.0, disc_area: float | None = None,
                   prop: cat.Propulsion = cat.PROPULSION) -> dict:
    """Electrical power [W] to fly at ``speed`` [m/s]: thrust = drag, through the propeller, motor and ESC."""
    if speed <= 0:
        return {"power": 0.0, "thrust": 0.0, "eta_prop": 0.0, "drag": 0.0}
    rho = float(st.density(altitude, delta_t))
    d = drag(ship, speed, altitude, delta_t).total
    A = disc_area if disc_area is not None else stern_disc_area(ship)
    eta = propeller_efficiency(d, speed, rho, A, prop.profile_efficiency)
    shaft = d * speed / eta
    return {"power": shaft / (prop.motor_efficiency * prop.esc_efficiency), "shaft": shaft, "thrust": d,
            "eta_prop": eta, "drag": d}


def max_speed(ship: sg.Strato, electric_w: float, altitude: float, delta_t: float = 0.0, disc_area: float | None = None) -> float:
    """The airspeed [m/s] that ``electric_w`` [W] buys (bisection: power grows monotonically with speed)."""
    if electric_w <= 0:
        return 0.0
    lo, hi = 0.0, 1.0
    while electric_power(ship, hi, altitude, delta_t, disc_area)["power"] < electric_w and hi < 200:
        hi *= 2
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if electric_power(ship, mid, altitude, delta_t, disc_area)["power"] < electric_w:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def stability(ship: sg.Strato, x_cg: float) -> dict:
    """Static pitch/yaw stability at small angle of attack: the fins' restoring moment derivative against the Munk
    (destabilising) moment derivative, both per radian and per unit dynamic pressure [m³]. ``margin`` > 0 means the
    fins win (statically stable, controls only trim); < 0 means the ship needs active control (Piccolo's job)."""
    h, lay = ship.hull, ship.layout
    k1, k2, _ = lamb_k(h.fineness)
    munk = 2 * (k2 - k1) * h.volume                     # dM/dα / q, from M = q V (k2 − k1) sin 2α
    S = ship.fin_planform_area
    span = lay.fin_span * h.diameter
    ar = 2 * span ** 2 / S                              # the hull reflects the fin root: effective aspect ratio doubles
    a = 2 * math.pi * ar / (2 + math.sqrt(ar ** 2 + 4))
    projected = sum(math.cos(math.radians(lay.fin_roll_deg + 360.0 * k / lay.fin_count)) ** 2 for k in range(lay.fin_count))
    c_root = lay.fin_root_chord * h.length
    x_ac = lay.fin_trailing_edge * h.length - 0.75 * c_root   # quarter chord of the mean chord, roughly
    fins = FIN_WAKE_EFFICIENCY * a * S * projected * (x_ac - x_cg)
    return {"munk": munk, "fins": fins, "margin": (fins - munk) / munk, "k1": k1, "k2": k2, "fin_lift_slope": a,
            "fin_aspect_ratio": ar}
