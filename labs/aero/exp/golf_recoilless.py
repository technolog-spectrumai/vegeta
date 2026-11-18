"""A recoilless golf-ball air cannon (after the recoilless rifle: momentum thrown backward cancels the recoil).

A rear venturi, on the golf cannon's three cases (``golf_gun.CASES``: tank, valve, pressure, 600 mm tube, bore 42.7):

**Rear venturi** (Krupp / Carl Gustaf style). The chamber behind the ball (the tank's gas arrives from the side,
   so the tank adds no axial momentum) has a rear nozzle: throat ``A_t``, a divergent cone of area ratio ``eps``.
   While the ball is in the tube the launcher feels
       F = -(p_c - p_a) A_b                         the gas pushing back on the breech (as in a plain gun)
           + C_F(p_c) p_c A_t                       the rear jet's thrust, forward
   and after the ball leaves, the gas also blows out of the muzzle (a sonic jet, backward on the launcher). The throat is
   sized so the whole shot's impulse is ~0. The price: the nozzle bleeds the chamber, so the same pressure throws the
   ball slower, and a lot more air is used; a back-blast zone opens behind the launcher.
   In a recoilless rifle the propellant burns in the chamber and makes gas faster than the nozzle lets it out. Here
   the gas comes through a valve, so the valve must be big compared with the throat (C_F A_t ~ A_b means a throat
   of ~35 mm for a 42.7 mm bore): ``valve_scale`` grows the valve.

0-D, ideal gas, adiabatic, as ``golf_gun.gun``; the nozzle's thrust coefficient from isentropic flow (an ideal
divergent cone; over-expanded at low chamber pressure, where it is taken as the sonic value if lower).
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

import golf_gun as gg

GAM, R, CP = 1.4, 287.05, 1005.0
PA, TA = gg.P_AMB, gg.T_AMB
A_B = gg.A_BORE
M_BALL = gg.M_BALL


def _mach_from_area(eps):
    """Supersonic Mach for an area ratio eps (A/A*), by bisection."""
    lo, hi = 1.0, 6.0
    for _ in range(60):
        m = (lo + hi) / 2
        f = 1 / m * ((2 / (GAM + 1)) * (1 + (GAM - 1) / 2 * m * m)) ** ((GAM + 1) / (2 * (GAM - 1)))
        lo, hi = (m, hi) if f < eps else (lo, m)
    return (lo + hi) / 2


def thrust_coeff(pc, eps):
    """C_F = F / (p_c A_t) for a choked nozzle of area ratio eps into PA; falls back to the sonic orifice (eps = 1)
    when that is better (a cone that over-expands badly separates). 0 when the nozzle is not choked."""
    if pc <= PA * ((GAM + 1) / 2) ** (GAM / (GAM - 1)):
        return 0.0 if pc <= PA else (pc - PA) / pc * 0.6         # subsonic tail-off: a rough jet
    def cf(e):
        me = _mach_from_area(e) if e > 1.0001 else 1.0
        pe = pc * (1 + (GAM - 1) / 2 * me * me) ** (-GAM / (GAM - 1))
        mom = math.sqrt(2 * GAM ** 2 / (GAM - 1) * (2 / (GAM + 1)) ** ((GAM + 1) / (GAM - 1)) * (1 - (pe / pc) ** ((GAM - 1) / GAM)))
        return mom + (pe - PA) / pc * e
    return max(cf(eps), cf(1.0))


def mdot_orifice(a, p0, T0, pb):
    if a <= 0 or pb >= p0:
        return 0.0
    r = pb / p0
    crit = (2 / (GAM + 1)) ** (GAM / (GAM - 1))
    if r <= crit:
        return a * p0 / math.sqrt(R * T0) * math.sqrt(GAM) * (2 / (GAM + 1)) ** ((GAM + 1) / (2 * (GAM - 1)))
    return a * p0 / math.sqrt(R * T0) * math.sqrt(2 * GAM / (GAM - 1) * (r ** (2 / GAM) - r ** ((GAM + 1) / GAM)))


def shot(case: gg.Case, throat_mm=0.0, eps=2.0, valve_scale=1.0, p_gauge=None, friction=4.0, dt=1e-6, t_after=0.05):
    """One shot of the venturi launcher (``throat_mm`` 0 = the plain golf cannon). The valve opens over its time,
    stays open until the ball leaves, then closes over the same time; the chamber then blows down through the muzzle
    and the nozzle. Returns the ball's speed, the launcher's impulse (+ forward), the air used, the peak chamber
    pressure and the rear jet's mean exit speed."""
    v = case.valve
    p0i = PA + (case.p_gauge if p_gauge is None else p_gauge) * 1e5
    V0 = case.tank.volume_l * 1e-3
    m_t = m_t0 = p0i * V0 / (R * TA)
    a_valve = v.cd * v.area * valve_scale ** 2
    a_t = math.pi / 4 * (throat_mm * 1e-3) ** 2
    x0 = 0.02
    travel = gg.BARREL - x0 - gg.D_BALL
    Vc = A_B * x0
    m_c = PA * Vc / (R * TA)
    E_c = PA * Vc / (GAM - 1)
    x = u = t = 0.0
    J = 0.0                    # launcher impulse, + forward
    out_t = None
    pc_max = 0.0
    jet_m = jet_J = 0.0
    a_snd = math.sqrt(GAM * R * TA)
    while True:
        rho_t = m_t / V0
        p_t = p0i * (rho_t / (p0i / (R * TA))) ** GAM
        T_t = p_t / (rho_t * R)
        Vc_now = Vc + A_B * min(x, travel)
        p_c = (GAM - 1) * E_c / Vc_now
        T_c = p_c * Vc_now / (m_c * R)
        pc_max = max(pc_max, p_c)
        # valve: ramp open, open until the ball leaves, then ramp shut
        if out_t is None:
            f = min(1.0, t / v.open_s)
        else:
            f = max(0.0, 1.0 - (t - out_t) / v.open_s)
        mdot_in = mdot_orifice(a_valve * f, p_t, T_t, p_c)
        mdot_n = mdot_orifice(a_t, p_c, T_c, PA)
        thrust_n = thrust_coeff(p_c, eps) * p_c * a_t if a_t else 0.0
        if out_t is None:
            M = u / a_snd
            pf = PA * (1 + GAM * (GAM + 1) / 4 * M * M + GAM * M * math.sqrt(1 + ((GAM + 1) / 4) ** 2 * M * M))
            F = A_B * (p_c - pf)
            Fn = F - friction if (u > 0 or F > friction) else 0.0
            u = max(0.0, u + Fn / M_BALL * dt)
            x += u * dt
            F_launch = -(p_c - PA) * A_B + thrust_n + (friction if u > 0 else 0.0)
            mdot_f = 0.0
            if x >= travel:
                out_t = t
                v_out = u
        else:
            mdot_f = mdot_orifice(A_B, p_c, T_c, PA)                 # blowdown out of the muzzle
            F_launch = thrust_n - thrust_coeff(p_c, 1.0) * p_c * A_B
            if (t - out_t > t_after) or (p_c < PA * 1.02 and f == 0):
                break
        J += F_launch * dt
        if mdot_n:
            jet_m += mdot_n * dt
            jet_J += thrust_n * dt
        m_t -= mdot_in * dt
        m_c += (mdot_in - mdot_n - mdot_f) * dt
        E_c += (mdot_in * CP * T_t - (mdot_n + mdot_f) * CP * T_c - (p_c * A_B * u if out_t is None else 0.0)) * dt
        m_c = max(m_c, 1e-9)
        E_c = max(E_c, 1e-9)
        t += dt
        if t > 0.5:
            v_out = u if out_t is None else v_out
            break
    return {"v": v_out, "impulse_Ns": J, "ball_momentum": M_BALL * v_out, "air_g": (m_t0 - m_t) * 1e3,
            "pc_max_bar": pc_max / 1e5, "jet_air_g": jet_m * 1e3, "jet_speed": jet_J / jet_m if jet_m else 0.0,
            "barrel_ms": (out_t or t) * 1e3}


def null_throat(case, eps=2.0, valve_scale=1.0, p_gauge=None):
    """The throat diameter (mm) that leaves the launcher no net impulse (bisection on the impulse's sign)."""
    lo, hi = 0.0, gg.BORE * 1e3
    for _ in range(30):
        mid = (lo + hi) / 2
        j = shot(case, mid, eps, valve_scale, p_gauge, dt=4e-6)["impulse_Ns"]
        lo, hi = (mid, hi) if j < 0 else (lo, mid)
    return (lo + hi) / 2


def restore_pressure(case, throat_mm, eps, valve_scale, target):
    """The tank pressure (bar g) that gives the plain gun's ball speed back with this nozzle; the throat is kept."""
    lo, hi = 0.1, 400.0
    for _ in range(30):
        mid = math.sqrt(lo * hi)
        v = shot(case, throat_mm, eps, valve_scale, mid, dt=4e-6)["v"]
        lo, hi = (mid, hi) if v < target else (lo, mid)
    return hi


def recoilless_design(case, target, eps=2.0, valve_scale=1.0, iters=4):
    """Throat AND tank pressure together: the plain gun's ball speed with no net recoil (alternate: null the impulse
    at this pressure, restore the speed with this throat)."""
    p = case.p_gauge
    th = 0.0
    for _ in range(iters):
        th = null_throat(case, eps, valve_scale, p)
        p = restore_pressure(case, th, eps, valve_scale, target)
    return th, p


def backblast(jet_speed, throat_mm, eps, safe=15.0):
    """How far behind the launcher the jet is still faster than ``safe`` m/s: a round free jet's centreline decays as
    u = 6 u_e d_e / x (Rajaratnam), d_e the nozzle exit's diameter."""
    de = throat_mm * 1e-3 * math.sqrt(eps)
    return 6 * jet_speed * de / safe


def summary(eps=2.0):
    rows = []
    for c in gg.CASES:
        c = gg.size(c)
        plain = shot(c, 0.0)
        # A at the same pressure: the throat that nulls the recoil, and what the ball loses
        vs = 1.0 if c.p_gauge > 5 else 2.0               # a low-pressure case needs a wider valve to feed the nozzle
        th_same = null_throat(c, eps, vs)
        ven = shot(c, th_same, eps, vs)
        # A designed: throat and pressure together for the plain gun's speed with no recoil
        dt_, p_back = recoilless_design(c, plain["v"], eps, vs)
        back = shot(c, dt_, eps, vs, p_back)
        rows.append(dict(case=c.name, p=c.p_gauge, plain_v=plain["v"], plain_J=plain["impulse_Ns"], plain_air=plain["air_g"],
                         valve_scale=vs, throat_same_p=th_same, throat_mm=dt_, ven_v=ven["v"], ven_J=ven["impulse_Ns"], ven_air=ven["air_g"],
                         jet_speed=ven["jet_speed"], p_restore=p_back, restore_v=back["v"], restore_J=back["impulse_Ns"],
                         restore_air=back["air_g"], backblast_m=backblast(back["jet_speed"], dt_, eps),
                         range_plain=gg.best(plain["v"])[1]["range"], range_ven=gg.best(ven["v"])[1]["range"] if ven["v"] > 1 else 0.0))
    return rows


if __name__ == "__main__":
    for r in summary():
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
