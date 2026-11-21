"""SALTY's hydraulic jump (``spider_actuation.salty_catapult``), simulated: the take-off as one degree of freedom
along the jump direction, coupled to the hydraulics.

The pump has charged a gas accumulator (pre-volume ``gas_cm3`` at ``p_charge_bar``) that stays connected to the two
jumping legs' extensor cylinders (``bore_mm``); a latch holds the legs folded. The trigger releases the latch: the
pressure drives the pistons, the leg levers multiply their travel by ``lever_ratio`` into body travel along the
take-off direction (45° unless ``angle_deg``), until the legs reach full stroke ``stroke_m`` — then flight.

    gas:      p_a = p0 · (V0 / (V0 + V_out))^n                (the gas expands as fluid leaves; n = 1 isothermal,
                                                               1.4 adiabatic: ``polytropic``)
    line:     p_a − p_c = R_lam·Q + K·ρ/2·(Q/A_line)²          (Hagen–Poiseuille + one velocity head per fitting)
    body:     m·a = 2·p_c·A_cyl / lever_ratio − m·g·sin(angle) − friction
    flow:     Q = 2·A_cyl·v / lever_ratio

Out: take-off speed, distance on level ground, apex height, take-off time, peak and mean power, the jump's share of
the gas energy spent, the worst line velocity and pressure loss, the pump's recharge time at ``pump_w``.
"""
from __future__ import annotations

import math

__all__ = ["simulate", "G"]

G = 9.81
RHO = 1050.0                 # water–glycol [kg/m³]
MU = 2.5e-3                  # its viscosity [Pa·s]


def simulate(mass_kg=0.25, p_charge_bar=16.0, gas_cm3=4.0, bore_mm=10.0, lever_ratio=7.5, stroke_m=0.05,
             line_id_mm=3.0, line_len_m=0.08, fittings_k=3.0, friction_n=0.5, angle_deg=45.0, polytropic=1.0,
             pump_w=3.0, pump_eff=0.4, dt=2e-5):
    A = math.pi * (bore_mm / 2e3) ** 2
    Al = math.pi * (line_id_mm / 2e3) ** 2
    R_lam = 128 * MU * line_len_m / (math.pi * (line_id_mm / 1e3) ** 4)
    V0 = gas_cm3 * 1e-6
    p0 = p_charge_bar * 1e5
    s = math.sin(math.radians(angle_deg))
    x = v = t = 0.0
    v_out = 0.0
    p_peak_loss = line_v_max = power_max = 0.0
    work = 0.0
    while x < stroke_m and t < 0.5:
        q = 2 * A * v / lever_ratio
        pa = p0 * (V0 / (V0 + v_out)) ** polytropic
        dp = R_lam * q + fittings_k * RHO / 2 * (q / Al) ** 2
        pc = max(pa - dp, 0.0)
        f = 2 * pc * A / lever_ratio - mass_kg * G * s - friction_n
        a = f / mass_kg
        if a <= 0 and v <= 0:                                             # not enough pressure to start
            break
        v_new = max(v + a * dt, 0.0)
        x += 0.5 * (v + v_new) * dt
        v_out += 2 * A * 0.5 * (v + v_new) * dt / lever_ratio
        power_max = max(power_max, f * v_new)
        work += (2 * pc * A / lever_ratio) * 0.5 * (v + v_new) * dt
        p_peak_loss = max(p_peak_loss, dp)
        line_v_max = max(line_v_max, q / Al)
        v = v_new
        t += dt
    took_off = x >= stroke_m
    rng = v ** 2 * math.sin(2 * math.radians(angle_deg)) / G if took_off else 0.0
    apex = (v * s) ** 2 / (2 * G) if took_off else 0.0
    # gas energy spent: ∫p dV over the delivered volume
    if polytropic == 1.0:
        e_gas = p0 * V0 * math.log((V0 + v_out) / V0)
    else:
        n = polytropic
        e_gas = p0 * V0 / (n - 1) * (1 - (V0 / (V0 + v_out)) ** (n - 1))
    ek = 0.5 * mass_kg * v ** 2
    recharge_s = (p0 * v_out) / (pump_w * pump_eff) if v_out > 0 else 0.0
    return {"took_off": took_off, "takeoff_speed_m_s": round(v, 4), "distance_m": round(rng, 4),
            "apex_m": round(apex, 4), "takeoff_s": round(t, 5), "fluid_cm3": round(v_out * 1e6, 3),
            "gas_energy_j": round(e_gas, 4), "kinetic_j": round(ek, 4),
            "efficiency": round(ek / e_gas, 4) if e_gas > 0 else None,
            "power_peak_w": round(power_max, 2), "power_mean_w": round(ek / t, 2) if t > 0 else 0.0,
            "line_velocity_max_m_s": round(line_v_max, 3), "line_loss_max_bar": round(p_peak_loss / 1e5, 3),
            "pressure_end_bar": round(p0 * (V0 / (V0 + v_out)) ** polytropic / 1e5, 3),
            "recharge_s": round(recharge_s, 3)}
