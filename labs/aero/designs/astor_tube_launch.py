"""ASTOR-T's first seconds out of the tube (``astor_t``): does it fly away, and what exit speed does the spring owe it?

The aircraft leaves the muzzle at the exit speed along the tube's elevation, stowed: wings along the body, the tail
blades trailing, the propeller folded. Then, in order: the tail blades and fins swing out as the tail clears the
muzzle; the wing panels spring open (``open_s``); the propeller opens and the motor comes to full power
(``motor_delay_s``, then a ramp). The flight controller holds the wings level and a pitch attitude: the tube's
elevation while slow, then a 12 deg climb; ASTOR's angle-of-attack limiter is under it.

The model: ``astor_v_flight.SimV`` for ASTOR-T. Until the panels are open the 60 deg (dash) state stands in for the
half-open wing (it has less lift than the open one, more than the stowed one: marked HAND); the opening is flown as
the sweep going 60 -> 12 deg in ``open_s``.

"Safe": 10 m above the muzzle at 1.3 x the spread stall or more. "Sink": how far below the muzzle it goes.
"""
from __future__ import annotations

import math

import numpy as np

import arigo1_model as am
import arigo1_mujoco as amj
import astor_v_flight as VF

MUZZLE_M = 30.0                    # the sim's height of the muzzle (high enough that a sinking launch is seen whole)


def launch(name="t", v_exit=18.0, elevation_deg=35.0, open_s=0.30, motor_delay_s=0.20, motor_ramp_s=0.30, headwind=0.0, crosswind=0.0,
           gust=0.0, seconds=8.0, record=None) -> dict:
    """One launch. ``headwind`` (+ against the launch), ``crosswind`` and a vertical ``gust`` (+ down) in m/s, steady."""
    import mujoco
    sim = VF.SimV(name, sweep=VF.SWEEPS[-1])
    C0 = VF.get(name, VF.SWEEPS[0])
    Vs = VF.stall(C0)
    sim.start(max(v_exit, 1.3 * Vs), MUZZLE_M)                        # a trim for the controller's elevator reference
    th = math.radians(elevation_deg)
    q = np.array([math.cos(-th / 2), 0.0, math.sin(-th / 2), 0.0])    # nose up by the elevation (Sim.start's convention)
    sim.d.qpos[3:7] = q
    sim.d.qvel[:] = 0.0
    sim.d.qvel[:3] = (v_exit * math.cos(th), 0.0, v_exit * math.sin(th))
    sim.wind[:] = (-headwind, crosswind, -gust)
    sim.sweep_cmd, sim.SWEEP_RATE = VF.SWEEPS[-1], (VF.SWEEPS[-1] - VF.SWEEPS[0]) / max(open_s, 0.05)
    mujoco.mj_forward(sim.m, sim.d)
    ap = amj.Autopilot(sim)
    t0 = sim.d.time
    t_clear = 1.0 / max(v_exit, 1.0)                                  # the tail leaves the muzzle (the aircraft is ~1 m long)
    z_min, v_min, a_max, n_max, t_safe, hist = 1e9, 1e9, -1e9, 0.0, None, []
    while sim.d.time - t0 < seconds:
        t = sim.d.time - t0
        s = sim.state()
        if t >= t_clear:
            sim.sweep_cmd = VF.SWEEPS[0]                              # the latch lets go: the panels spring open
        thr = 0.0 if t < motor_delay_s else min(1.0, (t - motor_delay_s) / max(motor_ramp_s, 0.01))
        slow = s["V"] < 1.25 * Vs
        theta_c = math.radians(min(elevation_deg, 20.0) if slow else 12.0)
        ap(phi_c=0.0, theta_c=theta_c, throttle=thr)
        sim.step(5)
        s, i = sim.state(), sim.info
        z = float(s["pos"][2]) - MUZZLE_M
        z_min, v_min = min(z_min, z), min(v_min, s["V"])
        a_max, n_max = max(a_max, math.degrees(i["alpha"])), max(n_max, abs(i["load_factor"]))
        if t_safe is None and z >= 10.0 and s["V"] >= 1.3 * Vs:
            t_safe = t
        hist.append((round(t, 2), round(float(s["pos"][0]), 1), round(z, 2), round(s["V"], 1), round(math.degrees(s["theta"]), 1),
                     round(math.degrees(i["alpha"]), 1), round(sim.sweep, 0), round(math.degrees(s["phi"]), 1)))
        if record is not None:
            record(sim, t)
        if z < -MUZZLE_M + 0.3:
            break
    s = sim.state()
    return {"design": name, "v_exit_m_s": v_exit, "elevation_deg": elevation_deg, "open_s": open_s, "headwind_m_s": headwind, "crosswind_m_s": crosswind,
            "gust_down_m_s": gust, "stall_spread_m_s": round(Vs, 1), "exit_over_stall": round(v_exit / Vs, 2),
            "sink_below_muzzle_m": round(max(0.0, -z_min), 2), "min_speed_m_s": round(v_min, 1), "max_alpha_deg": round(a_max, 1),
            "max_load_factor": round(n_max, 2), "max_bank_deg": round(max(abs(h[7]) for h in hist), 1),
            "time_to_safe_s": None if t_safe is None else round(t_safe, 2), "height_after_3s_m": next((h[2] for h in hist if h[0] >= 3.0), None),
            "ok": bool(t_safe is not None and z_min > -1.0), "history_t_x_z_V_theta_alpha_sweep_bank": hist[::10]}


def envelope(name="t", speeds=(12.0, 14.0, 16.0, 18.0, 20.0, 22.0), elevations=(25.0, 35.0, 45.0)) -> dict:
    """Calm air: sink and time to safe over exit speed and elevation; the lowest exit speed that stays above -1 m."""
    rows = [{k: v for k, v in launch(name, v, e).items() if not k.startswith("history")} for e in elevations for v in speeds]
    need = {f"{e:.0f}": next((r["v_exit_m_s"] for r in rows if r["elevation_deg"] == e and r["ok"]), None) for e in elevations}
    return {"rows": rows, "lowest_exit_speed_ok_m_s_by_elevation": need}
