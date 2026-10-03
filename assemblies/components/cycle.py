"""The microjet's thermodynamic cycle (``vegeta.boreas.microjet``) as a component: the catalogue engine, its
calibration from a CFD speed line, the design-point table, the performance map and the export notebook 29 reads."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from vegeta.boreas import microjet as mj

V_MAP = np.linspace(0.0, 220.0, 23)                       # m/s: the export's airspeed axis (notebook 28)


def rpm_map(e: mj.Microjet) -> np.ndarray:
    return e.rpm_idle + (e.rpm_max - e.rpm_idle) * np.linspace(0, 1, 14) ** 1.2


def from_catalogue(engine_class: str) -> mj.Microjet:
    """A calibrated engine of a ``microjet.CATALOGUE`` class (thrust, fuel flow and EGT of its datasheet)."""
    if engine_class not in mj.CATALOGUE:
        raise ValueError(f"engine_class must be one of {sorted(mj.CATALOGUE)}")
    return mj.from_catalogue(engine_class)


def design_point(e: mj.Microjet) -> dict:
    """The cycle at full speed, static, sea level (the table notebook 28 shows first)."""
    p = mj.solve(e, e.rpm_max)
    return {"thrust_N": p.thrust, "fuel_g_min": p.fuel_flow_g_min, "mass_flow_kg_s": p.mass_flow,
            "pressure_ratio": p.pressure_ratio, "tit_K": p.t4, "egt_K": p.t5, "jet_velocity_m_s": p.jet_velocity,
            "tip_speed_m_s": p.tip_speed, "tip_mach": p.tip_mach, "burner_efficiency": e.eta_burner, "tsfc_kg_N_h": p.tsfc,
            "nozzle_choked": bool(p.nozzle_choked), "mass_kg": e.mass_kg + e.system_mass_kg}


def calibrate_from_speedline(e: mj.Microjet, engine_class: str, speedline: dict | None) -> tuple[mj.Microjet, dict | None]:
    """Refit the cycle to the CFD compressor: the solved point nearest the cycle's design flow gives the work
    coefficient (slip) and the efficiency, the datasheet's thrust, fuel flow and EGT are fitted again (notebook 28
    section 5). Returns ``(engine, info)``; the engine unchanged and ``None`` without a usable point."""
    if not speedline:
        return e, None
    ok = [i for i, good in enumerate(speedline["ok"]) if good]
    if not ok:
        return e, None
    design_flow = mj.solve(e, e.rpm_max).mass_flow
    flows = np.asarray(speedline["mass_flow_kg_s"], float)
    i = min(ok, key=lambda k: abs(flows[k] - design_flow))
    work, eta = float(speedline["work_coefficient"][i]), float(speedline["efficiency_tt"][i])
    if not (np.isfinite(work) and np.isfinite(eta) and 0.5 < eta < 0.95):
        return e, None
    c = mj.CATALOGUE[engine_class]
    refit = mj.calibrate(replace(e, slip=work / e.power_input, eta_c=eta, a4=0.0, a8=0.0),
                         c["thrust_N"], c["fuel_g_min"] / 60e3, c["egt_K"])
    return refit, {"work_coefficient": work, "eta_c": eta, "mass_flow": float(flows[i]), "point": i}


def performance_map(e: mj.Microjet) -> dict:
    """Net thrust, fuel flow, air flow, TIT, EGT, pressure ratio and jet speed over ``V_MAP`` x ``rpm_map(e)``."""
    return mj.performance_map(e, V_MAP, rpm_map(e))


def export(e: mj.Microjet, path: str | Path, extra: dict) -> Path:
    """Write the engine and its map in the format ``microjet.load`` and ``aguya_flight.jet_unit_from_export`` read."""
    return mj.export(e, path, V_MAP, rpm_map(e), extra=extra)
