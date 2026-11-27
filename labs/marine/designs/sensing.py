"""Sensing and non-lethal effectors at sea, for the guard machines (the interceptor boat, the guard carrier, the
drones' surveillance payload): how far a camera detects, recognises and identifies a boat, by day and night, through
clear air, haze and fog; how far a radar sees a small craft over the horizon and through the sea's clutter; how far
a loudhailer is understood, a searchlight lights, a strobe dazzles, a water cannon throws. Plain physics with class
values for the constants (each marked); no vendor's equipment.

Conventions: ranges in metres unless a name says ``_km``; heights above the sea surface in metres.

* **Cameras: the Johnson criteria.** A task needs N line pairs (cycles) across the target's critical dimension (the
  square root of its visible side area): detection 1, recognition 4, identification 6.4, for a 50 % probability
  (Johnson 1958, as in STANAG 4347). Two pixels make a cycle, so the geometric range is ``dc / (2 N ifov)``, the
  pixel's angle blurred by the gimbal's residual jitter. The atmosphere then limits it: in the visible, the
  target's contrast against the sea decays as ``exp(-3.912 R / V)`` (Koschmieder; V the meteorological visibility)
  until it falls below the task's threshold; in the infrared, the target's temperature difference decays with the
  IR extinction (molecular absorption + aerosols, which a haze's small particles hardly scatter but a fog's droplets
  do) until it falls below the task's threshold. The horizon cuts all of it: the target's top must be above it.
* **Radar**: the radar equation's fourth root, from a reference range for a reference cross-section (the radar's
  class), cut by the radar horizon (4/3 earth) and shortened by the sea clutter with the sea state (class factors).
* **Sound**: spherical spreading (-20 log r) and the air's absorption; a voice is understood when it is 10 dB over
  the noise where the listener is.
* **Light**: a beam's intensity from its lumens and angle; illuminance ``I / r^2`` through the air's transmission.
* **Water**: a monitor's nozzle speed from its pressure; the throw a fraction of the vacuum range (fire-monitor
  class data), the reaction ``rho Q v``.
"""
from __future__ import annotations

import math

G = 9.81
RHO_SW = 1025.0

#: Cycles across the critical dimension for 50 % probability (Johnson 1958).
JOHNSON = {"detection": 1.0, "recognition": 4.0, "identification": 6.4}
#: Contrast thresholds in the visible per task (the more detail, the more contrast it needs; class values, assumed).
CONTRAST_MIN = {"detection": 0.02, "recognition": 0.05, "identification": 0.08}
#: Apparent temperature-difference thresholds in the IR per task (K; a cooled MWIR core of ~25 mK NETD, its MRTD at
#: the spatial frequencies each task needs; class values, assumed).
DELTA_T_MIN = {"detection": 0.10, "recognition": 0.25, "identification": 0.40}
#: Weather for the sensors: meteorological visibility (km) and the IR aerosol factor (how much of the visible
#: extinction the IR sees: haze's small particles little, fog's droplets nearly all). Class values.
WEATHER = {"clear": {"visibility_km": 20.0, "ir_aerosol": 0.10},
           "haze": {"visibility_km": 5.0, "ir_aerosol": 0.15},
           "fog": {"visibility_km": 0.5, "ir_aerosol": 0.80}}
IR_MOLECULAR_PER_KM = 0.12            # MWIR water-vapour absorption at sea, mid-latitude (class value)

#: Targets: visible side length and height above the water (m), the inherent visible contrast against the sea at
#: night-adapted/day levels, the apparent temperature difference of the whole target (K: engine, people, hull),
#: radar cross-section (m^2, broadside, an order of magnitude). Class values; task 4 (``small_craft.py``) has them
#: per craft with their basis.
TARGETS = {"skiff_6m": {"length_m": 6.0, "height_m": 1.2, "contrast": 0.30, "delta_T_K": 4.0, "rcs_m2": 1.0},
           "fishing_20m": {"length_m": 20.0, "height_m": 6.0, "contrast": 0.40, "delta_T_K": 6.0, "rcs_m2": 100.0}}

#: Sea clutter: the fraction of its clear-sea range at which a radar still holds a small craft, by Hs (m). Class
#: values for a coherent X-band navigation radar with clutter processing (assumed).
CLUTTER = ((0.0, 1.0), (0.5, 0.95), (1.0, 0.85), (1.5, 0.75), (2.0, 0.65), (2.5, 0.55), (3.0, 0.45))


def critical_dimension(t: dict) -> float:
    """The square root of the target's visible side area (m)."""
    return math.sqrt(t["length_m"] * t["height_m"])


def horizon_km(h_observer: float, h_target: float = 0.0, kind: str = "optical") -> float:
    """The distance at which a target's top (``h_target``) dips below the horizon for an observer at ``h_observer``:
    geometric 3.57 sqrt(h), optical with standard refraction 3.83, radar (4/3 earth) 4.12 (km, h in m)."""
    k = {"geometric": 3.57, "optical": 3.83, "radar": 4.12}[kind]
    return k * (math.sqrt(max(h_observer, 0.0)) + math.sqrt(max(h_target, 0.0)))


def johnson_range(dc_m: float, ifov_rad: float, task: str, jitter_rad: float = 0.0) -> float:
    """The geometric range at which ``dc_m`` spans the task's cycles (two pixels a cycle)."""
    return dc_m / (2 * JOHNSON[task] * math.hypot(ifov_rad, jitter_rad))


def visible_atm_range(visibility_km: float, contrast: float, task: str) -> float:
    """Koschmieder: the range at which the target's contrast falls to the task's threshold (m)."""
    c_min = CONTRAST_MIN[task]
    if contrast <= c_min:
        return 0.0
    return visibility_km * 1000 / 3.912 * math.log(contrast / c_min)


def ir_extinction_per_km(weather: str) -> float:
    w = WEATHER[weather]
    return IR_MOLECULAR_PER_KM + w["ir_aerosol"] * 3.912 / w["visibility_km"]


def ir_atm_range(weather: str, delta_T: float, task: str) -> float:
    """The range at which the target's apparent temperature difference falls to the task's threshold (m)."""
    dt_min = DELTA_T_MIN[task]
    if delta_T <= dt_min:
        return 0.0
    return 1000 * math.log(delta_T / dt_min) / ir_extinction_per_km(weather)


def camera_dri(camera: dict, target: dict, h_observer: float, weather: str = "clear", night: bool = False,
               jitter_rad: float = 0.0) -> dict:
    """Detection, recognition and identification ranges (m, 50 %) of one camera channel at its narrowest field of
    view. ``camera``: {"band": "visible"|"ir", "pixels_h": .., "hfov_narrow_deg": ..}. A visible channel sees
    nothing at night (no illumination assumed: a searchlight or low-light core is another channel). Each range is
    the least of the Johnson range, the atmosphere's and the horizon's; ``limited_by`` says which."""
    ifov = math.radians(camera["hfov_narrow_deg"]) / camera["pixels_h"]
    dc = critical_dimension(target)
    hz = 1000 * horizon_km(h_observer, target["height_m"])
    out, why = {}, {}
    for task in JOHNSON:
        geo = johnson_range(dc, ifov, task, jitter_rad)
        if camera["band"] == "visible":
            atm = 0.0 if night else visible_atm_range(WEATHER[weather]["visibility_km"], target["contrast"], task)
        else:
            atm = ir_atm_range(weather, target["delta_T_K"], task)
        cand = {"pixels": geo, "atmosphere": atm, "horizon": hz}
        k = min(cand, key=cand.get)
        out[task], why[task] = round(cand[k]), k
    return {"range_m": out, "limited_by": why, "ifov_urad": round(ifov * 1e6, 1)}


def eoir_dri(gimbal: dict, target: dict, h_observer: float, jitter_rad: float = 0.0) -> dict:
    """A gimbal's DRI table for one target: day (the better of the visible and IR channels) and night (IR), in
    clear air, haze and fog, in km."""
    rows = {}
    for weather in WEATHER:
        for light in ("day", "night"):
            best = {}
            for ch in gimbal["channels"]:
                r = camera_dri(ch, target, h_observer, weather, light == "night", jitter_rad)["range_m"]
                for task, v in r.items():
                    best[task] = max(best.get(task, 0), v)
            rows[f"{light}_{weather}"] = {t: round(v / 1000, 2) for t, v in best.items()}
    return rows


def radar_range(ref_range_km: float, ref_rcs_m2: float, rcs_m2: float, h_antenna: float, h_target: float,
                hs_m: float = 0.0) -> float:
    """A radar's detection range of a target (km): the fourth-root scaling from its class reference, shortened by
    the sea clutter at Hs, cut by the radar horizon."""
    free = ref_range_km * (rcs_m2 / ref_rcs_m2) ** 0.25
    hs = [h for h, _ in CLUTTER]
    k = CLUTTER[-1][1] if hs_m >= hs[-1] else next(
        f0 + (f1 - f0) * (hs_m - h0) / (h1 - h0) for (h0, f0), (h1, f1) in zip(CLUTTER, CLUTTER[1:]) if h0 <= hs_m <= h1)
    return min(free * k, horizon_km(h_antenna, h_target, "radar"))


def sound_range(spl_1m_db: float, needed_db: float, alpha_db_m: float = 0.01) -> float:
    """The distance (m) at which a source of ``spl_1m_db`` falls to ``needed_db`` (spherical spreading + the air's
    absorption, ``alpha`` ~0.01 dB/m at 2 kHz)."""
    lo, hi = 1.0, 1e5
    for _ in range(80):
        r = math.sqrt(lo * hi)
        if spl_1m_db - 20 * math.log10(r) - alpha_db_m * r > needed_db:
            lo = r
        else:
            hi = r
    return lo


def hailer(spl_1m_db: float) -> dict:
    """A long-range acoustic hailer: where a voice is understood (10 dB over the listener's noise) in a fast skiff
    (outboards at speed, ~95 dB at the helm), a slow one (idle, ~75 dB) and a quiet deck (~60 dB). Noise levels:
    class values for open outboard craft (assumed)."""
    return {"spl_1m_db": spl_1m_db,
            "intelligible_m": {"skiff_at_speed": round(sound_range(spl_1m_db, 95 + 10)),
                               "skiff_idling": round(sound_range(spl_1m_db, 75 + 10)),
                               "quiet_deck": round(sound_range(spl_1m_db, 60 + 10))},
            "audible_m": round(sound_range(spl_1m_db, 60.0))}


def searchlight(lumens: float, beam_deg: float, visibility_km: float = 20.0, lux_needed: float = 1.0,
                beam_efficiency: float = 0.6) -> dict:
    """A searchlight's peak intensity (cd) from its lumens and full beam angle, and the range at which it lights a
    target to ``lux_needed`` (1 lux: a boat and its people plain to a camera or an eye) through the air."""
    omega = 2 * math.pi * (1 - math.cos(math.radians(beam_deg / 2)))
    I = lumens * beam_efficiency / omega
    sigma = 3.912 / (visibility_km * 1000)
    lo, hi = 1.0, 1e5
    for _ in range(80):
        r = math.sqrt(lo * hi)
        if I / r ** 2 * math.exp(-sigma * r) > lux_needed:
            lo = r
        else:
            hi = r
    return {"lumens": lumens, "beam_deg": beam_deg, "peak_cd": round(I), "range_m": round(lo)}


def strobe(peak_cd: float, flash_s: float, hz: float, visibility_km: float = 20.0) -> dict:
    """A non-laser white strobe: its effective intensity (Blondel-Rey), the range to which it dazzles a
    dark-adapted eye (>= 10 lux at the eye in each flash: a disorienting glare at night, the eye cannot hold the
    boat in view; assumed threshold) and to which it is seen (Allard, 2e-7 lux at night)."""
    eff = peak_cd * flash_s / (0.2 + flash_s)
    sigma = 3.912 / (visibility_km * 1000)

    def reach(E):
        lo, hi = 1.0, 1e6
        for _ in range(80):
            r = math.sqrt(lo * hi)
            lo, hi = (r, hi) if peak_cd / r ** 2 * math.exp(-sigma * r) > E else (lo, r)
        return lo

    def seen(E):
        lo, hi = 1.0, 1e6
        for _ in range(80):
            r = math.sqrt(lo * hi)
            lo, hi = (r, hi) if eff / r ** 2 * math.exp(-sigma * r) > E else (lo, r)
        return lo

    return {"peak_cd": peak_cd, "flash_ms": flash_s * 1000, "hz": hz, "effective_cd": round(eff),
            "dazzle_m": round(reach(10.0)), "visible_km": round(seen(2e-7) / 1000, 1), "laser": False}


def water_cannon(flow_l_min: float, pressure_bar: float, elevation_deg: float = 32.0, pump_eff: float = 0.70) -> dict:
    """A water monitor: the nozzle speed from its pressure, the throw (a fraction of the vacuum range: solid
    streams of fire monitors throw ~1/3 of it at 1500 L/min, more for bigger streams; class data), the reaction on
    the boat, the pump's hydraulic and shaft power."""
    q = flow_l_min / 60000
    v = math.sqrt(2 * pressure_bar * 1e5 / RHO_SW)
    vac = v * v / G * math.sin(math.radians(2 * elevation_deg))
    k = 0.33 * (flow_l_min / 1500) ** 0.15
    hyd = q * pressure_bar * 1e5
    return {"flow_l_min": flow_l_min, "pressure_bar": pressure_bar, "nozzle_m_s": round(v, 1),
            "throw_m": round(k * vac), "vacuum_range_m": round(vac), "reaction_N": round(RHO_SW * q * v),
            "hydraulic_kW": round(hyd / 1000, 1), "shaft_kW": round(hyd / pump_eff / 1000, 1), "sea_water": True}
