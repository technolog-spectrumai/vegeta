"""Three architecture variants and their bottom-up budgets (mass, power, Δv). All inputs are ASSUMED class-typical
values unless the docstring says otherwise; see ``components.py`` for catalogue parts to be confirmed against datasheets.

A: storable bipropellant + robotic arm      B: green monopropellant + standard docking adapter (cooperative targets)
C: Hall-effect electric + chemical RCS + compliant gripper

Closure rule: the propellant load is sized so that the satellite ALONE has ``DV_TOTAL`` m/s of capability (the
requirement as written). What this does to the number of removals is the job of ``mission.py``."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import physics as ph
from .provenance import assumed

DV_TOTAL = 500.0                 # m/s, requirement (incl. reserves)
MASS_MIN, MASS_MAX, MASS_TARGET = 350.0, 500.0, 400.0
MARGIN = assumed(0.25, "-", "dry-mass growth margin at concept level (AIAA S-120 style 25-30 %)", "dry_margin")
ALT_KM, INC_DEG = 550.0, 97.6
LIFE_YEARS = 5.0


@dataclass(frozen=True)
class Propulsion:
    kind: str
    isp: float                      # s, delivered (vacuum, system level)
    thrust: float                   # N, total available for orbit manoeuvres
    tank_fraction: float            # tank + pressurant mass per kg propellant
    hardware_kg: float              # thrusters, valves, lines, PPU
    power_w: float = 0.0            # electrical power while thrusting
    rcs_isp: float = 0.0            # chemical RCS (variant C); 0 = RCS is the main system
    rcs_dv: float = 0.0             # m/s reserved to the RCS (variant C)
    rcs_hardware_kg: float = 0.0


@dataclass(frozen=True)
class Variant:
    key: str
    name: str
    propulsion: Propulsion
    capture: str                    # arm | docking | gripper
    capture_kg: float
    cooperative_only: bool
    notes: str
    subsystems: dict = field(default_factory=dict)


def _common_subsystems() -> dict:
    """Dry masses [kg], bottom-up, identical across variants (ASSUMED; replace with component masses in ``components``)."""
    return {"structure": assumed(52, "kg", "Al honeycomb + machined frames, ~13 % of 400 kg", "m_structure"),
            "thermal": assumed(13, "kg", "MLI, radiators, heaters, thermistors", "m_thermal"),
            "harness": assumed(22, "kg", "~5 % of mass, redundant", "m_harness"),
            "avionics": assumed(9, "kg", "2x rad-tolerant OBC + IO + safety-monitor unit", "m_avionics"),
            "gnc": assumed(24, "kg", "2 ST, 2 IMU, 2 GNSS, sun sensors, 4 wheels, 3 MTQ", "m_gnc"),
            "prox_sensors": assumed(14, "kg", "scanning LiDAR, wide + narrow cameras, thermal IR, 2nd independent range sensor", "m_prox"),
            "comms": assumed(9, "kg", "2 S-band transponders + patches, X-band tx", "m_comms"),
            "battery_pcdu": assumed(10, "kg", "Li-ion pack + PCDU + SADM", "m_power_elec")}


VARIANTS = {
    "A": Variant("A", "Bipropellant + robotic arm",
                 Propulsion("bipropellant", isp=assumed(290, "s", "small storable bipropellant thrusters, system level", "A_isp"),
                            thrust=assumed(44, "N", "2 x 22 N class main engines", "A_thrust"),
                            tank_fraction=assumed(0.10, "-", "", "A_tank"), hardware_kg=assumed(14, "kg", "", "A_prop_hw")),
                 "arm", assumed(38, "kg", "7-DoF-class 2 m arm + end effector + controller", "A_capture_kg"), False,
                 "Most agile for tumbling targets; highest mass and mechanism risk."),
    "B": Variant("B", "Green monopropellant + docking adapter",
                 Propulsion("green_monoprop", isp=assumed(235, "s", "ADN-based monoprop, system level", "B_isp"),
                            thrust=assumed(22, "N", "4 x 5 N + 1 N class thrusters", "B_thrust"),
                            tank_fraction=assumed(0.12, "-", "", "B_tank"), hardware_kg=assumed(10, "kg", "", "B_prop_hw")),
                 "docking", assumed(20, "kg", "active docking/latching adapter + force-torque sensing", "B_capture_kg"), True,
                 "Simple and light, but only for targets with a grapple/docking fixture: no uncooperative debris."),
    "C": Variant("C", "Hall-effect electric + chemical RCS + gripper",
                 Propulsion("hall_electric", isp=assumed(1500, "s", "Hall thruster ~600-1000 W class", "C_isp"),
                            thrust=ph.hall_thrust(800.0, 1500.0, 0.5),
                            tank_fraction=assumed(0.15, "-", "xenon pressure vessels", "C_tank"),
                            hardware_kg=assumed(16, "kg", "Hall thruster + spare, PPU, flow controller", "C_prop_hw"),
                            power_w=assumed(800, "W", "thruster input power", "C_power"),
                            rcs_isp=assumed(220, "s", "green monoprop RCS", "C_rcs_isp"),
                            rcs_dv=assumed(60, "m/s", "proximity ops + capture + abort, on chemical RCS", "C_rcs_dv"),
                            rcs_hardware_kg=assumed(9, "kg", "RCS thrusters + tank", "C_rcs_hw")),
                 "gripper", assumed(26, "kg", "compliant 3-finger/net-free gripper on a short boom", "C_capture_kg"), False,
                 "Best propellant economy by far; thrust-limited tows take days to weeks."),
}


def budget(v: Variant, dv_total: float = DV_TOTAL, margin: float | None = None, wet_target: float | None = None) -> dict:
    """Closed mass budget for ``v``.

    ``wet_target=None``: the propellant is sized so the satellite alone has ``dv_total`` m/s (the requirement as written).
    ``wet_target=400``: the tanks are filled up to that wet mass and the Δv capability is reported instead.
    Variant C splits the Δv: ``rcs_dv`` on the chemical RCS (burnt first, at full mass: conservative), the rest on xenon."""
    margin = float(MARGIN) if margin is None else margin
    p = v.propulsion
    sub = _common_subsystems()
    arrays = array_sizing()
    dry_nom = (sum(float(x) for x in sub.values()) + arrays["array_mass"] + arrays["battery_kg"] + float(v.capture_kg)
               + float(p.hardware_kg) + float(p.rcs_hardware_kg))
    dry = dry_nom * (1 + margin)
    ve, f = float(p.isp) * ph.G0, float(p.tank_fraction)
    rcs_dv = float(p.rcs_dv)
    ve_r = float(p.rcs_isp) * ph.G0 if rcs_dv else 1.0
    f_r = 0.12                                                  # ASSUMED RCS tank fraction
    prop_r = 0.0
    for _ in range(200):                                        # fixed point: tank masses depend on propellant masses
        if wet_target is None:
            m_e = dry + f_r * prop_r
            x = math.exp((dv_total - rcs_dv) / ve) - 1
            prop = m_e * x / (1 - f * x)
            new_r = (m_e + f * prop + prop) * (math.exp(rcs_dv / ve_r) - 1) if rcs_dv else 0.0
        else:
            new_r = wet_target * (1 - math.exp(-rcs_dv / ve_r)) if rcs_dv else 0.0
            prop = (wet_target - dry - (1 + f_r) * new_r) / (1 + f)
        if abs(new_r - prop_r) < 1e-10:
            prop_r = new_r
            break
        prop_r = new_r
    tank = f * prop + f_r * prop_r
    wet = dry + tank + prop + prop_r
    m_start_main = wet - prop_r
    dv_main = ve * math.log(m_start_main / (m_start_main - prop))
    dv_cap = dv_main + (rcs_dv if rcs_dv else 0.0)
    return {"variant": v.key, "dry_nominal": dry_nom, "dry_with_margin": dry, "tanks": tank, "propellant": prop,
            "rcs_propellant": prop_r, "wet": wet, "dv_capability": dv_cap, "dv_main": dv_main,
            "array_area_m2": arrays["area"], "battery_wh": arrays["battery_wh"],
            "subsystems": {k: float(x) for k, x in sub.items()} | {"arrays": arrays["array_mass"], "battery": arrays["battery_kg"],
                                                                   "capture": float(v.capture_kg),
                                                                   "propulsion_hw": float(p.hardware_kg) + float(p.rcs_hardware_kg)},
            "within_mass_window": MASS_MIN <= wet <= MASS_MAX, "dv_requirement_met": dv_cap >= dv_total - 1e-6}


def array_sizing(p_eol: float = 1500.0, years: float = LIFE_YEARS, beta_deg: float = 0.0) -> dict:
    """Array area for the 1.5 kW EOL requirement and the battery for the worst eclipse (beta = 0)."""
    eff = assumed(0.29, "-", "triple-junction cell BOL efficiency (class-typical)", "cell_eff")
    pack = assumed(0.85, "-", "cell packing factor on the panel", "packing")
    degr = assumed(0.02, "1/yr", "LEO radiation + thermal cycling + UV degradation", "degr")
    area = ph.array_area(p_eol, float(eff), float(pack), float(degr), years, incidence_loss=0.08, temp_loss=0.10, other_loss=0.03)
    t_ecl = ph.eclipse_fraction(ALT_KM, beta_deg) * ph.period(ph.radius(ALT_KM))
    p_ecl = assumed(380, "W", "average load in eclipse during rendezvous operations (sensors + arm + GNC)", "p_eclipse")
    dod = assumed(0.30, "-", "depth of discharge for ~28k cycles in 5 years", "dod")
    wh = ph.battery_energy_wh(float(p_ecl), t_ecl, float(dod))
    return {"area": area, "array_mass": area * float(assumed(3.6, "kg/m2", "deployable rigid panels incl. hinges, yoke", "array_areal_mass")),
            "battery_wh": wh, "battery_kg": wh / float(assumed(120, "Wh/kg", "pack-level specific energy", "battery_sp_energy")),
            "eclipse_min": t_ecl / 60, "p_eol_w": p_eol}


def power_modes(v: Variant) -> dict:
    """Load per operating mode [W] against 1500 W EOL generation (sunlit); a mode that exceeds it is flagged."""
    base = {"cruise": 190, "rendezvous": 420, "capture": 480, "safe": 120}
    if v.propulsion.power_w:
        base["tow (electric thrusting)"] = 300 + float(v.propulsion.power_w)
    else:
        base["tow (chemical burn)"] = 300
    return {k: {"load_w": w, "margin_w": 1500 - w, "ok": w <= 1500 * 0.9} for k, w in base.items()}
