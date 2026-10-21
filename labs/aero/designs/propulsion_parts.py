"""The aero machines' propellers as Yax part data (``vegeta.labkit.propulsion``): what ``designs/machines.py`` hands
``tools/export.py`` so a machine leaves with its propeller maps — Drongo's four rotors from notebook 08's Boreas system,
NISUS's pusher from its Ogma propulsion map, NISUS+'s and FALCO's from their electric drives — in the layers Yax
carries (geometry, CT/CP against J, the grid, the motor, the reduced two-constant model).

Each function takes the Yax ``Machine`` the export built (for the root link's pose) and returns the machine's part
data; Drongo keeps its ``ogma.rotors`` record as Piccolo's ``drako_drongo`` map reads it (``vehicles/drongo.map.json``),
``propulsion.reduced`` holds the same constants.

The spin signs of the single-propeller aircraft are **assumed** (``SPIN``): the notebooks apply thrust without a reaction
torque, and which way the bought propeller turns is a choice they never made. Yax's convention: +1 when the drag torque
on the frame is along +axis; a right-hand tractor propeller seen from behind turns clockwise, so its reaction on the
airframe is about −x: −1.
"""
from __future__ import annotations

import math

import numpy as np

from vegeta.labkit import propulsion as lp

from ogma.propulsion import RHO0

#: Assumed spin of the single-propeller aircraft (see the module): a right-hand propeller on a forward axis.
SPIN = {"nisus": -1, "nisus_plus": -1, "falco": -1}
SPIN_SOURCE = "spin assumed: a right-hand propeller (APC E/P) on the forward axis, reaction torque about −x"


def _axis(downthrust_deg: float) -> np.ndarray:
    """The thrust line in the Chiron body frame: forward, tilted nose-down by the down-thrust (as ``nisus_robot.Aero``)."""
    d = math.radians(downthrust_deg)
    return np.array([math.cos(d), 0.0, -math.sin(d)])


# ------------------------------------------------------------------------------------------------------------ Drongo
def drongo(m) -> dict:
    """Drongo's rotors: positions, spins and the motor-model constants for Gazebo (``ogma.rotors``, as before) and the
    full ``propulsion`` block — the 5x4.3 tri-blade's BEMT map on notebook 08's section, the 2306 on 4S, the reduced
    constants that ``ogma.rotors`` carries, checked against the map at hover."""
    import drongo_robot as dr
    from vegeta import boreas
    prop, airfoil, motor, battery = dr.boreas_system()
    p = dr.propulsion()
    pos, spin = dr.rotor_layout()
    g = dr.geometry()
    w_max = 0.8 * motor.kv_rpm_per_volt * battery.voltage * 2 * math.pi / 60
    kg = dr.PARTS_G["propellers 5x4.3 (4x)"] / 4 / 1000.0
    r = g["prop_r"]
    rotors = []
    for q, s in zip(np.asarray(pos, float), np.asarray(spin, float)):
        pw, _ = lp.in_machine_frame(m, q, (0.0, 0.0, 1.0))
        rotors.append({"pos": pw, "spin": int(s), "kg": kg, "inertia": [kg * r * r / 4, kg * r * r / 4, kg * r * r / 2], "radius": r,
                       "kF": p.max_thrust_N / w_max ** 2, "kM": p.torque_per_thrust_m, "w_max": w_max,
                       "tau_up": p.spin_up_s, "tau_down": 2 * p.spin_up_s, "T_max": p.max_thrust_N})
    ogma = {"rotors": rotors, "source": f"{p.source}; ω_max = 0.8 × KV × V ({motor.kv_rpm_per_volt:g} KV, {battery.cells}S; assumed)"}

    rho = RHO0
    grid = lp.grid_from_boreas(prop, airfoil, np.linspace(6000.0, 30000.0, 9), [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0], rho)
    entry = lp.propeller_entry(prop, grid, inertia_kgm2=kg * r * r / 2,
                               source=f"notebook 08: Boreas BEMT, {airfoil.name} ({airfoil.source}); {prop.notes}")
    P, rpm_full = lp.max_shaft_power(grid, battery_v=battery.voltage, i_max=motor.max_current_a, voltage_for=motor.voltage_for)
    mot = lp.motor_entry(motor.name, P, kv=motor.kv_rpm_per_volt, R_ohm=motor.resistance_ohm, I0_A=motor.no_load_current_a,
                         I_max_A=motor.max_current_a, battery_v=battery.voltage, esc_eff=0.96,
                         source=f"notebook 08's motor ({motor.source}); power_w: the map's largest shaft power within the pack and the current limit")
    hover = m.total_mass() * 9.81 / len(rotors)
    rpm_h = lp.rpm_for_thrust(grid, hover)
    red = {}
    yr = []
    for k, (q, s) in enumerate(zip(rotors, np.asarray(spin, float))):
        name = f"rotor{k}"
        yr.append(lp.rotor(name, q["pos"], (0.0, 0.0, 1.0), int(s), entry["name"], mot["name"], link=f"prop{k}"))
        red[name] = lp.reduced(grid, rpm=w_max * 60 / (2 * math.pi), taken_at=f"full throttle (T_max {p.max_thrust_N:g} N at ω_max, ogma.rotors)",
                               tau_up=p.spin_up_s, tau_down=2 * p.spin_up_s, check=(rpm_h, 0.0), kF=q["kF"], kM=q["kM"])
    blk = lp.block(yr, [entry], [mot], reduced=red,
                   battery={"name": battery.name, "cells": battery.cells, "capacity_ah": battery.capacity_ah, "usable_fraction": battery.usable_fraction,
                            "voltage_v": battery.voltage},
                   source="rotors: drongo_robot.rotor_layout; map: notebook 08's Boreas system (designs/drongo_robot.boreas_system); "
                          f"reduced: ogma.rotors ({p.source}); residual checked at hover ({hover:.2f} N per rotor)")
    return {"ogma": ogma, "propulsion": blk}


# ------------------------------------------------------------------------------------------------- the fixed wings
def _single(m, *, name: str, robot, prop, grid: dict, motor: dict, battery_v: float, i_max: float | None, pos_body, downthrust_deg: float,
            tau_up: float | None, source: str, prop_source: str, power_limit: tuple | None = None) -> dict:
    """One propeller on the thrust line: ``power_limit = (P, rpm)`` is the motor's full-throttle point (from a grid
    with the motor's voltage and current columns; by default this grid's)."""
    axis_b = _axis(downthrust_deg)
    pos_w, axis_w = lp.in_machine_frame(m, pos_body, axis_b)
    P, rpm_full = power_limit or lp.max_shaft_power(grid, battery_v=battery_v, i_max=i_max)
    mot = lp.motor_entry(motor["name"], P, kv=motor.get("kv"), R_ohm=motor.get("R"), I0_A=motor.get("I0"), I_max_A=i_max, battery_v=battery_v,
                         esc_eff=motor.get("esc_eff"), source=motor.get("source", "") + "; power_w: the map's largest shaft power at V = 0 within the pack")
    entry = lp.propeller_entry(prop, grid, source=prop_source)
    cruise = float(np.asarray(grid["airspeed_m_s"], float)[len(grid["airspeed_m_s"]) // 2])
    red = {"prop": lp.reduced(grid, rpm=rpm_full, taken_at=f"static, full throttle ({rpm_full:g} rpm)", tau_up=tau_up,
                              tau_down=(2 * tau_up) if tau_up else None, check=(rpm_full, cruise))}
    rot = lp.rotor("prop", pos_w, axis_w, SPIN[name], entry["name"], mot["name"], link=robot.root.name)
    return {"propulsion": lp.block([rot], [entry], [mot], reduced=red, source=f"{source}; {SPIN_SOURCE}"),
            "tags": {"data:propulsion/propellers*": "boreas", "data:propulsion/motors*": "fitted",     # the design's own tags
                     "data:propulsion/rotors*": "assumed", "data:propulsion/reduced*": "assumed"}}


def nisus(m) -> dict:
    """NISUS's 9x6 pusher: the Ogma propulsion map (``data/nisus_propulsion.json``: BEMT on the fitted X2216, thrust
    calibrated at the published static point), the propeller planform, the motor's fit; reduced at static full
    throttle, the residual at the map's middle airspeed."""
    import nisus_robot as nr
    import nisus_systems as ns
    robot = nr.nisus_robot()
    p, L = robot.params, robot.layout
    pm = ns.propulsion_map()
    fit = pm.notes.get("fit", {})
    motor = {"name": pm.motor_name, "kv": ns.PUBLISHED_STATIC["kv"], "R": fit.get("resistance_ohm"), "I0": fit.get("no_load_a"),
             "esc_eff": pm.esc_efficiency, "source": pm.notes.get("motor", "")}
    return _single(m, name="nisus", robot=robot, prop=ns.propeller_9x6(), grid=lp.grid_from_map(pm, RHO0), motor=motor, battery_v=pm.battery_v,
                   i_max=None, pos_body=nr.to_body(L["prop_x"], 0.0, p["motor_z"]), downthrust_deg=p["motor_downthrust_deg"],
                   tau_up=nr.Actuators.tau_throttle,
                   source="nisus_systems.propulsion_map (Boreas BEMT + fitted X2216); thrust line: nisus_robot.Aero (prop_x, motor_z, down-thrust)",
                   prop_source=f"{pm.notes.get('propeller', '')}; section: {pm.notes.get('section', '')}; {pm.notes.get('limits', '')}")


def _drive_machine(m, *, name: str, robot, dr, prop, pos_body, downthrust_deg: float, tau_up: float | None, source: str) -> dict:
    """A machine on an Ogma ``ElectricDrive`` (NISUS+, FALCO): the drive's sea-level thrust and torque tables as the
    grid, its fitted motor, the map at sea level for the power limit."""
    pm = dr.map_at(RHO0)
    grid = lp.grid_from_drive(dr)
    gm = lp.grid_from_map(pm, RHO0)                                            # voltage and current: the power limit
    limit = lp.max_shaft_power(gm, battery_v=dr.battery_v, i_max=dr.max_current_a)
    motor = {"name": dr.motor_name, "kv": dr.kv, "R": dr.R, "I0": dr.I0, "esc_eff": dr.esc_efficiency, "source": dr.notes.get("motor", "")}
    notes = dr.notes
    return _single(m, name=name, robot=robot, prop=prop, grid=grid, motor=motor, battery_v=dr.battery_v, i_max=dr.max_current_a, pos_body=pos_body,
                   downthrust_deg=downthrust_deg, tau_up=tau_up, source=source, power_limit=limit,
                   prop_source=f"{notes.get('propeller', '')}; section: {notes.get('section', '')}; {notes.get('brake region', '')}; {notes.get('limits', '')}")


def nisus_plus(m) -> dict:
    """NISUS+'s 15x8 pusher on the AT4125 (``data/nisus_plus_propulsion.json``)."""
    import nisus_plus_robot as npr
    import nisus_plus_systems as nps
    import nisus_robot as nr
    robot = npr.nisus_plus_robot()
    p, L = robot.params, robot.layout
    return _drive_machine(m, name="nisus_plus", robot=robot, dr=nps.drive(), prop=nps.propeller_15x8(), pos_body=nr.to_body(L["prop_x"], 0.0, p["motor_z"]),
                          downthrust_deg=p["motor_downthrust_deg"], tau_up=npr.NisusPlusActuators.tau_throttle,
                          source="nisus_plus_systems.drive (Boreas BEMT on the 15x8, the fitted AT4125); thrust line: nisus_robot.Aero (prop_x, motor_z, down-thrust)")


def falco(m) -> dict:
    """FALCO's 20x15 tractor on the AT5220-A (``data/falco_propulsion.json``)."""
    import falco_robot as fr
    import falco_systems as fsy
    import nisus_robot as nr
    robot = fr.falco_robot()
    p, L = robot.params, robot.layout
    return _drive_machine(m, name="falco", robot=robot, dr=fsy.drive(), prop=fsy.propeller(), pos_body=nr.to_body(L["prop_x"], 0.0, L["motor_z"]),
                          downthrust_deg=p["motor_downthrust_deg"], tau_up=fr.NisusPlusActuators.tau_throttle,
                          source="falco_systems.drive (Boreas BEMT on the 20x15, the fitted AT5220-A KV220); thrust line: falco_robot (prop_x, motor_z, down-thrust)")
