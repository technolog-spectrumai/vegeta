"""Ducted fan model: the ideal ducted actuator-disk limit, loss trends, the 90 mm EDF of notebook 25, nacelle drag."""
import json
import math
from dataclasses import replace

import numpy as np
import pytest

from vegeta import boreas
from vegeta.boreas import ducted

RHO = 1.225


def edf_rotor(blades: int = 12, n: int = 10) -> boreas.Propeller:
    """90 mm rotor, hub/tip 0.45, constant geometric pitch 160 mm, chord 16 -> 14 mm (notebook 25's EDF)."""
    R = 0.045
    r = np.linspace(0.45 * R, R, n)
    beta = np.degrees(np.arctan(0.160 / (2 * math.pi * r)))
    return boreas.Propeller("EDF 90 mm", blades, tuple(r), tuple(np.linspace(0.016, 0.014, n)), tuple(beta))


SECTION = boreas.Airfoil(cl_alpha=2 * math.pi * 0.9, alpha0_deg=-3.0, cl_max=1.1, cd0=0.018, k=0.04)
FAN = ducted.DuctedFan("EDF 90 mm, 7 stators", edf_rotor(), tip_clearance_m=0.0007, exit_area_ratio=0.9, stator_vanes=7)


def test_ideal_limit_is_the_ducted_actuator_disk():
    # no profile drag (and no stall: cl_max out of reach), no duct, stator or clearance loss
    af = boreas.Airfoil(cl_alpha=2 * math.pi * 0.9, alpha0_deg=-3.0, cl_max=5.0, cd0=0.0, k=0.0)
    fan = replace(FAN, duct_loss=0.0, stator_loss=0.0, tip_clearance_m=0.0)
    for V in (0.0, 20.0):
        op = ducted.solve(fan, af, 25000.0, V)
        assert op.converged and op.thrust > 0
        assert op.power == pytest.approx(op.thrust * (op.exit_velocity + V) / 2, rel=1e-3)          # energy = jet KE rise
        assert op.fan_velocity == pytest.approx(0.9 * op.exit_velocity) and op.mass_flow == pytest.approx(RHO * fan.fan_area * op.fan_velocity)
    hover = ducted.solve(fan, af, 25000.0)
    sigma = fan.exit_area_ratio
    assert hover.power == pytest.approx(0.5 * hover.thrust**1.5 / math.sqrt(RHO * sigma * fan.fan_area), rel=1e-3)
    assert hover.figure_of_merit == pytest.approx(math.sqrt(2 * sigma), rel=1e-3)                    # above 1: the duct
    # sigma = 1: 1/sqrt(2) of the ideal open rotor of the same area; the blades carry at most half the thrust
    one = ducted.solve(replace(fan, exit_area_ratio=1.0), af, 25000.0)
    open_rotor = one.thrust**1.5 / math.sqrt(2 * RHO * fan.fan_area)
    assert one.power == pytest.approx(open_rotor / math.sqrt(2), rel=1e-3)
    assert 0.3 < one.rotor_thrust / one.thrust <= 0.5 + 1e-9 and one.duct_thrust == pytest.approx(one.thrust - one.rotor_thrust)


def test_stator_raises_efficiency():
    plain = replace(FAN, stator_vanes=0)
    with_stator, without = (ducted.rpm_for_thrust(f, SECTION, 4.64, 20.0) for f in (FAN, plain))
    assert with_stator.efficiency > without.efficiency + 0.01
    assert ducted.solve(FAN, SECTION, 25000.0).figure_of_merit > ducted.solve(plain, SECTION, 25000.0).figure_of_merit


def test_tip_clearance_lowers_efficiency():
    tight, loose = (ducted.rpm_for_thrust(replace(FAN, tip_clearance_m=c), SECTION, 4.64, 20.0) for c in (0.0005, 0.0015))
    assert loose.efficiency < tight.efficiency and loose.power > tight.power
    # ~2 % of the rise per 1 % clearance: 1 mm more on 24.75 mm of blade costs several per cent of efficiency
    assert 0.02 < 1 - loose.efficiency / tight.efficiency < 0.15


def test_thrust_rises_with_rpm_and_rpm_for_thrust_inverts_solve():
    for V in (0.0, 20.0):
        T = [ducted.solve(FAN, SECTION, n, V).thrust for n in (8000.0, 15000.0, 25000.0, 35000.0)]
        assert np.all(np.diff(T) > 0)
    op = ducted.solve(FAN, SECTION, 23000.0, 10.0)
    back = ducted.rpm_for_thrust(FAN, SECTION, op.thrust, 10.0)
    assert back.rpm == pytest.approx(23000.0, rel=1e-3) and back.thrust == pytest.approx(op.thrust, rel=1e-3)
    with pytest.raises(ValueError, match="not reachable"):
        ducted.rpm_for_thrust(FAN, SECTION, 500.0, rpm_max=30000.0)
    slow = ducted.solve(FAN, SECTION, 2000.0, 30.0)                  # windmilling: negative thrust, no crash
    assert slow.converged and slow.thrust < 0 and slow.exit_velocity < 30.0 and slow.efficiency <= 0.0


def test_90mm_edf_of_notebook_25():
    static = ducted.rpm_for_thrust(FAN, SECTION, 22.0)
    assert static.converged and 15000 < static.rpm < 45000 and static.tip_mach < 0.5
    # with the default (clean) losses the fan reaches ~80 % of the ideal duct, sqrt(2 sigma) = 1.34
    assert 0.4 < static.figure_of_merit < math.sqrt(2 * FAN.exit_area_ratio)
    # a hobby-grade unit: sharp static inlet, struts, stator profile and tail cone lumped into duct_loss, 0.9 mm gap
    hobby = ducted.rpm_for_thrust(replace(FAN, duct_loss=0.25, tip_clearance_m=0.0009), SECTION, 22.0)
    assert 15000 < hobby.rpm < 45000 and 0.4 < hobby.figure_of_merit < 1.0
    cruise = ducted.rpm_for_thrust(FAN, SECTION, 4.64, 20.0)
    assert cruise.converged and 0 < cruise.efficiency < 2 * 20.0 / (20.0 + cruise.exit_velocity)
    d = cruise.to_dict()
    json.dumps(d)
    assert d["converged"] is True and len(d["radial"]["dp0"]) == 30 and d["thrust"] == pytest.approx(4.64, rel=1e-3)


def test_nacelle_drag():
    fan = replace(FAN, external_wetted_area_m2=0.035, duct_length_m=0.15)
    d = ducted.nacelle_drag(fan, 20.0)
    re = 20.0 * 0.15 / 1.5e-5
    assert d == pytest.approx(0.5 * RHO * 20.0**2 * 0.455 / math.log10(re) ** 2.58 * 0.035) and d > 0
    assert ducted.nacelle_drag(fan, 0.0) == 0.0 and ducted.nacelle_drag(FAN, 20.0) == 0.0
    assert ducted.nacelle_drag(fan, 40.0) > 3 * d                    # ~V^2, Cf falling slowly with Re


def test_validation():
    rotor = edf_rotor()
    for bad in ({"exit_area_ratio": 0.0}, {"duct_loss": -0.1}, {"stator_loss": -0.1}, {"stator_vanes": -1},
                {"tip_clearance_m": -1e-4}, {"tip_clearance_m": 0.03}, {"mass_kg": -1.0},
                {"external_wetted_area_m2": 0.02, "duct_length_m": 0.0}):
        with pytest.raises(ValueError):
            ducted.DuctedFan("bad", rotor, **bad)
    with pytest.raises(ValueError, match="Propeller"):
        ducted.DuctedFan("bad", "rotor")
    fan = ducted.DuctedFan("ok", rotor)
    assert fan.blade_height == pytest.approx(0.55 * 0.045) and fan.exit_area == pytest.approx(0.9 * fan.fan_area)
    assert fan.fan_area == pytest.approx(math.pi * 0.045**2 * (1 - 0.45**2))
    with pytest.raises(ValueError):
        ducted.solve(fan, SECTION, 0.0)
    with pytest.raises(ValueError):
        ducted.solve(fan, SECTION, 20000.0, -1.0)
    with pytest.raises(ValueError):
        ducted.nacelle_drag(fan, -1.0)
