"""H5: the power budget closes, the temperatures are a small thruster's, the erosion life is the class's, the test
stand's pumping speed is a real chamber's."""
import pytest

from vegeta.fulgur.hall import point_for_power, scale_channel
from vegeta.fulgur.hall_life import ThermalBody, erosion, facility, magnet_trade, power_budget, radiator_for, sputter_yield_bn, temperatures, total_impulse


def test_budget_closes():
    pt = point_for_power(300.0, 280.0)
    b = power_budget(pt)
    assert sum(v for k, v in b.items() if k != "discharge") == pytest.approx(b["discharge"], rel=1e-9)
    assert 0.5 < b["beam"] / b["discharge"] < 0.8


def test_temperatures_in_vacuum():
    pt = point_for_power(300.0, 280.0)
    ch = scale_channel(300.0, 280.0)
    tb = ThermalBody(ch, body_area=0.012, channel_front_area=0.003, coil_power=1.4)
    t = temperatures(pt, tb)
    assert 150 < t["body_C"] < 450 and t["channel_C"] > t["body_C"] and t["anode_C"] > t["body_C"]
    r = radiator_for(pt, tb, 190.0)
    assert r["area_m2"] > tb.body_area and r["body_C"] < 190.0


def test_erosion_and_life_are_the_class_s():
    ch = scale_channel(300.0, 280.0)
    assert sputter_yield_bn(40.0) == 0.0 and sputter_yield_bn(300.0) == pytest.approx(0.07, rel=1e-6)
    e = erosion(ch, wall_thickness=2e-3)
    assert 0.1 < e["rate_mm_per_kh"] < 5.0                     # the SPT class: ~1 mm per 1000 h early in life
    assert 500 < e["life_h"] < 20000
    pt = point_for_power(300.0, 280.0)
    imp = total_impulse(pt, e["life_h"])
    assert imp["total_impulse_Ns"] > 1e4


def test_magnets_and_test_stand():
    m = magnet_trade(293.0, 0.013, 0.041, 0.11, 1.4)
    assert 0.5 < m["magnet_length_mm"] < 5.0 and m["saved_power_W"] == 1.4
    pt = point_for_power(300.0, 280.0)
    ts = facility(pt)
    assert 5000 < ts["pumping_speed_l_s"] < 100000           # a cryopumped chamber, not a turbo on a bell jar
