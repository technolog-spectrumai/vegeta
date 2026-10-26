"""Strato-Micro (Stage V8.19): the maritime aerostat and micro blimp — lift at sea level, the tether's lean, the
sizing search, the free-flyer's power, and the wind statistics."""
import math
from dataclasses import replace

import pytest

import strato_micro as mi


def test_sea_level_lift_per_volume():
    r = mi.evaluate(6.0, mi.MicroDesign())
    assert r["net_lift"] / r["volume"] == pytest.approx(1.03, abs=0.04)    # ~1.0–1.06 kg/m³ of helium at sea level


def test_tether_leans_less_with_more_lift():
    d = mi.MicroDesign()
    small, big = mi.evaluate(6.0, d), mi.evaluate(9.0, d)
    assert big["tether_angle"] < small["tether_angle"]
    calm = mi.evaluate(6.0, d, replace(mi.Mission(), design_wind=4.0))
    assert calm["tether_angle"] < small["tether_angle"]


def test_too_big_overloads_the_tether():
    r = mi.evaluate(40.0, mi.MicroDesign())
    assert r["tether_sf"] < 5 and not r["feasible"]


def test_sizing_finds_the_threshold():
    d = mi.MicroDesign()
    r = mi.size(d)
    assert r["feasible"]
    assert not mi.evaluate(r["length"] - 0.05, d)["feasible"]
    assert r["tether_angle"] == pytest.approx(d.max_tether_angle, abs=0.5)


def test_free_flyer_power_grows_steeply_with_wind():
    d = mi.MicroDesign(concept="free", fin_scale=0.7)
    a = mi.evaluate(7.0, d, replace(mi.Mission(), design_wind=8.0))
    b = mi.evaluate(7.0, d, replace(mi.Mission(), design_wind=16.0))
    assert b["propulsion_w"] / a["propulsion_w"] > 4.0                        # ~V³ (19/11)³ ≈ 5


def test_weibull_operability():
    w = mi.SeaWind()
    assert w.fraction_below(1e-6, 200.0) == pytest.approx(0.0, abs=1e-9)
    assert w.fraction_below(100.0, 200.0) == pytest.approx(1.0)
    median_10 = w.mean_10m / math.gamma(1 + 1 / w.k) * math.log(2) ** (1 / w.k)
    assert w.fraction_below(median_10, 10.0) == pytest.approx(0.5)


def test_cheapest_is_reported():
    c = mi.cheapest()
    assert c["best"]["feasible"] and c["best"]["first_year_cost"] == min(
        r["first_year_cost"] for r in c["rows"] if r.get("feasible"))
