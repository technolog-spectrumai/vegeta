"""Walrus v0: it sits on the bed and can still come up empty, two pickers beat a diver over two shifts, the pick
cycle (not the bed) sets the rate, the boat's power carries it, the price pays back in a season; the CAD builds."""
import pytest

from walrus import REQUIREMENTS, Walrus


def test_sits_lifts_and_pays():
    s = Walrus.sizing()
    assert s["mass_wet_vehicle_kg"] < 40.0 and s["sink_kg"] > 0.5         # the tether and reel stay on the boat and s["lift_margin"] > 0.5
    assert s["v_swim_m_s"] > 0.4 and s["transit_min"] < 15 and s["hold_current_margin"] > 0 and s["nav_error_m"] < 3.0
    assert s["vs_diver"] > 1.5 and s["power_margin"] > 0
    assert s["price_eur"] < REQUIREMENTS["price_eur"] and s["payback_days"] < 60
    assert 0.0 < s["dig_duty"] < 1.0 and s["dig_m3_h"] > 0                 # the arms keep up with the lane


def test_the_picker_is_the_limit():
    dense, sparse = Walrus.sizing({"bed_density": 200.0}), Walrus.sizing({"bed_density": 4.0})
    assert dense["limited_by"] == "picking" and sparse["limited_by"] == "crawling"
    assert dense["rate_per_h"] == pytest.approx(2 * 3600 / 6.0)
    assert Walrus.sizing({"pickers": 4})["day_kg"] > dense["day_kg"]


def test_cad_builds():
    pytest.importorskip("cadquery")
    assert Walrus().build(Walrus.p()).val().isValid()
    assert len(Walrus().build(Walrus.p({"part": "arms"})).solids().vals()) == 2
