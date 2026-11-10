"""Odobenus v0: it sits on the bed and can still come up empty, two pickers beat a diver over two shifts, the pick
cycle (not the bed) sets the rate, the boat's power carries it, the price pays back in a season; the CAD builds."""
import pytest

from odobenus import REQUIREMENTS, Odobenus


def test_sits_lifts_and_pays():
    s = Odobenus.sizing()
    assert s["mass_in_air_kg"] < 40.0 and s["sink_kg"] > 0.5 and s["lift_margin"] > 0.5
    assert s["vs_diver"] > 1.5 and s["power_margin"] > 0
    assert s["price_eur"] < REQUIREMENTS["price_eur"] and s["payback_days"] < 60


def test_the_picker_is_the_limit():
    dense, sparse = Odobenus.sizing({"bed_density": 200.0}), Odobenus.sizing({"bed_density": 4.0})
    assert dense["limited_by"] == "picking" and sparse["limited_by"] == "crawling"
    assert dense["rate_per_h"] == pytest.approx(2 * 3600 / 6.0)
    assert Odobenus.sizing({"pickers": 4})["day_kg"] > dense["day_kg"]


def test_cad_builds():
    pytest.importorskip("cadquery")
    assert Odobenus().build(Odobenus.p()).val().isValid()
