"""Dolphin v0: it floats neutral with ballast to spare and blows up to a dry deck, a day on the farm on one charge, the
pearls fly out faster than they come in, the drones fit the hangar, the price holds; the CAD builds."""
import pytest

from dolphin import EMERILLION, REQUIREMENTS, Dolphin


def test_buoyancy_energy_and_price():
    s = Dolphin.sizing()
    assert s["trim_submerged_kg"] > 20.0                                # ballast water aboard when neutral: it can always lighten
    assert s["ballast_blow_kg"] < s["trim_submerged_kg"] + 60.0
    assert s["energy_margin"] > 0.1
    assert s["price_eur"] < REQUIREMENTS["price_eur"]


def test_the_pearls_fly_out():
    s = Dolphin.sizing()
    assert s["hangar_fit"]
    assert s["flight_reserve"] > 0.2 and s["flight_min"] < 25
    assert s["cycle_h"] == pytest.approx(s["capsule_h"])                # the drones wait for pearls, not the other way round
    assert s["pearls_day"] == pytest.approx(Dolphin.p()["oysters_per_h"] * 12.0)
    assert Dolphin.sizing({"oysters_per_h": 1500.0})["cycle_h"] > Dolphin.sizing({"oysters_per_h": 1500.0})["capsule_h"]   # then the drones limit


def test_cad_builds():
    pytest.importorskip("cadquery")
    assert Dolphin().build(Dolphin.p()).val().isValid()
