"""Gannet (the cargo quad) and Fulmar (the long-haul fixed-wing) for the drone carrier: they lift the crate, reach
the shore station / the port with the reserve, the fixed-wing is the cheaper per km, the deck limits are sane, the
CAD builds."""
import pytest

from fulmar import Fulmar
from gannet import SURVEY, Gannet


def test_gannet_lifts_the_crate_and_reaches_shore():
    s = Gannet.sizing()
    assert s["mtow_kg"] < 20 and s["thrust_weight"] > 2.0
    assert s["radius_km"] > 8.0                       # a shore station a few km off, with the reserve kept
    assert s["hover_W"] > s["cruise_W"] > s["cruise_empty_W"]
    assert 1.5 <= s["deck_hs_limit_m"] <= 3.0 and s["wind_limit_m_s"] >= 10
    assert any("kestrel" in k for k in SURVEY)


def test_fulmar_is_the_long_leg():
    f, g = Fulmar.sizing(), Gannet.sizing()
    assert f["radius_10ms_headwind_km"] > 40 and f["wh_per_km"] < g["wh_per_km"] / 2.5
    assert f["catapult_stroke_m"] < 4.0 and f["stall_m_s"] < 16
    assert Fulmar.sizing({"payload_kg": 0.6})["range_km"] > f["range_km"]   # the pearl case


def test_cad_builds():
    pytest.importorskip("cadquery")
    assert Gannet().build(Gannet.p()).val().isValid()
    assert Fulmar().build(Fulmar.p()).val().isValid()
