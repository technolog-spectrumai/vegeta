"""Pinna, the autonomous drone carrier: it floats with its load, the gensets carry the bed and the transit, the
deck's rates beat the packer, no crew aboard, the CAD builds."""
import pytest

from pinna import AUTONOMY, Pinna


def test_floats_and_powers():
    s = Pinna.sizing()
    assert 20 <= s["length_m"] <= 30 and s["draught_margin_t"] > 0
    assert s["gensets_margin"] > 0.3 and 12 < s["max_kn"] < 20 and s["endurance_days"] > 10


def test_deck_keeps_up_and_no_crew():
    s = Pinna.sizing()
    assert s["crew"] == 0 and {"radar", "ais", "colregs", "supervision"} <= set(AUTONOMY)
    assert s["air_kg_h_vtol_deck_limit"] > 200 and s["pack_kg_h"] >= 200 and s["baskets_per_h"] > 20
    assert s["deck_spots"]["vtol_pads"] == 4 and s["price_eur"] < 5e6


def test_cad_builds():
    pytest.importorskip("cadquery")
    assert Pinna().build(Pinna.p()).val().isValid()
