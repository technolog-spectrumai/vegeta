"""Pinna, the autonomous drone carrier: it floats with its load, the gensets carry the bed and the transit, the
deck's rates beat the packer, no crew aboard, the CAD builds."""
import pytest

from pinna import AUTONOMY, DAVIT_AHC, ROBOTS, Pinna, davit_card, davit_compensation, davit_hs_limit


def test_floats_and_powers():
    s = Pinna.sizing()
    assert 20 <= s["length_m"] <= 30 and s["draught_margin_t"] > 0
    assert s["gensets_margin"] > 0.3 and 12 < s["max_kn"] < 20 and s["endurance_days"] > 10


def test_deck_keeps_up_and_no_crew():
    s = Pinna.sizing()
    assert s["crew"] == 0 and {"radar", "ais", "colregs", "supervision"} <= set(AUTONOMY)
    assert s["air_kg_h_vtol_deck_limit"] > 200 and s["pack_kg_h"] >= 200 and s["baskets_per_h"] > 20
    assert s["deck_spots"]["vtol_pads"] == 4 and s["price_eur"] < 5e6


def test_every_task_has_a_robot():
    for k in ("davit", "basket_gantry", "packing_cell", "crate_loader", "swap_robot", "deck_tug", "cleaner"):
        r = ROBOTS[k]
        assert r["count"] >= 1 and r["cycle_s"] > 0 and r["power_kW"] > 0 and r["price_eur"] > 0 and r["faults_per_day"] >= 0
    p = Pinna.p()
    assert ROBOTS["basket_gantry"]["cycle_s"] == p["basket_haul_s"] and ROBOTS["packing_cell"]["cycle_s"] == p["pack_s"]
    assert ROBOTS["crate_loader"]["payload_kg"] >= p["crate_kg"] and ROBOTS["davit"]["count"] == p["davits"]


def test_cad_builds():
    pytest.importorskip("cadquery")
    assert Pinna().build(Pinna.p()).val().isValid()


def test_davit_compensation():
    """The compensated davit works past the plain davit's 2 m, every axis inside its limits there, the residual a
    small fraction of the sheave's motion; over its limit an axis runs out."""
    lim = davit_hs_limit()
    assert 2.5 <= lim <= 3.5 and lim > DAVIT_AHC["uncompensated_hs_m"]
    c = davit_compensation(2.0)
    assert c["ok"] and c["residual_vertical_m"] < 0.1 * c["sheave_vertical_m"] + 0.06 and c["residual_horizontal_m"] < 0.15
    assert not davit_compensation(lim + 0.5)["ok"]
    card = davit_card()
    assert card["hs_limit_m"] == lim == Pinna.sizing()["davit_hs_limit_m"] and card["max_roll_deg"] > 8
    assert ROBOTS["davit"]["price_eur"] >= 38000 + card["compensation_eur"] - 1
