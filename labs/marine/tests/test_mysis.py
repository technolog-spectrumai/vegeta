"""Mysis v0: a unit one person carries, pressing and holding on the hull with margin, a shift's worth of battery;
a group of six grooms a ship in a shift within the acoustic link's budget; the CAD builds where CadQuery is."""
import pytest

from mysis import REQUIREMENTS, Mysis


def test_the_unit():
    s = Mysis.sizing()
    assert s["mass_in_air_kg"] < 20.0 and 0.0 < s["trim_kg"] < 1.5
    assert s["press_margin"] >= 0 and s["hold_margin"] > 0 and s["liftoff_margin"] > 0
    assert s["press_W"] == 0.0                                       # the magnetic pad presses, not the thrusters
    assert s["groom_h"] > 1.0 and s["survey_h"] > 2.0
    assert s["unit_price_eur"] < REQUIREMENTS["unit_price_eur"] and s["node_price_eur"] < REQUIREMENTS["node_price_eur"]


@pytest.mark.parametrize("n", [1, 2, 4, 6])
def test_the_group_scales(n):
    g = Mysis.sizing({"group": n})
    assert g["link_margin"] > 0                                      # the modem carries the group's reports
    assert g["fix_period_s"] == n and g["drift_between_fixes_m"] < 0.1
    if n > 1:
        assert g["group_rate_m2_h"] > Mysis.sizing({"group": n - 1})["group_rate_m2_h"]


def test_a_ship_in_a_shift():
    g = Mysis.sizing({"group": 6})
    assert g["shift_h"] < 12.0
    assert Mysis.sizing({"group": 1})["shift_h"] > 4 * g["shift_h"]


def test_four_thrusters_against_six():
    four, six = Mysis.sizing({"thrusters": 4}), Mysis.sizing({"thrusters": 6})
    assert four["mass_in_air_kg"] < six["mass_in_air_kg"] and four["unit_price_eur"] < six["unit_price_eur"]
    assert four["hold_margin"] > 0                                   # the pad's grip holds the lane without sway
    assert four["liftoff_margin"] > 1                                # the clutch lets go: lift-off is easy


def test_cad_builds():
    pytest.importorskip("cadquery")
    assert Mysis().build(Mysis.p()).val().isValid()
