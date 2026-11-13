"""Shrimp v0: it fits the backpack and the price with every tool, trims with a little lead, holds in the pen's current
on a working tether, inspects a pen in a few dives; the CAD builds where CadQuery is."""
import pytest

from shrimp import REQUIREMENTS, TOOLS, Shrimp


@pytest.mark.parametrize("tool", list(TOOLS))
def test_each_tool_fits_the_pack_and_the_price(tool):
    s = Shrimp.sizing({"tool": tool})
    assert s["mass_in_air_kg"] < 5.0 and s["backpack_kg"] < 10.0
    assert 0.0 < s["trim_kg"] < 0.8                                   # a little lead, not foam
    assert s["kit_price_eur"] < REQUIREMENTS["price_eur"]
    assert s["dive_h"] > 1.5


def test_the_tether_sets_the_envelope():
    s = Shrimp.sizing()
    assert s["hold_margin"] > 0                                       # 0.5 m/s on 25 m of tether, just
    assert s["tether_drag_N"] > s["hull_drag_N"]                      # the tether, not the hull, is the load
    assert Shrimp.sizing({"tether_exposed_m": 50.0})["hold_margin"] < 0   # the full 50 m out in 0.5 m/s: no
    assert 20 < s["tether_max_at_0.5_m"] < 60
    assert 2 < s["dives_per_pen"] < 6
    assert 3 < s["profile_min"] < 10                                  # the daily oxygen profile is a short dive


def test_cad_builds_with_each_tool():
    pytest.importorskip("cadquery")
    for tool in TOOLS:
        assert Shrimp().build(Shrimp.p({"tool": tool})).val().isValid()
