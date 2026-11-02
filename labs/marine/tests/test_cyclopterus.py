"""Cyclopterus v0: every tool holds on the net with margin, a pen is inspected on one charge, morts are light in water,
the price stays under the target; the CAD builds where CadQuery is."""
import pytest

from cyclopterus import REQUIREMENTS, TOOLS, Cyclopterus


@pytest.mark.parametrize("tool", list(TOOLS))
def test_each_tool_holds_floats_and_fits_the_price(tool):
    s = Cyclopterus.sizing({"tool": tool})
    assert s["hold_margin"] > 0 and s["trim_kg"] > 0                  # holds on; lead (not foam) trims it
    assert s["unit_price_eur"] < REQUIREMENTS["price_eur"]
    assert s["shift_tool_work_h"] > 3.0


def test_a_pen_on_one_charge_and_light_morts():
    s = Cyclopterus.sizing()
    assert s["pens_per_charge"] >= 1.0
    assert s["morts_in_water_kg"] < 3.0
    assert s["price_with_all_tools_eur"] + s["dock_eur"] < REQUIREMENTS["price_eur"] + 1000


def test_cad_builds_with_each_tool():
    pytest.importorskip("cadquery")
    for tool in TOOLS:
        assert Cyclopterus().build(Cyclopterus.p({"tool": tool})).val().isValid()
