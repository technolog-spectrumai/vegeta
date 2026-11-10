"""Walrus v0: it sits on the bed and can still come up empty, two pickers beat a diver over two shifts, the pick
cycle (not the bed) sets the rate, the boat's power carries it, the price pays back in a season; the CAD builds."""
import pytest

from walrus import REQUIREMENTS, Walrus


def test_sits_lifts_and_pays():
    s = Walrus.sizing()
    assert s["mass_wet_vehicle_kg"] < 40.0 and s["sink_kg"] > 0.5         # the tether and reel stay on the boat and s["lift_margin"] > 0.5
    assert s["v_swim_m_s"] > 0.4 and s["transit_min"] < 15 and s["hold_current_margin"] > 0 and s["nav_error_m"] < 3.0
    assert s["vs_diver"] > 1.5 and s["power_margin"] > 0
    assert s["price_eur"] < 1.02 * REQUIREMENTS["price_eur"] and s["payback_days"] < 60   # round 2's thrusters: EUR 0.4k over
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


def test_thrusters_big_slow_and_full_control():
    import numpy as np
    from walrus import LAYOUT, THRUSTER_V0, allocation, compare_layouts, prop, thruster_layout
    p = Walrus.p()
    c = compare_layouts(p)
    assert LAYOUT == "C" and c["C"]["rank"] == 6 and c["C"]["fits"]
    assert np.linalg.matrix_rank(np.array(allocation(p))) == 6
    assert c["C"]["min_force_N"] > 30 and c["A"]["rank"] == 4 and c["B"]["rank"] == 5
    assert c["C"]["lift30_W"] <= min(c["A"]["lift30_W"], c["B"]["lift30_W"])
    new = prop(thruster_layout(p)[0])
    assert new["N_per_W"] > 2 * prop(THRUSTER_V0)["N_per_W"] and new["rpm"] < 2000
    s = Walrus.sizing()
    assert s["lift_margin"] > 0.5 and s["v_swim_m_s"] > 0.4
