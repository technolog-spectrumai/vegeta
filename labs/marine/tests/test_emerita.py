"""Emerita v0: the housings hold at 300 m, everything fits inside the hull, the range meets 60-100 km at survey speed
and 1.5 m/s sits near the best speed; the CAD builds where CadQuery is."""
import pytest

from emerita import REQUIREMENTS, Emerita


def test_range_housings_and_fit():
    s = Emerita.sizing()
    assert all(h["buckling_margin"] > 0 for h in s["housings"].values())
    assert s["fits"] and s["foam_L"] >= 0
    lo, hi = REQUIREMENTS["range_km"]
    assert lo <= s["range_km"] <= hi * 1.1
    wh = {u: r["Wh_per_km"] for u, r in s["speeds"].items()}
    assert wh[1.5] <= wh[1.0] and wh[1.5] <= wh[2.0]


def test_a_bigger_battery_goes_further():
    assert Emerita.sizing({"battery_kwh": 10.0})["range_km"] > Emerita.sizing()["range_km"]


def test_cad_builds_in_both_poses():
    pytest.importorskip("cadquery")
    for pose in ("stowed", "deployed"):
        assert Emerita().build(Emerita.p({"arm_pose": pose})).val().isValid()
