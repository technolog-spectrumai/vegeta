"""Gyrus's heat, wear and test: the analysis runs from the geometry and its numbers are a 300 W thruster's."""
import pytest

pytest.importorskip("cadquery")

import gyrus_life as gl


def test_analysis():
    a = gl.analysis()
    assert a["budget"]["discharge"] == pytest.approx(300.0)
    assert 0.005 < a["areas"]["body_m2"] < 0.03
    t = a["temperatures"]
    assert t["coils_C"] > t["body_C"] and a["radiator"]["body_C"] < t["body_C"]
    assert 1000 < a["erosion"]["life_h"] < 30000
    assert 5000 < a["test_stand"]["pumping_speed_l_s"] < 50000
    assert len(a["plan"]) == 6
