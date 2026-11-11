"""Perseus's chassis with other manipulators and a second hand (designs/perseus_options.py): the findings the
investigation's document states, held as tests.

Run: python tools/lab.py test land -- -k perseus_options
"""
import math

import pytest

import perseus_options as po
import perseus_robot as pbr


@pytest.fixture(scope="module")
def rows():
    return {c: po.evaluate(c) for c in po.CONFIGS}


def test_baseline_is_perseus(rows):
    assert rows["boom saw"]["mass_kg"] == pytest.approx(pbr.total_mass())
    assert rows["boom saw"]["added_kg"] == pytest.approx(0.0)


def test_boom_grab_is_wrist_then_stability_limited():
    a = po.boom_lift(1.4)
    assert a["limit"] == "drives" and a["capacity_kg"] == pytest.approx(22.3, abs=0.5)
    s = po.boom_lift(1.4, v=po.STRONG_WRIST)
    assert s["arm_kg"] > 2.5 * a["arm_kg"] and s["limit"] == "stability"
    assert po.boom_lift(2.1)["limit"] == "stability"


def test_no_good_second_hand_on_this_chassis(rows):
    light, mid, long_ = rows["boom saw + light hand"], rows["boom saw + arm6 grab"], rows["boom saw + long arm6 grab"]
    assert light["shared"] == 0.0                                   # cannot reach the cut
    assert mid["shared"] < 0.25 and mid["tip_margin_m"] < 0.10      # reaches little, nearly tips
    assert long_["shared"] > 0.4 and long_["tip_margin_m"] < 0.0    # reaches, tips
    assert mid["grade_cont_deg"] < rows["boom saw"]["grade_cont_deg"]


def test_boom_reaches_more_of_the_cut_zone_than_the_joint_arm(rows):
    assert rows["boom saw"]["cut_zone_reached_by_cutter"] > rows["arm6 saw"]["cut_zone_reached_by_cutter"]
    assert rows["arm6 saw"]["mass_kg"] < rows["boom saw"]["mass_kg"]


def test_bottom_cuts_pinch():
    assert all(c["kerf_bottom_cut"] == "pinches" for c in po.branch_cases())
