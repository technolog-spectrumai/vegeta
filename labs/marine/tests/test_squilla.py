"""Squilla v0: fits inside its hull, hovers with the lift, the work dive fits the battery with every vision set, the
default set is the best value of the full-score sets; the CAD builds where CadQuery is."""
import pytest

from squilla import DEFAULT_SENSORS, SENSOR_CONFIGS, Squilla


@pytest.mark.parametrize("sensors", list(SENSOR_CONFIGS))
def test_every_vision_set_fits_the_dive_and_the_jetson(sensors):
    s = Squilla.sizing({"sensors": sensors})
    assert s["fits"] and s["battery_margin"] > 0 and s["jetson_ok"]
    assert s["vertical_margin"] > 0 and s["surge_margin"] > 0


def test_default_vision_is_the_cheapest_set_scoring_eight():
    c = Squilla.compare_sensors()
    eights = sorted((v["price EUR"], k) for k, v in c.items() if v["score"] >= 8)
    assert eights[0][1] == DEFAULT_SENSORS


def test_squilla_is_shorter_and_lighter_than_nephro_with_two_arms():
    from nephro import Nephro
    assert Squilla.p()["length"] < Nephro.p()["length"]
    assert Squilla.sizing()["mass_in_air_kg"] < Nephro.sizing()["mass_in_air_kg"]


def test_cad_builds_in_both_poses():
    pytest.importorskip("cadquery")
    for pose in ("stowed", "work"):
        assert Squilla().build(Squilla.p({"arm_pose": pose})).val().isValid()
