"""The skewed propeller: same blade, sections swept back about the axis."""
import math

import pytest

from vegeta.dedalus.examples import Propeller

KW = dict(diameter=120.0, pitch=100.0, blades=1, hub_diameter=24.0, hub_height=16.0, bore=8.0, chord_root=18.0, chord_max=30.0,
          chord_tip=12.0, thickness=0.12, camber=0.05, stations=8)


def test_skew_sweeps_the_tip_back_and_keeps_the_volume():
    straight = Propeller().generate(**KW)
    skewed = Propeller().generate(**KW, skew_deg=30.0)
    assert skewed.measure()["valid"] and skewed.volume == pytest.approx(straight.volume, rel=0.02)
    (lo0, hi0), (lo1, hi1) = straight.bounding_box, skewed.bounding_box
    # the tip (at x ~ R) moves round toward -y: the blade reaches further to -y, and less far along +x
    assert lo1[1] < lo0[1] - 0.25 * 60 * math.sin(math.radians(30.0)) and hi1[0] < hi0[0]
    with pytest.raises(ValueError):
        Propeller().generate(**KW, skew_deg=90.0)
