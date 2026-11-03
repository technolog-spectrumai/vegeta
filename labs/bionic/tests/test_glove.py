"""The data glove's CAD: every part builds, it follows the hand model (size, side, pose), and carries one sensor pod
and one magnet per sensed joint of ``GloveSpec.default``."""
import pytest

pytest.importorskip("cadquery")

from vegeta.bionic.glove import GloveSpec

from glove import POD_AT, Glove

P = {q.name: q.default for q in Glove.parameters}


@pytest.mark.parametrize("part", ["fabric", "rings", "sensors", "magnets", "plate", "cuff", "cables"])
def test_every_part_builds(part):
    assert Glove().build({**P, "part": part}).val().isValid()


def test_one_pod_and_one_magnet_per_sensed_joint():
    sensed = GloveSpec.default().sensed
    assert set(sensed) == set(POD_AT)
    assert len(Glove().build({**P, "part": "sensors"}).solids().vals()) == len(sensed)
    assert len(Glove().build({**P, "part": "magnets"}).solids().vals()) == len(sensed)


def test_the_glove_follows_the_hand():
    d = Glove()
    small = d.build({**P, "part": "fabric", "hand_length": 165.0, "hand_breadth": 76.0, "pose": "flat"}).val().BoundingBox()
    large = d.build({**P, "part": "fabric", "hand_length": 205.0, "hand_breadth": 95.0, "pose": "flat"}).val().BoundingBox()
    assert large.xlen > small.xlen + 30
    right = d.build({**P, "part": "fabric", "side": "R"}).val().BoundingBox()
    left = d.build({**P, "part": "fabric", "side": "L"}).val().BoundingBox()
    assert right.ymax == pytest.approx(-left.ymin, abs=1.0)              # mirrored: the thumb on the other side
    flat = d.build({**P, "part": "fabric", "pose": "flat"}).val().BoundingBox()
    fist = d.build({**P, "part": "fabric", "pose": "fist"}).val().BoundingBox()
    assert fist.xlen < flat.xlen - 40                                      # the fingers curl in
