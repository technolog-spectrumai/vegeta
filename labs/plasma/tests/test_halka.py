"""Halka's CAD: every part builds for every preset, the rings match the fulgur preset, the masses sit near the
video's scale readings, and the layout scales."""
import pytest

pytest.importorskip("cadquery")

from vegeta.fulgur.thruster import PRESETS

from halka import Halka

P = {q.name: q.default for q in Halka.parameters}


@pytest.mark.parametrize("preset", list(PRESETS))
@pytest.mark.parametrize("part", ["frame", "grounds", "emitters", "supports", "fasteners"])
def test_every_part_builds(preset, part):
    s = Halka().build({**P, "preset": preset, "part": part})
    assert all(v.isValid() for v in s.solids().vals()) and len(s.solids().vals()) >= 1


def test_the_rings_are_the_fulgur_preset():
    for name, t in PRESETS.items():
        p = {**P, "preset": name}
        n_g = len(Halka().build({**p, "part": "grounds"}).solids().vals())
        n_e = len(Halka().build({**p, "part": "emitters"}).solids().vals())
        assert n_g == sum(1 for r in t.rings if r.role == "ground")
        assert n_e == sum(1 for r in t.rings if r.role == "emitter")
        bb = Halka().build({**p, "part": "grounds"}).val().BoundingBox()
        assert bb.xmax == pytest.approx(max(r.radius for r in t.rings) * 1000, abs=P["ground_w"] / 2 + 0.5)


def test_the_teeth_are_drawn_when_asked():
    plain = Halka().build({**P, "preset": "mark4", "part": "emitters", "teeth": "none"})
    full = Halka().build({**P, "preset": "mark4", "part": "emitters", "teeth": "full"})
    assert len(full.solids().vals()) > len(plain.solids().vals()) + 100


@pytest.mark.parametrize("preset,lo,hi", [("mark4", 130, 290), ("mark4b", 360, 670), ("mark4c", 340, 630)])
def test_masses_near_the_video(preset, lo, hi):
    m = Halka().masses({**P, "preset": preset})
    assert lo < m["total"] < hi                                   # the video: 209, 513, 484 g (±30 %)


def test_scale():
    small = Halka().build({**P, "part": "frame", "scale": 1.0}).val().BoundingBox()
    big = Halka().build({**P, "part": "frame", "scale": 1.5}).val().BoundingBox()
    assert big.xlen == pytest.approx(1.5 * small.xlen, rel=0.05)
