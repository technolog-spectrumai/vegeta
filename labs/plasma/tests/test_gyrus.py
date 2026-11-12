"""Gyrus's CAD: every part builds, the channel's dimensions are the scaled ones, the masses are a small thruster's."""
import pytest

pytest.importorskip("cadquery")

from vegeta.fulgur.hall import scale_channel

from gyrus import Gyrus

P = {q.name: q.default for q in Gyrus.parameters}


@pytest.mark.parametrize("part", ["channel", "anode", "core", "coil_in", "coils_out", "screen", "cathode", "feed"])
def test_every_part_builds(part):
    s = Gyrus().build({**P, "part": part})
    assert len(s.solids().vals()) >= 1 and all(v.isValid() for v in s.solids().vals())


def test_channel_is_the_scaled_one():
    ch = scale_channel(P["power"], P["v_d"])
    bb = Gyrus().build({**P, "part": "channel"}).val().BoundingBox()
    assert bb.xmax == pytest.approx(ch.r_out * 1e3 + P["wall"], abs=0.3)
    assert bb.zmax == pytest.approx(ch.L * 1e3, abs=0.3)
    assert len(Gyrus().build({**P, "part": "coils_out"}).solids().vals()) == 2 * P["outer_coils"]     # a core and a coil each


def test_masses():
    m = Gyrus().masses(P)
    assert 300 < m["total"] < 1500                                                   # a 300 W class thruster, iron circuit
    assert m["core + plate (iron)"] + m["screen + pole (iron)"] > m["channel (BN)"]
