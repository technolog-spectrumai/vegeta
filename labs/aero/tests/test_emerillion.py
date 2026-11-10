"""Emerillion (the X-wing): hovers with margin on the two wing-tip fans, makes the pearl run with reserve, builds."""
import pytest

from emerillion import Emerillion, hover, pearl_run, stall_speed


def test_hover_and_the_pearl_run():
    h = hover()
    assert h["hover_ok"] and 2.8 < h["mass_kg"] < 3.4 and h["hover_W"] < 1500
    r = pearl_run(10.0)
    assert r["reserve"] > 0.2 and r["minutes"] < 20
    assert pearl_run(15.0)["reserve"] > 0.0                             # the requirement's farthest vessel
    assert 15.0 < stall_speed() < 22.0                                  # a fast little wing: 28 m/s cruise clears it


def test_cad_builds_the_x_tail_and_the_aircraft():
    pytest.importorskip("cadquery")
    p = {q.name: q.default for q in Emerillion.parameters}
    tail = Emerillion().build({**p, "part": "tail"})
    assert tail.val().isValid()
    wing = Emerillion().build({**p, "part": "wing"})
    bb = wing.val().BoundingBox()
    assert abs((bb.ymax - bb.ymin) - p["span"]) < 1.0                   # tip to tip, the pods included
    assert Emerillion().build(p).val().isValid()
