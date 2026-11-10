"""Emerillion: hovers with margin on the 120 mm fan, makes the pearl run with reserve, builds with its legs."""
import pytest

from emerillion import Emerillion, hover, pearl_run


def test_hover_and_the_pearl_run():
    h = hover()
    assert h["hover_ok"] and 2.8 < h["mass_kg"] < 3.6 and h["hover_W"] < 2400
    r = pearl_run(10.0)
    assert r["reserve"] > 0.2 and r["minutes"] < 25
    assert pearl_run(15.0)["reserve"] > 0.0                             # the requirement's farthest vessel


def test_cad_builds_with_legs():
    pytest.importorskip("cadquery")
    p = {q.name: q.default for q in Emerillion.parameters}
    legs = Emerillion().build({**p, "part": "legs"})
    assert legs.val().isValid()
    assert Emerillion().build(p).val().isValid()
