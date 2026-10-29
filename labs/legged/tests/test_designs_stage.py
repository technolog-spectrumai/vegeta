"""The design-stage machines (no simulation yet): each CAD design builds, each robot model has the
expected joints and a plausible mass, and the robot dimensions match the CAD parameter defaults."""
import pytest

import body_kit  # noqa: F401
import humanoid_robot as hr

ROBOTS = {"humanoid": (hr.talia, 21, (30.0, 60.0))}


@pytest.mark.parametrize("name", sorted(ROBOTS))
def test_robot(name):
    build, n_act, (lo, hi) = ROBOTS[name]
    r = build()
    assert r.summary()["n_actuated"] == n_act
    assert lo < r.total_mass() < hi


def test_humanoid_cad():
    pytest.importorskip("cadquery")
    import humanoid

    defaults = {p.name: p.default for p in humanoid.Humanoid.parameters}
    assert {k: defaults[k] for k in hr.P} == hr.P
    d = humanoid.Humanoid()
    for part in ("humanoid", "trunk", "thigh", "shin", "foot"):
        assert d.generate(part=part).measure()["valid"]
    assert humanoid.Humanoid.total_height(defaults) == pytest.approx(1559.1, abs=0.1)
