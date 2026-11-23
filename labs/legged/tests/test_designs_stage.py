"""The design-stage machines (no simulation yet): each CAD design builds, each robot model has the
expected joints and a plausible mass, and the robot dimensions match the CAD parameter defaults."""
import pytest

import body_kit  # noqa: F401
import humanoid_robot as hr
import monster_robots as mr
import zoo_robots as zr

#: name -> (builder, actuated joints, mass range [kg])
ROBOTS = {"humanoid": (hr.talia, 21, (30.0, 60.0)),
          "zoo_dog": (zr.dog, 16, (15.0, 35.0)), "zoo_bear": (zr.bear, 12, (60.0, 160.0)),
          "zoo_deer": (zr.deer, 16, (20.0, 50.0)), "zoo_elephant": (zr.elephant, 28, (400.0, 1200.0)),
          "zoo_snake": (zr.snake, 32, (8.0, 25.0)), "zoo_bat": (zr.bat, 14, (1.0, 4.0)),
          "zoo_octopus": (zr.octopus, 128, (25.0, 70.0)),
          "monster_wolf": (mr.wolf, 16, (35.0, 80.0)), "monster_wyvern": (mr.wyvern, 8, (15.0, 50.0)),
          "monster_centaur": (mr.centaur, 24, (100.0, 220.0)), "monster_leech": (mr.leech, 20, (40.0, 100.0)),
          "monster_leshy": (mr.leshy, 16, (50.0, 120.0)), "monster_wraith": (mr.wraith, 96, (15.0, 60.0)),
          "monster_ghoul": (mr.ghoul, 21, (25.0, 60.0))}

#: CAD designs that must build as one valid solid: (module, class, overrides)
CAD = [("zoo_quadruped", c, {}) for c in ("Dog", "Bear", "Deer", "Elephant")] + \
      [("zoo_others", "Snake", {}), ("zoo_others", "Bat", {}), ("zoo_others", "Bat", {"fold": 1.0}),
       ("zoo_others", "Octopus", {})] + \
      [("monsters", c, {}) for c in ("Wolf", "Wyvern", "Centaur", "Leech", "Leshy", "Wraith", "Ghoul")]


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


@pytest.mark.parametrize("mod,cls,kw", CAD, ids=lambda v: str(v))
def test_cad_builds(mod, cls, kw):
    pytest.importorskip("cadquery")
    import importlib

    m = getattr(importlib.import_module(mod), cls)().generate(**kw).measure()
    assert m["valid"] and m["n_solids"] == 1


def test_registered():
    from pathlib import Path

    from vegeta.labkit import machines

    machines.discover(Path(__file__).resolve().parents[1] / "designs")
    assert set(ROBOTS) <= set(machines.names())
