"""The design-stage machines (no simulation yet): each CAD design builds, each robot model has the
expected joints and a plausible mass, and the robot dimensions match the CAD parameter defaults."""
import pytest

import body_kit  # noqa: F401
import humanoid_robot as hr
import fantasy_humanoid_robots as fhr
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
          "monster_ghoul": (mr.ghoul, 25, (25.0, 60.0)),
          "fantasy_ork": (fhr.ork, 21, (60.0, 95.0))}

#: CAD designs that must build as one valid solid: (module, class, overrides)
CAD = [("zoo_quadruped", c, {}) for c in ("Dog", "Bear", "Deer", "Elephant")] + \
      [("zoo_others", "Snake", {}), ("zoo_others", "Bat", {}), ("zoo_others", "Bat", {"fold": 1.0}),
       ("zoo_others", "Octopus", {})] + \
      [("monsters", c, {}) for c in ("Wolf", "Wyvern", "Centaur", "Leech", "Leshy", "Wraith", "Ghoul")] + \
      [("fantasy_humanoids", "Ork", {}), ("fantasy_humanoids", "Ork", {"part": "broom"}),
       ("fantasy_humanoids", "Ork", {"part": "lid"})]


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


def test_ork_carries_its_cad_masses_and_sweeps_on_the_ground():
    import fantasy_humanoid_motions as fm

    kin, ms = fm.motions_ork()
    kin.set(kin.q0)
    g = kin.model.geom("broom_bristles").id
    sole = kin.model.geom("L").id
    bottom = kin.data.geom_xpos[g][2] - kin.model.geom_size[g][2]
    ground = kin.data.geom_xpos[sole][2] - kin.model.geom_size[sole][2]
    assert bottom == pytest.approx(ground, abs=0.01)                    # the broom stands on the ground
    m = fhr.carried_masses()
    assert sum(m.values()) == pytest.approx(5.92, abs=0.01)
    sweep = {x.name: x for x in ms}["sweep"]
    assert sweep.check["ik_error_max_m"] < 0.002 and sweep.params["half_width_m"] == 0.2
    assert kin.q0[kin.index["R_elbow"]] == pytest.approx(-0.6109, abs=1e-4)   # the elbows stand bent 35°


def test_ork_carried_volumes_rebuild():
    pytest.importorskip("cadquery")
    import fantasy_humanoids as fhu

    p = fhu.Ork().resolve()
    wood, straw = fhu.broom(p)
    fresh = {"broom wood": wood.val().Volume(), "broom straw": straw.val().Volume(),
             "lid": fhu.bin_lid(p)[0].val().Volume()}
    for k, v in fhr.CAD_MM3.items():
        assert fresh[k] == pytest.approx(v, rel=1e-4), k


def test_ghoul_pincers_open_away_from_each_other():
    import motion_kit as mk

    kin = mk.Kin(mr.ghoul())
    for side in ("L", "R"):
        a, b = kin.index[f"{side}_jaw_a"], kin.index[f"{side}_jaw_b"]
        q = kin.q0.copy()
        q[a], q[b] = kin.hi[a], kin.lo[b]                                # both wide
        kin.set(q)
        ya = kin.world(*kin.capsule_end(f"{side}_jaw_a"))[1]
        yb = kin.world(*kin.capsule_end(f"{side}_jaw_b"))[1]
        assert ya - yb > 0.15                                            # the tips spread
