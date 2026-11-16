"""The design-stage machines (no simulation yet): each CAD design builds, each robot model has the
expected joints and a plausible mass, and the robot dimensions match the CAD parameter defaults."""
import pytest

import body_kit  # noqa: F401
import humanoid_robot as hr
import fantasy_humanoid_robots as fhr
import monster_robots as mr
import space_unit_robots as sr
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
          "fantasy_ork": (fhr.ork, 21, (60.0, 95.0)),
          "space_marine": (sr.marine, 21, (350.0, 520.0)), "space_marine_saw": (sr.marine_saw, 21, (350.0, 520.0)),
          "space_centurion": (sr.centurion, 21, (350.0, 520.0)), "space_breacher": (sr.breacher, 21, (380.0, 560.0)),
          "space_heavy": (sr.heavy, 21, (380.0, 560.0))}

#: CAD designs that must build as one valid solid: (module, class, overrides)
CAD = [("zoo_quadruped", c, {}) for c in ("Dog", "Bear", "Deer", "Elephant")] + \
      [("zoo_others", "Snake", {}), ("zoo_others", "Bat", {}), ("zoo_others", "Bat", {"fold": 1.0}),
       ("zoo_others", "Octopus", {})] + \
      [("monsters", c, {}) for c in ("Wolf", "Wyvern", "Centaur", "Leech", "Leshy", "Wraith", "Ghoul")] + \
      [("fantasy_humanoids", "Ork", {}), ("fantasy_humanoids", "Ork", {"part": "broom"}),
       ("fantasy_humanoids", "Ork", {"part": "lid"})] + \
      [("space_units", c, {}) for c in ("SpaceMarine", "SpaceMarineSaw", "Centurion", "Breacher", "Heavy")] + \
      [("space_units", "Heavy", {"part": "weapon"})]


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


def test_space_units_carry_their_cad_masses_and_aim_level():
    import numpy as np

    import space_unit_motions as sm

    for unit in ("marine", "marine_saw", "centurion", "breacher", "heavy"):
        m = sr.carried_masses(unit)
        assert all(v > 0 for v in m.values()) and sum(m.values()) < 200.0
        assert len(sm.__dict__[f"motions_{unit}"]()[1]) >= 2
    kin, ms = sm.motions_marine()
    aim = {x.name: x for x in ms}["aim"]
    assert aim.check["ik_error_max_m"] < 0.005
    q, _, _ = sm.aim_pose(kin)
    kin.set(q)
    b, front = kin.capsule_end("carbine_barrel")
    barrel, c = kin.geom_point("carbine_barrel")
    muzzle, mid = kin.world(b, front), kin.world(barrel, c)
    d = muzzle - mid
    assert d[0] > 0 and abs(d[2]) < 0.05 * d[0]                            # the barrel points ahead and level
    assert kin.q0[kin.index["R_elbow"]] == pytest.approx(-1.0472, abs=1e-4)   # the elbows stand bent 60°
    kin_b, ms_b = sm.motions_breacher()
    up = {x.name: x for x in ms_b}["shield_up"]
    kin_b.set(kin_b.q0)
    sb, c = kin_b.geom_point("shield")
    st, cs = kin_b.geom_point("shield_strap")
    assert (kin_b.world(sb, c) - kin_b.world(st, cs))[0] > 0.02          # at rest the shield stands ahead of its strap
    q = kin_b.q0.copy()
    for j in up.joints:
        q[kin_b.index[j]] = up.Q[len(up.t) // 2][up.joints.index(j)]      # the guard, mid-motion
    top = kin_b.world(sb, c + np.array([0.0, 0.0, 0.3]), q) - kin_b.world(sb, c)
    assert np.all(np.isfinite(q)) and top[2] > 0.15                        # the plate still stands up in the guard


def test_space_unit_volumes_rebuild():
    pytest.importorskip("cadquery")
    import space_units as sun

    for cls in (sun.SpaceMarine, sun.SpaceMarineSaw, sun.Centurion, sun.Breacher):
        m = cls().generate(part="weapon").measure()
        assert m["valid"] and m["n_solids"] == 2, cls                       # both hands busy: one solid in each
    for unit, cls in (("marine", sun.SpaceMarine), ("marine_saw", sun.SpaceMarineSaw), ("centurion", sun.Centurion),
                      ("breacher", sun.Breacher), ("heavy", sun.Heavy)):
        d = cls()
        p = d.resolve()
        fresh = {k: v.val().Volume() for k, v in d.armour_parts(p).items()}
        fresh.update({k: v.val().Volume() for k, v in d.weapon_parts(p).items()})
        for k, v in sr.CAD_MM3[unit].items():
            assert fresh[k] == pytest.approx(v, rel=1e-4), (unit, k)


def test_centurion_banner_stands_and_the_emblem_tops_it():
    import motion_kit as mk
    import numpy as np

    kin = mk.Kin(sr.centurion())
    kin.set(kin.q0)
    b, top = kin.capsule_end("banner_pole")
    g, c = kin.geom_point("emblem")
    pole_top, emblem = kin.world(b, top), kin.world(g, c)
    foot = kin.world(*kin.geom_point("L"))
    assert pole_top[2] - foot[2] > 2.2                                     # the pole stands well above the helmet
    assert abs(emblem[0] - pole_top[0]) < 0.05 and emblem[2] < pole_top[2]     # the gear hangs just under the top
    ant = kin.world(*kin.geom_point("antenna"))
    assert emblem[2] - 0.6 < ant[2] < emblem[2] and ant[0] > pole_top[0]        # the antenna under the gear, ahead of the pole
    # the body shields cover the body: from below the knees to above the helmet, on the Marine and the Breacher
    for build in (sr.marine, sr.breacher):
        k2 = mk.Kin(build())
        k2.set(k2.q0)
        g = k2.model.geom("shield")
        zc, hh = k2.world(*k2.geom_point("shield"))[2], float(k2.model.geom_size[g.id][2])
        knee = k2.world(k2.body("L_shin"), np.zeros(3))[2]
        head = k2.world(*k2.geom_point("head"))[2]
        assert zc - hh < knee - 0.1 and zc + hh > head + 0.05, build.__name__
    crest = kin.world(*kin.geom_point("crest"))
    head = kin.world(*kin.geom_point("head"))
    assert crest[2] > head[2] + 0.1                                        # the crest rides on the helmet
