"""The air match (branch dev_air_match): the arena in MuJoCo, the aircraft holding a hover, a match's bookkeeping.

Run: .venv/bin/python -m pytest -q labs/aero/tests/test_air_match.py   (about a minute; needs MuJoCo)
"""
import math

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
import emerillon_flight as ef  # noqa: E402
from air_match import arena as ar  # noqa: E402
from air_match import match  # noqa: E402


def test_arena_compiles_with_the_hoops_on_two_metre_sticks():
    a = ar.Arena()
    assert a.hoop_stick == 2.0 and a.hoops["A"][2] == pytest.approx(2.0 + a.hoop_radius) and a.hoops["A"][0] == -a.hoops["B"][0]
    m = mujoco.MjModel.from_xml_string(ar.to_mjcf(a, [("a1", "ER2", "A", a.start("A")), ("b1", "ER1", "B", a.start("B"))]))
    for name, v in (("a1", "ER2"), ("b1", "ER1")):
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        assert float(m.body_subtreemass[b]) == pytest.approx(ef.mass_budget(ef.VARIANTS[v]).mass, rel=1e-3)
        assert m.body_inertia[b][2] < m.body_inertia[b][0]                       # slender: it spins easiest about its nose axis
    assert sum(1 for i in range(m.nbody) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "").startswith("balloon")) == a.n_balloons
    n = ar.body_numbers("ER2")
    assert n["reach"] < a.hoop_radius and n["z_tail"] < 0 < n["z_nose"]          # it fits through the hoop; the CG is inside the body


def test_a_frozen_aircraft_holds_its_hover_and_the_balloons_only_drift():
    mt = match.Match(rules=match.Rules(duration=3.0, kickoff_s=10.0))
    start = [b.copy() for b in (mt.data.xpos[i].copy() for i in mt.balloons)]
    r = mt.run()
    for p in r["players"].values():
        h = p["hist"]
        home = mt.a.start(p["team"])
        assert np.abs(h["z"] - home[2]).max() < 0.15 and np.abs(h["x"] - home[0]).max() < 0.15 and h["bz"].min() > 0.98
        assert 0.05 < p["energy_wh"] < 0.6                                        # three seconds of hover power
    moved = [float(np.linalg.norm(mt.data.xpos[i] - s)) for i, s in zip(mt.balloons, start)]
    assert 0.0 < max(moved) < 4.0 and not r["events"]


def test_a_match_keeps_its_books():
    r = match.play(20.0, seed=2)
    goals = [e for e in r["events"] if e["what"] == "goal"]
    assert sum(r["score"].values()) == len(goals) == r["counts"]["goal"]
    assert sum(p["goals"] for p in r["players"].values()) == len(goals)
    assert any(e["what"] == "token" and e["why"] == "picked up" for e in r["events"])
    for e in goals:                                                               # only the carrier scores: it held the token just before
        h = r["players"][e["who"]]["hist"]
        assert h["token"][max(int(np.searchsorted(h["t"], e["t"])) - 3, 0)] == 1.0
    a = r["arena"]
    for p in r["players"].values():
        h = p["hist"]
        assert np.isfinite(h["x"]).all() and np.abs(h["x"]).max() < a.length / 2 + 4 and h["z"].min() > -0.05
        assert p["max_speed"] < 25.0 and p["fouls"] + p["crashes"] == sum(1 for e in r["events"] if e["what"] in ("foul", "crash") and e["who"] in (n for n, q in r["players"].items() if q is p))
    again = match.play(20.0, seed=2)
    assert again["score"] == r["score"] and len(again["events"]) == len(r["events"])   # the same seed, the same match


def test_the_plain_game_is_a_race_to_the_hoops():
    r = match.play(8.0, seed=1, possession=False)
    goals = [e for e in r["events"] if e["what"] == "goal"]
    assert goals and goals[0]["t"] < 7.0 and not any(e["what"] in ("token", "tag") for e in r["events"])
    assert match.Rules(score="over").score == "over"


def test_closest_approach():
    t, miss = match._closest(np.array([10.0, 1.0, 0.0]), np.array([-5.0, 0.0, 0.0]), 3.0)
    assert t == pytest.approx(2.0) and np.allclose(miss, [0.0, 1.0, 0.0])
    t, miss = match._closest(np.array([10.0, 1.0, 0.0]), np.array([5.0, 0.0, 0.0]), 3.0)
    assert t == 0.0 and math.isclose(float(np.linalg.norm(miss)), math.hypot(10.0, 1.0))
