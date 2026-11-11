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
    m = mujoco.MjModel.from_xml_string(ar.to_mjcf(a, [("a1", "ER2", "#d62728", a.start("A")), ("b1", "ER1", "#1f77b4", a.start("B"))]))
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


# ------------------------------------------------------------------------------- steps 2-4: the swarm, the formation, the group chase
def test_swarm_layouts_and_the_assignment():
    import itertools
    from air_match import swarm as sw
    w = sw.wedge_offsets(7, 2.4)
    assert np.allclose(w[0], 0) and np.allclose(np.linalg.norm(w[1]), 2.4) and np.allclose(np.linalg.norm(w[3] - w[1]), 2.4)
    assert math.degrees(math.atan2(abs(w[1][1]), -w[1][0])) == pytest.approx(sw.WEDGE_ANGLE) and w[1][1] == -w[2][1]
    assert np.allclose(sw.column_offsets(4, 2.0)[:, 0], [0, -2, -4, -6])
    rng = np.random.default_rng(3)
    cost = rng.uniform(0, 10, (6, 6)).tolist()
    best = min(sum(cost[i][p[i]] for i in range(6)) for p in itertools.permutations(range(6)))
    a = sw.min_cost_assignment(cost)
    assert sorted(a) == list(range(6)) and sum(cost[i][a[i]] for i in range(6)) == pytest.approx(best)
    here, there = [tuple(x) for x in sw.wedge_offsets(7, 2.4)], [tuple(x) for x in sw.column_offsets(7, 2.4)]
    m = sw.Morph(here, there, 14.0, 0.0)
    assert sw.MORPH_T[0] <= m.T <= sw.MORPH_T[1] and m.crossings <= sw.Morph.count_crossings(here, there, list(range(7)))
    assert sw.smoothstep(0.0) == (0.0, 0.0) and sw.smoothstep(1.0) == (1.0, 0.0) and sw.smoothstep(0.5)[0] == 0.5
    push = sw.separation((0, 0, 0), [(1.0, 0, 0), (9.0, 0, 0)], reach=0.7)
    assert push[0] < 0 and push[1] == 0 and np.allclose(sw.separation((0, 0, 0), [(5.0, 0, 0)], reach=0.7), 0)
    ref = sw.PathReference((0, 0, 3), 0.0, keep_m=10.0)
    for k in range(1, 200):
        ref.update((0.05 * k, 0, 3), (5.0, 0, 0), 0.01)
    p, v = ref.path_slot(4.0)
    assert p[0] == pytest.approx(ref.pos[0] - 4.0, abs=0.3) and v[0] == pytest.approx(5.0)
    p, v = ref.rigid_slot((-2.0, 1.0, 0.0))
    assert np.allclose(p, ref.pos + (-2.0, 1.0, 0.0)) and np.allclose(v, (5.0, 0, 0))


def test_the_formation_changes_shape_and_threads_a_hoop():
    from air_match import formation
    r = formation.fly(7, 26.0, seed=1)
    s = r["formation"]["summary"]
    assert s["members"] == 7 and s["changes"] >= 3 and {c["to"] for c in r["formation"]["changes"]} == {"wedge", "column"}
    assert all(c["crossings"] <= c["kept_numbers_crossings"] for c in r["formation"]["changes"])
    assert sum(1 for n, k in s["hoops_flown"].items() if n != "hare" and k >= 1) >= 6          # single file through the first hoop
    assert s["closest_pair_m"] > s["touching_at_m"] and s["slot_error_column_m"] < 1.5 and s["gap_to_hare_m"] < 9.0
    f = r["formation"]
    assert f["wedge"].max() == 1.0 and f["wedge"].min() == 0.0 and len(f["t"]) == len(r["players"]["f1"]["hist"]["t"])


@pytest.mark.parametrize("n", [3, 7])
def test_the_group_chase_counts_the_seconds_with_the_token(n):
    r = match.chase(n, 20.0, seed=2)
    assert len(r["players"]) == n == len(r["score"]) and r["rules"].mode == "keepaway"
    held = sum(p["hold_s"] for p in r["players"].values())
    assert 5.0 < held <= 20.0 and r["counts"]["tag"] >= 1 and r["counts"]["goal"] == 0
    assert sum(r["score"].values()) == pytest.approx(held + sum(p["bonus"] for p in r["players"].values()), abs=0.1 * n)
    for p in r["players"].values():
        dt = p["hist"]["t"][1] - p["hist"]["t"][0]
        assert p["hold_s"] <= float(p["hist"]["token"].sum()) * dt + 1.5        # the seconds counted are in the record (a restart's wait is not counted)
    assert r["winner"] == max(r["score"], key=r["score"].get)


# ------------------------------------------------------------------------------------------------------- the circus race
def test_the_stadium_and_its_racing_line():
    from air_match import circus
    a = circus.Stadium()
    assert a.hoops == {} and a.length == pytest.approx(circus.REAL["length"] * circus.SCALE) and a.half == pytest.approx(circus.REAL["spina"] * circus.SCALE / 2)
    r = circus.Race(circus.STRATEGIES[:2], seed=1, laps=1)
    names = {mujoco.mj_id2name(r.model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(r.model.ngeom)}
    assert {"spina", "meta0", "meta1"} <= names and not any((n or "").startswith("hoop") for n in names)
    st = circus.BY_NAME["wide"]
    pt, tan, to_turn, R = r.line(st, np.array([10.0, -20.0, 4.0]))               # the south straight: with the wind, toward the east turn
    assert np.allclose(pt[:2], [10.0, -st.radius]) and np.allclose(tan, [1, 0, 0]) and to_turn == pytest.approx(a.half - 10.0) and R == math.inf
    pt, tan, to_turn, R = r.line(st, np.array([a.half + 9.0, 0.0, 4.0]))         # the east turn: anticlockwise, so north
    assert np.allclose(pt[:2], [a.half + st.radius, 0.0]) and np.allclose(tan, [0, 1, 0]) and to_turn == 0.0 and R == st.radius
    pt, tan, _, _ = r.line(st, np.array([-10.0, 20.0, 4.0]))
    assert np.allclose(tan, [-1, 0, 0]) and pt[1] == st.radius
    low, high = r.wind_at(np.array([0, 0, 2.0])), r.wind_at(np.array([0, 0, 12.0]))
    assert 0 < low[0] < high[0] and low[1] == 0.0                                 # stronger aloft, down the south straight
    assert len({s.name for s in circus.STRATEGIES}) == 7


def test_a_racer_alone_laps_the_circus_and_the_wind_moves_the_balloons():
    from air_match import circus
    r = circus.solo(circus.BY_NAME["middle"], laps=1, balloons=6)
    s = r["race"]["standings"][0]
    assert s["finished"] and s["laps"] == 1 and 12.0 < s["time_s"] < 32.0 and s["faults"] == 0
    h = r["players"]["middle"]["hist"]
    assert h["y"].min() < -6.0 and h["y"].max() > 6.0 and np.abs(h["x"]).max() > circus.Stadium().half      # it went round both posts
    bal = r["balloons"]
    assert float(np.median(bal[-1, :, 0] - bal[0, :, 0])) > 5.0                   # they went downwind (or came back in upwind)
    calm = circus.Race([circus.BY_NAME["middle"]], 1, 1, 60.0, wind=dict(circus.WIND, speed=0.0, gust=0.0), n_balloons=6)
    assert np.allclose(calm.wind_at(np.array([0, 0, 5.0])), 0)


def test_a_race_ranks_its_racers():
    from air_match import circus
    r = circus.Race([circus.BY_NAME[n] for n in ("hot", "middle", "careful")], seed=2, laps=1, duration=60.0).run()
    st = r["race"]["standings"]
    assert [s["place"] for s in st] == [1, 2, 3] and {s["strategy"] for s in st} == {"hot", "middle", "careful"}
    done = [s for s in st if s["finished"]]
    assert done and [s["time_s"] for s in done] == sorted(s["time_s"] for s in done) and r["race"]["winner"] == st[0]["strategy"]
    assert all(s["faults"] == sum(1 for e in r["race"]["faults"] if e["who"] == s["strategy"]) for s in st)
    laps = [e for e in r["events"] if e["what"] == "lap"]
    assert len(laps) == len(done) and all(e["time"] > 10.0 for e in laps)


def test_the_slipstream_is_behind_an_aircraft_and_nowhere_else():
    from air_match import circus
    r = circus.Race([circus.BY_NAME["hot"], circus.BY_NAME["inside"]], seed=1, laps=1, wind=dict(circus.WIND, speed=0.0, gust=0.0), n_balloons=2)
    lead, me = r.players
    r.data.qvel[lead.va:lead.va + 3] = (30.0, 0.0, 0.0)
    p0 = r.pos(lead)
    behind, beside, ahead, far = p0 + (-4.0, 0, 0), p0 + (-4.0, 4.0, 0), p0 + (4.0, 0, 0), p0 + (-40.0, 0, 0)
    w = r.wake_at(me, behind)
    assert 2.0 < w[0] < 30.0 * circus.WAKE["gain"] and abs(w[1]) < 1e-9                    # the air follows it
    assert np.linalg.norm(r.wake_at(me, beside)) < 0.05 * w[0] and np.allclose(r.wake_at(me, ahead), 0) and np.allclose(r.wake_at(me, far), 0)
    assert np.linalg.norm(r.wake_at(me, p0 + (-10.0, 0, 0))) < w[0]                        # fading with distance
    assert np.allclose(r.wake_at(lead, behind), 0)                                         # nobody drafts itself (the other one stands still)
    off = circus.Race([circus.BY_NAME["hot"], circus.BY_NAME["inside"]], seed=1, laps=1, n_balloons=2, slipstream=False)
    off.data.qvel[off.players[0].va:off.players[0].va + 3] = (30.0, 0.0, 0.0)
    assert np.allclose(off.wake_at(off.players[1], off.pos(off.players[0]) + (-4.0, 0, 0)), 0)


def test_adaptive_strategies_change_what_they_fly():
    from air_match import circus
    assert {s.kind for s in circus.ADAPTIVE} == {"draft", "adapt", "temper"} and len(circus.FIELD) == 7
    r = circus.Race(circus.FIELD, seed=1, laps=1)
    by = {p.name: p for p in r.players}
    hot0 = circus.BY_NAME["tempered"]
    assert r.adapt(by["tempered"], hot0, 0.0, 50.0, math.inf).a_lat == hot0.a_lat
    r.state["tempered"]["faults"] = 2
    cooled = r.adapt(by["tempered"], hot0, 0.0, 50.0, math.inf)
    assert cooled.a_lat < hot0.a_lat and cooled.caution > hot0.caution and cooled.v_air < hot0.v_air
    r.state["tempered"]["bold"] = 4                                                        # clean laps give it back
    assert r.adapt(by["tempered"], hot0, 0.0, 50.0, math.inf).a_lat == hot0.a_lat
    ad, me = circus.BY_NAME["adaptive"], by["adaptive"]
    p = r.pos(me)
    for i, b in enumerate(r.balloons):                                                     # every balloon far away: the low line
        q = r.balloon_qadr[i]
        r.data.qpos[q:q + 3] = (-100.0, 25.0 + i, 5.0)
    mujoco.mj_forward(r.model, r.data)
    assert r.adapt(me, ad, 0.0, 50.0, math.inf).height == ad.height
    r.data.qpos[r.balloon_qadr[0]:r.balloon_qadr[0] + 3] = (p[0] + 12.0, -ad.radius, 4.0)  # one on its line ahead: up and over
    mujoco.mj_forward(r.model, r.data)
    assert r.adapt(me, ad, 0.0, 50.0, math.inf).height > 10.0
    assert circus.BY_NAME["hot"] is r.adapt(by["hot"], circus.BY_NAME["hot"], 0.0, 50.0, math.inf)   # a fixed one stays what it is


def test_a_changing_wind_changes():
    from air_match import circus
    r = circus.Race([circus.BY_NAME["middle"]], seed=3, laps=1, duration=25.0, wind=circus.CHANGING, n_balloons=6).run()
    wh = r["race"]["wind_hist"]
    assert wh[:, 1].std() > 0.5 and np.ptp(wh[:, 2]) > 20.0 and wh[:, 1].min() >= 0.0       # speed and direction wander
    assert all(4.0 <= f["ms"] <= 8.0 for f in r["race"]["gust_fronts"])
    steady = circus.Race([circus.BY_NAME["middle"]], seed=3, laps=1, duration=6.0, n_balloons=6).run()
    sh = steady["race"]["wind_hist"]
    assert np.allclose(sh[:, 1], circus.WIND["speed"]) and np.allclose(sh[:, 2], 0.0) and not steady["race"]["gust_fronts"]


# ------------------------------------------------------------------------------------------------------ the team chase
def test_the_team_chase_gives_the_reds_their_head_start_and_counts_the_tags():
    from air_match import teamchase as tc
    r = tc.play("scatter", "assigned", seed=1, duration=30.0, head_start_s=3.0)
    x = r["teamchase"]
    blues = [r["players"][f"blue{i}"]["hist"] for i in (1, 2, 3)]
    early = blues[0]["t"] < 2.8
    assert all(np.abs(h["x"][early] - h["x"][0]).max() < 0.3 for h in blues)              # the blues wait
    reds = [r["players"][f"red{i}"]["hist"] for i in (1, 2, 3)]
    assert max(float(np.hypot(h["x"][early][-1] - h["x"][0], h["y"][early][-1] - h["y"][0])) for h in reds) > 5.0   # the reds are off
    tags = [e for e in r["events"] if e["what"] == "tag"]
    assert len(tags) == len(x["caught"]) == sum(x["tags"].values()) and all(e["t"] > 3.0 for e in tags)
    assert x["winner"] == ("blue" if len(x["caught"]) == 3 else "red") and r["score"]["blue"] == len(x["caught"])
    assert x["red_seconds"] <= 3 * x["duration"] + 0.1


def test_assignment_spreads_the_blues():
    from air_match import teamchase as tc
    g = tc.TeamChase("scatter", "assigned", seed=1)
    a = g.targets(5.0)
    assert len({r.name for r in a.values()}) == 3                                          # one blue per red
    n = tc.TeamChase("scatter", "nearest", seed=1).targets(5.0)
    assert set(n) == {"blue1", "blue2", "blue3"}
