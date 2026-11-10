"""Heterogeneous fleet coordination (designs/fleet.py) and the mixed-fleet response scenario
(designs/fleet_scenario.py): planning, allocation, comms-loss recovery, supervision — deterministic.

Run: python tools/lab.py test land -- -k fleet
"""
import math

import numpy as np
import pytest

import fleet as fl
import fleet_scenario as fs


def _room(w=10.0, h=6.0, res=0.2):
    nx, ny = int(w / res), int(h / res)
    occ = np.zeros((nx, ny), np.int8)
    return fl.GridMap(occ, res)


def _profile(width=0.5, tags=("ground", "map"), height=0.4, **kw):
    base = dict(kind="test", tags=frozenset(tags), payload_kg=10.0, lift_kg=5.0, width_m=width, height_m=height,
                v_m_s=1.0, endurance_h=2.0, sensor_range_m=6.0)
    base.update(kw)
    return fl.Profile(**base)


# ----------------------------------------------------------------------------------------------- the map
def test_free_width_reads_the_gap():
    m = _room()
    m.occ[25, :] = fl.OCC                                     # a wall at x = 5 m ...
    m.occ[25, 10:15] = fl.FREE                                # ... with a 1.0 m gap
    w = m.free_width()
    assert w[25, 12] == pytest.approx(1.0)
    assert m.path((2.0, 1.0), (8.0, 1.0), 0.58) is not None   # Pekari-2M fits
    assert m.path((2.0, 1.0), (8.0, 1.0), 1.15) is None       # Pumba does not


def test_headroom_blocks_the_tall_robot():
    m = _room()
    m.occ[25, :] = fl.OCC
    m.occ[25, 5:25] = fl.FREE
    m.ceiling[20:30, :] = 0.55                                # a pipe rack across the room
    assert m.path((2.0, 3.0), (8.0, 3.0), 0.5, height=0.45) is not None
    assert m.path((2.0, 3.0), (8.0, 3.0), 0.5, height=0.90) is None


def test_sensing_stops_at_walls():
    truth = _room()
    truth.occ[25, :] = fl.OCC
    known = truth.copy_unknown()
    fl.sense(truth, known, (2.0, 3.0), 8.0)
    assert known.occ[25, 15] == fl.OCC                         # the wall is seen
    assert known.occ[35, 15] == fl.UNKNOWN                     # what is behind it is not


def test_third_party_vocabulary_is_mapped_not_invented():
    p = fl.third_party_profile("crawler X", {"capabilities": ["navigation", "camera", "narrow", "teleport"],
                                            "width_m": 0.4, "height_m": 0.3})
    assert p.tags == frozenset({"ground", "inspect", "confined"})


# ----------------------------------------------------------------------------------------------- allocation
def test_auction_gives_each_task_the_cheapest_capable_agent():
    truth = _room(20.0, 6.0)
    comms = fl.Comms((1.0, 3.0), range_m=100.0)
    c = fl.Coordinator(truth, comms)
    c.map.occ[:] = fl.FREE
    near = fl.Agent("near", _profile(), (2.0, 3.0), (2.0, 3.0), truth.copy_unknown())
    far = fl.Agent("far", _profile(), (12.0, 3.0), (12.0, 3.0), truth.copy_unknown())
    heavy = fl.Agent("heavy", _profile(tags=("ground", "heavy_lift"), lift_kg=60.0), (19.0, 3.0), (19.0, 3.0),
                     truth.copy_unknown())
    c.add(fl.Task("a", "inspect", (3.0, 3.0), tags=frozenset({"ground"})))
    c.add(fl.Task("b", "manipulate", (4.0, 3.0), tags=frozenset({"heavy_lift"}), lift_kg=50.0))
    agents = [near, far, heavy]
    c.allocate(0.0, agents, {a.id for a in agents})
    assert c.tasks["a"].agent == "near" and c.tasks["b"].agent == "heavy"
    assert far.state == "idle"


def test_dependencies_and_approvals_hold_tasks_back():
    truth = _room()
    c = fl.Coordinator(truth, fl.Comms((1.0, 1.0)))
    c.add(fl.Task("x", "inspect", (2.0, 2.0), after=("y",)))
    assert not c.ready(c.tasks["x"])                           # 'y' does not exist yet: not ready
    t = fl.Task("y", "clear_debris", (3.0, 3.0), requires_approval=True)
    c.add(t)
    assert t.status == "awaiting_approval"
    c.pre_approved.add("z")
    c.add(fl.Task("z", "clear_debris", (3.0, 3.0), requires_approval=True))
    assert c.tasks["z"].status == "open"


# ----------------------------------------------------------------------------------------------- the scenario
@pytest.fixture(scope="module")
def scenario():
    fleet = fs.build()
    res = fleet.run(2400.0)
    return fleet, res


def test_scenario_completes_every_operator_task(scenario):
    _, res = scenario
    for tid in ("inspect_control_room", "first_aid", "pump_to_plant_room", "close_plant_valve", "aerial_survey",
                "clear_debris1"):
        assert res["tasks"][tid]["status"] == "done", tid
    assert res["coverage"] > 0.99 and res["t_end"] < 1200


def test_capabilities_decide_who_does_what(scenario):
    _, res = scenario
    t = res["tasks"]
    assert t["clear_debris1"]["agent"] == "PUMBA" and t["pump_to_plant_room"]["agent"] == "PUMBA"
    assert t["inspect_control_room"]["agent"] in ("P1", "P2")          # only Pekari-2M fits under the pipe rack
    assert t["aerial_survey"]["agent"] == "UAV"
    assert t["pump_to_plant_room"]["done_t"] > t["clear_debris1"]["done_t"]


def test_failure_is_recovered_by_reassignment(scenario):
    _, res = scenario
    hist = res["tasks"]["inspect_control_room"]["history"]
    assert [h[1] for h in hist] == ["assigned", "released", "assigned"]
    assert hist[0][2] == "P2" and hist[2][2] == "P1"
    released_t = hist[1][0]
    assert released_t == pytest.approx(fs.FAULT_T + fl.T_RELEASE, abs=1.0)


def test_work_done_out_of_contact_is_reported(scenario):
    _, res = scenario
    ev = [e[1] for e in res["events"]]
    assert "done_offline" in ev and "done_reported" in ev and "link_lost" in ev and "reconnected" in ev


def test_emergency_stop_freezes_everyone(scenario):
    fleet = fs.build()
    while fleet.t < fs.ESTOP[0] + 1:
        fleet.step()
    pos = {a.id: a.pos for a in fleet.agents}
    while fleet.t < fs.ESTOP[1] - 1:
        fleet.step()
    assert all(fleet.agent(k).pos == v for k, v in pos.items())
    assert all(a.estop for a in fleet.agents)


def test_scenario_is_deterministic(scenario):
    _, res = scenario
    again = fs.run()
    assert again["events"] == res["events"] and again["t_end"] == res["t_end"]


def test_without_the_drone_the_ground_fleet_still_finishes():
    res = fs.run(with_drone=False)
    assert all(v["status"] in ("done", "cancelled") for v in res["tasks"].values())
    assert "aerial_survey" not in res["tasks"]
