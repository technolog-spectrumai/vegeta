"""Kerchak's quad, autonomous and drivable (``designs/gorilla_quad_autonomy``, ``designs/gorilla_quad_riders``, the sim
``sims/gorilla_quad_auto``): the planner goes round what blocks the lane and stops when nothing clears, the guardian
brakes whoever drives, the arbiter hands control over both ways, the riders fit, and the three scenarios end at the
goal with nothing hit."""
import math

import numpy as np
import pytest

import gorilla_quad as gq
import gorilla_quad_autonomy as ga
import gorilla_quad_riders as qr


def _box_points(x, y, half_x, half_y, z=0.4, n=12):
    xs = np.linspace(x - half_x, x + half_x, n)
    ys = np.linspace(y - half_y, y + half_y, n)
    face = [(x - half_x, yy, z) for yy in ys]
    sides = [(xx, y + s * half_y, z) for xx in xs for s in (-1, 1)]
    return np.array(face + sides)


def test_planner_detours_round_a_box_in_the_lane():
    p = ga.Planner(lambda x: 0.0, dict(qr.PROFILES["none"]))
    pts = _box_points(15.0, 0.0, 0.4, 0.45)
    plan = p.update(0.0, 4.0, pts, t=0.0)
    assert plan.status.startswith("detour")
    off = plan.offset
    need = ga.SAFETY["half_width"] + ga.SAFETY["clearance"] + 0.45
    assert abs(off) >= need - 0.25 + 1e-9                                 # the offset grid is 0.25 m
    for x in np.linspace(14.6, 15.4, 5):                                   # alongside the box the path is clear of it
        assert abs(plan.steer_path(x)) >= ga.SAFETY["half_width"] + 0.45
    assert plan.steer_path(0.0) == pytest.approx(0.0)                      # the route before and after
    assert plan.steer_path(40.0) == pytest.approx(0.0)


def test_planner_widens_a_detour_when_more_of_the_box_shows():
    p = ga.Planner(lambda x: 0.0, dict(qr.PROFILES["none"]))
    p.update(0.0, 4.0, np.array([[15.0, 0.0, 0.4]]), t=0.0)               # far away: one beam on it
    first = abs(p.detour["offset"])
    p.update(3.0, 4.0, _box_points(15.0, 0.0, 0.4, 1.2), t=0.5)           # closer: it is wider than it looked
    assert abs(p.detour["offset"]) > first


def test_planner_stops_when_nothing_clears():
    p = ga.Planner(lambda x: 0.0, dict(qr.PROFILES["none"]))
    wall = np.array([(12.0, y, 0.5) for y in np.linspace(-8.0, 8.0, 161)])
    plan = None
    for k, x in enumerate(np.linspace(0.0, 9.6, 9)):
        plan = p.update(x, 1.0, wall, t=0.1 * k)
    assert plan.status == "stopping: blocked"
    assert plan.v_max == pytest.approx(0.0, abs=1e-6)


def test_perception_drops_the_ground_and_the_bumps():
    per = ga.Perception()
    pts = np.array([[5, 0, 0.0], [6, 0, 0.11], [7, 0, 0.5]])
    assert per.obstacles(pts).tolist() == [[7, 0, 0.5]]


def test_guardian_brakes_for_what_is_on_the_arc_not_beside_it():
    g = ga.Guardian()
    assert g.check(0.0, 0.0, 0.0, 4.0, 0.0, np.array([[3.5, 0.0, 0.5]]), 1.7)
    g = ga.Guardian()
    assert not g.check(0.0, 0.0, 0.0, 4.0, 0.0, np.array([[3.5, 2.5, 0.5]]), 1.7)
    g = ga.Guardian()                                                      # turning away from it
    assert not g.check(0.0, 0.0, 0.0, 2.0, 0.5, np.array([[3.0, -1.0, 0.5]]), 1.7)


def test_arbiter_hands_over_both_ways():
    a = ga.Arbiter("none")
    assert a.update(0.0, 0.01) == "AUTO"
    assert a.update(0.0, 0.01, bar_torque=50.0) == "AUTO"                  # nobody aboard: no takeover
    a = ga.Arbiter("kerchak")
    assert a.mode == "MANUAL"
    assert a.update(1.0, 0.01, auto_button=True) == "AUTO"
    t = 2.0
    for _ in range(int(ga.STEER_BY_WIRE["takeover_s"] / 0.01) - 2):
        assert a.update(t, 0.01, bar_torque=25.0) == "AUTO"                # a brief push is not a takeover
        t += 0.01
    for _ in range(5):
        a.update(t, 0.01, bar_torque=25.0)
        t += 0.01
    assert a.mode == "MANUAL"
    assert a.update(t, 0.01, estop=True) == "STOP"
    assert a.update(t, 0.01, auto_button=True) == "STOP"                   # an e-stop latches


def test_seat_load_tells_who_rides():
    assert qr.classify_seat_load(2.0) == "none"
    assert qr.classify_seat_load(qr.HUMAN_MASS) == "human"
    assert qr.classify_seat_load(gq.RIDER["mass"]) == "kerchak"


def test_air_springs_keep_the_ride_whoever_rides():
    for who in ("none", "human", "kerchak"):
        s = gq.sizing(rider_mass=qr.PROFILES[who]["mass"])
        for c in gq.CORNERS:
            k, ms = s["corners"][c]["k_N_m"], s["corners"][c]["sprung_kg"]
            assert math.sqrt(k / ms) / (2 * math.pi) == pytest.approx(gq.RIDE_HZ, rel=1e-2)
            assert s["corners"][c]["sag_m"] == pytest.approx(9.81 / (2 * math.pi * gq.RIDE_HZ) ** 2, abs=2e-3)   # the same ride height


def test_the_human_fits_with_the_bar_brought_back():
    import motion_kit as mk

    kin = mk.Kin(qr.rider_robot("human"))
    assert sum(kin.model.body_mass) == pytest.approx(qr.HUMAN_MASS, rel=1e-3)
    off, err = qr.human_bar_offset(kin)
    assert err < 0.01
    assert 0.0 < off[0] <= qr.BAR_TRAVEL["back"] and 0.0 <= off[1] <= qr.BAR_TRAVEL["down"]


def test_kit_builds_and_the_lidars_see_past_the_frame():
    d = ga.AutonomyKit()
    for part in ("lidar_front", "rear_mast", "beacon", "estops"):
        assert d.generate(part=part).measure()["volume"] > 0
    fx = ga.LIDARS["front"]["pos"][0]
    assert fx > gq.GEOM["wheelbase"] / 2                                   # ahead of the front axle and the fenders' arc
    assert ga.LIDARS["rear"]["pos"][2] > gq.GEOM["trunk"]["floor_z"] + gq.GEOM["trunk"]["h"]   # above the trunk and its tools
    assert ga.lidar_rays("front").shape == (ga.LIDARS["front"]["beams"] * len(ga.LIDARS["front"]["layers"]), 3)


@pytest.mark.slow
@pytest.mark.parametrize("scenario", ["auto", "human", "kerchak"])
def test_scenarios_reach_the_goal_hitting_nothing(scenario):
    import gorilla_quad_auto as A

    ride = A.AutoRide(scenario)
    t = 0.0
    while t < ride.sc["seconds"]:
        ride.step(t, 1 / 15)
        t += 1 / 15
        x, y, yaw, v = ride.pose()
        if x > A.GOAL_X - 0.6 and abs(v) < 0.03 and t > 3.0:
            break
    rep = ride.report()
    assert rep["x_end_m"] == pytest.approx(A.GOAL_X, abs=0.7)
    assert rep["crate_moved_m"] < 0.01 and max(rep["cones_moved_m"]) < 0.01
    if scenario == "human":
        assert rep["aeb_s"] > 0.3                                          # the guardian stopped the distracted rider
    if scenario == "kerchak":
        assert [e for _, e in rep["events"]] == ["AUTO engaged by the rider", "MANUAL: the rider took over"]
