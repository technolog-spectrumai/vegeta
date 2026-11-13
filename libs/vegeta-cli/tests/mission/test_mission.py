"""vegeta.mission: the flight software's pieces in isolation, and a short closed loop with the kinematic aircraft.

    .venv/bin/python -m pytest -q vegeta-cli/tests/mission
"""
import math
import re
from pathlib import Path

import numpy as np
import pytest

import vegeta.mission as vm
from vegeta.mission import Bus, Executor, Node, compute, geometry
from vegeta.mission.camera import IMX219, IMX219_6MM, camera_pose, mount_rotation
from vegeta.mission.guidance import GuidanceConfig, PhotoPassGuidance
from vegeta.mission.messages import Detection, Detections, NavState, PhotoEvent, TargetSelection, TrackEstimate, Tracks
from vegeta.mission.metrics import Sample, evaluate
from vegeta.mission.photo import PhotoTrigger
from vegeta.mission.targeting import SchedulerConfig, TargetScheduler
from vegeta.mission.tracking import BirdTracker, TrackerConfig

PKG = Path(vm.__file__).parent


def level_nav(t=0.0, pos=(0, 0, 70), course=math.pi / 2, V=15.0):
    R = np.array([[math.cos(course), -math.sin(course), 0], [math.sin(course), math.cos(course), 0], [0, 0, 1.0]])
    return NavState(t, np.array(pos, float), V * np.array([math.cos(course), math.sin(course), 0.0]), R, np.zeros(3), V)


def test_flight_software_imports_no_simulator():
    for f in PKG.glob("*.py"):
        src = f.read_text()
        for mod in ("mujoco", "chiron", "rclpy", ".sim", "vegeta.mission.sim", "nisus"):
            assert not re.search(rf"^\s*(from|import)\s+[\w.]*{re.escape(mod)}\b", src, re.M), f"{f.name} imports {mod}"


def test_camera_projection_round_trip_and_mount():
    cam = IMX219
    assert cam.hfov_deg == pytest.approx(62.2, abs=0.3)
    assert IMX219_6MM.hfov_deg == pytest.approx(34.0, abs=0.5)
    Rbc = mount_rotation(15.0)
    assert Rbc[:, 2] @ np.array([1, 0, 0]) == pytest.approx(math.cos(math.radians(15)))
    assert Rbc[:, 2][2] < 0                                   # looking down
    nav = level_nav()
    R_wc, p_wc = camera_pose(cam, nav.R_wb, nav.pos)
    pts = np.array([[5.0, 80.0, 60.0], [-10.0, 40.0, 65.0]])
    uv, z = cam.project(pts, R_wc, p_wc)
    rays = cam.ray(uv, R_wc)
    back = p_wc + rays * np.linalg.norm(pts - p_wc, axis=1)[:, None]
    assert np.allclose(back, pts, atol=1e-6)
    assert cam.pixels_across(1.0, 20.0) == pytest.approx(cam.fx / 20.0)


def test_executor_rates_and_determinism():
    class Counter(Node):
        def __init__(self, name):
            self.name = name
            self.seen = []

        def step(self, t, bus):
            self.seen.append(round(t, 6))
            bus.publish(self.name, t)

    bus = Bus()
    ex = Executor(bus)
    a, b = ex.add(Counter("a"), 10.0), ex.add(Counter("b"), 4.0)
    ex.spin_until(1.0)
    assert len(a.seen) == 11 and len(b.seen) == 5
    assert a.seen == sorted(a.seen)
    assert bus.latest("a") == pytest.approx(1.0)


def test_compute_budget_orders_and_scales():
    tiny_orin = compute.budget(compute.DetectorConfig(device="orin_nano_super", model="yolox-tiny", search_tiles=2))
    tiny_orin1 = compute.budget(compute.DetectorConfig(device="orin_nano_super", model="yolox-tiny", search_tiles=1))
    tiny_nano = compute.budget(compute.DetectorConfig(device="nano_b01", model="yolox-tiny"))
    assert tiny_orin["feasible"] and tiny_orin["GPU load"] > tiny_orin1["GPU load"]
    assert not tiny_nano["feasible"] and tiny_nano["search [Hz] achieved"] < 5.0
    assert tiny_orin["status"].startswith("ASSUMED")
    t = compute.budget_table()
    assert len(t) == 12


def test_geometry_intercept_and_cpa():
    # a target 100 m north flying east at 10 m/s, pursuer at 20 m/s: |(10t, 100)| = 20t → t = 100/sqrt(300)
    t = geometry.intercept_time([0, 100, 0], [10, 0, 0], 20.0)
    assert t == pytest.approx(100 / math.sqrt(300), rel=1e-6)
    assert geometry.intercept_time([0, 100, 0], [0, 30, 0], 20.0) is None          # faster and going away
    tc, dc = geometry.closest_approach([0, 100, 0], [5, -10, 0])
    assert dc == pytest.approx(100 * 5 / math.hypot(5, 10), rel=1e-6)
    p = geometry.predict([0, 0, 0], [10, 0, 0], 0.5, math.pi / 0.5)                  # half a circle of radius 20
    assert p[:2] == pytest.approx([0.0, 40.0], abs=1e-6)


def _detections_for(cam, nav, birds, t, score=0.9, size=1.0):
    R_wc, p_wc = camera_pose(cam, nav.R_wb, nav.pos)
    uv, z = cam.project(birds, R_wc, p_wc)
    items = []
    for (u, v), zz in zip(uv, z):
        w = cam.fx * size / zz * 0.9
        items.append(Detection((u - w / 2, v - 0.2 * w, u + w / 2, v + 0.2 * w), score))
    return Detections(t, t, 0, R_wc, p_wc, tuple(items))


def test_tracker_follows_a_bird_from_a_moving_camera():
    cam = IMX219_6MM
    trk = BirdTracker(cam, TrackerConfig(size_prior_m=0.9))
    bird0, vb = np.array([0.0, 220.0, 55.0]), np.array([3.0, 2.0, 0.0])
    for k in range(40):
        t = 0.2 * k
        nav = level_nav(t, pos=(0, 15 * t, 70))
        d = _detections_for(cam, nav, (bird0 + vb * t)[None], t)
        u, v = d.items[0].centre
        assert 0 < u < cam.width and 0 < v < cam.height                   # in the frame all along
        trk.process(d)
    tr = trk.snapshot(7.8)
    assert len(tr.items) == 1 and tr.items[0].status == "confirmed"
    est = tr.items[0]
    truth = bird0 + vb * 7.8
    rng = np.linalg.norm(truth - level_nav(7.8, pos=(0, 117, 70)).pos)
    assert np.linalg.norm(est.pos - truth) < 0.3 * rng
    assert np.linalg.norm(est.vel[:2] - vb[:2]) < 5.0


def test_tracker_low_scores_keep_but_do_not_start_tracks():
    cam = IMX219_6MM
    trk = BirdTracker(cam)
    bird = np.array([[0.0, 120.0, 70.0]])
    for k in range(5):
        trk.process(_detections_for(cam, level_nav(0.2 * k), bird, 0.2 * k, score=0.9))
    hits = trk.tracks[0].hits
    for k in range(5, 10):
        trk.process(_detections_for(cam, level_nav(0.2 * k), bird, 0.2 * k, score=0.3))
    assert len(trk.tracks) == 1 and trk.tracks[0].hits == hits + 5           # the low-score boxes kept it alive
    trk2 = BirdTracker(cam)
    for k in range(10):
        trk2.process(_detections_for(cam, level_nav(0.2 * k), bird, 0.2 * k, score=0.3))
    assert trk2.tracks == []                                                    # ... but never start one


def test_tracker_rejects_a_range_far_off():
    """In log range, a track 200 m out does not take a box that says 15 m on the same ray (the old point fusion did)."""
    cam = IMX219_6MM
    trk = BirdTracker(cam)
    bird = np.array([[0.0, 200.0, 70.0]])
    for k in range(5):
        trk.process(_detections_for(cam, level_nav(0.2 * k), bird, 0.2 * k))
    det = _detections_for(cam, level_nav(1.0), bird, 1.0, size=1.0 * 200 / 15)    # same bearing, apparent range ~15 m
    trk.process(det)
    assert len(trk.tracks) == 2


def _track(tid, pos, vel=(0, 0, 0), status="confirmed", box=None, hits=10, since=0.0):
    return TrackEstimate(tid, status, np.array(pos, float), np.array(vel, float), np.eye(6) * 4.0, 0.0, 0.1, hits, 10.0, since, box, 0.9)


def test_scheduler_selects_commits_and_finishes():
    sch = TargetScheduler(SchedulerConfig(n_good=2, area_centre=(0, 200), commit_s=20.0))
    bus = Bus()
    sch.attach(bus)
    bus.publish("nav", level_nav())
    bus.publish("tracks", Tracks(0.0, (_track(1, (0, 150, 70)), _track(2, (0, 400, 70)))))
    sch.step(0.0, bus)
    assert bus.latest("target").track_id == 1
    bus.publish("tracks", Tracks(1.0, (_track(1, (0, 600, 70)), _track(2, (0, 60, 70)))))   # 2 is now much cheaper: still committed
    sch.step(1.0, bus)
    assert bus.latest("target").track_id == 1
    for k in range(2):
        bus.publish("photos", PhotoEvent(2.0 + k, 1, 20.0, 180.0, 0.5, 0.1, 0.9, True))
    sch.step(3.0, bus)
    sel = bus.latest("target")
    assert 1 in sel.done and sel.track_id == 2


def test_guidance_breaks_off_inside_the_separation():
    g = PhotoPassGuidance(GuidanceConfig(min_sep=10.0, range_margin=0.65))
    bus = Bus()
    bus.publish("nav", level_nav())
    bus.publish("target", TargetSelection(0.0, 1, "selected"))
    bus.publish("tracks", Tracks(0.0, (_track(1, (0, 12, 70)),)))                    # 12 m: 0.65 x 12 < 10
    g.step(0.0, bus)
    cmd = bus.latest("guidance")
    assert cmd.mode == "breakoff" and cmd.height > 70.0
    assert abs(abs(geometry_wrap(cmd.course - math.pi / 2)) - math.pi / 2) < 1e-6


def geometry_wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def test_guidance_search_then_intercept_from_behind():
    g = PhotoPassGuidance(GuidanceConfig(search_centre=(0, 300)))
    bus = Bus()
    bus.publish("nav", level_nav())
    g.step(0.0, bus)
    assert bus.latest("guidance").mode == "search"
    bus.publish("target", TargetSelection(1.0, 3, "selected"))
    bus.publish("tracks", Tracks(1.0, (_track(3, (100, 200, 60), vel=(-10, 0, 0)),)))
    g.step(1.0, bus)
    cmd = bus.latest("guidance")
    assert cmd.mode == "intercept"
    assert 60.0 <= cmd.height <= 115.0


def test_photo_trigger_thresholds():
    cam = IMX219_6MM
    ph = PhotoTrigger(cam)
    bus = Bus()
    bus.publish("nav", level_nav())
    bus.publish("target", TargetSelection(0.0, 1, "selected"))
    c = (cam.cx, cam.cy)
    bus.publish("tracks", Tracks(0.0, (_track(1, (0, 30, 70), vel=(0, 13, 0), box=(c[0] - 80, c[1] - 20, c[0] + 80, c[1] + 20)),)))
    ph.step(0.0, bus)
    ev = bus.latest("photos")
    assert ev is not None and ev.good and ev.pixels_across == pytest.approx(160)
    ph2 = PhotoTrigger(cam)
    bus.publish("tracks", Tracks(1.0, (_track(1, (0, 90, 70), box=(c[0] - 20, c[1] - 5, c[0] + 20, c[1] + 5)),)))
    n = len(bus.messages("photos"))
    ph2.step(1.0, bus)
    assert len(bus.messages("photos")) == n                                          # 40 px: no shot


def test_metrics_perfect_tracking():
    cam = IMX219_6MM
    samples = []
    for k in range(10):
        nav = level_nav(k * 0.5, pos=(0, 7.5 * k, 70))
        R_wc, p_wc = camera_pose(cam, nav.R_wb, nav.pos)
        birds = np.array([[2.0, 120.0, 65.0], [-8.0, 140.0, 66.0]])
        samples.append(Sample(k * 0.5, np.array([1, 2]), birds, np.array([1.0, 1.0]), R_wc, p_wc, np.array([7, 9]), birds.copy()))
    m = evaluate(samples, cam)
    assert m["MOTA"] == pytest.approx(1.0) and m["IDF1"] == pytest.approx(1.0) and m["identity switches"] == 0


def test_closed_loop_kinematic_short_and_repeatable():
    from vegeta.mission.sim import BirdField, BirdGroup, run_kinematic

    def once():
        birds = BirdField([BirdGroup("black-headed gull", 3, (0, 250), "thermal", (55, 80), 35)], seed=3)
        cfg = vm.MissionConfig(camera=IMX219_6MM, guidance=GuidanceConfig(search_centre=(0, 250)))
        return run_kinematic(birds, cfg, duration=60.0, seed=3)

    a, b = once(), once()
    sa, sb = a.scores(), b.scores()
    assert sa["detector frames"] > 100 and sa["matches"] > 0
    assert sa == sb                                                                      # deterministic
    modes = {e[2].split(" → ")[1].split(" ")[0] for e in a.stack.guidance.events}
    assert "intercept" in modes
