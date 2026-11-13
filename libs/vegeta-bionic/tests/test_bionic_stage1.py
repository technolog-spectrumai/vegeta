"""vegeta.bionic stage 1: the hand model, the simulated glove, calibration, orientation, the link and safety."""
import math

import numpy as np
import pytest

from vegeta.bionic import (JOINTS, Calibration, GloveSpec, HandPose, HandSize, InProcessLink, Limits, SimGlove, Watchdog,
                           decode, encode, estimate, fingertips, key_pose, motions, pinch_aperture)
from vegeta.bionic import _quat as Q
from vegeta.bionic.sensors import Imu

SENSED_RMS_DEG, SENSED_MAX_DEG, DIP_RMS_DEG = 2.0, 3.5, 6.0     # per-unit assembly scatter (phase sd 4 deg) sets them


def _calibrated(seed=1, poses=None):
    g = SimGlove(seed=seed)
    r = motions.calibration_routine()
    cal = Calibration.fit(g.run(r)[0], r.windows, g.spec, poses=poses)
    cal.with_magnetometer(g.run(motions.wrist_tour(20.0))[0])
    return g, cal


def test_hand_kinematics():
    size = HandSize()
    tips = fingertips(key_pose("flat"), size)
    mc, pp, mp, dp = size.segments["middle"]
    assert tips["middle"][0] == pytest.approx(mc + pp + mp + dp, rel=1e-9) and abs(tips["middle"][2]) < 1e-12   # straight finger
    fist = fingertips(key_pose("fist"), size)
    assert all(np.linalg.norm(fist[f]) < np.linalg.norm(tips[f]) for f in ("index", "middle", "ring", "little"))
    assert pinch_aperture(key_pose("pinch"), size) < 0.01 < 0.05 < pinch_aperture(key_pose("flat"), size)   # the pads meet
    left = fingertips(key_pose("flat"), HandSize(side="L"))
    assert left["thumb"][1] == pytest.approx(-tips["thumb"][1])                                # mirrored
    assert HandSize.percentile(5).length < HandSize.percentile(95).length
    wild = HandPose({j: 3.0 for j in JOINTS}).clamped()
    assert all(a <= math.radians(110) + 1e-9 for a in wild.joints.values())


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_calibration_recovers_sensed_joints(seed):
    g, cal = _calibrated(seed)
    s, truth = g.run(motions.grasp_cycles())
    est = estimate.estimate_hand(s, cal)
    err = np.degrees([[e.joints[j] - t.joints[j] for j in g.spec.sensed] for e, t in zip(est, truth)])
    assert np.sqrt((err ** 2).mean(axis=0)).max() < SENSED_RMS_DEG and np.abs(err).max() < SENSED_MAX_DEG
    dips = [j for j in JOINTS if j.endswith("_dip")]
    derr = np.degrees([[e.joints[j] - t.joints[j] for j in dips] for e, t in zip(est, truth)])
    assert np.sqrt((derr ** 2).mean(axis=0)).max() < DIP_RMS_DEG
    size = g.size
    ap = np.array([(pinch_aperture(e, size), pinch_aperture(t, size)) for e, t in zip(est, truth)])
    assert np.sqrt(np.mean((ap[:, 0] - ap[:, 1]) ** 2)) < 0.002


def test_calibration_with_pose_errors():
    """A real user holds the key poses a few degrees off: the error grows but stays bounded."""
    rng = np.random.default_rng(7)
    r = motions.calibration_routine()
    held = {n: key_pose(n) for n in r.windows}
    actual = {n: HandPose({j: a + math.radians(rng.normal(0, 5.0)) for j, a in p.joints.items()}).clamped() for n, p in held.items()}
    g = SimGlove(seed=1)
    samples = []
    for k in range(int(r.duration * g.spec.rate_hz) + 1):                    # the hand as actually held
        t = k / g.spec.rate_hz
        name = next((n for n, (t0, t1) in r.windows.items() if t0 - 0.3 <= t <= t1 + 0.1), None)
        samples.append(g.sample(t, actual[name] if name else r(t)))
    cal = Calibration.fit(samples, r.windows, g.spec)                         # fitted against the nominal poses
    s, truth = g.run(motions.grasp_cycles())
    err = np.degrees([[cal.angles(x)[j] - t.joints[j] for j in g.spec.sensed] for x, t in zip(s, truth)])
    assert np.sqrt((err ** 2).mean()) < 8.0


def test_calibration_refuses_static_joint():
    g = SimGlove(seed=1)
    flat = motions.Motion([(0.0, key_pose("flat")), (2.0, key_pose("flat"))], {"flat": (0.3, 1.9)})
    with pytest.raises(ValueError, match="do not move"):
        Calibration.fit(g.run(flat)[0], flat.windows, g.spec)


def test_orientation_filter_alone_is_accurate():
    g = SimGlove(seed=1)
    g.imu = Imu.unit(np.random.default_rng(5), acc_bias=np.zeros(3), mag_hard_iron=np.zeros(3))   # gyro bias only
    s, t = g.run(motions.wrist_tour(60.0))
    f = estimate.Mahony(1.0, 0.05, estimate.initial_orientation(s[0].acc, s[0].mag))
    tp, e = 0.0, []
    for x, tr in zip(s, t):
        e.append(math.degrees(Q.angle_between(f.update(x.gyro, x.acc, x.mag, x.t - tp), tr.wrist_quat)))
        tp = x.t
    assert np.mean(e[len(e) // 2:]) < 1.5 and max(e) < 3.0


def test_wrist_orientation_after_magnetometer_calibration():
    g, cal = _calibrated(1)
    assert np.linalg.norm(cal.mag_offset - g.imu.mag_hard_iron) < 0.3
    s, truth = g.run(motions.grasp_cycles())
    est = estimate.estimate_hand(s, cal)
    e = np.degrees([Q.angle_between(a.wrist_quat, b.wrist_quat) for a, b in zip(est, truth)])
    assert e.mean() < 4.0 and e.max() < 6.0


def test_link_round_trip_and_latency():
    g = SimGlove(seed=2)
    s = g.sample(0.0, key_pose("pinch"))
    d = decode(encode(s))
    assert d.seq == s.seq and d.side == s.side and np.array_equal(d.hall, s.hall)
    assert np.allclose(d.gyro, s.gyro, atol=1e-6) and np.allclose(d.mag, s.mag, atol=1e-4)
    link = InProcessLink(latency_s=0.02, jitter_s=0.0, loss=0.1, seed=3)
    samples, _ = g.run(motions.grasp_cycles(n=2))
    got = []
    for x in samples:
        link.send(x.t, encode(x))
        got += link.receive(x.t)
    got += link.receive(1e9)
    st = link.stats()
    assert st["delivered"] + st["lost"] == st["sent"] and 0.05 < st["lost"] / st["sent"] < 0.15
    assert all(ta - s.t == pytest.approx(0.02) for ta, s in got)


def test_link_jitter_reorders_and_stale_messages_are_dropped():
    g = SimGlove(seed=2)
    link = InProcessLink(latency_s=0.005, jitter_s=0.03, seed=4)
    samples, _ = g.run(motions.grasp_cycles(n=2))
    seqs = []
    for x in samples:
        link.send(x.t, encode(x))
        seqs += [s.seq for _, s in link.receive(x.t)]
    seqs += [s.seq for _, s in link.receive(1e9)]
    assert link.stats()["stale"] > 0 and seqs == sorted(seqs)


def test_decode_rejects_foreign_messages():
    with pytest.raises(ValueError):
        decode(b"XX" + bytes(40))


def test_watchdog_and_limits():
    w = Watchdog(0.1)
    assert not w.ok(0.0)
    w.feed(1.0)
    assert w.ok(1.05) and not w.ok(1.2)
    lim = Limits(0.0, 0.08, rate=0.1)
    v = [lim.apply(0.08, 0.01) for _ in range(5)]
    assert v[0] == 0.08                                                       # the first value is taken as is
    lim2 = Limits(0.0, 0.08, rate=0.1, value=0.0)
    assert lim2.apply(0.5, 0.1) == pytest.approx(0.01)                        # range clamp, then the rate
