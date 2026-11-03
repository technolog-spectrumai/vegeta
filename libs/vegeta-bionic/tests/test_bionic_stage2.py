"""vegeta.bionic stage 2 (without a simulator): the hand-to-pincer mapping in both modes and through the clutch, the
vibration feedback, the teleop channel's latency and watchdog, and the recording format."""
import math

import numpy as np
import pytest

from vegeta.bionic import (ContactFeedback, DelayLine, GloveChannel, HandPose, HandSize, OperatorUI, PincerMap,
                           PincerRetarget, Recorder, Recording, TeleopSide, VibrationMotor, key_pose, pinch_aperture)
from vegeta.bionic.hand import interpolate

SIZE = HandSize()


def _run(r, poses, dt=0.01, contact=lambda k: False):
    out = []
    for k, p in enumerate(poses):
        out.append(r.update(k * dt, p, SIZE, contact(k)))
    return out


def test_position_mode_follows_the_pinch_within_the_jaw_speed():
    r = PincerRetarget()
    flat, pinch = key_pose("flat"), key_pose("pinch")
    cmds = _run(r, [flat] * 50 + [pinch] * 100)
    assert cmds[49].opening == pytest.approx(r.map.opening(pinch_aperture(flat, SIZE)))
    assert cmds[-1].opening == pytest.approx(0.0, abs=1e-9)
    steps = np.diff([c.opening for c in cmds]) / 0.01
    assert np.abs(steps).max() <= r.map.jaw_speed + 1e-9


def test_squeeze_past_contact_commands_grip_and_holds_the_jaws():
    r = PincerRetarget()
    half = interpolate(key_pose("flat"), key_pose("pinch"), 0.6)
    poses = [half] * 60 + [interpolate(half, key_pose("pinch"), u) for u in np.linspace(0, 1, 60)]
    cmds = _run(r, poses, contact=lambda k: k >= 60)
    a0, a1 = pinch_aperture(half, SIZE), pinch_aperture(key_pose("pinch"), SIZE)
    assert cmds[-1].grip == pytest.approx(min(r.map.grip_gain * (a0 - a1) * 1000, r.map.grip_max), rel=0.05)
    assert cmds[-1].opening == pytest.approx(cmds[60].opening, abs=1e-6)       # held where they touched


def test_clutch_freezes_and_release_never_jumps():
    r = PincerRetarget()
    flat, pinch = key_pose("flat"), key_pose("pinch")
    _run(r, [flat] * 40)
    held = r.last.opening
    r.set_clutch(True)
    cmds = _run(r, [pinch] * 40)
    assert all(c.opening == held and c.clutched for c in cmds)
    r.set_clutch(False)
    c = r.update(1.0, pinch, SIZE)                                              # the hand is now pinched, the jaws open
    assert c.opening == pytest.approx(held)                                     # re-referenced: no jump


def test_rate_mode_closes_with_index_flexion():
    r = PincerRetarget()
    _run(r, [key_pose("flat")] * 30)
    r.set_mode("rate")
    o0 = r.last.opening
    bent = HandPose(dict(key_pose("flat").joints))
    bent.joints.update(index_mcp=math.radians(85), index_pip=math.radians(100))
    cmds = _run(r, [bent] * 100)
    assert cmds[-1].opening < o0 and abs((o0 - cmds[-1].opening) - r.map.rate_speed * 0.99 * (1.0 - 0.0)) < 0.03
    with pytest.raises(ValueError):
        r.set_mode("turbo")


def test_vibration_feedback():
    fb, motor = ContactFeedback(), VibrationMotor()
    assert fb.drive(0.0, False, 0.0) == 0.0
    assert fb.drive(1.0, True, 1.0) == 1.0                                      # the onset pulse
    light, heavy = fb.drive(1.5, True, 10.0), fb.drive(1.6, True, 1000.0)
    assert fb.floor <= light < heavy <= 1.0
    g = [motor.step(1.0, 0.01) for _ in range(30)]
    assert g[0] < 0.5 * motor.max_g and g[-1] > 0.95 * motor.max_g             # spins up in ~3 tau


def test_delay_line_and_operator_ui():
    d = DelayLine(0.1, default=0)
    d.put(0.0, 5)
    assert d.get(0.05) == 0 and d.get(0.1) == 5
    side = TeleopSide("L", seed=3)
    ui = OperatorUI([(0.5, "L", "clutch", True), (1.0, "L", "mode", "rate")])
    ui.apply(0.6, {"L": side})
    assert side.retarget.clutched and side.retarget.mode == "position" and len(ui.events) == 1


def test_glove_channel_latency_and_watchdog():
    ch = GloveChannel("R", seed=4, latency_s=0.05)
    flat = key_pose("flat")
    got = []
    for k in range(40):
        t = k * 0.01
        ch.operator_hand(t, flat)
        got.append(ch.receive(t))
    first = next(k for k, p in enumerate(got) if p is not None)
    assert first == 5                                                            # 50 ms
    assert ch.receive(1.0) is None                                               # nothing new for 0.6 s: tripped


def test_recording_round_trip(tmp_path):
    rec = Recorder(meta={"robot": "bench"})
    for k in range(5):
        rec.add("robot_state", 0.1 * k, q=np.arange(4) + k, opening=0.1)
    rec.camera("front", {"pos": [1, 0, 1], "fovy": 60, "size": [240, 320], "fps": 10})
    rec.frame("front", 0.0)
    out = Recording.load(rec.save(tmp_path / "demo"))
    assert out.streams["robot_state"]["q"].shape == (5, 4) and out.meta["robot"] == "bench"
    assert out.meta["cameras"]["front"]["video"].startswith("not rendered")
