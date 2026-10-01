"""The servo law: PD torque clipped to the DC-motor torque–speed line, zero beyond the no-load speed."""
import numpy as np
import pytest

from vegeta.chiron import servo_torque, torque_limit
from vegeta.chiron.servo import saturation

STALL, W0, KP, KD = 6.0, 5.76, 40.0, 0.8


def tau(q, qd, qt, qdt=0.0, ff=0.0):
    return servo_torque(q, qd, qt, qdt, ff, kp=KP, kd=KD, stall=STALL, no_load_speed=W0)


def test_unsaturated_is_plain_pd():
    assert tau(0.0, 0.0, 0.05) == pytest.approx(KP * 0.05)
    assert tau(0.1, 0.5, 0.05, 0.2, 0.3) == pytest.approx(KP * (-0.05) + KD * (0.2 - 0.5) + 0.3)


def test_clipped_to_stall_at_rest():
    assert tau(0.0, 0.0, 1.0) == pytest.approx(STALL)
    assert tau(0.0, 0.0, -1.0) == pytest.approx(-STALL)


def test_follows_the_torque_speed_line():
    for frac in (0.0, 0.25, 0.5, 0.9):
        w = frac * W0
        limit = STALL * (1 - frac)
        assert tau(0.0, w, 10.0, qdt=w) == pytest.approx(limit)          # motoring
        assert tau(0.0, -w, -10.0, qdt=-w) == pytest.approx(-limit)
        assert tau(0.0, w, -10.0, qdt=w) == pytest.approx(-limit)        # braking: same bound (protocol §1)
        assert torque_limit(w, STALL, W0) == pytest.approx(limit)


def test_zero_at_and_beyond_no_load_speed():
    for w in (W0, 1.2 * W0, -W0, -3 * W0):
        assert tau(0.0, w, 5.0) == 0.0
        assert tau(0.0, w, -5.0) == 0.0


def test_vectorised_over_joints_and_out_buffers_match():
    rng = np.random.default_rng(0)
    n = 36
    q, qd, qt = rng.normal(0, 0.5, n), rng.normal(0, 4.0, n), rng.normal(0, 0.5, n)
    qdt, ff = rng.normal(0, 1.0, n), rng.normal(0, 0.5, n)
    kp, kd = rng.uniform(10, 50, n), rng.uniform(0.1, 1.0, n)
    stall, w0 = rng.uniform(2, 24, n), rng.uniform(3, 30, n)
    ref = np.array([servo_torque(q[i], qd[i], qt[i], qdt[i], ff[i], kp=kp[i], kd=kd[i], stall=stall[i],
                                 no_load_speed=w0[i]) for i in range(n)])
    vec = servo_torque(q, qd, qt, qdt, ff, kp=kp, kd=kd, stall=stall, no_load_speed=w0)
    out, work = np.zeros(n), np.zeros(n)
    buf = servo_torque(q, qd, qt, qdt, ff, kp=kp, kd=kd, stall=stall, no_load_speed=w0, out=out, work=work)
    assert buf is out
    np.testing.assert_allclose(vec, ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)
    assert np.all(np.abs(vec) <= stall * np.maximum(0, 1 - np.abs(qd) / w0) + 1e-12)


def test_saturation_flags():
    ts, ss = saturation([6.0, 1.0, 0.0], [0.0, 0.0, 6.0], STALL, W0)
    assert ts.tolist() == [True, False, True]
    assert ss.tolist() == [False, False, True]
