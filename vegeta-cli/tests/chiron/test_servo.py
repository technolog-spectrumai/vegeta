"""The servo law: PD torque clipped to the DC-motor torque–speed line, zero beyond the no-load speed."""
import numpy as np
import pytest

from vegeta.chiron import servo_torque, torque_limit
from vegeta.chiron.servo import implicit_slope, saturation

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


# ----------------------------------------------------------------------------------------------- implicit slope
def _fd_slope(q, qd, qt, eps=1e-7):
    """Central finite difference ∂τ/∂q̇ [N·m·s/rad] of the servo law."""
    return (tau(q, qd + eps, qt) - tau(q, qd - eps, qt)) / (2 * eps)


def test_implicit_slope_is_the_laws_derivative_where_it_dissipates():
    cases = [(0.3, 0.05),       # (q̇ [rad/s], q* [rad]) at q = 0 — unclipped PD: −kd
             (2.0, 1.0),        # saturated, pulling on the line (τ > 0, q̇ > 0): −τ_stall/ω₀
             (-2.0, -1.0),      # the same, negative
             (7.0, 1.0)]        # beyond the no-load speed: τ ≡ 0
    for qd, qt in cases:
        t = tau(0.0, qd, qt)
        clipped = abs(KP * qt - KD * qd) > torque_limit(qd, STALL, W0)
        b = implicit_slope(t, qd, clipped, KD, STALL, W0)
        assert b == pytest.approx(_fd_slope(0.0, qd, qt), abs=1e-6), (qd, qt)
        assert b <= 0.0
    assert implicit_slope(tau(0, 0.3, 0.05), 0.3, False, KD, STALL, W0) == pytest.approx(-KD)
    assert implicit_slope(tau(0, 2.0, 1.0), 2.0, True, KD, STALL, W0) == pytest.approx(-STALL / W0)


def test_implicit_slope_while_braking_is_explicit_unless_the_joint_stops_within_the_step():
    qd, qt = 3.0, -1.0                              # moving forward, commanded far behind: braking on the line
    t = tau(0.0, qd, qt)
    assert t == pytest.approx(-STALL * (1 - qd / W0))
    assert _fd_slope(0.0, qd, qt) == pytest.approx(STALL / W0, rel=1e-6)    # positive: no implicit treatment
    assert implicit_slope(t, qd, True, KD, STALL, W0) == 0.0                 # explicit
    assert implicit_slope(t, qd, True, KD, STALL, W0, step_dv=0.5 * qd) == 0.0
    assert implicit_slope(t, qd, True, KD, STALL, W0, step_dv=1.5 * qd) == pytest.approx(-STALL / W0)


def test_implicit_slope_vectorised_and_out_buffer():
    t = np.array([0.5, 6.0, -0.79, 0.0, -6.0])
    qd = np.array([0.1, 0.0, 5.0, 7.0, -1.0])
    clipped = np.array([False, True, True, True, True])
    kd = np.array([0.8, 0.8, 0.5, 0.8, 0.8])
    out = np.full(5, np.nan)
    b = implicit_slope(t, qd, clipped, kd, STALL, W0, step_dv=np.array([0.0, 1.0, 1.0, 1.0, 0.0]), out=out)
    assert b is out
    np.testing.assert_allclose(b, [-0.8, -STALL / W0, 0.0, 0.0, -STALL / W0])
