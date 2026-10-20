"""og.motors reproduces Boreas's motor, Falco's static point and Chiron's servo law (tests/golden/motors.json)."""
import json
from pathlib import Path

import numpy as np
import pytest

from ogma import motors

G = json.loads((Path(__file__).parent / "golden" / "motors.json").read_text())


def test_dc_motor_is_boreas():
    for name, kv, R, I0, Imax in G["motors"]:
        m = motors.DCMotor(name, kv_rpm_per_volt=kv, resistance_ohm=R, no_load_current_a=I0, max_current_a=Imax)
        g = G["motor"][name]
        assert m.kt == g["kt"]
        assert [list(m.voltage_for(n, q)) for n, q in G["points"]] == g["voltage_for"]
        assert [m.efficiency(n, q) for n, q in G["points"]] == g["efficiency"]
    with pytest.raises(ValueError):
        motors.Motor("bad", 0.0, 0.1, 0.5, 10.0)


def test_operating_point_is_falcos_static_point():
    for kv, R, I0, V, cQ, n, I, Q in G["static"].values():
        got = motors.operating_point(kv, R, I0, V, lambda r: cQ * r * r)
        assert got == (n, I) and cQ * got[0] * got[0] == Q


def test_fit_resistance_puts_the_motor_at_its_point():
    R = motors.fit_resistance(22.2, 9000.0, 540.0, 40.0)
    m = motors.DCMotor("m", 540.0, R, 1.6, 85.0)
    V, I = m.voltage_for(9000.0, m.kt * (40.0 - 1.6))
    assert V == pytest.approx(22.2) and I == pytest.approx(40.0)


def test_servo_law_is_chirons():
    s = G["servo"]
    q, qd, qt, stall, w0 = (np.array(s[k]) for k in ("q", "qd", "qt", "stall", "w0"))
    kw = dict(kp=8.0, kd=0.3, stall=stall, no_load_speed=w0)
    tau = motors.servo_torque(q, qd, qt, 0.5, 0.1, **kw)
    np.testing.assert_array_equal(tau, s["tau"])
    np.testing.assert_array_equal(motors.servo_torque(q, qd, qt, 0.5, 0.1, four_quadrant=np.arange(12) % 2 == 0, **kw), s["four_quadrant"])
    out = np.empty(12)
    motors.servo_torque(q, qd, qt, 0.5, 0.1, out=out, **kw)
    np.testing.assert_array_equal(out, s["out"])
    np.testing.assert_array_equal(motors.torque_limit(qd, stall, w0), s["limit"])
    clipped = np.array(s["clipped"])
    np.testing.assert_array_equal(motors.implicit_slope(tau, qd, clipped, 0.3, stall, w0), s["slope"])
    np.testing.assert_array_equal(motors.implicit_slope(tau, qd, clipped, 0.3, stall, w0, step_dv=np.array(s["step_dv"])), s["slope_step"])
    ts, ss = motors.saturation(tau, qd, stall, w0)
    assert ts.tolist() == s["torque_sat"] and ss.tolist() == s["speed_sat"]


def test_transmissions_are_the_robots_hand_conversions():
    # Onager Atlas: no_load_speed = a.no_load_mm_s / 1000.0 / TILT_LEVER; Pekari: F / ROLLER_SHARE * r, v0 / r
    assert motors.lever(800.0, 25.0 / 1000.0, 0.12) == (800.0 * 0.12, 25.0 / 1000.0 / 0.12)
    assert motors.lever(300.0 / 0.4, 0.5, 0.03) == (300.0 / 0.4 * 0.03, 0.5 / 0.03)
    assert motors.gearbox(0.2, 900.0, 12.0, 0.9) == (0.2 * 12.0 * 0.9, 900.0 / 12.0)
