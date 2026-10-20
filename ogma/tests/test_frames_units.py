"""og.frames reproduces Vegeta's nisus_robot.quat_to_R / euler_from_R (the source of truth, c15e84e); og.units."""
import math

import numpy as np

import ogma as og


def test_quat_to_R_matches_vegeta():
    q = np.array([0.9, 0.1, -0.3, 0.2]); q /= np.linalg.norm(q)
    R = og.frames.quat_to_R(q)
    golden = np.array([[0.7263157894736842, -0.4421052631578947, -0.5263157894736842],      # Vegeta nisus_robot.quat_to_R
                       [0.31578947368421056, 0.8947368421052632, -0.3157894736842105],
                       [0.6105263157894737, 0.06315789473684214, 0.7894736842105263]])
    assert np.allclose(R, golden, atol=1e-15) and np.allclose(R @ R.T, np.eye(3), atol=1e-15)


def test_euler_round_trip():
    q = og.frames.yaw_quat(0.7)
    assert np.allclose(og.frames.euler_from_R(og.frames.quat_to_R(q)), (0.0, 0.0, 0.7))
    assert np.allclose(og.frames.AERO_FROM_BODY @ og.frames.AERO_FROM_BODY, np.eye(3))


def test_units():
    assert og.units.G == 9.80665 and og.units.G_VEGETA == 9.81
    assert math.isclose(og.units.rad_s_to_rpm(og.units.rpm_to_rad_s(1234.5)), 1234.5)


def test_planned_modules_say_so():
    import pytest
    with pytest.raises(AttributeError, match="planned"):
        og.batteries
