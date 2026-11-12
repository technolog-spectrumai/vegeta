"""labkit: the shared primitives, the machine registry and the bootstrap."""
import sys

import pytest

from vegeta.labkit import control, kinematics, machines


def test_smooth_and_phase():
    assert control.smooth(-1) == 0 and control.smooth(2) == 1 and control.smooth(0.5) == 0.5
    p = control.Phase("go", update=lambda m, o, t: True)
    assert p.timeout == 30.0 and p.start is None


def test_two_link_ik_round_trips_and_refuses():
    import math
    L1, L2 = 1.0, 0.8
    for x, z in ((0.0, -1.5), (0.3, -1.2), (-0.4, -1.0)):
        a1, a2 = (math.radians(a) for a in kinematics.ik_two_link_planar(x, z, L1, L2))
        assert -L1 * math.sin(a1) + L2 * math.sin(a2) == pytest.approx(x, abs=1e-9)      # as onager.axle_x
        assert -(L1 * math.cos(a1) + L2 * math.cos(a2)) == pytest.approx(z, abs=1e-9)
    assert kinematics.ik_two_link_planar(0.0, -3.0, L1, L2) is None


def test_registry_discovers_a_lab(tmp_path):
    d = tmp_path / "labs" / "toy" / "designs"
    d.mkdir(parents=True)
    (d / "toy_robot.py").write_text("def toy():\n    return 'a robot'\n")
    (d / "machines.py").write_text("from vegeta.labkit import machines\nmachines.register('toy', 'toy_robot:toy')\n")
    try:
        assert "toy" in machines.discover(d)
        m = machines.get("toy")
        assert m.lab == "toy" and m.robot() == "a robot"
        with pytest.raises(KeyError):
            machines.get("nothing")
    finally:
        machines._REGISTRY.pop("toy", None)
        sys.modules.pop("_vegeta_machines_toy", None)


def test_two_labs_cannot_register_one_name():
    machines._current_lab[0] = "a"
    machines.register("twin", lambda: None)
    machines._current_lab[0] = "b"
    try:
        with pytest.raises(ValueError):
            machines.register("twin", lambda: None)
    finally:
        machines._current_lab[0] = None
        machines._REGISTRY.pop("twin", None)


def test_propulsion_block_from_a_synthetic_grid():
    """vegeta.labkit.propulsion: a grid made from CT = 0.1 − 0.08 J, CP = 0.05 − 0.02 J gives those coefficients back
    (one value per distinct J), the reduced model at a point and its residual elsewhere, and the motor's power limit
    within the pack."""
    import math
    import numpy as np
    from vegeta.labkit import propulsion as lp
    rpm, V, D, rho = np.array([6000.0, 9000.0, 12000.0]), np.array([0.0, 5.08, 10.16, 15.24]), 0.254, 1.225
    n = (rpm / 60)[:, None]
    J = V[None, :] / (n * D)
    T, P = (0.10 - 0.08 * J) * rho * n ** 2 * D ** 4, (0.05 - 0.02 * J) * rho * n ** 3 * D ** 5
    grid = {"rpm": rpm.tolist(), "airspeed_m_s": V.tolist(), "rho": rho, "thrust_n": T.tolist(), "power_w": P.tolist(),
            "torque_nm": (P / (2 * math.pi * n)).tolist()}
    co = lp.coefficients(grid, D)
    Jc = np.asarray(co["J"])
    assert len(Jc) == 8 and np.all(np.diff(Jc) > 0)
    np.testing.assert_allclose(co["CT"], 0.10 - 0.08 * Jc, atol=1e-12)
    np.testing.assert_allclose(co["CP"], 0.05 - 0.02 * Jc, atol=1e-12)
    assert lp.at(grid, "thrust_n", 9000.0, 5.08) == pytest.approx(T[1, 1]) and lp.rpm_for_thrust(grid, T[1, 0]) == pytest.approx(9000.0)
    red = lp.reduced(grid, rpm=12000.0, taken_at="full", tau_up=0.03, tau_down=0.06, check=(12000.0, 10.16))
    w = 12000.0 * 2 * math.pi / 60
    assert red["kF"] == pytest.approx(T[2, 0] / w ** 2) and red["kM"] == pytest.approx(P[2, 0] / w / T[2, 0]) and red["tau_down"] == 0.06
    assert red["residual"] == pytest.approx(abs(T[2, 0] - T[2, 2]) / T[2, 2])
    volt = lambda r, q: (r / 2400 + (1.0 + q / 0.5) * 0.05, 1.0 + q / 0.5)             # a 2400 KV motor: V = rpm/KV + I R
    Pm, r = lp.max_shaft_power(grid, battery_v=4.0, voltage_for=volt)
    assert r == 9000.0 and Pm == pytest.approx(P[1, 0])                                   # 12000 rpm needs 5 V: out
    with pytest.raises(ValueError):
        lp.max_shaft_power(grid, battery_v=1.0, voltage_for=volt)
    blk = lp.block([lp.rotor("r0", (0, 0, 0), (0, 0, 2), 1, "p", "m")], [{"name": "p"}], [lp.motor_entry("m", Pm, kv=2400)], reduced={"r0": red})
    assert blk["rotors"][0]["axis"] == [0.0, 0.0, 1.0] and blk["motors"]["m"]["kv_rpm_per_v"] == 2400
    with pytest.raises(KeyError):
        lp.block([lp.rotor("r0", (0, 0, 0), (0, 0, 1), 1, "p", "x")], [{"name": "p"}], [])
