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
