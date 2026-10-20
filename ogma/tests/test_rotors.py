"""og.rotors reproduces Drongo's layout, mixer and rotor wrench and Kestrel's induced velocity (golden/rotors.json,
Vegeta 9a88dba)."""
import json
import math
from pathlib import Path

import numpy as np

from ogma import frames, rotors

G = json.loads((Path(__file__).with_name("golden") / "rotors.json").read_text())


def test_layout_and_mixer():
    pos, spin = rotors.x_layout(G["R_arm"], G["z"])
    assert np.allclose(pos, G["pos"], rtol=0, atol=1e-15)
    Ainv = np.linalg.inv(rotors.allocation(pos, spin, G["kappa"]))
    lo, hi = G["idle"] * G["max_thrust"], G["max_thrust"]
    for (T, M), (u, sat) in zip(G["cases"], G["mix"]):
        got, s = rotors.mix(Ainv, T, np.array(M), lo, hi)
        assert s == sat and np.allclose(got, u, rtol=1e-13, atol=1e-13)


def test_wrench_about_the_com():
    pos, spin = rotors.x_layout(G["R_arm"], G["z"])
    R = frames.quat_to_R(G["q"])
    F, M = rotors.wrench(G["T"], R, pos, spin, G["kappa"], com_offset=G["off"])
    assert np.allclose(F, G["force"], atol=1e-14) and np.allclose(M, G["torque"], atol=1e-14)


def test_induced_velocity_and_drag():
    got = [rotors.induced_velocity(t, V, a, 0.05) for t in (0.0, 5.0, 20.0) for V in (0.0, 8.0) for a in (0.0, 1.2, math.pi)]
    assert np.allclose(got, G["iv"], rtol=1e-14)
    assert np.allclose(rotors.quadratic_drag([10.0, 0, 0], 0.02, 1.2), [-1.2, 0, 0])
    assert math.isclose(rotors.lag_alpha(0.002, 0.035), 1 - math.exp(-0.002 / 0.035))
