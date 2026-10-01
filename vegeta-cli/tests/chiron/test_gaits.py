"""Phase generators: the fixed schedule (sigma = 0) and Tegotae load feedback."""
import math

import numpy as np
import pytest

from vegeta.chiron import PhaseGenerator, cycle_to_oscillator, in_stance, oscillator_to_cycle

LEGS = ["RL", "FL", "RR", "FR"]
BASE = {"RL": 0.0, "FL": 0.25, "RR": 0.5, "FR": 0.75}
DUTY = 0.75


def test_cycle_oscillator_map():
    assert cycle_to_oscillator(0.0, DUTY) == pytest.approx(math.pi)            # touchdown: start of stance
    assert cycle_to_oscillator(DUTY, DUTY) == pytest.approx(0.0)              # lift-off: start of swing
    assert cycle_to_oscillator(DUTY / 2, DUTY) == pytest.approx(1.5 * math.pi)
    assert cycle_to_oscillator(DUTY + (1 - DUTY) / 2, DUTY) == pytest.approx(0.5 * math.pi)
    c = np.linspace(0, 0.999, 400)
    np.testing.assert_allclose(oscillator_to_cycle(cycle_to_oscillator(c, DUTY), DUTY), c, atol=1e-12)
    phi = cycle_to_oscillator(c, DUTY)
    assert np.all((phi >= math.pi) == in_stance(c, DUTY))                      # stance <-> phi in [pi, 2pi)


def test_sigma_zero_is_the_fixed_schedule():
    f, dt = 2.0, 0.001
    pg = PhaseGenerator(LEGS, BASE, DUTY, f, sigma=0.0)
    stance_log = []
    for k in range(1, 3001):
        phases, st = pg.step(dt, normal_forces=np.full(4, 50.0))               # forces are ignored at sigma = 0
        stance_log.append(st.copy())
        if k % 500 == 0:
            exact = np.mod(np.array([BASE[leg] for leg in LEGS]) + f * k * dt, 1.0)
            np.testing.assert_allclose(phases, exact, atol=1e-9)
    st = np.array(stance_log)
    # every leg is in stance duty * T and in swing (1 - duty) * T of each period T
    frac = st.mean(axis=0)
    np.testing.assert_allclose(frac, DUTY, atol=2e-3)
    lift = np.nonzero(st[:-1, 0] & ~st[1:, 0])[0]
    touch = np.nonzero(~st[:-1, 0] & st[1:, 0])[0]
    swing_steps = touch[touch > lift[0]][0] - lift[0]
    assert swing_steps * dt == pytest.approx((1 - DUTY) / f, abs=2 * dt)


def test_tegotae_slows_loaded_late_stance_and_advances_early_stance():
    pg = PhaseGenerator(["a"], {"a": 0.0}, DUTY, 2.0, sigma=0.6)
    pg.reset(phases={"a": 0.9 * DUTY})                                         # late stance: cos(phi) > 0
    assert pg.rate([6.7])[0] < 2.0
    pg.reset(phases={"a": 0.1 * DUTY})                                         # early stance: cos(phi) < 0
    assert pg.rate([6.7])[0] > 2.0
    pg.reset(phases={"a": 0.5 * DUTY})                                         # mid stance: cos(phi) = 0
    assert pg.rate([6.7])[0] == pytest.approx(2.0)
    pg.reset(phases={"a": 0.9 * DUTY})
    assert pg.rate([0.0])[0] == pytest.approx(2.0)                            # unloaded: the free rhythm
    # the rule itself: d(phi)/dt = omega(phi) - sigma N cos(phi)
    phi = cycle_to_oscillator(0.9 * DUTY, DUTY)
    omega = math.pi / (DUTY * 0.5)
    dphidt = pg.rate([6.7])[0] * math.pi / DUTY
    assert dphidt == pytest.approx(omega - 0.6 * 6.7 * math.cos(phi))


def test_loaded_leg_delays_lift_off():
    free = PhaseGenerator(["a"], {"a": 0.5}, DUTY, 2.0, sigma=0.6)
    held = PhaseGenerator(["a"], {"a": 0.5}, DUTY, 2.0, sigma=0.6)
    for _ in range(200):
        free.step(0.001, {"a": 0.0})
        held.step(0.001, {"a": 10.0})
    assert held.phases[0] < free.phases[0]


def test_reset_offset_dict_and_validation():
    pg = PhaseGenerator(LEGS, BASE, DUTY, 1.0)
    np.testing.assert_allclose(pg.reset(offset=0.3), np.mod([0.3, 0.55, 0.8, 1.05], 1.0))
    assert pg.phase_of("FR") == pytest.approx(0.05)
    st, s = pg.progress()
    assert st.tolist() == [True, True, False, True]
    with pytest.raises(ValueError):
        PhaseGenerator(LEGS, {"RL": 0.0}, DUTY, 1.0)
    with pytest.raises(ValueError):
        PhaseGenerator(LEGS, BASE, 1.2, 1.0)
    with pytest.raises(ValueError):
        PhaseGenerator(LEGS, BASE, DUTY, 1.0, sigma=0.5).step(0.001, [1.0, 2.0])
