"""Wake field, load harmonics and their selection rules, dipole tones, slipstream model."""
import math

import numpy as np
import pytest

from vegeta import boreas
from vegeta.boreas import wake

PROP = boreas.Propeller.from_pitch("120 mm 3-blade", 0.12, 0.10, blades=3, chord_root_m=0.018, chord_max_m=0.030, chord_tip_m=0.012)
SEC = boreas.Airfoil(name="marine", cl_alpha=5.5, alpha0_deg=-2.0, cl_max=1.0, cd0=0.02, k=0.05)
RHO = 1025.0


def rel(a, ref):
    return abs(a) / abs(ref)


def test_wake_field_interpolation_and_validation():
    fw = wake.fin_wake(4, mean=0.15, depth=0.25, width_deg=12.0)
    assert fw(0.7, 0.0) == pytest.approx(0.40, abs=1e-6) and fw(0.7, 45.0) == pytest.approx(0.15, abs=1e-3)    # plus the tails of the two neighbouring fin wakes
    assert fw(0.7, 360.0) == pytest.approx(fw(0.7, 0.0)) and fw(2.0, 90.0) == pytest.approx(fw(1.0, 90.0))
    assert fw.mean() == pytest.approx(0.15 + 4 * 0.25 * 12 * math.sqrt(2 * math.pi) / 360, rel=1e-3)
    h = fw.harmonics(orders=12)
    assert h[3] > 0.05 and h[7] > 0.01 and max(h[[0, 1, 2, 4, 5, 6]]) < 1e-12          # orders 4, 8 only
    grid = wake.WakeField([0.5, 1.0], [0.0, 90.0, 180.0, 270.0], [[0.0, 0.2, 0.0, 0.2], [0.2, 0.4, 0.2, 0.4]])
    assert grid(0.75, 45.0) == pytest.approx(0.2) and grid([0.5, 1.0], [90.0, 90.0]).tolist() == pytest.approx([0.2, 0.4])
    with pytest.raises(ValueError):
        wake.WakeField([0.5, 1.0], [0.0, 90.0], [[0.0, 0.1, 0.2]])
    with pytest.raises(ValueError):
        wake.WakeField([1.0, 0.5], [0.0, 90.0], np.zeros((2, 2)))


def test_uniform_wake_gives_steady_loads_equal_to_the_steady_solution():
    h = wake.load_harmonics(PROP, SEC, 910.0, 1.5, wake.uniform_wake(0.15), RHO, n_angles=36)
    op = boreas.solve(PROP, SEC, 910.0, 1.5 * 0.85, RHO)
    assert h.amplitudes["shaft_thrust_N"][0] == pytest.approx(op.thrust, rel=1e-9)
    for name, a in h.amplitudes.items():
        assert np.all(np.abs(a[1:]) < 1e-9 * max(op.thrust, 1.0)), name


@pytest.mark.parametrize("rotation", [1, -1])
def test_fin_wake_selection_rules(rotation):
    h = wake.load_harmonics(PROP, SEC, 910.0, 1.5, wake.fin_wake(4, 0.15, 0.25, 12.0), RHO, rotation=rotation)
    A = h.amplitudes
    T0 = A["shaft_thrust_N"][0]
    # each blade sees the four fin wakes: orders 4, 8, 12
    assert rel(A["blade_thrust_N"][4], T0) > 1e-2 and rel(A["blade_thrust_N"][8], T0) > 1e-3
    assert max(rel(A["blade_thrust_N"][k], T0) for k in (1, 2, 3, 5, 6, 7)) < 1e-9
    # the shaft only keeps orders that are multiples of both the blade count and the fin count: 12
    assert rel(A["shaft_thrust_N"][12], T0) > 1e-3 and max(rel(A["shaft_thrust_N"][k], T0) for k in (3, 4, 6, 8, 9)) < 1e-9
    # side (bearing) forces: blade orders 4 and 8 shifted by one, where that is a multiple of 3: 3 and 9
    side = np.hypot(A["side_force_y_N"], A["side_force_z_N"])
    assert side[3] > 1e-3 * T0 and side[9] > 1e-4 * T0 and max(side[k] for k in (1, 2, 4, 6, 12)) < 1e-9 * T0
    rows = h.table(12)
    assert rows[11]["order"] == 12 and rows[11]["frequency_hz"] == pytest.approx(12 * 910 / 60)
    with pytest.raises(ValueError):
        wake.load_harmonics(PROP, SEC, 910.0, 1.5, wake.uniform_wake(), RHO, n_angles=100)


def test_unsteady_tones_are_dipoles():
    h = wake.load_harmonics(PROP, SEC, 910.0, 1.5, wake.fin_wake(4, 0.15, 0.25, 12.0), RHO)
    on_axis = {t["order"]: t for t in wake.unsteady_tones(h, 1.0, boreas.SEA_WATER, angle_deg=0.0)}
    assert set(on_axis) >= {12} and not {3, 9} & set(on_axis)                    # side forces do not radiate on the axis
    t = on_axis[12]
    p = 2 * math.pi * t["frequency_hz"] * t["thrust_amplitude_N"] / (4 * math.pi * 1500.0 * 1.0) / math.sqrt(2)
    assert t["p_rms_Pa"] == pytest.approx(p) and t["spl_db"] == pytest.approx(20 * math.log10(p / 1e-6))
    far = {x["order"]: x for x in wake.unsteady_tones(h, 10.0, boreas.SEA_WATER, angle_deg=0.0)}
    assert far[12]["spl_db"] == pytest.approx(t["spl_db"] - 20.0)
    broadside = {x["order"] for x in wake.unsteady_tones(h, 1.0, boreas.SEA_WATER, angle_deg=90.0)}
    assert {3, 9} <= broadside and 12 not in broadside


def test_sampled_wake_reads_a_velocity_field():
    def field(points):
        p = np.atleast_2d(points)
        u = np.zeros_like(p)
        u[:, 0] = 2.0 * (1 - 0.1 - 0.2 * (p[:, 1] > 0))            # slower in the +y half
        return u, np.abs(p[:, 2]) < 0.055                           # a strip of invalid points
    w = wake.sampled_wake(field, (0.0, 0.0, 0.0), 0.06, 2.0, n_phi=8)
    assert w(0.7, 0.0) == pytest.approx(0.3) and w(0.7, 180.0) == pytest.approx(0.1)
    with pytest.raises(ValueError, match="no valid samples"):
        wake.sampled_wake(lambda p: (np.zeros((len(p), 3)), np.zeros(len(p), bool)), (0, 0, 0), 0.06, 2.0)


def test_slipstream_model():
    op = boreas.solve(PROP, SEC, 910.0, 1.275, RHO)
    s = wake.slipstream_sampler(PROP, op, center=(0.1, 0.0, 0.0))
    R = PROP.radius
    pts = np.array([[0.1 - 30 * R, 0.5 * R, 0], [0.1, 0.6 * R, 0], [0.1 + 15 * R, 0.4 * R, 0], [0.1, 3 * R, 0], [0.1, 0.0, 0.0]])
    u, ok = s(pts)
    assert ok.tolist() == [False, True, True, True, False]            # beyond the extent; inside the hub
    vi = np.interp(0.6 * R, op.r, op.induced_velocity)
    assert u[1, 0] == pytest.approx(1.275 + vi, rel=0.1)               # at the disc: V + vi
    assert u[2, 0] > u[1, 0] and u[3].tolist() == pytest.approx([1.275, 0.0, 0.0], abs=1e-9)   # accelerating; outside: free stream
    assert u[1, 2] > 0                                                 # swirl in the sense of rotation (+z at +y)
    assert wake.slipstream_sampler(PROP, op, rotation=-1)(np.array([[0.0, 0.6 * R, 0.0]]))[0][0, 2] < 0
    # contraction satisfies continuity inside the tube: div U ~ 0
    h = 1e-4
    x0 = np.array([0.1 + 0.3 * R, 0.3 * R, 0.1 * R])
    div = sum((s(np.array([x0 + h * e]))[0][0, i] - s(np.array([x0 - h * e]))[0][0, i]) / (2 * h) for i, e in enumerate(np.eye(3)))
    assert abs(div) < 0.05 * (np.linalg.norm(s(np.array([x0]))[0]) / R)


def test_any_blade_count_works_with_the_default_angles():
    p7 = boreas.Propeller.from_pitch("7-blade", 0.12, 0.10, blades=7, chord_root_m=0.008, chord_max_m=0.013, chord_tip_m=0.005)
    h = wake.load_harmonics(p7, SEC, 1100.0, 1.5, wake.fin_wake(4, 0.15, 0.25, 12.0), RHO)
    A, T0 = h.amplitudes, h.amplitudes["shaft_thrust_N"][0]
    assert len(h.theta_deg) == 364 and rel(A["blade_thrust_N"][4], T0) > 1e-3
    assert max(rel(A["shaft_thrust_N"][k], T0) for k in (4, 7, 8, 12, 14)) < 1e-9          # 7 blades behind 4 fins: first shaft order 28
    side = np.hypot(A["side_force_y_N"], A["side_force_z_N"])
    assert side[7] > 1e-4 * T0 or side[8 - 1] > 1e-4 * T0                                  # blade order 8 -> side forces at 7
