"""og.envelope: membrane theory reduces to the cylinder and sphere formulas; beam loads close."""
import numpy as np
import pytest

from ogma import envelope as en


def test_cylinder_and_sphere():
    x = np.linspace(0, 10, 101)
    f = en.membrane_forces(x, np.full_like(x, 2.0), 300.0)
    assert np.allclose(f["hoop"], 600.0) and np.allclose(f["meridional"], 300.0)
    th = np.linspace(0.3, np.pi - 0.3, 2001)
    xs, rs = -3.0 * np.cos(th), 3.0 * np.sin(th)
    f = en.membrane_forces(xs, rs, 200.0)
    mid = slice(100, -100)
    assert np.allclose(f["hoop"][mid], 300.0, rtol=1e-3)
    assert np.allclose(f["meridional"][mid], 300.0, rtol=1e-3)


def test_wrinkling():
    M, r = 5e4, 2.0
    dp = en.wrinkling_pressure(M, r)
    assert dp * r / 2 == pytest.approx(en.bending_line_load(M, r))


def test_beam_loads_in_equilibrium():
    # lift on the middle half, weight on the two end quarters (in force and moment equilibrium): the hand solution
    # is M(5) = −625 N m at the middle, zero shear and moment at both free ends
    x = np.linspace(0, 10, 1000)                 # no station on the jumps
    w = np.where((x > 2.5) & (x < 7.5), 100.0, -100.0)
    shear, moment = en.beam_loads(x, w)
    assert shear[-1] == pytest.approx(0.0, abs=2.0)          # trapezoids across the two jumps: ±1 N
    assert moment[-1] == pytest.approx(0.0, abs=10.0)        # the ±1 N shear residual over 5 m
    assert moment.min() == pytest.approx(-625.0, rel=5e-3)
