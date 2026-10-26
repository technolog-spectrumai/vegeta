"""Equilibrium moments, BGK conservation, Guo forcing moments and parameter validation."""
import numpy as np
import pytest

from mancha import BGK, D3Q19, InvalidParameterError
from mancha.collision import equilibrium, from_parameters, moments, tau_from_viscosity, viscosity_from_tau
from mancha.collision.bgk import guo_source

LAT = D3Q19()
RNG = np.random.default_rng(1)


def _fields(shape=(4, 3, 2)):
    rho = 1.0 + 0.05 * RNG.standard_normal(shape)
    u = 0.05 * RNG.standard_normal((3,) + shape)
    return rho, u


def test_equilibrium_moments():
    rho, u = _fields()
    feq = equilibrium(rho, u, LAT.c, LAT.w)
    r, v = moments(feq, LAT.c)
    assert np.allclose(r, rho) and np.allclose(v, u)
    c = LAT.c.astype(float)
    pi = np.einsum("ia,ib,i...->ab...", c, c, feq)
    expected = rho * (np.einsum("a...,b...->ab...", u, u) + LAT.cs2 * np.eye(3)[:, :, None, None, None])
    assert np.allclose(pi, expected)


def test_bgk_conserves_mass_and_momentum():
    rho, u = _fields()
    f = equilibrium(rho, u, LAT.c, LAT.w) + 1e-3 * RNG.standard_normal((19,) + rho.shape)
    r, v = moments(f, LAT.c)
    out = BGK(0.8).collide(f, r, v, LAT.c, LAT.w)
    r2, v2 = moments(out, LAT.c)
    assert np.allclose(r2, r) and np.allclose(r2 * v2, r * v)


def test_bgk_relaxes_to_equilibrium_with_tau_one():
    rho, u = _fields()
    f = equilibrium(rho, u, LAT.c, LAT.w) + 1e-3 * RNG.standard_normal((19,) + rho.shape)
    r, v = moments(f, LAT.c)
    assert np.allclose(BGK(1.0).collide(f, r, v, LAT.c, LAT.w), equilibrium(r, v, LAT.c, LAT.w))


def test_guo_source_moments():
    _, u = _fields()
    force = np.array([1e-4, -2e-4, 5e-5])
    omega = 1 / 0.7
    s = guo_source(u, LAT.c, LAT.w, force, omega)
    assert np.allclose(s.sum(axis=0), 0.0)
    assert np.allclose(np.tensordot(LAT.c.T.astype(float), s, axes=(1, 0)),
                       (1 - omega / 2) * force.reshape(3, 1, 1, 1))


@pytest.mark.parametrize("tau", [0.5, 0.3, -1.0, float("nan"), float("inf")])
def test_invalid_tau_is_refused(tau):
    with pytest.raises(InvalidParameterError):
        BGK(tau)


def test_marginal_tau_warns():
    with pytest.warns(UserWarning):
        BGK(0.505)


def test_viscosity_tau_round_trip_and_parameters():
    assert np.isclose(viscosity_from_tau(tau_from_viscosity(0.02)), 0.02)
    assert np.isclose(BGK.from_viscosity(1 / 6).tau, 1.0)
    with pytest.raises(InvalidParameterError):
        BGK.from_viscosity(0.0)
    m = from_parameters(BGK(0.9).parameters())
    assert isinstance(m, BGK) and m.tau == 0.9
