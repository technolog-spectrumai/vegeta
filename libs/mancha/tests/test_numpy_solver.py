"""Regression tests of the NumPy reference solver against exact solutions (small grids, loose tolerances).

These guard the implementation; the scientific record (finer grids, refinement studies) is in benchmarks/.
"""
import numpy as np
import pytest

from mancha import Simulation
from mancha.diagnostics import observed_order
from mancha.validation import cavity, channel, poiseuille, sphere, taylor_green, uniform


def test_rest_state_is_preserved():
    r = uniform(shape=(8, 8, 8), steps=50)
    assert r["velocity_error_max"] < 1e-14 and r["density_error_max"] < 1e-14
    assert abs(r["mass_drift_rel"]) < 1e-14


def test_uniform_flow_is_preserved():
    r = uniform(shape=(8, 6, 4), velocity=(0.05, -0.02, 0.03), steps=50)
    assert r["velocity_error_max"] < 1e-13 and abs(r["mass_drift_rel"]) < 1e-13


def test_float32_reference():
    r = uniform(shape=(8, 6, 4), velocity=(0.05, -0.02, 0.03), steps=50, dtype=np.float32)
    assert r["velocity_error_max"] < 1e-6


def test_taylor_green_decay_and_second_order():
    errors = [abs(taylor_green(n=n)["decay_rate_error_rel"]) for n in (16, 32)]
    assert errors[1] < 0.03
    assert observed_order([16, 32], errors) > 1.8
    r = taylor_green(n=16, nz=3)
    assert abs(r["mass_drift_rel"]) < 1e-12
    assert np.allclose(r["simulation"].velocity()[2], 0.0, atol=1e-14)   # stays two-dimensional


def test_taylor_green_same_in_the_xz_plane():
    """The vortex rotated into the x–z plane gives the same field (the lattice treats the axes alike)."""
    r = taylor_green(n=16, steps=100)
    n, k, u0 = 16, 2 * np.pi / 16, 0.04
    x = np.arange(n, dtype=float)
    X, Z = np.meshgrid(x, x, indexing="ij")
    u = np.zeros((3, n, 1, n))
    u[0, :, 0, :] = u0 * np.sin(k * X) * np.cos(k * Z)
    u[2, :, 0, :] = -u0 * np.cos(k * X) * np.sin(k * Z)
    rho = (1 + 0.75 * u0 ** 2 * (np.cos(2 * k * X) + np.cos(2 * k * Z)))[:, None, :]
    sim = Simulation(shape=(n, 1, n), viscosity=0.02).initialize(density=rho, velocity=u)
    sim.run(100)
    a, b = r["simulation"].velocity()[:, :, :, 0], sim.velocity()[:, :, 0, :]
    assert np.allclose(a[0], b[0], atol=1e-14) and np.allclose(a[1], b[2], atol=1e-14)


def test_poiseuille_profile():
    r = poiseuille(ny=12)
    assert r["residual"] < 1e-8
    assert r["l2_error_rel"] < 0.01
    assert abs(r["mass_drift_rel"]) < 1e-12


def test_poiseuille_exact_at_magic_tau():
    """Halfway bounce-back + BGK is exact for Poiseuille flow at tau = 1/2 + sqrt(3/16)."""
    nu = np.sqrt(3 / 16) / 3
    r = poiseuille(ny=8, viscosity=nu, tol=1e-11, max_steps=60000)
    assert r["l2_error_rel"] < 1e-6


def test_closed_box_conserves_mass():
    sim = Simulation(shape=(10, 10, 10), viscosity=0.05)
    for face in ("x-", "x+", "y-", "z-", "z+"):
        sim.set_boundary(face, "wall")
    sim.set_boundary("y+", "wall", velocity=(0.05, 0, 0.02))
    m0 = sim.mass()
    sim.run(300)
    assert abs(sim.mass() / m0 - 1) < 1e-12
    assert np.abs(sim.velocity()).max() > 1e-3


@pytest.mark.slow
def test_channel_inlet_outlet():
    r = channel(nx=48, ny=12)
    assert r["flux_variation_rel"] < 1e-6
    assert r["profile_l2_error_rel"] < 0.01
    assert 1.45 < r["peak_ratio"] < 1.52


@pytest.mark.slow
def test_cavity_re100_against_ghia():
    r = cavity(n=24, tol=1e-6)
    assert r["ghia_u_max_abs_error"] < 0.03 and r["ghia_v_max_abs_error"] < 0.03
    assert r["vortex_centre_error"] < 0.05


@pytest.mark.slow
def test_sphere_runs_and_drags():
    r = sphere(diameter=6, reynolds=10, size=(5, 3, 3), tol=1e-5)
    assert r["cd"] > r["cd_schiller_naumann"]   # confinement only raises the drag
    assert abs(r["lateral_force"][0]) < 1e-3 * r["cd"] and abs(r["lateral_force"][1]) < 1e-3 * r["cd"]
