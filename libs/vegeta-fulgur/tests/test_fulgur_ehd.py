"""P3: the corona solved on FiPy against the exact coaxial space-charge solution."""
import math

import numpy as np
import pytest

pytest.importorskip("fipy")
from scipy.integrate import quad
from scipy.optimize import brentq

from vegeta.fulgur.ehd import CoronaProblem, coaxial, townsend_coaxial_current
from vegeta.fulgur.gas import DRY, EPS0

A, R, L = 0.5e-3, 20e-3, 1.0
MU, E0 = DRY.mobility(), DRY.peek_field(A)
V0 = E0 * A * math.log(R / A)


def exact_current(V: float) -> float:
    """The coaxial corona with Kaptzov at the wire: E(r)^2 = E0^2 (a/r)^2 + (I / 2 pi eps0 mu L)(1 - a^2/r^2),
    V = ∫ E dr — solved for I."""
    def v_of_i(I):
        k = I / (2 * math.pi * EPS0 * MU * L)
        return quad(lambda r: math.sqrt(E0 * E0 * (A / r) ** 2 + k * (1 - (A / r) ** 2)), A, R)[0]
    return 0.0 if V <= V0 else brentq(lambda I: v_of_i(I) - V, 1e-12, 10.0)


@pytest.fixture(scope="module")
def problem():
    mesh, em, gr = coaxial(A, R, L, nr=200)
    return CoronaProblem(mesh, em, gr, E0)


def test_laplace_field_at_the_wire(problem):
    problem._solve_poisson(1e4)
    assert problem.field_at(problem.emitter) == pytest.approx(1e4 / (A * math.log(R / A)), rel=0.01)


def test_below_onset_no_current(problem):
    r = problem.solve(0.9 * V0, open_boundary=False, warm=False)
    assert r.current == 0.0 and r.force_r == 0.0


@pytest.mark.parametrize("factor", [1.3, 2.5, 5.0, 8.0])
def test_current_matches_the_exact_solution(problem, factor):
    r = problem.solve(factor * V0, open_boundary=False)
    assert r.converged
    assert r.current == pytest.approx(exact_current(factor * V0), rel=0.05)
    assert r.e_emitter == pytest.approx(E0, rel=0.02)                      # Kaptzov held


def test_force_identity_and_conservation(problem):
    r = problem.solve(3.0 * V0, open_boundary=False)
    assert r.force_r == pytest.approx(r.current * (R - A) / MU, rel=0.03)    # ∫ rho E dV = I d / mu, outward
    rc = np.asarray(problem.mesh.cellCenters[0])
    I_r = 2 * math.pi * rc * L * np.asarray(problem.rho.value) * MU * problem._E_cell()[0]
    assert I_r.max() / I_r.min() < 1.03                                       # the same current at every radius


def test_townsend_is_the_approximation(problem):
    # Townsend's 1914 formula underestimates the exact solution by ~10-15 % here; the solver follows the exact one
    V = 2.5 * V0
    assert townsend_coaxial_current(V, V0, A, R, L) < exact_current(V) < 1.25 * townsend_coaxial_current(V, V0, A, R, L)


def test_ring_section_builds_and_solves_coarsely():
    from vegeta.fulgur.ehd import ring_section
    from vegeta.fulgur.thruster import MARK4C
    rings = [(r.radius, r.role, r.z) for r in MARK4C.rings]
    mesh, em, gr, edge = ring_section(rings, r_outer=0.16, z_lo=-0.06, z_hi=0.10, lc_edge=40e-6, lc_ground=1.5e-3, lc_far=12e-3)
    assert np.asarray(mesh.cellVolumes).min() > 0                            # no degenerate cells (the hub ring is narrowed)
    assert np.asarray(edge).sum() > 0 and (np.asarray(em) & np.asarray(edge)).sum() == np.asarray(edge).sum()
    prob = CoronaProblem(mesh, em, gr, DRY.peek_field(0.1e-3, 0.9), geometry="planar_axi", inject=edge)
    r = prob.solve(22e3)
    dz = 0.012
    assert r.converged and r.current > 0
    assert r.force_z == pytest.approx(r.current * dz / MU, rel=0.2)          # thrust = I dz / mu
    assert abs(r.force_r) < 0.5 * r.force_z                                   # the radial parts mostly cancel
