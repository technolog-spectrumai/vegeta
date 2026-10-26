"""Rotating frame, WALE LES and regularized BGK on the NumPy reference: exact solutions and properties."""
import itertools

import numpy as np
import pytest

from mancha import (
    BGK,
    D3Q19,
    WALE,
    InvalidParameterError,
    NumericalInstabilityError,
    RegularizedBGK,
    RotatingFrame,
    Simulation,
)
from mancha.collision import equilibrium, moments
from mancha.turbulence import velocity_gradient, wale_viscosity
from mancha.validation import cases as C

LAT = D3Q19()


def _far_field(shape=(6, 20, 20), omega=0.004, axial=0.0, collision=None, steps=1500, les=None):
    n = shape[1]
    frame = RotatingFrame(omega, center=((n - 1) / 2, (shape[2] - 1) / 2))
    kw = {"collision": collision} if collision is not None else {"viscosity": 0.05}
    sim = Simulation(shape=shape, frame=frame, velocity=(axial, 0, 0), turbulence=les, **kw)
    if axial:
        sim.set_boundary("x-", "velocity", velocity=(axial, 0, 0)).set_boundary("x+", "pressure", density=1.0)
    for face in ("y-", "y+", "z-", "z+"):
        sim.set_boundary(face, "velocity", velocity=(axial, 0, 0))
    return sim.run(steps)


# ---------------------------------------------------------------------------------------------- rotating frame
def test_fluid_at_rest_in_the_lab_stays_at_rest():
    """Seen from the rotating frame it is a rigid rotation; Coriolis + centrifugal must balance its advection
    with a uniform pressure (a wrong Coriolis term shows up as a density variation of order 1e-3)."""
    sim = _far_field()
    rho = sim.density()
    assert rho.max() - rho.min() < 5e-5
    assert np.abs(sim.absolute_velocity()).max() < 2e-4
    rel = sim.velocity() + sim.frame.frame_velocity(sim.shape)
    assert np.abs(rel).max() < 2e-4                       # velocity() is the relative velocity


def test_uniform_axial_flow_in_the_rotating_frame():
    sim = _far_field(shape=(12, 16, 16), axial=0.03, steps=2000)
    u = sim.absolute_velocity()
    u[0] -= 0.03
    assert np.abs(u).max() < 1e-3


def test_co_rotating_container_has_the_centrifugal_pressure():
    """A cylinder static in the frame: rigid co-rotation, ρ ∝ exp(3 ω² r² / 2), no relative motion."""
    n, om = 32, 0.006
    c = (n - 1) / 2
    y, z = np.meshgrid(np.arange(n) - c, np.arange(n) - c, indexing="ij")
    r2 = y ** 2 + z ** 2
    mask = np.broadcast_to((r2 > 14.0 ** 2)[None], (2, n, n)).copy()
    frame = RotatingFrame(om, center=(c, c))
    with pytest.warns(UserWarning):           # the default (lab-at-rest) start is fast at the corners
        sim = Simulation(shape=(2, n, n), viscosity=0.05, frame=frame)
    sim.set_solid(mask)
    sim.initialize(velocity=frame.frame_velocity(sim.shape))
    sim.run(3000)
    fluid = ~mask[0]
    rho = sim.density()[0][fluid]
    model = np.exp(1.5 * om ** 2 * r2[fluid])
    assert np.abs(sim.velocity()[:, ~mask]).max() < 1e-7
    assert rho.max() / rho.min() - 1 > 5e-3                  # the effect is there ...
    assert np.abs(rho / (rho.mean() / model.mean() * model) - 1).max() < 1e-6   # ... and has the right shape


def test_frame_api():
    frame = RotatingFrame(0.002, center=(7.5, 7.5))
    sim = Simulation(shape=(4, 16, 16), viscosity=0.05, frame=frame, velocity=(0.02, 0.0, 0.0))
    assert np.allclose(sim.absolute_velocity()[0], 0.02) and np.allclose(sim.absolute_velocity()[1:], 0.0)
    assert np.allclose(sim.velocity(), np.array([0.02, 0, 0]).reshape(3, 1, 1, 1) - frame.frame_velocity(sim.shape))
    with pytest.raises(InvalidParameterError):
        sim.set_boundary("y-", "wall")                       # a lab-fixed lateral wall cuts through the rotation
    sim.set_boundary("x-", "wall")                           # a lab-fixed end wall is fine (it slides in-plane)
    with pytest.raises(InvalidParameterError):
        RotatingFrame(0.01, axis="z")
    with pytest.raises(InvalidParameterError):              # frame speed at the faces reaches cs
        Simulation(shape=(4, 16, 16), viscosity=0.05, frame=RotatingFrame(0.08, center=(7.5, 7.5))
                   ).set_boundary("y-", "velocity", velocity=(0, 0, 0))


def test_frame_checkpoint_round_trip(tmp_path):
    a = _far_field(steps=20)
    b = Simulation.restore(a.checkpoint(tmp_path / "f.npz"))
    assert b.frame == a.frame
    a.run(10)
    b.run(10)
    assert np.array_equal(a.populations(), b.populations())


def test_moment_of_an_offset_body():
    """A body mirror-symmetric in y and z, in a periodic stream along x: no moment about its own centre, so its
    moment about any point is (centre − point) × F."""
    sim = Simulation(shape=(16, 12, 12), viscosity=0.05, velocity=(0.03, 0, 0))
    mask = np.zeros(sim.shape, bool)
    mask[6:9, 4:7, 5:8] = True                               # centre (7, 5, 6)
    sim.set_solid(mask).run(200)
    origin = np.array([1.0, 2.0, 3.0])
    force, moment = sim.solid_load(origin=origin)
    expected = np.cross(np.array([7.0, 5.0, 6.0]) - origin, force)
    assert np.allclose(moment, expected, rtol=1e-6, atol=1e-12 * np.abs(force).max())
    front = mask.copy()
    front[7:] = False
    assert np.allclose(sim.solid_force(front) + sim.solid_force(mask & ~front), force)   # parts add up (gauge)
    with pytest.raises(InvalidParameterError):
        sim.solid_load(part=~mask)


# ------------------------------------------------------------------------------------------------------- WALE
def _wale_reference(g, c_w):
    """WALE written out index by index (independent of the vectorised implementation)."""
    s = [[0.5 * (g[i][j] + g[j][i]) for j in range(3)] for i in range(3)]
    g2 = [[sum(g[i][k] * g[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    tr = sum(g2[i][i] for i in range(3)) / 3
    sd = [[0.5 * (g2[i][j] + g2[j][i]) - (tr if i == j else 0.0) for j in range(3)] for i in range(3)]
    ss = sum(s[i][j] ** 2 for i, j in itertools.product(range(3), repeat=2))
    sdsd = sum(sd[i][j] ** 2 for i, j in itertools.product(range(3), repeat=2))
    return c_w ** 2 * sdsd ** 1.5 / (ss ** 2.5 + sdsd ** 1.25)


def test_wale_formula():
    rng = np.random.default_rng(3)
    g = rng.standard_normal((3, 3, 5)) * 1e-2
    nut = wale_viscosity(g, 0.5)
    for k in range(5):
        assert np.isclose(nut[k], _wale_reference(g[:, :, k].tolist(), 0.5), rtol=1e-10)
    assert np.allclose(wale_viscosity(g, 1.0), 4 * nut)


def test_wale_vanishes_in_pure_shear_but_not_in_rotation():
    shear = np.array([[0, 1e-3, 0], [0, 0, 0], [0, 0, 0]], dtype=float).reshape(3, 3, 1)
    rotation = np.array([[0, -1e-3, 0], [1e-3, 0, 0], [0, 0, 0]], dtype=float).reshape(3, 3, 1)
    assert wale_viscosity(shear, 0.5)[0] < 1e-25
    assert wale_viscosity(rotation, 0.5)[0] > 1e-5        # WALE responds to rotation by design ...


def test_wale_in_a_rotating_frame_uses_the_lab_gradient():
    """... so in a rotating frame it must see the lab-frame gradient: the far field (at rest in the lab, a
    rigid rotation in the frame) gets no eddy viscosity, and the result equals the laminar one."""
    a = _far_field(steps=300)
    b = _far_field(steps=300, les=WALE(0.5))
    assert np.abs(a.populations() - b.populations()).max() < 1e-7       # ν_t only from the (tiny) deviations


def test_velocity_gradient_boundary_rule():
    u = np.zeros((3, 5, 4, 3))
    u[0] = np.arange(5).reshape(5, 1, 1) * 1.0                 # ∂u/∂x = 1
    g = velocity_gradient(u, None, (False, True, True))
    assert np.allclose(g[0, 0, 1:-1], 1.0)
    assert np.allclose(g[0, 0, 0], 0.5) and np.allclose(g[0, 0, -1], 0.5)   # one-sided, half weight
    solid = np.zeros((5, 4, 3), bool)
    solid[3] = True
    g = velocity_gradient(u, solid, (False, True, True))
    assert np.allclose(g[0, 0, 2], 0.5)                        # the solid neighbour is replaced by the node


def test_wale_leaves_laminar_channel_flow_alone():
    a = C.poiseuille(ny=10, max_steps=6000, tol=1e-9)
    b = C.poiseuille(ny=10, max_steps=6000, tol=1e-9, turbulence=WALE(0.5))
    assert np.abs(a["profile"] - b["profile"]).max() < 1e-9 * np.abs(a["profile"]).max() + 1e-14


# ---------------------------------------------------------------------------------------------- regularized
def test_regularized_collision_conserves_with_force():
    """Σ f* = ρ and Σ c f* = ρu + F/2 (the Guo invariant the first-order Hermite term protects)."""
    rng = np.random.default_rng(5)
    rho = 1 + 0.01 * rng.standard_normal((4, 3, 2))
    u = 0.03 * rng.standard_normal((3, 4, 3, 2))
    force = np.array([2e-4, -1e-4, 5e-5])
    f = equilibrium(rho, u - force.reshape(3, 1, 1, 1) / (2 * rho), LAT.c, LAT.w) + 1e-4 * rng.random((19, 4, 3, 2))
    r, v = moments(f, LAT.c)
    uh = v + force.reshape(3, 1, 1, 1) / (2 * r)
    for model in (BGK(0.7), RegularizedBGK(0.7)):
        out = model.collide(f, r, uh, LAT.c, LAT.w, force)
        r2, v2 = moments(out, LAT.c)
        assert np.allclose(r2, r)
        assert np.allclose(r2 * v2, r * uh + force.reshape(3, 1, 1, 1) / 2)


@pytest.mark.parametrize("tau", [0.6, 1.2])
def test_regularized_poiseuille_second_order(tau):
    nu = (tau - 0.5) / 3
    errs = [C.poiseuille(ny=n, collision=RegularizedBGK(tau), viscosity=None, tol=1e-11, max_steps=200000
                         )["l2_error_rel"] for n in (8, 16)]
    assert errs[1] < 0.01 and 3.5 < errs[0] / errs[1] < 4.5
    assert nu > 0


def test_regularized_les_survives_where_bgk_blows_up():
    n, u0 = 20, 0.05
    k = 2 * np.pi / n
    x = np.arange(n)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    u = np.stack([u0 * np.sin(k * X) * np.cos(k * Y) * np.cos(k * Z),
                  -u0 * np.cos(k * X) * np.sin(k * Y) * np.cos(k * Z), 0 * X])
    with pytest.warns(UserWarning):
        bgk = Simulation(shape=(n, n, n), tau=0.5002, check_interval=20)
    bgk.initialize(velocity=u)
    with pytest.raises(NumericalInstabilityError):
        bgk.run(1500)
    with pytest.warns(UserWarning):
        les = Simulation(shape=(n, n, n), collision=RegularizedBGK(0.5002), turbulence=WALE(), check_interval=20)
    les.initialize(velocity=u)
    e0 = les.kinetic_energy()
    les.run(1500)
    assert 0 < les.kinetic_energy() < e0
